import anndata
import miniball
import numpy as np
import pandas as pd
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.stats import poisson
from sklearn.cluster import DBSCAN


from .utils import *


# ============================================================ mcDETECT ============================================================ #


class mcDETECT:
    
    
    def __init__(self, type, transcripts, gnl_genes, nc_genes = None, eps = 1.5, minspl = None, grid_len = 1.0, cutoff_prob = 0.95, alpha = 5.0, low_bound = 3,
                 size_thr = 4.0, in_soma_thr = 0.1, l = 1.0, rho = 0.2, s = 1.0, nc_top = 20, nc_thr = 0.1):
        
        self.type = type                        # string, now supports "discrete" (e.g., MERSCOPE, CosMx) and "continuous" (e.g., Xenium)
        self.transcripts = transcripts          # dataframe, transcripts file
        self.gnl_genes = gnl_genes              # list, string, all granule markers
        self.nc_genes = nc_genes                # list, string, all negative controls
        self.eps = eps                          # numeric, searching radius epsilon
        self.minspl = minspl                    # integer, manually select min_samples, i.e., no automatic parameter selection
        self.grid_len = grid_len                # numeric, length of grids for computing the tissue area
        self.cutoff_prob = cutoff_prob          # numeric, cutoff probability in parameter selection for min_samples
        self.alpha = alpha                      # numeric, scaling factor in parameter selection for min_samples
        self.low_bound = low_bound              # integer, lower bound in parameter selection for min_samples
        self.size_thr = size_thr                # numeric, threshold for maximum radius of an aggregation
        self.in_soma_thr = in_soma_thr          # numeric, threshold for in-soma ratio
        self.l = l                              # numeric, scaling factor for seaching overlapped spheres
        self.rho = rho                          # numeric, threshold for determining overlaps
        self.s = s                              # numeric, scaling factor for merging overlapped spheres
        self.nc_top = nc_top                    # integer, number of negative controls retained for filtering
        self.nc_thr = nc_thr                    # numeric, threshold for negative control filtering
        
        if self.type == "discrete":
            self.z_grid = sorted(self.transcripts["global_z"].unique())     # sorted z-grid for discrete platforms
            self.z_grid_np = np.asarray(self.z_grid, dtype = float)         # numpy array of z-grid for discrete platforms
        elif self.type == "continuous":
            self.z_grid = None
            self.z_grid_np = None
        else:
            raise ValueError(f"Invalid platform type: {self.type}")
    
    
    # [INNER] construct grids, input for tissue_area()
    def construct_grid(self, grid_len = None):
        if grid_len is None:
            grid_len = self.grid_len
        x_min, x_max = np.min(self.transcripts["global_x"]), np.max(self.transcripts["global_x"])
        y_min, y_max = np.min(self.transcripts["global_y"]), np.max(self.transcripts["global_y"])
        x_min = np.floor(x_min / grid_len) * grid_len
        x_max = np.ceil(x_max / grid_len) * grid_len
        y_min = np.floor(y_min / grid_len) * grid_len
        y_max = np.ceil(y_max / grid_len) * grid_len
        x_bins = np.arange(x_min, x_max + grid_len, grid_len)
        y_bins = np.arange(y_min, y_max + grid_len, grid_len)
        return x_bins, y_bins
    
    
    # [INNER] calculate tissue area, input for poisson_select()
    def tissue_area(self):
        x_bins, y_bins = self.construct_grid(grid_len = None)
        hist, _, _ = np.histogram2d(self.transcripts["global_x"], self.transcripts["global_y"], bins = [x_bins, y_bins])
        area = np.count_nonzero(hist) * (self.grid_len ** 2)
        return area
    
    
    # [INNER] calculate optimal min_samples, input for dbscan()
    def poisson_select(self, gene_name):
        num_trans = np.sum(self.transcripts["target"] == gene_name)
        bg_density = num_trans / self.tissue_area()
        cutoff_density = poisson.ppf(self.cutoff_prob, mu = self.alpha * bg_density * (np.pi * self.eps ** 2))
        optimal_m = int(max(cutoff_density, self.low_bound))
        return optimal_m
    
    
    # [INTERMEDIATE] dictionary, low-in-soma spheres for each granule marker
    def dbscan(self, target_names = None, record_cell_id = False):
        
        if target_names is None:
            target_names = self.gnl_genes
        transcripts = self.transcripts[self.transcripts["target"].isin(target_names)]
        
        z_grid = self.z_grid_np if self.type == "discrete" else None
        
        sphere_dict = {}
        
        for j_idx, j in enumerate(target_names):
            
            # split transcripts
            target = transcripts[transcripts["target"] == j]
            others = transcripts[transcripts["target"] != j]
            tree = make_tree(d1 = np.array(others["global_x"]), d2 = np.array(others["global_y"]), d3 = np.array(others["global_z"]))
            
            # 3D DBSCAN
            if self.minspl is None:
                min_spl = self.poisson_select(j)
            else:
                min_spl = self.minspl
            X = np.array(target[["global_x", "global_y", "global_z"]])
            db = DBSCAN(eps = self.eps, min_samples = min_spl, algorithm = "kd_tree").fit(X)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # iterate over all aggregations
            cell_id, sphere_x, sphere_y, sphere_z, layer_z, sphere_r, sphere_size, sphere_comp, sphere_score = [], [], [], [], [], [], [], [], []
            
            for k in range(n_clusters):
                
                # ---------- find minimum enclosing spheres ---------- #
                mask = (labels == k)
                coords = X[mask]
                if coords.shape[0] == 0:
                    continue
                
                temp = pd.DataFrame(coords, columns=["global_x", "global_y", "global_z"])
                temp = temp.drop_duplicates()
                coords_unique = temp.to_numpy()
                
                # skip clusters with too few unique points
                if coords_unique.shape[0] < self.low_bound:
                    print(f"Skipping small cluster for gene {j}, cluster {k} (n = {coords_unique.shape[0]})")
                    continue
                
                # compute minimum enclosing sphere without singularity issues
                try:
                    center, r2 = miniball.get_bounding_ball(coords_unique, epsilon=1e-8)
                except np.linalg.LinAlgError:
                    print(f"Warning: singular matrix for gene {j}, cluster {k} —- using fallback sphere.")
                    center = coords_unique.mean(axis=0)
                    dists = np.linalg.norm(coords_unique - center, axis=1)
                    r2 = (dists.max() ** 2)
                radius = np.sqrt(r2)

                # record closest z-layer
                if z_grid is not None:
                    closest_z = z_grid[np.argmin(np.abs(z_grid - center[2]))]
                else:
                    closest_z = center[2]
                
                # record cell id after filtering
                if record_cell_id:
                    temp_target = target[labels == k]
                    temp_cell_id_mode = temp_target["cell_id"].mode()[0]
                    cell_id.append(temp_cell_id_mode)

                # ---------- compute sphere features (size, composition, and in-soma ratio) ---------- #
                temp_in_soma = np.sum(target["overlaps_nucleus"].values[mask])
                temp_size = coords.shape[0]
                other_idx = tree.query_ball_point([center[0], center[1], center[2]], radius)
                other_trans = others.iloc[other_idx]
                other_in_soma = np.sum(other_trans["overlaps_nucleus"])
                other_size = other_trans.shape[0]
                other_comp = len(other_trans["target"].unique())
                total_size = temp_size + other_size
                total_comp = 1 + other_comp
                in_soma_score = (temp_in_soma + other_in_soma) / total_size
                
                # record sphere features
                sphere_x.append(center[0])
                sphere_y.append(center[1])
                sphere_z.append(center[2])
                layer_z.append(closest_z)
                sphere_r.append(radius)
                sphere_size.append(total_size)
                sphere_comp.append(total_comp)
                sphere_score.append(in_soma_score)
            
            # basic features for all spheres from each granule marker
            sphere = pd.DataFrame(list(zip(sphere_x, sphere_y, sphere_z, layer_z, sphere_r, sphere_size, sphere_comp, sphere_score, [j] * len(sphere_x))),
                                      columns = ["sphere_x", "sphere_y", "sphere_z", "layer_z", "sphere_r", "size", "comp", "in_soma_ratio", "gene"])
            sphere = sphere.astype({"sphere_x": float, "sphere_y": float, "sphere_z": float, "layer_z": float, "sphere_r": float, "size": float, "comp": float, "in_soma_ratio": float, "gene": str})
            if record_cell_id:
                sphere["cell_id"] = cell_id
                sphere = sphere.astype({"cell_id": str})
            
            # size and in-soma ratio filtering
            sphere = sphere[(sphere["sphere_r"] < self.size_thr) & (sphere["in_soma_ratio"] < self.in_soma_thr)]
            sphere = sphere.reset_index(drop = True)
            sphere_dict[j_idx] = sphere
            
            print(f"{j_idx + 1} out of {len(target_names)} genes processed!")
        
        return sphere_dict
    
    
    # [INNER] ensure precomputed tree for all granule marker genes
    def _ensure_gnl_tree(self):
        if (not hasattr(self, "_gnl_transcripts")) or (not hasattr(self, "_gnl_tree")):
            self._gnl_transcripts = self.transcripts[self.transcripts["target"].isin(self.gnl_genes)].reset_index(drop=True)
            self._gnl_tree = make_tree(d1 = self._gnl_transcripts["global_x"].to_numpy(), d2 = self._gnl_transcripts["global_y"].to_numpy(), d3 = self._gnl_transcripts["global_z"].to_numpy())
    
    
    # [INNER] merge points from two overlapped spheres, input for _remove_overlaps(), use precomputed tree for all granule marker genes
    def _find_points(self, sphere_a, sphere_b):
        
        transcripts = self._gnl_transcripts
        tree = self._gnl_tree

        idx_a = tree.query_ball_point([sphere_a["sphere_x"], sphere_a["sphere_y"], sphere_a["sphere_z"]], sphere_a["sphere_r"])
        points_a = transcripts.iloc[idx_a]
        points_a = points_a[points_a["target"] == sphere_a["gene"]]

        idx_b = tree.query_ball_point([sphere_b["sphere_x"], sphere_b["sphere_y"], sphere_b["sphere_z"]], sphere_b["sphere_r"])
        points_b = transcripts.iloc[idx_b]
        points_b = points_b[points_b["target"] == sphere_b["gene"]]

        points = pd.concat([points_a, points_b])[["global_x", "global_y", "global_z"]]
        return points.to_numpy()
    
    
    # [INNER] merge spheres from different granule markers, input for detect()
    def _remove_overlaps(self, set_a, set_b):
        
        set_a = set_a.copy()
        set_b = set_b.copy()
        z_grid = self.z_grid_np if self.type == "discrete" else None

        # find possible overlaps on 2D by r-tree
        idx_b = make_rtree(set_b)
        for i, sphere_a in set_a.iterrows():
            center_a_3D = np.array([sphere_a.sphere_x, sphere_a.sphere_y, sphere_a.sphere_z])
            bounds_a = (sphere_a.sphere_x - sphere_a.sphere_r,
                        sphere_a.sphere_y - sphere_a.sphere_r,
                        sphere_a.sphere_x + sphere_a.sphere_r,
                        sphere_a.sphere_y + sphere_a.sphere_r)
            possible_overlaps = idx_b.intersection(bounds_a)

            # search 3D overlaps within possible overlaps
            for j in possible_overlaps:
                if j in set_b.index:
                    sphere_b = set_b.loc[j]
                    center_b_3D = np.array([sphere_b.sphere_x, sphere_b.sphere_y, sphere_b.sphere_z])
                    dist = np.linalg.norm(center_a_3D - center_b_3D)
                    radius_sum = sphere_a.sphere_r + sphere_b.sphere_r
                    radius_diff = sphere_a.sphere_r - sphere_b.sphere_r

                    # relative positions (0: internal & intersect, 1: internal, 2: intersect)
                    c0 = (dist < self.l * radius_sum)
                    c1 = (dist <= self.l * np.abs(radius_diff))
                    c1_1 = (radius_diff > 0)
                    c2_1 = (dist < self.rho * self.l * radius_sum)

                    # operations on dataframes
                    if c0:
                        if c1 and c1_1:                             # keep A and remove B
                            set_b.drop(index = j, inplace = True)
                        elif c1 and not c1_1:                       # replace A with B and remove B
                            set_a.loc[i] = set_b.loc[j]
                            set_b.drop(index = j, inplace = True)
                        elif not c1 and c2_1:                       # replace A with new sphere and remove B
                            points_union = self._find_points(sphere_a, sphere_b)
                            new_center, new_r2 = miniball.get_bounding_ball(points_union, epsilon=1e-8)
                            new_radius = np.sqrt(new_r2)
                            set_a.loc[i, "sphere_x"] = new_center[0]
                            set_a.loc[i, "sphere_y"] = new_center[1]
                            set_a.loc[i, "sphere_z"] = new_center[2]
                            if z_grid is not None:
                                set_a.loc[i, "layer_z"] = z_grid[np.argmin(np.abs(z_grid - new_center[2]))]
                            else:
                                set_a.loc[i, "layer_z"] = new_center[2]
                            set_a.loc[i, "sphere_r"] = self.s * new_radius
                            set_b.drop(index = j, inplace = True)
        
        set_a = set_a.reset_index(drop = True)
        set_b = set_b.reset_index(drop = True)
        return set_a, set_b
    
    
    # [INNER] merge spheres from different granule markers, input for detect()
    def merge_sphere(self, sphere_dict):
        self._ensure_gnl_tree()
        sphere = sphere_dict[0].copy()
        for j in range(1, len(sphere_dict)):
            target_sphere = sphere_dict[j]
            sphere, target_sphere_new = self._remove_overlaps(sphere, target_sphere)
            sphere = pd.concat([sphere, target_sphere_new])
            sphere = sphere.reset_index(drop = True)
        return sphere
    
    
    # [INNER] negative control filtering, input for detect()
    def nc_filter(self, sphere):
        
        # top nc_top negative control genes by expression level
        nc_transcripts = self.transcripts[self.transcripts["target"].isin(self.nc_genes)]
        nc_counts = nc_transcripts["target"].value_counts()
        nc_genes = list(nc_counts.index[:self.nc_top])
        
        # negative control filtering
        nc_transcripts_final = nc_transcripts[nc_transcripts["target"].isin(nc_genes)]
        tree = make_tree(d1 = np.array(nc_transcripts_final["global_x"]), d2 = np.array(nc_transcripts_final["global_y"]), d3 = np.array(nc_transcripts_final["global_z"]))
        centers = sphere[["sphere_x", "sphere_y", "layer_z"]].to_numpy()
        radii = sphere["sphere_r"].to_numpy()
        sizes = sphere["size"].to_numpy()
        counts = np.array([len(tree.query_ball_point(c, r)) for c, r in zip(centers, radii)])
        nc_ratio = counts / sizes
        sphere = sphere.copy().reset_index(drop=True)
        sphere["nc_ratio"] = nc_ratio
        if self.nc_thr is None:
            return sphere
        pass_idx = (counts == 0) | (nc_ratio < self.nc_thr)
        return sphere.loc[pass_idx].reset_index(drop=True)
    
    
    # [MAIN] dataframe, granule metadata
    def detect(self, record_cell_id = False):

        sphere_dict = self.dbscan(record_cell_id = record_cell_id)

        print("Merging spheres...")
        sphere = self.merge_sphere(sphere_dict)
        
        if self.nc_genes is None:
            return sphere
        else:
            print("Negative control filtering...")
            return self.nc_filter(sphere)
    
    
    # [MAIN] anndata, granule spatial transcriptome profile
    def profile(self, granule, genes = None, buffer = 0.0, print_itr = False, print_itr_interval = 5000):
        
        if genes is None:
            genes = list(self.transcripts["target"].unique())
            transcripts = self.transcripts
        else:
            transcripts = self.transcripts[self.transcripts["target"].isin(genes)]
        
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        gene_array = transcripts["target"].to_numpy()
        tree = make_tree(d1 = np.array(transcripts["global_x"]), d2 = np.array(transcripts["global_y"]), d3 = np.array(transcripts["global_z"]))
        
        n_gnl = granule.shape[0]
        n_gene = len(genes)
        data, row_idx, col_idx = [], [], []
        
        # iterate over all granules to count nearby transcripts
        for i in range(n_gnl):
            temp = granule.iloc[i]
            target_idx = tree.query_ball_point([temp["sphere_x"], temp["sphere_y"], temp["layer_z"]], temp["sphere_r"] + buffer)
            if not target_idx:
                continue
            local_genes = gene_array[target_idx]    # extract genes for those nearby transcripts
            counts = Counter(local_genes)      # count how many times each gene occurs
            for g, cnt in counts.items():           # append nonzero entries to sparse matrix lists
                j = gene_to_idx[g]                  # get gene column index
                data.append(cnt)                    # nonzero count
                row_idx.append(i)                   # row index = granule index
                col_idx.append(j)                   # column index = gene index
            if print_itr and (i % print_itr_interval == 0):
                print(f"{i} out of {n_gnl} granules profiled!")
        
        # construct sparse spatial transcriptome profile, (n_granules × n_genes)
        X = csr_matrix((data, (row_idx, col_idx)), shape = (n_gnl, n_gene), dtype = np.float32)
        adata = anndata.AnnData(X = X, obs = granule.copy())
        adata.obs["granule_id"] = [f"gnl_{i}" for i in range(n_gnl)]
        adata.obs = adata.obs.astype({"granule_id": str})
        adata.obs.rename(columns = {"sphere_x": "global_x", "sphere_y": "global_y", "sphere_z": "global_z"}, inplace = True)
        adata.var["genes"] = genes
        adata.var_names = genes
        adata.var_keys = genes
        return adata
    
    
    # [MAIN] anndata, spot-level gene expression
    def spot_expression(self, grid_len, genes = None, print_itr = False, print_itr_interval = 100):
        
        if genes is None:
            genes = list(self.transcripts["target"].unique())
            transcripts = self.transcripts
        else:
            transcripts = self.transcripts[self.transcripts["target"].isin(genes)]
        
        # construct bins
        x_bins, y_bins = self.construct_grid(grid_len = grid_len)
        
        # initialize data
        X = np.zeros((len(genes), (len(x_bins) - 1) * (len(y_bins) - 1)))
        global_x, global_y = [], []
        
        # coordinates
        for i in list(x_bins)[:-1]:
            center_x = i + 0.5 * grid_len
            for j in list(y_bins)[:-1]:
                center_y = j + 0.5 * grid_len
                global_x.append(center_x)
                global_y.append(center_y)
        
        # count matrix
        for k_idx, k in enumerate(genes):
            target_gene = transcripts[transcripts["target"] == k]
            count_gene, _, _ = np.histogram2d(target_gene["global_x"], target_gene["global_y"], bins = [x_bins, y_bins])
            X[k_idx, :] = count_gene.flatten()
            if print_itr and (k_idx % print_itr_interval == 0):
                print(f"{k_idx} out of {len(genes)} genes profiled!")
        
        # spot id
        spot_id = []
        for i in range(len(global_x)):
            id = "spot_" + str(i)
            spot_id.append(id)
        
        # assemble data
        adata = anndata.AnnData(X = np.transpose(X))
        adata.obs["spot_id"] = spot_id
        adata.obs["global_x"] = global_x
        adata.obs["global_y"] = global_y
        adata.var["genes"] = genes
        adata.var_names = genes
        adata.var_keys = genes
        return adata