import anndata
import math
import miniball
import numpy as np
import pandas as pd
import scanpy as sc
from collections import Counter
from rtree import index
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import poisson
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OneHotEncoder


from .utils import *


class mcDETECT:
    
    
    def __init__(self, type, transcripts, gnl_genes, nc_genes = None, eps = 1.5, minspl = None, grid_len = 1.0, cutoff_prob = 0.95, alpha = 5.0, low_bound = 3,
                 size_thr = 4.0, in_soma_thr = (0.5, 0.5), l = 1.0, rho = 0.2, s = 1.0, nc_top = 20, nc_thr = 0.1):
        
        self.type = type                        # string, iST platform, now support MERSCOPE, Xenium, and CosMx
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
        self.in_soma_thr = in_soma_thr          # 2-d tuple, threshold for low- and high-in-soma ratio
        self.l = l                              # numeric, scaling factor for seaching overlapped spheres
        self.rho = rho                          # numeric, threshold for determining overlaps
        self.s = s                              # numeric, scaling factor for merging overlapped spheres
        self.nc_top = nc_top                    # integer, number of negative controls retained for filtering
        self.nc_thr = nc_thr                    # numeric, threshold for negative control filtering
    
    
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
    
    
    # [INTERMEDIATE] dictionary, low- and high-in-soma spheres for each granule marker
    def dbscan(self, target_names = None, record_cell_id = False, write_csv = False, write_path = "./"):
        
        if self.type != "Xenium":
            z_grid = list(self.transcripts["global_z"].unique())
            z_grid.sort()
        
        if target_names is None:
            target_names = self.gnl_genes
        transcripts = self.transcripts[self.transcripts["target"].isin(target_names)]
        
        num_individual, data_low, data_high = [], {}, {}
        
        for j in target_names:
            
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

                # record closest z-layer
                if self.type != "Xenium":
                    closest_z = closest(z_grid, center[2])
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
                other_idx = tree.query_ball_point([center[0], center[1], center[2]], np.sqrt(r2))
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
                sphere_r.append(np.sqrt(r2))
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
            
            # split low- and high-in-soma spheres
            sphere_low = sphere[(sphere["sphere_r"] < self.size_thr) & (sphere["in_soma_ratio"] < self.in_soma_thr[0])]
            sphere_high = sphere[(sphere["sphere_r"] < self.size_thr) & (sphere["in_soma_ratio"] > self.in_soma_thr[1])]
            
            if write_csv:
                sphere_low.to_csv(write_path + j + " sphere.csv", index=0)
                sphere_high.to_csv(write_path + j + " sphere_high.csv", index=0)
            
            num_individual.append(sphere_low.shape[0])
            data_low[target_names.index(j)] = sphere_low
            data_high[target_names.index(j)] = sphere_high
            print("{} out of {} genes processed!".format(target_names.index(j) + 1, len(target_names)))
        
        return np.sum(num_individual), data_low, data_high
    
    
    # [INNER] merge points from two overlapped spheres, input for remove_overlaps()
    def find_points(self, sphere_a, sphere_b):
        transcripts = self.transcripts[self.transcripts["target"].isin(self.gnl_genes)]
        tree_temp = make_tree(d1 = np.array(transcripts["global_x"]), d2 = np.array(transcripts["global_y"]), d3 = np.array(transcripts["global_z"]))
        idx_a = tree_temp.query_ball_point([sphere_a["sphere_x"], sphere_a["sphere_y"], sphere_a["sphere_z"]], sphere_a["sphere_r"])
        points_a = transcripts.iloc[idx_a]
        points_a = points_a[points_a["target"] == sphere_a["gene"]]
        idx_b = tree_temp.query_ball_point([sphere_b["sphere_x"], sphere_b["sphere_y"], sphere_b["sphere_z"]], sphere_b["sphere_r"])
        points_b = transcripts.iloc[idx_b]
        points_b = points_b[points_b["target"] == sphere_b["gene"]]
        points = pd.concat([points_a, points_b])
        points = points[["global_x", "global_y", "global_z"]]
        return points
    
    
    def remove_overlaps(self, set_a, set_b):
        
        set_a = set_a.copy()
        set_b = set_b.copy()

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
                            points_union = np.array(self.find_points(sphere_a, sphere_b))
                            new_center, new_radius = miniball.get_bounding_ball(points_union, epsilon=1e-8)
                            set_a.loc[i, "sphere_x"] = new_center[0]
                            set_a.loc[i, "sphere_y"] = new_center[1]
                            set_a.loc[i, "sphere_z"] = new_center[2]
                            set_a.loc[i, "sphere_r"] = self.s * new_radius
                            set_b.drop(index = j, inplace = True)
        
        set_a = set_a.reset_index(drop = True)
        set_b = set_b.reset_index(drop = True)
        return set_a, set_b
    
    
    # [INNER] merge spheres from different granule markers, input for detect()
    def merge_sphere(self, sphere_dict):
        sphere = sphere_dict[0].copy()
        for j in range(1, len(self.gnl_genes)):
            target_sphere = sphere_dict[j]
            sphere, target_sphere_new = self.remove_overlaps(sphere, target_sphere)
            sphere = pd.concat([sphere, target_sphere_new])
            sphere = sphere.reset_index(drop = True)
        return sphere
    
    
    # [INNER] negative control filtering, input for detect()
    def nc_filter(self, sphere_low, sphere_high):
        
        # negative control gene profiling
        adata_low = self.profile(sphere_low, self.nc_genes)
        adata_high = self.profile(sphere_high, self.nc_genes)
        adata = anndata.concat([adata_low, adata_high], axis = 0, merge = "same")
        adata.var["genes"] = adata.var.index
        adata.obs_keys = list(np.arange(adata.shape[0]))
        adata.obs["type"] = ["low"] * adata_low.shape[0] + ["high"] * adata_high.shape[0]
        adata.obs["type"] = pd.Categorical(adata.obs["type"], categories = ["low", "high"], ordered = True)
        
        # DE analysis of negative control genes
        sc.tl.rank_genes_groups(adata, "type", method = "t-test")
        names = adata.uns["rank_genes_groups"]["names"]
        names = pd.DataFrame(names)
        logfc = adata.uns["rank_genes_groups"]["logfoldchanges"]
        logfc = pd.DataFrame(logfc)
        pvals = adata.uns["rank_genes_groups"]["pvals"]
        pvals = pd.DataFrame(pvals)

        # select top upregulated negative control genes
        df = pd.DataFrame({"names": names["high"], "logfc": logfc["high"], "pvals": pvals["high"]})
        df = df[df["logfc"] >= 0]
        df = df.sort_values(by = ["pvals"], ascending = True)
        nc_genes_final = list(df["names"].head(self.nc_top))
        
        # negative control filtering
        nc_transcripts_final = self.transcripts[self.transcripts["target"].isin(nc_genes_final)]
        tree = make_tree(d1 = np.array(nc_transcripts_final["global_x"]), d2 = np.array(nc_transcripts_final["global_y"]), d3 = np.array(nc_transcripts_final["global_z"]))
        centers = sphere_low[["sphere_x", "sphere_y", "sphere_z"]].to_numpy()
        radii = sphere_low["sphere_r"].to_numpy()
        sizes = sphere_low["size"].to_numpy()
        counts = np.array([len(tree.query_ball_point(c, r)) for c, r in zip(centers, radii)])
        nc_ratio = counts / sizes
        pass_idx = (counts == 0) | (nc_ratio < self.nc_thr)
        sphere = sphere_low[pass_idx].reset_index(drop = True)
        sphere["nc_ratio"] = nc_ratio[pass_idx]
        return sphere
    
    
    # [MAIN] dataframe, granule metadata
    def detect(self, record_cell_id = False):

        _, data_low, data_high = self.dbscan(record_cell_id = record_cell_id)

        print("Merging spheres...")
        sphere_low, sphere_high = self.merge_sphere(data_low), self.merge_sphere(data_high)
        
        if self.nc_genes is None:
            return sphere_low
        else:
            print("Negative control filtering...")
            return self.nc_filter(sphere_low, sphere_high)
    
    
    # [MAIN] anndata, granule spatial transcriptome profile
    def profile(self, granule, genes = None, buffer = 0.0, print_itr = False):
        
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
            counts = Counter(local_genes)           # count how many times each gene occurs
            for g, cnt in counts.items():           # append nonzero entries to sparse matrix lists
                j = gene_to_idx[g]                  # get gene column index
                data.append(cnt)                    # nonzero count
                row_idx.append(i)                   # row index = granule index
                col_idx.append(j)                   # column index = gene index
            if print_itr and (i % 5000 == 0):
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
    def spot_expression(self, grid_len, genes = None):
        
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
            if k_idx % 100 == 0:
                print("{} out of {} genes profiled!".format(k_idx, len(genes)))
        
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


# [MAIN] anndata, spot-level neuron metadata
def spot_neuron(adata_neuron, spot, grid_len = 50, neuron_loc_key = ["global_x", "global_y"], spot_loc_key = ["global_x", "global_y"]):
    
    adata_neuron = adata_neuron.copy()
    neurons = adata_neuron.obs
    spot = spot.copy()
    
    half_len = grid_len / 2
    
    indicator, neuron_count = [], []
    
    for _, row in spot.obs.iterrows():
        
        x = row[spot_loc_key[0]]
        y = row[spot_loc_key[1]]
        neuron_temp = neurons[(neurons[neuron_loc_key[0]] > x - half_len) & (neurons[neuron_loc_key[0]] < x + half_len) & (neurons[neuron_loc_key[1]] > y - half_len) & (neurons[neuron_loc_key[1]] < y + half_len)]
        indicator.append(int(len(neuron_temp) > 0))
        neuron_count.append(len(neuron_temp))
    
    spot.obs["indicator"] = indicator
    spot.obs["neuron_count"] = neuron_count
    return spot


# [MAIN] anndata, spot-level granule metadata
def spot_granule(granule, spot, grid_len = 50, gnl_loc_key = ["sphere_x", "sphere_y"], spot_loc_key = ["global_x", "global_y"]):
    
    granule = granule.copy()
    spot = spot.copy()
    
    half_len = grid_len / 2

    indicator, granule_count, granule_radius, granule_size, granule_score = [], [], [], [], []
    
    for _, row in spot.obs.iterrows():
        
        x = row[spot_loc_key[0]]
        y = row[spot_loc_key[1]]
        gnl_temp = granule[(granule[gnl_loc_key[0]] >= x - half_len) & (granule[gnl_loc_key[0]] < x + half_len) & (granule[gnl_loc_key[1]] >= y - half_len) & (granule[gnl_loc_key[1]] < y + half_len)]
        indicator.append(int(len(gnl_temp) > 0))
        granule_count.append(len(gnl_temp))

        if len(gnl_temp) == 0:
            granule_radius.append(0)
            granule_size.append(0)
            granule_score.append(0)
        else:
            granule_radius.append(np.nanmean(gnl_temp["sphere_r"]))
            granule_size.append(np.nanmean(gnl_temp["size"]))
            granule_score.append(np.nanmean(gnl_temp["in_soma_ratio"]))
    
    spot.obs["indicator"] = indicator
    spot.obs["gnl_count"] = granule_count
    spot.obs["gnl_radius"] = granule_radius
    spot.obs["gnl_size"] = granule_size
    spot.obs["gnl_score"] = granule_score
    return spot


# [Main] anndata, neuron-granule colocalization
def neighbor_granule(adata_neuron, granule_adata, radius = 10, sigma = None, loc_key = ["global_x", "global_y"]):
    
    adata_neuron = adata_neuron.copy()
    granule_adata = granule_adata.copy()
    
    if sigma is None:
        sigma = radius / 2
    
    # neuron and granule coordinates
    neuron_coords = adata_neuron.obs[loc_key].values
    gnl_coords = granule_adata.obs[loc_key].values
    
    # make tree
    tree = make_tree(d1 = gnl_coords[:, 0], d2 = gnl_coords[:, 1])
    
    # query neighboring granules for each neuron
    neighbor_indices = tree.query_ball_point(neuron_coords, r = radius)
    
    # record count and indices
    granule_counts = np.array([len(indices) for indices in neighbor_indices])
    adata_neuron.obs["neighbor_gnl_count"] = granule_counts
    adata_neuron.uns["neighbor_gnl_indices"] = neighbor_indices
    
    # ---------- neighboring granule expression matrix ---------- #
    n_neurons, n_genes = adata_neuron.n_obs, adata_neuron.n_vars
    weighted_expr = np.zeros((n_neurons, n_genes))
    
    for i, indices in enumerate(neighbor_indices):
        if len(indices) == 0:
            continue
        distances = np.linalg.norm(gnl_coords[indices] - neuron_coords[i], axis = 1)
        weights = np.exp(- (distances ** 2) / (2 * sigma ** 2))
        weights = weights / weights.sum()
        weighted_expr[i] = np.average(granule_adata.X[indices], axis = 0, weights = weights)

    adata_neuron.obsm["weighted_gnl_expression"] = weighted_expr
    
    # ---------- neighboring granule spatial feature ---------- #
    features = []

    for i, gnl_idx in enumerate(neighbor_indices):
        
        feats = {}
        feats["n_granules"] = len(gnl_idx)

        if len(gnl_idx) == 0:
            feats.update({"mean_distance": np.nan, "std_distance": np.nan, "radius_max": np.nan, "radius_min": np.nan, "density": 0, "center_offset_norm": np.nan, "anisotropy_ratio": np.nan})
        else:
            gnl_pos = gnl_coords[gnl_idx]
            neuron_pos = neuron_coords[i]
            dists = np.linalg.norm(gnl_pos - neuron_pos, axis = 1)
            feats["mean_distance"] = dists.mean()
            feats["std_distance"] = dists.std()
            feats["radius_max"] = dists.max()
            feats["radius_min"] = dists.min()
            feats["density"] = len(gnl_idx) / (np.pi * radius ** 2)
            centroid = gnl_pos.mean(axis = 0)
            offset = centroid - neuron_pos
            feats["center_offset_norm"] = np.linalg.norm(offset)
            cov = np.cov((gnl_pos - neuron_pos).T)
            eigvals = np.linalg.eigvalsh(cov)
            if np.min(eigvals) > 0:
                feats["anisotropy_ratio"] = np.max(eigvals) / np.min(eigvals)
            else:
                feats["anisotropy_ratio"] = np.nan

        features.append(feats)
    
    spatial_df = pd.DataFrame(features, index = adata_neuron.obs_names)
    return adata_neuron, spatial_df


# [MAIN] numpy array, neuron embeddings based on neighboring granules
def neuron_embedding_one_hot(adata_neuron, granule_adata, k = 10, radius = 10, loc_key = ["global_x", "global_y"], gnl_subtype_key = "granule_subtype_kmeans", padding_value = "Others"):
    
    adata_neuron = adata_neuron.copy()
    granule_adata = granule_adata.copy()
    
    # neuron and granule coordinates, granule subtypes
    neuron_coords = adata_neuron.obs[loc_key].to_numpy()
    granule_coords = granule_adata.obs[loc_key].to_numpy()
    granule_subtypes = granule_adata.obs[gnl_subtype_key].astype(str).to_numpy()
    
    # include padding category
    unique_subtypes = np.unique(granule_subtypes).tolist()
    if padding_value not in unique_subtypes:
        unique_subtypes.append(padding_value)
    
    encoder = OneHotEncoder(categories = [unique_subtypes], sparse = False, handle_unknown = "ignore")
    encoder.fit(np.array(unique_subtypes).reshape(-1, 1))
    S = len(unique_subtypes)
    
    # k-d tree
    tree = make_tree(d1 = granule_coords[:, 0], d2 = granule_coords[:, 1])
    distances, indices = tree.query(neuron_coords, k = k, distance_upper_bound = radius)
    
    # initialize output
    n_neurons = neuron_coords.shape[0]
    embeddings = np.zeros((n_neurons, k, S), dtype = float)

    for i in range(n_neurons):
        for k in range(k):
            idx = indices[i, k]
            dist = distances[i, k]
            if idx == granule_coords.shape[0] or np.isinf(dist):
                subtype = padding_value
            else:
                subtype = granule_subtypes[idx]
            onehot = encoder.transform([[subtype]])[0]
            embeddings[i, k, :] = onehot

    return embeddings, encoder.categories_[0]


# [MAIN] numpy array, neuron embeddings based on neighboring granules
def neuron_embedding_spatial_weight(adata_neuron, granule_adata, radius = 10, sigma = 10, loc_key = ["global_x", "global_y"], gnl_subtype_key = "granule_subtype_kmeans", padding_value = "Others"):
    
    adata_neuron = adata_neuron.copy()
    granule_adata = granule_adata.copy()
    
    # neuron and granule coordinates, granule subtypes
    neuron_coords = adata_neuron.obs[loc_key].to_numpy()
    granule_coords = granule_adata.obs[loc_key].to_numpy()
    granule_subtypes = granule_adata.obs[gnl_subtype_key].astype(str).to_numpy()
    
    # include padding category
    unique_subtypes = np.unique(granule_subtypes).tolist()
    if padding_value not in unique_subtypes:
        unique_subtypes.append(padding_value)
    
    encoder = OneHotEncoder(categories = [unique_subtypes], sparse = False, handle_unknown = "ignore")
    encoder.fit(np.array(unique_subtypes).reshape(-1, 1))
    S = len(unique_subtypes)
    
    # k-d tree
    tree = make_tree(d1 = granule_coords[:, 0], d2 = granule_coords[:, 1])
    all_neighbors = tree.query_ball_point(neuron_coords, r = radius)
    
    # initialize output
    n_neurons = neuron_coords.shape[0]
    embeddings = np.zeros((n_neurons, S), dtype = float)

    for i, neighbor_indices in enumerate(all_neighbors):
        if not neighbor_indices:
            # no neighbors, assign to padding subtype
            embeddings[i] = encoder.transform([[padding_value]])[0]
            continue

        # get neighbor subtypes and distances
        neighbor_coords = granule_coords[neighbor_indices]
        dists = np.linalg.norm(neuron_coords[i] - neighbor_coords, axis = 1)
        weights = np.exp(- dists / sigma)

        # encode subtypes to one-hot and weight them
        subtypes = granule_subtypes[neighbor_indices]
        onehots = encoder.transform(subtypes.reshape(-1, 1))
        weighted_sum = (weights[:, np.newaxis] * onehots).sum(axis = 0)

        # normalize to make it a composition vector
        embeddings[i] = weighted_sum / weights.sum()

    return embeddings, encoder.categories_[0]