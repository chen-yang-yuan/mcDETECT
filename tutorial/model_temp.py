import anndata
import math
import miniball
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import ConvexHull
from scipy.stats import poisson
from sklearn.cluster import DBSCAN
from closest import closest
from make_tree import make_tree
from make_rtree import make_rtree


class mcDETECT:
    
    
    def __init__(self, type, transcripts, syn_genes, nc_genes = None, eps = 1.5, minspl = None, grid_len = 1.0, cutoff_prob = 0.95, alpha = 5.0, low_bound = 3,
                 size_thr = 4.0, in_nucleus_thr = (0.5, 0.5), l = 1.0, rho = 0.2, s = 1.0, nc_top = 20, nc_thr = 0.1):
        
        self.type = type                        # string, iST platform, now support MERSCOPE, Xenium, and CosMx
        self.transcripts = transcripts          # dataframe, transcripts file
        self.syn_genes = syn_genes              # list, string, all synaptic markers
        self.nc_genes = nc_genes                # list, string, all negative controls
        self.eps = eps                          # numeric, searching radius epsilon
        self.minspl = minspl                    # integer, manually select min_samples, i.e., no automatic parameter selection
        self.grid_len = grid_len                # numeric, length of grids
        self.cutoff_prob = cutoff_prob          # numeric, cutoff probability in parameter selection for min_samples
        self.alpha = alpha                      # numeric, scaling factor in parameter selection for min_samples
        self.low_bound = low_bound              # integer, lower bound in parameter selection for min_samples
        self.size_thr = size_thr                # numeric, threshold for maximum radius of an aggregation
        self.in_nucleus_thr = in_nucleus_thr    # 2-d tuple, threshold for low- and high-in-nucleus ratio
        self.l = l                              # numeric, scaling factor for seaching overlapped spheres
        self.rho = rho                          # numeric, threshold for determining overlaps
        self.s = s                              # numeric, scaling factor for merging overlapped spheres
        self.nc_top = nc_top                    # integer, number of negative controls retained for filtering
        self.nc_thr = nc_thr                    # numeric, threshold for negative control filtering
    
    
    # [INNER] calculate tissue area, input for poisson_select()
    def effect_area(self):
        
        # construct grids
        x_min, x_max = np.min(self.transcripts['global_x']), np.max(self.transcripts['global_x'])
        y_min, y_max = np.min(self.transcripts['global_y']), np.max(self.transcripts['global_y'])
        x_min = np.floor(x_min / self.grid_len) * self.grid_len
        x_max = np.ceil(x_max / self.grid_len) * self.grid_len
        y_min = np.floor(y_min / self.grid_len) * self.grid_len
        y_max = np.ceil(y_max / self.grid_len) * self.grid_len

        # compute number of bins
        x_bins = np.arange(x_min, x_max + self.grid_len, self.grid_len)
        y_bins = np.arange(y_min, y_max + self.grid_len, self.grid_len)

        # count transcripts in each grid cell
        hist, _, _ = np.histogram2d(self.transcripts['global_x'], self.transcripts['global_y'], bins=[x_bins, y_bins])

        # count nonzero grids and compute effective area
        effective_area = np.count_nonzero(hist) * (self.grid_len ** 2)
        
        return effective_area
    
    
    # [INNER] calculate optimal min_samples, input for dbscan()
    def poisson_select(self, gene_name):
        num_trans = np.sum(self.transcripts['target'] == gene_name)
        bg_density = num_trans / self.effect_area()
        cutoff_density = poisson.ppf(self.cutoff_prob, mu = self.alpha * bg_density * (np.pi * self.eps ** 2))
        optimal_m = int(max(cutoff_density, self.low_bound))
        return optimal_m
    
    
    # [INTERMEDIATE] dictionary, low- and high-in-nucleus spheres for each synaptic marker
    def dbscan(self, target_names = None, write_csv = False, write_path = './'):
        
        if self.type != 'Xenium':
            z_grid = list(np.unique(self.transcripts['global_z']))
            z_grid.sort()
        
        if target_names is None:
            target_names = self.syn_genes
        transcripts = self.transcripts[self.transcripts['target'].isin(target_names)]
        
        num_individual, data_low, data_high = [], {}, {}
        
        for j in target_names:
            
            # split transcripts
            target = transcripts[transcripts['target'] == j]
            others = transcripts[transcripts['target'] != j]
            tree = make_tree(d1 = np.array(others['global_x']), d2 = np.array(others['global_y']), d3 = np.array(others['global_z']))
            
            # 3D DBSCAN
            if self.minspl is None:
                min_spl = self.poisson_select(j)
            else:
                min_spl = self.minspl
            X = np.array(target[['global_x', 'global_y', 'global_z']])
            db = DBSCAN(eps = self.eps, min_samples = min_spl, algorithm = 'kd_tree').fit(X)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # iterate over all aggregations
            sphere_x, sphere_y, sphere_z, layer_z, sphere_r, sphere_size, sphere_comp, sphere_score = [], [], [], [], [], [], [], []
            
            for k in range(n_clusters):
                
                # find minimum enclosing spheres
                temp = target[labels == k]
                temp_in_nucleus = np.sum(temp['CellComp'] == 'Nuclear')
                temp_size = temp.shape[0]
                temp = temp[['global_x', 'global_y', 'global_z']]
                temp = temp.drop_duplicates()
                center, r2 = miniball.get_bounding_ball(np.array(temp), epsilon=1e-8)
                if self.type != 'Xenium':
                    closest_z = closest(z_grid, center[2])
                else:
                    closest_z = center[2]
                
                # calculate size, composition, and in-nucleus score
                other_idx = tree.query_ball_point([center[0], center[1], center[2]], np.sqrt(r2))
                other_trans = others.iloc[other_idx]
                other_in_nucleus = np.sum(other_trans['CellComp'] == 'Nuclear')
                other_size = other_trans.shape[0]
                other_comp = len(np.unique(other_trans['target']))
                total_size = temp_size + other_size
                total_comp = 1 + other_comp
                local_score = (temp_in_nucleus + other_in_nucleus) / total_size
                
                # record coordinate, radius, size, composition, and in-nucleus score
                sphere_x.append(center[0])
                sphere_y.append(center[1])
                sphere_z.append(center[2])
                layer_z.append(closest_z)
                sphere_r.append(np.sqrt(r2))
                sphere_size.append(total_size)
                sphere_comp.append(total_comp)
                sphere_score.append(local_score)

            # basic features for all spheres from each synaptic marker
            sphere = pd.DataFrame(list(zip(sphere_x, sphere_y, sphere_z, layer_z, sphere_r, sphere_size, sphere_comp, sphere_score)),
                                  columns = ['sphere_x', 'sphere_y', 'sphere_z', 'layer_z', 'sphere_r', 'size', 'comp', 'in_nucleus'])
            sphere['gene'] = [j] * sphere.shape[0]
            
            # split low- and high-in-nucleus spheres
            sphere_low = sphere[(sphere['sphere_r'] < self.size_thr) & (sphere['in_nucleus'] < self.in_nucleus_thr[0])]
            sphere_high = sphere[(sphere['sphere_r'] < self.size_thr) & (sphere['in_nucleus'] > self.in_nucleus_thr[1])]
            
            if write_csv:
                sphere_low.to_csv(write_path + j + ' sphere.csv', index=0)
                sphere_high.to_csv(write_path + j + ' sphere_high.csv', index=0)
            
            num_individual.append(sphere_low.shape[0])
            data_low[target_names.index(j)] = sphere_low
            data_high[target_names.index(j)] = sphere_high
            print("{} out of {} genes processed!".format(target_names.index(j) + 1, len(target_names)))
        
        return np.sum(num_individual), data_low, data_high
    
    
    # [INNER] merge points from two overlapped spheres, input for remove_overlaps()
    def find_points(self, sphere_a, sphere_b):
        transcripts = self.transcripts[self.transcripts['target'].isin(self.syn_genes)]
        tree_temp = make_tree(d1 = np.array(transcripts['global_x']), d2 = np.array(transcripts['global_y']), d3 = np.array(transcripts['global_z']))
        idx_a = tree_temp.query_ball_point([sphere_a['sphere_x'], sphere_a['sphere_y'], sphere_a['sphere_z']], sphere_a['sphere_r'])
        points_a = transcripts.iloc[idx_a]
        points_a = points_a[points_a['target'] == sphere_a['gene']]
        idx_b = tree_temp.query_ball_point([sphere_b['sphere_x'], sphere_b['sphere_y'], sphere_b['sphere_z']], sphere_b['sphere_r'])
        points_b = transcripts.iloc[idx_b]
        points_b = points_b[points_b['target'] == sphere_b['gene']]
        points = pd.concat([points_a, points_b])
        points = points[['global_x', 'global_y', 'global_z']]
        return points

    
    def remove_overlaps(self, set_a, set_b):
        
        set_a = set_a.copy()
        set_b = set_b.copy()

        # find possible overlaps on 2D by r-tree
        idx_b = make_rtree(set_b)
        for i, sphere_a in set_a.iterrows():
            center_a_3D = (sphere_a.sphere_x, sphere_a.sphere_y, sphere_a.sphere_z)
            bounds_a = (
                sphere_a.sphere_x - sphere_a.sphere_r,
                sphere_a.sphere_y - sphere_a.sphere_r,
                sphere_a.sphere_x + sphere_a.sphere_r,
                sphere_a.sphere_y + sphere_a.sphere_r
            )
            possible_overlaps = idx_b.intersection(bounds_a)

            # search 3D overlaps within possible overlaps
            for j in possible_overlaps:
                if j in set_b.index:
                    sphere_b = set_b.loc[j]
                    center_b_3D = (sphere_b.sphere_x, sphere_b.sphere_y, sphere_b.sphere_z)
                    dist = math.dist(center_a_3D, center_b_3D)
                    radius_sum = sphere_a.sphere_r + sphere_b.sphere_r
                    radius_diff = sphere_a.sphere_r - sphere_b.sphere_r

                    # relative positions (0: internal & intersect, 1: internal, 2: intersect)
                    c0 = (dist < self.l * radius_sum)
                    c1 = (dist <= self.l * np.abs(radius_diff))
                    c1_1 = (radius_diff > 0)
                    c2_1 = (dist < self.rho * self.l * radius_sum)

                    # operations on dataframes
                    if c0:
                        if c1 and c1_1:         # keep A and remove B
                            set_b.drop(index = j, inplace = True)
                        elif c1 and not c1_1:   # replace A with B and remove B
                            set_a.loc[i] = set_b.loc[j]
                            set_b.drop(index = j, inplace = True)
                        elif not c1 and c2_1:   # replace A with new sphere and remove B
                            points_union = np.array(self.find_points(sphere_a, sphere_b))
                            new_center, new_radius = miniball.get_bounding_ball(points_union, epsilon=1e-8)
                            set_a.loc[i, 'sphere_x'] = new_center[0]
                            set_a.loc[i, 'sphere_y'] = new_center[1]
                            set_a.loc[i, 'sphere_z'] = new_center[2]
                            set_a.loc[i, 'sphere_r'] = self.s * new_radius
                            set_b.drop(index = j, inplace = True)

        # return output
        set_a = set_a.reset_index(drop = True)
        set_b = set_b.reset_index(drop = True)
        return set_a, set_b
    
    
    # [INNER] merge spheres from different synaptic markers, input for detect()
    def merge_sphere(self, sphere_dict):
        sphere = sphere_dict[0].copy()
        for j in range(1, len(self.syn_genes)):
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
        adata.var['genes'] = adata.var.index
        adata.obs_keys = list(np.arange(adata.shape[0]))
        adata.obs['type'] = ['low'] * adata_low.shape[0] + ['high'] * adata_high.shape[0]
        adata.obs['type'] = pd.Categorical(adata.obs['type'], categories = ["low", "high"], ordered = True)
        
        # DE analysis of negative control genes
        sc.tl.rank_genes_groups(adata, 'type', method = 't-test')
        names = adata.uns['rank_genes_groups']['names']
        names = pd.DataFrame(names)
        logfc = adata.uns['rank_genes_groups']['logfoldchanges']
        logfc = pd.DataFrame(logfc)
        pvals = adata.uns['rank_genes_groups']['pvals']
        pvals = pd.DataFrame(pvals)

        # select top upregulated negative control genes
        df = pd.DataFrame({'names': names["high"], 'logfc': logfc["high"], 'pvals': pvals["high"]})
        df = df[df['logfc'] >= 0]
        df = df.sort_values(by = ['pvals'], ascending = True)
        nc_genes_final = list(df['names'].head(self.nc_top))
        
        # negative control filtering
        nc_transcripts_final = self.transcripts[self.transcripts['target'].isin(nc_genes_final)]
        tree = make_tree(d1 = np.array(nc_transcripts_final['global_x']), d2 = np.array(nc_transcripts_final['global_y']), d3 = np.array(nc_transcripts_final['global_z']))
        pass_idx = [0] * sphere_low.shape[0]
        for i in range(sphere_low.shape[0]):
            temp = sphere_low.iloc[i]
            nc_idx = tree.query_ball_point([temp['sphere_x'], temp['sphere_y'], temp['sphere_z']], temp['sphere_r'])
            if len(nc_idx) == 0:
                pass_idx[i] = 1
            elif len(nc_idx) / temp['size'] < self.nc_thr:
                pass_idx[i] = 2
        sphere = sphere_low[np.array(pass_idx) != 0]
        return sphere
    
    
    # [MAIN] dataframe
    def detect(self):
        
        # multi-channel DBSCAN
        _, data_low, data_high = self.dbscan()
        
        # merge spheres
        print("Merging spheres...")
        sphere_low, sphere_high = self.merge_sphere(data_low), self.merge_sphere(data_high)
        
        if self.nc_genes is None:
            return sphere_low
        else:
            print("Negative control filtering...")
            return self.nc_filter(sphere_low, sphere_high)
    
    
    # [MAIN] anndata, spatial transcriptome profile of identified synapses
    def profile(self, synapse, genes = None, print_itr = False):
        
        if genes is None:
            genes = list(np.unique(self.transcripts['target']))
            transcripts = self.transcripts
        else:
            transcripts = self.transcripts[self.transcripts['target'].isin(genes)]
        tree = make_tree(d1 = np.array(transcripts['global_x']), d2 = np.array(transcripts['global_y']), d3 = np.array(transcripts['global_z']))
        
        # construct gene count matrix
        X = np.zeros((len(genes), synapse.shape[0]))
        for i in range(synapse.shape[0]):
            temp = synapse.iloc[i]
            target_idx = tree.query_ball_point([temp['sphere_x'], temp['sphere_y'], temp['layer_z']], temp['sphere_r'])
            target_trans = transcripts.iloc[target_idx]
            target_gene = list(target_trans['target'])
            for j in np.unique(target_gene):
                X[genes.index(j), i] = target_gene.count(j)
            if (print_itr) & (i % 5000 == 0):
                print('{} out of {} synapses profiled!'.format(i, synapse.shape[0]))
        
        # construct spatial transcriptome profile
        adata = anndata.AnnData(X = np.transpose(X), obs = synapse)
        adata.obs['synapse_id'] = ['syn_{}'.format(i) for i in range(synapse.shape[0])]
        adata.obs.rename(columns = {'sphere_x': 'global_x', 'sphere_y': 'global_y', 'sphere_z': 'global_z'}, inplace = True)
        adata.var['genes'] = genes
        adata.var_names = genes
        adata.var_keys = genes
        return adata