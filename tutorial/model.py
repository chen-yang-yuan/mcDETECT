import math
import miniball
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.cluster import DBSCAN
from closest import closest
from make_tree import make_tree
from make_rtree import make_rtree


class model:
    
    
    def __init__(self, shape, transcripts, target_all, eps = 1.5, cutoff = 0.95, alpha = 5, low = 3, in_thr = 0.5, comp_thr = 2, size_thr = 4, l = 1, p = 0.2, s = 1):
        self.shape = shape
        self.transcripts = transcripts
        self.target_all = target_all
        self.eps = eps
        self.cutoff = cutoff
        self.alpha = alpha
        self.low = low
        self.in_thr = in_thr
        self.comp_thr = comp_thr
        self.size_thr = size_thr
        self.l = l
        self.p = p
        self.s = s
    
    
    def effect_area(self):
        effective_area = self.shape[0] * self.shape[1]
        return effective_area
    
    
    def poisson_select(self, gene_name):
        num_trans = np.sum(self.transcripts['target'] == gene_name)
        bg_density = num_trans / self.effect_area()
        cutoff_density = poisson.ppf(self.cutoff, mu = self.alpha * bg_density * (np.pi * self.eps ** 2))
        optimal_m = int(max(cutoff_density, self.low))
        return optimal_m
    
    
    def dbscan_single(self, target_name, print_itr = False): # target_name is a single string
    
        target = self.transcripts[self.transcripts['target'] == target_name]
        
        min_spl = self.poisson_select(target_name)
        # min_spl = 3
        X = np.array(target[['global_x', 'global_y', 'global_z']])
        db = DBSCAN(eps = self.eps, min_samples = min_spl, algorithm = 'kd_tree').fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # z_list = list(np.unique(target['global_z']))
        # z_list.sort()
        sphere_x, sphere_y, sphere_z, sphere_r = [], [], [], []
        
        for k in range(n_clusters):
    
            temp = target[labels == k]
            temp_size = temp.shape[0]
            in_count = np.sum(temp['in_nucleus'])
            temp = temp[['global_x', 'global_y', 'global_z']]
            center, r2 = miniball.get_bounding_ball(np.array(temp), epsilon=1e-8)
            in_ratio = in_count / temp_size
            # closest_z = closest(z_list, center[2])
            # closest_idx = z_list.index(closest_z)

            if (in_ratio <= self.in_thr) & (np.sqrt(r2) <= self.size_thr):
                sphere_x.append(center[0])
                sphere_y.append(center[1])
                sphere_z.append(center[2])
                sphere_r.append(np.sqrt(r2))

        sphere = pd.DataFrame(list(zip(sphere_x, sphere_y, sphere_z, sphere_r)), columns = ['sphere_x', 'sphere_y', 'sphere_z', 'sphere_r'])
        sphere['gene'] = [target_name] * sphere.shape[0]
        
        if print_itr:
            print('Gene: {}; Transcripts: {}; Min_samples: {}; Filtered aggregations: {}'.format(target_name, target.shape[0], min_spl, sphere.shape[0]))
        
        return sphere # return a data frame
    
    
    def dbscan(self, target_names = None, print_itr = False): # target_names is a list of strings
        
        # iterate through all target genes
        if target_names is None:
            target_names = self.target_all
        transcripts = self.transcripts[self.transcripts['target'].isin(target_names)]
        result_data = {}
        
        for j in target_names:
            
            # split transcripts
            target = transcripts[transcripts['target'] == j]
            others = transcripts[transcripts['target'] != j]
            tree = make_tree(d1 = np.array(others['global_x']), d2 = np.array(others['global_y']), d3 = np.array(others['global_z']))
            
            # 3D DBSCAN
            min_spl = self.poisson_select(j)
            X = np.array(target[['global_x', 'global_y', 'global_z']])
            db = DBSCAN(eps = self.eps, min_samples = min_spl, algorithm = 'kd_tree').fit(X)
            labels = db.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # set up
            # z_list = list(np.unique(target['global_z']))
            # z_list.sort()
            sphere_x, sphere_y, sphere_z, sphere_r, sphere_size, sphere_comp = [], [], [], [], [], []
            
            # minimum enclosing sphere
            for k in range(n_clusters):
        
                # find all spheres
                temp = target[labels == k]
                temp_size = temp.shape[0]
                in_count = np.sum(temp['in_nucleus'])
                temp = temp[['global_x', 'global_y', 'global_z']]
                center, r2 = miniball.get_bounding_ball(np.array(temp), epsilon=1e-8)
                in_ratio = in_count / temp_size
                # closest_z = closest(z_list, center[2])
                # closest_idx = z_list.index(closest_z)
                
                # filter spheres by composition
                other_idx = tree.query_ball_point([center[0], center[1], center[2]], np.sqrt(r2))
                other_trans = others.iloc[other_idx]
                other_size = other_trans.shape[0]
                other_comp = len(np.unique(other_trans['target']))
                total_size = temp_size + other_size
                total_comp = 1 + other_comp
                
                if (in_ratio <= self.in_thr) & (np.sqrt(r2) <= self.size_thr) & (total_comp >= self.comp_thr):
                    sphere_x.append(center[0])
                    sphere_y.append(center[1])
                    sphere_z.append(center[2])
                    sphere_r.append(np.sqrt(r2))
                    sphere_size.append(total_size)
                    sphere_comp.append(total_comp)

            # output for each target gene
            sphere = pd.DataFrame(list(zip(sphere_x, sphere_y, sphere_z, sphere_r, sphere_size, sphere_comp)), columns = ['sphere_x', 'sphere_y', 'sphere_z', 'sphere_r', 'size', 'comp'])
            sphere['gene'] = [j] * sphere.shape[0]
            result_data[target_names.index(j)] = sphere
            
            if print_itr:
                print('Gene: {}; Transcripts: {}; Min_samples: {}; Filtered aggregations: {}'.format(j, target.shape[0], min_spl, sphere.shape[0]))
        
        return result_data # return a dictionary
    
    
    def find_points(self, sphere_a, sphere_b):
        transcripts = self.transcripts[self.transcripts['target'].isin(self.target_all)]
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
    
    
    # def remove_overlaps(self, set_a, set_b):
    
    #     # find possible overlaps on 2D by r-tree
    #     idx_b = make_rtree(set_b)
    #     for i, sphere_a in set_a.iterrows():
    #         center_a_3D = (sphere_a.sphere_x, sphere_a.sphere_y, sphere_a.sphere_z)
    #         bounds_a = (
    #             sphere_a.sphere_x - sphere_a.sphere_r,
    #             sphere_a.sphere_y - sphere_a.sphere_r,
    #             sphere_a.sphere_x + sphere_a.sphere_r,
    #             sphere_a.sphere_y + sphere_a.sphere_r
    #         )
    #         possible_overlaps = idx_b.intersection(bounds_a)
            
    #         # search 3D overlaps within possible overlaps
    #         for j in possible_overlaps:
    #             if j in set_b.index:
    #                 sphere_b = set_b.loc[j]
    #                 center_b_3D = (sphere_b.sphere_x, sphere_b.sphere_y, sphere_b.sphere_z)
    #                 dist = math.dist(center_a_3D, center_b_3D)
    #                 radius_sum = sphere_a.sphere_r + sphere_b.sphere_r
    #                 radius_diff = sphere_a.sphere_r - sphere_b.sphere_r
                    
    #                 # relative positions (0: internal & intersect, 1: internal, 2: intersect)
    #                 c0 = (dist < self.l * radius_sum)
    #                 c1 = (dist <= self.l * np.abs(radius_diff))
    #                 c1_1 = (radius_diff > 0)                      # internal: radius_A > radius_B
    #                 c2_1 = (dist < self.p * self.l * radius_sum)  # intersect: heavily overlap
                    
    #                 # operations on dataframes
    #                 if c0:
    #                     if c1 and c1_1:          # keep A and remove B
    #                         set_b = set_b.drop(j)
    #                     elif c1 and (not c1_1):  # replace A with B and remove B
    #                         set_a.loc[i] = set_b.loc[j]
    #                         set_b = set_b.drop(j)
    #                     elif (not c1) and c2_1:  # replace A with the new sphere and remove B
    #                         points_union = np.array(self.find_points(sphere_a, sphere_b))
    #                         new_center, new_radius = miniball.get_bounding_ball(points_union, epsilon = 1e-8)
    #                         set_a.sphere_x.loc[i] = new_center[0]
    #                         set_a.sphere_y.loc[i] = new_center[1]
    #                         set_a.sphere_z.loc[i] = new_center[2]
    #                         set_a.sphere_r.loc[i] = self.s * new_radius
    #                         set_b = set_b.drop(j)
        
    #     # return output
    #     set_a = set_a.reset_index(drop = True)
    #     set_b = set_b.reset_index(drop = True)           
    #     return set_a, set_b
    
    
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
                    c2_1 = (dist < self.p * self.l * radius_sum)

                    # operations on dataframes
                    if c0:
                        if c1 and c1_1:         # keep A and remove B
                            set_b.drop(index = j, inplace = True)
                        elif c1 and not c1_1:   # replace A with B and remove B
                            set_a.loc[i] = set_b.loc[j]
                            set_b.drop(index = j, inplace = True)
                        elif not c1 and c2_1:   # replace A with the new sphere and remove B
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
    
    
    def merge_sphere(self, sphere_dict, print_itr = False):
        sphere = sphere_dict[0].copy()
        for j in range(1, len(self.target_all)):
            target_sphere = sphere_dict[j]
            sphere, target_sphere_new = self.remove_overlaps(sphere, target_sphere)
            if print_itr:
                print(sphere.shape[0], target_sphere.shape[0])
            sphere = pd.concat([sphere, target_sphere_new])
            if print_itr:
                print(sphere.shape[0])
            sphere = sphere.reset_index(drop = True)
        return sphere
    
    
    def merge_data(self, print_dbscan = False, print_merge = False):
        result_data = self.dbscan(print_itr = print_dbscan)
        sphere = self.merge_sphere(result_data, print_itr = print_merge)
        return sphere