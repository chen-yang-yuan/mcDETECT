import cv2
import numpy as np
import pandas as pd
import os


class simulation: # for simulating one point process only, CSR and clustering
    
    
    def __init__(self, name, density, shape, layer_num, layer_gap, simulate_z, write_path, seed = 42):
        self.name = name                                                # string, gene name
        self.density = density                                          # numeric, transcript density
        self.shape = shape                                              # 2D array, simulation area
        self.total = int(shape[0] * shape[1] * density)                 # integer, total number of transcripts
        self.layer_num = layer_num                                      # integer, number of layers
        self.layer_gap = layer_gap                                      # numeric, distance between adjacent layers
        self.z_range = (layer_num - 1) * layer_gap                      # numeric, maximum of z-axis values 
        self.simulate_z = simulate_z                                    # boolean, whether to simulate 3D points
        self.write_path = write_path                                    # string, path to save output
        self.seed = seed                                                # integer, random seed
        
        # Ensure output directory exists
        if write_path and not os.path.exists(write_path):
            os.makedirs(write_path, exist_ok=True)
    
    
    def simulate_CSR(self, write_csv = False):
        np.random.seed(self.seed)
        x_coord = np.random.uniform(0, self.shape[0], self.total)
        y_coord = np.random.uniform(0, self.shape[1], self.total)
        if self.simulate_z:
            z_coord = np.random.uniform(0, self.z_range, self.total)
        else:
            z_coord = np.zeros(self.total)
        points = pd.DataFrame({'id': list(range(self.total)), 'target': [self.name] * self.total, 'global_x': x_coord, 'global_y': y_coord, 'global_z': z_coord, 'in_nucleus': [0] * self.total})
        if write_csv:
            points.to_csv(os.path.join(self.write_path, 'simulation_CSR_{}_density_{:.4f}.csv'.format(self.name, self.density)), index = 0)
        return points
    
    
    def simulate_cluster(self, num_clusters, beta, mean_dist, write_csv = False):
        
        """
        num_clusters: integer, number of clusters/parent nodes
        beta: tuple, two parameters in the beta distribution for simulating in-nucleus ratio
        mean_dist: numeric, mean distance between parent nodes and offsprings
        """
        
        # parent nodes
        np.random.seed(self.seed)
        parent_x = np.random.uniform(0, self.shape[0], num_clusters)
        parent_y = np.random.uniform(0, self.shape[1], num_clusters)
        parent_z = np.random.uniform(0, self.z_range, num_clusters)
        in_nucleus_ratio = np.random.beta(beta[0], beta[1], num_clusters)
        
        # poisson number of offsprings
        n_offspring = np.random.poisson(self.total/num_clusters, num_clusters)
        n = np.sum(n_offspring) # total number of offsprings, should be close to self.total
        
        # exponential distance
        d_offspring = np.random.exponential(mean_dist, n)
        
        # uniform angle with sine-weighted polar angle for uniform distribution on sphere
        theta = np.random.uniform(0, 2 * np.pi, n)      # azimuthal angle
        u = np.random.uniform(0, 1, n)                  # for sine-weighted polar angle
        phi = np.arccos(1 - 2 * u)                      # sine-weighted polar angle
        
        # make data
        id = []
        in_nucleus = []
        for i in range(num_clusters):
            np.random.seed(self.seed + i)
            id += [i] * n_offspring[i]
            in_nucleus += np.random.binomial(1, in_nucleus_ratio[i], n_offspring[i]).tolist()
        
        offspring_x = np.zeros(n)
        offspring_y = np.zeros(n)
        offspring_z = np.zeros(n)
        
        for i in range(n):
            offspring_x[i] = parent_x[id[i]] + d_offspring[i] * np.sin(phi[i]) * np.cos(theta[i])
            offspring_y[i] = parent_y[id[i]] + d_offspring[i] * np.sin(phi[i]) * np.sin(theta[i])
            offspring_z[i] = parent_z[id[i]] + d_offspring[i] * np.cos(phi[i])
        
        if not self.simulate_z:
            parent_z = np.zeros(num_clusters)
            offspring_z = np.zeros(n)
        
        parents = pd.DataFrame({'target': [self.name] * num_clusters, 'global_x': parent_x, 'global_y': parent_y, 'global_z': parent_z})
        points = pd.DataFrame({'id': id, 'target': [self.name] * n, 'global_x': offspring_x, 'global_y': offspring_y, 'global_z': offspring_z, 'in_nucleus': in_nucleus})
        if write_csv:
            parents.to_csv(os.path.join(self.write_path, 'simulation_cluster_{}_parents.csv'.format(self.name)), index = 0)
            points.to_csv(os.path.join(self.write_path, 'simulation_cluster_{}_density_{:.4f}.csv'.format(self.name, self.density)), index = 0)
        return parents, points
    
    
    def plot_points(self, points, radius = 1, color = (255, 0, 0), thickness = -1):
        img = np.full((self.shape[0], self.shape[1], 3), 255, dtype = np.uint8)
        n = points.shape[0]
        for i in range(n):
            x = int(points['global_x'][i])
            y = int(points['global_y'][i])
            img = cv2.circle(img, (x, y), radius, color, thickness)
        cv2.imwrite(os.path.join(self.write_path, 'simulation_{}_density_{:.4f}.png'.format(self.name, self.density)), img)




class multi_simulation: # for simulating multiple point processes, clustering
    
    
    def __init__(self, name, density, shape, layer_num, layer_gap, simulate_z, write_path, seed = 42):
        self.name = name                                                             # list, string, gene name
        self.density = density                                                       # list, numeric, transcript density
        self.shape = shape                                                           # 2D array, simulation area
        self.total = [int(shape[0] * shape[1] * i) for i in density]                 # list, integer, total number of transcripts
        self.layer_num = layer_num                                                   # integer, number of layers
        self.layer_gap = layer_gap                                                   # numeric, distance between adjacent layers
        self.z_range = (layer_num - 1) * layer_gap                                   # numeric, maximum of z-axis values
        self.simulate_z = simulate_z                                                 # boolean, whether to simulate 3D points
        self.write_path = write_path                                                 # string, path to save output
        self.seed = seed                                                             # integer, random seed
        
        # Ensure output directory exists
        if write_path and not os.path.exists(write_path):
            os.makedirs(write_path, exist_ok=True)
    
    
    def simulate_cluster(self, num_clusters, comp_prob, beta, mean_dist, comp_thr = 2, write_csv = False):
        
        """
        num_clusters: integer, number of total clusters/parent nodes
        comp_prob: list, probability of number of components in each cluster
        beta: tuple, two parameters in the beta distribution for simulating in-nucleus ratio
        mean_dist: numeric, mean distance between parent nodes and offsprings
        comp_thr: integer, composition threshold to filter the true aggregations
        """
        
        # parent nodes
        np.random.seed(self.seed)
        parent_x = np.random.uniform(0, self.shape[0], num_clusters)
        parent_y = np.random.uniform(0, self.shape[1], num_clusters)
        parent_z = np.random.uniform(0, self.z_range, num_clusters)
        in_nucleus_ratio = np.random.beta(beta[0], beta[1], num_clusters)
        
        # number of components in each cluster
        comp_list = [i + 1 for i in range(len(self.name))]
        num_comp = np.random.choice(comp_list, num_clusters, replace = True, p = comp_prob)
        
        # components in each cluster
        comp_name = {}
        for i in range(num_clusters):
            np.random.seed(self.seed + i)
            temp_name = list(np.random.choice(self.name, num_comp[i], replace = False))
            comp_name[i] = temp_name
        
        # simulate each individual gene
        np.random.seed(self.seed)
        
        parents_all = {}
        points_all = {}
        
        for j in self.name:
            
            j_idx = self.name.index(j)
            
            # clusters containing j
            valid_index = [i for i in range(num_clusters) if j in comp_name[i]]
            num_clusters_temp = len(valid_index)
            if num_clusters_temp == 0:
                continue
            parent_x_temp = parent_x[valid_index]
            parent_y_temp = parent_y[valid_index]
            parent_z_temp = parent_z[valid_index]
            
            # poisson number of offsprings
            n_offspring = np.random.poisson(self.total[j_idx]/num_clusters_temp, num_clusters_temp)
            n = np.sum(n_offspring)
            
            # exponential distance
            d_offspring = np.random.exponential(mean_dist, n)
            
            # uniform angle with sine-weighted polar angle for uniform distribution on sphere
            theta = np.random.uniform(0, 2 * np.pi, n)      # azimuthal angle
            u = np.random.uniform(0, 1, n)                  # for sine-weighted polar angle
            phi = np.arccos(1 - 2 * u)                      # sine-weighted polar angle
            
            # make data
            id = []
            in_nucleus = []
            for i in range(num_clusters_temp):
                np.random.seed(self.seed + i)
                id += [i] * n_offspring[i]
                in_nucleus += np.random.binomial(1, in_nucleus_ratio[i], n_offspring[i]).tolist()
        
            offspring_x = np.zeros(n)
            offspring_y = np.zeros(n)
            offspring_z = np.zeros(n)
            
            for i in range(n):
                offspring_x[i] = parent_x_temp[id[i]] + d_offspring[i] * np.sin(phi[i]) * np.cos(theta[i])
                offspring_y[i] = parent_y_temp[id[i]] + d_offspring[i] * np.sin(phi[i]) * np.sin(theta[i])
                offspring_z[i] = parent_z_temp[id[i]] + d_offspring[i] * np.cos(phi[i])
                
            if not self.simulate_z:
                parent_z_temp = np.zeros(num_clusters_temp)
                offspring_z = np.zeros(n)
                
            parents_temp = pd.DataFrame({'target': [j] * num_clusters_temp, 'global_x': parent_x_temp, 'global_y': parent_y_temp, 'global_z': parent_z_temp})
            parents_all[j_idx] = parents_temp
            points = pd.DataFrame({'id': id, 'target': [j] * n, 'global_x': offspring_x, 'global_y': offspring_y, 'global_z': offspring_z, 'in_nucleus': in_nucleus})
            points_all[j_idx] = points
        
        # filter clusters with composition >= comp_thr
        if self.simulate_z == False:
            parent_z = np.zeros(num_clusters)
        parents = pd.DataFrame({'id': range(np.sum(num_comp >= comp_thr)), 'global_x': parent_x[num_comp >= comp_thr], 'global_y': parent_y[num_comp >= comp_thr], 'global_z': parent_z[num_comp >= comp_thr]})
        
        # merge all genes
        parents_all = pd.concat(parents_all.values(), axis=0)
        points_all = pd.concat(points_all.values(), axis=0)
        if write_csv:
            parents.to_csv(os.path.join(self.write_path, 'simulation_cluster_true_parents.csv'), index = 0)
            parents_all.to_csv(os.path.join(self.write_path, 'simulation_cluster_all_parents.csv'), index = 0)
            points_all.to_csv(os.path.join(self.write_path, 'simulation_cluster_all_points.csv'), index = 0)
        return parents, parents_all, points_all
    
    
    def plot_points(self, points, color, radius = 1, thickness = -1):
        """
        color: list, bgr of each gene
        """
        img = np.full((self.shape[0], self.shape[1], 3), 255, dtype = np.uint8)
        for j in self.name:
            j_idx = self.name.index(j)
            points_temp = points[points['target'] == j]
            n = points_temp.shape[0]
            for i in range(n):
                x = int(points_temp['global_x'][i])
                y = int(points_temp['global_y'][i])
                img = cv2.circle(img, (x, y), radius, color[j_idx], thickness)
        cv2.imwrite(os.path.join(self.write_path, 'simulation_all.png'), img)
