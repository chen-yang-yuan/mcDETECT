# mcDETECT Tutorial

Authors: Chenyang Yuan, Krupa Patel, Hongshun Shi, Hsiao-Lin V. Wang, Feng Wang, Ronghua Li, Yangping Li, Victor G. Corces, Hailing Shi, Sulagna Das, Jindan Yu, Peng Jin, Bing Yao* and Jian Hu*

### Outline

1. [Installation](#1-installation)
2. [Import Python modules](#2-import-python-modules)
3. [Read in data](#3-read-in-data)
4. [Parameter settings](#4-parameter-settings)
5. [Synapse detection](#5-synapse-detection)
6. [Spatial domain assignment](#6-spatial-domain-assignment)
7. [Synapse transcriptome profiling](#7-synapse-transcriptome-profiling)
8. [Synapse subtyping](#8-synapse-subtyping)

### 1. Installation

The detailed installation procedure can be found in [Installation](../README.md/#installation). Here I directly install `mcDETECT` by running:

```bash
python3 -m pip install mcDETECT
```

Check the current version:


```python
import mcDETECT
mcDETECT.__version__
```




    '1.0.10'



### 2. Import Python modules

Compiling this tutorial file needs the following Python packages:


```python
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scanpy as sc
import SpaGCN as spg
import torch
from collections import defaultdict
from sklearn.cluster import KMeans

from mcDETECT import mcDETECT, closest

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0
```

You may also need to have `pyarrow` and `fastparquet` installed for reading `.parquet` files.

### 3. Read in data

The toy dataset used in this tutorial is part of the isocortex region from the [Xenium 5K mouse brain data](https://www.10xgenomics.com/datasets/xenium-prime-fresh-frozen-mouse-brain).

`mcDETECT` requires the following input:

* Transcript file: dataframe, records gene identity and 3D spatial coordinates of each mRNA molecule


```python
transcripts = pd.read_parquet("toy_data/transcripts.parquet")
```

We need to rename some columns of the transcript file to adapt to the input format. The input transcript file should look like:


```python
transcripts = transcripts[['cell_id', 'overlaps_nucleus', 'feature_name', 'x_location', 'y_location', 'z_location']]
transcripts = transcripts.rename(columns = {"feature_name": "target", "x_location": "global_x", "y_location": "global_y", "z_location": "global_z"})
print(transcripts.head().to_string())
```

                  cell_id  overlaps_nucleus target     global_x     global_y   global_z
    163006771  fgdhmaei-1                 0   A1cf  5994.734375  2021.468750  15.125000
    163006772  UNASSIGNED                 0    A2m  5763.109375  2043.625000  15.781250
    163006773  UNASSIGNED                 0    A2m  5951.984375  2085.984375  16.578125
    163006774  hieeideh-1                 1   Aatf  5757.593750  2163.453125  17.281250
    163006775  fghnlpdi-1                 1   Aatf  5969.406250  2149.406250  17.625000


* Synaptic markers: user-defined list


```python
syn_genes = ['Snap25', 'Camk2a', 'Slc17a7', 'Vamp2', 'Syp', 'Syn1', 'Dlg4', 'Gria2', 'Gap43', 'Gria1', 'Bsn', 'Slc32a1']
```

* (Optional) Negative control markers: user-defined list. If `None`, negative control filtering will be disabled.


```python
nc_genes = pd.read_csv('toy_data/negative_controls.csv')
nc_genes = list(nc_genes['Gene'])
print(nc_genes[:10])
```

    ['Neat1', 'Robo3', 'Sec1', 'Syne4', 'Xist', 'Thpo', 'Spaca6', 'Trmt13', 'Fbxl12', 'Cenpa']


### 4. Parameter settings

Instantiate an object `mc` from `mcDETECT`:


```python
mc = mcDETECT(type = "Xenium", transcripts = transcripts, syn_genes = syn_genes, nc_genes = nc_genes, eps = 1.5,
              minspl = None, grid_len = 1, cutoff_prob = 0.95, alpha = 5, low_bound = 3, size_thr = 5,
              in_nucleus_thr = (0.5, 0.5), l = 1, rho = 0.2, s = 1, nc_top = 20, nc_thr = 0.1)
```

Hyperparameters in `mcDETECT`:
* `eps` ($\epsilon$): numeric, searching radius in density-based clustering, default is 1.5 $\mu m$
* `minspl`: integer, min_samples in density-based clustering, default is `None`. Users can manually define `minspl` and thus disable the automatic parameter selection process.
* `grid_len`: numeric, side length of square grids over the tissue region (used in approximating the tissue area), default is 1 $\mu m$
* `cutoff_prob`: numeric, cutoff probability in parameter selection for min_samples, default is 0.95
* `alpha` ($\alpha$): numeric, enhancing factor in parameter selection for min_samples, default is 5
* `low_bound`: integer, lower bound in parameter selection for min_samples, default is 3
* `size_thr`: numeric, threshold for maximum radius of an aggregation, default is 5 $\mu m$
* `in_nucleus_thr`: 2-d tuple, thresholds for low and high in-nucleus ratio, default is (0.5, 0.5)
* `l`: numeric, scaling factor for seaching overlapped spheres, default is 1
* `rho` ($\rho$): numeric, threshold for determining overlaps, default is 0.2
* `s`: numeric, scaling factor for merging overlapped spheres, default is 1
* `nc_top`: integer, number of top negative control genes retained for filtering, default is 20
* `nc_thr`: numeric, threshold for negative control filtering, default is 0.1

### 5. Synapse detection

Synapse detection is implemented in the `detect()` function:


```python
synapses = mc.detect()
```

    1 out of 12 genes processed!
    2 out of 12 genes processed!
    3 out of 12 genes processed!
    4 out of 12 genes processed!
    5 out of 12 genes processed!
    6 out of 12 genes processed!
    7 out of 12 genes processed!
    8 out of 12 genes processed!
    9 out of 12 genes processed!
    10 out of 12 genes processed!
    11 out of 12 genes processed!
    12 out of 12 genes processed!
    Merging spheres...
    Negative control filtering...


The output is a dataframe of synapse metadata:


```python
print(synapses.head().to_string())
```

          sphere_x     sphere_y   sphere_z    layer_z  sphere_r  size  comp  in_nucleus    gene
    0  5861.525313  2021.429797  15.259961  15.259961  1.115372     8     3         0.0  Snap25
    1  5823.012341  2477.027071  18.744452  18.744452  1.400544    13     5         0.0  Snap25
    2  5805.578936  2419.213116  18.711572  18.711572  1.355859     9     3         0.0  Snap25
    3  5831.996698  2545.542771  18.262820  18.262820  1.168398     7     2         0.0  Snap25
    4  5800.522809  2731.226949  16.212029  16.212029  1.322234     7     2         0.0  Snap25


* `sphere_x`, `sphere_y`, `sphere_z`: 3D spatial coordinates of each identified synapse
* `layer_z`: the nearest z-layer of each identified synapse, only applicable in iST datasets with discrete z-coordinates, e.g., MERSCOPE and CosMx
* `sphere_r`: radius of each identified synapse
* `size`: number of synaptic mRNAs within each synapse
* `comp`: number of synaptic genes presented in each synapse
* `in_nucleus`: proportion of synaptic mRNAs located within cell nuclei
* `gene`: primary synaptic marker that defines the associated synapse

### 6. Spatial domain assignment

To detect spatial domains, we first need to create a spot-level gene expression data from the transcript file. Here we only retain the top 1,000 highly variable genes and use the `spot_expression()` function to construct such data:


```python
highly_variable_genes = pd.read_csv('toy_data/highly_variable_genes.csv')
highly_variable_genes = list(highly_variable_genes.iloc[:, 0])

spots = mc.spot_expression(grid_len = 50, genes = highly_variable_genes)
```

    0 out of 1000 genes profiled!
    100 out of 1000 genes profiled!
    200 out of 1000 genes profiled!
    300 out of 1000 genes profiled!
    400 out of 1000 genes profiled!
    500 out of 1000 genes profiled!
    600 out of 1000 genes profiled!
    700 out of 1000 genes profiled!
    800 out of 1000 genes profiled!
    900 out of 1000 genes profiled!


* `grid_len`: numeric, side length of square grids over the tissue region, default is 50 $\mu m$

The output is an anndata object representing the spot-level gene expression data:


```python
spots
```




    AnnData object with n_obs × n_vars = 400 × 1000
        obs: 'spot_id', 'global_x', 'global_y'
        var: 'genes'



Next, we apply a spatial clustering approach, `SpaGCN`, on this anndata object for spatial domain detection. For more details check its [GitHub page](https://github.com/jianhuupenn/SpaGCN/).


```python
%%capture

# Spot coordinates
x_array = spots.obs['global_x'].tolist()
y_array = spots.obs['global_y'].tolist()

# Adjacency matrix
s = 1
b = 49
adj = spg.calculate_adj_matrix(x = x_array, y = y_array, histology = False)

# Pre-processing
spots.var_names_make_unique()
spg.prefilter_genes(spots, min_cells = 3)
spg.prefilter_specialgenes(spots)
sc.pp.normalize_total(spots, target_sum = 1e4)
sc.pp.log1p(spots)

# Set hyperparameters
p = 0.5
l = spg.search_l(p, adj, start = 0.01, end = 1000, tol = 0.01, max_run = 100)

n_clusters = 6
r_seed = t_seed = n_seed = 1
res = spg.search_res(spots, adj, l, n_clusters, start = 0.7, step = 0.1, tol = 5e-3, lr = 0.05, max_epochs = 20, r_seed = r_seed, t_seed = t_seed, n_seed = n_seed)

# Run SpaGCN
clf = spg.SpaGCN()
clf.set_l(l)

random.seed(r_seed)
torch.manual_seed(t_seed)
np.random.seed(n_seed)

clf.train(spots, adj, init_spa = True, init = "louvain", res = res, tol = 5e-3, lr = 0.05, max_epochs = 200)
y_pred, prob = clf.predict()
spots.obs["pred"] = y_pred
spots.obs["pred"] = spots.obs["pred"].astype('category')

adj_2d = spg.calculate_adj_matrix(x = x_array, y = y_array, histology = False)
refined_pred = spg.refine(sample_id = spots.obs.index.tolist(), pred = spots.obs["pred"].tolist(), dis = adj_2d, shape = "square")
spots.obs["refined_pred"] = refined_pred
spots.obs["refined_pred"] = spots.obs["refined_pred"].astype('category')
```

Spot-level spatial domain assignment:


```python
sc.set_figure_params(scanpy = True, figsize = (5, 5))
ax = sc.pl.scatter(spots, alpha = 1, x = "global_y", y = "global_x", color = "refined_pred", size = 800, title = " ", show = False)
ax.grid(False)
ax.set_aspect('equal', 'box')
plt.savefig("tutorial_files/spatial_domain.png", dpi = 120)
plt.show()
```


    
![png](tutorial_files/tutorial_29_0.png)
    


We can replace the spatial domain labels with meaningful brain region labels, i.e., isocortex layers:


```python
area_dict = {'Others': [4],
             'Layer I': [3],
             'Layer II/III': [0],
             'Layer IV': [1],
             'Layer V': [2],
             'Layer VI': [5]}

spots.obs['brain_area'] = np.nan
for i in area_dict.keys():
    ind = pd.Series(spots.obs['refined_pred']).isin(area_dict[i])
    spots.obs.loc[ind, 'brain_area'] = i
```

The assigned brain region label for each spot is propagated to all synapses within it. The `closest()` function is designed to identify the element in a list that has the smallest distance to the query item.


```python
labels_df = pd.DataFrame({'global_x': spots.obs['global_x'], 'global_y': spots.obs['global_y'], 'brain_area': spots.obs['brain_area']})
x_grid, y_grid = list(np.unique(labels_df['global_x'])), list(np.unique(labels_df['global_y']))

synapses['brain_area'] = np.nan
for i in range(synapses.shape[0]):
    closest_x = closest(x_grid, synapses['sphere_x'].iloc[i])
    closest_y = closest(y_grid, synapses['sphere_y'].iloc[i])
    target_label = labels_df[(labels_df['global_x'] == closest_x) & (labels_df['global_y'] == closest_y)]
    synapses['brain_area'].iloc[i] = target_label['brain_area'][0]
synapses['brain_area'] = synapses['brain_area'].astype(str)
```

The resulting spatial distribution of all identified synapses, colored by brain region:


```python
synapse_adata = anndata.AnnData(X = np.zeros(synapses.shape), obs = synapses)
synapse_adata.obs['brain_area'] = pd.Categorical(synapse_adata.obs['brain_area'], categories = ['Layer I', 'Layer II/III', 'Layer IV', 'Layer V', 'Layer VI'], ordered = True)

color_map = ["#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3"]
ax = sc.pl.scatter(synapse_adata, alpha = 1, x = 'sphere_y', y = 'sphere_x', color = 'brain_area', palette = color_map, size = 30, title = " ", show = False)
ax.grid(False)
ax.set_aspect('equal', 'box')
plt.savefig("tutorial_files/synapses.png", dpi = 120)
plt.show()
```


    
![png](tutorial_files/tutorial_35_0.png)
    


### 7. Synapse transcriptome profiling

Synapse transcriptome profiling is implemented in the `profile()` function. The 


```python
syn_adata = mc.profile(synapses)
```

The output is an anndata object representing the spatial transcriptome profile of all identified synapses:


```python
syn_adata
```




    AnnData object with n_obs × n_vars = 1279 × 5006
        obs: 'global_x', 'global_y', 'global_z', 'layer_z', 'sphere_r', 'size', 'comp', 'in_nucleus', 'gene', 'brain_area', 'synapse_id'
        var: 'genes'



### 8. Synapse subtyping

We can classify the identified synapses into dinstinct subtypes, e.g., pre-synapses and post-synapses, based on their transcriptome profile. Here we use a list of pre- and post-synaptic markers for synapse subtyping:


```python
ref_genes = ['Bsn', 'Gap43', 'Nrxn1', 'Slc17a6', 'Slc17a7', 'Slc32a1', 'Snap25', 'Stx1a', 'Syn1', 'Syp', 'Syt1', 'Vamp2'] + ['Camk2a', 'Dlg3', 'Dlg4', 'Gphn', 'Gria1', 'Gria2', 'Homer1', 'Homer2', 'Nlgn1', 'Nlgn2', 'Nlgn3', 'Shank1', 'Shank3']
ref_genes = [i for i in ref_genes if i in syn_adata.var_names]

syn_adata_subset = syn_adata[:, ref_genes].copy()
syn_adata_subset
```




    AnnData object with n_obs × n_vars = 1279 × 20
        obs: 'global_x', 'global_y', 'global_z', 'layer_z', 'sphere_r', 'size', 'comp', 'in_nucleus', 'gene', 'brain_area', 'synapse_id'
        var: 'genes'



K-Means clustering on the synapses based on the enrichment of these markers:


```python
data = syn_adata_subset.X
if not isinstance(data, np.ndarray):
    data = data.toarray()

n_clusters = 10
kmeans = KMeans(n_clusters = n_clusters, random_state = 42, n_init = 25)
kmeans.fit(data)
syn_adata.obs['kmeans_pre_post'] = kmeans.labels_.astype(str)
```

Examine the enrichment of these markers in each resulting clusters:


```python
marker_genes = {'pre-syn': ['Bsn', 'Gap43', 'Slc17a6', 'Slc17a7', 'Slc32a1', 'Snap25', 'Stx1a', 'Syn1', 'Syp', 'Vamp2'],
                'post-syn': ['Camk2a', 'Dlg3', 'Dlg4', 'Gphn', 'Gria1', 'Gria2', 'Homer1', 'Nlgn2', 'Nlgn3', 'Shank3']}

expression_data = pd.DataFrame(syn_adata.X, columns=syn_adata.var_names, index=syn_adata.obs_names)
kmeans_labels = syn_adata.obs['kmeans_pre_post']
cluster_marker_avg = defaultdict(lambda: defaultdict(int))

for cluster, genes in marker_genes.items():
    num_genes = len(genes)
    for gene in genes:
        if gene in expression_data.columns:
            gene_expression = expression_data[gene].groupby(kmeans_labels).sum()
            for kmeans_cluster, expr_sum in gene_expression.items():
                cluster_marker_avg[kmeans_cluster][cluster] += expr_sum / num_genes

cluster_percentages = {}
for kmeans_cluster, cluster_counts in cluster_marker_avg.items():
    total_expression = sum(cluster_counts.values())
    if total_expression > 0:
        cluster_percentages[kmeans_cluster] = {cluster: (count / total_expression) * 100 for cluster, count in cluster_counts.items()}
    else:
        cluster_percentages[kmeans_cluster] = {cluster: 0 for cluster in cluster_counts.keys()}

cluster_labels = list(cluster_percentages.keys())
marker_clusters = list(marker_genes.keys())

plot_data = np.array([
    [cluster_percentages[cluster].get(marker, 0) for marker in marker_clusters]
    for cluster in cluster_labels
])

fig, ax = plt.subplots(figsize=(8, 6))
bottom = np.zeros(len(cluster_labels))

for i, marker_cluster in enumerate(marker_clusters):
    ax.bar(cluster_labels, plot_data[:, i], bottom=bottom, label=marker_cluster)
    bottom += plot_data[:, i]

ax.set_xlabel('mcDETECT Clusters')
ax.set_ylabel('Percentage of Marker Genes')
ax.grid(False)
ax.legend(title = "Type",loc = "upper right")
plt.savefig('tutorial_files/synapse_subtyping.png', dpi = 120)
plt.show()
```


    
![png](tutorial_files/tutorial_45_0.png)
    


Assign each cluster as representing pre- or post-synapses:


```python
pre_lst, post_lst, neutral_lst = [], [], []
for i in range(len(cluster_labels)):
    if plot_data[i, 0] > 60:
        pre_lst.append(cluster_labels[i])
    elif plot_data[i, 0] < 40:
        post_lst.append(cluster_labels[i])
    else:
        neutral_lst.append(cluster_labels[i])

pre_post_dict = {'pre-syn': pre_lst, 'post-syn': post_lst, 'neutral': neutral_lst}
syn_adata.obs['pre_post'] = np.nan
for i in pre_post_dict.keys():
    ind = pd.Series(syn_adata.obs['kmeans_pre_post']).isin(pre_post_dict[i])
    syn_adata.obs.loc[ind, 'pre_post'] = i
print(pre_lst, post_lst, neutral_lst)
```

    ['0', '1', '3', '4', '5', '6', '8', '9'] ['2', '7'] []


Spatial distribution of the identified pre- and post-synapses:


```python
syn_adata.obs["pre_post"] = pd.Categorical(syn_adata.obs["pre_post"], categories = ['pre-syn', 'post-syn'], ordered = True)

ax = sc.pl.scatter(syn_adata, alpha = 1, x = 'global_y', y = 'global_x', color = 'pre_post', size = 30, title = " ", show = False)
ax.grid(False)
ax.set_aspect('equal', 'box')
plt.savefig("tutorial_files/synapses_pre_post.png", dpi = 120)
plt.show()
```


    
![png](tutorial_files/tutorial_49_0.png)
    

