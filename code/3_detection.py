import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from mcDETECT.utils import *
from mcDETECT.model import *

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# File paths
dataset = "MERSCOPE_AD_1"
data_path = f"../data/{dataset}/"
output_path = f"../output/{dataset}/"

if dataset == "MERSCOPE_WT_1":
    flip = True
    cutoff = 6250
    theta = 10 * np.pi / 180
    coordinate_for_rotation = ["sphere_y", "sphere_x"]
    coordinate_for_flip = "global_y"
elif dataset == "MERSCOPE_AD_1":
    flip = False
    cutoff = -4200
    theta = 170 * np.pi / 180
    coordinate_for_rotation = ["sphere_x", "sphere_y"]
    coordinate_for_flip = "global_x"

# ==================== Read data ==================== #

# Cells
adata = sc.read_h5ad(data_path + "processed_data/adata.h5ad")

# Neurons
adata_neuron = adata[adata.obs["cell_type"].isin(["Glutamatergic", "GABAergic"])].copy()

# Transcripts
transcripts = pd.read_parquet(data_path + "processed_data/transcripts.parquet")

# Genes
genes = pd.read_csv(data_path + "processed_data/genes.csv")
genes = list(genes.iloc[:, 0])

# Negative control markers
nc_genes = pd.read_csv(data_path + "processed_data/negative_controls.csv")
nc_genes = list(nc_genes["Gene"])

# Spots
spots = sc.read_h5ad(data_path + "processed_data/spots.h5ad")

# Markers
syn_genes = ["Camk2a", "Cplx2", "Slc17a7", "Ddn", "Syp", "Map1a", "Shank1", "Syn1", "Gria1", "Gria2", "Cyfip2", "Vamp2", "Bsn", "Slc32a1", "Nfasc", "Syt1", "Tubb3", "Nav1", "Shank3", "Mapt"]
print("Number of syn_genes: ", len(syn_genes))

# ==================== Rough detection (run once) ==================== #
# no size filtering (size_thr = 1e5)
# no in-soma filtering (in_soma_thr = 1.01)
# no negative control filtering (nc_genes = None)
    
print(f"Running rough detection for {dataset}...")

mc = mcDETECT(type = "discrete", transcripts = transcripts, gnl_genes = syn_genes, nc_genes = None, eps = 1.5,
            minspl = 3, grid_len = 1, cutoff_prob = 0.95, alpha = 10, low_bound = 3, size_thr = 1e5,
            in_soma_thr = 1.01, l = 1, rho = 0.2, s = 1, nc_top = 20, nc_thr = 0.1)

sphere_dict = mc.dbscan(record_cell_id = True)
print("Merging spheres...")
granules = mc.merge_sphere(sphere_dict)
granules.to_parquet(output_path + "all_granules.parquet")
print("Granules shape: ", granules.shape)

# ==================== Fine detection ==================== #
# size filtering (size_thr = 4.0)
# in-soma filtering (in_soma_thr = 0.1)
# negative control filtering (nc_genes = nc_genes)

print(f"Running fine detection for {dataset}...")

mc = mcDETECT(type = "discrete", transcripts = transcripts, gnl_genes = syn_genes, nc_genes = nc_genes, eps = 1.5,
              minspl = 3, grid_len = 1, cutoff_prob = 0.95, alpha = 10, low_bound = 3, size_thr = 4.0,
              in_soma_thr = 0.1, l = 1, rho = 0.2, s = 1, nc_top = 20, nc_thr = 0.1)
granules = mc.detect()
granules.to_parquet(output_path + "granules.parquet")

# Assign region labels
labels_df = pd.DataFrame({"global_x": spots.obs["global_x"].to_numpy(), "global_y": spots.obs["global_y"].to_numpy(), "brain_area": spots.obs["brain_area"].to_numpy(),}).reset_index(drop=True)
spot_xy = labels_df[["global_x", "global_y"]].to_numpy()
tree = cKDTree(spot_xy)
gnl_xy = granules[["sphere_x", "sphere_y"]].to_numpy()
_, nn_idx = tree.query(gnl_xy, k=1)
granules = granules.copy()
granules["brain_area"] = labels_df.loc[nn_idx, "brain_area"].to_numpy()

rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
coords = granules[coordinate_for_rotation].to_numpy()
transformed_coords = coords @ rotation_matrix.T
granules["global_" + coordinate_for_rotation[0].split("_")[1] + "_new"] = transformed_coords[:, 0]
granules["global_" + coordinate_for_rotation[1].split("_")[1] + "_new"] = transformed_coords[:, 1]
if flip:
    granules[coordinate_for_flip + "_new"] = cutoff - granules[coordinate_for_flip + "_new"]

# Save granules
granules.to_parquet(output_path + "granules.parquet")
print("Granules shape: ", granules.shape)

# ==================== Granule expression profiling ==================== #

granule_adata = mc.profile(granules, genes = genes)
granule_adata.layers["counts"] = csr_matrix(granule_adata.X.copy())

sc.pp.normalize_total(granule_adata, target_sum=1e4)
sc.pp.log1p(granule_adata)
sc.tl.pca(granule_adata, n_comps=10, svd_solver="auto")
sc.tl.tsne(granule_adata, n_pcs=10)

granule_adata.write_h5ad(output_path + "granule_adata_tsne.h5ad")
print("Granule adata: ", granule_adata)