import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from mcDETECT.utils import *
from mcDETECT.model import *
from mcDETECT.downstream import *

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# Paths
compared_samples = ["MERSCOPE_WT_1", "MERSCOPE_AD_1"]
data_paths = [f"../data/{dataset}/" for dataset in compared_samples]
output_paths = [f"../output/{dataset}/" for dataset in compared_samples]
comparison_path = "../output/MERSCOPE_WT_AD_comparison/"

# Parameters
if compared_samples == ["MERSCOPE_WT_1", "MERSCOPE_AD_1"]:
    shift_x, shift_y = 12000, 7200
    cutoff = 6250
else:
    raise ValueError("Invalid compared_samples!")

# ==================== Read data ==================== #

# Transcripts
transcripts_1 = pd.read_parquet(data_paths[0] + "processed_data/transcripts.parquet")
transcripts_2 = pd.read_parquet(data_paths[1] + "processed_data/transcripts.parquet")

# Genes
genes = pd.read_csv(data_paths[0] + "processed_data/genes.csv")
genes = list(genes.iloc[:, 0])

# Negative control markers
nc_genes = pd.read_csv(data_paths[0] + "processed_data/negative_controls.csv")
nc_genes = list(nc_genes["Gene"])

# Markers
syn_genes = ["Camk2a", "Cplx2", "Slc17a7", "Ddn", "Syp", "Map1a", "Shank1", "Syn1", "Gria1", "Gria2", "Cyfip2", "Vamp2", "Bsn", "Slc32a1", "Nfasc", "Syt1", "Tubb3", "Nav1", "Shank3", "Mapt"]
print("Number of syn_genes: ", len(syn_genes))

# ==================== Granule expression profile ==================== #

# Read granules
granules_1 = pd.read_parquet(output_paths[0] + "granules.parquet")
granules_2 = pd.read_parquet(output_paths[1] + "granules.parquet")
print("Granules shape: ", granules_1.shape, granules_2.shape)

# Initialize mcDETECT
mc1 = mcDETECT(type = "discrete", transcripts = transcripts_1, gnl_genes = syn_genes, nc_genes = None, eps = 1.5,
               minspl = 3, grid_len = 1, cutoff_prob = 0.95, alpha = 10, low_bound = 3, size_thr = 1e5,
               in_soma_thr = 1.01, l = 1, rho = 0.2, s = 1, nc_top = 20, nc_thr = 0.1)
mc2 = mcDETECT(type = "discrete", transcripts = transcripts_2, gnl_genes = syn_genes, nc_genes = None, eps = 1.5,
               minspl = 3, grid_len = 1, cutoff_prob = 0.95, alpha = 10, low_bound = 3, size_thr = 1e5,
               in_soma_thr = 1.01, l = 1, rho = 0.2, s = 1, nc_top = 20, nc_thr = 0.1)

# Profile granules
granule_adata_1 = mc1.profile(granules_1, genes = genes)
granule_adata_2 = mc2.profile(granules_2, genes = genes)
print("Granule adata shape: ", granule_adata_1.shape, granule_adata_2.shape)

# Add metadata columns
mismatch1 = (granule_adata_1.obs["global_x"].to_numpy() != granules_1["sphere_x"].to_numpy()).sum()
mismatch2 = (granule_adata_2.obs["global_x"].to_numpy() != granules_2["sphere_x"].to_numpy()).sum()

if (mismatch1 > 0) or (mismatch2 > 0):
    raise ValueError(f"Granule metadata and expression profile do not match! mismatches: {mismatch1}, {mismatch2}")
else:
    granule_adata_1.obs["brain_area"] = granules_1["brain_area"]
    granule_adata_1.obs["global_x_new"] = granules_1["global_x_new"]
    granule_adata_1.obs["global_y_new"] = granules_1["global_y_new"]
    granule_adata_2.obs["brain_area"] = granules_2["brain_area"]
    granule_adata_2.obs["global_x_new"] = granules_2["global_x_new"]
    granule_adata_2.obs["global_y_new"] = granules_2["global_y_new"]

# Adjust coordinates (manual)
if compared_samples == ["MERSCOPE_WT_1", "MERSCOPE_AD_1"]:
    granule_adata_1.obs["global_y_new"] = cutoff - granule_adata_1.obs["global_y_new"]

granule_adata_1.obs["global_x_adjusted"] = granule_adata_1.obs["global_y_new"].copy()
granule_adata_1.obs["global_y_adjusted"] = granule_adata_1.obs["global_x_new"].copy()

granule_adata_2.obs["global_x_adjusted"] = granule_adata_2.obs["global_x_new"].copy()
granule_adata_2.obs["global_y_adjusted"] = granule_adata_2.obs["global_y_new"].copy()

granule_adata_2.obs["global_x_adjusted"] += shift_x
granule_adata_2.obs["global_y_adjusted"] += shift_y

# Concatenate granule adata
granule_adata_dict = {compared_samples[0]: granule_adata_1, compared_samples[1]: granule_adata_2}
granule_adata = anndata.concat(granule_adata_dict, axis = 0, merge = "same", label = "batch")
granule_adata.layers["counts"] = csr_matrix(granule_adata.X.copy())

# Normalize granule adata
sc.pp.normalize_total(granule_adata, target_sum=1e4)
sc.pp.log1p(granule_adata)

# t-SNE embedding
sc.tl.pca(granule_adata, n_comps=10, svd_solver="auto")
sc.tl.tsne(granule_adata, n_pcs=10)

# Save granule adata
granule_adata.write_h5ad(comparison_path + "granule_adata_tsne.h5ad")
print("Granule adata: ", granule_adata)