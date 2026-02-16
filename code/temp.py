import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from mcDETECT.utils import *
from mcDETECT.model import *

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# File paths
dataset = "MERSCOPE_WT_1"
data_path = f"../data/{dataset}/"
output_path = f"../output/{dataset}/"

if dataset == "MERSCOPE_WT_1":
    flip = True
    cutoff = 6250
    theta = 10 * np.pi / 180
    coordinate_for_rotation = ["sphere_y", "sphere_x"]
    coordinate_for_flip = "global_y"
else:
    flip = True
    cutoff = 6250
    theta = 10 * np.pi / 180
    coordinate_for_rotation = ["sphere_y", "sphere_x"]
    coordinate_for_flip = "global_y"

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

# ==================== Fine detection (run once) ==================== #
# size filtering (size_thr = 4.0)
# in-soma filtering (in_soma_thr = 0.1)
# negative control filtering (nc_genes = nc_genes)

mc = mcDETECT(type = "discrete", transcripts = transcripts, gnl_genes = syn_genes, nc_genes = nc_genes, eps = 1.5,
              minspl = 3, grid_len = 1, cutoff_prob = 0.95, alpha = 10, low_bound = 3, size_thr = 4.0,
              in_soma_thr = 0.1, l = 1, rho = 0.2, s = 1, nc_top = 20, nc_thr = 0.1)

granules = pd.read_parquet(output_path + "granules.parquet")
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