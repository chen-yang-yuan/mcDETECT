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
dataset = "Xenium_5K"
data_path = f"../../data/{dataset}/"
output_path = f"../../output/{dataset}/"

# -------------------- Read data -------------------- #

# Cells
adata = sc.read_h5ad(data_path + "adata.h5ad")

# Neurons
adata_neuron = adata[adata.obs["cell_type"].isin(["Glutamatergic", "GABAergic"])].copy()

# Transcripts
transcripts = pd.read_parquet(data_path + "transcripts.parquet")

# Genes
genes = pd.read_csv(data_path + "genes.csv")
genes = list(genes.iloc[:, 0])

# Negative control markers
nc_genes = pd.read_csv(data_path + "negative_controls.csv")
nc_genes = list(nc_genes["Gene"])

# Spots
spots = sc.read_h5ad(data_path + "spots.h5ad")

# Markers
syn_genes = ["Snap25", "Camk2a", "Slc17a7", "Cyfip2", "Map2", "Syp", "Syn1", "Slc32a1", "Vamp2", "Mapt", "Gria2", "Gap43", "Tubb3", "Dlg4", "Gria1", "Bsn"]
print("Number of syn_genes: ", len(syn_genes))

# -------------------- Detection -------------------- #

# Initialize mcDETECT
mc = mcDETECT(type = "discrete", transcripts = transcripts, gnl_genes = syn_genes, nc_genes = nc_genes, eps = 1.5,
              minspl = 4, grid_len = 1, cutoff_prob = 0.95, alpha = 10, low_bound = 3, size_thr = 4.0,
              in_soma_thr = 0.1, l = 1, rho = 0.2, s = 1, nc_top = 20, nc_thr = 0.1, merge_genes = True, merged_gene_label = "merged")

# Detection
granules = mc.detect()

# Assign region labels
labels_df = pd.DataFrame({"global_x": spots.obs["global_x"].to_numpy(), "global_y": spots.obs["global_y"].to_numpy(), "brain_area": spots.obs["brain_area"].to_numpy(),}).reset_index(drop=True)
spot_xy = labels_df[["global_x", "global_y"]].to_numpy()
tree = cKDTree(spot_xy)
gnl_xy = granules[["sphere_x", "sphere_y"]].to_numpy()
_, nn_idx = tree.query(gnl_xy, k=1)
granules = granules.copy()
granules["brain_area"] = labels_df.loc[nn_idx, "brain_area"].to_numpy()
granules.head()

# Save granules
granules.to_parquet(output_path + f"granules_merged_genes_minspl_{mc.minspl}.parquet")

# -------------------- Granule expression profiling -------------------- #

granule_adata = mc.profile(granules, genes = genes)
granule_adata.layers["counts"] = csr_matrix(granule_adata.X.copy())

sc.pp.normalize_total(granule_adata, target_sum=1e4)
sc.pp.log1p(granule_adata)
sc.tl.pca(granule_adata, n_comps=10, svd_solver="auto")
sc.tl.tsne(granule_adata, n_pcs=10)

granule_adata.write_h5ad(output_path + f"granule_adata_tsne_merged_genes_minspl_{mc.minspl}.h5ad")
print("Granule adata: ", granule_adata)