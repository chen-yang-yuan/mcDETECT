import pandas as pd
import scanpy as sc
from itertools import accumulate

from mcDETECT.utils import *
from mcDETECT.model import *

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# # ==================== MERSCOPE WT sample ==================== #

# # Paths
# data_path = "../data/MERSCOPE_WT_1/"
# output_path = "../output/MERSCOPE_WT_1/"

# # Read transcripts
# transcripts = pd.read_parquet(data_path + "processed_data/transcripts.parquet")

# # Read genes
# genes = pd.read_csv(data_path + "processed_data/genes.csv")
# genes = list(genes.iloc[:, 0])

# # All markers
# genes_syn_pre = ["Bsn", "Gap43", "Nrxn1", "Slc17a6", "Slc17a7", "Slc32a1", "Snap25", "Stx1a", "Syn1", "Syp", "Syt1", "Vamp2"]
# genes_syn_post = ["Camk2a", "Dlg3", "Dlg4", "Gphn", "Gria1", "Gria2", "Homer1", "Homer2", "Nlgn1", "Nlgn2", "Nlgn3", "Shank1", "Shank3"]
# genes_axon = ["Ank3", "Nav1", "Sptnb4", "Nfasc", "Mapt", "Tubb3"]
# genes_dendrite = ["Actb",  "Cplx2", "Cyfip2", "Ddn", "Dlg4", "Map1a", "Map2"]

# all_genes = list(set(genes_syn_pre + genes_syn_post + genes_axon + genes_dendrite))
# all_genes = list(set(all_genes) & set(genes))
# print(f"Number of granule markers: {len(all_genes)}")

# # Initialize mcDETECT
# mc = mcDETECT(type = "discrete", transcripts = transcripts, gnl_genes = all_genes, nc_genes = None, eps = 1.5,
#               minspl = 3, grid_len = 1, cutoff_prob = 0.95, alpha = 10, low_bound = 3, size_thr = 4.0,
#               in_soma_thr = 0.1, l = 1, rho = 0.2, s = 1, nc_top = 20, nc_thr = 0.1)

# # Single-marker detection
# num_granules = []
# for idx, gene in enumerate(all_genes):
#     sphere_dict = mc.dbscan(target_names = [gene])
#     num_granules.append(len(sphere_dict[0]))
#     print(f"{idx + 1} genes processed!")
    
# # Gene ranking
# gene_ranking_raw = pd.DataFrame({"all_genes": all_genes, "num_granules": num_granules})
# gene_ranking_raw = gene_ranking_raw.sort_values(by = "num_granules", ascending = False)
# gene_ranking_raw["num_granules_cumulative"] = list(accumulate(gene_ranking_raw["num_granules"]))
# gene_ranking_raw.to_csv(output_path + "gene_ranking_raw.csv", index = 0)

# # Find threshold index
# target_index = find_threshold_index(list(gene_ranking_raw["num_granules_cumulative"]), threshold = 0.99)
# print(target_index, list(gene_ranking_raw["all_genes"])[:target_index + 1])

# ==================== MERSCOPE AD sample ==================== #

# Paths
data_path = "../data/MERSCOPE_AD_1/"
output_path = "../output/MERSCOPE_AD_1/"

# Read transcripts
transcripts = pd.read_parquet(data_path + "processed_data/transcripts.parquet")

# Read genes
genes = pd.read_csv(data_path + "processed_data/genes.csv")
genes = list(genes.iloc[:, 0])

# All markers
genes_syn_pre = ["Bsn", "Gap43", "Nrxn1", "Slc17a6", "Slc17a7", "Slc32a1", "Snap25", "Stx1a", "Syn1", "Syp", "Syt1", "Vamp2"]
genes_syn_post = ["Camk2a", "Dlg3", "Dlg4", "Gphn", "Gria1", "Gria2", "Homer1", "Homer2", "Nlgn1", "Nlgn2", "Nlgn3", "Shank1", "Shank3"]
genes_axon = ["Ank3", "Nav1", "Sptnb4", "Nfasc", "Mapt", "Tubb3"]
genes_dendrite = ["Actb",  "Cplx2", "Cyfip2", "Ddn", "Dlg4", "Map1a", "Map2"]

all_genes = list(set(genes_syn_pre + genes_syn_post + genes_axon + genes_dendrite))
all_genes = list(set(all_genes) & set(genes))
print(f"Number of granule markers: {len(all_genes)}")

# Initialize mcDETECT
mc = mcDETECT(type = "discrete", transcripts = transcripts, gnl_genes = all_genes, nc_genes = None, eps = 1.5,
              minspl = 3, grid_len = 1, cutoff_prob = 0.95, alpha = 10, low_bound = 3, size_thr = 4.0,
              in_soma_thr = 0.1, l = 1, rho = 0.2, s = 1, nc_top = 20, nc_thr = 0.1)

# Single-marker detection
num_granules = []
for idx, gene in enumerate(all_genes):
    sphere_dict = mc.dbscan(target_names = [gene])
    num_granules.append(len(sphere_dict[0]))
    print(f"{idx + 1} genes processed!")
    
# Gene ranking
gene_ranking_raw = pd.DataFrame({"all_genes": all_genes, "num_granules": num_granules})
gene_ranking_raw = gene_ranking_raw.sort_values(by = "num_granules", ascending = False)
gene_ranking_raw["num_granules_cumulative"] = list(accumulate(gene_ranking_raw["num_granules"]))
gene_ranking_raw.to_csv(output_path + "gene_ranking_raw.csv", index = 0)

# Find threshold index
target_index = find_threshold_index(list(gene_ranking_raw["num_granules_cumulative"]), threshold = 0.99)
print(target_index, list(gene_ranking_raw["all_genes"])[:target_index + 1])

# # ==================== Xenium 5K sample ==================== #

# # Paths
# data_path = "../data/Xenium_5K/"
# output_path = "../output/Xenium_5K/"

# # Read transcripts
# transcripts = pd.read_parquet(data_path + "transcripts.parquet")

# # Read genes
# genes = pd.read_csv(data_path + "genes.csv")
# genes = list(genes.iloc[:, 0])

# # All markers
# genes_syn_pre = ["Bsn", "Gap43", "Nrxn1", "Slc17a6", "Slc17a7", "Slc32a1", "Snap25", "Stx1a", "Syn1", "Syp", "Syt1","Vamp2"]
# genes_syn_post = ["Camk2a", "Dlg3", "Dlg4", "Gphn", "Gria1", "Gria2", "Homer1", "Homer2", "Nlgn1", "Nlgn2", "Nlgn3", "Shank1", "Shank3"]
# genes_axon = ["Ank3", "Nav1", "Sptnb4", "Nfasc", "Mapt", "Tubb3"]
# genes_dendrite = ["Actb",  "Cplx2", "Cyfip2", "Ddn", "Dlg4", "Map1a", "Map2"]

# all_genes = list(set(genes_syn_pre + genes_syn_post + genes_axon + genes_dendrite))
# all_genes = list(set(all_genes) & set(genes))
# print(f"Number of granule markers: {len(all_genes)}")

# # Initialize mcDETECT
# mc = mcDETECT(type = "continuous", transcripts = transcripts, gnl_genes = all_genes, nc_genes = None, eps = 1.5,
#               minspl = 3, grid_len = 1, cutoff_prob = 0.95, alpha = 10, low_bound = 3, size_thr = 4.0,
#               in_soma_thr = 0.1, l = 1, rho = 0.2, s = 1, nc_top = 20, nc_thr = 0.1)

# # Single-marker detection
# num_granules = []
# for idx, gene in enumerate(all_genes):
#     sphere_dict = mc.dbscan(target_names = [gene])
#     num_granules.append(len(sphere_dict[0]))
#     print(f"{idx + 1} genes processed!")
    
# # Gene ranking
# gene_ranking_raw = pd.DataFrame({"all_genes": all_genes, "num_granules": num_granules})
# gene_ranking_raw = gene_ranking_raw.sort_values(by = "num_granules", ascending = False)
# gene_ranking_raw["num_granules_cumulative"] = list(accumulate(gene_ranking_raw["num_granules"]))
# gene_ranking_raw.to_csv(output_path + "gene_ranking_raw.csv", index = 0)

# # Find threshold index
# target_index = find_threshold_index(list(gene_ranking_raw["num_granules_cumulative"]), threshold = 0.99)
# print(target_index, list(gene_ranking_raw["all_genes"])[:target_index + 1])