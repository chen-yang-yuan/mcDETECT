"""
Benchmark number of clusters for granule clustering using Minibatch K-Means.

For detected granules in MERSCOPE WT 1 sample:
1. Build granule expression profile using mcDETECT's profile().
2. Normalize with Scanpy (normalize_total, log1p) before subsetting.
3. Subset the profile to a list of marker genes.
4. Run Minibatch K-Means for a range of cluster numbers; record:
   - Inertia (elbow method), Silhouette score, and cluster stability (mean pairwise ARI across runs with different random seeds).
5. Save all metrics to a single CSV and visualize. If stability stabilizes around a given K (e.g. K ≈ 15), that supports that choice biologically and statistically.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scanpy as sc
from mcDETECT.model import mcDETECT
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score, silhouette_samples

import warnings
warnings.filterwarnings("ignore")

# File paths
dataset = "MERSCOPE_WT_1"
data_path = f"../../data/{dataset}/"
output_path = f"../../output/{dataset}/"
benchmark_path = "../../output/benchmark/benchmark_clustering/"
os.makedirs(benchmark_path, exist_ok=True)

# ==================== Read data ==================== #

# Granules
granules = pd.read_parquet(output_path + "granules.parquet")
print(f"Number of granules: {len(granules)}")

# Transcripts
transcripts = pd.read_parquet(data_path + "processed_data/transcripts.parquet")
if "target" not in transcripts.columns and "gene" in transcripts.columns:
    transcripts["target"] = transcripts["gene"]

# Genes
genes = pd.read_csv(data_path + "processed_data/genes.csv")
genes = list(genes.iloc[:, 0])
print(f"Total genes in panel: {len(genes)}")

# Negative control genes
nc_genes = list(pd.read_csv(data_path + "processed_data/negative_controls.csv")["Gene"])
gnl_genes = ["Camk2a", "Cplx2", "Slc17a7", "Ddn", "Syp", "Map1a", "Shank1", "Syn1", "Gria1", "Gria2", "Cyfip2", "Vamp2", "Bsn", "Slc32a1", "Nfasc", "Syt1", "Tubb3", "Nav1", "Shank3", "Mapt"]

# Process transcripts
if "overlaps_nucleus" not in transcripts.columns and "overlaps_nucleus_5_dilation" in transcripts.columns:
    transcripts["overlaps_nucleus"] = transcripts["overlaps_nucleus_5_dilation"]
if "layer_z" not in granules.columns and "sphere_z" in granules.columns:
    granules["layer_z"] = granules["sphere_z"]

# ==================== Build expression profile ==================== #

# Run mcDETECT
mc = mcDETECT(type="discrete", transcripts=transcripts, gnl_genes=gnl_genes, nc_genes=nc_genes, eps=1.5,
              minspl=3, grid_len=1, cutoff_prob=0.95, alpha=10, low_bound=3, size_thr=4.0,
              in_soma_thr=0.1, l=1, rho=0.2, s=1, nc_top=20, nc_thr=0.1)

granule_adata = mc.profile(granules, genes=genes)
print(f"Expression profile shape: {granule_adata.shape}")

# Normalization
sc.pp.normalize_total(granule_adata, target_sum=1e4)
sc.pp.log1p(granule_adata)

# Subsetting
marker_genes = ["Bsn", "Gap43", "Nrxn1", "Slc17a6", "Slc17a7", "Slc32a1", "Stx1a", "Syn1", "Syp", "Syt1",
                "Vamp2", "Cplx2", "Camk2a", "Dlg3", "Dlg4", "Gphn", "Gria1", "Gria2", "Homer1", "Homer2",
                "Nlgn1", "Nlgn2", "Nlgn3", "Shank1", "Shank3", "Cyfip2", "Ddn", "Map1a", "Map2", "Ank3",
                "Nav1", "Nfasc", "Mapt", "Tubb3"]
marker_genes = list(dict.fromkeys(marker_genes))

marker_genes_in_data = [g for g in marker_genes if g in granule_adata.var_names]
missing = set(marker_genes) - set(marker_genes_in_data)
if missing:
    print(f"Marker genes not in profile (skipped): {missing}")
print(f"Using {len(marker_genes_in_data)} marker genes")

# Final expression matrix
adata_markers = granule_adata[:, marker_genes_in_data].copy()
X = adata_markers.X.toarray() if hasattr(adata_markers.X, "toarray") else np.asarray(adata_markers.X)
print(f"Final expression matrix shape: {X.shape}")

# ==================== Run Minibatch K-Means ==================== #

k_range = range(2, 31)
random_state = 0
batch_size = 5000

# Cluster stability: multiple seeds, mean pairwise ARI per k
n_seeds = 5
stability_seeds = [0, 42, 123, 456, 789][:n_seeds]

results = []
for k in k_range:
    km = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=batch_size)
    labels = km.fit_predict(X)
    inertia = km.inertia_
    # sil = silhouette_score(X, labels)

    # Cluster stability: re-run with different random seeds, compute mean pairwise ARI across runs
    label_list = [labels]
    for seed in stability_seeds[1:]:
        km_r = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=batch_size)
        label_list.append(km_r.fit_predict(X))
    ari_values = []
    for i in range(len(label_list)):
        for j in range(i + 1, len(label_list)):
            ari_values.append(adjusted_rand_score(label_list[i], label_list[j]))
    ari_stability_mean = np.mean(ari_values) if ari_values else np.nan

    results.append({
        "n_clusters": k,
        "inertia": inertia,
        # "silhouette_score": sil,
        "ari_stability_mean": ari_stability_mean,
    })

metrics_df = pd.DataFrame(results)

# Save results
out_csv = os.path.join(benchmark_path, "benchmark_clustering_results.csv")
metrics_df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")

# # Visualize Silhouette score distribution for target k
# target_k = 15
# km_target = MiniBatchKMeans(n_clusters=target_k, random_state=random_state, batch_size=batch_size)
# labels_target = km_target.fit_predict(X)
# sil_samples = silhouette_samples(X, labels_target)

# fig, ax = plt.subplots(figsize=(8, 6))
# ax.hist(sil_samples, bins=50, color="#9b59b6", alpha=0.7, edgecolor="black")
# ax.axvline(np.mean(sil_samples), color="red", linestyle="--", label=f"Mean = {np.mean(sil_samples):.4f}")
# ax.set_xlabel("Silhouette coefficient")
# ax.set_ylabel("Count")
# ax.set_title(f"Silhouette distribution (k = {target_k})")
# ax.legend()
# ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(benchmark_path + f"benchmark_clustering_silhouette_distribution_k_{target_k}.jpeg", dpi=500, bbox_inches="tight")
# plt.close()