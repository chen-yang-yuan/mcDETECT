import scanpy as sc
import time

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# ============================== Benchmark time on sparse expression ============================== #

adata = sc.read_h5ad("../../output/MERSCOPE_WT_AD_comparison/granule_adata_tsne.h5ad")
print(type(adata.X))
print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

# Neighbor graph (default: n_neighbors=15)
t0 = time.perf_counter()
sc.pp.neighbors(adata, use_rep="X")
t_neighbors = time.perf_counter() - t0
print(f"Neighbor graph: {t_neighbors:.3f} s")

# Leiden
t0 = time.perf_counter()
sc.tl.leiden(adata, resolution=0.1, key_added="leiden")
t_leiden = time.perf_counter() - t0
print(f"Leiden: {t_leiden:.3f} s")

# Louvain
t0 = time.perf_counter()
sc.tl.louvain(adata, resolution=0.1, key_added="louvain")
t_louvain = time.perf_counter() - t0
print(f"Louvain: {t_louvain:.3f} s")

print("\nSummary (seconds):")
print(f"  neighbors: {t_neighbors:.3f}")
print(f"  leiden:    {t_leiden:.3f}")
print(f"  louvain:   {t_louvain:.3f}")

print(f"Leiden clusters: {adata.obs['leiden'].nunique()}")
print(f"Louvain clusters: {adata.obs['louvain'].nunique()}")

# ============================== Benchmark time on dense expression ============================== #

adata_dense = adata.copy()
adata_dense.X = adata_dense.X.toarray()
print(type(adata_dense.X))
print(f"  Shape: {adata_dense.shape[0]} cells x {adata_dense.shape[1]} genes")

# Neighbor graph (default: n_neighbors=15)
t0 = time.perf_counter()
sc.pp.neighbors(adata_dense, use_rep="X")
t_neighbors = time.perf_counter() - t0
print(f"Neighbor graph: {t_neighbors:.3f} s")

# Leiden
t0 = time.perf_counter()
sc.tl.leiden(adata_dense, resolution=0.1, key_added="leiden")
t_leiden = time.perf_counter() - t0
print(f"Leiden: {t_leiden:.3f} s")

# Louvain
t0 = time.perf_counter()
sc.tl.louvain(adata_dense, resolution=0.1, key_added="louvain")
t_louvain = time.perf_counter() - t0
print(f"Louvain: {t_louvain:.3f} s")

print("\nSummary (seconds):")
print(f"  neighbors: {t_neighbors:.3f}")
print(f"  leiden:    {t_leiden:.3f}")
print(f"  louvain:   {t_louvain:.3f}")

print(f"Leiden clusters: {adata_dense.obs['leiden'].nunique()}")
print(f"Louvain clusters: {adata_dense.obs['louvain'].nunique()}")