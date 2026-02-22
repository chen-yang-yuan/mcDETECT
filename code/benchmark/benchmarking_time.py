"""
Benchmark wall-clock time for neighbor graph computation and Leiden/Louvain clustering.

Reads the combined granule adata (normalized, log-transformed) from
output/MERSCOPE_WT_AD_comparison/granule_adata_tsne.h5ad, computes the neighbor graph
with default parameters for sparse expression matrix, then runs Leiden and Louvain
with default parameters. Records and prints time for each step.
"""

import scanpy as sc
import time

def main():
    adata = sc.read_h5ad("../../output/MERSCOPE_WT_AD_comparison/granule_adata_tsne.h5ad")
    print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")

    # Neighbor graph (default: n_neighbors=15, use_rep=None for sparse X)
    t0 = time.perf_counter()
    sc.pp.neighbors(adata, use_rep=None)
    t_neighbors = time.perf_counter() - t0
    print(f"Neighbor graph: {t_neighbors:.3f} s")

    # Leiden (default resolution=1)
    t0 = time.perf_counter()
    sc.tl.leiden(adata, key_added="leiden")
    t_leiden = time.perf_counter() - t0
    print(f"Leiden:          {t_leiden:.3f} s")

    # Louvain (default resolution=1)
    t0 = time.perf_counter()
    sc.tl.louvain(adata, key_added="louvain")
    t_louvain = time.perf_counter() - t0
    print(f"Louvain:         {t_louvain:.3f} s")

    print("\nSummary (seconds):")
    print(f"  neighbors: {t_neighbors:.3f}")
    print(f"  leiden:    {t_leiden:.3f}")
    print(f"  louvain:   {t_louvain:.3f}")


if __name__ == "__main__":
    sc.settings.verbosity = 0
    main()