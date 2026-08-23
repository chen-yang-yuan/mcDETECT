#!/usr/bin/env python3
"""
A2b, stage 1 -- one permuted detection.

Reviewer #2: *"I am not convinced randomized data ... would not produce an essentially identical
embedding."* This script builds exactly that randomized data and runs the **identical** mcDETECT
chain on it, so the two can be compared on equal terms.

One SLURM array task = one (sample x permutation seed) unit. The permuted transcript table is
generated from the seed inside the job and never written to disk: five copies of a 1.75 GB table
buy nothing that the seed does not already record.

The chain is copied from code/3_detection.py:
    rough pass (all filters off) -> all_granules.parquet
    fine pass  (size + in-soma + NC on) -> granules.parquet
    profile -> normalize -> log1p -> PCA(10) -> t-SNE -> granule_adata_tsne.h5ad

Two deliberate departures from 3_detection.py, both irrelevant to what A2b measures:
  * the per-dataset rotation / flip is skipped -- it is canvas-alignment geometry for putting WT
    and AD on one canvas, and A2b never combines the samples;
  * `brain_area` is still assigned (cheap 1-NN against spots.h5ad), because the permuted granule
    tables are also the null that A2c will need.

Usage
-----
    python3 run_permutation_detect.py [task_id]        # defaults to $SLURM_ARRAY_TASK_ID
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2_config as C
import a2_common as A2

from mcDETECT.model import mcDETECT


def tasks():
    """Array index -> (sample, seed). Kept in one place so the SLURM array size is derivable."""
    return [(s, seed) for s in C.SAMPLES for seed in C.PERM_SEEDS]


def write_parquet_atomic(df, path):
    """Write to a temp file then rename.

    Same guard as R2_revision/baysor_ssam_merscope/common.py:15-24: an OOM- or timeout-killed
    task must not leave a truncated file that the skip-if-exists resume would then trust.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


def main(task_id):
    sample, seed = tasks()[task_id]
    out_dir = C.perm_dir(sample, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "granule_adata_tsne.h5ad"
    if done.exists():
        print(f"[{sample} seed {seed}] already finished -- {done}")
        return

    t0 = time.time()
    print(f"[{sample} seed {seed}] loading {C.transcripts_path(sample)}", flush=True)
    transcripts = pd.read_parquet(C.transcripts_path(sample))
    genes = list(pd.read_csv(C.genes_path(sample)).iloc[:, 0])
    nc_genes = list(pd.read_csv(C.nc_path(sample))["Gene"])
    print(f"[{sample} seed {seed}] {transcripts.shape[0]:,} transcripts, {len(genes)} genes",
          flush=True)

    # ---------------------------------------------------------------------------------------
    # The null. Global shuffle of the gene label within the sample: every molecule position, the
    # total density, and each gene's total count survive; only the label-position association is
    # destroyed. The assertions are cheap next to detection and the whole null rests on them, so
    # they run every time rather than behind a flag.
    # ---------------------------------------------------------------------------------------
    permuted = A2.permute_targets(transcripts, seed)
    A2.assert_permutation_valid(transcripts, permuted)
    print(f"[{sample} seed {seed}] permutation integrity OK", flush=True)
    marker_frac = float(permuted["target"].isin(C.SYN_GENES).mean())
    print(f"[{sample} seed {seed}] marker share after permutation: {marker_frac:.4f} "
          f"(unchanged by construction)", flush=True)
    del transcripts

    # -------------------- rough pass (all filters off) -------------------- #
    print(f"[{sample} seed {seed}] rough detection...", flush=True)
    mc_rough = mcDETECT(transcripts=permuted, gnl_genes=C.SYN_GENES, nc_genes=None,
                        **C.DETECT_KWARGS_ROUGH)
    sphere_dict = mc_rough.dbscan(record_cell_id=True)
    all_granules = mc_rough.merge_sphere(sphere_dict)
    write_parquet_atomic(all_granules, out_dir / "all_granules.parquet")
    print(f"[{sample} seed {seed}] rough granules: {all_granules.shape}", flush=True)
    del mc_rough, sphere_dict

    # -------------------- fine pass (size + in-soma + NC) -------------------- #
    print(f"[{sample} seed {seed}] fine detection...", flush=True)
    mc = mcDETECT(transcripts=permuted, gnl_genes=C.SYN_GENES, nc_genes=nc_genes,
                  **C.DETECT_KWARGS_FINE)
    granules = mc.detect()
    print(f"[{sample} seed {seed}] fine granules: {granules.shape}", flush=True)

    # Region labels, by nearest spot -- same 1-NN as code/3_detection.py:89-97.
    spots = sc.read_h5ad(C.spots_path(sample))
    labels_df = pd.DataFrame({"global_x": spots.obs["global_x"].to_numpy(),
                              "global_y": spots.obs["global_y"].to_numpy(),
                              "brain_area": spots.obs["brain_area"].to_numpy()})
    tree = cKDTree(labels_df[["global_x", "global_y"]].to_numpy())
    _, nn_idx = tree.query(granules[["sphere_x", "sphere_y"]].to_numpy(), k=1)
    granules = granules.copy()
    granules["brain_area"] = labels_df.loc[nn_idx, "brain_area"].to_numpy()
    write_parquet_atomic(granules, out_dir / "granules.parquet")

    # -------------------- profile + embedding -------------------- #
    print(f"[{sample} seed {seed}] profiling...", flush=True)
    granule_adata = mc.profile(granules, genes=genes)
    granule_adata.layers["counts"] = csr_matrix(granule_adata.X.copy())
    sc.pp.normalize_total(granule_adata, target_sum=1e4)
    sc.pp.log1p(granule_adata)
    sc.tl.pca(granule_adata, n_comps=10, svd_solver="auto")
    sc.tl.tsne(granule_adata, n_pcs=10)

    tmp = done.with_suffix(".h5ad.tmp")
    granule_adata.write_h5ad(tmp)
    os.replace(tmp, done)
    print(f"[{sample} seed {seed}] wrote {done} -- {granule_adata.shape} "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    else:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    n = len(tasks())
    if not 0 <= task_id < n:
        raise SystemExit(f"task_id {task_id} out of range (0..{n - 1})")
    main(task_id)
