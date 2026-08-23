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
    fine pass  (size + in-soma + NC on) -> granules.parquet + rotation/flip
    profile -> granule_profile.h5ad          (RAW counts in X)

**It stops at the profile, deliberately.** The embedding that matters is the COMBINED WT+AD one
built by code/4_post_detection.ipynb cell 19 -- every published granule result (Fig. 3f subtypes,
Fig. 4d t-SNE) rests on that object, not on a per-sample embedding. Normalising and embedding
here would produce ten per-sample t-SNEs that nothing reads, and `sc.tl.tsne` defaults to
single-threaded sklearn, so they were also the slowest step in the analysis. `score_embedding.py`
pairs (WT seed s, AD seed s) into one combined object and embeds that instead.

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


def register(granules, dataset):
    """Add `global_x_new` / `global_y_new` -- the canvas-alignment geometry.

    Verbatim from code/3_detection.py:99-106. It does not affect anything A2b measures (these are
    obs columns), but cell 19's coordinate step consumes them, and keeping the permuted granule
    tables structurally identical to the published ones costs two lines and makes them usable by
    A2c.
    """
    reg = C.REGISTRATION[dataset]
    theta = np.deg2rad(reg["theta_deg"])
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta)],
                                [-np.sin(theta), np.cos(theta)]])
    coords = granules[reg["rotate_cols"]].to_numpy()
    transformed = coords @ rotation_matrix.T
    granules["global_" + reg["rotate_cols"][0].split("_")[1] + "_new"] = transformed[:, 0]
    granules["global_" + reg["rotate_cols"][1].split("_")[1] + "_new"] = transformed[:, 1]
    if reg["flip"]:
        col = reg["flip_col"] + "_new"
        granules[col] = reg["cutoff"] - granules[col]
    return granules


def log_sphere_counts(sphere_dict, tag):
    """Report what is about to enter `merge_sphere`.

    `_remove_overlaps` is a Python row loop, so a sphere-count blow-up is the realistic failure
    mode for this stage -- and permuted markers sit on a denser, soma-dominated cloud, so the
    count need not resemble the real run's. Printing it here makes a runaway visible in the first
    minutes of the log rather than after a 240 h timeout.
    """
    per_gene = {}
    for key, df in sphere_dict.items():
        name = str(df["gene"].iloc[0]) if len(df) else f"idx{key}"
        per_gene[name] = len(df)
    total = sum(per_gene.values())
    print(f"    [{tag}] {total:,} spheres entering merge_sphere: {per_gene}", flush=True)
    return total


def main(task_id):
    sample, seed = tasks()[task_id]
    dataset = C.dataset(sample)
    out_dir = C.perm_dir(sample, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "granule_profile.h5ad"
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
    # destroyed. Permuted IN PLACE against a pre-captured fingerprint -- holding a second copy of
    # a 103 M-row table purely to diff it afterwards would roughly double peak memory. The
    # assertions are cheap next to detection and the whole null rests on them, so they run every
    # time rather than behind a flag.
    # ---------------------------------------------------------------------------------------
    fingerprint = A2.permutation_fingerprint(transcripts)
    A2.permute_targets_inplace(transcripts, seed)
    unchanged = A2.assert_permutation_valid(fingerprint, transcripts)
    del fingerprint
    print(f"[{sample} seed {seed}] permutation integrity OK "
          f"({unchanged:.2%} of probed labels unchanged, i.e. chance level)", flush=True)
    marker_frac = float(transcripts["target"].isin(C.SYN_GENES).mean())
    print(f"[{sample} seed {seed}] marker share after permutation: {marker_frac:.4f} "
          f"(unchanged by construction)", flush=True)

    # -------------------- rough pass (all filters off) -------------------- #
    if C.RUN_ROUGH_PASS:
        print(f"[{sample} seed {seed}] rough detection...", flush=True)
        mc_rough = mcDETECT(transcripts=transcripts, gnl_genes=C.SYN_GENES, nc_genes=None,
                            **C.DETECT_KWARGS_ROUGH)
        sphere_dict = mc_rough.dbscan(record_cell_id=True)
        log_sphere_counts(sphere_dict, "rough")
        all_granules = mc_rough.merge_sphere(sphere_dict)
        write_parquet_atomic(all_granules, out_dir / "all_granules.parquet")
        print(f"[{sample} seed {seed}] rough granules: {all_granules.shape}", flush=True)
        del mc_rough, sphere_dict, all_granules
    else:
        # Disabling this is the single biggest lever on this stage: the rough pass merges the
        # unfiltered sphere set, which is far larger than the fine pass's. The cost is the
        # in-soma survival statistic, which score_embedding.py then cannot report.
        print(f"[{sample} seed {seed}] rough pass SKIPPED (C.RUN_ROUGH_PASS = False)", flush=True)

    # -------------------- fine pass (size + in-soma + NC) -------------------- #
    print(f"[{sample} seed {seed}] fine detection...", flush=True)
    mc = mcDETECT(transcripts=transcripts, gnl_genes=C.SYN_GENES, nc_genes=nc_genes,
                  **C.DETECT_KWARGS_FINE)
    sphere_dict = mc.dbscan()
    log_sphere_counts(sphere_dict, "fine")
    granules = mc.nc_filter(mc.merge_sphere(sphere_dict))
    del sphere_dict
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
    granules = register(granules, dataset)
    write_parquet_atomic(granules, out_dir / "granules.parquet")

    # -------------------- profile (raw counts, no embedding) -------------------- #
    print(f"[{sample} seed {seed}] profiling...", flush=True)
    granule_adata = mc.profile(granules, genes=genes)
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
