#!/usr/bin/env python3
"""
A2b, stage 2 -- score one arm's embedding structure.

An *arm* is either a real sample (the published mcDETECT output) or one permuted detection from
`run_permutation_detect.py`. Both are scored by this one script on one code path: that identity
is the entire point of the comparison, so the real arm is re-scored here rather than quoting the
numbers already sitting in output/benchmark/benchmark_clustering/benchmark_clustering_results.csv
(which used a different `n_init` and no silhouette subsampling).

Exports, into output/a2b/metrics/, everything A2_figures.R needs -- R reads CSV and Parquet only,
never .h5ad, so anything it plots must be pre-exported here:

    <arm>_metrics.csv             k, inertia, silhouette, ARI stability
    <arm>_detection_summary.csv   granule counts and post-hoc filter predicates
    <arm>_distributions.parquet   pre-binned sphere_r / size / n_genes / n_reads
    <arm>_summary.csv             quantiles for the same four measures
    tsne_<arm>.jpeg               rendered here (scanpy), not in R

Usage
-----
    python3 score_embedding.py [task_id]       # defaults to $SLURM_ARRAY_TASK_ID
    python3 score_embedding.py --concat        # stitch the per-arm tables together
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2_config as C
import a2_common as A2

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0


def arm_paths(sample, seed):
    """Where this arm's three inputs live. Real arms read the published mcDETECT output."""
    if seed is None:
        d = C.mcdetect_out_dir(sample)
    else:
        d = C.perm_dir(sample, seed)
    return {"adata": d / "granule_adata_tsne.h5ad",
            "granules": d / "granules.parquet",
            "all_granules": d / "all_granules.parquet"}


def detection_summary(paths, keys):
    """
    Granule counts, plus the filter predicates evaluated post hoc on the rough (unfiltered) set.

    An exact pipeline survival chain is not recoverable and is not claimed here: mcDETECT applies
    the size and in-soma filters inside `dbscan()`, i.e. BEFORE `merge_sphere()`, so the fine set
    is not a subset of the rough set. What is reported instead is honest and comparable across
    arms: how many rough aggregates exist, how many of them would pass each threshold as a
    post-hoc predicate, and how many granules the real pipeline actually returns.

    The number that carries the argument is `frac_pass_in_soma`. Under a global label shuffle the
    marker transcripts inherit the panel-wide -- i.e. soma-dominated -- spatial distribution, so
    this is a measurement of how much of the permuted signal is somatic, not an argument about it.
    """
    row = dict(keys)
    rough = pd.read_parquet(paths["all_granules"])
    fine = pd.read_parquet(paths["granules"])
    row["n_rough"] = int(rough.shape[0])
    row["n_fine"] = int(fine.shape[0])

    size_ok = rough["sphere_r"] < C.DETECT_KWARGS_FINE["size_thr"]
    soma_ok = rough["in_soma_ratio"] < C.DETECT_KWARGS_FINE["in_soma_thr"]
    row["frac_pass_size"] = float(size_ok.mean())
    row["frac_pass_in_soma"] = float(soma_ok.mean())
    row["frac_pass_size_and_in_soma"] = float((size_ok & soma_ok).mean())
    row["median_rough_in_soma_ratio"] = float(rough["in_soma_ratio"].median())
    row["median_fine_sphere_r"] = float(fine["sphere_r"].median())
    row["median_fine_size"] = float(fine["size"].median())
    row["median_fine_comp"] = float(fine["comp"].median())
    return row


def main(task_id):
    sample, seed = C.all_arms()[task_id]
    arm = C.arm_name(sample, seed)
    paths = arm_paths(sample, seed)
    C.ensure_dirs()
    out = C.A2B_METRICS_DIR

    for name, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(
                f"{arm}: {p} is missing. Real arms need the published mcDETECT output under "
                f"output/<dataset>/; permuted arms need slurm/run_permutation.sh to have run "
                f"first.")

    keys = {"arm": arm, "sample": sample,
            "condition": "real" if seed is None else "permuted",
            "seed": -1 if seed is None else seed}
    print(f"[{arm}] scoring", flush=True)

    # -------------------- detection level -------------------- #
    pd.DataFrame([detection_summary(paths, keys)]).to_csv(
        out / f"{arm}_detection_summary.csv", index=False)

    # -------------------- distributions -------------------- #
    adata = sc.read_h5ad(paths["adata"])
    counts = adata.layers["counts"]
    genes_all = list(adata.var_names)
    nc_genes = [g for g in pd.read_csv(C.nc_path(sample))["Gene"] if g in genes_all]
    n_reads, n_genes_panel = A2.unique_gene_counts(counts, genes_all)
    _, n_genes_nonNC = A2.unique_gene_counts(counts, genes_all, exclude_genes=nc_genes)
    n_genes = n_genes_nonNC if C.EXCLUDE_NC_FROM_COMPLEXITY else n_genes_panel

    summary_rows, hist_rows = [], []
    for measure, values in [("sphere_r", adata.obs["sphere_r"].to_numpy()),
                            ("size", adata.obs["size"].to_numpy()),
                            ("n_reads", n_reads), ("n_genes", n_genes)]:
        s, h = A2.record_distribution(values, measure, C.HIST_BINS[measure], **keys)
        summary_rows.append(s)
        hist_rows.extend(h)
    pd.DataFrame(summary_rows).to_csv(out / f"{arm}_summary.csv", index=False)
    pd.DataFrame(hist_rows).to_parquet(out / f"{arm}_distributions.parquet", index=False)

    # -------------------- embedding structure -------------------- #
    # X is already normalize_total(1e4) + log1p in every arm's h5ad (both 3_detection.py and
    # run_permutation_detect.py end that way), so the marker matrix is taken as-is.
    var_names = [g for g in C.REF_GENES if g in adata.var_names]
    X = adata[:, var_names].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    print(f"[{arm}] marker matrix {X.shape}; scoring k = "
          f"{C.SCORE_K_RANGE.start}..{C.SCORE_K_RANGE.stop - 1}", flush=True)

    metrics = A2.score_embedding_structure(
        X, C.SCORE_K_RANGE, n_init=C.KMEANS_N_INIT, batch_size=C.KMEANS_BATCH_SIZE,
        stability_seeds=C.STABILITY_SEEDS,
        silhouette_sample_size=C.SILHOUETTE_SAMPLE_SIZE, silhouette_seed=C.SILHOUETTE_SEED)
    for k, v in keys.items():
        metrics[k] = v
    metrics["n_markers"] = len(var_names)
    metrics.to_csv(out / f"{arm}_metrics.csv", index=False)
    at15 = metrics.loc[metrics["n_clusters"] == C.K_SUBTYPE]
    if len(at15):
        r = at15.iloc[0]
        print(f"[{arm}] k=15: silhouette {r['silhouette_score']:.4f}, "
              f"ARI stability {r['ari_stability_mean']:.4f}", flush=True)

    # -------------------- t-SNE -------------------- #
    # Same rendering for every arm so real and permuted can sit side by side.
    if "X_tsne" in adata.obsm:
        sc.set_figure_params(figsize=(8, 8))
        ax = sc.pl.embedding(adata, basis="tsne", size=1, show=False)
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.gcf().savefig(out / f"tsne_{arm}.jpeg", dpi=500, bbox_inches="tight")
        plt.close()
    print(f"[{arm}] done", flush=True)


def concat():
    """Stitch the per-arm tables, and say plainly which arms are missing."""
    out = C.A2B_METRICS_DIR
    expected = [C.arm_name(s, seed) for s, seed in C.all_arms()]
    for stem, reader, writer in [("metrics", pd.read_csv, "a2b_metrics.csv"),
                                 ("detection_summary", pd.read_csv, "a2b_detection_summary.csv"),
                                 ("summary", pd.read_csv, "a2b_summary.csv"),
                                 ("distributions", pd.read_parquet, "a2b_distributions.parquet")]:
        parts, missing = [], []
        for arm in expected:
            ext = "parquet" if stem == "distributions" else "csv"
            p = out / f"{arm}_{stem}.{ext}"
            (parts if p.exists() else missing).append(reader(p) if p.exists() else arm)
        if not parts:
            print(f"{stem}: nothing to concatenate")
            continue
        df = pd.concat(parts, ignore_index=True)
        if writer.endswith(".parquet"):
            df.to_parquet(out / writer, index=False)
        else:
            df.to_csv(out / writer, index=False)
        print(f"{stem}: {len(parts)}/{len(expected)} arms -> {writer}"
              + (f"  MISSING: {missing}" if missing else ""))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--concat":
        concat()
    else:
        task_id = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
        n = len(C.all_arms())
        if not 0 <= task_id < n:
            raise SystemExit(f"task_id {task_id} out of range (0..{n - 1})")
        main(task_id)
