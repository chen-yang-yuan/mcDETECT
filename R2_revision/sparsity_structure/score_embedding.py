#!/usr/bin/env python3
"""
A2b, stage 2 -- build one combined WT+AD arm, score its embedding structure, and embed it.

An *arm* is one **combined WT+AD object**, because that is the embedding the paper reports:
`code/4_post_detection.ipynb` cell 19 concatenates the two samples' granule profiles, normalises,
log1p's and runs PCA(10) + t-SNE(n_pcs=10), and every published granule result (Fig. 3f subtypes,
Fig. 4d t-SNE) rests on that object. So permutation replicate s pairs (WT seed s, AD seed s) into
one object: 10 detections -> 5 null embeddings -> 6 arms (1 real + 5 permuted).

Real and permuted are scored by this one script on one code path; that identity is the entire
point of the comparison.

**The size-matching problem.** The permuted arms will not hold the same number of granules as the
real arm -- the shuffle preserves the marker transcript count exactly but moves the markers onto
the panel-wide (soma-dominated) distribution, and the count can end up either side of the real
one. Both silhouette and ARI stability depend on n, so an unmatched comparison invites exactly
the objection A2b exists to close. Every permuted arm therefore also emits a **size-matched
pair** -- both arms cut to min(n_real, n_perm), stratified by batch so the WT:AD ratio matches
too. The matched pair is the headline; the full-n series ride along with `n_obs` on every row.

Exports, into output/a2b/metrics/ -- R reads CSV and Parquet only, never .h5ad, so anything it
plots is pre-exported here:

    <arm>_metrics.csv             series, k, inertia, silhouette, ARI stability, n_obs
    <arm>_status.csv              one row per series: n_obs, embedded/skipped, reason
    <arm>_detection_summary.csv   per constituent sample: counts + post-hoc filter predicates
    <arm>_distributions.parquet   pre-binned sphere_r / size / n_genes / n_reads, per sample
    <arm>_summary.csv             quantiles for the same four measures
    tsne_<series>.jpeg            rendered here (scanpy), not in R

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
import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a2_config as C
import a2_common as A2

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0


# ==================================================================================================
# Building the combined arm
# ==================================================================================================

def combined_path(seed):
    return C.combined_dir(seed) / "granule_adata.h5ad"


def sample_detection_paths(sample, seed):
    """Where one constituent sample's detection tables live."""
    d = C.mcdetect_out_dir(sample) if seed is None else C.perm_dir(sample, seed)
    return {"granules": d / "granules.parquet",
            "all_granules": d / "all_granules.parquet",
            "profile": (C.mcdetect_granule_adata_path(sample) if seed is None
                        else d / "granule_profile.h5ad")}


def build_combined(seed, n_jobs=1):
    """
    Materialise one arm's combined WT+AD object, cached on disk.

    Real arm: read the published `granule_adata_tsne.h5ad` directly. That is legitimate and worth
    stating -- the metrics read `adata[:, REF_GENES].X`, which is normalize_total + log1p and does
    not depend on the PCA or t-SNE stored in that file, and the file came from the same
    mc.profile -> concat -> normalise code path this function reproduces.

    Permuted arm: reproduce code/4_post_detection.ipynb cell 19 exactly.
    """
    out = combined_path(seed)
    if out.exists():
        print(f"  combined object cached -- {out}", flush=True)
        return sc.read_h5ad(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if seed is None:
        print(f"  reading the published combined object {C.COMBINED_GRANULE_ADATA}", flush=True)
        adata = sc.read_h5ad(C.COMBINED_GRANULE_ADATA)
    else:
        parts = {}
        for sample in C.SAMPLES:
            p = sample_detection_paths(sample, seed)["profile"]
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} is missing -- run slurm/run_permutation.sh first "
                    f"(this arm needs both WT and AD at seed {seed}).")
            a = sc.read_h5ad(p)
            gran = pd.read_parquet(sample_detection_paths(sample, seed)["granules"])
            # profile() renames sphere_x/y/z -> global_x/y/z, so this is the same check cell 19
            # makes before trusting the row order.
            mismatch = int((a.obs["global_x"].to_numpy() != gran["sphere_x"].to_numpy()).sum())
            if mismatch:
                raise ValueError(f"{sample} seed {seed}: profile/granule row mismatch ({mismatch})")
            for col in ["brain_area", "global_x_new", "global_y_new"]:
                a.obs[col] = gran[col].to_numpy()
            parts[C.dataset(sample)] = a

        wt, ad = parts[C.dataset("WT")], parts[C.dataset("AD")]
        # Coordinate alignment, verbatim from cell 19. NOTE the WT y-flip is applied here for the
        # SECOND time -- 3_detection.py already flipped `global_y_new` with the same cutoff, so
        # the two cancel. That is what the published code does and what produced the published
        # object, so it is reproduced rather than "fixed". None of it affects the embedding.
        wt.obs["global_y_new"] = C.COMBINED_CUTOFF - wt.obs["global_y_new"]
        wt.obs["global_x_adjusted"] = wt.obs["global_y_new"].copy()
        wt.obs["global_y_adjusted"] = wt.obs["global_x_new"].copy()
        ad.obs["global_x_adjusted"] = ad.obs["global_x_new"] + C.COMBINED_SHIFT_X
        ad.obs["global_y_adjusted"] = ad.obs["global_y_new"] + C.COMBINED_SHIFT_Y

        adata = anndata.concat({C.dataset("WT"): wt, C.dataset("AD"): ad},
                               axis=0, merge="same", label="batch")
        adata.layers["counts"] = csr_matrix(adata.X.copy())
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        assert adata.n_obs == wt.n_obs + ad.n_obs, "concat lost granules"
        del parts, wt, ad

    if "X_pca" not in adata.obsm and min(adata.n_obs, adata.n_vars) > 10:
        sc.tl.pca(adata, n_comps=10, svd_solver="auto")

    tmp = out.with_suffix(".h5ad.tmp")
    adata.write_h5ad(tmp)
    os.replace(tmp, out)
    print(f"  wrote {out} -- {adata.shape}", flush=True)
    return adata


def marker_matrix(adata):
    """The 34-marker feature space the metrics are computed on, dense."""
    var_names = [g for g in C.REF_GENES if g in adata.var_names]
    X = adata[:, var_names].X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return np.ascontiguousarray(X, dtype=np.float64), var_names


# ==================================================================================================
# Per-arm outputs
# ==================================================================================================

def detection_summary(sample, seed):
    """
    Granule counts, plus the filter predicates evaluated post hoc on the rough (unfiltered) set.

    An exact pipeline survival chain is not recoverable and is not claimed here: mcDETECT applies
    the size and in-soma filters inside `dbscan()`, i.e. BEFORE `merge_sphere()`, so the fine set
    is not a subset of the rough set. What is reported instead is honest and comparable across
    arms: how many rough aggregates exist, how many of them would pass each threshold as a
    post-hoc predicate, and how many granules the pipeline actually returns.

    The number that carries the argument is `frac_pass_in_soma`. Under a global label shuffle the
    marker transcripts inherit the panel-wide distribution, whose in-nucleus fraction is 0.279
    (WT) / 0.304 (AD) against 0.218 / 0.242 for the real markers -- so this is a measurement of
    how much of the permuted signal is somatic, not an argument about it.
    """
    paths = sample_detection_paths(sample, seed)
    row = {"arm": C.arm_name(seed), "condition": "real" if seed is None else "permuted",
           "seed": -1 if seed is None else seed, "sample": sample}

    fine = pd.read_parquet(paths["granules"])
    row["n_fine"] = int(fine.shape[0])
    row["median_fine_sphere_r"] = float(fine["sphere_r"].median())
    row["median_fine_size"] = float(fine["size"].median())
    row["median_fine_comp"] = float(fine["comp"].median())

    if paths["all_granules"].exists():
        rough = pd.read_parquet(paths["all_granules"])
        size_ok = rough["sphere_r"] < C.DETECT_KWARGS_FINE["size_thr"]
        soma_ok = rough["in_soma_ratio"] < C.DETECT_KWARGS_FINE["in_soma_thr"]
        row["n_rough"] = int(rough.shape[0])
        row["frac_pass_size"] = float(size_ok.mean())
        row["frac_pass_in_soma"] = float(soma_ok.mean())
        row["frac_pass_size_and_in_soma"] = float((size_ok & soma_ok).mean())
        row["median_rough_in_soma_ratio"] = float(rough["in_soma_ratio"].median())
    else:
        # C.RUN_ROUGH_PASS was off for this arm. Say so rather than emitting silent NaNs.
        row.update({"n_rough": np.nan, "frac_pass_size": np.nan, "frac_pass_in_soma": np.nan,
                    "frac_pass_size_and_in_soma": np.nan, "median_rough_in_soma_ratio": np.nan,
                    "rough_pass": "not run"})
    return row


def distributions(adata, seed):
    """Pre-binned sphere_r / size / n_reads / n_genes, split by constituent sample."""
    counts = adata.layers["counts"]
    genes_all = list(adata.var_names)
    nc_genes = [g for g in pd.read_csv(C.nc_path("WT"))["Gene"] if g in genes_all]
    n_reads, n_genes_panel = A2.unique_gene_counts(counts, genes_all)
    _, n_genes_nonNC = A2.unique_gene_counts(counts, genes_all, exclude_genes=nc_genes)
    n_genes = n_genes_nonNC if C.EXCLUDE_NC_FROM_COMPLEXITY else n_genes_panel

    batch = adata.obs["batch"].astype(str).to_numpy()
    summary_rows, hist_rows = [], []
    for sample in C.SAMPLES:
        m = batch == C.dataset(sample)
        keys = {"arm": C.arm_name(seed), "condition": "real" if seed is None else "permuted",
                "seed": -1 if seed is None else seed, "sample": sample}
        for measure, values in [("sphere_r", adata.obs["sphere_r"].to_numpy()[m]),
                                ("size", adata.obs["size"].to_numpy()[m]),
                                ("n_reads", n_reads[m]), ("n_genes", n_genes[m])]:
            s, h = A2.record_distribution(values, measure, C.HIST_BINS[measure], **keys)
            summary_rows.append(s)
            hist_rows.extend(h)
    return pd.DataFrame(summary_rows), pd.DataFrame(hist_rows)


def embeddable(n_obs):
    """Guard. A collapsed null is a RESULT and must be recorded as one, never a traceback."""
    if n_obs < C.MIN_EMBED_N:
        return False, f"n_obs {n_obs} < MIN_EMBED_N {C.MIN_EMBED_N}"
    return True, "ok"


def score_series(X, series, n_obs, n_jobs):
    df = A2.score_embedding_structure(
        X, C.SCORE_K_RANGE, n_init=C.KMEANS_N_INIT, batch_size=C.KMEANS_BATCH_SIZE,
        stability_seeds=C.STABILITY_SEEDS, silhouette_sample_size=C.SILHOUETTE_SAMPLE_SIZE,
        silhouette_seed=C.SILHOUETTE_SEED, n_jobs=n_jobs)
    df["series"] = series
    at_k = df.loc[df["n_clusters"] == C.K_SUBTYPE]
    if len(at_k):
        r = at_k.iloc[0]
        print(f"    {series} (n={n_obs:,}) k=15: silhouette {r['silhouette_score']:.4f}, "
              f"ARI stability {r['ari_stability_mean']:.4f}", flush=True)
    return df


def render_tsne(adata, series, out_dir, n_jobs):
    """Full-population t-SNE, same implementation and parameters as the published figure.

    `sc.settings.n_jobs` defaults to 1, which is the only reason the published run was
    single-threaded; `sc.tl.tsne` forwards `n_jobs` into sklearn's TSNE, which uses it for both
    the neighbour search and the Barnes-Hut gradient. method, random_state and every other
    parameter are unchanged, so this is a threading change and nothing else.
    """
    f = out_dir / f"tsne_{series}.jpeg"
    if f.exists():
        print(f"    t-SNE cached -- {f.name}", flush=True)
        return
    if adata.n_obs <= 3 * C.TSNE_PERPLEXITY:
        print(f"    t-SNE skipped for {series}: n_obs {adata.n_obs} <= 3 * perplexity", flush=True)
        return
    print(f"    t-SNE {series} on {adata.n_obs:,} granules, n_jobs={n_jobs}...", flush=True)
    sc.tl.tsne(adata, n_pcs=10, n_jobs=n_jobs)
    sc.set_figure_params(figsize=(8, 8))
    ax = sc.pl.embedding(adata, basis="tsne", size=1, show=False)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.gcf().savefig(f, dpi=500, bbox_inches="tight")
    plt.close()
    print(f"    wrote {f.name}", flush=True)


# ==================================================================================================
# Main
# ==================================================================================================

def main(task_id):
    seed = C.all_arms()[task_id]
    arm = C.arm_name(seed)
    C.ensure_dirs()
    out = C.A2B_METRICS_DIR
    n_jobs = C.resolve_n_jobs()
    print(f"[{arm}] scoring with n_jobs={n_jobs}", flush=True)

    for sample in C.SAMPLES:
        p = sample_detection_paths(sample, seed)["granules"]
        if not p.exists():
            raise FileNotFoundError(
                f"{arm}: {p} is missing. The real arm needs the published mcDETECT output under "
                f"output/<dataset>/; permuted arms need slurm/run_permutation.sh to have run.")

    # -------------------- 1. combined object -------------------- #
    adata = build_combined(seed, n_jobs=n_jobs)
    print(f"[{arm}] combined object: {adata.shape}", flush=True)

    # -------------------- 2. detection summary -------------------- #
    f = out / f"{arm}_detection_summary.csv"
    if not f.exists():
        pd.DataFrame([detection_summary(s, seed) for s in C.SAMPLES]).to_csv(f, index=False)

    # -------------------- 3. distributions -------------------- #
    f = out / f"{arm}_distributions.parquet"
    if not f.exists():
        summ, hist = distributions(adata, seed)
        summ.to_csv(out / f"{arm}_summary.csv", index=False)
        hist.to_parquet(f, index=False)

    # -------------------- 4. embedding metrics -------------------- #
    # Series for this arm: the arm at full n, and (for permuted arms) the size-matched pair.
    metrics_file = out / f"{arm}_metrics.csv"
    need_metrics = not metrics_file.exists()
    want_tsne = {C.series_name(seed)} if C.RUN_TSNE else set()
    if C.RUN_TSNE and seed in C.TSNE_MATCHED_SEEDS:
        want_tsne |= {C.series_name(seed, matched=True, of="perm"),
                      C.series_name(seed, matched=True, of="real")}
    tsne_pending = {s_ for s_ in want_tsne if not (out / f"tsne_{s_}.jpeg").exists()}
    need_matched = seed is not None and (
        need_metrics or bool(tsne_pending & {C.series_name(seed, matched=True, of="perm"),
                                             C.series_name(seed, matched=True, of="real")}))

    X, var_names = marker_matrix(adata)
    jobs = [(C.series_name(seed), X, adata)]

    if need_matched:
        real = build_combined(None, n_jobs=n_jobs)
        X_real, _ = marker_matrix(real)
        idx_perm, idx_real = A2.size_matched_indices(
            adata.obs["batch"].astype(str).to_numpy(),
            real.obs["batch"].astype(str).to_numpy(), seed=C.MATCH_SEED)
        n_match = idx_perm.size
        assert idx_perm.size == idx_real.size, "size matching produced unequal arms"
        print(f"[{arm}] size matching: n_perm={X.shape[0]:,}, n_real={X_real.shape[0]:,} "
              f"-> n_match={n_match:,}", flush=True)
        jobs.append((C.series_name(seed, matched=True, of="perm"), X[idx_perm], adata[idx_perm]))
        jobs.append((C.series_name(seed, matched=True, of="real"), X_real[idx_real],
                     real[idx_real]))

    if need_metrics:
        status_rows, metric_frames = [], []
        for series, Xs, _ in jobs:
            n_obs = Xs.shape[0]
            ok, reason = embeddable(n_obs)
            status_rows.append({"arm": arm, "series": series,
                                "condition": C.series_condition(series),
                                "seed": -1 if seed is None else seed, "n_obs": n_obs,
                                "status": "embedded" if ok else "skipped", "reason": reason,
                                "n_markers": len(var_names)})
            if not ok:
                print(f"    {series}: SKIPPED -- {reason}", flush=True)
                continue
            metric_frames.append(score_series(Xs, series, n_obs, n_jobs))

        pd.DataFrame(status_rows).to_csv(out / f"{arm}_status.csv", index=False)
        if metric_frames:
            df = pd.concat(metric_frames, ignore_index=True)
            df["arm"] = arm
            df["condition"] = df["series"].map(C.series_condition)
            df["seed"] = -1 if seed is None else seed
            df["matched"] = df["series"].str.startswith("matched_")
            df.to_csv(metrics_file, index=False)
        else:
            print(f"[{arm}] no series was embeddable -- see {arm}_status.csv", flush=True)
    else:
        print(f"[{arm}] metrics cached -- {metrics_file.name}", flush=True)

    # -------------------- 5. t-SNE -------------------- #
    for series, _, ad_s in jobs:
        if series in want_tsne and ad_s.n_obs >= C.MIN_EMBED_N:
            render_tsne(ad_s.copy(), series, out, n_jobs)

    print(f"[{arm}] done", flush=True)


def concat():
    """Stitch the per-arm tables, and say plainly which arms are missing."""
    out = C.A2B_METRICS_DIR
    expected = [C.arm_name(seed) for seed in C.all_arms()]
    for stem, ext, writer in [("metrics", "csv", "a2b_metrics.csv"),
                              ("status", "csv", "a2b_status.csv"),
                              ("detection_summary", "csv", "a2b_detection_summary.csv"),
                              ("summary", "csv", "a2b_summary.csv"),
                              ("distributions", "parquet", "a2b_distributions.parquet")]:
        parts, missing = [], []
        for arm in expected:
            p = out / f"{arm}_{stem}.{ext}"
            if p.exists():
                parts.append(pd.read_csv(p) if ext == "csv" else pd.read_parquet(p))
            else:
                missing.append(arm)
        if not parts:
            print(f"{stem}: nothing to concatenate  MISSING: {missing}")
            continue
        df = pd.concat(parts, ignore_index=True)
        if ext == "parquet":
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
