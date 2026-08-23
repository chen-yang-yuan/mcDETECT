"""
Shared computation for analysis A2.

Everything here that reproduces published behaviour is a *port*, and each port names its source
file and lines. The ports exist because the originals live in notebooks, which cannot be
imported; nothing is re-derived or "improved" silently. Where a port is vectorised for speed
(the density primitives), a slow reference implementation sits beside it and
`A2a_multigene.ipynb` section 7 asserts the two agree.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.stats import ttest_ind
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits


# ==================================================================================================
# Granule complexity
# ==================================================================================================

def unique_gene_counts(counts, gene_names=None, exclude_genes=None):
    """
    Per-granule read count and unique-gene count from a granule x gene count matrix.

    This is the quantity the reviewer asked us to stratify on. It is NOT `granules.parquet["comp"]`
    -- see the note in a2_config.py: `comp` counts distinct granule MARKERS (<= 20) because
    mcDETECT_package/mcDETECT/model.py:102 restricts the frame to `gnl_genes` before detection,
    and it is never recomputed after merge_sphere(). The count below is over the panel, exactly
    as code/4_post_detection.ipynb cells 21-22 computes it from `mcDETECT.model.profile` output.

    Parameters
    ----------
    counts : sparse or dense (n_granules, n_genes) matrix -- raw counts, NOT log-normalised.
    gene_names, exclude_genes : if both given, columns in `exclude_genes` are dropped before
        counting unique genes (used to exclude the 19 negative-control genes).

    Returns
    -------
    n_reads, n_genes : int arrays of length n_granules.
    """
    if exclude_genes:
        if gene_names is None:
            raise ValueError("gene_names is required when exclude_genes is given")
        keep = ~pd.Index(gene_names).isin(list(exclude_genes))
        counts = counts[:, np.flatnonzero(keep)]

    if hasattr(counts, "getnnz"):                       # sparse
        n_reads = np.asarray(counts.sum(axis=1)).ravel()
        n_genes = np.asarray((counts > 0).sum(axis=1)).ravel()
    else:
        n_reads = np.asarray(counts).sum(axis=1)
        n_genes = (np.asarray(counts) > 0).sum(axis=1)
    return n_reads.astype(np.int64), n_genes.astype(np.int32)


def read_terciles(n_reads, labels=("low", "mid", "high")):
    """Split granules into read-count terciles. Ties go to the lower stratum (`qcut` semantics),
    and duplicate edges are tolerated because read counts are small integers."""
    ranks = pd.Series(n_reads).rank(method="first")
    return pd.qcut(ranks, q=3, labels=list(labels)).astype(str).to_numpy()


# ==================================================================================================
# Subtyping -- ported from code/benchmark/benchmark_subtyping.ipynb cell 15
# ==================================================================================================

def run_manual_subtyping(granule_adata, n_clusters, seed, batch_size=5000, n_init=20,
                         obs_key="granule_subtype_kmeans"):
    """K-means on the marker matrix -> obs[obs_key]; all randomness controlled by seed.

    Ported verbatim from code/benchmark/benchmark_subtyping.ipynb cell 15. `granule_adata` must
    already be subset to the 34 REF_GENES, and must have been normalised on the FULL panel
    before subsetting -- reversing that order changes the clustering.
    """
    data = granule_adata.X.copy()
    if hasattr(data, "toarray"):
        data = data.toarray()
    np.random.seed(seed)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=seed,
                             n_init=n_init)
    kmeans.fit(data)
    granule_adata.obs[obs_key] = kmeans.labels_.astype(str)
    desired_order = [str(i) for i in range(n_clusters)]
    granule_adata.obs[obs_key] = pd.Categorical(granule_adata.obs[obs_key],
                                                categories=desired_order, ordered=True)
    return granule_adata


def apply_manual_annotation(granule_adata, mapping, n_clusters,
                            cluster_column="granule_subtype_kmeans", strict=True):
    """
    Map cluster ids to compartment subtypes; add obs['granule_subtype_manual'] and
    obs['granule_subtype_manual_simple'].

    Ported from code/benchmark/benchmark_subtyping.ipynb cell 15, with validation added. The
    original silently produced an all-NaN column on a bad mapping; here a mistake in the
    hand-filled dict RAISES, because a stale or misspelt mapping otherwise fails invisibly and
    every downstream density and DE number inherits the error.
    """
    valid = {str(i) for i in range(n_clusters)}
    seen = {}
    for subtype, clusters in mapping.items():
        for c in clusters:
            c = str(c)
            if strict and c not in valid:
                raise ValueError(
                    f"MANUAL_SUBTYPE_MAPPING: cluster {c!r} (under {subtype!r}) is not in "
                    f"0..{n_clusters - 1}")
            if c in seen:
                raise ValueError(
                    f"MANUAL_SUBTYPE_MAPPING: cluster {c!r} listed twice -- under {seen[c]!r} "
                    f"and {subtype!r}")
            seen[c] = subtype

    if strict:
        missing = sorted(valid - set(seen), key=int)
        if missing:
            raise ValueError(
                f"MANUAL_SUBTYPE_MAPPING is incomplete: clusters {missing} are unassigned. "
                f"Put genuinely uninterpretable clusters under 'others' rather than omitting "
                f"them, so the count is explicit.")

    granule_adata.obs["granule_subtype_manual"] = (
        granule_adata.obs[cluster_column].astype(str).map(seen))
    granule_adata.obs["granule_subtype_manual_simple"] = (
        granule_adata.obs["granule_subtype_manual"].apply(
            lambda s: "mixed" if pd.notna(s) and " & " in str(s) else str(s)))
    return granule_adata


def ordered_cluster_ids(mapping, n_clusters, subtype_order):
    """Cluster ids re-ordered by compartment, for the 'ordered' verification heatmap.

    Ported from code/benchmark/benchmark_subtyping.ipynb cell 22 (main-result branch).
    """
    cluster_to_simple = {}
    for subtype, clusters in mapping.items():
        simple = "mixed" if (pd.notna(subtype) and " & " in str(subtype)) else str(subtype)
        for c in clusters:
            cluster_to_simple[str(c)] = simple
    out = []
    for simple in subtype_order:
        out.extend(sorted([c for c, s in cluster_to_simple.items() if s == simple], key=int))
    return out or [str(i) for i in range(n_clusters)]


def top_marker_table(granule_adata, ref_genes, cluster_column="granule_subtype_kmeans",
                     thr=0.5, max_markers=6):
    """
    Machine-readable reading aid for filling in MANUAL_SUBTYPE_MAPPING.

    Column-scales the per-cluster mean expression the same way `sc.pl.heatmap(...,
    standard_scale="var")` does, then lists each cluster's defining markers. Reading the mapping
    off this table is far less error-prone than eyeballing the figure.
    """
    X = granule_adata[:, ref_genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    df = pd.DataFrame(X, columns=ref_genes)
    df["_cluster"] = granule_adata.obs[cluster_column].astype(str).to_numpy()
    means = df.groupby("_cluster").mean()
    rng = means.max(axis=0) - means.min(axis=0)
    scaled = (means - means.min(axis=0)) / rng.replace(0, np.nan)

    rows = []
    for cluster in sorted(scaled.index, key=int):
        row = scaled.loc[cluster].dropna().sort_values(ascending=False)
        top = row[row >= thr].head(max_markers)
        if len(top) == 0:
            top = row.head(1)
        rows.append({"cluster": cluster,
                     "n_granules": int((df["_cluster"] == cluster).sum()),
                     "top_markers": ", ".join(f"{g} ({v:.2f})" for g, v in top.items())})
    return pd.DataFrame(rows)


# ==================================================================================================
# Density -- ported from code/benchmark/benchmark_subtyping.ipynb cell 19, vectorised
# ==================================================================================================

def spot_counts(granule_xy, spot_xy, grid_len=50):
    """
    Number of granules falling in each spot's box, `[cx - h, cx + h) x [cy - h, cy + h)`.

    Vectorised equivalent of the per-spot loop in `compute_subtype_density_per_region` /
    `compute_subtype_per_spot_counts` (benchmark_subtyping.ipynb cell 19). The published code
    tests every (spot, granule) pair; here each granule queries its 4 nearest spot centres and
    is tested against those boxes only. For a regular grid of pitch `grid_len` no box further
    than half the diagonal can contain the point, so the two agree exactly -- asserted against
    `_spot_counts_reference` in the notebook's validation section.

    Granules outside the grid match no spot and are dropped, as in the original.
    """
    granule_xy = np.asarray(granule_xy, dtype=float)
    spot_xy = np.asarray(spot_xy, dtype=float)
    n_spots = spot_xy.shape[0]
    counts = np.zeros(n_spots, dtype=np.int64)
    if granule_xy.shape[0] == 0 or n_spots == 0:
        return counts

    half = grid_len / 2.0
    tree = cKDTree(spot_xy)
    k = min(4, n_spots)
    _, idx = tree.query(granule_xy, k=k, distance_upper_bound=half * np.sqrt(2) + 1e-9)
    idx = np.atleast_2d(idx.T).T if k > 1 else idx.reshape(-1, 1)

    for col in range(idx.shape[1]):
        cand = idx[:, col]
        ok = cand < n_spots                                  # cKDTree pads misses with n_spots
        if not ok.any():
            continue
        g = granule_xy[ok]
        c = spot_xy[cand[ok]]
        inside = ((g[:, 0] >= c[:, 0] - half) & (g[:, 0] < c[:, 0] + half) &
                  (g[:, 1] >= c[:, 1] - half) & (g[:, 1] < c[:, 1] + half))
        if inside.any():
            np.add.at(counts, cand[ok][inside], 1)
    return counts


def _spot_counts_reference(granule_xy, spot_xy, grid_len=50):
    """Literal transcription of the published per-spot loop. Slow; used only to validate
    `spot_counts`."""
    half = grid_len / 2.0
    granule_xy = np.asarray(granule_xy, dtype=float)
    out = np.zeros(len(spot_xy), dtype=np.int64)
    for i, (x, y) in enumerate(np.asarray(spot_xy, dtype=float)):
        in_spot = ((granule_xy[:, 0] >= x - half) & (granule_xy[:, 0] < x + half) &
                   (granule_xy[:, 1] >= y - half) & (granule_xy[:, 1] < y + half))
        out[i] = in_spot.sum()
    return out


def per_spot_counts_table(granule_obs, spots, subtype_col="granule_subtype_manual_simple",
                          sample_label=None, area_col="brain_area",
                          coord_keys=("global_x", "global_y"), grid_len=50):
    """
    One row per (sample, brain_area, subtype, spot) with the granule count in that spot --
    the `compute_subtype_per_spot_counts` output, plus an `overall` subtype covering all granules.

    Call with ONE sample's granules and THAT sample's spots, as the published code requires;
    coordinates are not comparable across samples.
    """
    xcol = "global_x" if "global_x" in granule_obs.columns else "sphere_x"
    ycol = "global_y" if "global_y" in granule_obs.columns else "sphere_y"
    spot_xy = spots.obs[list(coord_keys)].to_numpy(dtype=float)
    areas = spots.obs[area_col].to_numpy()

    frames = []
    subtypes = [s for s in pd.unique(granule_obs[subtype_col].dropna())] + ["overall"]
    for subtype in subtypes:
        sub = granule_obs if subtype == "overall" else granule_obs[granule_obs[subtype_col] == subtype]
        counts = spot_counts(sub[[xcol, ycol]].to_numpy(dtype=float), spot_xy, grid_len=grid_len)
        frames.append(pd.DataFrame({"sample": sample_label, "brain_area": areas,
                                    "subtype": str(subtype), "count": counts}))
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["brain_area"])


def density_from_per_spot(per_spot):
    """Collapse per-spot counts to the published density table schema.

    density == mean per-spot count, which is exactly what
    `compute_subtype_density_per_region` computes (total granules / n_spots).
    """
    g = per_spot.groupby(["sample", "brain_area", "subtype"], dropna=False)["count"]
    out = g.agg(density="mean", n_spots="size", density_sd="std").reset_index()
    out["density_sem"] = out["density_sd"] / np.sqrt(out["n_spots"])
    return out


# --- significance -------------------- ported from benchmark_subtyping.ipynb cells 6 and 22 ---

def bonferroni(p_series):
    p = p_series.to_numpy(dtype=float)
    ok = np.isfinite(p)
    m = ok.sum()
    out = np.full_like(p, np.nan, dtype=float)
    if m > 0:
        out[ok] = np.minimum(p[ok] * m, 1.0)
    return pd.Series(out, index=p_series.index)


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    p_ok = p[ok]
    m = p_ok.size
    order = np.argsort(p_ok)
    ranked = p_ok[order]
    q_ranked = ranked * m / (np.arange(1, m + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q_ok = np.empty_like(p_ok)
    q_ok[order] = q_ranked
    q[ok] = q_ok
    return q


def p_val_to_star(p):
    """Same thresholds as mcDETECT.utils.p_val_to_star, restated so this module has no
    dependency on the package for a two-line helper."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def add_density_significance(density_df, per_spot_wt, per_spot_ad, n_bootstrap=500, seed=0):
    """
    Attach bootstrap 95% CI and the WT-vs-AD test to a density table.

    Ported from code/benchmark/benchmark_subtyping.ipynb cell 22: 500-replicate bootstrap of the
    mean per-spot count; two-sample t-test on log1p per-spot counts; Bonferroni WITHIN subtype
    (across brain areas) and BH FDR across all tests. The bootstrap RNG is seeded here -- the
    published code left it on the global numpy stream, so its CIs are not reproducible.
    """
    rng = np.random.default_rng(seed)
    per_spot_all = pd.concat([per_spot_wt, per_spot_ad], ignore_index=True)

    ci_records = []
    for (s, area, subtype), sub in per_spot_all.groupby(["sample", "brain_area", "subtype"],
                                                        dropna=False):
        vals = sub["count"].to_numpy(dtype=float)
        if len(vals) >= 2:
            boot = rng.choice(vals, size=(n_bootstrap, len(vals)), replace=True).mean(axis=1)
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        else:
            ci_low = ci_high = np.nan
        ci_records.append({"sample": s, "brain_area": area, "subtype": subtype,
                           "density_ci_low": ci_low, "density_ci_high": ci_high})
    density_df = density_df.merge(pd.DataFrame(ci_records),
                                  on=["sample", "brain_area", "subtype"], how="left")

    p_vals = []
    for (area, subtype), _ in density_df.groupby(["brain_area", "subtype"]):
        wt = per_spot_wt.loc[(per_spot_wt["brain_area"] == area) &
                             (per_spot_wt["subtype"] == subtype), "count"].to_numpy(dtype=float)
        ad = per_spot_ad.loc[(per_spot_ad["brain_area"] == area) &
                             (per_spot_ad["subtype"] == subtype), "count"].to_numpy(dtype=float)
        if len(wt) >= 2 and len(ad) >= 2:
            _, p = ttest_ind(np.log1p(wt), np.log1p(ad))
        else:
            p = np.nan
        p_vals.append({"brain_area": area, "subtype": subtype, "p_val": p})

    p_df = pd.DataFrame(p_vals)
    p_df["p_bonf"] = p_df.groupby("subtype", group_keys=False)["p_val"].apply(bonferroni)
    p_df["q_val"] = bh_fdr(p_df["p_val"].values)

    density_df = density_df.merge(p_df, on=["brain_area", "subtype"], how="left")
    for col in ["p_val", "p_bonf", "q_val"]:
        density_df[f"{col}_star"] = density_df[col].apply(p_val_to_star)
    return density_df


# ==================================================================================================
# A2b -- permutation and embedding-structure scoring
# ==================================================================================================

def permutation_fingerprint(transcripts, target_col="target",
                            coord_cols=("global_x", "global_y", "global_z"),
                            n_probe=100_000, seed=0):
    """
    Everything needed to validate a permutation, without keeping a second copy of the table.

    The transcript tables are 69-103 M rows, so holding `original` alongside `permuted` just to
    compare them afterwards roughly doubles peak memory for no reason. Capture this first,
    permute in place, then check against it.

    Per-gene totals are captured exactly (a value_counts is small). Positional integrity is
    captured on a fixed random probe of `n_probe` row positions instead: the checks must be
    ORDER-SENSITIVE, which rules out any whole-column summary that sums or xors per-element
    hashes -- those are invariant under exactly the permutation being tested. A 100 K-row probe
    costs nothing and catches any real coding error with overwhelming probability.
    """
    rng = np.random.default_rng(seed)
    n = len(transcripts)
    probe = np.sort(rng.choice(n, size=min(n_probe, n), replace=False))
    return {
        "n": n,
        "counts": transcripts[target_col].value_counts().sort_index(),
        "probe": probe,
        "labels": transcripts[target_col].to_numpy()[probe],
        "cols": {c: transcripts[c].to_numpy()[probe]
                 for c in list(coord_cols) + ["overlaps_nucleus"]},
    }


def permute_targets_inplace(transcripts, seed, target_col="target"):
    """
    Reviewer #2's null: permute gene labels across all transcripts of a sample, **in place**.

    Preserves every molecule position, the total transcript density, and each gene's total count
    -- it destroys only the association between a gene label and where its molecules sit. The
    label moves while the coordinates and `overlaps_nucleus` stay attached to their row, so each
    transcript keeps its own in-nucleus status.

    Mutates and returns `transcripts`. Pair with `permutation_fingerprint` (before) and
    `assert_permutation_valid` (after).
    """
    rng = np.random.default_rng(seed)
    labels = transcripts[target_col].to_numpy()
    transcripts[target_col] = labels[rng.permutation(labels.shape[0])]
    return transcripts


def assert_permutation_valid(fingerprint, permuted, target_col="target",
                             max_unchanged_frac=0.5):
    """The whole null rests on these facts, so they are checked every run, not behind a flag.
    Cheap relative to detection."""
    assert len(permuted) == fingerprint["n"], "permutation changed the number of transcripts"

    p = permuted[target_col].value_counts().sort_index()
    assert fingerprint["counts"].equals(p), "permutation changed per-gene totals"

    probe = fingerprint["probe"]
    for c, expected in fingerprint["cols"].items():
        assert np.array_equal(permuted[c].to_numpy()[probe], expected), (
            f"permutation moved column {c!r}; only the gene label may move")

    unchanged = float((permuted[target_col].to_numpy()[probe] == fingerprint["labels"]).mean())
    assert unchanged < max_unchanged_frac, (
        f"permutation looks like a no-op: {unchanged:.1%} of probed labels are unchanged")
    return unchanged


def size_matched_indices(batches_a, batches_b, seed=0):
    """
    Row indices subsampling two arms to the same size, stratified by batch.

    The permuted arms will not hold the same number of granules as the real arm, and both
    silhouette and ARI stability depend on n -- so an unmatched comparison invites the reading
    that the null looks less structured merely because it has less data. Matching is symmetric
    (whichever arm is larger gets cut) because the count can move in either direction, and it is
    done per batch so the WT:AD ratio matches as well as the total.

    Returns (idx_a, idx_b), sorted, with len(idx_a) == len(idx_b).
    """
    rng = np.random.default_rng(seed)
    batches_a = np.asarray(batches_a).astype(str)
    batches_b = np.asarray(batches_b).astype(str)
    out_a, out_b = [], []
    for b in sorted(set(batches_a) | set(batches_b)):
        ia = np.flatnonzero(batches_a == b)
        ib = np.flatnonzero(batches_b == b)
        n = min(ia.size, ib.size)
        if n == 0:
            continue
        out_a.append(ia if ia.size == n else rng.choice(ia, n, replace=False))
        out_b.append(ib if ib.size == n else rng.choice(ib, n, replace=False))
    if not out_a:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.sort(np.concatenate(out_a)), np.sort(np.concatenate(out_b))


def _score_one_k(X, k, n_init, batch_size, stability_seeds, silhouette_sample_size,
                 silhouette_seed):
    """One k of the sweep. Runs in a joblib worker, so BLAS is pinned to one thread -- 16 workers
    each spawning 16 BLAS threads oversubscribe the node and finish slower than serial."""
    with threadpool_limits(limits=1):
        km = MiniBatchKMeans(n_clusters=k, random_state=stability_seeds[0],
                             batch_size=batch_size, n_init=n_init)
        labels = km.fit_predict(X)

        n = X.shape[0]
        sample_size = None if silhouette_sample_size is None else min(silhouette_sample_size, n)
        try:
            sil = silhouette_score(X, labels, sample_size=sample_size,
                                   random_state=silhouette_seed)
        except ValueError:                                   # fewer than 2 populated clusters
            sil = np.nan

        label_list = [labels]
        for seed in stability_seeds[1:]:
            km_r = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=batch_size,
                                   n_init=n_init)
            label_list.append(km_r.fit_predict(X))
        ari = [adjusted_rand_score(label_list[i], label_list[j])
               for i in range(len(label_list)) for j in range(i + 1, len(label_list))]

    return {"n_clusters": k, "inertia": float(km.inertia_), "silhouette_score": sil,
            "ari_stability_mean": float(np.mean(ari)) if ari else np.nan,
            "n_obs": n, "silhouette_sample_size": sample_size}


def score_embedding_structure(X, k_range, n_init=20, batch_size=5000,
                              stability_seeds=(0, 42, 123, 456, 789),
                              silhouette_sample_size=50_000, silhouette_seed=0, n_jobs=1):
    """
    Inertia, silhouette and seed-to-seed ARI stability over a range of k.

    Ported from code/benchmark/benchmark_clustering.py:92-121, with `n_init` exposed (that script
    left it at the sklearn default while the published subtyping uses 20) and silhouette
    evaluated on a fixed subsample (it is O(n^2) in distances and the arms have ~10^5-10^6 rows).
    Both departures are applied identically to every arm, so real and permuted stay comparable
    with each other -- but NOT with the numbers in the published
    benchmark_clustering_results.csv.

    `n_jobs` parallelises across k. Each k is independent and every random_state is fixed, so
    this changes wall time and nothing else -- asserted by the n_jobs=1 vs n_jobs=4 equality
    check in the verification suite. joblib memmaps X (~300 MB) rather than pickling one copy
    per worker; set JOBLIB_TEMP_FOLDER to node-local scratch.

    Values of k >= n_obs are skipped rather than raising, so a collapsed null still returns a
    usable table.
    """
    stability_seeds = list(stability_seeds)
    ks = [k for k in k_range if k < X.shape[0]]
    if not ks:
        return pd.DataFrame(columns=["n_clusters", "inertia", "silhouette_score",
                                     "ari_stability_mean", "n_obs", "silhouette_sample_size"])

    rows = Parallel(n_jobs=n_jobs)(
        delayed(_score_one_k)(X, k, n_init, batch_size, stability_seeds,
                              silhouette_sample_size, silhouette_seed) for k in ks)
    # Sort by k so the table is deterministic regardless of completion order.
    return pd.DataFrame(rows).sort_values("n_clusters").reset_index(drop=True)


# ==================================================================================================
# Export helpers for the R hand-off
# ==================================================================================================

def record_distribution(values, measure, bin_spec, **keys):
    """
    Pre-binned histogram + quantiles for one distribution.

    A1 postproc idiom (`A1_filter_de.ipynb` section 3): the arms hold 10^5-10^6 values, so R gets
    bin counts and quantiles rather than the raw column. `bin_spec` is `(lo, hi, n_bins)`; values
    above `hi` land in a final `[hi, inf)` overflow bin so nothing is silently dropped.

    Returns (summary_row: dict, hist_rows: list[dict]).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lo, hi, n_bins = bin_spec
    edges = np.append(np.linspace(lo, hi, n_bins + 1), np.inf)

    summary = dict(keys, measure=measure, n=int(v.size))
    if v.size:
        qs = np.percentile(v, [5, 25, 50, 75, 95])
        summary.update(mean=float(v.mean()), q05=float(qs[0]), q25=float(qs[1]),
                       median=float(qs[2]), q75=float(qs[3]), q95=float(qs[4]),
                       frac_zero=float((v == 0).mean()),
                       frac_overflow=float((v > hi).mean()))
    else:
        summary.update(mean=np.nan, q05=np.nan, q25=np.nan, median=np.nan, q75=np.nan,
                       q95=np.nan, frac_zero=np.nan, frac_overflow=np.nan)

    counts, _ = np.histogram(v, bins=edges)
    total = counts.sum()
    hist_rows = [dict(keys, measure=measure, bin_lo=float(edges[i]), bin_hi=float(edges[i + 1]),
                      count=int(counts[i]),
                      frac=float(counts[i] / total) if total else np.nan)
                 for i in range(len(counts))]
    return summary, hist_rows
