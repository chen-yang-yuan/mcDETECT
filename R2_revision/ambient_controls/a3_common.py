"""
Shared computation for analysis A3 -- ambient RNA controls at the detection step.

Everything here that reproduces published behaviour is a PORT, and each port names its source
file and lines.

Every KD-tree helper is BATCHED. `cKDTree.query_ball_point` accepts arrays of points and radii
with `workers=-1` (and `return_length=True` when only counts are needed); the per-row Python loop
these functions originally used does not finish at this scale -- A3a section 4 alone would issue
~1.7e8 scalar queries.

Contents
--------
  io / setup            write_parquet_atomic, load_transcripts, load_nc_genes, load_genes,
                        load_panel, select_set0
  CSR threshold         tissue_area, csr_min_samples, csr_table
  sphere geometry       sphere_overlap, overlap_pairs
  profiling             profile_spheres (thin wrapper over A1's), funnel_counts
                        (note the two-list NC policy: new detections use 18 genes, reused
                         published data keeps 19 -- see a3_config.SET3_EXCLUDE)
  grids                 _grid_edges
  A3b                   tissue_mask, in_tissue, density_quintiles, place_vicinity_spheres,
                        dbscan_core_predicate
  A3c                   partition_transcripts, composition_logfc, axis1_table,
                        neutral_genes, fit_layer_count_model
  A3d                   grid_origin, grid_bin_id, bin_transcripts, local_null_matrices,
                        local_null_moments, local_null_permutation_check
  A3e                   sample_pseudo_arms, granule_members, local_pool_sizes,
                        draw_local_ambient, apply_relabel_patch, provenance_match,
                        match_spheres
  reporting             record_distribution, bh_fdr, p_val_to_star, write_run_info
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_config as C


# ============================================================ io / setup ============================================================ #

def write_parquet_atomic(df, path):
    """Temp file + os.replace, so a killed task never leaves a half-written parquet.

    from R2_revision/baysor_ssam_merscope/common.py:15-24
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_transcripts(sample, columns=None, verbose=True):
    """One sample's transcript table (~10^8 rows).

    Categorical strings keep `target` cheap. Required columns for A3:
    target, global_x, global_y, global_z, overlaps_nucleus.
    """
    import pyarrow.parquet as pq

    path = C.transcripts_path(sample)
    if verbose:
        print(f"[{sample}] reading {path}", flush=True)
    tbl = pq.read_table(path, columns=columns)
    df = tbl.to_pandas(strings_to_categorical=True)
    if verbose:
        print(f"[{sample}] {len(df):,} transcripts", flush=True)
    return df


def load_genes(sample="WT"):
    """The 290-gene panel, in file order."""
    g = pd.read_csv(C.genes_path(sample))
    return list(g.iloc[:, 0].dropna().unique())


def load_nc_genes(sample="WT", exclude=None):
    """The negative-control genes. From code/3_detection.py:48-49.

    Two lists, chosen by PROVENANCE (a3_config.SET3_EXCLUDE documents the policy in full):

        exclude=C.SET3_EXCLUDE  -> 18 genes, for any NEWLY DETECTED population. Gria2 is a granule
                                   marker mis-listed as a nuclear-enrichment negative control, so
                                   dropping it is a correction, not an approximation.
        exclude=None            -> 19 genes, for anything derived from the PUBLISHED data, which
                                   was built with that list.
    """
    nc = list(pd.read_csv(C.nc_path(sample))["Gene"])
    if exclude:
        nc = [g for g in nc if g not in set(exclude)]
    return nc


def load_panel(sample="WT"):
    return pd.read_csv(C.panel_path(sample))


def neutral_genes(sample="WT"):
    """The panel genes that took NO part in defining or filtering a granule.

    Two groups did: C.SYN_GENES seeded the DBSCAN pass that created every sphere, and the
    published 19-gene NC list was used by nc_filter to delete spheres. Gria2 is on both, so 38
    genes come out of the 290-gene panel and 252 remain.

    These are the genes A3c section 5 tests on, and -- equally important -- the gene set over
    which its composition is computed. Including the seed genes in the denominator would deflate
    every other gene's granule share by a constant (the markers are ~31% of transcripts and are
    concentrated in granules by construction), which moves the neutral point away from zero and
    reduces the analysis to a statement about relative ordering. Computed over this set, zero
    means "the same share of granule RNA as of the surrounding RNA", which is the quantity the
    reviewer's question is actually about.
    """
    panel = set(load_genes(sample))
    excluded = set(C.SYN_GENES) | set(load_nc_genes(sample))
    return sorted(panel - excluded)


def fit_layer_count_model(spot_counts, genes, layers=("granule", "residual_extrasomatic"),
                          samples=None, verbose=True):
    """Quasi-Poisson per gene: layers[0] vs layers[1], across spots, with an exposure offset.

    `genes` fixes BOTH which genes are fitted and which genes the per-spot layer totals are summed
    over -- the offset is the total of `genes` in that layer at that spot, so the coefficient is a
    share-ratio within exactly that pool. See neutral_genes() for why the pool matters.

    Spots are dropped unless both layers have a non-zero total: a spot with no exposure carries no
    information about a rate, and flooring the offset at log(1) against a median exposure of ~200
    fabricates data.
    """
    import statsmodels.api as sm

    gene_set = set(genes)
    marker_set = set(C.SYN_GENES)
    samples = list(C.SAMPLES if samples is None else samples)
    rows = []
    for sample in samples:
        sc_s = spot_counts[spot_counts["sample"] == sample]
        two = sc_s[sc_s["layer"].isin(layers) & sc_s["gene"].isin(gene_set)]
        tot = (two.groupby(["spot", "layer"], observed=True)["n"].sum().unstack(fill_value=0))
        for lay in layers:
            if lay not in tot.columns:
                tot[lay] = 0
        keep = (tot[layers[0]] > 0) & (tot[layers[1]] > 0)
        if verbose:
            print(f"[{sample}] spots with non-zero exposure in both layers: "
                  f"{int(keep.sum()):,}/{len(tot):,}", flush=True)
        tot = tot[keep]
        two = two[two["spot"].isin(tot.index)]
        off = np.log(np.concatenate([tot[layers[0]].to_numpy(),
                                     tot[layers[1]].to_numpy()]).astype(float))
        ind = np.concatenate([np.ones(len(tot)), np.zeros(len(tot))])
        X = sm.add_constant(ind, has_constant="add")
        for gene, sub in two.groupby("gene", observed=True):
            w = sub.pivot_table(index="spot", columns="layer", values="n",
                                fill_value=0, observed=True).reindex(tot.index, fill_value=0)
            y = np.concatenate([w.get(layers[0], pd.Series(0, index=tot.index)).to_numpy(),
                                w.get(layers[1], pd.Series(0, index=tot.index)).to_numpy()]
                               ).astype(float)
            base = dict(sample=sample, gene=gene, n_spots=len(tot),
                        is_marker=gene in marker_set)
            try:
                fit = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit(scale="X2")
                assert fit.df_resid > 0, "saturated model -- inference would be meaningless"
                rows.append(dict(base, df_resid=float(fit.df_resid),
                                 logFC_granule_vs_residual=float(fit.params[1]) / np.log(2),
                                 se=float(fit.bse[1]), pval=float(fit.pvalues[1]),
                                 dispersion=float(fit.scale)))
            except Exception as e:
                rows.append(dict(base, df_resid=np.nan, logFC_granule_vs_residual=np.nan,
                                 se=np.nan, pval=np.nan, dispersion=np.nan, error=str(e)[:80]))
    out = pd.DataFrame(rows)
    out["fdr"] = bh_fdr(out["pval"])
    return out


def select_set0(sample="WT", transcripts=None, n=None, verbose=True):
    """Pick Set 0: neutral panel genes abundance-matched to the 20 markers.

    Set 3 alone only shows 'nuclear genes stay in nuclei' -- circular, because the NC genes are
    DEFINED as nuclear-enriched and are then filtered on nuclear overlap. Set 0 shows that
    arbitrary genes AT MARKER ABUNDANCE do not form granule-like aggregates, and it neutralises
    the 15x abundance gap between markers (median 1,153,633 transcripts) and NC genes (74,893)
    that would otherwise explain an empty Set 3 for free.

    Excludes SYN_GENES, the 19 NC genes, and anything annotated in
    C.SET0_EXCLUDE_PANEL_COLUMNS; then greedily matches each marker to its nearest remaining gene
    in log10 transcript count, without replacement.

    Returns a DataFrame (marker, set0_gene, n_tx_marker, n_tx_set0, log10_gap) -- persisted to
    preflight/set0_genes.csv so the choice is auditable and stable across reruns.
    """
    n = n or C.SET0_N
    if transcripts is None:
        transcripts = load_transcripts(sample, columns=["target"], verbose=verbose)
    counts = transcripts["target"].value_counts()

    panel = load_panel(sample)
    annotated = set()
    for col in C.SET0_EXCLUDE_PANEL_COLUMNS:
        if col in panel.columns:
            hit = panel[col].astype(str).str.strip()
            annotated |= set(panel.loc[hit.ne("") & hit.ne("nan"), "Gene"])

    blocked = set(C.SYN_GENES) | set(load_nc_genes(sample)) | annotated
    pool = [g for g in load_genes(sample) if g not in blocked and counts.get(g, 0) > 0]
    if verbose:
        print(f"[set0] {len(pool)} candidate genes after excluding {len(blocked)} annotated/seed",
              flush=True)

    pool_log = {g: np.log10(counts[g]) for g in pool}
    rows, used = [], set()
    # Match rarest markers first: they have the fewest candidates, so greedy order matters.
    for m in sorted(C.SYN_GENES, key=lambda g: counts.get(g, 0)):
        target = np.log10(max(counts.get(m, 1), 1))
        cand = [(abs(v - target), g) for g, v in pool_log.items() if g not in used]
        if not cand:
            rows.append(dict(marker=m, set0_gene=None, n_tx_marker=int(counts.get(m, 0)),
                             n_tx_set0=np.nan, log10_gap=np.nan))
            continue
        gap, g = min(cand)
        used.add(g)
        rows.append(dict(marker=m, set0_gene=g, n_tx_marker=int(counts.get(m, 0)),
                         n_tx_set0=int(counts[g]), log10_gap=float(gap)))
    out = pd.DataFrame(rows)
    return out.head(n) if n < len(out) else out


# ============================================================ CSR threshold ============================================================ #

def tissue_area(transcripts, grid_len=1.0):
    """Occupied tissue area, um^2.

    Port of mcDETECT_package/mcDETECT/model.py:80-84 (construct_grid + tissue_area): a 2D
    histogram over ALL transcripts at `grid_len`, counting non-empty cells.
    """
    x, y = transcripts["global_x"].to_numpy(), transcripts["global_y"].to_numpy()
    xb, yb = _grid_edges(x, y, grid_len)
    hist, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    return float(np.count_nonzero(hist) * grid_len ** 2)


def csr_min_samples(n_tx, area, alpha, eps=None, cutoff_prob=None, low_bound=None):
    """The CSR / Poisson min_samples rule.

    Port of model.py:88-93. Note it is a 2D AREAL intensity against a 2D DISC pi*eps^2, even
    though DBSCAN runs in 3D -- stage D's local version must match that form.
    """
    eps = C.EPS if eps is None else eps
    cutoff_prob = C.CUTOFF_PROB if cutoff_prob is None else cutoff_prob
    low_bound = C.LOW_BOUND if low_bound is None else low_bound
    bg_density = np.asarray(n_tx, dtype=float) / area
    cutoff = poisson.ppf(cutoff_prob, mu=alpha * bg_density * (np.pi * eps ** 2))
    return np.maximum(cutoff, low_bound).astype(int)


def csr_table(sample, transcripts=None, genes=None, alphas=None, verbose=True):
    """What poisson_select WOULD have returned, per gene, over an alpha sweep.

    Backs the CSR disclosure. code/3_detection.py:66,83 passes minspl=3, so poisson_select never
    runs on real data. At the alpha=10 written in that call the rule would have been far STRICTER
    than what was used (Camk2a -> 32); at alpha = C.CSR_ALPHA_EQUIV = 0.5 it returns exactly 3 for
    all 20 markers in both samples, so min_samples = 3 IS the CSR rule at that alpha.
    """
    alphas = alphas or C.CSR_ALPHA_SWEEP
    if transcripts is None:
        transcripts = load_transcripts(sample, columns=["target", "global_x", "global_y"],
                                       verbose=verbose)
    area = tissue_area(transcripts)
    counts = transcripts["target"].value_counts()
    nc_list = load_nc_genes(sample)          # 19-gene published list: this table documents the
    nc = set(nc_list)                        # filter as published, so it is the right one here
    # dict.fromkeys de-duplicates while preserving order -- Gria2 is in BOTH lists, and without
    # this it appears twice, both rows labelled gene_set="marker".
    genes = genes or list(dict.fromkeys(list(C.SYN_GENES) + list(nc_list)))

    rows = []
    for g in genes:
        n = int(counts.get(g, 0))
        for a in alphas:
            rows.append(dict(sample=sample, gene=g,
                             gene_set=("marker" if g in C.SYN_GENES else
                                       "nc" if g in nc else "other"),
                             n_tx=n, tissue_area=area, bg_density=n / area, alpha=a,
                             min_samples=int(csr_min_samples(n, area, a))))
    return pd.DataFrame(rows)


# ============================================================ sphere geometry ============================================================ #


def sphere_overlap(d, r_a, r_b, criterion="intersect", rho=None, l=None):
    """Boolean overlap under one criterion of the ladder.

    mcDETECT's own merge predicate (model.py:349-353, with l=1, rho=0.2) is
        merge(A,B)  <=>  d <= |r_A - r_B|  (containment)  OR  d < rho*l*(r_A + r_B)
    i.e. two equal-radius spheres merge only within 0.4*r. That is VERY strict -- real granules
    routinely overlap without merging -- so `intersect` is the primary criterion: it is the
    loosest, it maximises apparent overlap, and a small value under it is uncontestable.
    """
    rho = C.MERGE_RHO if rho is None else rho
    l = C.MERGE_L if l is None else l
    d, r_a, r_b = (np.asarray(v, dtype=float) for v in (d, r_a, r_b))
    if criterion == "intersect":
        return d < r_a + r_b
    if criterion == "center_in":
        # ASYMMETRIC by design: "b's centre lies inside a". overlap_pairs(a, b) is therefore not
        # overlap_pairs(b, a) under this rung -- in A3a it is always called with the granule set
        # as `a`, i.e. "a Set-3 sphere's centre lies inside a Set-1/Set-2 granule".
        return d <= r_a
    if criterion == "merge":
        return (d <= np.abs(r_a - r_b)) | (d < rho * l * (r_a + r_b))
    raise ValueError(f"unknown criterion: {criterion}")


def overlap_pairs(a, b, criterion="intersect", z_col=None, max_r=None, chunk=200_000):
    """Which spheres of `a` overlap any sphere of `b`.

    Both frames need sphere_x, sphere_y, <z_col>, sphere_r. Uses a cKDTree ball query at
    r_a + max(r_b) so the candidate set is a superset, then applies the exact criterion.

    `criterion` may be a single name or a LIST of names. The candidate query is by far the
    expensive part and does not depend on the criterion, so passing the whole ladder at once
    costs one pass instead of three. A3a scores 2 controls x 2 granule sets x 3 rungs x 20 null
    re-placements per sample; without this the query is issued 3x more often than it needs to be.

    z_col defaults to C.OVERLAP_Z_COL. NOTE the inconsistency inherited from the package:
    dbscan / _remove_overlaps use `sphere_z`, while nc_filter and profile use `layer_z`. One
    column is chosen here and applied identically to every set; disclose it.

    Returns (mask over `a`, count of `b`-overlaps per row of `a`) for a single criterion, or
    {criterion: (mask, counts)} when a list is passed.
    """
    z_col = z_col or C.OVERLAP_Z_COL
    crits = [criterion] if isinstance(criterion, str) else list(criterion)
    counts = {c: np.zeros(len(a), dtype=int) for c in crits}

    if len(a) and len(b):
        pb = b[["sphere_x", "sphere_y", z_col]].to_numpy(dtype=float)
        rb = b["sphere_r"].to_numpy(dtype=float)
        tree = cKDTree(pb)
        pa = a[["sphere_x", "sphere_y", z_col]].to_numpy(dtype=float)
        ra = a["sphere_r"].to_numpy(dtype=float)
        max_r = float(rb.max()) if max_r is None else max_r

        # One batched query for the candidate supersets, then each criterion vectorised over the
        # flattened pairs. The per-row loop this replaces issued ~1.7e8 scalar queries in A3a's
        # overlap section and would not have finished.
        for lo in range(0, len(pa), chunk):
            hi = min(lo + chunk, len(pa))
            cand = tree.query_ball_point(pa[lo:hi], ra[lo:hi] + max_r, workers=-1)
            n_per = np.fromiter((len(c) for c in cand), dtype=np.int64, count=hi - lo)
            if n_per.sum() == 0:
                continue
            flat = np.fromiter((j for c in cand for j in c), dtype=np.int64,
                               count=int(n_per.sum()))
            owner = np.repeat(np.arange(lo, hi), n_per)
            d = np.linalg.norm(pb[flat] - pa[owner], axis=1)
            for c in crits:
                hit = sphere_overlap(d, ra[owner], rb[flat], c)
                np.add.at(counts[c], owner[hit], 1)

    if isinstance(criterion, str):
        return counts[criterion] > 0, counts[criterion]
    return {c: (counts[c] > 0, counts[c]) for c in crits}


# ============================================================ profiling ============================================================ #

def profile_spheres(spheres, sample=None, transcripts=None, genes=None, marker_genes=None,
                    nc_genes=None, **kwargs):
    """Count every transcript inside every sphere -> (features, count matrix, gene list).

    Thin wrapper over R2_revision/baysor_ssam_merscope/postproc/sphere_features.py::profile_spheres,
    which is a vectorised re-implementation of mcDETECT.model.profile (batched
    query_ball_point(workers=-1) instead of a per-sphere Python loop -- the package version would
    take 681,337 iterations). Works on ANY sphere table, which is what lets Sets 0/1/2/3 and the
    A3b pseudo-granules all be scored by identical code.

    `features` carries n_total, n_marker, n_nc, in_soma_ratio_all, in_soma_ratio_marker,
    nc_ratio_all, nc_ratio_marker, marker_frac. NOTE it returns a 3-tuple, not a 2-tuple.
    """
    sf = _import_sphere_features()
    # NC-policy guard. Left as None, the delegate resolves its OWN 19-gene published list for any
    # sphere table -- including Set 3 and the A3b pseudo-granules, which a3_config's provenance
    # rule says must use the 18-gene list. Resolve it here instead of inheriting silently.
    if nc_genes is None:
        nc_genes = load_nc_genes(sample or "WT", exclude=C.SET3_EXCLUDE)
    if genes is None:
        genes = load_genes(sample or "WT")
    if marker_genes is None:
        marker_genes = C.SYN_GENES          # a3_config's copy, not postproc_config's second one
    # sphere_r is a minimum-enclosing radius; see partition_transcripts' BUFFER note. The
    # delegate defaults to 0.0 so A1's cached feature tables stay byte-identical -- A3 opts in.
    kwargs.setdefault("buffer", C.KG_BUFFER)
    return sf.profile_spheres(spheres, sample=sample, transcripts=transcripts, genes=genes,
                              marker_genes=marker_genes, nc_genes=nc_genes, **kwargs)


def _import_sphere_features():
    """A1's postproc helpers, imported by path rather than copied.

    A2 chose to copy-and-cite; here the surface used is large (profiling, spot anchoring,
    density, significance) so importing keeps the two from drifting. If A1's postproc directory
    is ever moved, this is the single place to fix.
    """
    p = C.REPO_ROOT / "R2_revision" / "baysor_ssam_merscope" / "postproc"
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    import sphere_features as sf
    return sf


def funnel_counts(spheres_raw, size_thr=None, in_soma_thr=None, by=None, genes=None):
    """n surviving at each detection filter stage: raw -> size -> in_soma.

    Reported for EVERY set side by side. Set 3 shown only at the endpoint would be circular: NC
    genes are defined as nuclear-enriched, so if Set 3 empties ONLY at the in-soma step that
    proves nothing. If it is already near-empty at `raw`, that is a result.

    `genes` -- the full list actually seeded, when `by` is a single column of gene names. A gene
    that formed NO sphere at all never appears in `spheres_raw` (run_detection_sets.
    flatten_sphere_dict skips empty per-gene frames), so without this it silently VANISHES from
    the funnel instead of appearing as raw = 0. That is backwards: a zero is the strongest result
    a negative-control set can produce, and it was the one row missing (Set 3 seeded 18 genes and
    reported 16 in WT, 17 in AD -- C4a and Cyfip1 formed nothing anywhere).
    """
    size_thr = C.SIZE_THR if size_thr is None else size_thr
    in_soma_thr = C.IN_SOMA_THR if in_soma_thr is None else in_soma_thr

    def _count(df):
        n_raw = len(df)
        m_size = df["sphere_r"] < size_thr
        n_size = int(m_size.sum())
        n_in_soma = int((m_size & (df["in_soma_ratio"] < in_soma_thr)).sum())
        return dict(raw=n_raw, size=n_size, in_soma=n_in_soma)

    if by is None:
        if genes is not None:
            raise ValueError("genes= needs `by` to be the gene column")
        return pd.DataFrame([_count(spheres_raw)])
    rows = []
    for key, grp in spheres_raw.groupby(by, observed=True):
        row = {by: key} if isinstance(by, str) else dict(zip(by, key))
        row.update(_count(grp))
        rows.append(row)
    out = pd.DataFrame(rows, columns=[by, *C.FUNNEL_STAGES] if not rows else None)
    if genes is None:
        return out
    if not isinstance(by, str):
        raise ValueError("genes= needs `by` to be a single column name")
    # reindex over what was SEEDED, not over what happened to produce a sphere; seeded order is
    # preserved so the table reads the same way the run does.
    out = (out.set_index(by).reindex(list(genes)).rename_axis(by)
           .fillna(0).astype({stage: int for stage in C.FUNNEL_STAGES}).reset_index())
    return out


# ============================================================ grids ============================================================ #

def _grid_edges(x, y, grid_len):
    """The one place lattice edges are built -- tissue_area, tissue_mask and density_quintiles
    previously each carried their own verbatim copy of this."""
    xb = np.arange(np.floor(x.min() / grid_len) * grid_len,
                   np.ceil(x.max() / grid_len) * grid_len + grid_len, grid_len)
    yb = np.arange(np.floor(y.min() / grid_len) * grid_len,
                   np.ceil(y.max() / grid_len) * grid_len + grid_len, grid_len)
    return xb, yb


# ============================================================ A3b ============================================================ #

def tissue_mask(transcripts, grid_len=1.0):
    """Boolean occupancy grid + edges, the same mask tissue_area() counts.

    Used for the in-tissue test so A3b's areas match stage D's density denominators.
    """
    x, y = transcripts["global_x"].to_numpy(), transcripts["global_y"].to_numpy()
    xb, yb = _grid_edges(x, y, grid_len)
    hist, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    return hist > 0, xb, yb


def in_tissue(x, y, mask, xb, yb):
    ix = np.clip(np.searchsorted(xb, x, side="right") - 1, 0, mask.shape[0] - 1)
    iy = np.clip(np.searchsorted(yb, y, side="right") - 1, 0, mask.shape[1] - 1)
    inside = (x >= xb[0]) & (x < xb[-1]) & (y >= yb[0]) & (y < yb[-1])
    return inside & mask[ix, iy]


def density_quintiles(transcripts, grid_len=None, n_bins=None):
    """Total-transcript density per lattice cell, binned into quantiles.

    A3b matches each pseudo-granule to its source granule's density bin. This is the direct
    answer to 'ambient is denser near plaques/dense cells' -- the covariate we are accused of
    ignoring. Returns (bin index grid, x_edges, y_edges).
    """
    grid_len = C.VICINITY_DENSITY_GRID if grid_len is None else grid_len
    n_bins = C.VICINITY_DENSITY_N_BINS if n_bins is None else n_bins
    x, y = transcripts["global_x"].to_numpy(), transcripts["global_y"].to_numpy()
    xb, yb = _grid_edges(x, y, grid_len)
    h, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    occupied = h[h > 0]
    edges = np.quantile(occupied, np.linspace(0, 1, n_bins + 1)) if occupied.size else np.zeros(2)
    idx = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, n_bins - 1)
    return np.where(h > 0, idx, -1), xb, yb


def place_vicinity_spheres(granules, mask, xb, yb, d, arm, real_tree=None, real_r=None,
                           nuc_tree=None, rng=None, max_retry=None):
    """Matched-radius pseudo-granules at an IN-PLANE offset of distance `d`.

    In-plane is required: layer_z takes only 7 discrete values (0, 1.5 ... 9.0) and both
    profile() and nc_filter() query at layer_z, so a 3D direction would push the centre off the
    grid and break comparability.

    `arm` selects a C.VICINITY_ARMS entry. Per-plane 2D granule coverage is only 1.9% (WT) /
    1.5% (AD), so the 'rejected' arm removes few candidates and does not meaningfully bias the
    sample toward granule-sparse space -- but both arms are reported, and the fraction of
    offsets overlapping a real granule is itself a RESULT (it measures how much of the vicinity
    is already called).

    Returns the pseudo-granule frame plus per-row `n_retry` and `accepted`.
    """
    cfg = C.VICINITY_ARMS[arm]
    rng = rng or np.random.default_rng(C.VICINITY_SEED)
    max_retry = C.VICINITY_MAX_RETRY if max_retry is None else max_retry
    if cfg.get("reject_granule_overlap") and real_tree is None:
        raise ValueError(f"arm '{arm}' rejects granule overlap but no real_tree was given -- "
                         "silently skipping the rejection would make it an unlabelled duplicate "
                         "of the 'unrejected' arm")

    x0 = granules["sphere_x"].to_numpy(dtype=float)
    y0 = granules["sphere_y"].to_numpy(dtype=float)
    r = granules["sphere_r"].to_numpy(dtype=float)
    dist = np.broadcast_to(np.asarray(d, dtype=float), r.shape).copy()

    z = granules["layer_z"].to_numpy(dtype=float)
    n = len(granules)
    xs, ys = np.full(n, np.nan), np.full(n, np.nan)
    n_retry, accepted = np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
    pending = np.arange(n)

    for attempt in range(max_retry):
        if pending.size == 0:
            break
        theta = rng.uniform(0, 2 * np.pi, pending.size)
        cx = x0[pending] + dist[pending] * np.cos(theta)
        cy = y0[pending] + dist[pending] * np.sin(theta)
        r_pend, z_pend = r[pending], z[pending]
        ok = in_tissue(cx, cy, mask, xb, yb)

        # In-nucleus rejection. Documented in C.VICINITY_ARMS for BOTH arms and previously not
        # implemented at all, so every arm accepted offsets sitting inside nuclei.
        if nuc_tree is not None and ok.any():
            hit = np.asarray(nuc_tree.query_ball_point(
                np.column_stack([cx[ok], cy[ok], z_pend[ok]]), r_pend[ok],
                workers=-1, return_length=True), dtype=np.int64)
            ok[np.flatnonzero(ok)[hit > 0]] = False

        if cfg.get("reject_granule_overlap") and ok.any():
            # 3D, matching mcDETECT's own merge predicate. A 2D tree would reject an offset for
            # overlapping a granule on ANY of the 7 z-planes -- strictly stricter than the rule
            # C.VICINITY_ARMS claims to apply, and stricter than real granules obey.
            sel = np.flatnonzero(ok)
            q = np.column_stack([cx[sel], cy[sel], z_pend[sel]])
            cand = real_tree.query_ball_point(q, r_pend[sel] + float(real_r.max()), workers=-1)
            for m, c in enumerate(cand):
                if not c:
                    continue
                c = np.asarray(c)
                dd = np.linalg.norm(real_tree.data[c] - q[m], axis=1)
                if sphere_overlap(dd, np.full(c.size, r_pend[sel[m]]), real_r[c],
                                  cfg.get("criterion", "merge")).any():
                    ok[sel[m]] = False
        xs[pending[ok]], ys[pending[ok]] = cx[ok], cy[ok]
        accepted[pending[ok]] = True
        n_retry[pending[~ok]] += 1
        pending = pending[~ok]

    out = granules.copy().reset_index(drop=True)
    out["sphere_x"], out["sphere_y"] = xs, ys
    out["n_retry"], out["accepted"] = n_retry, accepted
    out["offset_d"] = dist
    out["arm"] = arm
    # Positional link back to the source granule. Without it the §7 distance gate cannot pair a
    # pseudo-granule with its source after concat renumbers the index, and it silently passes.
    out["src_i"] = np.arange(n)
    return out


def dbscan_core_predicate(pseudo, transcripts_by_gene, eps=None, min_samples=None):
    """Would the detector have FIRED here? eps-connectivity, not a count.

    Three transcripts scattered across a 4 um sphere are not eps=1.5-connected, so ">=3 inside"
    massively overstates pseudo-granule detectability. A DBSCAN **core point** is by definition a
    point with >= min_samples neighbours within eps, so the question "would DBSCAN have seeded a
    cluster here" is answered exactly by:

        does any transcript of the SEED GENE lying INSIDE the sphere have >= min_samples
        neighbours within eps, counted in that gene's full point cloud?

    That needs two batched ball queries and no clustering at all -- the previous implementation
    fitted one sklearn DBSCAN per row (~1e7 fits) and, worse, returned True if any cluster existed
    anywhere in the ball, including one lying entirely off to one side of it.

    Seed-gene matching matters: "any of 20 markers" is ~20x easier than "3 Camk2a", and Camk2a
    alone is 47% of the published granules. The seed gene must come from the persisted
    sphere_dict, NOT granules.parquet["gene"], which is stale after merging.
    """
    eps = C.EPS if eps is None else eps
    min_samples = C.DETECT_KWARGS_FINE["minspl"] if min_samples is None else min_samples

    if "seed_gene" not in pseudo.columns:
        raise KeyError("dbscan_core_predicate needs a `seed_gene` column -- without it every "
                       "row silently returns False, which would read as 'no pseudo-granule is "
                       "ever detectable'. Recover it from sphere_dict.parquet.")

    n = len(pseudo)
    out = np.zeros(n, dtype=bool)
    n_local = np.zeros(n, dtype=np.int64)
    skipped = np.zeros(n, dtype=bool)

    acc = pseudo["accepted"].to_numpy() if "accepted" in pseudo.columns else np.ones(n, bool)
    genes = pseudo["seed_gene"].to_numpy()
    cen = pseudo[["sphere_x", "sphere_y", "layer_z"]].to_numpy(dtype=float)
    rad = pseudo["sphere_r"].to_numpy(dtype=float)

    for g in pd.unique(genes):
        m = (genes == g) & acc
        if not m.any():
            continue
        if g not in transcripts_by_gene:
            skipped[m] = True          # gene has no transcripts: recorded, never silently False
            continue
        tree, coords = transcripts_by_gene[g]
        # (1) transcripts of the seed gene inside each sphere.
        # The ball MUST carry C.KG_BUFFER. sphere_r is the MINIMUM-ENCLOSING radius
        # (miniball.get_bounding_ball), so for a REAL granule its own support points sit exactly
        # ON the surface and a query at exactly sphere_r loses them to floating point -- which
        # showed up as median n_local = 2 for the real arm, below the min_samples = 3 that
        # provably held when DBSCAN formed that cluster. Pseudo-spheres have no support points on
        # their surface, so the buffer moves them only by the generic volume term (~3% at the
        # median radius): the correction is far larger for the real arm than for the pseudo arms,
        # and therefore WIDENS the gap rather than flattering it.
        inside = tree.query_ball_point(cen[m], rad[m] + C.KG_BUFFER, workers=-1)
        n_local[m] = np.fromiter((len(c) for c in inside), dtype=np.int64, count=int(m.sum()))
        # (2) is any of them a core point in the gene's full cloud?
        flat = [j for c in inside for j in c]
        if not flat:
            continue
        flat = np.fromiter(flat, dtype=np.int64, count=len(flat))
        owner = np.repeat(np.flatnonzero(m),
                          np.fromiter((len(c) for c in inside), dtype=np.int64,
                                      count=int(m.sum())))
        deg = np.asarray(tree.query_ball_point(coords[flat], eps, workers=-1,
                                               return_length=True), dtype=np.int64)
        core = deg >= min_samples      # query_ball_point counts the point itself, as DBSCAN does
        if core.any():
            out[np.unique(owner[core])] = True

    return pd.DataFrame({"would_detect": out, "n_local": n_local, "gene_missing": skipped})


# ============================================================ A3c ============================================================ #

def partition_transcripts(transcripts, granules, buffer=None, z_col="layer_z",
                          sample=None, cache=True, chunk=20_000, verbose=True):
    """Label every transcript intrasomatic / granule / residual_extrasomatic.

    Built at TRANSCRIPT level via batched ball queries, not by subtracting spot matrices. The
    published subtraction -- np.maximum(extrasomatic - spot_granule_expression, 0) in
    7_neuropil_subdomains.ipynb cell 9 and benchmark_ambient.ipynb cell 6 -- compounds three
    errors: spot_embedding assigns each granule to the spot holding its CENTRE
    (downstream.py:706-712) while the sphere spans neighbours; profile() counts ALL transcripts in
    the sphere including overlaps_nucleus == 1, so it over-subtracts from an extrasomatic-only
    layer; and overlapping granules double-count shared transcripts. The clip at 0 then makes the
    bias one-sided. MEASURED, the clip is small and is NOT marker-biased: non-markers are affected
    in a median 0.046% of spots against 0.000% for the markers, i.e. ~24x LESS where the published
    result lives. The transcript-level rebuild is justified by the three structural errors above,
    not by the clip.

    The three arms are disjoint and sum to len(transcripts) exactly.

    BUFFER. The containment query runs at sphere_r + C.KG_BUFFER, not at bare sphere_r. sphere_r
    is the MINIMUM-ENCLOSING radius, so a granule's own support points -- overwhelmingly its seed
    gene -- sit exactly on the surface. Measured on a WT window, a bare-radius query loses 11.6%
    of the granule layer and 93.6% of what it loses are SYN_GENES, which misfiles marker
    transcripts into residual_extrasomatic and attenuates the very contrast this table measures.

    CACHING. Assigning ~10^8 transcripts against ~10^6 spheres is the single most expensive
    operation in A3, so the result is cached to C.transcript_layer_path(sample) and reused within
    A3c. Pass `sample` to enable caching; `cache=False` forces a recompute.

    The cache is keyed on the sample alone, so it CANNOT detect a changed granule set, buffer or
    z_col -- the length check below is not a settings check. Callers must thread their OVERWRITE
    toggle into `cache` rather than rely on this function to notice.
    """
    buffer = C.KG_BUFFER if buffer is None else buffer

    if sample is not None and cache:
        cached = C.transcript_layer_path(sample)
        if cached.exists():
            lab = pd.read_parquet(cached)["layer"]
            if len(lab) == len(transcripts):
                if verbose:
                    print(f"[{sample}] transcript layers read from cache", flush=True)
                return pd.Series(pd.Categorical.from_codes(lab.to_numpy(), C.DE_LAYERS),
                                 index=transcripts.index, name="layer")
            print(f"[{sample}] cache length {len(lab):,} != {len(transcripts):,} -- recomputing",
                  flush=True)

    pts = transcripts[["global_x", "global_y", "global_z"]].to_numpy(dtype=float)
    tree = cKDTree(pts)
    in_granule = np.zeros(len(transcripts), dtype=bool)
    centers = granules[["sphere_x", "sphere_y", z_col]].to_numpy(dtype=float)
    radii = granules["sphere_r"].to_numpy(dtype=float) + buffer

    # Batched: one query per chunk of spheres rather than one per sphere. The per-sphere loop
    # this replaces ran 681,337 sequential queries against a 10^8-point tree, per call, and was
    # invoked four times across two notebooks.
    for lo in range(0, len(centers), chunk):
        hi = min(lo + chunk, len(centers))
        idx = tree.query_ball_point(centers[lo:hi], radii[lo:hi], workers=-1)
        flat = [j for c in idx for j in c]
        if flat:
            in_granule[np.fromiter(flat, dtype=np.int64, count=len(flat))] = True
        if verbose and (lo // chunk) % 10 == 0:
            print(f"    spheres {hi:,}/{len(centers):,}", flush=True)

    soma = transcripts["overlaps_nucleus"].to_numpy() == 1
    codes = np.where(soma, 0, np.where(in_granule, 1, 2)).astype(np.int8)  # order = C.DE_LAYERS

    if sample is not None and cache:
        write_parquet_atomic(pd.DataFrame({"layer": codes}), C.transcript_layer_path(sample))

    return pd.Series(pd.Categorical.from_codes(codes, C.DE_LAYERS),
                     index=transcripts.index, name="layer")


def composition_logfc(counts_a, counts_b, eps=None):
    """log2(share in A) - log2(share in B), per gene.

    Composition-based on both sides so the two are on one scale. This is the form
    code/old/benchmark_diffusion.ipynb cell 10 uses for the baseline; A3c applies it to the
    granule arm too, against the SAME soma reference -- the notebook's USE_ALT_GRANULE_VS_SOMA
    branch, which ships switched OFF. Without that, `delta` subtracts two logFCs that share no
    denominator.
    """
    eps = C.AXIS1_PSEUDOCOUNT if eps is None else eps
    a, b = np.asarray(counts_a, dtype=float), np.asarray(counts_b, dtype=float)
    # Denominator is sum + eps*n_genes so the shares actually sum to 1.
    # benchmark_diffusion.ipynb cell 10 uses `sum + eps`; that leaves a constant offset on every
    # logFC which is numerically negligible at these totals (~1e7) but wrong in form and material
    # for small layers (per-region or per-subtype splits). Deliberate deviation, disclosed.
    fa = (a + eps) / (a.sum() + eps * a.size)
    fb = (b + eps) / (b.sum() + eps * b.size)
    return np.log2(fa) - np.log2(fb)


def axis1_table(counts_by_layer, genes, markers=None):
    """The reviewer's Axis 1, computed against BOTH candidate baselines.

    The reviewer asked for "differential expression between somatic RNA and all non-somatic RNA,
    INDEPENDENT OF GRANULE DETECTION", then for the granule-specific differences to be compared
    against it. That fixes which baseline is primary:

    baseline_all_logFC       (granule + residual_extrasomatic) vs intrasomatic   -- PRIMARY.
                             All non-somatic RNA, exactly as worded. Genuinely independent of
                             granule detection: no sphere is needed to define either layer.
    baseline_logFC           residual_extrasomatic vs intrasomatic               -- SENSITIVITY.
                             Granule-free, so the baseline does not contain the signal it is a
                             null for. But `residual_extrasomatic` is "extrasomatic AND not
                             inside a called sphere", so it is detection-DEPENDENT by
                             construction. Do NOT describe it as detection-independent.
    granule_enrichment       granule vs intrasomatic  -- the same soma reference as both.

    Including the granule transcripts in the baseline biases the difference TOWARD ZERO, i.e. the
    primary is the more conservative of the two as well as the literal one. Both are reported;
    they agree (see axis1_divergence_test.csv).

    WHY SOMA IS THE REFERENCE, given that granules are extrasomatic. It is a presentation device,
    not a claim about biology: a shared denominator puts the two quantities on one axis so they
    can be plotted and regressed against each other, and it then CANCELS EXACTLY out of the
    difference --

        delta = [log2 sh_gnl - log2 sh_soma] - [log2 sh_res - log2 sh_soma]
              =  log2 sh_gnl - log2 sh_res

    so `delta` is granule-versus-extrasomatic and carries no soma term at all (asserted to 1e-9
    by A3c section 6). The predecessor, code/old/benchmark_diffusion.ipynb, used a DIFFERENT
    reference on each axis, so its `delta` subtracted two logFCs sharing no denominator; that is
    what the shared reference fixes. NOTE the cancellation does not extend to `residual`: the
    fitted slope is ~1.18, not 1, so that statistic does depend on the soma layer.

    Then a regression of granule_enrichment on the baseline, fitted on NON-markers, gives the
    reference line; markers above it are enriched beyond what the baseline predicts. Frame the
    claim as DIVERGENCE rather than excess -- the reviewer's own wording is "exceed OR DIVERGE
    FROM", and divergence survives compositional normalisation, which |logFC| does not.

    Returns (df, regression_dict) with keys slope/intercept (sensitivity) and
    slope_all/intercept_all (primary).
    """
    markers = set(markers or C.SYN_GENES)
    soma = np.array([counts_by_layer["intrasomatic"].get(g, 0) for g in genes], dtype=float)
    gnl = np.array([counts_by_layer["granule"].get(g, 0) for g in genes], dtype=float)
    res = np.array([counts_by_layer["residual_extrasomatic"].get(g, 0) for g in genes],
                   dtype=float)

    df = pd.DataFrame({
        "gene": genes,
        "n_intrasomatic": soma, "n_granule": gnl, "n_residual_extrasomatic": res,
        "n_all_extrasomatic": gnl + res,
        # PRIMARY -- the reviewer's literal "all non-somatic RNA"
        "baseline_all_logFC": composition_logfc(gnl + res, soma),
        # SENSITIVITY -- granule-free, but detection-dependent
        "baseline_logFC": composition_logfc(res, soma),
        "granule_enrichment": composition_logfc(gnl, soma),
    })
    df["delta_all"] = df["granule_enrichment"] - df["baseline_all_logFC"]
    df["delta"] = df["granule_enrichment"] - df["baseline_logFC"]
    df["is_marker"] = df["gene"].isin(markers)

    nm = ~df["is_marker"]
    reg = {}
    for suffix, xcol in [("_all", "baseline_all_logFC"), ("", "baseline_logFC")]:
        if nm.sum() >= 2:
            slope, intercept = np.polyfit(df.loc[nm, xcol], df.loc[nm, "granule_enrichment"], 1)
        else:
            slope, intercept = np.nan, np.nan
        df[f"expected_ge{suffix}"] = slope * df[xcol] + intercept
        df[f"residual{suffix}"] = df["granule_enrichment"] - df[f"expected_ge{suffix}"]
        df[f"above_regression{suffix}"] = df["granule_enrichment"] > df[f"expected_ge{suffix}"]
        reg[f"slope{suffix}"] = float(slope)
        reg[f"intercept{suffix}"] = float(intercept)
    # Returned explicitly, not stashed in df.attrs -- attrs is dropped by to_csv/to_parquet and
    # propagates inconsistently through copy/merge/groupby, so the slope would vanish silently.
    return df, reg


# ============================================================ A3d ============================================================ #

def grid_origin(x, y, grid):
    """The lattice a 2-D square grid of pitch `grid` is floored onto: (ox, oy, ny).

    ONE definition, because two callers need identical bin ids. `bin_transcripts` aggregates the
    section onto this lattice; A3e then has to ask which bin a granule CENTRE falls in. If either
    re-derived the arithmetic the ids could disagree silently -- and a granule drawing its ambient
    composition from the wrong bin is exactly the failure this analysis cannot afford.

    The origin comes from the WHOLE section before any filtering, so the lattice does not shift
    when the gene set or the layer set changes.
    """
    ox, oy = float(np.min(x)), float(np.min(y))
    ny = int(np.floor((np.max(y) - oy) / grid)) + 1
    return ox, oy, ny


def grid_bin_id(x, y, ox, oy, ny, grid):
    """Point coordinates -> the bin id used by `bin_transcripts`, on the lattice `grid_origin` gives."""
    ix = np.floor((np.asarray(x, dtype=float) - ox) / grid).astype(np.int64)
    iy = np.floor((np.asarray(y, dtype=float) - oy) / grid).astype(np.int64)
    return ix * ny + iy


def bin_transcripts(sample, grid=None, genes=None,
                    layers=("granule", "residual_extrasomatic"), return_grid=False,
                    verbose=True):
    """Aggregate one sample's transcripts onto a fresh square grid, by gene and compartment.

    Returns long `bin, gene, layer, n`, restricted to `genes` and `layers`.

    THE GRID IS BUILT FROM SCRATCH, and deliberately not the way A3c builds its 50 um one. That
    grid rounds onto the centres of the published spots object (A3c cell 10,
    `np.round((x - sx.min()) / 50)`) because the claim it quantifies is a claim about that object.
    There is no published 10 um spot object to anchor to, so this floors from the section's own
    minimum. Both are square lattices of the same pitch; only the origin convention differs, and
    nothing downstream depends on which one is used.

    2-D, pooling all seven z-planes, exactly as the 50 um grid does. A 10 x 10 um column through a
    9 um section is close to isotropic, so splitting z would buy no locality and would cost a
    great deal of sparsity in the local pool.

    `return_grid=True` additionally returns the lattice (grid, ox, oy, ny), so a caller that
    must map other points -- A3e maps granule centres -- gets its bin ids from `grid_bin_id` on
    the SAME lattice rather than re-deriving the arithmetic.

    Reads the per-transcript layer CACHED BY A3c rather than recomputing it. That file is one int8
    column in the transcript table's own row order, so this is a positional concatenation and the
    10^8-transcripts-against-10^6-spheres assignment -- the single most expensive operation in A3
    -- is not repeated. The cache is keyed on the sample alone and cannot detect a changed granule
    set (see partition_transcripts), so the caller must gate the result against the per-gene totals
    in partition_counts.csv.
    """
    grid = C.LOCAL_NULL_GRID if grid is None else float(grid)
    cached = C.transcript_layer_path(sample)
    if not Path(cached).exists():
        raise FileNotFoundError(
            f"missing {cached}\n  Run A3c_de_baseline.ipynb section 1 first. A3d reuses its "
            f"transcript partition and does not recompute one.")

    tx = load_transcripts(sample, columns=["global_x", "global_y", "target"], verbose=verbose)
    lab = pd.read_parquet(cached)["layer"].to_numpy()
    assert len(lab) == len(tx), (
        f"[{sample}] layer cache has {len(lab):,} rows against {len(tx):,} transcripts. The cache "
        f"is POSITIONAL, so a length mismatch means it was written for a different table.")

    layers = list(layers)
    genes = list(genes)
    n_gene, n_lay = len(genes), len(layers)
    gidx = {g: i for i, g in enumerate(genes)}

    # gene -> column, through the categorical's categories so the map runs once per category
    # rather than once per transcript.
    tgt = tx["target"]
    if isinstance(tgt.dtype, pd.CategoricalDtype):
        per_cat = np.array([gidx.get(c, -1) for c in tgt.cat.categories], dtype=np.int64)
        gcode = per_cat[tgt.cat.codes.to_numpy()]
    else:
        gcode = tgt.map(gidx).fillna(-1).to_numpy().astype(np.int64)

    lcode = np.full(len(tx), -1, dtype=np.int64)
    for j, lay in enumerate(layers):
        lcode[lab == C.DE_LAYERS.index(lay)] = j

    # Origin from the WHOLE section, before any filtering, so the lattice does not shift when the
    # gene set or the layer set changes.
    x = tx["global_x"].to_numpy()
    y = tx["global_y"].to_numpy()
    ox, oy, ny = grid_origin(x, y, grid)
    bin_id = grid_bin_id(x, y, ox, oy, ny, grid)
    del x, y, tx, tgt, lab

    keep = (gcode >= 0) & (lcode >= 0)
    if verbose:
        print(f"[{sample}] grid {grid:g} um, origin ({ox:.2f}, {oy:.2f}); "
              f"{int(keep.sum()):,} of {len(keep):,} transcripts in scope", flush=True)

    # One integer key per (bin, gene, layer) and a single sort, rather than a groupby over three
    # columns of 10^8 rows.
    key = (bin_id[keep] * n_gene + gcode[keep]) * n_lay + lcode[keep]
    del bin_id, gcode, lcode, keep
    uniq, cnt = np.unique(key, return_counts=True)
    del key

    lay_of = uniq % n_lay
    rest = uniq // n_lay
    # Narrow dtypes throughout: this table runs to ~3e7 rows per sample, and the obvious
    # int64/object version costs a gigabyte for nothing. Categories are pinned to `genes` and
    # `layers` order so the codes are directly usable as matrix column indices downstream.
    out = pd.DataFrame({
        "bin": (rest // n_gene).astype(np.int32),
        "gene": pd.Categorical.from_codes((rest % n_gene).astype(np.int16), genes),
        "layer": pd.Categorical.from_codes(lay_of.astype(np.int8), layers),
        "n": cnt.astype(np.int32),
        "sample": pd.Categorical([sample] * len(uniq), categories=list(C.SAMPLES)),
    })
    if verbose:
        print(f"[{sample}] {len(out):,} non-empty (bin, gene, layer) cells over "
              f"{out['bin'].nunique():,} bins", flush=True)
    if return_grid:
        return out, dict(grid=grid, ox=ox, oy=oy, ny=ny)
    return out


def local_null_matrices(binned, genes, layers=("granule", "residual_extrasomatic")):
    """Long bin/gene/layer counts -> two aligned sparse bins x genes matrices, in `layers` order.

    Rows are the sorted union of bins appearing in either layer and columns are `genes` in the
    given order, so the two matrices, every moment vector and every simulated total share one
    index. Alignment is the whole point of returning them together.
    """
    from scipy import sparse

    genes = list(genes)
    gidx = {g: i for i, g in enumerate(genes)}
    bins = np.unique(binned["bin"].to_numpy())
    row = np.searchsorted(bins, binned["bin"].to_numpy())

    # Through the CATEGORICAL CODES, never .astype(str): materialising 3e7 Python strings to look
    # them up in a dict is minutes and gigabytes, and it is pure waste when the codes already are
    # the lookup.
    def _codes(col, order):
        v = binned[col]
        if isinstance(v.dtype, pd.CategoricalDtype):
            remap = np.array([order.index(c) if c in order else -1 for c in v.cat.categories],
                             dtype=np.int64)
            return remap[v.cat.codes.to_numpy()]
        return pd.Series(v).map({o: i for i, o in enumerate(order)}).fillna(-1).to_numpy().astype(np.int64)

    col = _codes("gene", genes)
    assert (col >= 0).all(), "a gene in the binned table is absent from `genes`"
    lay = _codes("layer", list(layers))
    n = binned["n"].to_numpy().astype(np.float64)

    mats = []
    for j in range(len(layers)):
        m = lay == j
        mats.append(sparse.csr_matrix((n[m], (row[m], col[m])),
                                      shape=(len(bins), len(genes))))
    return bins, mats[0], mats[1]


def local_null_moments(O, Cres, mode):
    """Exact mean and variance of every gene's granule total under one of the two local nulls.

    Bins are independent, so the total over bins has a closed-form mean and variance and needs no
    simulation to get a p-value from. See a3_config's A3d block for the two definitions; in brief,
    "literal" redraws each bin's granule transcripts from the composition of the residual RNA
    beside them, and "permutation" relabels which of the bin's pooled non-somatic transcripts are
    granule ones.

    Returns (E, V), both length n_genes.
    """
    N = np.asarray(O.sum(axis=1)).ravel()
    n = np.asarray(Cres.sum(axis=1)).ravel()
    n_gene = O.shape[1]

    if mode == "literal":
        S = Cres.tocsr()
        base = n
    elif mode == "permutation":
        S = (O + Cres).tocsr()
        base = N + n
    else:
        raise ValueError(f"unknown mode {mode!r}; expected one of {C.LOCAL_NULL_MODES}")

    assert (base > 0).all(), "a bin has an empty pool -- filter bins before taking moments"
    rows = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
    q = S.data / base[rows]
    w = N[rows]
    E = np.bincount(S.indices, weights=w * q, minlength=n_gene)

    if mode == "literal":
        V = np.bincount(S.indices, weights=w * q * (1.0 - q), minlength=n_gene)
    else:
        # Finite-population correction: the permutation draws WITHOUT replacement, so a bin whose
        # pool is entirely granule (M == N) has no freedom at all and contributes no variance.
        # Without this the permutation p-values are anticonservative in exactly the dense bins
        # that carry the most weight.
        M = base
        fpc = np.where(M > 1, (M - N) / np.maximum(M - 1.0, 1.0), 0.0)[rows]
        V = np.bincount(S.indices, weights=w * q * (1.0 - q) * fpc, minlength=n_gene)
    return E, V


def local_null_permutation_check(O, Cres, n_bin_check=None, n_rep=None, rng=None, verbose=True):
    """Brute force: physically shuffle labels in real bins and compare against the closed form.

    This is the gate that makes `local_null_moments` believable. Two implementations of one
    formula agreeing proves only that the arithmetic was copied correctly; it cannot catch a WRONG
    formula. So this does the dumb thing instead -- in each of `n_bin_check` randomly chosen real
    bins, pool the granule and residual transcripts, draw N_b of them without replacement, and
    count genes. Repeat `n_rep` times. If the closed form is the permutation null it claims to be,
    the simulated per-gene mean lands on E within Monte-Carlo error and the simulated sd lands on
    sqrt(V).

    Bounded to a subset of bins on purpose: this is a correctness check on a formula, not a
    result, and a few thousand bins pin the moments to well under a percent while running in
    seconds. Whole-section brute force would take hours and prove nothing extra.

    Returns a frame with one row per gene: analytic_mean, mc_mean, analytic_sd, mc_sd, the z of
    the mean difference (expect ~N(0,1)) and the sd ratio (expect ~1), plus `granule_total_ok`,
    which records that every shuffle preserved the granule transcript count exactly -- the null
    moves composition, never abundance.
    """
    n_bin_check = C.LOCAL_NULL_CHECK_BINS if n_bin_check is None else int(n_bin_check)
    n_rep = C.LOCAL_NULL_CHECK_REPS if n_rep is None else int(n_rep)
    rng = np.random.default_rng(0) if rng is None else rng

    Ocsr, Rcsr = O.tocsr(), Cres.tocsr()
    n_bin, n_gene = Ocsr.shape
    pick = rng.choice(n_bin, size=min(n_bin_check, n_bin), replace=False)
    Osub, Rsub = Ocsr[pick], Rcsr[pick]

    E, V = local_null_moments(Osub, Rsub, "permutation")
    N = np.asarray(Osub.sum(axis=1)).ravel().astype(np.int64)

    # One explicit label array per bin: the pooled multiset of gene codes, granule + residual.
    K = (Osub + Rsub).tocsr()
    pools = [np.repeat(K.indices[K.indptr[i]:K.indptr[i + 1]],
                       K.data[K.indptr[i]:K.indptr[i + 1]].astype(np.int64))
             for i in range(K.shape[0])]
    if verbose:
        print(f"    brute force: {n_rep:,} shuffles over {len(pools):,} real bins "
              f"({sum(p.size for p in pools):,} transcripts)", flush=True)

    sim = np.empty((n_rep, n_gene), dtype=np.int64)
    for b in range(n_rep):
        acc = np.zeros(n_gene, dtype=np.int64)
        for i, pool in enumerate(pools):
            if N[i] == 0:
                continue
            # argpartition draws N of M without replacement in O(M) rather than shuffling all of M
            take = pool[np.argpartition(rng.random(pool.size), N[i])[:N[i]]]
            acc += np.bincount(take, minlength=n_gene)
        sim[b] = acc

    sd = np.sqrt(V)
    mc_mean, mc_sd = sim.mean(axis=0), sim.std(axis=0, ddof=1)
    return pd.DataFrame(dict(
        gene_index=np.arange(n_gene),
        analytic_mean=E, mc_mean=mc_mean, analytic_sd=sd, mc_sd=mc_sd,
        z_of_mean_diff=np.divide(mc_mean - E, sd / np.sqrt(n_rep),
                                 out=np.zeros_like(E), where=sd > 0),
        sd_ratio=np.divide(mc_sd, sd, out=np.ones_like(E), where=sd > 0),
        granule_total_ok=bool((sim.sum(axis=1) == N.sum()).all()),
        n_bin_check=len(pools), n_rep=n_rep))


# ============================================================ A3e ============================================================ #

def sample_pseudo_arms(n, rng, frac=None, arms=None):
    """Assign each granule to exactly one arm. Returns an int8 code array into `arms`.

    The converted arms are drawn as one disjoint block so no granule can land in both, which the
    obvious "draw 10%, then draw 10% again" would not guarantee.
    """
    frac = C.PSEUDO_FRAC if frac is None else float(frac)
    arms = list(C.PSEUDO_ARMS if arms is None else arms)
    conv = [a for a in arms if a != "untouched"]
    k = int(round(frac * n))

    code = np.full(n, arms.index("untouched"), dtype=np.int8)
    pick = rng.choice(n, size=k * len(conv), replace=False)
    for j, arm in enumerate(conv):
        code[pick[j * k:(j + 1) * k]] = arms.index(arm)
    return code


def granule_members(granules, points, buffer=None, z_col="layer_z", chunk=20_000, verbose=True):
    """Which of `points` lies inside each sphere, with every point owned by exactly ONE sphere.

    `points` is an (n, 3) array of transcript coordinates -- pass ONLY the granule-layer ones, so
    the tree is 6 M points rather than 10^8. Returns a frame `point, granule, dist`.

    THE BUFFER IS NOT OPTIONAL. The query runs at sphere_r + C.KG_BUFFER, matching
    partition_transcripts. sphere_r is the MINIMUM-ENCLOSING radius, so a granule's own support
    points -- overwhelmingly its seed gene -- sit exactly ON the surface, and a bare-radius query
    loses 11.6% of the granule layer of which 93.6% are markers. In A3e that failure mode is
    fatal rather than merely biased: the unrelabelled seed transcripts would stay in place and the
    pseudo-granule would be re-detected for free.

    SPHERES OVERLAP, so a transcript can fall in several. Each is assigned to the nearest centre,
    ties broken by the lower granule row, so the relabelling rewrites every transcript exactly
    once and the arms stay disjoint at transcript level.
    """
    buffer = C.KG_BUFFER if buffer is None else buffer
    cen = granules[["sphere_x", "sphere_y", z_col]].to_numpy(dtype=float)
    rad = granules["sphere_r"].to_numpy(dtype=float) + buffer
    tree = cKDTree(np.asarray(points, dtype=float))

    pt, gr = [], []
    for lo in range(0, len(cen), chunk):
        hi = min(lo + chunk, len(cen))
        idx = tree.query_ball_point(cen[lo:hi], rad[lo:hi], workers=-1)
        n_per = np.fromiter((len(c) for c in idx), dtype=np.int64, count=hi - lo)
        if n_per.sum() == 0:
            continue
        pt.append(np.fromiter((j for c in idx for j in c), dtype=np.int64,
                              count=int(n_per.sum())))
        gr.append(np.repeat(np.arange(lo, hi), n_per))
        if verbose and (lo // chunk) % 20 == 0:
            print(f"    spheres {hi:,}/{len(cen):,}", flush=True)
    if not pt:
        return pd.DataFrame(dict(point=np.array([], np.int64), granule=np.array([], np.int64),
                                 dist=np.array([], float)))

    pt = np.concatenate(pt)
    gr = np.concatenate(gr)
    d = np.linalg.norm(np.asarray(points, dtype=float)[pt] - cen[gr], axis=1)

    # nearest centre wins; ties by lower granule row. lexsort's last key is primary.
    order = np.lexsort((gr, d, pt))
    pt, gr, d = pt[order], gr[order], d[order]
    first = np.ones(len(pt), dtype=bool)
    first[1:] = pt[1:] != pt[:-1]
    return pd.DataFrame(dict(point=pt[first], granule=gr[first], dist=d[first]))


def local_pool_sizes(centres_xy, tree, radius, chunk=50_000, verbose=True):
    """How many residual transcripts lie within `radius` of each granule centre. Counts only.

    Counts, not indices: `return_length=True` never materialises the neighbour lists, so scoring a
    whole ladder of candidate radii costs one cheap pass each instead of gigabytes of int64. This
    is what lets the notebook report retention at several radii BEFORE committing to one.

    2-D on purpose. `tree` is built over (x, y) with all seven z-planes pooled, exactly as A3d's
    squares pool them: the section is 9 um deep, so a disc through it is a statement about the
    plane, which is the interpretable one.
    """
    centres_xy = np.asarray(centres_xy, dtype=float)
    out = np.empty(len(centres_xy), dtype=np.int64)
    for lo in range(0, len(centres_xy), chunk):
        hi = min(lo + chunk, len(centres_xy))
        out[lo:hi] = tree.query_ball_point(centres_xy[lo:hi], radius, workers=-1,
                                           return_length=True)
        if verbose and (lo // chunk) % 4 == 0:
            print(f"    centres {hi:,}/{len(centres_xy):,}", flush=True)
    return out


def draw_local_ambient(centres_xy, tree, res_code, radius, n_draw, rng, chunk=20_000,
                       verbose=True):
    """Redraw each granule's gene identities from the residual RNA within `radius` of its centre.

    Returns (owner, code): `code[owner == i]` is `n_draw[i]` gene codes drawn WITHOUT REPLACEMENT
    from the residual extrasomatic transcripts of granule i's own neighbourhood.

    Without replacement because the claim being tested is physical -- these identities came from
    real neighbouring ambient molecules, not from a fitted probability vector -- and because it is
    the same permutation spirit as the locked A3d null. Each granule draws independently: two
    granules in overlapping discs are two separate statements about that tissue, not one draw split
    in two.

    THE CALLER MUST HAVE FILTERED on pool >= max(min_pool, n_draw) already. This asserts it rather
    than falling back to drawing with replacement: a fallback would silently apply a different
    sampling scheme to exactly the largest granules, which are also the easiest to re-detect.

    The disc is centred on the granule, so there is no edge artefact to explain away -- unlike
    assigning each granule to whichever square of a fixed lattice its centre happens to land in.
    """
    centres_xy = np.asarray(centres_xy, dtype=float)
    n_draw = np.asarray(n_draw, dtype=np.int64)
    res_code = np.asarray(res_code)
    assert len(centres_xy) == len(n_draw)

    owner = np.repeat(np.arange(len(n_draw)), n_draw)
    code = np.empty(int(n_draw.sum()), dtype=res_code.dtype)
    at = np.concatenate([[0], np.cumsum(n_draw)])

    for lo in range(0, len(centres_xy), chunk):
        hi = min(lo + chunk, len(centres_xy))
        idx = tree.query_ball_point(centres_xy[lo:hi], radius, workers=-1)
        for j, nbrs in enumerate(idx):
            i = lo + j
            k = int(n_draw[i])
            if k == 0:
                continue
            labels = res_code[nbrs]
            if k > labels.size:
                raise AssertionError(
                    f"granule {i} asks for {k} labels from a pool of {labels.size}. The retention "
                    f"rule (pool >= max(min_pool, k)) was not applied before drawing -- drawing "
                    f"with replacement here would change the sampling scheme for exactly the "
                    f"largest granules.")
            # argpartition draws k of M without replacement in O(M), not O(M log M).
            # kth must be < size, so an exact-fit draw takes the whole pool.
            code[at[i]:at[i + 1]] = (labels if k == labels.size
                                     else labels[np.argpartition(rng.random(labels.size), k)[:k]])
        if verbose and (lo // chunk) % 4 == 0:
            print(f"    drew for {hi:,}/{len(centres_xy):,} granules", flush=True)
    return owner, code


def apply_relabel_patch(transcripts, patch, sample=None, expect_rows=None):
    """Rewrite `target` on the rows named by `patch`, in place, through the categorical's codes.

    `patch` has POSITIONAL `row` (into the transcript table's own row order, NOT its
    __index_level_0__, which carries gaps from Vizgen filtering) and `new_target`. Because every
    drawn label is an existing target, this is a code assignment: no category is added, no string
    is materialised, and 10^8 rows are never touched.

    Refuses to run if the table is not the length the patch was built against -- the patch is
    positional, so applying it to a different table would silently corrupt the wrong transcripts.
    """
    if expect_rows is not None and len(transcripts) != int(expect_rows):
        raise ValueError(
            f"[{sample}] patch was built against {int(expect_rows):,} transcripts but this table "
            f"has {len(transcripts):,}. The patch is POSITIONAL and must not be applied.")

    tgt = transcripts["target"]
    if not isinstance(tgt.dtype, pd.CategoricalDtype):
        tgt = tgt.astype("category")
    cats = list(tgt.cat.categories)
    cmap = {c: i for i, c in enumerate(cats)}
    new = patch["new_target"].astype(str).map(cmap)
    if new.isna().any():
        missing = sorted(set(patch["new_target"].astype(str)) - set(cats))[:5]
        raise ValueError(f"[{sample}] patch names targets absent from this table: {missing}")

    codes = tgt.cat.codes.to_numpy().copy()
    rows = patch["row"].to_numpy(dtype=np.int64)
    if rows.max() >= len(codes) or rows.min() < 0:
        raise ValueError(f"[{sample}] patch row index out of range")
    codes[rows] = new.to_numpy(dtype=codes.dtype)
    transcripts["target"] = pd.Categorical.from_codes(codes, cats)
    return transcripts


def provenance_match(new_spheres, points, owner, k_of, frac=None, buffer=None, z_col="layer_z",
                     chunk=20_000, verbose=True):
    """Which published granules did the re-run rebuild a sphere ON TOP OF?

    The identity criterion, and the primary one. A granule G counts as re-detected when some sphere
    of `new_spheres` contains at least `frac` of G's OWN transcripts -- `owner` says which granule
    each point of `points` belongs to, `k_of[G]` how many G has.

    WHY NOT GEOMETRY. `match_spheres`' rungs ask only whether a new sphere sits where G sat. They
    cannot tell "the detector called this object again" from "the detector called something else
    nearby", and 80% of granules are untouched and will certainly be called. Measured on the
    published granules matched against themselves, a granule already contains a DIFFERENT granule's
    centre 6.9% (WT) / 8.2% (AD) of the time under `center_in` and 34% / 39% under `intersect`. A
    pseudo-granule that was entirely destroyed would collect that rate for free. Credit that
    follows the molecules does not have the problem.

    The ball query runs at `sphere_r + KG_BUFFER` centred on (sphere_x, sphere_y, `z_col`) -- the
    `partition_transcripts` convention that defined ownership in the first place, so the two agree.

    Returns a frame indexed like `k_of` with `hit_provenance`, `best_recall` (the largest share of
    G's transcripts any single sphere managed, so a near miss is visible rather than a bare False)
    `scorable` (False where G has no transcripts to rebuild on -- those are never scored, and never
    silently divided by), and `n_crediting`, the number of spheres that cleared the threshold for
    G; self-matching the published set against itself turns that into the criterion's false-credit
    floor, directly comparable with the geometric ones.
    """
    frac = C.PSEUDO_PROVENANCE_FRAC if frac is None else float(frac)
    buffer = C.KG_BUFFER if buffer is None else buffer
    k_of = np.asarray(k_of, dtype=np.int64)
    owner = np.asarray(owner, dtype=np.int64)
    n_gran = len(k_of)

    need = np.maximum(1, np.ceil(frac * k_of)).astype(np.int64)
    best = np.zeros(n_gran, dtype=np.int64)          # most of G's own transcripts in any one sphere
    n_credit = np.zeros(n_gran, dtype=np.int64)      # how many spheres clear the threshold for G
    if len(new_spheres):
        tree = cKDTree(np.asarray(points, dtype=float))
        cen = new_spheres[["sphere_x", "sphere_y", z_col]].to_numpy(dtype=float)
        rad = new_spheres["sphere_r"].to_numpy(dtype=float) + buffer
        for lo in range(0, len(cen), chunk):
            hi = min(lo + chunk, len(cen))
            idx = tree.query_ball_point(cen[lo:hi], rad[lo:hi], workers=-1)
            n_per = np.fromiter((len(c) for c in idx), dtype=np.int64, count=hi - lo)
            if n_per.sum() == 0:
                continue
            flat = np.fromiter((j for c in idx for j in c), dtype=np.int64, count=int(n_per.sum()))
            sph = np.repeat(np.arange(hi - lo, dtype=np.int64), n_per)
            # one pass over (sphere, owner): pack both into a single key, count, then keep the
            # largest count each granule achieved in any single sphere
            key = sph * n_gran + owner[flat]
            uniq, cnt = np.unique(key, return_counts=True)
            own_of = uniq % n_gran
            np.maximum.at(best, own_of, cnt)
            np.add.at(n_credit, own_of[cnt >= need[own_of]], 1)
            if verbose and (lo // chunk) % 10 == 0:
                print(f"    spheres {hi:,}/{len(cen):,}", flush=True)

    scorable = k_of > 0
    return pd.DataFrame(dict(
        hit_provenance=scorable & (best >= need),
        best_recall=np.divide(best, k_of, out=np.zeros(n_gran, float), where=scorable),
        n_own_found=best, n_own=k_of, scorable=scorable,
        # >1 means more than one sphere holds half of G -- the self-match of the published set
        # against itself turns this into the criterion's false-credit floor.
        n_crediting=np.where(scorable, n_credit, 0)))


def match_spheres(reference, redetected, criteria=None, z_col=None):
    """For each sphere of `reference`, was it called again in `redetected`?

    A thin orientation of overlap_pairs: `reference` is `a`, so under the asymmetric `center_in`
    rung the question is "does a re-detected sphere's centre lie inside the original granule",
    which is the tightest statement of 'the same object was called again' that does not depend on
    radius drift. Returns a frame with one boolean and one count column per criterion.

    z_col defaults to C.OVERLAP_Z_COL (sphere_z, the true centre), matching A3a's ladder.
    """
    criteria = list(C.PSEUDO_MATCH_CRITERIA if criteria is None else criteria)
    res = overlap_pairs(reference, redetected, criterion=criteria, z_col=z_col)
    out = pd.DataFrame(index=np.arange(len(reference)))
    for c in criteria:
        mask, counts = res[c]
        out[f"hit_{c}"] = mask
        out[f"n_{c}"] = counts
    return out


# ============================================================ reporting ============================================================ #

def record_distribution(values, measure, bin_spec, **keys):
    """Pre-binned histogram + quantiles for one distribution.

    from R2_revision/sparsity_structure/a2_common.py:848 -- the A1 postproc idiom. R gets bin
    counts and quantiles, never the raw column. Values above `hi` land in a final [hi, inf)
    overflow bin so nothing is silently dropped.

    Returns (summary_row: dict, hist_rows: list[dict]).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lo, hi, n_bins = bin_spec
    # -inf ... lo catches underflow; hi ... inf catches overflow. Without the leading -inf,
    # np.histogram silently DISCARDS values below `lo` while the docstring promises otherwise.
    edges = np.concatenate([[-np.inf], np.linspace(lo, hi, n_bins + 1), [np.inf]])

    summary = dict(keys, measure=measure, n=int(v.size))
    if v.size:
        qs = np.percentile(v, [5, 25, 50, 75, 95])
        summary.update(mean=float(v.mean()), q05=float(qs[0]), q25=float(qs[1]),
                       median=float(qs[2]), q75=float(qs[3]), q95=float(qs[4]),
                       frac_zero=float((v == 0).mean()),
                       frac_underflow=float((v < lo).mean()),
                       frac_overflow=float((v > hi).mean()))
    else:
        summary.update(mean=np.nan, q05=np.nan, q25=np.nan, median=np.nan, q75=np.nan,
                       q95=np.nan, frac_zero=np.nan, frac_underflow=np.nan,
                       frac_overflow=np.nan)

    counts, _ = np.histogram(v, bins=edges)
    total = counts.sum()
    hist_rows = [dict(keys, measure=measure, bin_lo=float(edges[i]), bin_hi=float(edges[i + 1]),
                      count=int(counts[i]),
                      frac=float(counts[i] / total) if total else np.nan)
                 for i in range(len(counts))]
    return summary, hist_rows



def bh_fdr(p_values):
    """from R2_revision/sparsity_structure/a2_common.py:317"""
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    p_ok = p[ok]
    m = p_ok.size
    order = np.argsort(p_ok)
    ranked = p_ok[order]
    q_ranked = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    q_ok = np.empty_like(p_ok)
    q_ok[order] = np.clip(q_ranked, 0.0, 1.0)
    q[ok] = q_ok
    return q


def p_val_to_star(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def write_run_info(out_dir, **info):
    """The run record for one sub-analysis. Convention from a2/a1: every output folder carries
    one, so a table can always be traced to the settings that produced it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([info]).to_csv(out_dir / "run_info.csv", index=False)


def write_status(out_dir, rows, name="status.csv"):
    """Degenerate arms are a RESULT and must be recorded as one -- never a traceback, and never
    a silently dropped arm. Convention from a2_common / score_embedding.py."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / name, index=False)
