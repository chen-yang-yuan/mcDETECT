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
  sphere geometry       sphere_overlap, overlap_pairs, jaccard_balls
  profiling             profile_spheres (thin wrapper over A1's), funnel_counts
  NC filter forensics   nc_ratio_corrected, nc_leave_one_out, gria2_partition
                        (note the two-list NC policy: new detections use 18 genes, reused
                         published data keeps 19 -- see a3_config.SET3_EXCLUDE)
  stage D               local_lambda_grid, adaptive_min_samples, adaptive_survival
  A3b                   tissue_mask, in_tissue, density_quintiles, place_vicinity_spheres,
                        dbscan_core_predicate
  A3c                   partition_transcripts, composition_logfc, axis1_table
  reporting             record_distribution, bonferroni, bh_fdr, p_val_to_star, write_run_info
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


def jaccard_balls(d, r_a, r_b):
    """Volumetric Jaccard of two balls, closed form (spherical lens).

    Reported as a full distribution rather than a cutoff, so the overlap ladder does not rest on
    one arbitrary threshold.
    """
    d, r_a, r_b = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (d, r_a, r_b))
    d, r_a, r_b = np.broadcast_arrays(d, r_a, r_b)
    d = d.copy()
    v_a, v_b = 4 / 3 * np.pi * r_a ** 3, 4 / 3 * np.pi * r_b ** 3
    inter = np.zeros_like(d)
    contained = d <= np.abs(r_a - r_b)
    inter[contained] = np.minimum(v_a, v_b)[contained]
    lens = (~contained) & (d < r_a + r_b) & (d > 0)
    if lens.any():
        dl, ra, rb = d[lens], r_a[lens], r_b[lens]
        inter[lens] = (np.pi * (ra + rb - dl) ** 2
                       * (dl ** 2 + 2 * dl * rb - 3 * rb ** 2 + 2 * dl * ra + 6 * ra * rb
                          - 3 * ra ** 2) / (12 * dl))
    union = v_a + v_b - inter
    out = np.zeros_like(union)
    np.divide(inter, union, out=out, where=union > 0)
    return out


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
    """Which spheres of `a` overlap any sphere of `b`, under one criterion.

    Both frames need sphere_x, sphere_y, <z_col>, sphere_r. Uses a cKDTree ball query at
    r_a + max(r_b) so the candidate set is a superset, then applies the exact criterion.

    z_col defaults to C.OVERLAP_Z_COL. NOTE the inconsistency inherited from the package:
    dbscan / _remove_overlaps use `sphere_z`, while nc_filter and profile use `layer_z`. One
    column is chosen here and applied identically to every set; disclose it.

    Returns (mask over `a`, count of `b`-overlaps per row of `a`).
    """
    z_col = z_col or C.OVERLAP_Z_COL
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=bool), np.zeros(len(a), dtype=int)
    pb = b[["sphere_x", "sphere_y", z_col]].to_numpy(dtype=float)
    rb = b["sphere_r"].to_numpy(dtype=float)
    tree = cKDTree(pb)
    pa = a[["sphere_x", "sphere_y", z_col]].to_numpy(dtype=float)
    ra = a["sphere_r"].to_numpy(dtype=float)
    max_r = float(rb.max()) if max_r is None else max_r

    # One batched query for the candidate supersets, then the exact criterion vectorised over
    # the flattened pairs. The per-row loop this replaces issued ~1.7e8 scalar queries in A3a
    # section 4 and would not have finished.
    counts = np.zeros(len(a), dtype=int)
    for lo in range(0, len(pa), chunk):
        hi = min(lo + chunk, len(pa))
        cand = tree.query_ball_point(pa[lo:hi], ra[lo:hi] + max_r, workers=-1)
        n_per = np.fromiter((len(c) for c in cand), dtype=np.int64, count=hi - lo)
        if n_per.sum() == 0:
            continue
        flat = np.fromiter((j for c in cand for j in c), dtype=np.int64, count=int(n_per.sum()))
        owner = np.repeat(np.arange(lo, hi), n_per)
        d = np.linalg.norm(pb[flat] - pa[owner], axis=1)
        hit = sphere_overlap(d, ra[owner], rb[flat], criterion)
        np.add.at(counts, owner[hit], 1)
    return counts > 0, counts


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


def funnel_counts(spheres_raw, size_thr=None, in_soma_thr=None, by=None):
    """n surviving at each detection filter stage: raw -> size -> in_soma.

    Reported for EVERY set side by side. Set 3 shown only at the endpoint would be circular: NC
    genes are defined as nuclear-enriched, so if Set 3 empties ONLY at the in-soma step that
    proves nothing. If it is already near-empty at `raw`, that is a result.
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
        return pd.DataFrame([_count(spheres_raw)])
    rows = []
    for key, grp in spheres_raw.groupby(by, observed=True):
        row = {by: key} if isinstance(by, str) else dict(zip(by, key))
        row.update(_count(grp))
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================ NC filter forensics ============================================================ #

def nc_ratio_corrected(spheres, transcripts, nc_genes, z_col="layer_z"):
    """Recompute nc_ratio on ONE consistent geometry.

    mcDETECT's nc_filter (model.py:393-413) computes the numerator on the sphere's FINAL geometry
    (centre at layer_z, radius sphere_r) but divides by `size`, which _remove_overlaps never
    recomputes -- it updates only sphere_x/y/z, layer_z and sphere_r. So for every merged granule
    the ratio mixes a post-merge numerator with a pre-merge denominator, inflating nc_ratio
    exactly for the multi-marker (most confidently real) granules. Merging frequency is
    density-dependent, hence region- and condition-dependent.

    Here both come from the same ball query. Returns a frame with the corrected ratio, the
    published one, and whether the granule changes status at nc_thr.
    """
    nc = transcripts[transcripts["target"].isin(nc_genes)]
    marker = transcripts[transcripts["target"].isin(C.SYN_GENES)]
    centers = spheres[["sphere_x", "sphere_y", z_col]].to_numpy(dtype=float)
    radii = spheres["sphere_r"].to_numpy(dtype=float)

    def _counts(df):
        if len(df) == 0:
            return np.zeros(len(spheres), dtype=int)
        tree = cKDTree(df[["global_x", "global_y", "global_z"]].to_numpy(dtype=float))
        return np.asarray(tree.query_ball_point(centers, radii, workers=-1,
                                                return_length=True), dtype=np.int64)

    n_nc, n_marker = _counts(nc), _counts(marker)
    corrected = np.where(n_marker > 0, n_nc / np.maximum(n_marker, 1), np.nan)
    out = pd.DataFrame({
        "n_nc_in_sphere": n_nc,
        "n_marker_in_sphere": n_marker,
        "nc_ratio_corrected": corrected,
        "nc_ratio_published": spheres["nc_ratio"].to_numpy() if "nc_ratio" in spheres else np.nan,
    })
    keep_pub = (out["nc_ratio_published"] == 0) | (out["nc_ratio_published"] < C.NC_THR)
    keep_cor = (out["n_nc_in_sphere"] == 0) | (out["nc_ratio_corrected"] < C.NC_THR)
    out["status_changed"] = keep_pub.fillna(False) != keep_cor.fillna(False)
    return out


def nc_leave_one_out(spheres_set1, transcripts, nc_genes=None, sample="WT",
                     z_col="layer_z", gene_col="seed_gene"):
    """How many Set-1 granules each NC gene removes ON ITS OWN.

    The NC list is not gene-neutral -- it spans complement (C4a), AD-risk (Abca7), an
    oligodendrocyte gene (Opalin) and three dentate-gyrus genes -- so its background is itself
    spatially structured and plausibly condition-dependent, which is Reviewer #2's complaint
    applied to our own filter. If one structured gene drives most of the filtering and is
    elevated in AD, the published AD/WT granule difference is partly a filter artefact.

    `nc_genes` defaults to the 18-gene list, because Set 1 is a NEWLY DETECTED population and the
    provenance policy applies (a3_config.SET3_EXCLUDE). Pass the 19-gene list explicitly only to
    ask what the PUBLISHED filter did.
    """
    if nc_genes is None:
        nc_genes = load_nc_genes(sample, exclude=C.SET3_EXCLUDE)
    centers = spheres_set1[["sphere_x", "sphere_y", z_col]].to_numpy(dtype=float)
    radii = spheres_set1["sphere_r"].to_numpy(dtype=float)
    # `size` and `gene` are the POST-merge columns and both are stale (_remove_overlaps updates
    # only sphere_x/y/z, layer_z, sphere_r). That is deliberate here: this function reproduces
    # what the PUBLISHED nc_filter did, stale denominator included, so its numbers are directly
    # comparable to Set 2. Use nc_ratio_corrected() for the one-geometry version.
    size = spheres_set1["size"].to_numpy(dtype=float)
    seed = (spheres_set1[gene_col].to_numpy() if gene_col in spheres_set1
            else spheres_set1["gene"].to_numpy() if "gene" in spheres_set1 else None)

    rows = []
    for g in nc_genes:
        tx = transcripts[transcripts["target"] == g]
        if len(tx) == 0:
            rows.append(dict(nc_gene=g, n_tx=0, n_removed=0, n_removed_not_self_seeded=0))
            continue
        tree = cKDTree(tx[["global_x", "global_y", "global_z"]].to_numpy(dtype=float))
        cnt = np.asarray(tree.query_ball_point(centers, radii, workers=-1,
                                               return_length=True), dtype=np.int64)
        ratio = np.where(size > 0, cnt / np.maximum(size, 1), 0.0)
        removed = (cnt > 0) & (ratio >= C.NC_THR)
        not_self = removed & (seed != g) if seed is not None else removed
        rows.append(dict(nc_gene=g, n_tx=int(len(tx)), n_removed=int(removed.sum()),
                         n_removed_not_self_seeded=int(not_self.sum())))
    return pd.DataFrame(rows).sort_values("n_removed", ascending=False)


def gria2_partition(set1, set2, gene_col="gene"):
    """Split Set1 - Set2 into 'seeded on Gria2' vs 'dropped for nc_ratio >= thr', and give the
    matched 19-marker sensitivity.

    97.1% (WT) / 97.6% (AD) of published granules have nc_ratio EXACTLY 0, so this difference is
    dominated by the Gria2 list collision and must be partitioned rather than reported as one
    number. Frame Gria2 as a list inconsistency that makes Set 2 CONSERVATIVE -- Gria2 is a
    canonical dendritically-transported transcript, and its presence on a nuclear-enrichment list
    is a curation error, not evidence against granules.

    Both sets seed on all 20 markers (a3_config: the SEED list is never modified, only the NC
    list has two versions), which is what lets Set1 - Set2 isolate the NC filter -- see the
    merge-confound note in a3_config beside SET3_EXCLUDE.

    Columns
    -------
    n_removed_gria2   granules lost to the list collision. An UPPER BOUND on the gap left by
                      keeping Set 2 on the 19-gene NC list: Set 1 applies no NC filter at all, so
                      some of these would have been dropped by an 18-gene filter anyway.
    n_removed_other   the NC filter's genuine effect.
    *_ex_gria2        the free 19-marker sensitivity: Gria2-SEEDED rows stripped from BOTH sets,
                      so the two rest on the same effective marker population.

    NOTE both "invariants" this once claimed the notebooks assert --
    `n_removed_gria2 + n_removed_other == n_removed` and
    `n_removed_ex_gria2 == n_removed_other` -- are algebraic identities of these definitions and
    can never fail. They are not checks. The gate that means something re-derives
    `n_removed_other` from the nc_ratio predicate on Set 1 and compares.
    """
    for nm, frame in (("set1", set1), ("set2", set2)):
        if gene_col not in frame.columns:
            raise KeyError(f"{nm} has no '{gene_col}' column; pass gene_col='seed_gene' when "
                           "using the pre-merge sphere_dict, or 'gene' for a merged table")
    n1, n2 = len(set1), len(set2)
    g1 = int((set1[gene_col] == C.GRIA2).sum())
    g2 = int((set2[gene_col] == C.GRIA2).sum())
    # matched 19-marker populations: pure table filtering, no re-detection
    n1_ex, n2_ex = n1 - g1, n2 - g2
    return pd.DataFrame([dict(
        n_set1=n1, n_set2=n2, n_removed=n1 - n2,
        n_gria2_set1=g1, n_gria2_set2=g2, n_removed_gria2=g1 - g2,
        n_removed_other=(n1 - n2) - (g1 - g2),
        frac_removed_gria2=(g1 - g2) / (n1 - n2) if n1 != n2 else np.nan,
        # --- 19-marker sensitivity (Gria2-seeded rows dropped from both) ---
        n_set1_ex_gria2=n1_ex, n_set2_ex_gria2=n2_ex,
        n_removed_ex_gria2=n1_ex - n2_ex,
        frac_removed_ex_gria2=(n1_ex - n2_ex) / n1_ex if n1_ex else np.nan,
        # the gap accepted by keeping Set 2 on the 19-gene NC list, as a fraction of Set 1
        gap_frac_of_set1=(g1 - g2) / n1 if n1 else np.nan,
    )])


# ============================================================ stage D ============================================================ #

def _grid_edges(x, y, grid_len):
    """The one place lattice edges are built. tissue_area, tissue_mask, density_quintiles and
    local_lambda_grid previously each carried their own verbatim copy of this."""
    xb = np.arange(np.floor(x.min() / grid_len) * grid_len,
                   np.ceil(x.max() / grid_len) * grid_len + grid_len, grid_len)
    yb = np.arange(np.floor(y.min() / grid_len) * grid_len,
                   np.ceil(y.max() / grid_len) * grid_len + grid_len, grid_len)
    return xb, yb


def _occupancy_fraction(x, y, xb, yb, fine=1.0):
    """Fraction of `fine`-um cells that hold >=1 transcript, per coarse lattice cell.

    Mirrors the denominator in model.py::tissue_area (nonzero 1um cells x grid_len^2). Without it
    a granule beside a ventricle or at a section edge is credited with empty space as background
    and survives every threshold.
    """
    coarse_x, coarse_y = float(xb[1] - xb[0]), float(yb[1] - yb[0])
    kx, ky = int(round(coarse_x / fine)), int(round(coarse_y / fine))
    # A rounded k silently truncates the fine grid and misaligns every coarse cell against its
    # block -- no error, and a wrong lambda everywhere downstream. Exact at 25/1; assert it.
    if abs(kx * fine - coarse_x) > 1e-9 or abs(ky * fine - coarse_y) > 1e-9:
        raise ValueError(f"coarse grid ({coarse_x} x {coarse_y}) must be an exact multiple "
                         f"of fine ({fine})")

    nx, ny = (len(xb) - 1) * kx, (len(yb) - 1) * ky
    # built from the count, not from arange on floats, so the shape is exact by construction
    fxb = xb[0] + fine * np.arange(nx + 1)
    fyb = yb[0] + fine * np.arange(ny + 1)
    h, _, _ = np.histogram2d(x, y, bins=[fxb, fyb])
    occupied = (h > 0).astype(np.float32)
    return occupied.reshape(len(xb) - 1, kx, len(yb) - 1, ky).mean(axis=(1, 3))


def local_lambda_grid(transcripts, genes, grid_len=None, exclude_mask=None, fine=1.0):
    """Per-gene transcript COUNTS on a lattice, plus the tissue-occupancy fraction per cell.

    One `np.histogramdd` over (x, y, gene_code) instead of one `histogram2d` per gene -- 290
    separate passes over a 10^8-row column was the previous cost.

    `exclude_mask` drops transcripts (used to bracket neighbour contamination by removing all
    Set-2 granule transcripts). The lattice edges and the occupancy grid are BOTH derived from
    the full, unfiltered table, so the two `ADAPTIVE_EXCLUDE_GRANULE_TX` arms land on the
    identical grid and are cell-comparable -- deriving the edges from the filtered table, as this
    previously did, silently shifted every `searchsorted` index between arms.

    Returns (counts dict gene -> 2D array, occupancy 2D array, x_edges, y_edges).
    """
    grid_len = C.ADAPTIVE_GRID if grid_len is None else grid_len
    x_all = transcripts["global_x"].to_numpy()
    y_all = transcripts["global_y"].to_numpy()
    xb, yb = _grid_edges(x_all, y_all, grid_len)
    occ = _occupancy_fraction(x_all, y_all, xb, yb, fine=fine)

    genes = list(genes)
    keep = np.ones(len(transcripts), bool) if exclude_mask is None else ~np.asarray(exclude_mask)
    code = pd.Categorical(transcripts["target"], categories=genes).codes
    sel = keep & (code >= 0)
    H, _ = np.histogramdd(
        (x_all[sel], y_all[sel], code[sel].astype(float)),
        bins=[xb, yb, np.arange(len(genes) + 1) - 0.5])
    return {g: H[:, :, i] for i, g in enumerate(genes)}, occ, xb, yb


def disc_sum(counts_2d, occ, R, grid_len=None):
    """Box-sum a lattice over the ceil(R/grid) neighbourhood -> (summed counts, occupied area).

    This is what makes `R` enter the arithmetic at all. Without it every radius in C.ADAPTIVE_R
    reads the same single lattice cell and returns an identical survival curve.

    The window is a square of half-width k cells rather than a true disc; at R = 25 and 50 um on
    a 25 um lattice that is k = 1 and 2. The area denominator uses the SAME window, so the ratio
    (a density) stays correct -- only the window's shape is approximate, which is disclosed.
    """
    from scipy.ndimage import uniform_filter

    grid_len = C.ADAPTIVE_GRID if grid_len is None else grid_len
    k = int(np.ceil(R / grid_len))
    w = 2 * k + 1
    n = w * w
    tot = uniform_filter(counts_2d.astype(float), size=w, mode="constant", cval=0.0) * n
    occ_cells = uniform_filter(occ.astype(float), size=w, mode="constant", cval=0.0) * n
    return tot, occ_cells * (grid_len ** 2)          # OCCUPIED area only


def adaptive_min_samples(lambda_local, alpha, eps=None, cutoff_prob=None, low_bound=None):
    """m_local = max(poisson.ppf(cutoff_prob, alpha * lambda * pi * eps^2), low_bound).

    Same functional form as model.py:88-93, with lambda local rather than section-wide. 2D by
    construction, matching the published rule -- switching both locality and dimensionality at
    once would confound the sensitivity analysis.
    """
    eps = C.EPS if eps is None else eps
    cutoff_prob = C.CUTOFF_PROB if cutoff_prob is None else cutoff_prob
    low_bound = C.LOW_BOUND if low_bound is None else low_bound
    mu = alpha * np.asarray(lambda_local, dtype=float) * (np.pi * eps ** 2)
    return np.maximum(poisson.ppf(cutoff_prob, mu=mu), low_bound).astype(int)


def adaptive_survival(k_g, lambda_local, alphas=None):
    """survives <=> k_g >= m_local, over an alpha sweep -> a survival CURVE, not one number.

    `k_g` is the count of the SEED gene that formed this cluster, with the granule's own
    transcripts already subtracted from lambda_local -- otherwise the granule inflates its own
    background and the test is self-defeating. k_g is NOT `size` (which pools all markers and is
    stale after merging); it comes from the persisted sphere_dict.
    """
    alphas = alphas or C.ADAPTIVE_ALPHA_SWEEP
    k_g = np.asarray(k_g, dtype=float)
    rows = []
    for a in alphas:
        m = adaptive_min_samples(lambda_local, a)
        rows.append(dict(alpha=a, n=int(k_g.size), n_survive=int((k_g >= m).sum()),
                         frac_survive=float((k_g >= m).mean()) if k_g.size else np.nan,
                         median_m_local=float(np.median(m)) if k_g.size else np.nan))
    return pd.DataFrame(rows)


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
        # (1) transcripts of the seed gene inside each pseudo-sphere
        inside = tree.query_ball_point(cen[m], rad[m], workers=-1)
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
    bias one-sided and worst for the MARKER genes -- exactly where the result lives.

    The three arms are disjoint and sum to len(transcripts) exactly.

    CACHING. Assigning ~10^8 transcripts against ~10^6 spheres is the single most expensive
    operation in A3, so the result is cached to C.transcript_layer_path(sample) and reused: A3c
    section 1 computes it, A3a section 6 reads it. Pass `sample` to enable caching; `cache=False`
    forces a recompute.
    """
    buffer = C.DE_SPHERE_BUFFER if buffer is None else buffer

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
    """The reviewer's Axis 1, with a granule-free baseline.

    baseline_logFC     residual_extrasomatic vs intrasomatic  -- detection-INDEPENDENT, and
                       granule transcripts are excluded, so the baseline no longer contains the
                       signal it is supposed to be a null for
    granule_enrichment granule vs intrasomatic                -- SAME reference as the baseline
    delta              granule_enrichment - baseline_logFC

    Then a regression of granule_enrichment on baseline_logFC fitted on NON-markers gives the
    reference line; markers above it are enriched beyond what the baseline predicts.

    Returns (df, regression_dict). Frame the
    claim as DIVERGENCE rather than excess -- the reviewer's own wording is "exceed OR DIVERGE
    FROM", and divergence survives compositional normalisation, which |logFC| does not.
    """
    markers = set(markers or C.SYN_GENES)
    soma = np.array([counts_by_layer["intrasomatic"].get(g, 0) for g in genes], dtype=float)
    gnl = np.array([counts_by_layer["granule"].get(g, 0) for g in genes], dtype=float)
    res = np.array([counts_by_layer["residual_extrasomatic"].get(g, 0) for g in genes],
                   dtype=float)

    df = pd.DataFrame({
        "gene": genes,
        "n_intrasomatic": soma, "n_granule": gnl, "n_residual_extrasomatic": res,
        "baseline_logFC": composition_logfc(res, soma),
        "granule_enrichment": composition_logfc(gnl, soma),
    })
    df["delta"] = df["granule_enrichment"] - df["baseline_logFC"]
    df["is_marker"] = df["gene"].isin(markers)

    nm = ~df["is_marker"]
    if nm.sum() >= 2:
        slope, intercept = np.polyfit(df.loc[nm, "baseline_logFC"],
                                      df.loc[nm, "granule_enrichment"], 1)
    else:
        slope, intercept = np.nan, np.nan
    df["expected_ge"] = slope * df["baseline_logFC"] + intercept
    df["residual"] = df["granule_enrichment"] - df["expected_ge"]
    df["above_regression"] = df["granule_enrichment"] > df["expected_ge"]
    # Returned explicitly, not stashed in df.attrs -- attrs is dropped by to_csv/to_parquet and
    # propagates inconsistently through copy/merge/groupby, so the slope would vanish silently.
    return df, dict(slope=float(slope), intercept=float(intercept))


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
