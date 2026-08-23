"""
Profile Baysor / SSAM detections against the transcript table and derive the per-detection
granule-likeness features mcDETECT reports per granule.

Semantics are deliberately IDENTICAL to mcDETECT's own code, so the resulting numbers are
directly comparable to `output/MERSCOPE_{WT,AD}_1/granule_adata_tsne.h5ad`:

  * transcripts are collected inside the sphere with a 3D KD-tree query centred at
    [sphere_x, sphere_y, layer_z] with radius sphere_r          (model.py::profile, line 451)
  * layer_z is sphere_z snapped to the nearest platform z-plane (model.py:155-158)
  * in-soma counts use the precomputed `overlaps_nucleus` column (model.py:171, 262)
  * negative controls are the top-`nc_top` NC genes by count    (model.py::nc_filter, 396-402)

What differs is only the IMPLEMENTATION: mcDETECT loops one sphere at a time in Python, which is
fine for ~10^6 granules but not for the 3.5M-sphere tuned Baysor configs. Here the tree query is
batched (`query_ball_point` over a chunk of centres with per-sphere radii, `workers=-1`) and the
count matrix / scalar features are assembled with pure numpy on the flattened index array.

Main entry point:

    feats, adata = build_features("baysor", "WT")     # cached; re-reads from disk on re-run

Also usable on an arbitrary sphere table (used to validate against mcDETECT's own granules):

    feats, X = profile_spheres(granules, sample="WT")
"""

import itertools
import time
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, vstack

from mcDETECT.utils import make_tree

import postproc_config as C


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def snap_layer_z(z, z_grid):
    """Snap continuous z to the nearest platform z-plane (mirrors model.py:155-158)."""
    z_grid = np.asarray(sorted(z_grid), dtype=float)
    idx = np.abs(np.asarray(z, dtype=float)[:, None] - z_grid[None, :]).argmin(axis=1)
    return z_grid[idx]


def load_transcripts(sample, verbose=True):
    """
    Read one sample's transcript table (the columns the profiler needs).

    `target` is stored as a plain string column and these tables have ~10^8 rows, so it is read
    through pyarrow with `strings_to_categorical=True` -- as object-dtype Python strings it would
    cost several GB. `overlaps_nucleus` is narrowed from int64 to int8 for the same reason.
    """
    t0 = time.time()
    cols = ["global_x", "global_y", "global_z", "target", "overlaps_nucleus"]
    df = pq.read_table(C.transcripts_path(sample), columns=cols).to_pandas(
        strings_to_categorical=True)
    df["overlaps_nucleus"] = df["overlaps_nucleus"].astype(np.int8)
    if verbose:
        print(f"[{sample}] {df.shape[0]:,} transcripts loaded in {time.time() - t0:.1f}s")
    return df


def load_spheres(method, sample, geneset, param=C.PARAM, max_spheres=None, seed=0):
    """Read a config's raw sphere table; optionally subsample for a fast dry run."""
    sph = pd.read_parquet(C.spheres_path(method, sample, geneset, param))
    if max_spheres is not None and sph.shape[0] > max_spheres:
        sph = sph.sample(n=max_spheres, random_state=seed).sort_index()
        print(f"    [dry run] subsampled to {sph.shape[0]:,} spheres")
    return sph.reset_index(drop=True)


# ----------------------------------------------------------------------------- #
# Core: profile a sphere table
# ----------------------------------------------------------------------------- #
def profile_spheres(spheres, sample=None, transcripts=None, genes=None, marker_genes=None,
                    nc_genes=None, chunk_size=C.CHUNK_SIZE, verbose=True):
    """
    Count every transcript inside every sphere.

    Parameters
    ----------
    spheres : DataFrame with sphere_x, sphere_y, sphere_r and either sphere_z or layer_z.
    sample  : "WT" / "AD"; used to resolve transcripts / gene lists when they are not passed in.
    transcripts, genes, marker_genes, nc_genes : pass explicitly to profile several sphere tables
        against the same sample without re-reading the transcript table.

    Returns
    -------
    feats : DataFrame, one row per sphere (index aligned with `spheres`), with the raw counts and
            the derived ratios described in postproc_config.
    X     : csr_matrix (n_spheres x n_genes), counts of each panel gene inside each sphere.
    genes : the gene order of X's columns.
    """
    if transcripts is None:
        transcripts = load_transcripts(sample, verbose=verbose)
    if genes is None:
        genes = C.load_genes(sample)
    if marker_genes is None:
        marker_genes = C.SYN_GENES
    if nc_genes is None:
        nc_genes = C.load_nc_genes(sample)

    n_genes = len(genes)

    # ---------- transcript-side lookup arrays (built once) ---------- #
    target = transcripts["target"]
    if isinstance(target.dtype, pd.CategoricalDtype):
        gene_codes = target.cat.set_categories(genes).cat.codes.to_numpy().astype(np.int32)
    else:
        gene_codes = pd.Categorical(target, categories=genes).codes.astype(np.int32)
    overlaps = transcripts["overlaps_nucleus"].to_numpy(dtype=np.int8)
    marker_mask = transcripts["target"].isin(marker_genes).to_numpy()

    # negative controls: top-NC_TOP by count, exactly as model.py::nc_filter selects them
    # (value_counts on a categorical column also lists unobserved categories -- drop those first)
    nc_counts = transcripts.loc[transcripts["target"].isin(nc_genes), "target"].value_counts()
    nc_counts = nc_counts[nc_counts > 0]
    nc_top_genes = list(nc_counts.index[:C.NC_TOP])
    nc_mask = transcripts["target"].isin(nc_top_genes).to_numpy()

    if verbose:
        print(f"    panel genes: {n_genes}, markers: {marker_mask.sum():,} transcripts, "
              f"NC ({len(nc_top_genes)} genes): {nc_mask.sum():,} transcripts")

    # ---------- sphere-side query points ---------- #
    spheres = spheres.reset_index(drop=True)
    if "layer_z" in spheres.columns:
        layer_z = spheres["layer_z"].to_numpy(dtype=float)
    else:
        layer_z = snap_layer_z(spheres["sphere_z"].to_numpy(), transcripts["global_z"].unique())

    centers = np.column_stack([spheres["sphere_x"].to_numpy(dtype=float),
                               spheres["sphere_y"].to_numpy(dtype=float),
                               layer_z])
    radii = spheres["sphere_r"].to_numpy(dtype=float)
    n_sph = centers.shape[0]

    t0 = time.time()
    tree = make_tree(d1=transcripts["global_x"].to_numpy(dtype=float),
                     d2=transcripts["global_y"].to_numpy(dtype=float),
                     d3=transcripts["global_z"].to_numpy(dtype=float))
    if verbose:
        print(f"    KD-tree over {transcripts.shape[0]:,} transcripts built in {time.time() - t0:.1f}s")

    # ---------- chunked query + vectorised assembly ---------- #
    blocks, n_total, n_in_nuc, n_marker, n_marker_in_nuc, n_nc = [], [], [], [], [], []
    t0 = time.time()
    for start in range(0, n_sph, chunk_size):
        stop = min(start + chunk_size, n_sph)
        idx_lists = tree.query_ball_point(centers[start:stop], radii[start:stop],
                                          workers=C.KDTREE_WORKERS, return_sorted=False)
        m = stop - start

        lens = np.fromiter((len(i) for i in idx_lists), dtype=np.int64, count=m)
        total = int(lens.sum())
        flat = np.fromiter(itertools.chain.from_iterable(idx_lists), dtype=np.int64, count=total)
        rows = np.repeat(np.arange(m, dtype=np.int64), lens)
        del idx_lists

        # count matrix: duplicate (row, col) pairs are summed by csr_matrix
        cols = gene_codes[flat]
        keep = cols >= 0                      # guards against any transcript outside the panel
        blocks.append(csr_matrix((np.ones(int(keep.sum()), dtype=np.float32),
                                  (rows[keep], cols[keep].astype(np.int64))),
                                 shape=(m, n_genes)))

        # scalar features from the same flattened arrays
        in_nuc = overlaps[flat] > 0
        mk = marker_mask[flat]
        n_total.append(lens)
        n_in_nuc.append(np.bincount(rows[in_nuc], minlength=m))
        n_marker.append(np.bincount(rows[mk], minlength=m))
        n_marker_in_nuc.append(np.bincount(rows[mk & in_nuc], minlength=m))
        n_nc.append(np.bincount(rows[nc_mask[flat]], minlength=m))

        del flat, rows, cols, keep, in_nuc, mk
        if verbose:
            print(f"    profiled {stop:,}/{n_sph:,} spheres ({time.time() - t0:.1f}s)", flush=True)

    if not blocks:                                   # empty sphere table
        X = csr_matrix((0, n_genes), dtype=np.float32)
    else:
        X = vstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
    del blocks

    def _cat(parts):
        return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype=np.int64)

    n_total = _cat(n_total)
    n_in_nuc = _cat(n_in_nuc)
    n_marker = _cat(n_marker)
    n_marker_in_nuc = _cat(n_marker_in_nuc)
    n_nc = _cat(n_nc)

    # ---------- derived ratios ---------- #
    def _safe_div(num, den):
        out = np.full(num.shape, np.nan, dtype=float)
        ok = den > 0
        out[ok] = num[ok] / den[ok]
        return out

    feats = pd.DataFrame({
        "sphere_x": spheres["sphere_x"].to_numpy(dtype=float),
        "sphere_y": spheres["sphere_y"].to_numpy(dtype=float),
        "sphere_z": spheres["sphere_z"].to_numpy(dtype=float),
        "layer_z": layer_z,
        "sphere_r": radii,
        "n_total": n_total,
        "n_in_nucleus": n_in_nuc,
        "n_marker": n_marker,
        "n_marker_in_nucleus": n_marker_in_nuc,
        "n_nc": n_nc,
        # distinct genes inside the sphere -- mcDETECT's `comp`
        "comp": np.diff(X.indptr).astype(np.int64),
        # primary: all transcripts in the sphere
        "in_soma_ratio_all": _safe_div(n_in_nuc, n_total),
        # sensitivity: mcDETECT-identical, marker transcripts only (NaN when no markers inside)
        "in_soma_ratio_marker": _safe_div(n_marker_in_nuc, n_marker),
        "marker_frac": _safe_div(n_marker, n_total),
        "nc_ratio_all": _safe_div(n_nc, n_total),
        # mcDETECT's nc_filter divides NC counts by `size` (= the marker count)
        "nc_ratio_marker": _safe_div(n_nc, n_marker),
    })
    for extra in ("cell_id", "n_molecules"):        # Baysor-only passthrough columns
        if extra in spheres.columns:
            feats[extra] = spheres[extra].to_numpy()

    if verbose:
        print(f"    done: {n_sph:,} spheres, {X.nnz:,} nonzero counts, "
              f"{(n_total == 0).sum():,} empty spheres")
    return feats, X, genes


# ----------------------------------------------------------------------------- #
# Cached driver for one sweep config
# ----------------------------------------------------------------------------- #
def build_features(method, sample, geneset, param=C.PARAM, overwrite=False,
                   max_spheres=None, transcripts=None, chunk_size=C.CHUNK_SIZE, verbose=True):
    """
    Profile one sweep config and cache the result.

    Writes  <postproc>/<method>_<sample>_<param>_<geneset>_features.parquet
            <postproc>/<method>_<sample>_<param>_<geneset>_profile.h5ad   (counts in .X and .layers)
    and returns (features DataFrame, AnnData). Re-runs read the cache unless overwrite=True.

    Detections from the `markers` setting are profiled against the FULL 290-gene panel, exactly as
    mcDETECT seeds detection on 20 markers but profiles every gene (3_detection.py:111).
    """
    f_path = C.features_path(method, sample, geneset, param)
    p_path = C.profile_path(method, sample, geneset, param)

    if f_path.exists() and p_path.exists() and not overwrite and max_spheres is None:
        if verbose:
            print(f"[cached] {C.config_tag(method, sample, geneset, param)}")
        return pd.read_parquet(f_path), anndata.read_h5ad(p_path)

    print(f"[build] {C.config_tag(method, sample, geneset, param)}")
    spheres = load_spheres(method, sample, geneset, param, max_spheres=max_spheres)
    print(f"    {spheres.shape[0]:,} raw detections")

    feats, X, genes = profile_spheres(spheres, sample=sample, transcripts=transcripts,
                                      chunk_size=chunk_size, verbose=verbose)

    adata = anndata.AnnData(X=X, obs=feats.copy())
    adata.obs_names = [f"{method}_{geneset}_{sample}_{i}" for i in range(adata.n_obs)]
    adata.var_names = genes
    adata.var["genes"] = genes
    adata.layers["counts"] = X.copy()
    adata.obs["method"] = method
    adata.obs["sample"] = sample
    adata.obs["geneset"] = geneset

    if max_spheres is None:
        C.POSTPROC_DIR.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(f_path, index=False)
        adata.write_h5ad(p_path)
        print(f"    wrote {f_path.name} and {p_path.name}")
    else:
        print("    [dry run] results NOT cached to disk")
    return feats, adata


# ----------------------------------------------------------------------------- #
# Population definitions
# ----------------------------------------------------------------------------- #
def assign_populations(feats, geneset, in_soma_thr=C.IN_SOMA_THR,
                       marker_frac_thr=C.MARKER_FRAC_THR, verbose=True):
    """
    Add the nested population flags to a features table.

      pop1  all detections
      pop2  out-of-nucleus:  in_soma_ratio_all < in_soma_thr
      pop3  pop2 AND marker_frac >= marker_frac_thr        [`all` geneset only]

    pop3 exists only for the "all" setting: in the "markers" setting every transcript that formed
    the detection is already a granule marker, so a marker-enrichment filter is vacuous.

    pop2 uses the ALL-transcript in-soma ratio. mcDETECT's own marker-only denominator is reported
    alongside as `in_soma_ratio_marker` and gets its own panels in A1_figures.R section 2, which is
    where that sensitivity is read -- it is not turned into a separate population.
    """
    feats = feats.copy()

    feats["pop1"] = True
    feats["pop2"] = feats["in_soma_ratio_all"].to_numpy() < in_soma_thr

    msg = f"    pop1={feats['pop1'].sum():,}  pop2={feats['pop2'].sum():,}"
    if geneset == "all":
        feats["marker_enriched"] = feats["marker_frac"].to_numpy() >= marker_frac_thr  # NaN -> False
        feats["pop3"] = feats["pop2"] & feats["marker_enriched"]
        msg += f"  pop3={feats['pop3'].sum():,} (marker_frac >= {marker_frac_thr})"

    if verbose:
        print(msg)
    return feats


def sample_marker_share(sample, transcripts=None):
    """
    Fraction of a sample's panel transcripts that are granule markers. Not a filter any more --
    reported as the background against which the pop3 `marker_frac` threshold should be read.
    Counted with pyarrow so the 10^8-row `target` column is never materialised as Python strings.
    """
    genes = set(C.load_genes(sample))
    markers = set(C.SYN_GENES)

    if transcripts is not None:
        counts = transcripts["target"].value_counts()
        counts = {str(k): int(v) for k, v in counts.items()}
    else:
        import pyarrow.compute as pcomp
        col = pq.read_table(C.transcripts_path(sample), columns=["target"]).column("target")
        vc = pcomp.value_counts(col.combine_chunks())
        counts = {str(v.as_py()): c.as_py()
                  for v, c in zip(vc.field("values"), vc.field("counts"))}

    n_panel = sum(v for g, v in counts.items() if g in genes)
    n_marker = sum(v for g, v in counts.items() if g in markers)
    if n_panel == 0:
        raise ValueError(f"No panel transcripts found for sample {sample}")
    return n_marker / n_panel


def population_summary(feats, method, sample, geneset):
    """One row per population defined for this setting: n and the median granule-likeness features."""
    rows = []
    for pop in C.populations(geneset):
        sub = feats[feats[pop]]
        rows.append({
            "method": method, "sample": sample, "geneset": geneset, "population": pop,
            "n_detections": int(sub.shape[0]),
            "frac_of_all": float(sub.shape[0]) / float(feats.shape[0]) if len(feats) else np.nan,
            "median_sphere_r": float(sub["sphere_r"].median()) if len(sub) else np.nan,
            "median_n_total": float(sub["n_total"].median()) if len(sub) else np.nan,
            "median_n_marker": float(sub["n_marker"].median()) if len(sub) else np.nan,
            "median_comp": float(sub["comp"].median()) if len(sub) else np.nan,
            "median_in_soma_ratio_all": float(sub["in_soma_ratio_all"].median()) if len(sub) else np.nan,
            "median_nc_ratio_all": float(sub["nc_ratio_all"].median()) if len(sub) else np.nan,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- #
# Coordinate registration (correctness gate only)
# ----------------------------------------------------------------------------- #
def register_coords(feats, sample):
    """
    Put detections in the shared WT/AD frame mcDETECT registers its granules into.

    Not called by the pipeline any more -- nothing downstream reads the registered coordinates.
    It is kept because A1_filter_de.ipynb section 6 uses it as a correctness gate, checking that
    this transform reproduces the `global_x_new` / `global_y_new` stored on mcDETECT's own granules.

    Two stages, both copied from the mcDETECT pipeline so the frames coincide:
      1. per-sample rotation (+ optional flip)  -- code/3_detection.py:97-103
         -> global_x_new / global_y_new
      2. combined WT/AD registration            -- 4_post_detection.ipynb cell 19
         -> global_x_adjusted / global_y_adjusted  (WT y-flipped by `cutoff`, axes swapped;
            AD offset by shift_x / shift_y)
    A further WT `global_x = COMBINED_CUTOFF - global_x_adjusted` step exists downstream in
    mcDETECT (5_neuropil_subdomains_data.py:200-206) and is deliberately NOT applied here, so this
    stays a faithful copy of the granule-side transform.
    """
    reg = C.REGISTRATION[sample]
    feats = feats.copy()

    theta = reg["theta"]
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    coords = feats[reg["rotate_cols"]].to_numpy(dtype=float)
    transformed = coords @ rotation.T

    # column names follow mcDETECT: "sphere_y" -> "global_y_new", etc.
    col0 = "global_" + reg["rotate_cols"][0].split("_")[1] + "_new"
    col1 = "global_" + reg["rotate_cols"][1].split("_")[1] + "_new"
    feats[col0] = transformed[:, 0]
    feats[col1] = transformed[:, 1]
    if reg["flip"]:
        feats[reg["flip_col"] + "_new"] = reg["cutoff"] - feats[reg["flip_col"] + "_new"]

    # combined frame
    if sample == "WT":
        feats["global_y_new"] = C.COMBINED_CUTOFF - feats["global_y_new"]
        feats["global_x_adjusted"] = feats["global_y_new"].to_numpy()
        feats["global_y_adjusted"] = feats["global_x_new"].to_numpy()
    else:
        feats["global_x_adjusted"] = feats["global_x_new"].to_numpy() + C.COMBINED_SHIFT_X
        feats["global_y_adjusted"] = feats["global_y_new"].to_numpy() + C.COMBINED_SHIFT_Y
    return feats


# ----------------------------------------------------------------------------- #
# Compartment-marker entropy (how "mixed" is a single detection?)
# ----------------------------------------------------------------------------- #
def marker_entropy(adata, ref_genes=None, verbose=True):
    """
    Per-detection spread of the compartment-marker signal, over mcDETECT's 34 `REF_GENES`.

    mcDETECT's granule subtypes come from clustering detections on exactly these 34 markers
    (`benchmark_subtyping.ipynb`'s `main_result` arm), so "how concentrated is a detection's
    REF_GENES profile" is the per-detection precursor of "does it get a pure subtype label". It is
    measurable without the manual cluster -> compartment mapping, which is what makes it usable
    here: that mapping is a judgement call made per method, whereas this is a property of the data.

    For each detection, with p the profile over the markers normalised to sum 1:

        n_ref        total marker transcripts in the detection
        entropy      -sum(p log p), in nats
        entropy_norm entropy / log(n markers present)   -- 0 = one marker only, 1 = uniform
        perplexity   exp(entropy)                       -- effective number of distinct markers
        top1_share   largest single-marker share        -- 1 = the detection is one marker

    NO gene -> compartment grouping is applied. mcDETECT assigns compartments at the CLUSTER level,
    never at the gene level, so any gene-wise compartment map would be an invention of this script.
    All four statistics are properties of the count vector alone.

    Detections holding zero marker transcripts have no defined profile; they get NaN in all four
    columns and are counted separately rather than dropped, because that fraction is itself strongly
    method-dependent: 11.8% (WT) / 9.5% (AD) of tuned SSAM's all/pop1 detections contain none of the
    34 markers, against 0.7% / 0.8% of Baysor's.

    Returns a DataFrame aligned with `adata.obs_names`.
    """
    ref_genes = list(C.REF_GENES) if ref_genes is None else list(ref_genes)
    present = [g for g in ref_genes if g in set(map(str, adata.var_names))]
    missing = [g for g in ref_genes if g not in set(map(str, adata.var_names))]
    if verbose and missing:
        print(f"    marker_entropy: {len(missing)} of {len(ref_genes)} REF_GENES absent from the "
              f"panel and skipped: {', '.join(missing)}")
    if not present:
        raise ValueError("none of ref_genes are present in adata.var_names")

    sub = adata[:, present]
    X = sub.layers["counts"] if "counts" in sub.layers else sub.X
    X = csr_matrix(X, dtype=np.float64)
    X.eliminate_zeros()

    n_obs = X.shape[0]
    n_ref = np.asarray(X.sum(axis=1)).ravel()
    rows = np.repeat(np.arange(n_obs, dtype=np.int64), np.diff(X.indptr))
    vals = X.data

    defined = n_ref > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_ref[rows] > 0, vals / n_ref[rows], 0.0)
        contrib = np.where(p > 0, -p * np.log(p), 0.0)
    entropy = np.bincount(rows, weights=contrib, minlength=n_obs)

    top1 = np.zeros(n_obs, dtype=np.float64)
    np.maximum.at(top1, rows, vals)
    with np.errstate(divide="ignore", invalid="ignore"):
        top1_share = np.where(defined, top1 / np.where(defined, n_ref, 1.0), np.nan)

    entropy = np.where(defined, entropy, np.nan)
    # log(1) == 0 would make entropy_norm undefined; with a single marker present entropy is
    # identically 0 anyway, so the normalised value is 0 by definition rather than 0/0.
    denom = np.log(len(present))
    entropy_norm = (entropy / denom) if denom > 0 else np.where(defined, 0.0, np.nan)

    out = pd.DataFrame({
        "n_ref": n_ref.astype(np.int64),
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "perplexity": np.where(defined, np.exp(entropy), np.nan),
        "top1_share": top1_share,
    }, index=adata.obs_names.copy())
    if verbose:
        n_undef = int((~defined).sum())
        print(f"    marker_entropy: {n_obs:,} detections over {len(present)} markers | "
              f"undefined (n_ref == 0): {n_undef:,} ({100.0 * n_undef / max(n_obs, 1):.2f}%) | "
              f"median entropy_norm {np.nanmedian(entropy_norm):.3f}")
    return out


# ----------------------------------------------------------------------------- #
# Anchoring onto mcDETECT's own 50 um spot grid (for the subdomain-level DE)
# ----------------------------------------------------------------------------- #
def anchor_to_mcdetect_spots(adata, sample, grid_len=None, x_key="sphere_x", y_key="sphere_y",
                             verbose=True):
    """
    Aggregate detections onto mcDETECT's EXISTING spot grid, keeping mcDETECT's `spot_id`.

    The spatial partition here must be mcDETECT's, identical for every arm, so that the only thing
    varying between arms is which transcripts land in each spot. Building a fresh grid per method
    would not do: its spots would exist only where that method has detections, and its ids would
    mean nothing outside it.

    `data/<dataset>/processed_data/spots.h5ad` is a regular `grid_len` lattice with tile centres at
    `ix * grid_len + grid_len / 2` (verified exactly for both samples: WT 17,667 tiles, AD 12,604),
    so `ix = floor(x / grid_len)` recovers the tile a detection falls in, and `spot_id` joins
    straight to `4_hard_normalized_cluster_labels.parquet`.

    Coordinates must be the sample's OWN raw frame, matching the grid -- pass sphere_x / sphere_y,
    not the registered coordinates.

    Returns a spot-level AnnData carrying `spot_id`, `brain_area`, `n_detections` and every base-grid
    obs column, restricted to spots holding at least one detection.
    """
    grid_len = float(C.SPOT_GRID if grid_len is None else grid_len)

    base = anndata.read_h5ad(C.spots_path(sample))
    bx = np.asarray(base.obs["global_x"], dtype=float)
    by = np.asarray(base.obs["global_y"], dtype=float)
    bix = np.floor(bx / grid_len).astype(np.int64)
    biy = np.floor(by / grid_len).astype(np.int64)
    off = grid_len / 2.0
    if not (np.allclose(bx, bix * grid_len + off) and np.allclose(by, biy * grid_len + off)):
        raise ValueError(f"{C.spots_path(sample)} is not a regular {grid_len:g} um lattice; "
                         "the tile-centre assumption behind this join does not hold")

    # ix * 2^32 + iy is injective for |iy| < 2^31, which holds by many orders of magnitude here.
    M = np.int64(1) << np.int64(32)
    bkey = bix * M + biy
    order = np.argsort(bkey, kind="stable")
    bkey_sorted = bkey[order]

    x = np.asarray(adata.obs[x_key], dtype=float)
    y = np.asarray(adata.obs[y_key], dtype=float)
    dkey = np.floor(x / grid_len).astype(np.int64) * M + np.floor(y / grid_len).astype(np.int64)

    pos = np.searchsorted(bkey_sorted, dkey)
    inside = pos < bkey_sorted.size
    pos_clipped = np.where(inside, pos, 0)
    hit = inside & (bkey_sorted[pos_clipped] == dkey)
    row = order[pos_clipped]

    n_hit = int(hit.sum())
    if verbose:
        print(f"    {adata.n_obs:,} detections -> {n_hit:,} matched to a spot "
              f"({100.0 * n_hit / max(adata.n_obs, 1):.2f}%); "
              f"{adata.n_obs - n_hit:,} fell outside the grid")
    if n_hit == 0:
        raise ValueError(f"no detection fell inside {sample}'s spot grid -- check the coordinate "
                         f"frame of {x_key}/{y_key}")

    counts_layer = adata.layers["counts"] if "counts" in adata.layers else adata.X
    agg = csr_matrix((np.ones(n_hit, dtype=np.float32),
                      (row[hit], np.flatnonzero(hit))),
                     shape=(base.n_obs, adata.n_obs))
    X = (agg @ csr_matrix(counts_layer)).tocsr()

    obs = base.obs.copy()
    obs["n_detections"] = np.bincount(row[hit], minlength=base.n_obs).astype(np.int64)
    for col in ("method", "sample", "geneset"):
        if col in adata.obs.columns and adata.n_obs:
            obs[col] = adata.obs[col].iloc[0]

    spots = anndata.AnnData(X=X, obs=obs, var=adata.var.copy())
    spots.layers["counts"] = X.copy()
    spots = spots[spots.obs["n_detections"] > 0].copy()
    if verbose:
        print(f"    -> {spots.n_obs:,} of {base.n_obs:,} spots occupied "
              f"(median {np.median(spots.obs['n_detections']):.0f} detections/spot)")
    return spots


# ----------------------------------------------------------------------------- #
# Granule subtyping -- mcDETECT's procedure, used by both notebooks
#
#   A1_filter_de.ipynb section 4   heatmap per arm, labels discarded (all ten arms)
#   A1_ssam_subtypes.ipynb         labels kept and mapped to compartments (SSAM's five arms)
# ----------------------------------------------------------------------------- #
def subtype_cluster(adata, ref_genes=None, n_clusters=None, seed=None,
                    max_cells=None, subsample_seed=None, verbose=True):
    """
    Cluster one arm's detections on the 34 compartment markers, mcDETECT's procedure unchanged.

    Copied from mcDETECT's published `main_result` arm
    (code/benchmark/benchmark_subtyping.ipynb: `run_manual_subtyping()` in cell 15):

      1. normalise on the FULL panel (normalize_total 1e4 -> log1p), THEN subset to REF_GENES.
         The benchmark reads an already-normalised granule object and only then subsets, matching
         3_detection.py:114-115. Normalising after subsetting would rescale each detection by its
         marker content alone and change the clustering.
      2. densify, seed numpy, and fit MiniBatchKMeans(15, batch_size=5000, n_init=20,
         random_state=seed) on the marker matrix -- no PCA, no plain KMeans.
      3. store the labels as an ordered Categorical over "0".."k-1", as mcDETECT does.

    `max_cells=None` means NO subsampling -- every detection is clustered and the returned labels
    align 1:1 with `adata`, which is what a caller needs if it is going to attach them back to the
    object. Pass an integer to cap it: `subtype_heatmap` passes C.HEATMAP_MAX_CELLS, because the
    heatmap draws one column per detection and a 6 M-detection arm cannot be rendered.

    Returns (sub, labels, n_used): `sub` is the marker-subset AnnData carrying
    obs["granule_subtype_kmeans"], `labels` the same column, `n_used` how many detections went in.
    """
    import scanpy as sc
    from sklearn.cluster import MiniBatchKMeans

    ref_genes = list(C.REF_GENES) if ref_genes is None else list(ref_genes)
    n_clusters = C.K_SUBTYPE if n_clusters is None else n_clusters
    seed = C.SUBTYPE_SEED if seed is None else seed
    subsample_seed = C.HEATMAP_SEED if subsample_seed is None else subsample_seed

    n_obs = adata.n_obs
    if n_obs < n_clusters:
        if verbose:
            print(f"    only {n_obs} detections (< k = {n_clusters}) -- skipping")
        return None, None, 0

    if max_cells is not None and n_obs > max_cells:
        rng = np.random.default_rng(subsample_seed)
        idx = np.sort(rng.choice(n_obs, max_cells, replace=False))
        sub = adata[idx].copy()
    else:
        sub = adata.copy()
    n_used = sub.n_obs

    # ---- 1. normalise on the full panel, then subset (order matters) ---- #
    if "counts" in sub.layers:
        sub.X = sub.layers["counts"].copy()
    sc.pp.normalize_total(sub, target_sum=1e4)
    sc.pp.log1p(sub)

    present = [g for g in ref_genes if g in set(map(str, sub.var_names))]
    if not present:
        raise ValueError("none of ref_genes are present in adata.var_names")
    sub = sub[:, present].copy()

    # ---- 2. MiniBatchKMeans on the marker matrix ---- #
    X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    np.random.seed(seed)
    km = MiniBatchKMeans(n_clusters=n_clusters, batch_size=C.KMEANS_BATCH_SIZE,
                         random_state=seed, n_init=C.KMEANS_N_INIT)
    km.fit(X)
    sub.obs["granule_subtype_kmeans"] = pd.Categorical(
        km.labels_.astype(str), categories=[str(i) for i in range(n_clusters)], ordered=True)

    if verbose:
        note = f" (subsampled from {n_obs:,})" if n_used < n_obs else ""
        print(f"    clustered {n_used:,} detections{note} on {len(present)}/{len(ref_genes)} "
              f"markers, k = {n_clusters}, seed = {seed}")
    return sub, sub.obs["granule_subtype_kmeans"], n_used


def render_subtype_heatmap(sub, out_path, title=None, marker_means_path=None, verbose=True):
    """
    Draw mcDETECT's subtype heatmap from an ALREADY-clustered marker-subset AnnData -- i.e. the
    `sub` that `subtype_cluster` returns, carrying obs["granule_subtype_kmeans"].

    Split out so a caller that needs the labels does not have to cluster twice. The call matches
    benchmark_subtyping.ipynb cell 22: sc.pl.heatmap(cmap="Reds", standard_scale="var",
    swap_axes=True, dendrogram=False, figsize=(10, 6)) at dpi=500.

    With `marker_means_path` set, also writes the cluster x marker mean table, in numeric cluster
    order -- the table to read a manual cluster -> compartment mapping off.
    """
    import matplotlib.pyplot as plt
    import scanpy as sc

    present = [str(g) for g in sub.var_names]
    sc.pl.heatmap(sub, var_names=present, groupby="granule_subtype_kmeans", cmap="Reds",
                  standard_scale="var", dendrogram=False, swap_axes=True, show=False,
                  figsize=(10, 6))
    if title:
        plt.gcf().suptitle(title, fontsize=11, y=1.02)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.gcf().savefig(out_path, dpi=500, bbox_inches="tight")
    plt.close("all")

    if marker_means_path is not None:
        X = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
        means = (pd.DataFrame(X, columns=present)
                 .assign(cluster=sub.obs["granule_subtype_kmeans"].to_numpy())
                 .groupby("cluster", observed=True).mean())
        # groupby on the stringified labels sorts lexicographically (0, 1, 10, 11, ...);
        # reindex numerically so the table reads in cluster order
        means = means.reindex(sorted(means.index, key=int))
        Path(marker_means_path).parent.mkdir(parents=True, exist_ok=True)
        means.to_csv(marker_means_path)

    if verbose:
        print(f"    wrote {out_path.name}")
    return sub.n_obs


def subtype_heatmap(adata, out_path, ref_genes=None, n_clusters=None, seed=None,
                    max_cells="default", subsample_seed=None, title=None, verbose=True,
                    marker_means_path=None):
    """
    Render mcDETECT's subtype heatmap for one arm, so a Baysor / SSAM panel can be read directly
    against output/MERSCOPE_WT_AD_comparison/heatmap_subtype.jpeg.

    Convenience wrapper: cluster with `subtype_cluster`, then draw with
    `render_subtype_heatmap`. A caller that also needs the labels should call those two directly,
    so the arm is not clustered twice.

    `max_cells` defaults to C.HEATMAP_MAX_CELLS here, because the heatmap draws one column per
    detection; pass None to disable subsampling.

    Returns the number of detections plotted, or None if the arm was too small to cluster.
    """
    if max_cells == "default":
        max_cells = C.HEATMAP_MAX_CELLS

    sub, _, n_used = subtype_cluster(adata, ref_genes=ref_genes, n_clusters=n_clusters, seed=seed,
                                     max_cells=max_cells, subsample_seed=subsample_seed,
                                     verbose=verbose)
    if sub is None:
        return None
    render_subtype_heatmap(sub, out_path, title=title, marker_means_path=marker_means_path,
                           verbose=verbose)
    return n_used


# ----------------------------------------------------------------------------- #
# Granule-subtype density per brain region (WT vs AD)
#
# Re-derived from mcDETECT's own code/benchmark/benchmark_subtyping.ipynb -- the density and
# per-spot helpers in cell 19, the capture correction and statistics in cell 22, and the
# Bonferroni / BH helpers in cell 6 -- so the output is directly comparable to
# output/MERSCOPE_WT_AD_comparison/subtype_density_per_region_granule_adata_tsne.csv.
# ----------------------------------------------------------------------------- #
def grid_counts(x, y, spots, grid_len=None, spot_loc_key=("global_x", "global_y")):
    """
    Detections per spot, vectorised.

    mcDETECT tests membership as a half-open square around each spot centre
    (`gx >= x - half and gx < x + half`, same in y) by looping over every spot and rescanning the
    whole detection table -- O(n_spots x n_detections), which is 10^4 x 10^5 here and repeats per
    subtype. On a REGULAR lattice those squares tile without overlap, so "inside the square" and
    "nearest centre in the Chebyshev metric, within half an edge" select the same spot; the KD-tree
    answers that for all detections at once. `anchor_to_mcdetect_spots` asserts the lattice really
    is regular, and mcDETECT's spot objects are built on one.

    Returns an int array aligned with `spots.obs`.
    """
    from scipy.spatial import cKDTree

    grid_len = float(C.SPOT_GRID if grid_len is None else grid_len)
    centres = np.column_stack([np.asarray(spots.obs[spot_loc_key[0]], dtype=float),
                               np.asarray(spots.obs[spot_loc_key[1]], dtype=float)])
    tree = cKDTree(centres)
    _, idx = tree.query(np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)]),
                        k=1, p=np.inf, distance_upper_bound=grid_len / 2.0)
    inside = idx < centres.shape[0]          # detections outside every tile come back as n_spots
    return np.bincount(idx[inside], minlength=centres.shape[0]).astype(np.int64)


def subtype_density_per_region(det_x, det_y, det_subtype, spots, sample, apply_capture_coef,
                               grid_len=None, n_boot=None, seed=0, area_col="brain_area"):
    """
    Per-region detection density by granule subtype, matching mcDETECT's published analysis.

    For every (brain_area, subtype): density = detections in that area / spots in that area, on the
    sample's own 50 um spot grid -- a MEAN PER-SPOT COUNT, not a count per unit area. Plus one
    `subtype = "overall"` row per area covering ALL detections, **including those whose subtype is
    NaN**, exactly as mcDETECT does; so `overall` is >= the sum of the per-subtype rows and is
    strictly greater whenever some clusters were left out of the manual mapping.

    Brain areas are enumerated from `spots.obs`, in order of appearance, `Unknown` included -- again
    as mcDETECT does. The R side filters to the nine plotted regions.

    When `apply_capture_coef` is set, counts are DIVIDED by C.CAPTURE_EFFICIENCY_COEF before
    anything else, so density, sd, sem, CI and the test all inherit it -- the same place cell 22
    applies it, to both `density_df_ad["density"]` and `per_spot_ad["count"]`. That coefficient is
    1.0 here, i.e. the correction is deliberately NOT applied; see postproc_config.py for why and
    for what has to be disclosed about comparability with mcDETECT's published table.

    The bootstrap CI is seeded here; mcDETECT's draws from the global RNG downstream of
    MiniBatchKMeans and so is only reproducible by replaying its whole cell.

    Coordinates must be the sample's OWN raw frame and match `spots` -- pass sphere_x / sphere_y.

    Returns (density_df, per_spot_df); feed both to `add_density_significance`, which needs both
    samples at once.
    """
    grid_len = C.SPOT_GRID if grid_len is None else grid_len
    n_boot = C.N_BOOTSTRAP if n_boot is None else n_boot
    rng = np.random.default_rng(seed)

    det_x = np.asarray(det_x, dtype=float)
    det_y = np.asarray(det_y, dtype=float)
    det_subtype = np.asarray(det_subtype, dtype=object)
    area = np.asarray(spots.obs[area_col])

    scale = (1.0 / C.CAPTURE_EFFICIENCY_COEF) if apply_capture_coef else 1.0

    subtypes = [s for s in pd.unique(det_subtype) if pd.notna(s)]
    groups = [(str(s), det_subtype == s) for s in subtypes] + [("overall", np.ones(det_x.size, bool))]

    density_rows, per_spot_frames = [], []
    for label, mask in groups:
        counts = grid_counts(det_x[mask], det_y[mask], spots,
                             grid_len=grid_len).astype(float) * scale
        for a in pd.unique(area[pd.notna(area)]):
            sel = area == a
            n_spots = int(sel.sum())
            if n_spots == 0:
                continue
            vals = counts[sel]
            if vals.size >= 2:
                boot = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
                ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
                sd = float(vals.std(ddof=1))
            else:
                ci_low = ci_high = sd = np.nan
            density_rows.append({
                "sample": sample, "brain_area": str(a), "subtype": label,
                "density": float(vals.mean()), "n_spots": n_spots,
                "density_sd": sd, "density_sem": sd / np.sqrt(n_spots) if n_spots else np.nan,
                "density_ci_low": ci_low, "density_ci_high": ci_high,
            })
            per_spot_frames.append(pd.DataFrame({"sample": sample, "brain_area": str(a),
                                                 "subtype": label, "count": vals}))

    density_df = pd.DataFrame(density_rows)
    per_spot_df = (pd.concat(per_spot_frames, ignore_index=True) if per_spot_frames
                   else pd.DataFrame(columns=["sample", "brain_area", "subtype", "count"]))

    # mcDETECT loops brain_area OUTER and subtype INNER, with "overall" appended last per area;
    # the loop above is the other way round because the per-subtype spot counts are vectorised.
    # Reorder to mcDETECT's row order so the two CSVs can be diffed line for line.
    if not density_df.empty:
        area_rank = {str(a): i for i, a in enumerate(pd.unique(area[pd.notna(area)]))}
        sub_rank = {label: i for i, (label, _) in enumerate(groups)}   # "overall" is last by build
        density_df = (density_df
                      .assign(_a=density_df["brain_area"].map(area_rank),
                              _s=density_df["subtype"].map(sub_rank))
                      .sort_values(["_a", "_s"], kind="stable")
                      .drop(columns=["_a", "_s"])
                      .reset_index(drop=True))
    return density_df, per_spot_df


def add_density_significance(density_df, per_spot_df, wt="WT", ad="AD", setting=""):
    """
    Add mcDETECT's WT-vs-AD test columns to a stacked density table, and return it in mcDETECT's
    exact 16-column schema so the two CSVs can be read side by side.

    Matching benchmark_subtyping.ipynb cell 22 and its helpers in cell 6:
      * one two-sided `scipy.stats.ttest_ind` per (brain_area, subtype) -- INCLUDING "overall" --
        on `log1p` of the per-spot counts, n = spots per sample; NaN if either side has n < 2;
      * `p_bonf`  = Bonferroni WITHIN each subtype (m = the finite p-values for that subtype),
                    not globally;
      * `q_val`   = Benjamini-Hochberg across EVERY (area x subtype) test at once;
      * stars from mcDETECT.utils.p_val_to_star, which uses strict `>` at 0.05 / 0.01 / 0.001;
        a NaN p becomes "" rather than "ns", so the two are distinguishable.
    """
    from scipy.stats import ttest_ind
    from mcDETECT.utils import p_val_to_star

    def _bonferroni(p_series):
        p = p_series.to_numpy(dtype=float)
        ok = np.isfinite(p)
        out = np.full_like(p, np.nan, dtype=float)
        m = int(ok.sum())
        if m:
            out[ok] = np.minimum(p[ok] * m, 1.0)
        return pd.Series(out, index=p_series.index)

    def _bh_fdr(p_values):
        p = np.asarray(p_values, dtype=float)
        q = np.full_like(p, np.nan, dtype=float)
        ok = np.isfinite(p)
        if ok.sum() == 0:
            return q
        p_ok = p[ok]
        m = p_ok.size
        order = np.argsort(p_ok)
        ranked = p_ok[order] * m / np.arange(1, m + 1)
        ranked = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0.0, 1.0)
        q_ok = np.empty_like(p_ok)
        q_ok[order] = ranked
        q[ok] = q_ok
        return q

    ps = per_spot_df
    rows = []
    for (a, st), _ in density_df.groupby(["brain_area", "subtype"], observed=True):
        w = ps[(ps["sample"] == wt) & (ps.brain_area == a) & (ps.subtype == st)]["count"].to_numpy()
        d = ps[(ps["sample"] == ad) & (ps.brain_area == a) & (ps.subtype == st)]["count"].to_numpy()
        if w.size >= 2 and d.size >= 2:
            _, p = ttest_ind(np.log1p(w), np.log1p(d))
        else:
            p = np.nan
        rows.append({"brain_area": a, "subtype": st, "p_val": p})

    p_df = pd.DataFrame(rows)
    p_df["p_bonf"] = p_df.groupby("subtype", group_keys=False)["p_val"].apply(_bonferroni)
    p_df["q_val"] = _bh_fdr(p_df["p_val"].values)

    out = density_df.merge(p_df, on=["brain_area", "subtype"], how="left")
    out["setting"] = setting
    for src, dst in [("p_val", "p_val_star"), ("p_bonf", "p_bonf_star"), ("q_val", "q_val_star")]:
        out[dst] = out[src].apply(lambda v: "" if pd.isna(v) else p_val_to_star(v))

    cols = ["sample", "brain_area", "subtype", "density", "n_spots", "setting",
            "density_sd", "density_sem", "density_ci_low", "density_ci_high",
            "p_val", "p_bonf", "q_val", "p_val_star", "p_bonf_star", "q_val_star"]
    return out[cols]


# ----------------------------------------------------------------------------- #
# Synaptic density vs the genetic-labeling reference (WT only)
# ----------------------------------------------------------------------------- #
def synaptic_density_vs_reference(det_x, det_y, det_subtype_fine, spots, setting="",
                                  area_list=None, reference=None, drop_areas=None,
                                  synaptic_subtypes=None, grid_len=None, area_col="brain_area",
                                  verbose=True):
    """
    mcDETECT's external check on the subtyping, reproduced (benchmark_subtyping.ipynb cell 22, the
    WT correlation block): per-region density of the SYNAPTIC detections against synapse densities
    measured by volume EM + genetic labeling (Santuy et al. 2020), weighted by spots per region.

    Synaptic membership is decided on the FINE label -- `granule_subtype_manual` in
    `C.SYNAPTIC_SUBTYPES` -- not on the collapsed one, so `pre & den`, `post & den` and
    `pre & post & den` are excluded exactly as mcDETECT excludes them.

    Density is the same quantity `subtype_density_per_region` reports: a mean per-spot count over
    that region's 50 um spots. A region with no detections at all comes out 0.0, matching
    mcDETECT's `reindex(AREA_LIST)` + `nan_to_num(0.0)`. That function is not reused here because
    its bootstrap and its `overall` row have no part in this test.

    Both vectors are min-max rescaled with mcDETECT's `scale` before the correlation. That cannot
    change either coefficient -- both are invariant under a positive affine map -- and is kept only
    so this is line-for-line mcDETECT's computation.

    WT ONLY: the reference is a WT measurement, so C.CAPTURE_EFFICIENCY_COEF never enters. Pass the
    WT detections and the WT spot grid; coordinates must be that sample's own raw frame
    (sphere_x / sphere_y), as everywhere else in the density chain.

    Returns (per_area_df, summary):
      per_area_df  one row per region in `area_list` -- brain_area, density, n_spots, reference,
                   used. `used` is False for `drop_areas`, so the exclusions are visible in the
                   file rather than implied by its length.
      summary      a dict: n_detections, n_synaptic, synaptic_frac, n_areas, r_pearson, r_spearman
                   (weighted, mcDETECT's) and r_pearson_unw / r_spearman_unw (unweighted, ours),
                   plus `note` when the correlation could not be computed.
    """
    from scipy.stats import pearsonr, spearmanr

    from mcDETECT.utils import scale, weighted_corr, weighted_spearmanr

    area_list = list(C.AREA_LIST if area_list is None else area_list)
    reference = dict(C.GT_SYNAPSE_DENSITY if reference is None else reference)
    drop_areas = tuple(C.GT_DROP_AREAS if drop_areas is None else drop_areas)
    synaptic_subtypes = list(C.SYNAPTIC_SUBTYPES if synaptic_subtypes is None else synaptic_subtypes)

    det_x = np.asarray(det_x, dtype=float)
    det_y = np.asarray(det_y, dtype=float)
    fine = np.asarray(det_subtype_fine, dtype=object)
    keep = np.isin(fine, synaptic_subtypes)

    counts = grid_counts(det_x[keep], det_y[keep], spots, grid_len=grid_len).astype(float)
    area = np.asarray(spots.obs[area_col])

    rows = []
    for a in area_list:
        sel = area == a
        n_spots = int(sel.sum())
        rows.append({
            "setting": setting, "brain_area": a,
            "density": float(counts[sel].mean()) if n_spots else 0.0,
            "n_spots": n_spots,
            "reference": reference.get(a, np.nan),
            "used": (a not in drop_areas) and (a in reference),
        })
    per_area_df = pd.DataFrame(rows)

    used = per_area_df[per_area_df["used"]]
    obs = used["density"].to_numpy(dtype=float)
    ref = used["reference"].to_numpy(dtype=float)
    w = used["n_spots"].to_numpy(dtype=float)

    summary = {"setting": setting, "n_detections": int(det_x.size), "n_synaptic": int(keep.sum()),
               "synaptic_frac": float(keep.mean()) if det_x.size else np.nan,
               "n_areas": int(len(used)), "note": ""}

    if len(used) < 3:
        summary["note"] = f"only {len(used)} usable regions -- correlation not computed"
    elif np.ptp(obs) == 0:
        summary["note"] = ("every region has the same synaptic density (usually none at all) -- "
                           "correlation undefined")
    elif np.ptp(ref) == 0:
        summary["note"] = "the reference is constant over the usable regions -- correlation undefined"

    if summary["note"]:
        summary.update({k: np.nan for k in
                        ["r_pearson", "r_spearman", "r_pearson_unw", "r_spearman_unw"]})
    else:
        so, sr = scale(obs), scale(ref)
        summary["r_pearson"] = float(weighted_corr(so, sr, w))
        summary["r_spearman"] = float(weighted_spearmanr(so, sr, w))
        summary["r_pearson_unw"] = float(pearsonr(so, sr)[0])
        summary["r_spearman_unw"] = float(spearmanr(so, sr)[0])

    if verbose:
        tag = f"[{setting}] " if setting else ""
        print(f"    {tag}{summary['n_synaptic']:,} synaptic of {summary['n_detections']:,} "
              f"({summary['synaptic_frac']:.1%}) | {summary['n_areas']} regions "
              f"(dropped {', '.join(drop_areas)})")
        if summary["note"]:
            print(f"    {tag}{summary['note']}")
        else:
            print(f"    {tag}weighted Pearson = {summary['r_pearson']:.4f}, "
                  f"weighted Spearman = {summary['r_spearman']:.4f}")

    return per_area_df, summary


def _collapse_subtype_key(subtype, subtype_names):
    """
    mcDETECT's collapse rule for a mapping key: anything containing " & " is `mixed`, otherwise the
    key stands as itself. Raises if the result is not a name mcDETECT's Categorical would keep --
    see `apply_manual_subtypes` for why that matters. Shared so the guard, the label column and the
    heatmap ordering can never disagree about what a key means.
    """
    collapsed = "mixed" if " & " in str(subtype) else str(subtype)
    if collapsed not in subtype_names:
        raise ValueError(
            f"subtype {subtype!r} collapses to {collapsed!r}, which is not one of {subtype_names} "
            f'-- it would be silently dropped to NaN. Use one of those names, or a combination '
            f'joined with " & ".')
    return collapsed


def cluster_subtype_order(mapping, subtype_names=None, n_clusters=None, include_unlisted=True):
    """
    Cluster ids arranged so that clusters sharing a compartment sit together, in mcDETECT's subtype
    order (`benchmark_subtyping.ipynb` cell 22, the "Ordered heatmap" block): all `pre-syn`
    clusters first, then `post-syn`, `dendrites`, `axons`, `mixed`, `others`, each group sorted
    numerically.

    Returns (ordered_ids, cluster_to_simple). `ordered_ids` is a list of str.

    ONE DEPARTURE FROM mcDETECT, deliberate. mcDETECT's published mapping assigns all 15 clusters,
    so it can build the Categorical from the ordered list alone. Ours may be partial by design, and
    doing the same would silently DROP every unlisted cluster from the panel -- an ordered heatmap
    showing fewer detections than the unordered one it is meant to be compared against. With
    `include_unlisted` (the default) those clusters are appended, in numeric order, after the
    assigned ones, so the two panels always hold the same detections.
    """
    subtype_names = list(C.SUBTYPE_NAMES) if subtype_names is None else list(subtype_names)
    n_clusters = C.K_SUBTYPE if n_clusters is None else n_clusters

    cluster_to_simple = {}
    for subtype, clusters in mapping.items():
        simple = _collapse_subtype_key(subtype, subtype_names)
        for c in clusters:
            cluster_to_simple[str(c)] = simple

    ordered = []
    for simple in subtype_names:
        ordered.extend(sorted([c for c, v in cluster_to_simple.items() if v == simple], key=int))
    if include_unlisted:
        ordered.extend(sorted([str(i) for i in range(n_clusters) if str(i) not in cluster_to_simple],
                              key=int))
    return ordered, cluster_to_simple


def render_ordered_subtype_heatmap(sub, mapping, out_path, title=None, subtype_names=None,
                                   n_clusters=None, include_unlisted=True,
                                   cluster_map_path=None, verbose=True):
    """
    Redraw the subtype heatmap with the clusters REORDERED so that each compartment forms one
    contiguous block -- mcDETECT's `heatmap_subtype_ordered.jpeg`.

    This is the third step of mcDETECT's subtyping loop, and the one that shows whether the manual
    call actually holds up: read the unordered panel, assign clusters to compartments, then redraw
    and check that each block really is coherent. `benchmark_subtyping.ipynb` cell 22 draws it at
    figsize=(10, 6.15) -- very slightly taller than the unordered (10, 6) -- and that is kept so
    the two sit together the way mcDETECT's do.

    Pass the SAME `sub` that `subtype_cluster` returned and `render_subtype_heatmap` drew, so each
    cluster keeps the colour it had in the unordered panel: the group colours are reordered with the
    categories, exactly as mcDETECT does. `sub` is restored to its original category order and
    colours before returning, so it can be redrawn either way afterwards.

    Returns the ordered cluster ids, or None when `mapping` assigns nothing (the first pass, where
    the ordering would be the identity and the panel a duplicate of the unordered one).
    """
    import matplotlib.pyplot as plt
    import scanpy as sc

    subtype_names = list(C.SUBTYPE_NAMES) if subtype_names is None else list(subtype_names)
    n_clusters = C.K_SUBTYPE if n_clusters is None else n_clusters
    key = "granule_subtype_kmeans"

    ordered, cluster_to_simple = cluster_subtype_order(
        mapping, subtype_names=subtype_names, n_clusters=n_clusters,
        include_unlisted=include_unlisted)
    if not cluster_to_simple:
        if verbose:
            print("    mapping is empty -- ordered heatmap skipped (it would duplicate the "
                  "unordered one)")
        return None

    if cluster_map_path is not None:
        rows = [{"position": i, "cluster": c,
                 "subtype": next((k for k, v in mapping.items() if str(c) in map(str, v)), None),
                 "subtype_simple": cluster_to_simple.get(c),
                 "n_detections": int((sub.obs[key].astype(str) == c).sum())}
                for i, c in enumerate(ordered)]
        Path(cluster_map_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(cluster_map_path, index=False)

    orig_cats = list(sub.obs[key].cat.categories)
    orig_colors = list(sub.uns.get(key + "_colors", []))
    try:
        if len(orig_colors) >= len(orig_cats):
            sub.uns[key + "_colors"] = [orig_colors[orig_cats.index(c)] for c in ordered]
        sub.obs[key] = pd.Categorical(sub.obs[key].astype(str), categories=ordered, ordered=True)

        present = [str(g) for g in sub.var_names]
        sc.pl.heatmap(sub, var_names=present, groupby=key, cmap="Reds", standard_scale="var",
                      dendrogram=False, swap_axes=True, show=False, figsize=(10, 6.15))
        if title:
            plt.gcf().suptitle(title, fontsize=11, y=1.02)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(out_path, dpi=500, bbox_inches="tight")
        plt.close("all")
    finally:
        # leave `sub` exactly as it was found, as mcDETECT restores adata_combined
        sub.obs[key] = pd.Categorical(sub.obs[key].astype(str), categories=orig_cats, ordered=True)
        if orig_colors:
            sub.uns[key + "_colors"] = orig_colors

    if verbose:
        blocks = ", ".join(f"{s}:{sum(1 for v in cluster_to_simple.values() if v == s)}"
                           for s in subtype_names
                           if any(v == s for v in cluster_to_simple.values()))
        n_un = sum(1 for c in ordered if c not in cluster_to_simple)
        print(f"    wrote {out_path.name}  [{blocks}"
              + (f"; {n_un} unlisted appended" if n_un else "") + "]")
    return ordered


def apply_manual_subtypes(labels, mapping, subtype_names=None, n_clusters=None):
    """
    Map k-means cluster ids to compartment labels, mcDETECT's `apply_manual_annotation` semantics.

    `mapping` is {subtype: [cluster ids]}; keys containing " & " collapse to "mixed"; clusters left
    unlisted stay NaN, which the density step counts in `overall` but in no per-subtype row.

    Three guards mcDETECT does not have, all against the same silent failure: a mapping that looks
    filled in but produces an all-NaN column, whose only symptom is a density table containing
    nothing but `overall` rows.

      * mcDETECT's lookup is keyed on the STRING form of the cluster id -- its own mapping reads
        `"pre-syn": ["0", "11", ...]` -- so a mapping written with integers matches nothing. Ids
        are coerced with str().
      * A duplicate or out-of-range id raises instead of passing quietly.
      * A KEY that is neither a combination (" & ") nor one of `subtype_names` raises. mcDETECT
        casts the simple column to a Categorical over exactly those names, so a typo like
        "presyn" or "pre-synaptic" is silently dropped to NaN, taking that whole cluster with it.

    Returns (fine, simple): the full label (e.g. "pre & post") and the collapsed one ("mixed").
    """
    subtype_names = list(C.SUBTYPE_NAMES) if subtype_names is None else list(subtype_names)
    n_clusters = C.K_SUBTYPE if n_clusters is None else n_clusters

    k2sub, seen = {}, {}
    for subtype, clusters in mapping.items():
        _collapse_subtype_key(subtype, subtype_names)      # raises on a key that would vanish
        for c in clusters:
            key = str(c)
            if key not in {str(i) for i in range(n_clusters)}:
                raise ValueError(f"cluster id {c!r} under {subtype!r} is not in 0..{n_clusters - 1}")
            if key in seen:
                raise ValueError(f"cluster {key} is listed under both {seen[key]!r} and {subtype!r}")
            seen[key] = subtype
            k2sub[key] = subtype

    fine = pd.Series(labels).astype(str).map(k2sub)
    simple = pd.Categorical(
        fine.apply(lambda s: "mixed" if pd.notna(s) and " & " in str(s) else s),
        categories=subtype_names, ordered=True)
    return fine.to_numpy(), simple
