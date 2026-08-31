"""
Configuration for analysis A2 -- sparsity and the stochastic origin of granule-level structure
(`plans/Round2_response_analysis_plan.md` section A2).

Two sub-analyses share this module:

  A2a  multi-gene granule reanalysis   -- local, `A2a_multigene.ipynb`
  A2b  label-permutation null          -- HGCC, `run_permutation_detect.py` + `score_embedding.py`

Every constant that is *copied* from published code carries a `# from <file>:<lines>` comment.
Nothing here is derived at import time from a data file, so this module is safe to import on a
login node.
"""

import os
from pathlib import Path


# ============================================================ paths ============================================================ #

# Same resolution rule as R2_revision/baysor_ssam_merscope/config.py:34-40 -- resolve from the
# repo root so the identical code runs locally and on HGCC, with env-var escape hatches.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DATA_ROOT = Path(os.environ.get("MCDETECT_DATA_ROOT", REPO_ROOT / "data"))
MCDETECT_OUT_ROOT = Path(os.environ.get("MCDETECT_OUT_ROOT", REPO_ROOT / "output"))

# This analysis writes only here (git-ignored via the global `output/` rule).
OUT_ROOT = Path(os.environ.get("A2_OUT_ROOT", SCRIPT_DIR / "output"))
A2A_DIR = OUT_ROOT / "a2a"
A2A_MULTIGENE_DIR = A2A_DIR / "multigene"
A2A_READSTRATA_DIR = A2A_DIR / "readstrata"
A2B_DIR = OUT_ROOT / "a2b"
A2B_METRICS_DIR = A2B_DIR / "metrics"
A2C_DIR = OUT_ROOT / "a2c"
A2E_DIR = OUT_ROOT / "a2e"

# ---------------------------------------------------------------------------------------------
# Samples. Pair 1 only -- MERSCOPE_WT_2 / AD_2 do not enter this manuscript.
# ---------------------------------------------------------------------------------------------
SAMPLES = ["WT", "AD"]
SAMPLE_DATASET = {"WT": "MERSCOPE_WT_1", "AD": "MERSCOPE_AD_1"}
COMPARISON_DIR = MCDETECT_OUT_ROOT / "MERSCOPE_WT_AD_comparison"


def dataset(sample):
    """'WT' -> 'MERSCOPE_WT_1'."""
    return SAMPLE_DATASET[sample]


def data_dir(sample):
    return DATA_ROOT / dataset(sample) / "processed_data"


def transcripts_path(sample):
    return data_dir(sample) / "transcripts.parquet"


def genes_path(sample):
    return data_dir(sample) / "genes.csv"


def nc_path(sample):
    return data_dir(sample) / "negative_controls.csv"


def spots_path(sample):
    return data_dir(sample) / "spots.h5ad"


def count_matrix_path(sample):
    return data_dir(sample) / "count.parquet"


def mcdetect_out_dir(sample):
    return MCDETECT_OUT_ROOT / dataset(sample)


def mcdetect_granule_adata_path(sample):
    """The published per-sample granule profile -- the 'real' arm of A2b."""
    return mcdetect_out_dir(sample) / "granule_adata_tsne.h5ad"


def mcdetect_granules_path(sample):
    return mcdetect_out_dir(sample) / "granules.parquet"


def mcdetect_reads_genes_path(sample):
    """Fig. R9 source, written by code/4_post_detection.ipynb cells 21-22 (with buffer=0.01)."""
    return mcdetect_out_dir(sample) / "granule_reads_unique_genes_per_granule.parquet"


# --- published combined (WT+AD) artifacts, the A2a inputs ------------------------------------
# Built by code/4_post_detection.ipynb cell 19 and code/5_neuropil_subdomains_data.py.
COMBINED_GRANULE_ADATA = COMPARISON_DIR / "granule_adata_tsne.h5ad"
COMBINED_SUBTYPE_LABELS = COMPARISON_DIR / "granule_subtype_labels_granule_adata_tsne.parquet"
COMBINED_DENSITY_TABLE = COMPARISON_DIR / "subtype_density_per_region_granule_adata_tsne.csv"

# Spatial scaffold for the microdomain step. These are READ, never recomputed -- recomputing the
# grid or the SpaGCN layer labels for a granule subset would change the scaffold along with the
# granules and confound the comparison.
SUBDOMAIN_GRANULE_ADATA = COMPARISON_DIR / "neuropil_subdomains_granule_adata.h5ad"
SUBDOMAIN_ADATA = COMPARISON_DIR / "neuropil_subdomains_adata.h5ad"
SUBDOMAIN_SPOTS_50 = COMPARISON_DIR / "neuropil_subdomains_spots_ambient.h5ad"
PUBLISHED_SUBDOMAIN_DIR = COMPARISON_DIR / "neuropil_subdomains_Isocortex_50"

# Real-arm reference for sanity-checking the A2b metric port (k=15 -> silhouette 0.2967,
# ARI stability 0.8179). Produced by code/benchmark/benchmark_clustering.py on WT alone.
BENCHMARK_CLUSTERING_CSV = (
    MCDETECT_OUT_ROOT / "benchmark" / "benchmark_clustering" / "benchmark_clustering_results.csv"
)

# --- Xenium 5K, read-only, A2e only -----------------------------------------------------------
# Xenium is deliberately NOT in SAMPLES: everything else in this module is pair-1 MERSCOPE, and
# the two platforms do not share a panel, a seed list or a subtype feature space. Xenium enters
# only as a third arm of the A2e complexity / same-category tables, never as a fourth A2b arm.
#
# Note the layout: Xenium's processed data sits directly under data/Xenium_5K/, with no
# processed_data/ subdirectory, so data_dir() above does NOT apply to it.
XENIUM_DATASET = "Xenium_5K"
XENIUM_DATA_DIR = DATA_ROOT / XENIUM_DATASET
XENIUM_NC = XENIUM_DATA_DIR / "negative_controls.csv"          # 293 rows, vs 19 for MERSCOPE
XENIUM_GRANULE_ADATA = MCDETECT_OUT_ROOT / XENIUM_DATASET / "granule_adata_tsne.h5ad"
XENIUM_SUBTYPE_LABELS = MCDETECT_OUT_ROOT / XENIUM_DATASET / "granule_subtype_labels.parquet"

# The 16 markers Xenium seeds on -- a DIFFERENT list from SYN_GENES, not a subset of it.
#                                     # from other_analysis/Xenium_5K/1_detection.ipynb cell 2
# Only granules.parquet / granule_adata_tsne.h5ad retain per-granule seed identity. The
# *_merged_genes* variants in the same directory carry gene == "merged" on every row and are
# useless for anything keyed on the seed marker.
XENIUM_SYN_GENES = ["Snap25", "Camk2a", "Slc17a7", "Cyfip2", "Map2", "Syp", "Syn1", "Slc32a1",
                    "Vamp2", "Mapt", "Gria2", "Gap43", "Tubb3", "Dlg4", "Gria1", "Bsn"]


# ============================================================ gene sets ============================================================ #

# The 20 granule markers mcDETECT seeds on.                       # from code/3_detection.py:55
SYN_GENES = ["Camk2a", "Cplx2", "Slc17a7", "Ddn", "Syp", "Map1a", "Shank1", "Syn1", "Gria1",
             "Gria2", "Cyfip2", "Vamp2", "Bsn", "Slc32a1", "Nfasc", "Syt1", "Tubb3", "Nav1",
             "Shank3", "Mapt"]

# Compartment markers used for subtyping.       # from code/benchmark/benchmark_subtyping.ipynb cell 4
MARKER_GENES = {
    "pre-syn": ["Bsn", "Gap43", "Nrxn1", "Slc17a6", "Slc17a7", "Slc32a1", "Snap25", "Stx1a",
                "Syn1", "Syp", "Syt1", "Vamp2", "Cplx2"],
    "post-syn": ["Camk2a", "Dlg3", "Dlg4", "Gphn", "Gria1", "Gria2", "Homer1", "Homer2",
                 "Nlgn1", "Nlgn2", "Nlgn3", "Shank1", "Shank3"],
    "axons": ["Ank3", "Nav1", "Sptnb4", "Nfasc", "Mapt", "Tubb3"],
    "dendrites": ["Actb", "Cyfip2", "Ddn", "Dlg4", "Map1a", "Map2"],
}

# The 34-gene feature space for K-means. Order matters -- it is the heatmap column order.
REF_GENES = ["Bsn", "Gap43", "Nrxn1", "Slc17a6", "Slc17a7", "Slc32a1", "Stx1a", "Syn1", "Syp",
             "Syt1", "Vamp2", "Cplx2", "Camk2a", "Dlg3", "Dlg4", "Gphn", "Gria1", "Gria2",
             "Homer1", "Homer2", "Nlgn1", "Nlgn2", "Nlgn3", "Shank1", "Shank3", "Cyfip2", "Ddn",
             "Map1a", "Map2", "Ank3", "Nav1", "Nfasc", "Mapt", "Tubb3"]


def marker_category_map(available=None):
    """gene -> compartment category, for the subtype markers only.

    MARKER_GENES lists `Dlg4` twice -- under post-syn and under dendrites -- so a naive inversion
    of that dict is ambiguous. First match over MARKER_GENES key order resolves it to post-syn,
    which is where REF_GENES' column order also groups it, so this map reproduces exactly the
    grouping the published subtype heatmap draws rather than inventing a convention.

    `available = None` gives the MERSCOPE 34: pre-syn 12, post-syn 13, dendrites 4, axons 5.
    Otherwise the map is restricted to genes the panel actually carries, and `Snap25` -- in
    MARKER_GENES but off the MERSCOPE panel, hence absent from REF_GENES -- is admitted if
    present. A Xenium var index therefore yields its 24-gene feature space: 11 of the 34 are
    off-panel there and Snap25 takes the count from 23 back to 24.
    """
    if available is None:
        genes = list(REF_GENES)
    else:
        available = set(available)
        genes = [g for g in list(REF_GENES) + ["Snap25"] if g in available]

    gene_to_cat = {}
    for gene in genes:
        for category, members in MARKER_GENES.items():
            if gene in members:
                gene_to_cat[gene] = category
                break
    return gene_to_cat


# --- A2e ---------------------------------------------------------------------------------------
# `axons` is empty in every published cluster -> subtype mapping (benchmark_subtyping.ipynb cell
# 21 and other_analysis/Xenium_5K/2_subtyping.ipynb cell 10 both send it to []), and no
# axon-containing label exists in either subtype parquet. So the pure-subtype loop is three
# categories. That is a fact about the K-means clusters and says nothing about the SEED axis,
# which keeps all four.
PURE_SUBTYPES = ["pre-syn", "post-syn", "dendrites"]
CONTENT_CATEGORIES = ["pre-syn", "post-syn", "dendrites", "axons"]

# The four marker sets are not four compartments -- they are TWO anatomical compartments, each
# split into two overlapping labels. Presynaptic terminals are axonal structures, and the
# postsynaptic density sits inside dendritic spines (Camk2a, Dlg4, Shank, Homer and Gria are PSD
# proteins on a dendritic structure; Map2, Cyfip2, Ddn and Map1a are dendritic shaft markers).
# Asking a sub-label to separate from the compartment it physically lives in is not a test the data
# can pass, so the compartment collapse is where same-category association is actually testable.
#
# Fixed a priori from standard neuroanatomy -- NOT a regrouping chosen after inspecting any table.
# It lives here beside MARKER_GENES, rather than in the notebook, so that claim is checkable.
COMPARTMENT_OF = {"pre-syn": "axonal", "axons": "axonal",
                  "post-syn": "dendritic", "dendrites": "dendritic"}
COMPARTMENTS = ["axonal", "dendritic"]

# Complexity thresholds reported per sample; 3 is the headline and matches MIN_UNIQUE_GENES.
A2E_COMPLEXITY_LEVELS = [2, 3, 4]

# A2e runs the seed-category test twice. `all_content` is the analysis as specified -- remove only
# the granule's own seed gene. `nonseed_content` additionally removes every seed marker, because
# merge_sphere() keeps one marker of a merged pair and discards the other (model.py:323-377), and
# 8 of the 20 MERSCOPE seeds are pre-syn: a pre-syn-seeded granule is therefore likelier to carry
# a second pre-syn marker by construction. Only `nonseed_content` is immune to that.
A2E_CONTENT_ARMS = ["all_content", "nonseed_content"]


# ============================================================ A2a ============================================================ #

# --- complexity cutoff ------------------------------------------------------------------------
# The reviewer asked for a stratification by granule complexity. `comp` in granules.parquet is
# NOT the right column: mcDETECT_package/mcDETECT/model.py:102 restricts the working frame to
# `gnl_genes` before detection, so model.py:264-266 counts distinct GRANULE MARKERS (capped at
# 20, empirical max 19), not distinct panel genes; it is also never recomputed after
# merge_sphere(). The complexity we filter on is derived from profile(), which counts all panel
# genes inside the sphere:  n_genes = (granule_adata.layers["counts"] > 0).sum(axis=1).
MIN_UNIQUE_GENES = 3          # lower to 2 if retention is poor -- see A2a section 1

# The 19 negative controls are real nuclear-enriched panel genes (this panel has no blank
# probes), so "unique genes" is ambiguous. Primary count excludes them; the all-290 count is
# carried alongside as a sensitivity column.
EXCLUDE_NC_FROM_COMPLEXITY = True

READ_TERCILE_LABELS = ["low", "mid", "high"]

# --- subtyping ------------------------------- from benchmark_subtyping.ipynb cells 13, 22 ----
K_SUBTYPE = 15
KMEANS_BATCH_SIZE = 5000
KMEANS_N_INIT = 20
SUBTYPE_SEED = 1              # the published main result uses seed 1
SUBTYPE_ORDER = ["pre-syn", "post-syn", "dendrites", "axons", "mixed", "others"]

# --- density --------------------------------- from benchmark_subtyping.ipynb cells 13, 22 ----
AREA_LIST = ["Isocortex", "OLF", "HPF-CA", "HPF-DG", "HPF-SR", "CTXsp", "TH", "MB", "FT"]
SPOT_GRID = 50
N_BOOTSTRAP = 500
# AD counts/densities are divided by this to correct for capture efficiency.
CAPTURE_EFFICIENCY_COEF = 0.818691

# --- microdomains ---------------------------- from 7_neuropil_subdomains.ipynb cells 1, 9 ----
ROI = "Isocortex"
K_SUBDOMAIN = 4
SUBDOMAIN_SEED = 42
SUBDOMAIN_K_RANGE = range(2, 10)
SUBDOMAIN_BATCH_SIZE = 1000
STABILITY_SEEDS = [0, 42, 123, 456, 789]
PLOT_FIGSIZE = (12.5, 5.5)
PLOT_SPOT_SIZE = 80
LAYER_COLORS = ["#F56867", "#FEB915", "#C798EE", "#59BE86", "#7495D3", "#997273"]

# Published seed-1 cluster -> compartment mapping, kept ONLY as a worked reference for filling in
# the A2a notebook's MANUAL_SUBTYPE_MAPPING. It does NOT transfer: the subset is clustered
# separately, so its cluster ids mean something different.
PUBLISHED_SUBTYPE_MAPPING_SEED1 = {
    "pre-syn": ["0", "11", "12", "13"],
    "post-syn": ["1", "2", "3"],
    "dendrites": ["4", "8"],
    "axons": [],
    "pre & post": ["6"],
    "pre & den": [],
    "post & den": ["5", "9", "10"],
    "pre & post & den": ["7", "14"],
    "others": [],
}

# Published Subdomain contrast, kept as a reference anchor for the GSEA comparison in R.
# The A2a subdomains are recomputed and are NOT guaranteed to carry these labels -- see the
# SUBDOMAIN_PAIRS EDIT ME block in the notebook.
PUBLISHED_SUBDOMAIN_PAIR = ("Subdomain 1", "Subdomain 2")

# WT/AD colors, kept in step with A2_figures.R (R cannot import this module).
WT_COLOR = "#a0ccec"
AD_COLOR = "#f48488"


# ============================================================ registration ============================================================ #
# Canvas-alignment geometry, needed to rebuild the combined WT+AD object the way the published
# pipeline does. None of it affects the embedding -- these are obs columns only -- but keeping
# the permuted objects structurally identical to the published one costs nothing and makes them
# usable by A2c.
#
# theta is in DEGREES here so this module stays numpy-free and importable on a login node;
# the consumer converts.
REGISTRATION = {                                             # from code/3_detection.py:19-30
    "MERSCOPE_WT_1": dict(flip=True, cutoff=6250, theta_deg=10.0,
                          rotate_cols=["sphere_y", "sphere_x"], flip_col="global_y"),
    "MERSCOPE_AD_1": dict(flip=False, cutoff=-4200, theta_deg=170.0,
                          rotate_cols=["sphere_x", "sphere_y"], flip_col="global_x"),
}

# from code/4_post_detection.ipynb cell 19
COMBINED_SHIFT_X = 12000
COMBINED_SHIFT_Y = 7200
COMBINED_CUTOFF = 6250


# ============================================================ A2c ============================================================ #
# Functional co-clustering of co-detected genes.
#
# Runs on the FULL published detection, not A2a's subset: conditioning on unique-gene count would
# select on the very statistic being measured. Single-gene granules contribute no pairs under
# either the observed data or the null, so they are harmless.

def panel_annotation_path(sample="WT"):
    """The panel's own curated annotation. Richer than REF_GENES and, crucially, it carries both
    localization groups and co-expression groups -- see PANEL_GROUP_COLUMNS."""
    return data_dir(sample) / "gene_panel.csv"


# Which annotation columns to use, and what kind of gene programme each represents.
#
# This split is the whole point. If mcDETECT granules are packaged transport structures, the
# LOCALIZATION groups should co-occur above chance. If they were merely local co-expressed
# transcript clusters -- Reviewer #2's alternative -- then the CO-EXPRESSION groups (cell-type,
# regional and layer marker sets) would co-occur at least as strongly, because those genes are
# co-expressed by definition. The contrast is the test; neither arm is decoration.
PANEL_GROUP_COLUMNS = {
    "Synapse markers": "localization",       # pre-syn, post-syn
    "Neuropil": "localization",              # Neuropil, Dendrites, Axons
    "Cell type markers": "co-expression",    # 8 cell types
    "Region markers": "co-expression",       # 15 regions
    "Layer markers": "co-expression",        # 6 cortical layers
}

# "Xenium" appears in the Neuropil column as a panel-provenance tag (which genes came from the
# Xenium panel), not a functional group -- it would be meaningless as a co-occurrence set.
PANEL_EXCLUDE_LEVELS = {"Xenium"}

# A group needs this many NON-SEED genes to be tested; below it the within-group pair count is too
# small to say anything. Groups that fall out are reported, never dropped silently.
MIN_GROUP_SIZE = 4

# Null model. Degree-preserving CURVEBALL trade chain, implemented in
# a2_common.curveball / a2_common.cooccurrence_enrichment. It holds both margins EXACTLY -- each
# granule keeps its number of distinct genes, each gene keeps how many granules it appears in --
# which removes the two effects that would otherwise masquerade as co-occurrence.
#
# A bipartite configuration model (maximum entropy, degrees fixed only IN EXPECTATION) was the
# original choice and was rejected on measurement, so it has no constants here: a granule with
# exactly k genes contributes exactly C(k,2) pairs, while a soft-degree null contributes ~k^2/2 --
# a factor k/(k-1) that at the median complexity of ~5 genes over-estimates every pair by ~25%,
# and inflated z by ~96 sd on simulated data. Curveball's constraints are hard, so it has no such
# bias. See README.md, "How z is computed, for one gene pair".

GROUP_TEST_N_PERM = 2000        # gene -> group label permutations
GROUP_TEST_SEED = 0

# Validation (notebook section 6)
CURVEBALL_SUBSAMPLE = 50_000
CURVEBALL_N_RANDOM = 20
CURVEBALL_SEED = 0

# Programme colours, kept in step with A2_figures.R (R cannot import this module).
PROGRAMME_COLORS = {"localization": "#4f7fa8", "co-expression": "#d98b5f"}


# ============================================================ A2b ============================================================ #

# Detection kwargs, verbatim from the fine pass.               # from code/3_detection.py:82-84
DETECT_KWARGS_FINE = dict(type="discrete", eps=1.5, minspl=3, grid_len=1, cutoff_prob=0.95,
                          alpha=10, low_bound=3, size_thr=4.0, in_soma_thr=0.1, l=1, rho=0.2,
                          s=1, nc_top=20, nc_thr=0.1)

# Rough pass -- all filters off.                               # from code/3_detection.py:67-69
DETECT_KWARGS_ROUGH = dict(type="discrete", eps=1.5, minspl=3, grid_len=1, cutoff_prob=0.95,
                           alpha=10, low_bound=3, size_thr=1e5, in_soma_thr=1.01, l=1, rho=0.2,
                           s=1, nc_top=20, nc_thr=0.1)

N_PERM = 5                    # permutations per sample -> 2 x 5 = 10 detection tasks
PERM_SEEDS = list(range(N_PERM))

# Permutation replicate s pairs (WT seed s, AD seed s) into ONE combined embedding, because the
# embedding the paper reports (Fig. 3f subtypes, Fig. 4d t-SNE) is the combined WT+AD object from
# code/4_post_detection.ipynb cell 19 -- not a per-sample one. So: 10 detections -> 5 null
# embeddings -> 6 scoring arms (1 real + 5 permuted).

RUN_ROUGH_PASS = True         # the rough pass roughly doubles detection cost; first lever to pull
RUN_TSNE = True
RECOMPUTE_REAL_TSNE = True    # recompute the real arm's t-SNE so every arm is embedded by the
                              # same call with the same thread count; the published X_tsne in
                              # output/MERSCOPE_WT_AD_comparison/ is never overwritten

# Threads. Resolved at run time so the same code works locally and under SLURM.
N_JOBS = None                 # None -> $SLURM_CPUS_PER_TASK, else os.cpu_count()

# Degenerate-null guards. If the permuted arm collapses, that is a RESULT and must be recorded
# as one -- never a traceback, and never a silently dropped arm.
MIN_EMBED_N = 500             # below this, skip the embedding and write an explanatory row
TSNE_PERPLEXITY = 30          # scanpy's default, restated so the n_obs > 3 * perplexity guard
                              # has a single source
MATCH_SEED = 0                # subsample seed for the size-matched control series
# Full-population t-SNE is rendered for every arm. The size-matched PAIR (real and permuted at
# identical n, the honest side-by-side) is rendered only for these seeds -- one representative
# pair is what the figure needs, and each render is a full Barnes-Hut run.
TSNE_MATCHED_SEEDS = [0]

# Embedding-structure scoring. Same three metrics and the same seed list as
# code/benchmark/benchmark_clustering.py:92-121, with two deliberate departures, both applied
# identically to every arm so the comparison stays fair:
#   (1) n_init=20 (the published subtyping value) rather than that script's sklearn default --
#       so the numbers in output/benchmark/benchmark_clustering/benchmark_clustering_results.csv
#       are NOT directly comparable to the ones produced here;
#   (2) silhouette_score is O(n^2) in distances, so it is evaluated on a fixed random subsample.
SCORE_K_RANGE = range(2, 31)
SILHOUETTE_SAMPLE_SIZE = 50_000
SILHOUETTE_SEED = 0

# Pre-binned histogram edges for the R hand-off (A1 postproc idiom: export quantiles + bins,
# never millions of raw values).
HIST_BINS = {
    "sphere_r": (0.0, 4.0, 50),      # (lo, hi, n_bins); filtered at size_thr = 4.0
    "size": (0.0, 60.0, 60),
    "n_genes": (0.0, 60.0, 60),
    "n_reads": (0.0, 100.0, 50),
}


def perm_dir(sample, seed):
    """Per-sample permuted detection output (granule tables + raw profile)."""
    return A2B_DIR / f"perm_{sample}_seed{seed}"


def combined_dir(seed=None):
    """Combined WT+AD object for one arm. seed=None is the real arm."""
    return A2B_DIR / ("combined_real" if seed is None else f"combined_seed{seed}")


def arm_name(seed=None):
    """'real' or 'perm_seed3' -- the arm key used across every A2b output table."""
    return "real" if seed is None else f"perm_seed{seed}"


def all_arms():
    """Every arm scored by score_embedding.py, in array-index order: real, then each permutation.

    An arm is a COMBINED WT+AD object, so there are 1 + N_PERM of them, not 2 * (1 + N_PERM).
    """
    return [None] + list(PERM_SEEDS)


def series_name(seed, matched=False, of="perm"):
    """Label for one metric series within an arm.

    Because permuted and real arms will not contain the same number of granules, and both
    silhouette and ARI stability depend on n, every permuted arm also emits a size-matched pair:
    'matched_perm_seed<s>' and 'matched_real_seed<s>', both cut to min(n_real, n_perm). The
    matched pair is the headline comparison; the full-n series ride along with n_obs on every row.
    """
    if seed is None:
        return "real"
    return f"matched_{of}_seed{seed}" if matched else f"perm_seed{seed}"


def series_condition(series):
    """'real' or 'permuted' for a series label.

    Explicit rather than a substring test: 'matched_real_seed0' is a REAL series and
    'matched_perm_seed0' is a permuted one, and neither is caught by a naive endswith.
    """
    return "real" if series == "real" or series.startswith("matched_real") else "permuted"


def resolve_n_jobs(n_jobs=None):
    """Threads to use: explicit -> config -> $SLURM_CPUS_PER_TASK -> os.cpu_count()."""
    if n_jobs is None:
        n_jobs = N_JOBS
    if n_jobs is None:
        n_jobs = os.environ.get("SLURM_CPUS_PER_TASK")
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    return max(1, int(n_jobs))


def ensure_dirs():
    for d in [A2A_MULTIGENE_DIR, A2A_READSTRATA_DIR, A2B_DIR, A2B_METRICS_DIR, A2C_DIR,
              A2E_DIR]:
        d.mkdir(parents=True, exist_ok=True)
