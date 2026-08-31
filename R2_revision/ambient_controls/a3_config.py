"""
Configuration for analysis A3 -- ambient RNA controls at the DETECTION step
(`plans/Round2_response_analysis_plan.md` section A3, Reviewer #2 major point 9).

Three sub-analyses share this module:

  A3a  three-set NC split + locally-adaptive threshold re-test  -- local, `A3a_three_sets.ipynb`
  A3b  vicinity pseudo-granule control                          -- local, `A3b_vicinity.ipynb`
  A3c  somatic-vs-non-somatic DE baseline                       -- local, `A3c_de_baseline.ipynb`
  A3d  local-neighbourhood permutation null                     -- local, `A3d_local_null.ipynb`
  A3e  ambient-relabelled pseudo-granules, re-detected          -- both,  `A3e_pseudo_granules.ipynb`

plus two HGCC stages: `run_detection_sets.py`, which produces the Set 1 / Set 3 / Set 0 detections
that A3a and A3b consume, and `run_pseudo_detection.py`, which re-runs the published pipeline over
A3e's relabelled transcript table.

Every constant that is *copied* from published code carries a `# from <file>:<lines>` comment.
Nothing here is derived at import time from a data file, so this module is safe to import on a
login node. Anything that must be derived from data (Set 0's abundance-matched gene list) is a
function in `a3_common.py` whose result is persisted to `output/`.
"""

import os
from pathlib import Path


# ============================================================ paths ============================================================ #

# Same resolution rule as R2_revision/sparsity_structure/a2_config.py:22-30 -- resolve from the
# repo root so the identical code runs locally and on HGCC, with env-var escape hatches.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

DATA_ROOT = Path(os.environ.get("MCDETECT_DATA_ROOT", REPO_ROOT / "data"))
MCDETECT_OUT_ROOT = Path(os.environ.get("MCDETECT_OUT_ROOT", REPO_ROOT / "output"))

# This analysis writes only here (git-ignored via the global `output/` rule).
OUT_ROOT = Path(os.environ.get("A3_OUT_ROOT", SCRIPT_DIR / "output"))
PREFLIGHT_DIR = OUT_ROOT / "preflight"     # stage 0, local, must precede the HPC jobs
DETECT_DIR = OUT_ROOT / "detect"           # stage 1, HGCC -- Sets 0 / 1 / 3 (6 array tasks)
A3A_DIR = OUT_ROOT / "a3a"
A3B_DIR = OUT_ROOT / "a3b"
A3C_DIR = OUT_ROOT / "a3c"
A3D_DIR = OUT_ROOT / "a3d"           # stage 2, local -- the local-neighbourhood null
FIG_DIR = OUT_ROOT / "figures"             # written ONLY by A3_figures.R


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
    """The 19 negative-control genes. Real nuclear-enriched panel genes, NOT blank probes --
    this panel has none. See NC_GENES_ARE_REAL_GENES below."""
    return data_dir(sample) / "negative_controls.csv"


def panel_path(sample="WT"):
    """The panel's curated annotation, used to pick Set 0."""
    return data_dir(sample) / "gene_panel.csv"


def spots_path(sample):
    return data_dir(sample) / "spots.h5ad"


def mcdetect_out_dir(sample):
    return MCDETECT_OUT_ROOT / dataset(sample)


def mcdetect_granules_path(sample):
    """SET 2 -- the published fine-pass detection. Never recomputed by A3."""
    return mcdetect_out_dir(sample) / "granules.parquet"


def mcdetect_all_granules_path(sample):
    """The rough pass, all filters off. Used by A3b's zero-placement-bias variant."""
    return mcdetect_out_dir(sample) / "all_granules.parquet"


def mcdetect_granule_adata_path(sample):
    return mcdetect_out_dir(sample) / "granule_adata_tsne.h5ad"


# --- published combined (WT+AD) artifacts ----------------------------------------------------
# The granule object on the COMBINED CANVAS, already carrying granule_subtype_kmeans
# (5_neuropil_subdomains_data.py:203-219). Use this for anything paired with the spot grid.
#
# NOT granule_adata_tsne.h5ad: its obs["global_x"] is each sample's OWN raw frame (219..6967)
# while the spots are on the canvas (-14..12014), so pairing them assigns nearly every granule to
# the wrong spot or none. A3c asserts the ranges overlap before using it.
SUBDOMAIN_GRANULE_ADATA = COMPARISON_DIR / "neuropil_subdomains_granule_adata.h5ad"

# Spot scaffolds. Both are READ, never recomputed.
#   _spots.h5ad         25 um, 121,084 spots, WHOLE SECTION
#   _spots_ambient.h5ad 50 um,   4,310 spots, Isocortex + FT ONLY  (what 7_neuropil reads)
# NEITHER carries an `intrasomatic_transcripts` layer, despite
# code/5_neuropil_subdomains_data.py:242-243 writing one -- the saved files predate those lines.
# A3c therefore builds its somatic layer itself; see SOMATIC_LAYER_NOTE.
SUBDOMAIN_SPOTS_25 = COMPARISON_DIR / "neuropil_subdomains_spots.h5ad"
SUBDOMAIN_SPOTS_50 = COMPARISON_DIR / "neuropil_subdomains_spots_ambient.h5ad"
PUBLISHED_SUBDOMAIN_DIR = COMPARISON_DIR / "neuropil_subdomains_Isocortex_50"

SOMATIC_LAYER_NOTE = (
    "Do NOT derive the somatic layer as X - extrasomatic: the two were filled by different "
    "assignment rules and X >= extrasomatic fails in 3.07% of cells of the 50um object "
    "(worst deficit 46 transcripts). Rebuild it with fill_spot_expression on "
    "transcripts[overlaps_nucleus == 1], exactly as code/5_neuropil_subdomains_data.py:242-243 "
    "does, so somatic and non-somatic share one counting convention."
)

# A3's own Axis-1 precedent, revived and extended by A3c. Lives in gitignored code/old/, and its
# output CSV is no longer on disk. Its R panel is code/figures_response.Rmd:1424-1452, under a
# heading literally titled "Reviewer 2, Major Comment 9".


# ============================================================ gene sets ============================================================ #

# The 20 granule markers mcDETECT seeds on.                       # from code/3_detection.py:55
SYN_GENES = ["Camk2a", "Cplx2", "Slc17a7", "Ddn", "Syp", "Map1a", "Shank1", "Syn1", "Gria1",
             "Gria2", "Cyfip2", "Vamp2", "Bsn", "Slc32a1", "Nfasc", "Syt1", "Tubb3", "Nav1",
             "Shank3", "Mapt"]

# The 34-gene compartment feature space (heatmap column order).
# from R2_revision/sparsity_structure/a2_config.py:135-139
REF_GENES = ["Bsn", "Gap43", "Nrxn1", "Slc17a6", "Slc17a7", "Slc32a1", "Stx1a", "Syn1", "Syp",
             "Syt1", "Vamp2", "Cplx2", "Camk2a", "Dlg3", "Dlg4", "Gphn", "Gria1", "Gria2",
             "Homer1", "Homer2", "Nlgn1", "Nlgn2", "Nlgn3", "Shank1", "Shank3", "Cyfip2", "Ddn",
             "Map1a", "Map2", "Ank3", "Nav1", "Nfasc", "Mapt", "Tubb3"]

# --- the Gria2 collision ---------------------------------------------------------------------
# Gria2 is on BOTH the 20-gene marker list and the 19-gene negative-control list. The policy is
# fixed and not revisited anywhere in A3:
#
#   the 20-marker SEED list   -- never modified. Sets 1 and 2 both seed on all 20.
#   Set 3's SEED list         -- 18 (Gria2 dropped). Seeding a control on a canonical dendritic
#                                marker would manufacture the very overlap Set 3 exists to bound;
#                                Gria2 is ~13x more abundant than the median control gene, so
#                                keeping it could roughly double Set 3.
#   Set 2                     -- read from disk exactly as published, on the 19-gene filter it
#                                was actually built with.
#
# The residual discrepancy this accepts is under half a percent of Set 1, in the same direction in
# both samples, so it cannot move the WT/AD contrast. It is not analysed further.
GRIA2 = "Gria2"
SET3_EXCLUDE = [GRIA2]

# The 20-marker SEED list is never modified by A3.
SYN_GENES_UNCHANGED = True

# The two NC-list versions, by name, so consumers never have to re-derive the rule. Both strings
# say "NC list" on purpose: "18"/"19" alone reads as a seed count and has already caused that
# confusion once.
NC_LIST_NEW_USE = "NC list, 18 genes (Gria2 dropped)"
NC_LIST_PUBLISHED = "NC list, 19 genes (as published)"

# The NC list is nuclear-enriched (manuscript: "enriched in neuronal nuclei compared to the
# cytoplasm", Supp. Table 8) but is NOT gene-neutral: the file is an edgeR table whose Cluster
# column spans C4a (complement), Cyfip1, Abca7 (AD risk), Opalin (oligodendrocyte),
# Prox1/Npnt/Zfpm2 (dentate gyrus). So the NC background is itself spatially structured and
# plausibly condition-dependent -- Reviewer #2's own complaint, applied to the filter. Hence the
# leave-one-out and the AD/WT NC-density ratio in stage 0.
# nc_top=20 against a 19-gene list makes nc_filter's "top 20 by expression" step a no-op.


# --- Set 0: abundance-matched neutral genes ---------------------------------------------------
# Set 3 alone only shows "nuclear genes stay in nuclei" -- circular, since the NC genes are
# DEFINED as nuclear-enriched and are then filtered on nuclear overlap. Set 0 shows that
# arbitrary genes AT MARKER ABUNDANCE do not form granule-like aggregates, which is what actually
# addresses structured ambient. It also neutralises the abundance confound: median per-gene
# transcripts are 1,153,633 for markers vs 74,893 for NC genes -- a 15x gap, and DBSCAN yield is
# strongly superlinear in count.
#
# Selection (a3_common.select_set0, persisted to preflight/set0_genes.csv):
#   exclude SYN_GENES, the 19 NC genes, and any gene annotated in the panel columns below;
#   then for each marker pick the nearest unannotated gene in log10(transcript count).
SET0_N = 20
SET0_EXCLUDE_PANEL_COLUMNS = ["Synapse markers", "Neuropil", "Negative controls"]


# ============================================================ detection ============================================================ #

# Detection kwargs, verbatim from the published fine pass.     # from code/3_detection.py:82-84
# minspl=3 is kept EXACTLY as published so Sets 0/1/3 are constructed the same way as Set 2.
DETECT_KWARGS_FINE = dict(type="discrete", eps=1.5, minspl=3, grid_len=1, cutoff_prob=0.95,
                          alpha=10, low_bound=3, size_thr=4.0, in_soma_thr=0.1, l=1, rho=0.2,
                          s=1, nc_top=20, nc_thr=0.1)
# `nc_top` / `nc_thr` are inert for every A3 set: run_detection_sets.py calls dbscan() +
# merge_sphere() directly and never nc_filter() (all sets use nc_genes=None).

# Same, with the two per-gene filters OFF.                     # from code/3_detection.py:66-67
#
# This is how detection is actually run, and it is not a second pass -- it is the SAME pass.
# mcDETECT applies the size and in-soma filters at the END of dbscan() (model.py:288), i.e.
# BEFORE merge_sphere(). So calling dbscan() with the filters off and then applying the identical
# row-wise predicate per gene in pandas hands merge_sphere() the byte-identical input it would
# have received from the fine pass -- while also exposing the PRE-filter counts, without which
# the funnel's `raw` stage equals its `in_soma` stage and answers nothing.
DETECT_KWARGS_ROUGH = {**DETECT_KWARGS_FINE, "size_thr": 1e5, "in_soma_thr": 1.01}

# --- the CSR disclosure ------------------------------------------------------------------------
# code/3_detection.py:66,83 passes minspl=3, so poisson_select() -- the CSR background model
# Reviewer #2 objects to -- never runs on real data (alpha=10 and cutoff_prob=0.95 are inert).
# At the alpha=10 written in that call the CSR rule would have been far STRICTER than what was
# used (Camk2a -> 32; >3 for 19 of 20 markers), which would make the gap damaging.
#
# But alpha is a free scaling factor, and at alpha = 0.5 the CSR selector returns EXACTLY 3 for
# all 20 markers in BOTH samples (verified; 0.25 also works, 0.75 works for AD but not WT).
# So min_samples = 3 IS the CSR rule at alpha = 0.5, and no result changes.
#
# A3 RUNS NO DETECTION AT alpha = 0.5, and re-detects nothing to prove the equivalence.
# poisson_select
# is a deterministic function of one gene's transcript count and the tissue area, so
# "alpha = 0.5 => min_samples = 3 for every marker in both samples" is an ARITHMETIC IDENTITY --
# checked by preflight/csr_min_samples.csv and one assertion in the A3a gates. DBSCAN is
# deterministic given min_samples, so minspl=None/alpha=0.5 and minspl=3 are provably the same
# run; the published populations are therefore reused as they are, and every detection here uses
# minspl=3 verbatim (see DETECT_KWARGS_FINE above).
#
# Separate action item, outside A3 and not blocking it: change code/3_detection.py to
# minspl=None, alpha=0.5 and update Supp. Note 10 / Methods to state alpha = 0.5, so the code
# says what the response says. That is a Methods-consistency edit, NOT a re-run.
CSR_ALPHA_EQUIV = 0.5
CSR_ALPHA_SWEEP = [0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]
CSR_EXPECTED_MIN_SAMPLES = 3      # what poisson_select must return at CSR_ALPHA_EQUIV

# The set names used across every A3 output table.
SETS = ["set0", "set1", "set2", "set3"]
SET_LABEL = {
    "set0": "Set 0 (abundance-matched unannotated genes)",
    "set1": "Set 1 (granule markers, before NC filtering)",
    "set2": "Set 2 (published)",
    "set3": "Set 3 (nuclear-enriched control genes)",
}
# Which sets need a new detection run; set2 is read from output/ and never rewritten.
# 3 sets x 2 samples = 6 array tasks.
#
# Order is load-bearing: it fixes the SLURM array index, and the two cheap sets come first so an
# --array=0-3 slice validates the whole path end-to-end before set1's two 200G jobs are committed
# (set1 is then --array=4-5).
SETS_TO_DETECT = ["set0", "set3", "set1"]

# The funnel stages reported side by side for every set. Reporting Set 3 only at the end would be
# circular -- if it empties ONLY at the in-soma step that proves nothing, because NC genes are
# defined as nuclear-enriched. If it is already near-empty BEFORE that step, that is a result.
FUNNEL_STAGES = ["raw", "size", "in_soma"]


# ============================================================ A3a ============================================================ #

# --- overlap ladder ---------------------------------------------------------------------------
# mcDETECT's own merge predicate (model.py:349-353, with l=1, rho=0.2) is
#     merge(A,B)  <=>  d <= |r_A - r_B|   (containment)   OR   d < 0.2 * (r_A + r_B)
# i.e. two equal-radius spheres merge only within 0.4*r. That is VERY strict -- real granules
# routinely overlap without merging -- so reporting only that predicate would understate
# co-location and read as rigged. Lead with the loosest criterion instead.
# The three boolean rungs, in report order. Both directions are scored: what fraction of GRANULES
# meet a control aggregate, and what fraction of CONTROL aggregates meet a granule. `center_in` is
# asymmetric by construction ("b's centre lies inside a"), so under that rung the reverse
# direction is the mirrored predicate, not the same number read backwards.
OVERLAP_CRITERIA = ["intersect", "center_in", "merge"]
OVERLAP_PRIMARY = "intersect"          # d < r_A + r_B
MERGE_RHO = 0.2                        # from code/3_detection.py:84
MERGE_L = 1.0                          # from code/3_detection.py:84

# _remove_overlaps and dbscan use sphere_z; nc_filter and profile use layer_z. Pick one, apply it
# identically to every set, and disclose the inherited inconsistency.
OVERLAP_Z_COL = "sphere_z"

# A raw intersection count is uninterpretable without an expectation, so the granule-side
# fraction is reported as observed/expected against control spheres randomly re-placed in the
# tissue mask at matched radius and matched layer_z. Each control set gets its own draws (the
# seed carries the control name), because the null depends on that set's radii and z-plane
# distribution. The control-side fraction carries no null: it is a plain ceiling -- "at most this
# share of ambient aggregates touches a granule at all" -- and needs no expectation to be read.
OVERLAP_N_NULL = 20
OVERLAP_NULL_SEED = 0
# The two control populations, scored identically against both granule sets.
OVERLAP_CONTROLS = ["set0", "set3"]
OVERLAP_BASES = ["set1", "set2"]

# Any recount of a cluster's own transcripts by ball query on the FINAL sphere geometry must
# carry a buffer.
# sphere_r is the MINIMUM-ENCLOSING radius (miniball.get_bounding_ball), so the cluster's own
# support points sit exactly ON the surface and a query at exactly sphere_r loses them to
# floating point. Measured on the 321,053 Camk2a-seeded WT granules: at sphere_r only 43.6% have
# k >= 3, at sphere_r * (1 + 1e-9) it is 99.88%. Without it a containment query reports a
# floating-point artefact as a real rate; see the A3b detection-predicate note.
# 0.01 um is the published convention -- mcDETECT.profile() takes `sphere_r + buffer`
# (model.py:451) and code/4_post_detection.ipynb cells 21-22 use buffer = 0.01 for the Fig. R9
# distributions; the A2 record documents the same support-point effect.
KG_BUFFER = 0.01

# A3b section 7 gate: the fraction of REAL granules whose own seed gene still satisfies the DBSCAN
# core-point criterion inside their own sphere. This must be ~1 by construction -- the cluster was
# formed by that very criterion. The gate previously sat at 0.95, which was loose enough to accept
# the 0.982 produced by the missing KG_BUFFER; with the buffer applied it should clear 0.99.
PREDICATE_REAL_MIN = 0.99

# ============================================================ A3b ============================================================ #

# Offsets are IN-PLANE. layer_z takes only 7 discrete values (0, 1.5 ... 9.0) and both profile()
# and nc_filter() query at layer_z, so a 3D direction would push the centre off the grid.
VICINITY_D = [5.0, 10.0, 20.0, 50.0]      # um, absolute
VICINITY_D_RELATIVE = [2.0, 3.0]          # multiples of the source granule's sphere_r
VICINITY_MAX_RETRY = 20
VICINITY_SEED = 0

# Two arms. Per-plane 2D granule coverage is only 1.9% (WT) / 1.5% (AD)
# (sum(pi r^2) / (tissue_area * 7 planes)), so rejecting granule-overlapping offsets removes few
# candidates and does NOT meaningfully bias the sample toward granule-sparse space. (The 3D
# nearest-neighbour distance of 2.68 um is misleading -- it counts neighbours on adjacent
# z-planes.) Both arms are still reported.
VICINITY_ARMS = {
    # reject only in-nucleus and out-of-tissue; report the real-granule overlap fraction as a
    # RESULT (it measures how much of the vicinity is already called)
    "unrejected": dict(reject_granule_overlap=False),
    # additionally reject granule-overlapping offsets, using mcDETECT's OWN merge predicate and
    # nothing stricter -- a pseudo-granule that merely intersects a real granule is no more
    # inadmissible than two real granules that intersect
    "rejected": dict(reject_granule_overlap=True, criterion="merge"),
}

# The load-bearing statistic. A matched-radius COUNT comparison is a tautology: sphere_r is the
# minimum enclosing radius of the DBSCAN core points (miniball on deduplicated cluster coords),
# an order statistic, so the sphere is maximally dense by construction and any displaced copy of
# the same radius must capture <=. Report counts descriptively; decide on the DETECTION PREDICATE.
#
# The predicate is eps-CONNECTIVITY, not a count: three transcripts scattered across a 4um sphere
# are not eps=1.5-connected, so ">=3 inside" massively overstates detectability. A DBSCAN core
# point is by definition a point with >= min_samples neighbours within eps, so
# a3_common.dbscan_core_predicate asks exactly that -- for a transcript of the SEED GENE lying
# INSIDE the sphere. No margin and no clustering fit are involved.

# STRATIFY on the covariate we are accused of ignoring -- these are NOT placement constraints.
# place_vicinity_spheres draws a uniform-random in-plane angle and rejects only out-of-tissue,
# in-nucleus and (rejected arm) granule-overlapping offsets. Region and density quintile are
# labelled on the SOURCE granule and inherited by the copy, because a <= 50 um displacement
# rarely leaves a 25 um density cell or a brain region; the results are then reported WITHIN
# each level. That is a stratified comparison, not a matched design -- say so when describing it.
VICINITY_MATCH_ON = ["brain_area", "density_quintile"]
VICINITY_DENSITY_GRID = 25.0
VICINITY_DENSITY_N_BINS = 5

# 681K pseudo-granules from 681K overlapping sources are not independent, so paired p-values
# would be meaningless. For inference, thin to one granule per spot.
VICINITY_THIN_GRID = 25.0
VICINITY_THIN_SEED = 0


# ============================================================ A3c ============================================================ #

# The partition is built at TRANSCRIPT level, not by matrix subtraction. Three disjoint arms:
#   intrasomatic          overlaps_nucleus == 1
#   granule               inside some Set-2 sphere AND overlaps_nucleus == 0
#   residual_extrasomatic neither
# They must sum to the total transcript count per gene per sample, exactly (validation gate).
DE_LAYERS = ["intrasomatic", "granule", "residual_extrasomatic"]

# Why not the published subtraction. 7_neuropil_subdomains.ipynb cell 9 and
# benchmark_ambient.ipynb cell 6 both do
#     np.maximum(spots.layers["extrasomatic_transcripts"] - spot_granule_expression, 0)
# which compounds three errors: (1) spot_embedding assigns each granule to the spot containing its
# CENTRE (downstream.py:706-712) while the sphere spans neighbours; (2) profile() counts ALL
# transcripts in the sphere including overlaps_nucleus == 1, so it over-subtracts from an
# extrasomatic-only layer; (3) overlapping granules double-count shared transcripts. The
# maximum(...,0) clip then makes the bias one-sided. MEASURED (clip_bias_by_gene.csv), the clip is
# small and is NOT marker-biased: non-markers negative in a median 0.046% of spots (max 0.32%)
# against 0.000% (max 0.023%) for the 20 markers -- roughly 24x LESS exactly where the published
# result lives. So the transcript-level rebuild is justified by errors (1)-(3), which are
# structural, and the clip is reported as quantified-and-negligible rather than as a criticism.
# Quantify: per gene, the fraction of spots where the raw difference is negative before clipping.

# Ball radius for the transcript partition = sphere_r + DE_SPHERE_BUFFER.
#
# This was 0.0 ("matching profile()'s default"), which contradicted KG_BUFFER above and was wrong.
# model.profile() does default to 0.0, but code/4_post_detection.ipynb cells 21-22 -- the cells
# KG_BUFFER cites as the published convention -- pass buffer = 0.01. More decisively, sphere_r is
# the MINIMUM-ENCLOSING radius, so a granule's own support points lie exactly on the surface:
# measured on a WT window, a bare-radius partition loses 11.6% of the granule layer and 93.6% of
# what it loses are SYN_GENES (Camk2a -23%, Cplx2 -19%, Map1a -19%). Those transcripts were
# misfiled into residual_extrasomatic, attenuating granule enrichment for exactly the genes the
# analysis is about -- conservative, but wrong.
DE_SPHERE_BUFFER = KG_BUFFER

# Axis 1. granule_enrichment uses the SAME soma reference as the baseline
# (benchmark_diffusion.ipynb's USE_ALT_GRANULE_VS_SOMA branch, which ships switched OFF) --
# otherwise `delta` subtracts two logFCs that share no denominator.
AXIS1_PSEUDOCOUNT = 0.5           # from code/old/benchmark_diffusion.ipynb cell 10
# Frame as DIVERGENCE, not excess: the reviewer's own wording is "exceed OR DIVERGE FROM", and
# divergence is both stronger and safer under compositional normalisation.

# Normalisation. normalize_total(1e4) makes every comparison compositional, and the 20 markers
# are ~31% of transcripts (WT 0.3235 / AD 0.2960), so a real marker enrichment mechanically
# depletes every other gene and the baseline inherits the mirror image. Lead with a count model;
# keep the published Wilcoxon path as a secondary arm so numbers stay comparable to the
# existing CSVs.
DE_COUNT_MODEL = "quasipoisson"   # raw counts with a log(layer total) offset
DE_PUBLISHED_METHOD = "wilcoxon"  # from code/7_neuropil_subdomains.ipynb cell 9

# Axis 2. The Subdomain 1 vs 2 arm ALREADY EXISTS and is not recomputed --
# PUBLISHED_SUBDOMAIN_DIR/{granule,cell,ambient}_DE_genes_Subdomain 1_vs_Subdomain 2.csv
# (rho = 0.37 / 0.42 against the granule layer). What is missing is the WT-vs-AD contrast on the
# same three layers over the same grid, which is the one that maps onto R2's bias concern.

# --- the non-seed, non-control gene test (A3c section 5) -------------------------------------
# The marker-vs-non-marker divergence test is CIRCULAR as an answer to "is the granule compartment
# a passive sample of ambient RNA": mcDETECT draws a sphere around a dense cluster of seed-gene
# transcripts, so the 20 markers are mechanically concentrated inside granules. Section 5 therefore
# re-asks the question on genes that played no part in defining a granule -- neither seeding
# detection (SYN_GENES) nor entering nc_filter (the 19-gene published NC list). 290 - 38 = 252.
#
# The annotation is the panel's OWN curated design sheet (gene_panel.csv, the same file select_set0
# reads), written when the probe set was chosen and years before any granule was called. That is
# what makes it an independent label rather than a post-hoc grouping.
#
# The Neuropil column is tested and reported even though it does NOT separate: its values mix
# subcellular localisation with panel provenance ("Xenium" is one of them), and disclosing the null
# is cheaper than being asked why three annotation columns existed and one was shown.
PANEL_ANNOT_COLS = {"cell_type": "Cell type markers",
                    "synapse": "Synapse markers",
                    "neuropil": "Neuropil"}
NONSEED_NEURONAL = ["Excitatory neurons", "Inhibitory neurons"]
NONSEED_GLIAL = ["Astrocytes", "Oligodendrocytes", "Microglia", "OPC",
                 "Pericytes/Endothelial", "Fibroblast"]
NONSEED_SYNAPSE = ["pre-syn", "post-syn"]
NONSEED_NEUROPIL = ["Dendrites", "Axons", "Neuropil"]
# Primary statistic is the count model's granule-vs-residual logFC: it is measured WITHIN the same
# 50 um spot, so "granules simply sit in neuropil" cannot produce it. The compositional residual is
# carried as a robustness arm because it is built a completely different way.
NONSEED_STATS = ["logFC_granule_vs_residual", "residual_all"]
NONSEED_PRIMARY_STAT = "logFC_granule_vs_residual"
# Each gene contributes ONE value to this test, so the spot-level pseudo-replication that makes the
# count model's own p-values anticonservative does not apply to it. The model supplies effect
# sizes; the inference happens across genes.
NONSEED_TEST_NOTE = ("gene-level Mann-Whitney, one value per gene, so the spot-level "
                     "pseudo-replication caveat on the count model does not apply")


# Two reporting rules, both load-bearing.
REPORTING_RULES = [
    "Significant-gene COUNTS are not comparable across layers: the layers differ in counts per "
    "spot and in sparsity, and a rank test's power tracks that -- which is why ambient/cell show "
    "253/234 significant genes vs granule's 161. Compare RANKINGS and logFC correlations only. "
    "Quoting the tallies invites the reading 'the authors' ambient layer yields more DE genes "
    "than their granule layer'.",
    "n = 1 vs 1. One WT section, one AD section, so every spot-level WT/AD p-value is "
    "pseudo-replication. Frame WT/AD descriptively and put the inferential weight on the "
    "granule-vs-residual-ambient DIVERGENCE, which is a within-sample comparison.",
]


# ============================================================ A3d ============================================================ #
#
# A LOCAL SAMPLING NULL. A3c section 5 answers "is a granule a random draw from the RNA around it?"
# with summary statistics -- E(g), a share-ratio over 50 um squares, and R(g), a whole-section
# regression residual. Neither states the reviewer's hypothesis as a generative model and rejects
# it. A3d does: it preserves the number of transcripts each granule actually holds, redraws them
# from the composition of the RNA beside it, and asks whether the observed neuronal enrichment and
# glial depletion fall outside what those redraws produce.

# Neighbourhood side, in um. Five times tighter than C.SPOT_GRID, so the standing objection
# "granules sit in neuron-rich neighbourhoods" has five times less room to operate. A 10 x 10 um
# column through a 9 um section is close to isotropic, which is why the grid stays 2-D.
LOCAL_NULL_GRID = 10.0

# ONE NULL: THE PERMUTATION. Pool a bin's granule and residual extrasomatic transcripts and
# randomly relabel which N_b of them are "granule". That is exactly multivariate hypergeometric,
# and it is the reviewer's hypothesis stated as a generative model.
#
# A literal multinomial variant -- treat the residual composition p_b as KNOWN and draw
# X_b ~ Multinomial(N_b, p_b) -- was evaluated and retired. Two reasons, in order:
#   1. It assumes away the uncertainty in a composition estimated from a finite local pool, so its
#      p-values are anticonservative; measured on exchangeable data (the split-half gate) its
#      z-scores came out ~1.4x too wide, while the permutation's were unit-spread.
#   2. Its purpose -- to state the hypothesis as something you could actually GENERATE -- is now
#      served far better by A3e, which builds the pseudo-granules physically and runs the detector
#      over them.
# Nothing that was reported ever came from the literal variant, so nothing changes by dropping it.
#
# Bins are independent, so the null has closed-form totals over bins:
#   E_g = sum_b N_b K_bg / M_b
#   V_g = sum_b N_b (K_bg/M_b) (1 - K_bg/M_b) (M_b - N_b) / (M_b - 1)
# with K_bg = O_bg + c_bg and M_b = N_b + n_b. The last factor is the finite-population
# correction; without it the p-values are anticonservative in exactly the dense bins that carry
# most of the weight. Those moments are what the per-gene p-values are computed from, and a
# brute-force gate physically shuffles labels on a random subset of real bins to confirm they are
# the moments of the permutation actually being described.
LOCAL_NULL_MODES = ["permutation"]

# The GROUP statistic T has no closed form, so it is referred to a parametric null built from the
# per-gene moments above: independent normals, this many draws.
LOCAL_NULL_T_DRAWS = 100_000

# The brute-force check on the closed form (gate (c)): shuffle labels physically in this many
# randomly chosen real bins, this many times, and compare the simulated per-gene mean and sd
# against the algebra. Bounded rather than whole-section because it is a correctness check on a
# formula, not a result -- a few thousand bins pin the moments to well under a percent.
LOCAL_NULL_CHECK_BINS = 2000
LOCAL_NULL_CHECK_REPS = 1000

LOCAL_NULL_SEED = 0                         # combined with the sample name via crc32, never hash()

# A bin carries a null only if its local pool is big enough to estimate a composition from. Bins
# below this are dropped and the loss is reported in a3d_local_null_scope.csv rather than buried:
# if the kept bins do not cover most of the granule layer, the result is about a subset and must
# be described as one.
LOCAL_NULL_MIN_POOL = 20

# With millions of transcripts, significance alone proves very little -- a 1% compositional shift
# clears any p-value threshold. Every significance count is therefore reported beside a count of
# genes whose observed/expected ratio exceeds this fold change in either direction.
LOCAL_NULL_EFFECT_THR = 1.25

# The one statistic that answers the reviewer directly:
#   T = median(log2 obs/exp | neuronal) - median(log2 obs/exp | glial + vascular)
# over the same NONSEED_NEURONAL / NONSEED_GLIAL labels section 5 uses, on the same 252 neutral
# genes. Its null distribution comes from LOCAL_NULL_T_DRAWS parametric draws.
LOCAL_NULL_GROUP_STAT = ("median log2(obs/exp) in neuronal genes minus the same in glial and "
                         "vascular genes, both from the panel's own annotation sheet")



# ============================================================ A3e ============================================================ #
#
# THE REVIEWER'S HYPOTHESIS, BUILT AND FED BACK TO THE DETECTOR. A3d answers "could a granule's
# composition be a random draw from the RNA around it?" in z-scores. A3e answers the same question
# with the detector itself: take real granules, keep every transcript exactly where it is, replace
# only the GENE IDENTITIES with labels drawn from the surrounding ambient RNA, and re-run mcDETECT
# over the whole section. If mcDETECT is calling locally dense patches of ambient RNA, it calls
# these back; if it is calling compositionally distinct compartments, it misses them.
#
# WHY THIS IS NOT CIRCULAR. mcDETECT seeds DBSCAN on the 20 markers, so "remove the markers and it
# stops detecting" would prove nothing on its own. Three things make the number informative:
#   1. Local density is held EXACTLY fixed -- every transcript keeps its own (x, y, z) and only
#      `target` changes. The reviewer's "locally elevated ambient RNA" is fully preserved, and
#      composition is the only thing that varies.
#   2. Ambient is not marker-free. The 20 markers are ~32% of all transcripts, so a local ambient
#      draw puts real marker labels back into the pseudo-granule. The re-detection rate is
#      MEASURED and could have come out high.
#   3. The `scramble` arm below separates composition from geometry.
#
# NOT A REPEAT OF A3b. A3b displaces a sphere to a nearby empty location and applies a
# detectability predicate. A3e keeps the location, changes the contents, and runs the real detector
# end to end.

A3E_DIR = OUT_ROOT / "a3e"                 # stage 3 -- construction local, detection on HGCC


def pseudo_detect_dir(sample):
    """Per-sample re-detection output for the patched transcript table."""
    return A3E_DIR / f"detect_{sample}"


def pseudo_relabel_path(sample):
    """The patch: positional row index + new target code, for the changed rows only.

    Deliberately NOT a copy of the transcript table. The patch is ~10% of the granule layer
    (~1.2 M rows in WT against 103 M transcripts), so it is small enough to rsync and to read
    beside the diff, and it is the auditable record of exactly what was changed.

    POSITIONAL, like transcript_layer_<sample>.parquet: `row` indexes the transcript table's own
    row order, which is NOT its __index_level_0__ (that carries gaps from Vizgen filtering).
    a3e_relabel_scope.csv records the table length the patch was built against, and the driver
    asserts it before applying anything.
    """
    return A3E_DIR / f"a3e_relabel_{sample}.parquet"


# --- the three arms ---------------------------------------------------------------------------
# One random sample and ONE detection run per section. What makes a single run sufficient is the
# untouched arm: it is simultaneously the proof that the re-run reproduces the published pipeline
# and the proof that the perturbation stayed local.
#
#   ambient    labels redrawn from the local residual extrasomatic composition -- the hypothesis
#   scramble   the granule's OWN labels permuted among its own points. Composition preserved
#              exactly, geometry scrambled identically to the ambient arm. Without this, a low
#              ambient re-detection rate is ambiguous: relabelling also scrambles WHICH point
#              carries which gene, and that alone could break DBSCAN's eps-connectivity. The
#              ambient arm read against the scramble arm is the purely COMPOSITIONAL effect.
#   untouched  the remainder, changed in no way
PSEUDO_ARMS = ["ambient", "scramble", "untouched"]
PSEUDO_CONVERTED_ARMS = ["ambient", "scramble"]
PSEUDO_FRAC = 0.10                          # of published granules, per converted arm
PSEUDO_SEED = 0                             # crc32 with the sample name, as everywhere in A3

# Relabelling is applied to the transcripts of the GRANULE LAYER inside the sphere, i.e. those
# with overlaps_nucleus == 0. Intrasomatic transcripts inside a granule sphere are left alone:
# they belong to the somatic layer by A3c's partition, in_soma_ratio < 0.1 caps them at a tenth of
# the sphere, and rewriting them would be modifying soma content to answer a question about
# extrasomatic RNA.
PSEUDO_RELABEL_LAYER = "granule"

# --- the local ambient pool -------------------------------------------------------------------
# A DISC CENTRED ON THE GRANULE, not a square of a fixed lattice.
#
# The obvious implementation reuses A3d's 10 um grid and gives each granule the residual RNA of
# whichever square its centre happens to fall in. That was the first version and it is wrong for
# this analysis: the granule is not centred in that square. Its centre sits a median ~2.5 um -- and
# up to 7 um at a corner -- from the middle of the neighbourhood it is being compared against, so
# "the RNA around this granule" is measurably off-centre and asymmetric. A3d can live with that,
# because it needs a PARTITION of the section to sum closed-form moments over. A3e needs no
# partition at all; it needs one neighbourhood per granule, and a disc states that literally.
#
# The disc is 2-D, pooling all seven z-planes, as A3d's squares are. The section is 9 um deep, so
# this is a statement about the plane, which is the interpretable one.
#
# RADIUS. Area pi*5^2 = 78.5 um^2 is 21% TIGHTER than A3d's 100 um^2 square, so the locality claim
# is at least as strong as A3d's, and it is a round number to state. Measured: granules are small
# (median sphere_r 0.93 um WT / 0.95 AD, max 4.0) and residual RNA runs 3.3 / um^2 (WT) and
# 2.3 / um^2 (AD), so the annulus left outside a median granule holds ~250 (WT) / ~174 (AD)
# transcripts against the ~9 a granule contains. The ladder below is scored and reported before
# anything is drawn, so the choice is checked against the data rather than asserted.
PSEUDO_POOL_RADIUS = 5.0
PSEUDO_POOL_RADIUS_LADDER = [4.0, 5.0, 6.0, 7.0]   # diagnostic only; reported, never used to draw

# RETENTION. A granule is redrawn only if its own neighbourhood can actually supply the draw:
#
#     pool_size >= max(PSEUDO_MIN_POOL, k)        k = the granule's own transcript count
#
# The `k` half makes the without-replacement draw well defined -- without it a granule larger than
# its own surroundings would have to be drawn WITH replacement, which is a different sampling
# scheme applied to exactly the largest granules. `draw_local_ambient` asserts the rule rather than
# falling back, so a violation stops the run instead of quietly changing the model.
#
# APPLIED TO EVERY ARM, not just `ambient`. Neighbourhood density plausibly predicts how
# re-detectable a granule is, so restricting only the arm that draws from the pool would confound
# `ambient` against `scramble` with local density. Granules that fail are labelled
# `excluded_thin_pool` and reported beside `excluded_contaminated`; they are never folded into an
# arm. Precedence when both apply: thin pool first.
PSEUDO_MIN_POOL = 50

# The residual layer already excludes EVERY granule's transcripts, which satisfies "excluding the
# granule's own transcripts" more strongly than asked: a granule draws from RNA that no granule was
# built from, its own included.
#
# Labels are drawn WITHOUT REPLACEMENT from the pool's actual label multiset -- the same
# permutation spirit as the locked A3d null, and a physical statement ("these identities came from
# real neighbouring ambient molecules") rather than a parametric one.
#
# THE POOL IS OVER ALL TARGETS -- 290 panel genes plus the Blank probes -- NOT A3d's 252 neutral
# genes. Restricting to neutral genes would remove every marker from the pool and make
# non-detection true by construction.
PSEUDO_EXCLUDED_LABELS = ["excluded_thin_pool", "excluded_contaminated"]

# --- what counts as "re-detected" -------------------------------------------------------------
#
# CREDIT FOLLOWS THE MOLECULES, NOT PROXIMITY. A purely geometric rule cannot distinguish "the
# detector called this object again" from "the detector called something else nearby". That matters
# here more than anywhere else in A3: 80% of granules are untouched and will certainly be called,
# so a pseudo-granule whose contents were entirely replaced can still be credited to a neighbour.
#
# MEASURED, by matching the published granules against themselves -- every granule matches itself,
# so a count > 1 means a DIFFERENT granule also satisfies the rule, i.e. the rate at which a
# destroyed pseudo-granule would be scored re-detected for free:
#
#       criterion      WT       AD
#       center_in     6.93%    8.20%     <- unusable as the primary
#       intersect    33.91%   38.98%     <- barely an identity statement at all
#       merge         0.34%    0.65%
#
# So the primary is PROVENANCE: a granule counts as re-detected when the re-run produced a sphere
# containing at least PSEUDO_PROVENANCE_FRAC of THAT GRANULE'S OWN transcripts -- the granule-layer
# transcripts assigned to it by a3_common.granule_members. Neighbour bleed goes to essentially zero
# because the credit follows the actual molecules, and it states plainly in the response: the
# detector rebuilt a sphere on the same transcripts.
#
# The three geometric rungs are mcDETECT's own predicates and are still scored and reported beside
# it, so the answer can be read under every definition and cannot be an artefact of ours. The floor
# table above is regenerated from data into a3e_match_floor.csv rather than trusted from this
# comment.
PSEUDO_MATCH_GEOMETRIC = ["center_in", "intersect", "merge"]
PSEUDO_MATCH_CRITERIA = ["provenance"] + PSEUDO_MATCH_GEOMETRIC
PSEUDO_MATCH_PRIMARY = "provenance"

# Half of a granule's own transcripts. A recall statement about the object, not a threshold tuned
# to a result: below 0.5 a sphere could be credited with two different granules at once, and at 0.5
# it cannot.
PSEUDO_PROVENANCE_FRAC = 0.5

# THE LOAD-BEARING GATE. If the untouched arm is not re-detected at essentially 100%, no other
# number in A3e means anything -- it would mean the re-run does not reproduce the published
# pipeline, or that the perturbation did not stay local. A3a's Set-2 reproduction rebuilt 681,346
# spheres against 681,337 published (a 9-sphere slack out of 681 K), and miniball is randomised at
# the 1e-13 level, so the threshold is set just below 1 rather than at it.
PSEUDO_CONTROL_MIN = 0.99

# Published Set 2 was built with the 19-gene NC list (Gria2 included), so reproducing it uses that
# list, not the 18-gene SET3_EXCLUDE one. Same policy as everywhere in A3: new detections use 18,
# anything reproducing published data keeps the 19 it was built with.
PSEUDO_NC_LIST = "published"

# ============================================================ shared ============================================================ #

AREA_LIST = ["Isocortex", "OLF", "HPF-CA", "HPF-DG", "HPF-SR", "CTXsp", "TH", "MB", "FT"]
SPOT_GRID = 50                              # from benchmark_subtyping.ipynb cells 13, 22
N_BOOTSTRAP = 500
# AD counts/densities are divided by this. NOTE: it is a GLOBAL scalar and should not be assumed
# spatially uniform under structured ambient -- report per-region WT/AD total-transcript ratios
# and their spread alongside any table that uses it.
CAPTURE_EFFICIENCY_COEF = 0.818691

IN_SOMA_THR = 0.1                           # from code/3_detection.py:84
SIZE_THR = 4.0                              # from code/3_detection.py:83
NC_THR = 0.1                                # from code/3_detection.py:84
EPS = 1.5                                   # from code/3_detection.py:82
LOW_BOUND = 3                               # from code/3_detection.py:83
CUTOFF_PROB = 0.95                          # from code/3_detection.py:83
Z_GRID = [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0]

# Pre-binned histogram edges for the R hand-off (A1 postproc idiom: export quantiles + bins,
# never millions of raw values).
HIST_BINS = {
    "sphere_r": (0.0, 4.0, 50),
    "size": (0.0, 60.0, 60),
    "n_total": (0.0, 100.0, 50),
    "n_marker": (0.0, 60.0, 60),
    "in_soma_ratio": (0.0, 1.0, 50),
    "nc_ratio": (0.0, 1.0, 50),
    "lambda_local": (0.0, 1.0, 50),
}

# WT/AD colors, kept in step with A3_figures.R (R cannot import this module).
WT_COLOR = "#a0ccec"
AD_COLOR = "#f48488"



# ------------------------------------------------------------------------------------------- #
# KNOWN ISSUE, flagged only -- NOT investigated by A3 (see README.md).
#
# The AD section thins with depth while WT is flat, so granule counts follow:
#
#   z (um)        0     1.5     3.0     4.5     6.0     7.5     9.0
#   WT tx (M)   13.6    15.0    15.4    15.4    15.2    14.7    14.0
#   AD tx (M)   16.0    15.5    14.5    11.6     6.9     3.1     1.4
#   WT granules 95,352 102,896 102,409 100,692 97,970  93,667  88,351
#   AD granules 122,594 114,874 98,265  48,256  12,128  2,068     624
#
# Restricted to the three fully covered planes (z <= 3), WT = 300,657 and AD = 335,733 -- AD is
# HIGHER, reversing the direction of the total-granule difference. CAPTURE_EFFICIENCY_COEF does
# not correct for this: the raw transcript ratio is 0.666 and the deficit is z-structured, not
# uniform. Whether the published PER-SUBTYPE, PER-REGION claim reverses is UNCHECKED.
# ------------------------------------------------------------------------------------------- #


def resolve_n_jobs(n_jobs=None):
    """Threads to use: explicit -> $SLURM_CPUS_PER_TASK -> os.cpu_count()."""
    if n_jobs is None:
        n_jobs = os.environ.get("SLURM_CPUS_PER_TASK")
    if n_jobs is None:
        n_jobs = os.cpu_count() or 1
    return max(1, int(n_jobs))


def detect_dir(set_name, sample):
    """Per-(set, sample) detection output: merged table + per-gene sphere_dict."""
    return DETECT_DIR / f"{set_name}_{sample}"


def spheres_path(set_name, sample):
    """The merged granule table for one set. Set 2 is the published file, never rewritten."""
    if set_name == "set2":
        return mcdetect_granules_path(sample)
    return detect_dir(set_name, sample) / "spheres.parquet"


def sphere_dict_path(set_name, sample):
    """Per-gene pre-merge spheres, with the seed gene and its own-gene member count k_g.

    Required because merge_sphere() is many-to-one and _remove_overlaps updates only
    sphere_x/y/z, layer_z and sphere_r -- `size`, `comp`, `gene` and `in_soma_ratio` are all
    stale afterwards. So granules.parquet["gene"] is not reliably the seed gene (A3b needs it)
    and `size` is not k_g (A3a stage D needs it).
    """
    return detect_dir(set_name, sample) / "sphere_dict.parquet"


def transcript_layer_path(sample):
    """Cached per-transcript layer label (intrasomatic / granule / residual_extrasomatic).

    One int8 column per sample, ~100 MB. Computed once by A3c section 1 and reused by A3a
    section 6 -- assigning 10^8 transcripts against 681 K spheres is the single most expensive
    operation in A3, and it was previously recomputed four times across two notebooks.
    """
    return A3C_DIR / f"transcript_layer_{sample}.parquet"


def ensure_dirs():
    for d in [PREFLIGHT_DIR, DETECT_DIR, A3A_DIR, A3B_DIR, A3C_DIR, A3D_DIR, A3E_DIR,
              FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
