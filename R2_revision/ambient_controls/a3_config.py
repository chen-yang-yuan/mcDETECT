"""
Configuration for analysis A3 -- ambient RNA controls at the DETECTION step
(`plans/Round2_response_analysis_plan.md` section A3, Reviewer #2 major point 9).

Three sub-analyses share this module:

  A3a  three-set NC split + locally-adaptive threshold re-test  -- local, `A3a_three_sets.ipynb`
  A3b  vicinity pseudo-granule control                          -- local, `A3b_vicinity.ipynb`
  A3c  somatic-vs-non-somatic DE baseline                       -- local, `A3c_de_baseline.ipynb`

plus one HGCC stage, `run_detection_sets.py`, which produces the Set 1 / Set 3 / Set 0 detections
that A3a and A3b consume.

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
# Gria2 is in BOTH SYN_GENES and the 19-gene negative-control list. nc_filter counts it in the
# numerator and `size` counts it in the denominator, so Gria2-seeded granules self-filter:
#
#   Gria2 spheres      rough    + size/in-soma    published (Set 2)
#   WT                 41,544         2,737              4
#   AD                 20,525         1,139              0
#
# Quote the MIDDLE column: most Gria2 aggregates are intranuclear (consistent with Gria2 sitting
# on a nuclear-enrichment list), so the loss attributable to the NC FILTER is ~2.7K / ~1.1K, not
# 41K / 20K. For context, 97.1% (WT) / 97.6% (AD) of published granules have nc_ratio EXACTLY 0,
# so the Set1-vs-Set2 difference is dominated by Gria2 and must be partitioned, never reported as
# one number.
#
# POLICY. TWO SEPARATE GENE LISTS ARE INVOLVED, AND ONLY ONE OF THEM IS EVER MODIFIED:
#
#   SYN_GENES  -- 20 granule MARKERS, the DETECTION SEEDS (code/3_detection.py:55)
#   nc_genes   -- 19 NEGATIVE CONTROLS, the NC FILTER list (negative_controls.csv)
#
# Gria2 is on both. Everything below concerns the NC LIST ONLY. The 20-marker seed list is used
# unchanged everywhere in A3 -- Sets 1 and 2 both seed on all 20 (see SYN_GENES_UNCHANGED).
#
# The NC list has two versions, chosen by provenance:
#
#   Set 3's SEED list                    -> 18 (Gria2 dropped)
#   the leave-one-out enumerated on Set 1 -> 18 (Gria2 dropped)
#   nc_ratio recomputed on Set 2          -> 19 (as published)
#
# Excluding Gria2 from a NEW use is a correction, not an approximation: Gria2 is a granule marker
# mis-listed as a nuclear-enrichment control, so dropping it makes Set 3 MORE faithful to
# "nuclear-enriched negative controls". Keeping it would have been large relative to SET 3 --
# Gria2 is ~13x more abundant than the median NC gene, so it could roughly double Set 3 and
# manufacture a spurious Set1-intersect-Set3 overlap. Set 2 keeps 19 because that is the filter
# the data on disk was actually built with.
#
# WHY SET 1 STILL SEEDS ON ALL 20 MARKERS. Set 1's whole job is to be "Set 2 minus the NC filter".
# Seeding it on 19 would make it differ from Set 2 in TWO ways at once, and the difference would
# no longer isolate the filter. The confound is real, not hypothetical: _remove_overlaps
# (model.py:323-377) is order-dependent and propagates whole rows --
#     containment with B larger -> `set_a.loc[i] = set_b.loc[j]`, A's row replaced WHOLESALE
#                                  (gene label included)
#     deep intersection         -> A's geometry refit over the union, A's gene label SURVIVES
# so a Gria2 sphere can absorb, or be absorbed by, another marker's sphere. Dropping Gria2 from
# the seeds would therefore change the geometry AND the labels of NON-Gria2 granules too.
#
# A consequence, noted rather than fixed: some SURVIVING Set-2 granules have geometry enlarged by
# a merge with a Gria2 sphere that was only NC-filtered away afterwards. That is baked into the
# published result.
#
# Dropping Gria2 from the MARKER list is also unnecessary and not free: it is a genuine
# post-synaptic marker (it is in REF_GENES and MARKER_GENES["post-syn"]), and the published
# population already contains effectively none of it (4 WT / 0 AD) -- so the effective marker set
# is already 19, and saying so costs nothing. Re-running the published detection on 19 markers
# would change Fig. 3-5 for a gene contributing 4 granules: out of scope.
#
# THE GAP. Keeping Set 2 on 19 while Set 3 seeds on 18 leaves the published population short of
# the Gria2-seeded granules an 18-gene NC filter would have kept: at most ~2,737 of ~737,000 in
# WT (0.37%) and ~1,139 of ~427,000 in AD (0.27%). These are UPPER BOUNDS taken from Set 1, which
# applies no NC filter at all; most of those granules would clear an 18-gene filter anyway, since
# the remaining controls are rare. Under half a percent, same direction in both samples, so it
# cannot move the WT/AD contrast. Reported once in gria2_partition.csv, alongside a free
# post-hoc sensitivity that strips Gria2-SEEDED rows from BOTH Set 1 and Set 2 so the two rest on
# the same effective 19-marker population. No arm is built on any of it.
#
# Because of this policy there is no Set 3prime (all-19) detection -- it existed only to show the
# choice does not matter, and Set 3prime - Set 3 is exactly the Gria2-seeded spheres, which Set 1
# already contains.
GRIA2 = "Gria2"
SET3_EXCLUDE = [GRIA2]

# The 20-marker SEED list is never modified by A3. Asserted in the A3a correctness gates.
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
# (alpha = 0.5 does appear in ADAPTIVE_ALPHA_SWEEP below -- that is the stage-D local
#  threshold sweep, a post-hoc re-test of already-called granules, not a detection.) poisson_select
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
    "set0": "Set 0 (abundance-matched neutral genes)",
    "set1": "Set 1 (markers, no NC filter)",
    "set2": "Set 2 (published)",
    "set3": "Set 3 (NC genes minus Gria2)",
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
# The three boolean rungs, in report order. `jaccard` is NOT here: it is a continuous overlap
# fraction reported as a distribution (a3_common.jaccard_balls), not a predicate, and
# sphere_overlap() raises on it.
OVERLAP_CRITERIA = ["intersect", "center_in", "merge"]
OVERLAP_PRIMARY = "intersect"          # d < r_A + r_B
MERGE_RHO = 0.2                        # from code/3_detection.py:84
MERGE_L = 1.0                          # from code/3_detection.py:84

# _remove_overlaps and dbscan use sphere_z; nc_filter and profile use layer_z. Pick one, apply it
# identically to every set, and disclose the inherited inconsistency.
OVERLAP_Z_COL = "sphere_z"

# Raw |Set1 & Set3| is uninterpretable without an expectation, so it is reported as
# observed/expected against Set-3 spheres randomly re-placed in the tissue mask at matched radius
# and matched layer_z.
OVERLAP_N_NULL = 20
OVERLAP_NULL_SEED = 0

# --- stage D: locally-adaptive threshold re-test ----------------------------------------------
# poisson_select is a 2D AREAL intensity (tissue_area() is 2D grid occupancy * grid_len^2) against
# a 2D DISC pi*eps^2, even though DBSCAN runs in 3D. The local version must match that functional
# form or the comparison is meaningless -- so: 2D, per gene, same alpha/cutoff_prob/low_bound.
#
#   lambda_local(g,i) = [ N_g(disc R at (x,y)) - k_g(i) ] / [ pi R^2 * occ(x,y,R) ]
#   m_local(g,i)      = max( poisson.ppf(0.95, alpha * lambda_local * pi * eps^2), low_bound )
#   survives          <=> k_g(i) >= m_local(g,i)
#
# k_g(i) is the count of gene g that formed THIS cluster. Subtracting it is essential, otherwise
# the granule inflates its own background and the test is self-defeating. It is NOT in
# granules.parquet (`size` pools all markers and is stale after merging) -- it comes from the
# per-gene sphere_dict persisted by run_detection_sets.py.
ADAPTIVE_R = [25.0, 50.0]          # primary radii, um
ADAPTIVE_R_SENSITIVITY = [10.0]    # below ~10um the disc holds ~6 transcripts -- Poisson noise
ADAPTIVE_ALPHA_SWEEP = [0.5, 1.0, 5.0, 10.0]   # report a survival CURVE, not one number
# Lattice on which the per-gene counts are binned. It must DIVIDE the smallest R in the sweep,
# or radii collapse onto the same window: at 25 um, R = 10 and R = 25 both give k = 1 and the
# "sensitivity" radius is not a sensitivity at all. At 10 um, R = 10/25/50 -> k = 1/3/5.
# It must also be an exact multiple of the 1 um occupancy grid (asserted in _occupancy_fraction).
ADAPTIVE_GRID = 10.0
# lambda is a disc of radius R, not one lattice cell: the counts are box-summed over the
# ceil(R / ADAPTIVE_GRID) neighbourhood and divided by the OCCUPIED area of that same disc.
# Without this, R never enters the arithmetic and every radius returns the identical curve.
ADAPTIVE_EXCLUDE_GRANULE_TX = [False, True]    # bracket the neighbour-contamination range

ADAPTIVE_CAVEATS = [
    "Post-hoc re-test, not a re-detection: a truly adaptive min_samples changes which points are "
    "core, hence cluster membership, the enclosing sphere, and k_g itself. Fixed-cluster "
    "re-testing can only REMOVE granules, never add or reshape them.",
    "It therefore bounds false-positive inflation but is SILENT on false negatives in "
    "low-density regions, where an adaptive rule would be more permissive. AD is the "
    "lower-density arm, so this cuts against our own effect direction.",
    "lambda_local is contaminated by neighbouring granules; excluding the granule's own "
    "transcripts fixes self-contamination only. Report it both with and without all Set-2 "
    "granule transcripts removed -- the truth is bracketed by the two.",
    "This tests spatial homogeneity, not Poisson-ness. If ambient is overdispersed even locally, "
    "a Poisson cutoff under-corrects at every scale; fit a quasi-Poisson dispersion phi per gene "
    "across 25um spots and add an NB arm if phi >> 1.",
]


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

# Match on the covariate we are accused of ignoring.
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
# maximum(...,0) clip then makes the bias one-sided and worst for the MARKER genes -- exactly
# where the result lives. Quantify: per gene, the fraction of spots where the raw difference is
# negative before clipping.

DE_SPHERE_BUFFER = 0.0            # ball radius = sphere_r + buffer, matching profile()'s default

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
    for d in [PREFLIGHT_DIR, DETECT_DIR, A3A_DIR, A3B_DIR, A3C_DIR, FIG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
