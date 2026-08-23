"""
Shared configuration for the A1 POST-HOC analysis of the Baysor / SSAM sweep.

The detection stage (../build_manifest.py, ../run_{baysor,ssam}_tile.py, ../concat_spheres.py)
ran on HGCC and produced raw, unfiltered spheres. THIS stage runs LOCALLY and interactively:
it recomputes each detection's granule-likeness features, splits the detections into populations, and
runs the mixing and subdomain-contrast analyses on each -- plus, for SSAM alone, mcDETECT's granule
subtyping chain (cluster -> manual mapping -> ordered heatmap -> subtype density -> the WT
genetic-labeling correlation).

Scope: `param = tuned` only, with **two parallel gene-set settings** analyzed separately:

    geneset "all"      full 290-gene panel  -> populations pop1, pop2, pop3
    geneset "markers"  the 20 mcDETECT syn_genes -> populations pop1, pop2

        pop1  all detections
        pop2  out-of-nucleus            (in_soma_ratio < IN_SOMA_THR, mcDETECT's 0.1 cutoff)
        pop3  pop2 AND granule-marker-enriched (marker_frac >= MARKER_FRAC_THR)  [`all` only]

`param = default` is used ONLY as a scale check (number of detections vs mcDETECT).
Marker enrichment is meaningless in the `markers` setting (every transcript that formed the
detection is a marker), so pop3 exists only for `all`.

NO size or NC filtering is applied; sphere_r / nc_ratio are computed for reporting only.

This is load-bearing, not incidental. `pop2` IS DEFINED BY in_soma_ratio < 0.1 and `pop3` adds
marker_frac >= 0.4, so the in-soma distributions of pop2/pop3 are truncated by construction and
cannot evidence "filtering recovers granules". `nc_ratio` is filtered on NOWHERE -- not in the
detection sweep, not in pop2, not in pop3 -- so however it moves across the populations is a real,
independent result, in whichever direction it turns out to run. One caveat: nc_ratio_MARKER
(= n_nc / n_marker) shares a denominator with the marker_frac cut that defines pop3, so it falls
mechanically there; nc_ratio_ALL is the one to read for pop3.

Two conventions worth remembering:

* **In-soma denominator.** mcDETECT's own `in_soma_ratio` counts ONLY granule-marker transcripts
  (model.py:102 subsets to gnl_genes; model.py:258-267 forms total_size from the seed gene plus the
  other markers). Under the "all" geneset many tuned Baysor spheres contain zero marker transcripts,
  so a marker-only denominator would be undefined for them and would collapse pop2 into pop3. Both
  settings therefore use ALL transcripts as the primary denominator and report the mcDETECT-identical
  marker-only ratio as a sensitivity column.

* **Expression quantification.** Every detection is quantified by the transcripts inside its
  ENCLOSING SPHERE, for both methods. mcDETECT's granule expression is sphere capture and SSAM emits
  no molecule assignment at all, so sphere-for-everyone is the only quantification that isolates the
  detection difference rather than confounding it with the counting rule. For Baysor this captures a
  median ~1.4x more transcripts than Baysor itself assigns to the cell (q95 ~2.7x) -- disclose in
  Methods. (Baysor's segmentation.csv is not retained by the detection stage.)
"""

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent      # .../baysor_ssam_merscope/postproc
DETECT_DIR = SCRIPT_DIR.parent                    # .../baysor_ssam_merscope
REPO_ROOT = DETECT_DIR.parents[1]                 # .../mcDETECT

# Reuse the detection sweep's config verbatim (paths, sweep axes, SYN_GENES, locked params) so
# the two stages can never drift apart.
if str(DETECT_DIR) not in sys.path:
    sys.path.insert(0, str(DETECT_DIR))
import config as DET  # noqa: E402

# ----------------------------------------------------------------------------- #
# Paths
# ----------------------------------------------------------------------------- #
DATA_ROOT = DET.DATA_ROOT                          # .../mcDETECT/data
OUT_ROOT = DET.OUT_ROOT                            # .../baysor_ssam_merscope/output
POSTPROC_DIR = OUT_ROOT / "postproc"               # everything this stage writes
ARMS_DIR = POSTPROC_DIR / "wt_ad"                  # one directory per subdomain-DE arm (part B)
MCDETECT_OUT = REPO_ROOT / "output"                # mcDETECT's own granule outputs

# mcDETECT reference material
WT_AD_DIR = MCDETECT_OUT / "MERSCOPE_WT_AD_comparison"
MCDETECT_GRANULE_ADATA = WT_AD_DIR / "granule_adata_tsne.h5ad"          # 1,080,146 x 290, obs["batch"]
SUBDOMAIN_DIR_50 = WT_AD_DIR / "neuropil_subdomains_Isocortex_50"

# The published microdomain partition, reused verbatim as the anchor for the subdomain-level DE:
# spot_id -> subdomain_kmeans for the 4,310 Isocortex 50 um spots (Subdomain 1 = 1,117 spots,
# Subdomain 2 = 1,096). spot_id joins 1:1 onto each sample's own spots.h5ad grid.
SUBDOMAIN_LABELS = SUBDOMAIN_DIR_50 / "4_hard_normalized_cluster_labels.parquet"

# mcDETECT's own DE over that same pair, on three expression layers. These are the anchors the
# Baysor/SSAM arms are scored against: `granule` is the target, while `cell` and `ambient` show what
# the same subdomains yield when NON-granule transcripts are aggregated over them (their logFC
# correlates only r = 0.37 / 0.42 with the granule layer, yet they carry MORE significant genes --
# 234 and 253 of 290 vs the granule layer's 161). Reproducing the granule ranking is the hard part.
SUBDOMAIN_ANCHOR_LAYERS = ["granule", "cell", "ambient"]
SUBDOMAIN_PAIR = ("Subdomain 1", "Subdomain 2")

# GO terms that land on the pre-synaptic side of the subdomain contrast but whose leading edge is
# Vamp2 / Cplx2 / Stxbp1 -- three SNARE proteins annotated to immune GO terms via *degranulation*.
# On a 290-gene panel these sets reduce to those few genes, so they are a relabelling of the
# synaptic-exocytosis programme rather than immune biology. A1_figures.R section 5 reports the
# pre-synaptic tally BOTH ways -- including and excluding them -- so this judgement is visible in
# the output rather than hidden in it. NOTE: no Python code reads this constant -- R cannot import
# this module, so A1_figures.R keeps its own copy as ARTIFACT_TERMS and that is the one that runs.
# This is the canonical record of the judgement; change both together.
GSEA_ANNOTATION_ARTIFACTS = ["GOBP_LEUKOCYTE_MEDIATED_IMMUNITY", "GOBP_IMMUNE_EFFECTOR_PROCESS"]

# The 50 um Isocortex spot object those labels were built on. A1_filter_de.ipynb section 5 reads it
# for one reason only: spot_id is per-sample, so this is what resolves each labelled spot back to
# WT or AD.
SUBDOMAIN_SPOTS_50 = WT_AD_DIR / "neuropil_subdomains_spots_ambient.h5ad"   # 4,310 spots, 50 um, Isocortex

# ----------------------------------------------------------------------------- #
# Sweep slice analyzed here
# ----------------------------------------------------------------------------- #
METHODS = ["baysor", "ssam"]
SAMPLES = DET.SAMPLES                              # ["WT", "AD"]
PARAM = "tuned"
GENESETS = ["all", "markers"]                      # the two parallel settings

SYN_GENES = list(DET.SYN_GENES)                    # the 20 markers mcDETECT seeds DETECTION on

# The 34 compartment markers mcDETECT CLUSTERS on when subtyping granules -- a superset of
# SYN_GENES that adds post-synaptic / dendritic / axonal genes which seed nothing but do
# discriminate compartments. Copied verbatim from code/benchmark/benchmark_subtyping.ipynb
# (cell 4, `ref_genes`), the source of the published granule_subtype_kmeans labels; identical
# in code/old/8_neuropil_subdomains_pair2.py:71.
REF_GENES = [
    "Bsn", "Gap43", "Nrxn1", "Slc17a6", "Slc17a7", "Slc32a1", "Stx1a", "Syn1", "Syp", "Syt1",
    "Vamp2", "Cplx2", "Camk2a", "Dlg3", "Dlg4", "Gphn", "Gria1", "Gria2", "Homer1", "Homer2",
    "Nlgn1", "Nlgn2", "Nlgn3", "Shank1", "Shank3", "Cyfip2", "Ddn", "Map1a", "Map2", "Ank3",
    "Nav1", "Nfasc", "Mapt", "Tubb3",
]


def populations(geneset):
    """Populations defined for a gene-set setting (pop3 is meaningless for `markers`)."""
    return ["pop1", "pop2", "pop3"] if geneset == "all" else ["pop1", "pop2"]


# ----------------------------------------------------------------------------- #
# Thresholds (mcDETECT's own values where one exists)
# ----------------------------------------------------------------------------- #
IN_SOMA_THR = 0.1        # model.py default in_soma_thr; pop2 keeps in_soma_ratio < this

# pop3: >= this fraction of the sphere's transcripts are granule markers.
# Deliberately ABSOLUTE and uniform across samples -- a per-sample or background-derived cutoff
# would filter WT and AD asymmetrically (AD's background is lower, so it would get a looser bar),
# and both samples are pooled in the subdomain contrast.
# Reference points: the panel-wide marker share, i.e. the fraction expected by chance, is
# WT 0.3235 / AD 0.2960 / pooled 0.3125 (see SF.sample_marker_share). 0.4 is therefore ~1.28x
# background -- still "enriched" rather than "typical". Anything at or below ~0.31 is at or below
# chance and would not mean enrichment at all, which puts a floor of roughly 0.35 on this value.
# Surviving pop2 detections at 0.4: Baysor 804,184 (WT) / 416,989 (AD); SSAM 21,713 / 18,594.
# Retune with the scan table printed in A1_filter_de.ipynb section 3.
MARKER_FRAC_THR = 0.4
NC_TOP = 20              # model.py::nc_filter keeps the top-20 NC genes by count (only 19 exist)

# ----------------------------------------------------------------------------- #
# Spatial scaffold
# ----------------------------------------------------------------------------- #
SPOT_GRID = 50           # um; mcDETECT's own spot lattice, which section 5 aggregates onto

# ----------------------------------------------------------------------------- #
# Granule subtyping -- every value copied from mcDETECT's published `main_result` arm
# (code/benchmark/benchmark_subtyping.ipynb, `run_manual_subtyping()` in cell 15, driven by cell 22).
# mcDETECT clusters the REF_GENES matrix directly with MiniBatchKMeans: no PCA, no plain KMeans.
#
# Two consumers, with different needs:
#   A1_filter_de.ipynb section 4  renders a heatmap per arm and exports NO labels -- the panel is
#                                 the result, for all ten arms.
#   A1_ssam_subtypes.ipynb        re-clusters SSAM's five arms from scratch and DOES export labels,
#                                 which feed the manual mapping and the subtype density. It sets
#                                 its own per-population seed; SUBTYPE_SEED is only its default.
# ----------------------------------------------------------------------------- #
K_SUBTYPE = 15
KMEANS_BATCH_SIZE = 5000
KMEANS_N_INIT = 20
SUBTYPE_SEED = 1

# sc.pl.heatmap draws one column per detection and baysor/all/pop1 holds ~6 M of them, so each arm
# is reduced to a seeded random subsample for both the clustering fit and the plot. Disclosed in the
# figure caption; arms smaller than this use every detection.
HEATMAP_MAX_CELLS = 200_000
HEATMAP_SEED = 0

# `granule_subtype_manual_simple` categories, in mcDETECT's order (benchmark_subtyping.ipynb cell 22).
# Mapping keys containing " & " collapse to "mixed"; clusters left unlisted stay NaN.
SUBTYPE_NAMES = ["pre-syn", "post-syn", "dendrites", "axons", "mixed", "others"]

# ----------------------------------------------------------------------------- #
# Subtype density per brain region -- matched to mcDETECT's published analysis
# (code/benchmark/benchmark_subtyping.ipynb cells 19 and 22)
# ----------------------------------------------------------------------------- #
# Capture-efficiency correction between the two sections, applied to the AD side only: the AD
# per-spot counts are DIVIDED by this before anything else, so density, sd, sem, bootstrap CI and
# the t-test all inherit it. WT is never touched.
#
# SET TO 1.0 -- NO CORRECTION IS APPLIED HERE. mcDETECT's own analysis divides AD by 0.818691
# (benchmark_subtyping.ipynb cell 22, hard-coded there with no derivation recorded anywhere in the
# repo). It is deliberately not carried over, because at 1 / 0.818691 = 1.22x it is large enough to
# manufacture the effect it would then be read as evidence for: it is essentially the whole of
# SSAM's apparent AD density increase, which is ~1.05 -- i.e. flat -- uncorrected. See
# postproc/old/README.md, the retired overall-density section, for those numbers.
#
# TO DISCLOSE: the 16-column schema and the WT densities are directly comparable to mcDETECT's
# published subtype_density_per_region_granule_adata_tsne.csv; the AD densities are NOT on the same
# scale as its AD densities. To restore mcDETECT's convention, set this to 0.818691 and re-run
# A1_ssam_subtypes.ipynb section 5 (nothing there is cached), then A1_figures.R section 6.
CAPTURE_EFFICIENCY_COEF = 1.0
N_BOOTSTRAP = 500        # resamples for the per-spot bootstrap CI (percentiles 2.5 / 97.5)


# ----------------------------------------------------------------------------- #
# Synaptic density vs the genetic-labeling reference -- WT only
# (code/benchmark/benchmark_subtyping.ipynb cell 13 for the constants, cell 22 for the test)
# ----------------------------------------------------------------------------- #
# mcDETECT's one external check on the subtyping: do the SYNAPTIC detections distribute across brain
# regions the way synapses actually do? Nothing in detection or clustering could have been fitted to
# this, which is what makes it worth reproducing for SSAM.
#
# The nine regions, in mcDETECT's order. Retired from this config once (see postproc/old/README.md)
# and reinstated here; A1_figures.R:1433 keeps its own copy for plotting and must stay in step.
AREA_LIST = ["Isocortex", "OLF", "HPF-CA", "HPF-DG", "HPF-SR", "CTXsp", "TH", "MB", "FT"]

# Synapse density per region measured by volume electron microscopy + genetic labeling in an
# age-matched WT mouse (Santuy et al. 2020, Sci Rep; the manuscript's Fig. S9 reference). Values
# verbatim from benchmark_subtyping.ipynb cell 13, where they are a list positionally aligned to
# AREA_LIST with 99 standing in for HPF-SR. A dict is the same numbers minus the placeholder, and
# GT_DROP_AREAS states both exclusions instead of leaving them as np.delete([4, 8]).
GT_SYNAPSE_DENSITY = {
    "Isocortex": 1.71603565, "OLF": 1.964351308, "HPF-CA": 2.052720791, "HPF-DG": 1.139278326,
    "CTXsp": 1.678527951, "TH": 1.082904337, "MB": 0.444031185, "FT": 0.0199885,
}

# HPF-SR has no reference value at all; FT is mcDETECT's own exclusion (its value, 0.0199, exists).
# Kept identical so the coefficients are comparable -- 7 regions enter the correlation.
GT_DROP_AREAS = ("HPF-SR", "FT")
GT_AREAS_USED = [a for a in AREA_LIST if a in GT_SYNAPSE_DENSITY and a not in GT_DROP_AREAS]

# The reference is a WT measurement, so only WT detections are compared and CAPTURE_EFFICIENCY_COEF
# never applies. Synaptic membership is decided on the FINE label (`granule_subtype_manual`), not
# the collapsed one, so `pre & den` / `post & den` / `pre & post & den` are excluded.
GT_SAMPLE = "WT"
SYNAPTIC_SUBTYPES = ["pre-syn", "post-syn", "pre & post"]

# mcDETECT's own published coefficients for this test on its granules, transcribed from
# benchmark_subtyping.ipynb cell 24 (setting `main_results`, k = 15). Cited, not recomputed: cell 22
# prints them but never persists the per-region density vector behind them.
MCDETECT_GT_PEARSON = 0.9098
MCDETECT_GT_SPEARMAN = 0.8784
MCDETECT_GT_SOURCE = "published: benchmark_subtyping.ipynb cell 24 (main_results, k = 15)"


# ----------------------------------------------------------------------------- #
# Runtime knobs (the profiling step is the expensive one)
# ----------------------------------------------------------------------------- #
CHUNK_SIZE = 250_000     # spheres per KD-tree query batch
KDTREE_WORKERS = -1      # -1 = all cores

# ----------------------------------------------------------------------------- #
# WT / AD coordinate registration
#
# Lifted verbatim from code/3_detection.py:20-31 (per-sample rotation / flip) and
# code/4_post_detection.ipynb cell 19 + code/5_neuropil_subdomains_data.py:200-206 (the combined
# WT/AD registration), so Baysor/SSAM detections land in the same frame as mcDETECT's granules.
# ----------------------------------------------------------------------------- #
REGISTRATION = {
    "WT": {"theta": 10 * np.pi / 180, "flip": True, "cutoff": 6250,
           "rotate_cols": ["sphere_y", "sphere_x"], "flip_col": "global_y"},
    "AD": {"theta": 170 * np.pi / 180, "flip": False, "cutoff": -4200,
           "rotate_cols": ["sphere_x", "sphere_y"], "flip_col": "global_x"},
}
COMBINED_SHIFT_X = 12000     # AD offset when WT and AD are placed side by side
COMBINED_SHIFT_Y = 7200
COMBINED_CUTOFF = 6250       # WT y-flip applied when building the combined frame


# ----------------------------------------------------------------------------- #
# Path helpers -- `geneset` is always explicit so a config can never be mislabelled
# ----------------------------------------------------------------------------- #
def spheres_path(method, sample, geneset, param=PARAM):
    """Raw detections from the HGCC sweep."""
    return DET.spheres_path(method, sample, param, geneset)


def config_tag(method, sample, geneset, param=PARAM):
    return f"{method}_{sample}_{param}_{geneset}"


def features_path(method, sample, geneset, param=PARAM):
    """Per-detection features. After A1_filter_de.ipynb section 3 this also carries the population
    flags -- it is the single source of truth for detection-level annotation."""
    return POSTPROC_DIR / f"{config_tag(method, sample, geneset, param)}_features.parquet"


def profile_path(method, sample, geneset, param=PARAM):
    return POSTPROC_DIR / f"{config_tag(method, sample, geneset, param)}_profile.h5ad"


def arm_dir(method, geneset, population):
    """Run directory for one arm: wt_ad/<method>_<geneset>_<population>/.

    Written by A1_filter_de.ipynb section 5 (subdomain-anchored DE) and discovered one level below
    ARMS_DIR by A1_figures.R section 5. mcDETECT's own anchor tables sit alongside in
    ARMS_DIR / "mcdetect_reference"."""
    return ARMS_DIR / f"{method}_{geneset}_{population}"


def transcripts_path(sample):
    return DET.transcripts_path(sample)


def processed_dir(sample):
    return DATA_ROOT / DET.sample_dataset(sample) / "processed_data"


def genes_path(sample):
    return processed_dir(sample) / "genes.csv"


def nc_path(sample):
    return processed_dir(sample) / "negative_controls.csv"


def spots_path(sample):
    return processed_dir(sample) / "spots.h5ad"


def mcdetect_granules_path(sample):
    """mcDETECT's final (filtered) granules for this sample."""
    return MCDETECT_OUT / DET.sample_dataset(sample) / "granules.parquet"


def dataset_name(sample):
    """'WT' -> 'MERSCOPE_WT_1' (the label mcDETECT uses in obs['batch'])."""
    return DET.sample_dataset(sample)


def load_genes(sample):
    import pandas as pd
    return list(pd.read_csv(genes_path(sample)).iloc[:, 0])


def load_nc_genes(sample):
    import pandas as pd
    return list(pd.read_csv(nc_path(sample))["Gene"])
