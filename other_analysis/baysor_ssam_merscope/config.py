"""
Shared configuration for the Baysor / SSAM real-data benchmark on MERSCOPE WT & AD.

This is the A1 analysis of plans/Round2_response_analysis_plan.md: run the two
assumption-light methods on the real MERSCOPE_WT_1 / MERSCOPE_AD_1 samples and
compare their native detections against mcDETECT granules.

Design (see the plan for rationale):
  - Sweep sample     : WT, AD
  - Sweep gene input : all panel genes  vs  the 20 mcDETECT granule markers
  - Sweep parameters : default vs tuned (values LOCKED from the simulation,
                       code/figures_response.Rmd / simulation/analyze_crosstab.py)
  - Filtering (size / in-soma / NC) is applied POST-HOC in a later step, NOT here,
    because Baysor/SSAM have no cross-marker merging step. Detection scripts here
    emit the raw native spheres only.

Parallelism: every (sample x param x geneset x tile) unit is one independent job,
dispatched as a SLURM array task (see slurm/). Within a detection the tissue is
cut into TILE_SIZE x TILE_SIZE um squares; tiles are detected independently and
concatenated afterwards (edge cases across tile borders are ignored, per design).

Nothing here is run locally; scripts are transferred to HGCC. Paths resolve from
the repository root (two levels up from this file) and can be overridden with the
MCDETECT_DATA_ROOT / MCDETECT_OUT_ROOT environment variables.
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------------- #
# Paths
# ----------------------------------------------------------------------------- #
# other_analysis/baysor_ssam_merscope/config.py  ->  parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(os.environ.get("MCDETECT_DATA_ROOT", REPO_ROOT / "data"))
# Outputs land in the repo-level (git-ignored) output area, beside the existing
# WT/AD comparison figures.
OUT_ROOT = Path(
    os.environ.get(
        "MCDETECT_OUT_ROOT",
        REPO_ROOT / "output" / "MERSCOPE_WT_AD_comparison" / "baysor_ssam",
    )
)

MANIFEST_DIR = OUT_ROOT / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "manifest.csv"
N_JOBS_PATH = MANIFEST_DIR / "n_jobs.txt"
# Per-tile transcript shards (all genes), written once by build_manifest.py so the
# array tasks each read only their own tile instead of the whole sample.
SHARD_DIR = MANIFEST_DIR / "shards"


def shard_path(sample: str, tile_row: int, tile_col: int) -> Path:
    return SHARD_DIR / sample / f"tile_{tile_row:04d}_{tile_col:04d}.parquet"


def sample_dataset(sample: str) -> str:
    """'WT' -> 'MERSCOPE_WT_1', 'AD' -> 'MERSCOPE_AD_1'."""
    return {"WT": "MERSCOPE_WT_1", "AD": "MERSCOPE_AD_1"}[sample]


def transcripts_path(sample: str) -> Path:
    return DATA_ROOT / sample_dataset(sample) / "processed_data" / "transcripts.parquet"


# ----------------------------------------------------------------------------- #
# Transcript-table schema (MERSCOPE native; matches mcDETECT & the sim converters)
# ----------------------------------------------------------------------------- #
X_COL, Y_COL, Z_COL, GENE_COL = "global_x", "global_y", "global_z", "target"
READ_COLS = [X_COL, Y_COL, Z_COL, GENE_COL]

# MERSCOPE has a discrete z-grid; we run the methods in 3D for parity with mcDETECT.
IS_3D = True

# ----------------------------------------------------------------------------- #
# Sweep axes
# ----------------------------------------------------------------------------- #
SAMPLES = ["WT", "AD"]
PARAMS = ["default", "tuned"]
GENESETS = ["all", "markers"]

# The 20 synaptic granule markers mcDETECT seeds on (code/3_detection.py::syn_genes).
# geneset == "markers" restricts the transcript input to exactly these; "all" keeps
# the full panel (the methods' intended usage).
SYN_GENES = [
    "Camk2a", "Cplx2", "Slc17a7", "Ddn", "Syp", "Map1a", "Shank1", "Syn1",
    "Gria1", "Gria2", "Cyfip2", "Vamp2", "Bsn", "Slc32a1", "Nfasc", "Syt1",
    "Tubb3", "Nav1", "Shank3", "Mapt",
]

# ----------------------------------------------------------------------------- #
# Tiling
# ----------------------------------------------------------------------------- #
TILE_SIZE = 1000.0  # microns; 1000 x 1000 um squares detected in parallel

# ----------------------------------------------------------------------------- #
# Method parameters -- LOCKED from the simulation comparison
#   Baysor : simulation/output/Baysor_output_{min_mols}_{scale}
#            default = 30_30, tuned = 30_1.5  (only `scale` moves; -m stays 30)
#   SSAM   : simulation/output/SSAM_output_{norm}_{expression}
#            default = 0_0 (no thresholds), tuned = 0.2_0.027 (VISp preset)
# ----------------------------------------------------------------------------- #
BAYSOR_PARAMS = {
    "default": {"min_molecules_per_cell": 30, "scale": 30.0},
    "tuned":   {"min_molecules_per_cell": 30, "scale": 1.5},
}
# Baysor threads per task and the miniball drop applied when turning a Baysor cell
# into a sphere (matches the simulation converter; kept in the "no-filter" native
# set -- it is Baysor's own minimum-cell drop, not an mcDETECT filter).
BAYSOR_THREADS = int(os.environ.get("BAYSOR_THREADS", "8"))
BAYSOR_MIN_POINTS_PER_SEGMENT = 12.5
BAYSOR_MINIBALL_EPSILON = 1e-4

SSAM_PARAMS = {
    "default": {"expression_threshold": 0.0, "norm_threshold": 0.0},
    "tuned":   {"expression_threshold": 0.027, "norm_threshold": 0.2},
}
# Held constant in both settings (as in simulation).
SSAM_BANDWIDTH = 2.5
SSAM_SAMPLING_DISTANCE = 2.0
SSAM_SEARCH_SIZE = 3
SSAM_DETECTION_RADIUS = 1.5  # SSAM emits centers only; fixed sphere radius


# ----------------------------------------------------------------------------- #
# Output layout helpers
# ----------------------------------------------------------------------------- #
def config_tag(sample: str, param: str, geneset: str) -> str:
    return f"{sample}_{param}_{geneset}"


def config_dir(method: str, sample: str, param: str, geneset: str) -> Path:
    """.../baysor_ssam/<method>/<sample>_<param>_<geneset>/"""
    return OUT_ROOT / method / config_tag(sample, param, geneset)


def tile_path(method: str, sample: str, param: str, geneset: str,
              tile_row: int, tile_col: int) -> Path:
    return (config_dir(method, sample, param, geneset)
            / "tiles" / f"tile_{tile_row:04d}_{tile_col:04d}.parquet")


def spheres_path(method: str, sample: str, param: str, geneset: str) -> Path:
    """Concatenated per-config sphere table (produced by concat_spheres.py)."""
    return config_dir(method, sample, param, geneset) / "spheres.parquet"


def resolve_geneset(geneset: str):
    """Return the list of genes to keep, or None for the full panel."""
    if geneset == "all":
        return None
    if geneset == "markers":
        return list(SYN_GENES)
    raise ValueError(f"Unknown geneset: {geneset!r}")
