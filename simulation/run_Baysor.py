import glob
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import miniball
import numpy as np
import pandas as pd

from evaluation_utils import make_tree, metric_main, metric_main_polygons


# ==================== POLYGON vs SPHERE approach (switch here) ==================== #
# To use Baysor polygons (segmentation_polygons_3d.json) for evaluation:
#   - Set USE_POLYGONS = True below.
#   - In "Main loop" (run_all_baysor): the POLYGON branch is active; SPHERE branch is commented.
#   - In "Evaluation" section: the POLYGON branch (metric_main_polygons) is active; SPHERE branch is commented.
# To use minimum enclosing spheres again:
#   - Set USE_POLYGONS = False.
#   - In run_all_baysor: uncomment the SPHERE branch and comment the POLYGON branch.
#   - In Evaluation: uncomment the SPHERE branch and comment the POLYGON branch.
USE_POLYGONS = True

# ==================== User configurations ==================== #

# Roots relative to this script's working directory
SIM_DATA_ROOT = "simulated_data"
BAYSOR_OUT_ROOT = "output/Baysor_output"
os.makedirs(BAYSOR_OUT_ROOT, exist_ok=True)

# Baysor parameters (tune once then freeze)
DEFAULT_MIN_MOLS = 20
DEFAULT_SCALE = 3.0
DEFAULT_THREADS = 16

# Resume logic: skip if baysor_spheres.parquet already exists
RESUME_IF_DONE = True

# Run only a subset (e.g. 10) or None for all
LIMIT_N = None

# Restrict to 3D data only for parity with SSAM / mcDETECT
BAYSOR_3D_ONLY = True

PROGRESS_EVERY_N_SEEDS = 10

# Debug: optionally run on a small subset first (e.g. 10 seeds of single_marker 3D A)
# DEBUG_SAMPLE = True
# DEBUG_MODE = {
#     "mode": "single_marker",
#     "dimension": "3D",
#     "scenario": "A",
#     "n": 10,
# }
DEBUG_SAMPLE = True
DEBUG_MODE = {
    "mode": "multi_marker",
    "dimension": "3D",
    "scenario": "all",
    "n": 3,
}

# How to invoke Baysor on HGCC:
# - By default, we call "baysor" and expect it to be on PATH inside baysor_env.
# - Alternatively, set environment variable BAYSOR_BINARY to a full path, e.g.
#     export BAYSOR_BINARY="$HOME/.julia/bin/baysor"
BAYSOR_BINARY = os.environ.get("BAYSOR_BINARY", "baysor")

print("SIM_DATA_ROOT:", SIM_DATA_ROOT)
print("BAYSOR_OUT_ROOT:", BAYSOR_OUT_ROOT)
print("BAYSOR_3D_ONLY:", BAYSOR_3D_ONLY)
print("BAYSOR_BINARY:", BAYSOR_BINARY or "(baysor from PATH)")


# ==================== Discover simulated parquet files ==================== #

def discover_simulated_data(sim_root: str) -> pd.DataFrame:
    rows = []

    # single_marker/{dimension}/{name}/seed_{seed}.parquet
    pattern_single = os.path.join(sim_root, "single_marker", "*", "*", "seed_*.parquet")
    for f in sorted(glob.glob(pattern_single)):
        parts = f.split(os.sep)
        dimension = parts[-3]
        name = parts[-2]
        m = re.search(r"seed_(\d+)\.parquet$", parts[-1])
        if not m:
            continue
        seed = int(m.group(1))
        rows.append(
            {
                "mode": "single_marker",
                "dimension": dimension,
                "scenario": name,
                "seed": seed,
                "is_3d": ("3D" in dimension.upper()),
                "input_parquet": f,
            }
        )

    # multi_marker/{dimension}/all/seed_{seed}.parquet
    pattern_multi = os.path.join(sim_root, "multi_marker", "*", "all", "seed_*.parquet")
    for f in sorted(glob.glob(pattern_multi)):
        parts = f.split(os.sep)
        dimension = parts[-3]
        m = re.search(r"seed_(\d+)\.parquet$", parts[-1])
        if not m:
            continue
        seed = int(m.group(1))
        rows.append(
            {
                "mode": "multi_marker",
                "dimension": dimension,
                "scenario": "all",
                "seed": seed,
                "is_3d": ("3D" in dimension.upper()),
                "input_parquet": f,
            }
        )

    df = pd.DataFrame(rows)
    if df.shape[0] == 0:
        print("No simulated parquet files found. Check SIM_DATA_ROOT.")
    return df


inputs_df = discover_simulated_data(SIM_DATA_ROOT)
if BAYSOR_3D_ONLY:
    inputs_df = inputs_df[inputs_df["is_3d"]].copy().reset_index(drop=True)
    print("Restricted to 3D data only. Simulations to run:", inputs_df.shape[0])
else:
    print("Total simulations found:", inputs_df.shape[0])
print(inputs_df.head())

# Optional debug subset: limit to a small slice (default: 10 seeds of single_marker 3D A)
if DEBUG_SAMPLE:
    mask = (
        (inputs_df["mode"] == DEBUG_MODE["mode"])
        & (inputs_df["dimension"] == DEBUG_MODE["dimension"])
        & (inputs_df["scenario"] == DEBUG_MODE["scenario"])
    )
    inputs_df = (
        inputs_df[mask]
        .head(int(DEBUG_MODE.get("n", 10)))
        .copy()
        .reset_index(drop=True)
    )
    print(
        "DEBUG_SAMPLE enabled – running only on",
        f"{inputs_df.shape[0]} rows from",
        f"{DEBUG_MODE['mode']} {DEBUG_MODE['dimension']} {DEBUG_MODE['scenario']}",
    )
    print(inputs_df)


# ==================== Convert simulated parquet → Baysor molecule table ==================== #

def simulated_to_baysor_table(sim_parquet: str, is_3d: bool) -> pd.DataFrame:
    """
    Convert simulated parquet to Baysor molecule table with columns:
      transcript_id, x, y, z, gene
    Accepts either (x,y,z,gene) or (global_x,global_y,global_z,target) in input.
    """
    df = pd.read_parquet(sim_parquet).reset_index(drop=True)

    if {"x", "y", "gene"}.issubset(df.columns):
        x_col, y_col, z_col, gene_col = "x", "y", "z", "gene"
    elif {"global_x", "global_y", "target"}.issubset(df.columns):
        x_col, y_col, z_col, gene_col = "global_x", "global_y", "global_z", "target"
    else:
        raise ValueError(
            f"{sim_parquet}: need (x,y,z,gene) or (global_x,global_y,global_z,target). "
            f"Got: {list(df.columns)}"
        )

    if "transcript_id" not in df.columns:
        df["transcript_id"] = df.index.astype(int)

    z_vals = df[z_col].astype(float) if (is_3d and z_col in df.columns) else np.zeros(len(df))
    baysor_df = pd.DataFrame(
        {
            "transcript_id": df["transcript_id"].astype(int),
            "x": df[x_col].astype(float),
            "y": df[y_col].astype(float),
            "z": z_vals,
            "gene": df[gene_col].astype(str),
        }
    )
    return baysor_df


# ==================== Output path mapping (mirror directory layout) ==================== #

# Baysor 3D polygon output (when USE_POLYGONS=True we use this for evaluation)
BAYSOR_POLYGONS_JSON = "segmentation_polygons_3d.json"


def make_baysor_out_paths(sim_parquet: str, sim_root: str, out_root: str):
    rel = os.path.relpath(sim_parquet, sim_root)
    rel_no_ext = os.path.splitext(rel)[0]
    out_dir = os.path.join(out_root, rel_no_ext)
    spheres_parquet = os.path.join(out_dir, "baysor_spheres.parquet")
    polygons_json = os.path.join(out_dir, BAYSOR_POLYGONS_JSON)
    return out_dir, spheres_parquet, polygons_json


# ==================== Run Baysor CLI and convert to detection spheres ==================== #

def run_baysor_cli(
    in_csv: str,
    out_dir: str,
    is_3d: bool,
    min_molecules_per_cell: int,
    scale: float,
    n_threads: int = 8,
    make_plots: bool = False,
    timeout_sec: int | None = None,
) -> str:
    """
    Run Baysor CLI and return path to segmentation.csv in out_dir.
    Assumes Baysor is installed in baysor_env and callable as `baysor` or via BAYSOR_BINARY.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = str(n_threads)

    baysor_cmd = BAYSOR_BINARY or "baysor"
    baysor_cmd = baysor_cmd.strip() or "baysor"
    baysor_exe = os.path.expanduser(baysor_cmd)
    if os.sep in baysor_exe:
        baysor_exe = os.path.abspath(baysor_exe)

    # Coordinates file as first positional after "run"
    cmd = [baysor_exe, "run", in_csv, "-x", "x", "-y", "y", "-g", "gene"]
    if is_3d:
        cmd += ["-z", "z"]

    cmd += ["-m", str(min_molecules_per_cell), "-s", str(scale)]

    if make_plots:
        cmd += ["--plot"]

    cmd += ["-o", str(out_dir_path)]

    print("    Baysor command:", " ".join(cmd))
    result = subprocess.run(
        cmd,
        env=env,
        timeout=timeout_sec,
        capture_output=True,
        text=True,
    )
    seg_csv = out_dir_path / "segmentation.csv"
    if seg_csv.exists():
        # Baysor sometimes returns exit code 1 even when it completes successfully
        # (e.g. progress-bar or non-interactive quirks). Treat output presence as success.
        return str(seg_csv)
    if result.returncode != 0:
        err_msg = (
            f"Baysor exited with code {result.returncode}. "
            f"stderr: {result.stderr or '(empty)'}. stdout: {result.stdout or '(empty)'}"
        )
        raise RuntimeError(err_msg)
    raise FileNotFoundError(
        f"Expected {seg_csv} not found. Files in {out_dir_path}: {list(out_dir_path.glob('*'))}"
    )


# ==================== Polygon-based detection (used when USE_POLYGONS=True) ==================== #

def _parse_z_range(z_key: str):
    """Parse Baysor z-slice key like '[-35.14, 0.89)' into (z_lo, z_hi)."""
    import re
    m = re.match(r"\[\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)", z_key.strip())
    if not m:
        raise ValueError(f"Invalid z range key: {z_key!r}")
    return float(m.group(1)), float(m.group(2))


def load_baysor_polygons_3d(json_path: str):
    """
    Load Baysor segmentation_polygons_3d.json and return a list of cells suitable for
    metric_main_polygons. Each cell is a list of (z_lo, z_hi, ring) where ring is a list
    of [x, y] (exterior boundary). Keys in the JSON are z-intervals; features are
    GeoJSON Polygon features with id (cell id).
    """
    import json
    with open(json_path) as f:
        data = json.load(f)
    # Group by feature id (cell id) across z-slices: cell_id -> [(z_lo, z_hi, ring), ...]
    by_cell = {}
    for z_key, obj in data.items():
        z_lo, z_hi = _parse_z_range(z_key)
        for feat in obj.get("features", []):
            geom = feat.get("geometry")
            fid = feat.get("id", "")
            if not geom or geom.get("type") != "Polygon":
                continue
            coords = geom.get("coordinates")
            if not coords or not coords[0]:
                continue
            ring = [[float(p[0]), float(p[1])] for p in coords[0]]
            if len(ring) < 3:
                continue
            by_cell.setdefault(fid, []).append((z_lo, z_hi, ring))
    return list(by_cell.values())


# ==================== Sphere-based detection (used when USE_POLYGONS=False) ==================== #
# def baysor_segmentation_to_spheres(
#     seg_csv: str,
#     baysor_input_csv: str,
#     miniball_epsilon: float = 1e-4,
#     min_points_per_segment: int = 3,
# ) -> pd.DataFrame:
#     """
#     Read segmentation.csv + Baysor input CSV and build a sphere table:
#       sphere_x, sphere_y, sphere_z, sphere_r
#     """
#     seg = pd.read_csv(seg_csv)
#     mol = pd.read_csv(baysor_input_csv, usecols=["transcript_id", "x", "y", "z"])

#     if "transcript_id" not in seg.columns:
#         raise ValueError(f"{seg_csv} missing transcript_id. Columns: {list(seg.columns)}")
#     if "cell" not in seg.columns:
#         raise ValueError(f"{seg_csv} missing cell. Columns: {list(seg.columns)}")

#     merged = seg.merge(mol, on="transcript_id", how="inner")

#     # Conventions: cell==0 means unassigned
#     merged = merged[merged["cell"] != 0]

#     # Optional noise filter if present
#     if "is_noise" in merged.columns:
#         merged = merged[merged["is_noise"].astype(str).str.lower() != "true"]

#     rows = []
#     for cell_id, g in merged.groupby("cell", sort=False):
#         coords = g[["x", "y", "z"]].to_numpy(dtype=float)
#         n = coords.shape[0]
#         if n < min_points_per_segment:
#             continue
#         center, r2 = miniball.get_bounding_ball(coords, epsilon=miniball_epsilon)
#         rows.append(
#             {
#                 "sphere_x": float(center[0]),
#                 "sphere_y": float(center[1]),
#                 "sphere_z": float(center[2]),
#                 "sphere_r": float(np.sqrt(r2)),
#                 "cell_id": int(cell_id),
#                 "n_molecules": int(n),
#             }
#         )

#     return pd.DataFrame(rows, columns=["sphere_x", "sphere_y", "sphere_z", "sphere_r", "cell_id", "n_molecules"])


def baysor_segmentation_to_spheres(
    seg_csv: str,
    miniball_epsilon: float = 1e-4,
    min_points_per_segment: int = 3,
) -> pd.DataFrame:
    """
    Read Baysor segmentation.csv and build a sphere table:
      sphere_x, sphere_y, sphere_z, sphere_r
    """
    seg = pd.read_csv(seg_csv)

    required = {"transcript_id", "x", "y", "z", "cell"}
    missing = required - set(seg.columns)
    if missing:
        raise ValueError(f"{seg_csv} missing required columns: {missing}")

    # cell==0 means unassigned
    seg = seg[seg["cell"] != 0]

    # Optional noise filter
    if "is_noise" in seg.columns:
        seg = seg[seg["is_noise"].astype(str).str.lower() != "true"]

    rows = []
    for cell_id, g in seg.groupby("cell", sort=False):
        coords = g[["x", "y", "z"]].to_numpy(dtype=float)
        n = coords.shape[0]
        if n < min_points_per_segment:
            continue

        center, r2 = miniball.get_bounding_ball(coords, epsilon=miniball_epsilon)
        rows.append(
            {
                "sphere_x": float(center[0]),
                "sphere_y": float(center[1]),
                "sphere_z": float(center[2]),
                "sphere_r": float(np.sqrt(r2)),
                "cell_id": str(cell_id),
                "n_molecules": int(n),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["sphere_x", "sphere_y", "sphere_z", "sphere_r", "cell_id", "n_molecules"],
    )


# ==================== Main loop: run Baysor over all simulations ==================== #

def run_all_baysor(
    inputs_df: pd.DataFrame,
    sim_root: str,
    out_root: str,
    min_mols: int,
    scale: float,
    threads: int,
    resume_if_done: bool = True,
    limit_n: int | None = None,
    progress_every_n: int | None = None,
) -> pd.DataFrame:
    logs = []
    df = inputs_df.copy()
    if limit_n is not None:
        df = df.head(limit_n)

    progress_every = progress_every_n or PROGRESS_EVERY_N_SEEDS
    setting_totals = df.groupby(["mode", "dimension", "scenario"]).size()
    setting_count = {}

    def _maybe_print(row):
        key = (row["mode"], row["dimension"], row["scenario"])
        setting_count[key] = setting_count.get(key, 0) + 1
        n, total = setting_count[key], int(setting_totals.get(key, 0))
        if progress_every and (n % progress_every == 0 or n == total):
            print(f"  [{row['mode']}] {row['dimension']} {row['scenario']}: {n}/{total} seeds.")

    for _, row in df.iterrows():
        sim_parquet = row["input_parquet"]
        is_3d = bool(row["is_3d"])
        out_dir, spheres_parquet, polygons_json = make_baysor_out_paths(sim_parquet, sim_root, out_root)

        # Resume: POLYGON path checks for segmentation_polygons_3d.json; SPHERE path checks for baysor_spheres.parquet
        if USE_POLYGONS:
            if resume_if_done and os.path.exists(polygons_json):
                cells = load_baysor_polygons_3d(polygons_json)
                logs.append(
                    {
                        **row.to_dict(),
                        "status": "skipped_exists",
                        "out_dir": out_dir,
                        "spheres_parquet": spheres_parquet,
                        "detection_type": "polygons",
                        "detection_path": polygons_json,
                        "n_spheres": len(cells),
                        "runtime_sec": 0.0,
                        "error": None,
                    }
                )
                _maybe_print(row)
                continue
        else:
            if resume_if_done and os.path.exists(spheres_parquet):
                n_spheres = pd.read_parquet(spheres_parquet).shape[0]
                logs.append(
                    {
                        **row.to_dict(),
                        "status": "skipped_exists",
                        "out_dir": out_dir,
                        "spheres_parquet": spheres_parquet,
                        "detection_type": "spheres",
                        "detection_path": spheres_parquet,
                        "n_spheres": n_spheres,
                        "runtime_sec": 0.0,
                        "error": None,
                    }
                )
                _maybe_print(row)
                continue

        t0 = time.time()
        print(
            f"  Starting: {row['mode']} {row['dimension']} {row['scenario']} seed_{row['seed']} ...",
            flush=True,
        )
        try:
            baysor_df = simulated_to_baysor_table(sim_parquet, is_3d)
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                in_csv = os.path.join(tmpdir, "molecules.csv")
                baysor_df.to_csv(in_csv, index=False)

                seg_csv = run_baysor_cli(
                    in_csv=in_csv,
                    out_dir=out_dir,
                    is_3d=is_3d,
                    min_molecules_per_cell=min_mols,
                    scale=scale,
                    n_threads=threads,
                    make_plots=False,
                )

                if USE_POLYGONS:
                    # -------- POLYGON approach: use segmentation_polygons_3d.json for evaluation --------
                    if not os.path.exists(polygons_json):
                        raise FileNotFoundError(
                            f"USE_POLYGONS=True but {BAYSOR_POLYGONS_JSON} not found in {out_dir}"
                        )
                    cells = load_baysor_polygons_3d(polygons_json)
                    detection_type, detection_path = "polygons", polygons_json
                    n_detections = len(cells)
                else:
                    # -------- SPHERE approach: miniball enclosing spheres from segmentation.csv --------
                    # spheres = baysor_segmentation_to_spheres(seg_csv, in_csv)  # older signature
                    spheres = baysor_segmentation_to_spheres(seg_csv)
                    spheres.to_parquet(spheres_parquet, index=False)
                    detection_type, detection_path = "spheres", spheres_parquet
                    n_detections = int(spheres.shape[0])

            logs.append(
                {
                    **row.to_dict(),
                    "status": "ok",
                    "out_dir": out_dir,
                    "spheres_parquet": spheres_parquet,
                    "detection_type": detection_type,
                    "detection_path": detection_path,
                    "n_spheres": n_detections,
                    "runtime_sec": float(time.time() - t0),
                    "error": None,
                }
            )
            _maybe_print(row)
        except Exception as e:
            logs.append(
                {
                    **row.to_dict(),
                    "status": "failed",
                    "out_dir": out_dir,
                    "spheres_parquet": spheres_parquet,
                    "detection_type": "spheres" if not USE_POLYGONS else "polygons",
                    "detection_path": spheres_parquet if not USE_POLYGONS else polygons_json,
                    "n_spheres": None,
                    "runtime_sec": float(time.time() - t0),
                    "error": repr(e),
                }
            )
            _maybe_print(row)

    return pd.DataFrame(logs)


logs_df = run_all_baysor(
    inputs_df=inputs_df,
    sim_root=SIM_DATA_ROOT,
    out_root=BAYSOR_OUT_ROOT,
    min_mols=DEFAULT_MIN_MOLS,
    scale=DEFAULT_SCALE,
    threads=DEFAULT_THREADS,
    resume_if_done=RESUME_IF_DONE,
    limit_n=LIMIT_N,
)

print("Status counts:", logs_df["status"].value_counts().to_dict())


# ==================== Save logs and create index for evaluation ==================== #

Path(BAYSOR_OUT_ROOT).mkdir(parents=True, exist_ok=True)
log_path = os.path.join(BAYSOR_OUT_ROOT, "baysor_run_log.csv")
logs_df.to_csv(log_path, index=False)
print("Saved log:", log_path)

index_df = logs_df[logs_df["status"].isin(["ok", "skipped_exists"])][
    ["mode", "dimension", "scenario", "seed", "spheres_parquet", "out_dir", "detection_type", "detection_path"]
].copy()
index_path = os.path.join(
    BAYSOR_OUT_ROOT,
    "baysor_polygons_index.csv" if USE_POLYGONS else "baysor_spheres_index.csv",
)
index_df.to_csv(index_path, index=False)
print("Saved index:", index_path)
print(index_df.head())


# ==================== Evaluation: ground truth and metrics ==================== #
# When USE_POLYGONS=True: the loop below uses detection_type to call either
# metric_main_polygons (polygons) or metric_main (spheres). No need to comment/uncomment here;
# switch via USE_POLYGONS at the top of the file.

def get_ground_truth_single(dimension: str, scenario: str, seed: int):
    """Load single-marker ground-truth parents from Parquet and build KD-tree."""
    gt_path = os.path.join(
        SIM_DATA_ROOT,
        "single_marker",
        dimension,
        scenario,
        f"seed_{seed}_ground_truth.parquet",
    )
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
    parents_all = pd.read_parquet(gt_path)
    tree = make_tree(
        d1=np.array(parents_all["x"]),
        d2=np.array(parents_all["y"]),
        d3=np.array(parents_all["z"]),
    )
    return parents_all, tree


def get_ground_truth_multi(dimension: str, seed: int):
    """Load multi-marker ground-truth parents from Parquet (scenario 'all') and build KD-tree."""
    gt_path = os.path.join(
        SIM_DATA_ROOT,
        "multi_marker",
        dimension,
        "all",
        f"seed_{seed}_ground_truth.parquet",
    )
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
    parents_all = pd.read_parquet(gt_path)
    tree = make_tree(
        d1=np.array(parents_all["x"]),
        d2=np.array(parents_all["y"]),
        d3=np.array(parents_all["z"]),
    )
    return parents_all, tree


# Same constants as main.ipynb / run_SSAM.py (for naming)
POINT_TYPE = ["CSR", "Extranuclear", "Intranuclear"]
RATIO = [0.5, 0.25, 0.25]
MEAN_DIST_EXTRA = 1
MEAN_DIST_INTRA = 4
BETA_EXTRA = (1, 19)
BETA_INTRA = (19, 1)
MARKER_SETTINGS = {
    "A": {"density": 0.08, "num_clusters_extra": 5000, "num_clusters_intra": 2000},
    "B": {"density": 0.04, "num_clusters_extra": 3000, "num_clusters_intra": 1200},
    "C": {"density": 0.02, "num_clusters_extra": 2000, "num_clusters_intra": 800},
}
SHAPE = (2000, 2000)
LAYER_NUM = 8
LAYER_GAP = 1.5
NAME_MULTI = ["A", "B", "C"]
CSR_DENSITY = [0.04, 0.02, 0.01]
EXTRA_DENSITY = [0.02, 0.01, 0.005]
EXTRA_NUM_CLUSTERS = 5000
EXTRA_BETA = (1, 19)
EXTRA_COMP_PROB = [0.4, 0.3, 0.3]
EXTRA_MEAN_DIST = 1
INTRA_DENSITY = [0.02, 0.01, 0.005]
INTRA_NUM_CLUSTERS = 1000
INTRA_BETA = (19, 1)
INTRA_COMP_PROB = [0.8, 0.1, 0.1]
INTRA_MEAN_DIST = 4
INTRA_COMP_THR = 2
EXTRA_COMP_THR = 2

EVAL_OUT_DIR = BAYSOR_OUT_ROOT
Path(EVAL_OUT_DIR).mkdir(parents=True, exist_ok=True)

single_results: list[tuple] = []
multi_results: list[tuple] = []

for _, row in index_df.iterrows():
    mode = row["mode"]
    dimension = row["dimension"]
    scenario = row["scenario"]
    seed = row["seed"]
    detection_path = row["detection_path"]
    detection_type = row["detection_type"]
    if not os.path.exists(detection_path):
        continue
    try:
        if mode == "single_marker":
            parents_all, tree = get_ground_truth_single(dimension, scenario, seed)
        else:
            parents_all, tree = get_ground_truth_multi(dimension, seed)
        ground_truth_indices = set(parents_all.index)

        if detection_type == "polygons":
            # -------- POLYGON approach: point-in-polygon matching --------
            cells_polygons = load_baysor_polygons_3d(detection_path)
            if len(cells_polygons) == 0:
                if mode == "single_marker":
                    single_results.append((dimension, scenario, seed, 0.0, 0.0, 0.0, 0.0))
                else:
                    multi_results.append((dimension, seed, 0.0, 0.0, 0.0, 0.0))
                continue
            precision, recall, accuracy, f1 = metric_main_polygons(
                parents_all, ground_truth_indices, cells_polygons
            )
        else:
            # -------- SPHERE approach: point-in-sphere matching --------
            sphere = pd.read_parquet(detection_path)
            if sphere.shape[0] == 0:
                if mode == "single_marker":
                    single_results.append((dimension, scenario, seed, 0.0, 0.0, 0.0, 0.0))
                else:
                    multi_results.append((dimension, seed, 0.0, 0.0, 0.0, 0.0))
                continue
            precision, recall, accuracy, f1 = metric_main(tree, ground_truth_indices, sphere)

        if mode == "single_marker":
            single_results.append((dimension, scenario, seed, precision, recall, accuracy, f1))
        else:
            multi_results.append((dimension, seed, precision, recall, accuracy, f1))
    except Exception as e:
        print(f"Error evaluating {detection_path}: {e}")
        if mode == "single_marker":
            single_results.append((dimension, scenario, seed, np.nan, np.nan, np.nan, np.nan))
        else:
            multi_results.append((dimension, seed, np.nan, np.nan, np.nan, np.nan))

print("Single-marker results:", len(single_results))
print("Multi-marker results:", len(multi_results))

for (dimension, scenario) in set((r[0], r[1]) for r in single_results):
    rows = [r for r in single_results if r[0] == dimension and r[1] == scenario]
    rows.sort(key=lambda x: x[2])
    extra = MARKER_SETTINGS[scenario]["num_clusters_extra"]
    intra = MARKER_SETTINGS[scenario]["num_clusters_intra"]
    df = pd.DataFrame(
        {
            "Simulation": [r[2] for r in rows],
            "Precision": [r[3] for r in rows],
            "Recall": [r[4] for r in rows],
            "Accuracy": [r[5] for r in rows],
            "F1 Score": [r[6] for r in rows],
        }
    )
    path = os.path.join(
        EVAL_OUT_DIR,
        f"single_marker_{dimension}_{scenario}_{extra}_{intra}_baysor.csv",
    )
    df.to_csv(path, index=False)
    print("Saved:", path)

for dimension in set(r[0] for r in multi_results):
    rows = [r for r in multi_results if r[0] == dimension]
    rows.sort(key=lambda x: x[1])
    df = pd.DataFrame(
        {
            "Simulation": [r[1] for r in rows],
            "Precision": [r[2] for r in rows],
            "Recall": [r[3] for r in rows],
            "Accuracy": [r[4] for r in rows],
            "F1": [r[5] for r in rows],
        }
    )
    path = os.path.join(
        EVAL_OUT_DIR,
        f"multi_marker_{dimension}_all_{EXTRA_NUM_CLUSTERS}_{INTRA_NUM_CLUSTERS}_baysor.csv",
    )
    df.to_csv(path, index=False)
    print("Saved:", path)