import glob
import numpy as np
import os
import pandas as pd
import re
import tempfile
import time

from evaluation_utils import make_tree, metric_main
from pathlib import Path

import ssam
print("ssam:", getattr(ssam, "__version__", "unknown"))

# ==================== User configurations ==================== #

# Roots relative to this script's working directory
SIM_DATA_ROOT = "simulated_data"
SSAM_OUT_ROOT = "output/SSAM_output"
os.makedirs(SSAM_OUT_ROOT, exist_ok=True)

# SSAM parameters (tune once then freeze)
DEFAULT_BANDWIDTH = 2.5
DEFAULT_SAMPLING_DISTANCE = 1.0
DEFAULT_FIND_LOCALMAX_SEARCH_SIZE = 3

# Fixed radius for SSAM "detection spheres" (SSAM outputs cell centers only; we need sphere_r for metric_main)
SSAM_DETECTION_RADIUS = 5.0

# Resume: skip if ssam_spheres.parquet already exists
RESUME_IF_DONE = True

# Run only a subset (e.g. 10) or None for all
LIMIT_N = None

# Restrict to 3D data only for parity with run_Baysor; SSAM run_kde supports depth for 3D
SSAM_3D_ONLY = True

PROGRESS_EVERY_N_SEEDS = 10

print("SIM_DATA_ROOT:", SIM_DATA_ROOT)
print("SSAM_OUT_ROOT:", SSAM_OUT_ROOT)
print("SSAM_3D_ONLY:", SSAM_3D_ONLY)

# ==================== Discover simulated parquet files ==================== #

def discover_simulated_data(sim_root: str) -> pd.DataFrame:
    rows = []
    pattern_single = os.path.join(sim_root, "single_marker", "*", "*", "seed_*.parquet")
    for f in sorted(glob.glob(pattern_single)):
        parts = f.split(os.sep)
        dimension = parts[-3]
        name = parts[-2]
        m = re.search(r"seed_(\d+)\.parquet$", parts[-1])
        if not m:
            continue
        seed = int(m.group(1))
        rows.append({
            "mode": "single_marker",
            "dimension": dimension,
            "scenario": name,
            "seed": seed,
            "is_3d": ("3D" in dimension.upper()),
            "input_parquet": f,
        })

    pattern_multi = os.path.join(sim_root, "multi_marker", "*", "all", "seed_*.parquet")
    for f in sorted(glob.glob(pattern_multi)):
        parts = f.split(os.sep)
        dimension = parts[-3]
        m = re.search(r"seed_(\d+)\.parquet$", parts[-1])
        if not m:
            continue
        seed = int(m.group(1))
        rows.append({
            "mode": "multi_marker",
            "dimension": dimension,
            "scenario": "all",
            "seed": seed,
            "is_3d": ("3D" in dimension.upper()),
            "input_parquet": f,
        })

    df = pd.DataFrame(rows)
    if df.shape[0] == 0:
        print("No simulated parquet files found. Check SIM_DATA_ROOT.")
    return df

inputs_df = discover_simulated_data(SIM_DATA_ROOT)
if SSAM_3D_ONLY:
    inputs_df = inputs_df[inputs_df["is_3d"]].copy().reset_index(drop=True)
    print("Restricted to 3D data only. Simulations to run:", inputs_df.shape[0])
else:
    print("Total simulations found:", inputs_df.shape[0])
print(inputs_df.head())

# ==================== Convert simulated parquet file → SSAM input (x, y, z, gene) ==================== #

def simulated_to_ssam_df(sim_parquet: str, is_3d: bool) -> pd.DataFrame:
    """Convert simulated parquet file to DataFrame with x, y, gene (and z if 3D) for SSAM."""
    df = pd.read_parquet(sim_parquet).reset_index(drop=True)
    if "x" in df.columns and "y" in df.columns and "gene" in df.columns:
        x_col, y_col, z_col, gene_col = "x", "y", "z", "gene"
    elif "global_x" in df.columns and "global_y" in df.columns and "target" in df.columns:
        x_col, y_col, z_col, gene_col = "global_x", "global_y", "global_z", "target"
    else:
        raise ValueError(f"{sim_parquet}: need (x,y,z,gene) or (global_x,global_y,global_z,target). Got: {list(df.columns)}")

    out = pd.DataFrame({
        "x": df[x_col].astype(float),
        "y": df[y_col].astype(float),
        "gene": df[gene_col].astype(str),
    })
    if is_3d and z_col in df.columns:
        out["z"] = df[z_col].astype(float)
    else:
        out["z"] = 0.0
    return out

# ==================== Output path mapping (mirror directory layout) ==================== #

def make_ssam_out_paths(sim_parquet: str, sim_root: str, out_root: str):
    rel = os.path.relpath(sim_parquet, sim_root)
    rel_no_ext = os.path.splitext(rel)[0]
    out_dir = os.path.join(out_root, rel_no_ext)
    spheres_parquet = os.path.join(out_dir, "ssam_spheres.parquet")
    return out_dir, spheres_parquet

# ==================== Run SSAM (KDE + local maxima) and convert to detection spheres ==================== #

"""
SSAM is segmentation-free: it builds a gene-expression vector field via KDE and finds local maxima as probable cell locations. We use those local maxima as detected cell centers
and assign a fixed radius for evaluation with metric_main (same protocol as Baysor spheres). No clustering or cell-type mapping is applied for this benchmark.
"""

def ssam_localmax_to_spheres(dataset, detection_radius: float, is_3d: bool) -> pd.DataFrame:
    """
    Get physical coordinates of SSAM local maxima and build a sphere table (sphere_x, sphere_y, sphere_z, sphere_r).
    dataset.local_maxs are grid indices; vf_params[0] is sampling_distance.
    """
    lm = dataset.local_maxs
    if lm is None or len(lm[0]) == 0:
        return pd.DataFrame(columns=["sphere_x", "sphere_y", "sphere_z", "sphere_r"])

    sampling = float(dataset.zarr_group["vf_params"][0])
    # Grid indices -> physical coordinates (SSAM uses order consistent with vf shape: width, height, depth)
    x_phys = np.array(lm[0], dtype=float) * sampling
    y_phys = np.array(lm[1], dtype=float) * sampling
    if is_3d and len(lm) >= 3:
        z_phys = np.array(lm[2], dtype=float) * sampling
    else:
        z_phys = np.zeros_like(x_phys)

    return pd.DataFrame({
        "sphere_x": x_phys,
        "sphere_y": y_phys,
        "sphere_z": z_phys,
        "sphere_r": detection_radius,
    })

def run_ssam_one(
    sim_parquet: str,
    out_dir: str,
    spheres_parquet: str,
    is_3d: bool,
    bandwidth: float,
    sampling_distance: float,
    search_size: int,
    detection_radius: float,
    verbose: bool = False,
) -> None:
    """Run SSAM KDE + find_localmax for one simulation and save ssam_spheres.parquet."""
    df = simulated_to_ssam_df(sim_parquet, is_3d)
    width = float(df["x"].max())
    height = float(df["y"].max())
    depth = float(df["z"].max()) + 1.0 if is_3d else 1.0

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        ds = ssam.SSAMDataset(tmpdir)
        analysis = ssam.SSAMAnalysis(ds, verbose=verbose)
        # run_kde expects DataFrame with columns x, y, gene (and z for 3D)
        analysis.run_kde(
            df,
            width=width,
            height=height,
            depth=depth,
            bandwidth=bandwidth,
            sampling_distance=sampling_distance,
        )
        analysis.find_localmax(search_size=search_size)

        spheres = ssam_localmax_to_spheres(ds, detection_radius, is_3d)

    spheres.to_parquet(spheres_parquet, index=False)
    
# ==================== Main loop: run SSAM over all simulations ==================== #

def run_all_ssam(
    inputs_df: pd.DataFrame,
    sim_root: str,
    out_root: str,
    bandwidth: float,
    sampling_distance: float,
    search_size: int,
    detection_radius: float,
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
        out_dir, spheres_parquet = make_ssam_out_paths(sim_parquet, sim_root, out_root)

        if resume_if_done and os.path.exists(spheres_parquet):
            n_spheres = pd.read_parquet(spheres_parquet).shape[0]
            logs.append({
                **row.to_dict(),
                "status": "skipped_exists",
                "out_dir": out_dir,
                "spheres_parquet": spheres_parquet,
                "n_spheres": n_spheres,
                "runtime_sec": 0.0,
                "error": None,
            })
            _maybe_print(row)
            continue

        t0 = time.time()
        print(f"  Starting: {row['mode']} {row['dimension']} {row['scenario']} seed_{row['seed']} ...", flush=True)
        try:
            run_ssam_one(
                sim_parquet=sim_parquet,
                out_dir=out_dir,
                spheres_parquet=spheres_parquet,
                is_3d=is_3d,
                bandwidth=bandwidth,
                sampling_distance=sampling_distance,
                search_size=search_size,
                detection_radius=detection_radius,
            )
            spheres = pd.read_parquet(spheres_parquet)
            logs.append({
                **row.to_dict(),
                "status": "ok",
                "out_dir": out_dir,
                "spheres_parquet": spheres_parquet,
                "n_spheres": int(spheres.shape[0]),
                "runtime_sec": float(time.time() - t0),
                "error": None,
            })
            _maybe_print(row)
        except Exception as e:
            logs.append({
                **row.to_dict(),
                "status": "failed",
                "out_dir": out_dir,
                "spheres_parquet": spheres_parquet,
                "n_spheres": None,
                "runtime_sec": float(time.time() - t0),
                "error": repr(e),
            })
            _maybe_print(row)

    return pd.DataFrame(logs)

logs_df = run_all_ssam(
    inputs_df=inputs_df,
    sim_root=SIM_DATA_ROOT,
    out_root=SSAM_OUT_ROOT,
    bandwidth=DEFAULT_BANDWIDTH,
    sampling_distance=DEFAULT_SAMPLING_DISTANCE,
    search_size=DEFAULT_FIND_LOCALMAX_SEARCH_SIZE,
    detection_radius=SSAM_DETECTION_RADIUS,
    resume_if_done=RESUME_IF_DONE,
    limit_n=LIMIT_N,
)

logs_df["status"].value_counts()

# ==================== Save logs and create index for evaluation ==================== #

Path(SSAM_OUT_ROOT).mkdir(parents=True, exist_ok=True)
log_path = os.path.join(SSAM_OUT_ROOT, "ssam_run_log.csv")
logs_df.to_csv(log_path, index=False)
print("Saved log:", log_path)

index_df = logs_df[logs_df["status"].isin(["ok", "skipped_exists"])][
    ["mode", "dimension", "scenario", "seed", "spheres_parquet", "out_dir"]
].copy()
index_path = os.path.join(SSAM_OUT_ROOT, "ssam_spheres_index.csv")
index_df.to_csv(index_path, index=False)
print("Saved index:", index_path)
print("\nStatus counts:", logs_df["status"].value_counts().to_dict())
print(index_df.head())

# # ==================== Evaluation: ground truth and metrics ==================== #

# def get_ground_truth_single(dimension: str, scenario: str, seed: int):
#     """Load single-marker ground-truth parents from Parquet and build KD-tree."""
#     gt_path = os.path.join(
#         SIM_DATA_ROOT,
#         "single_marker",
#         dimension,
#         scenario,
#         f"seed_{seed}_ground_truth.parquet",
#     )
#     if not os.path.exists(gt_path):
#         raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
#     parents_all = pd.read_parquet(gt_path)
#     tree = make_tree(
#         d1=np.array(parents_all["x"]),
#         d2=np.array(parents_all["y"]),
#         d3=np.array(parents_all["z"]),
#     )
#     return parents_all, tree

# def get_ground_truth_multi(dimension: str, seed: int):
#     """Load multi-marker ground-truth parents from Parquet and build KD-tree."""
#     gt_path = os.path.join(
#         SIM_DATA_ROOT,
#         "multi_marker",
#         dimension,
#         "all",
#         f"seed_{seed}_ground_truth.parquet",
#     )
#     if not os.path.exists(gt_path):
#         raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")
#     parents_all = pd.read_parquet(gt_path)
#     tree = make_tree(
#         d1=np.array(parents_all["x"]),
#         d2=np.array(parents_all["y"]),
#         d3=np.array(parents_all["z"]),
#     )
#     return parents_all, tree

# # Same constants as main.ipynb / run_Baysor.ipynb
# POINT_TYPE = ["CSR", "Extranuclear", "Intranuclear"]
# RATIO = [0.5, 0.25, 0.25]
# MEAN_DIST_EXTRA = 1
# MEAN_DIST_INTRA = 4
# BETA_EXTRA = (1, 19)
# BETA_INTRA = (19, 1)
# MARKER_SETTINGS = {
#     "A": {"density": 0.08, "num_clusters_extra": 5000, "num_clusters_intra": 2000},
#     "B": {"density": 0.04, "num_clusters_extra": 3000, "num_clusters_intra": 1200},
#     "C": {"density": 0.02, "num_clusters_extra": 2000, "num_clusters_intra": 800},
# }
# SHAPE = (2000, 2000)
# LAYER_NUM = 8
# LAYER_GAP = 1.5
# NAME_MULTI = ["A", "B", "C"]
# CSR_DENSITY = [0.04, 0.02, 0.01]
# EXTRA_DENSITY = [0.02, 0.01, 0.005]
# EXTRA_NUM_CLUSTERS = 5000
# EXTRA_BETA = (1, 19)
# EXTRA_COMP_PROB = [0.4, 0.3, 0.3]
# EXTRA_MEAN_DIST = 1
# INTRA_DENSITY = [0.02, 0.01, 0.005]
# INTRA_NUM_CLUSTERS = 1000
# INTRA_BETA = (19, 1)
# INTRA_COMP_PROB = [0.8, 0.1, 0.1]
# INTRA_MEAN_DIST = 4
# INTRA_COMP_THR = 2
# EXTRA_COMP_THR = 2

# EVAL_OUT_DIR = SSAM_OUT_ROOT
# Path(EVAL_OUT_DIR).mkdir(parents=True, exist_ok=True)

# single_results = []
# multi_results = []

# for _, row in index_df.iterrows():
#     mode = row["mode"]
#     dimension = row["dimension"]
#     scenario = row["scenario"]
#     seed = row["seed"]
#     spheres_parquet = row["spheres_parquet"]
#     if not os.path.exists(spheres_parquet):
#         continue
#     sphere = pd.read_parquet(spheres_parquet)
#     if sphere.shape[0] == 0:
#         if mode == "single_marker":
#             single_results.append((dimension, scenario, seed, 0.0, 0.0, 0.0, 0.0))
#         else:
#             multi_results.append((dimension, seed, 0.0, 0.0, 0.0, 0.0))
#         continue
#     try:
#         if mode == "single_marker":
#             parents_all, tree = get_ground_truth_single(dimension, scenario, seed)
#         else:
#             parents_all, tree = get_ground_truth_multi(dimension, seed)
#         ground_truth_indices = set(parents_all.index)
#         precision, recall, accuracy, f1 = metric_main(tree, ground_truth_indices, sphere)
#         if mode == "single_marker":
#             single_results.append((dimension, scenario, seed, precision, recall, accuracy, f1))
#         else:
#             multi_results.append((dimension, seed, precision, recall, accuracy, f1))
#     except Exception as e:
#         print(f"Error evaluating {spheres_parquet}: {e}")
#         if mode == "single_marker":
#             single_results.append((dimension, scenario, seed, np.nan, np.nan, np.nan, np.nan))
#         else:
#             multi_results.append((dimension, seed, np.nan, np.nan, np.nan, np.nan))

# print("Single-marker results:", len(single_results))
# print("Multi-marker results:", len(multi_results))

# for (dimension, scenario) in set((r[0], r[1]) for r in single_results):
#     rows = [r for r in single_results if r[0] == dimension and r[1] == scenario]
#     rows.sort(key=lambda x: x[2])
#     extra = MARKER_SETTINGS[scenario]["num_clusters_extra"]
#     intra = MARKER_SETTINGS[scenario]["num_clusters_intra"]
#     df = pd.DataFrame({
#         "Simulation": [r[2] for r in rows],
#         "Precision": [r[3] for r in rows],
#         "Recall": [r[4] for r in rows],
#         "Accuracy": [r[5] for r in rows],
#         "F1 Score": [r[6] for r in rows],
#     })
#     path = os.path.join(EVAL_OUT_DIR, f"single_marker_{dimension}_{scenario}_{extra}_{intra}_ssam.csv")
#     df.to_csv(path, index=False)
#     print("Saved:", path)

# for dimension in set(r[0] for r in multi_results):
#     rows = [r for r in multi_results if r[0] == dimension]
#     rows.sort(key=lambda x: x[1])
#     df = pd.DataFrame({
#         "Simulation": [r[1] for r in rows],
#         "Precision": [r[2] for r in rows],
#         "Recall": [r[3] for r in rows],
#         "Accuracy": [r[4] for r in rows],
#         "F1": [r[5] for r in rows],
#     })
#     path = os.path.join(EVAL_OUT_DIR, f"multi_marker_{dimension}_all_{EXTRA_NUM_CLUSTERS}_{INTRA_NUM_CLUSTERS}_ssam.csv")
#     df.to_csv(path, index=False)
#     print("Saved:", path)