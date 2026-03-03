import os
import re
import glob
import time
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

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

# Resume: skip if ssam_spheres.csv already exists
RESUME_IF_DONE = True

# Run only a subset (e.g. 10) or None for all
LIMIT_N = None

# Restrict to 3D data only for parity with run_Baysor; SSAM run_kde supports depth for 3D
SSAM_3D_ONLY = True

PROGRESS_EVERY_N_SEEDS = 10

print("SIM_DATA_ROOT:", SIM_DATA_ROOT)
print("SSAM_OUT_ROOT:", SSAM_OUT_ROOT)
print("SSAM_3D_ONLY:", SSAM_3D_ONLY)

# ==================== Discover simulated CSVs ==================== #

def discover_simulated_data(sim_root: str) -> pd.DataFrame:
    rows = []
    pattern_single = os.path.join(sim_root, "single_marker", "*", "*", "seed_*.csv")
    for f in sorted(glob.glob(pattern_single)):
        parts = f.split(os.sep)
        dimension = parts[-3]
        name = parts[-2]
        m = re.search(r"seed_(\d+)\.csv$", parts[-1])
        if not m:
            continue
        seed = int(m.group(1))
        rows.append({
            "mode": "single_marker",
            "dimension": dimension,
            "scenario": name,
            "seed": seed,
            "is_3d": ("3D" in dimension.upper()),
            "input_csv": f,
        })

    pattern_multi = os.path.join(sim_root, "multi_marker", "*", "all", "seed_*.csv")
    for f in sorted(glob.glob(pattern_multi)):
        parts = f.split(os.sep)
        dimension = parts[-3]
        m = re.search(r"seed_(\d+)\.csv$", parts[-1])
        if not m:
            continue
        seed = int(m.group(1))
        rows.append({
            "mode": "multi_marker",
            "dimension": dimension,
            "scenario": "all",
            "seed": seed,
            "is_3d": ("3D" in dimension.upper()),
            "input_csv": f,
        })

    df = pd.DataFrame(rows)
    if df.shape[0] == 0:
        print("No simulated CSVs found. Check SIM_DATA_ROOT.")
    return df


inputs_df = discover_simulated_data(SIM_DATA_ROOT)
if SSAM_3D_ONLY:
    inputs_df = inputs_df[inputs_df["is_3d"]].copy().reset_index(drop=True)
    print("Restricted to 3D data only. Simulations to run:", inputs_df.shape[0])
else:
    print("Total simulations found:", inputs_df.shape[0])
print(inputs_df.head())

# ==================== Convert simulated CSV → SSAM input (x, y, z, gene) ==================== #

def simulated_to_ssam_df(sim_csv: str, is_3d: bool) -> pd.DataFrame:
    """Convert simulated CSV to DataFrame with x, y, gene (and z if 3D) for SSAM."""
    df = pd.read_csv(sim_csv).reset_index(drop=True)
    if "x" in df.columns and "y" in df.columns and "gene" in df.columns:
        x_col, y_col, z_col, gene_col = "x", "y", "z", "gene"
    elif "global_x" in df.columns and "global_y" in df.columns and "target" in df.columns:
        x_col, y_col, z_col, gene_col = "global_x", "global_y", "global_z", "target"
    else:
        raise ValueError(f"{sim_csv}: need (x,y,z,gene) or (global_x,global_y,global_z,target). Got: {list(df.columns)}")

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

def make_ssam_out_paths(sim_csv: str, sim_root: str, out_root: str):
    rel = os.path.relpath(sim_csv, sim_root)
    rel_no_ext = os.path.splitext(rel)[0]
    out_dir = os.path.join(out_root, rel_no_ext)
    spheres_csv = os.path.join(out_dir, "ssam_spheres.csv")
    return out_dir, spheres_csv

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
    sim_csv: str,
    out_dir: str,
    spheres_csv: str,
    is_3d: bool,
    bandwidth: float,
    sampling_distance: float,
    search_size: int,
    detection_radius: float,
    verbose: bool = False,
) -> None:
    """Run SSAM KDE + find_localmax for one simulation and save ssam_spheres.csv."""
    df = simulated_to_ssam_df(sim_csv, is_3d)
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

    spheres.to_csv(spheres_csv, index=False)
    
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
        sim_csv = row["input_csv"]
        is_3d = bool(row["is_3d"])
        out_dir, spheres_csv = make_ssam_out_paths(sim_csv, sim_root, out_root)

        if resume_if_done and os.path.exists(spheres_csv):
            n_spheres = pd.read_csv(spheres_csv).shape[0]
            logs.append({
                **row.to_dict(),
                "status": "skipped_exists",
                "out_dir": out_dir,
                "spheres_csv": spheres_csv,
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
                sim_csv=sim_csv,
                out_dir=out_dir,
                spheres_csv=spheres_csv,
                is_3d=is_3d,
                bandwidth=bandwidth,
                sampling_distance=sampling_distance,
                search_size=search_size,
                detection_radius=detection_radius,
            )
            spheres = pd.read_csv(spheres_csv)
            logs.append({
                **row.to_dict(),
                "status": "ok",
                "out_dir": out_dir,
                "spheres_csv": spheres_csv,
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
                "spheres_csv": spheres_csv,
                "n_spheres": None,
                "runtime_sec": float(time.time() - t0),
                "error": repr(e),
            })
            _maybe_print(row)

    return pd.DataFrame(logs)


# logs_df = run_all_ssam(
#     inputs_df=inputs_df,
#     sim_root=SIM_DATA_ROOT,
#     out_root=SSAM_OUT_ROOT,
#     bandwidth=DEFAULT_BANDWIDTH,
#     sampling_distance=DEFAULT_SAMPLING_DISTANCE,
#     search_size=DEFAULT_FIND_LOCALMAX_SEARCH_SIZE,
#     detection_radius=SSAM_DETECTION_RADIUS,
#     resume_if_done=RESUME_IF_DONE,
#     limit_n=LIMIT_N,
# )

# logs_df["status"].value_counts()