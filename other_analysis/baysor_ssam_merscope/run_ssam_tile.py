"""
SSAM detection on ONE (sample x param x geneset x tile) unit of the manifest.

Dispatched as a SLURM array task: the array index selects the manifest row.
Loads the sample's transcripts, restricts to the gene set and the tile, localizes
coordinates (SSAM builds a KDE grid from 0..max, so tiles must be origin-shifted),
runs SSAM's KDE + local-maxima detection (locked default/tuned thresholds), turns
each local maximum into a fixed-radius sphere, re-globalizes, and writes the tile's
sphere table. NO in-soma/NC filtering here -- that is a post-hoc step.

Run inside `ssam_hpc` (ssam==1.1.3).

Usage:
    python run_ssam_tile.py --task-id $SLURM_ARRAY_TASK_ID
    python run_ssam_tile.py --task-id 0 --manifest /path/to/manifest.csv
"""

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import ssam

import config as C
import common


def localmax_to_spheres(dataset, detection_radius: float, is_3d: bool) -> pd.DataFrame:
    """SSAM local maxima (grid indices) -> fixed-radius spheres in physical coords."""
    lm = dataset.local_maxs
    if lm is None or len(lm[0]) == 0:
        return pd.DataFrame(columns=["sphere_x", "sphere_y", "sphere_z", "sphere_r"])

    sampling = float(dataset.zarr_group["vf_params"][0])
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
        "sphere_r": float(detection_radius),
    })


def run_ssam_tile(local: pd.DataFrame, is_3d: bool,
                  expression_threshold: float, norm_threshold: float) -> pd.DataFrame:
    """KDE + find_localmax on one localized tile; returns a sphere table."""
    width = float(local["x"].max())
    height = float(local["y"].max())
    depth = (float(local["z"].max()) + 1.0) if is_3d else 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        ds = ssam.SSAMDataset(tmpdir)
        analysis = ssam.SSAMAnalysis(ds, verbose=False)
        analysis.run_kde(
            local[["x", "y", "z", "gene"]] if is_3d else local[["x", "y", "gene"]],
            width=width, height=height, depth=depth,
            bandwidth=C.SSAM_BANDWIDTH,
            sampling_distance=C.SSAM_SAMPLING_DISTANCE,
        )
        # default arm uses thresholds of 0/0 (no removal); tuned uses the VISp preset.
        analysis.set_thresholds(
            expression_threshold=expression_threshold,
            norm_threshold=norm_threshold,
        )
        analysis.find_localmax(search_size=C.SSAM_SEARCH_SIZE)
        return localmax_to_spheres(ds, C.SSAM_DETECTION_RADIUS, is_3d)


def main():
    ap = argparse.ArgumentParser(description="SSAM detection on one manifest tile.")
    ap.add_argument("--task-id", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")),
                    help="Manifest job_id (defaults to $SLURM_ARRAY_TASK_ID).")
    ap.add_argument("--manifest", type=str, default=str(C.MANIFEST_PATH))
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if the tile output already exists.")
    args = ap.parse_args()

    if args.task_id < 0:
        raise SystemExit("No --task-id and no $SLURM_ARRAY_TASK_ID set.")

    row = common.read_manifest_row(args.manifest, args.task_id)
    sample, param, geneset = row["sample"], row["param"], row["geneset"]
    tile_row, tile_col = int(row["tile_row"]), int(row["tile_col"])
    out_path = C.tile_path("ssam", sample, param, geneset, tile_row, tile_col)

    if out_path.exists() and not args.overwrite:
        print(f"[skip] exists: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[job {args.task_id}] ssam {sample} {param} {geneset} "
          f"tile r{tile_row} c{tile_col}", flush=True)

    t0 = time.time()
    tile = common.load_tile_transcripts(sample, tile_row, tile_col, geneset)
    print(f"    transcripts in tile: {tile.shape[0]}", flush=True)

    if tile.shape[0] == 0:
        pd.DataFrame(columns=["sphere_x", "sphere_y", "sphere_z", "sphere_r"]).to_parquet(
            out_path, index=False)
        print(f"    empty tile -> wrote 0 spheres to {out_path}")
        return

    local, offset = common.localize(tile)
    p = C.SSAM_PARAMS[param]

    spheres = run_ssam_tile(
        local, is_3d=C.IS_3D,
        expression_threshold=p["expression_threshold"],
        norm_threshold=p["norm_threshold"],
    )
    spheres = common.globalize_spheres(spheres, offset)
    spheres.to_parquet(out_path, index=False)
    print(f"    wrote {spheres.shape[0]} spheres to {out_path} "
          f"in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
