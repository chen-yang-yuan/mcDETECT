"""
Baysor detection on ONE (sample x param x geneset x tile) unit of the manifest.

Dispatched as a SLURM array task: the array index selects the manifest row.
Loads that tile's transcript shard, restricts to the gene set, localizes
coordinates, runs Baysor (locked default/tuned params), converts each Baysor cell
to a minimum-enclosing sphere (miniball), re-globalizes, and writes the tile's
sphere table. NO size/in-soma/NC filtering here -- that is a post-hoc step.

Run inside `baysor_env`, with Baysor either on PATH as `baysor` or pointed to by
the BAYSOR_BINARY environment variable.

Usage:
    python run_baysor_tile.py --task-id $SLURM_ARRAY_TASK_ID
    python run_baysor_tile.py --task-id 0 --manifest /path/to/manifest.csv
"""

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

import miniball
import numpy as np
import pandas as pd

import config as C
import common

BAYSOR_BINARY = os.environ.get("BAYSOR_BINARY", "baysor")


def run_baysor_cli(in_csv: str, out_dir: str, is_3d: bool,
                   min_molecules_per_cell: int, scale: float,
                   n_threads: int) -> str:
    """Run the Baysor CLI; return the path to segmentation.csv."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = str(n_threads)

    baysor_exe = os.path.expanduser((BAYSOR_BINARY or "baysor").strip() or "baysor")
    if os.sep in baysor_exe:
        baysor_exe = os.path.abspath(baysor_exe)

    cmd = [baysor_exe, "run", in_csv, "-x", "x", "-y", "y", "-g", "gene"]
    if is_3d:
        cmd += ["-z", "z"]
    cmd += ["-m", str(min_molecules_per_cell), "-s", str(scale), "-o", str(out_dir_path)]

    print("    Baysor command:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    seg_csv = out_dir_path / "segmentation.csv"
    if seg_csv.exists():
        # Baysor can return exit code 1 while still completing (progress-bar quirk
        # in non-interactive runs); presence of the output is the success signal.
        return str(seg_csv)
    raise RuntimeError(
        f"Baysor produced no segmentation.csv (exit {result.returncode}). "
        f"stderr: {result.stderr or '(empty)'}"
    )


def segmentation_to_spheres(seg_csv: str) -> pd.DataFrame:
    """Each Baysor cell -> one minimum-enclosing sphere via miniball."""
    seg = pd.read_csv(seg_csv)
    required = {"transcript_id", "x", "y", "z", "cell"}
    missing = required - set(seg.columns)
    if missing:
        raise ValueError(f"{seg_csv} missing columns: {missing}")

    seg = seg[seg["cell"] != 0]  # cell 0 == unassigned
    if "is_noise" in seg.columns:
        seg = seg[seg["is_noise"].astype(str).str.lower() != "true"]

    rows = []
    for cell_id, g in seg.groupby("cell", sort=False):
        coords = g[["x", "y", "z"]].to_numpy(dtype=float)
        n = coords.shape[0]
        if n < C.BAYSOR_MIN_POINTS_PER_SEGMENT:
            continue
        center, r2 = miniball.get_bounding_ball(coords, epsilon=C.BAYSOR_MINIBALL_EPSILON)
        rows.append({
            "sphere_x": float(center[0]),
            "sphere_y": float(center[1]),
            "sphere_z": float(center[2]),
            "sphere_r": float(np.sqrt(r2)),
            "cell_id": str(cell_id),
            "n_molecules": int(n),
        })
    return pd.DataFrame(
        rows,
        columns=["sphere_x", "sphere_y", "sphere_z", "sphere_r", "cell_id", "n_molecules"],
    )


def main():
    ap = argparse.ArgumentParser(description="Baysor detection on one manifest tile.")
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
    out_path = C.tile_path("baysor", sample, param, geneset, tile_row, tile_col)

    if out_path.exists() and not args.overwrite:
        print(f"[skip] exists: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[job {args.task_id}] baysor {sample} {param} {geneset} "
          f"tile r{tile_row} c{tile_col}", flush=True)

    t0 = time.time()
    tile = common.load_tile_transcripts(sample, tile_row, tile_col, geneset)
    print(f"    transcripts in tile: {tile.shape[0]}", flush=True)

    if tile.shape[0] == 0:
        pd.DataFrame(
            columns=["sphere_x", "sphere_y", "sphere_z", "sphere_r", "cell_id", "n_molecules"]
        ).to_parquet(out_path, index=False)
        print(f"    empty tile -> wrote 0 spheres to {out_path}")
        return

    local, offset = common.localize(tile)
    p = C.BAYSOR_PARAMS[param]

    with tempfile.TemporaryDirectory() as tmpdir:
        in_csv = os.path.join(tmpdir, "molecules.csv")
        mol = local[["x", "y", "z", "gene"]].copy()
        mol.insert(0, "transcript_id", np.arange(len(mol), dtype=int))
        mol.to_csv(in_csv, index=False)

        seg_csv = run_baysor_cli(
            in_csv=in_csv,
            out_dir=os.path.join(tmpdir, "baysor_out"),
            is_3d=C.IS_3D,
            min_molecules_per_cell=p["min_molecules_per_cell"],
            scale=p["scale"],
            n_threads=C.BAYSOR_THREADS,
        )
        spheres = segmentation_to_spheres(seg_csv)

    spheres = common.globalize_spheres(spheres, offset)
    spheres.to_parquet(out_path, index=False)
    print(f"    wrote {spheres.shape[0]} spheres to {out_path} "
          f"in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
