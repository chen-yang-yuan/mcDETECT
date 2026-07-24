"""
Concatenate per-tile detections into one sphere table per (method x sample x
param x geneset) config. Run after all detection array tasks finish.

Each transcript belongs to exactly one tile (half-open tiling), so tiles are a
clean partition -- concatenation is a simple stack, no de-duplication. Objects
straddling a tile border may be split or dropped; this edge effect is ignored by
design (see the plan / config.py).

Completeness is checked against the manifest: every occupied tile of a sample is
expected in each of its configs. Missing tile outputs (e.g. a failed/oversubscribed
array task) are reported per config -- NOT silently dropped -- so an incomplete
config can be spotted and its tasks resubmitted with --overwrite.

Output: <config_dir>/spheres.parquet  (+ spheres_summary.csv across all configs)

Run in any env with pandas/pyarrow (e.g. mcDETECT-env).

Usage:
    python concat_spheres.py                 # both methods
    python concat_spheres.py --method baysor
"""

import argparse
import re

import pandas as pd

import config as C

_TILE_RE = re.compile(r"tile_(\d+)_(\d+)\.parquet$")


def _expected_tiles(manifest: pd.DataFrame, sample: str, param: str, geneset: str):
    """Set of (tile_row, tile_col) the manifest says this config should have."""
    m = manifest[(manifest["sample"] == sample)
                 & (manifest["param"] == param)
                 & (manifest["geneset"] == geneset)]
    return {(int(r), int(c)) for r, c in zip(m["tile_row"], m["tile_col"])}


def _found_tiles(tiles_dir):
    """Map (tile_row, tile_col) -> parquet path for tiles present on disk."""
    found = {}
    for p in tiles_dir.glob("tile_*.parquet"):
        mm = _TILE_RE.search(p.name)
        if mm:
            found[(int(mm.group(1)), int(mm.group(2)))] = p
    return found


def concat_config(method: str, sample: str, param: str, geneset: str, manifest):
    tiles_dir = C.config_dir(method, sample, param, geneset) / "tiles"
    tag = C.config_tag(sample, param, geneset)
    if not tiles_dir.exists():
        print(f"[{method}/{tag}] no tiles dir ({tiles_dir}); skipping.")
        return None

    found = _found_tiles(tiles_dir)
    if not found:
        print(f"[{method}/{tag}] no tile parquets found; skipping.")
        return None

    expected = _expected_tiles(manifest, sample, param, geneset) if manifest is not None else set(found)
    missing = sorted(expected - set(found)) if expected else []

    frames = [pd.read_parquet(found[key]) for key in sorted(found)]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    out = C.spheres_path(method, sample, param, geneset)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)

    warn = ""
    if missing:
        warn = f"  !! MISSING {len(missing)} expected tiles (resubmit those tasks)"
    print(f"[{method}/{tag}] tiles found/expected={len(found)}/{len(expected) or len(found)} "
          f"-> {merged.shape[0]} spheres -> {out}{warn}")

    return {
        "method": method, "sample": sample, "param": param, "geneset": geneset,
        "n_tiles_expected": len(expected) or len(found),
        "n_tiles_found": len(found),
        "n_tiles_missing": len(missing),
        "n_spheres": int(merged.shape[0]),
        "spheres_parquet": str(out),
    }


def main():
    ap = argparse.ArgumentParser(description="Concatenate per-tile spheres per config.")
    ap.add_argument("--method", choices=["baysor", "ssam", "both"], default="both")
    args = ap.parse_args()

    methods = ["baysor", "ssam"] if args.method == "both" else [args.method]

    manifest = pd.read_csv(C.MANIFEST_PATH) if C.MANIFEST_PATH.exists() else None
    if manifest is None:
        print(f"WARNING: manifest not found ({C.MANIFEST_PATH}); "
              f"completeness cannot be checked (will stitch whatever tiles exist).")

    summary = []
    for method in methods:
        for sample in C.SAMPLES:
            for param in C.PARAMS:
                for geneset in C.GENESETS:
                    rec = concat_config(method, sample, param, geneset, manifest)
                    if rec is not None:
                        summary.append(rec)

    if summary:
        sdf = pd.DataFrame(summary)
        C.OUT_ROOT.mkdir(parents=True, exist_ok=True)
        spath = C.OUT_ROOT / "spheres_summary.csv"
        sdf.to_csv(spath, index=False)
        print("\nSummary:")
        print(sdf.to_string(index=False))
        n_incomplete = int((sdf["n_tiles_missing"] > 0).sum())
        if n_incomplete:
            print(f"\n!! {n_incomplete} config(s) are INCOMPLETE (missing tiles). "
                  f"Resubmit the missing array tasks (they skip finished tiles), then rerun concat.")
        print(f"\nWrote {spath}")


if __name__ == "__main__":
    main()
