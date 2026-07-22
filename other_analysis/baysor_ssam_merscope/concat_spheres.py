"""
Concatenate per-tile detections into one sphere table per (method x sample x
param x geneset) config. Run after all detection array tasks finish.

Each transcript belongs to exactly one tile (half-open tiling), so tiles are a
clean partition -- concatenation is a simple stack, no de-duplication. Objects
straddling a tile border may be split or dropped; this edge effect is ignored by
design (see the plan / config.py).

Output: <config_dir>/spheres.parquet  (+ a summary CSV across all configs)

Run in any env with pandas/pyarrow (e.g. mcDETECT-env).

Usage:
    python concat_spheres.py                 # both methods
    python concat_spheres.py --method baysor
"""

import argparse

import pandas as pd

import config as C


def concat_config(method: str, sample: str, param: str, geneset: str):
    tiles_dir = C.config_dir(method, sample, param, geneset) / "tiles"
    tag = C.config_tag(sample, param, geneset)
    if not tiles_dir.exists():
        print(f"[{method}/{tag}] no tiles dir ({tiles_dir}); skipping.")
        return None

    parts = sorted(tiles_dir.glob("tile_*.parquet"))
    if not parts:
        print(f"[{method}/{tag}] no tile parquets found; skipping.")
        return None

    frames = [pd.read_parquet(p) for p in parts]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    out = C.spheres_path(method, sample, param, geneset)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    print(f"[{method}/{tag}] {len(parts)} tiles -> {merged.shape[0]} spheres -> {out}")

    return {
        "method": method, "sample": sample, "param": param, "geneset": geneset,
        "n_tiles": len(parts), "n_spheres": int(merged.shape[0]),
        "spheres_parquet": str(out),
    }


def main():
    ap = argparse.ArgumentParser(description="Concatenate per-tile spheres per config.")
    ap.add_argument("--method", choices=["baysor", "ssam", "both"], default="both")
    args = ap.parse_args()

    methods = ["baysor", "ssam"] if args.method == "both" else [args.method]

    summary = []
    for method in methods:
        for sample in C.SAMPLES:
            for param in C.PARAMS:
                for geneset in C.GENESETS:
                    rec = concat_config(method, sample, param, geneset)
                    if rec is not None:
                        summary.append(rec)

    if summary:
        sdf = pd.DataFrame(summary)
        C.OUT_ROOT.mkdir(parents=True, exist_ok=True)
        spath = C.OUT_ROOT / "spheres_summary.csv"
        sdf.to_csv(spath, index=False)
        print("\nSummary:")
        print(sdf.to_string(index=False))
        print(f"\nWrote {spath}")


if __name__ == "__main__":
    main()
