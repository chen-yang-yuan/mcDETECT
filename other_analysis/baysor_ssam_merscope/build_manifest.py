"""
Build the job manifest for the Baysor/SSAM MERSCOPE benchmark.

For every sample we read the transcript table once, compute the tissue bounding
box, cut it into TILE_SIZE x TILE_SIZE um tiles, and keep only the tiles that
actually contain transcripts. Each occupied tile is crossed with every
(param x geneset) combination to produce one manifest row = one SLURM array task.

The same manifest drives BOTH methods (tiling is transcript-defined, hence
method-independent); run_baysor_tile.py and run_ssam_tile.py each resolve the
`param` label to their own locked values via config.py.

Output: <OUT_ROOT>/manifests/manifest.csv  (job_id is the array index, 0..N-1)
        <OUT_ROOT>/manifests/n_jobs.txt    (N, for the --array range)

Run this ONCE (in any env with pandas/pyarrow, e.g. mcDETECT-env) before
submitting the detection arrays. Reads real data -- run on HGCC, not locally.
"""

import numpy as np
import pandas as pd

import config as C


def enumerate_tiles(sample: str):
    """
    Read a sample's transcripts once, enumerate occupied tiles, and write each
    tile's transcript shard (all genes) so array tasks can read just their tile.
    Returns a list of occupied-tile dicts.
    """
    df = pd.read_parquet(C.transcripts_path(sample), columns=C.READ_COLS)
    x = df[C.X_COL].to_numpy(dtype=float)
    y = df[C.Y_COL].to_numpy(dtype=float)

    xmin, ymin = float(x.min()), float(y.min())
    xmax, ymax = float(x.max()), float(y.max())

    ncol = max(int(np.ceil((xmax - xmin) / C.TILE_SIZE)), 1)
    nrow = max(int(np.ceil((ymax - ymin) / C.TILE_SIZE)), 1)

    # Assign every transcript to exactly one tile (half-open); this both counts
    # occupancy and groups rows for sharding.
    col_of = np.clip(((x - xmin) / C.TILE_SIZE).astype(int), 0, ncol - 1)
    row_of = np.clip(((y - ymin) / C.TILE_SIZE).astype(int), 0, nrow - 1)
    df = df.assign(_tile_row=row_of, _tile_col=col_of)

    (C.SHARD_DIR / sample).mkdir(parents=True, exist_ok=True)

    tiles = []
    for (r, c), g in df.groupby(["_tile_row", "_tile_col"], sort=True):
        r, c = int(r), int(c)
        shard = C.shard_path(sample, r, c)
        g.drop(columns=["_tile_row", "_tile_col"]).reset_index(drop=True).to_parquet(
            shard, index=False)
        x0 = xmin + c * C.TILE_SIZE
        y0 = ymin + r * C.TILE_SIZE
        tiles.append({
            "tile_row": r,
            "tile_col": c,
            "x0": x0,
            "x1": x0 + C.TILE_SIZE,
            "y0": y0,
            "y1": y0 + C.TILE_SIZE,
            "n_tx": int(g.shape[0]),
        })
    print(f"[{sample}] bbox=({xmin:.1f},{ymin:.1f})-({xmax:.1f},{ymax:.1f}) "
          f"grid={nrow}x{ncol} occupied_tiles={len(tiles)} "
          f"shards->{C.SHARD_DIR / sample}")
    return tiles


def main():
    rows = []
    for sample in C.SAMPLES:
        tiles = enumerate_tiles(sample)
        for t in tiles:
            for param in C.PARAMS:
                for geneset in C.GENESETS:
                    rows.append({
                        "sample": sample,
                        "param": param,
                        "geneset": geneset,
                        **t,
                    })

    man = pd.DataFrame(rows)
    man.insert(0, "job_id", np.arange(len(man), dtype=int))

    C.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    man.to_csv(C.MANIFEST_PATH, index=False)
    C.N_JOBS_PATH.write_text(str(len(man)) + "\n")

    print(f"\nWrote {C.MANIFEST_PATH}  ({len(man)} jobs)")
    print(f"Wrote {C.N_JOBS_PATH}")
    print("Per-config tile counts:")
    print(man.groupby(["sample", "param", "geneset"]).size().to_string())
    print(f"\nSubmit arrays with:  --array=0-{len(man) - 1}%<concurrency>")


if __name__ == "__main__":
    main()
