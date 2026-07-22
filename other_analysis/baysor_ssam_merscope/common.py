"""
Shared data helpers for the Baysor/SSAM MERSCOPE benchmark: loading a tile's
transcript shard, restricting to a gene set, and localizing / re-globalizing
coordinates so per-tile detections can be stitched back together.
"""

import pandas as pd

import config as C


def load_tile_transcripts(sample: str, tile_row: int, tile_col: int, geneset: str) -> pd.DataFrame:
    """
    Read one tile's pre-written transcript shard (all genes) and restrict to the
    requested gene set. Shards are produced once by build_manifest.py so array
    tasks avoid re-reading the whole sample table.
    """
    df = pd.read_parquet(C.shard_path(sample, tile_row, tile_col))
    genes = C.resolve_geneset(geneset)
    if genes is not None:
        df = df[df[C.GENE_COL].isin(genes)]
    return df.reset_index(drop=True)


def localize(df: pd.DataFrame):
    """
    Shift coordinates so the tile starts near the origin (SSAM allocates a grid
    from 0..max, so global coords would blow up memory). Returns (local_df, offset)
    where offset = (ox, oy, oz) is what was subtracted; add it back to sphere
    centers to return to global coordinates.
    """
    ox = float(df[C.X_COL].min())
    oy = float(df[C.Y_COL].min())
    oz = float(df[C.Z_COL].min()) if C.IS_3D else 0.0
    local = pd.DataFrame({
        "x": df[C.X_COL].to_numpy(dtype=float) - ox,
        "y": df[C.Y_COL].to_numpy(dtype=float) - oy,
        "gene": df[C.GENE_COL].astype(str).to_numpy(),
    })
    local["z"] = (df[C.Z_COL].to_numpy(dtype=float) - oz) if C.IS_3D else 0.0
    return local, (ox, oy, oz)


def globalize_spheres(spheres: pd.DataFrame, offset) -> pd.DataFrame:
    """Add the tile offset back onto sphere centers (radius is unaffected)."""
    if spheres.shape[0] == 0:
        return spheres
    ox, oy, oz = offset
    spheres = spheres.copy()
    spheres["sphere_x"] = spheres["sphere_x"] + ox
    spheres["sphere_y"] = spheres["sphere_y"] + oy
    spheres["sphere_z"] = spheres["sphere_z"] + oz
    return spheres


def read_manifest_row(manifest_path, task_id: int) -> pd.Series:
    man = pd.read_csv(manifest_path)
    hit = man[man["job_id"] == task_id]
    if hit.shape[0] != 1:
        raise ValueError(
            f"job_id {task_id} matched {hit.shape[0]} manifest rows in {manifest_path} "
            f"(expected exactly 1; manifest has {man.shape[0]} rows)."
        )
    return hit.iloc[0]
