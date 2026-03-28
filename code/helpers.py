import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree


# --- Low-res grid → sub-tiles (zeros) → transcript counts into X (same coords) -----------------
# 1) subdivide_spots(spots_raw, sub_um)     → AnnData with 4× more rows, X=0, new centroids & spot_id
# 2) construct_grid_from_spots(spots, sub_um) → histogram bin edges aligned to those centroids
# 3) fill_spot_expression_from_transcripts(spots, transcripts, grid_len=sub_um) → histogram2d (layers)
# 4) fill_spot_expression_from_transcripts_exact(...) → per-tile containment [cx±h]×[cy±h], half-open (X or layer)
# Use the same value for sub_um and grid_len (e.g. 25 when parents are 50×50 μm).


def subdivide_spots(spots_raw: "sc.AnnData", sub_um: float = 25.0) -> "sc.AnnData":
    """Split each spot into four sub-tiles (half parent edge length `sub_um`); X is all zeros."""
    half_sub = float(sub_um) / 2.0
    offsets = np.array([[-half_sub, -half_sub],
                        [half_sub, -half_sub],
                        [-half_sub, half_sub],
                        [half_sub, half_sub]])
    n = spots_raw.n_obs
    gx = spots_raw.obs["global_x"].to_numpy(dtype=float)
    gy = spots_raw.obs["global_y"].to_numpy(dtype=float)
    new_gx = np.column_stack([gx + offsets[i, 0] for i in range(4)]).ravel()
    new_gy = np.column_stack([gy + offsets[i, 1] for i in range(4)]).ravel()
    ix = np.arange(n).repeat(4)
    inherited = ["region_labels", "brain_area", "batch"]
    obs_new = spots_raw.obs.iloc[ix][inherited].reset_index(drop=True)
    obs_new["global_x"] = new_gx
    obs_new["global_y"] = new_gy
    parent_ids = spots_raw.obs["spot_id"].astype(str).to_numpy()[ix]
    quad = np.tile(np.arange(4), n)
    obs_new["spot_id"] = parent_ids + "_25um_q" + quad.astype(str)
    out = anndata.AnnData(X=np.zeros((obs_new.shape[0], spots_raw.shape[1])), obs=obs_new, var=spots_raw.var.copy())
    if spots_raw.uns:
        out.uns = dict(spots_raw.uns)
    return out


def construct_grid_from_spots(spots: "sc.AnnData", grid_len: float = 25.0, x_col: str = "global_x", y_col: str = "global_y") -> tuple[np.ndarray, np.ndarray]:
    """Edges for np.histogram2d: square tiles [cx±h]×[cy±h], h=grid_len/2, spanning all spot centroids."""
    half = float(grid_len) / 2.0
    cx = spots.obs[x_col].to_numpy(dtype=float)
    cy = spots.obs[y_col].to_numpy(dtype=float)
    x_min_edge = float(np.min(cx - half))
    x_max_edge = float(np.max(cx + half))
    y_min_edge = float(np.min(cy - half))
    y_max_edge = float(np.max(cy + half))
    x_bins = np.arange(x_min_edge, x_max_edge + grid_len, grid_len)
    y_bins = np.arange(y_min_edge, y_max_edge + grid_len, grid_len)
    return x_bins, y_bins


def fill_spot_expression_from_transcripts(
    spots: "sc.AnnData",
    transcripts: pd.DataFrame,
    grid_len: float = 25.0,
    gene_col: str = "target",
    x_col: str = "global_x",
    y_col: str = "global_y",
    batch_col: str = "batch",
    inplace: bool = True,
    layer_name: str = "all_transcripts",
) -> "sc.AnnData":
    """Count transcripts per gene into spots.X using the same bins as construct_grid_from_spots (per batch).

    Multiple spot rows can map to the same histogram bin (e.g. duplicate centroids after concat); each
    row receives the same bin count. Deduplicate obs upstream if you need one row per tile.
    """
    if not inplace:
        spots = spots.copy()
    genes = list(spots.var_names)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    cx = spots.obs[x_col].to_numpy(dtype=float)
    cy = spots.obs[y_col].to_numpy(dtype=float)
    bsp = spots.obs[batch_col].astype(str).to_numpy()
    n_obs, n_vars = spots.n_obs, spots.n_vars
    X = np.zeros((n_obs, n_vars), dtype=np.float64)

    x_bins, y_bins = construct_grid_from_spots(spots, grid_len, x_col, y_col)
    nx, ny = len(x_bins) - 1, len(y_bins) - 1
    ix_sp = np.searchsorted(x_bins, cx, side="right") - 1
    iy_sp = np.searchsorted(y_bins, cy, side="right") - 1
    ix_sp = np.clip(ix_sp, 0, nx - 1)
    iy_sp = np.clip(iy_sp, 0, ny - 1)
    batches = transcripts[batch_col].unique()
    for g in genes:
        gi = gene_to_idx[g]
        tg = transcripts[transcripts[gene_col] == g]
        if len(tg) == 0:
            continue
        for b in batches:
            t_b = tg[tg[batch_col].astype(str) == str(b)]
            if len(t_b) == 0:
                continue
            count_gene, _, _ = np.histogram2d(
                t_b[x_col].to_numpy(dtype=float),
                t_b[y_col].to_numpy(dtype=float),
                bins=[x_bins, y_bins],
            )
            rows = np.flatnonzero(bsp == str(b))
            if len(rows) == 0:
                continue
            X[rows, gi] = count_gene[ix_sp[rows], iy_sp[rows]]
    spots.layers[layer_name] = X
    return spots


def fill_spot_expression(
    spots: "sc.AnnData",
    transcripts: pd.DataFrame,
    grid_len: float = 25.0,
    gene_col: str = "target",
    x_col: str = "global_x",
    y_col: str = "global_y",
    batch_col: str = "batch",
    inplace: bool = True,
    assign_to: str = "all_transcripts",
    tx_chunk: int = 65_536,
    spot_chunk: int = 512,
) -> "sc.AnnData":
    """Assign each transcript to spot tiles by exact axis-aligned containment (half-open intervals).

    Tile for spot center (cx, cy) is [cx - h, cx + h) x [cy - h, cy + h) with h = grid_len / 2, matching
    subdivide_spots. Counts are per (gene, spot); batch must match between transcript and spot rows.

    This avoids shared histogram bins: each spot gets only transcripts inside its own square. If two tiles
    overlap, a transcript in the overlap is counted toward every spot whose tile contains it.

    Parameters
    ----------
    assign_to : str
        "X" to write ``spots.X``, or a layer name to write ``spots.layers[assign_to]``.
    tx_chunk, spot_chunk : int
        Chunk sizes to bound peak memory for the boolean overlap array (spots_chunk x transcripts_chunk).
    """
    if not inplace:
        spots = spots.copy()
    half = float(grid_len) / 2.0
    genes = list(spots.var_names)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    cx = spots.obs[x_col].to_numpy(dtype=float)
    cy = spots.obs[y_col].to_numpy(dtype=float)
    bsp = spots.obs[batch_col].astype(str).to_numpy()
    n_obs, n_vars = spots.n_obs, spots.n_vars
    X = np.zeros((n_obs, n_vars), dtype=np.float64)
    batches = transcripts[batch_col].unique()

    for idx, g in enumerate(genes):
        gi = gene_to_idx[g]
        tg = transcripts[transcripts[gene_col] == g]
        if len(tg) == 0:
            continue
        for b in batches:
            bs = str(b)
            t_b = tg[tg[batch_col].astype(str) == bs]
            if len(t_b) == 0:
                continue
            rows = np.flatnonzero(bsp == bs)
            if rows.size == 0:
                continue
            tx_all = t_b[x_col].to_numpy(dtype=float)
            ty_all = t_b[y_col].to_numpy(dtype=float)
            cx_b = cx[rows]
            cy_b = cy[rows]
            counts = np.zeros(rows.size, dtype=np.float64)
            for r0 in range(0, rows.size, spot_chunk):
                r1 = min(r0 + spot_chunk, rows.size)
                cx_c = cx_b[r0:r1, None]
                cy_c = cy_b[r0:r1, None]
                acc = np.zeros(r1 - r0, dtype=np.float64)
                for t0 in range(0, tx_all.shape[0], tx_chunk):
                    t1 = min(t0 + tx_chunk, tx_all.shape[0])
                    xc = tx_all[t0:t1][None, :]
                    yc = ty_all[t0:t1][None, :]
                    acc += np.sum(
                        (xc >= cx_c - half)
                        & (xc < cx_c + half)
                        & (yc >= cy_c - half)
                        & (yc < cy_c + half),
                        axis=1,
                    ).astype(np.float64)
                counts[r0:r1] = acc
            X[rows, gi] = counts
        if idx % 10 == 0:
            print(f"Processed {idx} out of {len(genes)} genes!")

    if assign_to == "X":
        spots.X = X
    else:
        spots.layers[assign_to] = X
    return spots


def spot_embedding_granule_composition(
    spots,
    granule_adata,
    adata=None,
    spot_loc_key=("global_x", "global_y"),
    spot_width=50.0,
    spot_height=50.0,
    spot_width_col=None,
    spot_height_col=None,
    granule_loc_key=("global_x_adjusted", "global_y_adjusted"),
    granule_subtype_key="granule_subtype",
    subtype_names=["pre-syn", "post-syn", "dendrites", "mixed"],
    cell_loc_key=("global_x", "global_y"),
    include_soma_density=False,
    smoothing=False,
    smoothing_radius=None,
    smoothing_k=None,
    smoothing_mode="mean",
):
    """
    Compute spot/grid-level embeddings from granule composition.

    Each granule is assigned to exactly one centroid-defined grid.
    The embedding for each spot is:
        [ log1p(total_granule_count), prop_subtype_1, ..., prop_subtype_K ]
    Optionally append soma density as an extra feature.
    Optionally smooth spot-level features across neighboring spots.

    Parameters
    ----------
    spots : AnnData
        Spot/grid-level AnnData. spots.obs must contain centroid coordinates.
    granule_adata : AnnData
        Granule-level AnnData with coordinates and subtype labels in .obs.
    adata : AnnData, optional
        Cell-level AnnData for soma density calculation.
    spot_loc_key : tuple/list of str
        Columns in spots.obs containing spot centroid coordinates.
    spot_width, spot_height : float
        Constant width and height of each grid square/rectangle.
    spot_width_col, spot_height_col : str or None
        Optional per-spot width/height columns in spots.obs.
    granule_loc_key : tuple/list of str
        Columns in granule_adata.obs containing granule coordinates.
    granule_subtype_key : str
        Column in granule_adata.obs containing granule subtype labels.
    cell_loc_key : tuple/list of str
        Columns in adata.obs containing cell coordinates.
    include_soma_density : bool
        If True, append soma density as an additional feature.
    smoothing : bool
        If True, smooth embeddings across nearby spots after construction.
    smoothing_radius : float or None
        Radius-based smoothing on spot centroids.
    smoothing_k : int or None
        k-nearest-neighbor smoothing on spot centroids.
    smoothing_mode : {"mean", "gaussian"}
        Smoothing type.

    Returns
    -------
    embeddings : np.ndarray
        Array of shape (n_spots, n_features).
    feature_names : list of str
        Names of embedding features.
    """
    # ----------------------------
    # Spot centroids and geometry
    # ----------------------------
    sx = spots.obs[spot_loc_key[0]].to_numpy(dtype=float)
    sy = spots.obs[spot_loc_key[1]].to_numpy(dtype=float)
    n_spots = spots.n_obs

    if spot_width_col is not None:
        sw = spots.obs[spot_width_col].to_numpy(dtype=float)
    else:
        sw = np.full(n_spots, float(spot_width))

    if spot_height_col is not None:
        sh = spots.obs[spot_height_col].to_numpy(dtype=float)
    else:
        sh = np.full(n_spots, float(spot_height))

    x_min = sx - sw / 2.0
    x_max = sx + sw / 2.0
    y_min = sy - sh / 2.0
    y_max = sy + sh / 2.0
    area = sw * sh

    # ----------------------------
    # Granule data
    # ----------------------------
    gx = granule_adata.obs[granule_loc_key[0]].to_numpy(dtype=float)
    gy = granule_adata.obs[granule_loc_key[1]].to_numpy(dtype=float)
    gsub = granule_adata.obs[granule_subtype_key].astype(str).to_numpy()

    # subtype_names = np.sort(pd.unique(gsub))
    subtype_to_idx = {s: i for i, s in enumerate(subtype_names)}
    K = len(subtype_names)

    granule_counts = np.zeros(n_spots, dtype=int)
    subtype_counts = np.zeros((n_spots, K), dtype=int)

    # ----------------------------
    # Assign each granule to one spot
    # ----------------------------
    spot_tree = cKDTree(np.column_stack([sx, sy]))
    max_halfdiag = np.max(np.sqrt((sw / 2.0) ** 2 + (sh / 2.0) ** 2))

    granule_points = np.column_stack([gx, gy])
    candidate_lists = spot_tree.query_ball_point(granule_points, r=max_halfdiag + 1e-8)

    for i, candidates in enumerate(candidate_lists):
        assigned_spot = None
        for j in candidates:
            if (x_min[j] <= gx[i] < x_max[j]) and (y_min[j] <= gy[i] < y_max[j]):
                assigned_spot = j
                break

        if assigned_spot is not None:
            granule_counts[assigned_spot] += 1
            subtype_counts[assigned_spot, subtype_to_idx[gsub[i]]] += 1

    # ----------------------------
    # Build embedding
    # ----------------------------
    subtype_props = np.zeros((n_spots, K), dtype=float)
    nonzero = granule_counts > 0
    subtype_props[nonzero] = subtype_counts[nonzero] / granule_counts[nonzero, None]

    embeddings = np.concatenate(
        [np.log1p(granule_counts).reshape(-1, 1), subtype_props],
        axis=1
    )
    feature_names = ["log1p_granule_count"] + [f"prop_{s}" for s in subtype_names]

    # ----------------------------
    # Optional soma density
    # ----------------------------
    if include_soma_density:
        if adata is None:
            raise ValueError("`adata` must be provided when include_soma_density=True.")

        cx = adata.obs[cell_loc_key[0]].to_numpy(dtype=float)
        cy = adata.obs[cell_loc_key[1]].to_numpy(dtype=float)

        cell_counts = np.zeros(n_spots, dtype=int)
        cell_points = np.column_stack([cx, cy])
        candidate_lists = spot_tree.query_ball_point(cell_points, r=max_halfdiag + 1e-8)

        for i, candidates in enumerate(candidate_lists):
            assigned_spot = None
            for j in candidates:
                if (x_min[j] <= cx[i] < x_max[j]) and (y_min[j] <= cy[i] < y_max[j]):
                    assigned_spot = j
                    break

            if assigned_spot is not None:
                cell_counts[assigned_spot] += 1

        soma_density = cell_counts / area
        embeddings = np.concatenate([embeddings, soma_density.reshape(-1, 1)], axis=1)
        feature_names.append("soma_density")

    # ----------------------------
    # Optional smoothing
    # ----------------------------
    if smoothing:
        if smoothing_radius is None and smoothing_k is None:
            raise ValueError("Provide either `smoothing_radius` or `smoothing_k` when smoothing=True.")

        center_coords = np.column_stack([sx, sy])
        tree = cKDTree(center_coords)
        smoothed = np.zeros_like(embeddings, dtype=float)

        if smoothing_radius is not None:
            nbrs_all = tree.query_ball_point(center_coords, r=smoothing_radius)

            for i, nbrs in enumerate(nbrs_all):
                nbrs = np.array(nbrs, dtype=int)
                if len(nbrs) == 0:
                    smoothed[i] = embeddings[i]
                    continue

                if smoothing_mode == "mean":
                    smoothed[i] = embeddings[nbrs].mean(axis=0)

                elif smoothing_mode == "gaussian":
                    d = np.linalg.norm(center_coords[nbrs] - center_coords[i], axis=1)
                    sigma = max(smoothing_radius / 2.0, 1e-8)
                    wts = np.exp(-(d ** 2) / (2 * sigma ** 2))
                    wts /= wts.sum()
                    smoothed[i] = (wts[:, None] * embeddings[nbrs]).sum(axis=0)

                else:
                    raise ValueError("smoothing_mode must be 'mean' or 'gaussian'.")

        else:
            dists, nbrs = tree.query(center_coords, k=min(smoothing_k, n_spots))
            if nbrs.ndim == 1:
                nbrs = nbrs[:, None]
                dists = dists[:, None]

            for i in range(n_spots):
                ids = nbrs[i]
                ds = dists[i]

                if smoothing_mode == "mean":
                    smoothed[i] = embeddings[ids].mean(axis=0)

                elif smoothing_mode == "gaussian":
                    positive_ds = ds[ds > 0]
                    sigma = np.median(positive_ds) if len(positive_ds) > 0 else 1.0
                    sigma = max(sigma, 1e-8)
                    wts = np.exp(-(ds ** 2) / (2 * sigma ** 2))
                    wts /= wts.sum()
                    smoothed[i] = (wts[:, None] * embeddings[ids]).sum(axis=0)

                else:
                    raise ValueError("smoothing_mode must be 'mean' or 'gaussian'.")

        embeddings = smoothed

    return embeddings, feature_names