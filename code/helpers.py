import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def spot_embedding_granule_subtype_counts(
    spots,
    granule_adata,
    adata=None,
    spot_loc_key=("global_x", "global_y"),
    spot_width=25.0,
    spot_height=25.0,
    granule_loc_key=("global_x", "global_y"),
    granule_subtype_key="granule_subtype",
    subtype_names=("pre-syn", "post-syn", "dendrites", "mixed"),
    cell_loc_key=("global_x", "global_y"),
    include_soma_features=False,
    smoothing=False,
    smoothing_radius=None,
    smoothing_k=None,
    smoothing_mode="mean",
    return_dataframe=False,
):
    """
    Compute spot/grid-level granule embeddings using subtype counts only.

    Each granule is assigned to exactly one centroid-defined spot.
    The returned embedding matrix contains only subtype counts per spot:
        [count_subtype_1, ..., count_subtype_K]

    Total granule count is returned separately and is not included in the embedding.
    Optionally, soma count and soma density are also returned separately.
    Optionally, local smoothing is applied at the spot-feature level.

    Parameters
    ----------
    spots : AnnData
        Spot/grid-level AnnData. spots.obs must contain centroid coordinates.
    granule_adata : AnnData
        Granule-level AnnData with coordinates and subtype labels in .obs.
    adata : AnnData, optional
        Cell-level AnnData for soma features.
    spot_loc_key : tuple/list of str
        Columns in spots.obs containing spot centroid coordinates.
    spot_width, spot_height : float
        Constant width and height of each spot.
    granule_loc_key : tuple/list of str
        Columns in granule_adata.obs containing granule coordinates.
    granule_subtype_key : str
        Column in granule_adata.obs containing granule subtype labels.
    subtype_names : sequence of str
        Ordered subtype names to include in the embedding.
    cell_loc_key : tuple/list of str
        Columns in adata.obs containing cell coordinates.
    include_soma_features : bool
        If True, compute and return soma_count.
    smoothing : bool
        If True, smooth spot-level features across neighboring spots.
    smoothing_radius : float or None
        Radius-based smoothing on spot centroids.
    smoothing_k : int or None
        k-nearest-neighbor smoothing on spot centroids.
    smoothing_mode : {"mean", "gaussian"}
        Smoothing type.
    return_dataframe : bool
        If True, also return pandas DataFrames/Series.

    Returns
    -------
    subtype_count_embedding : np.ndarray, shape (n_spots, K)
        Subtype count matrix for clustering/transformation downstream.
    subtype_feature_names : list of str
        Names of subtype count columns.
    aux_features : dict
        Dictionary containing:
            - "granule_count": total granule count per spot
            - "soma_count": optional
    optionally DataFrame objects if return_dataframe=True
    """
    # ----------------------------
    # Spot centroids and geometry
    # ----------------------------
    sx = spots.obs[spot_loc_key[0]].to_numpy(dtype=float)
    sy = spots.obs[spot_loc_key[1]].to_numpy(dtype=float)
    n_spots = spots.n_obs

    sw = np.full(n_spots, float(spot_width))
    sh = np.full(n_spots, float(spot_height))

    x_min = sx - sw / 2.0
    x_max = sx + sw / 2.0
    y_min = sy - sh / 2.0
    y_max = sy + sh / 2.0

    # ----------------------------
    # Granule data
    # ----------------------------
    gx = granule_adata.obs[granule_loc_key[0]].to_numpy(dtype=float)
    gy = granule_adata.obs[granule_loc_key[1]].to_numpy(dtype=float)
    gsub = granule_adata.obs[granule_subtype_key].astype(str).to_numpy()

    subtype_names = list(subtype_names)
    subtype_to_idx = {s: i for i, s in enumerate(subtype_names)}
    K = len(subtype_names)

    granule_counts = np.zeros(n_spots, dtype=float)
    subtype_counts = np.zeros((n_spots, K), dtype=float)

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
            if gsub[i] in subtype_to_idx:
                subtype_counts[assigned_spot, subtype_to_idx[gsub[i]]] += 1

    # ----------------------------
    # Optional soma features
    # ----------------------------
    soma_counts = None

    if include_soma_features:
        if adata is None:
            raise ValueError("`adata` must be provided when include_soma_features=True.")

        cx = adata.obs[cell_loc_key[0]].to_numpy(dtype=float)
        cy = adata.obs[cell_loc_key[1]].to_numpy(dtype=float)

        soma_counts = np.zeros(n_spots, dtype=float)
        cell_points = np.column_stack([cx, cy])
        candidate_lists = spot_tree.query_ball_point(cell_points, r=max_halfdiag + 1e-8)

        for i, candidates in enumerate(candidate_lists):
            assigned_spot = None
            for j in candidates:
                if (x_min[j] <= cx[i] < x_max[j]) and (y_min[j] <= cy[i] < y_max[j]):
                    assigned_spot = j
                    break

            if assigned_spot is not None:
                soma_counts[assigned_spot] += 1

    # ----------------------------
    # Optional smoothing
    # ----------------------------
    if smoothing:
        if smoothing_radius is None and smoothing_k is None:
            raise ValueError("Provide either `smoothing_radius` or `smoothing_k` when smoothing=True.")

        center_coords = np.column_stack([sx, sy])
        tree = cKDTree(center_coords)

        def _smooth_matrix(mat):
            smoothed = np.zeros_like(mat, dtype=float)

            if smoothing_radius is not None:
                nbrs_all = tree.query_ball_point(center_coords, r=smoothing_radius)

                for i, nbrs in enumerate(nbrs_all):
                    nbrs = np.array(nbrs, dtype=int)
                    if len(nbrs) == 0:
                        smoothed[i] = mat[i]
                        continue

                    if smoothing_mode == "mean":
                        smoothed[i] = mat[nbrs].mean(axis=0)

                    elif smoothing_mode == "gaussian":
                        d = np.linalg.norm(center_coords[nbrs] - center_coords[i], axis=1)
                        sigma = max(smoothing_radius / 2.0, 1e-8)
                        wts = np.exp(-(d ** 2) / (2 * sigma ** 2))
                        wts /= wts.sum()
                        smoothed[i] = (wts[:, None] * mat[nbrs]).sum(axis=0)

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
                        smoothed[i] = mat[ids].mean(axis=0)

                    elif smoothing_mode == "gaussian":
                        positive_ds = ds[ds > 0]
                        sigma = np.median(positive_ds) if len(positive_ds) > 0 else 1.0
                        sigma = max(sigma, 1e-8)
                        wts = np.exp(-(ds ** 2) / (2 * sigma ** 2))
                        wts /= wts.sum()
                        smoothed[i] = (wts[:, None] * mat[ids]).sum(axis=0)

                    else:
                        raise ValueError("smoothing_mode must be 'mean' or 'gaussian'.")

            return smoothed

        subtype_counts = _smooth_matrix(subtype_counts)
        granule_counts = _smooth_matrix(granule_counts.reshape(-1, 1)).ravel()

        if include_soma_features:
            soma_counts = _smooth_matrix(soma_counts.reshape(-1, 1)).ravel()

    # ----------------------------
    # Outputs
    # ----------------------------
    subtype_feature_names = [f"count_{s}" for s in subtype_names]
    aux_features = {
        "granule_count": granule_counts,
    }

    if include_soma_features:
        aux_features["soma_count"] = soma_counts

    if return_dataframe:
        embedding_df = pd.DataFrame(
            subtype_counts,
            index=spots.obs_names,
            columns=subtype_feature_names
        )
        aux_df = pd.DataFrame(
            aux_features,
            index=spots.obs_names
        )
        return subtype_counts, subtype_feature_names, aux_features, embedding_df, aux_df

    return subtype_counts, subtype_feature_names, aux_features