import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from mcDETECT.utils import *


def spot_embedding(
    spots,
    granule_adata,
    adata=None,
    spot_loc_key=("global_x", "global_y"),
    spot_width=25.0,
    spot_height=25.0,
    granule_loc_key=("global_x", "global_y"),
    granule_subtype_key="granule_subtype",
    subtype_names=("pre-syn", "post-syn", "dendrites", "mixed"),
    granule_count_layer="counts",
    cell_loc_key=("global_x", "global_y"),
    include_soma_features=False,
    smoothing=False,
    smoothing_radius=None,
    smoothing_k=None,
    smoothing_mode="mean",
):
    """
    Compute spot/grid-level granule embeddings using subtype counts, and additionally
    aggregate raw granule expression counts into a spot-by-gene count matrix.

    Assignment rule:
    - Each granule/cell is assigned based on its coordinate.
    - Spots are centroid-defined squares of size spot_width x spot_height.
    - If multiple candidate spots satisfy the containment check, the first valid match
      is used, matching the behavior of the original function.

    Parameters
    ----------
    spots : AnnData
        Spot/grid-level AnnData. spots.obs must contain centroid coordinates.
    granule_adata : AnnData
        Granule-level AnnData with coordinates and subtype labels in .obs.
        Raw granule expression counts must be stored in granule_adata.layers[granule_count_layer].
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
        Ordered subtype names to include in the subtype-count embedding.
    granule_count_layer : str
        Layer in granule_adata containing raw gene counts.
    cell_loc_key : tuple/list of str
        Columns in adata.obs containing cell coordinates.
    include_soma_features : bool
        If True, compute and return soma_count.
    smoothing : bool
        If True, smooth spot-level outputs across neighboring spots.
    smoothing_radius : float or None
        Radius-based smoothing on spot centroids.
    smoothing_k : int or None
        k-nearest-neighbor smoothing on spot centroids.
    smoothing_mode : {"mean", "gaussian"}
        Smoothing type.

    Returns
    -------
    subtype_counts : np.ndarray, shape (n_spots, K)
        Spot-by-subtype count matrix.
    subtype_feature_names : list of str
        Names of subtype-count columns.
    aux_features : dict
        Dictionary containing:
            - "granule_count": total granule count per spot
            - "soma_count": optional
    spot_gene_counts : np.ndarray, shape (n_spots, n_genes)
        Spot-by-gene aggregated raw count matrix.
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
    n_granules = granule_adata.n_obs

    subtype_names = list(subtype_names)
    subtype_to_idx = {s: i for i, s in enumerate(subtype_names)}
    K = len(subtype_names)

    if granule_count_layer not in granule_adata.layers:
        raise ValueError(f"Layer '{granule_count_layer}' not found in granule_adata.layers.")

    G = granule_adata.layers[granule_count_layer]
    gene_names = np.array(granule_adata.var_names)
    n_genes = granule_adata.n_vars

    granule_counts = np.zeros(n_spots, dtype=float)
    subtype_counts = np.zeros((n_spots, K), dtype=float)

    # ----------------------------
    # Assign each granule to one spot
    # ----------------------------
    spot_tree = cKDTree(np.column_stack([sx, sy]))
    max_halfdiag = np.max(np.sqrt((sw / 2.0) ** 2 + (sh / 2.0) ** 2))

    granule_points = np.column_stack([gx, gy])
    candidate_lists = spot_tree.query_ball_point(granule_points, r=max_halfdiag + 1e-8)

    assigned_spot = np.full(n_granules, -1, dtype=int)

    for i, candidates in enumerate(candidate_lists):
        for j in candidates:
            if (x_min[j] <= gx[i] < x_max[j]) and (y_min[j] <= gy[i] < y_max[j]):
                assigned_spot[i] = j
                granule_counts[j] += 1
                if gsub[i] in subtype_to_idx:
                    subtype_counts[j, subtype_to_idx[gsub[i]]] += 1
                break

    # ----------------------------
    # Aggregate granule expression
    # ----------------------------
    valid_mask = assigned_spot >= 0

    if sparse.issparse(G):
        G_valid = G[valid_mask]
        rows = assigned_spot[valid_mask]
        cols = np.arange(rows.shape[0])

        assign_mat = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols)),
            shape=(n_spots, len(rows))
        )

        spot_gene_counts = assign_mat @ G_valid
        if sparse.issparse(spot_gene_counts):
            spot_gene_counts = spot_gene_counts.toarray()
    else:
        G_valid = np.asarray(G)[valid_mask]
        spot_gene_counts = np.zeros((n_spots, n_genes), dtype=np.float64)
        valid_spots = assigned_spot[valid_mask]
        for i, s in enumerate(valid_spots):
            spot_gene_counts[s] += G_valid[i]

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
            for j in candidates:
                if (x_min[j] <= cx[i] < x_max[j]) and (y_min[j] <= cy[i] < y_max[j]):
                    soma_counts[j] += 1
                    break

    # ----------------------------
    # Optional smoothing
    # ----------------------------
    if smoothing:
        if smoothing_radius is None and smoothing_k is None:
            raise ValueError("Provide either `smoothing_radius` or `smoothing_k` when smoothing=True.")

        center_coords = np.column_stack([sx, sy])
        tree = cKDTree(center_coords)

        def _smooth_matrix(mat):
            mat = np.asarray(mat, dtype=float)
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
                        if mat.ndim == 1:
                            smoothed[i] = (wts * mat[nbrs]).sum()
                        else:
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
                        if mat.ndim == 1:
                            smoothed[i] = (wts * mat[ids]).sum()
                        else:
                            smoothed[i] = (wts[:, None] * mat[ids]).sum(axis=0)

                    else:
                        raise ValueError("smoothing_mode must be 'mean' or 'gaussian'.")

            return smoothed

        subtype_counts = _smooth_matrix(subtype_counts)
        granule_counts = _smooth_matrix(granule_counts)
        spot_gene_counts = _smooth_matrix(spot_gene_counts)

        if include_soma_features:
            soma_counts = _smooth_matrix(soma_counts)

    # ----------------------------
    # Outputs
    # ----------------------------
    subtype_feature_names = [f"count_{s}" for s in subtype_names]
    aux_features = {
        "granule_count": granule_counts,
    }

    if include_soma_features:
        aux_features["soma_count"] = soma_counts

    return subtype_counts, subtype_feature_names, aux_features, spot_gene_counts


def spot_embedding_soft(
    spots,
    granule_adata,
    adata=None,
    spot_loc_key=("global_x", "global_y"),
    spot_width=25.0,
    spot_height=25.0,
    granule_loc_key=("global_x", "global_y"),
    granule_subtype_key="granule_subtype",
    subtype_names=("pre-syn", "post-syn", "dendrites", "mixed"),
    granule_count_layer="counts",
    cell_loc_key=("global_x", "global_y"),
    include_soma_features=False,
    kernel="gaussian",
    sigma=None,
    support_radius=None,
    normalize_subtype_embedding=False,
    normalize_gene_counts=False,
):
    """
    Grid-based spot embedding using kernel-weighted neighborhoods around spot centers.

    Main idea
    ---------
    Each spot is still treated as a grid location, but granules are not assigned by
    strict box membership. Instead, every granule within a local support region around
    the spot center contributes with a distance-based kernel weight.

    Parameters
    ----------
    spots : AnnData
        Spot/grid-level AnnData.
    granule_adata : AnnData
        Granule-level AnnData with coordinates and subtype labels in .obs.
    adata : AnnData, optional
        Cell-level AnnData for optional soma features.
    spot_loc_key, granule_loc_key, cell_loc_key : tuple/list of str
        Coordinate column names in .obs.
    spot_width, spot_height : float
        Grid width/height, used to choose default kernel scale.
    granule_subtype_key : str
        Column in granule_adata.obs containing granule subtype labels.
    subtype_names : sequence of str
        Ordered subtype names for the embedding.
    granule_count_layer : str
        Layer in granule_adata containing raw gene counts.
    include_soma_features : bool
        If True, also compute kernel-weighted soma abundance around each spot center.
    kernel : {"gaussian", "exponential", "uniform"}
        Weighting kernel.
    sigma : float or None
        Kernel scale. If None, defaults to min(spot_width, spot_height)/2.
    support_radius : float or None
        Search radius for candidate granules. If None, defaults to 3*sigma for
        gaussian, 4*sigma for exponential, or max(spot_width, spot_height)/2 for uniform.
    normalize_subtype_embedding : bool
        If True, subtype embedding becomes a local composition vector (weights sum to 1).
        If False, returns weighted subtype counts.
    normalize_gene_counts : bool
        If True, gene matrix is divided by total kernel weight per spot.

    Returns
    -------
    subtype_counts : np.ndarray, shape (n_spots, K)
        Spot-by-subtype weighted count matrix or composition matrix.
    subtype_feature_names : list of str
        Feature names.
    aux_features : dict
        Contains:
            - "granule_count": total kernel-weighted granule mass per spot
            - "soma_count": optional kernel-weighted soma mass
            - "kernel_weight_sum": same as granule_count
    spot_gene_counts : np.ndarray, shape (n_spots, n_genes)
        Spot-by-gene weighted aggregated count matrix.
    """
    sx = spots.obs[spot_loc_key[0]].to_numpy(dtype=float)
    sy = spots.obs[spot_loc_key[1]].to_numpy(dtype=float)
    spot_coords = np.column_stack([sx, sy])
    n_spots = spots.n_obs

    gx = granule_adata.obs[granule_loc_key[0]].to_numpy(dtype=float)
    gy = granule_adata.obs[granule_loc_key[1]].to_numpy(dtype=float)
    granule_coords = np.column_stack([gx, gy])
    gsub = granule_adata.obs[granule_subtype_key].astype(str).to_numpy()

    subtype_names = list(subtype_names)
    subtype_to_idx = {s: i for i, s in enumerate(subtype_names)}
    K = len(subtype_names)

    if granule_count_layer not in granule_adata.layers:
        raise ValueError(f"Layer '{granule_count_layer}' not found in granule_adata.layers.")

    G = granule_adata.layers[granule_count_layer]
    n_genes = granule_adata.n_vars

    if sigma is None:
        sigma = min(float(spot_width), float(spot_height)) / 2.0

    if support_radius is None:
        if kernel == "gaussian":
            support_radius = 3.0 * sigma
        elif kernel == "exponential":
            support_radius = 4.0 * sigma
        elif kernel == "uniform":
            support_radius = max(float(spot_width), float(spot_height)) / 2.0
        else:
            raise ValueError("kernel must be one of {'gaussian', 'exponential', 'uniform'}")

    def _kernel_weight(dist):
        if kernel == "gaussian":
            return np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
        elif kernel == "exponential":
            return np.exp(-dist / sigma)
        elif kernel == "uniform":
            return np.ones_like(dist, dtype=float)
        else:
            raise ValueError("kernel must be one of {'gaussian', 'exponential', 'uniform'}")

    granule_tree = cKDTree(granule_coords)
    neighbor_lists = granule_tree.query_ball_point(spot_coords, r=support_radius)

    subtype_counts = np.zeros((n_spots, K), dtype=float)
    granule_counts = np.zeros(n_spots, dtype=float)
    spot_gene_counts = np.zeros((n_spots, n_genes), dtype=float)

    for i, nbrs in enumerate(neighbor_lists):
        if len(nbrs) == 0:
            continue

        nbrs = np.asarray(nbrs, dtype=int)
        d = np.linalg.norm(granule_coords[nbrs] - spot_coords[i], axis=1)
        w = _kernel_weight(d)
        wsum = w.sum()

        granule_counts[i] = wsum

        sub_i = gsub[nbrs]
        for j, s in enumerate(sub_i):
            if s in subtype_to_idx:
                subtype_counts[i, subtype_to_idx[s]] += w[j]

        if sparse.issparse(G):
            Gi = G[nbrs]
            weighted = Gi.multiply(w[:, None])
            spot_gene_counts[i] = np.asarray(weighted.sum(axis=0)).ravel()
        else:
            Gi = np.asarray(G)[nbrs]
            spot_gene_counts[i] = (w[:, None] * Gi).sum(axis=0)

    if normalize_subtype_embedding:
        row_sums = subtype_counts.sum(axis=1, keepdims=True)
        subtype_counts = np.divide(
            subtype_counts,
            row_sums,
            out=np.zeros_like(subtype_counts),
            where=row_sums > 0
        )

    if normalize_gene_counts:
        denom = granule_counts[:, None]
        spot_gene_counts = np.divide(
            spot_gene_counts,
            denom,
            out=np.zeros_like(spot_gene_counts),
            where=denom > 0
        )

    soma_counts = None
    if include_soma_features:
        if adata is None:
            raise ValueError("`adata` must be provided when include_soma_features=True.")

        cx = adata.obs[cell_loc_key[0]].to_numpy(dtype=float)
        cy = adata.obs[cell_loc_key[1]].to_numpy(dtype=float)
        cell_coords = np.column_stack([cx, cy])

        cell_tree = cKDTree(cell_coords)
        cell_neighbor_lists = cell_tree.query_ball_point(spot_coords, r=support_radius)

        soma_counts = np.zeros(n_spots, dtype=float)
        for i, nbrs in enumerate(cell_neighbor_lists):
            if len(nbrs) == 0:
                continue
            nbrs = np.asarray(nbrs, dtype=int)
            d = np.linalg.norm(cell_coords[nbrs] - spot_coords[i], axis=1)
            w = _kernel_weight(d)
            soma_counts[i] = w.sum()

    subtype_feature_names = [f"count_{s}" for s in subtype_names]
    aux_features = {
        "granule_count": granule_counts,
        "kernel_weight_sum": granule_counts,
    }
    if include_soma_features:
        aux_features["soma_count"] = soma_counts

    return subtype_counts, subtype_feature_names, aux_features, spot_gene_counts


def neuron_embedding_spatial_weight(adata_neuron, granule_adata, radius = 10, sigma = 10, loc_key = ["global_x", "global_y"], gnl_subtype_key = "granule_subtype_kmeans", padding_value = "Others"):
    
    adata_neuron = adata_neuron.copy()
    granule_adata = granule_adata.copy()
    
    # neuron and granule coordinates, granule subtypes
    neuron_coords = adata_neuron.obs[loc_key].to_numpy()
    granule_coords = granule_adata.obs[loc_key].to_numpy()
    granule_subtypes = granule_adata.obs[gnl_subtype_key].astype(str).to_numpy()
    
    # include padding category
    unique_subtypes = np.unique(granule_subtypes).tolist()
    if padding_value not in unique_subtypes:
        unique_subtypes.append(padding_value)
    
    encoder = OneHotEncoder(categories = [unique_subtypes], sparse_output = False, handle_unknown = "ignore")
    encoder.fit(np.array(unique_subtypes).reshape(-1, 1))
    S = len(unique_subtypes)
    
    # k-d tree
    tree = make_tree(d1 = granule_coords[:, 0], d2 = granule_coords[:, 1])
    all_neighbors = tree.query_ball_point(neuron_coords, r = radius)
    
    # initialize output
    n_neurons = neuron_coords.shape[0]
    embeddings = np.zeros((n_neurons, S), dtype = float)
    
    # NEW: store number of granules per neuron
    granule_counts = np.zeros(n_neurons, dtype=int)

    for i, neighbor_indices in enumerate(all_neighbors):
        # NEW: count neighbors
        granule_counts[i] = len(neighbor_indices)

        if not neighbor_indices:
            # no neighbors, assign to padding subtype
            embeddings[i] = encoder.transform([[padding_value]])[0]
            continue

        # get neighbor subtypes and distances
        neighbor_coords = granule_coords[neighbor_indices]
        dists = np.linalg.norm(neuron_coords[i] - neighbor_coords, axis = 1)
        weights = np.exp(- dists / sigma)

        # encode subtypes to one-hot and weight them
        subtypes = granule_subtypes[neighbor_indices]
        onehots = encoder.transform(subtypes.reshape(-1, 1))
        weighted_sum = (weights[:, np.newaxis] * onehots).sum(axis = 0)

        # normalize to make it a composition vector
        embeddings[i] = weighted_sum / weights.sum()

    # UPDATED return
    return embeddings, encoder.categories_[0], granule_counts