import anndata
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Dict, List, Optional

from .utils import *


# ============================================================ Rule-based automated granule subtyping ============================================================ #


class GranuleSubtyper:
    
    """
    Automatic granule subtyping that mimics manual annotation from z-scored heatmaps.
    
    This classifier:
    1. Z-score normalizes gene expression (per gene across clusters/granules)
    2. Computes category scores based on z-scored values
    3. Assigns subtypes based on which category has highest z-score
    """
    
    def __init__(self, genes_syn_pre: Optional[List[str]] = None, genes_syn_post: Optional[List[str]] = None, genes_dendrite: Optional[List[str]] = None, genes_axon: Optional[List[str]] = None, enrichment_threshold: float = 0.35, min_zscore_threshold: float = 0.0):
        self.genes_syn_pre = genes_syn_pre
        self.genes_syn_post = genes_syn_post
        self.genes_dendrite = genes_dendrite
        self.genes_axon = genes_axon
        self.enrichment_threshold = enrichment_threshold    # minimum proportion of z-score sum to consider category enriched
        self.min_zscore_threshold = min_zscore_threshold    # minimum mean z-score for a category to be considered present (0.0 = at least average expression)
    
    
    def _compute_zscore_matrix(self, expression_matrix: np.ndarray, cluster_labels: Optional[np.ndarray] = None) -> np.ndarray:
        
        """
        Compute z-score normalized expression matrix.
        If cluster_labels provided: z-score per gene across clusters (cluster-level means); If no cluster_labels: z-score per gene across all granules (granule-level).
        
        Parameters
        ----------
        expression_matrix : np.ndarray, raw expression matrix (granules × genes).
        cluster_labels : np.ndarray, optional, cluster assignments for each granule.
            
        Returns
        -------
        np.ndarray, z-scored expression matrix.
        """
        
        if cluster_labels is not None:
            
            # Cluster-level z-scoring
            unique_clusters = np.unique(cluster_labels)
            cluster_means = np.zeros((len(unique_clusters), expression_matrix.shape[1]))
            
            for i, cluster in enumerate(unique_clusters):
                mask = cluster_labels == cluster
                cluster_means[i, :] = expression_matrix[mask, :].mean(axis=0)
            
            # Z-score across clusters (per gene)
            scaler = StandardScaler()
            zscore_means = scaler.fit_transform(cluster_means.T).T
            
            # Map back to individual granules
            zscore_matrix = np.zeros_like(expression_matrix)
            for i, cluster in enumerate(unique_clusters):
                mask = cluster_labels == cluster
                zscore_matrix[mask, :] = zscore_means[i, :]
            
            return zscore_matrix
        
        else:
            
            # Granule-level z-scoring
            scaler = StandardScaler()
            return scaler.fit_transform(expression_matrix)
    
    
    def _compute_category_scores_zscore(self, zscore_matrix: np.ndarray, gene_names: List[str], cluster_labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        
        """
        Compute category scores based on z-scored expression.
        For each category, sum the z-scores of genes in that category.
        
        Parameters
        ----------
        zscore_matrix : np.ndarray, z-scored expression matrix.
        gene_names : List[str], gene names.
        cluster_labels : np.ndarray, optional, cluster assignments.
        
        Returns
        -------
        pd.DataFrame, DataFrame with category z-score sums.
        """
        
        gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
        
        if cluster_labels is not None:
            
            # Compute cluster-level scores
            unique_clusters = np.unique(cluster_labels)
            n_clusters = len(unique_clusters)
            
            scores = {"cluster": unique_clusters,
                      "pre_zscore": np.zeros(n_clusters),
                      "post_zscore": np.zeros(n_clusters),
                      "den_zscore": np.zeros(n_clusters),
                      "axon_zscore": np.zeros(n_clusters)}
            
            for i, cluster in enumerate(unique_clusters):
                mask = cluster_labels == cluster
                cluster_zscore = zscore_matrix[mask, :].mean(axis=0)
                
                # Sum z-scores for each category
                for category, genes in [("pre", self.genes_syn_pre), ("post", self.genes_syn_post), ("den", self.genes_dendrite), ("axon", self.genes_axon)]:
                    gene_indices = [gene_to_idx[g] for g in genes if g in gene_to_idx]
                    if len(gene_indices) > 0:
                        scores[f"{category}_zscore"][i] = cluster_zscore[gene_indices].sum()    # sum of z-scores for each category
        
        else:
            
            n_granules = zscore_matrix.shape[0]                                                 # compute granule-level scores
            
            scores = {"pre_zscore": np.zeros(n_granules),
                      "post_zscore": np.zeros(n_granules),
                      "den_zscore": np.zeros(n_granules),
                      "axon_zscore": np.zeros(n_granules)}
            
            for category, genes in [("pre", self.genes_syn_pre), ("post", self.genes_syn_post), ("den", self.genes_dendrite), ("axon", self.genes_axon)]:
                gene_indices = [gene_to_idx[g] for g in genes if g in gene_to_idx]
                if len(gene_indices) > 0:
                    scores[f"{category}_zscore"] = zscore_matrix[:, gene_indices].sum(axis=1)   # sum of z-scores for each category
        
        return pd.DataFrame(scores)
    
    
    def _classify_from_zscores(self, pre_zscore: float, post_zscore: float, den_zscore: float, axon_zscore: float) -> str:
        
        """
        Classify based on z-score sums.
        
        Logic:
        1. Compute total positive z-score
        2. Compute proportion of each category
        3. Identify enriched categories (above threshold)
        4. Assign subtype based on enriched categories
        
        Parameters
        ----------
        pre_zscore, post_zscore, den_zscore, axon_zscore : float, summed z-scores for each category.
            
        Returns
        -------
        str, assigned subtype.
        """
        
        # Only consider positive z-scores
        pre_pos = max(0, pre_zscore)
        post_pos = max(0, post_zscore)
        den_pos = max(0, den_zscore)
        axon_pos = max(0, axon_zscore)
        
        total_pos = pre_pos + post_pos + den_pos + axon_pos
        
        if total_pos == 0:
            return "others"
        
        # Compute proportions
        pre_prop = pre_pos / total_pos
        post_prop = post_pos / total_pos
        den_prop = den_pos / total_pos
        axon_prop = axon_pos / total_pos
        
        # Identify enriched categories
        is_pre = pre_prop >= self.enrichment_threshold and pre_zscore >= self.min_zscore_threshold
        is_post = post_prop >= self.enrichment_threshold and post_zscore >= self.min_zscore_threshold
        is_den = den_prop >= self.enrichment_threshold and den_zscore >= self.min_zscore_threshold
        is_axon = axon_prop >= self.enrichment_threshold and axon_zscore >= self.min_zscore_threshold
        
        # Classification logic
        # Pure types
        if is_pre and not is_post and not is_den and not is_axon:
            return "pre-syn"
        elif is_post and not is_pre and not is_den and not is_axon:
            return "post-syn"
        elif is_den and not is_pre and not is_post and not is_axon:
            return "dendrites"
        elif is_axon and not is_pre and not is_post and not is_den:
            return "axons"
        
        # Two-category combinations
        elif is_pre and is_post and not is_den and not is_axon:
            return "pre & post"
        elif is_pre and is_den and not is_post and not is_axon:
            return "pre & den"
        elif is_post and is_den and not is_pre and not is_axon:
            return "post & den"
        elif is_pre and is_axon and not is_post and not is_den:
            return "pre & axon"
        elif is_post and is_axon and not is_pre and not is_den:
            return "post & axon"
        elif is_den and is_axon and not is_pre and not is_post:
            return "den & axon"
        
        # Three-category combinations
        elif is_pre and is_post and is_den and not is_axon:
            return "pre & post & den"
        elif is_pre and is_post and is_axon and not is_den:
            return "pre & post & axon"
        elif is_pre and is_den and is_axon and not is_post:
            return "pre & den & axon"
        elif is_post and is_den and is_axon and not is_pre:
            return "post & den & axon"
        
        # Four-category combination
        elif is_pre and is_post and is_den and is_axon:
            return "pre & post & den & axon"
        
        # No enriched categories or other patterns
        else:
            return "others"
    
    
    def predict(self, granule_adata: anndata.AnnData, cluster_column: Optional[str] = None, add_scores: bool = True) -> pd.Series:
        
        """
        Predict granule subtypes using z-score-based approach.
        
        Parameters
        ----------
        granule_adata : anndata.AnnData, AnnData object containing granule expression data.
        cluster_column : str, optional, column in obs containing cluster labels. If provided, classification is done at the cluster level (mimicking manual annotation of clusters). If None, classification is done at the granule level.
        add_scores : bool, default=True, whether to add z-score sums to granule_adata.obs.
            
        Returns
        -------  
        pd.Series, predicted subtypes.
        """
        
        # Extract expression matrix
        expr_matrix = granule_adata.X
        if hasattr(expr_matrix, "toarray"):
            expr_matrix = expr_matrix.toarray()
        
        gene_names = list(granule_adata.var_names)
        
        # Get cluster labels if provided
        cluster_labels = None
        if cluster_column is not None and cluster_column in granule_adata.obs.columns:
            cluster_labels = granule_adata.obs[cluster_column].values
        
        # Compute z-score matrix
        zscore_matrix = self._compute_zscore_matrix(expr_matrix, cluster_labels)
        
        # Compute category scores
        scores_df = self._compute_category_scores_zscore(zscore_matrix, gene_names, cluster_labels)
        
        # Classify
        if cluster_labels is not None:
            
            # Cluster-level classification
            cluster_subtypes = {}
            for idx, row in scores_df.iterrows():
                cluster = row["cluster"]
                subtype = self._classify_from_zscores(row["pre_zscore"],
                                                      row["post_zscore"],
                                                      row["den_zscore"],
                                                      row["axon_zscore"])
                cluster_subtypes[cluster] = subtype     # map cluster subtypes to granules
            
            # Map cluster subtypes to granules
            subtypes = [cluster_subtypes[c] for c in cluster_labels]
            
            # Add cluster-level scores to obs
            if add_scores:
                for idx, row in scores_df.iterrows():
                    cluster = row["cluster"]
                    mask = cluster_labels == cluster
                    for col in ["pre_zscore", "post_zscore", "den_zscore", "axon_zscore"]:
                        if col not in granule_adata.obs.columns:
                            granule_adata.obs[col] = 0.0
                        granule_adata.obs.loc[mask, col] = row[col]

        else:
            
            # Granule-level classification
            subtypes = []
            for idx in range(len(scores_df)):
                subtype = self._classify_from_zscores(scores_df.loc[idx, "pre_zscore"],
                                                      scores_df.loc[idx, "post_zscore"],
                                                      scores_df.loc[idx, "den_zscore"],
                                                      scores_df.loc[idx, "axon_zscore"])
                subtypes.append(subtype)                # append subtype to list
            
            # Add scores to obs
            if add_scores:
                for col in ["pre_zscore", "post_zscore", "den_zscore", "axon_zscore"]:
                    granule_adata.obs[col] = scores_df[col].values
        
        return pd.Series(subtypes, index=granule_adata.obs.index, name="granule_subtype_automated")


# Convenience function
def classify_granules(granule_adata: anndata.AnnData, cluster_column: Optional[str] = None, enrichment_threshold: float = 0.35, min_zscore_threshold: float = 0.0, custom_markers: Optional[Dict[str, List[str]]] = None) -> pd.Series:
    
    """
    Automatic granule subtyping that mimics manual annotation from z-scored heatmaps.
    
    Parameters
    ----------
    granule_adata : anndata.AnnData, AnnData object with granule expression data.
    cluster_column : str, optional, column in obs with cluster labels. If provided, classification is done at cluster level. If None, done at granule level.
    enrichment_threshold : float, default=0.35, minimum proportion of positive z-score to consider category enriched.
    min_zscore_threshold : float, default=0.0, minimum z-score for category to be considered (0 = average).
    custom_markers : Dict[str, List[str]], optional, custom marker gene sets.
        
    Returns
    -------
    pd.Series, predicted subtypes.
    """
    
    subtyper = GranuleSubtyper(genes_syn_pre=custom_markers.get("pre-syn"),
                               genes_syn_post=custom_markers.get("post-syn"),
                               genes_dendrite=custom_markers.get("dendrites"),
                               genes_axon=custom_markers.get("axons"),
                               enrichment_threshold=enrichment_threshold,
                               min_zscore_threshold=min_zscore_threshold)
    
    subtypes = subtyper.predict(granule_adata=granule_adata, cluster_column=cluster_column, add_scores=True)
    subtypes_simple = subtypes.apply(lambda s: "mixed" if " & " in str(s) else str(s))
    return subtypes, subtypes_simple


# ============================================================ Granule & neuron spatial analysis ============================================================ #


# [MAIN] anndata, spot-level neuron metadata
def spot_neuron(adata_neuron, spot, grid_len = 50, neuron_loc_key = ["global_x", "global_y"], spot_loc_key = ["global_x", "global_y"]):
    
    adata_neuron = adata_neuron.copy()
    neurons = adata_neuron.obs
    spot = spot.copy()
    
    half_len = grid_len / 2
    
    indicator, neuron_count = [], []
    
    for _, row in spot.obs.iterrows():
        
        x = row[spot_loc_key[0]]
        y = row[spot_loc_key[1]]
        neuron_temp = neurons[(neurons[neuron_loc_key[0]] > x - half_len) & (neurons[neuron_loc_key[0]] < x + half_len) & (neurons[neuron_loc_key[1]] > y - half_len) & (neurons[neuron_loc_key[1]] < y + half_len)]
        indicator.append(int(len(neuron_temp) > 0))
        neuron_count.append(len(neuron_temp))
    
    spot.obs["indicator"] = indicator
    spot.obs["neuron_count"] = neuron_count
    return spot


# [MAIN] anndata, spot-level granule metadata
def spot_granule(granule, spot, grid_len = 50, gnl_loc_key = ["sphere_x", "sphere_y"], spot_loc_key = ["global_x", "global_y"]):
    
    granule = granule.copy()
    spot = spot.copy()
    
    half_len = grid_len / 2

    indicator, granule_count, granule_radius, granule_size, granule_score = [], [], [], [], []
    
    for _, row in spot.obs.iterrows():
        
        x = row[spot_loc_key[0]]
        y = row[spot_loc_key[1]]
        gnl_temp = granule[(granule[gnl_loc_key[0]] >= x - half_len) & (granule[gnl_loc_key[0]] < x + half_len) & (granule[gnl_loc_key[1]] >= y - half_len) & (granule[gnl_loc_key[1]] < y + half_len)]
        indicator.append(int(len(gnl_temp) > 0))
        granule_count.append(len(gnl_temp))

        if len(gnl_temp) == 0:
            granule_radius.append(0)
            granule_size.append(0)
            granule_score.append(0)
        else:
            granule_radius.append(np.nanmean(gnl_temp["sphere_r"]))
            granule_size.append(np.nanmean(gnl_temp["size"]))
            granule_score.append(np.nanmean(gnl_temp["in_soma_ratio"]))
    
    spot.obs["indicator"] = indicator
    spot.obs["gnl_count"] = granule_count
    spot.obs["gnl_radius"] = granule_radius
    spot.obs["gnl_size"] = granule_size
    spot.obs["gnl_score"] = granule_score
    return spot


# [Main] anndata, neuron-granule colocalization
def neighbor_granule(adata_neuron, granule_adata, radius = 10, sigma = None, loc_key = ["global_x", "global_y"]):
    
    adata_neuron = adata_neuron.copy()
    granule_adata = granule_adata.copy()
    
    if sigma is None:
        sigma = radius / 2
    
    # neuron and granule coordinates
    neuron_coords = adata_neuron.obs[loc_key].values
    gnl_coords = granule_adata.obs[loc_key].values
    
    # make tree
    tree = make_tree(d1 = gnl_coords[:, 0], d2 = gnl_coords[:, 1])
    
    # query neighboring granules for each neuron
    neighbor_indices = tree.query_ball_point(neuron_coords, r = radius)
    
    # record count and indices
    granule_counts = np.array([len(indices) for indices in neighbor_indices])
    adata_neuron.obs["neighbor_gnl_count"] = granule_counts
    adata_neuron.uns["neighbor_gnl_indices"] = neighbor_indices
    
    # ---------- neighboring granule expression matrix ---------- #
    n_neurons, n_genes = adata_neuron.n_obs, adata_neuron.n_vars
    weighted_expr = np.zeros((n_neurons, n_genes))
    
    for i, indices in enumerate(neighbor_indices):
        if len(indices) == 0:
            continue
        distances = np.linalg.norm(gnl_coords[indices] - neuron_coords[i], axis = 1)
        weights = np.exp(- (distances ** 2) / (2 * sigma ** 2))
        weights = weights / weights.sum()
        weighted_expr[i] = np.average(granule_adata.X[indices], axis = 0, weights = weights)

    adata_neuron.obsm["weighted_gnl_expression"] = weighted_expr
    
    # ---------- neighboring granule spatial feature ---------- #
    features = []

    for i, gnl_idx in enumerate(neighbor_indices):
        
        feats = {}
        feats["n_granules"] = len(gnl_idx)

        if len(gnl_idx) == 0:
            feats.update({"mean_distance": np.nan, "std_distance": np.nan, "radius_max": np.nan, "radius_min": np.nan, "density": 0, "center_offset_norm": np.nan, "anisotropy_ratio": np.nan})
        else:
            gnl_pos = gnl_coords[gnl_idx]
            neuron_pos = neuron_coords[i]
            dists = np.linalg.norm(gnl_pos - neuron_pos, axis = 1)
            feats["mean_distance"] = dists.mean()
            feats["std_distance"] = dists.std()
            feats["radius_max"] = dists.max()
            feats["radius_min"] = dists.min()
            feats["density"] = len(gnl_idx) / (np.pi * radius ** 2)
            centroid = gnl_pos.mean(axis = 0)
            offset = centroid - neuron_pos
            feats["center_offset_norm"] = np.linalg.norm(offset)
            cov = np.cov((gnl_pos - neuron_pos).T)
            eigvals = np.linalg.eigvalsh(cov)
            if np.min(eigvals) > 0:
                feats["anisotropy_ratio"] = np.max(eigvals) / np.min(eigvals)
            else:
                feats["anisotropy_ratio"] = np.nan

        features.append(feats)
    
    spatial_df = pd.DataFrame(features, index = adata_neuron.obs_names)
    return adata_neuron, spatial_df


# [MAIN] numpy array, neuron embeddings based on neighboring granules
def neuron_embedding_one_hot(adata_neuron, granule_adata, k = 10, radius = 10, loc_key = ["global_x", "global_y"], gnl_subtype_key = "granule_subtype_kmeans", padding_value = "Others"):
    
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
    
    encoder = OneHotEncoder(categories = [unique_subtypes], sparse = False, handle_unknown = "ignore")
    encoder.fit(np.array(unique_subtypes).reshape(-1, 1))
    S = len(unique_subtypes)
    
    # k-d tree
    tree = make_tree(d1 = granule_coords[:, 0], d2 = granule_coords[:, 1])
    distances, indices = tree.query(neuron_coords, k = k, distance_upper_bound = radius)
    
    # initialize output
    n_neurons = neuron_coords.shape[0]
    embeddings = np.zeros((n_neurons, k, S), dtype = float)

    for i in range(n_neurons):
        for k in range(k):
            idx = indices[i, k]
            dist = distances[i, k]
            if idx == granule_coords.shape[0] or np.isinf(dist):
                subtype = padding_value
            else:
                subtype = granule_subtypes[idx]
            onehot = encoder.transform([[subtype]])[0]
            embeddings[i, k, :] = onehot

    return embeddings, encoder.categories_[0]


# [MAIN] numpy array, neuron embeddings based on neighboring granules
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


# ============================================================ Spot-level granule embeddings ============================================================ #


# Hard embedding with optional smoothing
def spot_embedding(
    spots,
    granule_adata,
    adata=None,
    count_matrix=None,
    spot_loc_key=("global_x", "global_y"),
    spot_width=25.0,
    spot_height=25.0,
    granule_loc_key=("global_x", "global_y"),
    granule_subtype_key="granule_subtype",
    subtype_names=("pre-syn", "post-syn", "dendrites", "mixed"),
    granule_count_layer="counts",
    cell_loc_key=("global_x", "global_y"),
    cell_id_key="cell_id",
    count_matrix_cell_id_key="cell_id",
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
    spot_granule_expression : np.ndarray, shape (n_spots, n_genes)
        Spot-by-gene aggregated raw granule count matrix, computed by summing
        `granule_adata.layers[granule_count_layer]` over granules assigned to each spot.
    spot_cell_expression : np.ndarray or None, shape (n_spots, n_genes_count)
        Spot-by-gene aggregated raw cell count matrix from `count_matrix`, computed by
        summing counts over cells assigned to each spot (via `adata.obs[cell_id_key]` mapped
        into `count_matrix[count_matrix_cell_id_key]`). Returned only when
        `include_soma_features=True` and `adata is not None` and `count_matrix is not None`;
        otherwise None.
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

        spot_granule_expression = assign_mat @ G_valid
        if sparse.issparse(spot_granule_expression):
            spot_granule_expression = spot_granule_expression.toarray()
    else:
        G_valid = np.asarray(G)[valid_mask]
        spot_granule_expression = np.zeros((n_spots, n_genes), dtype=np.float64)
        valid_spots = assigned_spot[valid_mask]
        for i, s in enumerate(valid_spots):
            spot_granule_expression[s] += G_valid[i]

    # ----------------------------
    # Optional soma features
    # ----------------------------
    soma_counts = None
    spot_cell_expression = None

    if include_soma_features:
        if adata is None:
            raise ValueError("`adata` must be provided when include_soma_features=True.")

        cx = adata.obs[cell_loc_key[0]].to_numpy(dtype=float)
        cy = adata.obs[cell_loc_key[1]].to_numpy(dtype=float)

        soma_counts = np.zeros(n_spots, dtype=float)
        cell_points = np.column_stack([cx, cy])
        candidate_lists = spot_tree.query_ball_point(cell_points, r=max_halfdiag + 1e-8)

        assigned_spot_cell = np.full(adata.n_obs, -1, dtype=int)
        for i, candidates in enumerate(candidate_lists):
            for j in candidates:
                if (x_min[j] <= cx[i] < x_max[j]) and (y_min[j] <= cy[i] < y_max[j]):
                    soma_counts[j] += 1
                    assigned_spot_cell[i] = j
                    break

        if count_matrix is None:
            raise ValueError("`count_matrix` must be provided when include_soma_features=True to compute spot_cell_expression.")

        if cell_id_key not in adata.obs.columns:
            raise ValueError(f"`adata.obs` must contain column '{cell_id_key}' to map cells into count_matrix.")
        if not isinstance(count_matrix, pd.DataFrame):
            raise TypeError("`count_matrix` must be a pandas DataFrame with a cell id column and gene columns.")
        if count_matrix_cell_id_key not in count_matrix.columns:
            raise ValueError(f"`count_matrix` must contain column '{count_matrix_cell_id_key}' to map cells by id.")

        cell_ids = adata.obs[cell_id_key].astype(str).to_numpy()
        count_cell_ids = pd.Index(count_matrix[count_matrix_cell_id_key].astype(str))
        mapped_rows = count_cell_ids.get_indexer(cell_ids)
        gene_cols = [c for c in count_matrix.columns if c != count_matrix_cell_id_key]
        if len(gene_cols) == 0:
            raise ValueError("`count_matrix` has no gene columns (only cell_id column found).")

        valid_cell_mask = (assigned_spot_cell >= 0) & (mapped_rows >= 0)
        if np.any(valid_cell_mask):
            rows = assigned_spot_cell[valid_cell_mask]
            cols = np.arange(rows.shape[0])
            assign_mat = sparse.csr_matrix(
                (np.ones(len(rows), dtype=np.float64), (rows, cols)),
                shape=(n_spots, len(rows)),
            )

            Xc = count_matrix.iloc[mapped_rows[valid_cell_mask]][gene_cols].to_numpy(dtype=np.float64, copy=False)
            spot_cell_expression = assign_mat @ Xc
            if sparse.issparse(spot_cell_expression):
                spot_cell_expression = spot_cell_expression.toarray()
        else:
            spot_cell_expression = np.zeros((n_spots, len(gene_cols)), dtype=np.float64)

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
        # granule_counts = _smooth_matrix(granule_counts)
        # spot_granule_expression = _smooth_matrix(spot_granule_expression)

        # if include_soma_features:
        #     soma_counts = _smooth_matrix(soma_counts)
        #     if spot_cell_expression is not None:
        #         spot_cell_expression = _smooth_matrix(spot_cell_expression)

    # ----------------------------
    # Outputs
    # ----------------------------
    subtype_feature_names = [f"count_{s}" for s in subtype_names]
    aux_features = {
        "granule_count": granule_counts,
    }

    if include_soma_features:
        aux_features["soma_count"] = soma_counts

    return subtype_counts, subtype_feature_names, aux_features, spot_granule_expression, spot_cell_expression


# Soft embedding
def spot_embedding_soft(
    spots,
    granule_adata,
    adata=None,
    count_matrix=None,
    spot_loc_key=("global_x", "global_y"),
    spot_width=25.0,
    spot_height=25.0,
    granule_loc_key=("global_x", "global_y"),
    granule_subtype_key="granule_subtype",
    subtype_names=("pre-syn", "post-syn", "dendrites", "mixed"),
    granule_count_layer="counts",
    cell_loc_key=("global_x", "global_y"),
    cell_id_key="cell_id",
    count_matrix_cell_id_key="cell_id",
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
    spot_granule_expression : np.ndarray, shape (n_spots, n_genes)
        Spot-by-gene kernel-weighted aggregated granule count matrix derived from
        `granule_adata.layers[granule_count_layer]`.
    spot_cell_expression : np.ndarray or None, shape (n_spots, n_genes_count)
        Spot-by-gene kernel-weighted aggregated cell count matrix from `count_matrix`,
        using cells in `adata` mapped by `cell_id`. Returned only when
        `include_soma_features=True` and `adata is not None` and `count_matrix is not None`;
        otherwise None.
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
    spot_granule_expression = np.zeros((n_spots, n_genes), dtype=float)

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
            spot_granule_expression[i] = np.asarray(weighted.sum(axis=0)).ravel()
        else:
            Gi = np.asarray(G)[nbrs]
            spot_granule_expression[i] = (w[:, None] * Gi).sum(axis=0)

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
        spot_granule_expression = np.divide(
            spot_granule_expression,
            denom,
            out=np.zeros_like(spot_granule_expression),
            where=denom > 0
        )

    soma_counts = None
    spot_cell_expression = None
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

        if count_matrix is None:
            raise ValueError("`count_matrix` must be provided when include_soma_features=True to compute spot_cell_expression.")

        if cell_id_key not in adata.obs.columns:
            raise ValueError(f"`adata.obs` must contain column '{cell_id_key}' to map cells into count_matrix.")
        if not isinstance(count_matrix, pd.DataFrame):
            raise TypeError("`count_matrix` must be a pandas DataFrame with a cell id column and gene columns.")
        if count_matrix_cell_id_key not in count_matrix.columns:
            raise ValueError(f"`count_matrix` must contain column '{count_matrix_cell_id_key}' to map cells by id.")

        cell_ids = adata.obs[cell_id_key].astype(str).to_numpy()
        count_cell_ids = pd.Index(count_matrix[count_matrix_cell_id_key].astype(str))
        mapped_rows = count_cell_ids.get_indexer(cell_ids)
        gene_cols = [c for c in count_matrix.columns if c != count_matrix_cell_id_key]
        if len(gene_cols) == 0:
            raise ValueError("`count_matrix` has no gene columns (only cell_id column found).")

        # Aggregate per spot with the same kernel weights used for soma_counts.
        spot_cell_expression = np.zeros((n_spots, len(gene_cols)), dtype=np.float64)

        for i, nbrs in enumerate(cell_neighbor_lists):
            if len(nbrs) == 0:
                continue
            nbrs = np.asarray(nbrs, dtype=int)
            mapped = mapped_rows[nbrs]
            keep = mapped >= 0
            if not np.any(keep):
                continue

            nbrs_kept = nbrs[keep]
            mapped_kept = mapped[keep]

            d = np.linalg.norm(cell_coords[nbrs_kept] - spot_coords[i], axis=1)
            w = _kernel_weight(d)

            Xi = count_matrix.iloc[mapped_kept][gene_cols].to_numpy(dtype=np.float64, copy=False)
            spot_cell_expression[i] = (w[:, None] * Xi).sum(axis=0)

    subtype_feature_names = [f"count_{s}" for s in subtype_names]
    aux_features = {
        "granule_count": granule_counts,
        "kernel_weight_sum": granule_counts,
    }
    if include_soma_features:
        aux_features["soma_count"] = soma_counts

    return subtype_counts, subtype_feature_names, aux_features, spot_granule_expression, spot_cell_expression