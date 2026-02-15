import anndata
import numpy as np
import pandas as pd
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
    
    encoder = OneHotEncoder(categories = [unique_subtypes], sparse = False, handle_unknown = "ignore")
    encoder.fit(np.array(unique_subtypes).reshape(-1, 1))
    S = len(unique_subtypes)
    
    # k-d tree
    tree = make_tree(d1 = granule_coords[:, 0], d2 = granule_coords[:, 1])
    all_neighbors = tree.query_ball_point(neuron_coords, r = radius)
    
    # initialize output
    n_neurons = neuron_coords.shape[0]
    embeddings = np.zeros((n_neurons, S), dtype = float)

    for i, neighbor_indices in enumerate(all_neighbors):
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

    return embeddings, encoder.categories_[0]