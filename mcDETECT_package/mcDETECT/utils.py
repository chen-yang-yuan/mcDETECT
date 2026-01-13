import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from rtree import index
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from shapely.geometry import Point


def find_threshold_index(cumsum_list, threshold = 0.99):
    total = cumsum_list[-1]
    for i, value in enumerate(cumsum_list):
        if value >= threshold * total:
            return i
    return None


def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i] - K))]


def make_tree(d1 = None, d2 = None, d3 = None):
    active_dimensions = [dimension for dimension in [d1, d2, d3] if dimension is not None]
    if len(active_dimensions) == 1:
        points = np.c_[active_dimensions[0].ravel()]
    elif len(active_dimensions) == 2:
        points = np.c_[active_dimensions[0].ravel(), active_dimensions[1].ravel()]
    elif len(active_dimensions) == 3:
        points = np.c_[active_dimensions[0].ravel(), active_dimensions[1].ravel(), active_dimensions[2].ravel()]
    return cKDTree(points)


def make_rtree(spheres):
    p = index.Property()
    idx = index.Index(properties = p)
    for i, sphere in enumerate(spheres.itertuples()):
        center = Point(sphere.sphere_x, sphere.sphere_y)
        bounds = (center.x - sphere.sphere_r,
                  center.y - sphere.sphere_r,
                  center.x + sphere.sphere_r,
                  center.y + sphere.sphere_r)
        idx.insert(i, bounds)
    return idx


def scale(array, max = 1):
    new_array = (array - np.min(array)) / (np.max(array) - np.min(array)) * max
    return new_array


def weighted_corr(estimated, actual, weights):
    
    estimated = np.array(estimated)
    actual = np.array(actual)
    weights = np.array(weights)

    # weighted mean
    mean_estimated = np.average(estimated, weights = weights)
    mean_actual = np.average(actual, weights = weights)

    # weighted covariance
    cov_w = np.sum(weights * (estimated - mean_estimated) * (actual - mean_actual)) / np.sum(weights)

    # weighted variances
    var_estimated = np.sum(weights * (estimated - mean_estimated) ** 2) / np.sum(weights)
    var_actual = np.sum(weights * (actual - mean_actual) ** 2) / np.sum(weights)

    # weighted correlation coefficient
    weighted_corr = cov_w / np.sqrt(var_estimated * var_actual)
    
    return weighted_corr


def weighted_spearmanr(A, B, weights):
    
    A = np.array(A)
    B = np.array(B)
    weights = np.array(weights)

    # rank the data
    R_A = rankdata(A)
    R_B = rankdata(B)

    # weighted mean
    mean_R_A_w = np.average(R_A, weights=weights)
    mean_R_B_w = np.average(R_B, weights=weights)

    # weighted covariance
    cov_w = np.sum(weights * (R_A - mean_R_A_w) * (R_B - mean_R_B_w)) / np.sum(weights)

    # weighted variances
    var_R_A_w = np.sum(weights * (R_A - mean_R_A_w)**2) / np.sum(weights)
    var_R_B_w = np.sum(weights * (R_B - mean_R_B_w)**2) / np.sum(weights)

    # weighted Spearman correlation coefficient
    weighted_spearman_corr = cov_w / np.sqrt(var_R_A_w * var_R_B_w)
    
    return weighted_spearman_corr


def assign_palette_to_adata(adata, obs_key = "granule_expr_cluster_hierarchical", self_defined = False, cmap_name = "tab10"):
    
    adata = adata.copy()
    
    if not pd.api.types.is_categorical_dtype(adata.obs[obs_key]):
        adata.obs[obs_key] = adata.obs[obs_key].astype("category")
    
    categories = adata.obs[obs_key].cat.categories
    n_categories = len(categories)
    
    if self_defined:
        cmap = plt.colormaps[cmap_name]
        color_palette = [cmap(i) for i in range(n_categories)]
    else:
        base_colors = plt.get_cmap(cmap_name).colors
        if n_categories > len(base_colors):
            color_palette = sns.color_palette(cmap_name, n_categories)
        else:
            color_palette = base_colors[:n_categories]
    
    adata.uns[f"{obs_key}_colors"] = [mcolors.to_hex(c) for c in color_palette]
    
    return adata


def p_val_to_star(p):
    if p > 0.05:
        return "ns"
    elif p > 0.01:
        return "*"
    elif p > 0.001:
        return "**"
    else:
        return "***"


def top_columns_above_threshold(row, threshold=0.5):
    sorted_row = row.sort_values(ascending=False)
    cumsum = sorted_row.cumsum()
    # Find how many top columns are needed to exceed the threshold
    n = (cumsum > threshold).idxmax()
    # Slice up to and including the index that crosses the threshold
    return sorted_row.loc[:n].index.tolist()