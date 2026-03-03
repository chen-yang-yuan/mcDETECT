"""
Shared evaluation metrics for mcDETECT, Baysor, and SSAM benchmarking.
Used by main.ipynb (mcDETECT), run_Baysor.ipynb (Baysor), and run_SSAM.py (SSAM).
"""
import numpy as np
from scipy.spatial import cKDTree


def make_tree(d1=None, d2=None, d3=None):
    """Build KD-tree from 1D/2D/3D coordinates (from model.py)."""
    active_dimensions = [dimension for dimension in [d1, d2, d3] if dimension is not None]
    if len(active_dimensions) == 1:
        points = np.c_[active_dimensions[0].ravel()]
    elif len(active_dimensions) == 2:
        points = np.c_[active_dimensions[0].ravel(), active_dimensions[1].ravel()]
    elif len(active_dimensions) == 3:
        points = np.c_[active_dimensions[0].ravel(), active_dimensions[1].ravel(), active_dimensions[2].ravel()]
    return cKDTree(points)


def calculate_metric(ground_truth_indices, matched_index):
    """Compute precision, recall, accuracy, F1 from ground truth and matched detections."""
    flattened_matches = []
    for match in matched_index:
        if isinstance(match, tuple):
            flattened_matches.extend(match)
        elif match != -1:
            flattened_matches.append(match)

    unique_matched_points = set(flattened_matches)
    true_positives = len(unique_matched_points & ground_truth_indices)
    false_positives = len([x for x in matched_index if x == -1])
    false_negatives = len(ground_truth_indices - unique_matched_points)
    total_detections = len(matched_index)
    true_matches = len([x for x in matched_index if x != -1])

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    accuracy = true_matches / total_detections if total_detections > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, accuracy, f1


def metric_main(tree, ground_truth_indices, sphere):
    """
    Match detection spheres to ground truth points and return precision, recall, accuracy, F1.
    sphere: DataFrame with columns sphere_x, sphere_y, sphere_z, sphere_r.
    """
    matched_index = []
    for k in range(sphere.shape[0]):
        idx = tree.query_ball_point(
            [sphere["sphere_x"].iloc[k], sphere["sphere_y"].iloc[k], sphere["sphere_z"].iloc[k]],
            sphere["sphere_r"].iloc[k],
        )
        if len(idx) == 0:
            matched_index.append(-1)
        elif len(idx) == 1:
            matched_index += idx
        else:
            matched_index.append(tuple(idx))
    return calculate_metric(ground_truth_indices, matched_index)