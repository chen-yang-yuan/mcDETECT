"""
Shared evaluation metrics for mcDETECT, Baysor, and SSAM benchmarking.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def make_tree(d1=None, d2=None, d3=None):
    active_dimensions = [dimension for dimension in [d1, d2, d3] if dimension is not None]
    if len(active_dimensions) == 1:
        points = np.c_[active_dimensions[0].ravel()]
    elif len(active_dimensions) == 2:
        points = np.c_[active_dimensions[0].ravel(), active_dimensions[1].ravel()]
    elif len(active_dimensions) == 3:
        points = np.c_[active_dimensions[0].ravel(), active_dimensions[1].ravel(), active_dimensions[2].ravel()]
    else:
        raise ValueError("At least one coordinate array (d1, d2, or d3) must be non-None.")
    return cKDTree(points)


# ==================== Object-level metrics ==================== #

def compute_object_level_metrics(
    transcripts,
    spheres,
    tau_c=0.9,
    tau_p=0.9,
    type_col="type",
    granule_col="granule_id",
    x_col="x",
    y_col="y",
    z_col="z",
    extranuclear_label="Extranuclear",
    return_crosstab: bool = True,
):
    """
    Object-level precision/recall/F1 using purity + completeness + one-to-one matching.

    This implementation assumes:
      - Each transcript row has:
          * `type_col` (e.g. "CSR", "Extranuclear", "Intranuclear")
          * `granule_col` where:
              -1  -> CSR / background
              >=0 -> ID of an aggregate this transcript belongs to
      - Ground-truth *objects* are extranuclear aggregates:
          G_k = set of transcripts with type == extranuclear_label
                                      and granule_col == k (k != -1)
      - Each detection sphere D_j encloses a set of transcripts defined
        geometrically by (x_col, y_col, z_col) within radius sphere_r.

    Steps:
      1. Build KD-tree on all transcripts (for fast sphere -> transcripts lookup).
      2. Build ground-truth aggregates G_k (one set per granule_id).
      3. For each sphere j, build D_j as the set of transcript indices within its radius.
      4. For each (k, j), compute:
           completeness c_{k,j} = |G_k ∩ D_j| / |G_k|
           purity       p_{k,j} = |G_k ∩ D_j| / |D_j|
         Keep only pairs with c_{k,j} >= tau_c and p_{k,j} >= tau_p.
         Define pair score S_{k,j} = 2 * |G_k ∩ D_j| / (|G_k| + |D_j|).
      5. Run maximum-weight bipartite matching on S_{k,j} (Hungarian algorithm).
      6. Let TP be the number of matched pairs with S_{k,j} > 0.
         Then:
           FN = #GT_aggregates - TP
           FP = #detections    - TP
         Precision = TP / (TP + FP)
         Recall    = TP / (TP + FN)
         F1        = harmonic mean of Precision and Recall.

    Parameters
    ----------
    transcripts : pandas.DataFrame
        Simulated (or real) transcripts with at least the columns specified
        by `type_col`, `granule_col`, `x_col`, `y_col`, `z_col`.
    spheres : pandas.DataFrame
        Detection results with columns:
          'sphere_x', 'sphere_y', 'sphere_z', 'sphere_r'.
    tau_c : float, optional
        Completeness threshold (default 0.9).
    tau_p : float, optional
        Purity threshold (default 0.9).
    type_col : str, optional
        Column name holding CSR/Extranuclear/Intranuclear labels.
    granule_col : str, optional
        Column name holding aggregate IDs; -1 marks CSR/background.
    x_col, y_col, z_col : str, optional
        Coordinate column names (default to simulation's global_* columns).
    extranuclear_label : str, optional
        Value in `type_col` that denotes extranuclear transcripts.

    Returns
    -------
    metrics : dict
        {
          "precision": float,
          "recall": float,
          "f1": float,
          "tp": int,
          "fp": int,
          "fn": int,
          "num_gt_objects": int,
          "num_detections": int,
        }

    crosstab : pandas.DataFrame, optional
        Only returned when `return_crosstab=True`. A contingency table of
        transcript counts with shape (num_detections + 1, num_gt_objects + 1),
        where:
          - rows 0..J-1 are detection spheres (sphere_0, sphere_1, ...)
          - last row 'no_sphere' aggregates transcripts not inside any sphere
          - columns 0..K-1 are GT aggregates (gt_<granule_id>)
          - last column 'no_gt' aggregates transcripts not belonging to any
            GT aggregate (e.g. CSR, intranuclear, or filtered-out clusters).
    """
    if transcripts.empty or spheres.empty:
        metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": int(spheres.shape[0]),
            "fn": 0,
            "num_gt_objects": 0,
            "num_detections": int(spheres.shape[0]),
        }
        if return_crosstab:
            return metrics, None
        return metrics

    required_cols = {type_col, granule_col, x_col, y_col, z_col}
    missing = required_cols - set(transcripts.columns)
    if missing:
        raise ValueError(f"transcripts is missing required columns: {sorted(missing)}")

    # Work on a copy with a clean RangeIndex so that integer indices
    # (0..N-1) can be used consistently for both GT aggregates and spheres.
    df = transcripts.reset_index(drop=True).copy()

    # Ground-truth extranuclear aggregates: group by granule_col (excluding -1).
    #
    # Be defensive about dtype here: when concatenating DataFrames where some parts
    # lack `granule_col`, pandas will create it and fill with NaN, which would crash
    # on astype(int). We treat NaN as "not a GT aggregate" (same as -1).
    granule_series = pd.to_numeric(df[granule_col], errors="coerce")
    mask_gt = (df[type_col] == extranuclear_label) & granule_series.notna() & (granule_series != -1)
    df_gt = pd.DataFrame({granule_col: granule_series[mask_gt]})

    if df_gt.empty:
        # No ground-truth extranuclear aggregates; all detections are FPs.
        num_det = int(spheres.shape[0])
        metrics = {
            "precision": 0.0 if num_det > 0 else 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": num_det,
            "fn": 0,
            "num_gt_objects": 0,
            "num_detections": num_det,
        }
        if return_crosstab:
            return metrics, None
        return metrics

    # Map each aggregate ID to the set of transcript indices (0..N-1) it contains.
    granule_ids = df_gt[granule_col].astype(int).to_numpy()
    gt_indices = df_gt.index.to_numpy()

    gt_agg_dict: dict[int, set[int]] = {}
    for idx, gid in zip(gt_indices, granule_ids):
        if gid not in gt_agg_dict:
            gt_agg_dict[gid] = set()
        gt_agg_dict[gid].add(int(idx))

    gt_ids = sorted(gt_agg_dict.keys())
    gt_sets = [gt_agg_dict[k] for k in gt_ids]
    num_gt = len(gt_sets)

    # Build KD-tree on all transcripts for detection -> transcripts lookup.
    coords = np.column_stack(
        [
            df[x_col].astype(float).to_numpy(),
            df[y_col].astype(float).to_numpy(),
            df[z_col].astype(float).to_numpy(),
        ]
    )
    tree_all = cKDTree(coords)

    # Build detection sets D_j (sets of transcript indices within each sphere).
    det_sets = []
    for _, row in spheres.iterrows():
        center = [float(row["sphere_x"]), float(row["sphere_y"]), float(row["sphere_z"])]
        radius = float(row["sphere_r"])
        idx = tree_all.query_ball_point(center, radius)
        det_sets.append(set(int(i) for i in idx))

    num_det = len(det_sets)

    if num_det == 0:
        # No detections: all GT objects are missed (FN).
        fn = num_gt
        tp = 0
        fp = 0
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_gt_objects": num_gt,
            "num_detections": num_det,
        }
        if return_crosstab:
            return metrics, None
        return metrics

    # Optional cross-tabulation of raw overlaps between detections and GT aggregates.
    xtab = None
    if return_crosstab:
        all_indices = set(range(df.shape[0]))
        gt_union = set().union(*gt_sets) if gt_sets else set()
        bg_gt = all_indices - gt_union
        det_union = set().union(*det_sets) if det_sets else set()
        bg_det = all_indices - det_union

        table = np.zeros((num_det + 1, num_gt + 1), dtype=int)

    # Build weight matrix S_{k,j} with purity/completeness filtering.
    weights = np.zeros((num_gt, num_det), dtype=float)
    for k, G_k in enumerate(gt_sets):
        size_G = float(len(G_k))
        if size_G == 0:
            continue
        for j, D_j in enumerate(det_sets):
            size_D = float(len(D_j))
            if size_D == 0:
                continue
            inter_size = float(len(G_k & D_j))
            if return_crosstab:
                # Raw intersection count for cross-tab (no thresholding).
                table[j, k] = int(inter_size)
            if inter_size == 0.0:
                continue
            c_kj = inter_size / size_G
            p_kj = inter_size / size_D
            if c_kj < tau_c or p_kj < tau_p:
                continue
            # F1-like overlap score for matching weight.
            S_kj = 2.0 * inter_size / (size_G + size_D)
            weights[k, j] = S_kj

    if return_crosstab:
        # Complete background row/column.
        for j, D_j in enumerate(det_sets):
            table[j, num_gt] = len(D_j & bg_gt)
        for k, G_k in enumerate(gt_sets):
            table[num_det, k] = len(bg_det & G_k)
        table[num_det, num_gt] = len(bg_det & bg_gt)

        row_labels = [f"sphere_{j}" for j in range(num_det)] + ["no_sphere"]
        col_labels = [f"gt_{gid}" for gid in gt_ids] + ["no_gt"]
        xtab = pd.DataFrame(table, index=row_labels, columns=col_labels)

    # If no eligible pairs, all GT objects are FN and all detections are FP.
    if not np.any(weights > 0):
        tp = 0
        fn = num_gt
        fp = num_det
        precision = 0.0 if (tp + fp) > 0 else 0.0
        recall = 0.0 if (tp + fn) > 0 else 0.0
        f1 = 0.0
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_gt_objects": num_gt,
            "num_detections": num_det,
        }
        if return_crosstab:
            return metrics, xtab
        return metrics

    # Maximum-weight bipartite matching via Hungarian algorithm.
    max_w = float(weights.max())
    cost = max_w - weights  # linear_sum_assignment minimizes cost
    row_ind, col_ind = linear_sum_assignment(cost)

    # Count only pairs with strictly positive weight as true matches.
    tp_pairs = [(r, c) for r, c in zip(row_ind, col_ind) if weights[r, c] > 0.0]
    tp = len(tp_pairs)
    fn = num_gt - tp
    fp = num_det - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_gt_objects": num_gt,
        "num_detections": num_det,
    }

    if return_crosstab:
        return metrics, xtab
    return metrics


# ==================== Legacy point-level metrics ==================== #

def calculate_metric(ground_truth_indices, matched_index):
    """
    Legacy point-level metric.

    - `ground_truth_indices`: set of point indices considered true objects.
    - `matched_index`: for each detection, either:
        * -1 (no GT point hit)
        * a single GT index
        * a tuple of GT indices

    Returns (precision, recall, accuracy, F1) using the original definition.
    """
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
    Legacy point-level metric.

    Match detection spheres to ground truth points and return precision,
    recall, accuracy, F1 under the "≥1 GT point per sphere" rule.

    Parameters
    ----------
    tree : cKDTree
        KD-tree built on ground truth point coordinates.
    ground_truth_indices : set
        Indices of ground-truth points in the KD-tree.
    sphere : pandas.DataFrame
        Columns: 'sphere_x', 'sphere_y', 'sphere_z', 'sphere_r'.
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