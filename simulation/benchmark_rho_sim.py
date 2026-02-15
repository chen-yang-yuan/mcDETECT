import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, f1_score

from model import *
from simulate import *

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

# ==================== Function for calculating precision, recall, accuracy, F1 score ==================== #

def calculate_metric(ground_truth_indices, matched_index):
    
    flattened_matches = []
    for match in matched_index:
        if isinstance(match, tuple):
            flattened_matches.extend(match)
        elif match != -1:
            flattened_matches.append(match)

    # 1. True Positives (TP): Unique ground truth points correctly detected
    unique_matched_points = set(flattened_matches)
    true_positives = len(unique_matched_points & ground_truth_indices)

    # 2. False Positives (FP): Detections that didn"t match any ground truth
    false_positives = len([x for x in matched_index if x == -1])

    # 3. False Negatives (FN): Ground truth points that were never matched
    false_negatives = len(ground_truth_indices - unique_matched_points)

    # 4. Total ground truth points (used for recall)
    total_ground_truth_points = len(ground_truth_indices)

    # 5. Total detections (used for accuracy)
    total_detections = len(matched_index)

    # 6. Precision
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    # 7. Recall
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0

    # 8. Revised Accuracy
    true_matches = len([x for x in matched_index if x != -1])  # Count of detections correctly matched
    if total_detections > 0:
        accuracy = true_matches / total_detections
    else:
        accuracy = 0.0
    
    # 9. F1 Score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return precision, recall, accuracy, f1

# Main metric calculation function
def metric_main(tree, ground_truth_indices, sphere):
    matched_index = []
    for k in range(sphere.shape[0]):
        idx = tree.query_ball_point([sphere["sphere_x"].iloc[k], sphere["sphere_y"].iloc[k], sphere["sphere_z"].iloc[k]], sphere["sphere_r"].iloc[k])
        if len(idx) == 0:
            matched_index.append(-1)
        elif len(idx) == 1:
            matched_index += idx
        elif len(idx) > 1:
            matched_index.append(tuple(idx))
    return calculate_metric(ground_truth_indices, matched_index)

# Helper function for computing average detections per ground truth
def compute_avg_detections_per_GT(sphere_all, parents_all):
    detections_per_GT = []
    for _, gt_row in parents_all.iterrows():
        gx, gy, gz = gt_row["global_x"], gt_row["global_y"], gt_row["global_z"]
        count = sum(1 for _, sphere in sphere_all.iterrows()
                    if np.sqrt((gx - sphere["sphere_x"])**2 + (gy - sphere["sphere_y"])**2 + (gz - sphere["sphere_z"])**2) <= sphere["sphere_r"])
        detections_per_GT.append(count)
    return np.mean(detections_per_GT) if detections_per_GT else 0.0

# ==================== Benchmark parameter p in the multi-marker scenario ==================== #

"""
Note on raw count vs precision: Raw detection count (e.g., 5137 at p=0.2) exceeds ground truth (~3000) because many detections are split (one aggregate detected as multiple overlapping spheres).
Precision/recall use spatial matching: FP = detections with NO ground truth overlap. At p=0.2, precision ~95% and recall ~99% (same as Multi-marker CSR section), so few true FPs.
The model always applies "drop contained" (see `remove_overlaps` in model.py); p only controls merging of overlapping-but-not-containing pairs.
"""

# Set up
name = ["A", "B", "C"]

shape = (2000, 2000)
layer_num = 8
layer_gap = 1.5
write_path = ""

CSR_density = [0.04, 0.02, 0.01]

extra_density = [0.02, 0.01, 0.005]
extra_num_clusters = 5000
extra_beta = (1, 19)
extra_comp_prob = [0.4, 0.3, 0.3]
extra_mean_dist = 1

intra_density = [0.02, 0.01, 0.005]
intra_num_clusters = 1000
intra_beta = (19, 1)
intra_comp_prob = [0.8, 0.1, 0.1]
intra_mean_dist = 4

# Benchmark merging parameter p
p_values = np.arange(0, 1.1, 0.1)
benchmark_seeds = np.arange(1, 11)

results_rows = []

for benchmark_seed in benchmark_seeds:
    
    multi_simulate_extra = multi_simulation(name=name, density=extra_density, shape=shape, layer_num=layer_num, layer_gap=layer_gap, simulate_z=True, write_path=write_path, seed=benchmark_seed)
    parents_extra, _, points_extra = multi_simulate_extra.simulate_cluster(num_clusters=extra_num_clusters, beta=extra_beta, comp_prob=extra_comp_prob, mean_dist=extra_mean_dist, comp_thr=2)
    multi_simulate_intra = multi_simulation(name=name, density=intra_density, shape=shape, layer_num=layer_num, layer_gap=layer_gap, simulate_z=True, write_path=write_path, seed=benchmark_seed + 10)
    _, _, points_intra = multi_simulate_intra.simulate_cluster(num_clusters=intra_num_clusters, beta=intra_beta, comp_prob=intra_comp_prob, mean_dist=intra_mean_dist, comp_thr=2)
    simulate_A = simulation(name=name[0], density=CSR_density[0], shape=shape, layer_num=layer_num, layer_gap=layer_gap, simulate_z=True, write_path=write_path, seed=benchmark_seed + 20)
    points_CSR_A = simulate_A.simulate_CSR()
    simulate_B = simulation(name=name[1], density=CSR_density[1], shape=shape, layer_num=layer_num, layer_gap=layer_gap, simulate_z=True, write_path=write_path, seed=benchmark_seed + 30)
    points_CSR_B = simulate_B.simulate_CSR()
    simulate_C = simulation(name=name[2], density=CSR_density[2], shape=shape, layer_num=layer_num, layer_gap=layer_gap, simulate_z=True, write_path=write_path, seed=benchmark_seed + 40)
    points_CSR_C = simulate_C.simulate_CSR()
    parents_all = parents_extra
    points_all = pd.concat([points_extra, points_intra, points_CSR_A, points_CSR_B, points_CSR_C], axis=0, ignore_index=True)
    
    tree_gt = make_tree(d1=np.array(parents_all["global_x"]), d2=np.array(parents_all["global_y"]), d3=np.array(parents_all["global_z"]))
    ground_truth_indices = set(parents_all.index)
    
    detect_base = model(shape=(2000, 2000), transcripts=points_all, target_all=name, eps=1.5, in_thr=0.25, comp_thr=2, size_thr=4, p=0.5, l=2.5)
    sphere_dict = detect_base.dbscan()
    
    for l_value in [2, 2.5, 3]:
        for param in p_values:
            detect = model(shape=(2000, 2000), transcripts=points_all, target_all=name, eps=1.5, in_thr=0.25, comp_thr=2, size_thr=4, p=param, l=l_value)
            sphere_dist = detect.merge_sphere(sphere_dict)
            precision, recall, accuracy, f1 = metric_main(tree_gt, ground_truth_indices, sphere_dist)
            results_rows.append({"l": l_value, "param": param, "seed": benchmark_seed, "num_detections": sphere_dist.shape[0], "avg_detections_per_GT": compute_avg_detections_per_GT(sphere_dist, parents_all), "precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1})
            print(f"l={l_value}, p={param} in seed {benchmark_seed} done!")

results_df = pd.DataFrame(results_rows)
results_df.to_csv(os.path.join(output_dir, "p_l_benchmark_multi_marker_3D_detailed.csv"), index=False)
print(f"Saved: {os.path.join(output_dir, 'p_l_benchmark_multi_marker_3D_detailed.csv')}")

mean_df = results_df.groupby(["l", "param"]).agg({"num_detections": "mean", "avg_detections_per_GT": "mean", "precision": "mean", "recall": "mean", "accuracy": "mean", "f1": "mean"}).reset_index()
mean_df.columns = ["l", "param", "num_detections_mean", "avg_detections_per_GT_mean", "precision_mean", "recall_mean", "accuracy_mean", "f1_mean"]
mean_df.to_csv(os.path.join(output_dir, "p_l_benchmark_multi_marker_3D_mean.csv"), index=False)
print(f"Saved: {os.path.join(output_dir, 'p_l_benchmark_multi_marker_3D_mean.csv')}")