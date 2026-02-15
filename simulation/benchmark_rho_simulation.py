import miniball
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

# ==================== Helper functions for volume-, Jaccard-, and Dice-based merge ==================== #

"""
Each has an l-like parameter (l_vol, l_jaccard, l_dice) fixed at 2.5 to match distance l=2.5
"""

def sphere_volume(r):
    return (4.0 / 3.0) * np.pi * (r ** 3)

def sphere_intersection_volume(d, r1, r2):
    if d <= 1e-12:
        return sphere_volume(min(r1, r2))
    if d >= r1 + r2:
        return 0.0
    if d <= r2 - r1:
        return sphere_volume(r1)
    term = (r1 + r2 - d) ** 2 * (d**2 + 2 * d * (r1 + r2) - 3 * (r1 - r2) ** 2)
    return (np.pi / (12 * d)) * term

def get_points_in_sphere(center_xyz, radius, gene, tree_transcripts, transcript_coords, gene_per_transcript):
    idx = np.array(tree_transcripts.query_ball_point(center_xyz, radius), dtype=np.intp)
    if len(idx) == 0:
        return np.empty((0, 3))
    mask = gene_per_transcript[idx] == gene
    return transcript_coords[idx[mask]]

def _do_merge_two_spheres_sim(set_a, i, set_b, j, sphere_a, sphere_b, s, tree_transcripts, transcript_coords, gene_per_transcript):
    ca = np.array([sphere_a["sphere_x"], sphere_a["sphere_y"], sphere_a["sphere_z"]])
    ra = float(sphere_a["sphere_r"])
    cb = np.array([sphere_b["sphere_x"], sphere_b["sphere_y"], sphere_b["sphere_z"]])
    rb = float(sphere_b["sphere_r"])
    pts_a = get_points_in_sphere(ca, ra, sphere_a["gene"], tree_transcripts, transcript_coords, gene_per_transcript)
    pts_b = get_points_in_sphere(cb, rb, sphere_b["gene"], tree_transcripts, transcript_coords, gene_per_transcript)
    pts_union = np.vstack([pts_a, pts_b]) if pts_a.size and pts_b.size else (pts_a if pts_a.size else pts_b)
    if pts_union.size == 0:
        return False
    try:
        new_center, r2 = miniball.get_bounding_ball(pts_union, epsilon=1e-8)
    except Exception:
        return False
    new_r = np.sqrt(r2) * s
    set_a.loc[i, "sphere_x"] = new_center[0]
    set_a.loc[i, "sphere_y"] = new_center[1]
    set_a.loc[i, "sphere_z"] = new_center[2]
    set_a.loc[i, "sphere_r"] = new_r
    set_b.drop(index=j, inplace=True)
    return True

def remove_overlaps_by_volume_sim(set_a, set_b, gamma, l_vol, s, tree_transcripts, transcript_coords, gene_per_transcript):
    gamma_eff = gamma / l_vol
    set_a, set_b = set_a.copy(), set_b.copy()
    if set_a.shape[0] == 0 or set_b.shape[0] == 0:
        return set_a, set_b
    idx_b = make_rtree(set_b)
    for i, sphere_a in set_a.iterrows():
        ca = np.array([sphere_a.sphere_x, sphere_a.sphere_y, sphere_a.sphere_z])
        ra = float(sphere_a.sphere_r)
        va = sphere_volume(ra)
        bounds_a = (sphere_a.sphere_x - sphere_a.sphere_r, sphere_a.sphere_y - sphere_a.sphere_r,
                    sphere_a.sphere_x + sphere_a.sphere_r, sphere_a.sphere_y + sphere_a.sphere_r)
        for j in idx_b.intersection(bounds_a):
            if j not in set_b.index:
                continue
            sphere_b = set_b.loc[j]
            cb = np.array([sphere_b.sphere_x, sphere_b.sphere_y, sphere_b.sphere_z])
            rb = float(sphere_b.sphere_r)
            vb = sphere_volume(rb)
            d = np.linalg.norm(ca - cb)
            if d >= ra + rb:
                continue
            r_small, r_large = (ra, rb) if ra <= rb else (rb, ra)
            v_small = sphere_volume(r_small)
            if d <= r_large - r_small:
                if ra > rb:
                    set_b.drop(index=j, inplace=True)
                else:
                    set_a.loc[i] = set_b.loc[j]
                    set_b.drop(index=j, inplace=True)
                continue
            inter_vol = sphere_intersection_volume(d, r_small, r_large)
            if inter_vol / v_small < gamma_eff:
                continue
            pts_a = get_points_in_sphere(ca, ra, sphere_a["gene"], tree_transcripts, transcript_coords, gene_per_transcript)
            pts_b = get_points_in_sphere(cb, rb, sphere_b["gene"], tree_transcripts, transcript_coords, gene_per_transcript)
            pts_union = np.vstack([pts_a, pts_b]) if pts_a.size and pts_b.size else (pts_a if pts_a.size else pts_b)
            if pts_union.size == 0:
                continue
            try:
                new_center, r2 = miniball.get_bounding_ball(pts_union, epsilon=1e-8)
            except Exception:
                continue
            new_r = np.sqrt(r2) * s
            set_a.loc[i, "sphere_x"], set_a.loc[i, "sphere_y"], set_a.loc[i, "sphere_z"] = new_center[0], new_center[1], new_center[2]
            set_a.loc[i, "sphere_r"] = new_r
            set_b.drop(index=j, inplace=True)
    return set_a.reset_index(drop=True), set_b.reset_index(drop=True)

def remove_overlaps_by_jaccard_sim(set_a, set_b, jaccard_thr, l_jaccard, s, tree_transcripts, transcript_coords, gene_per_transcript):
    jaccard_eff = jaccard_thr / l_jaccard
    set_a, set_b = set_a.copy(), set_b.copy()
    if set_a.shape[0] == 0 or set_b.shape[0] == 0:
        return set_a, set_b
    idx_b = make_rtree(set_b)
    for i, sphere_a in set_a.iterrows():
        ca = np.array([sphere_a.sphere_x, sphere_a.sphere_y, sphere_a.sphere_z])
        ra, va = float(sphere_a.sphere_r), sphere_volume(float(sphere_a.sphere_r))
        bounds_a = (sphere_a.sphere_x - sphere_a.sphere_r, sphere_a.sphere_y - sphere_a.sphere_r,
                    sphere_a.sphere_x + sphere_a.sphere_r, sphere_a.sphere_y + sphere_a.sphere_r)
        for j in idx_b.intersection(bounds_a):
            if j not in set_b.index:
                continue
            sphere_b = set_b.loc[j]
            cb = np.array([sphere_b.sphere_x, sphere_b.sphere_y, sphere_b.sphere_z])
            rb, vb = float(sphere_b.sphere_r), sphere_volume(float(sphere_b.sphere_r))
            d = np.linalg.norm(ca - cb)
            if d >= ra + rb:
                continue
            r_small, r_large = (ra, rb) if ra <= rb else (rb, ra)
            if d <= r_large - r_small:
                if ra > rb:
                    set_b.drop(index=j, inplace=True)
                else:
                    set_a.loc[i] = set_b.loc[j]
                    set_b.drop(index=j, inplace=True)
                continue
            inter_vol = sphere_intersection_volume(d, r_small, r_large)
            union_vol = va + vb - inter_vol
            if union_vol <= 0 or inter_vol / union_vol < jaccard_eff:
                continue
            _do_merge_two_spheres_sim(set_a, i, set_b, j, sphere_a, sphere_b, s, tree_transcripts, transcript_coords, gene_per_transcript)
    return set_a.reset_index(drop=True), set_b.reset_index(drop=True)

def remove_overlaps_by_dice_sim(set_a, set_b, dice_thr, l_dice, s, tree_transcripts, transcript_coords, gene_per_transcript):
    dice_eff = dice_thr / l_dice
    set_a, set_b = set_a.copy(), set_b.copy()
    if set_a.shape[0] == 0 or set_b.shape[0] == 0:
        return set_a, set_b
    idx_b = make_rtree(set_b)
    for i, sphere_a in set_a.iterrows():
        ca = np.array([sphere_a.sphere_x, sphere_a.sphere_y, sphere_a.sphere_z])
        ra, va = float(sphere_a.sphere_r), sphere_volume(float(sphere_a.sphere_r))
        bounds_a = (sphere_a.sphere_x - sphere_a.sphere_r, sphere_a.sphere_y - sphere_a.sphere_r,
                    sphere_a.sphere_x + sphere_a.sphere_r, sphere_a.sphere_y + sphere_a.sphere_r)
        for j in idx_b.intersection(bounds_a):
            if j not in set_b.index:
                continue
            sphere_b = set_b.loc[j]
            cb = np.array([sphere_b.sphere_x, sphere_b.sphere_y, sphere_b.sphere_z])
            rb, vb = float(sphere_b.sphere_r), sphere_volume(float(sphere_b.sphere_r))
            d = np.linalg.norm(ca - cb)
            if d >= ra + rb:
                continue
            r_small, r_large = (ra, rb) if ra <= rb else (rb, ra)
            if d <= r_large - r_small:
                if ra > rb:
                    set_b.drop(index=j, inplace=True)
                else:
                    set_a.loc[i] = set_b.loc[j]
                    set_b.drop(index=j, inplace=True)
                continue
            inter_vol = sphere_intersection_volume(d, r_small, r_large)
            if (va + vb) <= 0 or (2.0 * inter_vol) / (va + vb) < dice_eff:
                continue
            _do_merge_two_spheres_sim(set_a, i, set_b, j, sphere_a, sphere_b, s, tree_transcripts, transcript_coords, gene_per_transcript)
    return set_a.reset_index(drop=True), set_b.reset_index(drop=True)

def merge_sphere_by_volume_sim(sphere_dict, target_names, param, l_vol, s, tree_transcripts, transcript_coords, gene_per_transcript):
    gamma = 1.0 - param  # param 0->drop only, param 1->full
    sphere = sphere_dict[0].copy()
    for j in range(1, len(target_names)):
        set_b = sphere_dict[j]
        sphere, set_b_new = remove_overlaps_by_volume_sim(sphere, set_b, gamma, l_vol, s, tree_transcripts, transcript_coords, gene_per_transcript)
        sphere = pd.concat([sphere, set_b_new], ignore_index=True).reset_index(drop=True)
    return sphere

def merge_sphere_by_jaccard_sim(sphere_dict, target_names, param, l_jaccard, s, tree_transcripts, transcript_coords, gene_per_transcript):
    jaccard_thr = 1.0 - param
    sphere = sphere_dict[0].copy()
    for j in range(1, len(target_names)):
        set_b = sphere_dict[j]
        sphere, set_b_new = remove_overlaps_by_jaccard_sim(sphere, set_b, jaccard_thr, l_jaccard, s, tree_transcripts, transcript_coords, gene_per_transcript)
        sphere = pd.concat([sphere, set_b_new], ignore_index=True).reset_index(drop=True)
    return sphere

def merge_sphere_by_dice_sim(sphere_dict, target_names, param, l_dice, s, tree_transcripts, transcript_coords, gene_per_transcript):
    dice_thr = 1.0 - param
    sphere = sphere_dict[0].copy()
    for j in range(1, len(target_names)):
        set_b = sphere_dict[j]
        sphere, set_b_new = remove_overlaps_by_dice_sim(sphere, set_b, dice_thr, l_dice, s, tree_transcripts, transcript_coords, gene_per_transcript)
        sphere = pd.concat([sphere, set_b_new], ignore_index=True).reset_index(drop=True)
    return sphere

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

# Benchmark merge strategies: distance (default), volume, Jaccard, Dice
# param 0..1 for each; l=2.5 for distance; l_vol=l_jaccard=l_dice=2.5 for others
p_values = np.arange(0, 1.1, 0.1)
benchmark_seeds = np.arange(1, 11)
l_distance = l_vol = l_jaccard = l_dice = 2.5
s_volume = 1.0

def compute_avg_detections_per_GT(sphere_all, parents_all):
    detections_per_GT = []
    for _, gt_row in parents_all.iterrows():
        gx, gy, gz = gt_row["global_x"], gt_row["global_y"], gt_row["global_z"]
        count = sum(1 for _, sphere in sphere_all.iterrows()
                    if np.sqrt((gx - sphere["sphere_x"])**2 + (gy - sphere["sphere_y"])**2 + (gz - sphere["sphere_z"])**2) <= sphere["sphere_r"])
        detections_per_GT.append(count)
    return np.mean(detections_per_GT) if detections_per_GT else 0.0

results_rows = []

print("Benchmarking merge strategies (distance, volume, Jaccard, Dice)...")

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
    
    # Ground truth tree for precision/recall/accuracy/F1
    tree_gt = make_tree(d1=np.array(parents_all["global_x"]), d2=np.array(parents_all["global_y"]), d3=np.array(parents_all["global_z"]))
    ground_truth_indices = set(parents_all.index)
    
    transcript_coords = points_all[["global_x", "global_y", "global_z"]].values
    gene_per_transcript = points_all["target"].values
    tree_transcripts = make_tree(d1=transcript_coords[:, 0], d2=transcript_coords[:, 1], d3=transcript_coords[:, 2])
    
    detect_base = model(shape=(2000, 2000), transcripts=points_all, target_all=name, eps=1.5, in_thr=0.25, comp_thr=2, size_thr=4, p=0.5, l=l_distance)
    sphere_dict = detect_base.dbscan()
    
    for param in p_values:
        
        # Distance (default)
        detect = model(shape=(2000, 2000), transcripts=points_all, target_all=name, eps=1.5, in_thr=0.25, comp_thr=2, size_thr=4, p=param, l=l_distance)
        sphere_dist = detect.merge_sphere(sphere_dict)
        precision, recall, accuracy, f1 = metric_main(tree_gt, ground_truth_indices, sphere_dist)
        results_rows.append({"strategy": "distance", "param": param, "seed": benchmark_seed, "num_detections": sphere_dist.shape[0], "avg_detections_per_GT": compute_avg_detections_per_GT(sphere_dist, parents_all), "precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1})
        
        # Volume
        sphere_vol = merge_sphere_by_volume_sim(sphere_dict, name, param, l_vol, s_volume, tree_transcripts, transcript_coords, gene_per_transcript)
        precision, recall, accuracy, f1 = metric_main(tree_gt, ground_truth_indices, sphere_vol)
        results_rows.append({"strategy": "volume", "param": param, "seed": benchmark_seed, "num_detections": sphere_vol.shape[0], "avg_detections_per_GT": compute_avg_detections_per_GT(sphere_vol, parents_all), "precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1})
        
        # Jaccard
        sphere_jaccard = merge_sphere_by_jaccard_sim(sphere_dict, name, param, l_jaccard, s_volume, tree_transcripts, transcript_coords, gene_per_transcript)
        precision, recall, accuracy, f1 = metric_main(tree_gt, ground_truth_indices, sphere_jaccard)
        results_rows.append({"strategy": "jaccard", "param": param, "seed": benchmark_seed, "num_detections": sphere_jaccard.shape[0], "avg_detections_per_GT": compute_avg_detections_per_GT(sphere_jaccard, parents_all), "precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1})
        
        # Dice
        sphere_dice = merge_sphere_by_dice_sim(sphere_dict, name, param, l_dice, s_volume, tree_transcripts, transcript_coords, gene_per_transcript)
        precision, recall, accuracy, f1 = metric_main(tree_gt, ground_truth_indices, sphere_dice)
        results_rows.append({"strategy": "dice", "param": param, "seed": benchmark_seed, "num_detections": sphere_dice.shape[0], "avg_detections_per_GT": compute_avg_detections_per_GT(sphere_dice, parents_all), "precision": precision, "recall": recall, "accuracy": accuracy, "f1": f1 })
        
        print(f"p={param} in seed {benchmark_seed} done!")

results_df = pd.DataFrame(results_rows)
mean_df = results_df.groupby(["strategy", "param"]).agg({"num_detections": "mean", "avg_detections_per_GT": "mean", "precision": "mean", "recall": "mean", "accuracy": "mean", "f1": "mean"}).reset_index()
mean_df.columns = ["strategy", "param", "num_detections_mean", "avg_detections_per_GT_mean", "precision_mean", "recall_mean", "accuracy_mean", "f1_mean"]

mean_df.to_csv(os.path.join(output_dir, "p_benchmark_multi_marker_3D_all_strategies.csv"), index=False)
print(f"Saved: {os.path.join(output_dir, 'p_benchmark_multi_marker_3D_all_strategies.csv')}")