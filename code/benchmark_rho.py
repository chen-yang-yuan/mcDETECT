import numpy as np
import os
import pandas as pd

from mcDETECT.utils import *
from mcDETECT.model import mcDETECT

# Paths
dataset = "MERSCOPE_WT_1"
data_path = f"../data/{dataset}/"
output_path = f"../output/benchmark/"
os.makedirs(output_path, exist_ok=True)

# Load transcripts
transcripts = pd.read_parquet(data_path + "processed_data/transcripts.parquet")

# Ensure mcDETECT expects 'target' and 'overlaps_nucleus'
if "target" not in transcripts.columns and "gene" in transcripts.columns:
    transcripts = transcripts.rename(columns={"gene": "target"})
if "overlaps_nucleus" not in transcripts.columns and "overlaps_nucleus_5_dilation" in transcripts.columns:
    transcripts["overlaps_nucleus"] = transcripts["overlaps_nucleus_5_dilation"]

# Pre-extract arrays for faster access (acceleration: avoid repeated column access in loops)
transcript_coords = transcripts[["global_x", "global_y", "global_z"]].values
gene_per_transcript = transcripts["target"].values

# Build a single cKDTree on all transcripts (used for fast "transcripts in sphere" queries)
# Avoids O(num_spheres * num_transcripts) full scans; each query is O(log N + k)
print("Building cKDTree on transcripts for fast spatial queries...")
tree_transcripts = make_tree(
    transcript_coords[:, 0],
    transcript_coords[:, 1],
    transcript_coords[:, 2],
)

# Granule markers (same as detection.ipynb)
gnl_genes = ["Camk2a", "Cplx2", "Slc17a7", "Ddn", "Syp", "Map1a", "Shank1", "Syn1", "Gria1", "Gria2", "Cyfip2", "Vamp2", "Bsn", "Slc32a1", "Nfasc", "Syt1", "Tubb3", "Nav1", "Shank3", "Mapt"]

# Negative control genes
nc_genes = pd.read_csv(data_path + "processed_data/negative_controls.csv")
nc_genes = list(nc_genes["Gene"])

print(f"Transcripts: {transcripts.shape[0]}")
print(f"Granule markers: {len(gnl_genes)}")

# --- Benchmark settings: span from no operations to full merge ---
# "no_ops": no dropping, no merging — just concatenate per-gene spheres (baseline).
# rho=0: drop contained spheres, but do NOT merge "intersect but not contain" pairs.
# rho=0.2,...,1: drop contained + merge overlapping (rho=1 = merge even touching).
rho_values = np.arange(0, 1.01, 0.1)  # 0, 0.1, ..., 1.0

num_detections_records = []
aggregates_per_transcript_distributions = []
unique_genes_per_granule_dfs = []

# Use nc_genes for filtering if desired (set to None to skip nc_filter)
use_nc_genes = None  # or nc_genes to enable negative control filtering

# --- Step 1: Run dbscan() once ---
mc_base = mcDETECT(
    type="MERSCOPE",
    transcripts=transcripts,
    gnl_genes=gnl_genes,
    nc_genes=use_nc_genes,
    eps=1.5,
    minspl=3,
    grid_len=1,
    cutoff_prob=0.95,
    alpha=10,
    low_bound=3,
    size_thr=4,
    in_soma_thr=(0.1, 0.9),
    l=1.0,
    rho=0.2,
    s=1.0,
    nc_top=15,
    nc_thr=0.1,
)
print("Running dbscan() once (per-gene aggregates)...")
_, data_low, data_high = mc_base.dbscan()


def compute_metrics(sphere_all, scenario_label, rho_val):
    """Compute num_detections, avg_aggregates_per_transcript, and unique_genes_per_granule for a sphere table."""
    num_detections = sphere_all.shape[0]
    all_transcript_indices = []
    unique_genes_per_granule = []
    use_z = "layer_z" if "layer_z" in sphere_all.columns else "sphere_z"
    sphere_cols = sphere_all[["sphere_x", "sphere_y", use_z, "sphere_r"]]
    for k in range(num_detections):
        row = sphere_cols.iloc[k]
        center = [float(row["sphere_x"]), float(row["sphere_y"]), float(row[use_z])]
        r = float(row["sphere_r"])
        transcript_indices = np.array(tree_transcripts.query_ball_point(center, r), dtype=np.intp)
        all_transcript_indices.append(transcript_indices)
        if len(transcript_indices) > 0:
            n_unique = len(np.unique(gene_per_transcript[transcript_indices]))
        else:
            n_unique = 0
        unique_genes_per_granule.append(n_unique)
    if all_transcript_indices:
        all_tis = np.concatenate(all_transcript_indices)
        _, counts = np.unique(all_tis, return_counts=True)
        avg_aggregates_per_transcript = float(np.mean(counts))
        aggregates_per_transcript_distributions.append({"scenario": scenario_label, "rho": rho_val, "counts": counts.tolist()})
    else:
        avg_aggregates_per_transcript = 0.0
        aggregates_per_transcript_distributions.append({"scenario": scenario_label, "rho": rho_val, "counts": []})
    num_detections_records.append({
        "scenario": scenario_label,
        "rho": rho_val,
        "num_detections": num_detections,
        "avg_aggregates_per_transcript": avg_aggregates_per_transcript,
    })
    unique_genes_per_granule_dfs.append(
        pd.DataFrame({
            "scenario": scenario_label,
            "rho": rho_val,
            "granule_idx": np.arange(num_detections),
            "n_unique_genes": unique_genes_per_granule,
        })
    )
    return num_detections, avg_aggregates_per_transcript


# --- Step 2a: No operations — concatenate per-gene spheres only (no dropping, no merging) ---
print("Benchmarking: no_ops (concat only, no drop, no merge)...")
sphere_no_ops = pd.concat([data_low[j] for j in range(len(gnl_genes))], ignore_index=True)
n_det, avg_agg = compute_metrics(sphere_no_ops, "no_ops", np.nan)
print(f"  no_ops: {n_det} detections, mean aggregates/transcript (in-aggregate) = {avg_agg:.4f}")

# --- Step 2b: For each rho, run merge_sphere (drop contained + merge overlapping when rho allows) ---
print("Benchmarking: rho = 0, 0.1, ..., 1 (drop contained + merge overlapping)...")
for rho in rho_values:
    mc = mcDETECT(
        type="MERSCOPE",
        transcripts=transcripts,
        gnl_genes=gnl_genes,
        nc_genes=use_nc_genes,
        eps=1.5,
        minspl=3,
        grid_len=1,
        cutoff_prob=0.95,
        alpha=10,
        low_bound=3,
        size_thr=4,
        in_soma_thr=(0.1, 0.9),
        l=1.0,
        rho=rho,
        s=1.0,
        nc_top=15,
        nc_thr=0.1,
    )
    sphere_low = mc.merge_sphere(data_low)
    if use_nc_genes is not None:
        sphere_high = mc.merge_sphere(data_high)
        sphere_all = mc.nc_filter(sphere_low, sphere_high)
    else:
        sphere_all = sphere_low
    n_det, avg_agg = compute_metrics(sphere_all, "merge", float(rho))
    print(f"  rho = {rho:.1f}: {n_det} detections, mean aggregates/transcript (in-aggregate) = {avg_agg:.4f}")

# Save summary results
rho_benchmark_df = pd.DataFrame(num_detections_records)
rho_benchmark_df.to_csv(os.path.join(output_path, "benchmark_rho_MERSCOPE_WT1.csv"), index=False)
print("Saved:", os.path.join(output_path, "benchmark_rho_MERSCOPE_WT1.csv"))

# Save unique genes per granule
unique_genes_per_granule_df = pd.concat(unique_genes_per_granule_dfs, ignore_index=True)
unique_genes_per_granule_path = os.path.join(output_path, "benchmark_rho_unique_genes_per_granule_MERSCOPE_WT1.csv")
unique_genes_per_granule_df.to_csv(unique_genes_per_granule_path, index=False)
print("Saved:", unique_genes_per_granule_path)

# Interpretation
print("\nInterpretation:")
print("  no_ops: no dropping, no merging (concat per-gene spheres only) — baseline.")
print("  rho=0: drop contained spheres, no merging of 'intersect but not contain' pairs.")
print("  rho=1: drop contained + merge all overlapping/touching spheres (full merge).")
print("  Difference (no_ops vs rho=0) = effect of dropping contained; (rho=0 vs rho=1) = effect of merging overlapping.")