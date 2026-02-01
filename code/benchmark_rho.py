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

# Granule markers (same as detection.ipynb)
gnl_genes = ["Camk2a", "Cplx2", "Slc17a7", "Ddn", "Syp", "Map1a", "Shank1", "Syn1", "Gria1", "Gria2", "Cyfip2", "Vamp2", "Bsn", "Slc32a1", "Nfasc", "Syt1", "Tubb3", "Nav1", "Shank3", "Mapt"]

# Negative control genes
nc_genes = pd.read_csv(data_path + "processed_data/negative_controls.csv")
nc_genes = list(nc_genes["Gene"])

print(f"Transcripts: {transcripts.shape[0]}")
print(f"Granule markers: {len(gnl_genes)}")

# Rho values to benchmark
rho_values = np.arange(0, 1.1, 0.1)

num_detections_vs_rho = []
avg_aggregates_per_transcript_vs_rho = []
aggregates_per_transcript_distributions = []

transcript_coords = transcripts[["global_x", "global_y", "global_z"]].values

# Use nc_genes for filtering if desired (set to None to skip nc_filter)
use_nc_genes = None  # or nc_genes to enable negative control filtering

# --- Step 1: Run dbscan() once (rho is not used in dbscan) ---
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

# --- Step 2: For each rho, run only merge_sphere (and nc_filter if applicable) ---
print("Benchmarking rho (merge only)...")
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

    # (1) Number of detections
    num_detections = sphere_all.shape[0]
    num_detections_vs_rho.append(num_detections)

    # (2) For each transcript in at least one aggregate, count how many aggregates it belongs to
    transcript_aggregate_count = {}
    for idx, sphere in sphere_all.iterrows():
        center = np.array([sphere["sphere_x"], sphere["sphere_y"], sphere["sphere_z"]])
        distances = np.sqrt(((transcript_coords - center) ** 2).sum(axis=1))
        within_sphere = distances <= sphere["sphere_r"]
        transcript_indices = np.where(within_sphere)[0]
        for ti in transcript_indices:
            transcript_aggregate_count[ti] = transcript_aggregate_count.get(ti, 0) + 1

    if len(transcript_aggregate_count) > 0:
        counts = list(transcript_aggregate_count.values())
        avg_aggregates_per_transcript = np.mean(counts)
        aggregates_per_transcript_distributions.append({"rho": rho, "counts": counts})
    else:
        avg_aggregates_per_transcript = 0.0
        aggregates_per_transcript_distributions.append({"rho": rho, "counts": []})

    avg_aggregates_per_transcript_vs_rho.append(avg_aggregates_per_transcript)
    print(f"rho = {rho:.1f}: {num_detections} detections, mean aggregates per transcript (in-aggregate) = {avg_aggregates_per_transcript:.4f}")

# Save summary results
rho_benchmark_df = pd.DataFrame({
    "rho": rho_values,
    "num_detections": num_detections_vs_rho,
    "avg_aggregates_per_transcript": avg_aggregates_per_transcript_vs_rho,
})
rho_benchmark_df.to_csv(os.path.join(output_path, "benchmark_rho_MERSCOPE_WT1.csv"), index=False)
print("Saved:", os.path.join(output_path, "benchmark_rho_MERSCOPE_WT1.csv"))