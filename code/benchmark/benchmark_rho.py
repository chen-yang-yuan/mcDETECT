"""
Benchmark the merge threshold rho on MERSCOPE WT 1.

Scenarios:
  no_ops  — Concat per-gene spheres only (no drop, no merge). Baseline.
  rho 0   — Drop contained spheres; do not merge intersect-but-not-contain pairs.
  rho 1   — Drop contained + merge all overlapping/touching (full merge).

Interpretation: (no_ops vs rho=0) = effect of dropping contained;
               (rho=0 vs rho=1) = effect of merging overlapping.
"""

import numpy as np
import os
import pandas as pd

from mcDETECT.utils import make_tree
from mcDETECT.model import mcDETECT


# ---------------------------------------------------------------------------
# Paths and data
# ---------------------------------------------------------------------------

dataset = "MERSCOPE_WT_1"
data_path = f"../../data/{dataset}/"
output_path = f"../../output/benchmark/"
os.makedirs(output_path, exist_ok=True)

transcripts = pd.read_parquet(data_path + "processed_data/transcripts.parquet")
if "target" not in transcripts.columns and "gene" in transcripts.columns:
    transcripts = transcripts.rename(columns={"gene": "target"})
if "overlaps_nucleus" not in transcripts.columns and "overlaps_nucleus_5_dilation" in transcripts.columns:
    transcripts["overlaps_nucleus"] = transcripts["overlaps_nucleus_5_dilation"]

transcript_coords = transcripts[["global_x", "global_y", "global_z"]].values
gene_per_transcript = transcripts["target"].values

gnl_genes = [
    "Camk2a", "Cplx2", "Slc17a7", "Ddn", "Syp", "Map1a", "Shank1", "Syn1",
    "Gria1", "Gria2", "Cyfip2", "Vamp2", "Bsn", "Slc32a1", "Nfasc", "Syt1",
    "Tubb3", "Nav1", "Shank3", "Mapt",
]
nc_genes = list(pd.read_csv(data_path + "processed_data/negative_controls.csv")["Gene"])
use_nc_genes = None  # set to nc_genes to enable negative-control filtering

print(f"Transcripts: {transcripts.shape[0]}, granule markers: {len(gnl_genes)}")

# Spatial index for fast "transcripts in sphere" queries (O(log N + k) per sphere)
tree_transcripts = make_tree(
    transcript_coords[:, 0], transcript_coords[:, 1], transcript_coords[:, 2]
)


# ---------------------------------------------------------------------------
# mcDETECT config (shared; only rho varies in the merge sweep)
# ---------------------------------------------------------------------------

def mc_kwargs(rho=0.2):
    return dict(
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


def compute_metrics(spheres, scenario, rho_val):
    """
    For each transcript in >= 1 aggregate: count how many unique aggregates it belongs to; report mean.

    Two denominators:
      - All genes: any transcript (any gene) that falls in >= 1 sphere. Non-granule genes can lie
        inside spheres defined by granule markers, so they can be counted in multiple spheres even
        when spheres don't overlap in terms of granule markers.
      - Granule markers only: restrict to transcripts whose gene is in gnl_genes; then mean over
        those (ideal = 1 if no sphere overlap for granule-marker transcripts).
    """
    n = spheres.shape[0]
    z_col = "layer_z" if "layer_z" in spheres.columns else "sphere_z"
    cols = spheres[["sphere_x", "sphere_y", z_col, "sphere_r"]]

    all_indices = []
    n_unique_genes = []
    for i in range(n):
        r = cols.iloc[i]
        center = [float(r["sphere_x"]), float(r["sphere_y"]), float(r[z_col])]
        rad = float(r["sphere_r"])
        idx = np.array(tree_transcripts.query_ball_point(center, rad), dtype=np.intp)
        all_indices.append(idx)
        n_unique_genes.append(len(np.unique(gene_per_transcript[idx])) if len(idx) > 0 else 0)

    if all_indices:
        flat = np.concatenate(all_indices)
        unique_idx, counts = np.unique(flat, return_counts=True)
        # Mean over all transcripts (any gene) in >= 1 sphere
        avg_all = float(np.mean(counts))
        # Mean over granule-marker transcripts only (gene in gnl_genes)
        gnl_mask = np.isin(gene_per_transcript[unique_idx], gnl_genes)
        avg_gnl = float(np.mean(counts[gnl_mask])) if gnl_mask.any() else np.nan
        aggregates_per_transcript_distributions.append({"scenario": scenario, "rho": rho_val, "counts": counts.tolist()})
    else:
        avg_all = 0.0
        avg_gnl = np.nan
        aggregates_per_transcript_distributions.append({"scenario": scenario, "rho": rho_val, "counts": []})

    num_detections_records.append({
        "scenario": scenario,
        "rho": rho_val,
        "num_detections": n,
        "avg_aggregates_per_transcript_all_genes": avg_all,
        "avg_aggregates_per_transcript_gnl_only": avg_gnl,
    })
    unique_genes_per_granule_dfs.append(
        pd.DataFrame({"scenario": scenario, "rho": rho_val, "granule_idx": np.arange(n), "n_unique_genes": n_unique_genes})
    )
    return n, avg_all, avg_gnl


# ---------------------------------------------------------------------------
# Benchmark: no_ops then rho sweep
# ---------------------------------------------------------------------------

rho_values = np.arange(0, 1.01, 0.1)
num_detections_records = []
aggregates_per_transcript_distributions = []
unique_genes_per_granule_dfs = []

# Per-gene aggregates (dbscan once; rho not used)
print("Running dbscan() once...")
mc_base = mcDETECT(**mc_kwargs())
_, data_low, data_high = mc_base.dbscan()

# no_ops: concat per-gene spheres, no remove_overlaps
print("Benchmarking no_ops...")
sphere_no_ops = pd.concat([data_low[j] for j in range(len(gnl_genes))], ignore_index=True)
n, avg_all, avg_gnl = compute_metrics(sphere_no_ops, "no_ops", np.nan)
print(f"  no_ops: {n} detections | mean(# agg/transcript) all_genes = {avg_all:.4f}, gnl_only = {avg_gnl:.4f}")

# rho sweep: merge_sphere (drop contained + merge overlapping when dist < rho*l*radius_sum)
print("Benchmarking rho = 0 .. 1...")
for rho in rho_values:
    mc = mcDETECT(**mc_kwargs(rho=float(rho)))
    sphere_all = mc.merge_sphere(data_low)
    if use_nc_genes is not None:
        sphere_high = mc.merge_sphere(data_high)
        sphere_all = mc.nc_filter(sphere_all, sphere_high)
    n, avg_all, avg_gnl = compute_metrics(sphere_all, "merge", float(rho))
    gnl_str = f"{avg_gnl:.4f}" if not np.isnan(avg_gnl) else "n/a"
    print(f"  rho = {rho:.1f}: {n} detections | mean(# agg/transcript) all_genes = {avg_all:.4f}, gnl_only = {gnl_str}")


# ---------------------------------------------------------------------------
# Save and summary
# ---------------------------------------------------------------------------

summary_path = os.path.join(output_path, "benchmark_rho_MERSCOPE_WT1.csv")
pd.DataFrame(num_detections_records).to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")

granule_path = os.path.join(output_path, "benchmark_rho_unique_genes_per_granule_MERSCOPE_WT1.csv")
pd.concat(unique_genes_per_granule_dfs, ignore_index=True).to_csv(granule_path, index=False)
print(f"Saved: {granule_path}")

# Interpretation
print("\n--- Interpretation ---")
print("Metric: For each transcript in >= 1 aggregate, # of unique aggregates it belongs to; we report the mean. Ideal = 1 (no overlap).")
print("  all_genes: denominator = any transcript (any gene) in >= 1 sphere.")
print("  gnl_only:  denominator = transcripts whose gene is in gnl_genes, in >= 1 sphere.")
recs = num_detections_records
if len(recs) >= 2:
    no_ops_det = recs[0]["num_detections"]
    rho0_rec = next((r for r in recs if r.get("scenario") == "merge" and r.get("rho") == 0.0), None)
    rho1_rec = next((r for r in recs if r.get("scenario") == "merge" and r.get("rho") == 1.0), None)
    if rho0_rec is not None:
        print(f"  no_ops vs rho=0: {no_ops_det} -> {rho0_rec['num_detections']} detections (dropping contained: -{no_ops_det - rho0_rec['num_detections']}).")
    if rho0_rec is not None and rho1_rec is not None:
        print(f"  rho=0 vs rho=1: {rho0_rec['num_detections']} -> {rho1_rec['num_detections']} detections (merging overlapping: -{rho0_rec['num_detections'] - rho1_rec['num_detections']}).")
print("  Mean > 1: many transcripts lie in multiple aggregates. merge_sphere only merges *cross-gene* overlaps;")
print("  *within-gene* overlaps (same gene, different DBSCAN clusters) are never merged, so mean stays > 1.")