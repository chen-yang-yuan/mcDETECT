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
data_path = f"../data/{dataset}/"
output_path = f"../output/benchmark/"
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
    """Count detections, avg aggregates per transcript (in-aggregate), and n_unique_genes per granule."""
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
        _, counts = np.unique(flat, return_counts=True)
        avg_agg = float(np.mean(counts))
        aggregates_per_transcript_distributions.append({"scenario": scenario, "rho": rho_val, "counts": counts.tolist()})
    else:
        avg_agg = 0.0
        aggregates_per_transcript_distributions.append({"scenario": scenario, "rho": rho_val, "counts": []})

    num_detections_records.append({
        "scenario": scenario,
        "rho": rho_val,
        "num_detections": n,
        "avg_aggregates_per_transcript": avg_agg,
    })
    unique_genes_per_granule_dfs.append(
        pd.DataFrame({"scenario": scenario, "rho": rho_val, "granule_idx": np.arange(n), "n_unique_genes": n_unique_genes})
    )
    return n, avg_agg


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
n, avg = compute_metrics(sphere_no_ops, "no_ops", np.nan)
print(f"  no_ops: {n} detections, avg aggregates/transcript = {avg:.4f}")

# rho sweep: merge_sphere (drop contained + merge overlapping when dist < rho*l*radius_sum)
print("Benchmarking rho = 0 .. 1...")
for rho in rho_values:
    mc = mcDETECT(**mc_kwargs(rho=float(rho)))
    sphere_all = mc.merge_sphere(data_low)
    if use_nc_genes is not None:
        sphere_high = mc.merge_sphere(data_high)
        sphere_all = mc.nc_filter(sphere_all, sphere_high)
    n, avg = compute_metrics(sphere_all, "merge", float(rho))
    print(f"  rho = {rho:.1f}: {n} detections, avg aggregates/transcript = {avg:.4f}")


# ---------------------------------------------------------------------------
# Save and summary
# ---------------------------------------------------------------------------

summary_path = os.path.join(output_path, "benchmark_rho_MERSCOPE_WT1.csv")
pd.DataFrame(num_detections_records).to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")

granule_path = os.path.join(output_path, "benchmark_rho_unique_genes_per_granule_MERSCOPE_WT1.csv")
pd.concat(unique_genes_per_granule_dfs, ignore_index=True).to_csv(granule_path, index=False)
print(f"Saved: {granule_path}")

print("\nInterpretation: no_ops = baseline; rho=0 = drop contained only; rho=1 = full merge. Compare no_ops vs rho=0 (contained) and rho=0 vs rho=1 (overlapping).")