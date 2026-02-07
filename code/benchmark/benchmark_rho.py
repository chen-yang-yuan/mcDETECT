"""
Benchmark merge criteria on MERSCOPE WT 1.

Each parameter sweep spans three scenarios (model.py unchanged):
  (1) No operation     — no drop, no merge (no_ops: concat per-gene spheres only).
  (2) Drop contained   — drop the smaller sphere when one contains the other; do NOT merge intersect-but-not-contain pairs.
  (3) Full operation   — drop contained + merge to maximum extent (e.g. merge even touching in distance-based).

Parameter → (2) drop only / (3) full:
  rho=0         → (2); rho=1            → (3) [distance: merge when dist < rho*l*(r_a+r_b)]
  gamma=1       → (2); gamma=0          → (3) [volume: merge when inter/min(V_a,V_b) >= gamma]
  jaccard_thr=1 → (2); jaccard_thr=0    → (3) [Jaccard: merge when inter/union >= jaccard_thr]
  dice_thr=1    → (2); dice_thr=0       → (3) [Dice: merge when 2*inter/(V_a+V_b) >= dice_thr]

Scenarios: no_ops, merge (rho), merge_volume (gamma), merge_jaccard (jaccard_thr), merge_dice (dice_thr).
"""

import numpy as np
import os
import pandas as pd
import miniball

from mcDETECT.utils import make_tree, make_rtree
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
# Alternative merge criterion: sphere-sphere intersection volume (helper only)
# ---------------------------------------------------------------------------

def sphere_volume(r):
    """Volume of a sphere with radius r."""
    return (4.0 / 3.0) * np.pi * (r ** 3)


def sphere_intersection_volume(d, r1, r2):
    """
    Volume of intersection of two spheres with distance d between centers; r1 <= r2.
    Returns 0 if disjoint, volume of smaller sphere if one inside the other, else closed-form cap formula.
    """
    if d >= r1 + r2:
        return 0.0
    if d <= r2 - r1:
        return sphere_volume(r1)
    # r2 - r1 < d < r1 + r2: intersection = lens from two spherical caps
    term = (r1 + r2 - d) ** 2 * (d**2 + 2 * d * (r1 + r2) - 3 * (r1 - r2) ** 2)
    return (np.pi / (12 * d)) * term


def get_points_in_sphere(center_xyz, radius, gene, tree_transcripts, transcript_coords, gene_per_transcript):
    """Return (N, 3) array of transcript coordinates inside the sphere that belong to the given gene."""
    idx = np.array(tree_transcripts.query_ball_point(center_xyz, radius), dtype=np.intp)
    if len(idx) == 0:
        return np.empty((0, 3))
    mask = gene_per_transcript[idx] == gene
    return transcript_coords[idx[mask]]


def remove_overlaps_by_volume(set_a, set_b, gamma, s, tree_transcripts, transcript_coords, gene_per_transcript, gnl_genes):
    """
    Resolve overlaps between set_a and set_b using intersection-volume ratio (no model change).
    - If one sphere contains the other: drop the smaller (same as model).
    - Else if intersection_volume / min(V_a, V_b) >= gamma: merge (union of points, new bounding ball) and drop b.
    Returns (set_a, set_b) after in-place updates and reset_index.
    """
    set_a = set_a.copy()
    set_b = set_b.copy()
    idx_b = make_rtree(set_b)

    for i, sphere_a in set_a.iterrows():
        ca = np.array([sphere_a.sphere_x, sphere_a.sphere_y, sphere_a.sphere_z])
        ra = float(sphere_a.sphere_r)
        va = sphere_volume(ra)
        bounds_a = (
            sphere_a.sphere_x - sphere_a.sphere_r,
            sphere_a.sphere_y - sphere_a.sphere_r,
            sphere_a.sphere_x + sphere_a.sphere_r,
            sphere_a.sphere_y + sphere_a.sphere_r,
        )
        for j in idx_b.intersection(bounds_a):
            if j not in set_b.index:
                continue
            sphere_b = set_b.loc[j]
            cb = np.array([sphere_b.sphere_x, sphere_b.sphere_y, sphere_b.sphere_z])
            rb = float(sphere_b.sphere_r)
            vb = sphere_volume(rb)
            d = np.linalg.norm(ca - cb)

            # Disjoint
            if d >= ra + rb:
                continue
            r_small, r_large = (ra, rb) if ra <= rb else (rb, ra)
            v_small = sphere_volume(r_small)

            # One contains the other: keep larger, drop smaller (same as model)
            if d <= r_large - r_small:
                if ra > rb:
                    set_b.drop(index=j, inplace=True)
                else:
                    set_a.loc[i] = set_b.loc[j]
                    set_b.drop(index=j, inplace=True)
                continue

            # Intersect but not contain: merge only if intersection_volume / min_vol >= gamma
            inter_vol = sphere_intersection_volume(d, r_small, r_large)
            if inter_vol / v_small < gamma:
                continue
            # Merge: union of points, new bounding ball
            pts_a = get_points_in_sphere(
                ca, ra, sphere_a["gene"], tree_transcripts, transcript_coords, gene_per_transcript
            )
            pts_b = get_points_in_sphere(
                cb, rb, sphere_b["gene"], tree_transcripts, transcript_coords, gene_per_transcript
            )
            pts_union = np.vstack([pts_a, pts_b]) if pts_a.size and pts_b.size else (pts_a if pts_a.size else pts_b)
            if pts_union.size == 0:
                continue
            try:
                new_center, r2 = miniball.get_bounding_ball(pts_union, epsilon=1e-8)
            except Exception:
                continue
            new_r = np.sqrt(r2) * s
            set_a.loc[i, "sphere_x"] = new_center[0]
            set_a.loc[i, "sphere_y"] = new_center[1]
            set_a.loc[i, "sphere_z"] = new_center[2]
            set_a.loc[i, "sphere_r"] = new_r
            set_b.drop(index=j, inplace=True)

    set_a = set_a.reset_index(drop=True)
    set_b = set_b.reset_index(drop=True)
    return set_a, set_b


def merge_sphere_by_intersection(data_low, gnl_genes, gamma, s, tree_transcripts, transcript_coords, gene_per_transcript):
    """
    Merge per-gene spheres using intersection/min_volume criterion (helper only).
    data_low: dict mapping gene index -> DataFrame of spheres (columns sphere_x, sphere_y, sphere_z, sphere_r, gene, ...).
    """
    sphere = data_low[0].copy()
    for j in range(1, len(gnl_genes)):
        set_b = data_low[j]
        sphere, set_b_new = remove_overlaps_by_volume(
            sphere, set_b, gamma, s, tree_transcripts, transcript_coords, gene_per_transcript, gnl_genes
        )
        sphere = pd.concat([sphere, set_b_new], ignore_index=True)
        sphere = sphere.reset_index(drop=True)
    return sphere


def _do_merge_two_spheres(set_a, i, set_b, j, sphere_a, sphere_b, s, tree_transcripts, transcript_coords, gene_per_transcript):
    """Perform merge: replace set_a[i] with bounding ball of union of points from sphere_a and sphere_b; drop set_b[j]."""
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


def remove_overlaps_by_jaccard(set_a, set_b, jaccard_thr, s, tree_transcripts, transcript_coords, gene_per_transcript, gnl_genes):
    """
    Resolve overlaps using Jaccard: intersection_volume / union_volume (no model change).
    - If one sphere contains the other: drop the smaller (same as model).
    - Else if inter_vol / (V_a + V_b - inter_vol) >= jaccard_thr: merge and drop b.
    """
    set_a = set_a.copy()
    set_b = set_b.copy()
    idx_b = make_rtree(set_b)

    for i, sphere_a in set_a.iterrows():
        ca = np.array([sphere_a.sphere_x, sphere_a.sphere_y, sphere_a.sphere_z])
        ra = float(sphere_a.sphere_r)
        va = sphere_volume(ra)
        bounds_a = (
            sphere_a.sphere_x - sphere_a.sphere_r,
            sphere_a.sphere_y - sphere_a.sphere_r,
            sphere_a.sphere_x + sphere_a.sphere_r,
            sphere_a.sphere_y + sphere_a.sphere_r,
        )
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
            if d <= r_large - r_small:
                if ra > rb:
                    set_b.drop(index=j, inplace=True)
                else:
                    set_a.loc[i] = set_b.loc[j]
                    set_b.drop(index=j, inplace=True)
                continue
            inter_vol = sphere_intersection_volume(d, r_small, r_large)
            union_vol = va + vb - inter_vol
            if union_vol <= 0 or inter_vol / union_vol < jaccard_thr:
                continue
            _do_merge_two_spheres(set_a, i, set_b, j, sphere_a, sphere_b, s, tree_transcripts, transcript_coords, gene_per_transcript)

    set_a = set_a.reset_index(drop=True)
    set_b = set_b.reset_index(drop=True)
    return set_a, set_b


def remove_overlaps_by_dice(set_a, set_b, dice_thr, s, tree_transcripts, transcript_coords, gene_per_transcript, gnl_genes):
    """
    Resolve overlaps using Dice: 2 * intersection_volume / (V_a + V_b) (no model change).
    - If one sphere contains the other: drop the smaller (same as model).
    - Else if 2*inter_vol / (V_a + V_b) >= dice_thr: merge and drop b.
    """
    set_a = set_a.copy()
    set_b = set_b.copy()
    idx_b = make_rtree(set_b)

    for i, sphere_a in set_a.iterrows():
        ca = np.array([sphere_a.sphere_x, sphere_a.sphere_y, sphere_a.sphere_z])
        ra = float(sphere_a.sphere_r)
        va = sphere_volume(ra)
        bounds_a = (
            sphere_a.sphere_x - sphere_a.sphere_r,
            sphere_a.sphere_y - sphere_a.sphere_r,
            sphere_a.sphere_x + sphere_a.sphere_r,
            sphere_a.sphere_y + sphere_a.sphere_r,
        )
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
            if d <= r_large - r_small:
                if ra > rb:
                    set_b.drop(index=j, inplace=True)
                else:
                    set_a.loc[i] = set_b.loc[j]
                    set_b.drop(index=j, inplace=True)
                continue
            inter_vol = sphere_intersection_volume(d, r_small, r_large)
            if (va + vb) <= 0 or (2.0 * inter_vol) / (va + vb) < dice_thr:
                continue
            _do_merge_two_spheres(set_a, i, set_b, j, sphere_a, sphere_b, s, tree_transcripts, transcript_coords, gene_per_transcript)

    set_a = set_a.reset_index(drop=True)
    set_b = set_b.reset_index(drop=True)
    return set_a, set_b


def merge_sphere_by_jaccard(data_low, gnl_genes, jaccard_thr, s, tree_transcripts, transcript_coords, gene_per_transcript):
    """Merge per-gene spheres using Jaccard criterion (helper only)."""
    sphere = data_low[0].copy()
    for j in range(1, len(gnl_genes)):
        set_b = data_low[j]
        sphere, set_b_new = remove_overlaps_by_jaccard(
            sphere, set_b, jaccard_thr, s, tree_transcripts, transcript_coords, gene_per_transcript, gnl_genes
        )
        sphere = pd.concat([sphere, set_b_new], ignore_index=True)
        sphere = sphere.reset_index(drop=True)
    return sphere


def merge_sphere_by_dice(data_low, gnl_genes, dice_thr, s, tree_transcripts, transcript_coords, gene_per_transcript):
    """Merge per-gene spheres using Dice criterion (helper only)."""
    sphere = data_low[0].copy()
    for j in range(1, len(gnl_genes)):
        set_b = data_low[j]
        sphere, set_b_new = remove_overlaps_by_dice(
            sphere, set_b, dice_thr, s, tree_transcripts, transcript_coords, gene_per_transcript, gnl_genes
        )
        sphere = pd.concat([sphere, set_b_new], ignore_index=True)
        sphere = sphere.reset_index(drop=True)
    return sphere


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


def compute_metrics(spheres, scenario, rho_val, gamma_val=np.nan, jaccard_thr_val=np.nan, dice_thr_val=np.nan):
    """
    For each transcript in >= 1 aggregate: count how many unique aggregates it belongs to; report mean.

    Two denominators:
      - All genes: any transcript (any gene) that falls in >= 1 sphere.
      - Granule markers only: restrict to transcripts whose gene is in gnl_genes (ideal mean = 1 if no overlap).
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
        avg_all = float(np.mean(counts))
        gnl_mask = np.isin(gene_per_transcript[unique_idx], gnl_genes)
        avg_gnl = float(np.mean(counts[gnl_mask])) if gnl_mask.any() else np.nan
        aggregates_per_transcript_distributions.append({"scenario": scenario, "counts": counts.tolist()})
    else:
        avg_all = 0.0
        avg_gnl = np.nan
        aggregates_per_transcript_distributions.append({"scenario": scenario, "counts": []})

    num_detections_records.append({
        "scenario": scenario,
        "rho": rho_val,
        "gamma": gamma_val,
        "jaccard_thr": jaccard_thr_val,
        "dice_thr": dice_thr_val,
        "num_detections": n,
        "avg_aggregates_per_transcript_all_genes": avg_all,
        "avg_aggregates_per_transcript_gnl_only": avg_gnl,
    })
    unique_genes_per_granule_dfs.append(
        pd.DataFrame({
            "scenario": scenario, "rho": rho_val, "gamma": gamma_val, "jaccard_thr": jaccard_thr_val, "dice_thr": dice_thr_val,
            "granule_idx": np.arange(n), "n_unique_genes": n_unique_genes,
        })
    )
    return n, avg_all, avg_gnl


# ---------------------------------------------------------------------------
# Benchmark: (1) no_ops, then each parameter from (2) drop only to (3) full
# ---------------------------------------------------------------------------

# Sweep 0..1: rho=0 = (2) drop only, rho=1 = (3) full
rho_values = np.arange(0, 1.01, 0.1)
# Sweep 1..0: gamma/jaccard/dice: 1 = (2) drop only, 0 = (3) full
gamma_values = np.arange(1.0, -0.01, -0.1)
jaccard_thr_values = np.arange(1.0, -0.01, -0.1)
dice_thr_values = np.arange(1.0, -0.01, -0.1)

num_detections_records = []
aggregates_per_transcript_distributions = []
unique_genes_per_granule_dfs = []

# Per-gene aggregates (dbscan once; rho not used)
print("Running dbscan() once...")
mc_base = mcDETECT(**mc_kwargs())
_, data_low, data_high = mc_base.dbscan()

# (1) No operation: concat per-gene spheres only (no drop, no merge)
print("Benchmarking (1) no_ops...")
sphere_no_ops = pd.concat([data_low[j] for j in range(len(gnl_genes))], ignore_index=True)
n, avg_all, avg_gnl = compute_metrics(sphere_no_ops, "no_ops", np.nan, np.nan, np.nan, np.nan)
print(f"  no_ops: {n} detections | mean(# agg/transcript) all_genes = {avg_all:.4f}, gnl_only = {avg_gnl:.4f}")

# rho: (2) rho=0 = drop only, ... (3) rho=1 = full (merge even touching)
print("Benchmarking rho: (2) drop only (rho=0) .. (3) full (rho=1)...")
for rho in rho_values:
    mc = mcDETECT(**mc_kwargs(rho=float(rho)))
    sphere_all = mc.merge_sphere(data_low)
    if use_nc_genes is not None:
        sphere_high = mc.merge_sphere(data_high)
        sphere_all = mc.nc_filter(sphere_all, sphere_high)
    n, avg_all, avg_gnl = compute_metrics(sphere_all, "merge", float(rho), np.nan, np.nan, np.nan)
    gnl_str = f"{avg_gnl:.4f}" if not np.isnan(avg_gnl) else "n/a"
    print(f"  rho = {rho:.1f}: {n} detections | mean(# agg/transcript) all_genes = {avg_all:.4f}, gnl_only = {gnl_str}")

# gamma: (2) gamma=1 = drop only, ... (3) gamma=0 = full (merge any overlap)
s_volume = 1.0
print("Benchmarking gamma: (2) drop only (gamma=1) .. (3) full (gamma=0)...")
for gamma in gamma_values:
    sphere_all = merge_sphere_by_intersection(
        data_low, gnl_genes, float(gamma), s_volume,
        tree_transcripts, transcript_coords, gene_per_transcript,
    )
    n, avg_all, avg_gnl = compute_metrics(sphere_all, "merge_volume", np.nan, float(gamma), np.nan, np.nan)
    gnl_str = f"{avg_gnl:.4f}" if not np.isnan(avg_gnl) else "n/a"
    print(f"  gamma = {gamma:.1f}: {n} detections | mean(# agg/transcript) all_genes = {avg_all:.4f}, gnl_only = {gnl_str}")

# jaccard_thr: (2) jaccard_thr=1 = drop only, ... (3) jaccard_thr=0 = full
print("Benchmarking jaccard_thr: (2) drop only (jaccard_thr=1) .. (3) full (jaccard_thr=0)...")
for jaccard_thr in jaccard_thr_values:
    sphere_all = merge_sphere_by_jaccard(
        data_low, gnl_genes, float(jaccard_thr), s_volume,
        tree_transcripts, transcript_coords, gene_per_transcript,
    )
    n, avg_all, avg_gnl = compute_metrics(sphere_all, "merge_jaccard", np.nan, np.nan, float(jaccard_thr), np.nan)
    gnl_str = f"{avg_gnl:.4f}" if not np.isnan(avg_gnl) else "n/a"
    print(f"  jaccard_thr = {jaccard_thr:.1f}: {n} detections | mean(# agg/transcript) all_genes = {avg_all:.4f}, gnl_only = {gnl_str}")

# dice_thr: (2) dice_thr=1 = drop only, ... (3) dice_thr=0 = full
print("Benchmarking dice_thr: (2) drop only (dice_thr=1) .. (3) full (dice_thr=0)...")
for dice_thr in dice_thr_values:
    sphere_all = merge_sphere_by_dice(
        data_low, gnl_genes, float(dice_thr), s_volume,
        tree_transcripts, transcript_coords, gene_per_transcript,
    )
    n, avg_all, avg_gnl = compute_metrics(sphere_all, "merge_dice", np.nan, np.nan, np.nan, float(dice_thr))
    gnl_str = f"{avg_gnl:.4f}" if not np.isnan(avg_gnl) else "n/a"
    print(f"  dice_thr = {dice_thr:.1f}: {n} detections | mean(# agg/transcript) all_genes = {avg_all:.4f}, gnl_only = {gnl_str}")


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
print("Scenarios spanned: (1) no_ops = no drop, no merge; (2) drop contained only (rho=0, gamma=1, jaccard_thr=1, dice_thr=1); (3) full (rho=1, gamma=0, jaccard_thr=0, dice_thr=0).")
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
print("  Mean > 1: many transcripts lie in multiple aggregates (cross-gene and within-gene).")
gamma_recs = [r for r in recs if r.get("scenario") == "merge_volume"]
if gamma_recs:
    gamma0 = next((r for r in gamma_recs if r.get("gamma") == 0.0), None)
    gamma1 = next((r for r in gamma_recs if r.get("gamma") == 1.0), None)
    if gamma0 is not None and gamma1 is not None:
        print(f"  gamma=0 vs gamma=1 (volume): {gamma0['num_detections']} -> {gamma1['num_detections']} detections.")
jaccard_recs = [r for r in recs if r.get("scenario") == "merge_jaccard"]
if jaccard_recs:
    j0 = next((r for r in jaccard_recs if r.get("jaccard_thr") == 0.0), None)
    j1 = next((r for r in jaccard_recs if r.get("jaccard_thr") == 1.0), None)
    if j0 is not None and j1 is not None:
        print(f"  jaccard_thr=0 vs 1 (Jaccard): {j0['num_detections']} -> {j1['num_detections']} detections.")
dice_recs = [r for r in recs if r.get("scenario") == "merge_dice"]
if dice_recs:
    d0 = next((r for r in dice_recs if r.get("dice_thr") == 0.0), None)
    d1 = next((r for r in dice_recs if r.get("dice_thr") == 1.0), None)
    if d0 is not None and d1 is not None:
        print(f"  dice_thr=0 vs 1 (Dice): {d0['num_detections']} -> {d1['num_detections']} detections.")