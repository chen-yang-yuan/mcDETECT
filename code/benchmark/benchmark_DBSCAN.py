import os
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from mcDETECT.model import mcDETECT


# ==================== Paths and dataset ==================== #

dataset = "MERSCOPE_WT_1"
data_path = f"../../data/{dataset}/"
benchmark_path = "../../output/benchmark/benchmark_DBSCAN/"
os.makedirs(benchmark_path, exist_ok=True)


# ==================== Load data ==================== #

# Transcripts
transcripts = pd.read_parquet(os.path.join(data_path, "processed_data", "transcripts_small_region.parquet"))

# Genes (not strictly needed here but kept for completeness/consistency)
genes = pd.read_csv(os.path.join(data_path, "processed_data", "genes.csv"))
genes = list(genes.iloc[:, 0])

# Negative control markers
nc_genes = pd.read_csv(os.path.join(data_path, "processed_data", "negative_controls.csv"))
nc_genes = list(nc_genes["Gene"])

# Precompute NC transcripts + 3D tree once (reused across all runs)
nc_trans = transcripts[transcripts["target"].isin(nc_genes)]
if nc_trans.shape[0] > 0:
    NC_TREE = cKDTree(
        np.c_[
            nc_trans["global_x"].to_numpy(),
            nc_trans["global_y"].to_numpy(),
            nc_trans["global_z"].to_numpy(),
        ]
    )
else:
    NC_TREE = None

# Granule marker genes (same as syn_genes in code/2_detection.py)
gnl_genes = [
    "Camk2a",
    "Cplx2",
    "Slc17a7",
    "Ddn",
    "Syp",
    "Map1a",
    "Shank1",
    "Syn1",
    "Gria1",
    "Gria2",
    "Cyfip2",
    "Vamp2",
    "Bsn",
    "Slc32a1",
    "Nfasc",
    "Syt1",
    "Tubb3",
    "Nav1",
    "Shank3",
    "Mapt",
]


# ==================== Helper functions ==================== #

def make_tree_3d(d1, d2, d3):
    """Build 3D cKDTree from coordinate arrays (similar to benchmark_filtering.ipynb)."""
    points = np.c_[np.ravel(d1), np.ravel(d2), np.ravel(d3)]
    return cKDTree(points)


def compute_nc_ratio(granules_df):
    """
    Per granule: nc_ratio = (NC transcript count in sphere) / size.
    Uses precomputed NC_TREE on all negative-control transcripts.
    """
    if NC_TREE is None or granules_df.shape[0] == 0:
        return np.zeros(granules_df.shape[0], dtype=float)

    centers = granules_df[["sphere_x", "sphere_y", "layer_z"]].to_numpy()
    radii = granules_df["sphere_r"].to_numpy()
    sizes = granules_df["size"].to_numpy().astype(float)

    counts = np.array([len(NC_TREE.query_ball_point(c, r)) for c, r in zip(centers, radii)])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(sizes > 0, counts / sizes, 0.0)
    return ratios


def run_detection(eps, minspl):
    """
    Run mcDETECT rough detection with given DBSCAN parameters.
    Downstream filters (size_thr, in_soma_thr, nc_genes) are effectively disabled.
    """
    mc = mcDETECT(
        type="discrete",
        transcripts=transcripts,
        gnl_genes=gnl_genes,
        nc_genes=None,      # turn off negative control filtering during detection
        eps=eps,
        minspl=minspl,
        grid_len=1.0,
        cutoff_prob=0.95,
        alpha=10.0,
        low_bound=3,
        size_thr=1e5,       # effectively no size filtering
        in_soma_thr=1.01,   # effectively no in-soma filtering
        l=1.0,
        rho=0.2,
        s=1.0,
        nc_top=20,
        nc_thr=0.1,
    )

    sphere_dict = mc.dbscan(record_cell_id=False)
    granules = mc.merge_sphere(sphere_dict)
    granules = granules.reset_index(drop=True)
    return granules


def summarize_run(granules_df, eps, minspl, scenario):
    """Compute summary statistics for one DBSCAN configuration."""
    n = granules_df.shape[0]
    if n == 0:
        nc_ratio = np.array([], dtype=float)
    else:
        # Add nc_ratio in-place to avoid copying the full DataFrame
        nc_ratio = compute_nc_ratio(granules_df)
        granules_df["nc_ratio"] = nc_ratio

    mean_aggregate_radius = granules_df["sphere_r"].mean() if n else np.nan
    mean_in_soma_ratio = granules_df["in_soma_ratio"].mean() if n else np.nan
    mean_nc_ratio = nc_ratio.mean() if n else np.nan

    return {
        "dataset": dataset,
        "scenario": scenario,
        "eps": eps,
        "minspl": minspl,
        "n_detections": int(n),
        "mean_aggregate_radius": float(mean_aggregate_radius) if not np.isnan(mean_aggregate_radius) else np.nan,
        "mean_in_soma_ratio": float(mean_in_soma_ratio) if not np.isnan(mean_in_soma_ratio) else np.nan,
        "mean_nc_ratio": float(mean_nc_ratio) if not np.isnan(mean_nc_ratio) else np.nan,
    }, granules_df


def build_configs(args):
    """
    Build list of (scenario, eps, minspl) configs.
    If --eps and --minspl are provided, run only that pair; otherwise run full sweep.
    """
    configs = []
    if args.eps is not None and args.minspl is not None:
        scenario = args.scenario if args.scenario is not None else "custom"
        configs.append((scenario, float(args.eps), int(args.minspl)))
    else:
        # Baseline configuration (current defaults)
        configs.append(("baseline", 1.5, 3))
        # Sweep eps while fixing minspl at default (3)
        for eps_val in [1.0, 2.0, 2.5, 5.0]:
            configs.append(("eps_sweep", eps_val, 3))
        # Sweep minspl while fixing eps at default (1.5)
        for minspl_val in [4, 5]:
            configs.append(("minspl_sweep", 1.5, minspl_val))
    return configs


def main():
    parser = argparse.ArgumentParser(description="Benchmark mcDETECT DBSCAN parameters.")
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="If set together with --minspl, run only this eps/minspl configuration.",
    )
    parser.add_argument(
        "--minspl",
        type=int,
        default=None,
        help="If set together with --eps, run only this eps/minspl configuration.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Optional scenario label when using --eps/--minspl (default: 'custom').",
    )
    args = parser.parse_args()

    results = []
    configs = build_configs(args)

    for scenario, eps_val, minspl_val in configs:
        print(f"Running DBSCAN benchmark: scenario={scenario}, eps={eps_val}, minspl={minspl_val}")
        granules = run_detection(eps=eps_val, minspl=minspl_val)

        summary_row, granules_with_nc = summarize_run(granules, eps_val, minspl_val, scenario)
        results.append(summary_row)

        # Save per-run granules with DBSCAN parameters in filename
        fname = f"{dataset}_granules_eps{eps_val}_minspl{minspl_val}.parquet"
        granules_with_nc.to_parquet(os.path.join(benchmark_path, fname), index=False)
        print(f"  Saved granules to {os.path.join(benchmark_path, fname)} (n={granules_with_nc.shape[0]})")

    # Save all summary statistics into a single CSV file
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(benchmark_path, f"{dataset}_benchmark_DBSCAN_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Saved DBSCAN benchmark summary to {results_csv}")


if __name__ == "__main__":
    main()