import glob
import os
import pandas as pd
from typing import Dict, List


ROW_SHARE_THRESHOLD = 0.5


def load_xtab(path: str) -> pd.DataFrame:
    """
    Load a cross-table CSV saved by the simulation scripts.

    Assumes:
      - Rows: spheres + last row 'no_sphere'
      - Columns: GT aggregates + last column 'no_gt'
      - Index column saved in the first CSV column.
    """
    return pd.read_csv(path)


def summarize_sphere_matches(
    xtab: pd.DataFrame, share_threshold: float = ROW_SHARE_THRESHOLD
) -> Dict[str, int]:
    """
    Advisor-style simplification:
      - For each GT column, pick argmax sphere row among detection rows.
      - Keep the match only if purity share > 0.5.

    Purity share:
      share = count(sphere, gt) / sum_all_columns(sphere_row)

    Output categories over detection spheres:
      - no_match_or_low_purity
      - good_match_high_purity
    """
    # Detection sphere rows (exclude final "no_sphere" row).
    sph_rows = xtab.iloc[:-1, :].copy()
    # GT columns (exclude final "no_gt" column).
    gt_cols = xtab.columns[:-1].tolist()

    good_spheres = set()

    for gt_col in gt_cols:
        col_vals = sph_rows[gt_col]
        if float(col_vals.sum()) <= 0:
            continue

        sph_idx = col_vals.idxmax()
        row_sum = float(sph_rows.loc[sph_idx].sum())
        if row_sum <= 0:
            continue

        share = float(sph_rows.loc[sph_idx, gt_col]) / row_sum
        if share > share_threshold:
            good_spheres.add(sph_idx)

    total_spheres = int(sph_rows.shape[0])
    good_count = int(len(good_spheres))
    bad_count = int(total_spheres - good_count)

    return {
        "no_match_or_low_purity": bad_count,
        "good_match_high_purity": good_count,
    }


def summarize_across_seeds(xtab_dir: str) -> pd.DataFrame:
    """
    Read all seed xtabs under one method and compute per-seed sphere counts.

    Returns:
      - sphere_matrix: index = seed, columns = simplified sphere categories
    """
    pattern = os.path.join(xtab_dir, "multi_marker_3D_all_5000_1000_seed_*_xtab.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No cross-table CSVs found under {xtab_dir}")

    sphere_rows: List[pd.Series] = []

    for p in paths:
        base = os.path.basename(p)
        try:
            seed_str = base.split("_seed_")[1].split("_xtab")[0]
            seed = int(seed_str)
        except Exception:
            continue

        xtab = load_xtab(p)
        sphere_counts = summarize_sphere_matches(xtab)
        sphere_rows.append(pd.Series(sphere_counts, name=seed))

    sphere_matrix = pd.DataFrame(sphere_rows).sort_index()
    sphere_matrix = sphere_matrix.reindex(
        columns=["no_match_or_low_purity", "good_match_high_purity"],
        fill_value=0,
    )
    return sphere_matrix


def main():
    base_out = "output"
    methods = {
        "mcDETECT": os.path.join(base_out, "mcDETECT_output", "xtabs"),
        "Baysor_30_30": os.path.join(base_out, "Baysor_output_30_30", "xtabs"),
        "Baysor_30_1.5": os.path.join(base_out, "Baysor_output_30_1.5", "xtabs"),
        "SSAM_0_0": os.path.join(base_out, "SSAM_output_0_0", "xtabs"),
        "SSAM_0.2_0.027": os.path.join(base_out, "SSAM_output_0.2_0.027", "xtabs"),
    }

    for method_name, xtab_dir in methods.items():
        if not os.path.isdir(xtab_dir):
            print(f"[{method_name}] xtab directory not found, skipping: {xtab_dir}")
            continue

        print(f"[{method_name}] summarizing crosstabs in {xtab_dir} ...")
        sphere_matrix = summarize_across_seeds(xtab_dir)

        sphere_path = os.path.join(xtab_dir, f"{method_name}_sphere_scenarios_by_seed_3.csv")
        sphere_matrix.to_csv(sphere_path)

        print(f"[{method_name}] sphere per-seed counts saved to {sphere_path}")


if __name__ == "__main__":
    main()