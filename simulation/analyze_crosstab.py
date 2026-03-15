import os
import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


TAU_P = 0.5
TAU_C = 0.5


def load_xtab(path: str) -> pd.DataFrame:
    """
    Load a cross-table CSV saved by the simulation scripts.

    Assumes:
      - Rows: spheres + last row 'no_sphere'
      - Columns: GT aggregates + last column 'no_gt'
      - Index column saved in the first CSV column.
    """
    # Match the notebook behavior: read the CSV as-is (no special index column),
    # so rows are 0..N-1 and columns include all GT labels plus 'no_gt'.
    return pd.read_csv(path)


def compute_purity_completeness(xtab: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Given a cross-table xtab, return:
      - xtab_det: sphere × GT table (excluding 'no_sphere' and 'no_gt')
      - purity:  row-normalized xtab_det  (per-sphere purity to each GT)
      - completeness: column-normalized xtab_det (per-GT completeness from each sphere)
    """
    # Strip background row/col (last ones) to get detections × GT aggregates
    xtab_det = xtab.iloc[:-1, :-1].copy()  # rows: spheres, cols: GTs

    row_sums = xtab_det.sum(axis=1)  # total transcripts per sphere (GT only)
    col_sums = xtab_det.sum(axis=0)  # total transcripts per GT (over all spheres)

    # Avoid division by zero
    row_sums_safe = row_sums.replace(0, np.nan)
    col_sums_safe = col_sums.replace(0, np.nan)

    purity = xtab_det.div(row_sums_safe, axis=0)
    completeness = xtab_det.div(col_sums_safe, axis=1)

    return xtab_det, purity, completeness


def summarize_gt_scenarios(xtab: pd.DataFrame) -> Dict[str, int]:
    """
    For one seed, compute counts of GT-side scenarios based on the cross-table.

    Scenarios (mutually exclusive, exhaustive over GTs):
      - missed
      - good_single_match
      - good_multi_sphere
      - high_p_low_c
      - high_c_low_p
      - low_p_low_c
    """
    xtab_det, purity, completeness = compute_purity_completeness(xtab)

    gt_labels = xtab_det.columns.tolist()
    counts: Dict[str, int] = {
        "missed": 0,
        "good_single_match": 0,
        "good_multi_sphere": 0,
        "high_p_low_c": 0,
        "high_c_low_p": 0,
        "low_p_low_c": 0,
    }

    for gt_col in gt_labels:
        counts_col = xtab_det[gt_col]
        total_gt_inside_spheres = counts_col.sum()
        n_spheres_touch = int((counts_col > 0).sum())

        # Include GT mass in no_sphere as well when checking for degeneracy
        total_gt = int(xtab[gt_col].sum())

        if total_gt == 0:
            # Degenerate GT with zero transcripts; treat as "missed" for counting purposes
            scenario = "missed"
        else:
            # Best sphere by F1 over purity/completeness (among spheres that touch this GT)
            p_col = purity[gt_col]
            c_col = completeness[gt_col]
            f1_col = 2 * p_col * c_col / (p_col + c_col)

            if n_spheres_touch == 0:
                # All GT mass is in no_sphere row
                scenario = "missed"
            else:
                best_idx = f1_col.idxmax()
                # In case of duplicated index labels, .loc may return a Series.
                # We coerce to float after filling NaNs with 0 for robustness.
                best_p_val = p_col.loc[best_idx]
                best_c_val = c_col.loc[best_idx]
                if isinstance(best_p_val, pd.Series):
                    best_p_val = best_p_val.iloc[0]
                if isinstance(best_c_val, pd.Series):
                    best_c_val = best_c_val.iloc[0]
                best_p = float(0.0 if pd.isna(best_p_val) else best_p_val)
                best_c = float(0.0 if pd.isna(best_c_val) else best_c_val)

                p_pass = best_p >= TAU_P
                c_pass = best_c >= TAU_C

                if p_pass and c_pass and n_spheres_touch == 1:
                    scenario = "good_single_match"
                elif p_pass and c_pass and n_spheres_touch > 1:
                    scenario = "good_multi_sphere"
                elif p_pass and not c_pass:
                    scenario = "high_p_low_c"
                elif (not p_pass) and c_pass:
                    scenario = "high_c_low_p"
                else:
                    scenario = "low_p_low_c"

        counts[scenario] += 1

    return counts


def summarize_sphere_scenarios(xtab: pd.DataFrame) -> Dict[str, int]:
    """
    For one seed, compute counts of detection-side scenarios based on the cross-table.

    Scenarios (mutually exclusive, exhaustive over spheres), matching
    the logic in env/analyze_crosstab.ipynb:
      - background_sphere
      - good_sphere
      - good_but_mixed
      - mixed_gt_sphere
      - low_quality_sphere
    """
    xtab_det, purity, completeness = compute_purity_completeness(xtab)

    sphere_labels = xtab_det.index.tolist()
    counts: Dict[str, int] = {
        "background_sphere": 0,
        "good_sphere": 0,
        "good_but_mixed": 0,
        "mixed_gt_sphere": 0,
        "low_quality_sphere": 0,
    }

    for sph_label in sphere_labels:
        row = xtab_det.loc[sph_label]
        # If index labels are duplicated and weren't aggregated for some reason,
        # `.loc` can return a DataFrame. Collapse it to a single Series by summing.
        if isinstance(row, pd.DataFrame):
            row = row.sum(axis=0)
        total_sphere = int(row.sum())
        n_gts_touch = int((row > 0).sum())

        if total_sphere == 0:
            # Empty sphere (rare) or nonempty sphere containing only background (no_gt),
            # since gt-only mass is zero.
            scenario = "background_sphere"
        else:
            p_row = purity.loc[sph_label]
            c_row = completeness.loc[sph_label]
            f1_row = 2 * p_row * c_row / (p_row + c_row)
            best_gt = f1_row.idxmax()
            best_p = float(p_row.loc[best_gt]) if not pd.isna(p_row.loc[best_gt]) else 0.0
            best_c = float(c_row.loc[best_gt]) if not pd.isna(c_row.loc[best_gt]) else 0.0

            mass_to_gt = total_sphere
            mass_to_no_gt = int(xtab.loc[sph_label, "no_gt"]) if "no_gt" in xtab.columns else 0
            # frac_no_gt is not used for scenario branching here, but could be logged if needed
            # frac_no_gt = mass_to_no_gt / (mass_to_gt + mass_to_no_gt) if (mass_to_gt + mass_to_no_gt) > 0 else 0.0

            p_pass = best_p >= TAU_P
            c_pass = best_c >= TAU_C

            if mass_to_gt == 0 and mass_to_no_gt > 0:
                scenario = "background_sphere"
            elif p_pass and c_pass and n_gts_touch == 1:
                scenario = "good_sphere"
            elif p_pass and c_pass and n_gts_touch > 1:
                scenario = "good_but_mixed"
            elif n_gts_touch > 1 and not p_pass:
                scenario = "mixed_gt_sphere"
            else:
                scenario = "low_quality_sphere"

        counts[scenario] += 1

    return counts


def summarize_across_seeds(
    xtab_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a given xtab directory, read all per-seed cross-tables and compute
    per-seed scenario counts for:
      - GT side
      - Sphere side

    Returns two DataFrames:
      - gt_matrix:    index = seed, columns = GT scenarios
      - sphere_matrix: index = seed, columns = sphere scenarios
    """
    pattern = os.path.join(
        xtab_dir, "multi_marker_3D_all_5000_1000_seed_*_xtab.csv"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No cross-table CSVs found under {xtab_dir}")

    gt_rows: List[pd.Series] = []
    sphere_rows: List[pd.Series] = []
    seeds: List[int] = []

    for p in paths:
        # Extract seed from filename
        base = os.path.basename(p)
        # Expect ..._seed_{N}_xtab.csv
        try:
            seed_str = base.split("_seed_")[1].split("_xtab")[0]
            seed = int(seed_str)
        except Exception:
            # Fallback: ignore files that don't match pattern well
            continue

        xtab = load_xtab(p)

        gt_counts = summarize_gt_scenarios(xtab)
        sphere_counts = summarize_sphere_scenarios(xtab)

        gt_rows.append(pd.Series(gt_counts, name=seed))
        sphere_rows.append(pd.Series(sphere_counts, name=seed))
        seeds.append(seed)

    gt_matrix = pd.DataFrame(gt_rows).sort_index()
    sphere_matrix = pd.DataFrame(sphere_rows).sort_index()

    # Ensure consistent column order
    gt_matrix = gt_matrix.reindex(
        columns=[
            "missed",
            "good_single_match",
            "good_multi_sphere",
            "high_p_low_c",
            "high_c_low_p",
            "low_p_low_c",
        ],
        fill_value=0,
    )
    sphere_matrix = sphere_matrix.reindex(
        columns=[
            "background_sphere",
            "good_sphere",
            "good_but_mixed",
            "mixed_gt_sphere",
            "low_quality_sphere",
        ],
        fill_value=0,
    )

    return gt_matrix, sphere_matrix


def main():
    base_out = "output"
    methods = {
        # "mcDETECT": os.path.join(base_out, "mcDETECT_output", "xtabs"),
        "Baysor": os.path.join(base_out, "Baysor_output", "xtabs"),
        # "SSAM": os.path.join(base_out, "SSAM_output", "xtabs"),
    }

    for method_name, xtab_dir in methods.items():
        if not os.path.isdir(xtab_dir):
            print(f"[{method_name}] xtab directory not found, skipping: {xtab_dir}")
            continue

        print(f"[{method_name}] summarizing crosstabs in {xtab_dir} ...")
        gt_matrix, sphere_matrix = summarize_across_seeds(xtab_dir)

        gt_path = os.path.join(xtab_dir, f"{method_name}_gt_scenarios_by_seed.csv")
        sphere_path = os.path.join(xtab_dir, f"{method_name}_sphere_scenarios_by_seed.csv")

        gt_matrix.to_csv(gt_path)
        sphere_matrix.to_csv(sphere_path)

        print(f"[{method_name}] GT per-seed scenario counts saved to {gt_path}")
        print(f"[{method_name}] sphere per-seed scenario counts saved to {sphere_path}")


if __name__ == "__main__":
    main()