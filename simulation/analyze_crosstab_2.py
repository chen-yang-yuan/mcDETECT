import glob
import os
import pandas as pd
from typing import Any, Dict, List


ROW_SHARE_THRESHOLD = 0.5


def _scalar_prob(prob: pd.DataFrame, row: Any, col: Any) -> float:
    v = prob.loc[row, col]
    if isinstance(v, pd.Series):
        v = v.iloc[0]
    return float(0.0 if pd.isna(v) else v)


def load_xtab(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def summarize_sphere_matches(xtab: pd.DataFrame, share_threshold: float = ROW_SHARE_THRESHOLD) -> Dict[str, int]:

    df = xtab.copy()
    prob = df.div(df.sum(axis=1), axis=0)

    all_cols = df.columns.tolist()
    good_count = 0

    for col in all_cols:
        col_vals = df[col]
        j = col_vals.idxmax()
        share = _scalar_prob(prob, j, col)
        if share > share_threshold:
            good_count += 1

    bad_count = df.shape[0] - good_count

    return {
        "no_match_or_low_purity": int(bad_count),
        "good_match_high_purity": int(good_count),
    }


def summarize_across_seeds(xtab_dir: str) -> pd.DataFrame:
    
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

        sphere_path = os.path.join(xtab_dir, f"{method_name}_sphere_scenarios_by_seed_4.csv")
        sphere_matrix.to_csv(sphere_path)

        print(f"[{method_name}] sphere per-seed counts saved to {sphere_path}")


if __name__ == "__main__":
    main()