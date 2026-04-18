# 6. Step E — Granule density comparison WT vs AD (`benchmark_subtyping.ipynb`)

[← Tutorial index](./README.md)

This step assumes each granule has **`granule_subtype_manual_simple`**, coordinates (**`global_x` / `global_y`** in **`obs`** after `profile`, or `sphere_*` before rename), and a **sample** column (**`"WT"`** / **`"AD"`** or batch names).

### Step E1 — Spatial reference: spots per sample

Load **`spots.h5ad`** for WT and AD separately. Each should expose **`brain_area`** and spot centroids **`global_x`**, **`global_y`** (after any sample-specific alignment used in your pipeline).

### Step E2 — Density definition (50 µm grid by default in the notebook helpers)

The helper **`compute_subtype_density_per_region`** (in the benchmark notebook) implements:

- For each **sample**, each **brain_area**, and each **subtype** (plus an **"overall"** row):
  - Sum over spots: for each spot center, count granules whose \((x,y)\) falls in a **square window** of half-width **`grid_len/2`** (default **`grid_len=50`**).
  - **Density** = (total granule–spot hits) / (**number of spots** in that brain area).

So density is “expected granules per spot” under that counting rule, not volume density in µm³.

### Step E3 — AD capture-efficiency correction

The notebook scales **AD** densities and per-spot counts by a fixed factor to compare to WT:

```python
CAPTURE_EFFICIENCY_COEF = 0.818691
# After computing AD densities or counts:
# density_ad = density_ad / CAPTURE_EFFICIENCY_COEF
```

Adjust or omit if your study does not use this calibration.

### Step E4 — Per-spot counts for statistics

**`compute_subtype_per_spot_counts`** builds one row per (sample, brain_area, subtype, spot) with the number of granules in that spot’s window. These streams feed:

- Bootstrap **95% CI** for mean density (optional loop in the notebook).
- **Welch t-test** on **`log1p(count)`** between WT and AD per (brain_area, subtype).
- **Bonferroni** and **Benjamini–Hochberg FDR** on p-values.

### Step E5 — Export

Results are merged into tables such as **`subtype_density_per_region_{setting_key}.csv`** and label Parquets (**`granule_subtype_labels_{setting_key}.parquet`**). Use the same **`setting_key`** string your benchmark loop uses for traceability.

**Next:** [Step F — Neuropil subdomains](./07_neuropil_subdomains.md)
