# 3. Step B — Granule detection (pattern from `3_detection.py`)

[← Tutorial index](./README.md)

### Step B1 — Load inputs (conceptual)

- Read **`transcripts`** (e.g. Parquet) with the columns required in [Section 1](./01_what_mcdetect_expects.md).
- Read **`gnl_genes`** list (synaptic markers) and **`nc_genes`** (negative controls CSV).

### Step B2 — Optional “rough” detection

**Goal:** See candidate granules without strict biological or NC filtering.

**Actions:**

1. Build `mcDETECT` with **`nc_genes=None`**, **`size_thr`** very large, **`in_soma_thr`** above 1.
2. Call **`sphere_dict = mc.dbscan(record_cell_id=True)`** if you need per-sphere `cell_id`.
3. Call **`granules_rough = mc.merge_sphere(sphere_dict)`**.
4. Save or inspect **`granules_rough`** (e.g. Parquet).

**Output shapes:**

- **`sphere_dict`:** `dict[int, pandas.DataFrame]` — one table per marker index (columns include `sphere_x`, `sphere_y`, `sphere_z`, `layer_z`, `sphere_r`, `size`, `comp`, `in_soma_ratio`, `gene`, optional `cell_id`).
- **`granules_rough`:** single **`DataFrame`** after cross-gene merge.

Example call sequence (rough configuration values match the idea in `3_detection.py`):

```python
mc_rough = mcDETECT(
    type="discrete",
    transcripts=transcripts,
    gnl_genes=syn_genes,
    nc_genes=None,
    eps=1.5,
    minspl=3,
    grid_len=1,
    cutoff_prob=0.95,
    alpha=10,
    low_bound=3,
    size_thr=1e5,
    in_soma_thr=1.01,
    l=1,
    rho=0.2,
    s=1,
    nc_top=20,
    nc_thr=0.1,
)

sphere_dict = mc_rough.dbscan(record_cell_id=True)
granules_rough = mc_rough.merge_sphere(sphere_dict)
```

### Step B3 — “Fine” detection (recommended)

**Goal:** Spheres with realistic maximum radius, low in-soma contamination, and NC filtering.

**Action:** Rebuild **`mcDETECT`** with **`nc_genes`**, **`size_thr=4.0`**, **`in_soma_thr=0.1`** (example values from `3_detection.py`), then:

```python
granules = mc.detect()
```

**Internal pipeline:** `dbscan` → `merge_sphere` → `nc_filter` (if `nc_genes` is not `None`).

| Method | Purpose | Main output |
|--------|---------|---------------|
| `dbscan(target_names=None, record_cell_id=False)` | 3D DBSCAN per marker, minimum enclosing sphere, feature filters | `dict[int, DataFrame]` |
| `merge_sphere(sphere_dict)` | Resolve overlaps between genes | `DataFrame` |
| `detect(record_cell_id=False)` | Full pipeline including NC filter | `DataFrame` |

### Step B4 — Optional region labels

If you have a spot/grid `AnnData` with `global_x`, `global_y`, and `brain_area`, you can assign each granule to the nearest spot’s region (e.g. `cKDTree` query) before saving. This is dataset-specific; see `3_detection.py` for the MERSCOPE pattern.

**Next:** [Step C — Profile granules](./04_profile_granules.md)
