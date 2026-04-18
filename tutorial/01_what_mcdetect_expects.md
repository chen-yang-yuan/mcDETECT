# 1. What mcDETECT expects

[← Tutorial index](./README.md)

### 1.1 Platform type

| Value | Use case |
|--------|----------|
| `"discrete"` | Z is a small set of optical planes (MERSCOPE, CosMx). Unique `global_z` values define the internal z-grid. |
| `"continuous"` | Z is continuous (e.g. Xenium). |

**MERSCOPE:** use `"discrete"`.

### 1.2 Transcript table (`pandas.DataFrame`)

Each row is one transcript. The detector relies on:

| Column | Role |
|--------|------|
| `global_x`, `global_y`, `global_z` | Spatial coordinates (same units as `eps`, `grid_len`, sphere radii). |
| `target` | Gene name string; granule markers must appear here and in `gnl_genes`. |
| `overlaps_nucleus` | Boolean or 0/1; used to compute **in-soma ratio** per sphere. |
| `cell_id` | Optional; if you set `record_cell_id=True` in `dbscan` / `detect`, each sphere gets a modal `cell_id`. |

Negative controls are passed as **`nc_genes`** and matched via `target` when filtering is enabled.

**Next:** [Step A — Construct `mcDETECT`](./02_construct_mcdetect.md)
