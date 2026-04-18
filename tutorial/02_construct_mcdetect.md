# 2. Step A — Construct `mcDETECT` and understand every parameter

[← Tutorial index](./README.md)

**Step A1.** Import the class:

```python
from mcDETECT.model import mcDETECT
```

**Step A2.** Read the parameter list below in order. Each entry matches one argument of `mcDETECT(...)`.

1. **`type`** — `"discrete"` or `"continuous"`; see [Section 1](./01_what_mcdetect_expects.md).
2. **`transcripts`** — Full transcript `DataFrame`; column requirements are in [Section 1](./01_what_mcdetect_expects.md).
3. **`gnl_genes`** — `list[str]`: all granule / synaptic marker genes to run DBSCAN on.
4. **`nc_genes`** — `list[str]` or **`None`**. If **`None`**, `detect()` skips negative-control filtering. Otherwise `nc_filter` runs at the end of `detect()`.
5. **`eps`** — DBSCAN neighborhood radius in coordinate units (3D).
6. **`minspl`** — If **`None`**, `poisson_select(gene)` sets per-gene `min_samples` from tissue area and background density. If an **integer**, that value is used for all genes (as in `3_detection.py` with `minspl=3`).
7. **`grid_len`** — Bin width when estimating tissue area for the Poisson model (not the spot grid in downstream notebooks).
8. **`cutoff_prob`** — Quantile for the Poisson background (used when `minspl is None`).
9. **`alpha`** — Multiplier on expected local density inside `π eps²` when selecting `min_samples`.
10. **`low_bound`** — Lower floor for automatic `min_samples` and minimum number of **unique** transcript positions per DBSCAN cluster before the cluster is skipped.
11. **`size_thr`** — Discard spheres whose minimum enclosing ball has **radius ≥ `size_thr`**. Use a very large value (e.g. `1e5`) for a “no radius cap” exploratory run.
12. **`in_soma_thr`** — Discard spheres with **in-soma ratio ≥ `in_soma_thr`**. Use a value slightly above 1 (e.g. `1.01`) to effectively disable.
13. **`l`** — Scales center–center distance vs. sum of radii when resolving overlaps between spheres from **different** genes.
14. **`rho`** — Threshold pairing with `l` for deciding when two spheres “intersect” vs. nest for merge logic in `merge_sphere`.
15. **`s`** — Scales the radius of a **new** sphere after merging two overlapping spheres (miniball over combined points).
16. **`nc_top`** — When `nc_genes` is set, only the top **`nc_top`** negative-control genes by transcript count enter the NC filter.
17. **`nc_thr`** — Maximum allowed ratio of NC transcripts to total transcripts inside the sphere; spheres above this are removed. Use **`None`** to keep all spheres but still compute `nc_ratio` if applicable.
18. **`merge_genes`** — If **`True`**, granule markers are collapsed to a single pseudo-marker column for detection (see `model.py`).
19. **`merged_gene_label`** — Label used for that pseudo-marker when `merge_genes=True`.

**Step A3.** Instantiate without inline comments — parameters are documented above:

```python
mc = mcDETECT(
    type="discrete",
    transcripts=transcripts,
    gnl_genes=syn_genes,
    nc_genes=nc_genes,
    eps=1.5,
    minspl=3,
    grid_len=1,
    cutoff_prob=0.95,
    alpha=10,
    low_bound=3,
    size_thr=4.0,
    in_soma_thr=0.1,
    l=1,
    rho=0.2,
    s=1,
    nc_top=20,
    nc_thr=0.1,
    merge_genes=False,
    merged_gene_label="merged",
)
```

The same hyperparameter names apply to **rough** vs **fine** runs; only the values of `nc_genes`, `size_thr`, `in_soma_thr`, and related flags change; see [Section 3](./03_granule_detection.md).

**Next:** [Step B — Granule detection](./03_granule_detection.md)
