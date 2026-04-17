# mcDETECT tutorial: MERSCOPE analysis (step by step)

This tutorial follows the repository workflow for **MERSCOPE** data: transcript-based `**mcDETECT`** setup → granule detection → expression profiling → **manual** granule subtyping → **WT vs AD granule density** comparison → **Isocortex neuropil subdomain** analysis on a **50×50** grid with **hard** embeddings, **row-normalized** subtype vectors, and **K-Means with k = 4**.

Primary references: `mcDETECT_package/mcDETECT/model.py`, `code/3_detection.py`, `code/benchmark/benchmark_subtyping.ipynb`, `code/7_neuropil_subdomains.ipynb`.

---

## Analysis roadmap

1. **Prepare** transcript and auxiliary tables (coordinates, gene names, nucleus overlap, optional `cell_id`).
2. **Initialize** `mcDETECT` with platform type, marker lists, and numerical hyperparameters (see §2).
3. **Detect granules** — optional rough pass, then fine pass with filters (`§3`).
4. **Profile** granules into an `AnnData` count matrix (`§4`).
5. **Subtype manually** — normalize expression, k-means, heatmap, map clusters to biology, derive a simple mixed/pure label (`§5`).
6. **Compare density** WT vs AD by brain region using spots as spatial units (`§6`).
7. **Neuropil subdomains (Isocortex, 50×50)** — hard `spot_embedding`, row-wise normalization, K-Means **k = 4** (`§7`).

Dataset-specific coordinate transforms and multi-setting benchmark loops are omitted here.

---

## 1. What mcDETECT expects

### 1.1 Platform type


| Value          | Use case                                                                                                   |
| -------------- | ---------------------------------------------------------------------------------------------------------- |
| `"discrete"`   | Z is a small set of optical planes (MERSCOPE, CosMx). Unique `global_z` values define the internal z-grid. |
| `"continuous"` | Z is continuous (e.g. Xenium).                                                                             |


**MERSCOPE:** use `"discrete"`.

### 1.2 Transcript table (`pandas.DataFrame`)

Each row is one transcript. The detector relies on:


| Column                             | Role                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `global_x`, `global_y`, `global_z` | Spatial coordinates (same units as `eps`, `grid_len`, sphere radii).                                   |
| `target`                           | Gene name string; granule markers must appear here and in `gnl_genes`.                                 |
| `overlaps_nucleus`                 | Boolean or 0/1; used to compute **in-soma ratio** per sphere.                                          |
| `cell_id`                          | Optional; if you set `record_cell_id=True` in `dbscan` / `detect`, each sphere gets a modal `cell_id`. |


Negative controls are passed as `**nc_genes`** and matched via `target` when filtering is enabled.

---

## 2. Step A — Construct `mcDETECT` and understand every parameter

**Step A1.** Import the class:

```python
from mcDETECT.model import mcDETECT
```

**Step A2.** Read the parameter list below in order. Each entry matches one argument of `mcDETECT(...)`.

1. `**type`** — `"discrete"` or `"continuous"` (§1.1).
2. `**transcripts**` — Full transcript `DataFrame` (§1.2).
3. `**gnl_genes**` — `list[str]`: all granule / synaptic marker genes to run DBSCAN on.
4. `**nc_genes**` — `list[str]` or `**None**`. If `**None**`, `detect()` skips negative-control filtering. Otherwise `nc_filter` runs at the end of `detect()`.
5. `**eps**` — DBSCAN neighborhood radius in coordinate units (3D).
6. `**minspl**` — If `**None**`, `poisson_select(gene)` sets per-gene `min_samples` from tissue area and background density. If an **integer**, that value is used for all genes (as in `3_detection.py` with `minspl=3`).
7. `**grid_len`** — Bin width when estimating tissue area for the Poisson model (not the spot grid in downstream notebooks).
8. `**cutoff_prob**` — Quantile for the Poisson background (used when `minspl is None`).
9. `**alpha**` — Multiplier on expected local density inside `π eps²` when selecting `min_samples`.
10. `**low_bound**` — Lower floor for automatic `min_samples` and minimum number of **unique** transcript positions per DBSCAN cluster before the cluster is skipped.
11. `**size_thr`** — Discard spheres whose minimum enclosing ball has **radius ≥ `size_thr`**. Use a very large value (e.g. `1e5`) for a “no radius cap” exploratory run.
12. `**in_soma_thr**` — Discard spheres with **in-soma ratio ≥ `in_soma_thr`**. Use a value slightly above 1 (e.g. `1.01`) to effectively disable.
13. `**l**` — Scales center–center distance vs. sum of radii when resolving overlaps between spheres from **different** genes.
14. `**rho`** — Threshold pairing with `l` for deciding when two spheres “intersect” vs. nest for merge logic in `merge_sphere`.
15. `**s**` — Scales the radius of a **new** sphere after merging two overlapping spheres (miniball over combined points).
16. `**nc_top`** — When `nc_genes` is set, only the top `**nc_top**` negative-control genes by transcript count enter the NC filter.
17. `**nc_thr**` — Maximum allowed ratio of NC transcripts to total transcripts inside the sphere; spheres above this are removed. Use `**None**` to keep all spheres but still compute `nc_ratio` if applicable.
18. `**merge_genes**` — If `**True**`, granule markers are collapsed to a single pseudo-marker column for detection (see `model.py`).
19. `**merged_gene_label**` — Label used for that pseudo-marker when `merge_genes=True`.

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

The same hyperparameter names apply to **rough** vs **fine** runs; only the values of `nc_genes`, `size_thr`, `in_soma_thr`, and related flags change (§3).

---

## 3. Step B — Granule detection (pattern from `3_detection.py`)

### Step B1 — Load inputs (conceptual)

- Read `**transcripts`** (e.g. Parquet) with columns from §1.2.
- Read `**gnl_genes**` list (synaptic markers) and `**nc_genes**` (negative controls CSV).

### Step B2 — Optional “rough” detection

**Goal:** See candidate granules without strict biological or NC filtering.

**Actions:**

1. Build `mcDETECT` with `**nc_genes=None`**, `**size_thr**` very large, `**in_soma_thr**` above 1.
2. Call `**sphere_dict = mc.dbscan(record_cell_id=True)**` if you need per-sphere `cell_id`.
3. Call `**granules_rough = mc.merge_sphere(sphere_dict)**`.
4. Save or inspect `**granules_rough**` (e.g. Parquet).

**Output shapes:**

- `**sphere_dict`:** `dict[int, pandas.DataFrame]` — one table per marker index (columns include `sphere_x`, `sphere_y`, `sphere_z`, `layer_z`, `sphere_r`, `size`, `comp`, `in_soma_ratio`, `gene`, optional `cell_id`).
- `**granules_rough`:** single `**DataFrame`** after cross-gene merge.

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

**Action:** Rebuild `**mcDETECT`** with `**nc_genes**`, `**size_thr=4.0**`, `**in_soma_thr=0.1**` (example values from `3_detection.py`), then:

```python
granules = mc.detect()
```

**Internal pipeline:** `dbscan` → `merge_sphere` → `nc_filter` (if `nc_genes` is not `None`).


| Method                                            | Purpose                                                         | Main output            |
| ------------------------------------------------- | --------------------------------------------------------------- | ---------------------- |
| `dbscan(target_names=None, record_cell_id=False)` | 3D DBSCAN per marker, minimum enclosing sphere, feature filters | `dict[int, DataFrame]` |
| `merge_sphere(sphere_dict)`                       | Resolve overlaps between genes                                  | `DataFrame`            |
| `detect(record_cell_id=False)`                    | Full pipeline including NC filter                               | `DataFrame`            |


### Step B4 — Optional region labels

If you have a spot/grid `AnnData` with `global_x`, `global_y`, and `brain_area`, you can assign each granule to the nearest spot’s region (e.g. `cKDTree` query) before saving. This is dataset-specific; see `3_detection.py` for the MERSCOPE pattern.

---

## 4. Step C — Profile granules (`profile`)

**Step C1.** Ensure the same `**mc`** instance still holds `**transcripts**` used for detection.

**Step C2.** Choose the gene list `**genes`** (full panel or a subset).

**Step C3.** Call:

```python
granule_adata = mc.profile(
    granules,
    genes=genes,
    buffer=0.0,
    print_itr=False,
    print_itr_interval=5000,
)
```


| Argument   | Meaning                                                             |
| ---------- | ------------------------------------------------------------------- |
| `granules` | `DataFrame` from `detect()` (or merged rough table).                |
| `genes`    | Genes to count; `**None**` uses all genes present in `transcripts`. |
| `buffer`   | Added to each sphere’s radius when querying transcripts.            |


**Output:** `**anndata.AnnData`** — sparse `**X**`: `n_granules × n_genes`; `**obs**`: granule metadata with coordinates renamed to `global_x` / `global_y` / `global_z` and `**granule_id**` added.

**Step C4.** Typical Scanpy QC for visualization (as in `3_detection.py`): copy counts to a layer, `normalize_total`, `log1p`, `pca`, `tsne`.

---

## 5. Step D — Manual granule subtyping only (`benchmark_subtyping.ipynb`)

Automated rule-based subtyping (`classify_granules`) is **not** covered here; the benchmark notebook’s primary annotation path is **manual**.

### Step D1 — Normalize for clustering

After `**profile`**, store raw counts in `**layers["counts"]**`, then apply `**sc.pp.normalize_total**` and `**sc.pp.log1p**` on `**X**` (or follow your notebook’s exact normalization for consistency with saved `h5ad` files).

### Step D2 — Choose k and run MiniBatch k-means

Fix `**n_clusters**` (e.g. 15 in the benchmark), `**seed**`, `**batch_size**`, `**n_init**`. Fit on the matrix used for clustering (dense `X` if sparse).

```python
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

def run_manual_subtyping(granule_adata, n_clusters, seed, batch_size=5000, n_init=20, obs_key="granule_subtype_kmeans"):
    data = granule_adata.X.copy()
    if hasattr(data, "toarray"):
        data = data.toarray()
    np.random.seed(seed)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=seed,
        n_init=n_init,
    )
    kmeans.fit(data)
    granule_adata.obs[obs_key] = kmeans.labels_.astype(str)
    granule_adata.obs[obs_key] = pd.Categorical(
        granule_adata.obs[obs_key],
        categories=[str(i) for i in range(n_clusters)],
        ordered=True,
    )
    return granule_adata
```

**Output:** `**obs[obs_key]`** — string cluster ids `"0"`, `"1"`, ….

### Step D3 — Heatmap-driven biology

1. Pick a **reference gene list** (e.g. synaptic markers overlapping `var_names`).
2. Plot `**scanpy.pl.heatmap`** with `**groupby=obs_key**`, `**standard_scale="var"**`, to see which clusters look pre-synaptic, post-synaptic, dendritic, mixed, etc.

### Step D4 — Manual mapping dictionary

Build a `**mapping**` from biological subtype names to **lists of cluster id strings**:

```python
def apply_manual_annotation(granule_adata, mapping, cluster_column="granule_subtype_kmeans"):
    k2sub = {}
    for subtype, clusters in mapping.items():
        for c in clusters:
            k2sub[c] = subtype
    granule_adata.obs["granule_subtype_manual"] = (
        granule_adata.obs[cluster_column].astype(str).map(k2sub)
    )
    granule_adata.obs["granule_subtype_manual_simple"] = granule_adata.obs["granule_subtype_manual"].apply(
        lambda s: "mixed" if pd.notna(s) and " & " in str(s) else str(s)
    )
    return granule_adata
```

**Convention:** finer labels live in `**granule_subtype_manual`** (e.g. `"pre & post"`); `**granule_subtype_manual_simple**` collapses any label containing `**" & "**` to `**"mixed"**` for density and summaries.

### Step D5 — Paired WT + AD objects (if applicable)

For cross-sample workflows, concatenate WT and AD `**granule_adata**` objects, restrict to common genes, normalize, run k-means once on the combined matrix, then annotate with a single `**MANUAL_SUBTYPE_MAPPING**` keyed by filename or setting. The benchmark notebook uses `**obs["sample"]**` in `**("WT", "AD")**` or batch labels.

---

## 6. Step E — Granule density comparison WT vs AD (`benchmark_subtyping.ipynb`)

This step assumes each granule has `**granule_subtype_manual_simple**`, coordinates (`**global_x` / `global_y**` in `**obs**` after `profile`, or `sphere_*` before rename), and a **sample** column (`**"WT"`** / `**"AD"**` or batch names).

### Step E1 — Spatial reference: spots per sample

Load `**spots.h5ad**` for WT and AD separately. Each should expose `**brain_area**` and spot centroids `**global_x**`, `**global_y**` (after any sample-specific alignment used in your pipeline).

### Step E2 — Density definition (50 µm grid by default in the notebook helpers)

The helper `**compute_subtype_density_per_region**` (in the benchmark notebook) implements:

- For each **sample**, each **brain_area**, and each **subtype** (plus an **"overall"** row):
  - Sum over spots: for each spot center, count granules whose (x,y) falls in a **square window** of half-width `**grid_len/2`** (default `**grid_len=50**`).
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

`**compute_subtype_per_spot_counts**` builds one row per (sample, brain_area, subtype, spot) with the number of granules in that spot’s window. These streams feed:

- Bootstrap **95% CI** for mean density (optional loop in the notebook).
- **Welch t-test** on `**log1p(count)`** between WT and AD per (brain_area, subtype).
- **Bonferroni** and **Benjamini–Hochberg FDR** on p-values.

### Step E5 — Export

Results are merged into tables such as `**subtype_density_per_region_{setting_key}.csv`** and label Parquets (`**granule_subtype_labels_{setting_key}.parquet**`). Use the same `**setting_key**` string your benchmark loop uses for traceability.

---

## 7. Step F — Neuropil subdomain analysis, Isocortex (`7_neuropil_subdomains.ipynb`)

The following is the **final-used** setting: **Isocortex**, **50×50** µm grids, **hard** `spot_embedding`, **row-normalized** subtype embedding, **K-Means with k = 4**. Soft embeddings, 25×25 grids, unnormalized embeddings, inertia/ARI sweeps, LDA/GMM/MiniBatch variants, and **k = 5** are omitted.

### Step F1 — Region and grid constants

```python
ROI = "Isocortex"
grid_len = "50_by_50"
grid_len_num = 50
```

### Step F2 — Inputs (prepared elsewhere in the pipeline)

You need consistent objects under your comparison output directory (paths in the notebook are relative to `code/`):

- `**spots**` — Combined WT + AD spot `AnnData` for Isocortex with `**global_x**`, `**global_y**`, `**layer_labels**`, `**batch**`, and aligned coordinates (the notebook uses `**recover_spots**` for 50×50 to attach `**layer_labels**` from a reference `h5ad`).
- `**adata**` — Cell-level `AnnData` (e.g. `**neuropil_subdomains_adata.h5ad**`).
- `**granule_adata**` — Granule `AnnData` restricted to the neuropil workflow (e.g. `**neuropil_subdomains_granule_adata.h5ad**`) with `**granule_subtype_kmeans**` in `**obs**` and raw counts in `**layers["counts"]**`.
- `**count_matrix**` — Per-cell gene counts with `**cell_id**`, gene columns aligned to `**granule_adata.var_names**`.

**Neurons for soma features:** subset cells to Isocortex and excitatory/inhibitory types, e.g. `brain_area` matching `**ROI`** and `**cell_type**` in `["Glutamatergic", "GABAergic"]` → `**adata_neuron**`.

### Step F3 — Hard embedding (`spot_embedding`)

Call `**spot_embedding**` from `**mcDETECT.downstream**` with **hard** assignment to 50×50 windows, Gaussian smoothing of subtype counts, and soma features:

```python
import numpy as np
from mcDETECT.downstream import spot_embedding

embeddings, embeddings_features, aux_features, spot_granule_expression, spot_cell_expression = spot_embedding(
    spots=spots,
    granule_adata=granule_adata,
    adata=adata_neuron,
    count_matrix=count_matrix,
    spot_loc_key=("global_x", "global_y"),
    spot_width=grid_len_num,
    spot_height=grid_len_num,
    granule_loc_key=("global_x", "global_y"),
    granule_subtype_key="granule_subtype_kmeans",
    subtype_names=[str(i) for i in range(granule_adata.obs["granule_subtype_kmeans"].nunique())],
    granule_count_layer="counts",
    cell_loc_key=("global_x", "global_y"),
    cell_id_key="cell_id",
    count_matrix_cell_id_key="cell_id",
    include_soma_features=True,
    smoothing=True,
    smoothing_radius=np.sqrt(2) * grid_len_num + 1,
    smoothing_mode="gaussian",
)

for aux_key, aux_val in aux_features.items():
    spots.obs[aux_key] = aux_val
```

**Meaning:** `**embeddings`** is a 2D array (`**n_spots × n_subtype_columns**`): per spot, counts (and soma-related columns if included) aggregated into that grid cell. Granules/cells are assigned to **one** spot by containment in the `**spot_width × spot_height`** square around each centroid.

### Step F4 — Drop empty spots

```python
mask = spots.obs["granule_count"] > 0
spots = spots[mask].copy()
embeddings = embeddings[mask].copy()
```

### Step F5 — Hard **normalized** embedding for clustering

Normalize **each row** of `**embeddings`** to sum to 1 (subtype proportion per spot). This is the **“normalized”** mode used for clustering in the notebook:

```python
row_sums = embeddings.sum(axis=1, keepdims=True)
X = np.divide(embeddings, row_sums, out=np.zeros_like(embeddings, dtype=float), where=row_sums > 0)
```

### Step F6 — K-Means with **k = 4**

```python
from sklearn.cluster import KMeans

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
kmeans_labels = kmeans.fit_predict(X)
spots.obs["subdomain_kmeans"] = [f"Subdomain {l + 1}" for l in kmeans_labels]
```

The full notebook optionally **relabels** clusters for consistent ordering across plots (a `**relabel_map`**); treat that as cosmetic once `**k = 4**` is fixed.

### Step F7 — Visualization and follow-ups (optional)

- Spatial scatter of `**subdomain_kmeans**` colored by layer or subdomain.
- **Clustermap** of mean `**X`** per subdomain × k-means cluster column (granule subtype dimensions).
- For **differential expression** between two subdomains, the notebook uses `**spot_granule_expression`** / `**spot_cell_expression**` with `**sc.tl.rank_genes_groups**` on `**log1p**`-normalized counts — only if you need gene-level contrasts.

---

## 8. Quick reference checklist


| Step | Action                                                                            |
| ---- | --------------------------------------------------------------------------------- |
| 1    | Build `transcripts` with required columns (§1.2).                                 |
| 2    | Instantiate `mcDETECT` using the numbered parameter list (§2).                    |
| 3    | Rough then fine detection, or only `detect()` (§3).                               |
| 4    | `profile` → `granule_adata` (§4).                                                 |
| 5    | Normalize → MiniBatch k-means → heatmap → `apply_manual_annotation` (§5).         |
| 6    | WT/AD density via spot windows, optional AD calibration, stats, export (§6).      |
| 7    | Isocortex, 50×50, `spot_embedding` (hard), row-normalize, K-Means **k = 4** (§7). |


This document is intentionally aligned with `**model.py`** APIs and the **analysis order** in `**3_detection.py`**, `**benchmark_subtyping.ipynb**`, and the **Isocortex / 50×50 / hard-normalized / k = 4** branch of `**7_neuropil_subdomains.ipynb`**.