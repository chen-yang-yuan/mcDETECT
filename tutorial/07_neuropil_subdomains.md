# 7. Step F — Neuropil subdomain analysis, Isocortex (`7_neuropil_subdomains.ipynb`)

[← Tutorial index](./README.md)

The following is the **final-used** setting: **Isocortex**, **50×50** µm grids, **hard** `spot_embedding`, **row-normalized** subtype embedding, **K-Means with k = 4**. Soft embeddings, 25×25 grids, unnormalized embeddings, inertia/ARI sweeps, LDA/GMM/MiniBatch variants, and **k = 5** are omitted.

### Step F1 — Region and grid constants

```python
ROI = "Isocortex"
grid_len = "50_by_50"
grid_len_num = 50
```

### Step F2 — Inputs (prepared elsewhere in the pipeline)

You need consistent objects under your comparison output directory (paths in the notebook are relative to `code/`):

- **`spots`** — Combined WT + AD spot `AnnData` for Isocortex with **`global_x`**, **`global_y`**, **`layer_labels`**, **`batch`**, and aligned coordinates (the notebook uses **`recover_spots`** for 50×50 to attach **`layer_labels`** from a reference `h5ad`).
- **`adata`** — Cell-level `AnnData` (e.g. **`neuropil_subdomains_adata.h5ad`**).
- **`granule_adata`** — Granule `AnnData` restricted to the neuropil workflow (e.g. **`neuropil_subdomains_granule_adata.h5ad`**) with **`granule_subtype_kmeans`** in **`obs`** and raw counts in **`layers["counts"]`**.
- **`count_matrix`** — Per-cell gene counts with **`cell_id`**, gene columns aligned to **`granule_adata.var_names`**.

**Neurons for soma features:** subset cells to Isocortex and excitatory/inhibitory types, e.g. `brain_area` matching **`ROI`** and **`cell_type`** in `["Glutamatergic", "GABAergic"]` → **`adata_neuron`**.

### Step F3 — Hard embedding (`spot_embedding`)

Call **`spot_embedding`** from **`mcDETECT.downstream`** with **hard** assignment to 50×50 windows, Gaussian smoothing of subtype counts, and soma features:

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

**Meaning:** **`embeddings`** is a 2D array (**`n_spots × n_subtype_columns`**): per spot, counts (and soma-related columns if included) aggregated into that grid cell. Granules/cells are assigned to **one** spot by containment in the **`spot_width × spot_height`** square around each centroid.

### Step F4 — Drop empty spots

```python
mask = spots.obs["granule_count"] > 0
spots = spots[mask].copy()
embeddings = embeddings[mask].copy()
```

### Step F5 — Hard **normalized** embedding for clustering

Normalize **each row** of **`embeddings`** to sum to 1 (subtype proportion per spot). This is the **“normalized”** mode used for clustering in the notebook:

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

The full notebook optionally **relabels** clusters for consistent ordering across plots (a **`relabel_map`**); treat that as cosmetic once **`k = 4`** is fixed.

### Step F7 — Visualization and follow-ups (optional)

- Spatial scatter of **`subdomain_kmeans`** colored by layer or subdomain.
- **Clustermap** of mean **`X`** per subdomain × k-means cluster column (granule subtype dimensions).
- For **differential expression** between two subdomains, the notebook uses **`spot_granule_expression`** / **`spot_cell_expression`** with **`sc.tl.rank_genes_groups`** on **`log1p`**-normalized counts — only if you need gene-level contrasts.

**Next:** [Quick reference checklist](./08_checklist.md)
