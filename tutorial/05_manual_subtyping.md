# 5. Step D — Manual granule subtyping only (`benchmark_subtyping.ipynb`)

[← Tutorial index](./README.md)

Automated rule-based subtyping (`classify_granules`) is **not** covered here; the benchmark notebook’s primary annotation path is **manual**.

### Step D1 — Normalize for clustering

After **`profile`**, store raw counts in **`layers["counts"]`**, then apply **`sc.pp.normalize_total`** and **`sc.pp.log1p`** on **`X`** (or follow your notebook’s exact normalization for consistency with saved `h5ad` files).

### Step D2 — Choose k and run MiniBatch k-means

Fix **`n_clusters`** (e.g. 15 in the benchmark), **`seed`**, **`batch_size`**, **`n_init`**. Fit on the matrix used for clustering (dense `X` if sparse).

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

**Output:** **`obs[obs_key]`** — string cluster ids `"0"`, `"1"`, ….

### Step D3 — Heatmap-driven biology

1. Pick a **reference gene list** (e.g. synaptic markers overlapping `var_names`).
2. Plot **`scanpy.pl.heatmap`** with **`groupby=obs_key`**, **`standard_scale="var"`**, to see which clusters look pre-synaptic, post-synaptic, dendritic, mixed, etc.

### Step D4 — Manual mapping dictionary

Build a **`mapping`** from biological subtype names to **lists of cluster id strings**:

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

**Convention:** finer labels live in **`granule_subtype_manual`** (e.g. `"pre & post"`); **`granule_subtype_manual_simple`** collapses any label containing **`" & "`** to **`"mixed"`** for density and summaries.

### Step D5 — Paired WT + AD objects (if applicable)

For cross-sample workflows, concatenate WT and AD **`granule_adata`** objects, restrict to common genes, normalize, run k-means once on the combined matrix, then annotate with a single **`MANUAL_SUBTYPE_MAPPING`** keyed by filename or setting. The benchmark notebook uses **`obs["sample"]`** in **`("WT", "AD")`** or batch labels.

**Next:** [Step E — WT vs AD density](./06_density_wt_ad.md)
