# mcDETECT tutorial: MERSCOPE analysis (step by step)

This tutorial follows the repository workflow for **MERSCOPE** data: transcript-based **`mcDETECT`** setup → granule detection → expression profiling → **manual** granule subtyping → **WT vs AD granule density** comparison → **Isocortex neuropil subdomain** analysis on a **50×50** grid with **hard** embeddings, **row-normalized** subtype vectors, and **K-Means with k = 4**.

**Primary references:** `mcDETECT_package/mcDETECT/model.py`, `code/3_detection.py`, `code/benchmark/benchmark_subtyping.ipynb`, `code/7_neuropil_subdomains.ipynb`.

---

## Analysis roadmap

1. [**Prepare**](./01_what_mcdetect_expects.md) — transcript and auxiliary tables (coordinates, gene names, nucleus overlap, optional `cell_id`).
2. [**Initialize**](./02_construct_mcdetect.md) `mcDETECT` with platform type, marker lists, and numerical hyperparameters.
3. [**Detect granules**](./03_granule_detection.md) — optional rough pass, then fine pass with filters.
4. [**Profile**](./04_profile_granules.md) granules into an `AnnData` count matrix.
5. [**Subtype manually**](./05_manual_subtyping.md) — normalize expression, k-means, heatmap, map clusters to biology, derive a simple mixed/pure label.
6. [**Compare density**](./06_density_wt_ad.md) WT vs AD by brain region using spots as spatial units.
7. [**Neuropil subdomains (Isocortex, 50×50)**](./07_neuropil_subdomains.md) — hard `spot_embedding`, row-wise normalization, K-Means **k = 4**.

See also the [quick reference checklist](./08_checklist.md).

Dataset-specific coordinate transforms and multi-setting benchmark loops are omitted here.

---

## Pages

| File | Section |
|------|---------|
| [01_what_mcdetect_expects.md](./01_what_mcdetect_expects.md) | What mcDETECT expects |
| [02_construct_mcdetect.md](./02_construct_mcdetect.md) | Step A — Construct `mcDETECT` |
| [03_granule_detection.md](./03_granule_detection.md) | Step B — Granule detection |
| [04_profile_granules.md](./04_profile_granules.md) | Step C — Profile granules |
| [05_manual_subtyping.md](./05_manual_subtyping.md) | Step D — Manual granule subtyping |
| [06_density_wt_ad.md](./06_density_wt_ad.md) | Step E — WT vs AD density |
| [07_neuropil_subdomains.md](./07_neuropil_subdomains.md) | Step F — Neuropil subdomains (Isocortex) |
| [08_checklist.md](./08_checklist.md) | Quick reference checklist |

**Read the Docs:** edit → sync → push workflow: **[PUBLISHING_READTHEDOCS.md](https://github.com/chen-yang-yuan/mcDETECT/blob/main/tutorial/PUBLISHING_READTHEDOCS.md)** (also on disk in this folder). Maintainer-only CI notes: **[readthedocs_github_action.md](https://github.com/chen-yang-yuan/mcDETECT/blob/main/tutorial/readthedocs_github_action.md)** (not copied to the RTD site).
