# 8. Quick reference checklist

[← Tutorial index](./README.md)

| Step | Action |
|------|--------|
| 1 | Build `transcripts` with required columns ([Section 1](./01_what_mcdetect_expects.md)). |
| 2 | Instantiate `mcDETECT` using the numbered parameter list ([Section 2](./02_construct_mcdetect.md)). |
| 3 | Rough then fine detection, or only `detect()` ([Section 3](./03_granule_detection.md)). |
| 4 | `profile` → `granule_adata` ([Section 4](./04_profile_granules.md)). |
| 5 | Normalize → MiniBatch k-means → heatmap → `apply_manual_annotation` ([Section 5](./05_manual_subtyping.md)). |
| 6 | WT/AD density via spot windows, optional AD calibration, stats, export ([Section 6](./06_density_wt_ad.md)). |
| 7 | Isocortex, 50×50, `spot_embedding` (hard), row-normalize, K-Means **k = 4** ([Section 7](./07_neuropil_subdomains.md)). |

This document is intentionally aligned with **`model.py`** APIs and the **analysis order** in **`3_detection.py`**, **`benchmark_subtyping.ipynb`**, and the **Isocortex / 50×50 / hard-normalized / k = 4** branch of **`7_neuropil_subdomains.ipynb`**.
