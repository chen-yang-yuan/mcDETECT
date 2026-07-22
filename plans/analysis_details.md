# mcDETECT — Full Analysis & Methodology Reference

This document is the deep reference for the entire mcDETECT project: the algorithm, every
analysis script/notebook, the simulation benchmark, and the biological validation. `CLAUDE.md`
stays intentionally short and points here for detail. Nothing in this file needs to be run to be
useful — it is documentation. (Per repo policy, do not execute any code unless explicitly asked.)

- Package version: `2.1.6` (`mcDETECT_package/mcDETECT/__init__.py`)
- Package path: `mcDETECT_package/mcDETECT/{model,utils,downstream}.py`
- Analysis: `code/` (numbered pipeline), `code/benchmark/`, `simulation/`, `validation/`, `other_analysis/`

---

## 1. Scientific goal

mcDETECT finds the **"dark transcriptome"**: mRNA that forms small aggregates *outside* the
nucleus/soma (in dendrites, axons, synapses) — missed by standard cell-segmentation pipelines that
assign transcripts to cell bodies. Working from *in situ* spatial transcriptomics (iST) point
clouds (MERSCOPE, CosMx, Xenium, MERFISH), it treats every mRNA molecule as a 3D point and uses
density-based clustering to call **RNA granules**, then profiles, subtypes, and spatially maps them.

The biological payoff (downstream): granules are subtyped (pre-synaptic / post-synaptic / dendritic
/ axonal / mixed), and cortical tissue is partitioned into **neuropil subdomains** — spot clusters
defined by their *granule-subtype composition* rather than by cell types — enabling WT-vs-AD
(Alzheimer's) differential analysis of the compartmentalized transcriptome.

---

## 2. The algorithm — `mcDETECT` class (`model.py`)

### 2.1 Input schema
A **transcripts DataFrame**, one row per molecule:
- `global_x`, `global_y`, `global_z` (float) — 3D coordinates. For `type="discrete"` the unique
  `global_z` values form a finite imaging-plane grid; for `"continuous"` z is truly continuous.
- `target` (str) — gene name (primary label; preserved for downstream counting).
- `overlaps_nucleus` (0/1) — whether the molecule lies inside a segmented nucleus/soma (from DAPI
  dilation, computed in step 1). Drives the in-soma filter.
- `cell_id` — segmented cell (only needed with `record_cell_id=True`).

### 2.2 Constructor parameters
`mcDETECT(type, transcripts, gnl_genes, nc_genes=None, eps=1.5, minspl=None, grid_len=1.0,
cutoff_prob=0.95, alpha=5.0, low_bound=3, size_thr=4.0, in_soma_thr=0.1, l=1.0, rho=0.2, s=1.0,
nc_top=20, nc_thr=0.1, merge_genes=False, merged_gene_label="merged")`

| Param | Default | Meaning |
|---|---|---|
| `type` | — | `"discrete"` (MERSCOPE/CosMx, z-grid) or `"continuous"` (Xenium). Else `ValueError`. |
| `gnl_genes` | — | Granule marker genes (the panel to detect on). |
| `nc_genes` | `None` | Negative-control genes; `None` skips NC filtering. |
| `eps` | `1.5` | DBSCAN radius ε (µm); also the disk radius `π·eps²` in the Poisson model. |
| `minspl` | `None` | Manual `min_samples`; `None` ⇒ auto per gene via `poisson_select`. |
| `grid_len` | `1.0` | Grid cell size for tissue-area estimate. |
| `cutoff_prob` | `0.95` | Poisson quantile for auto `min_samples`. |
| `alpha` | `5.0` | Inflates background density in the Poisson model (analysis scripts use `10`). |
| `low_bound` | `3` | Floor on `min_samples`; also min unique points to keep a cluster. |
| `size_thr` | `4.0` | Max sphere radius (µm); larger spheres discarded. |
| `in_soma_thr` | `0.1` | Max in-soma fraction; higher ⇒ discarded (extrasomatic filter). |
| `l` | `1.0` | Distance scale in overlap tests. |
| `rho` | `0.2` | Merge threshold: intersecting spheres merge only if `dist < rho·l·(r_a+r_b)`. |
| `s` | `1.0` | Radius scale applied to a merged sphere. |
| `nc_top` | `20` | # most-expressed NC genes retained for filtering. |
| `nc_thr` | `0.1` | Max NC ratio; `None` annotates but doesn't drop. |
| `merge_genes` | `False` | If `True`, pool all `gnl_genes` into one label and DBSCAN once. |
| `merged_gene_label` | `"merged"` | Label for the merged marker. |

### 2.3 Method pipeline (execution order)
1. **`construct_grid` / `tissue_area`** — occupied tissue area = (# nonzero cells of a `grid_len` 2D
   histogram) × `grid_len²`.
2. **`poisson_select(gene)`** — auto `min_samples`. Under a homogeneous-Poisson null, expected count
   of this gene in a radius-`eps` disk is `bg_density·π·eps²`; inflate by `alpha`; take the
   `cutoff_prob` quantile: `min_samples = int(max(poisson.ppf(cutoff_prob, mu=alpha·bg_density·π·eps²), low_bound))`.
3. **`dbscan(target_names, record_cell_id)`** — per marker: 3D DBSCAN (`eps`, kd_tree). Each cluster →
   **minimum enclosing sphere** via `miniball.get_bounding_ball` (fallback: centroid + max distance on
   singular matrices) fit on *deduplicated* coordinates. Computes per sphere: `size` (transcripts
   inside, incl. neighboring non-target granule transcripts via kd-tree), `comp` (# distinct genes
   represented), `in_soma_ratio` (fraction with `overlaps_nucleus==1`), `layer_z` (z snapped to
   nearest plane for discrete; = z for continuous). Filters `sphere_r < size_thr` AND
   `in_soma_ratio < in_soma_thr`. Returns `{gene_idx: sphere_df}`.
   - **`merge_genes=True` branch**: runs DBSCAN once on the pooled merged-marker cloud (higher
     sensitivity). Used by `other_analysis/Xenium_5K/1_detection_merged_genes.py`.
4. **`merge_sphere` → `_remove_overlaps`** — merge spheres *across* markers. A 2D R-tree
   (`make_rtree`) finds candidate overlaps; a 3D distance test then classifies each pair:
   containment ⇒ drop the smaller (or replace with the larger); genuine deep intersection
   (`dist < rho·l·(r_a+r_b)`) ⇒ refit a new miniball over the union of member transcripts, radius
   scaled by `s`; weak intersection ⇒ keep both. `rho` (default 0.2) means mere geometric overlap is
   **not** enough — centers must be substantially close to fuse.
5. **`nc_filter(sphere)`** — build a kd-tree over the top-`nc_top` negative-control transcripts;
   `nc_ratio = (NC transcripts inside sphere) / size`; keep spheres with `nc_ratio==0` or
   `< nc_thr`. Removes aggregates that are just nuclear/somatic background.
6. **`detect(record_cell_id)`** = `dbscan → merge_sphere → nc_filter` (NC step skipped if
   `nc_genes is None`). Returns the **granule metadata DataFrame**.
7. **`profile(granule, genes, buffer)`** — for each granule sphere, count *all* transcripts inside
   (kd-tree `query_ball_point` at `layer_z`) → sparse **granule × gene AnnData**; adds `granule_id`,
   renames `sphere_x/y/z → global_x/y/z` in `obs`.
8. **`spot_expression(grid_len, genes)`** — 2D-histogram pseudo-spot × gene AnnData (Visium-like).

### 2.4 Granule metadata columns
`sphere_x/y/z` (miniball center), `layer_z` (plane-snapped z, used for all downstream queries),
`sphere_r` (radius), `size` (# transcripts inside), `comp` (# distinct genes), `in_soma_ratio`,
`gene` (seeding marker or `"merged"`), `nc_ratio` (added by `nc_filter`), `cell_id` (optional).

### 2.5 `utils.py`
`make_tree` (scipy `cKDTree`), `make_rtree` (2D bbox R-tree), `find_threshold_index` (cumulative-
fraction cutoff, used for the 99% marker-panel cut), `closest`, `scale`, `weighted_corr`,
`weighted_spearmanr`, `assign_palette_to_adata`, `p_val_to_star`, `top_columns_above_threshold`.

### 2.6 `downstream.py`
- **`GranuleSubtyper` / `classify_granules`** — rule-based subtyping mimicking manual heatmap
  annotation: z-score genes (per-cluster or per-granule), sum per marker category (pre-syn / post-syn
  / dendrites / axons), classify a category as "enriched" if its share of total positive z-score
  ≥ `enrichment_threshold` (0.35) and raw z ≥ `min_zscore_threshold` (0.0). Emits pure/pair/triple/
  quad labels or `"others"`; `classify_granules` also returns a `_simple` series collapsing any
  combination to `"mixed"`. **`custom_markers` must be supplied** (default `None` would error).
- **`spot_neuron` / `spot_granule`** — per-spot box counts/means of neurons / granules.
- **`neighbor_granule`** — neuron↔granule colocalization; Gaussian-distance-weighted granule
  expression per neuron plus spatial features (density, anisotropy, offset).
- **`neuron_embedding_one_hot`** — per neuron, one-hot subtypes of k nearest granules within radius.
- **`neuron_embedding_spatial_weight`** — per neuron, distance-weighted subtype composition vector.
- **`spot_embedding` (hard/box)** and **`spot_embedding_soft` (kernel-weighted)** — the workhorses of
  steps 7–9. Assign granules to spots (box containment vs Gaussian/exponential/uniform kernel),
  returning `(subtype_counts, feature_names, aux_features{granule_count[,soma_count]},
  spot_granule_expression, spot_cell_expression)`. `spot_embedding` supports Gaussian smoothing and
  optional soma features; only `subtype_counts` is smoothed.

---

## 3. Main analysis pipeline (`code/`, numbered = execution order)

Paths are relative to `code/` (`../data/...`, `../output/...`). All heavy `.py` steps have a matching
SLURM `.sh` wrapper (`module load miniconda3; conda activate mcDETECT-env; python3 <step>.py`).

| Step | File | Purpose / key I/O |
|---|---|---|
| 1 | `1_clean_transcripts.ipynb` | **Ingestion + DAPI in-soma labeling.** Raw `transcripts.csv` + per-z `mosaic_DAPI_z*.tif` → adaptive-threshold + contour size filter + morphological **dilation** (radius 5 & 10) of nuclei; a transcript is `overlaps_nucleus` if its DAPI pixel is nonzero (radius-5 dilation is the one used downstream). Rescales z ×1.5 µm, applies per-dataset rotation `theta` + hemisphere `cutoff` crop. → `data/<dataset>/processed_data/transcripts.parquet` (+ companion `adata.h5ad`, `spots.h5ad`, `genes.csv`, `negative_controls.csv`). |
| 2 | `2_gene_ranking.py` | **Marker-panel selection.** For each candidate marker, `mc.dbscan([gene])` counts granules; rank descending, cumulative sum, `find_threshold_index(0.99)`. → `output/<dataset>/gene_ranking_raw.csv`. |
| 3 | `3_detection.py` | **Main detection (two passes).** *Rough*: all filters off (`size_thr=1e5`, `in_soma_thr=1.01`, `nc_genes=None`) → `all_granules.parquet`. *Fine*: filters on (`size_thr=4.0`, `in_soma_thr=0.1`, NC on) → `mc.detect()` → assign `brain_area` by nearest spot, apply per-dataset rotation/flip → `granules.parquet`. Then `mc.profile` → normalize/log1p/PCA/t-SNE → `granule_adata_tsne.h5ad`. Per-dataset geometry hard-coded near top (`flip`, `cutoff`, `theta`, coord columns); `dataset` set at line ~15. |
| 4 | `4_post_detection.ipynb` | **Post-detection + combined WT+AD object.** `spot_neuron`/`spot_granule` → per-region density CSVs and granule↔neuron heatmaps; **Combined mode** aligns WT+AD coords, concatenates, PCA/t-SNE → `MERSCOPE_WT_AD_comparison/granule_adata_tsne.h5ad` (feeds steps 5–7). |
| 5 | `5_neuropil_subdomains_data.py` | **Data assembly for subdomains (pair 1).** Align + concat WT_1/AD_1 spots/cells/transcripts (`batch` label); merge k-means + manual granule-subtype labels; `subdivide_spots` (50 µm → 25 µm sub-tiles); `fill_spot_expression` fills all / extrasomatic (`overlaps_nucleus==0`) / intrasomatic layers. → `neuropil_subdomains_{adata,transcripts,granule_adata,spots}` in the comparison folder. |
| 6 | `6_neuropil_subdomains_SpaGCN.ipynb` | **Cortical-layer annotation (needs Python 3.11 kernel).** SpaGCN over spot grid → cluster → map to layer labels (L1…L6, RSP for Isocortex; CA1/CA3/DG/Neuropil for HPF); manual geometric masks refine. → `neuropil_subdomains_spots_<ROI>.h5ad` (+ `HPF_annotation.csv`). |
| 7 | `7_neuropil_subdomains.ipynb` | **MAIN downstream result (pair 1).** `spot_embedding` (hard, Gaussian smoothing, soma features) on `granule_subtype_kmeans` → spot×subtype embeddings + `spot_granule/cell/ambient` expression; benchmark #clusters (inertia + ARI); cluster spots into **4 subdomains** (LDA/GMM/KMeans); composition + log2FC heatmaps; Wilcoxon **DE** (Subdomain 1 vs 2) on granule/cell/ambient. → `neuropil_subdomains_Isocortex_50/`. |
| 8 | `8_neuropil_subdomains_pair2.ipynb` + `.py` | **Same analysis for pair 2** (`MERSCOPE_WT_2`/`AD_2`, 290-gene panel, *rough* granule adata, coords NOT aligned across samples so granules never mix). K-means(15) subtyping (drop junk cluster "9"), per-sample `build_spot_grid` + `spot_embedding` → KMeans(4) subdomains. **Deliverable: AD-vs-WT granule DEGs per subdomain** (`granule_DE_genes_AD_vs_WT_Subdomain_*.csv`). → `MERSCOPE_WT_AD_2_comparison/neuropil_subdomains_Isocortex_50/`. |
| 9 | `9_pathway_p1_expression_pair2.ipynb` | **Pathway/AD gene-set spatial expression (pair 2).** Gene sets from `data/gene_panel_v2.xlsx` (`pathway_p1` = Priority-1, `pathway_p2`, `AD_genes`); AD up/down direction parsed from `Subcategory`. `spot_embedding` (subtype `"all"`, no smoothing) → per-gene spatial scatter of `log2(counts+1)` (WT left, AD flipped right) + split violins. → `{pathway_p1,pathway_p2,AD_genes}_expression_Isocortex_50/`. |

**Vocabulary:** *pair 1* = `MERSCOPE_WT_1`/`AD_1`; *pair 2* = `MERSCOPE_WT_2`/`AD_2`. *Neuropil
subdomains* = spot clusters defined by granule-subtype composition within a cortical ROI. *pathway
p1* = Priority-1 pathway gene set. Per-dataset `theta`/`flip`/`cutoff`/coord-column values (steps 1,
3, 4, 5, 6) are **canvas-alignment geometry**, not algorithm parameters.

### 3.1 Benchmarks (`code/benchmark/`)
Each answers a "why this parameter/choice" question. Precomputed inputs live under `output/`.

| File | Benchmarks |
|---|---|
| `benchmark_DBSCAN.py` | `eps` × `min_samples` sweep on a small ROI (n detections, radius, in-soma, NC ratio). |
| `benchmark_clustering.py` | # granule K-means clusters (elbow inertia, silhouette, ARI stability) — justifies K≈15. |
| `benchmark_rho.py` | Sphere-merge rule + overlap metric (distance/`rho` vs volume/Jaccard/Dice). |
| `benchmark_time.py` | Runtime of neighbors + Leiden/Louvain, sparse vs dense. |
| `benchmark_DAPI.ipynb` | DAPI thresholding/dilation → soma area & in-soma ratio (validates the soma filter). |
| `benchmark_ambient.ipynb` | OLS test that subdomain signal isn't driven by ambient background. |
| `benchmark_collapse.ipynb` | Platform spatial resolution + k-NN transcript distances (informs `eps`). |
| `benchmark_filtering.ipynb` | in-soma & NC filtering thresholds (none / 0.1 / 0.1 / both). |
| `benchmark_sphere.ipynb` | `profile()` radius setting (default/fixed/expand/shrink) vs expression correlation. |
| `benchmark_subtyping.ipynb` | Manual K-means vs automatic `GranuleSubtyper` (grid-search to match manual proportions). |

---

## 4. Simulation benchmark (`simulation/`)

Purpose: on synthetic data with **known ground truth**, show mcDETECT out-detects competing methods.

- **`simulate.py`** — generates labeled point clouds. `simulation` = single gene: `simulate_CSR`
  (uniform background noise) + `simulate_cluster` (**Thomas / Neyman-Scott** process: parents uniform,
  Poisson offspring displaced by exponential radial distance in random 3D direction; per-cluster
  `in_nucleus_ratio ~ Beta(β)` — `β=(1,19)` extranuclear granules vs `β=(19,1)` intranuclear).
  `multi_simulation` = multi-gene co-localized granules with a global `granule_id`; a cluster is
  ground-truth **only if it contains ≥ `comp_thr` (=2) marker genes**. → `simulated_data/{single,multi}_marker/3D/*.parquet` (cols `x,y,z,gene,granule_id,type∈{CSR,Extranuclear,Intranuclear}`).
- **`model.py`** — a **standalone, speed-optimized reimplementation** of the mcDETECT core for the
  benchmark (global kd-tree, integer-coded genes, `miniball_epsilon=1e-4`). Note parameter-name
  drift vs the package: `p` here = `rho`; `in_thr` here = `in_soma_thr`. Not the installable package.
- **Competing methods** (same `sphere_x/y/z/r` output schema, scored identically):
  - `run_Baysor.py` (+`.sh`) — external **Baysor** Julia CLI segmentation; each segmented cell →
    miniball sphere. Multi-marker 3D, 200 seeds, 20 SLURM blocks × 10 seeds. Uses `baysor_env`.
  - `run_SSAM.py` (+`.sh`) — **SSAM** segmentation-free KDE vector field + local maxima → fixed
    radius-1.5 spheres. Uses `ssam_hpc` env.
- **`evaluation_utils.py`** — primary metric `compute_object_level_metrics`: ground-truth objects
  (by `granule_id`, extranuclear only) vs detection sets (transcripts in each sphere); keep pairs with
  completeness ≥ `tau_c` and purity ≥ `tau_p`; **Hungarian** max-weight one-to-one matching →
  precision/recall/F1 + cross-tab. Default τ=0.9, but **Baysor/SSAM evals call it with τ=0.5**.
- **Drivers:** `benchmark_rho_sim.py` (sweep `p`×`l`), `benchmark_rho_sim_all_strategies.py`
  (distance vs volume/Jaccard/Dice merge criteria), `analyze_crosstab.py` (per-method purity
  summary), orchestrated in `run_mcDETECT.ipynb`. Final figures via `figures.R`
  (`detection_side.jpeg`, `running_time.jpeg`).

---

## 5. Biological validation (`validation/`)

Driver `validation/figures.Rmd` (organized by reviewer response). Each subfolder = one line of
real-world evidence that detected granules are real and biologically meaningful:

- **`EM_data/`** — electron-microscopy corroboration of granule sizes/abundances (Mishchenko,
  Santuy, MICrONS `meta_all_with_length.csv` axon/dendrite lengths by cortical layer).
- **`DAPI_dilation/`** — justifies MERSCOPE nucleus-dilation vs Xenium cell boundaries: dilated DAPI
  recovers object areas and in-soma ratios matching Xenium (`object_areas.jpeg`, `in_soma_ratios.jpeg`).
- **`Hailing_data/`** — independent N2A-cell smFISH/deconvolution imaging (DAPI/Malat1/Calm1/Actb/
  Snap25); validates in-vs-out-of-nucleus localization against pixel-level truth (`analysis.ipynb`).
- **`isoform/`** — `analysis.R`: granule-enriched genes (detection_ratio ≥ 0.02) have longer/more
  complex transcript architecture (isoform diversity, junctions, UTR/CDS lengths) via one-sided
  Wilcoxon; commented blocks add SPLISOSM spatial-isoform overlap and GO/KEGG enrichment.
- **`RIBOmap/` + `scRNA-seq_RIBOmap/`** — granule markers enriched in ribosome-bound / neuropil
  translation (Mann-Whitney / beta regression), concordant with scRNA-seq references.
- **`scRNA-seq_vs_bulk/`** — synaptic-gene UMI proportion across bulk/sc/snRNA-seq by region/age;
  WT/AD expression ratios for pre-/post-synaptic panels.
- **`justify_simulation/`** — figures justifying simulation field size & in-soma-ratio choices.

---

## 6. Cross-platform application (`other_analysis/`)

All notebooks import the **installed package** (`from mcDETECT.model import *`) and follow a common
template: detection → `granules.parquet` → granule-vs-neuron marker heatmap → synapse-density vs
brain-region correlation (region assigned by nearest-spot `cKDTree`). Driver `figures.Rmd` = Fig 6.

- **`Xenium_5K/`** (primary, 5000-plex): `1_detection.ipynb` (per-gene) and the production
  `1_detection_merged_genes.py` (+`.sh`) which sets **`merge_genes=True`** (pool 16 synaptic markers
  into one cloud, `minspl=4`) → `granules_merged_genes_minspl_4.parquet` +
  `granule_adata_tsne_merged_genes_minspl_4.h5ad`. `2_subtyping*` = K-means(15) + manual mapping +
  per-region density with bootstrap CIs. `3_enriched_genes.ipynb` → `gene_detection_counts.xlsx`
  (feeds the isoform validation).
- **`CosMx/main.ipynb`**, **`MERFISH/main.ipynb`** — same pipeline on other platforms (`minspl=3`;
  MERFISH uses `in_soma_thr=0.5`), demonstrating generalization.
- **`RIBOmap/RIBOmap.R`** — process-read-fraction ratio table + violins.
- **`in_soma_ratio/`** — granule markers are nucleus-depleted across all three platforms (one-sided
  t-tests, violins).

---

## 7. Operational notes (see `CLAUDE.md` for the short version)

- **Datasets:** `MERSCOPE_WT_1`/`AD_1` (pair 1, discrete), `MERSCOPE_WT_2`/`AD_2` (pair 2),
  `Xenium_5K` (continuous, primary), `CosMx`, `MERFISH`.
- **Two-pass detection:** rough (filters off) → `all_granules.parquet`; fine (filters on) →
  `granules.parquet`.
- **Config lives at the top of each script** (dataset name, hard-coded per-dataset geometry) — no
  CLI args. To run another dataset, edit those constants.
- **Gitignored working dirs** (`data/`, `output/`, `validation/`, `other_analysis/`, `code/old/`,
  `simulation/simulated_data/`, `simulation/output/`, `figures.*`) won't exist in a fresh clone.
- **Two algorithm copies:** the packaged `mcDETECT` (used by `code/` and `other_analysis/`) vs the
  benchmark-local `simulation/model.py` (renamed params: `p`↔`rho`, `in_thr`↔`in_soma_thr`).
- **Environments:** `mcDETECT-env` (main, Python 3.10); step 6 needs Python 3.11 for SpaGCN; the
  simulation competitors need `baysor_env` and `ssam_hpc`.

---

*Maintenance: when the algorithm or a pipeline step changes materially, update this file and keep the
`CLAUDE.md` summary in sync. This file is documentation only — do not run any code unless the user
explicitly asks.*
