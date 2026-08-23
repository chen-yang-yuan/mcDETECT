# A2 — sparsity and the stochastic origin of granule-level structure

This is analysis **A2** of `plans/Round2_response_analysis_plan.md`. It answers the two things
Reviewer #2 says are still outstanding:

> *"My original request, to stratify by granule complexity and show the structure is not a
> low-count artifact, has not been met."*
> *"I am not convinced randomized data … would not produce an essentially identical embedding."*

**A2a** reruns the published pair-1 downstream chain on the multi-gene granule subset and on
read-count strata — local, in a notebook, because it has two manual-annotation pauses.
**A2b** builds the reviewer's randomized data and runs the identical detection → profile →
embedding chain on it — entirely on HGCC under SLURM, no notebook. Only `A2_figures.R` is local
for A2b.

A2c (gene–gene functional co-clustering) is deferred; A2d (promoting Fig. R9 to the supplement)
needs no new computation and is superseded by A2a section 1.

---

## The column the reviewer's request hinges on

`granules.parquet["comp"]` is **not** granule complexity, and using it would have answered the
wrong question.

`mcDETECT_package/mcDETECT/model.py:102` restricts the transcript frame to `gnl_genes` before
detection begins. So at `model.py:264-266`,

```python
other_comp  = len(other_trans[detect_col].unique())   # other_trans is markers-only
total_comp  = 1 + other_comp
```

`comp` counts **distinct granule markers**, capped at 20 — empirical max 19, mean 2.53 (WT) /
2.44 (AD). It is also never recomputed after `merge_sphere()`: `_remove_overlaps` updates only
`sphere_x/y/z/r`, or replaces the row with the other sphere's wholesale, so post-merge `comp` is
stale even as a marker count.

The complexity used throughout comes from `profile()` (`model.py:432-473`), which counts **all**
panel genes inside the sphere:

```python
n_genes = (granule_adata.layers["counts"] > 0).sum(axis=1)
```

This is the same quantity as `granule_reads_unique_genes_per_granule.parquet`
(`code/4_post_detection.ipynb` cells 21–22, the Fig. R9 source). That table was built with
`mc.profile(buffer=0.01)` while `granule_adata_tsne.h5ad` was built with `buffer=0.0`, so A2a
recomputes from the AnnData for internal consistency and writes the agreement rate to
`complexity_crosscheck.csv` rather than choosing silently.

`plans/analysis_details.md` described `comp` as "# distinct genes"; that line has been corrected.

**Counting convention.** This panel has no blank probes — `negative_controls.csv` lists 19 real
nuclear-enriched panel genes — so "unique genes" needs a stated denominator. The primary count
excludes those 19 (`C.EXCLUDE_NC_FROM_COMPLEXITY = True`); the all-290 count is exported beside
it as a sensitivity column.

---

## Files

```
a2_config.py                 paths, gene sets, every constant; the only place to change a setting
a2_common.py                 ported computation (subtyping, density, permutation, scoring, exports)

A2a_multigene.ipynb          A2a end to end            [local, mcDETECT-env]
run_permutation_detect.py    A2b stage 1, one permuted detection (10 tasks)  [HGCC, SLURM array]
score_embedding.py           A2b stage 2, one COMBINED arm (6 tasks); --concat  [HGCC, array]
slurm/run_permutation.sh     detection array wrapper
slurm/score_embedding.sh     scoring array wrapper
slurm/concat.sh              stitches the per-arm tables
slurm/submit.sh              submits all three with the right dependencies
A2_figures.R                 all figures, per-section toggles         [local]
```

Outputs (git-ignored):

```
output/
├── a2a/
│   ├── multigene/           complexity tables, subtype heatmaps + labels, density,
│   │   └── neuropil_subdomains_Isocortex_50/    subdomain maps, heatmaps, DE tables
│   └── readstrata/          tercile edges, composition, per-tercile density
├── a2b/
│   ├── perm_<sample>_seed<N>/   all_granules.parquet, granules.parquet, granule_profile.h5ad
│   ├── combined_{real,seed<N>}/ the combined WT+AD object that is actually embedded
│   └── metrics/             per-arm + concatenated CSV/Parquet, t-SNE jpegs
└── figures/                 everything A2_figures.R draws
```

---

## A2a — multi-gene granule reanalysis

Reads the published combined granule profile and subsets it. **No detection is rerun and
`transcripts.parquet` is never opened** (except by the optional validation section): `profile()`
queries each sphere independently, so subsetting rows of the granule × gene matrix is identical to
re-profiling the retained spheres. Section 7 asserts this on 1,000 sampled granules.

| § | What |
|---|---|
| 1 | Complexity distributions, the `comp`-vs-`n_genes` cross-tab, the ≥ 3-unique-gene subset, retention by sample and region |
| 2 | Re-normalise → log1p → PCA(10) → t-SNE on the subset, exactly `code/3_detection.py:113-117` |
| 3 | `MiniBatchKMeans(k=15, batch_size=5000, n_init=20, seed=1)` on the 34 compartment markers → heatmap → **manual mapping** |
| 4 | WT-vs-AD subtype density per region, bootstrap CI + t-test on log1p per-spot counts |
| 5 | Neuropil microdomains on the published spatial scaffold → recomputed `subdomain_kmeans` → **manual pair choice** → granule/cell/ambient DE |
| 6 | Read-count terciles over **all** granules, with the published subtype labels held fixed |
| 7 | Correctness gates (`VALIDATE = False`) |

### The two manual pauses

Run the notebook twice.

1. **`MANUAL_SUBTYPE_MAPPING`** (§3). Leave it empty on the first pass — sections 3–5 detect that
   and stop after writing `heatmap_subtype.jpeg` and `subtype_top_markers.csv`. Fill the dict from
   those two, then rerun. `A2.apply_manual_annotation` **raises** on a misspelt key, an
   out-of-range cluster id, a duplicate, or an omission; the published seed-1 mapping is quoted in
   the notebook only as a format example, because the subset is clustered separately and its
   cluster ids mean something different. `heatmap_subtype_ordered.jpeg` is the verdict on the
   mapping — read it as a check, not a result. The mapping and its seed are recorded to
   `run_info.csv`.
2. **`SUBDOMAIN_PAIRS`** (§5). The published contrast was `Subdomain 1 vs Subdomain 2`, but that
   depended on a cosmetic `relabel_map` fitted to the published clustering
   (`7_neuropil_subdomains.ipynb` cell 9) which does **not** transfer. K-means labels here are
   arbitrary, so **the pair to contrast is not necessarily 1 and 2**. Pick it from
   `4_hard_normalized_kmeans.jpeg` and `4_hard_normalized_heatmap_log2fc_kmeans.jpeg`. Several
   pairs are allowed; DE file names match the published
   `neuropil_subdomains_Isocortex_50/` exactly, so `A2_figures.R` discovers and scores them
   without repointing.

Nothing in either section is cached. The prescribed workflow is "run empty → fill in → rerun",
which any output-exists check would defeat by skipping exactly the second pass.

### What is inherited and what is recomputed

This distinction is the whole design of §5, and it cuts both ways:

- **Inherited, read not recomputed:** the 50 µm Isocortex spot grid and its SpaGCN `layer_labels`
  (`neuropil_subdomains_spots_ambient.h5ad`), the cell object, the per-cell count matrix.
  Recomputing the scaffold alongside the granules would change two things at once.
- **Recomputed:** `granule_subtype_kmeans` and `subdomain_kmeans`. Microdomains are *defined* by
  granule-subtype composition, so inheriting the published labels would be circular.
- **Held at the published values, not swept:** ROI = Isocortex, 50 µm spots, K_subtype = 15,
  K_subdomain = 4, hard embedding with gaussian smoothing.

### If retention is poor

Section 1 prints the worst-sample retention and warns below 20 %. The cutoff lives at
`C.MIN_UNIQUE_GENES` — drop it to 2 there and rerun from section 1. Do not change it in the
notebook.

---

## A2b — label-permutation null

**The null.** `A2.permute_targets_inplace` shuffles the `target` column across all transcripts of
a sample. Every molecule position, the total transcript density, and each gene's total count
survive; only the association between a gene label and where its molecules sit is destroyed.
`permutation_fingerprint` / `assert_permutation_valid` check all three every run — not behind a
flag, because the whole argument rests on them and they are free next to detection. The checks
use a fixed 100 K-row positional probe rather than a whole-column hash, because they must be
**order-sensitive**: any summary that sums or xors per-element hashes is invariant under exactly
the permutation being tested. Permuting in place against that probe also avoids holding two
103 M-row tables at once.

### An arm is a combined WT+AD object

This is the point that governs the whole stage. The embedding every published granule result
rests on — Fig. 3f subtypes, Fig. 4d t-SNE — is the **combined** object built by
`code/4_post_detection.ipynb` cell 19: `anndata.concat({WT, AD}, label="batch")` →
`layers["counts"]` → `normalize_total(1e4)` → `log1p` → `PCA(10)` → `t-SNE(n_pcs=10)`. A
per-sample embedding is not what the reviewer is talking about.

So permutation replicate *s* pairs **(WT seed s, AD seed s)** into one object:

```
10 detections  ->  5 null embeddings  +  1 real  =  6 scoring arms
```

`score_embedding.py` reproduces cell 19 exactly for the permuted arms, including its coordinate
alignment. (Note the WT y-flip is applied there for the *second* time — `3_detection.py` already
flipped `global_y_new` with the same cutoff, so the two cancel. That is what the published code
does and what produced the published object, so it is reproduced rather than "fixed". None of it
affects the embedding, which reads only expression.)

The real arm reads the published `granule_adata_tsne.h5ad` directly. That is legitimate and
worth stating: the metrics read `adata[:, REF_GENES].X`, which is `normalize_total` + `log1p`
and does not depend on the PCA or t-SNE stored in that file, and that file came from the same
`mc.profile` → concat → normalise code path.

### Size matching, and why it is the headline

The permuted arms will **not** hold the same number of granules as the real arm. The shuffle
preserves the marker transcript count exactly (33.3 M in WT, 20.3 M in AD — markers are 32.2 % /
29.5 % of the panel) but moves those markers onto the panel-wide distribution, raising the
in-nucleus fraction from 0.218 → 0.279 (WT) and 0.242 → 0.304 (AD). Per sphere that is decisive:
real granules average `in_soma_ratio` 0.0005 against a `< 0.1` cut, while a permuted
~6-transcript sphere averages ~0.28. The count can still move in **either** direction, since a
soma-dominated point cloud yields more DBSCAN clusters that are individually larger and then
fail `sphere_r < 4`.

Both silhouette and ARI stability depend on n. An unmatched comparison would therefore invite
exactly the objection A2b exists to close — *"of course the null looks less structured, you gave
it a tenth of the data."* So every permuted arm also emits a **size-matched pair**,
`matched_perm_seed<s>` and `matched_real_seed<s>`, both cut to `min(n_real, n_perm)` and
stratified by `batch` so the WT:AD ratio matches as well as the total. Matching is symmetric —
whichever arm is larger gets subsampled. **The matched pair is the headline comparison**; the
full-n series ride along with `n_obs` on every row, and `A2_figures.R` plots them as two facets.

The count difference itself is reported as a result (`a2b_granule_counts.jpeg`), not hidden.

### Comparability of the metrics

Real and permuted are scored by one script on one code path. Two deliberate departures from
`code/benchmark/benchmark_clustering.py:92-121`, applied identically to every arm: `n_init = 20`
(the published subtyping value, where that script left the sklearn default), and silhouette on a
fixed `SILHOUETTE_SAMPLE_SIZE` subsample, since it is O(n²) in distances. Consequence to
disclose: these numbers are comparable **with each other** but **not** with the published
`benchmark_clustering_results.csv`.

### t-SNE — full population, published implementation

`sc.settings.n_jobs` defaults to **1**, and `sc.tl.tsne` forwards `n_jobs` straight into
`sklearn.manifold.TSNE`, which uses it for both the nearest-neighbour search and the Barnes-Hut
gradient. So the published run — and the first version of this code — was single-threaded purely
by default. Passing `n_jobs` gives the **same implementation, same `method="barnes_hut"`, same
`random_state`, same parameters**; only the thread count changes. That is what makes
full-population t-SNE affordable, and why `openTSNE`/`MulticoreTSNE` are not installed (scanpy
itself notes MulticoreTSNE "is not actually faster anymore").

Every arm gets a full-population t-SNE, the real arm included, so all are produced by the same
call; the published `X_tsne` is never overwritten. The size-matched **pair** is additionally
rendered for `C.TSNE_MATCHED_SEEDS` (seed 0 by default) — one honest side-by-side at identical
n, since a t-SNE of 1.08 M points beside one of 50 K differs visually from point density alone.

### Detection-level reporting is deliberately modest

An exact filter-survival chain is *not* recoverable and is not claimed: mcDETECT applies the size
and in-soma filters inside `dbscan()`, i.e. before `merge_sphere()`, so the fine set is not a
subset of the rough set. `score_embedding.py` reports the rough and fine counts plus each
threshold evaluated as a **post-hoc predicate on the rough set** — comparable across arms and
labelled as such in the figure caption.

### If the null collapses

That is a **result**, not a crash. Any series with `n_obs < C.MIN_EMBED_N` (500) is skipped with
an explanatory row in `<arm>_status.csv`; the k-sweep drops `k >= n_obs`; t-SNE needs
`n_obs > 3 * perplexity`. Nothing is ever silently dropped — `--concat` names every missing arm,
and `A2_figures.R` §5 prints the skipped series before plotting.

### It stops at the embedding

No subtyping, no density, no microdomains for A2b.

## Run order

```bash
# ---------- A2a: local, mcDETECT-env, run from this directory ----------
#   pass 1: run A2a_multigene.ipynb with MANUAL_SUBTYPE_MAPPING empty
#           -> read output/a2a/multigene/heatmap_subtype.jpeg + subtype_top_markers.csv
#           -> fill MANUAL_SUBTYPE_MAPPING, rerun from section 3
#           -> read 4_hard_normalized_kmeans.jpeg + ..._heatmap_log2fc_kmeans.jpeg
#           -> fill SUBDOMAIN_PAIRS, rerun section 5
#   pass 2: run end to end

# ---------- A2b: HGCC ----------
cd ~/hulab/projects/mcDETECT/R2_revision/sparsity_structure
mkdir -p logs                                  # SLURM opens log files before the job body runs

sbatch --array=0 slurm/run_permutation.sh      # smoke-test one task before the full array
bash slurm/submit.sh 10                        # 10 detections -> 6 scoring arms (afterok)
                                               #   -> concat (afterany)

cat output/a2b/metrics/a2b_detection_summary.csv   # expect 12 rows (6 arms x 2 samples)
cat output/a2b/metrics/a2b_status.csv              # any series that was too small to embed

#   transfer back only output/a2b/metrics/ -- small CSVs and jpegs, not the h5ad files

# ---------- figures: local ----------
Rscript A2_figures.R
```

**Reruns.** Finished detection tasks are skipped, so resubmit only the failed ids
(`sbatch --array=<id1>,<id2> slurm/run_permutation.sh`). `concat` runs on `afterany` and names
any arm that is missing rather than dropping it silently.

**Where the time goes.**

| Stage | Tasks | Within a task |
|---|---|---|
| Detection | 10 concurrent | **No parallelism available.** `DBSCAN(...)` at `model.py:127,211` passes no `n_jobs`, `dbscan()` loops the 20 markers serially, and `merge_sphere()`/`_remove_overlaps` is a Python row loop. The log prints the sphere count entering the merge, so a blow-up shows up in the first minutes rather than after a 240 h timeout. |
| Combine + score | 6 concurrent | PCA on BLAS threads; the k-sweep is joblib-parallel over 29 k values → ~2 rounds on 16 cores instead of 29 serial blocks |
| t-SNE | in the same 6 tasks | `n_jobs` → sklearn's neighbour search **and** Barnes-Hut gradient |

Two caveats, both consequences of the count difference. t-SNE is Barnes-Hut, so cost scales
roughly `n log n`: the arms will **not** take equal time and the stage finishes when the
**largest** one does, not "in one t-SNE". Detection time is likewise not guaranteed to match the
real run.

**Levers if a stage runs long.** Detection: set `C.RUN_ROUGH_PASS = False` — it merges the
unfiltered sphere set and is the larger half of the stage; the cost is the in-soma survival
statistic. Scoring: narrow `C.SCORE_K_RANGE`, but narrow it for every arm, never for one.
t-SNE: `C.RUN_TSNE = False` leaves the metrics untouched.

**Why detection is not parallelised further.** `dbscan(target_names=[gene])` per gene looks like
an easy win but is wrong — `model.py:202` defines `others` as the other markers *within the
passed subset*, so single-gene calls would silently yield wrong `size`, `comp` and
`in_soma_ratio`. Reimplementing the loop is worse: A2b's entire claim is that the null went
through the **identical** pipeline, so it must call the published code, not a faster lookalike.

---

## Which output backs which claim

There is no intermediate summary document; every number in the response letter should be
re-derivable from `output/` alone.

| Response element | Source |
|---|---|
| "`comp` counts markers, not genes" | `a2a/multigene/comp_vs_ngenes.parquet`, `figures/comp_vs_unique_genes.jpeg` |
| Reads / unique genes per granule (Fig. R9) | `a2a/multigene/complexity_summary.csv`, `figures/complexity_n_{reads,genes}_all.jpeg` |
| "n granules retained at ≥ 3 unique genes" | `a2a/multigene/retention_by_region.csv`, `figures/multigene_retention.jpeg` |
| Subtype structure persists | `a2a/multigene/heatmap_subtype{,_ordered}.jpeg`, `subtype_composition.csv` |
| AD pre-synaptic reduction persists | `a2a/multigene/subtype_density_per_region_multigene.csv`, `figures/granule_density_multigene_pre-syn.jpeg`, `figures/granule_density_all_vs_multigene.jpeg` |
| Microdomain contrast persists | `a2a/multigene/neuropil_subdomains_Isocortex_50/`, `figures/gsea_terms_published_vs_multigene.csv` |
| "not a low-count artifact" | `a2a/readstrata/readstrata_density.csv`, `figures/readstrata_density_*.jpeg` |
| "randomized data does not give the same embedding" | `a2b/metrics/a2b_metrics.csv`, `figures/a2b_silhouette_score.jpeg`, `figures/a2b_ari_stability_mean.jpeg`, `figures/a2b_structure_at_k15.csv` — quote the **size-matched** facet |
| Permutation yields a different granule count | `figures/a2b_granule_counts.jpeg`, `a2b/metrics/a2b_detection_summary.csv` |
| Any null arm too small to embed | `a2b/metrics/a2b_status.csv` |
| Permuted detections are somatic | `figures/a2b_in_soma_survival.jpeg`, `a2b/metrics/a2b_detection_summary.csv` |
| Real vs permuted t-SNE | `a2b/metrics/tsne_matched_{real,perm}_seed0.jpeg` (equal n — use this pair), `tsne_real.jpeg` / `tsne_perm_seed*.jpeg` (full n) |

Images are JPEG at dpi 500; convert to PNG before embedding in Word (the Adobe APP14 marker
trips Word up).

---

## What this does and does not settle

**Does.** A2a shows the subtype structure, the AD pre-synaptic density reduction and the
microdomain contrast survive restriction to granules that cannot be explained by their seeding
marker alone, and that the WT/AD effect is not confined to the lowest read tercile. A2b shows the
embedding structure is not reproduced by data with identical positions, identical density and
identical per-gene totals — and, because of the size-matched arms, not merely because the null
has fewer granules to work with.

**Does not.** Neither is a test of whether granules are biologically real — that burden sits with
A3 (ambient / pseudo-granule controls) and the EM validation. A2b's null is a *global* shuffle:
it destroys all spatial gene structure, including the regional composition gradients that any
real tissue has, so it is the reviewer's stated hypothetical rather than the strictest possible
null. A block-wise permutation preserving regional composition would be a harder test; it is not
what was asked for, and it is noted here so the choice is on the record rather than implied.

Two further caveats worth disclosing in Methods: A2a's absolute densities are lower than the
published ones purely because the subset is smaller — only the WT-vs-AD direction transfers; and
A2a's subdomains are recomputed, so their numbering carries no relation to the published
Subdomain 1–4.
