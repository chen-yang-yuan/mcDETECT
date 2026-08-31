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

**A2c** asks whether the genes co-detected inside a granule are non-randomly associated — local, in
a notebook, and independent of both A2a and A2b. It answers **yes** for the genes detection never
touches, and **no** to the follow-up question of whether that association is organised by
localization rather than co-expression; both halves are reported. A2d (promoting Fig. R9 to the
supplement) needs no new computation and is superseded by A2a section 1.

**A2e** covers the two per-sample results the response needs and the other three do not produce:
the fraction of granules carrying ≥ 3 unique genes in **each** of MERSCOPE WT, MERSCOPE AD and
Xenium 5K, and a direct test of whether granule content tracks the seeding marker's compartment
category. Local, in a notebook, and independent of A2a, A2b and A2c — it reads only published
artifacts.

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

### Two provenance traps, both real, both already handled

**1. `granule_id` has two conventions.** They label the same positional key and joining them
naively matches nothing:

| Artifact | `granule_id` | Why |
|---|---|---|
| `output/<dataset>/granule_reads_unique_genes_per_granule.parquet` | `'0'`, `'1'`, … | `4_post_detection.ipynb` cells 21-22 wrote `granule_adata.obs_names`, and `profile()` builds `AnnData(X, obs=granule.copy())`, so obs_names inherit the granules DataFrame's RangeIndex |
| every AnnData's `obs["granule_id"]` | `'gnl_0'`, `'gnl_1'`, … | set separately at `model.py:467` |

`A2.normalise_granule_id` reconciles them and **raises** on anything that is neither, rather than
returning something that quietly fails to join. Everything else — the subtype-label parquet and
`neuropil_subdomains_granule_adata.h5ad` — uses `gnl_i`, so only the Fig. R9 cross-check needs it.

**2. Only one published artifact used `buffer = 0.01`.** `profile()` defaults to `buffer = 0.0`
(`model.py:432`), and every call producing a published *result* omits the argument:

| Call site | buffer | Produces |
|---|---|---|
| `code/3_detection.py:111` | **0.00** | per-sample `granule_adata_tsne.h5ad` |
| `code/4_post_detection.ipynb` cell 19 | **0.00** | the combined `granule_adata_tsne.h5ad` — the object subtyping, t-SNE, microdomains and DE all run on |
| `code/benchmark/benchmark_subtyping.ipynb` cell 10 | **0.00** | the published k = 15 subtype labels |
| `code/benchmark/benchmark_clustering.py:65` | **0.00** | the published silhouette / ARI benchmark |
| `code/4_post_detection.ipynb` cells 21-22 | **0.01** | *only* the Fig. R9 reads / unique-genes parquet |

The two radii disagree very differently by measure — a minimum-enclosing sphere has 2-4 support
points sitting *exactly* on its surface, so a 0.01 µm buffer systematically captures a few more
transcripts:

| Measure | exact agreement | delta (0.00 − 0.01) | median at 0.00 | median at 0.01 |
|---|---|---|---|---|
| unique genes | 93.3 % | mean −0.07, range [−4, 0] | 4 | 4 |
| reads | 12.0 % | mean −1.15, range [−9, 0] | 5 (pooled) | **6** (WT) / **7** (AD) |

Against a median of ~5 reads a difference of 1–3 reads is a large *relative* difference; neither
number is wrong, they are two measurements of the same spheres. So section 1 splits them:

- the **≥ 3 filter** runs on the **buffer = 0.00** unique-gene count, i.e. the same matrix the
  subset is taken from, so every retained granule genuinely has ≥ 3 genes in the profile carried
  forward — and it is the conservative direction, since 0.00 yields equal-or-fewer genes;
- the **reported read distribution** is the **published buffer = 0.01** one, so the supplement,
  the manuscript and the reviewer's *"median 6–7 reads / 4 genes"* all agree;
- the unique-gene distribution is reported at 0.00, matching the filter, which costs nothing
  because the medians are identical at 4.

`complexity_summary.csv` and `complexity_histogram.parquet` carry a `buffer` column so each panel
is self-documenting, and `complexity_crosscheck.csv` holds every number in the table above.

---

## Files

```
a2_config.py                 paths, gene sets, every constant; the only place to change a setting
a2_common.py                 ported computation (subtyping, density, permutation, scoring, exports)

A2a_multigene.ipynb          A2a end to end            [local, mcDETECT-env]
A2c_cooccurrence.ipynb       A2c end to end            [local, mcDETECT-env]
A2e_seed_content.ipynb       A2e end to end            [local, mcDETECT-env]
run_permutation_detect.py    A2b stage 1, one permuted detection (10 tasks)  [HGCC, SLURM array]
score_embedding.py           A2b stage 2, one COMBINED arm (6 tasks); --concat  [HGCC, array]
slurm/run_permutation.sh     detection array wrapper
slurm/score_embedding.sh     scoring array wrapper
slurm/concat.sh              stitches the per-arm tables
slurm/submit.sh              submits all three with the right dependencies
A2_figures.R                 all figures, per-section toggles         [local]
build_response_doc.py        builds plans/Response_R2_comment6_sparsity.docx  [local]
verify_response_doc.py       re-reads every asserted cell from its source     [local]
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
├── a2c/                     pair + group enrichment, clustermap, groups tested/dropped,
│                            go_abundance_stratified.csv
├── a2e/                     marker inventory, per-sample complexity, same-category content,
│                            seed × content table, chi-square / Fisher / permutation results,
│                            the a-priori compartment collapse, granule-level companion
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

---

## A2c — functional co-clustering of co-detected genes

**What it targets.** Reviewer #2, major point 6: *"because detection seeds on individual marker
genes, each granule essentially expresses just the single marker it was detected on … The granule
subtypes … therefore largely reflect the seeding marker rather than any genuine multi-gene granule
transcriptome."* A2c tests that directly, on the part of a granule mcDETECT **never touches** — the
270 non-seed genes.

The reviewer did not ask for a co-clustering analysis; that comes from our own action notes under
the same point. The claim quoted above is the thing to disprove.

### What came out — one positive result and two negative ones

Stated first, because the section below is method and the method is not the news.

**Positive, and it is the answer to the quoted claim.** Among the 270 genes detection never uses,
co-occurrence is strongly non-random: **33.9 %** of the 36,315 gene pairs reach z > 2 against the
2.5 % expected, **17.8 %** survive a Bonferroni threshold of z > 4.83, and the strongest pairs are
transparently interpretable modules — the neurofilament triplet (`Nefm`–`Nefh`, `Nefm`–`Nefl`), the
two GABA-synthesis enzymes (`Gad2`–`Gad1`), an oligodendrocyte pair (`Cnp`–`Sox10`), and
synaptic-vesicle machinery around `Stxbp1`. A granule whose content were exhausted by its seeding
marker could not produce this.

**Negative 1 — the localization-versus-co-expression contrast did not separate.** This section was
built so that it could fail, and it did. Taking the median across groups, **co-expression reaches
3.10 and localization 2.24** — the control programme scores *higher*. Ranked individually, one
localization group leads (Axons, 37.3, but on only four genes) and the next five are all
co-expression: Astrocytes 10.5, Microglia 10.5, OPC 9.0, Oligodendrocytes 7.9, Inhibitory neurons
7.8. `pre-syn` reaches 5.4, while **`post-syn` (16 genes, −0.90) and `Neuropil` (39 genes, the
largest localization group, −1.65) sit at or below the 0.35 background.** The ordering survives
every robustness arm: co-expression leads localization in `all` (3.10 / 2.24), `Isocortex`
(0.75 / 0.56) and `WT` (2.31 / 1.50), and only in `AD` does localization edge ahead, by 1.57 to
1.40. So A2c shows the non-seed content is
structured, but it does **not** show that structure is organised by localization rather than by
co-expression.

**Negative 2 — the external GO check is negative too.** See "The external GO check" below.

Both negatives are reported in the response document
(`build_response_doc.py` §3.0, §3.3) and §3.3 is written to be removable for exactly this reason.

### Scope: the full detection, non-seed genes only

**Not A2a's subset.** A2c runs on the **full** published detection, all 1,080,146 granules.
Conditioning on unique-gene count, as A2a does, would select on the very statistic being measured.
Granules with fewer than two genes contribute no pairs under either the data or the null, so
dropping them costs nothing.

**Not A2b's null.** A2b permutes at the detection level to answer a different question. Here the
null is a degree-preserving shuffle of the granule × gene table.

**Non-seed only.** `merge_sphere()` merges overlapping spheres seeded by *different* markers, so
co-occurrence among the 20 seed markers is partly manufactured by detection — 64.7 % of granules
carry ≥ 2 of them. Running the analysis on the seeds would reproduce exactly the circularity the
reviewer objects to. The seed arm is still computed as a **positive control for the statistic**, and
labelled detection-confounded wherever it appears. It behaves as designed: median z **16.5** across
its 190 pairs, against **0.35** for the 36,315 non-seed pairs.

### How `z` is computed, for one gene pair

This is the metric the whole section rests on. Implementation: `a2_common.py:607-744`
(`curveball`, `cooccurrence_enrichment`), driven by `run_arm` in `A2c_cooccurrence.ipynb` §2.
Tables named below live in `output/a2c/`; figures and their backing CSVs in `output/figures/`.

**0 — the matrix.** Binarise the published combined granule object's `layers["counts"]` to a
granule × gene presence/absence matrix `B`. Co-occurrence is a question about which genes are
*there*, not how many copies. Two filters, and the order matters:

1. **Columns first** (`run_arm`): restrict to the arm's genes — 270 non-seed for the primary arm.
2. **Rows second** (`a2_common.py:692-694`): drop granules with fewer than 2 genes *within that
   column set*. The primary arm keeps **593,195** granules and drops 486,951.

Because the row filter runs after the column subset, each arm uses a different granule set (593,195
for non-seed, 698,526 for the seed control). That is correct — the dropped granules carry no pairs
either way — but the arms are not on identical granules and should not be described as if they were.

One guard that is not optional (`a2_common.py:687-691`): `B` is cast to `float64` **before** any
multiplication. A boolean sparse matmul is *logical* — `True + True = True` — so `BᵀB` would
saturate every observed count at 1 while the null, built numerically, counted properly. That
produces a large abundance-dependent deficit that looks exactly like real biology.

**1 — observed.** `O = Bᵀ B`, so `O_ij` is the number of granules containing **both** gene *i* and
gene *j*. Asserted `O.max() <= n_granules`, which fails loudly if the matrix ever arrives
non-binary.

**2 — the null ensemble.** The null must remove the two effects that would otherwise masquerade as
co-occurrence: **complex granules pair everything with everything**, and **abundant genes pair with
everything**. So it holds both margins — each granule's gene count, each gene's granule count —
**exactly**, not in expectation, using the curveball trade algorithm (Strona et al., *Nat Commun*
5:4114, 2014; `a2_common.py:607`). One trade, on two rows held as index sets:

```
shared = ra ∩ rb;   only_a = ra − shared;   only_b = rb − shared
pool   = shuffle(only_a ∪ only_b)
ra ← shared ∪ pool[:|only_a|]      rb ← shared ∪ pool[|only_a|:]
```

Row sums are preserved because each row gets back exactly as many non-shared elements as it gave.
Column sums are preserved because every index in the pool appears exactly once going in and exactly
once coming out. Both constraints are hard — no rejection step, no approximation, no tolerance.

Chain schedule for the primary arm (`nnz` = 3,257,809 detections; all recorded in `run_info.csv`):

| | trades |
|---|---|
| burn-in | 5 × nnz = **16,289,045** |
| then **20** states, each separated by | 1 × nnz = **3,257,809** |

**3 — expectation and spread.** Each null state is rescored the same way, `O⁽ᵇ⁾ = B⁽ᵇ⁾ᵀ B⁽ᵇ⁾`.
Running sums of `O⁽ᵇ⁾` and `O⁽ᵇ⁾²` give

```
Ê   = mean_b O⁽ᵇ⁾
Var = max( mean_b O⁽ᵇ⁾² − Ê² , 0 ) × 20/19        sd = √Var
```

The `max(·, 0)` clamps floating-point cancellation; the 20/19 is Bessel's correction. **Both `Ê` and
`sd` are empirical** — nothing analytic enters at any point.

**4 — the statistic.**

```
z_ij = (O_ij − Ê_ij) / sd_ij
```

Non-finite values (a pair that never co-occurs in any null state gives sd = 0) become `NaN`; the
diagonal is `NaN`. Only the upper triangle is emitted — 36,315 rows for 270 genes — alongside
`log2_obs_over_exp = log2((O+1)/(Ê+1))` as a pseudocounted effect size. Worked example, the top pair
in `pair_enrichment.parquet`:

```
Vamp1–Nefh:   O = 4513    Ê = 1487.95    sd = 25.63    →    z = 118.0
```

For scale when reading any single value, the primary arm's z runs
**−20.0 / −2.18 / 0.35 / 3.29 / 25.4** at the 1st / 25th / 50th / 75th / 99th percentiles.

**Why not an analytic null.** The obvious shortcut is a bipartite configuration model that fixes the
degrees only *in expectation*. It was tried and rejected on measurement, and the bias is not subtle:
a granule with exactly *k* genes contributes exactly `C(k,2)` pairs, whereas a soft-degree null
contributes about `k²/2` — a factor `k/(k−1)`, which at the median complexity of ~5 genes is a
**~25 % over-estimate applied to every pair**. On simulated data it inflated z by ~96 sd. Curveball
has no such bias because its constraints are hard.

**What `z` is and is not used for.** It is **never** converted to a per-pair p-value. It feeds
exactly two things: the 270 × 270 clustermap, and the group test below, whose significance comes
from permuting gene → group labels rather than from assuming z is normal. So the only property z
needs is that it is computed identically for every pair — which it is.

**Three caveats, stated rather than buried.**

- **`n_null = 20` is small.** The Monte-Carlo error on `Ê` is `1/√20` = **0.224 null sd**, recorded
  as `mc_error_frac_of_sd` in `run_info.csv`; `sd` itself carries only 19 df (~16 % relative error).
  Roughly a fifth of a standard deviation of any z is estimation, not signal. This is a second
  reason z gets no p-value: z = 118 is far past what 20 draws could resolve as a tail probability.
- **The 20 states come from one chain**, spaced `nnz` trades apart, not from independent restarts.
  The variance estimate treats them as independent. At that spacing this is reasonable, but it is an
  assumption, not a proof.
- **Calibration is checked, not assumed — but the gate is off by default.** Setting
  `VALIDATE = True` (notebook §6a) curveballs a 50 K-granule submatrix 10 × `nnz` to draw a matrix
  *from the estimator's own null*, rescores it, and asserts z comes back ~N(0,1)
  (`|mean| < 0.25`, `0.8 < sd < 1.3`). **`output/a2c/null_calibration.csv` does not currently
  exist.** `build_response_doc.py` now reads that file rather than quoting a remembered number, so
  the notebook must be run once with `VALIDATE = True` before the response document will build.

### The group-level test

**Statistic:** the median z over within-group pairs. **Significance:** permuting the gene → group
assignment across all non-seed genes at fixed group size, 2,000 replicates, BH-corrected across
groups (`a2_common.py:755`). Pairs share genes and are emphatically not independent, so a test
treating them as independent observations would be badly anti-conservative.

**The permutation is abundance-matched, and it has to be.** Rare genes carry systematically higher
z, and the two programmes differ ~5.5-fold in abundance — localization groups have a median of
**~17,800** detections against **~3,250** for co-expression groups. An unmatched permutation would
compare abundant groups against mostly-rare random sets and bias the comparison before any biology
entered. Each replicate is therefore drawn with the group's own abundance-bin composition.

**Groups** come from the panel's own curated annotation (`gene_panel.csv`), which supplies both
kinds of programme. **28 groups** clear `MIN_GROUP_SIZE = 4` non-seed genes:

| programme | groups tested |
|---|---|
| **localization** | 4 — pre-syn (19 genes), post-syn (16), Neuropil (39), Axons (4) |
| **co-expression** | 24 — 8 cell-type sets, 11 regional sets, 5 cortical-layer sets |

Six fall below the threshold and are listed in `groups_dropped.csv` rather than dropped silently —
including **Dendrites** (3 genes), so one localization group is lost to panel size.

**The logic of the contrast.** Functional coherence on its own does *not* separate a granule from
"any co-expressed transcript cluster" — coherence is what co-expression looks like. What could
separate them is *which* programme organises the co-occurrence: if these are packaged transport
structures, localization sets should stand out and co-expression sets have no particular reason to.
As reported above, they did not separate.

**The Isocortex arm is the check on the regional confound**, and it does work as intended: within
one region the regional marker sets collapse (HPF-CA −3.97 → −0.33, HY 5.30 → −0.16, TH 3.39 → 0.55,
STR 1.17 → −0.10), confirming they were reporting tissue composition. But the background falls too
(median z 0.35 → 0.05) and the cell-type sets remain the leaders, so the collapse does not rescue
the contrast.

### The external GO check

Independent of the panel's annotation, and it covers every non-seed gene rather than only the
annotated ones: do pairs sharing a GO biological-process term co-occur more strongly?

**Pooled, the answer is no** — and in the wrong direction. Pairs sharing a term have a *lower*
median z than pairs that do not: **0.078 vs 0.464** (Δ −0.386, n = 8,571 vs 21,564).

**Stratified by abundance, it is a null rather than a reversal.** GO-annotated genes are the
better-studied, more abundant ones, and abundance drives z — the same confound that forced
abundance matching into the group permutation. Re-testing within expected-count deciles and
combining with a signed-rank test on the ten per-decile differences
(`a2c/go_abundance_stratified.csv`) gives 8/10 deciles negative, median Δ −0.357, **p = 0.105**.
So the pooled negative is largely the
confound; what remains is not significant, but it is not positive either.

**And the structure is not concentrated in a few modules.** Stratifying pairs by z
(`figures/a2c_go_by_z_bin.csv`) gives a U-shape against a 28.4 % baseline: 1.22× in the
most-depleted decile and 1.12× in the most-enriched, flat in between. Neither the "few strong
functional modules" reading
nor the "broad similarity gradient" reading survives.

**Runtime.** The null is an exact curveball chain over ~3.3 M detections. On the recorded run the
primary arm took **4.9 min** and all five arms **14.4 min** (`run_info.csv`); budget more on a
loaded machine. `DRY_RUN = True` subsamples and uses five null draws — run that first; its numbers
are indicative only.

---

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

# ---------- A2e: local, mcDETECT-env, run from this directory ----------
#   Run All, once. No gates, no manual pause, nothing to fill in. ~6-9 min: most of it is
#   loading the two h5ads and the 4,000 permutation replicates. The correctness gates always
#   run, and the Xenium arm skips itself with a printed note if its outputs are absent.
#   Independent of A2a/A2b/A2c.

# ---------- figures: local ----------
Rscript A2_figures.R

# ---------- response document: local, after every figure run ----------
python3 build_response_doc.py       # -> plans/Response_R2_comment6_sparsity.docx
python3 verify_response_doc.py      # re-reads every asserted cell from its source
```

`A2_figures.R` covers **all three** analyses: sections 1-4 plot A2a, section 5 plots A2b, section 6
plots A2c. Every block is guarded by a file check, so running it before the HGCC sweep prints
`[skip 5] missing: ...` and carries on; `RUN_A2B <- FALSE` at the top silences it.

Toggles worth knowing about, all at the top of the script:

| toggle | section | effect |
|---|---|---|
| `RUN_GSEA` | 3 | microdomain DE → GSEA. Pulls in clusterProfiler and is the slow one; also gates the GO test in section 6 |
| `RUN_GSEA_CHORD` | 3 | gene × pathway chord diagrams in the published mcDETECT style, on by default. Each is a 4000 × 4000 JPEG, so it roughly doubles section 3's output count |
| `COMPOSITION_BY_SAMPLE` | 2 | additionally facets the all-vs-multi-gene subtype composition bars by WT / AD |
| `RUN_A2B` | 5 | set `FALSE` to silence the permutation block before the HGCC sweep lands |

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

Every number in the response letter is re-derivable from `output/` alone — and that rule is
enforced rather than trusted. `build_response_doc.py` composes
`plans/Response_R2_comment6_sparsity.docx` by reading these files, so its prose and its tables
cannot drift apart, and `verify_response_doc.py` re-reads every asserted cell from its source
afterwards. Nothing in the document is typed by hand.

| Response element | Source |
|---|---|
| "`comp` counts markers, not genes" | `a2a/multigene/comp_vs_ngenes.parquet`, `figures/comp_vs_unique_genes.jpeg` |
| Reads / unique genes per granule (Fig. R9) | `a2a/multigene/complexity_summary.csv`, `figures/complexity_n_{reads,genes}_all.jpeg` — reads at the published buffer = 0.01 |
| Buffer / `granule_id` provenance, if queried | `a2a/multigene/complexity_crosscheck.csv` |
| "n granules retained at ≥ 3 unique genes" | `a2a/multigene/retention_by_region.csv`, `figures/multigene_retention.jpeg` |
| Subtype structure persists | `a2a/multigene/heatmap_subtype{,_ordered}.jpeg`, `subtype_composition.csv` |
| Subtype **composition** shifts but keeps its WT/AD direction | `figures/subtype_composition_all_vs_multigene.{csv,jpeg}` (pooled two-bar), `figures/subtype_composition_all_vs_multigene_by_sample.{csv,jpeg}` (WT/AD facets) |
| AD pre-synaptic reduction persists | `a2a/multigene/subtype_density_per_region_multigene.csv`, `figures/granule_density_multigene_pre-syn.jpeg`, `figures/granule_density_all_vs_multigene.jpeg` |
| Microdomain contrast persists | `a2a/multigene/neuropil_subdomains_Isocortex_50/`, `figures/gsea_terms_published_vs_multigene.csv` (carries `layer`/`target`/`reference`, so each NES's contrast is explicit) |
| Enriched pathways, in the published mcDETECT figure style | `a2a/multigene/neuropil_subdomains_Isocortex_50/*_{positive,negative}_chord_{diagram,legend}.jpeg` — gene × pathway chords, one pair per DE table per direction, beside the `*_{target,reference}_GSEA.jpeg` dotplots |
| Microdomain **partition** persists, not only the DE | `figures/subdomain_correspondence.{csv,jpeg}` — spot-level map of multi-gene vs published subdomains on the same inherited grid; section 3 also states at run time whether the chosen contrast points the same way as the published one |
| "not a low-count artifact" | `a2a/readstrata/readstrata_density.csv`, `figures/readstrata_density_*.jpeg` |
| "randomized data does not give the same embedding" | `a2b/metrics/a2b_metrics.csv`, `figures/a2b_silhouette_score.jpeg`, `figures/a2b_ari_stability_mean.jpeg`, `figures/a2b_structure_at_k15.csv` — quote the **size-matched** facet |
| Permutation yields a different granule count | `figures/a2b_granule_counts.jpeg`, `a2b/metrics/a2b_detection_summary.csv` |
| Any null arm too small to embed | `a2b/metrics/a2b_status.csv` |
| Granules carry structured non-seed content | `a2c/pair_enrichment.parquet` (per-pair `observed`/`expected`/`null_sd`/`z`), `a2c/run_info.csv` (null schedule + MC error), `a2c/group_enrichment.csv`, `figures/a2c_group_enrichment.jpeg` |
| Localization vs co-expression — **the contrast did not separate** | `figures/a2c_programme_summary.csv`, `figures/a2c_group_enrichment_by_arm.jpeg` (Isocortex column is the regional control). Quote this as the negative result it is, not as support |
| GO check — **also negative** | `figures/a2c_go_shared_term_test.csv` + `a2c_go_shared_term.jpeg` (pooled), `a2c/go_abundance_stratified.csv` (the confound, and the null that survives it), `figures/a2c_go_by_z_bin.{csv,jpeg}` + `a2c_pair_go.parquet` (neither concentrated nor a gradient) |
| Co-occurrence block structure | `a2c/cooccurrence_clustermap.jpeg`, `a2c/clustermap_gene_order.csv` |
| Permuted detections are somatic | `figures/a2b_in_soma_survival.jpeg`, `a2b/metrics/a2b_detection_summary.csv` |
| Real vs permuted t-SNE | `a2b/metrics/tsne_matched_{real,perm}_seed0.jpeg` (equal n — use this pair), `tsne_real.jpeg` / `tsne_perm_seed*.jpeg` (full n) |
| Granules are multi-gene **per sample**, incl. Xenium | `a2e/complexity_by_sample.csv` (WT / AD / Xenium, both NC conventions), `a2e/same_category_content.csv` (pure-subtype granules carrying a second marker of their own category) |
| Content is not randomly distributed w.r.t. seed category | `a2e/seed_content_tests.csv` — the asymptotic chi-square **and** the 2,000-shuffle granule-label permutation side by side, with `design_effect` connecting them. `a2e/seed_content_table.csv` carries `fold_vs_expected` on all 16 cells, which is where the block structure is visible. Quote the **`nonseed_content`** arm whenever the merge confound is raised |
| Content tracks the seed's **compartment** | `a2e/seed_content_compartment.csv` — the a-priori 2 × 2 collapse (`C.COMPARTMENT_OF`), carrying a chi-square (`chi2`, `chi2_neg_log10_p`, `cramers_v`), a two-sided Fisher's exact test on each diagonal cell (`odds_ratio`, `p`, `neg_log10_p`, `direction`) and a permutation p, so "Fisher's exact test on each diagonal cell" in Methods covers this table as well as the four-way one. **This is the table that tests the claim**, not the four-way diagonal; see the caveat below. `a2e/seed_content_granule_level.csv` is the granule-unit companion if the transcript-level p-value is challenged |
| Per-category detail | `a2e/seed_content_diagonal.csv` — two-sided Fisher with a `direction` column and the permutation null band. Same-category enrichment holds for pre-syn and axons; post-syn and dendrites do **not** show it and the reason is anatomical, not a failure |
| Denominators behind every A2e number | `a2e/marker_inventory.csv` — markers per category per platform, split into seed and non-seed. Xenium has 24 markers, two of them dendritic and two axonal; its fractions are **not** comparable to MERSCOPE's |

Images are JPEG at dpi 500; convert to PNG before embedding in Word (the Adobe APP14 marker
trips Word up).

---

## What this does and does not settle

**Does.** A2a shows the subtype structure, the AD pre-synaptic density reduction and the
microdomain contrast survive restriction to granules that cannot be explained by their seeding
marker alone, and that the WT/AD effect is not confined to the lowest read tercile. A2b shows the
embedding structure is not reproduced by data with identical positions, identical density and
identical per-gene totals — and, because of the size-matched arms, not merely because the null
has fewer granules to work with. A2c shows the genes detection never touches are non-randomly
co-detected: 33.9 % of non-seed pairs reach z > 2 against 2.5 % expected, 17.8 % survive
Bonferroni, and the strongest pairs are interpretable modules. That is a direct answer to
*"each granule essentially expresses just the single marker it was detected on"*. A2e adds the
per-sample version of that answer — including Xenium 5K, which none of the other three touch — and
shows that co-detected content is associated with the seeding marker's **compartment** — axonal
vs somatodendritic — rather than distributed at random. That association is what
`a2e/seed_content_compartment.csv` reports; the four-way diagonal is a finer split than the
anatomy supports and is not the test.

**Does not.** None of the three is a test of whether granules are biologically real — that burden
sits with A3 (ambient / pseudo-granule controls) and the EM validation.

**A2c specifically does not establish that the co-occurrence is *localization*-organised.** Its
own discriminating contrast failed: co-expression groups score at or above localization groups in
every arm, `post-syn` and `Neuropil` sit at background, and the external GO check is negative
pooled and null once abundance-stratified. Showing that co-detected genes are functionally related
would not by itself separate a granule from "any co-expressed transcript cluster" anyway — that is
why the control was built in, and the control did not come out our way. Report it; do not lean the
argument on it.

**A2b's null is a *global* shuffle.** It destroys all spatial gene structure, including the
regional composition gradients that any real tissue has, so it is the reviewer's stated
hypothetical rather than the strictest possible null. A block-wise permutation preserving regional
composition would be a harder test; it is not what was asked for, and it is noted here so the
choice is on the record rather than implied.

Three further caveats worth disclosing in Methods:

- A2a's absolute densities are lower than the published ones purely because the subset is smaller —
  only the WT-vs-AD direction transfers.
- A2a's subdomains are recomputed, so their numbering carries no relation to the published
  Subdomain 1–4.
- A2c's single strongest group, Axons, rests on **four** non-seed genes and six pairs. It is the
  one localization group that behaves as the design predicted, and it is too small to carry weight
  on its own.
- A2e's `all_content` arm is partly manufactured by detection: `merge_sphere()` keeps one seed
  marker of a merged pair and discards the other (`model.py:323-377`), and 8 of the 20 MERSCOPE
  seeds are pre-syn, so a pre-syn-seeded granule carries a second pre-syn marker more often than
  chance would give even with no biology. That is what the `nonseed_content` arm controls, and it
  is the arm to quote. It is thin, though — 4 pre-syn, 8 post-syn, **1** dendritic (`Map2`) and
  **1** axonal (`Ank3`) marker — so the last two rows carry a fold change, not independent support.
- A2e's omnibus chi-square counts **transcripts**, which are not independent within a granule. The
  size of that problem is measured rather than assumed: shuffling the seed-category label across
  granules (which keeps each granule's whole content vector, so all within-granule dependence
  survives) puts the null χ² at 17.2 against the asymptotic 9.0 — a **design effect of 1.9**, and
  1.2 for `nonseed_content`. Correcting for it takes χ² from 206,412 to ~108,000, both far beyond
  any threshold, and the observed statistic exceeds the largest of 2,000 null draws by 5,340×. It is
  mild because the clusters are tiny: after removing each granule's own seed gene, granules carry a
  mean of 2.36 marker transcripts (median 1; 32.5 % contribute none). Report the asymptotic and
  permutation columns together — `seed_content_tests.csv` carries both — and quote
  `seed_content_granule_level.csv` if the unit itself is disputed.
- **A2e's four-way diagonal is not the frame in which the claim is testable, and reporting it alone
  would look like two failures.** Same-category enrichment holds for pre-syn (1.33, and 1.67 with
  seeds removed) and axons (2.60 / 2.54) but *not* for post-syn (0.69 / 0.90) or dendrites
  (0.48 / 1.07). The reason is anatomical: the four marker sets are **two compartments split into
  two overlapping labels each** — the postsynaptic density sits inside dendritic spines, and
  presynaptic terminals are axonal structures — so `post-syn` and `dendrites` label one physical
  compartment and cannot separate from each other. `seed_content_table.csv` shows the resulting
  block structure directly (post-syn → dendrites 1.40, dendrites → post-syn 1.51). Collapsed to the
  two compartments the association is clean and **identical in both arms** — axonal-content fold
  1.405 and 1.395 — which is `seed_content_compartment.csv`. `C.COMPARTMENT_OF` is fixed a priori
  from standard neuroanatomy and lives in `a2_config.py`, so "specified before the table was seen"
  is checkable; present it that way, because presented as a regrouping it reads as p-hacking.
- In a 2 × 2 the odds ratio is invariant to swapping both rows and columns, so **both diagonal rows
  of `seed_content_compartment.csv` carry the same `odds_ratio` and the same Fisher `p`** — that is
  arithmetic, not a duplicated record. The two compartments are told apart by `fold_enrichment`
  (1.405 axonal vs 1.322 somatodendritic in `all_content`), which is a ratio of proportions and is
  the number to quote. Never use the odds ratio and the fold interchangeably.
- A2e's cleanest internal control sits in the `nonseed_content` arm, whose dendritic content is
  **`Map2` alone**: it comes in at 0.27 × expected in pre-syn-seeded granules and 0.23 × in
  axon-seeded ones. MAP2 is the canonical somatodendritic marker, actively excluded from axons.
- A2e's Xenium arm uses a **24-marker** feature space, not 34: eleven of the 34 are off the Xenium
  panel and `Snap25`, absent from MERSCOPE, is on it. Its seed list is also a different 16 markers,
  not a subset of the MERSCOPE 20. Report Xenium beside MERSCOPE, never pooled with it.
