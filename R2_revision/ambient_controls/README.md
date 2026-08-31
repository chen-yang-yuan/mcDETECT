# A3 — ambient RNA controls at the detection step

Answers **Reviewer #2, major point 9** (`plans/Round2_response_analysis_plan.md` §A3). Round 2's
comment is a continuation of round 1's, and the reason it is still open is that the two controls
the reviewer offered in round 1 were **never run**.

> "The ambient background is modeled as complete spatial randomness (CSR), yet ambient RNA from
> debris, dying cells, and extracellular vesicles is typically spatially structured rather than
> uniform, and is likely to be denser precisely where cells are denser, or in regions of severe AD
> pathology, such as around amyloid plaques… A CSR-based threshold could therefore under-correct in
> such regions and inflate granule calls locally, which is a particular concern for the WT versus
> AD comparison. **The downstream density regression is reassuring but does not test this, since it
> operates on granules that have already been called.** A direct check at the detection step, such
> as the pseudo-granule negative control I suggested, would settle the question."

The round-1 offer, verbatim:

> "As a potential control, the authors could consider performing a **differential expression
> analysis between somatic RNA and all non-somatic RNA, independent of granule detection**, and then
> assess to what extent the observed granule-specific differences exceed or diverge from this
> baseline non-somatic signal. **Or alternatively, define pseudo-granules in the direct vicinity of
> actual granules** as a negative control."

Three sub-analyses, all local except the detection runs they consume:

| | what | where |
|---|---|---|
| **A3a** | the PI's three-set negative-control split, + the locally-adaptive threshold re-test | `A3a_three_sets.ipynb` [local] |
| **A3b** | the reviewer's vicinity pseudo-granule control | `A3b_vicinity.ipynb` [local] |
| **A3c** | the reviewer's somatic-vs-non-somatic DE baseline | `A3c_de_baseline.ipynb` [local] |
| — | the Set 0 / 1 / 3 detections all three consume — **6 array tasks** | `run_detection_sets.py` [HGCC, SLURM array] |

---

## What was already on disk, and whether it answered the point

**`code/benchmark/benchmark_ambient.ipynb` does not.** It is an OLS regression of per-spot granule
*density* on `AD + ambient_marker_cov`, per (subtype, region):

| the reviewer asked for | that notebook does |
|---|---|
| a **differential expression** analysis | a regression on granule **density** — no gene-level contrast anywhere |
| between **somatic and all non-somatic RNA** | no somatic layer at all |
| **independent of granule detection** | `ambient = extrasomatic − granule_expression`, i.e. defined *by subtracting called granules* |

It is precisely the analysis round 2 dismissed. Reused here only as scaffolding
(`CAPTURE_EFFICIENCY_COEF`, `GRID_LEN`, `bonferroni_within_group`).

**`code/old/benchmark_diffusion.ipynb` largely does.** It already computes the reviewer's Axis 1 —
`baseline_logFC` (all extrasomatic vs somatic — which A3c now adopts as its **primary** baseline,
being the reviewer's literal wording and the only detection-independent form), `granule_enrichment`, their
difference `delta`, and a regression fitted on non-markers — and `code/figures_response.Rmd:1421-1452`
already carries its scatter under a heading literally titled *"Reviewer 2, Major Comment 9"*. It was
built for round 1, then **not used** in the response; its output CSV is no longer on disk and the
notebook sits in gitignored `code/old/`. A3c revives it and closes six gaps:

| # | gap | fixed in |
|---|---|---|
| 1 | WT only — the reviewer names *conditions* | A3c §5 |
| 2 | `delta` subtracts two logFCs with different references; the correct branch (`USE_ALT_GRANULE_VS_SOMA`) ships **off** | A3c §3 |
| 3 | the baseline was never tested, and no granule-free variant existed to check it against | A3c §1, §4 |
| 4 | no significance on `granule_enrichment` or `delta` | A3c §4 |
| 5 | uses `all_granules` + post-hoc filtering; keys `nc_ratio` on `sphere_z` where `nc_filter` uses `layer_z` | A3c §1 |
| 6 | cell 7 is O(n_spots × n_transcripts) — a 103M-element mask per spot | A3c §1 |

---

## Provenance traps

Five things about the published pipeline that a reviewer reading the code can find, and that the
response is better off stating first.

### 1. The CSR threshold never runs on real data — but α = 0.5 makes that a non-issue

`code/3_detection.py:66,83` passes `minspl = 3`, so `poisson_select()` — the CSR background model
the reviewer objects to — is bypassed, and `alpha=10` / `cutoff_prob=0.95` are inert arguments.
(It *is* used in `simulation/model.py`, which is what Fig. 2 rests on.)

At the `alpha = 10` written in that call, the CSR rule would have been far **stricter** than what
was used — Camk2a → 32, and > 3 for 19 of the 20 markers — which would make the gap damaging. But α
is a free scaling factor, and **at α = 0.5 the rule returns exactly 3 for all 20 markers in both
samples** (0.25 also works; 0.75 works for AD but not WT). So `min_samples = 3` *is* the CSR rule at
α = 0.5, no result changes, and the disclosure is two sentences.

**A3 runs nothing at α = 0.5, and re-detects nothing to prove the equivalence.** `poisson_select`
is a deterministic function of one gene's transcript count and the tissue area, so
"α = 0.5 ⇒ `min_samples` = 3 for every marker in both samples" is an **arithmetic identity** —
`output/preflight/csr_min_samples.csv` plus one assertion in the A3a gates. Since DBSCAN is
deterministic given `min_samples`, `minspl=None, alpha=0.5` and `minspl=3` are provably the same
run, so the published populations are reused as they are and every detection here passes
`minspl = 3` verbatim.

**Separate action item, outside A3 and not blocking it:** change `code/3_detection.py` to
`minspl = None, alpha = 0.5` and update Supp. Note 10 / Methods to state α = 0.5, so the code says
what the response says. A Methods-consistency edit, **not** a re-run — the output is unchanged by
the identity above.

### 2. `Gria2` is on both the marker list and the NC list

`nc_filter` counts it in the numerator and `size` counts it in the denominator, so Gria2-seeded
granules self-filter. **This is a fixed policy, not an analysis** — A3a no longer partitions or
quantifies it, and the response does not raise it.

| list | role | in A3 |
|---|---|---|
| `syn_genes` — **20 granule markers** | the **detection seeds** (`3_detection.py:55`) | **unchanged everywhere.** Sets 1 and 2 both seed on all 20 |
| `nc_genes` — **19 negative controls** | the **NC filter**, and Set 3's seed list | two versions, chosen by provenance |

Only the **NC list** is adjusted:

| NC-list use | version | why |
|---|---|---|
| Set 3's seed list | **18**, Gria2 dropped | seeding a control population on a canonical dendritic marker would manufacture the very overlap Set 3 exists to bound. Gria2 is ~13× more abundant than the median control gene, so keeping it could roughly double Set 3 |
| `nc_ratio` recomputed on Set 2 | **19**, as published | must reproduce the filter the data on disk was built with |

**Set 1 still seeds on all 20**, because its job is to be "Set 2 minus the NC filter" and seeding
it on 19 would make it differ from Set 2 in two ways at once. `_remove_overlaps`
(`model.py:323-377`) propagates whole rows — containment with B larger does
`set_a.loc[i] = set_b.loc[j]`, replacing A's row *including its gene label* — so a Gria2 sphere can
absorb, or be absorbed by, another marker's sphere; dropping it from the seeds would change the
geometry and labels of **non-Gria2** granules too.

The residual discrepancy this accepts is under half a percent of Set 1, in the same direction in
both samples, so it cannot move the WT/AD contrast. There is no all-19 detection arm:
`Set 3′ − Set 3` is exactly the Gria2-seeded spheres, which Set 1 already contains.

### 3. `size` is stale after merging, so `nc_ratio` mixes two geometries

`_remove_overlaps` updates only `sphere_x/y/z`, `layer_z` and `sphere_r`. So for every **merged**
granule, `nc_filter` computes its numerator on the enlarged post-merge sphere and divides by the
pre-merge `size` — inflating `nc_ratio` exactly for the multi-marker, most confidently real
granules. Merging frequency is density-dependent, hence region- and condition-dependent.
`comp`, `gene` and `in_soma_ratio` are stale the same way (`plans/analysis_details.md` already
documents the `comp` case).

Two consequences the pipeline has to work around, and why every detection run here persists the
per-gene `sphere_dict`:

* `granules.parquet["gene"]` is **not reliably the seed gene**, which A3b's seed-matched detection
  predicate needs;
* `size` is **not** `k_g`, the own-gene count on the final geometry, so anything that needs a
  cluster's own transcript count must recount it by ball query (with `C.KG_BUFFER` — `sphere_r` is
  the *minimum-enclosing* radius, so support points sit exactly on the surface).

A3a does not analyse this defect: it is a property of the published filter, not of the ambient
question, and raising it would audit our own result without answering the reviewer.

### 4. The NC list is nuclear-enriched but not gene-neutral

The manuscript describes it as "enriched in neuronal nuclei compared to the cytoplasm"
(Supp. Table 8), which is right. But the 19-row file is an edgeR table whose `Cluster` column spans
`C4a` (complement), `Cyfip1`, `Abca7` (AD risk), `Opalin` (oligodendrocyte),
`Prox1`/`Npnt`/`Zfpm2` (dentate gyrus) — so the NC background is itself **spatially structured and
plausibly condition-dependent**. Under the current framing that is a *feature*, not a problem:
Set 3 is not claimed to be a spatially uniform null, it is claimed to be the ambient population,
and structured ambient is exactly what the reviewer says exists. What must be disclosed once and
not chased is narrower — two of the eighteen (`Cpeb1`, `Cyfip1`) are known dendritic RNA-binding
proteins, so the panel is not perfectly soma-restricted. Note also that `nc_top = 20` against a
19-gene list makes the "top 20 by expression" step a no-op.

### 5. Markers are 15× more abundant than NC genes

Median per-gene transcripts: **markers 1,153,633 vs NC 74,893**. DBSCAN yield is strongly
superlinear in count, so a near-empty Set 3 would be trivially explained by rarity. This is why
**Set 0** exists and why every set is reported as a per-million-transcript **rate**, not a count.

---

## Files

```
ambient_controls/
├── README.md                  this file
├── a3_config.py               paths, gene sets, thresholds; every copied constant carries
│                              a "# from code/3_detection.py:55" provenance comment
├── a3_common.py               ported computation; vectorised primitives with named sources
├── run_detection_sets.py      [HGCC, SLURM array] Sets 0 / 1 / 3 (6 tasks)
├── run_pseudo_detection.py    [HGCC, SLURM array] A3e's re-detection over the relabelled
│                              transcript table (2 tasks); builds nothing, applies a patch
├── A3_preflight.ipynb         [local, BEFORE the array] CSR table, Set-0 gene list, Set-2
│                              diagnostics, z profile
├── A3a_three_sets.ipynb       [local] the ambient control population: funnels, overlap, density
├── A3b_vicinity.ipynb         [local] the vicinity pseudo-granule control
├── A3c_de_baseline.ipynb      [local] the somatic-vs-non-somatic DE baseline
├── A3d_local_null.ipynb       [local] the local-neighbourhood permutation null
├── A3e_pseudo_granules.ipynb  [local + HGCC] ambient pseudo-granules, re-detected
├── A3_figures.R               all figures, sections numbered 1:1 to the notebooks
└── slurm/
    ├── run_detection.sh       one array task = one (set x sample)
    ├── run_pseudo_detection.sh  one array task = one sample (A3e)
    └── submit.sh              derives the array size from a3_config
```

### Outputs (git-ignored)

```
output/
├── preflight/    csr_min_samples.csv, set0_genes.csv, set2_diagnostics.csv, z_profile_<sample>.csv
├── detect/       <set>_<sample>/{spheres,sphere_dict}.parquet, funnel_by_gene.csv, run_info.csv
├── a3a/          set_inventory.csv, funnel_by_gene.csv, overlap_ladder.csv,
│                 overlap_transcript_level.csv, set2_reproduction.csv,
│                 set_density_per_region.csv, capture_ratio_per_region.csv
│                 (seven files, and every one of them is quoted in the response)
├── a3b/          source_summary.csv, placement_status.csv, vicinity_overlap_with_real.csv,
│                 profile_summary.csv + profile_histogram.parquet, profile_funnel.csv,
│                 detection_predicate.csv (+ _stratified, _thinned),
│                 rough_variant_by_distance.csv
├── a3c/          partition_counts.csv, transcript_layer_<sample>.parquet (shared cache),
│                 spot_layer_counts.parquet, clip_bias_by_gene.csv, clip_bias_scope.csv,
│                 axis1_gene_table.csv,
│                 axis1_summary.csv, axis1_count_model.csv, axis1_divergence_test.csv,
│                 axis2_wt_ad_by_layer.csv, axis2_layer_correlation.csv,
│                 axis1_nonseed_{annotation,scope,genes,reproducibility}.csv,
│                 axis1_count_model_neutral.csv
├── a3d/          a3d_bin_layer_counts.parquet (the 10 um grid),
│                 a3d_local_null_scope.csv, a3d_local_null_genes.csv,
│                 a3d_local_null_group.csv + _group_null.csv,
│                 a3d_local_null_calibration.csv, a3d_local_null_negative_control.csv
├── a3e/          a3e_relabel_<sample>.parquet (THE PATCH -- this is what goes to HGCC),
│                 a3e_pool_ladder.csv, a3e_relabel_scope.csv, a3e_relabel_audit.csv,
│                 a3e_precheck.csv, a3e_match_floor.csv,
│                 a3e_redetection_rate.csv, a3e_marker_shift.csv,
│                 detect_<sample>/{spheres,sphere_dict}.parquet, funnel_by_gene.csv, run_info.csv
└── figures/      everything A3_figures.R draws
```

---

## The sets

| set | seeds on | filters | source |
|---|---|---|---|
| **Set 0** | 20 panel genes unannotated for synapse / neuropil / NC, **abundance-matched** to the markers | size + in-soma | new |
| **Set 1** | the 20 granule markers | size + in-soma, **no NC** | new |
| **Set 2** | the 20 granule markers | size + in-soma + NC | **published** `granules.parquet` |
| **Set 3** | the NC genes **minus Gria2** (18 nuclear-enriched genes) | size + in-soma, **no NC** | new |

**Set 3 is the ambient population, not a null.** The NC filter is deliberately not applied to it:
those genes are the seeds now, so filtering on them would be circular. What survives the in-soma
filter is extrasomatic aggregates of soma-restricted transcripts — the debris / dying cells /
extracellular vesicles the reviewer names.

**Set 1 is a true re-detection, not a post-hoc filter of `all_granules.parquet`.** mcDETECT applies
the size and in-soma filters at the end of `dbscan()`, i.e. *before* `merge_sphere()`, so filtering
the rough pass afterwards does not reproduce Set 2's construction (737,063 vs 681,337 spheres in WT)
and the comparison would measure filter *order* rather than the NC filter. That Set 1 and Set 2
differ in **nothing else** is verified, not assumed — see `a3a/set2_reproduction.csv`.

**Set 0 answers rarity, and only rarity.** Set 3's genes are ~15× rarer than the markers and DBSCAN
yield is superlinear in count, so "Set 3 is small" could be abundance alone. Set 0 removes that
objection; it is *not* a soma-restricted panel and must never be described as one. Its primary
statistic is the **per-million-transcript rate**, not the sphere count: the panel has no
unannotated gene above ~300 K transcripts, so the rarest markers match within ~2 % while the most
abundant are matched several-fold low — worst `Camk2a` 6,237,713 vs `Zbtb20` 266,796 (23×), and in
aggregate 33.3 M marker transcripts against 9.7 M for Set 0 (3.4×). Four of the twenty (`Grin2b`,
`Dner`, `Epha4`, `Ncam1`) are documented dendritic transcripts, so a higher Set 0 yield is
expected on biology. `A3_preflight.ipynb` prints the match quality and `preflight/set0_genes.csv`
records it per marker.

Every set is reported as a **funnel** (raw → size → in-soma), because for Set 3 the stage at which
it thins *is* the mechanism: soma-restricted transcripts cluster densely inside somata, and it is
the extrasomatic residue that the analysis bounds.

---

## A3a — `A3a_three_sets.ipynb`

**Four sections, one per claim, and nothing else.** The notebook computes only what the response
quotes: every table it writes is cited in §1 of `plans/Response_R2_comment9_ambient.docx`, and
there is no correctness-gate section, no forensics section and no "for the record" output.

| § | claim it produces | writes |
|---|---|---|
| 1 | structured ambient RNA exists, and it is **sparse** | `set_inventory.csv`, `funnel_by_gene.csv` |
| 2 | control aggregates **rarely coincide** with granules, and the NC filter removes most of those that do | `overlap_ladder.csv`, `overlap_transcript_level.csv`, `set2_reproduction.csv` |
| 3 | control density **cannot reproduce** the WT-vs-AD result | `set_density_per_region.csv`, `capture_ratio_per_region.csv` |

Section numbering maps 1:1 onto `A3_figures.R`'s `[1]`–`[3]`. Runs **once, top to bottom, with
nothing to adjust** — everything it needs is on disk by the time it starts. Requires the
`mcDETECT-env` kernel: §2 imports `mcDETECT` to reproduce Set 2 from Set 1.

### The argument, in the order the response makes it

**Set 3 is the reviewer's own mechanism, run through our detector.** The reviewer names debris,
dying cells and extracellular vesicles — all of which release *somatic* RNA. The NC list is
defined in the manuscript as transcripts *"enriched in neuronal nuclei compared to the cytoplasm"*
(Supp. Table 8), so an **extrasomatic** aggregate of them cannot be a granule; it can only be
released material. Seed on those genes, apply the size and in-soma filters, apply **no NC filter**
(they are the seeds now), and what survives *is* the structured ambient population.

Write **nuclear-enriched (hence soma-restricted)**, not "soma-enriched" — that is what the
manuscript says, and it is the stronger claim. One thing to disclose once and not chase: two of
the eighteen (`Cpeb1`, `Cyfip1`) are known dendritic RNA-binding proteins.

**Set 3's separation appears AT the in-soma filter, and that is the design, not an artefact.**
Raw rates differ only ~2× between Set 3 and the markers; after the in-soma filter they differ
~10× (WT) and ~15× (AD). Soma-restricted transcripts do cluster densely — inside somata. The
filter removes those, and the residue is what the analysis bounds. The funnel is reported for
exactly this reason: it makes the stage visible instead of asserting an endpoint.

**Set 0 answers rarity, not ambient.** NC genes are ~15× rarer than the markers and DBSCAN yield
is superlinear in count, so a small Set 3 could be rarity alone. Set 0 — panel genes unannotated
in the *Synapse markers* / *Neuropil* / *Negative controls* columns, matched 1:1 on
`log10(count)` — closes that off, and every count is additionally reported per million
transcripts of the seeding gene. Two limits to state wherever Set 0 is used: the match degrades
badly at the top (`Camk2a` 6,237,713 vs `Zbtb20` 266,796, 23×; the pool has nothing above ~300 K),
and *unannotated* is not *non-dendritic* — `Grin2b`, `Dner`, `Epha4`, `Ncam1` are documented
dendritic transcripts, so a higher Set 0 yield is expected on biology and is **not** evidence of
ambient contamination. Set 3 carries the ambient reading; Set 0 carries the abundance reading.

**Set 1 is the pivot, and its premise is measured.** Set 1 minus Set 2 isolates the NC filter only
if the two differ in nothing else — so §2 re-applies the published filter to Set 1 and records the
agreement in `set2_reproduction.csv` (WT 681,346 vs 681,337 published; AD exact). The residual is
`miniball`'s randomised fit. This replaces the old correctness gate: it is a reported result, not
a check.

**Set 1 is also the built-in positive control for §3.** The statistic that carries the biology is
the per-region AD/WT density *ratio*. Set 1 is Set 2 minus one filter, so if that profile is
recoverable at all Set 1 must recover it — and it does (ρ ≈ 0.98). Neither control does. Lead §3
on that contrast, not on per-region sign agreement, which is a coin flip on as few as 2 spheres in
the sparsest AD region.

### The overlap criterion, and both directions

mcDETECT's own merge predicate (`model.py:349-353`, with `l=1`, `rho=0.2`) is

```
merge(A,B)  ⟺  d ≤ |r_A − r_B|   (containment)   OR   d < 0.2·(r_A + r_B)
```

so two equal-radius spheres merge only when their centres are within `0.4·r`. That is very strict —
real granules routinely overlap without merging — so quoting only that predicate would understate
co-location and read as rigged. The ladder therefore **leads with `intersect`** (`d < r_A + r_B`),
the loosest criterion: a small overlap under it cannot be argued with. Overlap is also reported at
**transcript level**, which is merge-invariant (granule-level cardinality is partly an artefact of
`merge_sphere`'s gene order, whose base is `sphere_dict[0]`).

**Both controls are scored against both granule sets, in both directions** — 2 × 2 × 3 criteria ×
2 samples = 24 rows.

| column | question |
|---|---|
| `frac_overlapping` | what share of **granules** meets a control aggregate — bounds contamination of the published result |
| `frac_control_overlapping` | what share of **control aggregates** meets a granule — how much of the ambient population the detector would have had to pick up |

**Both carry a null**, and the second one needs it more than the first: Set 1 holds 741 K spheres
over an 18.8 mm² section, so a sphere dropped anywhere in the tissue meets one at ~9 % by geometry
alone. Quoting the raw 42.9 % without its 9.4 % expectation beside it would read as damning when
the enrichment is 4.6×. Nulls are 20 uniform re-placements in the tissue mask at matched radius and
matched `layer_z`, seeded per `(sample, control)` with `crc32` so kernel restarts reproduce.
`center_in` is asymmetric by construction, so under that rung the reverse direction is the
**mirrored predicate**, not the same number read backwards.

**Report the enrichment honestly.** Obs/exp stays above 1 for Set 3 against Set 1 in both
directions — ambient aggregates and granule candidates are not independently placed, i.e. the
reviewer's mechanism is real. The argument is magnitude plus the fall after filtering (WT 6.9× →
3.0× granule-side, 4.6× → 2.5× control-side; AD 3.2× → 1.2× and 2.6× → 1.2×), never "co-location
is at chance."

**The transcript-level specificity contrast is the best single number in §2.** The NC filter cuts
the Set-3-coinciding transcript fraction ~3.9× (WT) and ~5.1× (AD) but the Set-0-coinciding
fraction only ~1.1×. The filter is not trimming co-located material indiscriminately; it is
preferentially removing what coincides with soma-restricted transcripts.

### What was removed, and why

| removed | why |
|---|---|
| corrected `nc_ratio` (old §2a) | audits our own published filter; supports no claim |
| NC leave-one-out (old §2b) | argues the NC list is not gene-neutral — cuts *against* the filter-works claim |
| Gria2 partition + gap sensitivity (old §2c/2d) | not central; the policy is stated once and not analysed |
| the Jaccard rung | orphan — nothing ever read `overlap_jaccard_summary.csv` |
| the locally-adaptive threshold re-test (old §6) | Set 2 only, Set 3 never entered, equivocal, and its specified per-region deliverable was never produced. **A3b's vicinity control answers the same reviewer sentence with a better design.** |
| the correctness-gate section (old §7) | checks, not results. The one load-bearing gate became `set2_reproduction.csv` |

Also gone from `a3_common.py`: `jaccard_balls`, `nc_leave_one_out`, `gria2_partition`,
`nc_ratio_corrected`, `local_lambda_grid`, `_occupancy_fraction`, `disc_sum`,
`adaptive_min_samples`, `adaptive_survival` — and from `a3_config.py` the whole `ADAPTIVE_*` block
and the ~70-line Gria2 policy essay, now three lines. The Set 2 reproduction calls
`mcDETECT.nc_filter` directly, so no local re-implementation of the filter is needed at all.

`overlap_pairs` now accepts a **list** of criteria and scores the whole ladder from one candidate
query, which is what makes 2 controls × 2 bases × 21 pairings × 2 directions affordable.

---

## Corrections applied in the rebuild (2026-08-27)

Four defects were found by audit and fixed; all four changed reported numbers.

| defect | effect | now |
|---|---|---|
| `KG_BUFFER` defined but **never read** — three sites queried at a bare minimum-enclosing radius | A3b real arm read 98.2% with `median_n_local = 2`, below the `min_samples = 3` that formed the cluster; A3c's granule layer lost ~12% of its transcripts, 94% of them markers | applied at all three sites. Real arm **99.97%**, `median_n_local = 3`; granule layer up ~13%; every A3c divergence statistic strengthened (AD markers above the line 16/20 → **18/20**) |
| A3b's in-soma funnel counted **empty** spheres as somatic | reported a fictitious "28% removed at the in-soma step"; the response asserted the inverse of the truth | funnel reports `n_empty` separately and computes in-soma over non-empty spheres only. Emptiness is now a result: **27% of 5 µm copies contain nothing at all** |
| the tissue-wide random floor rejected only out-of-tissue while the vicinity arms also rejected in-nucleus | the elevation ratio compared two differently-built populations | floor built under the identical rule; elevation 3.4×/4.3× → **3.2×/3.9×** |
| 25 Blank probes leaked into `partition_counts`, the BH family, Axis 2 and the figures | Axis 2 on n = 315 not 290; the response reported 295 "non-marker genes" one paragraph after saying 270 | filtered once into `parts_panel`; `parts` stays complete so the exhaustiveness gate remains meaningful |

**One reading had to change with the fix.** `rel:2.0` places the copy *externally tangent* to its
source, so with the containment buffer it admits a shell of the source granule's own boundary
transcripts. It is now the highest pseudo arm (WT 6.6%, AD 9.2%) for a geometric reason, in both
placement arms. `rel:3.0` clears its source and sits with the absolute offsets (WT 3.7%, AD 7.1%).
**Read the curve on the absolute offsets and on `rel:3.0`.** The earlier claim that `rel:2.0` sat
below `abs:5.0` "because a tangent copy captures none of the source's own points" was true of a
bare-radius query and is false with the buffer.

---

## A3b — `A3b_vicinity.ipynb`

| § | what |
|---|---|
| 1 | source granules, tissue mask, local-density strata, seed gene from the `sphere_dict` |
| 2 | placement — two arms × the offset sweep |
| 3 | profiling, real vs pseudo (**descriptive**) |
| 4 | **the detection predicate** — the load-bearing statistic |
| 5 | the distance curve, stratified by density and thinned for inference |
| 6 | the zero-placement-bias variant |
| 7 | correctness gates |

Three decisions that determine whether this survives scrutiny:

**Offsets are in-plane.** `layer_z` takes only **7 discrete values** (0, 1.5 … 9.0 µm) and both
`profile()` and `nc_filter()` query at `layer_z`, so a 3D direction would push the centre off the
grid.

**The count comparison is a tautology and does not carry the argument.** `sphere_r` is the *minimum
enclosing radius* of the exact DBSCAN core points (`miniball` on deduplicated cluster coords) — an
order statistic. The real sphere is maximally dense by construction, so any displaced copy at the
same radius must capture no more. Reporting "pseudo-granules capture fewer transcripts" and stopping
would be reporting an algebraic identity.

**The predicate is eps-connectivity, not a count.** Three transcripts scattered across a 4 µm sphere
are not `eps = 1.5`-connected, so "≥ 3 inside" badly overstates detectability. §4 runs the actual
`DBSCAN(eps, min_samples)` on the **same seed gene** — "any of 20 markers" is ~20× easier than
"3 Camk2a", and Camk2a alone is 47 % of the published granules.

**On rejecting offsets that land on a real granule** — measured before deciding: per-plane 2D granule
coverage is only **1.9 % (WT) / 1.5 % (AD)**, so the rejection discards few candidates and does not
meaningfully bias the sample toward granule-sparse space. (A 3D nearest-neighbour distance of 2.68 µm
looks alarming but counts neighbours on adjacent z-planes.) Both arms run anyway, and in the
unrejected arm the fraction landing on a real granule is itself a **result**.

**§6 needs no HPC output and has no placement rule at all.** `all_granules.parquet` is the rough
pass — every candidate the detector found and the filters then rejected, ambient-driven ones
included, at real positions. Comparing Set 2 against `all_granules \ Set 2` by distance asks the
same question with nothing synthesised. Run *in addition* to the literal control, since the reviewer
asked for that one by name.

---

## A3c — `A3c_de_baseline.ipynb`

| § | what |
|---|---|
| 1 | the transcript-level partition: intrasomatic / granule / residual-extrasomatic |
| 2 | how biased the published spot-matrix subtraction is |
| 3 | Axis 1 — compartment |
| 4 | Axis 1 significance, on a non-compositional primary |
| 5 | Axis 2 — conditions and regions |
| 6 | correctness gates |

**The partition is built at transcript level, and it supports both baselines §3 needs.**

| baseline | layers | role |
|---|---|---|
| **all non-somatic RNA** | `granule + residual_extrasomatic` vs `intrasomatic` | **primary** — the reviewer's literal wording, and the only form independent of granule detection: neither side needs a sphere to define |
| **granule-free** | `residual_extrasomatic` vs `intrasomatic` | **sensitivity** — excludes the in-granule transcripts so the baseline does not contain the signal it is a null for, but is defined as "extrasomatic *and not inside a called sphere*" and is therefore detection-**dependent**. Never label it otherwise |

Including the in-granule transcripts biases the contrast **toward zero**, so the primary is the
conservative choice as well as the literal one; the two arms agree (`axis1_divergence_test.csv`).
The §6 gate asserts the three layers sum to the transcript count exactly, per gene, per sample, and
that the somatic reference cancels out of `delta` to 1e-9.

**Do not use the published subtraction.** `np.maximum(extrasomatic − spot_granule_expression, 0)`
(in `7_neuropil_subdomains.ipynb` cell 9 and `benchmark_ambient.ipynb` cell 6) compounds three
errors: `spot_embedding` assigns each granule to the spot holding its **centre**
(`downstream.py:706-712`) while the sphere spans neighbours; `profile()` counts **all** transcripts
in the sphere including `overlaps_nucleus == 1`, so it over-subtracts from an extrasomatic-only
layer; and overlapping granules double-count shared transcripts. The clip at 0 then makes the bias
**one-sided and worst for the marker genes** — exactly where the result lives. §2 quantifies it.

**Two reporting rules, both load-bearing:**

1. **Significant-gene counts are not comparable across layers.** The layers differ in depth and
   sparsity and a rank test's power tracks that — which is why the published ambient and cell layers
   show 253 and 234 significant genes against the granule layer's 161. Compare **rankings and logFC
   correlations only**. Quoting the tallies invites the reading *"the authors' ambient layer yields
   more DE genes than their granule layer."*
2. **n = 1 vs 1.** One WT section, one AD section, so every spot-level WT/AD p-value is
   pseudo-replication — and this applies to the published result too. WT/AD is descriptive; the
   inferential weight sits on the within-sample granule-vs-residual-ambient **divergence**.

The Subdomain 1 vs 2 arm **already exists** in
`output/MERSCOPE_WT_AD_comparison/neuropil_subdomains_Isocortex_50/` and is not recomputed.

---

## A3d — `A3d_local_null.ipynb`

| § | what |
|---|---|
| 1 | the 10 µm grid, built from scratch, over the 252 neutral genes |
| 2 | which bins can carry a null, and how much of the granule layer they cover |
| 3 | the null: closed-form moments over bins |
| 4 | per-gene observed vs expected |
| 5 | the neuronal-minus-glial contrast against its null |
| 6 | correctness gates |

**Why this exists when A3c §5 already answers the same question.** A3c §5 reports that granules
*rank* neuronal genes above glial ones. It never states the reviewer's hypothesis as a model and
rejects it. A3d does: hold fixed the number of transcripts each granule contains, ask what it would
hold if they had been drawn from the RNA in its own 10 µm square, and measure how far the observed
separation lies from that. The neighbourhood is **25× smaller in area** than A3c's 50 µm squares, so
"granules sit in neuron-rich neuropil" has correspondingly less room to operate.

**One null: the permutation.** Pool a bin's granule and residual extrasomatic transcripts and
relabel which `N_b` of them are "granule" — exactly multivariate hypergeometric. It carries the
uncertainty in the local composition instead of assuming it away, and it is mildly **conservative**,
because the granule's own transcripts sit in the pool it is measured against and so dilute the
contrast being tested. Gate (d) confirms it is calibrated: split each bin's residual pool into two
random halves and test one against the other, and the z-scores come back centred on zero with unit
spread and no gene called.

**A literal multinomial variant was retired** (2026-08-31). It treated the residual composition
`p_b` as known and drew `Multinomial(N_b, p_b)`; measured on those same exchangeable halves its
z-scores were ~1.4× too wide. Nothing reported ever came from it, so no number changed when it was
dropped. Its other purpose — stating the hypothesis as something one could physically *generate* —
is now served far better by **A3e**, which builds the pseudo-granules and runs the detector over
them. The `a3_config.py` A3d block records this in full.

**Reuses A3c's partition, does not recompute it.** `transcript_layer_<sample>.parquet` is one int8
column in the transcript table's row order, so pairing it back is a positional concatenation — gate
(a) checks the totals against `partition_counts.csv` rather than trusting that. This is the only
thing A3d needs from another notebook: not the Set 0/1/3 detections, not the vicinity controls.

**Gate (c) is brute force, on purpose.** Everything reported comes out of the closed-form `E_g` and
`V_g`, and two implementations of one formula agreeing would say nothing about whether the formula
is right. So gate (c) does not check algebra against algebra: it takes a few thousand *real* bins,
physically pools each one's granule and residual transcripts, shuffles which of them are "granule",
counts genes, and repeats a thousand times. The simulated mean must land on `E_g` and the simulated
spread on `sqrt(V_g)` — and every shuffle must preserve the granule transcript total exactly, since
the null moves composition and never abundance.

**Report effect sizes beside significance.** The granule layer holds millions of transcripts, so a
compositional shift of a fraction of a percent clears any threshold and a bare count of significant
genes is close to uninformative. Every such count is reported next to the number of genes whose
observed/expected ratio exceeds `LOCAL_NULL_EFFECT_THR`.

---

## A3e — `A3e_pseudo_granules.ipynb`

| § | what |
|---|---|
| 1 | the published granules, split into three arms |
| 2 | which transcripts belong to each converted granule |
| 3 | the local ambient pool — a 5 µm disc centred on the granule — the radius ladder, and the draw |
| 4 | the patch it writes |
| 5 | a local pre-check, before spending four hours on a node |
| 6 | *(second pass)* what came back through the detector, and what "re-detected" means |
| 7 | correctness gates |

**The most direct answer in A3.** A3d states the reviewer's hypothesis as a model and rejects it in
z-scores. A3e states it as an *object* and hands it back to the detector: take real granules, keep
every transcript exactly where it is, replace only the gene identities with labels drawn from the
ambient RNA around that granule, and re-run mcDETECT over the whole section. If mcDETECT is calling
locally dense ambient patches, it calls these back.

**Three arms, one detection run per sample.**

| arm | share | what changes | what it tells us |
|---|---|---|---|
| `ambient` | 10% | labels redrawn from the residual extrasomatic RNA within 5 µm of the granule centre | the hypothesis |
| `scramble` | 10% | the granule's **own** labels permuted among its own points | the machinery control |
| `untouched` | 80% | nothing | the load-bearing control |

**Why the `scramble` arm is not optional.** Relabelling also scrambles *which* point carries which
gene, and that alone could break DBSCAN's ε = 1.5 connectivity. Without an arm that scrambles
identically while preserving the granule's own composition, a low `ambient` rate is ambiguous and
the objection *"you scrambled the transcripts, of course DBSCAN broke"* has no answer. `ambient`
read against `scramble` is the compositional effect on its own.

**Why the `untouched` arm is what makes one run enough.** It is simultaneously the proof that the
re-run reproduces the published pipeline and the proof that the perturbation stayed local.
`minspl = 3` is fixed in `DETECT_KWARGS_FINE`, so `poisson_select` never runs and DBSCAN is a purely
local, deterministic function of the point pattern — but gate (d) measures that rather than assuming
it, against `PSEUDO_CONTROL_MIN`. If it fails, no other number in A3e is readable.

**This is not a repeat of A3b.** A3b *displaces* a sphere to a nearby empty location and applies a
detectability predicate. A3e keeps the location, changes the contents, and runs the real detector
end to end.

**It is also not circular, and the reason matters.** mcDETECT seeds on the 20 markers, so "strip the
markers and it stops detecting" would prove nothing. Three things make the number informative: local
density is held *exactly* fixed (only `target` changes); the ambient pool is over **all targets**, so
it contains markers at their real ambient frequency and the re-detection rate could have come out
high; and the `scramble` arm separates composition from geometry.

### The neighbourhood is a disc, not a square of A3d's grid

The first version reused A3d's 10 µm lattice and gave each granule the residual RNA of whichever
square its centre fell in. That is the wrong instrument here: the granule is **not centred** in that
square — its centre sits a median ~2.5 µm, and up to 7 µm at a corner, from the middle of the
neighbourhood it is being compared against. A3d can live with that because it needs a *partition* of
the section to sum closed-form moments over; A3e needs no partition, only one neighbourhood per
granule, and a disc says that literally. It also retires the 3 × 3 fallback: a disc sized in µm
cannot be "too thin" the way an arbitrary lattice square can.

`PSEUDO_POOL_RADIUS = 5.0` µm, 2-D with all seven z-planes pooled as A3d's squares are. Area
78.5 µm² is **21% tighter** than A3d's 100 µm² square, so the locality claim is at least as strong.

### Retention: the pool must be able to supply the draw

    pool_size  >=  max(PSEUDO_MIN_POOL, k)        k = the granule's own transcript count

The `k` half is what makes the without-replacement draw well defined. Without it a granule larger
than its own surroundings would have to be drawn *with* replacement — a different sampling scheme
applied to exactly the largest granules, which are also the easiest to re-detect.
`a3_common.draw_local_ambient` **asserts** the rule rather than falling back, so a violation stops
the run instead of quietly changing the model.

Applied to **every arm**, not just `ambient`: neighbourhood density plausibly predicts how
re-detectable a granule is, so restricting only the arm that draws from the pool would confound
`ambient` against `scramble` with local density. Failures are labelled `excluded_thin_pool` and
reported beside `excluded_contaminated`, never folded into an arm. Gate (f) checks the two converted
arms came out matched in size, granule transcript count and local pool size.

`a3e_pool_ladder.csv` scores the rule at r ∈ {4, 5, 6, 7} µm before anything is drawn — counts only,
so a whole ladder is cheap. **Measured on the real sections:**

| r (µm) | WT pool median | WT retention | AD pool median | AD retention |
|---|---|---|---|---|
| 4 | 215 | 99.69% | 126 | 96.96% |
| **5** | **329** | **99.92%** | **193** | **99.37%** |
| 6 | 464 | 99.96% | 274 | 99.74% |
| 7 | 619 | 99.97% | 367 | 99.83% |

against a median granule of 6 transcripts. 5 µm is the tightest radius that keeps both sections
above 99%; 4 µm would lose 3% of AD.

### What counts as "re-detected"

A granule counts as re-detected when the re-run produced a sphere that **(1)** contains at least
`PSEUDO_PROVENANCE_FRAC` = half of that granule's own transcripts, and **(2)** contains more of that
granule's transcripts than of any other granule's. Plainly: *the detector rebuilt a sphere on the
same transcripts, and that sphere is more that granule than anything else.*

Geometry cannot say this. It cannot separate "the detector called this object again" from "the
detector called something else nearby", and 80% of granules are untouched and will certainly be
called. The loophole is **measured** — matching the published granules against themselves, where a
credit from any sphere but the granule's own is one it collects for free:

| criterion | WT | AD |
|---|---|---|
| `center_in` | 6.93% | 8.20% |
| `intersect` | 33.91% | 38.98% |
| `merge` | 0.34% | 0.65% |
| condition (1) alone | 5.98% | 7.82% |
| **(1) and (2) — the primary** | **0.25%** | **0.28%** |

**Condition (2) is the one that matters, and that was not obvious.** (1) alone was expected to solve
it and does not: published granules overlap so heavily that a neighbour's sphere already holds half
of a given granule's transcripts 6–8% of the time — no better than `center_in`. Requiring the sphere
to be *mostly* that granule takes it to a quarter of a percent, below even `merge`, and costs
nothing: a granule is credited by its own sphere 100.000% of the time at every threshold from 0.5
to 1.0. `merge` reaches a low floor the other way, by being strict enough to miss genuine
re-detections, which is why it is reported but never primary.

All four criteria are scored into `a3e_redetection_rate.csv`, and the floor table is regenerated
from data into `a3e_match_floor.csv` rather than trusted from this README. Gate (g) asserts the
primary beats `center_in`.

`a3_common.provenance_match` has to know whose transcripts each point is, so section 2 runs a
**second** ownership pass over all granules. It must not replace the converted-only one:
`granule_members` assigns to the nearest granule *among those passed in*, so recomputing it over all
681 K would move transcripts to untouched neighbours and desynchronise the gates from the patch
already written.

### The membership trap

**`KG_BUFFER`.** A granule's members are collected at `sphere_r + 0.01` centred on
`(sphere_x, sphere_y, layer_z)` — the `partition_transcripts` convention. `sphere_r` is the
*minimum-enclosing* radius, so support points sit exactly on the surface and a bare-radius query
loses 11.6% of the granule layer, 93.6% of it markers. Elsewhere in A3 that biases a comparison;
here it would be fatal, because the unrelabelled seed transcripts would stay in place and the
pseudo-granule would be re-detected for free.

Spheres overlap, so a transcript can fall inside several. Each is assigned to the nearest
centre (ties by the lower granule row), so every transcript is rewritten exactly once. Untouched
granules that share a rewritten transcript are **excluded from the control and counted**.

### A patch, not a copy

The notebook writes `a3e_relabel_<sample>.parquet` — positional row index plus new target, for the
rewritten rows only — rather than a second 1.6 GB transcript table. It is what travels to HGCC, it
is small enough to read beside the diff, and it is the auditable record of exactly what changed.
`a3e_relabel_scope.csv` carries the table length it was built against; `run_pseudo_detection.py`
refuses to apply it to a table of any other length, because a positional patch applied to the wrong
table would silently rewrite the wrong transcripts.

The node builds nothing. It applies a patch that was already decided locally, then runs the
published pipeline: `dbscan → size/in-soma filters → merge_sphere → nc_filter`, with the **19-gene**
NC list Set 2 was built with. A3a already validated that reproduction on the unmodified table
(681,346 rebuilt against 681,337 published in WT), which is where `PSEUDO_CONTROL_MIN` comes from.

---

## Runbook

**Ten steps, each run exactly once. No parameter is adjusted at any point** — every notebook's
defaults produce the final tables. Run all notebooks from `R2_revision/ambient_controls/` on the
`mcDETECT-env` kernel (A3a §2 imports `mcDETECT` to reproduce Set 2 from Set 1).

| # | step | where | needs | produces |
|---|---|---|---|---|
| 1 | `A3_preflight.ipynb`, top to bottom | local | nothing | `output/preflight/` |
| 2 | upload `set0_genes.csv` | local → HGCC | 1 | the array's only non-tracked input |
| 3 | the detection array | HGCC | 2 | `output/detect/` |
| 4 | download `output/detect/` | HGCC → local | 3 | the four sets on disk locally |
| 5 | `A3c` → `A3b` → `A3a`, each top to bottom | local | 4 | `output/a3{a,b,c}/` |
| 6 | `A3d_local_null.ipynb`, top to bottom | local | 5 (`A3c` §1 only) | `output/a3d/` |
| 7 | `A3e_pseudo_granules.ipynb` §0–5 | local | 5 (`A3c` §1 only) | `output/a3e/a3e_relabel_*` |
| 8 | upload `output/a3e/`, run the pseudo array | local → HGCC | 7 | `output/a3e/detect_<sample>/` |
| 9 | download it, re-run `A3e` top to bottom | HGCC → local | 8 | `a3e_redetection_rate.csv` |
| 10 | `Rscript A3_figures.R` | local | 6, 9 | `output/figures/` |

```bash
cd R2_revision/ambient_controls
HGCC=hgcc:~/hulab/projects/mcDETECT/R2_revision/ambient_controls    # as in slurm/run_detection.sh

# ---- 1. pre-flight, LOCAL. Writes the Set-0 gene list the detection script reads. ----
jupyter lab A3_preflight.ipynb

# ---- 2. upload it. This is the ONLY file the array needs that git does not carry. ----
ssh hgcc "mkdir -p ~/hulab/projects/mcDETECT/R2_revision/ambient_controls/output/preflight"
scp output/preflight/set0_genes.csv "$HGCC/output/preflight/"

# ---- 3. detection, HGCC. 3 sets x 2 samples = 6 tasks. ----
#    0-1 set0 WT/AD  |  2-3 set3 WT/AD  |  4-5 set1 WT/AD   (python3 run_detection_sets.py --list)
#    Cheap sets first, so the whole path is validated before set1's two 200G jobs.
sbatch --array=0-3 slurm/run_detection.sh
sbatch --array=4-5 slurm/run_detection.sh
#    or all at once, 4 concurrent:   bash slurm/submit.sh 4
#    Finished tasks are skipped on resubmit, so failures can be rerun by id: --array=2,5

# ---- 4. download the results. ----
rsync -av "$HGCC/output/detect/" output/detect/
#    Expect 6 directories, each with spheres.parquet, sphere_dict.parquet, funnel_by_gene.csv,
#    run_info.csv (+ status.csv if a zero-count gene was dropped). A few hundred MB, dominated by
#    set1_*/sphere_dict.parquet (the unfiltered pre-merge pass).

# ---- 5. local analysis, in this order. ----
jupyter lab A3c_de_baseline.ipynb    # needs nothing from step 4
jupyter lab A3b_vicinity.ipynb       # sections 1-5 need set1_*/sphere_dict.parquet
jupyter lab A3a_three_sets.ipynb     # needs all four sets; A3a section 2 is the slow one

# ---- 6. the local-neighbourhood null. Needs only A3c section 1. ----
jupyter lab A3d_local_null.ipynb

# ---- 7. A3e pass 1, LOCAL: build the relabelling. Sections 6-7 report "waiting" and the
#         notebook still completes. ----
jupyter lab A3e_pseudo_granules.ipynb

# ---- 8. A3e detection, HGCC. 2 tasks (WT, AD); expect ~4 h and ~2 h at 200G. ----
ssh hgcc "mkdir -p ~/hulab/projects/mcDETECT/R2_revision/ambient_controls/output/a3e"
scp output/a3e/a3e_relabel_*.parquet output/a3e/a3e_relabel_scope.csv "$HGCC/output/a3e/"
sbatch slurm/run_pseudo_detection.sh          # python3 run_pseudo_detection.py --list
#    Finished tasks are skipped on resubmit: --array=1 reruns AD alone.

# ---- 9. download, then re-run A3e top to bottom. Sections 6-7 now light up. ----
rsync -av "$HGCC/output/a3e/detect_WT/" output/a3e/detect_WT/
rsync -av "$HGCC/output/a3e/detect_AD/" output/a3e/detect_AD/
jupyter lab A3e_pseudo_granules.ipynb

# ---- 10. figures. Every section degrades to "[skip]" on a missing input, so this is safe
#          to run at any point. ----
Rscript A3_figures.R
```

**A partial download fails quietly.** Every set-dependent cell prints `[skip] ... missing` rather
than raising, so check `output/a3a/set_inventory.csv` has **4 sets x 2 samples = 8 rows** and
`output/a3a/overlap_ladder.csv` has **24 rows** (2 controls x 2 granule sets x 3 criteria x 2
samples) before trusting A3a. `A3c` and `A3b` §6 need nothing from HGCC and can run while the
array is still queued.

### Toggles

Defaults are the final configuration; these exist for debugging, not for the normal run.

| notebook | toggle | default | effect |
|---|---|---|---|
| A3b, A3c, A3d, A3e | `VALIDATE` | **`True`** | correctness gates. On by default here, unlike A1/A2: the gates are the last section, so every table is already written when they run, and they are what catches a bad run — A3c's gate (a) caught a bad edit during the rebuild. **A3a has none** — it computes only what the response quotes |
| A3a, A3b | `DRY_RUN` | `False` | `True` subsamples (`MAX_SPHERES` / `MAX_GRANULES` = 200K) for a cheap smoke pass over every cell. **The tables a dry run writes are not final** |
| A3b | `RUN_PREDICATE` | `True` | §4-5, the slow and load-bearing step |
| A3b | `RUN_ROUGH_VARIANT` | `True` | §6 — needs no HPC output |
| A3c | `OVERWRITE` | `False` | `True` recomputes the cached transcript partition |
| A3c | `RUN_CLIP_BIAS` | `True` | §2 — needs `spot_embedding` |
| A3c | `RUN_COUNT_MODEL` | `True` | §4 — the non-compositional primary |
| A3c | `RUN_AXIS2` | `True` | §5 — WT/AD on the three layers |
| A3d | `OVERWRITE` | `False` | `True` rebuilds the 10 µm bin counts from the transcript tables |
| A3e | `OVERWRITE` | `False` | `True` redraws the relabelling even if a patch is already on disk. **Changes the patch, so the detection run must be redone** |
| A3e | `RUN_PRECHECK` | `True` | §5 — the expensive local step (one KD-tree per marker over ~30 M transcripts). Indicative only; §6 is authoritative |
| `A3_figures.R` | `RUN_*` | `TRUE` | one per section, numbered to the notebooks |

`A3_preflight.ipynb` has no toggles at all.

### Tables that are deliberately not plotted

Several outputs are quotable tables for the response rather than figures, and `A3_figures.R` does
not read them. The load-bearing ones are `a3a/set2_reproduction.csv`,
`a3a/overlap_transcript_level.csv`, `a3c/axis1_divergence_test.csv` (the divergence claim stated
as a test), `a3c/axis1_count_model.csv`, `a3b/profile_funnel.csv`,
`a3b/detection_predicate_thinned.csv` and `a3b/rough_variant_by_distance.csv`. This is a choice,
not an oversight. What is *not* here any more is anything unplotted **and** unquoted — that is the
rule the A3a rewrite applied.

### The one expensive local step

Assigning ~10⁸ transcripts to ~10⁶ spheres is the single heaviest operation in A3. It is done
**once**, by `A3c` §1, and cached to `output/a3c/transcript_layer_<sample>.parquet` (one int8
column, ~100 MB). Nothing in A3a reads that cache any more — the analysis that did (the adaptive
threshold re-test) was removed — but A3c still needs it. Expect several GB of RAM for the KD-tree
during that first pass. Every KD-tree call in `a3_common` is batched (`query_ball_point` over arrays with
`workers=-1`) — the per-sphere Python loops these replaced would not have finished.

### Where the time goes

`set1` detection dominates the A3 HPC stage — it is 2 of the 6 tasks and the only pair needing
200 G — and **A3e's re-detection costs the same again**, because it is the same 20-marker pass over
the same section plus an `nc_filter`. Budget roughly 4 h (WT) and 2 h (AD) for it, from Set 1's
recorded 225.9 and 95.8 minutes.
Note detection now runs DBSCAN with the size/in-soma filters **off** and applies them per gene
afterwards: identical merged output (verified to 1e-12; `miniball` is randomised, so mcDETECT's own
fine pass is not bitwise reproducible either), but it also yields the real `raw → size → in-soma`
funnel, which a filtered `sphere_dict` cannot. `set0`/`set3` are cheap by comparison (Set 3 seeds
on ~2.0 M NC transcripts against Set 1's 33.3 M).

Locally, **A3b §4** is the slow step — one core-point test per pseudo-granule — which is what
`DRY_RUN` exists to shorten while debugging; the final run must have it `False`. **A3a §2** is the
next heaviest: 2 controls × 2 granule sets × 21 sphere-set pairings (observed + 20 nulls) × 2
directions, over up to 741 K spheres. It runs in a few minutes because `overlap_pairs` scores the
whole three-rung ladder from **one** candidate query — passing the criteria as a list is what keeps
this affordable, and reverting it triples the cost. **A3c §1**'s partition is one batched ball
query over ~10⁸ transcripts per sample, cached to `partition_counts.csv`. **A3e §2** reuses that
cache and builds its KD-tree over the ~6 M granule-layer transcripts only, so membership costs
seconds rather than repeating the 10⁸-point pass; **A3e §5** is the expensive local step and is the
one thing in that notebook worth switching off while debugging.

---

## Which output backs which claim

| response element | source file |
|---|---|
| "structured ambient RNA is present and sparse" | `a3a/funnel_by_gene.csv` (rate columns), `a3a/set_inventory.csv` |
| "ambient aggregates rarely coincide with granules, and the enrichment over chance falls once the NC filter is applied" | `a3a/overlap_ladder.csv`, `a3a/overlap_transcript_level.csv` |
| "Set 1 and Set 2 differ only by the NC filter" | `a3a/set2_reproduction.csv` |
| "the ambient population carries none of the WT/AD condition contrast" | `a3a/set_density_per_region.csv` (Set 1 is the built-in positive control) |
| **"the detector would not have fired a few µm away"** | `a3b/detection_predicate.csv` (+ `_stratified`, `_thinned`) |
| "…and that is not a density effect" | `a3b/detection_predicate_stratified.csv` |
| "granule enrichment diverges from the non-somatic baseline (granule + residual vs intrasomatic, the reviewer's literal wording)" | `a3c/axis1_gene_table.csv`, `a3c/axis1_summary.csv`, `a3c/axis1_divergence_test.csv` |
| "…and is not an artefact of compositional normalisation" | `a3c/axis1_count_model.csv` |
| "the AD granule signal is not reproduced by the raw non-somatic layer" | `a3c/axis2_wt_ad_by_layer.csv`, `a3c/axis2_layer_correlation.csv` |
| **"granules are not random samples of the RNA around them"** — gene by gene, and as one neuronal-minus-glial contrast | `a3d/a3d_local_null_genes.csv`, `a3d/a3d_local_null_group.csv` |
| "…and the test is not simply over-powered" | `a3d/a3d_local_null_negative_control.csv` (gate (d): exchangeable halves, nothing called) |
| **"pseudo-granules built from local ambient RNA are not re-detected"** | `a3e/a3e_redetection_rate.csv` (`untouched` calibrates it, `scramble` isolates composition) |
| "…and 're-detected' means the same transcripts, not merely the same place" | `a3e/a3e_match_floor.csv` (0.25% / 0.28% false-credit floor, against `center_in`'s 6.93% / 8.20%) |
| "…and the reason is that local ambient supplies too few marker transcripts" | `a3e/a3e_marker_shift.csv`, `a3e/a3e_relabel_audit.csv` |

---

## What this does and does not settle

**Does.**

- Runs, at last, both controls the reviewer offered in round 1 — the vicinity pseudo-granules and
  the somatic-vs-non-somatic DE baseline — the latter against all non-somatic RNA, exactly as
  worded, with a granule-free variant reported alongside as a sensitivity arm.
- Tests at the **detection step**, not on already-called granules: A3b asks whether the detector
  would have fired at a matched sphere a few µm away, using eps-connectivity on the same seed gene.
- Answers the *substantive* version of the CSR worry — that a **global** threshold under-corrects
  where background is locally denser — with A3b's stratified detection predicate: the pseudo-granule
  rate stays an order of magnitude below the real-granule rate **within every local-density
  quintile**, so the effect is not a density effect.
- Makes the ambient population itself **visible and measurable** (A3a Set 3), instead of arguing
  from its absence, and neutralises the abundance objection with per-million-transcript rates plus
  an abundance-matched second control (Set 0).

**Does not.**

- **Ambient aggregates and granule candidates are not independently placed.** Obs/exp against
  random re-placement is 6.9× (granule side) and 4.6× (control side) for Set 3 against Set 1 in WT.
  The reviewer's mechanism is **real**; the case rests on magnitude and on the fall after NC
  filtering (to 3.0× and 2.5× in WT, 1.2× in AD). Say so — a claim that co-location is at chance
  would be false and is checkable in one table.
- **No adaptive re-detection was run.** The locally-adaptive `min_samples` re-test was removed: it
  could only ever remove granules, never add the ones an adaptive rule would call in sparse regions,
  and AD is the lower-density arm, so the untested direction was the one that worked against us.
  A3b's vicinity control answers the same reviewer sentence with a design that is not one-sided.
- **n = 1 vs 1.** Nothing here fixes the single-section design; every WT/AD p-value remains
  pseudo-replication, and the inferential weight is deliberately on within-sample comparisons.
- **A3b's literal control still requires a placement rule.** The rough-pass variant is the
  no-placement-rule cross-check, but the reviewer asked for the literal control by name and it
  carries its own assumptions.
- **Per-region ambient counts are small** — as few as 2 aggregates in the sparsest AD region — so
  A3a §3 rests on the rank correlation of the AD/WT ratio profile (Set 1 ρ = 0.98 vs Set 3
  ρ = −0.13) and on magnitude, never on an individual region.
- **The three-set design cannot prove ambient is unstructured** — only that structured ambient does
  not produce granule-shaped, marker-enriched, non-overlapping aggregates at these thresholds.
- **Set 0 is neither abundance-matched at the top nor a clean negative.** The panel runs out of
  unannotated genes above ~300 K transcripts, so the most abundant markers are matched up to 23×
  low (see *The sets*) — the per-million-transcript rate is what makes the comparison valid there.
  And four of its twenty genes (`Grin2b`, `Dner`, `Epha4`, `Ncam1`) are documented dendritic
  transcripts, so it bounds what arbitrary genes produce, not what ambient produces.

### Known issue, flagged and NOT investigated here

The AD section thins with depth while WT is flat, so granule counts follow:

| z (µm) | 0 | 1.5 | 3.0 | 4.5 | 6.0 | 7.5 | 9.0 |
|---|---|---|---|---|---|---|---|
| WT tx (M) | 13.6 | 15.0 | 15.4 | 15.4 | 15.2 | 14.7 | 14.0 |
| AD tx (M) | 16.0 | 15.5 | 14.5 | 11.6 | 6.9 | 3.1 | 1.4 |
| WT granules | 95,352 | 102,896 | 102,409 | 100,692 | 97,970 | 93,667 | 88,351 |
| AD granules | 122,594 | 114,874 | 98,265 | 48,256 | 12,128 | 2,068 | 624 |

Restricted to the three fully covered planes (z ≤ 3), WT = 300,657 and AD = 335,733 — **AD is
higher**, reversing the direction of the total-granule difference. `CAPTURE_EFFICIENCY_COEF = 0.818691`
does not correct for this: the raw transcript ratio is 0.666 and the deficit is z-structured, not
uniform. **Whether the published per-subtype, per-region claim reverses is unchecked.** Recorded here
because A3 is defending that comparison; the decision on what to do about it sits outside this
analysis. Source: `output/preflight/z_profile_<sample>.csv`.
