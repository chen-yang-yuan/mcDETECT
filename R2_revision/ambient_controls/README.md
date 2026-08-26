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
`baseline_logFC` (extrasomatic vs somatic, detection-independent), `granule_enrichment`, their
difference `delta`, and a regression fitted on non-markers — and `code/figures_response.Rmd:1421-1452`
already carries its scatter under a heading literally titled *"Reviewer 2, Major Comment 9"*. It was
built for round 1, then **not used** in the response; its output CSV is no longer on disk and the
notebook sits in gitignored `code/old/`. A3c revives it and closes six gaps:

| # | gap | fixed in |
|---|---|---|
| 1 | WT only — the reviewer names *conditions* | A3c §5 |
| 2 | `delta` subtracts two logFCs with different references; the correct branch (`USE_ALT_GRANULE_VS_SOMA`) ships **off** | A3c §3 |
| 3 | the baseline includes in-granule transcripts, so it contains the signal it is a null for | A3c §1 |
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
granules self-filter:

| Gria2 spheres | rough | + size / in-soma | published (Set 2) |
|---|---|---|---|
| WT | 41,544 | 2,737 | **4** |
| AD | 20,525 | 1,139 | **0** |

**Quote the middle column.** Most Gria2 aggregates are intranuclear — consistent with Gria2 sitting
on a nuclear-enrichment list — so the loss attributable to the **NC filter** is ~2.7 K / ~1.1 K, not
41 K / 20 K. And 97.1 % (WT) / 97.6 % (AD) of published granules have `nc_ratio` **exactly 0**, so
this single collision dominates the Set1-vs-Set2 difference and must be partitioned rather than
reported as one number.

Frame it as a **curation error that makes Set 2 conservative**: Gria2 is a canonical dendritically
transported transcript, so its presence on a nuclear-enrichment list is a list inconsistency, not
evidence against granules.

**Policy — two separate lists, and only one of them changes.**

| list | role | in A3 |
|---|---|---|
| `syn_genes` — **20 granule markers** | the **detection seeds** (`3_detection.py:55`) | **unchanged everywhere.** Sets 1 and 2 both seed on all 20 |
| `nc_genes` — **19 negative controls** | the **NC filter**, and Set 3's seed list | two versions, chosen by provenance |

Only the **NC list** is adjusted:

| NC-list use | version | why |
|---|---|---|
| Set 3's seed list | **18**, Gria2 dropped | a correction, not an approximation — it makes Set 3 more faithful to "nuclear-enriched negative controls" |
| the leave-one-out enumerated over Set 1 | **18**, Gria2 dropped | same reason — it asks what the *other* controls do |
| `nc_ratio` recomputed on Set 2 | **19**, as published | must reproduce the filter the data on disk was built with |

**Why Set 1 still seeds on all 20 markers.** Its whole job is to be "Set 2 minus the NC filter";
seeding it on 19 would make it differ from Set 2 in *two* ways at once and the difference would no
longer isolate the filter. The confound is real, not hypothetical — `_remove_overlaps`
(`model.py:323-377`) is order-dependent and propagates whole rows: containment with B larger does
`set_a.loc[i] = set_b.loc[j]`, replacing A's row **wholesale, gene label included**, while a deep
intersection refits A's geometry and keeps A's label. So a Gria2 sphere can absorb, or be absorbed
by, another marker's sphere, and dropping it from the seeds would change the geometry *and the
labels* of **non-Gria2** granules too.

A consequence noted rather than fixed: some *surviving* Set-2 granules have geometry enlarged by a
merge with a Gria2 sphere that was only NC-filtered away afterwards. That is baked into the
published result.

**Dropping Gria2 from the marker list is unnecessary and not free.** It is a genuine post-synaptic
marker (in `REF_GENES` and `MARKER_GENES["post-syn"]`), and the published population already holds
effectively none of it (4 WT / 0 AD) — so the effective marker set *is already 19*, and saying so
costs nothing. Re-running the published detection on 19 markers would change Fig. 3–5 for a gene
contributing 4 granules: out of scope.

**There is no all-19 (Set 3′) detection arm.** It existed only to show the choice does not matter,
and `Set 3′ − Set 3` is exactly the Gria2-seeded spheres, which Set 1 already contains — so the
figure comes free and 2 of the original 8 array tasks disappear.

**The gap this leaves is negligible, and quantified once.** Keeping Set 2 on 19 genes means the
published population is short of the Gria2-seeded granules an 18-gene filter would have kept: at
most **≈2,737 of ~737,000 in WT (0.37 %)** and **≈1,139 of ~427,000 in AD (0.27 %)**. Those are
**upper bounds** — Set 1 applies no NC filter at all, so some of those granules would have been
dropped by an 18-gene filter anyway. Under half a percent, same direction in both samples, so it
cannot move the WT/AD contrast.

**And a free sensitivity closes it.** Strip Gria2-*seeded* rows from **both** Set 1 and Set 2 and
recompute: both then rest on the same effective 19-marker population, isolating the NC filter's
genuine effect from the list collision. Pure table filtering on outputs that already exist —
reported as the `*_ex_gria2` columns of `a3a/gria2_partition.csv`, with a gate asserting
`n_removed_ex_gria2 == n_removed_other`.

One thing worth stating plainly, because it runs the other way: the gap is small relative to
Set 1 / Set 2, but Gria2 is ~13× more abundant than the median NC gene, so *including* it would
have been large relative to **Set 3** — potentially doubling it and manufacturing a spurious
Set1∩Set3 overlap. That, not convenience, is why it is excluded there.

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
* `size` is **not** `k_g`, the own-gene count A3a §6 must subtract from the local background.

### 4. The NC list is nuclear-enriched but not gene-neutral

The manuscript describes it as "enriched in neuronal nuclei compared to the cytoplasm"
(Supp. Table 8), which is right. But the 19-row file is an edgeR table whose `Cluster` column spans
`C4a` (complement), `Cyfip1`, `Abca7` (AD risk), `Opalin` (oligodendrocyte),
`Prox1`/`Npnt`/`Zfpm2` (dentate gyrus) — so the NC background is itself **spatially structured and
plausibly condition-dependent**, which is the reviewer's own complaint applied to our filter. Hence
the leave-one-out (over the **18**-gene NC list, per the provenance policy above) and the per-sample
AD/WT NC-density ratio in A3a §2. Note also that `nc_top = 20`
against a 19-gene list makes the "top 20 by expression" step a no-op.

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
├── A3_preflight.ipynb         [local, BEFORE the array] CSR table, Set-0 gene list, Set-2
│                              diagnostics, z profile
├── A3a_three_sets.ipynb       [local] NC forensics, funnels, overlap, density, stage D
├── A3b_vicinity.ipynb         [local] the vicinity pseudo-granule control
├── A3c_de_baseline.ipynb      [local] the somatic-vs-non-somatic DE baseline
├── A3_figures.R               all figures, sections numbered 1:1 to the notebooks
└── slurm/
    ├── run_detection.sh       one array task = one (set x sample)
    └── submit.sh              derives the array size from a3_config
```

### Outputs (git-ignored)

```
output/
├── preflight/    csr_min_samples.csv, set0_genes.csv, set2_diagnostics.csv, z_profile_<sample>.csv
├── detect/       <set>_<sample>/{spheres,sphere_dict}.parquet, funnel_by_gene.csv, run_info.csv
├── a3a/          nc_ratio_corrected_summary.csv, nc_leave_one_out.csv, gria2_partition.csv,
│                 capture_ratio_per_region.csv,
│                 set_inventory.csv, funnel_by_gene.csv, overlap_ladder.csv,
│                 overlap_transcript_level.csv, set_density_per_region.csv,
│                 capture_ratio_per_region.csv, adaptive_survival.csv, adaptive_caveats.csv
├── a3b/          source_summary.csv, placement_status.csv, vicinity_overlap_with_real.csv,
│                 profile_summary.csv + profile_histogram.parquet, profile_funnel.csv,
│                 detection_predicate.csv (+ _stratified, _thinned),
│                 rough_variant_by_distance.csv
├── a3c/          partition_counts.csv, transcript_layer_<sample>.parquet (shared cache),
│                 spot_layer_counts.parquet, clip_bias_by_gene.csv, axis1_gene_table.csv,
│                 axis1_summary.csv, axis1_count_model.csv, axis1_divergence_test.csv,
│                 axis2_wt_ad_by_layer.csv, axis2_layer_correlation.csv
└── figures/      everything A3_figures.R draws
```

---

## The sets

| set | seeds on | filters | source |
|---|---|---|---|
| **Set 0** | ~20 neutral panel genes, **abundance-matched** to the markers | size + in-soma | new |
| **Set 1** | the 20 granule markers | size + in-soma, **no NC** | new |
| **Set 2** | the 20 granule markers | size + in-soma + NC | **published** `granules.parquet` |
| **Set 3** | the NC genes **minus Gria2** (18 genes) | size + in-soma | new |

**Set 1 is a true re-detection, not a post-hoc filter of `all_granules.parquet`.** mcDETECT applies
the size and in-soma filters at the end of `dbscan()`, i.e. *before* `merge_sphere()`, so filtering
the rough pass afterwards does not reproduce Set 2's construction (737,063 vs 681,337 spheres in WT)
and the comparison would measure filter *order* rather than the NC filter.

**Set 0 is the highest value-per-CPU-hour item here.** Set 3 alone only shows "nuclear genes stay in
nuclei" — circular, since the NC genes are *defined* as nuclear-enriched and are then filtered on
nuclear overlap. Set 0 shows that arbitrary genes *at marker abundance* do not form granule-like
aggregates. That is why every set is reported as a **funnel** (raw → size → in-soma) with the
marker funnel printed beside it: if Set 3 is already near-empty *before* the in-soma filter, that is
a result; if it only empties at that step, it is circular.

**The abundance match holds only below ~700K transcripts, so Set 0's primary statistic is the
per-million-transcript rate, not the sphere count.** The panel has no unannotated gene more
abundant than that, so the eight rarest markers match within ~2 % while the seven most abundant
are matched several-fold low — worst `Camk2a` 6,237,713 vs `Zbtb20` 266,796 (23×), and in aggregate
33.3 M marker transcripts against 9.7 M for Set 0 (3.4×). Comparing raw counts would hand back
precisely the abundance objection Set 0 exists to remove; the rate (`rate_<stage>_per_Mtx` in
`a3a/funnel_by_gene.csv`) does not. `A3_preflight.ipynb` prints the match quality and
`preflight/set0_genes.csv` records it per marker.

---

## A3a — `A3a_three_sets.ipynb`

Sections are numbered from **2**, because §1 is the pre-flight and lives in
`A3_preflight.ipynb`. The numbering is global across the A3 notebooks and maps 1:1 onto
`A3_figures.R`'s sections.

| § | what |
|---|---|
| 2 | NC-filter forensics: one-geometry `nc_ratio`, per-NC-gene leave-one-out, the Gria2 partition, and (§2d) the gap + the 19-marker sensitivity |
| 3 | set inventory and funnels, as counts and as per-million-transcript rates |
| 4 | the overlap ladder — Set1∩Set3 and Set2∩Set3, plus merge-invariant transcript-level overlap |
| 5 | per-region density per set, WT vs AD, with the per-region capture-ratio spread |
| 6 | the locally-adaptive threshold re-test |
| 7 | correctness gates (`VALIDATE`, **on** by default) |

Runs **once, top to bottom, with nothing to adjust** — everything it needs is on disk by the time
it starts. Requires the `mcDETECT-env` kernel: §7's gate (b) imports `mcDETECT`.

### The overlap criterion

mcDETECT's own merge predicate (`model.py:349-353`, with `l=1`, `rho=0.2`) is

```
merge(A,B)  ⟺  d ≤ |r_A − r_B|   (containment)   OR   d < 0.2·(r_A + r_B)
```

so two equal-radius spheres merge only when their centres are within `0.4·r`. That is very strict —
real granules routinely overlap without merging — so quoting only that predicate would understate
co-location and read as rigged. The ladder therefore **leads with `intersect`** (`d < r_A + r_B`),
the loosest criterion: a small overlap under it cannot be argued with. Overlap is also reported at
**transcript level**, which is merge-invariant (granule-level cardinality is partly an artefact of
`merge_sphere`'s gene order, whose base is `sphere_dict[0]`), and as **observed/expected** against
Set-3 spheres re-placed uniformly in the tissue mask.

### §6 — the locally-adaptive threshold

The local rule matches the published functional form exactly, or the comparison means nothing.
`poisson_select` is a **2D areal** intensity (`tissue_area()` is 2D grid occupancy × `grid_len²`)
against a **2D disc** `π·eps²`, even though DBSCAN runs in 3D:

```
λ_local(g,i) = [ N_g(disc R at (x,y)) − k_g(i) ] / [ π R² · occ(x,y,R) ]
m_local(g,i) = max( poisson.ppf(0.95, α · λ_local · π · eps²), 3 )
survives     ⟺ k_g(i) ≥ m_local(g,i)
```

Subtracting `k_g(i)` is essential — otherwise the granule inflates its own background and the test
is self-defeating. `occ` comes from the same 1 µm occupancy grid `tissue_area()` counts; without it
a granule beside a ventricle or at a section edge gets a spuriously low λ and survives everything.

**Caveats that ship with the table (`adaptive_caveats.csv`) and belong in the legend:**

1. This is a **post-hoc re-test, not a re-detection**. A truly adaptive `min_samples` changes which
   points are core, hence cluster membership, the enclosing sphere, and `k_g` itself. Fixed-cluster
   re-testing can only *remove* granules.
2. So it **bounds false-positive inflation but is silent on false negatives** in low-density
   regions, where an adaptive rule would be *more* permissive. **AD is the lower-density arm, so the
   silent direction works against our own effect.**
3. λ_local is contaminated by neighbouring granules; excluding the granule's own transcripts fixes
   self-contamination only. Reported with and without all Set-2 granule transcripts removed, so the
   truth is bracketed.
4. This tests spatial **homogeneity**, not Poisson-ness.

The deliverable is **not** "X % survive" but whether the WT/AD per-region result holds on survivors —
plus the survival rate *per sample*, because a differential survival rate between WT and AD would be
the reviewer's hypothesis confirmed, and it is better reported by us than found by them.

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

**The partition is built at transcript level, and that is what makes the baseline honest.**
`benchmark_diffusion.ipynb`'s baseline uses *all* extrasomatic transcripts, which includes the
in-granule ones — so its "baseline" partly contains the signal it is supposed to be a null for, and
`delta` is biased toward zero. Removing them makes the three arms a true partition; the §6 gate
asserts they sum to the transcript count exactly, per gene, per sample.

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

## Runbook

**Six steps, each run exactly once. No parameter is adjusted at any point** — every notebook's
defaults produce the final tables. Run all notebooks from `R2_revision/ambient_controls/` on the
`mcDETECT-env` kernel (A3a §7 imports `mcDETECT`).

| # | step | where | needs | produces |
|---|---|---|---|---|
| 1 | `A3_preflight.ipynb`, top to bottom | local | nothing | `output/preflight/` |
| 2 | upload `set0_genes.csv` | local → HGCC | 1 | the array's only non-tracked input |
| 3 | the detection array | HGCC | 2 | `output/detect/` |
| 4 | download `output/detect/` | HGCC → local | 3 | the four sets on disk locally |
| 5 | `A3c` → `A3b` → `A3a`, each top to bottom | local | 4 | `output/a3{a,b,c}/` |
| 6 | `Rscript A3_figures.R` | local | 5 | `output/figures/` |

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
jupyter lab A3c_de_baseline.ipynb    # needs nothing from step 4; its section 1 caches the
                                     # transcript partition that A3a section 6 reuses
jupyter lab A3b_vicinity.ipynb       # sections 1-5 need set1_*/sphere_dict.parquet
jupyter lab A3a_three_sets.ipynb

# ---- 6. figures. Every section degrades to "[skip]" on a missing input, so this is safe
#         to run at any point. ----
Rscript A3_figures.R
```

**A partial download fails quietly.** Every set-dependent cell prints `[skip] ... missing` rather
than raising, so check `output/a3a/set_inventory.csv` has **4 sets x 2 samples = 8 rows** before
trusting §4-§6. `A3c` and `A3b` §6 need nothing from HGCC and can run while the array is still
queued.

### Toggles

Defaults are the final configuration; these exist for debugging, not for the normal run.

| notebook | toggle | default | effect |
|---|---|---|---|
| A3a, A3b, A3c | `VALIDATE` | **`True`** | correctness gates. On by default here, unlike A1/A2: the gates are the last section, so every table is already written when they run, and they are what catches a bad run |
| A3a, A3b | `DRY_RUN` | `False` | `True` subsamples (`MAX_SPHERES` / `MAX_GRANULES` = 200K) for a cheap smoke pass over every cell. **The tables a dry run writes are not final** |
| A3a | `RUN_LEAVE_ONE_OUT` | `True` | §2b — one KD-tree per NC gene |
| A3b | `RUN_PREDICATE` | `True` | §4-5, the slow and load-bearing step |
| A3b | `RUN_ROUGH_VARIANT` | `True` | §6 — needs no HPC output |
| A3c | `OVERWRITE` | `False` | `True` recomputes the cached transcript partition |
| A3c | `RUN_CLIP_BIAS` | `True` | §2 — needs `spot_embedding` |
| A3c | `RUN_COUNT_MODEL` | `True` | §4 — the non-compositional primary |
| A3c | `RUN_AXIS2` | `True` | §5 — WT/AD on the three layers |
| `A3_figures.R` | `RUN_*` | `TRUE` | one per section, numbered to the notebooks |

`A3_preflight.ipynb` has no toggles at all.

### Tables that are deliberately not plotted

Seventeen outputs are quotable tables for the Supplementary Note rather than figures, and
`A3_figures.R` does not read them. The load-bearing ones are `a3c/clip_bias_by_gene.csv` (how
biased the published spot-matrix subtraction is), `a3c/axis1_divergence_test.csv` (the divergence
claim stated as a test), `a3c/axis1_count_model.csv`, `a3b/profile_funnel.csv`,
`a3b/detection_predicate_thinned.csv` and `a3b/rough_variant_by_distance.csv`. This is a choice,
not an oversight.

### The one expensive local step

Assigning ~10⁸ transcripts to ~10⁶ spheres is the single heaviest operation in A3. It is done
**once**, by `A3c` §1, and cached to `output/a3c/transcript_layer_<sample>.parquet` (one int8
column, ~100 MB); `A3a` §6 reads that cache. Expect several GB of RAM for the KD-tree during that
first pass. Every KD-tree call in `a3_common` is batched (`query_ball_point` over arrays with
`workers=-1`) — the per-sphere Python loops these replaced would not have finished.

### Where the time goes

`set1` detection dominates the HPC stage — it is 2 of the 6 tasks and the only pair needing 200 G.
Note detection now runs DBSCAN with the size/in-soma filters **off** and applies them per gene
afterwards: identical merged output (verified to 1e-12; `miniball` is randomised, so mcDETECT's own
fine pass is not bitwise reproducible either), but it also yields the real `raw → size → in-soma`
funnel, which a filtered `sphere_dict` cannot. `set0`/`set3` (Set 3 seeds on ~2.9 M NC transcripts against Set 1's 33.3 M). Locally, A3b §4 is the
slow step — one core-point test per pseudo-granule — which is what `DRY_RUN` exists to shorten
while debugging; the final run must have it `False`. A3c §1's partition
is one batched ball query over ~10⁸ transcripts per sample and is cached to `partition_counts.csv`.

---

## Which output backs which claim

| response element | source file |
|---|---|
| "the CSR selector at α = 0.5 returns the threshold we used" | `preflight/csr_min_samples.csv` |
| "negative-control genes do not form granule-like aggregates, at matched abundance" | `a3a/funnel_by_gene.csv` (rate columns), `a3a/set_inventory.csv` |
| "NC pseudo-granules do not overlap real granules above chance" | `a3a/overlap_ladder.csv`, `a3a/overlap_transcript_level.csv` |
| "NC pseudo-granule density shows no WT/AD difference" | `a3a/set_density_per_region.csv` |
| "the conclusion survives a locally-adaptive threshold" | `a3a/adaptive_survival.csv` + `a3a/adaptive_caveats.csv` |
| "the NC filter is conservative, and what it removes is dominated by a list collision" | `a3a/gria2_partition.csv`, `a3a/nc_leave_one_out.csv`, `a3a/nc_ratio_corrected_summary.csv` |
| "and the Gria2 list policy leaves a <0.4 % discrepancy, which a 19-marker sensitivity closes" | `a3a/gria2_partition.csv` (`gap_frac_of_set1`, `*_ex_gria2`) |
| **"the detector would not have fired a few µm away"** | `a3b/detection_predicate.csv` (+ `_stratified`, `_thinned`) |
| "…and that is not a density effect" | `a3b/detection_predicate_stratified.csv` |
| "granule enrichment diverges from the detection-independent non-somatic baseline" | `a3c/axis1_gene_table.csv`, `a3c/axis1_summary.csv`, `a3c/axis1_divergence_test.csv` |
| "…and is not an artefact of compositional normalisation" | `a3c/axis1_count_model.csv` |
| "the AD granule signal is not reproduced by the raw non-somatic layer" | `a3c/axis2_wt_ad_by_layer.csv`, `a3c/axis2_layer_correlation.csv` |

---

## What this does and does not settle

**Does.**

- Runs, at last, both controls the reviewer offered in round 1 — the vicinity pseudo-granules and
  the somatic-vs-non-somatic DE baseline — with the baseline made genuinely granule-free.
- Tests at the **detection step**, not on already-called granules: A3b asks whether the detector
  would have fired at a matched sphere a few µm away, using eps-connectivity on the same seed gene.
- Answers the *substantive* version of the CSR worry — that a **global** threshold under-corrects
  where background is locally denser — by re-testing every call against a locally estimated one.
- Neutralises the two objections that would otherwise sink the three-set design: circularity (Set 0)
  and abundance (rates, and abundance matching).
- Puts four provenance traps on the record before a reviewer finds them.

**Does not.**

- **A3a §6 is one-sided.** It can only remove granules, never add ones a locally adaptive rule would
  have called in sparse regions. AD is the lower-density arm, so the untested direction is the one
  that would work against us. A true re-detection with an adaptive `bg_density` is the strict
  version and was scoped out.
- **n = 1 vs 1.** Nothing here fixes the single-section design; every WT/AD p-value remains
  pseudo-replication, and the inferential weight is deliberately on within-sample comparisons.
- **λ_local cannot fully separate signal from background.** At R = 25 µm with granules this dense,
  neighbouring granules contaminate the estimate; the answer is bracketed, not pinned.
- **A3b's literal control still requires a placement rule.** §6's rough-pass variant is the
  no-placement-rule cross-check, but the reviewer asked for the literal control by name and it
  carries its own assumptions.
- **The three-set design cannot prove ambient is unstructured** — only that structured ambient does
  not produce granule-shaped, marker-enriched, non-overlapping aggregates at these thresholds.
- **Set 0 is not abundance-matched at the top of the range.** The panel runs out of unannotated
  genes above ~700 K transcripts, so for the seven most abundant markers Set 0 is 5–23× rarer (see
  *The sets*). The per-million-transcript rate is what makes the comparison valid there; a raw
  count comparison would not be.

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
