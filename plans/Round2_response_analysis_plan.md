# Round-2 Reviewer Response — Computational Analysis Plan

**Manuscript:** *Uncovering the dark spatial transcriptome in polarized neuronal compartments with mcDETECT* (nBME, revision round 2).
**Scope:** the **computational** analyses needed to address the round-2 reviewer comments — design, prior-round history, and expected results. Wet-lab / protein-staining asks and pure text edits are tracked briefly in Section 6. Planning document only; do not run code unless explicitly asked.

**Sources synthesized:** `R2_Reviewer_comments.docx` (this round), `R1_Response_to_reviewers.docx` + `R1_Supplementary_Notes.docx` (what was already done and why R2 is unconvinced), `R1_Manuscript.docx`, `analysis_details.md` (codebase reference).

---

## 0. Where we stand — and why these points are still open

- **Reviewer #1** — signs off. No action.
- **Reviewer #3** — one comment: single WT + single 5xFAD sample; **soften AD claims to proof-of-concept / hypothesis-generating** across Abstract/Results/Discussion ("down tongue"). Text-only (Section 6). Explicitly *not* a barrier to publication.
- **Reviewer #2** — the only holdout. **Every round-2 point is a continuation of a round-1 major point that R2 considers only partially resolved.** This is the crucial framing: we are not answering fresh questions, we are answering "your fix didn't fully close my original concern." The plan below is organized so that for each point we state *what was tried in round 1, why R2 still objects, and the analysis that actually closes it.*

R2's unifying worry across all points is **circularity**: that mcDETECT detects "mcDETECT-shaped" structures and that its granule-level structure / WT-AD signal could be a seeding or ambient artifact. The antidote in every case is an **independent method or an explicit null model** whose result does not depend on mcDETECT's own generative assumptions.

### Round-1 → Round-2 lineage & summary

| R1 major pt | R2 round-2 status | What was tried in R1 | Why R2 still objects | Analysis |
|---|---|---|---|---|
| **1, 7** (circularity, method comparison) | Open | Baysor/SSAM on **simulation** (Fig. R4); leave-out 5-marker enrichment (Fig. R11) | Simulation ground truth is mcDETECT-shaped; leave-out genes are themselves co-aggregating markers → doesn't prove genuine granules. "Argument now rests entirely on orthogonal validations." | **A1** |
| **4** (simulation / size) | Open | Revised soma-overlap 20%→5%, soma radius→5.5 µm (Fig. R7); kept ~1 µm granule mean; eps justified by NN distance (Fig. R10, Note 15) | 1 µm is built into the parameterization (eps=1.5 chosen to match); literature disagrees (150 nm–>1 µm); reported size-agreement is circular | **A4** (+ A1) |
| **6** (sparsity / stochastic origin) | **Open — "original request has not been met"** | Added read/gene-per-granule dist (Fig. R9, not in manuscript); moved WT/AD to microdomains (Fig. 5) | Median 6–7 reads / 4 genes; each granule ≈ its seeding marker; subtypes/t-SNE may follow from seeding alone; **stratify-by-complexity never done**; not convinced randomized data wouldn't give identical embedding; Fig. R9 still absent from manuscript | **A2** |
| **9** (ambient RNA) | Open | Diffusion check (Malat1, Fig. R12); CSR filtering rationale (Note 10); post-hoc ambient **regression** on granule density (the big table) | Ambient modeled as **CSR (spatially uniform)** but real ambient is structured & denser near plaques/dense cells → CSR under-corrects **at detection**; regression "operates on already-called granules." **The two controls R2 offered in R1 (somatic-vs-non-somatic DE baseline; vicinity pseudo-granules) were never done.** | **A3** |
| summary (protein-level) | Open (rebut) | — | All orthogonal validation is transcriptomic/anatomical; wants RBP colocalization | Wet-lab + rebuttal (Section 6) |

**Priority:** A2 (explicitly unmet request) ≈ A3 (repeat request, controls never run — these two carry granule *reality* vs the deeper circularity worry) > A1 (real-data method comparison — carries *specificity* on the native/default arms **and** assumption-independent *biology confirmation* on the tuned+filtered arms) > A4 (mostly framing; eps already justified).

---

## A1. Real-data benchmark: Baysor & SSAM on MERSCOPE WT vs AD — do assumption-light methods recover the granule population?

**R2 (round 2).** The Baysor/SSAM comparison is on simulated data "whose ground truth is generated under mcDETECT's own model." Run these assumption-light methods **on real data**: *if they converge on the same granule populations / WT-AD differences → assumption-independent support; if they diverge, that is itself informative.*

**Round-1 history / why still open.** In R1 the authors ran Baysor/SSAM only on **simulation** (Fig. 2g–h; good-match / false-positive counts from `code/figures_response.Rmd`: mcDETECT 2861 / 276; Baysor default 22 / 975; Baysor tuned 2873 / 2923; SSAM default 2425 / 26801; SSAM tuned 2396 / 2118) and added the 5-marker leave-out enrichment (Fig. R11). R2 rejects both: simulation is mcDETECT-shaped, and the held-out markers co-aggregate with the seeds so R11 only shows "detection doesn't recover its own seeds." Since R11 is dismissed, the **real-data comparison** is the remaining route. Our expectation (per PI) is **not** uniform failure but a *sweep that separates two claims*: some arms show mcDETECT's **specificity** (Baysor/SSAM native output is somatic, low-precision), while other arms — tuned + filtered — should **recover the granule population and reproduce the WT/AD biology**, giving the assumption-independent confirmation R2 called "strongest." The simulation counts already foreshadow this: only the *tuned* settings recover a granule-scale population, and even then at ≥ 50 % false-positive rates that mcDETECT's filtering avoids.

**What Baysor and SSAM actually are (methods primer — this governs the whole design).**
- **Baysor** (Petukhov et al., *Nat. Biotechnol.* 2022) is a **cell-segmentation** method: a Markov-random-field + Bayesian-mixture model that assigns every molecule to a **cell** using the joint transcriptional composition of the **entire panel**. Its one principal user parameter is `min-molecules-per-cell` (`-m`); `scale` (`-s`, expected cell radius) and MRF β(=1) set the rest. It has **no concept of an extrasomatic granule** and **no gene-subset targeting** — it consumes all genes. First-hand output: a per-molecule **cell assignment** (`segmentation.csv`).
- **SSAM** (Park et al., *Nat. Commun.* 2021) is **segmentation-free**: it builds a Gaussian-KDE **gene-expression vector field over all genes** (bandwidth 2.5 µm default), then reports **local maxima** of that field as putative **cell centers** and classifies pixels into cell types. It, too, uses the **whole panel** and has **no granule concept**. First-hand output: the KDE **local maxima** (candidate cell centers).
- **Consequence:** both are built to find **cells / cell types**, seeded at transcript-dense (**somatic**) locations — the opposite of mcDETECT's small, sparse, **extrasomatic** aggregates. The honest question is therefore not "can we tune them into granule detectors," but "run each as its authors intend and ask whether its native output coincides with the granule population." We expect it does **not**, and we quantify *why* (their objects are somata/cells: large, high in-soma, NC-contaminated).

**Gene input — sweep two sets (user directive #2).** Neither method accepts a marker list; **both use the full panel by design** (Baysor models joint composition across all genes; SSAM builds the KDE field over all genes). So we sweep the input transcript table between:
- **(i) Full 290-gene panel** — the methods' intended usage (`genes.csv`).
- **(ii) Granule markers only** — the exact 20 synaptic markers mcDETECT seeds on (`syn_genes` in `code/3_detection.py`), pre-filtering the transcript `target` column before the converters. This hands Baysor/SSAM **mcDETECT's own input** and asks whether the gene set was ever the limiting factor.

**Parameters — the two settings locked from simulation (user directive #3).** The published simulation comparison (`code/figures_response.Rmd`, `simulation/analyze_crosstab.py`) used exactly **two** settings per method. Reuse these verbatim:

| Method | **default** | **tuned** |
|---|---|---|
| **Baysor** | `-m 30 -s 30` (`min-molecules-per-cell = 30`, `scale = 30` → expects large cells) | `-m 30 -s 1.5` (`scale = 1.5` µm → granule-sized) |
| **SSAM** | **no thresholds** (`expression_threshold = 0`, `norm_threshold = 0`) | `expression_threshold = 0.027`, `norm_threshold = 0.2` (VISp preset) |

SSAM `bandwidth = 2.5 µm`, `sampling_distance = 2.0`, and fixed sphere radius `1.5 µm` are held constant in both (as in simulation); Baysor `min-molecules-per-cell = 30` is constant, only `scale` moves. Reporting **both** pre-empts a "you hand-tuned them" objection.

**Filtering — a third swept dimension (no-filter / filter), applied post-hoc (user directive #1).** Because we can recompute each detection's size, in-soma ratio, and NC ratio (Eval b), filtering becomes an explicit **arm** rather than a fixed convention:
- **No-filter:** each method's **native** set — every Baysor cell (only Baysor's own `< 12.5`-molecule drop) and every SSAM local maximum. Tests the methods as published.
- **Filter:** apply mcDETECT's own criteria to their outputs, **sequentially** — for **Baysor**: size (`sphere_r < size_thr = 4`) → in-soma (`in_soma_ratio < 0.1`) → NC (`nc_ratio < nc_thr = 0.1`); for **SSAM**: in-soma → NC **only** (SSAM radii are a fixed 1.5 µm, so a size filter is meaningless). Tests whether their outputs, cleaned the way mcDETECT cleans, recover the granule population.

**Run economy — many scenarios, few runs.** Filtering needs **no extra detection runs**: unlike mcDETECT there is **no cross-marker merging step**, so size / in-soma / NC are pure per-sphere post-hoc predicates on a single output. So the full per-method sweep is gene-input {all, markers} × parameter {default, tuned} × filtering {off, on} = **8 configs from only 4 detection runs** (gene × parameter); filter/no-filter is a cheap branch on each run's spheres. Across WT + AD × 2 methods = **16 detection runs** total, each scored by the identical code below.

**Reusable code.** `simulation/run_Baysor.py` (molecule CSV → Baysor CLI → per-cell **miniball** min-enclosing spheres → `baysor_spheres.parquet`) and `simulation/run_SSAM.py` (`run_kde → set_thresholds → find_localmax` → **fixed-radius** spheres → `ssam_spheres.parquet`) — **both already accept the MERSCOPE-native `(global_x, global_y, global_z, target)` schema**, so real transcript tables flow through unchanged; only pre-filter the gene column for sweep (ii), and set width/height/depth from the real coordinate ranges. `simulation/analyze_crosstab.py::summarize_sphere_matches` (good-match / low-purity per-object tally) and the `code/figures_response.Rmd` "detection_side" stacked bar are the exact reporting scaffolds. Bundled macOS Baysor at `code/utils/baysor_macos-latest_x64_build/` (export `$BAYSOR_BINARY`); envs `baysor_env` and `ssam_hpc` (`code/utils/ssam_hpc.yml`, Python 3.11, `ssam==1.1.3`). mcDETECT granules from `code/3_detection.py`. All sphere sets scored by the **same** code below.

**Evaluation — first-hand output → spheres → three lenses (user directive #4), computed across the whole sweep.**
Both converters already emit `(sphere_x, sphere_y, sphere_z, sphere_r)` (Baysor: per-cell miniball radius; SSAM: fixed radius). In simulation all three methods were scored by `simulation/evaluation_utils.py::compute_object_level_metrics` (per-object **purity** ≥ 0.5 and **completeness** ≥ 0.5, Dice-weighted **Hungarian** one-to-one match → good match vs false positive). Real data has **no per-transcript ground truth**, so we adapt:
1. **(a) Concordance with mcDETECT granules (per sample).** Treat mcDETECT granules as the reference and reuse `compute_object_level_metrics` (purity/completeness ≥ 0.5, Hungarian) + a nearest-centroid overlap fraction. Report the fraction of mcDETECT granules a Baysor/SSAM object recovers, and the fraction of their objects matching **no** granule — the real-data analogue of the good-match / false-positive stacked bar.
2. **(b) Intrinsic granule-likeness — the crux (user directive #4: in-nucleus / NC ratio).** For each method's spheres, recompute — from the **same** transcript table, using the precomputed `overlaps_nucleus` column and the NC gene set (`negative_controls.csv`) — the identical features mcDETECT reports per granule: **`in_soma_ratio`** (enclosed transcripts overlapping a nucleus / size), **`nc_ratio`** (NC transcripts / size), **`sphere_r`** (size), **`size`** (reads), **`comp`** (distinct genes). Compare distributions mcDETECT vs Baysor vs SSAM. *Expected:* native Baysor cells are large and soma-centred (high `in_soma_ratio`, large `sphere_r`, high read count); native SSAM maxima sit at transcript-dense somatic locations (high `in_soma_ratio`); both carry more NC signal than mcDETECT granules → they capture **somata / cells, not extrasomatic granules**. These same features **are** the filter predicates, so (b) both characterizes and drives the filter arm.
3. **(c) WT/AD biology — granule density + subtype density only (user directive).** From each config's detections, recompute (i) **granule density** WT vs AD per region and (ii) **granule-subtype density** — pre- vs post-synaptic, subtyped by enclosed-marker composition (`downstream.py::GranuleSubtyper` / `classify_granules`) — WT vs AD. **Stop here: no neuropil-microdomain and no pathway/GSEA analysis for Baysor/SSAM.** Scored by the same `downstream` / `code/4_post_detection.ipynb` code as mcDETECT.

**Data.** Fig. 4–5 dataset = **`MERSCOPE_WT_1` / `MERSCOPE_AD_1`** (custom 290-gene panel) — transcript lists under `data/MERSCOPE_{WT,AD}_1/processed_data/transcripts.parquet` (already carries `overlaps_nucleus`), granule outputs under `output/MERSCOPE_{WT,AD}_1/` and `output/MERSCOPE_WT_AD_comparison/`. (`MERSCOPE_WT_2/AD_2` do not enter this manuscript — ignore.)

**Expected results (balanced — the sweep separates two claims, not uniform failure).**
- **Specificity (mcDETECT better) — default and/or no-filter arms.** Baysor-default (`-s 30`) segments whole **cells** → near-zero granule concordance, high `in_soma_ratio`, large radius; SSAM-default (no thresholds) massively over-detects → huge false-positive count. Native (unfiltered) outputs are somatic and NC-contaminated relative to mcDETECT granules. (Mirrors simulation: Baysor default 22 good / 975 FP; SSAM default 2425 / 26801 FP.)
- **Biology confirmed (mcDETECT correct) — tuned + filtered arms, esp. marker-only input.** Once Baysor-tuned (`-s 1.5`) / SSAM-tuned outputs are size/in-soma/NC-filtered, the survivors should overlap mcDETECT granules and — the key point — **reproduce the same WT/AD granule-density reduction and pre-synaptic-subtype vulnerability**. Independent methods converging on the same disease biology is the assumption-independent support R2 called "strongest."
- **The trade-off is the story.** Even where tuned methods recover the population, they do so at far lower precision without filtering (≈ 50 %+ false positives in simulation); filtering their outputs closes much of the gap. So: the granule population is **real and recoverable by independent methods** (answers circularity), while mcDETECT reaches it **more precisely and without per-dataset tuning or a size/soma-aware model bolted on afterward** (answers specificity).

**Positioning (author's-eyes note).** As reframed, A1 now carries **both** messages R2 raised: *specificity* (native Baysor/SSAM recover cells, not granules) **and** — via the tuned+filtered convergence on WT/AD biology — *assumption-independent confirmation* of the granule population and its disease signal. It still does not, alone, settle R2's deeper worry that the granules could be an ambient/seeding artifact; that burden stays with the null-model / ambient controls (**A2, A3**) and the EM validation. Frame the rebuttal as: "we ran the assumption-light methods on real data as suggested — their native output is somatic (specificity), but tuned and filtered they recover the same granule population and the same WT/AD biology (convergence); mcDETECT reaches this more precisely and without per-dataset tuning."

**Deliverable.** New Supp. figure — per method × gene-set × parameter × filter arm: (i) concordance with mcDETECT granules (good-match / false-positive bar, à la `detection_side`), (ii) `in_soma_ratio` / `nc_ratio` / `sphere_r` distributions vs mcDETECT, (iii) WT/AD granule density + pre-/post-synaptic subtype density — plus a Supp. Note and 1–2 Results sentences. **Risk:** Baysor on a full section is heavy (paper reports ≤ 51 min / 3.7 M molecules; a 290-plex section is larger) — scope HPC time early; the 16 runs are fixed up front (locked params, post-hoc filtering), so run the marker-only input first (cheapest, most diagnostic).

---

## A2. Sparsity and the stochastic origin of granule-level structure

**R2 (round 2).** Median MERSCOPE granule ≈ 6–7 reads / 4 genes; because detection **seeds on one marker**, each granule ≈ its seeding marker (Fig. 3e), so subtypes (Fig. 3f) and the discrete t-SNE (Fig. 4d) "may follow from the seeding alone." *"I am not convinced randomized data … would not produce an essentially identical embedding."* **"My original request, to stratify by granule complexity and show the structure is not a low-count artifact, has not been met."** Also: put the per-granule read/gene distributions (Fig. R9) in the manuscript.

**Round-1 history / why still open.** In R1 the authors produced Fig. R9 (read/gene distributions) and pivoted the WT/AD analysis to aggregated neuropil microdomains (Fig. 5) — but **never stratified by complexity** and **never ran the randomization/null test**. R2 explicitly notes the request was not met, and reads the microdomain pivot as tacitly conceding single-granule profiles are too sparse. So A2 must directly deliver the unmet asks.

### A2a. Multi-gene granule (>2 genes) reanalysis — the explicitly requested stratification
Use the `comp` column (distinct genes per granule; from `mcDETECT.dbscan`/`detect`). Subset to **comp ≥ 3** and rerun the pair-1 pipeline: subtyping + t-SNE (`code/3_detection.py`) and the WT/AD density + microdomain analysis (`code/5_neuropil_subdomains_data.py`, `code/7_neuropil_subdomains.ipynb`, `downstream.py::spot_embedding`) on `MERSCOPE_WT_1/AD_1`. Report the multi-gene fraction and show the subtype structure, AD pre-synaptic reduction, and microdomain contrast **persist** (reduced n). *Also stratify by read count* (e.g. terciles) to show findings are not confined to the lowest-count granules.

### A2b. Label-permutation null vs real embedding — the direct rebuttal to "randomized data"
Build R2's own hypothetical and show it fails: **permute gene labels across transcripts** (preserve every position and overall density), then run the **identical** pipeline (single-marker seeding → detect → `profile` → K-means → t-SNE). Quantify structure (silhouette, subtype-cluster ARI/stability across seeds; reuse `code/benchmark/benchmark_clustering.py`) for **real vs permuted** over many permutations. **Expected:** real granules show significantly stronger, reproducible cluster structure than the permuted null (which collapses) → the embedding is **not** a pure seeding artifact. This is the specific claim R2 said they were "not convinced" about.

### A2c. Functional co-clustering of co-detected genes
Within multi-gene granules, build the gene–gene **co-occurrence** matrix; compare to the A2b permutation null to get pair-wise enrichment; test whether **within-functional-group** pairs (pre/post/dendritic/axonal from `downstream.py::GranuleSubtyper`, or GO terms) co-occur above chance; hierarchically cluster to see functional blocks emerge. **Expected:** same-compartment genes co-occur above chance and form blocks → granules capture genuine multi-gene functional modules, not random single-marker seeds. Answers R2's "does the co-detected gene come from the same functional group?" (per advisor note).

### A2d. Include Fig R9 in the manuscript
Already computed (`code/4_post_detection.ipynb` → `granule_reads_unique_genes_per_granule.parquet`). Promote to a supplementary figure — R2 asked twice. No new computation.

**Deliverable.** 1–2 Supp. figures (multi-gene + read-count stratification; real-vs-permuted embedding + co-occurrence blocks) + Supp. Note; Fig. R9 in supplement; a limitation paragraph on sparsity (Section 6).

---

## A3. Ambient RNA and the CSR (spatial-uniformity) assumption — at the detection step

**R2 (round 2).** Ambient is modeled as **CSR (spatially uniform)**, but real ambient (debris, dying cells, EVs) is **structured** — denser where cells are denser and near amyloid plaques, i.e. where AD pathology is worst. A CSR threshold could **under-correct locally and inflate granule calls** there, biasing WT/AD. The existing density regression "operates on granules that have already been called" and does not test the **detection step**. Wants a direct detection-step check — "such as the pseudo-granule negative control I suggested."

**Round-1 history / why still open (important).** R2's round-1 Major 9 offered **two** concrete controls: (a) DE between somatic RNA and **all non-somatic RNA**, independent of granule detection, then check granule-specific differences exceed that baseline; **or** (b) **pseudo-granules in the direct vicinity of actual granules** as a negative control. The authors did **neither** — they argued diffusion is negligible (Malat1, Fig. R12), explained the CSR filter (Note 10), and ran a post-hoc **ambient regression** on granule density (the big table; effect survives adjustment in most regions). R2 now says that regression is post-detection and re-requests the pseudo-granule control. **So A3's job is to finally run the detection-step controls that were skipped.**

**Reusable code.** `mcDETECT` with `gnl_genes = nc_genes` (Set 3); `nc_filter` (Set 2); `code/benchmark/benchmark_filtering.ipynb` (already compares none / in-soma / NC / both) as scaffold; `code/benchmark/benchmark_ambient.ipynb` (ambient regression) for the local-ambient extension; `evaluation_utils.py::compute_object_level_metrics` for overlap.

**Design.**
1. **NC-gene pseudo-granule control (advisor's 4-set design).** Detect on both samples:
   - **Set 1:** granule-marker detection, in-soma filter only (no NC).
   - **Set 2:** Set 1 + NC filter (= standard pipeline).
   - **Set 3:** **negative-control-gene** detection (nuclear-enriched), in-soma filter only → pseudo-granules.
   Compute overlap **Set1∩Set3** and **Set2∩Set3**. If structured ambient drove calls, NC genes would form spurious aggregates overlapping real granules. **Expected:** Set 3 is sparse with minimal overlap; NC filter (Set 2) suppresses residual overlap.
2. **Vicinity pseudo-granule control (R2's literal round-1 suggestion).** Place fake spheres offset from real granule centroids (matched radius, into neighboring extrasomatic space), profile them the same way, and show they capture far fewer transcripts / fail the granule criteria vs real granules — a within-sample negative control at random nearby locations.
3. **Somatic-vs-non-somatic DE baseline (R2's alternative round-1 suggestion).** Independent of detection, DE between somatic RNA and *all* non-somatic RNA (WT vs AD); show the granule-specific WT/AD differences **exceed / diverge from** this baseline non-somatic signal → the effect is granule-specific, not generic ambient.
4. **WT/AD test on NC pseudo-granules + plaque/structure robustness.** Set-3 density WT vs AD per region → **expected: no significant difference** (advisor's stated expectation), so the AD granule loss isn't ambient-driven. Extend `benchmark_ambient.ipynb` to condition the pre-synaptic reduction on **local cell density** (proxy for structured ambient / plaque regions). *Optional strongest test:* replace the global CSR `bg_density` in `poisson_select` with a **spatially-local (adaptive) background density**, re-detect, and show the WT/AD conclusion is unchanged — a direct rebuttal to "CSR under-corrects locally."

**Expected results (summary).** NC-seeded and vicinity pseudo-granules are sparse and spatially non-overlapping with real granules; NC pseudo-granule density shows **no WT/AD difference**; granule-specific WT/AD signal exceeds the non-somatic baseline; the pre-synaptic reduction survives conditioning on local cell density and a locally-adaptive threshold → the CSR assumption does not drive the disease effect, tested **at detection**.

**Deliverable.** Supp. figure (4-set + vicinity overlap; somatic-vs-non-somatic baseline; NC-pseudo-granule WT/AD; local-adaptive robustness) + Supp. Note; a Methods sentence noting the CSR step is optional and robust to a locally-adaptive alternative.

---

## A4. Granule size assumption

**R2 (round 2).** ~1 µm is built into the parameterization (eps=1.5 chosen to match assumed size; 90% of radii 0.51–1.57 µm, upper end ≈ eps); literature disagrees (≈150 nm to >1 µm; Knowles 1996; Krichevsky & Kosik 2001; Batish 2012; Bauer 2022). Presenting size-agreement as "consistency with prior reports" without acknowledging the disagreement is unconvincing.

**Round-1 history / why still open.** R1 already reframed size as a **plausibility check** (not validation) and justified eps via nearest-neighbor distance (Fig. R10 = 1.3–1.7 µm; Note 15). R2 accepts the reframing but still wants the **size disagreement explicitly acknowledged** and the modality caveat stated.

**Design (framing + light optional compute).**
- Add a limitation paragraph: our radii are an **RNA-aggregate, minimum-enclosing-sphere** measure in 3D, not comparable to protein/RBP-puncta or diameter-based 2D estimates (measurement-method **and** modality differences — advisor note); cite the disagreeing range explicitly rather than implying consensus.
- **Optional panel:** eps ∈ {1.3, 1.5, 1.7} µm sensitivity (sweep already in `code/benchmark/benchmark_DBSCAN.py`) showing downstream biology (density trends, subtypes) is stable → conclusions don't hinge on the exact size. Reinforced by A1 (assumption-light methods recover the same populations).

**Deliverable.** Limitation text + optional Supp. sensitivity panel (largely already computed).

---

## 5. Execution order & shared scaffolding

1. **A3** — mostly reuses detection + `benchmark_filtering`/`benchmark_ambient`; no external tools; closes a repeat request with controls that were skipped in R1. Fast, high-credibility.
2. **A2** — reuses `comp`, `benchmark_clustering`, subtyping/embedding; the permutation null (A2b) and stratification (A2a) are the explicitly unmet asks. Self-contained.
3. **A1** — heaviest (external Baysor/SSAM on full sections), but only **16 detection runs** (WT+AD × 2 methods × gene-set{all, markers} × param{default, tuned}); the filter{off,on} arm is **post-hoc** (no merging step). Params **locked** from simulation (Baysor `-m 30 -s 30` / `-s 1.5`; SSAM `0/0` / `0.2/0.027`). Both `run_Baysor.py`/`run_SSAM.py` already accept the real `(global_x,y,z, target)` schema; score all methods through the **same** concordance + in-soma/NC/size + WT/AD-density/subtype code (`analyze_crosstab.py`, `figures_response.Rmd` scaffolds). Start env/runtime scoping now; run marker-only input first.
4. **A4** — text + a panel lifted from `benchmark_DBSCAN`.

All outputs land in `output/MERSCOPE_WT_AD_comparison/` (the `MERSCOPE_WT_1/AD_1` comparison) beside existing figures; each becomes a Supp. Note + figure with a one-line Results pointer. Because most points are *repeat* concerns, the rebuttal letter should explicitly say, per point, "in round 1 we did X; the reviewer noted it did not fully address Y; we have now done Z" — matching the lineage table above.

---

## 6. Non-computational items (tracked, not designed here)

- **Reviewer #3 — soften AD claims.** Reframe AD findings as **proof-of-concept / hypothesis-generating** consistently in Abstract, Results, Discussion; avoid "biomarker / therapeutic target / confirmed selective vulnerability" as definitive ("down tongue"). Text only.
- **Limitation acknowledgments (Reviewer #2).** Add explicit text for: (i) genuine granules **vs** co-expressed transcript clusters — both biologically meaningful; simultaneous RBP+RNA spatial profiling is not currently feasible; (ii) **size** measurement/modality caveat (A4); (iii) **sparsity** of single-granule profiles (A2) — 290-gene panel, higher-plex future panels will improve detection, current power already yields robust, biologically meaningful results.
- **Include Fig R9** (per-granule reads/genes) in the supplement (A2d).
- **Protein / RBP colocalization (R2 summary).** **Out of computational scope** (wet-lab). Planned **rebuttal**: RBPs have insufficient **specificity** (also in soluble RNPs, other granule types, diffuse pools; e.g. Staufen1, FMRP) and insufficient **sensitivity** (heterogeneous, partially overlapping, maturation-/compartment-/state-dependent composition; absence of one RBP ≠ absence of a granule), so no single/small RBP set is a ground truth; standard granule ID uses physical isolation with RBPs supportive only; and we already provide **EM validation** beyond the transcript level. If pursued, stain two RBPs and show their **partial/dis-colocalization** with each other (reinforcing that no single RBP marks all granules). No computational deliverable.

---

*Maintenance: targets the round-2 rebuttal. When an analysis completes, link its Supp. Note/figure here and update `plans/analysis_details.md` if the pipeline changes. Documentation only — do not run code unless explicitly asked.*
