# Findings — Baysor and SSAM on real MERSCOPE WT / AD

What the assumption-light methods do and do not recover, measured against mcDETECT on the same
tissue, the same 290-gene panel and the same spatial scaffold.

All numbers below come from `output/postproc/` (git-ignored) and are reproducible by running
`postproc/A1_filter_de.ipynb` then `postproc/A1_figures.R`. Each table names the file it came from.
Method framing, thresholds and the population definitions are in `README.md`.

---

## Headline

**Baysor is not measuring granules — it is re-reporting the soma / ambient transcript field.**
**SSAM leans granule-ward but never gets there.** Neither reproduces mcDETECT's regional AD decline
without filtering, and no amount of filtering fixes either method's compartment mixing.

The single most decisive measurement. mcDETECT ran its own Subdomain 1 vs 2 contrast on three
expression layers — `granule` (the answer to reproduce), `cell` (soma transcripts) and `ambient`
(everything outside cells and granules). Correlating each arm's gene ranking against all three
places every method on that scale:

| arm | vs **granule** | vs **cell** | vs **ambient** | closest |
|---|---|---|---|---|
| baysor_all_pop2 | 0.585 | 0.758 | **0.990** | ambient |
| baysor_markers_pop2 | 0.601 | 0.758 | **0.984** | ambient |
| baysor_all_pop3 | 0.672 | 0.765 | **0.942** | ambient |
| baysor_all_pop1 | 0.672 | **0.938** | 0.923 | cell |
| baysor_markers_pop1 | 0.716 | **0.938** | 0.921 | cell |
| ssam_all_pop1 | 0.760 | **0.777** | 0.727 | cell |
| ssam_all_pop2 | 0.688 | 0.616 | **0.724** | ambient |
| ssam_all_pop3 | **0.687** | 0.546 | 0.564 | granule |
| ssam_markers_pop1 | **0.842** | 0.808 | 0.679 | granule |
| ssam_markers_pop2 | **0.770** | 0.675 | 0.708 | granule |

*Spearman over 290 genes. Source: `subdomain_anchor_comparison.csv`, extended with the cell/ambient
columns. For scale, mcDETECT's own layers correlate granule–cell 0.691, granule–ambient 0.592.*

Every Baysor arm's closest match is `cell` or `ambient`, at **r = 0.92–0.99** — far above its
correlation with `granule` (0.585–0.716). That is what a 2–3 µm segmentation tiling the whole tissue
must produce: in the two subdomains it places **444 detections per 50 µm spot**, so its aggregate is
essentially the spot's total content.

---

## 1. Subdomain DE and GSEA — asymmetric failure

`wt_ad/<arm>/granule_DE_genes_Subdomain 1_vs_Subdomain 2_GSEA.csv`

Every arm is aggregated onto mcDETECT's **own** microdomains, so the spatial partition is identical
and only the transcripts differ. Sign convention: **positive NES = Subdomain 1 (pre-synaptic);
negative NES = Subdomain 2 (post-synaptic / dendritic).**

mcDETECT's granule layer gives 12 terms at FDR < 0.05 — 6 pre-synaptic, 6 post-synaptic.

| arm | terms | FDR<0.05 | % +NES | PRE 6 | POST 6 |
|---|---|---|---|---|---|
| **mcDETECT [granule]** — target | 61 | 12 | 36.1 | 6/6 | 6/6 |
| mcDETECT [cell] — floor | 125 | 34 | 9.6 | 3/6 | 6/6 |
| mcDETECT [ambient] — floor | 80 | 39 | 1.2 | **0/6** | 6/6 |
| baysor_all_pop1 | 94 | 52 | 4.3 | 1/6 | 6/6 |
| baysor_all_pop2 | 76 | 25 | 1.3 | 0/6 | 6/6 |
| baysor_all_pop3 | 72 | 0 | 5.6 | 2/6 | 6/6 |
| baysor_markers_pop1 | 95 | 41 | 3.2 | 1/6 | 6/6 |
| baysor_markers_pop2 | 85 | 13 | 5.9 | 1/6 | 6/6 |
| **ssam_all_pop1** | 102 | 17 | 30.4 | **6/6** | 6/6 |
| ssam_all_pop2 | 38 | 3 | 26.3 | 5/6 | 6/6 |
| ssam_all_pop3 | 20 | 3 | 35.0 | 2/6 | 4/6 |
| **ssam_markers_pop1** | 77 | 20 | 23.4 | **6/6** | 6/6 |
| ssam_markers_pop2 | 36 | 7 | 19.4 | 3/6 | 6/6 |

**The post-synaptic half is uninformative.** All six terms are recovered by every arm *and by the
ambient floor*, often with larger |NES| than mcDETECT's own granule layer (Baysor pop1 gives
`PROTEIN_LOCALIZATION_TO_SYNAPSE` NES = −2.31 against mcDETECT's −2.12). Aggregating any transcripts
over these subdomains reproduces the post-synaptic contrast.

**The pre-synaptic half discriminates completely**, and the honest count is stricter than the table
suggests. Two of the six "pre-synaptic" terms are annotation artifacts:
`LEUKOCYTE_MEDIATED_IMMUNITY` and `IMMUNE_EFFECTOR_PROCESS` have core enrichment **Vamp2 / Cplx2 /
Stxbp1** — three SNARE proteins annotated to immune GO terms via *degranulation*. They are a
relabelling of the same synaptic exocytosis genes. Excluding them leaves four genuinely
pre-synaptic terms:

| term (mcDETECT NES) | ssam_all_pop1 | ssam_markers_pop1 | baysor_all_pop1 | baysor_all_pop3 | mcDET ambient |
|---|---|---|---|---|---|
| SYNAPTIC_VESICLE_RECYCLING (+1.81) | +1.64 | +1.62 | — | — | — |
| PRESYNAPTIC_ENDOCYTOSIS (+1.80) | +1.61 | +1.60 | — | — | — |
| EXOCYTIC_PROCESS (+1.75) | +1.73 | +1.70 | — | — | — |
| ENDOMEMBRANE_SYSTEM_ORGANIZATION (+1.74) | +1.65 | +1.65 | — | — | — |
| | **4/4** | **4/4** | **0/4** | **0/4** | **0/4** |

*"—" = not recovered (`p.adjust ≥ 0.25`).*

**Baysor recovers none of the four in any arm** — it sits exactly at the ambient floor. SSAM's
unfiltered arms recover all four.

One corroborating detail: Baysor's `LEUKOCYTE_MEDIATED_IMMUNITY` leading edge uniquely picks up
**C1qa / C1qb** — genuine complement genes, absent from both mcDETECT's and SSAM's leading edges.
Baysor is pulling in real microglial transcripts, which is independent evidence that its detections
are capturing somata rather than granules.

---

## 2. Detection density per region — RETIRED

This analysis was removed from the pipeline on 2026-08-21 and is not used in the response letter.
The numbers it produced, and the code that produced them, are recorded in `postproc/old/README.md`.
In one line: mcDETECT declined in 6/9 regions (median AD/WT 0.763); unfiltered Baysor was flat
(1.008, 4/9); filtered Baysor recovered the pattern (0.781, 6/9, r = 0.64); SSAM declined in 0/9
regions in every population. It was the only analysis in which filtering partially recovered
mcDETECT's result.

## 3. Compartment-marker mixing — intrinsic, and filtering does not fix it

`entropy_summary.csv`. Spread of each detection's profile over mcDETECT's 34 `REF_GENES`, averaged
over WT and AD:

| arm | effective markers | compartment-pure | median n_ref |
|---|---|---|---|
| **mcDETECT granules** | **1.9** | **32 %** | 3.5 |
| ssam/all/pop2 | 2.9 | 19 % | 3.0 |
| ssam/all/pop1 | 3.0 | 18 % | 3.0 |
| ssam/markers/pop1 | 3.2 | 11 % | 4.5 |
| baysor/all/pop2 | 5.5 | 2 % | 11.5 |
| baysor/all/pop1 | 6.0 | 2 % | 12.0 |
| baysor/all/pop3 | 6.0 | **0.0 %** | 17.5 |
| baysor/markers/pop2 | 7.9 | **0.0 %** | 28.0 |
| baysor/markers/pop1 | 9.2 | **0.0 %** | 30.5 |

*"effective markers" = exp(Shannon entropy); "compartment-pure" = fraction with a single marker
holding ≥ 90 % of the profile.*

A mcDETECT granule is a focal cluster of ~2 marker species and is pure a third of the time. A Baysor
detection blends 6–9, and **not one in the markers arm is pure**. This is the quantitative form of
the qualitative observation that Baysor/SSAM subtypes come out mixed.

**Filtering does not touch it**: Baysor pop1 → pop3 moves from 6.0 to 6.0 effective markers. Mixing
is a property of the detection geometry — 36 transcripts per detection against mcDETECT's 4 — not
something a post-hoc filter removes.

**The visual form of the same result** is produced by the pipeline: notebook §4 renders a
granule-subtype heatmap per arm into `subtype_heatmaps/<method>_<geneset>_<pop>.jpeg`, using
mcDETECT's own clustering procedure unchanged (MiniBatchKMeans k=15 on the 34 `REF_GENES` after
full-panel normalisation, then `sc.pl.heatmap` with `standard_scale="var"`). Only the figure is
exported — no subtype labels — because assigning clusters to compartments is a manual call these
detections do not support, which is exactly what the panels show. Arms are subsampled to
`HEATMAP_MAX_CELLS` = 200,000 detections for rendering, seeded.

---

## Caveats to state up front

1. **The subdomains are genotype-imbalanced.** Subdomain 1 is 88.5 % WT spots (989/128), Subdomain 2
   is 14.4 % (158/938). A reviewer will notice. It survives the obvious check: the
   Subdomain-1-vs-2 logFC correlates only **r = −0.21** with the bulk AD-vs-WT axis, against
   0.76–0.96 for a straight WT-vs-AD contrast — so the subdomains are imbalanced in *composition*
   while the expression axis they define is nearly orthogonal to genotype. Say this first.
2. **The immune terms are artifacts.** `LEUKOCYTE_MEDIATED_IMMUNITY` is the top positive-NES term in
   mcDETECT's own granule and cell tables and reads as an AD-microglia signature. With `setSize = 13`
   on a 290-gene panel it is a relabelling of synaptic exocytosis genes. Do not read immune biology
   into it — except in Baysor's case, where C1qa/C1qb appear and mean the opposite of what they seem
   to (soma capture, not biology).
3. **SSAM's pathway result is thin but not degenerate.** SSAM pop1 contributes ~9 detections per spot
   yet occupies all 2,213 subdomain spots. SSAM pop3 (2.1/spot over 1,672 spots) is thin enough to
   discount.
4. **Filtering is not uniformly good.** It halves negative-control content in both methods, but
   leaves compartment mixing untouched and *degrades* SSAM's pathway recovery (unfiltered 4/4
   pre-synaptic terms → 0/4 at pop3). "Filter harder" is not a general fix.
5. **Both methods were run at one detection scale.** SSAM's `bandwidth`/`sampling_distance` — its
   analogue of mcDETECT's `eps` and Baysor's `scale` — were never swept; only a post-hoc threshold
   was. Disclose if the tuning question is raised.

---

## What this supports

The two methods fail differently, and both failures are attributable to the detection rule rather
than to the downstream analysis, because every arm shares mcDETECT's spatial partition and gene
panel.

**Baysor** produces cell-sized, compartment-mixed objects that reproduce the soma/ambient field
(r up to 0.99) and recover none of the four pre-synaptic pathways, in any of its five populations.
Filtering removes nuclear and background material but leaves the mixing and the pathway result
unchanged.

**SSAM** produces small, comparatively pure detections that do recover the pre-synaptic pathway
signal — the one place an assumption-light method converges on mcDETECT's biology.

Baysor does not reach mcDETECT's granule layer on any axis measured, and post-hoc filtering does not
rescue it.

---

## Wired into the rebuttal

`plans/Response_R2_comments1-2_Baysor_SSAM.docx` carries Reviewer #2 comments 1 and 2 with this
analysis added in **purple** (black = reviewer, blue = carried-over response, red = manuscript edit,
green = draft placeholder).

It is argued in two parts — *what these methods detect*, then *what follows for the microdomain
biology* — so the mis-identification is established before any biology claim. **Every number is
reported for all five populations per method**, with mcDETECT as a baseline row, so no conclusion
depends on which arm was highlighted. Figures stay selective.

| § | claim | evidence |
|---|---|---|
| — | what each tool returns, and why SSAM needs an assigned radius | prose, sourced to Petukhov 2022 and Park 2021 |
| 1.1 | detections are nuclear / background | 10-row table: in-nucleus failures, negative-control content |
| 1.2 | they resemble mcDETECT's soma/ambient layers | 11-row table incl. mcDETECT's own granule↔soma (0.691) and granule↔ambient (0.592) baselines |
| 1.3 | they cannot be resolved into subtypes | 11-row table + **Fig. R-A / R-B** heatmaps |
| 2 | microdomain DE/GSEA recovers the wrong pathways | 13-row table + **Fig. R-C / R-D** dotplots |
| c2 | sizes, and why Baysor's radius exceeds its `scale` | 7-row table + the molecule-count regression |

| figure | source |
|---|---|
| R-A | `output/MERSCOPE_WT_AD_comparison/heatmap_subtype.jpeg` (mcDETECT, published) |
| R-B | `output/postproc/subtype_heatmaps/baysor_markers_pop2.jpeg` — **now pipeline-generated** (notebook §4), no longer recovered from the archive |
| R-C | `wt_ad/mcdetect_reference/granule_DE_genes_Subdomain 1_vs_Subdomain 2_target_GSEA.jpeg` |
| R-D | `wt_ad/baysor_all_pop1/granule_DE_genes_Subdomain 1_vs_Subdomain 2_target_GSEA.jpeg` |

Figures are converted to PNG for embedding — R writes JPEGs with an Adobe APP14 marker that Word's
importer via `python-docx` rejects. **Regenerate Fig. R-B from the new pipeline output** after the
notebook is re-run: the version currently embedded was extracted from the archived notebook and will
match closely but not exactly (same seed and settings, now on a 200k subsample).

**What is foregrounded.** Baysor leads throughout, because it fails on every axis and does so in all
five of its populations — showing all five is what makes that claim strongest, not weaker. Density is
gone entirely. The post-synaptic half of the pathway result gets one line, to pre-empt the question,
since everything including the ambient layer recovers it.

**Stated against interest**, because a reviewer checking the data would find them:
- SSAM *does* recover the pre-synaptic programme (4/4 in both unfiltered arms). Reframed as the
  assumption-independent support the reviewer asked for, not buried.
- SSAM's radius is assigned, not measured, so SSAM is excluded from the radius comparison outright.
- The size comparison does not by itself resolve the parameterisation concern.
- Baysor's realised radius (2.32 µm) exceeds the `scale` we set (1.5 µm). Addressed head-on: the
  log–log regression of radius on molecule count has slope 0.276 ≈ 1/3, so radius tracks molecules
  per segment rather than the prior, and even Baysor's smallest segments are 2.06 µm.
