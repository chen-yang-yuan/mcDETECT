# Baysor & SSAM real-data benchmark on MERSCOPE WT / AD (plan A1)

Runs the two assumption-light methods (Baysor cell segmentation; SSAM
segmentation-free KDE) on the real `MERSCOPE_WT_1` / `MERSCOPE_AD_1` samples, so
their **native** detections can be compared against mcDETECT granules. This is the
detection stage of analysis **A1** in `plans/Round2_response_analysis_plan.md`.

These scripts are meant to be transferred to **HGCC** and run under SLURM — they
are **not** run locally (and never run automatically). Detection only: raw native
spheres are produced here; size / in-soma / negative-control **filtering is a
separate post-hoc step** (Baysor/SSAM have no merging step, so filters are pure
per-sphere predicates applied after concatenation).

## Sweep

| axis | values |
|---|---|
| sample | WT, AD |
| gene input | `all` (full 290-gene panel) · `markers` (the 20 mcDETECT `syn_genes`) |
| parameters | `default` · `tuned` (locked from the simulation) |

Locked parameters (from `code/figures_response.Rmd` / `simulation/analyze_crosstab.py`):

| method | default | tuned |
|---|---|---|
| Baysor | `-m 30 -s 30` | `-m 30 -s 1.5` |
| SSAM | thresholds `0 / 0` | `expression 0.027 / norm 0.2` (VISp) |

SSAM `bandwidth 2.5`, `sampling_distance 2.0`, `search_size 3`, sphere radius `1.5`
are held constant; Baysor `-m 30` is held constant. All values live in `config.py`.

## Parallelism

Every `(sample × param × geneset × tile)` unit is one SLURM **array task**. Within
a detection the tissue is cut into **1000 × 1000 µm tiles** (`TILE_SIZE`), detected
independently, then concatenated. Each transcript falls in exactly one tile
(half-open tiling) so concatenation is a clean stack; objects split across a tile
border are ignored by design. Coordinates are localized per tile before detection
(so SSAM's grid stays small) and re-globalized on the resulting sphere centers.

Filtering is **not** a run axis: it is applied to each config's `spheres.parquet`
afterward, so only `sample × param × geneset × tile` runs actually execute
(4 configs/sample × #tiles, per method).

## Files

```
config.py            shared config: paths, sweep axes, locked params, tiling, layout
common.py            load a tile shard, restrict genes, localize/globalize coords
build_manifest.py    tile each sample -> write per-tile shards + manifest.csv
run_baysor_tile.py   Baysor on one manifest tile -> miniball spheres (tile parquet)
run_ssam_tile.py     SSAM on one manifest tile   -> fixed-radius spheres (tile parquet)
concat_spheres.py    stitch tiles -> one spheres.parquet per config (+ completeness check)
slurm/               SLURM wrappers (see below)
```

Outputs (git-ignored) go to this analysis dir's own `output/` subfolder
(`R2_revision/baysor_ssam_merscope/output/`):

```
manifests/manifest.csv, n_jobs.txt
manifests/shards/<sample>/tile_RRRR_CCCC.parquet   (per-tile transcript shards, all genes)
<method>/<sample>_<param>_<geneset>/tiles/tile_RRRR_CCCC.parquet
<method>/<sample>_<param>_<geneset>/spheres.parquet
spheres_summary.csv
```

`build_manifest.py` reads each sample's transcript table **once** and writes the
per-tile shards; every array task then reads only its own shard (not the whole
sample), which is the main I/O win of tiling.

## How to run on HGCC

Assumes the repo is at `~/hulab/projects/mcDETECT` (edit the `cd` lines in
`slurm/*.sh` if not) with `data/MERSCOPE_{WT,AD}_1/processed_data/transcripts.parquet`
present, and conda envs `mcDETECT-env`, `baysor_env` (Baysor on PATH), `ssam_hpc`.

**Required Baysor patch:** Baysor 0.7.1's cosmetic NCV color-embedding
(`gene_composition_colors(bm_data.x, …)` in `cli_wrappers.jl::run_segmentation`)
crashes on this data (`UmapFit: size(X,2) must be greater than n_neighbors`)
*before* `segmentation.csv` is written, killing the tile. Wrap that call in a
`try/catch` that fills `ncv_color` with a placeholder on failure — cosmetic only,
segmentation and locked params unchanged (the wrapper runs from source, so the
edit applies on the next run). See the plan's HPC section for the exact edit.
Sparse edge tiles are handled as 0 detections by `run_{baysor,ssam}_tile.py`, so
smoke-test a *dense* tile (high `n_tx` in `manifest.csv`), not just `--array=0`.

```bash
cd ~/hulab/projects/mcDETECT/R2_revision/baysor_ssam_merscope

# 1) Build the manifest (fast; login node is fine).
bash slurm/build_manifest.sh            # -> manifests/manifest.csv, n_jobs.txt

# 2) Submit both detection arrays + concat (afterany) in one shot.
bash slurm/submit_arrays.sh 50          # 50 = max concurrent array tasks

#    ...or submit manually:
# N=$(cat output/manifests/n_jobs.txt)
# sbatch --array=0-$((N-1))%50 slurm/run_baysor.sh
# sbatch --array=0-$((N-1))%50 slurm/run_ssam.sh
# sbatch slurm/concat.sh                # after both arrays finish
```

Per-tile outputs mean reruns are cheap: finished tiles are skipped, so a partially
failed array can simply be resubmitted (add `--overwrite` to force recompute).
`concat` cross-checks tiles against the manifest and lists any config with missing
tiles in `spheres_summary.csv`, so partial failures are never silently dropped.

## Post-hoc stage — `postproc/` (runs locally, not on HGCC)

Once the detections are transferred back, `postproc/` filters them and runs the WT/AD
comparison. Scope is `param = tuned` with **two parallel gene-set settings analyzed
separately**; `param = default` is used *only* as a detection-count scale check.

```
postproc/postproc_config.py   paths, thresholds, registration constants, analyzed slice
postproc/sphere_features.py   vectorised re-implementation of mcDETECT.model.profile, plus
                              region annotation, marker entropy, granule-subtype clustering and
                              heatmaps, subtype density per region, and aggregation onto
                              mcDETECT's spot grid
postproc/A1_filter_de.ipynb   the notebook: [1] scale check  [2] profile  [3] populations and
                              the three quality measures  [4] mixing + subtype heatmaps
                              [5] subdomain-anchored DE  [6] correctness gates
postproc/A1_ssam_subtypes.ipynb
                              SSAM only, mcDETECT's three-step subtyping loop: [2] cluster from
                              scratch (per-population seed) + heatmap  [3] manual cluster ->
                              compartment mapping  [4] the same heatmap redrawn with clusters
                              grouped by compartment, which is the check on [3]  [5] subtype
                              density per brain region  [6] WT synaptic density vs the
                              genetic-labeling reference (Santuy et al. 2020), mcDETECT's own
                              external check -- tables only, as mcDETECT reports it. Baysor is excluded by design -- its
                              detections do not resolve into compartment-specific clusters, so no
                              honest mapping exists for it. Stops at density; microdomains stay in
                              A1_filter_de.ipynb section 5.
postproc/A1_figures.R         the figures, numbered to match the notebooks:
                              PART A  [1] detection radius + detection size
                                      [2] in-nucleus ratio, both denominators
                                      [3] NC ratio, both denominators
                              PART B  [4] compartment-marker mixing (vs mcDETECT granules)
                                      [5] subdomain DE -> GSEA / NES dotplots + the anchor
                                          comparison + pathway recovery
                                          (chord diagrams: RUN_GSEA_CHORD, off by default)
                                      [6] SSAM subtype density bars <- A1_ssam_subtypes.ipynb
                              (subtype heatmaps are rendered by the notebooks, not by R)
                              Each section has its own toggle.
postproc/old/                 retired analyses, with the reasoning in its README.md
```

**Part A (sections 1–3)** describes what a single detection looks like, for every population and
both gene sets. **Part B (sections 4–6)** asks what the detections recover: mcDETECT's
granule-subtype purity, and its subdomain pathway result.

**Where mcDETECT appears.** No mcDETECT data enters a part-A output — those panels describe the
Baysor / SSAM detections on their own terms, and mcDETECT is cited there only as the source of a
threshold or a plotting style. Part B is the opposite by design: it exists to ask whether these
detections reproduce results mcDETECT reports. The single exception in part A is the notebook's
section 1, which compares detection *counts* against mcDETECT's granule count — that is parameter
selection, and it is what justifies `param = tuned`.

**Two arms were retired** and are archived under `postproc/old/` with the numbers behind each
decision. A WT-vs-AD detection-level and pseudo-spot-level DE/GSEA arm could not separate the
methods, because the contrast is a property of the transcript field rather than of the detector —
unfiltered Baysor's per-gene logFC is **r = 0.96** with a logFC computed from raw transcripts with no
detection step at all, and mcDETECT's own WT-vs-AD ranking has **0 of 69 terms at FDR < 0.05**. And
`A1_granule_subtypes.ipynb`, which recomputed granule subtypes and microdomains for **both**
methods, was superseded: section 4 reproduces its subtype clustering and heatmaps without needing
the manual cluster→compartment mapping, section 5 gives a better-controlled subdomain DE, and
section 4 also turns its central observation — "the subtypes come out highly mixed" — into a
measurement. Its subtyping half was later **reinstated for SSAM alone** as
`A1_ssam_subtypes.ipynb`, because SSAM's clusters — unlike Baysor's — do resolve into
compartments, so the manual mapping that blocked the original notebook is answerable for it.

| measure | reported as | section |
|---|---|---|
| `sphere_r` | detection radius (µm) | 1 |
| `n_total`, `n_marker` | transcripts and granule-marker transcripts per detection | 1 |
| `in_soma_ratio` | in-nucleus fraction, **two denominators** | 2 |
| `nc_ratio` | negative-control fraction, **two denominators** | 3 |

Both ratios are reported on two denominators — `_all` over every transcript inside the detection,
and `_marker` over the granule-marker transcripts only (mcDETECT's own definition). `n_nc` counts
the top-`NC_TOP` negative-control genes, selected exactly as `model.py::nc_filter` selects them.

Detection size is the measure that actually separates the methods — Baysor holds a median of ~37
transcripts per detection against SSAM's ~10, which is what makes Baysor's
in-nucleus ratio a continuum rather than a spike at 0 and 1. It is plotted for **both** methods
(unlike the radius, which is a constant we set for SSAM) and styled after the mcDETECT counterpart in
`code/figures_response.Rmd:1804-1815`. `n_marker` is genuinely 0 for some detections — 13.9% of
SSAM's `all`/pop1 in WT, 12.4% in AD, against 1.2% / 1.6% for Baysor — so the linear per-population panels keep that bar visible and the log-scaled
overlay/boxplot state the fraction they cannot draw.

Each measure gets a summary table, an **overlay** (all populations on one axis) and a **standalone
per-population panel** for every method × gene set × population × sample; the panels land in
`output/postproc/per_population/` under the stem
`<measure>[_<denominator>]_<method>_<geneset>_<pop>_<sample>.jpeg`.

There is deliberately **no mcDETECT comparison** in these sections — no reference distribution, no
gate-pass panel, no enrichment-over-chance. The panels describe the Baysor / SSAM detections on
their own terms. The exports in section 3 loop over the Baysor/SSAM feature tables, so no
mcDETECT row can reach those panels in the first place — nothing is filtered on read.

### What is and is not circular

`pop2` **is defined by** `in_soma_ratio_all < 0.1`, and `pop3` adds `marker_frac >= 0.4`. So the
in-nucleus distributions of `pop2`/`pop3` (section 2) are truncated **by construction** — `pop1` is
the informative panel there, and the rest is bookkeeping.

**`nc_ratio` is filtered on nowhere** — not in the detection sweep, not in `pop2`, not in `pop3`,
and no population definition touches `nc_top` or `nc_thr`. Its `pop1 → pop2 → pop3` trend is
therefore a genuine, independent result, in whichever direction it runs; a flat or worsening trend
would say the nucleus/marker filters do not buy back granule specificity, which is equally
reportable. Nothing is tuned on it.

The one caveat: `nc_ratio_marker = n_nc / n_marker` shares its denominator with the `marker_frac`
cut defining `pop3`, so it shifts there partly mechanically. Kept as is and disclosed — read
`nc_ratio_all` for `pop3`; the marker one is a sensitivity check.

Section 2 reports the in-nucleus ratio of every population on both denominators, with no external
comparator. The distributions pile up at exactly zero, so `A1_filter_de.ipynb` exports a pre-binned
histogram plus quantiles rather than ~8 M raw values, and the script emits a full-range panel, a
non-zero-only panel and a printed numeric table. Note when reading it that `pop2`/`pop3` are
*defined* by `in_soma_ratio_all < 0.1` and so are truncated by construction — `pop1` is the
informative panel.

| geneset | detection input | populations |
|---|---|---|
| `all` | full 290-gene panel | `pop1`, `pop2`, `pop3` |
| `markers` | the 20 `syn_genes` | `pop1`, `pop2` |

`pop1` = all detections; `pop2` = out-of-nucleus (`in_soma_ratio < 0.1`, mcDETECT's cutoff);
`pop3` = `pop2` **and** `marker_frac >= 0.4` (`MARKER_FRAC_THR`; an absolute, uniform cutoff so the
WT and AD arms are filtered identically — ~1.28× the panel-wide marker background of 0.3125, and
section 3 of the notebook prints a scan of `pop3` sizes across candidate cutoffs for retuning).
`pop3` exists only for `all` — in the `markers`
setting every transcript that formed the detection is already a marker, so the filter is vacuous.
Detections from **both** settings are profiled against the full 290-gene panel, exactly as mcDETECT
seeds on 20 markers but profiles every gene (`3_detection.py:111`).

### The three analyses beyond the per-detection measures

**Compartment-marker mixing (notebook §4 → R §4).** mcDETECT's granule subtypes come from
clustering on the 34 `REF_GENES`, so how *concentrated* a detection's profile is over those markers
is the per-detection precursor of whether it earns a pure subtype label — measurable without the
manual mapping, which is what makes it usable here. Reported as `entropy_norm` (0 = one marker,
1 = uniform), `perplexity` (effective number of markers), `top1_share`, and the fraction of
detections holding **no** marker at all, which is itself method-dependent. mcDETECT's own granules
are exported on the same statistic as the reference distribution. **No gene → compartment grouping
is applied**: mcDETECT assigns compartments at the cluster level, never gene by gene, so a gene-wise
map would be this repository's invention rather than the manuscript's.

**Subdomain-anchored DE → GSEA (notebook §5 → R §5).** The closest test to the original intent.
Rather than asking whether Baysor/SSAM can *build* a microdomain map — they cannot; their detections
subtype as highly mixed — each arm is aggregated onto mcDETECT's **own** 50 µm spot grid
(`SF.anchor_to_mcdetect_spots`), restricted to `Subdomain 1` / `Subdomain 2`, and run through the DE
call from `code/7_neuropil_subdomains.ipynb` cell 9 verbatim. The spatial partition is therefore
identical across arms and only the transcripts differ. GSEA is `clusterProfiler::GSEA` on `msigdbr`
mouse **C5:BP** — the same database as the manuscript.

Each arm is written to `wt_ad/<method>_<geneset>_<population>/`, and mcDETECT's anchor tables sit
beside them in `wt_ad/mcdetect_reference/`.

`wt_ad/mcdetect_reference/` holds mcDETECT's own DE over the same pair on three expression
layers, and it supplies the scale the arms are read against:

| anchor | r(logFC) vs granule | genes at padj < 0.05 | role |
|---|---|---|---|
| `granule` | — | 161 / 290 | the answer to reproduce |
| `cell` | 0.371 | 234 / 290 | floor: soma transcripts over the same subdomains |
| `ambient` | 0.416 | 253 / 290 | floor: everything outside cells and granules |

The floors carry *more* significant genes than the target, so "this arm produced a significant
result" proves nothing — an arm landing at `ambient` has recovered the spatial contrast without
recovering anything granule-specific. R §5 writes `subdomain_anchor_comparison.csv/.jpeg` placing
every arm on that scale.

**Disclose with this analysis.** The two subdomains are strongly genotype-imbalanced: Subdomain 1 is
**88.5 %** WT spots (989 / 128) and Subdomain 2 is **14.4 %** (158 / 938). The contrast nevertheless
is not a genotype contrast in disguise — its logFC correlates only **r = −0.21** with the bulk
AD-vs-WT axis, against 0.76–0.96 for the retired WT-AD DE arms — but this should be stated up front
rather than left for a reviewer to find. Both samples are pooled, exactly as mcDETECT pools them.

One caution on the published GSEA tables: `LEUKOCYTE_MEDIATED_IMMUNITY` is the top positive-NES term
in the granule and cell 50 µm outputs and reads as an AD-microglia signature, but its
`core_enrichment` is **Vamp2 / Cplx2 / Stxbp1** — three SNARE proteins annotated there via immune
*degranulation*. At `setSize = 13` on a 290-gene panel it is a relabelling of the synaptic exocytosis
genes that already drive `SYNAPTIC_VESICLE_RECYCLING`. The same applies to the other immune terms in
the `cell` table. Do not read immune biology into them.

**In-soma denominator.** mcDETECT's `in_soma_ratio` counts **granule markers only**
(`model.py:102`, `258-267`). Under the all-genes arm many tuned Baysor spheres hold zero marker
transcripts, so the primary ratio here uses **all** transcripts in the sphere and the marker-only
ratio is reported alongside as a sensitivity column.

**Expression quantification.** Detections are quantified by the transcripts inside their **enclosing
sphere** for *both* methods. mcDETECT's granule expression is sphere capture and SSAM emits no
molecule assignment, so this is the only rule that isolates the detection difference rather than
confounding it with the counting rule. For Baysor the sphere captures a median ~1.4× more transcripts
than Baysor assigns to that cell (q95 ≈ 2.7×) — disclose in Methods. (Baysor's `segmentation.csv`
is not retained by the detection stage, so its native assignment would require a re-run.)

**Capture efficiency.** mcDETECT's published subtype-density analysis divides the AD side by
`CAPTURE_EFFICIENCY_COEF = 0.818691` (hard-coded in `benchmark_subtyping.ipynb` cell 22, with no
derivation recorded anywhere in the repo). **That correction is deliberately not applied here** —
`postproc_config.py` sets the constant to `1.0`. At 1/0.818691 = 1.22× it is essentially the whole of
SSAM's apparent AD density increase, which is ~1.05, i.e. flat, uncorrected (`postproc/old/README.md`,
the retired overall-density section). Consequence to state wherever the two are shown together: the
16-column schema and the **WT** densities are directly comparable to mcDETECT's published table, the
**AD** densities are not on the same scale. Restoring it is a one-constant change plus a re-run of
`A1_ssam_subtypes.ipynb` §5 and `A1_figures.R` §6.

No size or NC filtering is applied; `sphere_r` and `nc_ratio` are computed for reporting only — see
*What is and is not circular* above, which is what makes the NC panels worth anything.

**Radius panels.** Only Baysor is plotted: SSAM's `sphere_r` is the constant we set
(`SSAM_DETECTION_RADIUS = 1.5` µm), so it has no distribution to compare — it is carried in
`radius_summary.csv` so the omission is visible rather than silent, and the notebook asserts it
really is constant. Section 1 writes one overlay, `radius_overlay.jpeg`, on a log-x axis; the
per-population panels cut the x-axis at `RADIUS_QUANTILE = 0.999` so the tail does not flatten the
bulk. Baysor's median radius is ~2.3 µm (`all`) and ~3.1 µm (`markers`).

**Which two subdomains are contrasted** is fixed, not chosen per run: `C.SUBDOMAIN_PAIR` in
`postproc_config.py` is `("Subdomain 1", "Subdomain 2")`, mcDETECT's own published pair. The DE
tables use the same filenames and columns as `code/7_neuropil_subdomains.ipynb`, and **section 5 of
`A1_figures.R` discovers every one of them automatically**, producing the GSEA table, NES dotplots
beside each, matching `output/MERSCOPE_WT_AD_comparison/neuropil_subdomains_Isocortex_50/` for
mcDETECT (chord diagrams complete the set when `RUN_GSEA_CHORD` is on). The GSEA and
`make_gsea_chord` code is lifted from `code/figures_response.Rmd`, so no path in that Rmd has to be
repointed.

**Run order**, from `postproc/` in `mcDETECT-env`:

1. `A1_filter_de.ipynb` — everything except the SSAM subtypes. Set `OVERWRITE = True` when re-running
   after a change, or the cached section-4 heatmaps and section-5 DE tables survive untouched.
2. `A1_ssam_subtypes.ipynb` — run once with `MANUAL_SUBTYPE_MAPPING` empty, read each arm's
   `heatmap_subtype.jpeg` / `subtype_marker_means.csv`, fill the five blocks in, then re-run §3–§4
   and check the ordered heatmap before letting §5–§6 run. Nothing here is cached. §6 writes no
   figure — it is two correlation coefficients per arm, which is how mcDETECT reports it.
3. `A1_figures.R` — every figure and summary table. Each section has its own toggle.

Outputs land in `output/postproc/` (git-ignored). The profiling step is the expensive one (3.5 M
Baysor spheres against a 103 M-row transcript table) — it is chunked and cached, with `MAX_SPHERES`
for a dry run.

## Which output backs which rebuttal figure

`plans/Response_R2_comments1-2_Baysor_SSAM.docx` is written directly from the files below; there is
no intermediate summary document. Every number in it should be re-derivable from `output/postproc/`
alone.

| response element | source |
|---|---|
| in-nucleus failure rate, negative-control content, per arm | `in_soma_summary.csv`, `nc_summary.csv` |
| resemblance to mcDETECT's granule / soma / ambient layers, incl. mcDETECT's own baselines | `subdomain_anchor_comparison.csv` |
| compartment mixing per arm | `entropy_summary.csv` |
| Fig. R-A — mcDETECT subtype heatmap | `output/MERSCOPE_WT_AD_comparison/heatmap_subtype.jpeg` (published) |
| Fig. R-B — Baysor subtype heatmap | `subtype_heatmaps/<arm>.jpeg`, n from `subtype_heatmap_index.csv` |
| GSEA significance and pathway recovery per arm | `subdomain_pathway_recovery.csv` |
| synaptic density vs the genetic-labeling reference, per SSAM arm | `ssam_subtypes/ground_truth_correlation.csv` (mcDETECT's published 0.9098 / 0.8784 cited in the same table) |
| Figs. R-C / R-D — pre-synaptic GSEA dotplots | `wt_ad/<arm>/granule_DE_genes_Subdomain 1_vs_Subdomain 2_target_GSEA.jpeg` |
| radius and transcripts per detection | `radius_summary.csv`, `size_summary.csv` |
| mcDETECT's own radius / transcript baseline | `mcdetect_baseline.csv` |
| why Baysor's radius exceeds its `scale` prior | `radius_vs_molecules.csv` |

Figures are converted to PNG before embedding: R writes JPEGs carrying an Adobe APP14 marker that
Word's importer (via `python-docx`) rejects.
