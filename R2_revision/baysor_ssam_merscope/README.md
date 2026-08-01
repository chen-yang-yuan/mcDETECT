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

## Next (not in these scripts)

Post-hoc **filtering + evaluation** on each `spheres.parquet`: recompute
`in_soma_ratio` (via the transcript `overlaps_nucleus` column) and `nc_ratio` (via
`negative_controls.csv`); apply the filter arm (Baysor: size→in-soma→NC; SSAM:
in-soma→NC); then score concordance vs mcDETECT granules and WT/AD granule +
subtype density. See A1 Eval (a)–(c) in the plan.
