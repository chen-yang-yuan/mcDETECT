# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working rule: do not run the code unless asked

When editing or updating any code in this repository, do NOT test, execute, or run it — not locally and not on the backend/HPC (no `python3 ...`, `sbatch`, `make`, notebook execution, etc.) — unless the user explicitly asks you to run it. Make the edits and stop.

## Deep reference

`plans/ANALYSIS.md` is the full methodology + per-script reference (algorithm, every numbered step, benchmarks, simulation, validation, cross-platform analysis). Read it when you need detail beyond this summary; keep it in sync when the algorithm or pipeline changes materially. This CLAUDE.md stays intentionally short — only what's needed to orient any run.

## What this repository is

mcDETECT is a computational framework for uncovering the "dark transcriptome" in polarized neuronal compartments (dendrites, axons, synapses) from *in situ* spatial transcriptomics (iST) data — MERSCOPE, CosMx, Xenium, MERFISH. It treats each mRNA molecule as a 3D point and uses density-based clustering (DBSCAN) to find extrasomatic mRNA aggregates ("granules") outside cell somata.

This is a **research repository** backing a paper, not just a library. It contains both:
- **`mcDETECT_package/`** — the pip-installable `mcDETECT` package (the reusable algorithm).
- **`code/`, `simulation/`, `validation/`, `other_analysis/`, `benchmark/`** — analysis scripts and notebooks that reproduce paper figures and consume the package.

## Environment & commands

There is no test suite, linter, or build step. Work is run via the conda environment and (on HPC) SLURM.

```bash
# Create/activate the reproducible environment (Python 3.10)
conda env create -f code/utils/env.yaml
conda activate mcDETECT-env

# Install the package for local development
cd mcDETECT_package && python setup.py install --user   # or: pip install -e .

# Run a pipeline step locally
cd code && python3 3_detection.py

# Submit a pipeline step to SLURM (each *.py has a matching *.sh wrapper)
sbatch code/3_detection.sh
```

`make push` (default `make` target) stages everything, auto-commits with a timestamp message, and pushes `main` to `origin`. The recent git history is dominated by these auto-commits — commit only when the user asks.

## Package architecture (`mcDETECT_package/mcDETECT/`)

Three modules, re-exported from `__init__.py` as `utils`, `model`, `downstream`.

- **`model.py`** — the `mcDETECT` class, the algorithm core. Constructed with a transcripts DataFrame plus granule markers (`gnl_genes`) and negative-control genes (`nc_genes`), and a `type` of `"discrete"` (MERSCOPE/CosMx — has a fixed z-grid) or `"continuous"` (Xenium). Key pipeline methods:
  - `dbscan()` — per-marker DBSCAN clustering into candidate spheres (min-enclosing spheres via `miniball`). `min_samples` is auto-selected per gene from a Poisson background-density model (`poisson_select`), unless `minspl` is set. Filters by sphere size and in-soma ratio (somata estimated by dilating nuclear masks).
  - `merge_sphere()` — merges spatially overlapping spheres across markers (overlap threshold `rho`).
  - `nc_filter()` — drops spheres enriched in negative-control genes.
  - `detect()` — orchestrates `dbscan → merge_sphere → nc_filter`, returning a granule-metadata DataFrame.
  - `profile()` — builds the granule × gene spatial-transcriptome `AnnData` by counting all transcripts within each sphere.
  - `spot_expression()` — grid/spot-level pseudo-expression `AnnData`.
- **`utils.py`** — helpers: KD-tree / R-tree builders (`make_tree`, `make_rtree`), correlation metrics, palette/plotting helpers.
- **`downstream.py`** — post-detection analysis: `GranuleSubtyper` / `classify_granules` (marker-based granule subtyping), and neuron/spot embedding functions (`neighbor_granule`, `neuron_embedding_*`, `spot_embedding*`) that relate granules back to neurons and spatial neighborhoods.

Transcript DataFrames use columns `global_x`, `global_y`, `global_z`, and `target` (gene name). Discrete platforms also have `global_z` on a discrete grid. When `merge_genes=True`, all `gnl_genes` are collapsed into one synthetic marker (`target_detect`) for detection while `target` is preserved for downstream counting.

## Analysis pipeline (`code/`)

Scripts and notebooks are **numbered in execution order** — the leading integer is the intended sequence, not part of a module system:

1. `1_clean_transcripts.ipynb` — ingest raw platform output → `data/<dataset>/processed_data/` (`transcripts.parquet`, `adata.h5ad`, `spots.h5ad`, `genes.csv`, `negative_controls.csv`).
2. `2_gene_ranking.py` — rank candidate granule markers by granule yield.
3. `3_detection.py` — main granule detection; writes `output/<dataset>/all_granules.parquet` etc. Does a "rough" pass (no filtering) then a "fine" pass (size + in-soma + nc filtering). Per-dataset parameters (rotation/flip/cutoff) are hard-coded near the top of the file.
4. `4_post_detection.ipynb` → `9_pathway_*.ipynb` — granule profiling, neuropil subdomain analysis (SpaGCN), pathway/expression analysis.

`code/benchmark/` compares mcDETECT against alternatives and parameter choices. `simulation/` runs the synthetic-data benchmark against Baysor (`run_Baysor.py`) and SSAM (`run_SSAM.py`); `code/utils/` bundles a macOS Baysor build and env specs.

Dataset naming: `MERSCOPE_WT_1` / `MERSCOPE_AD_1` (wild-type vs Alzheimer's disease) are **pair 1** (steps 4–7); `MERSCOPE_WT_2` / `MERSCOPE_AD_2` are **pair 2** (steps 8–9, the `_pair2` files). Plus `Xenium_5K` (primary, `continuous`), `CosMx`, `MERFISH`. "Neuropil subdomains" = spot clusters defined by granule-subtype composition within a cortical ROI.

## Important: gitignored working directories

The following are **gitignored** and hold large data/results that will NOT be in a fresh clone — do not assume they exist and do not try to commit them: `data/`, `output/`, `validation/`, `other_analysis/`, `code/old/`, `simulation/simulated_data/`, `simulation/output/`, and all `figures.*` / `figures_response.*` files. Scripts reference these via relative paths (`../data/...`, `../output/...`) and must be run from `code/`.

## Conventions

- Scripts hard-code file paths and per-dataset parameters at the top rather than taking CLI arguments; follow that pattern when adding a dataset.
- The package version lives in both `mcDETECT_package/mcDETECT/__init__.py` and `setup.py` — keep them in sync (currently 2.1.6).
- Tutorial/docs are external: https://mcdetect-tutorial.readthedocs.io/
