#!/usr/bin/env python3
"""
A3e, stage 2 -- re-run the published detection over the relabelled transcript table.

Reviewer #2, major point 9 asks whether granules might be locally elevated ambient RNA rather than
real compartments. A3d answers that statistically. This answers it with the detector itself.

`A3e_pseudo_granules.ipynb` takes the published granules, samples two disjoint tenths of them, and
rewrites the GENE IDENTITIES of the transcripts inside them -- one tenth from the composition of
the residual extrasomatic RNA in the granule's own 10 um square, the other by permuting the
granule's own labels among its own points. Every transcript keeps its exact (x, y, z). It writes
the change as a patch. This script applies that patch and runs the published pipeline again.

The question each arm answers
-----------------------------
    ambient    would mcDETECT call this object if its contents really were a random sample of the
               RNA around it? This is the reviewer's hypothesis, built and handed back.
    scramble   the machinery control. Relabelling also scrambles WHICH point carries which gene,
               and that alone could break DBSCAN's eps-connectivity. This arm scrambles exactly
               the same way while preserving the granule's composition, so `ambient` read against
               `scramble` is the compositional effect on its own.
    untouched  everything else. Must come back at ~100%: it is simultaneously the proof that this
               script reproduces the published pipeline and the proof that the perturbation stayed
               local. minspl = 3 is fixed, so poisson_select never runs and DBSCAN is a purely
               local function of the point pattern -- but that is measured here, not assumed.

Why this reproduces Set 2 rather than Set 1
-------------------------------------------
The granules being converted are the PUBLISHED ones, so the re-run has to be the published
pipeline: dbscan -> size/in-soma filters -> merge_sphere -> nc_filter, with the 19-gene NC list
Set 2 was built with (a3_config.PSEUDO_NC_LIST). A3a already validated exactly this reproduction
on the unmodified table -- 681,346 rebuilt against 681,337 published in WT -- so the code path is
known-good and that 9-sphere slack is what a3_config.PSEUDO_CONTROL_MIN is set against.

`flatten_sphere_dict` and `apply_fine_filters` are IMPORTED from run_detection_sets rather than
copied: the identity that makes the one-pass design correct (model.py:288 applies the size and
in-soma predicate row-wise at the END of dbscan, i.e. before merging) is documented there and
verified there, and two copies of it would eventually disagree.

Usage
-----
    python3 run_pseudo_detection.py [task_id]      # defaults to $SLURM_ARRAY_TASK_ID
    python3 run_pseudo_detection.py --list         # print the task table and exit
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_config as C
import a3_common as A3
from run_detection_sets import apply_fine_filters, flatten_sphere_dict

from mcDETECT.model import mcDETECT


def tasks():
    """Array index -> sample. One place, so the SLURM array size is derivable."""
    return list(C.SAMPLES)


def load_patch(sample):
    """The relabelling built by the notebook, plus the guard that makes it safe to apply.

    The patch is POSITIONAL -- `row` indexes the transcript table's own row order, not its
    __index_level_0__, which carries gaps from Vizgen filtering. a3e_relabel_scope.csv records the
    table length it was built against; applying it to any other table would silently rewrite the
    wrong transcripts, so the length is checked before anything is touched.
    """
    patch_path = C.pseudo_relabel_path(sample)
    scope_path = C.A3E_DIR / "a3e_relabel_scope.csv"
    for f in (patch_path, scope_path):
        if not f.exists():
            raise SystemExit(
                f"{f} is missing. Run A3e_pseudo_granules.ipynb sections 1-4 locally and copy "
                f"output/a3e/ here before submitting the array -- this script builds nothing, it "
                f"only applies what that notebook decided. See README.md, runbook step 8.")

    patch = pd.read_parquet(patch_path)
    scope = pd.read_csv(scope_path).set_index("sample")
    return patch, int(scope.loc[sample, "n_transcripts"])


def main(task_id):
    sample = tasks()[task_id]
    out_dir = C.pseudo_detect_dir(sample)
    out_dir.mkdir(parents=True, exist_ok=True)

    spheres_out = out_dir / "spheres.parquet"
    dict_out = out_dir / "sphere_dict.parquet"
    if spheres_out.exists() and dict_out.exists():
        print(f"[{sample}] already done -- skipping", flush=True)
        return

    t0 = time.time()
    patch, expect_rows = load_patch(sample)
    transcripts = A3.load_transcripts(sample)
    print(f"[{sample}] {len(transcripts):,} transcripts; patch rewrites {len(patch):,} "
          f"({len(patch) / len(transcripts):.3%})", flush=True)

    A3.apply_relabel_patch(transcripts, patch, sample=sample, expect_rows=expect_rows)

    genes = list(C.SYN_GENES)
    nc_genes = A3.load_nc_genes(sample)          # the 19-gene published list, per PSEUDO_NC_LIST
    print(f"[{sample}] seeding on {len(genes)} markers, NC filtering on {len(nc_genes)} genes",
          flush=True)

    mc = mcDETECT(transcripts=transcripts, gnl_genes=genes, nc_genes=nc_genes,
                  **C.DETECT_KWARGS_ROUGH)

    print(f"[{sample}] dbscan (filters off)...", flush=True)
    rough_dict = mc.dbscan()
    flat = flatten_sphere_dict(rough_dict, genes)
    A3.write_parquet_atomic(flat, dict_out)
    print(f"[{sample}] pre-merge spheres (raw): {len(flat):,}", flush=True)

    funnel = A3.funnel_counts(flat.rename(columns={"size_premerge": "size"}), by="seed_gene",
                              genes=genes)
    funnel.insert(0, "sample", sample)
    funnel.to_csv(out_dir / "funnel_by_gene.csv", index=False)

    fine_dict = apply_fine_filters(rough_dict, C.DETECT_KWARGS_FINE["size_thr"],
                                   C.DETECT_KWARGS_FINE["in_soma_thr"])
    print(f"[{sample}] after size + in-soma filters: {sum(len(v) for v in fine_dict.values()):,}",
          flush=True)

    print(f"[{sample}] merging...", flush=True)
    merged = mc.merge_sphere(fine_dict)
    print(f"[{sample}] merged: {len(merged):,}; negative-control filtering...", flush=True)
    spheres = mc.nc_filter(merged)
    A3.write_parquet_atomic(spheres.reset_index(drop=True), spheres_out)

    A3.write_run_info(out_dir, sample=sample, n_genes=len(genes), genes=";".join(genes),
                      n_nc_genes=len(nc_genes), nc_list=C.PSEUDO_NC_LIST,
                      n_transcripts=len(transcripts), n_relabelled=len(patch),
                      n_spheres_premerge=len(flat), n_spheres_merged=len(merged),
                      n_spheres_final=len(spheres),
                      minutes=round((time.time() - t0) / 60, 1),
                      **{k: v for k, v in C.DETECT_KWARGS_FINE.items()})
    print(f"[{sample}] done: {len(spheres):,} spheres in {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        for i, smp in enumerate(tasks()):
            print(i, smp)
        raise SystemExit(0)
    task_id = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= task_id < len(tasks()):
        raise SystemExit(f"task_id {task_id} out of range (0..{len(tasks()) - 1})")
    main(task_id)
