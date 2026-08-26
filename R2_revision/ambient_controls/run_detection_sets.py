#!/usr/bin/env python3
"""
A3, stage 1 -- the detection runs that Sets 0 / 1 / 3 need.

Reviewer #2, major point 9: the ambient background is modelled as complete spatial randomness,
so a threshold derived from it *"could under-correct in such regions and inflate granule calls
locally"*, and the round-1 density regression does not test that because it *"operates on
granules that have already been called"*. Answering at the detection step needs detections that
do not exist on disk.

One SLURM array task = one (set x sample) unit:

    set0        ~20 panel genes with no synaptic / nuclear annotation, ABUNDANCE-MATCHED to the
                20 markers. The control that actually addresses structured ambient: Set 3 alone
                only shows "nuclear genes stay in nuclei", which is circular, because the NC
                genes are DEFINED as nuclear-enriched and are then filtered on nuclear overlap.
                Set 0 shows that arbitrary genes AT MARKER ABUNDANCE do not form granule-like
                aggregates, and it neutralises the 15x abundance gap (markers median 1,153,633
                transcripts vs NC 74,893) that would otherwise explain an empty Set 3 for free.
    set1        the 20 markers with the size + in-soma filters but NO NC filter.
    set3        seeded directly on the 18-gene NC list (Gria2 dropped).

    set2        NOT produced here -- it is the published output/<dataset>/granules.parquet,
                reused exactly as it is.

3 sets x 2 samples = 6 array tasks. Indices 0-3 are the two cheap sets, 4-5 are set1.

There is no set3prime (all-19) arm. The NC policy is one rule by provenance -- new detections use
the 18-gene list, reused published data keeps the 19-gene list it was built with -- and
set3prime - set3 is exactly the Gria2-seeded spheres, which set1 already contains. See
a3_config.SET3_EXCLUDE for the full rationale and the size of the resulting gap (<0.4%).

Set 1 is a TRUE re-detection, not a post-hoc filter of all_granules.parquet. mcDETECT applies the
size and in-soma filters at the end of dbscan(), i.e. BEFORE merge_sphere(), so filtering the
rough pass afterwards does not reproduce Set 2's construction (737,063 vs 681,337 spheres in WT)
and the Set1-vs-Set2 comparison would be measuring the filter ORDER rather than the NC filter.

Two things every run persists that the published pipeline does not
------------------------------------------------------------------
`merge_sphere()` is many-to-one, and `_remove_overlaps` updates only sphere_x/y/z, layer_z and
sphere_r -- `size`, `comp`, `gene` and `in_soma_ratio` are all stale afterwards. So:

  * granules.parquet["gene"] is NOT reliably the seed gene, which A3b needs for its
    seed-gene-matched detection predicate ("any of 20 markers" is ~20x easier than "3 Camk2a");
  * `size` is NOT k_g, the own-gene member count that A3a stage D must subtract from the local
    background (otherwise each granule inflates its own background and the test is self-defeating).

Both come from the PRE-MERGE per-gene sphere_dict, so this script writes it alongside the merged
table as sphere_dict.parquet.

On min_samples
--------------
DETECT_KWARGS_FINE keeps minspl=3 verbatim from code/3_detection.py:83, so Sets 0/1/3 are built
exactly the way Set 2 was -- which is what makes them comparable to it.

Nothing here runs at alpha = 0.5. That value only appears in the preflight CSR table, where it is
an arithmetic identity showing poisson_select would have returned 3 for every marker anyway; since
DBSCAN is deterministic given min_samples, the two settings are provably the same run, so no
re-detection is needed to establish it.

Usage
-----
    python3 run_detection_sets.py [task_id]        # defaults to $SLURM_ARRAY_TASK_ID
    python3 run_detection_sets.py --list           # print the task table and exit
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3_config as C
import a3_common as A3

from mcDETECT.model import mcDETECT


def tasks():
    """Array index -> (set_name, sample). One place, so the SLURM array size is derivable."""
    return [(s, smp) for s in C.SETS_TO_DETECT for smp in C.SAMPLES]


def gene_set_for(set_name, sample, transcripts=None):
    """The genes each set seeds detection on.

    set0 is data-dependent (abundance matching), so it is resolved here and the resulting list is
    persisted to preflight/set0_genes.csv -- the choice must be auditable and stable across
    reruns, and both samples must use the SAME list or the WT/AD comparison is not a comparison.
    """
    if set_name == "set1":
        return list(C.SYN_GENES)
    if set_name == "set3":
        # 18 genes: this is a NEW detection, so the Gria2 collision is corrected (a3_config policy)
        return A3.load_nc_genes(sample, exclude=C.SET3_EXCLUDE)
    if set_name == "set0":
        path = C.PREFLIGHT_DIR / "set0_genes.csv"
        if not path.exists():
            # Deliberately fail rather than generate it here: tasks 0 and 1 run concurrently and
            # would race on the same file, and the AD task would have to load the WT transcript
            # table to select. A3a section 1 writes it -- that is why it must run first.
            raise SystemExit(
                f"{path} is missing. Run A3a_three_sets.ipynb section 1 (RUN_PREFLIGHT) before "
                "submitting the array -- it selects Set 0's abundance-matched genes on WT and "
                "the same list is reused for AD.")
        sel = pd.read_csv(path)
        return sel["set0_gene"].dropna().unique().tolist()
    raise ValueError(f"unknown set: {set_name}")


def flatten_sphere_dict(sphere_dict, genes):
    """Per-gene pre-merge spheres -> one frame carrying the seed gene.

    Written with the size and in-soma filters OFF (see `main`), so this is the genuinely RAW
    per-gene output and the funnel's three stages are distinguishable.

    `size_premerge` is dbscan's `size` before merging: (own-gene cluster members) + (other
    granule-marker transcripts in the ball), per model.py:258-267. It is NOT k_g, the own-gene
    count -- A3a recomputes that by ball query on the final geometry.
    """
    frames = []
    for idx, df in sphere_dict.items():
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        d["seed_gene"] = genes[idx] if idx < len(genes) else str(idx)
        d["gene_index"] = idx
        d = d.rename(columns={"size": "size_premerge", "comp": "comp_premerge"})
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=["sphere_x", "sphere_y", "sphere_z", "layer_z", "sphere_r",
                                     "size_premerge", "comp_premerge", "in_soma_ratio", "gene",
                                     "seed_gene", "gene_index"])
    return pd.concat(frames, ignore_index=True)


def apply_fine_filters(sphere_dict, size_thr, in_soma_thr):
    """Apply mcDETECT's own per-gene filters to a rough sphere_dict.

    This is the identity that makes the one-pass design work. model.py:288 ends dbscan() with

        sphere = sphere[(sphere.sphere_r < size_thr) & (sphere.in_soma_ratio < in_soma_thr)]

    i.e. a row-wise predicate applied BEFORE merge_sphere(). Re-applying exactly that predicate
    here hands merge_sphere() the same input the fine pass would have produced, while the
    unfiltered dict remains available for the funnel's `raw` stage.

    Verified on a real 1.3 K-transcript crop: same sphere count, same genes, values equal to
    within 1e-12. NOT bitwise -- `miniball.get_bounding_ball` is randomised, so two runs of
    mcDETECT's OWN fine pass differ by ~9e-13 too. Any reproduction check needs a tolerance,
    not `.equals()`.
    """
    return {k: v[(v["sphere_r"] < size_thr) & (v["in_soma_ratio"] < in_soma_thr)]
                 .reset_index(drop=True)
            for k, v in sphere_dict.items()}


def main(task_id):
    set_name, sample = tasks()[task_id]
    tag = f"{set_name} {sample}"
    out_dir = C.detect_dir(set_name, sample)
    out_dir.mkdir(parents=True, exist_ok=True)

    spheres_out = C.spheres_path(set_name, sample)
    dict_out = C.sphere_dict_path(set_name, sample)
    if spheres_out.exists() and dict_out.exists():
        print(f"[{tag}] already done -- skipping", flush=True)
        return

    t0 = time.time()
    transcripts = A3.load_transcripts(sample)
    genes = gene_set_for(set_name, sample, transcripts=transcripts)

    # A gene with zero transcripts in THIS sample makes DBSCAN.fit raise on an empty array
    # (model.py:210). Set 0's list is chosen on WT and reused for AD on purpose, so this is a
    # live risk; drop such genes and record it rather than failing the task.
    present = transcripts["target"].value_counts()
    missing = [g for g in genes if int(present.get(g, 0)) == 0]
    if missing:
        genes = [g for g in genes if g not in set(missing)]
        A3.write_status(out_dir, [dict(set=set_name, sample=sample, dropped_gene=g,
                                       reason="zero transcripts in this sample")
                                  for g in missing])
        print(f"[{tag}] dropped {len(missing)} zero-count genes: {missing}", flush=True)
    if not genes:
        raise SystemExit(f"[{tag}] no seed genes with transcripts -- nothing to detect")
    print(f"[{tag}] seeding on {len(genes)} genes: {genes}", flush=True)

    # DBSCAN with the two per-gene filters OFF, so the funnel has a real `raw` stage. The fine
    # filters are then applied per gene below -- provably the same input to merge_sphere() as the
    # fine pass, because model.py:288 applies exactly that row-wise predicate before merging.
    mc = mcDETECT(transcripts=transcripts, gnl_genes=genes, nc_genes=None,
                  **C.DETECT_KWARGS_ROUGH)

    print(f"[{tag}] dbscan (filters off)...", flush=True)
    rough_dict = mc.dbscan()
    flat = flatten_sphere_dict(rough_dict, genes)
    A3.write_parquet_atomic(flat, dict_out)
    print(f"[{tag}] pre-merge spheres (raw): {len(flat):,}", flush=True)

    # The funnel, from the genuinely unfiltered per-gene output.
    funnel = A3.funnel_counts(flat.rename(columns={"size_premerge": "size"}), by="seed_gene")
    funnel.insert(0, "sample", sample)
    funnel.insert(0, "set", set_name)
    funnel.to_csv(out_dir / "funnel_by_gene.csv", index=False)
    if len(funnel) and (funnel["raw"] == funnel["in_soma"]).all():
        print(f"[{tag}] WARNING: funnel stages are identical -- the filters did not bite",
              flush=True)

    fine_dict = apply_fine_filters(rough_dict, C.DETECT_KWARGS_FINE["size_thr"],
                                   C.DETECT_KWARGS_FINE["in_soma_thr"])
    n_fine = sum(len(v) for v in fine_dict.values())
    print(f"[{tag}] after size + in-soma filters: {n_fine:,}", flush=True)

    print(f"[{tag}] merging...", flush=True)
    # merge_sphere reads self.gnl_genes/_find_points off `mc`, whose thresholds are the rough
    # ones -- irrelevant here, since merging never re-applies the size/in-soma predicate.
    spheres = mc.merge_sphere(fine_dict)
    A3.write_parquet_atomic(spheres.reset_index(drop=True), spheres_out)

    A3.write_run_info(out_dir, set=set_name, sample=sample, n_genes=len(genes),
                      genes=";".join(genes), n_spheres_premerge=len(flat),
                      n_spheres_merged=len(spheres),
                      minutes=round((time.time() - t0) / 60, 1),
                      **{k: v for k, v in C.DETECT_KWARGS_FINE.items()})
    print(f"[{tag}] done: {len(spheres):,} spheres in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        for i, (s, smp) in enumerate(tasks()):
            print(i, s, smp)
        raise SystemExit(0)
    if len(sys.argv) > 1:
        task_id = int(sys.argv[1])
    else:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    n = len(tasks())
    if not 0 <= task_id < n:
        raise SystemExit(f"task_id {task_id} out of range (0..{n - 1})")
    main(task_id)
