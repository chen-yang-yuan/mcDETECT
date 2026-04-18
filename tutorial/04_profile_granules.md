# 4. Step C — Profile granules (`profile`)

[← Tutorial index](./README.md)

**Step C1.** Ensure the same **`mc`** instance still holds **`transcripts`** used for detection.

**Step C2.** Choose the gene list **`genes`** (full panel or a subset).

**Step C3.** Call:

```python
granule_adata = mc.profile(
    granules,
    genes=genes,
    buffer=0.0,
    print_itr=False,
    print_itr_interval=5000,
)
```

| Argument | Meaning |
|----------|---------|
| `granules` | `DataFrame` from `detect()` (or merged rough table). |
| `genes` | Genes to count; **`None`** uses all genes present in `transcripts`. |
| `buffer` | Added to each sphere’s radius when querying transcripts. |

**Output:** **`anndata.AnnData`** — sparse **`X`**: `n_granules × n_genes`; **`obs`**: granule metadata with coordinates renamed to `global_x` / `global_y` / `global_z` and **`granule_id`** added.

**Step C4.** Typical Scanpy QC for visualization (as in `3_detection.py`): copy counts to a layer, `normalize_total`, `log1p`, `pca`, `tsne`.

**Next:** [Step D — Manual granule subtyping](./05_manual_subtyping.md)
