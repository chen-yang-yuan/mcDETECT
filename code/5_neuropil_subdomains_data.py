import anndata
import numpy as np
import pandas as pd
import scanpy as sc

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

# File paths
comparison_path = f"../output/MERSCOPE_WT_AD_comparison/"

# ==================== Helper functions ==================== #

def subdivide_spots(spots_raw: "sc.AnnData", sub_um: float = 25.0) -> "sc.AnnData":
    
    """Split each spot into four sub-tiles (half parent edge length `sub_um`); X is all zeros."""
    
    half_sub = float(sub_um) / 2.0
    offsets = np.array([[-half_sub, -half_sub],
                        [half_sub, -half_sub],
                        [-half_sub, half_sub],
                        [half_sub, half_sub]])
    n = spots_raw.n_obs
    gx = spots_raw.obs["global_x"].to_numpy(dtype=float)
    gy = spots_raw.obs["global_y"].to_numpy(dtype=float)
    new_gx = np.column_stack([gx + offsets[i, 0] for i in range(4)]).ravel()
    new_gy = np.column_stack([gy + offsets[i, 1] for i in range(4)]).ravel()
    ix = np.arange(n).repeat(4)
    inherited = ["region_labels", "brain_area", "batch"]
    obs_new = spots_raw.obs.iloc[ix][inherited].reset_index(drop=True)
    obs_new["global_x"] = new_gx
    obs_new["global_y"] = new_gy
    parent_ids = spots_raw.obs["spot_id"].astype(str).to_numpy()[ix]
    quad = np.tile(np.arange(4), n)
    obs_new["spot_id"] = parent_ids + "_25um_q" + quad.astype(str)
    out = anndata.AnnData(X=np.zeros((obs_new.shape[0], spots_raw.shape[1])), obs=obs_new, var=spots_raw.var.copy())
    if spots_raw.uns:
        out.uns = dict(spots_raw.uns)
    return out

def fill_spot_expression(spots: "sc.AnnData", transcripts: pd.DataFrame, grid_len: float = 25.0, gene_col: str = "target", spots_x_col: str = "global_x", spots_y_col: str = "global_y", transcripts_x_col = "global_x", transcripts_y_col = "global_y", batch_col: str = "batch", inplace: bool = True, assign_to: str = "X", tx_chunk: int = 65_536, spot_chunk: int = 512) -> "sc.AnnData":
    
    """Assign each transcript to spot tiles by exact axis-aligned containment (half-open intervals).

    Tile for spot center (cx, cy) is [cx - h, cx + h) x [cy - h, cy + h) with h = grid_len / 2, matching
    subdivide_spots. Counts are per (gene, spot); batch must match between transcript and spot rows.

    This avoids shared histogram bins: each spot gets only transcripts inside its own square. If two tiles
    overlap, a transcript in the overlap is counted toward every spot whose tile contains it.

    Parameters
    ----------
    assign_to : str
        "X" to write ``spots.X``, or a layer name to write ``spots.layers[assign_to]``.
    tx_chunk, spot_chunk : int
        Chunk sizes to bound peak memory for the boolean overlap array (spots_chunk x transcripts_chunk).
    """
    
    if not inplace:
        spots = spots.copy()
    half = float(grid_len) / 2.0
    genes = list(spots.var_names)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    cx = spots.obs[spots_x_col].to_numpy(dtype=float)
    cy = spots.obs[spots_y_col].to_numpy(dtype=float)
    bsp = spots.obs[batch_col].astype(str).to_numpy()
    n_obs, n_vars = spots.n_obs, spots.n_vars
    X = np.zeros((n_obs, n_vars), dtype=np.float64)
    batches = transcripts[batch_col].unique()

    for idx, g in enumerate(genes):
        gi = gene_to_idx[g]
        tg = transcripts[transcripts[gene_col] == g]
        if len(tg) == 0:
            continue
        for b in batches:
            bs = str(b)
            t_b = tg[tg[batch_col].astype(str) == bs]
            if len(t_b) == 0:
                continue
            rows = np.flatnonzero(bsp == bs)
            if rows.size == 0:
                continue
            tx_all = t_b[transcripts_x_col].to_numpy(dtype=float)
            ty_all = t_b[transcripts_y_col].to_numpy(dtype=float)
            cx_b = cx[rows]
            cy_b = cy[rows]
            counts = np.zeros(rows.size, dtype=np.float64)
            for r0 in range(0, rows.size, spot_chunk):
                r1 = min(r0 + spot_chunk, rows.size)
                cx_c = cx_b[r0:r1, None]
                cy_c = cy_b[r0:r1, None]
                acc = np.zeros(r1 - r0, dtype=np.float64)
                for t0 in range(0, tx_all.shape[0], tx_chunk):
                    t1 = min(t0 + tx_chunk, tx_all.shape[0])
                    xc = tx_all[t0:t1][None, :]
                    yc = ty_all[t0:t1][None, :]
                    acc += np.sum(
                        (xc >= cx_c - half)
                        & (xc < cx_c + half)
                        & (yc >= cy_c - half)
                        & (yc < cy_c + half),
                        axis=1,
                    ).astype(np.float64)
                counts[r0:r1] = acc
            X[rows, gi] = counts
        if idx % 10 == 0:
            print(f"Processed {idx} out of {len(genes)} genes!")

    if assign_to == "X":
        spots.X = X
    else:
        spots.layers[assign_to] = X
    return spots

# ==================== Read data ==================== #

# -------------------- Spots -------------------- #
spots_WT = sc.read_h5ad(f"../data/MERSCOPE_WT_1/processed_data/spots.h5ad")
spots_AD = sc.read_h5ad(f"../data/MERSCOPE_AD_1/processed_data/spots.h5ad")

# Adjust coordinates
spots_WT.obs["global_x"] = spots_WT.obs["global_y_new"].copy()
spots_WT.obs["global_y"] = spots_WT.obs["global_x_new"].copy()

spots_AD.obs["global_x"] = spots_AD.obs["global_x_new"].copy()
spots_AD.obs["global_y"] = spots_AD.obs["global_y_new"].copy()
spots_AD.obs["global_x"] += 12000
spots_AD.obs["global_y"] += 7200

# Merge spots
spots_dict = {"MERSCOPE_WT_1": spots_WT, "MERSCOPE_AD_1": spots_AD}
spots_raw = anndata.concat(spots_dict, axis = 0, merge = "same", label = "batch")

del spots_WT
del spots_AD

# -------------------- Cells -------------------- #
adata_1 = sc.read_h5ad(f"../data/MERSCOPE_WT_1/processed_data/adata.h5ad")
adata_2 = sc.read_h5ad(f"../data/MERSCOPE_AD_1/processed_data/adata.h5ad")

# Adjust coordinates
adata_1.obs["global_x"] = adata_1.obs["global_y_new"].copy()
adata_1.obs["global_y"] = adata_1.obs["global_x_new"].copy()

adata_2.obs["global_x"] = adata_2.obs["global_x_new"].copy()
adata_2.obs["global_y"] = adata_2.obs["global_y_new"].copy()
adata_2.obs["global_x"] += 12000
adata_2.obs["global_y"] += 7200

# Merge cells
adata_dict = {"MERSCOPE_WT_1": adata_1, "MERSCOPE_AD_1": adata_2}
adata = anndata.concat(adata_dict, axis = 0, merge = "same", label = "batch")
adata.write_h5ad(comparison_path + "neuropil_subdomains_adata.h5ad")

# -------------------- Transcripts -------------------- #
theta_WT = 10 * np.pi / 180
theta_AD = 170 * np.pi / 180

transcripts_WT = pd.read_parquet(f"../data/MERSCOPE_WT_1/processed_data/transcripts.parquet")
rotation_matrix = np.array([[np.cos(theta_WT), np.sin(theta_WT)], [-np.sin(theta_WT), np.cos(theta_WT)]])
coords = transcripts_WT[["global_y", "global_x"]].to_numpy()
transformed_coords = coords @ rotation_matrix.T
transcripts_WT["global_y_new"] = transformed_coords[:, 0]
transcripts_WT["global_x_new"] = transformed_coords[:, 1]
transcripts_WT["global_y_new"] = 6250 - transcripts_WT["global_y_new"]

transcripts_WT["global_x"] = transcripts_WT["global_y_new"].copy()
transcripts_WT["global_y"] = transcripts_WT["global_x_new"].copy()

del transcripts_WT["global_x_new"]
del transcripts_WT["global_y_new"]

transcripts_AD = pd.read_parquet(f"../data/MERSCOPE_AD_1/processed_data/transcripts.parquet")
rotation_matrix = np.array([[np.cos(theta_AD), np.sin(theta_AD)], [-np.sin(theta_AD), np.cos(theta_AD)]])
coords = transcripts_AD[["global_x", "global_y"]].to_numpy()
transformed_coords = coords @ rotation_matrix.T
transcripts_AD["global_x_new"] = transformed_coords[:, 0]
transcripts_AD["global_y_new"] = transformed_coords[:, 1]

transcripts_AD["global_x"] = transcripts_AD["global_x_new"].copy()
transcripts_AD["global_y"] = transcripts_AD["global_y_new"].copy()
transcripts_AD["global_x"] += 12000
transcripts_AD["global_y"] += 7200

del transcripts_AD["global_x_new"]
del transcripts_AD["global_y_new"]

# Merge transcripts
transcripts_WT["batch"] = "MERSCOPE_WT_1"
transcripts_AD["batch"] = "MERSCOPE_AD_1"
transcripts = pd.concat([transcripts_WT, transcripts_AD], axis = 0)
transcripts.to_parquet(comparison_path + "neuropil_subdomains_transcripts.parquet")

del transcripts_WT
del transcripts_AD

# -------------------- Granules -------------------- #
granule_adata = sc.read_h5ad(comparison_path + "granule_adata_tsne.h5ad")

# Adjust coordinates
granule_adata.obs["global_x"] = granule_adata.obs["global_x_adjusted"].copy()
granule_adata.obs["global_y"] = granule_adata.obs["global_y_adjusted"].copy()
mask = granule_adata.obs["batch"] == "MERSCOPE_WT_1"
granule_adata.obs.loc[mask, "global_x"] = (6250 - granule_adata.obs.loc[mask, "global_x"])

# Read granule subtype labels
granule_subtype_df = pd.read_parquet(comparison_path + "granule_subtype_labels_granule_adata_tsne.parquet")
cols_keep = ["sample", "granule_id", "granule_subtype_kmeans", "granule_subtype_manual", "granule_subtype_manual_simple"]
granule_subtype_df = granule_subtype_df[cols_keep].drop_duplicates(["sample", "granule_id"])

# Merge granule subtype labels
granule_adata.obs = granule_adata.obs.reset_index(names="obs_name")
granule_adata.obs = granule_adata.obs.merge(granule_subtype_df,
                                            left_on=["batch", "granule_id"],
                                            right_on=["sample", "granule_id"],
                                            how="left",
                                            validate="one_to_one").set_index("obs_name")
if "sample" in granule_adata.obs.columns:
    granule_adata.obs = granule_adata.obs.drop(columns=["sample"])

# Convert granule subtype labels to category
granule_adata.obs["granule_subtype_kmeans"] = granule_adata.obs["granule_subtype_kmeans"].astype("category")
granule_adata.obs["granule_subtype_manual"] = granule_adata.obs["granule_subtype_manual"].astype("category")
granule_adata.obs["granule_subtype"] = granule_adata.obs["granule_subtype_manual_simple"].astype("category")

# Save results
granule_adata.write_h5ad(comparison_path + "neuropil_subdomains_granule_adata.h5ad")

# ==================== Enhance spatial resolution ==================== #

# All transcripts
spots = subdivide_spots(spots_raw)
spots = fill_spot_expression(spots, transcripts, grid_len=25.0, assign_to="X")

# Extrasomatic transcripts
transcripts_extrasomatic = transcripts[transcripts["overlaps_nucleus"] == 0]
spots = fill_spot_expression(spots, transcripts_extrasomatic, grid_len=25.0, assign_to="extrasomatic_transcripts")

# Intrasomatic transcripts
transcripts_intrasomatic = transcripts[transcripts["overlaps_nucleus"] == 1]
spots = fill_spot_expression(spots, transcripts_intrasomatic, grid_len=25.0, assign_to="intrasomatic_transcripts")

# Save results
spots.write_h5ad(comparison_path + "neuropil_subdomains_spots.h5ad")