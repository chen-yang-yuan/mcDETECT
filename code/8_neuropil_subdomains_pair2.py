"""
Rough neuropil-subdomain analysis for the SECOND AD-WT MERSCOPE pair.

Pipeline (mirrors code/benchmark/benchmark_subtyping.ipynb + code/7_neuropil_subdomains.ipynb,
but self-contained and only using the rough granule adata for the second sample pair):

1. Concatenate the two granule adata (output/MERSCOPE_AD_2 and output/MERSCOPE_WT_2).
   Coordinates are NOT aligned across samples; each sample keeps its own coordinate frame
   and is gridded independently so granules from the two samples are never mixed in a spot.
2. Manual granule subtyping (k-means on marker genes -> heatmap -> manual cluster->subtype
   mapping). Only the manual route is implemented; MANUAL_SUBTYPE_MAPPING is a PLACEHOLDER
   to be filled after inspecting heatmap_subtype.jpeg.
3. Build four cortical (Isocortex) neuropil subdomains from the granule-subtype composition
   of 50x50 um spots (spot_embedding on granule_subtype_kmeans -> row-normalize -> KMeans, k=4).
4. Deliverable: for each neuropil subdomain, differentially expressed genes (DEGs) of the
   spot-level GRANULE expression comparing AD vs WT (Wilcoxon).

Enrichment analysis is intentionally left out (see note in step 4).

Run with the mcDETECT-env interpreter, e.g.:
    /Users/chenyang/miniconda3/envs/mcDETECT-env/bin/python code/8_neuropil_subdomains_pair2.py
"""

import os
import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.sparse import csr_matrix, issparse
from sklearn.cluster import KMeans, MiniBatchKMeans

from mcDETECT.downstream import spot_embedding

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0


# ==================================================================== #
#                               Config                                  #
# ==================================================================== #

# Repo root (this file lives in code/)
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

AD_PATH = os.path.join(BASE, "output/MERSCOPE_AD_2/granule_adata_raw.h5ad")
WT_PATH = os.path.join(BASE, "output/MERSCOPE_WT_2/granule_adata_raw.h5ad")

ROI = "Isocortex"          # cortical neuropil subdomains are built within the isocortex
GRID_LEN = 50              # spot size in microns (50 x 50)
N_CLUSTERS = 15            # k-means clusters for manual subtyping (as in benchmark_subtyping)
N_SUBDOMAINS = 4           # four cortical neuropil subdomains
SEED = 42

# k-means clusters that do not correspond to any granule subtype and are dropped from ALL
# downstream analysis (subtyping annotation, subdomain construction, DEGs).
DROP_CLUSTERS = ["9"]

OUTPUT_PATH = os.path.join(BASE, f"output/MERSCOPE_WT_AD_2_comparison/neuropil_subdomains_{ROI}_{GRID_LEN}/")
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Marker genes (from benchmark_subtyping.ipynb). Used for k-means subtyping + heatmap.
marker_genes = {
    "pre-syn": ["Bsn", "Gap43", "Nrxn1", "Slc17a6", "Slc17a7", "Slc32a1", "Snap25", "Stx1a", "Syn1", "Syp", "Syt1", "Vamp2", "Cplx2"],
    "post-syn": ["Camk2a", "Dlg3", "Dlg4", "Gphn", "Gria1", "Gria2", "Homer1", "Homer2", "Nlgn1", "Nlgn2", "Nlgn3", "Shank1", "Shank3"],
    "axons": ["Ank3", "Nav1", "Sptnb4", "Nfasc", "Mapt", "Tubb3"],
    "dendrites": ["Actb", "Cyfip2", "Ddn", "Dlg4", "Map1a", "Map2"],
}
ref_genes = ["Bsn", "Gap43", "Nrxn1", "Slc17a6", "Slc17a7", "Slc32a1", "Stx1a", "Syn1", "Syp", "Syt1", "Vamp2", "Cplx2",
             "Camk2a", "Dlg3", "Dlg4", "Gphn", "Gria1", "Gria2", "Homer1", "Homer2", "Nlgn1", "Nlgn2", "Nlgn3", "Shank1", "Shank3",
             "Cyfip2", "Ddn", "Map1a", "Map2", "Ank3", "Nav1", "Nfasc", "Mapt", "Tubb3"]

# ------------------------------------------------------------------ #
# PLACEHOLDER: manual cluster -> subtype mapping.
# After running once, open OUTPUT_PATH/heatmap_subtype.jpeg, read off which of the
# N_CLUSTERS k-means clusters (labelled "0".."14") look pre-syn / post-syn / dendrites /
# axons / mixed, and fill in the lists below. Combination keys with " & " are treated as
# "mixed" in *_manual_simple. Clusters left unlisted stay unannotated.
# This mapping only drives the manual-subtype annotation + heatmap coloring; the subdomain
# construction and the AD-vs-WT DEG deliverable do NOT depend on it, so the script runs
# end-to-end even while the mapping is still empty.
# ------------------------------------------------------------------ #
MANUAL_SUBTYPE_MAPPING = {
    "pre-syn": ["1", "2", "3", "4", "7", "8"],
    "post-syn": ["0", "5"],
    "dendrites": ["6", "13"],
    "axons": [],
    "pre & post": [],
    "pre & den": [],
    "post & den": ["11", "14"],
    "pre & post & den": ["10", "12"],
    "others": [],
}


# ==================================================================== #
#                              Helpers                                  #
# ==================================================================== #

def apply_manual_annotation(granule, mapping, cluster_col="granule_subtype_kmeans"):
    """Map k-means cluster ids -> manual subtype (fine) + _simple (mixed for combinations)."""
    k2sub = {}
    for subtype, clusters in mapping.items():
        for c in clusters:
            k2sub[str(c)] = subtype
    fine = granule.obs[cluster_col].astype(str).map(k2sub)
    granule.obs["granule_subtype_manual"] = fine
    granule.obs["granule_subtype_manual_simple"] = fine.apply(
        lambda s: "mixed" if (pd.notna(s) and " & " in str(s)) else (str(s) if pd.notna(s) else np.nan)
    )
    return granule


def cluster_to_simple_category(mapping):
    """cluster id (str) -> simple subtype category (for heatmap column coloring)."""
    c2cat = {}
    for subtype, clusters in mapping.items():
        simple = "mixed" if (pd.notna(subtype) and " & " in str(subtype)) else str(subtype)
        for c in clusters:
            c2cat[str(c)] = simple
    return c2cat


def build_spot_grid(obs, grid_len, x_col="global_x", y_col="global_y"):
    """Occupied-cell grid: one spot per 50x50 tile that contains >=1 granule.

    Centroid of tile (ix, iy) is ((ix+0.5)*grid_len, (iy+0.5)*grid_len), so spot_embedding's
    half-open box [cx-h, cx+h) exactly matches the tile [ix*grid_len, (ix+1)*grid_len).
    """
    gx = obs[x_col].to_numpy(dtype=float)
    gy = obs[y_col].to_numpy(dtype=float)
    ix = np.floor(gx / grid_len).astype(int)
    iy = np.floor(gy / grid_len).astype(int)
    cells = np.unique(np.stack([ix, iy], axis=1), axis=0)
    cx = (cells[:, 0] + 0.5) * grid_len
    cy = (cells[:, 1] + 0.5) * grid_len
    spots = anndata.AnnData(X=np.zeros((len(cx), 1), dtype=np.float32))
    spots.obs["global_x"] = cx
    spots.obs["global_y"] = cy
    spots.obs["spot_id"] = [f"spot_{i}" for i in range(len(cx))]
    return spots


def styled_scatter(obs_df, color_series, path, title=""):
    """Simple spatial scatter of spots colored by a categorical/continuous series."""
    fig, ax = plt.subplots(figsize=(8, 8))
    if color_series.dtype.name in ("category", "object"):
        cats = pd.Categorical(color_series)
        palette = sns.color_palette("Set2", n_colors=len(cats.categories))
        for cat, col in zip(cats.categories, palette):
            m = color_series == cat
            ax.scatter(obs_df.loc[m, "global_x"], obs_df.loc[m, "global_y"], s=6, color=col, label=str(cat))
        ax.legend(markerscale=2, frameon=False, fontsize=8)
    else:
        sca = ax.scatter(obs_df["global_x"], obs_df["global_y"], s=6, c=color_series, cmap="Reds")
        fig.colorbar(sca, ax=ax, shrink=0.6)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ==================================================================== #
#              1. Load and concatenate the two granule adata            #
# ==================================================================== #

print("==================== 1. Load & concatenate ====================")
ad = sc.read_h5ad(AD_PATH)
wt = sc.read_h5ad(WT_PATH)
ad.obs["batch"] = "MERSCOPE_AD_2"
ad.obs["sample"] = "AD"
wt.obs["batch"] = "MERSCOPE_WT_2"
wt.obs["sample"] = "WT"

# Same 290-gene panel; concat on shared genes. index_unique avoids obs_name collisions.
granule = anndata.concat([wt, ad], join="inner", index_unique="-")
granule.obs["sample"] = granule.obs["sample"].astype(str)
granule.obs["batch"] = granule.obs["batch"].astype(str)

# Preserve raw counts for spot-level expression aggregation.
granule.layers["counts"] = csr_matrix(granule.X.copy())
print(f"Concatenated granule adata: {granule.shape}")
print(granule.obs["sample"].value_counts())


# ==================================================================== #
#         2. Manual granule subtyping (k-means -> heatmap -> map)       #
# ==================================================================== #

print("\n==================== 2. Manual subtyping ====================")
ref_genes_present = [g for g in ref_genes if g in granule.var_names]
sub = granule[:, ref_genes_present].copy()
sc.pp.normalize_total(sub, target_sum=1e4)
sc.pp.log1p(sub)

X_sub = sub.X.toarray() if issparse(sub.X) else np.asarray(sub.X)
np.random.seed(SEED)
km = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=5000, random_state=SEED, n_init=20)
km.fit(X_sub)
labels = km.labels_.astype(str)
cats = [str(i) for i in range(N_CLUSTERS)]
granule.obs["granule_subtype_kmeans"] = pd.Categorical(labels, categories=cats, ordered=True)
sub.obs["granule_subtype_kmeans"] = granule.obs["granule_subtype_kmeans"].values
print(granule.obs["granule_subtype_kmeans"].value_counts().sort_index())

# Heatmap of marker expression per k-means cluster (use to fill MANUAL_SUBTYPE_MAPPING).
sc.pl.heatmap(sub, var_names=ref_genes_present, groupby="granule_subtype_kmeans",
              cmap="Reds", standard_scale="var", dendrogram=False, swap_axes=True,
              show=False, figsize=(10, 6))
plt.gcf().savefig(os.path.join(OUTPUT_PATH, "heatmap_subtype.jpeg"), dpi=500, bbox_inches="tight")
plt.close()
print(f"Saved heatmap_subtype.jpeg to {OUTPUT_PATH}")

# Apply the (placeholder) manual mapping.
granule = apply_manual_annotation(granule, MANUAL_SUBTYPE_MAPPING)
n_annot = granule.obs["granule_subtype_manual"].notna().sum()
print(f"Granules with a manual subtype assigned: {n_annot} / {granule.n_obs} "
      f"(0 is expected until MANUAL_SUBTYPE_MAPPING is filled in)")

# Drop clusters that do not correspond to any subtype (identified from heatmap_subtype.jpeg).
# Removed from ALL downstream analysis: subdomain construction and AD-vs-WT DEGs.
if DROP_CLUSTERS:
    before = granule.n_obs
    granule = granule[~granule.obs["granule_subtype_kmeans"].isin(DROP_CLUSTERS)].copy()
    granule.obs["granule_subtype_kmeans"] = granule.obs["granule_subtype_kmeans"].cat.remove_unused_categories()
    print(f"Dropped clusters {DROP_CLUSTERS}: {before} -> {granule.n_obs} granules")
    print("Remaining clusters:", list(granule.obs["granule_subtype_kmeans"].cat.categories))


# ==================================================================== #
#     3. Build cortical neuropil subdomains from subtype composition    #
# ==================================================================== #

print(f"\n==================== 3. Neuropil subdomains ({ROI}) ====================")
granule_roi = granule[granule.obs["brain_area"] == ROI].copy()
print(f"{ROI} granules: {granule_roi.n_obs}")

# Composition features = remaining k-means clusters (dropped clusters already removed).
subtype_names = list(granule.obs["granule_subtype_kmeans"].cat.categories)
smoothing_radius = np.sqrt(2) * GRID_LEN + 1

emb_list, expr_list, obs_list = [], [], []
for s in ["WT", "AD"]:
    g_s = granule_roi[granule_roi.obs["sample"] == s].copy()
    spots_s = build_spot_grid(g_s.obs, GRID_LEN)
    print(f"  [{s}] granules={g_s.n_obs}, spots={spots_s.n_obs}")

    emb, _, aux, spot_granule_expr, _ = spot_embedding(
        spots=spots_s,
        granule_adata=g_s,
        adata=None,
        count_matrix=None,
        spot_loc_key=("global_x", "global_y"),
        spot_width=GRID_LEN,
        spot_height=GRID_LEN,
        granule_loc_key=("global_x", "global_y"),
        granule_subtype_key="granule_subtype_kmeans",
        subtype_names=subtype_names,
        granule_count_layer="counts",
        include_soma_features=False,
        smoothing=True,
        smoothing_radius=smoothing_radius,
        smoothing_mode="gaussian",
    )

    keep = aux["granule_count"] > 0
    obs_s = spots_s.obs.loc[keep].copy()
    obs_s["sample"] = s
    obs_s["batch"] = "MERSCOPE_AD_2" if s == "AD" else "MERSCOPE_WT_2"
    obs_s["granule_count"] = aux["granule_count"][keep]

    emb_list.append(emb[keep])
    expr_list.append(spot_granule_expr[keep])
    obs_list.append(obs_s)

embeddings = np.vstack(emb_list)
spot_granule_expression = np.vstack(expr_list)
spot_obs = pd.concat(obs_list, ignore_index=True)
print(f"Total spots with granules: {spot_obs.shape[0]}")

# Row-normalize the subtype composition, then cluster into N_SUBDOMAINS.
row_sums = embeddings.sum(axis=1, keepdims=True)
X_prop = np.divide(embeddings, row_sums, out=np.zeros_like(embeddings, dtype=float), where=row_sums > 0)

kmeans = KMeans(n_clusters=N_SUBDOMAINS, random_state=SEED, n_init=20)
sd_labels = kmeans.fit_predict(X_prop)
spot_obs["subdomain"] = pd.Categorical([f"Subdomain {l + 1}" for l in sd_labels],
                                       categories=[f"Subdomain {i + 1}" for i in range(N_SUBDOMAINS)],
                                       ordered=True)
print(spot_obs.groupby(["subdomain", "sample"]).size().unstack(fill_value=0))

# Save spot-level subdomain table.
spot_obs.to_parquet(os.path.join(OUTPUT_PATH, "spot_subdomain_labels.parquet"), index=False)

# Subdomain x subtype composition (mean over spots; X_prop is the normalized composition).
comp_df = pd.DataFrame(X_prop, columns=subtype_names)
comp_df["subdomain"] = spot_obs["subdomain"].values
subdomain_order = [f"Subdomain {i + 1}" for i in range(N_SUBDOMAINS)]
subdomain_means = comp_df.groupby("subdomain").mean().loc[subdomain_order]

# ----- Group granule subtypes (k-means clusters) by their manual category ----- #
# Column order = clusters sorted within category, categories in a fixed order (like the
# original granule_col_order). Requires MANUAL_SUBTYPE_MAPPING to be filled to see real
# grouping; unlisted clusters fall under "unannotated".
c2cat = cluster_to_simple_category(MANUAL_SUBTYPE_MAPPING)
ordered_cats = ["pre-syn", "post-syn", "dendrites", "axons", "mixed", "others", "unannotated"]
col_order = []
for cat in ordered_cats:
    clusters_here = sorted([s for s in subtype_names if c2cat.get(s, "unannotated") == cat], key=lambda x: int(x))
    col_order.extend(clusters_here)

subdomain_means = subdomain_means[col_order]
category_series = pd.Series([c2cat.get(s, "unannotated") for s in col_order], index=col_order, name="Category")
present_cats = [c for c in ordered_cats if c in set(category_series)]
cat_to_color = dict(zip(present_cats, sns.color_palette("Set2", n_colors=len(present_cats))))
col_colors = category_series.map(cat_to_color)


def _style_clustermap(g):
    g.ax_heatmap.set_position([0.1, 0.2, 0.75, 0.7])
    g.cax.set_position([0.88, 0.2, 0.02, 0.5])
    g.ax_col_colors.set_position([0.1, 0.92, 0.75, 0.04])
    g.ax_heatmap.tick_params(axis="both", which="both", length=0)
    g.ax_heatmap.grid(False)
    g.ax_heatmap.set_xlabel("Granule Subtype")
    g.ax_heatmap.set_ylabel("Neuropil Subdomain", labelpad=10)
    g.ax_heatmap.yaxis.set_label_position("left")
    g.ax_heatmap.yaxis.tick_left()
    g.ax_heatmap.set_title(" ")
    g.ax_row_dendrogram.set_visible(False)
    g.ax_col_dendrogram.set_visible(False)
    for spine in g.ax_col_colors.spines.values():
        spine.set_visible(False)
    g.cax.grid(False)
    g.ax_col_colors.set_xticks([])
    g.ax_col_colors.set_yticks([])
    g.ax_col_colors.grid(False)
    g.ax_heatmap.spines["top"].set_visible(False)
    # Legend mapping the top color bar to granule subtype categories.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=cat_to_color[c], label=c) for c in present_cats]
    g.figure.legend(handles=handles, title="Granule subtype", loc="lower left",
                    bbox_to_anchor=(0.1, 1.0), ncol=len(present_cats), frameon=False,
                    fontsize=8, title_fontsize=9)


# ----- Mean composition heatmap (Reds) ----- #
g = sns.clustermap(subdomain_means, cmap="Reds", annot=False, col_colors=col_colors,
                   row_cluster=False, col_cluster=False, linewidths=0.5, linecolor="lightgray",
                   xticklabels=True, yticklabels=True, figsize=(10, 2.5), cbar_pos=(0.02, 0.2, 0.02, 0.5))
_style_clustermap(g)
g.savefig(os.path.join(OUTPUT_PATH, "heatmap_subdomain_composition.jpeg"), dpi=500, bbox_inches="tight")
plt.close()

# ----- Enrichment heatmap (log2 fold-change vs global mean; red/blue) ----- #
# log2FC of each subdomain's mean subtype proportion relative to the global mean proportion.
global_means = comp_df[col_order].mean(axis=0)
eps = 1e-8
log2fc_mat = np.log2((subdomain_means + eps).divide(global_means + eps, axis=1))
vmax = np.nanmax(np.abs(log2fc_mat.values))
vmin = -vmax

g = sns.clustermap(log2fc_mat, cmap="bwr", center=0, vmin=vmin, vmax=vmax, annot=False,
                   col_colors=col_colors, row_cluster=False, col_cluster=False,
                   linewidths=0.5, linecolor="lightgray", xticklabels=True, yticklabels=True,
                   figsize=(10, 3), cbar_pos=(0.02, 0.2, 0.02, 0.5))
_style_clustermap(g)
g.savefig(os.path.join(OUTPUT_PATH, "heatmap_subdomain_composition_log2fc.jpeg"), dpi=500, bbox_inches="tight")
plt.close()

# Spatial scatter of subdomains, per sample (separate coordinate frames).
for s in ["WT", "AD"]:
    m = spot_obs["sample"] == s
    styled_scatter(spot_obs.loc[m], spot_obs.loc[m, "subdomain"],
                   os.path.join(OUTPUT_PATH, f"subdomains_{s}.jpeg"), title=f"{s} neuropil subdomains")
print(f"Saved subdomain heatmap and scatter plots to {OUTPUT_PATH}")


# ==================================================================== #
#   4. Deliverable: AD-vs-WT granule DEGs within each neuropil subdomain #
# ==================================================================== #
# NOTE: pathway/enrichment analysis is intentionally omitted; this produces the ranked DEG
#       tables only. Run enrichment downstream on these gene lists if needed.

print("\n==================== 4. AD vs WT granule DEGs per subdomain ====================")
all_deg = []
for sd in [f"Subdomain {i + 1}" for i in range(N_SUBDOMAINS)]:
    m = (spot_obs["subdomain"] == sd).to_numpy()
    obs_sd = spot_obs.loc[m].copy()
    n_ad = int((obs_sd["sample"] == "AD").sum())
    n_wt = int((obs_sd["sample"] == "WT").sum())
    print(f"  {sd}: AD spots={n_ad}, WT spots={n_wt}")
    if n_ad < 2 or n_wt < 2:
        print(f"    Skipping {sd}: not enough spots in one condition.")
        continue

    expr_sd = spot_granule_expression[m].copy()
    de_adata = anndata.AnnData(X=expr_sd, obs=obs_sd.reset_index(drop=True), var=granule.var.copy())
    de_adata.var_names = granule.var_names
    sc.pp.normalize_total(de_adata, target_sum=1e4)
    sc.pp.log1p(de_adata)

    sc.tl.rank_genes_groups(de_adata, groupby="sample", groups=["AD"], reference="WT", method="wilcoxon")
    df = sc.get.rank_genes_groups_df(de_adata, group="AD")
    df = df.sort_values("logfoldchanges", ascending=False)
    df["subdomain"] = sd
    df["n_spots_AD"] = n_ad
    df["n_spots_WT"] = n_wt

    safe_sd = sd.replace(" ", "_")
    out_csv = os.path.join(OUTPUT_PATH, f"granule_DE_genes_AD_vs_WT_{safe_sd}.csv")
    df.to_csv(out_csv, index=False)
    print(f"    Saved {out_csv} ({df.shape[0]} genes)")
    all_deg.append(df)

if all_deg:
    combined = pd.concat(all_deg, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_PATH, "granule_DE_genes_AD_vs_WT_all_subdomains.csv"), index=False)
    print(f"\nSaved combined DEG table to {OUTPUT_PATH}granule_DE_genes_AD_vs_WT_all_subdomains.csv")

print("\nDone.")
