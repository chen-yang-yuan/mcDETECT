# ==============================================================================================
# A3 -- ambient RNA controls at the detection step: all figures
#
# Reviewer #2, major point 9. Section numbers match the notebooks 1:1:
#
#   [1]-[3]  A3a_three_sets.ipynb      set funnels / overlap ladder / per-region density
#   [4]      A3b_vicinity.ipynb        the detection predicate
#   [5]-[6]  A3c_de_baseline.ipynb     Axis 1, the compartment contrast
#   [7]      A3d_local_null.ipynb      the 10 um local-sampling null
#
# SEVEN FIGURES, and every one is placed in the response letter. Panels that were drawn but never
# cited -- the reciprocal-overlap bars, the vicinity-overlap map, the marker-count histogram, the
# Axis-1 residual violin and both Axis-2 panels -- have been removed. The Axis-2 ANALYSIS still
# runs and still writes its CSVs; it is simply not in the letter.
#
# This script reads CSV and Parquet ONLY. R cannot open an .h5ad, so anything plotted here was
# pre-exported by the Python side -- distributions arrive as a `*_summary.csv` + `*_histogram.parquet`
# pair (bin counts and quantiles), never as millions of raw values.
#
# WHERE THE ARGUMENT LIVES. Section [5] is the load-bearing panel: the fraction of vicinity
# pseudo-granules that would have been DETECTED, as a function of offset distance, against the real
# granules as a ceiling and tissue-wide random locations as the floor. Section [4] contains
# comparisons that are partly pre-ordained by the geometry (a matched-radius sphere cannot capture
# more than the minimum enclosing sphere it was copied from) -- description, not evidence.
#
# NOTHING IS DRAWN THAT IS NOT PLACED IN THE RESPONSE. Panels that existed only as an internal
# record -- the CSR/z-coverage preflight, the NC-filter forensics, the adaptive-threshold survival
# -- have been removed along with the analyses behind them.
#
# Every section is independently toggleable. Missing inputs degrade to a [skip] message rather than
# an error, so a partial run still produces every figure it can.
#
# Style: dpi 500, WT #a0ccec / AD #f48488 (kept in step with a3_config.py -- R cannot import it),
# theme_classic() for distributions, theme_bw() for dotplots, histogram outline #e9ecef.
#
# Usage:  Rscript A3_figures.R
# ==============================================================================================

suppressPackageStartupMessages({
  library(arrow); library(dplyr); library(ggplot2); library(tidyr); library(scales)
})

here::i_am("R2_revision/ambient_controls/A3_figures.R")

# -------------------- section toggles -------------------- #
# One panel per claim, and nothing else. Every figure below is placed in the response; a panel
# that is not is a panel that gets cut in review, so it is not drawn here.
RUN_SETS      <- TRUE    # 1. funnels, per million transcripts of the seeding gene
RUN_OVERLAP   <- TRUE    # 2. the overlap ladder, granule side, both controls
RUN_DENSITY   <- TRUE    # 3. per-region density per set, WT vs AD
RUN_PREDICATE <- TRUE    # 4. THE detection-predicate curve
RUN_AXIS1     <- TRUE    # 5. compartment: baseline vs granule enrichment
RUN_NONSEED   <- TRUE    # 6. THE non-circular panel: genes that seeded nothing
RUN_LOCALNULL <- TRUE    # 7. THE local-sampling null: is a granule a random draw locally?
RUN_PSEUDO    <- TRUE    # 8. THE pseudo-granule re-detection: build the hypothesis, feed it back

dpi <- 500

# -------------------- paths -------------------- #
root    <- here::here("R2_revision/ambient_controls/output")
a3a_dir <- file.path(root, "a3a")
a3b_dir <- file.path(root, "a3b")
a3c_dir <- file.path(root, "a3c")
a3d_dir <- file.path(root, "a3d")
a3e_dir <- file.path(root, "a3e")
fig_dir <- file.path(root, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# -------------------- palette and shared helpers -------------------- #
fill_colors <- c(WT = "#a0ccec", AD = "#f48488")     # a3_config.py holds the same two hex values
set_colors  <- c(set0 = "#b6b6b6", set1 = "#7fb069", set2 = "#4f7fa8", set3 = "#d98b5f")
area_order <- c("Isocortex", "OLF", "HPF-CA", "HPF-DG", "HPF-SR", "CTXsp", "TH", "MB", "FT")

need <- function(path, section) {
  if (!file.exists(path)) { message("  [skip ", section, "] missing: ", path); return(FALSE) }
  TRUE
}

save_fig <- function(p, name, width, height) {
  ggsave(file.path(fig_dir, name), p, width = width, height = height, dpi = dpi)
  message("  wrote ", name)
}

# ==============================================================================================
# 1. The sets, as funnels
# ==============================================================================================
#
# Rate, not count, and a funnel, not an endpoint. The control genes are ~15x rarer than the
# markers and DBSCAN yield is superlinear in count, so a raw comparison would be an abundance
# comparison; Set 0 -- unannotated genes at marker abundance -- is the arm that closes that off.
# The funnel also makes visible WHERE each population thins, which for Set 3 is the whole point:
# soma-restricted transcripts aggregate mostly inside somata, and it is the extrasomatic residue
# that this analysis counts.

if (RUN_SETS) {
  message("[1] set funnels")

  f <- file.path(a3a_dir, "funnel_by_gene.csv")
  if (need(f, "1")) {
    fun <- read.csv(f)
    # raw/size/in_soma are counted PER SEED GENE (the input to merge_sphere), so an aggregate
    # detected on several markers is counted several times. set_inventory.csv holds the merged
    # count, which is the population every later section uses; it is the funnel's last stage.
    finv <- file.path(a3a_dir, "set_inventory.csv")
    if (!need(finv, "1")) return(invisible(NULL))
    merged <- read.csv(finv) %>% select(set, sample, merged = n_spheres)
    agg <- fun %>%
      group_by(set, sample) %>%
      summarise(across(c(raw, size, in_soma, n_tx_gene), sum), .groups = "drop") %>%
      left_join(merged, by = c("set", "sample")) %>%
      pivot_longer(c(raw, size, in_soma, merged), names_to = "stage", values_to = "n") %>%
      mutate(stage = factor(stage, levels = c("raw", "size", "in_soma", "merged")),
             rate = n / (n_tx_gene / 1e6))

    p <- ggplot(agg, aes(x = stage, y = rate, colour = set, group = set)) +
      geom_line(linewidth = 0.9) + geom_point(size = 2.2) +
      facet_wrap(~ sample) + scale_y_log10() +
      scale_colour_manual(values = set_colors) +
      labs(x = NULL, y = "spheres per million transcripts of the seeding gene (log)",
           colour = NULL,
           caption = paste("Rate, not count: the NC genes are ~15x rarer than the markers and",
                           "DBSCAN yield is superlinear in count,\nso a raw comparison would be",
                           "an abundance comparison. Set 0 is abundance-matched to the markers.",
                           "The first three\nstages are counted per seed gene; `merged` is after",
                           "cross-gene merging and is the population used downstream.")) +
      theme_bw() + theme(legend.position = "bottom",
                         plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3a_funnel_rates.jpeg", 8, 5)
  }
}


# ==============================================================================================
# 2. The overlap ladder
# ==============================================================================================
#
# `intersect` (d < r_A + r_B) leads because it is the LOOSEST criterion and therefore maximises
# apparent overlap -- a small value under it cannot be argued with. mcDETECT's own merge predicate
# requires centres within 0.4*r and would understate co-location badly if quoted alone.
#
# ONE DIRECTION ONLY: the share of GRANULES meeting a control aggregate. That is what bounds
# contamination of the published result, and the Set1 -> Set2 fall in it -- which steepens as the
# criterion tightens -- is the evidence that the NC filter works. The reciprocal share and the
# re-placement null stay in overlap_ladder.csv but are not drawn; see the editorial note in
# section 1.3 of the response document. Set 0 and Set 3 are drawn side by side throughout.

if (RUN_OVERLAP) {
  message("[2] overlap ladder")

  f <- file.path(a3a_dir, "overlap_ladder.csv")
  if (need(f, "2")) {
    ov <- read.csv(f) %>%
      mutate(criterion = factor(criterion, levels = c("intersect", "center_in", "merge")),
             base = factor(base, levels = c("set1", "set2"),
                           labels = c("Set 1 (before NC filter)", "Set 2 (published)")),
             control = factor(control, levels = c("set0", "set3"),
                              labels = c("Set 0 (abundance-matched)",
                                         "Set 3 (nuclear-enriched)")))

    p <- ggplot(ov, aes(x = criterion, y = frac_overlapping, fill = sample)) +
      geom_col(position = "dodge") +
      facet_grid(control ~ base, scales = "free_y") +
      scale_fill_manual(values = fill_colors) +
      labs(x = NULL, y = "fraction of granules meeting a control aggregate", fill = NULL,
           caption = paste("`intersect` (spheres touch at all) is the loosest criterion and is",
                           "drawn first on purpose; `merge` is mcDETECT's own\nmerge predicate,",
                           "i.e. the criterion under which the detector would have combined the",
                           "two objects. Free y-axis per row.")) +
      theme_bw() + theme(legend.position = "bottom",
                         plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3a_overlap_ladder.jpeg", 9, 7)

  }
}


# ==============================================================================================
# 3. Per-region density per set, WT vs AD
# ==============================================================================================
#
# The claim: neither control population reproduces the granules' regional WT/AD pattern. Set 1 is
# the built-in positive control for that statistic -- it is Set 2 minus one filter, so if the
# per-region AD/WT ratio profile is recoverable at all, Set 1 must recover it.
# n = 1 vs 1, so this is descriptive -- and the capture-efficiency coefficient is a single global
# scalar whose per-region spread is reported beside it.

if (RUN_DENSITY) {
  message("[3] per-region density")

  f <- file.path(a3a_dir, "set_density_per_region.csv")
  if (need(f, "3")) {
    d <- read.csv(f)
    # subtype_density_per_region emits BOTH an "all" row and an identical "overall" row per
    # (area, sample); without this filter position="dodge" stacks the pair and every bar is
    # drawn at twice its true density. The notebook already filters, this is belt-and-braces.
    if ("subtype" %in% names(d)) d <- d %>% filter(subtype == "overall")
    if ("brain_area" %in% names(d)) {
      d <- d %>% filter(brain_area %in% area_order) %>%
        mutate(brain_area = factor(brain_area, levels = area_order))
      ycol <- intersect(c("density", "mean_count", "value"), names(d))[1]
      if (!is.na(ycol)) {
        p <- ggplot(d, aes(x = brain_area, y = .data[[ycol]], fill = sample)) +
          geom_col(position = "dodge") + facet_wrap(~ set, scales = "free_y") +
          scale_fill_manual(values = fill_colors) +
          labs(x = NULL, y = "granules per spot", fill = NULL,
               caption = "n = 1 WT vs 1 AD section: descriptive, not inferential.") +
          theme_bw() +
          theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom",
                plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
        save_fig(p, "a3a_density_per_region.jpeg", 10, 6)
      }
    }
  }
}


# ==============================================================================================
# 4. A3b -- THE detection predicate
# ==============================================================================================
#
# This is the panel the response rests on. For each pseudo-granule: would DBSCAN(eps=1.5,
# min_samples=3) on the SAME SEED GENE actually have fired at that location? eps-connectivity, not
# a count -- three transcripts scattered across a 4 um sphere are not connected, so a count-based
# version would badly overstate detectability. Seed-matched because "any of 20 markers" is ~20x
# easier than "3 Camk2a", and Camk2a alone is 47% of the published granules.
#
# The SHAPE answers the reviewer. Rising sharply as the offset shrinks toward the granule = the
# call has genuine local specificity. Flat from 5 to 50 um, i.e. already at the tissue-wide
# asymptote right next to a real granule = it does not, and that is his hypothesis confirmed.

if (RUN_PREDICATE) {
  message("[4] detection predicate")

  f <- file.path(a3b_dir, "detection_predicate.csv")
  if (need(f, "4")) {
    pr <- read.csv(f)
    ref <- pr %>% filter(arm %in% c("real", "random_tissue"))
    cur <- pr %>% filter(!arm %in% c("real", "random_tissue")) %>%
      mutate(d_num = suppressWarnings(as.numeric(d_label)))

    p <- ggplot(cur, aes(x = d_num, y = frac_detect, colour = sample, shape = arm)) +
      geom_line(aes(linetype = arm), linewidth = 0.9) + geom_point(size = 2.4) +
      geom_hline(data = ref %>% filter(arm == "real"),
                 aes(yintercept = frac_detect, colour = sample), linetype = "dashed") +
      geom_hline(data = ref %>% filter(arm == "random_tissue"),
                 aes(yintercept = frac_detect, colour = sample), linetype = "dotted") +
      facet_wrap(~ d_kind, scales = "free_x") +
      scale_colour_manual(values = fill_colors) +
      scale_y_continuous(limits = c(0, 1)) +
      labs(x = "offset distance", y = "fraction that would have been detected",
           colour = NULL, shape = NULL, linetype = NULL,
           caption = paste("dashed = real granules (the ceiling, must be ~1 by construction);",
                           "dotted = tissue-wide random locations (the floor).\nThe SHAPE is the",
                           "answer: flat from 5 to 50 um would mean the call has no local",
                           "specificity.")) +
      theme_bw() + theme(legend.position = "bottom",
                         plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3b_detection_predicate.jpeg", 9, 5.5)
  }

  f <- file.path(a3b_dir, "detection_predicate_stratified.csv")
  if (need(f, "4")) {
    st <- read.csv(f)
    if ("density_quintile" %in% names(st)) {
      sq <- st %>% filter(!is.na(density_quintile)) %>%
        mutate(d = paste0(d_kind, ":", d_label))
      p <- ggplot(sq, aes(x = factor(density_quintile), y = frac_detect,
                          colour = sample, group = interaction(sample, d))) +
        geom_line(alpha = 0.6) + geom_point(size = 1.8) +
        facet_wrap(~ d) + scale_colour_manual(values = fill_colors) +
        labs(x = "local transcript-density quintile", y = "fraction detected", colour = NULL,
             caption = paste("Within-stratum: a difference that survives here cannot be a local",
                             "density difference,\nwhich is the mechanism the reviewer",
                             "proposes.")) +
        theme_bw() + theme(legend.position = "bottom",
                           plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
      save_fig(p, "a3b_predicate_by_density.jpeg", 9, 6)
    }
  }
}


# ==============================================================================================
# 5. A3c -- Axis 1, the compartment contrast
# ==============================================================================================
#
# The reviewer's literal ask: DE between somatic and all non-somatic RNA, independent of granule
# detection, then whether the granule-specific differences exceed or DIVERGE FROM that baseline.
#
# The scatter is lifted from code/figures_response.Rmd:1424-1452 (which already sits under a
# heading titled "Reviewer 2, Major Comment 9") with two changes. (1) Granule enrichment now uses
# the SAME soma reference as the baseline, so the two axes share a denominator and their
# difference is meaningful -- and the soma term then cancels exactly out of `delta`, leaving
# granule vs extrasomatic. (2) The baseline plotted is `baseline_all_logFC`, ALL non-somatic RNA
# (granule + residual extrasomatic) vs soma, which is the reviewer's literal wording and is the
# one baseline genuinely independent of granule detection.
#
# `baseline_logFC` (residual extrasomatic alone) is the SENSITIVITY arm, kept in the CSV. It is
# granule-free, which is desirable, but it is defined as "extrasomatic AND not inside a called
# sphere" and is therefore detection-DEPENDENT. Never label it detection-independent.

if (RUN_AXIS1) {
  message("[5] axis 1 -- compartment")

  f <- file.path(a3c_dir, "axis1_gene_table.csv")
  if (need(f, "5")) {
    df <- read.csv(f) %>%
      mutate(marker_group = ifelse(is_marker == "True" | is_marker == TRUE,
                                   "Granule markers", "Others"))
    # fall back to the granule-free baseline when reading a pre-`_all` CSV
    xvar  <- if ("baseline_all_logFC" %in% names(df)) "baseline_all_logFC" else "baseline_logFC"
    xlab  <- if (xvar == "baseline_all_logFC")
      "Baseline logFC (all non-somatic RNA vs soma)" else
      "Baseline logFC (residual extrasomatic vs soma)"

    p <- ggplot(df, aes(x = .data[[xvar]], y = granule_enrichment)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dotted", colour = "grey50") +
      geom_smooth(data = df %>% filter(marker_group == "Others"),
                  method = "lm", se = TRUE, colour = "black", linewidth = 0.75) +
      geom_point(aes(fill = marker_group), shape = 21, size = 2, colour = "black",
                 stroke = 0.1) +
      facet_wrap(~ sample) +
      scale_fill_manual(values = c("Granule markers" = "#f48488", "Others" = "#a0ccec")) +
      labs(x = xlab,
           y = "Granule enrichment logFC (granule vs soma)", fill = NULL,
           caption = paste("Line fitted on NON-markers. Both axes share the soma reference, so",
                           "their difference is meaningful and the\nsoma term cancels out of it.",
                           "The baseline is all non-somatic RNA -- the reviewer's wording, and",
                           "the only\nform independent of granule detection. Including the",
                           "in-granule transcripts biases the contrast\ntoward zero, so this is",
                           "also the conservative choice.")) +
      theme_classic() +
      theme(axis.text = element_text(size = 12), axis.title = element_text(size = 13),
            legend.position = "bottom",
            plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3c_axis1_scatter.jpeg", 9, 5.5)

  }
}


# ==============================================================================================
# 6. A3c -- the non-seed, non-control gene test
# ==============================================================================================
#
# THE LOAD-BEARING PANEL OF SECTION 3, and the only one in it that is not circular. Section [5]'s
# scatter is marker-anchored: mcDETECT builds a granule by clustering marker transcripts, so the
# markers sitting above the non-marker line is guaranteed by construction, not evidence.
#
# This panel drops every gene that seeded detection (SYN_GENES) or entered nc_filter (the 19-gene
# published NC list) -- 38 genes -- and plots the remaining 252 by the panel's OWN curated
# cell-type annotation, written when the probe set was designed. y is the count model's
# granule-vs-residual logFC, measured WITHIN each 50 um spot, so a "granules just sit in neuropil"
# reading cannot produce a separation here.
#
# Nearly every gene is negative: a granule sphere is dominated by the transcripts that formed it,
# so everything else is diluted. The claim is the ORDERING, which is why the zero line is drawn
# but not emphasised.

if (RUN_NONSEED) {
  message("[6] non-seed gene test")

  f <- file.path(a3c_dir, "axis1_nonseed_genes.csv")
  if (need(f, "6")) {
    NEURONAL <- c("Excitatory neurons", "Inhibitory neurons")
    GLIAL    <- c("Astrocytes", "Oligodendrocytes", "Microglia", "OPC",
                  "Pericytes/Endothelial", "Fibroblast")

    ns <- read.csv(f) %>%
      filter(cell_type %in% c(NEURONAL, GLIAL)) %>%
      mutate(klass = ifelse(cell_type %in% NEURONAL, "Neuronal", "Glial / vascular"),
             cell_type = factor(cell_type, levels = c(NEURONAL, GLIAL)))

    p <- ggplot(ns, aes(x = cell_type, y = logFC_granule_vs_residual, fill = klass)) +
      geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
      geom_boxplot(outlier.shape = NA, alpha = 0.85, width = 0.65) +
      geom_point(shape = 21, size = 1.6, colour = "black", stroke = 0.15,
                 position = position_jitter(width = 0.12, height = 0), alpha = 0.9) +
      facet_wrap(~ sample) +
      scale_fill_manual(values = c("Neuronal" = "#f48488", "Glial / vascular" = "#a0ccec")) +
      labs(x = NULL, y = "granule vs surrounding\nnon-somatic RNA (log2)",
           fill = NULL,
           caption = paste("Every gene shown seeded no detection and entered no filter: the 20",
                           "marker genes and the 19 negative\ncontrols are excluded. The",
                           "annotation is the panel's own curated design sheet.",
                           "\nThe dashed line at zero is equal representation inside and",
                           "outside granules; shares are computed over these 252 genes only.")) +
      theme_bw() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1),
            legend.position = "bottom",
            plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3c_nonseed_celltype.jpeg", 11, 6.2)
  }
}


# ==============================================================================================
# 7. A3d -- the local sampling null
# ==============================================================================================
#
# Section [6] shows that granules RANK neuronal genes above glial ones. This one asks the
# reviewer's question in its literal form -- if a granule were a random sample of the non-somatic
# RNA in its own 10 um neighbourhood, could it look like this? -- and answers it with a
# probability rather than an ordering.
#
# (a) is the effect size: how far each gene's share of granule RNA departs from what the local
# RNA predicts. Unlike section [6]'s panel, zero here is not merely a reference line -- it is the
# reviewer's hypothesis. A gene at zero is exactly as common inside granules as in the material
# 10 um around them.
#
# (b) is the test. T is the gap between the neuronal and glial medians in (a), recomputed on
# every one of the simulated sections in which granules ARE random local samples. The null is a
# spike because a thousand redraws produce nothing remotely like the observation; that the spike
# looks small next to the observed line is the entire point of the panel, not a scaling problem.
#
# ONE null: the permutation (pool each 10 um square's granule and residual transcripts, relabel
# which are "granule"). The literal multinomial variant was retired -- see the A3d block in
# a3_config.py, which holds the canonical mode name that R cannot import.

if (RUN_LOCALNULL) {
  message("[7] local sampling null")

  fg <- file.path(a3d_dir, "a3d_local_null_genes.csv")
  fn <- file.path(a3d_dir, "a3d_local_null_group_null.csv")
  fk <- file.path(a3d_dir, "a3d_local_null_group.csv")
  if (need(fg, "7") && need(fn, "7") && need(fk, "7")) {
    NEURONAL <- c("Excitatory neurons", "Inhibitory neurons")
    GLIAL    <- c("Astrocytes", "Oligodendrocytes", "Microglia", "OPC",
                  "Pericytes/Endothelial", "Fibroblast")

    gn_all <- read.csv(fg)
    n_gene <- length(unique(gn_all$gene))           # 252, read off the table not typed in
    gn <- gn_all %>%
      filter(cell_type %in% c(NEURONAL, GLIAL)) %>%
      mutate(klass = ifelse(cell_type %in% NEURONAL, "Neuronal", "Glial / vascular"),
             cell_type = factor(cell_type, levels = c(NEURONAL, GLIAL)))

    pa <- ggplot(gn, aes(x = cell_type, y = log2_obs_over_exp, fill = klass)) +
      geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
      geom_boxplot(outlier.shape = NA, alpha = 0.85, width = 0.65) +
      geom_point(shape = 21, size = 1.6, colour = "black", stroke = 0.15,
                 position = position_jitter(width = 0.12, height = 0), alpha = 0.9) +
      facet_wrap(~ sample) +
      scale_fill_manual(values = c("Neuronal" = "#f48488", "Glial / vascular" = "#a0ccec")) +
      labs(x = NULL, fill = NULL,
           y = "observed / expected in granules,\nagainst the local 10 um pool (log2)",
           title = "a  Departure from a random local sample, gene by gene") +
      theme_bw() +
      theme(axis.text.x = element_text(angle = 30, hjust = 1),
            legend.position = "bottom",
            plot.title = element_text(size = 11, face = "bold"))

    tn <- read.csv(fn)
    n_rep <- nrow(tn) / length(unique(tn$sample))   # not hardcoded: read off the null itself
    tk  <- read.csv(fk)
    lab <- tk %>%
      mutate(txt = sprintf("observed T = %.2f\n(%.0f null SD away)", T_obs, z))

    pb <- ggplot(tn, aes(x = T)) +
      geom_histogram(bins = 40, fill = "grey75", colour = "#e9ecef", linewidth = 0.2) +
      geom_vline(data = tk, aes(xintercept = T_obs), colour = "#b03030", linewidth = 0.7) +
      geom_text(data = lab, aes(x = -Inf, y = Inf, label = txt),
                hjust = -0.05, vjust = 1.35, size = 3.0, colour = "#b03030", lineheight = 0.95) +
      facet_wrap(~ sample) +
      expand_limits(x = c(0, max(tk$T_obs) * 1.10)) +
      labs(x = paste("T = median log2(obs/exp) in neuronal genes",
                     "minus the same in glial and vascular genes"),
           y = "draws under the hypothesis",
           title = "b  The same statistic under the hypothesis that granules are random local samples",
           caption = paste(sprintf("Grey: %s draws of T under the hypothesis that a granule's",
                                   format(n_rep, big.mark = ",", scientific = FALSE)),
                           "contents are a random relabelling of it and\nthe RNA in its own 10 um",
                           "square, keeping the number of transcripts per granule fixed.",
                           "Red: the observed value.\nAll", n_gene, "genes tested seeded no",
                           "detection and entered no filter, so nothing here follows\nfrom how a",
                           "granule was defined.")) +
      theme_classic() +
      theme(plot.title = element_text(size = 11, face = "bold"),
            plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"),
            strip.background = element_blank())

    p <- patchwork::wrap_plots(pa, pb, ncol = 1, heights = c(1.15, 1))
    save_fig(p, "a3d_local_null.jpeg", 11, 9.4)
  }
}


# ============================================================================================== #
# 8. A3e -- ambient pseudo-granules, put back through the detector
# ============================================================================================== #
# The most direct answer in the whole response, and the one panel a reader can grasp without any
# statistics: three groups of real granules, identical in every transcript position, differing only
# in what the transcripts are CALLED, run through the published pipeline again.
#
#   untouched  nothing changed         -- must come back at ~100%, or nothing else here is readable
#   scramble   own labels permuted     -- composition preserved exactly, geometry scrambled
#   ambient    labels drawn from the surrounding residual RNA
#
# The `scramble` bar is what makes the `ambient` bar mean something. Both arms scramble which point
# carries which gene; only `ambient` also changes the composition. So the gap between them is the
# compositional effect on its own, and the gap between `untouched` and `scramble` is whatever the
# scrambling costs by itself.

if (RUN_PSEUDO) {
  message("[8] pseudo-granule re-detection")

  fr <- file.path(a3e_dir, "a3e_redetection_rate.csv")
  fm <- file.path(a3e_dir, "a3e_marker_shift.csv")
  if (need(fr, "8")) {
    arm_lab <- c(untouched = "Untouched\n(control)",
                 scramble  = "Own labels\npermuted",
                 ambient   = "Labels drawn from\nlocal ambient RNA")
    arm_col <- c(untouched = "#4f7fa8", scramble = "#b6b6b6", ambient = "#b03030")

    # `is_primary` is written by the notebook from a3_config.PSEUDO_MATCH_PRIMARY -- R cannot
    # import a3_config, so the choice travels in the data rather than being retyped here.
    # The table also carries an `excluded_contaminated` row -- untouched granules that shared a
    # rewritten transcript with a converted one. They are reported in the notebook and are not a
    # study arm, so they are dropped here rather than plotted as an unlabelled fourth bar.
    rr <- read.csv(fr) %>%
      filter(is_primary == "True" | is_primary == TRUE, arm %in% names(arm_lab)) %>%
      mutate(arm = factor(arm, levels = names(arm_lab)))
    stopifnot(nrow(rr) == length(unique(rr$sample)) * length(arm_lab))

    pa <- ggplot(rr, aes(x = arm, y = rate, fill = arm)) +
      geom_col(width = 0.68, alpha = 0.9) +
      geom_text(aes(label = sprintf("%.1f%%", 100 * rate)), vjust = -0.45, size = 3.4) +
      facet_wrap(~ sample) +
      scale_y_continuous(labels = scales::percent, limits = c(0, 1.08),
                         expand = expansion(mult = c(0, 0.02))) +
      scale_x_discrete(labels = arm_lab) +
      scale_fill_manual(values = arm_col, guide = "none") +
      labs(x = NULL, y = "re-detected by mcDETECT",
           title = "a  Granules whose transcripts were relabelled, put back through the detector") +
      theme_bw() +
      theme(plot.title = element_text(size = 11, face = "bold"),
            strip.background = element_blank())

    p <- pa
    if (file.exists(fm)) {
      mk <- read.csv(fm) %>%
        filter(arm %in% names(arm_lab)) %>%
        mutate(arm = factor(arm, levels = names(arm_lab)))
      pb <- ggplot(mk, aes(x = n_marker, y = frac, colour = arm)) +
        geom_step(linewidth = 0.7) +
        facet_wrap(~ sample) +
        scale_colour_manual(values = arm_col, labels = arm_lab, name = NULL) +
        labs(x = "marker transcripts in the sphere after relabelling",
             y = "fraction of granules",
             title = "b  Why: the local ambient supplies too few marker transcripts",
             caption = paste("mcDETECT seeds DBSCAN on the 20 granule markers and needs",
                             "min_samples = 3 of them within eps = 1.5 um.\nEvery transcript keeps",
                             "its own position in all three arms, so local density is identical",
                             "throughout and\ncomposition is the only thing that varies.")) +
        theme_bw() +
        theme(plot.title = element_text(size = 11, face = "bold"),
              plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"),
              legend.position = "bottom", strip.background = element_blank())
      p <- patchwork::wrap_plots(pa, pb, ncol = 1, heights = c(1, 1.15))
    }
    save_fig(p, "a3e_pseudo_granules.jpeg", 10, if (file.exists(fm)) 9.0 else 4.6)
  }
}


message("done -- figures in ", fig_dir)
