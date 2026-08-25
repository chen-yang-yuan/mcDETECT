# ==============================================================================================
# A2 -- figures for the sparsity / stochastic-origin analyses
#
#   plans/Round2_response_analysis_plan.md, section A2.
#
# Sections 1-4 plot A2a (`A2a_multigene.ipynb`); section 5 plots A2b; section 6 plots A2c
# (`run_permutation_detect.py` -> `score_embedding.py` on HGCC).
#
# This script reads CSV and Parquet only. R cannot open an .h5ad, so anything plotted here was
# pre-exported by the Python side; conversely the granule-subtype HEATMAPS and the t-SNE panels
# are rendered by scanpy in the notebook / scoring script, not here.
#
# Every section is independently toggleable below -- the GSEA section in particular is slow and
# pulls in clusterProfiler.
#
# Conventions follow code/figures.Rmd and R2_revision/baysor_ssam_merscope/postproc/A1_figures.R:
# dpi 500, WT #a0ccec / AD #f48488, theme_classic() for distributions and theme_bw() for
# dotplots, histogram outline #e9ecef.
#
#   Rscript A2_figures.R
# ==============================================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

here::i_am("R2_revision/sparsity_structure/A2_figures.R")

# -------------------- section toggles -------------------- #
RUN_COMPLEXITY  <- TRUE    # 1. complexity distributions, comp-vs-n_genes, retention
RUN_DENSITY     <- TRUE    # 2. subtype composition + WT/AD density bars
RUN_GSEA        <- TRUE    # 3. microdomain DE -> GSEA + NES dotplots  (slow)
# 3b. Gene x pathway chord diagrams, the mcDETECT pipeline's own view of the same GSEA result.
# A separate flag from RUN_GSEA because each diagram is a 4000x4000 JPEG: turning this on roughly
# doubles section 3's runtime and quadruples its file count, for a figure the dotplots already
# summarise. On here (unlike A1_figures.R:77) because the response calls for it.
RUN_GSEA_CHORD  <- TRUE
RUN_READSTRATA  <- TRUE    # 4. read-count terciles
RUN_A2B         <- TRUE    # 5. permutation null vs real embedding
RUN_A2C         <- TRUE    # 6. gene co-occurrence: localization vs co-expression programmes

# Optional selectors for section 3; NULL = every DE table found.
GSEA_PAIRS <- NULL         # e.g. c("Subdomain 3_vs_Subdomain 1")

# Section 2 companion panel: the same two composition bars split WT | AD. Self-contained --
# set FALSE to drop it and nothing else changes.
COMPOSITION_BY_SAMPLE <- TRUE

dpi <- 500

# -------------------- paths -------------------- #
root        <- here::here("R2_revision/sparsity_structure/output")
a2a_dir     <- file.path(root, "a2a", "multigene")
strata_dir  <- file.path(root, "a2a", "readstrata")
a2b_dir     <- file.path(root, "a2b", "metrics")
a2c_dir     <- file.path(root, "a2c")
sub_dir     <- file.path(a2a_dir, "neuropil_subdomains_Isocortex_50")
fig_dir     <- file.path(root, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# The published reference tables, for side-by-side anchors.
pub_dir       <- here::here("output/MERSCOPE_WT_AD_comparison")
pub_density   <- file.path(pub_dir, "subtype_density_per_region_granule_adata_tsne.csv")
pub_subdomain <- file.path(pub_dir, "neuropil_subdomains_Isocortex_50")
pub_subtype_labels <- file.path(pub_dir, "granule_subtype_labels_granule_adata_tsne.parquet")

# -------------------- palette and shared helpers -------------------- #
# a2_config.py keeps the same two hex values; R cannot import it, so change both together.
fill_colors <- c(WT = "#a0ccec", AD = "#f48488")
cond_colors <- c(real = "#4f7fa8", permuted = "#b0b0b0")
# a2_config.py PROGRAMME_COLORS keeps the same two values; change both together.
prog_colors <- c(localization = "#4f7fa8", "co-expression" = "#d98b5f")
area_order  <- c("Isocortex", "OLF", "HPF-CA", "HPF-DG", "HPF-SR", "CTXsp", "TH", "MB", "FT")
subtype_order <- c("overall", "pre-syn", "post-syn", "dendrites", "axons", "mixed")
# Separate from `subtype_order` on purpose: the composition panels mirror the mcDETECT benchmarking
# figures, whose level order is this one (code/figures_response.Rmd:2077). `subtype_order` above
# leads with the "overall" pseudo-row, which exists only in the DENSITY tables, and carries "axons"
# rather than "others".
composition_order <- c("pre-syn", "post-syn", "dendrites", "mixed", "others")
# matplotlib's tab20, used for the chord-diagram pathway sectors. Copied from A1_figures.R:107-110,
# itself from code/figures_response.Rmd:176 -- the published chord JPEGs use these exact hex values,
# so the multi-gene panels only sit beside them convincingly if this vector is left alone.
tab20_colors <- c("#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
                  "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
                  "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
                  "#17becf", "#9edae5")

need <- function(path, section) {
  if (!file.exists(path)) {
    message("  [skip ", section, "] missing: ", path)
    return(FALSE)
  }
  TRUE
}

# One function draws every pre-binned distribution panel, in sections 1 and 5, so they cannot
# drift apart. `df` carries bin_lo / bin_hi / frac; the final [hi, Inf) overflow bin is folded
# onto the last finite bin and flagged in the caption rather than dropped.
binned_hist <- function(df, out_file, x_lab, fill_var, palette, caption = NULL,
                        width = 8, height = 5) {
  finite_hi <- max(df$bin_hi[is.finite(df$bin_hi)])
  overflow <- df %>% filter(!is.finite(bin_hi))
  df <- df %>%
    filter(is.finite(bin_hi)) %>%
    mutate(x = (bin_lo + bin_hi) / 2, w = bin_hi - bin_lo)

  if (nrow(overflow) && sum(overflow$frac, na.rm = TRUE) > 0) {
    caption <- paste0(caption %||% "",
                      sprintf("%s%.2f%% of values exceed %.3g and are not shown.",
                              if (is.null(caption)) "" else " ",
                              100 * sum(overflow$frac, na.rm = TRUE), finite_hi))
  }

  p <- ggplot(df, aes(x = x, y = frac, fill = .data[[fill_var]])) +
    geom_col(position = "identity", alpha = 0.55, colour = "#e9ecef",
             width = df$w[1], linewidth = 0.2) +
    scale_fill_manual(values = palette) +
    labs(x = x_lab, y = "Fraction of detections", fill = NULL, caption = caption) +
    theme_classic() +
    theme(axis.text = element_text(size = 12), axis.title = element_text(size = 13),
          legend.position = "bottom",
          plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
  ggsave(out_file, p, width = width, height = height, dpi = dpi)
  message("  wrote ", basename(out_file))
}

`%||%` <- function(a, b) if (is.null(a)) b else a


# ==============================================================================================
# 1. Granule complexity and retention
# ==============================================================================================

if (RUN_COMPLEXITY) {
  message("[1] complexity and retention")

  f_hist <- file.path(a2a_dir, "complexity_histogram.parquet")
  f_summ <- file.path(a2a_dir, "complexity_summary.csv")
  f_ct   <- file.path(a2a_dir, "comp_vs_ngenes.parquet")
  f_ret  <- file.path(a2a_dir, "retention_by_region.csv")

  if (need(f_hist, "1")) {
    hist_df <- read_parquet(f_hist) %>%
      mutate(sample = factor(sample, levels = c("WT", "AD")))

    # (a) all granules, WT vs AD -- this is the Fig. R9 content the reviewer asked twice to see.
    # Provenance differs by measure and travels with the data in a `buffer` column: reads come
    # from the PUBLISHED buffer = 0.01 table (the source of the manuscript's and the reviewer's
    # "median 6-7 reads"), unique genes from the buffer = 0.00 column the >= 3 filter acts on, so
    # this panel and multigene_retention.jpeg cannot disagree. Stated in the caption rather than
    # left to the README.
    buf_note <- function(df) {
      b <- unique(df$buffer)
      if (length(b) != 1) return("")
      if (b == 0.01) "Profiling radius buffer = 0.01 um, as published for Fig. R9."
      else "Profiling radius buffer = 0.00 um, matching the matrix the subset is taken from."
    }
    for (m in c("n_reads", "n_genes")) {
      d <- hist_df %>% filter(measure == m, population == "all")
      binned_hist(d, file.path(fig_dir, paste0("complexity_", m, "_all.jpeg")),
                  x_lab = if (m == "n_reads") "Reads per granule" else "Unique genes per granule",
                  fill_var = "sample", palette = fill_colors, caption = buf_note(d))
    }

    # (b) all vs the multi-gene subset, per sample -- what the >= 3 cutoff removes.
    for (smp in c("WT", "AD")) {
      d <- hist_df %>%
        filter(measure == "n_reads", sample == smp) %>%
        mutate(population = factor(population, levels = c("all", "multigene"),
                                   labels = c("All granules", "Multi-gene subset")))
      binned_hist(d, file.path(fig_dir, paste0("complexity_reads_subset_", smp, ".jpeg")),
                  x_lab = "Reads per granule", fill_var = "population",
                  palette = c("All granules" = "#c9c9c9",
                              "Multi-gene subset" = unname(fill_colors[smp])),
                  caption = paste0(smp, ": read-count distribution before and after the",
                                   " unique-gene cutoff. ", buf_note(d)))
    }
  }

  # (c) comp vs unique genes -- the panel showing what `comp` actually counted.
  if (need(f_ct, "1")) {
    ct <- read_parquet(f_ct) %>% filter(count > 0)
    p <- ggplot(ct, aes(x = comp, y = n_genes, fill = log10(count))) +
      geom_tile() +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey40") +
      scale_fill_distiller(palette = "Reds", direction = 1) +
      labs(x = "comp (distinct granule markers, mcDETECT output)",
           y = "Unique genes per granule (whole panel, buffer = 0.00)",
           fill = expression(log[10]~"granules"),
           caption = paste("`comp` is capped at the 20 granule markers and is not recomputed",
                           "after sphere merging, so it is not a measure of granule complexity;",
                           "the y axis is.")) +
      theme_classic() +
      theme(axis.text = element_text(size = 12), axis.title = element_text(size = 13),
            plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
    ggsave(file.path(fig_dir, "comp_vs_unique_genes.jpeg"), p, width = 7.5, height = 6, dpi = dpi)
    message("  wrote comp_vs_unique_genes.jpeg")
  }

  # (d) retention by region.
  if (need(f_ret, "1")) {
    ret <- read.csv(f_ret, check.names = FALSE) %>%
      filter(brain_area %in% c("overall", area_order)) %>%
      mutate(sample = ifelse(grepl("WT", batch), "WT", "AD"),
             sample = factor(sample, levels = c("WT", "AD")),
             brain_area = factor(brain_area, levels = c("overall", area_order)))
    p <- ggplot(ret, aes(x = brain_area, y = retention, fill = sample)) +
      geom_col(position = position_dodge(width = 0.8), width = 0.7, colour = "#3b3b3b") +
      scale_fill_manual(values = fill_colors) +
      scale_y_continuous(labels = scales::percent, expand = c(0, 0),
                         limits = c(0, max(ret$retention, na.rm = TRUE) * 1.15)) +
      labs(x = NULL, y = "Granules retained", fill = NULL,
           caption = sprintf("Retained = at least %d unique genes per granule.",
                             ret$min_unique_genes[1])) +
      theme_classic() +
      theme(axis.text = element_text(size = 12), axis.title = element_text(size = 13),
            legend.position = "bottom",
            plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
    ggsave(file.path(fig_dir, "multigene_retention.jpeg"), p, width = 9, height = 5, dpi = dpi)
    message("  wrote multigene_retention.jpeg")
  }

  if (need(f_summ, "1")) {
    message("  quantiles: ", f_summ)
  }
}


# ==============================================================================================
# 2. Subtype composition and WT-vs-AD density
# ==============================================================================================

# Bar-plot code lifted from code/figures_response.Rmd:2224-2299, so the multi-gene panels are
# drawn exactly like the published ones and can be placed beside them.
density_bars <- function(density_df, subtype_to_plot, out_file, caption = NULL) {
  df_plot <- density_df %>%
    filter(subtype == subtype_to_plot, sample %in% c("WT", "AD"), brain_area %in% area_order) %>%
    mutate(brain_area = factor(brain_area, levels = area_order),
           sample = factor(sample, levels = c("WT", "AD")))
  if (nrow(df_plot) == 0) {
    message("  [skip] no rows for subtype ", subtype_to_plot)
    return(invisible(NULL))
  }

  has_ci <- all(c("density_ci_low", "density_ci_high") %in% colnames(df_plot))
  y_range <- if (has_ci) max(df_plot$density_ci_high, na.rm = TRUE) else max(df_plot$density, na.rm = TRUE)
  gap <- 0.03 * y_range

  df_star <- df_plot %>%
    group_by(brain_area) %>%
    summarise(y_max = if (has_ci) max(density_ci_high, na.rm = TRUE) else max(density, na.rm = TRUE),
              .groups = "drop") %>%
    left_join(df_plot %>% distinct(brain_area, p_bonf_star), by = "brain_area") %>%
    mutate(x = brain_area, y = y_max + gap,
           label_size = ifelse(p_bonf_star == "ns", 5, 7))
  y_top <- max(df_star$y, na.rm = TRUE) + gap
  dodge <- position_dodge(width = 0.8)

  p <- ggplot(df_plot, aes(x = brain_area, y = density, fill = sample)) +
    geom_col(position = dodge, width = 0.7, color = "#3b3b3b") +
    {if (has_ci) geom_errorbar(aes(ymin = density_ci_low, ymax = density_ci_high),
                               position = dodge, color = "#3b3b3b", width = 0.2, linewidth = 0.6)
     else NULL} +
    geom_text(data = df_star, aes(x = x, y = y, label = p_bonf_star, size = label_size),
              inherit.aes = FALSE, vjust = 0) +
    scale_size_identity() +
    scale_fill_manual(values = fill_colors) +
    scale_y_continuous(expand = c(0, 0)) +
    coord_cartesian(clip = "off", ylim = c(0, y_top)) +
    labs(x = " ", y = "Density", fill = " ", caption = caption) +
    theme_classic() +
    theme(axis.title.y = element_text(size = 15),
          axis.text.x  = element_text(size = 15),
          axis.text.y  = element_text(size = 15),
          legend.position = "bottom",
          legend.box.margin = margin(t = -20, b = 0),
          legend.title = element_text(size = 15),
          legend.text  = element_text(size = 15),
          plot.caption = element_text(size = 9, colour = "grey30", hjust = 0),
          plot.margin  = margin(t = 10, r = 10, b = 10, l = 10))
  ggsave(out_file, p, width = 9, height = 6, dpi = dpi)
  message("  wrote ", basename(out_file))
}

# Stacked 100%-composition bars, one bar per population.
#
# Layout copied verbatim from the mcDETECT benchmarking panels (code/figures_response.Rmd:2102-2119
# -- the sphere-construction, rho and filtering scenarios), so the output can be placed directly
# beside output/benchmark/benchmark_sphere/WT_AD_comparison/granule_subtype_composition.jpeg. Two
# deliberate departures from that source: it filters to `sample == "WT"` and we pool WT+AD, and its
# x-axis is a detection PARAMETER rather than a granule population.
#
# No caption, unlike every other panel in this script -- the benchmarking figures carry none, and
# matching them is the whole point. The one caveat worth stating (the two bars are labelled by two
# independent clusterings) is emitted to the console instead.
#
# The fill scale deliberately KEEPS ggplot's default `drop = TRUE`. "others" is empty under every
# current mapping, and pinning the scale would put a legend key on the panel for a subtype holding
# zero granules. Colours still match the published benchmarking figure without pinning: brewer
# assigns in factor-level order, and `composition_order` is that figure's own order, so the four
# populated subtypes take the same first four Set2 colours there as here.
composition_bars <- function(df, out_file, x_labels, facet = NULL, width = 7, height = 6) {
  p <- ggplot(df, aes(x = population, y = pct, fill = granule_subtype_manual_simple)) +
    geom_col(width = 0.6, color = NA) +
    scale_x_discrete(drop = FALSE, labels = x_labels) +
    scale_y_continuous(limits = c(0, 101), breaks = seq(0, 101, 20), expand = c(0, 0)) +
    scale_fill_brewer(palette = "Set2") +
    labs(x = NULL, y = "Subtype composition (%)", fill = "Subtype") +
    coord_cartesian(ylim = c(0, 100)) +
    {if (!is.null(facet)) facet_wrap(stats::as.formula(paste("~", facet))) else NULL} +
    theme_classic() +
    theme(axis.text = element_text(size = 18),
          axis.title = element_text(size = 18),
          plot.margin = margin(10, 20, 10, 10),
          legend.position = "bottom",
          legend.box.margin = margin(t = -10, b = 0),
          legend.title = element_text(size = 15),
          legend.text = element_text(size = 15),
          strip.background = element_blank(),
          strip.text = element_text(size = 16))
  ggsave(out_file, p, width = width, height = height, dpi = dpi)
  message("  wrote ", basename(out_file))
}

if (RUN_DENSITY) {
  message("[2] subtype composition and density")

  f_den  <- file.path(a2a_dir, "subtype_density_per_region_multigene.csv")
  f_comp <- file.path(a2a_dir, "subtype_composition.csv")

  if (need(f_den, "2")) {
    den <- read.csv(f_den, check.names = FALSE)
    for (s in intersect(subtype_order, unique(den$subtype))) {
      density_bars(den, s,
                   file.path(fig_dir, paste0("granule_density_multigene_", s, ".jpeg")),
                   caption = "Multi-gene granules only. Bonferroni-corrected within subtype.")
    }

    # Published full-set densities beside them, so the persistence claim is legible as a
    # comparison rather than an assertion.
    if (file.exists(pub_density)) {
      pub <- read.csv(pub_density, check.names = FALSE)
      cmp <- bind_rows(pub %>% mutate(population = "All granules"),
                       den %>% mutate(population = "Multi-gene")) %>%
        filter(brain_area %in% area_order, sample %in% c("WT", "AD")) %>%
        mutate(brain_area = factor(brain_area, levels = area_order),
               sample = factor(sample, levels = c("WT", "AD")))
      p <- ggplot(cmp %>% filter(subtype %in% c("overall", "pre-syn", "post-syn", "dendrites",
                                                "mixed")),
                  aes(x = brain_area, y = density, fill = sample)) +
        geom_col(position = position_dodge(width = 0.8), width = 0.7, colour = "#3b3b3b") +
        facet_grid(subtype ~ population, scales = "free_y") +
        scale_fill_manual(values = fill_colors) +
        labs(x = NULL, y = "Density", fill = NULL,
             caption = paste("Left: published full granule set. Right: multi-gene subset.",
                             "Absolute densities differ by construction; the WT-vs-AD",
                             "direction is what carries over.")) +
        theme_classic() +
        theme(axis.text.x = element_text(size = 10, angle = 45, hjust = 1),
              strip.background = element_blank(), legend.position = "bottom",
              plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
      ggsave(file.path(fig_dir, "granule_density_all_vs_multigene.jpeg"), p,
             width = 11, height = 11, dpi = dpi)
      message("  wrote granule_density_all_vs_multigene.jpeg")
    } else {
      message("  [note] published density table not found; skipping the side-by-side panel")
    }
  }

  if (need(f_comp, "2")) {
    comp <- read.csv(f_comp, check.names = FALSE) %>%
      mutate(subtype = factor(subtype, levels = intersect(subtype_order, subtype)))
    p <- ggplot(comp, aes(x = subtype, y = fraction, fill = subtype)) +
      geom_col(colour = "#3b3b3b", width = 0.7, show.legend = FALSE) +
      scale_fill_brewer(palette = "Set2") +
      scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
      labs(x = NULL, y = "Fraction of multi-gene granules") +
      theme_classic() +
      theme(axis.text = element_text(size = 12), axis.title = element_text(size = 13))
    ggsave(file.path(fig_dir, "subtype_composition_multigene.jpeg"), p,
           width = 7, height = 5, dpi = dpi)
    message("  wrote subtype_composition_multigene.jpeg")
  }

  # -------------------- all granules vs the multi-gene subset -------------------- #
  # The composition counterpart to granule_density_all_vs_multigene.jpeg: same two populations and
  # the same question, composition rather than density. Both label tables are already on disk, so
  # nothing is recomputed and the notebook does not need re-running.
  f_lab_new <- file.path(a2a_dir, "granule_subtype_labels_multigene.parquet")

  if (need(pub_subtype_labels, "2") && need(f_lab_new, "2")) {
    POP_ALL <- "All granules"
    POP_MG  <- "Multi-gene (>= 3 genes)"

    read_labels <- function(path, pop_label) {
      read_parquet(path, col_select = c("sample", "granule_subtype_manual_simple")) %>%
        mutate(population = pop_label,
               # The published tables store the full dataset name, e.g. MERSCOPE_WT_1.
               sample_simple = ifelse(grepl("WT", as.character(sample)), "WT", "AD"),
               granule_subtype_manual_simple = as.character(granule_subtype_manual_simple))
    }

    labs_df <- bind_rows(read_labels(pub_subtype_labels, POP_ALL),
                         read_labels(f_lab_new, POP_MG)) %>%
      mutate(
        # Same collapse as the benchmarking panels: anything outside the canonical set becomes
        # "others" rather than vanishing, so every bar still sums to 100%.
        granule_subtype_manual_simple = if_else(
          granule_subtype_manual_simple %in% composition_order,
          granule_subtype_manual_simple, "others"),
        granule_subtype_manual_simple = factor(granule_subtype_manual_simple,
                                               levels = composition_order),
        population = factor(population, levels = c(POP_ALL, POP_MG)),
        sample_simple = factor(sample_simple, levels = c("WT", "AD")))

    comp_pop <- labs_df %>%
      count(population, granule_subtype_manual_simple, name = "n") %>%
      group_by(population) %>%
      mutate(pct = 100 * n / sum(n)) %>%
      ungroup()

    write.csv(comp_pop, file.path(fig_dir, "subtype_composition_all_vs_multigene.csv"),
              row.names = FALSE)
    message("  wrote subtype_composition_all_vs_multigene.csv")

    # ---- cross-checks ----
    # These bars are counted from the label parquets, whereas the response document quotes
    # subtype_composition.csv and the read-strata table. A drift between those sources -- one file
    # regenerated and another not -- would be completely invisible in the rendered figure, so
    # assert the agreement instead of trusting it.
    if (file.exists(f_comp)) {
      chk <- read.csv(f_comp, check.names = FALSE)
      got <- comp_pop %>%
        filter(population == POP_MG) %>%
        transmute(subtype = as.character(granule_subtype_manual_simple), got = pct / 100)
      j <- merge(chk, got, by = "subtype")
      stopifnot(nrow(j) == nrow(chk))
      bad <- j$subtype[abs(j$fraction - j$got) > 1e-6]
      if (length(bad))
        stop("multi-gene bar disagrees with subtype_composition.csv for: ",
             paste(bad, collapse = ", "))
      message("  cross-check OK: multi-gene bar matches subtype_composition.csv")
    }

    f_strata_summ <- file.path(strata_dir, "readstrata_summary.csv")
    if (file.exists(f_strata_summ)) {
      # Pool on the COUNT column. Terciles hold unequal numbers of granules per sample, so
      # averaging the `fraction` column would silently give the wrong reference.
      pooled <- read.csv(f_strata_summ, check.names = FALSE) %>%
        group_by(subtype = granule_subtype_manual_simple) %>%
        summarise(n = sum(n), .groups = "drop") %>%
        mutate(expect = n / sum(n)) %>%
        dplyr::select(subtype, expect)
      got <- comp_pop %>%
        filter(population == POP_ALL) %>%
        transmute(subtype = as.character(granule_subtype_manual_simple), got = pct / 100)
      j <- merge(pooled, got, by = "subtype")
      stopifnot(nrow(j) == nrow(pooled))
      bad <- j$subtype[abs(j$expect - j$got) > 1e-6]
      if (length(bad))
        stop("all-granule bar disagrees with pooled readstrata_summary.csv for: ",
             paste(bad, collapse = ", "))
      message("  cross-check OK: all-granule bar matches pooled readstrata_summary.csv")
    }

    message("  label provenance: the all-granule bar carries the PUBLISHED clustering; the ",
            "multi-gene bar carries an independent k-means re-clustering of the subset with its ",
            "own manual mapping. Subtypes correspond by marker interpretation, not by cluster id.")

    n_by_pop <- comp_pop %>% group_by(population) %>% summarise(n = sum(n), .groups = "drop")
    x_labels <- setNames(
      sprintf("%s\n(n = %s)", as.character(n_by_pop$population),
              format(n_by_pop$n, big.mark = ",", trim = TRUE)),
      as.character(n_by_pop$population))

    composition_bars(comp_pop, file.path(fig_dir, "subtype_composition_all_vs_multigene.jpeg"),
                     x_labels)

    # Companion panel: the same two bars per sample. The published WT-vs-AD direction survives the
    # restriction (pre-syn higher in WT, dendrites higher in AD, in both populations), which is the
    # claim A2a rests on and is not visible once the samples are pooled.
    if (COMPOSITION_BY_SAMPLE) {
      comp_smp <- labs_df %>%
        count(sample_simple, population, granule_subtype_manual_simple, name = "n") %>%
        group_by(sample_simple, population) %>%
        mutate(pct = 100 * n / sum(n)) %>%
        ungroup()
      write.csv(comp_smp,
                file.path(fig_dir, "subtype_composition_all_vs_multigene_by_sample.csv"),
                row.names = FALSE)
      message("  wrote subtype_composition_all_vs_multigene_by_sample.csv")
      # n differs per facet, so the x labels drop it here rather than showing a pooled figure.
      composition_bars(comp_smp,
                       file.path(fig_dir,
                                 "subtype_composition_all_vs_multigene_by_sample.jpeg"),
                       x_labels = setNames(c("All\ngranules", "Multi-gene"), c(POP_ALL, POP_MG)),
                       facet = "sample_simple", width = 10, height = 6)
    }
  }
}


# ==============================================================================================
# 3. Neuropil-microdomain DE -> GSEA
# ==============================================================================================
#
# NAMESPACE NOTE (inherited from A1_figures.R): org.Mm.eg.db is deliberately NOT loaded --
# AnnotationDbi's S4 `select` masks dplyr::select. Everything below is dplyr:: qualified as a
# guard. The GSEA core is copied verbatim from code/figures_response.Rmd:561-700; only the file
# discovery around it is new, so a run with several SUBDOMAIN_PAIRS needs no edit here.
#
# Per DE table this section writes, matching the published set one for one:
#   <stem>_GSEA.csv
#   <stem>_target_GSEA.jpeg   <stem>_reference_GSEA.jpeg                      [NES dotplots]
#   <stem>_{positive,negative}_chord_diagram.jpeg  + _chord_legend.jpeg       [RUN_GSEA_CHORD]

# Gene x pathway chord diagram -- the mcDETECT pipeline's own view of a GSEA result.
#
# Copied VERBATIM from R2_revision/baysor_ssam_merscope/postproc/A1_figures.R:947-1037, which in
# turn copied it verbatim from code/figures_response.Rmd:315-430 (the Rmd is a document, not a
# sourceable library, so it cannot be imported). Keeping it byte-identical is the point: the
# published chord JPEGs under output/MERSCOPE_WT_AD_comparison/neuropil_subdomains_Isocortex_50/
# came out of this code, and the multi-gene panels are only comparable to them if nothing drifts.
#
# Every call is namespace-qualified, so this needs no library() of its own -- which matters here,
# because section 3 loads clusterProfiler (see the NAMESPACE NOTE above).
#
# Shape of the figure: genes become grey sectors carrying rotated labels, the top `top_n` pathways
# become coloured sectors left UNLABELLED, and each ribbon links a pathway to one of its
# leading-edge genes. Pathway names go to a separate _chord_legend.jpeg, which is why the diagram
# itself carries no legend. Ribbons are unweighted -- one gene-pathway pair is one link -- so
# ribbon width reads as "how many leading-edge genes", never as fold change.
make_gsea_chord <- function(gsea_df, direction = c("positive", "negative"), out_prefix, top_n = 20) {

  direction <- match.arg(direction)

  if (direction == "positive") {
    plot_df <- gsea_df %>%
      filter(NES > 0) %>%
      arrange(p.adjust, desc(NES)) %>%
      slice_head(n = top_n)
  } else {
    plot_df <- gsea_df %>%
      filter(NES < 0) %>%
      arrange(p.adjust, NES) %>%
      slice_head(n = top_n)
  }

  if (nrow(plot_df) == 0) {
    message("No pathways available for chord diagram: ", out_prefix, " [", direction, "]")
    return(invisible(NULL))
  }

  df_long <- plot_df %>%
    dplyr::select(Description, core_enrichment) %>%
    dplyr::rename(Pathway = Description) %>%
    dplyr::filter(!is.na(core_enrichment), core_enrichment != "") %>%
    dplyr::mutate(core_enrichment = strsplit(core_enrichment, "/", fixed = TRUE)) %>%
    tidyr::unnest(core_enrichment) %>%
    dplyr::rename(Gene = core_enrichment) %>%
    dplyr::filter(!is.na(Gene), Gene != "") %>%
    dplyr::mutate(Gene = stringi::stri_trans_totitle(Gene),
                  Pathway = as.character(Pathway)) %>%
    dplyr::distinct(Gene, Pathway)

  if (nrow(df_long) == 0) {
    message("No gene-pathway pairs available for chord diagram: ", out_prefix, " [", direction, "]")
    return(invisible(NULL))
  }

  pathways <- unique(df_long$Pathway)
  genes <- unique(df_long$Gene)

  pathway_colors <- setNames(rep(tab20_colors, length.out = length(pathways)), pathways)
  grid_colors <- c(setNames(rep("grey70", length(genes)), genes), pathway_colors)

  jpeg(paste0(out_prefix, "_", direction, "_chord_diagram.jpeg"),
       width = 4000, height = 4000, res = 500)

  circlize::circos.clear()
  circlize::circos.par(gap.degree = 3)

  circlize::chordDiagram(
    x = df_long,
    order = c(genes, pathways),
    grid.col = grid_colors,
    col = pathway_colors[df_long$Pathway],
    transparency = 0.35,
    annotationTrack = "grid",
    preAllocateTracks = list(track.height = 0.14)
  )

  circlize::circos.trackPlotRegion(
    track.index = 1,
    panel.fun = function(x, y) {
      sector.name <- circlize::get.cell.meta.data("sector.index")
      xlim <- circlize::get.cell.meta.data("xlim")
      if (!(sector.name %in% pathways)) {
        circlize::circos.text(x = mean(xlim), y = 0.05, labels = sector.name,
                              facing = "clockwise", niceFacing = TRUE,
                              adj = c(0, 0.5), cex = 0.45)
      }
    },
    bg.border = NA
  )

  dev.off()
  circlize::circos.clear()

  jpeg(paste0(out_prefix, "_", direction, "_chord_legend.jpeg"),
       width = 3000, height = 3000, res = 500)
  legend_obj <- ComplexHeatmap::Legend(
    labels = names(pathway_colors),
    title = paste("Pathways (", direction, " NES)", sep = ""),
    legend_gp = grid::gpar(fill = pathway_colors),
    ncol = 1
  )
  ComplexHeatmap::draw(legend_obj, x = grid::unit(0.5, "npc"), y = grid::unit(0.5, "npc"),
                       just = c("center", "center"))
  dev.off()

  invisible(NULL)
}


gsea_dotplot <- function(df, out_file, x_var, width = 10, height = 6) {
  p <- ggplot(df, aes(x = .data[[x_var]], y = Description)) +
    geom_point(aes(size = setSize, fill = p.adjust), shape = 21, stroke = 0.8, color = "black") +
    scale_fill_distiller(palette = "Reds") +
    labs(title = " ", x = "Normalized Enrichment Score (NES)", y = NULL,
         fill = "Adjusted p-value", size = "Gene set size") +
    theme_bw() +
    theme(axis.text.x = element_text(size = 15),
          axis.text.y = element_text(size = 13),
          axis.title = element_text(size = 15),
          legend.title = element_text(size = 13),
          legend.text = element_text(size = 13))
  ggsave(out_file, p, width = width, height = height, dpi = dpi)
}

run_gsea_on <- function(de_csv, out_dir, term2gene) {
  stem <- sub("\\.csv$", "", basename(de_csv))
  df <- read.csv(de_csv, check.names = FALSE)

  gene_list <- df %>%
    dplyr::select(names, scores) %>%
    dplyr::filter(!is.na(names), !is.na(scores)) %>%
    dplyr::group_by(names) %>%
    dplyr::summarise(rank_metric = mean(scores), .groups = "drop") %>%
    dplyr::arrange(dplyr::desc(rank_metric), names)

  gene_vector <- gene_list$rank_metric
  names(gene_vector) <- gene_list$names
  gene_vector <- gene_vector + runif(length(gene_vector), -1e-10, 1e-10)  # deterministic ties
  gene_vector <- sort(gene_vector, decreasing = TRUE)

  overlap_n <- sum(names(gene_vector) %in% term2gene$gene_symbol)
  message("  ", stem, ": ", length(gene_vector), " genes, ", overlap_n, " in TERM2GENE")
  if (length(gene_vector) < 50 || overlap_n < 20) {
    message("  [skip] ranking or overlap too small")
    return(invisible(NULL))
  }

  gsea_res <- tryCatch(
    clusterProfiler::GSEA(geneList = gene_vector, TERM2GENE = term2gene,
                          minGSSize = 10, maxGSSize = 500, pvalueCutoff = 0.25,
                          pAdjustMethod = "BH", eps = 1e-10, seed = TRUE,
                          by = "fgsea", verbose = FALSE),
    error = function(e) {
      message("  [skip] GSEA failed: ", e$message)
      NULL
    })
  if (is.null(gsea_res)) return(invisible(NULL))

  result_df <- as.data.frame(gsea_res)
  if (nrow(result_df) == 0) {
    message("  [skip] no enriched pathways")
    return(invisible(NULL))
  }

  result_df <- result_df %>%
    dplyr::mutate(label_clean = stringr::str_remove(ID, "^GOBP_"),
                  label_clean = stringr::str_replace_all(label_clean, "_", " "),
                  label_clean = stringr::str_to_sentence(label_clean),
                  Description = label_clean)
  write.csv(result_df %>% dplyr::arrange(dplyr::desc(NES)),
            file.path(out_dir, paste0(stem, "_GSEA.csv")))

  top_target <- result_df %>%
    dplyr::filter(NES > 0) %>%
    dplyr::arrange(p.adjust, dplyr::desc(NES)) %>%
    dplyr::slice_head(n = 10) %>%
    dplyr::mutate(Description = forcats::fct_reorder(Description, NES))
  top_reference <- result_df %>%
    dplyr::filter(NES < 0) %>%
    dplyr::arrange(p.adjust, NES) %>%
    dplyr::slice_head(n = 10) %>%
    dplyr::mutate(NES_plot = abs(NES),
                  Description = forcats::fct_reorder(Description, NES_plot))

  if (nrow(top_target) > 0)
    gsea_dotplot(top_target, file.path(out_dir, paste0(stem, "_target_GSEA.jpeg")), "NES")
  if (nrow(top_reference) > 0)
    gsea_dotplot(top_reference, file.path(out_dir, paste0(stem, "_reference_GSEA.jpeg")), "NES_plot")
  # -------------------- chord diagrams -------------------- #
  # `out_prefix` must carry out_dir: A2's run_gsea_on() takes `stem` as a bare basename and routes
  # every write through out_dir, unlike A1 where `stem` is already a full path. Passing `stem`
  # alone would silently drop 4 JPEGs per table into the working directory.
  if (RUN_GSEA_CHORD) {
    make_gsea_chord(result_df, "positive", file.path(out_dir, stem), top_n = 20)
    make_gsea_chord(result_df, "negative", file.path(out_dir, stem), top_n = 20)
  }

  message("  wrote ", stem, "_GSEA.csv + dotplots",
          if (RUN_GSEA_CHORD) " + chord diagrams" else "")
  invisible(result_df)
}

if (RUN_GSEA) {
  message("[3] microdomain GSEA")
  if (!dir.exists(sub_dir)) {
    message("  [skip 3] missing: ", sub_dir, " -- run A2a section 5 first")
  } else {
    suppressPackageStartupMessages({
      library(clusterProfiler)
      library(forcats)
      library(msigdbr)
      library(stringr)
    })

    msig_mouse <- msigdbr(species = "Mus musculus", category = "C5", subcategory = "BP")
    term2gene <- msig_mouse %>% dplyr::select(gs_name, gene_symbol) %>% dplyr::distinct()
    set.seed(42)

    # Discover every DE table the notebook wrote. `_GSEA.csv` is excluded explicitly: subdomain
    # names contain spaces, so the greedy `.*_vs_.*` pattern would otherwise match GSEA outputs
    # from a previous run and re-score them.
    de_files <- list.files(sub_dir, pattern = "_DE_genes_.*_vs_.*\\.csv$", full.names = TRUE)
    de_files <- de_files[!grepl("_GSEA\\.csv$", de_files)]
    if (!is.null(GSEA_PAIRS)) {
      de_files <- de_files[Reduce(`|`, lapply(GSEA_PAIRS, function(p) grepl(p, de_files, fixed = TRUE)))]
    }
    message("  ", length(de_files), " DE table(s) found")
    for (f in de_files) run_gsea_on(f, sub_dir, term2gene)

    # ---------------------------------------------------------------------------------------
    # Subdomain correspondence -- run BEFORE the anchor comparison, because it is what makes
    # that comparison readable.
    #
    # The multi-gene run recomputes `subdomain_kmeans`, so its labels are arbitrary: the
    # published cosmetic relabel_map (7_neuropil_subdomains.ipynb cell 9) was fitted to the
    # published clustering and does not transfer. Whether an NES from this run can be set beside
    # a published one therefore depends on which multi-gene subdomain corresponds to which
    # published one -- and both runs share the same inherited spot grid, so that is directly
    # measurable by spot_id rather than something to assume.
    # ---------------------------------------------------------------------------------------
    pub_labels <- file.path(pub_subdomain, "4_hard_normalized_cluster_labels.parquet")
    new_labels <- file.path(sub_dir, "4_hard_normalized_cluster_labels.parquet")
    corr_best <- NULL

    if (file.exists(pub_labels) && file.exists(new_labels)) {
      lp <- read_parquet(pub_labels) %>%
        dplyr::transmute(spot_id = as.character(spot_id),
                         published = as.character(subdomain_kmeans))
      ln <- read_parquet(new_labels) %>%
        dplyr::transmute(spot_id = as.character(spot_id),
                         multigene = as.character(subdomain_kmeans))
      j <- dplyr::inner_join(lp, ln, by = "spot_id")
      message("  subdomain correspondence: ", nrow(j), " shared spots (",
              nrow(lp), " published, ", nrow(ln), " multi-gene)")

      ct <- j %>%
        dplyr::count(multigene, published, name = "n_spots") %>%
        dplyr::group_by(multigene) %>%
        dplyr::mutate(frac_of_multigene = n_spots / sum(n_spots)) %>%
        dplyr::ungroup()

      corr_best <- ct %>%
        dplyr::group_by(multigene) %>%
        dplyr::slice_max(n_spots, n = 1, with_ties = FALSE) %>%
        dplyr::ungroup() %>%
        dplyr::transmute(multigene,
                         best_published_match = published,
                         best_match_frac = round(frac_of_multigene, 4))

      write.csv(ct %>% dplyr::left_join(corr_best, by = "multigene") %>%
                  dplyr::mutate(n_shared_spots = nrow(j)),
                file.path(fig_dir, "subdomain_correspondence.csv"), row.names = FALSE)
      message("  wrote subdomain_correspondence.csv")
      for (i in seq_len(nrow(corr_best)))
        message("    multi-gene ", corr_best$multigene[i], " -> published ",
                corr_best$best_published_match[i], " (",
                round(100 * corr_best$best_match_frac[i], 1), "% of its spots)")

      p <- ggplot(ct, aes(x = published, y = multigene, fill = frac_of_multigene)) +
        geom_tile() +
        geom_text(aes(label = scales::comma(n_spots)), size = 3.5, colour = "grey20") +
        scale_fill_distiller(palette = "Reds", direction = 1, labels = scales::percent) +
        labs(x = "Published subdomain (all granules)", y = "Multi-gene subdomain",
             fill = "Share of the\nmulti-gene subdomain",
             caption = paste("Same inherited 50 um spot grid, joined by spot_id. A near-diagonal",
                             "mapping means the microdomain partition itself survives the",
                             "multi-gene restriction, not only the differential expression.")) +
        theme_classic() +
        theme(axis.text = element_text(size = 12), axis.title = element_text(size = 13),
              plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
      ggsave(file.path(fig_dir, "subdomain_correspondence.jpeg"), p,
             width = 7.5, height = 5.5, dpi = dpi)
      message("  wrote subdomain_correspondence.jpeg")
    } else {
      message("  [note] cluster-label parquet missing on one side; skipping the correspondence check")
    }

    # The published Subdomain 1-vs-2 GSEA is the anchor: the question is whether the multi-gene
    # subset recovers the same pathways, so both must be on the table.
    #
    # Subdomain names contain spaces, so the contrast regex is anchored on `_DE_genes_` and
    # `_GSEA.csv$` and only then splits on `_vs_`; a greedy unanchored pattern would mis-split.
    parse_contrast <- function(fn) {
      m <- stringr::str_match(fn,
        "^(granule|cell|ambient)_DE_genes_(.+)_vs_(.+)_GSEA\\.csv$")
      list(layer = m[, 2], target = m[, 3], reference = m[, 4])
    }

    pub_gsea <- list.files(pub_subdomain, pattern = "_GSEA\\.csv$", full.names = TRUE)
    if (length(pub_gsea)) {
      new_gsea <- list.files(sub_dir, pattern = "_GSEA\\.csv$", full.names = TRUE)

      # `write.csv()` defaults to row.names = TRUE, so every GSEA table -- ours AND the published
      # ones, which we do not control -- carries a leading unnamed ("") column. `check.names =
      # FALSE` preserves that empty name, and any dplyr verb that CONSTRUCTS a new frame from it
      # (transmute, and select in recent versions) refuses to run. Base-R `$` extraction is
      # indifferent to it. Same handling as A1_figures.R:1225 and :1364-1371 -- do not "fix" this
      # by dropping check.names = FALSE.
      read_terms <- function(f, tag) {
        d <- suppressWarnings(try(read.csv(f, check.names = FALSE), silent = TRUE))
        if (inherits(d, "try-error")) {
          message("  [note] unreadable, skipped: ", basename(f))
          return(NULL)
        }
        if (!all(c("ID", "NES", "p.adjust") %in% colnames(d))) {
          message("  [note] not a GSEA table, skipped: ", basename(f))
          return(NULL)
        }
        cx <- parse_contrast(basename(f))
        data.frame(source = tag, file = basename(f),
                   layer = cx$layer, target = cx$target, reference = cx$reference,
                   ID = d$ID, NES = d$NES, p.adjust = d$p.adjust,
                   stringsAsFactors = FALSE)
      }

      cmp <- dplyr::bind_rows(
        dplyr::bind_rows(lapply(pub_gsea, read_terms, tag = "published_all_granules")),
        dplyr::bind_rows(lapply(new_gsea, read_terms, tag = "multigene")))
      write.csv(cmp, file.path(fig_dir, "gsea_terms_published_vs_multigene.csv"), row.names = FALSE)
      message("  wrote gsea_terms_published_vs_multigene.csv")

      # Say out loud whether the two contrasts point the same way. Getting this wrong would flip
      # every NES sign in the comparison, and it is invisible in the table itself.
      ours <- cmp %>% dplyr::filter(source == "multigene") %>%
        dplyr::distinct(target, reference)
      if (!is.null(corr_best) && nrow(ours) == 1) {
        map_to <- function(x) {
          i <- match(x, corr_best$multigene)
          if (is.na(i)) NA_character_ else corr_best$best_published_match[i]
        }
        t_pub <- map_to(ours$target[1]); r_pub <- map_to(ours$reference[1])
        message("  contrast: ", ours$target[1], " vs ", ours$reference[1],
                "  ->  published ", t_pub, " vs ", r_pub)
        if (identical(t_pub, "Subdomain 1") && identical(r_pub, "Subdomain 2")) {
          message("  direction MATCHES the published Subdomain 1 vs 2 -- NES signs are directly comparable.")
        } else if (identical(t_pub, "Subdomain 2") && identical(r_pub, "Subdomain 1")) {
          message("  direction is REVERSED relative to the published contrast -- NES signs are flipped.")
        } else {
          message("  this pair does not map onto the published Subdomain 1 vs 2; compare with care.")
        }
      }
    } else {
      message("  [note] no published GSEA tables found for the anchor comparison")
    }
  }
}


# ==============================================================================================
# 4. Read-count terciles
# ==============================================================================================

if (RUN_READSTRATA) {
  message("[4] read-count terciles")

  f_cplx <- file.path(strata_dir, "readstrata_complexity.csv")
  f_comp <- file.path(strata_dir, "readstrata_summary.csv")
  f_den  <- file.path(strata_dir, "readstrata_density.csv")
  terc <- c("low", "mid", "high")

  if (need(f_comp, "4")) {
    comp <- read.csv(f_comp, check.names = FALSE) %>%
      mutate(read_tercile = factor(read_tercile, levels = terc),
             sample_simple = factor(sample_simple, levels = c("WT", "AD")))
    p <- ggplot(comp, aes(x = read_tercile, y = fraction,
                          fill = granule_subtype_manual_simple)) +
      geom_col(colour = "#3b3b3b", width = 0.7) +
      facet_wrap(~ sample_simple) +
      scale_fill_brewer(palette = "Set2") +
      scale_y_continuous(labels = scales::percent, expand = c(0, 0)) +
      labs(x = "Read-count tercile", y = "Subtype composition", fill = NULL,
           caption = paste("Subtype labels are the published ones, held fixed, so this measures",
                           "read depth alone.")) +
      theme_classic() +
      theme(strip.background = element_blank(), legend.position = "bottom",
            plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
    ggsave(file.path(fig_dir, "readstrata_composition.jpeg"), p, width = 9, height = 5, dpi = dpi)
    message("  wrote readstrata_composition.jpeg")
  }

  if (need(f_den, "4")) {
    den <- read.csv(f_den, check.names = FALSE) %>%
      filter(brain_area %in% area_order, sample %in% c("WT", "AD")) %>%
      mutate(read_tercile = factor(read_tercile, levels = terc),
             brain_area = factor(brain_area, levels = area_order),
             sample = factor(sample, levels = c("WT", "AD")))

    # Per-tercile WT/AD bars for each subtype -- the panel that answers "not a low-count artifact".
    for (s in intersect(subtype_order, unique(den$subtype))) {
      d <- den %>% filter(subtype == s)
      if (nrow(d) == 0) next
      p <- ggplot(d, aes(x = brain_area, y = density, fill = sample)) +
        geom_col(position = position_dodge(width = 0.8), width = 0.7, colour = "#3b3b3b") +
        geom_errorbar(aes(ymin = density_ci_low, ymax = density_ci_high),
                      position = position_dodge(width = 0.8), colour = "#3b3b3b",
                      width = 0.2, linewidth = 0.5) +
        geom_text(aes(y = density_ci_high, label = p_bonf_star),
                  position = position_dodge(width = 0.8), vjust = -0.4, size = 3.5) +
        facet_wrap(~ read_tercile, ncol = 1, scales = "free_y") +
        scale_fill_manual(values = fill_colors) +
        labs(x = NULL, y = "Density", fill = NULL,
             caption = paste0("Subtype: ", s,
                              ". If the WT-AD difference were a low-count artifact it would be",
                              " confined to the low tercile.")) +
        theme_classic() +
        theme(strip.background = element_blank(), legend.position = "bottom",
              axis.text.x = element_text(size = 11),
              plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
      ggsave(file.path(fig_dir, paste0("readstrata_density_", s, ".jpeg")), p,
             width = 9, height = 9, dpi = dpi)
      message("  wrote readstrata_density_", s, ".jpeg")
    }
  }

  if (need(f_cplx, "4")) message("  tercile edges and complexity: ", f_cplx)
}


# ==============================================================================================
# 5. A2b -- permutation null vs real embedding
# ==============================================================================================
#
# An arm here is one COMBINED WT+AD object -- 1 real + 5 permuted -- because that is the
# embedding the paper reports. Each permuted arm carries three metric series: itself at full n,
# and a size-matched pair (`matched_perm_seed<s>` / `matched_real_seed<s>`) cut to
# min(n_real, n_perm). The **size-matched panel is the headline**: silhouette and ARI stability
# both depend on n, so the full-n panel alone would invite the reading that the null looks less
# structured merely because it holds less data.

if (RUN_A2B) {
  message("[5] permutation null")

  f_metrics <- file.path(a2b_dir, "a2b_metrics.csv")
  f_status  <- file.path(a2b_dir, "a2b_status.csv")
  f_detect  <- file.path(a2b_dir, "a2b_detection_summary.csv")
  f_dist    <- file.path(a2b_dir, "a2b_distributions.parquet")

  # Any arm whose null collapsed below MIN_EMBED_N is recorded, not dropped -- surface it.
  if (need(f_status, "5")) {
    st <- read.csv(f_status, check.names = FALSE)
    skipped <- st %>% filter(status != "embedded")
    if (nrow(skipped)) {
      message("  NOTE ", nrow(skipped), " series were not embedded (too few granules):")
      for (i in seq_len(nrow(skipped)))
        message("    ", skipped$series[i], " (n = ", skipped$n_obs[i], "): ", skipped$reason[i])
    }
    write.csv(st, file.path(fig_dir, "a2b_series_status.csv"), row.names = FALSE)
  }

  if (need(f_metrics, "5")) {
    met <- read.csv(f_metrics, check.names = FALSE) %>%
      mutate(condition = factor(condition, levels = c("real", "permuted")),
             panel = factor(ifelse(matched, "Size-matched (headline)", "Full n"),
                            levels = c("Size-matched (headline)", "Full n")))

    # Mean with a min-max ribbon over arms; plotting each of the 5 permuted seeds separately
    # would say the same thing less legibly.
    band <- met %>%
      group_by(panel, condition, n_clusters) %>%
      summarise(across(c(silhouette_score, ari_stability_mean, inertia),
                       list(mean = ~mean(.x, na.rm = TRUE),
                            lo = ~min(.x, na.rm = TRUE),
                            hi = ~max(.x, na.rm = TRUE))),
                n_arms = dplyr::n(), n_obs = mean(n_obs), .groups = "drop")

    n_note <- band %>%
      group_by(panel, condition) %>%
      summarise(n_obs = round(mean(n_obs)), .groups = "drop") %>%
      mutate(txt = paste0(condition, " n=", format(n_obs, big.mark = ","))) %>%
      group_by(panel) %>%
      summarise(txt = paste(txt, collapse = "; "), .groups = "drop")

    for (m in c("silhouette_score", "ari_stability_mean")) {
      d <- band %>%
        transmute(panel, condition, n_clusters,
                  mean = .data[[paste0(m, "_mean")]],
                  lo = .data[[paste0(m, "_lo")]],
                  hi = .data[[paste0(m, "_hi")]])
      y_lab <- if (m == "silhouette_score") "Silhouette score" else
        "Cluster stability (mean pairwise ARI)"
      p <- ggplot(d, aes(x = n_clusters, y = mean, colour = condition, fill = condition)) +
        geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.2, colour = NA) +
        geom_line(linewidth = 0.8) +
        geom_point(size = 1.6) +
        geom_vline(xintercept = 15, linetype = "dashed", colour = "grey50") +
        geom_text(data = n_note, aes(x = Inf, y = Inf, label = txt), inherit.aes = FALSE,
                  hjust = 1.05, vjust = 1.6, size = 3.2, colour = "grey30") +
        facet_wrap(~ panel) +
        scale_colour_manual(values = cond_colors) +
        scale_fill_manual(values = cond_colors) +
        labs(x = "Number of granule clusters (k)", y = y_lab, colour = NULL, fill = NULL,
             caption = paste("Combined WT+AD embedding. Ribbon spans arms; dashed line marks the",
                             "published k = 15. Left panel equalises n between real and permuted;",
                             "right panel shows each arm at its own size. Both arms use",
                             "n_init = 20 and the same silhouette subsample, so they are",
                             "comparable with each other but not with the published",
                             "benchmark_clustering_results.csv.")) +
        theme_classic() +
        theme(strip.background = element_blank(), legend.position = "bottom",
              axis.text = element_text(size = 12), axis.title = element_text(size = 13),
              plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
      ggsave(file.path(fig_dir, paste0("a2b_", m, ".jpeg")), p, width = 10, height = 5, dpi = dpi)
      message("  wrote a2b_", m, ".jpeg")
    }

    # The headline number: structure at the published k.
    at_k <- met %>%
      filter(n_clusters == 15) %>%
      dplyr::select(arm, series, condition, matched, seed, silhouette_score,
                    ari_stability_mean, n_obs)
    write.csv(at_k, file.path(fig_dir, "a2b_structure_at_k15.csv"), row.names = FALSE)
    message("  wrote a2b_structure_at_k15.csv")
  }

  if (need(f_detect, "5")) {
    det <- read.csv(f_detect, check.names = FALSE) %>%
      mutate(condition = factor(condition, levels = c("real", "permuted")),
             sample = factor(sample, levels = c("WT", "AD")))

    # The count difference is a result in its own right, not a nuisance -- show it directly.
    p <- ggplot(det, aes(x = sample, y = n_fine, fill = condition)) +
      geom_boxplot(outlier.size = 0.8, colour = "#3b3b3b", alpha = 0.8) +
      scale_fill_manual(values = cond_colors) +
      scale_y_continuous(labels = scales::comma) +
      labs(x = NULL, y = "Granules detected (fine pass)", fill = NULL,
           caption = paste("One box per arm set. The shuffle preserves the marker transcript",
                           "count exactly, so any difference here comes from where those",
                           "markers land, not how many there are.")) +
      theme_classic() +
      theme(legend.position = "bottom", axis.text = element_text(size = 12),
            plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
    ggsave(file.path(fig_dir, "a2b_granule_counts.jpeg"), p, width = 7, height = 5, dpi = dpi)
    message("  wrote a2b_granule_counts.jpeg")

    if (all(c("n_rough", "n_fine") %in% names(det)) && any(!is.na(det$n_rough))) {
      d <- det %>%
        dplyr::select(sample, condition, arm, n_rough, n_fine) %>%
        pivot_longer(c(n_rough, n_fine), names_to = "stage", values_to = "n") %>%
        mutate(stage = factor(stage, levels = c("n_rough", "n_fine"),
                              labels = c("Rough (no filters)", "Fine (size + soma + NC)")))
      p <- ggplot(d, aes(x = stage, y = n, fill = condition)) +
        geom_boxplot(outlier.size = 0.8, colour = "#3b3b3b", alpha = 0.8) +
        facet_wrap(~ sample) +
        scale_fill_manual(values = cond_colors) +
        scale_y_continuous(labels = scales::comma) +
        labs(x = NULL, y = "Detections", fill = NULL,
             caption = paste("Permuted markers inherit the panel-wide, soma-dominated",
                             "distribution, so the in-soma filter is where they are expected",
                             "to fall away.")) +
        theme_classic() +
        theme(strip.background = element_blank(), legend.position = "bottom",
              axis.text = element_text(size = 12),
              plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
      ggsave(file.path(fig_dir, "a2b_detection_counts.jpeg"), p, width = 9, height = 5, dpi = dpi)
      message("  wrote a2b_detection_counts.jpeg")

      p <- ggplot(det %>% filter(!is.na(frac_pass_in_soma)),
                  aes(x = sample, y = frac_pass_in_soma, fill = condition)) +
        geom_boxplot(outlier.size = 0.8, colour = "#3b3b3b", alpha = 0.8) +
        scale_fill_manual(values = cond_colors) +
        scale_y_continuous(labels = scales::percent) +
        labs(x = NULL, y = "Rough detections with in-soma ratio < 0.1", fill = NULL,
             caption = paste("Evaluated post hoc on the unfiltered (rough) set, identically for",
                             "every arm; mcDETECT itself applies this filter before sphere",
                             "merging, so this is not the pipeline's own survival chain.")) +
        theme_classic() +
        theme(legend.position = "bottom", axis.text = element_text(size = 12),
              plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
      ggsave(file.path(fig_dir, "a2b_in_soma_survival.jpeg"), p, width = 7, height = 5, dpi = dpi)
      message("  wrote a2b_in_soma_survival.jpeg")
    } else {
      message("  [note] no rough-pass tables (C.RUN_ROUGH_PASS was off); skipping survival panels")
    }
  }

  if (need(f_dist, "5")) {
    dist_df <- read_parquet(f_dist) %>%
      mutate(condition = factor(condition, levels = c("real", "permuted")))
    for (m in c("sphere_r", "size", "n_genes", "n_reads")) {
      d <- dist_df %>%
        filter(measure == m) %>%
        group_by(sample, condition, bin_lo, bin_hi) %>%
        summarise(frac = mean(frac, na.rm = TRUE), .groups = "drop")
      for (smp in unique(d$sample)) {
        binned_hist(d %>% filter(sample == smp),
                    file.path(fig_dir, paste0("a2b_", m, "_", smp, ".jpeg")),
                    x_lab = m, fill_var = "condition", palette = cond_colors,
                    caption = paste0(smp, ": permuted arms averaged over seeds."))
      }
    }
  }
}

# ==============================================================================================
# 6. A2c -- gene co-occurrence: localization versus co-expression programmes
# ==============================================================================================
#
# Everything here is on the 270 NON-SEED genes. mcDETECT's merge_sphere() merges spheres seeded by
# different markers, so co-occurrence among the 20 seed markers is partly manufactured by
# detection; that arm appears only as a positive control and is labelled as such.
#
# The contrast is the argument. Functional coherence on its own does not separate a granule from
# "any co-expressed transcript cluster" -- coherence is what co-expression looks like. What would
# separate them is WHICH programme organises the co-occurrence: localization groups (pre/post-syn,
# Neuropil, Axons) or co-expression groups (cell-type, region, layer marker sets).

if (RUN_A2C) {
  message("[6] gene co-occurrence")

  f_grp   <- file.path(a2c_dir, "group_enrichment.csv")
  f_pairs <- file.path(a2c_dir, "pair_enrichment.parquet")
  f_drop  <- file.path(a2c_dir, "groups_dropped.csv")

  if (need(f_grp, "6")) {
    grp <- read.csv(f_grp, check.names = FALSE) %>%
      mutate(programme = factor(programme, levels = c("localization", "co-expression")))

    if (file.exists(f_drop)) {
      dr <- read.csv(f_drop, check.names = FALSE)
      if (nrow(dr)) message("  NOTE ", nrow(dr), " group(s) too small to test: ",
                            paste(dr$group, collapse = ", "))
    }

    prim <- grp %>% filter(arm == "all") %>% arrange(median_z) %>%
      mutate(group = factor(group, levels = group))
    p <- ggplot(prim, aes(x = median_z, y = group, fill = programme)) +
      geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50") +
      geom_col(colour = "#3b3b3b", width = 0.75) +
      geom_text(aes(label = q_upper_star,
                    x = median_z + sign(median_z) * 0.15 * max(abs(median_z))),
                size = 3.5, colour = "grey20") +
      scale_fill_manual(values = prog_colors) +
      labs(x = "Median co-occurrence enrichment (z) within group", y = NULL, fill = NULL,
           caption = paste("Non-seed genes only. Null preserves both margins exactly (granule",
                           "complexity and gene detection frequency); the permutation is",
                           "abundance-matched, because rare genes carry systematically higher z",
                           "and the two programmes differ ~5-fold in abundance.",
                           "Stars are BH-adjusted.")) +
      theme_classic() +
      theme(axis.text = element_text(size = 11), axis.title = element_text(size = 13),
            legend.position = "bottom",
            plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
    ggsave(file.path(fig_dir, "a2c_group_enrichment.jpeg"), p, width = 8.5, height = 8, dpi = dpi)
    message("  wrote a2c_group_enrichment.jpeg")

    # Robustness across arms. Isocortex is the check on the regional confound: a group whose
    # signal vanishes there was reporting tissue composition, not granule packaging.
    arms <- intersect(c("all", "Isocortex", "WT", "AD"), unique(grp$arm))
    d <- grp %>% filter(arm %in% arms) %>%
      mutate(arm = factor(arm, levels = arms),
             group = factor(group, levels = levels(prim$group)))
    p <- ggplot(d, aes(x = arm, y = group, fill = median_z)) +
      geom_tile() +
      geom_text(aes(label = sprintf("%.1f", median_z)), size = 2.8, colour = "grey15") +
      scale_fill_distiller(palette = "RdBu", limits = c(-1, 1) * max(abs(d$median_z), na.rm = TRUE)) +
      facet_grid(programme ~ ., scales = "free_y", space = "free_y") +
      labs(x = NULL, y = NULL, fill = "median z",
           caption = paste("Isocortex restricts to one region: groups that are really reporting",
                           "regional or tissue-compartment composition should collapse there.")) +
      theme_classic() +
      theme(axis.text = element_text(size = 10), strip.background = element_blank(),
            plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
    ggsave(file.path(fig_dir, "a2c_group_enrichment_by_arm.jpeg"), p, width = 7, height = 9,
           dpi = dpi)
    message("  wrote a2c_group_enrichment_by_arm.jpeg")

    write.csv(grp %>% group_by(arm, programme) %>%
                summarise(n_groups = dplyr::n(), median_of_medians = median(median_z),
                          n_sig = sum(q_upper < 0.05, na.rm = TRUE), .groups = "drop"),
              file.path(fig_dir, "a2c_programme_summary.csv"), row.names = FALSE)
    message("  wrote a2c_programme_summary.csv")
  }

  # -------------------- GO shared-term test -------------------- #
  # External and unbiased, and it covers every non-seed gene rather than only the annotated ones.
  if (need(f_pairs, "6") && RUN_GSEA) {
    suppressPackageStartupMessages({ library(msigdbr) })
    pr <- read_parquet(f_pairs) %>% filter(arm == "all")

    msig <- msigdbr(species = "Mus musculus", category = "C5", subcategory = "BP")
    sizes <- msig %>% dplyr::count(gs_name, name = "n")
    keep_sets <- sizes$gs_name[sizes$n >= 10 & sizes$n <= 500]     # same bounds as the GSEA above
    m <- msig %>% dplyr::filter(gs_name %in% keep_sets) %>%
      dplyr::select(gs_name, gene_symbol) %>% dplyr::distinct()
    terms <- split(m$gs_name, m$gene_symbol)

    jac <- function(a, b) {
      ta <- terms[[a]]; tb <- terms[[b]]
      if (is.null(ta) || is.null(tb)) return(NA_real_)
      i <- length(intersect(ta, tb)); u <- length(union(ta, tb))
      if (u == 0) NA_real_ else i / u
    }
    pr$go_jaccard <- mapply(jac, pr$gene_i, pr$gene_j)
    pr$go_shared <- pr$go_jaccard > 0
    ok <- !is.na(pr$go_shared)
    message("  GO: ", sum(ok), " of ", nrow(pr), " pairs have both genes annotated; ",
            sum(pr$go_shared[ok]), " share >= 1 BP term")

    if (sum(ok) > 100 && length(unique(pr$go_shared[ok])) == 2) {
      d <- pr[ok, ]
      w <- suppressWarnings(wilcox.test(z ~ go_shared, data = d))
      med <- tapply(d$z, d$go_shared, median)
      # Report the effect, not only the p-value: with ~36k pairs everything is significant.
      message(sprintf("  median z: shares a GO term %.3f vs does not %.3f (delta %+.3f), p = %.3g",
                      med["TRUE"], med["FALSE"], med["TRUE"] - med["FALSE"], w$p.value))
      write.csv(data.frame(median_z_shared = med["TRUE"], median_z_not = med["FALSE"],
                           delta = med["TRUE"] - med["FALSE"], p_wilcox = w$p.value,
                           n_shared = sum(d$go_shared), n_not = sum(!d$go_shared)),
                file.path(fig_dir, "a2c_go_shared_term_test.csv"), row.names = FALSE)

      p <- ggplot(d, aes(x = go_shared, y = z, fill = go_shared)) +
        geom_boxplot(outlier.size = 0.4, colour = "#3b3b3b", alpha = 0.85, show.legend = FALSE) +
        scale_fill_manual(values = c("FALSE" = "#c9c9c9", "TRUE" = unname(prog_colors[1]))) +
        scale_x_discrete(labels = c("FALSE" = "no shared\nGO BP term",
                                    "TRUE" = "shares >= 1\nGO BP term")) +
        coord_cartesian(ylim = quantile(d$z, c(0.005, 0.995), na.rm = TRUE)) +
        labs(x = NULL, y = "Co-occurrence enrichment (z)",
             caption = paste("Non-seed pairs, external GO BP annotation, gene sets of size",
                             "10-500 as in the GSEA above. Effect size is the point; with tens",
                             "of thousands of pairs the p-value is not informative on its own.")) +
        theme_classic() +
        theme(axis.text = element_text(size = 12), axis.title = element_text(size = 13),
              plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
      ggsave(file.path(fig_dir, "a2c_go_shared_term.jpeg"), p, width = 6, height = 5, dpi = dpi)
      message("  wrote a2c_go_shared_term.jpeg")

      # ---- Is the structure concentrated, or spread as a similarity gradient? ----
      # The overall test above is negative -- pairs sharing a GO term have LOWER z. That sits
      # oddly against a top-pair list that is obviously functional (Gad2-Gad1, the neurofilament
      # triplet, Cnp-Sox10), so the two readings have to be separated rather than argued about:
      #   concentrated -> the shared-GO fraction rises in the strongly-enriched stratum while the
      #                   middle stays flat, i.e. a few strong functional modules;
      #   gradient     -> it tracks z monotonically, which the negative test already rules out;
      #   neither      -> claim 2 is simply negative, and the document says so.
      arrow::write_parquet(d[, c("gene_i", "gene_j", "z", "go_shared", "go_jaccard")],
                           file.path(fig_dir, "a2c_pair_go.parquet"))
      message("  wrote a2c_pair_go.parquet (", nrow(d), " annotated pairs)")

      baseline <- mean(d$go_shared)
      z_bonf <- qnorm(0.025 / nrow(pr), lower.tail = FALSE)   # same threshold quoted in the doc

      by_bin <- d %>%
        dplyr::mutate(bin = dplyr::ntile(z, 10)) %>%
        dplyr::group_by(bin) %>%
        dplyr::summarise(stratum = paste0("z decile ", dplyr::first(bin)),
                         n_pairs = dplyr::n(), median_z = median(z),
                         frac_shared_go = mean(go_shared),
                         median_jaccard = median(go_jaccard, na.rm = TRUE), .groups = "drop") %>%
        dplyr::select(-bin)

      strata <- d %>%
        dplyr::mutate(stratum = dplyr::case_when(
          z >  z_bonf ~ "Bonferroni enriched",
          z < -z_bonf ~ "Bonferroni depleted",
          TRUE        ~ "middle")) %>%
        dplyr::group_by(stratum) %>%
        dplyr::summarise(n_pairs = dplyr::n(), median_z = median(z),
                         frac_shared_go = mean(go_shared),
                         median_jaccard = median(go_jaccard, na.rm = TRUE), .groups = "drop")

      out <- dplyr::bind_rows(by_bin, strata) %>%
        dplyr::mutate(baseline_frac_shared_go = baseline,
                      enrichment_over_baseline = frac_shared_go / baseline,
                      z_bonferroni_threshold = z_bonf)
      write.csv(out, file.path(fig_dir, "a2c_go_by_z_bin.csv"), row.names = FALSE)
      message("  wrote a2c_go_by_z_bin.csv (baseline shared-GO fraction ",
              sprintf("%.3f", baseline), ")")
      for (i in seq_len(nrow(strata)))
        message(sprintf("    %-20s n=%6d  median z %+7.2f  shares GO %.3f  (%.2fx baseline)",
                        strata$stratum[i], strata$n_pairs[i], strata$median_z[i],
                        strata$frac_shared_go[i], strata$frac_shared_go[i] / baseline))

      pb <- ggplot(by_bin, aes(x = median_z, y = frac_shared_go)) +
        geom_hline(yintercept = baseline, linetype = "dashed", colour = "grey50") +
        geom_line(colour = unname(prog_colors[1]), linewidth = 0.8) +
        geom_point(size = 2.2, colour = unname(prog_colors[1])) +
        scale_y_continuous(labels = scales::percent) +
        labs(x = "Median co-occurrence enrichment (z) of the decile",
             y = "Pairs sharing >= 1 GO BP term",
             caption = paste0("Dashed line is the overall shared-GO fraction (",
                              sprintf("%.1f%%", 100 * baseline), "). A rise confined to the",
                              " top decile means co-occurrence is carried by a few strong",
                              " functional modules rather than by broad functional similarity.")) +
        theme_classic() +
        theme(axis.text = element_text(size = 12), axis.title = element_text(size = 13),
              plot.caption = element_text(size = 9, colour = "grey30", hjust = 0))
      ggsave(file.path(fig_dir, "a2c_go_by_z_bin.jpeg"), pb, width = 7, height = 5, dpi = dpi)
      message("  wrote a2c_go_by_z_bin.jpeg")
    } else {
      message("  [skip] too few annotated pairs for the GO test")
    }
  } else if (!RUN_GSEA) {
    message("  [note] GO test skipped because RUN_GSEA is FALSE (it needs msigdbr)")
  }
}


message("done -- figures in ", fig_dir)
