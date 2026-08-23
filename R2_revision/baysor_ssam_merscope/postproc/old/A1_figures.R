#!/usr/bin/env Rscript
# ============================================================================== #
# A1 -- R-side figures for the Baysor / SSAM post-hoc analysis
#
# THE THREE FINISHED ANALYSES -- one per-detection quality measure each:
#
#   Section 1  detection radius, and detection SIZE (transcript counts)
#   Section 2  in-nucleus (in-soma) ratio       <- A1_filter_de.ipynb
#   Section 3  negative-control (NC) ratio      <- A1_filter_de.ipynb
#
# THE DEFERRED WT-vs-AD ARM -- off by default:
#
#   Section 4  microdomain DE -> GSEA / dotplots / chords  <- A1_granule_subtypes.ipynb  [OFF]
#   Section 5  WT-vs-AD granule-subtype density            <- A1_granule_subtypes.ipynb  [OFF]
#
# SCOPE. Sections 1-3 do one thing: for every population, summarize and visualize the
# three per-detection quality measures, for both methods and both gene sets.
#
#   population   geneset "all"      pop1, pop2, pop3
#                geneset "markers"  pop1, pop2      (marker enrichment is vacuous there)
#
# The in-nucleus and NC ratios are each reported on TWO denominators:
#   "all"     every transcript inside the detection
#   "marker"  the granule-marker transcripts only (mcDETECT's own definition)
#
# There is deliberately NO comparison against mcDETECT granules here -- no reference
# distribution, no gate-pass panel, no enrichment-over-chance. These panels describe the
# Baysor / SSAM detections on their own terms.
#
# READ pop2 / pop3 WITH CARE. pop2 IS DEFINED BY in_soma_ratio_all < 0.1 and pop3 adds
# marker_frac >= 0.4, so the section-2 distributions for those populations are truncated BY
# CONSTRUCTION -- pop1 is the informative panel there. Nothing defines a population by NC
# ratio, so section 3 is free of that caveat; the one thing to note is that the MARKER
# denominator (n_nc / n_marker) shares its denominator with the marker_frac cut defining
# pop3, so it shifts there partly mechanically. Read the "all" denominator for pop3.
#
# Two views per measure:
#   overlay          all populations on one axis, for direct comparison
#   per-population   one standalone panel per method x geneset x population x sample
#
# Each section has its own toggle. Sections 4 and 5 are OFF by default because the WT-vs-AD
# arm (subtypes, microdomains, pathways) is deferred; turn them back on once
# A1_granule_subtypes.ipynb has been run.
#
# Usage:  Rscript A1_figures.R
# ============================================================================== #

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

here::i_am("R2_revision/baysor_ssam_merscope/postproc/A1_figures.R")

# ------------------------------------------------------------------ #
# What to run
# ------------------------------------------------------------------ #
RUN_RADIUS <- TRUE
RUN_SIZE <- TRUE           # transcript-count panels; reads the features parquets directly
RUN_IN_SOMA <- TRUE
RUN_NC <- TRUE
RUN_MICRODOMAIN <- FALSE   # WT-vs-AD arm deferred; needs A1_granule_subtypes.ipynb
RUN_DENSITY <- FALSE       # ditto

# X-axis cutoff for the per-population RADIUS panels, as a quantile of sphere_r.
# 0.99 stops mid-tail (the histogram is still 0.5-3% of peak height there, which reads as an
# abrupt cut); 0.999 tapers to <0.4% everywhere. 0.9999 over-corrects -- it stretches the axis
# to ~9 um on the markers arm to chase a handful of detections and squashes the informative bulk.
RADIUS_QUANTILE <- 0.999

# Target number of bars in each per-population SIZE panel. Transcript counts span very
# different ranges across methods (SSAM n_total reaches ~40, Baysor's markers arm ~537), so
# one-bar-per-integer would draw ~40 bars for SSAM against ~537 for Baysor -- the same
# distribution rendered at wildly different bar densities. Binning to a fixed bar COUNT
# instead of a fixed bar WIDTH makes the panels visually comparable. 50 also matches the
# ratio histograms in sections 2-3 (0.02-wide bins over [0,1]), so every per-population
# panel in the script now carries roughly the same number of bars.
# Bin width is forced to a whole number of transcripts and to at least 1: the data are
# integer counts, so sub-integer bins would leave empty gaps between attainable values.
SIZE_TARGET_BINS <- 50

# Section 4 selectors. NULL = process everything found on disk.
MICRODOMAIN_RUNS <- NULL   # e.g. c("baysor_markers_pop2"); NULL = every run directory
PAIRS <- NULL              # e.g. c("Subdomain 1_vs_Subdomain 3"); NULL = every pair found

dpi <- 500
out_dir <- here::here("R2_revision/baysor_ssam_merscope/output/postproc")
subtype_dir <- file.path(out_dir, "subtypes")
pop_dir <- file.path(out_dir, "per_population")   # the standalone panels land here
dir.create(pop_dir, showWarnings = FALSE, recursive = TRUE)

fill_colors <- c(WT = "#6fafd2", AD = "#f48488")
tab20_colors <- c("#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
                  "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
                  "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
                  "#17becf", "#9edae5")


# ------------------------------------------------------------------ #
# Shared helpers -- sections 1-3 all use these
# ------------------------------------------------------------------ #
POP_LABEL <- c(pop1 = "All detections",
               pop2 = "Out-of-nucleus",
               pop3 = "Out-of-nucleus + marker-enriched")

relabel <- function(x) unname(ifelse(x %in% names(POP_LABEL), POP_LABEL[x], x))

# both ratio measures are reported on two denominators
denom_label <- function(x) ifelse(x == "all", "all transcripts",
                                  "granule markers")

# The notebook may still write mcDETECT reference rows into the summary/histogram tables.
# They are not plotted here -- these panels describe Baysor / SSAM on their own terms --
# so they are dropped on read, which keeps the script correct whether or not the notebook
# has been re-run since.
drop_reference <- function(df) df %>% filter(method != "mcDETECT", geneset != "-")

# One standalone panel, shared by all three sections so they look identical.
# `df` is either raw values (value_col, radius) or pre-binned counts (bin_lo/bin_hi, ratios).
pop_hist <- function(df, out_file, smp, x_lab, binned, xmax = NA_real_,
                     caption = NULL, bar_width = 0.02, width = 4.5, height = 4) {
  p <- if (binned) {
    ggplot(df, aes(x = (bin_lo + bin_hi) / 2, y = frac)) +
      geom_col(width = bar_width, fill = fill_colors[[smp]], color = "#e9ecef", linewidth = 0.15) +
      labs(x = x_lab, y = "Fraction of detections")
  } else {
    ggplot(df, aes(x = value)) +
      geom_histogram(binwidth = 0.1, fill = fill_colors[[smp]], color = "#e9ecef") +
      labs(x = x_lab, y = "Count")
  }
  if (is.finite(xmax)) p <- p + coord_cartesian(xlim = c(0, xmax))
  if (!is.null(caption)) p <- p + labs(caption = caption)
  p <- p + theme_classic() +
    theme(axis.title = element_text(size = 13),
          axis.text = element_text(size = 12),
          plot.title = element_text(size = 12, face = "bold"),
          plot.caption = element_text(size = 9, hjust = 0, color = "grey30"))
  ggsave(out_file, p, width = width, height = height, dpi = dpi)
  invisible(p)
}

# The NC marker-denominator ratio (n_nc / n_marker) is not bounded by 1, so its histogram
# carries a final [1, Inf) overflow bin. It must be shown, not clipped: it holds up to ~6.7%
# of SSAM's detections. These two helpers keep that handling in one place.
has_overflow <- function(d) any(d$bin_lo >= 1 & d$count > 0)

# The two denominators reach that bin for different reasons, so they get different labels:
#   marker  n_nc / n_marker is genuinely unbounded (NC and marker transcripts are disjoint),
#           so this bar aggregates everything from 1 upwards -- up to ~6.7% of SSAM's pop1.
#   all     n_nc / n_total cannot exceed 1 (NC transcripts are a subset of all transcripts),
#           so the bar is exactly ratio = 1: detections made up entirely of negative controls.
overflow_caption <- function(rt) {
  if (rt == "all") "final bar = ratio 1.0 (every transcript is a negative control)"
  else "final bar = ratio >= 1 (n_nc >= n_marker; unbounded above)"
}

# Walk every (method, geneset, population, sample) group present in `df` and call `fun`.
# One place decides which populations exist for which gene set, so the three sections
# cannot drift apart.
for_each_group <- function(df, fun) {
  n <- 0
  for (mth in sort(unique(df$method))) {
    for (gs in sort(unique(df$geneset))) {
      for (pop in names(POP_LABEL)) {
        if (pop == "pop3" && gs != "all") next   # pop3 is defined only for the all-genes arm
        for (smp in c("WT", "AD")) {
          if (fun(mth, gs, pop, smp)) n <- n + 1
        }
      }
    }
  }
  n
}


# ============================================================================== #
# SECTION 1 -- detection radius
#
# Companion to A1_filter_de.ipynb, which writes
#   ../output/postproc/radius_populations.parquet
#   (method, geneset, sample, sphere_r, pop1, pop2, pop3)
#
# PLOTS ARE BAYSOR ONLY. SSAM emits centres, not extents -- its radius is the constant we
# set ourselves (SSAM_DETECTION_RADIUS = 1.5 um in ../config.py), so it has no distribution
# to show. It IS carried in the summary table, so the omission is visible rather than
# silent, and the script asserts it really is constant.
#
# Panels follow the mcDETECT granule-radius figure's style (code/figures.Rmd, Fig. 3b / 4b):
# geom_histogram(binwidth = 0.1), WT #6fafd2 / AD #f48488, theme_classic, dpi 500.
# ============================================================================== #

summarise_radius <- function(x, method, geneset, population, smp) {
  q <- quantile(x, c(0.05, 0.5, 0.95, 0.99), names = FALSE)
  data.frame(
    method = method, geneset = geneset, population = population, sample = smp,
    n = length(x), mean = mean(x),
    q05 = q[1], median = q[2], q95 = q[3], q99 = q[4], max = max(x)
  )
}

if (RUN_RADIUS) {

  rad_path <- file.path(out_dir, "radius_populations.parquet")
  if (!file.exists(rad_path)) {
    stop("Missing ", rad_path, " -- run A1_filter_de.ipynb (section 3) first.")
  }
  rad_all <- read_parquet(rad_path)

  # SSAM's radius must be one value, or the note above stops being true
  ssam_r <- unique(rad_all$sphere_r[rad_all$method == "ssam"])
  if (length(ssam_r) > 1) {
    message("NOTE: SSAM radius is no longer constant (", length(ssam_r),
            " distinct values) -- it is still excluded from the plots; revisit that.")
  }

  # -------------------- 1a. per-population histograms (Baysor) -------------------- #
  # ONE x-limit per (method, geneset) -- the widest RADIUS_QUANTILE across that gene set's
  # populations and samples. Per-panel limits would draw pop1 and pop2 on different scales,
  # which defeats the point of standalone panels meant to be compared side by side.
  rad_limit <- new.env(parent = emptyenv())
  invisible(for_each_group(rad_all, function(mth, gs, pop, smp) {
    d <- rad_all %>% filter(method == mth, geneset == gs, sample == smp, .data[[pop]])
    if (nrow(d) == 0) return(FALSE)
    key <- paste(mth, gs)
    q <- as.numeric(quantile(d$sphere_r, RADIUS_QUANTILE))
    assign(key, max(q, mget(key, envir = rad_limit, ifnotfound = 0)[[1]]), envir = rad_limit)
    TRUE
  }))

  n_rad <- for_each_group(rad_all %>% filter(method == "baysor"), function(mth, gs, pop, smp) {
    d <- rad_all %>% filter(method == mth, geneset == gs, sample == smp, .data[[pop]])
    if (nrow(d) == 0) return(FALSE)
    stem <- paste("radius", mth, gs, pop, smp, sep = "_")
    pop_hist(d %>% transmute(value = sphere_r),
             file.path(pop_dir, paste0(stem, ".jpeg")), smp,
             "Detection radius (um)", binned = FALSE,
             xmax = get(paste(mth, gs), envir = rad_limit))
    TRUE
  })

  cat("\n[section 1] shared x-limits (q", RADIUS_QUANTILE, ") per method x geneset:\n", sep = "")
  for (k in sort(ls(rad_limit))) cat(sprintf("  %-16s %.2f um\n", k, get(k, envir = rad_limit)))

  # -------------------- 1b. overlay across populations -------------------- #
  overlay <- bind_rows(lapply(names(POP_LABEL), function(pop) {
    rad_all %>%
      filter(method == "baysor", .data[[pop]]) %>%
      transmute(sphere_r, sample, geneset, population = POP_LABEL[[pop]])
  })) %>%
    mutate(population = factor(population, levels = unname(POP_LABEL)),
           sample = factor(sample, levels = c("WT", "AD")))

  p <- ggplot(overlay %>% filter(sphere_r > 0), aes(x = sphere_r, color = population)) +
    geom_density(linewidth = 0.9) +
    scale_x_log10() +
    facet_grid(geneset ~ sample) +
    labs(x = "Detection radius (um), log scale", y = "Density", color = NULL) +
    theme_classic() +
    theme(axis.title = element_text(size = 15), axis.text = element_text(size = 13),
          strip.text = element_text(size = 14), legend.text = element_text(size = 11),
          legend.position = "bottom")
  ggsave(file.path(out_dir, "radius_overlay.jpeg"), p,
         width = 10, height = 4 * length(unique(overlay$geneset)), dpi = dpi)

  # -------------------- 1c. quantile table (both methods) -------------------- #
  radius_summary <- NULL
  invisible(for_each_group(rad_all, function(mth, gs, pop, smp) {
    d <- rad_all %>% filter(method == mth, geneset == gs, sample == smp, .data[[pop]])
    if (nrow(d) == 0) return(FALSE)
    radius_summary <<- bind_rows(radius_summary,
      summarise_radius(d$sphere_r, mth, gs, relabel(pop), smp))
    TRUE
  }))

  write.csv(radius_summary, file.path(out_dir, "radius_summary.csv"), row.names = FALSE)
  cat("\n[section 1] detection radius (um)\n")
  print(radius_summary %>% mutate(across(where(is.numeric), ~round(.x, 4))), row.names = FALSE)
  cat("\nSSAM's radius is the constant set in ../config.py -- tabulated above for",
      "completeness,\nnot plotted, since a constant has no distribution.\n")
  cat("\n[section 1] wrote", n_rad, "per-population panels +",
      "radius_overlay.jpeg + radius_summary.csv\n\n")
}




# ============================================================================== #
# SECTION 1 (cont.) -- detection SIZE: transcripts and granule-marker transcripts
#
# Reads the per-detection features directly from
#   ../output/postproc/<method>_<sample>_tuned_<geneset>_features.parquet
# (columns n_total, n_marker + the population flags), so nothing has to be
# regenerated by the notebook.
#
# BOTH METHODS ARE PLOTTED HERE, unlike the radius: SSAM's radius is a constant we
# set, but its transcript count is a real measurement.
#
# This is the measure that actually separates the methods, and it is what explains
# section 2: Baysor's detections hold a median of ~37 transcripts against SSAM's ~10
# and mcDETECT granules' ~4, which is why Baysor's in-nucleus ratio is a continuum
# rather than a spike at 0 and 1.
#
# Style follows the mcDETECT counterpart in code/figures_response.Rmd:1804-1815
# (the benchmark_sphere panel): boxplots on a log10 axis, fill #1f77b4, "Number of
# transcripts". That figure has no marker-count panel -- benchmark_sphere_metrics_*.csv
# carries only n_transcripts / n_unique_genes / nc_ratio / in_soma_ratio -- so
# n_marker extends the same style rather than copying an existing panel.
#
# ZEROS. n_marker is genuinely 0 for some detections (13.9% of SSAM's all/pop1: no
# granule marker at all). That is a finding, not a nuisance, so:
#   per-population panels are LINEAR    -> the zero bar is visible
#   overlay and boxplot are LOG10       -> zeros cannot be drawn, so each carries a
#                                          caption naming the fraction it dropped.
# The mcDETECT precedent silently filtered n_transcripts > 0; we disclose instead.
#
# Everything is derived from a compact value-frequency table built once per group,
# never by handing millions of raw values to ggplot.
# ============================================================================== #

SIZE_MEASURES <- c(n_total  = "Number of transcripts",
                   n_marker = "Number of granule-marker transcripts")

if (RUN_SIZE) {

  size_counts <- NULL    # value -> count, per (measure, method, geneset, population, sample)
  size_summary <- NULL

  for (mth in c("baysor", "ssam")) {
    for (gs in c("all", "markers")) {
      for (smp in c("WT", "AD")) {

        f <- file.path(out_dir, sprintf("%s_%s_tuned_%s_features.parquet", mth, smp, gs))
        if (!file.exists(f)) {
          message("Missing ", basename(f), " -- skipping that group in section 1 (size).")
          next
        }
        pops <- c("pop1", "pop2", if (gs == "all") "pop3")
        d <- read_parquet(f, col_select = all_of(c(names(SIZE_MEASURES), pops)))

        for (pop in pops) {
          sel <- d[[pop]]
          if (!any(sel)) next
          for (ms in names(SIZE_MEASURES)) {
            x <- d[[ms]][sel]
            q <- quantile(x, c(0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999), names = FALSE)
            size_summary <- bind_rows(size_summary, data.frame(
              measure = ms, method = mth, geneset = gs,
              population = relabel(pop), pop_key = pop, sample = smp,
              n = length(x), mean = mean(x),
              q05 = q[1], q25 = q[2], median = q[3], q75 = q[4], q95 = q[5],
              q99 = q[6], q999 = q[7], max = max(x),
              frac_zero = mean(x == 0)))
            size_counts <- bind_rows(size_counts,
              data.frame(value = x) %>% count(value, name = "count") %>%
                mutate(measure = ms, method = mth, geneset = gs,
                       population = relabel(pop), pop_key = pop, sample = smp))
          }
        }
        rm(d)
      }
    }
  }

  if (is.null(size_summary)) {
    message("No features parquets found -- skipping section 1 (size).")
  } else {

    size_counts <- size_counts %>%
      group_by(measure, method, geneset, population, sample) %>%
      mutate(frac = count / sum(count)) %>%
      ungroup() %>%
      mutate(population = factor(population, levels = unname(POP_LABEL)),
             sample = factor(sample, levels = c("WT", "AD")))

    write.csv(size_summary %>% select(-pop_key),
              file.path(out_dir, "size_summary.csv"), row.names = FALSE)

    for (ms in names(SIZE_MEASURES)) {
      cat("\n[section 1] ", SIZE_MEASURES[[ms]], " per detection\n", sep = "")
      print(size_summary %>% filter(measure == ms) %>%
              select(method, geneset, population, sample, n, mean,
                     q05, median, q95, q99, max, frac_zero) %>%
              mutate(across(where(is.numeric), ~round(.x, 3))), row.names = FALSE)
    }

    # -------------------- 1d. per-population panels (linear, zeros visible) ------ #
    # Shared x-limit per (measure, method, geneset) -- the widest q999 across that
    # group's populations and samples, same rule the radius panels use.
    n_size <- 0
    size_binwidth <- NULL
    for (ms in names(SIZE_MEASURES)) {
      for (mth in unique(size_summary$method)) {
        for (gs in unique(size_summary$geneset)) {

          lim <- size_summary %>%
            filter(measure == ms, method == mth, geneset == gs) %>% pull(q999)
          if (!length(lim)) next
          lim <- max(lim)

          # One bin width per arm, derived from that arm's shared x-limit, so every panel
          # ends up with ~SIZE_TARGET_BINS bars whatever the underlying range.
          bw <- max(1, ceiling(lim / SIZE_TARGET_BINS))
          size_binwidth <- bind_rows(size_binwidth, data.frame(
            measure = ms, method = mth, geneset = gs, xlim = lim,
            bin_width = bw, n_bars = ceiling(lim / bw)))

          for (pop in names(POP_LABEL)) {
            if (pop == "pop3" && gs != "all") next
            for (smp in c("WT", "AD")) {
              d <- size_counts %>%
                filter(measure == ms, method == mth, geneset == gs,
                       population == relabel(pop), sample == smp)
              if (nrow(d) == 0) next
              # aggregate the integer value-frequency table into the arm's bins;
              # `frac` is already normalised per group, so summing it is exact
              d <- d %>%
                mutate(bin = floor(value / bw)) %>%
                group_by(bin) %>%
                summarise(frac = sum(frac), .groups = "drop") %>%
                transmute(bin_lo = bin * bw, bin_hi = (bin + 1) * bw, frac)
              stem <- paste("size", ms, mth, gs, pop, smp, sep = "_")
              pop_hist(d, file.path(pop_dir, paste0(stem, ".jpeg")), smp,
                       SIZE_MEASURES[[ms]], binned = TRUE,
                       xmax = lim, bar_width = bw,
                       caption = if (bw > 1)
                         sprintf("bin width = %d transcripts", bw) else NULL)
              n_size <- n_size + 1
            }
          }
        }
      }
    }

    # -------------------- 1e. overlay across populations (log10 x) --------------- #
    for (ms in names(SIZE_MEASURES)) {

      d <- size_counts %>% filter(measure == ms)
      dropped <- size_summary %>% filter(measure == ms) %>% pull(frac_zero) %>% max()
      cap <- if (dropped > 0) sprintf(
        "log axis: detections with %s = 0 are not shown (up to %.1f%% of a population)",
        ms, 100 * dropped) else NULL

      p <- ggplot(d %>% filter(value > 0),
                  aes(x = value, y = frac, color = population)) +
        geom_line(linewidth = 0.8) +
        scale_x_log10() +
        facet_grid(method + geneset ~ sample) +
        labs(x = paste0(SIZE_MEASURES[[ms]], ", log scale"),
             y = "Fraction of detections", color = NULL, caption = cap) +
        theme_classic() +
        theme(axis.title = element_text(size = 15), axis.text = element_text(size = 13),
              strip.text = element_text(size = 12), legend.text = element_text(size = 11),
              legend.position = "bottom",
              plot.caption = element_text(size = 9, hjust = 0, color = "grey30"))
      ggsave(file.path(out_dir, paste0("size_", ms, "_overlay.jpeg")), p,
             width = 10, height = 9, dpi = dpi)
    }

    # -------------------- 1f. mcDETECT-style boxplots (log10 y) ------------------ #
    # geom_boxplot(stat = "identity") from precomputed quantiles: the raw vectors run
    # to millions of rows, and the drawn figure is identical either way.
    for (ms in names(SIZE_MEASURES)) {

      bx <- size_summary %>%
        filter(measure == ms) %>%
        mutate(population = factor(population, levels = unname(POP_LABEL)),
               sample = factor(sample, levels = c("WT", "AD")),
               arm = paste(method, geneset, sep = " / "),
               ymin = pmax(q05, 0.5), lower = pmax(q25, 0.5),
               middle = pmax(median, 0.5), upper = pmax(q75, 0.5),
               ymax = pmax(q95, 0.5))

      dropped <- max(bx$frac_zero)
      cap <- paste0("box = q25-q75, whiskers = q05-q95, line = median.",
                    if (dropped > 0) sprintf(
                      "\nlog axis: %s = 0 cannot be shown (up to %.1f%% of a population); ",
                      ms, 100 * dropped) else "\n",
                    "values below 1 are clamped to 0.5 for display.")

      p <- ggplot(bx, aes(x = population, ymin = ymin, lower = lower, middle = middle,
                          upper = upper, ymax = ymax, fill = sample)) +
        geom_boxplot(stat = "identity", position = position_dodge(width = 0.7),
                     width = 0.6, fatten = 1.2, color = "#3b3b3b") +
        scale_y_log10() +
        scale_fill_manual(values = scales::alpha(c(WT = "#1f77b4", AD = "#f48488"), 0.8)) +
        facet_wrap(~ arm, nrow = 1) +
        labs(x = NULL, y = SIZE_MEASURES[[ms]], fill = NULL, caption = cap) +
        theme_classic() +
        theme(axis.title = element_text(size = 15),
              axis.text.x = element_text(size = 11, angle = 30, hjust = 1),
              axis.text.y = element_text(size = 13),
              strip.text = element_text(size = 12), legend.position = "bottom",
              plot.caption = element_text(size = 9, hjust = 0, color = "grey30"))
      ggsave(file.path(out_dir, paste0("size_", ms, "_boxplot.jpeg")), p,
             width = 12, height = 6, dpi = dpi)
    }

    cat("\n[section 1] size-panel binning (target ", SIZE_TARGET_BINS, " bars/panel):\n", sep = "")
    print(size_binwidth, row.names = FALSE)

    cat("\n[section 1] wrote", n_size, "per-population size panels +",
        "size_{n_total,n_marker}_{overlay,boxplot}.jpeg + size_summary.csv\n\n")
  }
}


# ============================================================================== #
# SECTION 2 -- in-nucleus (in-soma) ratio
#
# Companion to A1_filter_de.ipynb, which writes
#   ../output/postproc/in_soma_summary.csv       quantiles / frac_zero per group
#   ../output/postproc/in_soma_histogram.parquet 0.02-wide binned counts per group
#
# Covers BOTH methods (unlike the radius, SSAM's in-nucleus ratio is a real measurement),
# both gene sets, every population, and both denominators.
#
# READ WITH CARE: pop2 and pop3 are *defined* by in_soma_ratio_all < 0.1, so their
# distributions are truncated by construction. pop1 is the informative panel -- it says how
# somatic each method's raw detections are.
#
# The mass piles up at exactly zero, so the overlay is drawn twice: the full distribution,
# and the non-zero part renormalised, where the shape is actually visible.
# ============================================================================== #

if (RUN_IN_SOMA) {

  sum_path <- file.path(out_dir, "in_soma_summary.csv")
  hist_path <- file.path(out_dir, "in_soma_histogram.parquet")

  if (!file.exists(sum_path) || !file.exists(hist_path)) {
    message("Missing in_soma_summary.csv / in_soma_histogram.parquet -- ",
            "run A1_filter_de.ipynb (section 3) first. Skipping section 2.")
  } else {

    in_soma <- read.csv(sum_path, check.names = FALSE) %>% drop_reference()
    hist_df <- read_parquet(hist_path) %>% drop_reference()

    # -------------------- 2a. numeric table -------------------- #
    tbl <- in_soma %>%
      mutate(population = relabel(population)) %>%
      arrange(ratio, method, geneset, population, sample) %>%
      select(ratio, method, geneset, population, sample, n, n_undefined, mean,
             q05, median, q95, frac_zero)

    write.csv(tbl, file.path(out_dir, "in_soma_summary_table.csv"), row.names = FALSE)

    for (rt in sort(unique(tbl$ratio))) {
      cat("\n[section 2] in-nucleus ratio, denominator =", toupper(denom_label(rt)), "\n")
      print(tbl %>% filter(ratio == rt) %>% select(-ratio) %>%
              mutate(across(where(is.numeric), ~round(.x, 4))), row.names = FALSE)
    }

    # -------------------- 2b. per-population panels -------------------- #
    n_is <- 0
    for (rt in sort(unique(hist_df$ratio))) {
      n_is <- n_is + for_each_group(hist_df %>% filter(ratio == rt),
                                    function(mth, gs, pop, smp) {
        d <- hist_df %>% filter(ratio == rt, method == mth, geneset == gs,
                                population == pop, sample == smp)
        if (nrow(d) == 0) return(FALSE)
        stem <- paste("in_soma", rt, mth, gs, pop, smp, sep = "_")
        pop_hist(d, file.path(pop_dir, paste0(stem, ".jpeg")), smp,
                 paste0("in-nucleus ratio (", denom_label(rt), ")"),
                 binned = TRUE, xmax = 1)
        TRUE
      })
    }

    # -------------------- 2c. overlays -------------------- #
    for (rt in sort(unique(hist_df$ratio))) {

      d <- hist_df %>%
        filter(ratio == rt) %>%
        mutate(group = paste0(method, " / ", relabel(population)),
               sample = factor(sample, levels = c("WT", "AD")))
      if (nrow(d) == 0) next

      p <- ggplot(d, aes(x = (bin_lo + bin_hi) / 2, y = frac, fill = group)) +
        geom_col(position = "identity", alpha = 0.45, width = 0.02) +
        facet_grid(geneset ~ sample) +
        labs(x = paste0("in-nucleus ratio (", denom_label(rt), ")"),
             y = "fraction of detections", fill = NULL) +
        theme_classic() +
        theme(axis.title = element_text(size = 14), axis.text = element_text(size = 12),
              strip.text = element_text(size = 13), legend.text = element_text(size = 10),
              legend.position = "bottom")
      ggsave(file.path(out_dir, paste0("in_soma_", rt, "_bars.jpeg")), p,
             width = 11, height = 3.6 * length(unique(d$geneset)), dpi = dpi)

      d_nz <- d %>%
        filter(bin_lo > 0) %>%
        group_by(ratio, method, geneset, sample, group) %>%
        mutate(frac_nz = ifelse(sum(count) > 0, count / sum(count), 0)) %>%
        ungroup()

      p2 <- ggplot(d_nz, aes(x = (bin_lo + bin_hi) / 2, y = frac_nz, color = group)) +
        geom_line(linewidth = 0.8) +
        facet_grid(geneset ~ sample) +
        labs(x = paste0("in-nucleus ratio > 0 (", denom_label(rt), ")"),
             y = "fraction of non-zero detections", color = NULL) +
        theme_classic() +
        theme(axis.title = element_text(size = 14), axis.text = element_text(size = 12),
              strip.text = element_text(size = 13), legend.text = element_text(size = 10),
              legend.position = "bottom")
      ggsave(file.path(out_dir, paste0("in_soma_", rt, "_nonzero.jpeg")), p2,
             width = 11, height = 3.6 * length(unique(d$geneset)), dpi = dpi)
    }

    cat("\n[section 2] wrote", n_is, "per-population panels + the in_soma_*.jpeg overlays",
        "+ in_soma_summary_table.csv\n")
  }
}


# ============================================================================== #
# SECTION 3 -- negative-control (NC) ratio
#
# Companion to A1_filter_de.ipynb, which writes
#   ../output/postproc/nc_summary.csv        quantiles / frac_zero per group
#   ../output/postproc/nc_histogram.parquet  0.02-wide binned counts (+ an overflow bin)
#
# Structured exactly like section 2 so the two read identically: a table, per-population
# panels, and two overlays, on both denominators.
#
#   nc_ratio_all     = n_nc / n_total   (every transcript in the detection)
#   nc_ratio_marker  = n_nc / n_marker  (granule-marker transcripts only)
#
# NOTHING defines a population by NC ratio -- not the detection sweep, not pop2, not pop3,
# and no population definition touches nc_top or nc_thr. So unlike section 2 these
# distributions are not truncated by construction, and the pop1 -> pop2 -> pop3 trend is a
# real read in whichever direction it runs.
#
# The one caveat: the MARKER denominator shares n_marker with the marker_frac >= 0.4 cut
# that defines pop3, so it shifts there partly mechanically. Read the "all" denominator for
# pop3; the marker one is a sensitivity check.
# ============================================================================== #

if (RUN_NC) {

  nc_sum_path <- file.path(out_dir, "nc_summary.csv")
  nc_hist_path <- file.path(out_dir, "nc_histogram.parquet")

  if (!file.exists(nc_sum_path) || !file.exists(nc_hist_path)) {
    message("Missing nc_summary.csv / nc_histogram.parquet -- ",
            "run A1_filter_de.ipynb (section 3) first. Skipping section 3.")
  } else {

    nc <- read.csv(nc_sum_path, check.names = FALSE) %>% drop_reference()
    nc_hist <- read_parquet(nc_hist_path) %>% drop_reference()

    # -------------------- 3a. numeric table -------------------- #
    nc_tbl <- nc %>%
      mutate(population = relabel(population)) %>%
      arrange(ratio, method, geneset, population, sample) %>%
      select(ratio, method, geneset, population, sample, n, n_undefined, mean,
             q05, median, q95, frac_zero)

    write.csv(nc_tbl, file.path(out_dir, "nc_summary_table.csv"), row.names = FALSE)

    for (rt in sort(unique(nc_tbl$ratio))) {
      cat("\n[section 3] NC ratio, denominator =", toupper(denom_label(rt)), "\n")
      if (rt == "marker") {
        cat("  NOTE: pop3 shifts partly mechanically here -- the marker_frac cut that",
            "defines it\n  inflates this denominator. Read the all-transcript table for pop3.\n")
      }
      print(nc_tbl %>% filter(ratio == rt) %>% select(-ratio) %>%
              mutate(across(where(is.numeric), ~round(.x, 4))), row.names = FALSE)
    }

    # -------------------- 3b. per-population panels -------------------- #
    # nc_ratio_marker is unbounded above 1, so its last bin runs to +Inf; place that bin
    # just past 1 rather than letting a non-finite coordinate drop the row silently.
    nc_hist <- nc_hist %>%
      mutate(bin_hi = ifelse(is.finite(bin_hi), bin_hi, bin_lo + 0.02))

    n_nc <- 0
    for (rt in sort(unique(nc_hist$ratio))) {
      n_nc <- n_nc + for_each_group(nc_hist %>% filter(ratio == rt),
                                    function(mth, gs, pop, smp) {
        d <- nc_hist %>% filter(ratio == rt, method == mth, geneset == gs,
                                population == pop, sample == smp)
        if (nrow(d) == 0) return(FALSE)
        stem <- paste("nc", rt, mth, gs, pop, smp, sep = "_")
        ov <- has_overflow(d)
        pop_hist(d, file.path(pop_dir, paste0(stem, ".jpeg")), smp,
                 paste0("NC ratio (", denom_label(rt), ")"),
                 binned = TRUE,
                 xmax = if (ov) max(d$bin_hi) else 1,
                 caption = if (ov) overflow_caption(rt) else NULL)
        TRUE
      })
    }

    # -------------------- 3c. overlays -------------------- #
    for (rt in sort(unique(nc_hist$ratio))) {

      d <- nc_hist %>%
        filter(ratio == rt) %>%
        mutate(group = paste0(method, " / ", relabel(population)),
               sample = factor(sample, levels = c("WT", "AD")))
      if (nrow(d) == 0) next

      ov <- has_overflow(d)

      p <- ggplot(d, aes(x = (bin_lo + bin_hi) / 2, y = frac, fill = group)) +
        geom_col(position = "identity", alpha = 0.45, width = 0.02) +
        facet_grid(geneset ~ sample) +
        labs(x = paste0("NC ratio (", denom_label(rt), ")"),
             y = "fraction of detections", fill = NULL,
             caption = if (ov) overflow_caption(rt) else NULL) +
        theme_classic() +
        theme(axis.title = element_text(size = 14), axis.text = element_text(size = 12),
              strip.text = element_text(size = 13), legend.text = element_text(size = 10),
              legend.position = "bottom",
              plot.caption = element_text(size = 9, hjust = 0, color = "grey30"))
      ggsave(file.path(out_dir, paste0("nc_", rt, "_bars.jpeg")), p,
             width = 11, height = 3.6 * length(unique(d$geneset)), dpi = dpi)

      d_nz <- d %>%
        filter(bin_lo > 0) %>%
        group_by(ratio, method, geneset, sample, group) %>%
        mutate(frac_nz = ifelse(sum(count) > 0, count / sum(count), 0)) %>%
        ungroup()

      p2 <- ggplot(d_nz, aes(x = (bin_lo + bin_hi) / 2, y = frac_nz, color = group)) +
        geom_line(linewidth = 0.8) +
        facet_grid(geneset ~ sample) +
        labs(x = paste0("NC ratio > 0 (", denom_label(rt), ")"),
             y = "fraction of non-zero detections", color = NULL) +
        theme_classic() +
        theme(axis.title = element_text(size = 14), axis.text = element_text(size = 12),
              strip.text = element_text(size = 13), legend.text = element_text(size = 10),
              legend.position = "bottom")
      ggsave(file.path(out_dir, paste0("nc_", rt, "_nonzero.jpeg")), p2,
             width = 11, height = 3.6 * length(unique(d$geneset)), dpi = dpi)
    }

    cat("\n[section 3] wrote", n_nc, "per-population panels + the nc_*.jpeg overlays",
        "+ nc_summary_table.csv\n")
    cat("\n[sections 1-3] per-population panels are in", pop_dir, "\n\n")
  }
}


# ============================================================================== #
# SECTION 4 -- microdomain DE -> GSEA, NES dotplots, chord diagrams
#
# Companion to A1_granule_subtypes.ipynb, which writes, per run directory
# ../output/postproc/subtypes/<method>_markers_<population>/ :
#   {granule,cell,ambient}_DE_genes_<target>_vs_<reference>.csv
#
# Which two subdomains to contrast is your call after looking at the subdomain map,
# so it is set in the notebook (SUBDOMAIN_PAIRS) and simply discovered here -- every
# pair the notebook produced is processed. Restrict with MICRODOMAIN_RUNS / PAIRS above.
#
# Everything below is lifted from code/figures_response.Rmd (GSEA chunk, lines
# ~562-689, and make_gsea_chord, lines ~315-430) so the output files match
# output/MERSCOPE_WT_AD_comparison/neuropil_subdomains_Isocortex_50/ one for one:
#   <stem>_GSEA.csv
#   <stem>_target_GSEA.jpeg     <stem>_reference_GSEA.jpeg
#   <stem>_{positive,negative}_chord_diagram.jpeg  + _chord_legend.jpeg
# ============================================================================== #

# Copied verbatim from code/figures_response.Rmd:315-430 (the Rmd is a document, not
# a sourceable library, so it cannot be imported).
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


gsea_dotplot <- function(df, out_file, width = 10, height = 6) {
  p <- ggplot(df, aes(x = NES_plot, y = Description)) +
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
  invisible(p)
}


if (RUN_MICRODOMAIN) {

  if (!dir.exists(subtype_dir)) {
    message("No ", subtype_dir, " -- run A1_granule_subtypes.ipynb first. Skipping section 4.")
  } else {

    suppressPackageStartupMessages({
      library(clusterProfiler)
      library(ComplexHeatmap)
      library(circlize)
      library(forcats)
      library(msigdbr)
      library(org.Mm.eg.db)
      library(stringi)
      library(stringr)
    })

    # -------------------- discover the DE tables -------------------- #
    runs <- list.dirs(subtype_dir, full.names = FALSE, recursive = FALSE)
    if (!is.null(MICRODOMAIN_RUNS)) runs <- intersect(runs, MICRODOMAIN_RUNS)

    jobs <- bind_rows(lapply(runs, function(run) {
      files <- list.files(file.path(subtype_dir, run), pattern = "_DE_genes_.*_vs_.*\\.csv$")
      # drop this script's own output: subdomain names contain spaces, so the greedy
      # match below would otherwise swallow "_GSEA" into the reference name
      files <- files[!grepl("_GSEA\\.csv$", files)]
      if (length(files) == 0) return(NULL)
      m <- stringr::str_match(files, "^(granule|cell|ambient)_DE_genes_(.+)_vs_(.+)\\.csv$")
      keep <- !is.na(m[, 1])
      if (!any(keep)) return(NULL)
      data.frame(run = run, file = files[keep], layer = m[keep, 2],
                 target = m[keep, 3], reference = m[keep, 4], stringsAsFactors = FALSE)
    }))

    if (is.null(jobs) || nrow(jobs) == 0) {
      message("No microdomain DE tables found under ", subtype_dir, " -- skipping section 4.")
    } else {
      jobs$pair <- paste0(jobs$target, "_vs_", jobs$reference)
      if (!is.null(PAIRS)) jobs <- jobs[jobs$pair %in% PAIRS, , drop = FALSE]

      cat("[section 4] DE tables to process:\n")
      print(jobs[, c("run", "layer", "target", "reference")], row.names = FALSE)

      # -------------------- gene sets (once) -------------------- #
      msig_mouse <- msigdbr(species = "Mus musculus", category = "C5", subcategory = "BP")
      term2gene <- msig_mouse %>% dplyr::select(gs_name, gene_symbol) %>% distinct()
      set.seed(42)

      for (i in seq_len(nrow(jobs))) {

        job <- jobs[i, ]
        in_path <- file.path(subtype_dir, job$run, job$file)
        stem <- file.path(subtype_dir, job$run, sub("\\.csv$", "", job$file))
        cat("\n---", job$run, "/", job$file, "---\n")

        df <- read.csv(in_path, check.names = FALSE)
        if (!all(c("names", "scores") %in% names(df))) {
          message("  missing names/scores columns -- skipping.")
          next
        }

        gene_list <- df %>%
          dplyr::select(names, scores) %>%
          filter(!is.na(names), !is.na(scores)) %>%
          group_by(names) %>%
          summarise(rank_metric = mean(scores), .groups = "drop") %>%
          arrange(desc(rank_metric), names)

        gene_vector <- gene_list$rank_metric
        names(gene_vector) <- gene_list$names
        gene_vector <- gene_vector + runif(length(gene_vector), -1e-10, 1e-10)  # deterministic ties
        gene_vector <- sort(gene_vector, decreasing = TRUE)

        overlap_n <- sum(names(gene_vector) %in% term2gene$gene_symbol)
        message("  unique genes: ", length(gene_vector), "; overlap with TERM2GENE: ", overlap_n)
        if (length(gene_vector) < 50 || overlap_n < 20) {
          message("  ranking or overlap too small -- skipping.")
          next
        }

        gsea_res <- tryCatch(
          GSEA(geneList = gene_vector, TERM2GENE = term2gene,
               minGSSize = 10, maxGSSize = 500, pvalueCutoff = 0.25,
               pAdjustMethod = "BH", eps = 1e-10, seed = TRUE, by = "fgsea", verbose = FALSE),
          error = function(e) {
            message("  GSEA failed: ", e$message)
            NULL
          }
        )
        if (is.null(gsea_res)) next

        result_df <- as.data.frame(gsea_res)
        if (nrow(result_df) == 0) {
          message("  no enriched pathways.")
          next
        }

        result_df <- result_df %>%
          mutate(label_clean = str_remove(ID, "^GOBP_"),
                 label_clean = str_replace_all(label_clean, "_", " "),
                 label_clean = str_to_sentence(label_clean),
                 Description = label_clean)

        write.csv(result_df %>% arrange(desc(NES)), paste0(stem, "_GSEA.csv"))

        # -------------------- NES dotplots -------------------- #
        top_target_df <- result_df %>%
          filter(NES > 0) %>% arrange(p.adjust, desc(NES)) %>% slice_head(n = 10) %>%
          mutate(NES_plot = NES, Description = fct_reorder(Description, NES))
        top_reference_df <- result_df %>%
          filter(NES < 0) %>% arrange(p.adjust, NES) %>% slice_head(n = 10) %>%
          mutate(NES_plot = abs(NES), Description = fct_reorder(Description, NES_plot))

        if (nrow(top_target_df) > 0) {
          gsea_dotplot(top_target_df, paste0(stem, "_target_GSEA.jpeg"))
        }
        if (nrow(top_reference_df) > 0) {
          gsea_dotplot(top_reference_df, paste0(stem, "_reference_GSEA.jpeg"))
        }

        # -------------------- chord diagrams -------------------- #
        make_gsea_chord(gsea_df = result_df, direction = "positive", out_prefix = stem, top_n = 20)
        make_gsea_chord(gsea_df = result_df, direction = "negative", out_prefix = stem, top_n = 20)

        cat("  wrote", basename(stem), "GSEA table, dotplots and chord diagrams\n")
      }

      cat("\n[section 4] done --", nrow(jobs), "DE table(s) processed under", subtype_dir, "\n")
    }
  }
}


# ============================================================================== #
# SECTION 5 -- WT vs AD granule-subtype density
#
# Companion to A1_granule_subtypes.ipynb section 4, which writes, per run directory
# ../output/postproc/subtypes/<method>_markers_<population>/subtype_density_per_region.csv
# in mcDETECT's exact schema (sample, brain_area, subtype, density, n_spots, setting,
# density_sd, density_sem, density_ci_low, density_ci_high, p_val, p_bonf, q_val, *_star).
#
# The AD densities in that file are ALREADY divided by CAPTURE_EFFICIENCY_COEF = 0.818691,
# as mcDETECT does -- nothing further is corrected here.
#
# The panels are lifted from code/figures_response.Rmd:2246-2293, so they can be put next to
# mcDETECT's own output/MERSCOPE_WT_AD_comparison/granule_density_<subtype>.jpeg and read the
# same way: one file per subtype, bars per brain region dodged by sample, bootstrap 95% CI
# errorbars, and the Bonferroni-within-subtype star above each pair.
# ============================================================================== #

# mcDETECT's plotting order; any other subtype present is appended after these.
subtype_plot_order <- c("overall", "pre-syn", "post-syn", "dendrites", "mixed", "axons", "others")
area_order <- c("Isocortex", "OLF", "HPF-CA", "HPF-DG", "HPF-SR", "CTXsp", "TH", "MB", "FT")

density_barplot <- function(df_plot, out_file) {
  has_ci <- all(c("density_ci_low", "density_ci_high") %in% colnames(df_plot))

  y_range <- if (has_ci) max(df_plot$density_ci_high, na.rm = TRUE) else max(df_plot$density, na.rm = TRUE)
  if (!is.finite(y_range) || y_range <= 0) return(invisible(NULL))
  gap <- 0.03 * y_range

  df_star <- df_plot %>%
    group_by(brain_area) %>%
    summarise(y_max = if (has_ci) max(density_ci_high, na.rm = TRUE) else max(density, na.rm = TRUE),
              .groups = "drop") %>%
    left_join(df_plot %>% distinct(brain_area, p_bonf_star), by = "brain_area") %>%
    mutate(x = brain_area, y = y_max + gap,
           label_size = ifelse(is.na(p_bonf_star) | p_bonf_star == "ns", 5, 7),
           p_bonf_star = ifelse(is.na(p_bonf_star), "", p_bonf_star))

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
    scale_fill_manual(values = c("WT" = "#a0ccec", "AD" = "#f48488")) +
    scale_y_continuous(expand = c(0, 0)) +
    coord_cartesian(clip = "off", ylim = c(0, y_top)) +
    labs(x = " ", y = "Density", fill = " ") +
    theme_classic() +
    theme(axis.title.y = element_text(size = 15),
          axis.text.x  = element_text(size = 15),
          axis.text.y  = element_text(size = 15),
          legend.position = "bottom",
          legend.box.margin = margin(t = -20, b = 0),
          legend.title = element_text(size = 15),
          legend.text  = element_text(size = 15),
          plot.margin  = margin(t = 10, r = 10, b = 10, l = 10))

  ggsave(out_file, p, width = 9, height = 6, dpi = dpi)
  invisible(p)
}

if (RUN_DENSITY) {

  if (!dir.exists(subtype_dir)) {
    message("No ", subtype_dir, " -- run A1_granule_subtypes.ipynb first. Skipping section 5.")
  } else {

    runs <- list.dirs(subtype_dir, full.names = FALSE, recursive = FALSE)
    if (!is.null(MICRODOMAIN_RUNS)) runs <- intersect(runs, MICRODOMAIN_RUNS)

    n_done <- 0
    for (run in runs) {

      csv_path <- file.path(subtype_dir, run, "subtype_density_per_region.csv")
      if (!file.exists(csv_path)) {
        message("[", run, "] no subtype_density_per_region.csv -- skipping.")
        next
      }

      dens <- read.csv(csv_path, check.names = FALSE)
      subtypes <- unique(dens$subtype)
      subtypes <- c(intersect(subtype_plot_order, subtypes), setdiff(subtypes, subtype_plot_order))

      cat("\n---", run, "--- subtypes:", paste(subtypes, collapse = ", "), "\n")

      for (st in subtypes) {
        df_plot <- dens %>%
          filter(subtype == st, sample %in% c("WT", "AD"), brain_area %in% area_order) %>%
          mutate(brain_area = factor(brain_area, levels = area_order),
                 sample = factor(sample, levels = c("WT", "AD")))
        if (nrow(df_plot) == 0) {
          message("  ", st, ": no rows in the plotted regions -- skipping.")
          next
        }
        out_file <- file.path(subtype_dir, run, paste0("granule_density_", st, ".jpeg"))
        density_barplot(df_plot, out_file)
        cat("  wrote", basename(out_file), "\n")
        n_done <- n_done + 1
      }
    }

    cat("\n[section 5] wrote", n_done, "density panel(s) under", subtype_dir, "\n")
  }
}


