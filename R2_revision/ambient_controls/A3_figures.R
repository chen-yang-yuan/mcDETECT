# ==============================================================================================
# A3 -- ambient RNA controls at the detection step: all figures
#
# Reviewer #2, major point 9. Section numbers match the notebooks 1:1:
#
#   [1]-[6]   A3a_three_sets.ipynb      preflight / NC forensics / sets / overlap / density / stage D
#   [7]-[8]   A3b_vicinity.ipynb        placement + profiles / the detection predicate
#   [9]-[10]  A3c_de_baseline.ipynb     Axis 1 (compartment) / Axis 2 (conditions and regions)
#
# This script reads CSV and Parquet ONLY. R cannot open an .h5ad, so anything plotted here was
# pre-exported by the Python side -- distributions arrive as a `*_summary.csv` + `*_histogram.parquet`
# pair (bin counts and quantiles), never as millions of raw values.
#
# WHERE THE ARGUMENT LIVES. Section [8] is the load-bearing panel: the fraction of vicinity
# pseudo-granules that would have been DETECTED, as a function of offset distance, against the real
# granules as a ceiling and tissue-wide random locations as the floor. Sections [3] and [7] contain
# comparisons that are partly pre-ordained by the geometry (a matched-radius sphere cannot capture
# more than the minimum enclosing sphere it was copied from) -- they are description, not evidence.
#
# READ [6] WITH CARE. The adaptive-threshold survival is ONE-SIDED: it can only remove granules,
# never add ones the global threshold missed. It bounds false-positive inflation and is silent on
# false negatives in low-density regions -- and AD is the lower-density arm. The caveat text ships
# alongside in adaptive_caveats.csv and belongs in any figure legend drawn from that panel.
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
RUN_PREFLIGHT <- TRUE    # 1. CSR min_samples table + z-plane profile
RUN_NC        <- TRUE    # 2. NC-filter forensics: corrected ratio, leave-one-out, Gria2
RUN_SETS      <- TRUE    # 3. set inventory + funnels, raw and per-million-transcript
RUN_OVERLAP   <- TRUE    # 4. the overlap ladder, observed vs expected
RUN_DENSITY   <- TRUE    # 5. per-region density per set, WT vs AD
RUN_ADAPTIVE  <- TRUE    # 6. locally-adaptive threshold survival curves
RUN_VICINITY  <- TRUE    # 7. placement acceptance + real-vs-pseudo distributions
RUN_PREDICATE <- TRUE    # 8. THE detection-predicate curve
RUN_AXIS1     <- TRUE    # 9. compartment: baseline vs granule enrichment
RUN_AXIS2     <- TRUE    # 10. conditions and regions

dpi <- 500

# -------------------- paths -------------------- #
root    <- here::here("R2_revision/ambient_controls/output")
pre_dir <- file.path(root, "preflight")
a3a_dir <- file.path(root, "a3a")
a3b_dir <- file.path(root, "a3b")
a3c_dir <- file.path(root, "a3c")
fig_dir <- file.path(root, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# -------------------- palette and shared helpers -------------------- #
fill_colors <- c(WT = "#a0ccec", AD = "#f48488")     # a3_config.py holds the same two hex values
set_colors  <- c(set0 = "#b6b6b6", set1 = "#7fb069", set2 = "#4f7fa8", set3 = "#d98b5f")
hist_outline <- "#e9ecef"
area_order <- c("Isocortex", "OLF", "HPF-CA", "HPF-DG", "HPF-SR", "CTXsp", "TH", "MB", "FT")

need <- function(path, section) {
  if (!file.exists(path)) { message("  [skip ", section, "] missing: ", path); return(FALSE) }
  TRUE
}

save_fig <- function(p, name, width, height) {
  ggsave(file.path(fig_dir, name), p, width = width, height = height, dpi = dpi)
  message("  wrote ", name)
}

# One standalone binned-histogram panel, shared by every distribution section so they cannot
# drift apart visually.
binned_hist <- function(df, out_file, x_lab, fill_var = "sample", palette = NULL,
                        facet = NULL, caption = NULL, width = 7, height = 4.5) {
  p <- ggplot(df, aes(x = bin_lo, y = frac, fill = .data[[fill_var]])) +
    geom_col(width = df$bin_hi[1] - df$bin_lo[1], colour = hist_outline, linewidth = 0.1,
             position = "identity", alpha = 0.65) +
    labs(x = x_lab, y = "fraction", fill = NULL, caption = caption) +
    theme_classic() +
    theme(axis.text = element_text(size = 12), axis.title = element_text(size = 13),
          legend.position = "bottom",
          plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
  pal <- if (!is.null(palette)) palette else
    if (all(df[[fill_var]] %in% names(fill_colors))) fill_colors else NULL
  if (!is.null(pal)) p <- p + scale_fill_manual(values = pal)
  if (!is.null(facet)) p <- p + facet_wrap(as.formula(paste("~", facet)))
  save_fig(p, out_file, width, height)
}


# ==============================================================================================
# 1. Pre-flight -- the CSR disclosure and the flagged z-coverage issue
# ==============================================================================================
#
# The CSR panel is the evidence for one sentence in the response: code/3_detection.py passes
# minspl=3, so poisson_select never runs on real data, and at alpha = 0.5 that rule returns
# exactly 3 for every marker in both samples. Plotting min_samples against alpha makes both halves
# visible at once -- that alpha = 10 (the inert value in the call) would have been far stricter,
# and that alpha = 0.5 reproduces what was used.
#
# The z panel is the KNOWN ISSUE flagged in README.md and deliberately NOT investigated by A3: the
# AD section thins with depth while WT is flat. It is drawn so the number is on record.

if (RUN_PREFLIGHT) {
  message("[1] pre-flight")

  f <- file.path(pre_dir, "csr_min_samples.csv")
  if (need(f, "1")) {
    csr <- read.csv(f) %>% filter(gene_set %in% c("marker", "nc"))
    p <- ggplot(csr, aes(x = alpha, y = min_samples, group = gene, colour = gene_set)) +
      geom_line(alpha = 0.5) + geom_point(size = 0.8, alpha = 0.7) +
      geom_hline(yintercept = 3, linetype = "dashed", colour = "black") +
      geom_vline(xintercept = 0.5, linetype = "dotted", colour = "#4f7fa8") +
      facet_wrap(~ sample) +
      scale_x_log10() + scale_y_log10() +
      scale_colour_manual(values = c(marker = "#4f7fa8", nc = "#d98b5f")) +
      labs(x = expression(alpha), y = "min_samples returned by poisson_select",
           colour = NULL,
           caption = paste("dashed = the value actually used (3); dotted = alpha 0.5,",
                           "at which the CSR rule returns 3 for every marker")) +
      theme_bw() + theme(legend.position = "bottom",
                         plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3a_csr_min_samples.jpeg", 8, 4.5)
  }

  zf <- list.files(pre_dir, pattern = "^z_profile_.*\\.csv$", full.names = TRUE)
  if (length(zf)) {
    z <- do.call(rbind, lapply(zf, read.csv))
    zl <- z %>%
      pivot_longer(c(n_tx, n_granules), names_to = "measure", values_to = "n") %>%
      group_by(sample, measure) %>% mutate(frac = n / sum(n, na.rm = TRUE)) %>% ungroup()
    p <- ggplot(zl, aes(x = z, y = frac, colour = sample)) +
      geom_line(linewidth = 0.9) + geom_point(size = 1.8) +
      facet_wrap(~ measure, scales = "free_y") +
      scale_colour_manual(values = fill_colors) +
      labs(x = "z plane (um)", y = "fraction of total", colour = NULL,
           caption = paste("KNOWN ISSUE, flagged not investigated: the AD section thins with",
                           "depth while WT is flat, so granule counts follow.")) +
      theme_bw() + theme(legend.position = "bottom",
                         plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3a_z_coverage.jpeg", 8, 4)
  } else message("  [skip 1] no z_profile_*.csv")
}


# ==============================================================================================
# 2. What the NC filter actually does
# ==============================================================================================
#
# Three panels, all pre-emptive: a reviewer reading model.py can find every one of these.
#   2a  nc_ratio pairs a post-merge numerator with a pre-merge denominator (`size` is never
#       recomputed by _remove_overlaps), inflating it for the multi-marker granules
#   2b  the NC list is not gene-neutral -- leave-one-out (over the 18-gene list, since Set 1 is a
#       newly detected population) shows which genes carry the filtering
#   2c  Gria2 is on BOTH lists, so Gria2-seeded granules self-filter

if (RUN_NC) {
  message("[2] NC-filter forensics")

  f <- file.path(a3a_dir, "nc_leave_one_out.csv")
  if (need(f, "2")) {
    loo <- read.csv(f)
    p <- ggplot(loo, aes(x = reorder(nc_gene, n_removed), y = n_removed, fill = sample)) +
      geom_col(position = "dodge") + coord_flip() +
      scale_fill_manual(values = fill_colors) +
      scale_y_continuous(labels = comma) +
      labs(x = NULL, y = "Set-1 granules removed by this NC gene alone", fill = NULL,
           caption = paste("The NC list spans complement (C4a), AD risk (Abca7), an",
                           "oligodendrocyte gene (Opalin) and three dentate-gyrus genes,\nso its",
                           "background is itself spatially structured -- the reviewer's own",
                           "concern applied to our filter.")) +
      theme_bw() + theme(legend.position = "bottom",
                         plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3a_nc_leave_one_out.jpeg", 7.5, 6)
  }

  f <- file.path(a3a_dir, "gria2_partition.csv")
  if (need(f, "2")) {
    g <- read.csv(f) %>%
      select(sample, n_removed_gria2, n_removed_other) %>%
      pivot_longer(-sample, names_to = "component", values_to = "n") %>%
      mutate(component = recode(component,
                                n_removed_gria2 = "seeded on Gria2 (list collision)",
                                n_removed_other = "dropped for nc_ratio >= 0.1"))
    p <- ggplot(g, aes(x = sample, y = n, fill = component)) +
      geom_col() + scale_y_continuous(labels = comma) +
      scale_fill_manual(values = c("seeded on Gria2 (list collision)" = "#d98b5f",
                                   "dropped for nc_ratio >= 0.1" = "#4f7fa8")) +
      labs(x = NULL, y = "granules removed (Set 1 -> Set 2)", fill = NULL,
           caption = paste("Gria2 is on both the marker list and the NC list, so nc_filter counts",
                           "it in the numerator and `size` counts it\nin the denominator. The",
                           "collision makes Set 2 CONSERVATIVE -- Gria2 is a canonical",
                           "dendritically transported transcript.")) +
      theme_bw() + theme(legend.position = "bottom",
                         plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3a_gria2_partition.jpeg", 7, 5)
  }
}


# ==============================================================================================
# 3. The sets, as funnels
# ==============================================================================================
#
# Endpoints alone would be circular for Set 3: the NC genes are DEFINED as nuclear-enriched, so
# "few survive the in-soma filter" proves nothing. The funnel shows WHERE each set empties, and the
# per-million-transcript rate removes the 15x abundance gap that would otherwise explain an empty
# Set 3 for free. Set 0 -- neutral genes at marker abundance -- is the arm that makes the point.

if (RUN_SETS) {
  message("[3] set funnels")

  f <- file.path(a3a_dir, "funnel_by_gene.csv")
  if (need(f, "3")) {
    fun <- read.csv(f)
    agg <- fun %>%
      group_by(set, sample) %>%
      summarise(across(c(raw, size, in_soma, n_tx_gene), sum), .groups = "drop") %>%
      pivot_longer(c(raw, size, in_soma), names_to = "stage", values_to = "n") %>%
      mutate(stage = factor(stage, levels = c("raw", "size", "in_soma")),
             rate = n / (n_tx_gene / 1e6))

    p <- ggplot(agg, aes(x = stage, y = n, colour = set, group = set)) +
      geom_line(linewidth = 0.9) + geom_point(size = 2.2) +
      facet_wrap(~ sample) + scale_y_log10(labels = comma) +
      scale_colour_manual(values = set_colors) +
      labs(x = NULL, y = "spheres surviving (log)", colour = NULL) +
      theme_bw() + theme(legend.position = "bottom")
    save_fig(p, "a3a_funnel_counts.jpeg", 8, 4.5)

    p <- ggplot(agg, aes(x = stage, y = rate, colour = set, group = set)) +
      geom_line(linewidth = 0.9) + geom_point(size = 2.2) +
      facet_wrap(~ sample) + scale_y_log10() +
      scale_colour_manual(values = set_colors) +
      labs(x = NULL, y = "spheres per million transcripts of the seeding gene (log)",
           colour = NULL,
           caption = paste("Rate, not count: the NC genes are ~15x rarer than the markers and",
                           "DBSCAN yield is superlinear in count,\nso a raw comparison would be",
                           "an abundance comparison. Set 0 is abundance-matched to the markers.")) +
      theme_bw() + theme(legend.position = "bottom",
                         plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3a_funnel_rates.jpeg", 8, 5)
  }
}


# ==============================================================================================
# 4. The overlap ladder
# ==============================================================================================
#
# `intersect` (d < r_A + r_B) leads because it is the LOOSEST criterion and therefore maximises
# apparent overlap -- a small value under it cannot be argued with. mcDETECT's own merge predicate
# requires centres within 0.4*r and would understate co-location badly if quoted alone.
# Observed/expected is against Set-3 spheres re-placed uniformly in the tissue mask.

if (RUN_OVERLAP) {
  message("[4] overlap ladder")

  f <- file.path(a3a_dir, "overlap_ladder.csv")
  if (need(f, "4")) {
    ov <- read.csv(f) %>%
      mutate(criterion = factor(criterion, levels = c("intersect", "center_in", "merge")))
    p <- ggplot(ov, aes(x = criterion, y = frac_overlapping, fill = sample)) +
      geom_col(position = "dodge") +
      geom_errorbar(aes(ymin = expected_frac - null_sd, ymax = expected_frac + null_sd),
                    position = position_dodge(width = 0.9), width = 0.25, colour = "grey30") +
      geom_point(aes(y = expected_frac), position = position_dodge(width = 0.9),
                 shape = 4, size = 2, colour = "grey20") +
      facet_wrap(~ base) + scale_fill_manual(values = fill_colors) +
      labs(x = NULL, y = "fraction of granules overlapping a Set-3 sphere", fill = NULL,
           caption = paste("x = expected under Set-3 spheres re-placed uniformly in the tissue",
                           "mask at matched radius and layer_z.\n`intersect` is the loosest",
                           "criterion and is reported first on purpose.")) +
      theme_bw() + theme(legend.position = "bottom",
                         plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3a_overlap_ladder.jpeg", 8, 5)
  }
}


# ==============================================================================================
# 5. Per-region density per set, WT vs AD
# ==============================================================================================
#
# The advisor's expectation: Set-3 (negative-control) pseudo-granules show NO WT/AD difference.
# n = 1 vs 1, so this is descriptive -- and the capture-efficiency coefficient is a single global
# scalar whose per-region spread is reported beside it.

if (RUN_DENSITY) {
  message("[5] per-region density")

  f <- file.path(a3a_dir, "set_density_per_region.csv")
  if (need(f, "5")) {
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
# 6. Stage D -- locally-adaptive threshold survival
# ==============================================================================================
#
# READ THIS PANEL WITH THE CAVEAT ATTACHED. The test is one-sided: it re-applies a locally
# estimated min_samples to already-called granules, so it can only REMOVE them. It bounds
# false-positive inflation where background is denser -- which is exactly the reviewer's stated
# failure mode -- and says nothing about false negatives in sparse regions. AD is the lower-density
# arm, so the silent direction works against our own effect.
#
# The deliverable is not "X% survive" but whether the WT/AD result holds on survivors, plus the
# survival rate per sample: a DIFFERENTIAL survival rate between WT and AD would be the reviewer's
# hypothesis confirmed, and it is better reported by us than found by them.

if (RUN_ADAPTIVE) {
  message("[6] adaptive-threshold survival")

  f <- file.path(a3a_dir, "adaptive_survival.csv")
  if (need(f, "6")) {
    s <- read.csv(f)
    cav <- file.path(a3a_dir, "adaptive_caveats.csv")
    cap <- if (file.exists(cav)) paste(strwrap(read.csv(cav)$caveat[1], 110), collapse = "\n")
           else "One-sided: this test can only remove granules, never add them."

    p <- ggplot(s, aes(x = alpha, y = frac_survive, colour = sample,
                       linetype = factor(exclude_granule_tx))) +
      geom_line(linewidth = 0.9) + geom_point(size = 1.8) +
      facet_grid(k_variant ~ R, labeller = label_both) +
      scale_x_log10() + scale_y_continuous(limits = c(0, 1)) +
      scale_colour_manual(values = fill_colors) +
      labs(x = expression(alpha), y = "fraction of granules surviving",
           colour = NULL, linetype = "granule tx excluded", caption = cap) +
      theme_bw() + theme(legend.position = "bottom",
                         plot.caption = element_text(hjust = 0, size = 8, colour = "grey30"))
    save_fig(p, "a3a_adaptive_survival.jpeg", 10, 6.5)
  }
}


# ==============================================================================================
# 7. A3b -- placement and real-vs-pseudo distributions
# ==============================================================================================
#
# DESCRIPTION, NOT EVIDENCE. A matched-radius count comparison is an algebraic identity: sphere_r is
# the minimum enclosing radius of the exact DBSCAN core points, so the real sphere is maximally
# dense by construction and any displaced copy must capture no more. What is informative here is
# the SHAPE of the gap and the filter funnel, plus the fraction of offsets that land on a real
# granule -- which measures how much of the immediate vicinity is already called.

if (RUN_VICINITY) {
  message("[7] vicinity placement + profiles")

  f <- file.path(a3b_dir, "vicinity_overlap_with_real.csv")
  if (need(f, "7")) {
    ov <- read.csv(f) %>% mutate(d = paste0(d_kind, ":", d_label))
    p <- ggplot(ov, aes(x = reorder(d, frac_on_real_granule), y = frac_on_real_granule,
                        fill = sample)) +
      geom_col(position = "dodge") + coord_flip() +
      scale_fill_manual(values = fill_colors) +
      labs(x = "offset", y = "fraction of offsets landing on a real granule", fill = NULL,
           caption = paste("A result, not a nuisance: this is how much of the immediate vicinity",
                           "is already called.\nPer-plane granule coverage is only ~1.9% (WT) /",
                           "1.5% (AD), which is why rejecting overlaps barely biases the sample.")) +
      theme_bw() + theme(legend.position = "bottom",
                         plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3b_vicinity_overlap.jpeg", 7.5, 5)
  }

  f <- file.path(a3b_dir, "profile_histogram.parquet")
  if (need(f, "7")) {
    h <- read_parquet(f) %>% filter(measure == "n_marker")
    h$kind <- ifelse(grepl("^real", h$arm), "real", "pseudo")
    hh <- h %>% group_by(sample, kind, bin_lo, bin_hi) %>%
      summarise(frac = mean(frac), .groups = "drop") %>% filter(is.finite(bin_hi))
    binned_hist(hh, "a3b_marker_counts.jpeg",
                x_lab = "marker transcripts inside the sphere", fill_var = "kind",
                palette = c(real = "#4f7fa8", pseudo = "#d98b5f"), facet = "sample",
                caption = paste("Descriptive only: a matched-radius displaced copy of a minimum",
                                "enclosing sphere must capture no more.\nThe decision is",
                                "section 8."),
                width = 8, height = 4.5)
  }
}


# ==============================================================================================
# 8. A3b -- THE detection predicate
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
  message("[8] detection predicate")

  f <- file.path(a3b_dir, "detection_predicate.csv")
  if (need(f, "8")) {
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
  if (need(f, "8")) {
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
# 9. A3c -- Axis 1, the compartment contrast
# ==============================================================================================
#
# The reviewer's literal ask: DE between somatic and all non-somatic RNA, independent of granule
# detection, then whether the granule-specific differences exceed or DIVERGE FROM that baseline.
#
# The scatter is lifted from code/figures_response.Rmd:1424-1452 (which already sits under a
# heading titled "Reviewer 2, Major Comment 9") with two changes: the baseline is now
# granule-FREE -- residual extrasomatic vs soma, so it no longer contains the signal it is a null
# for -- and granule enrichment now uses the SAME soma reference, so the two axes share a
# denominator and their difference is meaningful.

if (RUN_AXIS1) {
  message("[9] axis 1 -- compartment")

  f <- file.path(a3c_dir, "axis1_gene_table.csv")
  if (need(f, "9")) {
    df <- read.csv(f) %>%
      mutate(marker_group = ifelse(is_marker == "True" | is_marker == TRUE,
                                   "Granule markers", "Others"))
    p <- ggplot(df, aes(x = baseline_logFC, y = granule_enrichment)) +
      geom_abline(slope = 1, intercept = 0, linetype = "dotted", colour = "grey50") +
      geom_smooth(data = df %>% filter(marker_group == "Others"),
                  method = "lm", se = TRUE, colour = "black", linewidth = 0.75) +
      geom_point(aes(fill = marker_group), shape = 21, size = 2, colour = "black",
                 stroke = 0.1) +
      facet_wrap(~ sample) +
      scale_fill_manual(values = c("Granule markers" = "#f48488", "Others" = "#a0ccec")) +
      labs(x = "Baseline logFC (residual extrasomatic vs soma)",
           y = "Granule enrichment logFC (granule vs soma)", fill = NULL,
           caption = paste("Line fitted on NON-markers. Both axes share the soma reference, so",
                           "their difference is meaningful;\nthe baseline excludes in-granule",
                           "transcripts, so it is a genuine null rather than a diluted signal.")) +
      theme_classic() +
      theme(axis.text = element_text(size = 12), axis.title = element_text(size = 13),
            legend.position = "bottom",
            plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3c_axis1_scatter.jpeg", 9, 5.5)

    p <- ggplot(df, aes(x = marker_group, y = residual, fill = marker_group)) +
      geom_violin(alpha = 0.7, colour = "grey30") +
      geom_boxplot(width = 0.15, outlier.size = 0.4, fill = "white") +
      facet_wrap(~ sample) +
      scale_fill_manual(values = c("Granule markers" = "#f48488", "Others" = "#a0ccec")) +
      labs(x = NULL, y = "residual from the non-marker line", fill = NULL,
           caption = paste("Divergence, not excess: the reviewer's wording is 'exceed OR DIVERGE",
                           "FROM'. A residual test is invariant\nto the compositional rescaling",
                           "that makes an absolute |logFC| comparison fragile.")) +
      theme_classic() + theme(legend.position = "none",
                              plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3c_axis1_residual.jpeg", 8, 5)
  }
}


# ==============================================================================================
# 10. A3c -- Axis 2, conditions and regions
# ==============================================================================================
#
# RANKINGS ONLY. The three layers differ enormously in counts per spot and in sparsity, and a rank
# test's power tracks that -- which is why the published ambient and cell layers show 253 and 234
# significant genes against the granule layer's 161. Plotting those tallies would invite the
# reading "the authors' ambient layer yields more DE genes than their granule layer". So this
# section plots correlations and logFC agreement, never counts.
#
# And n = 1 vs 1: the WT/AD contrast here is descriptive.

if (RUN_AXIS2) {
  message("[10] axis 2 -- conditions and regions")

  f <- file.path(a3c_dir, "axis2_layer_correlation.csv")
  if (need(f, "10")) {
    cr <- read.csv(f) %>% mutate(pair = paste(layer_a, "vs", layer_b))
    p <- ggplot(cr, aes(x = reorder(pair, spearman_rho), y = spearman_rho)) +
      geom_col(fill = "#4f7fa8") + coord_flip() +
      geom_text(aes(label = sprintf("%.3f", spearman_rho)), hjust = -0.15, size = 3.5) +
      scale_y_continuous(limits = c(0, 1.05)) +
      labs(x = NULL, y = "Spearman rho of WT-vs-AD logFC between layers",
           caption = paste("Rankings only -- significant-gene COUNTS are not comparable across",
                           "layers, because the layers differ\nin depth and sparsity and a rank",
                           "test's power tracks that.")) +
      theme_bw() + theme(plot.caption = element_text(hjust = 0, size = 9, colour = "grey30"))
    save_fig(p, "a3c_axis2_layer_correlation.jpeg", 7.5, 4)
  }

  f <- file.path(a3c_dir, "axis2_wt_ad_by_layer.csv")
  if (need(f, "10")) {
    a2 <- read.csv(f)
    w <- a2 %>% select(gene, layer, logFC_AD_vs_WT, is_marker) %>%
      pivot_wider(names_from = layer, values_from = logFC_AD_vs_WT)
    if (all(c("granule", "residual_extrasomatic") %in% names(w))) {
      w$marker_group <- ifelse(w$is_marker == "True" | w$is_marker == TRUE,
                               "Granule markers", "Others")
      p <- ggplot(w, aes(x = residual_extrasomatic, y = granule)) +
        geom_abline(slope = 1, intercept = 0, linetype = "dotted", colour = "grey50") +
        geom_point(aes(fill = marker_group), shape = 21, size = 2, colour = "black",
                   stroke = 0.1) +
        scale_fill_manual(values = c("Granule markers" = "#f48488", "Others" = "#a0ccec")) +
        labs(x = "AD vs WT logFC, residual extrasomatic (detection-independent)",
             y = "AD vs WT logFC, granule layer", fill = NULL,
             caption = paste("If the AD granule signal were an ambient artefact these would lie",
                             "on the dotted line.\nn = 1 vs 1 section: descriptive.")) +
        theme_classic() + theme(legend.position = "bottom",
                                plot.caption = element_text(hjust = 0, size = 9,
                                                            colour = "grey30"))
      save_fig(p, "a3c_axis2_granule_vs_ambient.jpeg", 7, 5.5)
    }
  }
}

message("done -- figures in ", fig_dir)
