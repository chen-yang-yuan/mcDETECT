# ==============================================================================================
# A2 -- figures for the sparsity / stochastic-origin analyses
#
#   plans/Round2_response_analysis_plan.md, section A2.
#
# Sections 1-4 plot A2a (`A2a_multigene.ipynb`); section 5 plots A2b
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
RUN_READSTRATA  <- TRUE    # 4. read-count terciles
RUN_A2B         <- TRUE    # 5. permutation null vs real embedding

# Optional selectors for section 3; NULL = every DE table found.
GSEA_PAIRS <- NULL         # e.g. c("Subdomain 3_vs_Subdomain 1")

dpi <- 500

# -------------------- paths -------------------- #
root        <- here::here("R2_revision/sparsity_structure/output")
a2a_dir     <- file.path(root, "a2a", "multigene")
strata_dir  <- file.path(root, "a2a", "readstrata")
a2b_dir     <- file.path(root, "a2b", "metrics")
sub_dir     <- file.path(a2a_dir, "neuropil_subdomains_Isocortex_50")
fig_dir     <- file.path(root, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

# The published reference tables, for side-by-side anchors.
pub_dir       <- here::here("output/MERSCOPE_WT_AD_comparison")
pub_density   <- file.path(pub_dir, "subtype_density_per_region_granule_adata_tsne.csv")
pub_subdomain <- file.path(pub_dir, "neuropil_subdomains_Isocortex_50")

# -------------------- palette and shared helpers -------------------- #
# a2_config.py keeps the same two hex values; R cannot import it, so change both together.
fill_colors <- c(WT = "#a0ccec", AD = "#f48488")
cond_colors <- c(real = "#4f7fa8", permuted = "#b0b0b0")
area_order  <- c("Isocortex", "OLF", "HPF-CA", "HPF-DG", "HPF-SR", "CTXsp", "TH", "MB", "FT")
subtype_order <- c("overall", "pre-syn", "post-syn", "dendrites", "axons", "mixed")

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
    for (m in c("n_reads", "n_genes")) {
      binned_hist(hist_df %>% filter(measure == m, population == "all"),
                  file.path(fig_dir, paste0("complexity_", m, "_all.jpeg")),
                  x_lab = if (m == "n_reads") "Reads per granule" else "Unique genes per granule",
                  fill_var = "sample", palette = fill_colors)
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
                                   " unique-gene cutoff."))
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
           y = "Unique genes per granule (whole panel)",
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
}


# ==============================================================================================
# 3. Neuropil-microdomain DE -> GSEA
# ==============================================================================================
#
# NAMESPACE NOTE (inherited from A1_figures.R): org.Mm.eg.db is deliberately NOT loaded --
# AnnotationDbi's S4 `select` masks dplyr::select. Everything below is dplyr:: qualified as a
# guard. The GSEA core is copied verbatim from code/figures_response.Rmd:561-700; only the file
# discovery around it is new, so a run with several SUBDOMAIN_PAIRS needs no edit here.

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
  message("  wrote ", stem, "_GSEA.csv + dotplots")
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

    # The published Subdomain 1-vs-2 GSEA is the anchor: the question is whether the multi-gene
    # subset recovers the same pathways, so both must be on the table.
    pub_gsea <- list.files(pub_subdomain, pattern = "_GSEA\\.csv$", full.names = TRUE)
    if (length(pub_gsea)) {
      new_gsea <- list.files(sub_dir, pattern = "_GSEA\\.csv$", full.names = TRUE)
      read_terms <- function(f, tag) {
        read.csv(f, check.names = FALSE) %>%
          dplyr::transmute(source = tag, file = basename(f), ID, NES, p.adjust)
      }
      cmp <- dplyr::bind_rows(
        dplyr::bind_rows(lapply(pub_gsea, read_terms, tag = "published_all_granules")),
        dplyr::bind_rows(lapply(new_gsea, read_terms, tag = "multigene")))
      write.csv(cmp, file.path(fig_dir, "gsea_terms_published_vs_multigene.csv"), row.names = FALSE)
      message("  wrote gsea_terms_published_vs_multigene.csv")
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

message("done -- figures in ", fig_dir)
