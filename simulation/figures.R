library(tidyverse)
library(reshape)

here::i_am("simulation/figures.R")

dpi <- 500 # adjust the dpi for all figures



#-------------------- Single-marker 3D/2D simulation results --------------------#

for (setting in c("3D", "2D")){

  data_A <- read.csv(here::here(paste0("simulation/output/single_marker_", setting, "_A_5000_2000.csv"))) %>%
    mutate(Scenario = "Marker A")
  
  data_B <- read.csv(here::here(paste0("simulation/output/single_marker_", setting, "_B_3000_1200.csv"))) %>%
    mutate(Scenario = "Marker B")
  
  data_C <- read.csv(here::here(paste0("simulation/output/single_marker_", setting, "_C_2000_800.csv"))) %>%
    mutate(Scenario = "Marker C")
  
  combined_data <- bind_rows(data_A, data_B, data_C)
  
  long_data <- combined_data %>%
    pivot_longer(cols = c(Accuracy, Precision, Recall, F1.Score), names_to = "Metric", values_to = "Value") %>%
    mutate(Metric = factor(Metric, levels = c("Accuracy", "Precision", "Recall", "F1.Score"), labels = c("Accuracy", "Precision", "Recall", "F1 Score")))
  
  summary_data <- long_data %>%
    group_by(Scenario, Metric) %>%
    summarise(mean_value = mean(Value, na.rm = TRUE), sd_value = sd(Value, na.rm = TRUE), .groups = "drop") %>%
    mutate(ymin = round(mean_value - sd_value, 4), ymax = round(mean_value + sd_value, 4))
  
  p <- ggplot(summary_data, aes(x = Scenario, y = mean_value, fill = Metric)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.75), color = "black", width = 0.6) +
    geom_errorbar(aes(ymin = ymin, ymax = ymax), position = position_dodge(width = 0.75), width = 0.3) +
    coord_cartesian(ylim = c(0.85, 1)) +
    scale_fill_manual(values = c("#f48488", "#a0ccec", "#f0c47c", "#8BBFA8")) +
    labs(x = " ", y = "Metric Value", fill = " ") +
    theme_classic() +
    theme(axis.title.y = element_text(size = 15),
          axis.text.x = element_text(size = 15),
          axis.text.y = element_text(size = 15),
          legend.position = "bottom",
          legend.box.margin = margin(t = -20, b = 0),
          legend.text = element_text(size = 15))
  
  ggsave(here::here(paste0("simulation/output/single_marker_", setting, ".jpeg")), p, width = 6, height = 6, dpi = dpi)

}



#-------------------- Multi-marker 3D/2D simulation results --------------------#

for (setting in c("3D", "2D")){

  data_A <- read.csv(here::here(paste0("simulation/output/multi_marker_", setting, "_A_5000_1000.csv"))) %>%
    mutate(Scenario = "Marker A")
  
  data_B <- read.csv(here::here(paste0("simulation/output/multi_marker_", setting, "_B_5000_1000.csv"))) %>%
    mutate(Scenario = "Marker B")
  
  data_C <- read.csv(here::here(paste0("simulation/output/multi_marker_", setting, "_C_5000_1000.csv"))) %>%
    mutate(Scenario = "Marker C")
  
  data_all <- read.csv(here::here(paste0("simulation/output/multi_marker_", setting, "_all_5000_1000.csv"))) %>%
    mutate(Scenario = "Combined")
  
  combined_data <- bind_rows(data_A, data_B, data_C, data_all)
  
  long_data <- combined_data %>%
    pivot_longer(cols = c(Accuracy, Precision, Recall, F1), names_to = "Metric", values_to = "Value")  %>%
    mutate(Metric = factor(Metric, levels = c("Accuracy", "Precision", "Recall", "F1"), labels = c("Accuracy", "Precision", "Recall", "F1 Score")),
           Scenario = factor(Scenario, levels = c("Marker A", "Marker B", "Marker C", "Combined")))
  
  long_data <- long_data[long_data$Scenario == "Combined", ] # only visualize the combined scenario!
  
  summary_data <- long_data %>%
    group_by(Scenario, Metric) %>%
    summarise(mean_value = mean(Value, na.rm = TRUE), sd_value = sd(Value, na.rm = TRUE), .groups = "drop") %>%
    mutate(ymin = round(mean_value - sd_value, 4), ymax = round(mean_value + sd_value, 4))
  
  p <- ggplot(summary_data, aes(x = Scenario, y = mean_value, fill = Metric)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.6), color = "black", width = 0.475) +
    geom_errorbar(aes(ymin = ymin, ymax = ymax), position = position_dodge(width = 0.6), width = 0.2375) +
    coord_cartesian(ylim = c(0.85, 1)) +
    scale_fill_manual(values = c("#f48488", "#a0ccec", "#f0c47c", "#8BBFA8")) +
    labs(x = " ", y = "Metric Value", fill = " ") +
    theme_classic() +
    theme(axis.title.y = element_text(size = 15),
          axis.text.x = element_text(size = 15),
          axis.text.y = element_text(size = 15),
          legend.position = "bottom",
          legend.box.margin = margin(t = -20, b = 0),
          legend.text = element_text(size = 15))
  
  ggsave(here::here(paste0("simulation/output/multi_marker_", setting, ".jpeg")), p, width = 3, height = 6, dpi = dpi)

}



#-------------------- Benchmark rho, the merging threshold --------------------#

data <- read.csv(here::here("simulation/output/p_benchmark_multi_marker_3D_detailed.csv"))

# calculate summary statistics for each p value
summary_data <- data %>%
  group_by(p) %>%
  summarise(
    mean_detections = mean(num_detections),
    sd_detections = sd(num_detections),
    mean_aggregates_per_transcript = mean(avg_aggregates_per_transcript),
    sd_aggregates_per_transcript = sd(avg_aggregates_per_transcript),
    .groups = 'drop'
  )

# number of detections vs rho
p1 <- ggplot(summary_data, aes(x = p, y = mean_detections)) +
  geom_hline(yintercept = 3000, linetype = "dashed", color = "red", linewidth = 0.8) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(shape = 21, fill = "steelblue", color = "black", size = 3, stroke = 0.7) +
  # geom_pointrange(aes(ymin = mean_detections - sd_detections, 
  #                     ymax = mean_detections + sd_detections),
  #                 color = "black",
  #                 fill = "steelblue",
  #                 shape = 21,
  #                 stroke = 0.7,
  #                 fatten = 6,
  #                 linewidth = 0.7) +
  scale_x_continuous(breaks = seq(0, 1, 0.1), limits = c(0, 1)) +
  labs(x = "\u03C1 (merging threshold parameter)",
       y = "Number of Detections") +
  theme_classic() +
  theme(axis.title = element_text(size = 15),
        axis.text = element_text(size = 15))

ggsave(here::here("simulation/output/benckmark_p_number_of_detections.jpeg"), p1, width = 10, height = 6, dpi = dpi)

# average number of aggregates per transcript vs rho
p2 <- ggplot(summary_data, aes(x = p, y = mean_aggregates_per_transcript)) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 0.8) +
  geom_line(color = "darkgreen", linewidth = 1) +
  geom_point(shape = 21, fill = "darkgreen", color = "black", size = 3, stroke = 0.7) +
  # geom_pointrange(aes(ymin = mean_aggregates_per_transcript - sd_aggregates_per_transcript, 
  #                     ymax = mean_aggregates_per_transcript + sd_aggregates_per_transcript),
  #                 color = "black",
  #                 fill = "darkgreen",
  #                 shape = 21,
  #                 stroke = 0.7,
  #                 fatten = 2,
  #                 linewidth = 0.7) +
  scale_x_continuous(breaks = seq(0, 1, 0.1), limits = c(0, 1)) +
  labs(x = "\u03C1 (merging threshold parameter)",
       y = "Average Number of Aggregates per Transcript") +
  theme_classic() +
  theme(axis.title = element_text(size = 15),
        axis.text = element_text(size = 15))

ggsave(here::here("simulation/output/benckmark_p_number_of_aggregates_per_transcripts.jpeg"), p2, width = 10, height = 6, dpi = dpi)



#-------------------- Benchmark additional markers with varying densities --------------------#

for (setting in c("3D", "2D")){
  
  data_A <- read.csv(here::here(paste0("simulation/output/single_marker_", setting, "_A_5000_2000.csv"))) %>%
    mutate(Scenario = "Marker A")
  
  data_B <- read.csv(here::here(paste0("simulation/output/single_marker_", setting, "_B_3000_1200.csv"))) %>%
    mutate(Scenario = "Marker B")
  
  data_C <- read.csv(here::here(paste0("simulation/output/single_marker_", setting, "_C_2000_800.csv"))) %>%
    mutate(Scenario = "Marker C")
  
  data_D <- read.csv(here::here(paste0("simulation/output/single_marker_", setting, "_D_1250_500.csv"))) %>%
    mutate(Scenario = "Marker D")
  
  data_E <- read.csv(here::here(paste0("simulation/output/single_marker_", setting, "_E_800_300.csv"))) %>%
    mutate(Scenario = "Marker E")
  
  data_F <- read.csv(here::here(paste0("simulation/output/single_marker_", setting, "_F_500_200.csv"))) %>%
    mutate(Scenario = "Marker F")
  
  combined_data <- bind_rows(data_A, data_B, data_C, data_D, data_E, data_F)
  
  long_data <- combined_data %>%
    pivot_longer(cols = c(Accuracy, Precision, Recall, F1.Score), names_to = "Metric", values_to = "Value") %>%
    mutate(Metric = factor(Metric, levels = c("Accuracy", "Precision", "Recall", "F1.Score"), labels = c("Accuracy", "Precision", "Recall", "F1 Score")))
  
  summary_data <- long_data %>%
    group_by(Scenario, Metric) %>%
    summarise(mean_value = mean(Value, na.rm = TRUE), sd_value = sd(Value, na.rm = TRUE), .groups = "drop") %>%
    mutate(ymin = round(mean_value - sd_value, 4), ymax = round(mean_value + sd_value, 4))
  
  p <- ggplot(summary_data, aes(x = Scenario, y = mean_value, fill = Metric)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), color = "black", width = 0.6) +
    geom_errorbar(aes(ymin = ymin, ymax = ymax), position = position_dodge(width = 0.8), width = 0.3) +
    coord_cartesian(ylim = c(0.6, 1)) +
    scale_fill_manual(values = c("#f48488", "#a0ccec", "#f0c47c", "#8BBFA8")) +
    labs(x = " ", y = "Metric Value", fill = " ") +
    theme_classic() +
    theme(axis.title.y = element_text(size = 15),
          axis.text.x = element_text(size = 15),
          axis.text.y = element_text(size = 15),
          legend.position = "bottom",
          legend.box.margin = margin(t = -20, b = 0),
          legend.text = element_text(size = 15))
  
  ggsave(here::here(paste0("simulation/output/single_marker_", setting, "_additional_markers.jpeg")), p, width = 15, height = 5, dpi = dpi)
  
}



#-------------------- Benchmark CSR ratio --------------------#

all_data <- list()
k <- 1

for (i in sprintf("%.2f", seq(0.2, 0.6, 0.1))) {
  
  csr_ratio <- i
  extra_ratio <- sprintf("%.2f", (1 - as.numeric(i)) / 2)
  intra_ratio <- extra_ratio
  
  data_A <- read.csv(here::here(paste0("simulation/output/benchmark_ratio_csr_A_csr", csr_ratio, "_extra", extra_ratio, "_intra", intra_ratio, ".csv"))) %>%
    mutate(Scenario = "Marker A")
  
  data_B <- read.csv(here::here(paste0("simulation/output/benchmark_ratio_csr_B_csr", csr_ratio, "_extra", extra_ratio, "_intra", intra_ratio, ".csv"))) %>%
    mutate(Scenario = "Marker B")
  
  data_C <- read.csv(here::here(paste0("simulation/output/benchmark_ratio_csr_C_csr", csr_ratio, "_extra", extra_ratio, "_intra", intra_ratio, ".csv"))) %>%
    mutate(Scenario = "Marker C")
  
  all_data[[k]] <- data_A; k <- k + 1
  all_data[[k]] <- data_B; k <- k + 1
  all_data[[k]] <- data_C; k <- k + 1
  
}

combined_data <- dplyr::bind_rows(all_data)

summary_df <- combined_data %>%
  mutate(CSR_ratio = as.numeric(CSR_ratio)) %>%
  group_by(CSR_ratio, Scenario) %>%
  summarise(Precision_mean = mean(Precision, na.rm = TRUE),
            Precision_sd = sd(Precision, na.rm = TRUE),
            Recall_mean = mean(Recall, na.rm = TRUE),
            Recall_sd = sd(Recall, na.rm = TRUE),
            Accuracy_mean = mean(Accuracy, na.rm = TRUE),
            Accuracy_sd = sd(Accuracy, na.rm = TRUE),
            F1_mean = mean(F1_Score, na.rm = TRUE),
            F1_sd = sd(F1_Score, na.rm = TRUE),
            .groups = "drop")

summary_long <- summary_df %>%
  pivot_longer(cols = -c(CSR_ratio, Scenario),
               names_to = c("Metric", ".value"),
               names_pattern = "(.*)_(mean|sd)")

metric_labels <- c("Precision" = "Precision", "Recall" = "Recall", "Accuracy" = "Accuracy", "F1" = "F1 Score")

plot_metric <- function(metric_name){
  
  df <- summary_long %>% filter(Metric == metric_name)
  display_name <- metric_labels[[metric_name]]
  
  ggplot(df, aes(x = CSR_ratio, y = mean, color = Scenario, group = Scenario)) +
    geom_line(linewidth = 1, show.legend = FALSE) +
    geom_point(aes(fill = Scenario), shape = 21, color = "black", size = 3, stroke = 0.7) +
    # geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.02, linewidth = 0.6) +
    coord_cartesian(ylim = c(0.85, 1)) +
    scale_x_continuous(breaks = seq(0.2, 0.6, 0.1), limits = c(0.2, 0.6)) +
    labs(x = "CSR ratio", y = paste0("Mean ", display_name), color = " ") +
    theme_classic() + 
    theme(axis.title = element_text(size = 15),
          axis.text = element_text(size = 15),
          legend.position = "bottom",
          legend.box.margin = margin(t = -10, b = 0),
          legend.title = element_blank(),
          legend.text = element_text(size = 15))
}

for (metric in c("Precision", "Recall", "Accuracy", "F1")){
  p <- plot_metric(metric)
  ggsave(here::here(paste0("simulation/output/benchmark_CSR_ratio_", metric, ".jpeg")), p, width = 7, height = 6, dpi = dpi)
}



#-------------------- Benchmark extrasomatic ratio --------------------#

all_data <- list()
k <- 1

for (i in sprintf("%.3f", seq(0.1, 0.4, 0.075))) {
  
  extra_ratio <- i
  intra_ratio <- sprintf("%.3f", 0.5 - as.numeric(i))
  
  data_A <- read.csv(here::here(paste0("simulation/output/benchmark_ratio_fixedcsr_A_csr0.50_extra", extra_ratio, "_intra", intra_ratio, ".csv"))) %>%
    mutate(Scenario = "Marker A")
  
  data_B <- read.csv(here::here(paste0("simulation/output/benchmark_ratio_fixedcsr_B_csr0.50_extra", extra_ratio, "_intra", intra_ratio, ".csv"))) %>%
    mutate(Scenario = "Marker B")
  
  data_C <- read.csv(here::here(paste0("simulation/output/benchmark_ratio_fixedcsr_C_csr0.50_extra", extra_ratio, "_intra", intra_ratio, ".csv"))) %>%
    mutate(Scenario = "Marker C")
  
  all_data[[k]] <- data_A; k <- k + 1
  all_data[[k]] <- data_B; k <- k + 1
  all_data[[k]] <- data_C; k <- k + 1
  
}

combined_data <- dplyr::bind_rows(all_data)

summary_df <- combined_data %>%
  mutate(Extra_ratio = as.numeric(Extra_ratio)) %>%
  group_by(Extra_ratio, Scenario) %>%
  summarise(Precision_mean = mean(Precision, na.rm = TRUE),
            Precision_sd = sd(Precision, na.rm = TRUE),
            Recall_mean = mean(Recall, na.rm = TRUE),
            Recall_sd = sd(Recall, na.rm = TRUE),
            Accuracy_mean = mean(Accuracy, na.rm = TRUE),
            Accuracy_sd = sd(Accuracy, na.rm = TRUE),
            F1_mean = mean(F1_Score, na.rm = TRUE),
            F1_sd = sd(F1_Score, na.rm = TRUE),
            .groups = "drop")

summary_long <- summary_df %>%
  pivot_longer(cols = -c(Extra_ratio, Scenario),
               names_to = c("Metric", ".value"),
               names_pattern = "(.*)_(mean|sd)")

metric_labels <- c("Precision" = "Precision", "Recall" = "Recall", "Accuracy" = "Accuracy", "F1" = "F1 Score")

plot_metric <- function(metric_name){
  
  df <- summary_long %>% filter(Metric == metric_name)
  display_name <- metric_labels[[metric_name]]
  
  ggplot(df, aes(x = Extra_ratio, y = mean, color = Scenario, group = Scenario)) +
    geom_line(linewidth = 1, show.legend = FALSE) +
    geom_point(aes(fill = Scenario), shape = 21, color = "black", size = 3, stroke = 0.7) +
    # geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.02, linewidth = 0.6) +
    coord_cartesian(ylim = c(0.85, 1)) +
    scale_x_continuous(breaks = seq(0.1, 0.4, 0.075), limits = c(0.1, 0.4)) +
    labs(x = "Extrasomatic ratio", y = paste0("Mean ", display_name), color = " ") +
    theme_classic() + 
    theme(axis.title = element_text(size = 15),
          axis.text = element_text(size = 15),
          legend.position = "bottom",
          legend.box.margin = margin(t = -10, b = 0),
          legend.title = element_blank(),
          legend.text = element_text(size = 15))
}

for (metric in c("Precision", "Recall", "Accuracy", "F1")){
  p <- plot_metric(metric)
  ggsave(here::here(paste0("simulation/output/benchmark_Extrasomatic_ratio_", metric, ".jpeg")), p, width = 7, height = 6, dpi = dpi)
}











