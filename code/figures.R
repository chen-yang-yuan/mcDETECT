library(arrow)
library(tidyverse)
library(reshape)

here::i_am("code/figures.R")

dpi <- 500 # adjust the dpi for all figures



#-------------------- [Reviewer 2, Major Comment 4] Justify simulation size --------------------#

areas_Xenium <- read.csv(here::here("validation/DAPI_dilation/areas_Xenium.csv")) %>%
  mutate(dataset = "Xenium 5K")

areas_MERSCOPE <- read.csv(here::here("validation/DAPI_dilation/areas_MERSCOPE_small_size_moderate_dilation.csv")) %>%
  filter(area >= quantile(area, 0.05, na.rm = TRUE), area <= quantile(area, 0.98, na.rm = TRUE)) %>%
  mutate(dataset = "MERSCOPE")

areas_simulation <- read.csv(here::here("simulation/output/intranuclear_area.csv")) %>%
  mutate(dataset = "Simulation")

df <- bind_rows(areas_Xenium, areas_MERSCOPE, areas_simulation) %>%
  mutate(dataset = factor(dataset, levels = c("Xenium 5K", "MERSCOPE", "Simulation")))

df_outliers <- df %>%
  group_by(dataset) %>%
  mutate(q1 = quantile(area, 0.25, na.rm = TRUE),
         q3 = quantile(area, 0.75, na.rm = TRUE),
         iqr = q3 - q1,
         lower = q1 - 1.5 * iqr,
         upper = q3 + 1.5 * iqr) %>%
  filter(area < lower | area > upper) %>%
  ungroup()

p <- ggplot(df, aes(x = dataset, y = area, fill = dataset)) +
  geom_boxplot(position = position_dodge(width = 0.6), width = 0.4, outlier.shape = NA, fatten = 1.2) +
  geom_point(data = df_outliers, aes(x = dataset, y = area, group = dataset), position = position_dodge(width = 0.6), size = 0.2, alpha = 0.5, show.legend = FALSE) +
  scale_fill_manual(values = scales::alpha(c("#1f77b4", "#1f77b4", "#1f77b4"), 0.8)) +
  labs(x = NULL, y = "Cell / Object Area", fill = NULL) +
  theme_classic() +
  theme(axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15),
        legend.position = "none")

ggsave(here::here("validation/justify_simulation/simulation_area.jpeg"), p, width = 5, height = 6, dpi = dpi)



#-------------------- [Reviewer 2, Major Comment 4] Justify simulation in-soma ratio --------------------#

df <- read_parquet(here::here("output/MERSCOPE_WT_1/all_granules.parquet"))

cutoff <- 0.2
df_cut <- df[df$in_soma_ratio < cutoff, ]
print(nrow(df_cut[df_cut$in_soma_ratio == 0, ]) / nrow(df_cut))

p <- ggplot(df, aes(x = in_soma_ratio)) +
  geom_histogram(binwidth = 0.025, color = "black", linewidth = 0.25, fill = "#6fafd2", alpha = 1) +
  geom_vline(xintercept = cutoff, color = "#b71c2c", linetype = "dashed", linewidth = 0.8) +
  # annotate("text", x = cutoff + 0.02, y = Inf, label = cutoff, color = "#b71c2c", hjust = 0, vjust = 1.5, size = 5) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.2)) +
  scale_y_continuous(expand = c(0, 0)) +
  labs(x = " ", y = "Count") +
  theme_classic() +
  theme(axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 15),
        axis.text.y = element_text(size = 15))

ggsave(here::here("validation/justify_simulation/all_granules_in_soma_ratio.jpeg"), p, width = 7, height = 6, dpi = dpi)
