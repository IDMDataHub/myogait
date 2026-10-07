# =============================================================================
# LEVEL 3 -- cohort curve agreement (Sapiens2 vs Vicon, per joint)
# Two statistical units: run (n = paired runs) and patient (n = 15).
# Stratified all / SAIN / NMD. Metrics from 01_data_prepared/agreement_curves.csv
# (per-run mean cycle curves, side-matched):
#   rmse, bias (video - Vicon), rmse_centered (offset removed = shape error),
#   shape_r (Pearson), cmc, rom_err.
# Outputs: tables/L3_curve_agreement_summary.csv, figures/L3_cohort/agreement_forest.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

metrics <- c(rmse = "RMSE (°)", bias = "Bias (°)", rmse_centered = "Centred RMSE (°)",
             shape_r = "Pearson r", cmc = "CMC", rom_err = "ROM error (°)")
run_lvl <- agree %>% group_by(pair_id, setting, patient, group, joint) %>%
  summarise(across(all_of(names(metrics)), ~ mean(.x, na.rm = TRUE)), .groups = "drop")
pat_lvl <- run_lvl %>% group_by(setting, patient, group, joint) %>%
  summarise(across(all_of(names(metrics)), ~ mean(.x, na.rm = TRUE)), .groups = "drop")

summ <- function(df, level) {
  bind_rows(df %>% mutate(stratum = "all"),
            df %>% filter(group %in% c("SAIN", "NMD")) %>% mutate(stratum = as.character(group))) %>%
    pivot_longer(all_of(names(metrics)), names_to = "metric") %>%
    group_by(setting, stratum, joint, metric) %>%
    summarise(mean_ci(value), .groups = "drop") %>%
    mutate(level = level, .before = 1)
}
tab <- bind_rows(summ(run_lvl, "run"), summ(pat_lvl, "patient"))
save_tab(tab, "L3_curve_agreement_summary.csv")

p <- tab %>% filter(level == "patient", stratum == "all") %>%
  mutate(metric = factor(metrics[metric], metrics), joint = factor(joint, JOINTS)) %>%
  ggplot(aes(mean, joint, colour = setting)) +
  geom_vline(xintercept = 0, colour = "grey70") +
  geom_pointrange(aes(xmin = ci_low, xmax = ci_high), position = position_dodge(width = .6)) +
  facet_wrap(~metric, scales = "free_x", nrow = 2) +
  scale_colour_manual(values = COL_SETTING, name = NULL) + guides(colour = guide_legend(ncol = 2)) +
  labs(title = "Sapiens2 vs Vicon - curve agreement per joint and setting (subject level, mean ± 95% CI)",
       x = NULL, y = NULL) + theme_mk()
save_fig(p, "L3_cohort", "agreement_forest.png", 12, 7)
msg("L3 agreement tables done")
