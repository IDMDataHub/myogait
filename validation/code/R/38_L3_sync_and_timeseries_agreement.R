# =============================================================================
# LEVEL 3 -- quality of the frame-level synchronisation and agreement of the
# synchronised 100 Hz series (no cycle normalisation of the video).
#  * sync.csv: lag (video time at Vicon t = 0), cross-correlation r, L/R swap
#  * agreement_synced_timeseries.csv: RMSE / bias / centred RMSE / r per joint
# Outputs: tables/L3_sync_quality.csv, tables/L3_synced_agreement_summary.csv,
#          figures/L3_cohort/sync_quality.png, figures/L3_cohort/synced_agreement.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

q <- sync %>% group_by(setting) %>% summarise(n_runs = n(), n_sync_ok = sum(sync_ok), r_mean = mean(r), r_min = min(r), r_median = median(r),
                        n_side_swap = sum(side_swap), overlap_mean_s = mean(overlap_s), .groups = "drop")
save_tab(q, "L3_sync_quality.csv")
p1 <- ggplot(sync, aes(r, fill = sync_ok)) + geom_histogram(bins = 25, colour = "white") +
  geom_vline(xintercept = 0.9, linetype = 2) + facet_wrap(~setting, ncol = 1, scales = "free_y") +
  scale_fill_manual(values = c(`FALSE` = "#E4572E", `TRUE` = "#22314E"), name = "Reliable (r >= 0.9)") +
  labs(title = "Frame-level synchronisation quality (cross-correlation of hip/knee/ankle)",
       x = "Cross-correlation r at the best lag", y = "Runs") + theme_mk()
save_fig(p1, "L3_cohort", "sync_quality.png", 8, 2 + 1.8 * n_distinct(sync$setting))

s <- agree_ts %>% filter(sync_ok) %>% pivot_longer(c(rmse, bias, rmse_centered, r), names_to = "metric") %>%
  group_by(setting, joint, metric) %>% summarise(mean_ci(value), .groups = "drop")
save_tab(s, "L3_synced_agreement_summary.csv")
p2 <- agree_ts %>% filter(sync_ok) %>% pivot_longer(c(rmse, bias, rmse_centered, r), names_to = "metric") %>%
  mutate(joint = factor(joint, JOINTS)) %>%
  ggplot(aes(joint, value, fill = setting)) + geom_boxplot(alpha = .5, outlier.shape = NA) +
  facet_wrap(~metric, scales = "free_y") + scale_fill_manual(values = COL_SETTING, name = NULL) +
  guides(fill = guide_legend(ncol = 2)) +
  labs(title = "Agreement on the time-synchronised series (per run and side)", x = NULL, y = NULL,
       colour = NULL) + theme_mk()
save_fig(p2, "L3_cohort", "synced_agreement.png", 10, 6)
msg("L3 sync done")
