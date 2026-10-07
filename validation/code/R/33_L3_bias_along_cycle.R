# =============================================================================
# LEVEL 3 -- WHERE in the gait cycle does Sapiens2 differ from Vicon?
#  (a) pointwise bias curve (video - Vicon) per joint: thin line per patient,
#      cohort mean ± 95% CI (across patients) and ± SD; % cycle where the CI
#      excludes 0 marked at the bottom (pointwise, NOT corrected for multiple
#      comparisons -- use SPM (spm1d) for a formal test).
#  (b) heatmap patient x % cycle of the mean bias (diverging palette).
#  (c) pointwise RMSE along the cycle (cohort).
# Outputs: tables/L3_bias_along_cycle.csv, figures/L3_cohort/bias_along_cycle*.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

d_run <- means %>% filter(joint %in% JOINTS) %>%
  select(pair_id, setting, patient, group, system, side_ref, joint, pct, mean) %>%
  pivot_wider(names_from = system, values_from = mean) %>%
  filter(is.finite(video), is.finite(vicon)) %>% mutate(d = video - vicon)
d_pat <- d_run %>% group_by(setting, patient, group, joint, pct) %>%
  summarise(d = mean(d), .groups = "drop") %>% mutate(joint = factor(joint, JOINTS))
coh <- d_pat %>% group_by(setting, joint, pct) %>%
  summarise(n = n(), mean = mean(d), sd = sd(d),
            ci_low = mean - qt(.975, n - 1) * sd / sqrt(n),
            ci_high = mean + qt(.975, n - 1) * sd / sqrt(n),
            rmse = sqrt(mean(d^2)), .groups = "drop") %>%
  mutate(ci_excludes_0 = ci_low > 0 | ci_high < 0)
save_tab(coh, "L3_bias_along_cycle.csv")

p1 <- ggplot() +
  geom_hline(yintercept = 0, colour = "grey50") +
  geom_line(data = d_pat, aes(pct, d, group = patient, colour = group), alpha = .35, linewidth = .4) +
  geom_ribbon(data = coh, aes(pct, ymin = mean - sd, ymax = mean + sd), fill = "grey60", alpha = .15) +
  geom_ribbon(data = coh, aes(pct, ymin = ci_low, ymax = ci_high), fill = "#22314E", alpha = .3) +
  geom_line(data = coh, aes(pct, mean), colour = "#22314E", linewidth = 1.1) +
  geom_rug(data = filter(coh, ci_excludes_0), aes(x = pct), sides = "b", colour = "#E4572E") +
  facet_grid(setting ~ joint, scales = "free_y", labeller = labeller(joint = LAB_JOINT, setting = label_wrap_gen(18))) +
  scale_colour_manual(values = COL_GROUP, name = NULL) +
  labs(title = "Bias along the gait cycle (Sapiens2 - Vicon)",
       subtitle = "Mean ± 95% CI (dark) and ± SD (light) across patients; red rug = CI excludes 0 (pointwise)",
       x = "Gait cycle (%)", y = "Sapiens2 - Vicon (°)") + theme_mk()
save_fig(p1, "L3_cohort", "bias_along_cycle.png", 12, 3 + 2.6 * n_distinct(coh$setting))

lim <- max(abs(d_pat$d))
for (st_ in unique(as.character(d_pat$setting))) {
p2 <- ggplot(filter(d_pat, setting == st_), aes(pct, paste(patient, group), fill = d)) + geom_tile() +
  facet_wrap(~joint, labeller = labeller(joint = LAB_JOINT)) +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", limits = c(-lim, lim),
                       name = "Bias (°)") +
  labs(title = paste0("Bias heatmap - subject x gait-cycle phase - ", st_), x = "Gait cycle (%)", y = NULL) +
  theme_mk(9)
save_fig(p2, "L3_cohort", paste0("bias_heatmap__", slug(st_), ".png"), 12, 6)
}

p3 <- ggplot(coh, aes(pct, rmse)) + geom_area(fill = "#E4572E", alpha = .25) +
  geom_line(colour = "#E4572E", linewidth = 1) +
  facet_grid(setting ~ joint, labeller = labeller(joint = LAB_JOINT, setting = label_wrap_gen(18))) +
  labs(title = "Pointwise RMSE along the gait cycle (across subjects)",
       x = "Gait cycle (%)", y = "RMSE (°)") + theme_mk()
save_fig(p3, "L3_cohort", "rmse_along_cycle.png", 12, 2.5 + 2.4 * n_distinct(coh$setting))
msg("L3 bias along cycle done")
