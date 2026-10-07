# =============================================================================
# LEVEL 3 -- which recording setting works best?
#  (a) Bath: the SAME walk filmed by 4 synchronised cameras (lateral L cam01,
#      lateral R cam05, frontal cam03, rear cam07) -> accuracy per view, paired by
#      trial (differences are purely due to the viewpoint).
#  (b) all settings side by side, incl. Myokinesis (hand-held iPhone following the
#      subject, clinic, patients) vs Bath lab lateral views (fixed camera, healthy).
# Unit: run x joint (sides averaged); subject-level summary in the table.
# Outputs: tables/L3_setting_comparison.csv, tables/L3_bath_view_paired.csv,
#          figures/L3_views/*.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

mets <- c(rmse = "RMSE (°)", rmse_centered = "Centred RMSE (°)", bias = "Bias (°)",
          shape_r = "Pearson r", rom_err = "ROM error (°)")
run_lvl <- agree %>% group_by(pair_id, dataset, setting, patient, joint) %>%
  summarise(across(all_of(names(mets)), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
  left_join(runs %>% select(pair_id, vicon_trial), by = "pair_id")

# (b) all settings, subject level
tab <- run_lvl %>% group_by(setting, patient, joint) %>%
  summarise(across(all_of(names(mets)), mean), n_runs = n(), .groups = "drop") %>%
  pivot_longer(all_of(names(mets)), names_to = "metric") %>%
  group_by(setting, joint, metric) %>% summarise(mean_ci(value), .groups = "drop")
save_tab(tab, "L3_setting_comparison.csv")

p1 <- run_lvl %>% pivot_longer(all_of(names(mets)), names_to = "metric") %>%
  mutate(metric = factor(mets[metric], mets), joint = factor(joint, JOINTS)) %>%
  ggplot(aes(joint, value, fill = setting)) +
  geom_boxplot(alpha = .55, outlier.size = .5, position = position_dodge(width = .8)) +
  facet_wrap(~metric, scales = "free_y", nrow = 2) +
  scale_fill_manual(values = COL_SETTING, name = NULL) + guides(fill = guide_legend(ncol = 2)) +
  labs(title = "Accuracy vs Vicon by recording setting (one value per run and joint)",
       subtitle = "Myokinesis = hand-held iPhone following patients in clinic; Bath = fixed lab cameras, healthy adults",
       x = NULL, y = NULL) + theme_mk(10)
save_fig(p1, "L3_views", "setting_comparison_boxplots.png", 13, 8)

# (a) Bath paired by trial: each view vs lateral-left (cam01) on the same walk
bath <- run_lvl %>% filter(dataset == "bath") %>%
  mutate(view = sub("Bath - ", "", as.character(setting)))
paired <- bath %>% select(vicon_trial, patient, joint, view, rmse, rmse_centered, shape_r) %>%
  pivot_longer(c(rmse, rmse_centered, shape_r), names_to = "metric") %>%
  pivot_wider(names_from = view, values_from = value)
ref <- "lateral left (cam01)"
others <- setdiff(names(paired), c("vicon_trial", "patient", "joint", "metric", ref))
pt <- map_dfr(others, function(v) paired %>% filter(is.finite(.data[[ref]]), is.finite(.data[[v]])) %>%
  group_by(joint, metric) %>%
  summarise(view = v, n_trials = n(), mean_ref = mean(.data[[ref]]), mean_view = mean(.data[[v]]),
            diff = mean(.data[[v]] - .data[[ref]]),
            p_paired_wilcoxon = if (n() >= 5) wilcox.test(.data[[v]], .data[[ref]], paired = TRUE,
                                                        exact = FALSE)$p.value else NA_real_,
            .groups = "drop"))
save_tab(pt, "L3_bath_view_paired.csv")

p2 <- bath %>% mutate(joint = factor(joint, JOINTS)) %>%
  ggplot(aes(view, rmse_centered, fill = view)) +
  geom_boxplot(alpha = .5, outlier.shape = NA) +
  geom_line(aes(group = vicon_trial), colour = "grey70", alpha = .4, linewidth = .3) +
  geom_point(size = .8, alpha = .6) +
  facet_wrap(~joint, labeller = labeller(joint = LAB_JOINT)) +
  scale_fill_manual(values = setNames(COL_SETTING[-1], sub("Bath - ", "", names(COL_SETTING)[-1])), guide = "none") +
  labs(title = "Bath - same walks seen from 4 cameras: waveform error (centred RMSE) per view",
       subtitle = "Grey lines join the same trial across views; paired tests in tables/L3_bath_view_paired.csv",
       x = NULL, y = "Centred RMSE vs Vicon (°)") + theme_mk(10) +
  theme(axis.text.x = element_text(angle = 20, hjust = 1))
save_fig(p2, "L3_views", "bath_views_paired.png", 12, 5.5)
msg("L3 view & setting comparison done")
