# =============================================================================
# LEVEL 3 -- the analyses of the myogait accuracy reference report (Bath,
# 05_report_reference/bath/myogait_accuracy_full.pdf), recomputed here from the
# prepared data for EVERY setting (Myokinesis + each Bath camera view):
#
#  (1) per-cycle clinical parameters (both systems), derived from the 101-point
#      cycle curves + cycle timing:
#        stride_time_s, cadence_spm, stance_pct, swing_pct,
#        <joint>_rom, <joint>_at_ic (angle at initial contact = 0 %),
#        peak_hip_flex (max), peak_hip_ext (min),
#        knee_stance_flex (max knee flexion during 0-40 % = loading response),
#        peak_knee_flex (max during swing, 60-100 %),
#        peak_ankle_df (max), peak_ankle_pf (min),
#        <joint>_peak_vel (deg/s, 95th percentile of |d angle / dt| -- NOT the
#        max, which catches single noise spikes and inflates the LoA ~5x)
#  (2) parameter agreement: bias, LoA, ICC per parameter (run level = mean of
#      the run's cycles, sides matched to the Vicon)
#  (3) MDC95 of the MARKERLESS measurement as a function of the number of
#      cycles averaged: Sw = pooled within-run SD across cycles,
#      MDC95(n) = 1.96 * sqrt(2) * Sw / sqrt(n), n = 1, 3, 5, 10, 20
#      (same definition as parameter_mdc.csv in the Bath report)
# Outputs: tables/L3_cycle_clinical_parameters.csv, L3_parameter_agreement.csv,
#          L3_parameter_mdc.csv, figures/L3_cohort/parameter_agreement_forest.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

cyc_t <- cycles %>% select(cycle_uid, duration_s, stance_pct, swing_pct)
cp <- curves %>% filter(joint %in% JOINTS) %>%
  left_join(cyc_t, by = "cycle_uid") %>%
  group_by(cycle_uid, pair_id, dataset, setting, patient, group, system, side_ref, joint) %>%
  arrange(pct, .by_group = TRUE) %>%
  summarise(
    duration_s = first(duration_s), stance_pct = first(stance_pct), swing_pct = first(swing_pct),
    rom = max(angle) - min(angle), at_ic = angle[1], mx = max(angle), mn = min(angle),
    stance_flex = max(angle[pct <= 40]), swing_flex = max(angle[pct >= 60]),
    peak_vel = quantile(abs(diff(angle)) / (first(duration_s) / 100), .95, na.rm = TRUE),
    .groups = "drop")

wide_cycle <- cp %>%
  transmute(cycle_uid, pair_id, dataset, setting, patient, group, system, side_ref, joint,
            duration_s, stance_pct, swing_pct, rom, at_ic, mx, mn, stance_flex, swing_flex, peak_vel) %>%
  pivot_wider(names_from = joint, values_from = c(rom, at_ic, mx, mn, stance_flex, swing_flex, peak_vel),
              names_glue = "{joint}_{.value}") %>%
  transmute(cycle_uid, pair_id, dataset, setting, patient, group, system, side_ref,
            stride_time_s = duration_s, cadence_spm = 120 / duration_s, stance_pct, swing_pct,
            hip_rom, knee_rom, ankle_rom, hip_at_ic, knee_at_ic, ankle_at_ic,
            peak_hip_flex = hip_mx, peak_hip_ext = hip_mn,
            knee_stance_flex = knee_stance_flex, peak_knee_flex = knee_swing_flex,
            peak_ankle_df = ankle_mx, peak_ankle_pf = ankle_mn,
            hip_peak_vel = hip_peak_vel, knee_peak_vel = knee_peak_vel, ankle_peak_vel = ankle_peak_vel)
save_tab(wide_cycle, "L3_cycle_clinical_parameters.csv")

PARAMS <- c(stride_time_s = "Stride time (s)", cadence_spm = "Cadence (steps/min)",
            stance_pct = "Stance (%)", swing_pct = "Swing (%)",
            hip_rom = "Hip ROM (°)", knee_rom = "Knee ROM (°)", ankle_rom = "Ankle ROM (°)",
            hip_at_ic = "Hip at IC (°)", knee_at_ic = "Knee at IC (°)", ankle_at_ic = "Ankle at IC (°)",
            peak_hip_flex = "Peak hip flexion (°)", peak_hip_ext = "Peak hip extension (°)",
            knee_stance_flex = "Knee stance flexion (°)", peak_knee_flex = "Peak knee flexion (°)",
            peak_ankle_df = "Peak ankle DF (°)", peak_ankle_pf = "Peak ankle PF (°)",
            hip_peak_vel = "Hip peak velocity (°/s)", knee_peak_vel = "Knee peak velocity (°/s)",
            ankle_peak_vel = "Ankle peak velocity (°/s)")
long_cycle <- wide_cycle %>% pivot_longer(all_of(names(PARAMS)), names_to = "param") %>%
  filter(is.finite(value))

# (2) agreement, run level (mean of cycles), side-matched
run_lvl <- long_cycle %>% group_by(pair_id, dataset, setting, patient, group, system, side_ref, param) %>%
  summarise(value = mean(value), .groups = "drop") %>%
  pivot_wider(names_from = system, values_from = value) %>% filter(is.finite(video), is.finite(vicon))
agr <- run_lvl %>% group_by(setting, param) %>%
  group_modify(~ bind_cols(ba_stats(.x$video, .x$vicon),
                           icc_pair(.x$video, .x$vicon) %>% select(-n),
                           tibble(vicon_mean = mean(.x$vicon), video_mean = mean(.x$video),
                                  mae = mean(abs(.x$video - .x$vicon))))) %>%
  ungroup() %>% mutate(label = PARAMS[param], loa_width = loa_high - loa_low, .after = param)
save_tab(agr, "L3_parameter_agreement.csv")

# (3) MDC of the markerless measurement vs number of cycles averaged
mdc <- long_cycle %>% filter(system == "video") %>%
  group_by(setting, param, pair_id, side_ref) %>% filter(n() >= 2) %>%
  summarise(v = var(value), k = n(), .groups = "drop") %>%
  group_by(setting, param) %>%
  summarise(Sw = sqrt(sum(v * (k - 1)) / sum(k - 1)), n_runs = n(), .groups = "drop") %>%
  mutate(label = PARAMS[param], .after = param)
for (n in c(1, 3, 5, 10, 20)) mdc[[paste0("mdc95_n", n)]] <- 1.96 * sqrt(2) * mdc$Sw / sqrt(n)
save_tab(mdc, "L3_parameter_mdc.csv")

# figure: bias ± LoA per parameter, one colour per setting
# frontal / rear views left out of the figure (unusable, they crush the scale; kept in the table)
p <- agr %>% filter(!grepl("frontal|rear", setting)) %>% mutate(label = factor(label, rev(PARAMS))) %>%
  ggplot(aes(bias, label, colour = setting)) +
  geom_vline(xintercept = 0, colour = "grey60") +
  geom_errorbarh(aes(xmin = loa_low, xmax = loa_high), height = 0, alpha = .6,
                 position = position_dodge(width = .7)) +
  geom_point(size = 1.8, position = position_dodge(width = .7)) +
  facet_wrap(~ grepl("vel", param), scales = "free",
             labeller = labeller(.cols = c(`FALSE` = "Timing & angles", `TRUE` = "Peak angular velocities"))) +
  scale_colour_manual(values = COL_SETTING, name = NULL) +
  labs(title = "Clinical parameter agreement, markerless - Vicon (bias and 95% limits of agreement)",
       x = "Sapiens2 - Vicon", y = NULL) + theme_mk(9) + guides(colour = guide_legend(ncol = 2))
save_fig(p, "L3_cohort", "parameter_agreement_forest.png", 13, 9)
msg("L3 clinical parameters + MDC done")
