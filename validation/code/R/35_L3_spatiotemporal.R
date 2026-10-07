# =============================================================================
# LEVEL 3 -- spatio-temporal validation (cadence, stride/step time, stance,
# double support, step & stride length, speed), Sapiens2 vs Vicon.
# NOTE: lengths/speed require myogait >= 0.8.9 (scale calibrated on the whole
# recording; 0.8.8 over-estimated them ~1.5x on camera-following videos).
# Figures: scatter with identity + regression; summary table with ICC, bias,
# mean absolute error and mean absolute % error.
# Outputs: tables/L3_spatiotemporal_validation.csv, figures/L3_cohort/spatiotemporal_scatter.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

tab <- st_wide %>% group_by(setting, param) %>%
  group_modify(~ bind_cols(icc_pair(.x$video, .x$vicon) %>% select(-n),
                           ba_stats(.x$video, .x$vicon),
                           tibble(mae = mean(abs(.x$video - .x$vicon)),
                                  mape_pct = 100 * mean(abs(.x$video - .x$vicon) / abs(.x$vicon)),
                                  vicon_mean = mean(.x$vicon), video_mean = mean(.x$video)))) %>%
  ungroup() %>% mutate(label = ST_PARAMS[param], .after = param)
save_tab(tab, "L3_spatiotemporal_validation.csv")

p <- st_wide %>% mutate(label = factor(ST_PARAMS[param], ST_PARAMS)) %>%
  ggplot(aes(vicon, video)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
  geom_smooth(aes(colour = setting), method = "lm", formula = y ~ x, se = FALSE, linewidth = .5) +
  geom_point(aes(colour = setting), size = 1.4, alpha = .7) +
  facet_wrap(~label, scales = "free") +
  scale_colour_manual(values = COL_SETTING, name = NULL) + guides(colour = guide_legend(ncol = 2)) +
  labs(title = "Spatio-temporal parameters - Sapiens2 (y) vs Vicon (x), one point per paired run",
       subtitle = "ICC, bias, MAPE per setting: tables/L3_spatiotemporal_validation.csv",
       x = "Vicon", y = "Sapiens2") + theme_mk(9)
save_fig(p, "L3_cohort", "spatiotemporal_scatter.png", 13, 10)
msg("L3 spatio-temporal done")
