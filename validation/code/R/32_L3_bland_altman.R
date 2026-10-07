# =============================================================================
# LEVEL 3 -- Bland-Altman (video - Vicon) with 95% CI of bias (orange band)
# and of each limit of agreement (grey bands), plus a proportional-bias
# regression (dotted line, p-value in the label), for:
#   * kinematic parameters (ROM / max / min, hip-knee-ankle), run level
#   * spatio-temporal parameters, run level
# Points coloured by patient, shaped by group.
# Outputs: tables/L3_bland_altman.csv, figures/L3_cohort/BA_*.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

kin <- run_params_wide %>% filter(param %in% c("rom", "max", "min")) %>%
  group_by(pair_id, setting, patient, group, joint, param) %>%
  summarise(video = mean(video), vicon = mean(vicon), .groups = "drop") %>%
  mutate(joint = factor(joint, JOINTS), param = factor(param, c("rom", "max", "min")))

ba_plot <- function(df, facets, title, file, w, h) {
  st_ <- df %>% group_by(across(all_of(facets))) %>%
    group_modify(~ ba_stats(.x$video, .x$vicon)) %>% ungroup()
  pts <- df %>% mutate(m = (video + vicon) / 2, d = video - vicon)
  lab <- st_ %>% mutate(txt = sprintf("bias %.2f [%.2f; %.2f]\nLoA %.2f / %.2f\nprop. bias p = %.3f",
                                      bias, bias_ci_low, bias_ci_high, loa_low, loa_high, prop_bias_p))
  p <- ggplot(pts, aes(m, d)) +
    geom_rect(data = st_, inherit.aes = FALSE, fill = "grey85", alpha = .5,
              aes(xmin = -Inf, xmax = Inf, ymin = loa_low_ci_low, ymax = loa_low_ci_high)) +
    geom_rect(data = st_, inherit.aes = FALSE, fill = "grey85", alpha = .5,
              aes(xmin = -Inf, xmax = Inf, ymin = loa_high_ci_low, ymax = loa_high_ci_high)) +
    geom_rect(data = st_, inherit.aes = FALSE, fill = "#E4572E", alpha = .15,
              aes(xmin = -Inf, xmax = Inf, ymin = bias_ci_low, ymax = bias_ci_high)) +
    geom_hline(yintercept = 0, colour = "grey60") +
    geom_hline(data = st_, aes(yintercept = bias), colour = "#E4572E", linewidth = .8) +
    geom_hline(data = st_, aes(yintercept = loa_low), linetype = 2) +
    geom_hline(data = st_, aes(yintercept = loa_high), linetype = 2) +
    geom_smooth(method = "lm", formula = y ~ x, se = FALSE, colour = "grey40",
                linewidth = .5, linetype = 3) +
    geom_point(aes(colour = patient, shape = group), size = 1.8, alpha = .85) +
    geom_text(data = lab, aes(x = Inf, y = Inf, label = txt), hjust = 1.05, vjust = 1.15,
              size = 2.3, colour = "grey20") +
    facet_wrap(facets, scales = "free") +
    scale_colour_viridis_d(option = "turbo", name = "Patient") +
    labs(title = title, x = "Mean of Sapiens2 and Vicon", y = "Sapiens2 - Vicon") + theme_mk(9)
  save_fig(p, "L3_cohort", file, w, h)
  st_
}
out <- list()
for (st_ in unique(as.character(runs$setting))) {
  t1 <- ba_plot(filter(kin, setting == st_), c("joint", "param"),
                paste0("Bland-Altman - kinematic parameters (°), run level - ", st_),
                paste0("BA_kinematics__", slug(st_), ".png"), 13, 11)
  t2 <- ba_plot(st_wide %>% filter(setting == st_) %>% mutate(param = factor(ST_PARAMS[param], ST_PARAMS)),
                "param", paste0("Bland-Altman - spatio-temporal parameters, run level (myogait 0.8.9) - ", st_),
                paste0("BA_spatiotemporal__", slug(st_), ".png"), 13, 10)
  out[[st_]] <- bind_rows(t1 %>% mutate(family = "kinematics", across(c(joint, param), as.character)),
                          t2 %>% mutate(family = "spatiotemporal", param = as.character(param))) %>%
    mutate(setting = st_, .before = 1)
}
save_tab(bind_rows(out), "L3_bland_altman.csv")
msg("L3 Bland-Altman done")
