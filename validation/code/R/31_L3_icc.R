# =============================================================================
# LEVEL 3 -- ICC(2,1) absolute agreement & ICC(3,1) consistency (+ Pearson r)
# for kinematic parameters (ROM / max / min per joint) and spatio-temporal
# parameters, at run level (sides averaged) and patient level.
# Outputs: tables/L3_icc.csv, figures/L3_cohort/icc_dotplot.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

kin_run <- run_params_wide %>% filter(param %in% c("rom", "max", "min")) %>%
  group_by(pair_id, setting, patient, group, joint, param) %>%
  summarise(video = mean(video), vicon = mean(vicon), .groups = "drop") %>%
  mutate(variable = paste(joint, param, sep = "_"))
kin_pat <- patient_params_wide %>% filter(param %in% c("rom", "max", "min")) %>%
  mutate(variable = paste(joint, param, sep = "_"))
st_run <- st_wide %>% mutate(variable = param)
st_pat <- st_wide %>% group_by(setting, patient, group, param) %>%
  summarise(video = mean(video), vicon = mean(vicon), .groups = "drop") %>% mutate(variable = param)

icc_tab <- function(df, level, family) df %>% group_by(setting, variable) %>%
  group_modify(~ icc_pair(.x$video, .x$vicon)) %>% ungroup() %>%
  mutate(level = level, family = family, .before = 1)
tab <- bind_rows(icc_tab(kin_run, "run", "kinematics"), icc_tab(kin_pat, "patient", "kinematics"),
                 icc_tab(st_run, "run", "spatiotemporal"), icc_tab(st_pat, "patient", "spatiotemporal"))
save_tab(tab, "L3_icc.csv")

for (st_ in unique(as.character(tab$setting))) {
p <- tab %>% filter(is.finite(icc_2_1), setting == st_) %>%
  pivot_longer(c(icc_2_1, icc_3_1), names_to = "form", values_to = "icc") %>%
  mutate(lo = ifelse(form == "icc_2_1", icc_2_1_low, icc_3_1_low),
         hi = ifelse(form == "icc_2_1", icc_2_1_high, icc_3_1_high),
         form = recode(form, icc_2_1 = "ICC(2,1) agreement", icc_3_1 = "ICC(3,1) consistency")) %>%
  ggplot(aes(icc, reorder(variable, icc), colour = form)) +
  annotate("rect", xmin = .75, xmax = .9, ymin = -Inf, ymax = Inf, alpha = .06) +
  annotate("rect", xmin = .9, xmax = 1, ymin = -Inf, ymax = Inf, alpha = .12) +
  geom_pointrange(aes(xmin = lo, xmax = hi), position = position_dodge(width = .6), size = .25) +
  facet_grid(family ~ level, scales = "free_y", space = "free_y") +
  scale_colour_manual(values = c("#22314E", "#E4572E"), name = NULL) +
  coord_cartesian(xlim = c(-.2, 1)) +
  labs(title = paste0("ICC Sapiens2 vs Vicon - ", st_),
       subtitle = "Shaded: good 0.75-0.9, excellent > 0.9 (Koo & Li 2016)",
       x = "ICC (95% CI)", y = NULL) + theme_mk(9)
save_fig(p, "L3_cohort", paste0("icc_dotplot__", slug(st_), ".png"), 11, 10)
}
msg("L3 ICC done")
