# =============================================================================
# LEVEL 3 -- clinical validity: does Sapiens2 reproduce the SAIN vs NMD
# difference that Vicon measures?
#  (a) group mean curves, both systems (colour = group, linetype = system)
#  (b) per parameter (patient level): Hedges g + Welch p in EACH system, and
#      scatter g_video vs g_vicon (identity = same clinical conclusion)
#  (c) boxplots group x system for kinematic + spatio-temporal parameters
# Outputs: tables/L3_group_effects.csv, figures/L3_cohort/groups_*.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

# SAIN vs NMD exists only in Myokinesis (Bath = healthy lab adults)
gm <- means %>% filter(joint %in% JOINTS, dataset == "myokinesis") %>%
  group_by(patient, group, system, joint, pct) %>% summarise(m = mean(mean), .groups = "drop") %>%
  group_by(group, system, joint, pct) %>% summarise(mm = mean(m), sd = sd(m), n = n(), .groups = "drop") %>%
  mutate(joint = factor(joint, JOINTS))
pa <- ggplot(gm, aes(pct, mm, colour = group, linetype = system)) +
  geom_ribbon(aes(ymin = mm - sd, ymax = mm + sd, fill = group, group = interaction(group, system)),
              alpha = .08, colour = NA) +
  geom_line(linewidth = .9) +
  facet_wrap(~joint, scales = "free_y", labeller = labeller(joint = LAB_JOINT)) +
  scale_colour_manual(values = COL_GROUP, drop = TRUE) + scale_fill_manual(values = COL_GROUP, guide = "none") +
  scale_linetype_manual(values = c(vicon = "solid", video = "22"), labels = LAB_SYS) +
  labs(title = "Group mean kinematics (± SD across patients), Vicon vs Sapiens2",
       x = "Gait cycle (%)", y = "Angle (°)", colour = NULL, linetype = NULL) + theme_mk()
save_fig(pa, "L3_cohort", "groups_mean_curves.png", 12, 5)

kin_long <- run_params %>% filter(joint %in% JOINTS, is.finite(rom), dataset == "myokinesis") %>%
  group_by(patient, group, system, joint) %>%
  summarise(rom = mean(rom), max = mean(max), min = mean(min), .groups = "drop") %>%
  pivot_longer(c(rom, max, min), names_to = "p") %>% mutate(param = paste(joint, p, sep = "_")) %>%
  select(patient, group, system, param, value)
st_long <- st %>% filter(dataset == "myokinesis") %>% select(patient, group, system, all_of(names(ST_PARAMS))) %>%
  pivot_longer(all_of(names(ST_PARAMS)), names_to = "param") %>%
  group_by(patient, group, system, param) %>% summarise(value = mean(value, na.rm = TRUE), .groups = "drop")
allp <- bind_rows(kin_long, st_long) %>% filter(is.finite(value)) %>% mutate(group = droplevels(factor(group)))

eff <- allp %>% group_by(param, system) %>%
  summarise(n_SAIN = sum(group == "SAIN"), n_NMD = sum(group == "NMD"),
            mean_SAIN = mean(value[group == "SAIN"]), mean_NMD = mean(value[group == "NMD"]),
            hedges_g = hedges_g(value[group == "SAIN"], value[group == "NMD"]),
            p_welch = tryCatch(t.test(value[group == "SAIN"], value[group == "NMD"])$p.value, error = function(e) NA),
            p_mannwhitney = tryCatch(wilcox.test(value[group == "SAIN"], value[group == "NMD"], exact = FALSE)$p.value,
                                     error = function(e) NA), .groups = "drop")
save_tab(eff, "L3_group_effects.csv")

gg <- eff %>% select(param, system, hedges_g) %>% pivot_wider(names_from = system, values_from = hedges_g)
pb <- ggplot(gg, aes(vicon, video, label = param)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, colour = "grey50") +
  geom_hline(yintercept = 0, colour = "grey85") + geom_vline(xintercept = 0, colour = "grey85") +
  geom_point(colour = "#E4572E", size = 2) +
  geom_text(size = 2.3, vjust = -.7, colour = "grey30", check_overlap = TRUE) +
  labs(title = "Group effect size (SAIN - NMD, Hedges g): Sapiens2 vs Vicon",
       subtitle = "Points on the diagonal = markerless reaches the same clinical conclusion as Vicon",
       x = "Hedges g measured with Vicon", y = "Hedges g measured with Sapiens2") + theme_mk()
save_fig(pb, "L3_cohort", "groups_effect_size_agreement.png", 8, 7)

pc <- ggplot(allp, aes(group, value, fill = group)) +
  geom_boxplot(alpha = .35, outlier.shape = NA, width = .6) +
  geom_point(aes(colour = group), position = position_jitter(width = .12, seed = 1), size = 1.2) +
  facet_grid(param ~ system, scales = "free_y", labeller = labeller(system = LAB_SYS)) +
  scale_fill_manual(values = COL_GROUP) + scale_colour_manual(values = COL_GROUP) +
  labs(title = "SAIN vs NMD per parameter and system (patient means)", x = NULL, y = NULL) +
  theme_mk(8) + theme(legend.position = "none", strip.text.y = element_text(angle = 0, size = 7))
save_fig(pc, "L3_cohort", "groups_boxplots.png", 8, 22)
msg("L3 groups done")
