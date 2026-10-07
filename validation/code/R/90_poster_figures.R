# =============================================================================
# POSTER figures (English), same visual identity as the report:
#  poster_knee_agreement.png : best-agreement run, every knee cycle + mean ± 95% CI
#  poster_group_angles.png   : SAIN vs NMD group mean curves, Sapiens2
# Outputs: figures/L3_cohort/poster_*.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

best <- agree %>% filter(joint == "knee", dataset == "myokinesis") %>% arrange(desc(shape_r), rmse_centered) %>% slice(1)
d <- curves %>% filter(pair_id == best$pair_id, joint == "knee", side_ref == best$side)
mn <- d %>% group_by(system, pct) %>%
  summarise(m = mean(angle), n = n(), s = sd(angle), .groups = "drop") %>%
  mutate(h = qt(.975, pmax(n - 1, 1)) * s / sqrt(n))
p1 <- ggplot() +
  geom_line(data = d, aes(pct, angle, group = cycle_uid, colour = system), alpha = .18, linewidth = .4) +
  geom_ribbon(data = mn, aes(pct, ymin = m - h, ymax = m + h, fill = system), alpha = .25) +
  geom_line(data = mn, aes(pct, m, colour = system), linewidth = 1.4) +
  scale_colour_manual(values = COL_SYS, labels = LAB_SYS, name = NULL) +
  scale_fill_manual(values = COL_SYS, labels = LAB_SYS, name = NULL) +
  annotate("label", x = 100, y = -Inf, hjust = 1, vjust = -.2, size = 3, label.size = .2,
           label = sprintf("Single walk · %s · %s limb\nPearson r = %.3f   RMSE = %.1f°   CMC = %.2f",
                           best$pair_id, best$side, best$shape_r, best$rmse, best$cmc)) +
  labs(title = "Knee kinematics: markerless vs marker-based reference",
       x = "Gait cycle (%)", y = "Knee flexion (°)") + theme_mk(12)
save_fig(p1, "L3_cohort", "poster_knee_agreement.png", 8.6, 6)

gm <- means %>% filter(system == "video", joint %in% c("ankle", "knee"), dataset == "myokinesis") %>%
  group_by(patient, group, joint, pct) %>% summarise(m = mean(mean), .groups = "drop") %>%
  group_by(group, joint, pct) %>% summarise(mm = mean(m), sd = sd(m), .groups = "drop")
p2 <- ggplot(gm, aes(pct, mm, colour = group, fill = group)) +
  geom_ribbon(aes(ymin = mm - sd, ymax = mm + sd), alpha = .15, colour = NA) +
  geom_line(linewidth = 1.2) +
  facet_wrap(~joint, scales = "free_y", labeller = labeller(joint = LAB_JOINT)) +
  scale_colour_manual(values = COL_GROUP, drop = TRUE) + scale_fill_manual(values = COL_GROUP, drop = TRUE) +
  labs(title = "Group mean joint kinematics - SAIN vs NMD (markerless)",
       x = "Gait cycle (%)", y = "Angle (°)", colour = NULL, fill = NULL) + theme_mk(12)
save_fig(p2, "L3_cohort", "poster_group_angles.png", 11, 5)
msg("poster figures done")
