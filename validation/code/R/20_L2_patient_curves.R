# =============================================================================
# LEVEL 2 -- per patient: mean of the run-mean curves, ± SD across runs,
# Vicon vs Sapiens2 (one figure per patient) + a cohort grid of all patients
# for each joint. Output: 04_outputs/figures/L2_patient/
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

# frontal / rear Bath views excluded: no usable sagittal angles (see contexte)
pm <- means %>% filter(joint %in% JOINTS, !view %in% c("frontal", "rear")) %>%
  group_by(setting, patient, group, system, side_ref, joint, pct) %>%
  summarise(m = mean(mean), s = sd(mean), n_runs = n(), .groups = "drop") %>%
  mutate(joint = factor(joint, JOINTS), s = coalesce(s, 0))

for (key in unique(paste(pm$setting, pm$patient, sep = "||"))) {
  kp <- strsplit(key, "||", fixed = TRUE)[[1]]; st_ <- kp[1]; p_ <- kp[2]
  d <- filter(pm, setting == st_, patient == p_)
  p <- ggplot(d, aes(pct, m, colour = system, fill = system)) +
    geom_ribbon(aes(ymin = m - s, ymax = m + s), alpha = .15, colour = NA) +
    geom_line(linewidth = 1) +
    facet_grid(joint ~ side_ref, scales = "free_y", labeller = labeller(joint = LAB_JOINT)) +
    scale_colour_manual(values = COL_SYS, labels = LAB_SYS, name = NULL) +
    scale_fill_manual(values = COL_SYS, labels = LAB_SYS, name = NULL) +
    labs(title = sprintf("%s — subject %s (%s), mean of %d paired runs (± SD across runs)",
                         st_, p_, d$group[1], max(d$n_runs)),
         x = "Gait cycle (%)", y = "Angle (°)") + theme_mk()
  save_fig(p, "L2_patient", paste0(slug(st_), "__", p_, ".png"), 9, 9)
}

# all patients on one page, per joint (both sides averaged)
pj <- pm %>% group_by(setting, patient, group, system, joint, pct) %>% summarise(m = mean(m), .groups = "drop")
for (st_ in unique(as.character(pj$setting))) for (j in JOINTS) {
  p <- ggplot(filter(pj, joint == j, setting == st_), aes(pct, m, colour = system)) +
    geom_line(linewidth = .8) + facet_wrap(~ paste(patient, group), ncol = 5) +
    scale_colour_manual(values = COL_SYS, labels = LAB_SYS, name = NULL) +
    labs(title = paste0(LAB_JOINT[j], " — every subject, Vicon vs Sapiens2 — ", st_),
         x = "Gait cycle (%)", y = "Angle (°)") + theme_mk(9)
  save_fig(p, "L2_patient", paste0("grid__", slug(st_), "__", j, ".png"), 12, 8)
}
msg("L2 patient curves done")
