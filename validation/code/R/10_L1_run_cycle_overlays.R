# =============================================================================
# LEVEL 1 -- one figure per paired run (64 figures)
# ON DEMAND only (Rscript 03_R/run_all.R L1). Lateral / Myokinesis settings only:
# frontal and rear Bath views do not give usable sagittal angles (see contexte).
# Every gait cycle (thin) + mean ± SD (bold/ribbon) for Vicon and Sapiens2,
# hip/knee/ankle x anatomical side (Vicon side reference), with the run's
# curve-agreement numbers. Output: 04_outputs/figures/L1_run_cycles/<pair_id>.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

keep_pairs <- runs %>% filter(!view %in% c("frontal", "rear")) %>% pull(pair_id)
dat <- curves %>% filter(joint %in% JOINTS, pair_id %in% keep_pairs) %>% mutate(joint = factor(joint, JOINTS))
mn  <- dat %>% group_by(pair_id, system, side_ref, joint, pct) %>%
  summarise(m = mean(angle), s = sd(angle), .groups = "drop")
lab <- agree %>% transmute(pair_id, side_ref = side, joint = factor(joint, JOINTS),
                           txt = sprintf("RMSE %.1f°  bias %+.1f°  r %.2f", rmse, bias, shape_r))
side_lab <- c(L = "Left (Vicon side)", R = "Right (Vicon side)")

for (pid in sort(unique(dat$pair_id))) {
  r <- filter(runs, pair_id == pid)
  p <- ggplot() +
    geom_line(data = filter(dat, pair_id == pid),
              aes(pct, angle, group = cycle_uid, colour = system), alpha = .18, linewidth = .35) +
    geom_ribbon(data = filter(mn, pair_id == pid),
                aes(pct, ymin = m - s, ymax = m + s, fill = system), alpha = .15) +
    geom_line(data = filter(mn, pair_id == pid), aes(pct, m, colour = system), linewidth = 1.1) +
    geom_text(data = filter(lab, pair_id == pid), aes(x = 50, y = Inf, label = txt),
              vjust = 1.4, size = 2.8, colour = "grey30") +
    facet_grid(joint ~ side_ref, scales = "free_y",
               labeller = labeller(joint = LAB_JOINT, side_ref = side_lab)) +
    scale_colour_manual(values = COL_SYS, labels = LAB_SYS, name = NULL) +
    scale_fill_manual(values = COL_SYS, labels = LAB_SYS, name = NULL) +
    labs(title = sprintf("%s  —  %s vs Vicon %s", pid, r$video_run, r$vicon_trial),
         subtitle = sprintf("%s · subject %s (%s) · %d video cycles / %d Vicon cycles",
                            r$setting, r$patient, r$group, r$video_n_cycles, r$vicon_n_cycles),
         x = "Gait cycle (%)", y = "Angle (°)") + theme_mk()
  save_fig(p, "L1_run_cycles", paste0(pid, ".png"), 9, 9)
}
msg("L1 cycle overlays: ", n_distinct(dat$pair_id), " figures")
