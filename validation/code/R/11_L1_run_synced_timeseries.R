# =============================================================================
# LEVEL 1 -- frame-level, time-synchronised series (one figure per run)
# ON DEMAND only (Rscript 03_R/run_all.R L1). Only pairs with a reliable sync
# (sync_ok, r >= 0.9) -- the others would show a meaningless alignment.
# The Vicon trial is placed inside the longer video by cross-correlation
# (01_data_prepared/synced/<pair_id>_synced.csv, 100 Hz, Vicon time base).
# Vertical lines = Vicon heel strikes. Bottom row = video - Vicon difference.
# Output: 04_outputs/figures/L1_run_synced/<pair_id>.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

for (pid in sort(sync$pair_id[sync$sync_ok])) {
  f <- file.path(DATA, "synced", paste0(pid, "_synced.csv"))
  if (!file.exists(f)) next
  s <- read_csv(f)
  long <- s %>% pivot_longer(-c(t_vicon_s, t_video_s), names_to = c("system", "joint", "side"),
                              names_sep = "_", values_to = "angle") %>%
    mutate(joint = factor(joint, JOINTS))
  diff <- long %>% pivot_wider(names_from = system, values_from = angle) %>%
    mutate(d = video - vicon)
  vt0 <- min(read_csv(file.path(DATA, "timeseries", paste0(pid, "_vicon.csv")))$time_s, na.rm = TRUE)
  hs <- events %>% filter(pair_id == pid, system == "vicon", event == "HS") %>%
    transmute(side, t = time_s - vt0)
  sy <- filter(sync, pair_id == pid)
  p1 <- ggplot(long, aes(t_vicon_s, angle, colour = system)) +
    geom_vline(data = hs, aes(xintercept = t), colour = "grey75", linewidth = .3) +
    geom_line(linewidth = .7) + facet_grid(joint ~ side, scales = "free_y",
                                           labeller = labeller(joint = LAB_JOINT)) +
    scale_colour_manual(values = COL_SYS, labels = LAB_SYS, name = NULL) +
    labs(x = NULL, y = "Angle (°)") + theme_mk()
  p2 <- ggplot(diff, aes(t_vicon_s, d, colour = joint)) + geom_hline(yintercept = 0) +
    geom_line(linewidth = .6) + facet_wrap(~side, nrow = 1) +
    scale_colour_viridis_d(end = .85, name = NULL) +
    labs(x = "Time from Vicon start (s)", y = "Video − Vicon (°)") + theme_mk()
  p <- (p1 / p2) + plot_layout(heights = c(3, 1.2)) +
    plot_annotation(title = paste0(pid, " — frame-synchronised kinematics (", sy$setting, ")"),
                    subtitle = sprintf("Cross-correlation r = %.3f%s · Vicon starts at video t = %.2f s · L/R swapped: %s",
                                       sy$r, ifelse(sy$sync_ok, "", "  [SYNC UNRELIABLE - excluded from frame-level stats]"), sy$lag_s, sy$side_swap),
                    theme = theme_mk())
  save_fig(p, "L1_run_synced", paste0(pid, ".png"), 10, 10)
}
msg("L1 synced series: ", nrow(sync), " figures")
