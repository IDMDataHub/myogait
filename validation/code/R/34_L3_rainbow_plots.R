# =============================================================================
# LEVEL 3 -- "rainbow" plots: the gait-cycle phase as colour (0% red -> 100% violet)
#  (a) angle-angle: x = Vicon, y = Sapiens2, one path per patient + cohort path,
#      identity line = perfect agreement. Loops/offsets show WHERE they differ.
#  (b) pointwise Bland-Altman: x = mean, y = difference, every run x % cycle,
#      coloured by phase -> bias that depends on phase/magnitude shows as colour bands.
#  (c) frame-level (no cycle normalisation of the video!): the synchronised 100 Hz
#      series, phase taken from the Vicon heel strikes of the same side.
# Outputs: figures/L3_cohort/rainbow_*.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

rainbow_scale <- scale_colour_gradientn(colours = rev(rainbow(7, end = .8)), limits = c(0, 100),
                                        name = "Gait cycle (%)")
w <- means %>% filter(joint %in% JOINTS) %>%
  select(pair_id, setting, patient, group, system, side_ref, joint, pct, mean) %>%
  pivot_wider(names_from = system, values_from = mean) %>%
  filter(is.finite(video), is.finite(vicon)) %>% mutate(joint = factor(joint, JOINTS))
pat <- w %>% group_by(setting, patient, group, joint, pct) %>%
  summarise(video = mean(video), vicon = mean(vicon), .groups = "drop")
coh <- pat %>% group_by(setting, joint, pct) %>% summarise(video = mean(video), vicon = mean(vicon), .groups = "drop")

# (a) angle-angle
pa <- ggplot(pat, aes(vicon, video, colour = pct)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey40", linetype = 2) +
  geom_path(aes(group = patient), alpha = .35, linewidth = .5) +
  geom_path(data = coh, linewidth = 1.8) +
  facet_grid(setting ~ joint, scales = "free", labeller = labeller(joint = LAB_JOINT, setting = label_wrap_gen(18))) +
  rainbow_scale +
  labs(title = "Angle-angle plot along the gait cycle (colour = phase)",
       subtitle = "Thin: each patient (mean of runs) · bold: cohort · dashed: identity (perfect agreement)",
       x = "Vicon angle (°)", y = "Sapiens2 angle (°)") + theme_mk() +
  theme(legend.position = "right", aspect.ratio = 1)
save_fig(pa, "L3_cohort", "rainbow_angle_angle.png", 13, 2 + 3.6 * n_distinct(coh$setting))

# (b) pointwise Bland-Altman coloured by phase
pb <- ggplot(w %>% mutate(m = (video + vicon) / 2, d = video - vicon), aes(m, d, colour = pct)) +
  geom_hline(yintercept = 0, colour = "grey40") +
  geom_point(size = .5, alpha = .35) +
  geom_path(data = coh %>% mutate(m = (video + vicon) / 2, d = video - vicon), linewidth = 1.6) +
  facet_grid(setting ~ joint, scales = "free", labeller = labeller(joint = LAB_JOINT, setting = label_wrap_gen(18))) +
  rainbow_scale +
  labs(title = "Pointwise Bland-Altman along the gait cycle (every run x % cycle)",
       subtitle = "Bold path = cohort mean trajectory; colour = gait-cycle phase",
       x = "Mean of Sapiens2 and Vicon (°)", y = "Sapiens2 - Vicon (°)") + theme_mk() +
  theme(legend.position = "right", aspect.ratio = 1)
save_fig(pb, "L3_cohort", "rainbow_bland_altman_pointwise.png", 13, 2 + 3.6 * n_distinct(coh$setting))

# (c) frame-level synchronised series, phase from Vicon heel strikes
phase_from_hs <- function(t, hs) {
  ph <- rep(NA_real_, length(t)); hs <- sort(hs)
  if (length(hs) < 2) return(ph)
  for (i in seq_len(length(hs) - 1)) {
    k <- t >= hs[i] & t < hs[i + 1]; ph[k] <- 100 * (t[k] - hs[i]) / (hs[i + 1] - hs[i])
  }
  ph
}
fl <- map_dfr(sync$pair_id[sync$sync_ok], function(pid) {   # reliable syncs only
  f <- file.path(DATA, "synced", paste0(pid, "_synced.csv"))
  if (!file.exists(f)) return(NULL)
  s <- read_csv(f)
  vt0 <- min(read_csv(file.path(DATA, "timeseries", paste0(pid, "_vicon.csv")))$time_s, na.rm = TRUE)
  map_dfr(c("L", "R"), function(sd_) {
    hs <- events %>% filter(pair_id == pid, system == "vicon", event == "HS", side == sd_) %>%
      pull(time_s) - vt0
    ph <- phase_from_hs(s$t_vicon_s, hs)
    map_dfr(JOINTS, ~ tibble(pair_id = pid, side = sd_, joint = .x, phase = ph,
                             vicon = s[[paste0("vicon_", .x, "_", sd_)]],
                             video = s[[paste0("video_", .x, "_", sd_)]]))
  })
}) %>% filter(is.finite(phase), is.finite(vicon), is.finite(video)) %>%
  left_join(runs %>% select(pair_id, setting), by = "pair_id") %>%
  mutate(joint = factor(joint, JOINTS))
pc <- ggplot(fl, aes(vicon, video, colour = phase)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey40", linetype = 2) +
  geom_point(size = .25, alpha = .25) +
  facet_grid(setting ~ joint, scales = "free", labeller = labeller(joint = LAB_JOINT, setting = label_wrap_gen(18))) +
  rainbow_scale +
  labs(title = "Frame-level agreement on the time-synchronised series (100 Hz)",
       subtitle = sprintf("%s samples from %d runs with a reliable sync (r >= 0.9); phase from Vicon heel strikes",
                          format(nrow(fl), big.mark = " "), n_distinct(fl$pair_id)),
       x = "Vicon angle (°)", y = "Sapiens2 angle (°)") + theme_mk() +
  theme(legend.position = "right", aspect.ratio = 1)
save_fig(pc, "L3_cohort", "rainbow_frame_level_synced.png", 13, 2 + 3.6 * n_distinct(fl$setting))
msg("L3 rainbow plots done")
