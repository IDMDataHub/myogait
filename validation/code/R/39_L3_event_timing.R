# =============================================================================
# LEVEL 3 -- temporal validity of gait events (heel strike HS, toe off TO)
# Video event times are expressed on the Vicon clock with the synchronisation
# lag (t_vicon = t_video - lag_s) and their side mapped to the Vicon side; each
# Vicon event is matched to the nearest video event of the same type/side
# within 250 ms. Error = video - Vicon (ms).
# NOTE: the lag itself comes from the kinematics, so a constant timing offset
# common to all events is absorbed by the sync; this measures RELATIVE timing
# (event placement within the stride), not absolute clock accuracy.
# Outputs: tables/L3_event_timing_matches.csv, tables/L3_event_timing_summary.csv,
#          figures/L3_cohort/event_timing.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

ev <- events %>% left_join(sync %>% select(pair_id, lag_s, sync_ok), by = "pair_id") %>%
  left_join(side_map %>% select(pair_id, side_swap), by = "pair_id") %>%
  filter(sync_ok, is.finite(lag_s)) %>%   # reliable syncs only
  mutate(t = ifelse(system == "video", time_s - lag_s, time_s),
         side_ref = ifelse(system == "video" & side_swap, ifelse(side == "L", "R", "L"), side))
vic <- ev %>% filter(system == "vicon"); vid <- ev %>% filter(system == "video")

m <- vic %>% select(pair_id, event, side_ref, t_vicon = t) %>%
  inner_join(vid %>% select(pair_id, event, side_ref, t_video = t),
             by = c("pair_id", "event", "side_ref"), relationship = "many-to-many") %>%
  mutate(err_ms = 1000 * (t_video - t_vicon)) %>%
  group_by(pair_id, event, side_ref, t_vicon) %>% slice_min(abs(err_ms), n = 1, with_ties = FALSE) %>%
  ungroup() %>% filter(abs(err_ms) <= 250) %>%
  left_join(runs %>% select(pair_id, setting, patient, group), by = "pair_id")
save_tab(m, "L3_event_timing_matches.csv")
sm <- m %>% group_by(setting, event) %>%
  summarise(n = n(), bias_ms = mean(err_ms), sd_ms = sd(err_ms), mae_ms = mean(abs(err_ms)),
            loa_low_ms = bias_ms - 1.96 * sd_ms, loa_high_ms = bias_ms + 1.96 * sd_ms, .groups = "drop")
save_tab(sm, "L3_event_timing_summary.csv")

p <- ggplot(m, aes(err_ms, fill = event)) + geom_vline(xintercept = 0, colour = "grey40") +
  geom_histogram(binwidth = 10, colour = "white", alpha = .85) +
  facet_grid(setting ~ event, scales = "free_y",
             labeller = labeller(event = c(HS = "Heel strike", TO = "Toe off"), setting = label_wrap_gen(22))) +
  scale_fill_manual(values = c(HS = "#22314E", TO = "#E4572E"), guide = "none") +
  labs(title = "Gait-event timing error, Sapiens2 - Vicon (after synchronisation)",
       subtitle = "Bias, SD, MAE and limits per setting: tables/L3_event_timing_summary.csv",
       x = "Timing error (ms)", y = "Events") + theme_mk()
save_fig(p, "L3_cohort", "event_timing.png", 10, 2 + 2.2 * n_distinct(m$setting))
msg("L3 event timing done")
