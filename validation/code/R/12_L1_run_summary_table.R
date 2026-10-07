# =============================================================================
# LEVEL 1 -- one row per paired run: pairing, sync, curve agreement per joint,
# spatio-temporal video vs Vicon. Output: 04_outputs/tables/L1_runs_summary.csv
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

ag <- agree %>% group_by(pair_id, joint) %>%
  summarise(rmse = mean(rmse), bias = mean(bias), rmse_c = mean(rmse_centered),
            r = mean(shape_r), cmc = mean(cmc), .groups = "drop") %>%
  pivot_wider(names_from = joint, values_from = c(rmse, bias, rmse_c, r, cmc), names_glue = "{joint}_{.value}")
stw <- st_wide %>% filter(param %in% c("cadence_spm", "stride_time_s", "stride_length_m", "speed_mps")) %>%
  transmute(pair_id, param, video, vicon) %>%
  pivot_wider(names_from = param, values_from = c(video, vicon), names_glue = "{param}_{.value}")
out <- runs %>% select(pair_id, dataset, view, setting, patient, group, video_run, vicon_trial, pairing_residual_s,
                       video_n_cycles, vicon_n_cycles) %>%
  left_join(sync %>% select(pair_id, sync_r = r, sync_ok, sync_lag_s = lag_s), by = "pair_id") %>%
  left_join(side_map %>% select(pair_id, side_swap, side_method), by = "pair_id") %>%
  left_join(ag, by = "pair_id") %>% left_join(stw, by = "pair_id")
save_tab(out, "L1_runs_summary.csv")
msg("L1 summary table: ", nrow(out), " runs")
