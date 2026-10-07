# =============================================================================
# LEVEL 2 -- patient-level tables (patient = statistical unit)
#  L2_patient_curve_agreement.csv : RMSE / bias / centred RMSE / r / CMC per joint
#  L2_patient_kinematic_params.csv: ROM, max, min per joint, video vs Vicon
#  L2_patient_spatiotemporal.csv  : spatio-temporal parameters, video vs Vicon
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

a <- agree %>% group_by(setting, patient, group, joint) %>%
  summarise(n_runs = n_distinct(pair_id), rmse = mean(rmse), mae = mean(mae), bias = mean(bias),
            rmse_centered = mean(rmse_centered), r = mean(shape_r), cmc = mean(cmc, na.rm = TRUE),
            rom_err = mean(rom_err), peak_t_err_abs = mean(abs(peak_t_err)), .groups = "drop")
save_tab(a, "L2_patient_curve_agreement.csv")

k <- patient_params_wide %>% mutate(diff = video - vicon)
save_tab(k, "L2_patient_kinematic_params.csv")

s <- st_wide %>% group_by(setting, patient, group, param) %>%
  summarise(n_runs = n(), video = mean(video), vicon = mean(vicon), .groups = "drop") %>%
  mutate(diff = video - vicon, diff_pct = 100 * diff / vicon)
save_tab(s, "L2_patient_spatiotemporal.csv")
msg("L2 patient tables done")
