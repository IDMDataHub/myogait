# =============================================================================
# 01_load_data.R -- read every prepared table once, harmonise types and sides
#
# SIDE CONVENTION: the Vicon side is the reference label ("side_ref").
# The pose model's left/right can be mirrored w.r.t. the anatomical side when
# the subject walks the other way; the frame-level synchronisation detected it
# (sync.csv / side_map.csv, column side_swap). All cross-system comparisons
# below use side_ref, so video "L" is compared with Vicon "L" anatomically.
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))

rd <- function(f) read_csv(file.path(DATA, f),
                           col_types = cols(patient = col_character(), .default = col_guess()))
# Myokinesis IDs are 4-digit numbers (keep the leading zero); Bath IDs are "P08"
pad_pid <- function(df) {
  if ("patient" %in% names(df)) {
    num <- grepl("^[0-9]+$", df$patient)
    df$patient[num] <- sprintf("%04d", as.integer(df$patient[num]))
  }
  df
}

subjects  <- rd("subjects.csv")                     %>% pad_pid()
runs      <- rd("runs.csv")                         %>% pad_pid()
side_map  <- read_csv(file.path(DATA, "side_map.csv"), col_types = cols(.default = col_guess()))
sync      <- rd("sync.csv")                         %>% pad_pid()
events    <- read_csv(file.path(DATA, "events.csv"))
st        <- rd("spatiotemporal.csv")               %>% pad_pid()
bio       <- rd("biomarkers.csv")                   %>% pad_pid()
agree     <- rd("agreement_curves.csv")             %>% pad_pid()
agree_ts  <- rd("agreement_synced_timeseries.csv")  %>% pad_pid()
means     <- rd("curves_run_mean_side_matched_long.csv") %>% pad_pid()   # side = Vicon side
cycles    <- rd("cycles.csv")                       %>% pad_pid()
params    <- rd("cycle_params_long.csv")            %>% pad_pid()
curves    <- rd("curves_cycles_long.csv")           %>% pad_pid()

swap_side <- function(side, system, swap) ifelse(system == "video" & swap,
                                                 ifelse(side == "L", "R", "L"), side)
add_side_ref <- function(df) df %>%
  left_join(side_map %>% select(pair_id, side_swap), by = "pair_id") %>%
  mutate(side_swap = coalesce(side_swap, FALSE),
         side_ref  = swap_side(side, system, side_swap))
cycles <- add_side_ref(cycles)
params <- add_side_ref(params)
curves <- add_side_ref(curves)
means  <- means %>% mutate(side_ref = side)

for (nm in c("subjects", "runs", "st", "bio", "means", "cycles", "params", "curves", "agree",
              "agree_ts", "sync")) {
  df <- get(nm)
  if ("group" %in% names(df)) df <- mutate(df, group = factor(group, c("SAIN", "NMD", "HEALTHY")))
  if ("setting" %in% names(df)) df <- mutate(df, setting = factor(setting, SETTINGS))
  assign(nm, df)
}

msg("data loaded: ", nrow(runs), " paired runs (",
    paste(names(table(runs$dataset)), table(runs$dataset), collapse = ", "), "), ",
    n_distinct(runs$patient), " subjects, ", nrow(curves), " cycle-curve samples")

# ---- shared aggregates ---------------------------------------------------------
# run-level kinematic parameters: mean over the run's cycles, per system/side/joint
run_params <- params %>%
  group_by(pair_id, dataset, setting, patient, group, system, side_ref, joint) %>%
  summarise(n_cycles = n(), rom = mean(rom), max = mean(max), min = mean(min),
            mean = mean(mean), pct_at_max = mean(pct_at_max), .groups = "drop")
# paired (video vs vicon) wide version on the common joints
run_params_wide <- run_params %>% filter(joint %in% JOINTS) %>%
  select(-n_cycles) %>%
  pivot_longer(c(rom, max, min, mean, pct_at_max), names_to = "param") %>%
  pivot_wider(names_from = system, values_from = value) %>%
  filter(is.finite(video), is.finite(vicon))
# patient-level (mean over runs and both sides)
patient_params_wide <- run_params_wide %>%
  group_by(dataset, setting, patient, group, joint, param) %>%
  summarise(video = mean(video), vicon = mean(vicon), n_runs = n_distinct(pair_id), .groups = "drop")

ST_PARAMS <- c(cadence_spm = "Cadence (steps/min)", stride_time_s = "Stride time (s)",
               step_time_s = "Step time (s)", stance_pct_L = "Stance L (%)",
               stance_pct_R = "Stance R (%)", double_support_pct = "Double support (%)",
               step_length_L_m = "Step length L (m)", step_length_R_m = "Step length R (m)",
               stride_length_m = "Stride length (m)", speed_mps = "Walking speed (m/s)")
# lateralised video parameters follow the same L/R mapping as the curves
st <- st %>% left_join(side_map %>% select(pair_id, side_swap), by = "pair_id") %>%
  mutate(side_swap = coalesce(side_swap, FALSE))
sw <- st$system == "video" & st$side_swap
for (a in list(c("stance_pct_L", "stance_pct_R"), c("swing_pct_L", "swing_pct_R"),
               c("step_length_L_m", "step_length_R_m"))) {
  tmp <- st[[a[1]]][sw]; st[[a[1]]][sw] <- st[[a[2]]][sw]; st[[a[2]]][sw] <- tmp
}
st_wide <- st %>%
  select(pair_id, dataset, setting, patient, group, system, all_of(names(ST_PARAMS))) %>%
  pivot_longer(all_of(names(ST_PARAMS)), names_to = "param") %>%
  pivot_wider(names_from = system, values_from = value) %>%
  filter(is.finite(video), is.finite(vicon))
