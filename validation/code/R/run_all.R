# =============================================================================
# run_all.R -- regenerate EVERY table and figure of 04_outputs/ from 01_data_prepared/
#   From the package root:   Rscript 03_R/run_all.R
#   Only some levels:        Rscript 03_R/run_all.R L3        (prefix filter: L1, L2, L3, 90)
#   Per-run figures (L1, one figure per pair, slow) are NOT made by default:
#                            Rscript 03_R/run_all.R L1        (on demand)
# Each script is independent (can also be sourced alone); a failure in one does
# not stop the others -- see 04_outputs/run_all_log.txt.
# =============================================================================
source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
source(file.path(PKG, "03_R", "01_load_data.R"))

args <- commandArgs(trailingOnly = TRUE)
scripts <- sort(list.files(file.path(PKG, "03_R"), pattern = "^[1-9][0-9]_.*\\.R$", full.names = TRUE))
if (length(args)) {
  scripts <- scripts[grepl(paste(args, collapse = "|"), basename(scripts))]
} else {
  # default: tables + subject/cohort figures; skip the per-run figure scripts
  scripts <- scripts[!grepl("^1[01]_L1_", basename(scripts))]
}

log_file <- file.path(PKG, "04_outputs", "run_all_log.txt")
cat("run_all", format(Sys.time()), "\n", file = log_file)
for (s in scripts) {
  t0 <- Sys.time()
  res <- tryCatch({ source(s, local = new.env(parent = globalenv())); "OK" },
                  error = function(e) paste("ERROR:", conditionMessage(e)))
  line <- sprintf("%-45s %-6s %5.1fs %s", basename(s), substr(res, 1, 5),
                  as.numeric(difftime(Sys.time(), t0, units = "secs")),
                  if (res == "OK") "" else res)
  cat(line, "\n"); cat(line, "\n", file = log_file, append = TRUE)
}
