# =============================================================================
# 00_config.R -- paths, packages, palette, theme and shared statistics helpers
# Sourced by every other script. Run everything from the PACKAGE ROOT
# (the folder that contains 01_data_prepared/), e.g.  Rscript 03_R/run_all.R
# =============================================================================

pkgs <- c("readr", "dplyr", "tidyr", "ggplot2", "purrr", "stringr",
          "patchwork", "scales", "irr")
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE))
  install.packages(p, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(tidyr); library(ggplot2)
  library(purrr); library(stringr); library(patchwork); library(scales)
})
options(dplyr.summarise.inform = FALSE, readr.show_col_types = FALSE)

# ---- package root ------------------------------------------------------------
PKG <- Sys.getenv("MYOKIN_PKG", unset = "")
if (PKG == "") {
  PKG <- if (dir.exists("01_data_prepared")) "." else
         if (dir.exists("../01_data_prepared")) ".." else
         stop("Run from the package root (folder containing 01_data_prepared/), ",
              "or set the MYOKIN_PKG environment variable.")
}
DATA <- file.path(PKG, "01_data_prepared")
TAB  <- file.path(PKG, "04_outputs", "tables")
FIG  <- file.path(PKG, "04_outputs", "figures")
for (d in c(TAB, file.path(FIG, c("L1_run_cycles", "L1_run_synced", "L2_patient",
                                  "L3_cohort", "L3_views"))))
  dir.create(d, recursive = TRUE, showWarnings = FALSE)

# ---- palette & theme (same identity as the report) --------------------------
COL_SYS   <- c(vicon = "#22314E", video = "#E4572E")
LAB_SYS   <- c(vicon = "Vicon (200 Hz)", video = "Sapiens2 (iPhone)")
COL_GROUP <- c(SAIN = "#2A9D8F", NMD = "#E76F51", HEALTHY = "#7B8CDE")
# one colour per recording setting (dataset x camera view)
SETTINGS <- c("Myokinesis - iPhone following the subject", "Bath - lateral left (cam01)",
              "Bath - lateral right (cam05)", "Bath - frontal (cam03)", "Bath - rear (cam07)")
COL_SETTING <- setNames(c("#E4572E", "#22314E", "#4F7CAC", "#9BC53D", "#C3A995"), SETTINGS)
slug <- function(x) gsub("[^A-Za-z0-9]+", "_", x)
JOINTS    <- c("hip", "knee", "ankle")
LAB_JOINT <- c(hip = "Hip flexion", knee = "Knee flexion", ankle = "Ankle dorsiflexion",
               trunk = "Trunk")

theme_mk <- function(base = 11) {
  theme_minimal(base_size = base) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(colour = "#E3E7ED"),
          strip.text = element_text(face = "bold"),
          plot.title = element_text(face = "bold"),
          plot.title.position = "plot",
          legend.position = "bottom",
          plot.background = element_rect(fill = "white", colour = NA))
}

save_fig <- function(p, subdir, file, w = 9, h = 5.5) {
  ggsave(file.path(FIG, subdir, file), p, width = w, height = h, dpi = 300, bg = "white")
}
save_tab <- function(df, file) write_csv(df, file.path(TAB, file))

# ---- statistics helpers -------------------------------------------------------
# Bland-Altman with 95% CI of bias and of each limit (Bland & Altman 1999),
# plus a proportional-bias test (regression of difference on mean).
ba_stats <- function(video, vicon) {
  d <- video - vicon; m <- (video + vicon) / 2
  ok <- is.finite(d) & is.finite(m); d <- d[ok]; m <- m[ok]; n <- length(d)
  if (n < 3) return(tibble(n = n))
  b <- mean(d); s <- sd(d); tq <- qt(0.975, n - 1)
  se_b <- s / sqrt(n); se_l <- sqrt(3 * s^2 / n)
  cf <- summary(lm(d ~ m))$coefficients
  tibble(n = n, bias = b, sd_diff = s,
         bias_ci_low = b - tq * se_b, bias_ci_high = b + tq * se_b,
         loa_low = b - 1.96 * s, loa_high = b + 1.96 * s,
         loa_low_ci_low = b - 1.96 * s - tq * se_l, loa_low_ci_high = b - 1.96 * s + tq * se_l,
         loa_high_ci_low = b + 1.96 * s - tq * se_l, loa_high_ci_high = b + 1.96 * s + tq * se_l,
         prop_bias_slope = cf[2, 1], prop_bias_p = cf[2, 4])
}

# ICC(2,1) absolute agreement and ICC(3,1) consistency, two-way, single measure.
icc_pair <- function(video, vicon) {
  ok <- is.finite(video) & is.finite(vicon)
  m <- cbind(vicon[ok], video[ok])
  if (nrow(m) < 5) return(tibble(n = nrow(m)))
  a <- irr::icc(m, "twoway", "agreement", "single")
  c <- irr::icc(m, "twoway", "consistency", "single")
  tibble(n = nrow(m),
         icc_2_1 = a$value, icc_2_1_low = a$lbound, icc_2_1_high = a$ubound,
         icc_3_1 = c$value, icc_3_1_low = c$lbound, icc_3_1_high = c$ubound,
         pearson_r = suppressWarnings(cor(m[, 1], m[, 2])))
}

hedges_g <- function(a, b) {
  a <- a[is.finite(a)]; b <- b[is.finite(b)]
  if (length(a) < 2 || length(b) < 2) return(NA_real_)
  sp <- sqrt(((length(a) - 1) * var(a) + (length(b) - 1) * var(b)) / (length(a) + length(b) - 2))
  if (sp == 0) return(NA_real_)
  (mean(a) - mean(b)) / sp * (1 - 3 / (4 * (length(a) + length(b)) - 9))
}

mean_ci <- function(x) {
  x <- x[is.finite(x)]; n <- length(x)
  if (n < 2) return(tibble(n = n, mean = mean(x), sd = NA, ci_low = NA, ci_high = NA))
  h <- qt(0.975, n - 1) * sd(x) / sqrt(n)
  tibble(n = n, mean = mean(x), sd = sd(x), ci_low = mean(x) - h, ci_high = mean(x) + h,
         median = median(x), q1 = quantile(x, .25), q3 = quantile(x, .75))
}

msg <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")), ..., "\n", sep = "")
