# =============================================================================
# LEVEL 3 -- accelerometry-style biomarkers (virtual accelerometer at the pelvis)
# IH (index of harmonicity), RMS (AP, vertical), LF/HF, harmonic ratio (AP, vertical).
# CAUTION: markerless RMS is in image-normalised units while Vicon RMS is in
# metric units -> compare them by association (Spearman / ICC(3,1) consistency),
# not by absolute agreement. Dimensionless ratios (IH, HR, LF/HF) are closer to
# comparable, but still depend on sampling rate (60 vs 200 Hz) and filtering.
# Outputs: tables/L3_biomarkers_validity.csv, tables/L3_biomarkers_groups.csv,
#          figures/L3_cohort/biomarkers_*.png
# =============================================================================
if (!exists("PKG")) source(file.path(if (dir.exists("03_R")) "03_R" else ".", "00_config.R"))
if (!exists("curves")) source(file.path(PKG, "03_R", "01_load_data.R"))

BIO <- c(IH_ap = "Index of harmonicity (AP)", RMS_ap = "RMS acceleration (AP)",
         RMS_vert = "RMS acceleration (vertical)", LF_HF_ap = "LF/HF ratio (AP)",
         HR_ap = "Harmonic ratio (AP)", HR_vert = "Harmonic ratio (vertical)")
bl <- bio %>% pivot_longer(all_of(names(BIO)), names_to = "biomarker")
bw <- bl %>% select(pair_id, setting, patient, group, system, biomarker, value) %>%
  pivot_wider(names_from = system, values_from = value) %>% filter(is.finite(video), is.finite(vicon))

val <- bw %>% group_by(setting, biomarker) %>%
  group_modify(~ bind_cols(icc_pair(.x$video, .x$vicon),
                           tibble(spearman = suppressWarnings(cor(.x$video, .x$vicon, method = "spearman"))))) %>%
  ungroup()
save_tab(val, "L3_biomarkers_validity.csv")

pp <- bl %>% filter(dataset == "myokinesis") %>% group_by(patient, group, system, biomarker) %>% summarise(value = mean(value, na.rm = TRUE), .groups = "drop")
grp <- pp %>% filter(is.finite(value)) %>% group_by(biomarker, system) %>%
  summarise(hedges_g = hedges_g(value[group == "SAIN"], value[group == "NMD"]),
            p_welch = tryCatch(t.test(value[group == "SAIN"], value[group == "NMD"])$p.value, error = function(e) NA),
            .groups = "drop")
save_tab(grp, "L3_biomarkers_groups.csv")

p1 <- ggplot(bw %>% mutate(biomarker = factor(BIO[biomarker], BIO)), aes(vicon, video, colour = setting)) +
  geom_point(size = 1.4, alpha = .7) +
  facet_wrap(~biomarker, scales = "free") + scale_colour_manual(values = COL_SETTING, name = NULL) +
  guides(colour = guide_legend(ncol = 2)) +
  labs(title = "Biomarkers: Sapiens2 (y) vs Vicon (x), per run", x = "Vicon", y = "Sapiens2") + theme_mk(9)
save_fig(p1, "L3_cohort", "biomarkers_scatter.png", 12, 7)

p2 <- ggplot(pp %>% mutate(biomarker = factor(BIO[biomarker], BIO)), aes(group, value, fill = group)) +
  geom_boxplot(alpha = .35, outlier.shape = NA, width = .6) +
  geom_point(aes(colour = group), position = position_jitter(width = .1, seed = 2), size = 1.3) +
  facet_grid(biomarker ~ system, scales = "free_y", labeller = labeller(system = LAB_SYS)) +
  scale_fill_manual(values = COL_GROUP) + scale_colour_manual(values = COL_GROUP) +
  labs(title = "Biomarkers by group and system (Myokinesis, patient means)", x = NULL, y = NULL) +
  theme_mk(8) + theme(legend.position = "none", strip.text.y = element_text(angle = 0))
save_fig(p2, "L3_cohort", "biomarkers_groups.png", 8, 12)
msg("L3 biomarkers done")
