# Load libraries
library(brms)
library(readr)
library(dplyr)
library(tibble)
library(ggplot2)

# Load and preprocess data
df <- read_tsv("Metadata.Final.tsv")

df <- df %>%
  mutate(
    MainHap = factor(MainHap),
    site = factor(site),
    PC1_group = cut(PC1, breaks = 5, include.lowest = TRUE)
  )

# Define fixed effect covariates
covars <- "BMI + PW_AGE"

# Define reference haplogroups
refs <- c("M", "L3")

for (ref in refs) {
  cat("Running PTB models with ref =", ref, "\n")

  df <- df %>% mutate(MainHap = relevel(MainHap, ref = ref))

  # Model 1: Random intercept for site
  formula_site <- as.formula(paste0("PTB ~ MainHap + ", covars, " + (1 | site)"))
  model_site <- brm(
    formula = formula_site,
    data = df,
    family = bernoulli(),
    chains = 4,
    iter = 4000,
    cores = 4,
    control = list(adapt_delta = 0.95)
  )

  site_fixed <- as.data.frame(summary(model_site)$fixed)
  site_fixed <- site_fixed %>%
    rownames_to_column("term") %>%
    mutate(
      OR = exp(Estimate),
      OR_CI_lower = exp(`l-95% CI`),
      OR_CI_upper = exp(`u-95% CI`),
      sig = ifelse(OR_CI_lower > 1 | OR_CI_upper < 1, "*", "")
    )

  write.csv(site_fixed, paste0("ptb_fixed_effects_site_", ref, ".csv"), row.names = FALSE)

  # Plot ORs for haplogroups only
  hap_terms <- site_fixed %>%
    filter(grepl("MainHap", term)) %>%
    mutate(term_label = paste0(term, sig))

  p <- ggplot(hap_terms, aes(x = reorder(term_label, OR), y = OR)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = OR_CI_lower, ymax = OR_CI_upper), width = 0.2) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray40") +
    labs(title = paste("Odds Ratios (site model) — ref =", ref),
         x = "Haplogroup",
         y = "Odds Ratio") +
    theme_bw(base_size = 14) +
    coord_flip()

  ggsave(paste0("ptb_OR_plot_site_", ref, ".png"), plot = p, width = 7, height = 4.5, dpi = 300)

  # Model 2: Random intercept for PC1 group
  formula_pc1 <- as.formula(paste0("PTB ~ MainHap + ", covars, " + (1 | PC1_group)"))
  model_pc1 <- brm(
    formula = formula_pc1,
    data = df,
    family = bernoulli(),
    chains = 4,
    iter = 4000,
    cores = 4,
    control = list(adapt_delta = 0.95)
  )

  pc1_fixed <- as.data.frame(summary(model_pc1)$fixed)
  pc1_fixed <- pc1_fixed %>%
    rownames_to_column("term") %>%
    mutate(
      OR = exp(Estimate),
      OR_CI_lower = exp(`l-95% CI`),
      OR_CI_upper = exp(`u-95% CI`),
      sig = ifelse(OR_CI_lower > 1 | OR_CI_upper < 1, "*", "")
    )

  write.csv(pc1_fixed, paste0("ptb_fixed_effects_pc1group_", ref, ".csv"), row.names = FALSE)

  # Plot ORs for haplogroups only
  hap_terms_pc1 <- pc1_fixed %>%
    filter(grepl("MainHap", term)) %>%
    mutate(term_label = paste0(term, sig))

  p_pc1 <- ggplot(hap_terms_pc1, aes(x = reorder(term_label, OR), y = OR)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = OR_CI_lower, ymax = OR_CI_upper), width = 0.2) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray40") +
    labs(title = paste("Odds Ratios (PC1 group model) — ref =", ref),
         x = "Haplogroup",
         y = "Odds Ratio") +
    theme_bw(base_size = 14) +
    coord_flip()

  ggsave(paste0("ptb_OR_plot_pc1group_", ref, ".png"), plot = p_pc1, width = 7, height = 4.5, dpi = 300)
}
