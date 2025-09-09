# =====================================================
# Single-site PTB (Bayesian, brms)
# - No random effects (single site)
# - Adjust for BMI, AGE, PC1..PC5
# - Skeptical priors on haplogroup coefficients
# - Exports ORs, 95% CrIs, Pr(OR>1), ROPE
# =====================================================

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(forcats); library(stringr); library(tidyr)
  library(brms); library(posterior)
})

set.seed(2025)

# ----------------------------
# Config
# ----------------------------
INFILE   <- "Metadata.Final.tsv"
OUTDIR   <- "model_outputs_single_site_bayes"
if (!dir.exists(OUTDIR)) dir.create(OUTDIR, recursive = TRUE)

# Minimum per-hap sample size within site (to avoid extreme singletons).
# Set to 1 to disable filtering; set to 5 (default) for stability.
MIN_PER_HAP <- 5

# ROPE (region of practical equivalence) on log-odds:
# |log(OR)| < 0.2 ~ "too small to matter" (~OR 0.82–1.22)
ROPE_LOGOR <- 0.2

# Sampler controls
CTRL <- list(adapt_delta = 0.99, max_treedepth = 13)

# ----------------------------
# Helpers
# ----------------------------
zscore_by <- function(x, g) (x - ave(x, g, FUN = mean, na.rm = TRUE)) / ave(x, g, FUN = sd, na.rm = TRUE)

choose_ref <- function(fct) {
  tab <- sort(table(fct), decreasing = TRUE)
  names(tab)[1]
}

make_hap_priors <- function(d) {
  ref <- levels(d$MainHap)[1]
  hlv <- setdiff(levels(d$MainHap), ref)
  c(
    lapply(hlv, function(h) prior(normal(0, 0.5), class = "b", coef = paste0("MainHap", h))), # skeptical on hap
    prior(normal(0, 1), class = "b")  # BMI_s, AGE_s, PCs
  )
}

tidy_from_summary <- function(fit, d, rope = ROPE_LOGOR) {
  fx <- as.data.frame(summary(fit)$fixed) |>
    tibble::rownames_to_column("term") |>
    mutate(
      label = case_when(
        grepl("^MainHap\\[T\\.", term) ~ sub("^MainHap\\[T\\.(.+)\\]$", "\\1", term),
        grepl("^MainHap", term)        ~ sub("^MainHap", "", term),
        TRUE ~ term
      ),
      OR     = exp(Estimate),
      OR_low = exp(`l-95% CI`),
      OR_hi  = exp(`u-95% CI`)
    )

  # Posterior draws for Pr(OR>1) and ROPE
  draws <- as_draws_df(fit)
  hap_cols <- grep("^b_MainHap", names(draws), value = TRUE)
  prob_tbl <- NULL
  if (length(hap_cols)) {
    prob_tbl <- tibble(
      term  = sub("^b_", "", hap_cols),
      Pr_OR_gt_1 = colMeans(exp(as.matrix(draws[, hap_cols, drop = FALSE])) > 1),
      ROPE_small = colMeans(abs(as.matrix(draws[, hap_cols, drop = FALSE])) < rope)
    )
    prob_tbl$label <- prob_tbl$term |>
      sub("^MainHap\\[T\\.(.+)\\]$", "\\1", x = _) |>
      sub("^MainHap", "", x = _)
  }

  out <- fx |>
    left_join(prob_tbl, by = c("term", "label"))

  # attach n per hap
  n_per <- as.data.frame(table(d$MainHap)) |> dplyr::rename(label = Var1, n = Freq)
  out <- out |> left_join(n_per, by = "label")
  out
}

# ----------------------------
# Load & prepare
# ----------------------------
df <- read_tsv(INFILE, show_col_types = FALSE)

# Ensure needed columns exist
need <- c("PTB","MainHap","BMI","PW_AGE","site")
miss <- setdiff(need, names(df))
if (length(miss)) stop("Missing columns: ", paste(miss, collapse = ", "))

# Factorize
df <- df |>
  mutate(
    site    = factor(site),
    MainHap = factor(MainHap)
  )

# Add PCs if missing (fill with 0 so formulas still work)
for (k in 1:5) {
  pc <- paste0("PC", k)
  if (!pc %in% names(df)) df[[pc]] <- 0
}

# Scale covariates WITHIN SITE (safer for single-site fits)
df <- df |>
  group_by(site) |>
  mutate(
    BMI_s = zscore_by(BMI, site),
    AGE_s = zscore_by(PW_AGE, site)
  ) |>
  ungroup()

# ----------------------------
# Per-site loop
# ----------------------------
all_results <- list()

for (s in levels(df$site)) {
  dd <- df |> filter(site == s) |> droplevels()

  # Choose reference = most common hap in this site
  ref <- choose_ref(dd$MainHap)
  dd$MainHap <- fct_relevel(dd$MainHap, ref)

  # Filter ultra-rare haplogroups for stability (optional)
  keep <- names(which(table(dd$MainHap) >= MIN_PER_HAP))
  if (length(keep) <= 1) {
    message(sprintf("[%s] Skipped: <2 haplogroups after filtering (min per hap = %d).", s, MIN_PER_HAP))
    next
  }
  dd <- dd |> filter(MainHap %in% keep) |> droplevels()

  # Build priors targeted at hap terms
  pri <- make_hap_priors(dd)

  # Fit Bayesian logistic (no random effects)
  form <- bf(PTB ~ MainHap + BMI_s + AGE_s + PC1 + PC2 + PC3 + PC4 + PC5)
  fit <- brm(
    form, data = dd, family = bernoulli(),
    prior = pri,
    chains = 4, iter = 4000, cores = 4,
    control = CTRL, seed = 2025
  )

  # Tidy & export
  tab <- tidy_from_summary(fit, dd)
  tab$site <- s
  tab$ref  <- ref

  # Write per-site CSV
  out_site <- file.path(OUTDIR, paste0("PTB_bayes_", gsub("[^A-Za-z0-9]+", "_", s), ".csv"))
  write_csv(tab, out_site)

  # Keep only hap rows for combined summary
  all_results[[s]] <- tab |> filter(grepl("^MainHap", term))
}

# Combined summary across sites
if (length(all_results)) {
  comb <- bind_rows(all_results)
  write_csv(comb, file.path(OUTDIR, "PTB_bayes_single_site_summary.csv"))
  cat("Done. Wrote per-site tables and combined summary to:", normalizePath(OUTDIR), "\n")
} else {
  cat("No sites produced results. Check MIN_PER_HAP or data availability.\n")
}
