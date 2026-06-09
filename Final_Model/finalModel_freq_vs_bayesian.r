# ============================
# Joint Cohort (All sites pooled): GA & PTB models
# Dynamic covariate PTB/GA pipeline
# Supports fixed site or random site
# Supports optional PCs and clinical/environmental covariates
# - GA: Student-t; back-transform to days
# - PTB: Binomial; forest plot, EMMs, Pr(OR>1)
# ============================

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(stringr); library(forcats)
  library(ggplot2); library(broom); library(broom.mixed)
  library(glmmTMB); library(emmeans); library(brms)
  library(DHARMa); library(posterior); library(tidyr); library(tibble)
})

set.seed(2025)


# ---- brms convergence diagnostics helper ----
save_brms_diagnostics <- function(fit, prefix, outdir) {
  # Summary table
  summ <- as.data.frame(summary(fit)$fixed)
  summ$term <- rownames(summ)
  write_csv(summ, file.path(outdir, paste0(prefix, "_fixed_effects_summary.csv")))

  # Full posterior summary with Rhat and ESS
  draw_summ <- posterior::summarise_draws(
    posterior::as_draws_df(fit),
    mean,
    sd,
    median,
    mad,
    ~posterior::quantile2(.x, probs = 0.025),
    ~posterior::quantile2(.x, probs = 0.975),
    posterior::rhat,
    posterior::ess_bulk,
    posterior::ess_tail
  )
  draw_summ <- as.data.frame(draw_summ)

  names(draw_summ)[names(draw_summ) == "posterior::rhat"] <- "rhat"
  names(draw_summ)[names(draw_summ) == "posterior::ess_bulk"] <- "ess_bulk"
  names(draw_summ)[names(draw_summ) == "posterior::ess_tail"] <- "ess_tail"
    
  write_csv(draw_summ, file.path(outdir, paste0(prefix, "_draws_summary.csv")))

  # NUTS diagnostics
  np <- nuts_params(fit)
  write_csv(np, file.path(outdir, paste0(prefix, "_nuts_params.csv")))

  # Divergences
  n_div <- sum(np$Parameter == "divergent__" & np$Value == 1, na.rm = TRUE)

  # Max treedepth hits
  n_treedepth <- sum(np$Parameter == "treedepth__" & np$Value >= 15, na.rm = TRUE)

  # BFMI
  # BFMI / energy diagnostic (version-safe)
  bfmi_text <- tryCatch({
    if ("nuts_energy" %in% getNamespaceExports("bayesplot")) {
      capture.output(print(bayesplot::nuts_energy(fit)))
    } else {
      "BFMI check not available in this bayesplot version."
    }
  }, error = function(e) {
    paste("BFMI check failed:", e$message)
  })


  # Rhat / ESS flags
  bad_rhat <- draw_summ %>% filter(!is.na(rhat) & rhat > 1.01)
  low_ess_bulk <- draw_summ %>% filter(!is.na(ess_bulk) & ess_bulk < 400)
  low_ess_tail <- draw_summ %>% filter(!is.na(ess_tail) & ess_tail < 400)

  write_csv(bad_rhat, file.path(outdir, paste0(prefix, "_bad_rhat.csv")))
  write_csv(low_ess_bulk, file.path(outdir, paste0(prefix, "_low_ess_bulk.csv")))
  write_csv(low_ess_tail, file.path(outdir, paste0(prefix, "_low_ess_tail.csv")))

  diag_lines <- c(
    paste("Model:", prefix),
    paste("Divergences:", n_div),
    paste("Treedepth hits:", n_treedepth),
    "",
    "BFMI output:",
    bfmi_text
  )

  writeLines(diag_lines, file.path(outdir, paste0(prefix, "_diagnostics.txt")))

  # Trace / mixing plots
  png(file.path(outdir, paste0(prefix, "_traceplot.png")), width = 1600, height = 1200)
  plot(fit)
  dev.off()

  # Posterior predictive check
  png(file.path(outdir, paste0(prefix, "_pp_check.png")), width = 1200, height = 900)
  pp_check(fit, ndraws = 200)
  dev.off()
}



# ---------------------------------
# USER SETTING: only edit these lines
# ---------------------------------
# ==== CONFIG ====

args <- commandArgs(trailingOnly = TRUE)

# Choose a default reference for the Joint cohort
DEFAULT_Ref <- args[1]
covariates <- args[2]
INFILE <- "Metadata.Final.tsv"
# ---------------------------------
# Build output directory name
# ---------------------------------

cov_string <- covariates %>%
  gsub("\\s+", "", .) %>%
  gsub("\\(1\\|site\\)", "SITE_RANDOM", .) %>%  # temporary placeholder
  gsub("\\+", "_", .) %>%
  gsub("\\bsite\\b", "siteFE", .) %>%           # whole word only
  gsub("SITE_RANDOM", "siteRE", .) %>%
  gsub("[()|]", "", .)

OUTDIR <- file.path(
  "model_outputs",
  paste0(
    "All_",
    DEFAULT_Ref,
    "_",
    cov_string
  )
)

if (!dir.exists(OUTDIR)) {
  dir.create(OUTDIR, recursive = TRUE)
}










# ---------------------------------
# CONFIG
# ---------------------------------

columnCat <- c(
  "FUEL_FOR_COOK",
  "site",
  "MainHap"
)

columnCont <- c(
  "PW_AGE",
  "PW_EDUCATION",
  "MAT_HEIGHT",
  "MAT_WEIGHT",
  "BMI",
  "TOILET",
  "WEALTH_INDEX",
  "DRINKING_SOURCE",
  "PC1",
  "PC2",
  "PC3",
  "PC4",
  "PC5"
)

columnBin <- c(
  "BABY_SEX",
  "CHRON_HTN",
  "DIABETES",
  "HH_ELECTRICITY",
  "TB",
  "THYROID",
  "TYP_HOUSE"
)



# LOAD RAW DATA,PREPROCESS, AND SAVE GA SCALE
# ---------------------------------

df_raw <- read_tsv(INFILE, show_col_types = FALSE)

ga_mean_raw <- mean(df_raw$GAGEBRTH, na.rm = TRUE)
ga_sd_raw   <- sd(df_raw$GAGEBRTH, na.rm = TRUE)


# ---------------------------------
# LOAD & PREPROCESS FOR MODELING
# ---------------------------------

df <- df_raw %>%
  mutate(

    # categorical
    across(all_of(columnCat), as.factor),

    # continuous / ordinal scaled IN PLACE
    across(
      all_of(columnCont),
      ~ as.numeric(scale(.x))
    ),

    # binary
    across(all_of(columnBin), as.numeric),

    # GA outcome scaled for modeling
    GAGEBRTH = as.numeric(scale(GAGEBRTH))
  )


# Validate binary columns are 0/1/NA only
bad_bin <- sapply(df[columnBin], function(x) any(!is.na(x) & !x %in% c(0, 1)))
if (any(bad_bin)) {
  stop("Binary columns contain non-0/1 values: ",
       paste(names(bad_bin)[bad_bin], collapse = ", "))
}


# Ensure reference haplogroup is present; otherwise pick the most frequent
if (!(DEFAULT_Ref %in% levels(df$MainHap))) {
  message(sprintf("Note: default ref '%s' not found; using most frequent MainHap as ref.", DEFAULT_Ref))
  fallback <- names(sort(table(df$MainHap), decreasing = TRUE))[1]
  df$MainHap <- fct_relevel(df$MainHap, fallback)
} else {
  df$MainHap <- fct_relevel(df$MainHap, DEFAULT_Ref)
}
ref_name <- levels(df$MainHap)[1]
n_ref <- sum(df$MainHap == ref_name, na.rm = TRUE)
if (n_ref < 100) message(sprintf("Warning: reference '%s' has n=%d", ref_name, n_ref))

writeLines(
  c(
    paste("Reference:", ref_name),
    paste("Original reference requested:", DEFAULT_Ref),
    paste("Covariates:", covariates),
    paste("GA formula:", paste("GAGEBRTH ~ MainHap +", covariates)),
    paste("PTB formula:", paste("PTB ~ MainHap +", covariates)),
    paste("Raw GA mean:", ga_mean_raw),
    paste("Raw GA SD:", ga_sd_raw)
  ),
  file.path(OUTDIR, "model_formula_used.txt")
)

hap_names <- paste0("MainHap", levels(df$MainHap)[-1])

# ---- Helpers ----
hap_mask <- function(terms, var = "MainHap") grepl(paste0("^", var), terms)

bh_on_hap <- function(tbl, term_col = "term", p_col = "p.value", var = "MainHap") {
  if (!p_col %in% names(tbl)) stop("bh_on_hap needs a p.value column.")
  m <- hap_mask(tbl[[term_col]], var)
  q <- rep(NA_real_, nrow(tbl))
  if (any(m)) q[m] <- p.adjust(tbl[[p_col]][m], method = "BH")
  dplyr::mutate(tbl, padj = q)
}

bh_on_hap_wald <- function(tbl, term_col="term", mean_col="Estimate", se_col="Est.Error", var="MainHap") {
  stopifnot(all(c(term_col, mean_col, se_col) %in% names(tbl)))
  z <- abs(tbl[[mean_col]] / tbl[[se_col]])
  p <- 2 * pnorm(-z)
  m <- hap_mask(tbl[[term_col]], var)
  q <- rep(NA_real_, nrow(tbl))
  if (any(m)) q[m] <- p.adjust(p[m], method = "BH")
  tibble::add_column(tbl, p.value = p, padj = q, .after = se_col)
}

to_or <- function(tbl, est = "estimate", lo = "conf.low", hi = "conf.high") {
  tbl %>% mutate(OR = exp(.data[[est]]),
                 OR_low = exp(.data[[lo]]),
                 OR_hi  = exp(.data[[hi]]))
}

# Friendly labels from either "MainHap[T.X]" or "MainHapX"
robust_hap_label <- function(x) {
  dplyr::case_when(
    grepl("^MainHap\\[T\\.", x) ~ sub("^MainHap\\[T\\.(.+)\\]$", "\\1", x),
    grepl("^MainHap", x)        ~ sub("^MainHap", "", x),
    TRUE ~ x
  )
}

save_forest_ptb <- function(tbl, title, file, label_col = "term", or = "OR", lo = "OR_low", hi = "OR_hi") {
  d <- tbl %>% filter(hap_mask(.data[[label_col]]))
  if (!nrow(d)) return(invisible(NULL))
  d$label <- robust_hap_label(d[[label_col]])
  gg <- ggplot(d, aes(x = reorder(label, .data[[or]]), y = .data[[or]])) +
    geom_point(size = 2.6) +
    geom_errorbar(aes(ymin = .data[[lo]], ymax = .data[[hi]]), width = 0.2) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
    coord_flip() + theme_bw(13) +
    labs(title = title, x = sprintf("Haplogroup (ref: %s)", ref_name), y = "Odds ratio (PTB)")
  ggsave(file, gg, width = 7, height = 4.8, dpi = 300)
}


# ---- Detect whether formula includes a site random effect ----
has_site_re <- grepl("\\(1\\s*\\|\\s*site\\)", covariates)
has_site_fe <- grepl("(^|\\+)\\s*site\\s*($|\\+)", covariates) && !has_site_re

message("covariates = ", covariates)
message("has_site_re = ", has_site_re)
message("has_site_fe = ", has_site_fe)

# ---- Helper to build hap-only priors ----
make_hap_priors <- function(hap_names, sd_hap = 0.5) {
  pri_list <- lapply(hap_names, function(nm)
    prior_string(sprintf("normal(0, %g)", sd_hap), class = "b", coef = nm)
  )
  do.call(c, pri_list)
}

# ---- GA priors ----
make_pri_ga <- function(covariates, hap_names, sd_hap = 0.5) {
  pri <- make_hap_priors(hap_names, sd_hap = sd_hap)

  pri <- c(
    pri,
    prior(student_t(3, 0, 2.5), class = "sigma")
  )

  if (has_site_re) {
    pri <- c(
      pri,
      prior(student_t(3, 0, 2.5), class = "sd")
    )
  }

  pri
}
pri_ga <- make_pri_ga(covariates, hap_names, sd_hap = 0.5)
ctrl_ga  <- list(adapt_delta = 0.999, max_treedepth = 15)      
                  
# ---- PTB priors ----
make_pri_ptb <- function(covariates, hap_names, sd_hap = 1.0) {
  pri <- make_hap_priors(hap_names, sd_hap = sd_hap)

  if (has_site_re) {
    pri <- c(
      pri,
      prior(student_t(3, 0, 2.5), class = "sd")
    )
  }

  pri
}
pri_ptb <- make_pri_ptb(covariates, hap_names, sd_hap = 1.0)
ctrl_ptb <- list(adapt_delta = 0.99,  max_treedepth = 13)

# ============================
# Frequentist: glmmTMB
# ============================

# ============================
# GA frequentist models: Gaussian + Student-t
# ============================

ga_formula <- as.formula(paste("GAGEBRTH ~ MainHap +", covariates))

# ---- GA Gaussian ----
ga_tmb_gaussian <- glmmTMB(
  ga_formula,
  data = df,
  family = gaussian()
)

tidy_ga_gaussian <- broom.mixed::tidy(
  ga_tmb_gaussian,
  effects = "fixed",
  conf.int = TRUE
) %>%
  bh_on_hap() %>%
  mutate(
    beta_days = estimate * ga_sd_raw,
    lo_days   = conf.low * ga_sd_raw,
    hi_days   = conf.high * ga_sd_raw
  )

write_csv(
  tidy_ga_gaussian,
  file.path(OUTDIR, "ga_glmmtmb_gaussian.csv")
)

# Diagnostics: DHARMa Gaussian
png(file.path(OUTDIR, "ga_glmmtmb_gaussian_DHARMa.png"), width = 1200, height = 900)
plot(simulateResiduals(ga_tmb_gaussian))
dev.off()


# ---- GA Student-t ----
ga_tmb_student <- glmmTMB(
  ga_formula,
  data = df,
  family = t_family()
)

tidy_ga_student <- broom.mixed::tidy(
  ga_tmb_student,
  effects = "fixed",
  conf.int = TRUE
) %>%
  bh_on_hap() %>%
  mutate(
    beta_days = estimate * ga_sd_raw,
    lo_days   = conf.low * ga_sd_raw,
    hi_days   = conf.high * ga_sd_raw
  )

write_csv(
  tidy_ga_student,
  file.path(OUTDIR, "ga_glmmtmb_student_t.csv")
)

# Diagnostics: DHARMa Student-t
png(file.path(OUTDIR, "ga_glmmtmb_student_t_DHARMa.png"), width = 1200, height = 900)
plot(simulateResiduals(ga_tmb_student))
dev.off()


# ---- Compare GA Gaussian vs Student-t ----
ga_model_compare <- tibble::tibble(
  model = c("Gaussian", "StudentT"),
  AIC = c(
    AIC(ga_tmb_gaussian),
    AIC(ga_tmb_student)
  )
)

write_csv(
  ga_model_compare,
  file.path(OUTDIR, "ga_glmmtmb_gaussian_vs_student_t_AIC.csv")
)

# PTB (Binomial logit), site random intercept
ptb_tmb <- glmmTMB(as.formula(paste("PTB ~ MainHap +", covariates)),
                   data = df, family = binomial())
fx_ptb  <- broom.mixed::tidy(ptb_tmb, effects = "fixed", conf.int = TRUE) %>%
  bh_on_hap() %>% to_or()
write_csv(fx_ptb, file.path(OUTDIR, "ptb_glmmtmb.csv"))

analysis_label <- if ("site" %in% names(df) &&
                      length(unique(df$site)) == 1) {
  unique(as.character(df$site))
} else {
  "All Sites"
}

save_forest_ptb(
  fx_ptb,
  paste0("PTB GLMM (glmmTMB, ", analysis_label, ")"),
  file.path(OUTDIR, "ptb_glmmtmb_site_forest.png")
)

# Diagnostics: DHARMa (PTB)
png(file.path(OUTDIR, "ptb_glmmtmb_DHARMa.png"), width=1200, height=900)
plot(simulateResiduals(ptb_tmb))
dev.off()

# Predicted PTB probability by haplogroup (marginal over random site)
emm_ptb <- try(emmeans(ptb_tmb, ~ MainHap, type = "response"), silent = TRUE)
if (!inherits(emm_ptb, "try-error")) {
  write.csv(as.data.frame(emm_ptb),
            file.path(OUTDIR, "ptb_glmmtmb_emmeans_probs.csv"),
            row.names = FALSE)
  pair_tab <- as.data.frame(pairs(emm_ptb, adjust = "BH"))
  write.csv(pair_tab,
            file.path(OUTDIR, "ptb_glmmtmb_emmeans_pairs_BH.csv"),
            row.names = FALSE)
}

# ============================
# Bayesian: brms mirrors with robust priors/controls
# ============================

# GA (Student-t) on standardized outcome, with back-transform to days
brm_ga <- brm(
  as.formula(paste("GAGEBRTH ~ MainHap +", covariates)), 
  data = df, family = student(),
  prior = pri_ga,
  chains = 4, iter = 4000, cores = 4,
  control = ctrl_ga, init = 0, seed = 2025
)
sink(file.path(OUTDIR, "ga_brm_summary.txt")); print(summary(brm_ga)); sink()

sd_ga <- ga_sd_raw



png(file.path(OUTDIR, "ga_brm_pp_check.png"), width=1200, height=900)
pp_check(brm_ga, ndraws=200)
dev.off()
capture.output(bayes_R2(brm_ga), file = file.path(OUTDIR, "ga_brm_bayesR2.txt"))


saveRDS(brm_ga, file.path(OUTDIR, "ga_brm.rds"))


# --- Posterior probabilities for GA (Student-t model) ---
# Assumes: brm_ga already fit; df in memory; your robust_hap_label() & hap_mask() exist.





# 1) Posterior draws for fixed effects (each column is a parameter draw on the standardized scale)
dr <- posterior::as_draws_df(brm_ga)
fix_cols <- grep("^b_", names(dr), value = TRUE)  # all fixed-effect betas, e.g., b_Intercept, b_MainHap[T.L2], etc.

# 2) Compute posterior probabilities per coefficient
post_tab <- lapply(fix_cols, function(nm) {
  s <- as.numeric(dr[[nm]])                        # draws on standardized GA scale
  data.frame(
    param          = nm,
    Pr_beta_gt0    = mean(s > 0),                  # Pr(β > 0)
    p_two          = 2 * pmin(mean(s > 0), mean(s < 0)),  # two-sided posterior sign probability (Bayesian analog)
    Pr_days_gt_1   = mean(sd_ga * s >  1),         # Pr(effect > +1 day)
    Pr_days_lt_m1  = mean(sd_ga * s < -1)          # Pr(effect < -1 day)
  )
}) %>% bind_rows()

# 3) Tidy names to match your summary table 'term' column
post_tab <- post_tab %>%
  mutate(term = sub("^b_", "", param),
         label = robust_hap_label(term)) %>%
  select(term, label, Pr_beta_gt0, p_two, Pr_days_gt_1, Pr_days_lt_m1)

# 4) Join to your existing GA summary, add back-transformed days, and (optionally) BH within hap terms
fx_brm_ga <- as.data.frame(summary(brm_ga)$fixed) %>%
  rownames_to_column("term") %>%
  mutate(
    beta_days = Estimate   * sd_ga,
    lo_days   = `l-95% CI` * sd_ga,
    hi_days   = `u-95% CI` * sd_ga,
    label     = robust_hap_label(term)
  ) %>%
  left_join(post_tab, by = c("term","label"))

# Optional: BH adjust the Bayesian two-sided sign probs across hap terms only
m <- hap_mask(fx_brm_ga$term, var = "MainHap")
fx_brm_ga$padj_signprob <- NA_real_
fx_brm_ga$padj_signprob[m] <- p.adjust(fx_brm_ga$p_two[m], method = "BH")

# 5) Write it out
readr::write_csv(fx_brm_ga, file.path(OUTDIR, "ga_brm_posterior_probs.csv"))
































# 2) Define prior settings to try
prior_grid <- list(
  shrink_05 = make_pri_ptb(covariates, hap_names, sd_hap = 0.5),
  shrink_10 = make_pri_ptb(covariates, hap_names, sd_hap = 1.0),
  wide_25   = make_pri_ptb(covariates, hap_names, sd_hap = 2.5),
  brms_default       = c()
)

# 3) Fit under each prior (same model structure)
fit_under_prior <- function(pr) {
  brm(
    as.formula(paste("PTB ~ MainHap +", covariates)),
    data = df, family = bernoulli(),
    prior = pr,
    chains = 2, iter = 3000, warmup = 1000, cores = 2,
    control = ctrl_ptb ,
    init = 0, seed = 2025
  )
}

fits <- lapply(prior_grid, fit_under_prior)


# Save sensitivity fits immediately in case of downstream crash
for (nm in names(fits)) {
  saveRDS(fits[[nm]], file.path(OUTDIR, paste0("ptb_brm_sensitivity_", nm, ".rds")))
}

# 4) Extract per-hap results
summarize_haps <- function(fit, label) {
  fx   <- as.data.frame(summary(fit)$fixed) %>% rownames_to_column("term")
  fx_h <- fx %>% filter(grepl("^MainHap", term)) %>%
    transmute(term,
              OR    = exp(Estimate),
              OR_lo = exp(`l-95% CI`),
              OR_hi = exp(`u-95% CI`))

  draws <- as_draws_df(fit)
  hap_cols <- grep("^b_MainHap", names(draws), value = TRUE)
  post <- lapply(hap_cols, function(nm) {
    s <- as.numeric(draws[[nm]])                 # log-OR draws
    tibble(term = sub("^b_", "", nm),
           Pr_OR_gt_1 = mean(exp(s) > 1),
           p_two      = 2 * pmin(mean(s > 0), mean(s < 0)))
  }) %>% bind_rows()

  left_join(fx_h, post, by = "term") %>%
    mutate(prior_setting = label)
}



                  
results <- bind_rows(
  summarize_haps(fits$shrink_05, "Normal(0,0.5)"),
  summarize_haps(fits$shrink_10, "Normal(0,1.0)"),
  summarize_haps(fits$wide_25,   "Normal(0,2.5)"),
  summarize_haps(fits$brms_default,      "brms_default")
) %>%
  mutate(label = sub("^MainHap\\[T\\.(.+)\\]$", "\\1", sub("^MainHap", "", term))) %>%
  select(prior_setting, label, OR, OR_lo, OR_hi, Pr_OR_gt_1, p_two) %>%
  arrange(prior_setting, label)

# Optional: BH across hap terms within each prior setting
results <- results %>%
  group_by(prior_setting) %>%
  mutate(padj = p.adjust(p_two, method = "BH")) %>%
  ungroup()

readr::write_csv(results, file.path(OUTDIR, "ptb_brm_prior_sensitivity_haps.csv"))



ptb_brm_final <- brm(as.formula(paste("PTB ~ MainHap +", covariates)), 
              data=df, family=bernoulli(),
              prior=pri_ptb,
              chains=2, iter=3000, warmup=1000,
              control = ctrl_ptb 


fx_RE <- as.data.frame(summary(ptb_brm_final)$fixed)
write_csv(fx_RE, file.path(OUTDIR, "ptb_brm_final_fixed_effects.csv"))

sink(file.path(OUTDIR, "ptb_brm_summary.txt"))
print(summary(ptb_brm_final))   # or your final PTB brms object name
sink()


#Diagnostics:
save_brms_diagnostics(brm_ga, "ga_brm", OUTDIR)
save_brms_diagnostics(ptb_brm_final, "ptb_brm_final", OUTDIR)


saveRDS(ptb_brm_final, file.path(OUTDIR, "ptb_brm_final.rds"))
                  
###What to do if convergence is bad? 
##Increase adapt_delta control = list(adapt_delta = 0.999, max_treedepth = 15)
# Increase iterations chains = 4, iter = 6000, warmup = 2000
# Use slightly stronger priors




#Save PTB sparsity counts by haplogroup
# ---- PTB counts by haplogroup ----
hap_ptb_counts <- df %>%
  group_by(MainHap) %>%
  summarise(
    n_total = n(),
    n_ptb   = sum(PTB == 1, na.rm = TRUE),
    n_term  = sum(PTB == 0, na.rm = TRUE),
    n_sites = n_distinct(site),
    .groups = "drop"
  ) %>%
  arrange(desc(n_total))

write_csv(hap_ptb_counts, file.path(OUTDIR, "hap_ptb_counts.csv"))
print(hap_ptb_counts)




# ---------------------------------
# Site-level summaries for model covariates only
# ---------------------------------



# Split covariate formula string into individual variable names
covariate_vars <- covariates %>%
  strsplit("\\+") %>%
  unlist() %>%
  trimws()

# Remove random effect syntax if present, e.g. "(1 | site)"
covariate_vars <- covariate_vars[!grepl("\\|", covariate_vars)]

# Keep only real column names
covariate_vars <- covariate_vars[covariate_vars %in% names(df_raw)]

# Identify which selected covariates are continuous/binary/categorical
summary_cont <- intersect(covariate_vars, columnCont)
summary_bin  <- intersect(covariate_vars, columnBin)
summary_cat  <- intersect(covariate_vars, columnCat)

site_summary <- df_raw %>%
  group_by(site) %>%
  summarise(
    n_total  = n(),
    n_ptb    = sum(PTB == 1, na.rm = TRUE),
    ptb_rate = mean(PTB == 1, na.rm = TRUE),

    mean_ga  = mean(GAGEBRTH, na.rm = TRUE),
    sd_ga    = sd(GAGEBRTH, na.rm = TRUE),

    across(
      all_of(summary_cont),
      list(
        mean = ~ mean(.x, na.rm = TRUE),
        sd   = ~ sd(.x, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    ),

    across(
      all_of(summary_bin),
      list(
        n_yes = ~ sum(.x == 1, na.rm = TRUE),
        prop_yes = ~ mean(.x == 1, na.rm = TRUE)
      ),
      .names = "{.col}_{.fn}"
    ),

    .groups = "drop"
  )

write_csv(site_summary, file.path(OUTDIR, "site_summary.csv"))
print(site_summary)


site_cat_summary <- df_raw %>%
  select(site, all_of(summary_cat)) %>%
  pivot_longer(-site, names_to = "variable", values_to = "value") %>%
  group_by(site, variable, value) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(site, variable) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()

write_csv(site_cat_summary, file.path(OUTDIR, "site_categorical_summary.csv"))


# ============================
# PTB sparsity / site structure table
# ============================

hap_site_ptb_table <- df %>%
  group_by(MainHap, site) %>%
  summarise(
    n_total  = n(),
    n_ptb    = sum(PTB == 1, na.rm = TRUE),
    n_term   = sum(PTB == 0, na.rm = TRUE),
    ptb_rate = mean(PTB == 1, na.rm = TRUE),
    .groups  = "drop"
  ) %>%
  arrange(MainHap, site)

write_csv(hap_site_ptb_table,
          file.path(OUTDIR, "hap_site_ptb_table.csv"))

print(hap_site_ptb_table)

# ---- Flag sparse / problematic cells ----
hap_site_ptb_flags <- hap_site_ptb_table %>%
  mutate(
    zero_ptb    = n_ptb == 0,
    zero_term   = n_term == 0,
    sparse_cell = n_total < 5,
    low_events  = n_ptb < 2
  )

write_csv(hap_site_ptb_flags,
          file.path(OUTDIR, "hap_site_ptb_flags.csv"))

problem_cells <- hap_site_ptb_flags %>%
  filter(zero_ptb | zero_term | sparse_cell | low_events)

write_csv(problem_cells,
          file.path(OUTDIR, "hap_site_ptb_problem_cells.csv"))

print(problem_cells)
