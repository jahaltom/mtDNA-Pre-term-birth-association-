# ============================
# Joint Cohort (All sites pooled): GA & PTB models
# - No PCs (primary)
# - Random intercept for site: (1 | site)
# - GA: Student-t; back-transform to days
# - PTB: Binomial; forest plot, EMMs, Pr(OR>1)
# ============================

suppressPackageStartupMessages({
  library(readr); library(dplyr); library(stringr); library(forcats)
  library(ggplot2); library(broom); library(broom.mixed)
  library(glmmTMB); library(emmeans); library(brms)
  library(DHARMa); library(loo); library(posterior); library(tidyr)
})

set.seed(2025)

# ==== CONFIG ====
INFILE <- "Metadata.Final.tsv"
OUTDIR <- file.path("model_outputs", "All")
if (!dir.exists(OUTDIR)) dir.create(OUTDIR, recursive = TRUE)


covariates <- "BMI_s + AGE_s + (1|site)"
#covariates <- "BMI_s + AGE_s + PC1 + PC2 + PC3 + PC4 +PC5"

# Choose a default reference for the Joint cohort
DEFAULT_REF <- "R"  # set to "M" if you prefer; script will fall back if absent

# ---- Load & preprocess ----
df <- read_tsv(INFILE, show_col_types = FALSE) %>%
  mutate(
    MainHap     = factor(MainHap),
    site        = factor(site),
    BMI_s       = as.numeric(scale(BMI)),
    AGE_s       = as.numeric(scale(PW_AGE)),
    GAGEBRTH_s  = as.numeric(scale(GAGEBRTH))
  )

# Ensure reference haplogroup is present; otherwise pick the most frequent
if (!(DEFAULT_REF %in% levels(df$MainHap))) {
  message(sprintf("Note: default ref '%s' not found; using most frequent MainHap as ref.", DEFAULT_REF))
  fallback <- names(sort(table(df$MainHap), decreasing = TRUE))[1]
  df$MainHap <- fct_relevel(df$MainHap, fallback)
} else {
  df$MainHap <- fct_relevel(df$MainHap, DEFAULT_REF)
}
ref_name <- levels(df$MainHap)[1]
n_ref <- sum(df$MainHap == ref_name, na.rm = TRUE)
if (n_ref < 100) message(sprintf("Warning: reference '%s' has n=%d", ref_name, n_ref))

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


pri_ga <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(student_t(3, 0, 2.5), class = "sd"),
  prior(student_t(3, 0, 2.5), class = "sigma")
)

ctrl_ga  <- list(adapt_delta = 0.999, max_treedepth = 15)
ctrl_ptb <- list(adapt_delta = 0.99,  max_treedepth = 13)

# ============================
# Frequentist: glmmTMB
# ============================

# GA (Gaussian), site random intercept
ga_tmb <- glmmTMB(as.formula(paste("GAGEBRTH ~ MainHap +", covariates)),    
                  data = df, family = gaussian())
tidy_ga <- broom.mixed::tidy(ga_tmb, effects = "fixed", conf.int = TRUE) %>% bh_on_hap()
write_csv(tidy_ga, file.path(OUTDIR, "ga_glmmtmb.csv"))

# Diagnostics: DHARMa (GA)
png(file.path(OUTDIR, "ga_glmmtmb_DHARMa.png"), width=1200, height=900)
plot(simulateResiduals(ga_tmb))
dev.off()

# PTB (Binomial logit), site random intercept
ptb_tmb <- glmmTMB(as.formula(paste("PTB ~ MainHap +", covariates)),
                   data = df, family = binomial())
fx_ptb  <- broom.mixed::tidy(ptb_tmb, effects = "fixed", conf.int = TRUE) %>%
  bh_on_hap() %>% to_or()
write_csv(fx_ptb, file.path(OUTDIR, "ptb_glmmtmb.csv"))
save_forest_ptb(fx_ptb, "PTB GLMM (glmmTMB, Joint/All)", file.path(OUTDIR, "ptb_glmmtmb_site_forest.png"))

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
  as.formula(paste("GAGEBRTH_s ~ MainHap +", covariates)), 
  data = df, family = student(),
  prior = pri_ga,
  chains = 4, iter = 4000, cores = 4,
  control = ctrl_ga, inits = 0, seed = 2025
)
sink(file.path(OUTDIR, "ga_brm_summary.txt")); print(summary(brm_ga)); sink()

sd_ga <- sd(df$GAGEBRTH, na.rm = TRUE)
fx_brm_ga <- as.data.frame(summary(brm_ga)$fixed) %>%
  tibble::rownames_to_column("term") %>%
 # bh_on_hap_wald(term_col="term", mean_col="Estimate", se_col="Est.Error") %>%
  mutate(beta_days = Estimate * sd_ga,
         lo_days   = `l-95% CI` * sd_ga,
         hi_days   = `u-95% CI` * sd_ga)
write_csv(fx_brm_ga, file.path(OUTDIR, "ga_brm.csv"))

png(file.path(OUTDIR, "ga_brm_pp_check.png"), width=1200, height=900)
pp_check(brm_ga, ndraws=200)
dev.off()
capture.output(bayes_R2(brm_ga), file = file.path(OUTDIR, "ga_brm_bayesR2.txt"))





# --- Posterior probabilities for GA (Student-t model) ---
# Assumes: brm_ga already fit; df in memory; your robust_hap_label() & hap_mask() exist.

library(posterior); library(dplyr); library(tibble)

sd_ga <- sd(df$GAGEBRTH, na.rm = TRUE)

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




























library(dplyr); library(tibble); library(posterior)

# 0) From your earlier step:
tmp_ptb <- brm(as.formula(paste("PTB ~ MainHap +", covariates)), data=df, family=bernoulli(),  prior=NULL, chains=1, iter=1000, warmup=500, control=ctrl_ptb, init=0, seed=2025)  
coef_names <- rownames(as.data.frame(summary(tmp_ptb)$fixed))
hap_names  <- coef_names[grepl("^MainHap", coef_names)]

# 1) Helper to build hap-only priors with a given SD (use prior_string to avoid NSE)
make_hap_priors <- function(hap_names, sd_hap = 0.5) {
  pri_list <- lapply(hap_names, function(nm)
    prior_string(sprintf("normal(0, %g)", sd_hap), class = "b", coef = nm)
  )
  do.call(c, pri_list)
}

# 2) Define prior settings to try
prior_grid <- list(
  shrink_05 = make_hap_priors(hap_names, sd_hap = 0.5),   # your current skeptical prior
  shrink_10 = make_hap_priors(hap_names, sd_hap = 1.0),   # milder
  wide_25   = make_hap_priors(hap_names, sd_hap = 2.5),   # weakly-informative
  flat      = NULL                                        # no hap prior (brms default for b’s)
)

# 3) Fit under each prior (same model structure)
fit_under_prior <- function(pr) {
  brm(
    as.formula(paste("PTB ~ MainHap +", covariates)),
    data = df, family = bernoulli(),
    prior = pr,
    chains = 2, iter = 3000, warmup = 1000, cores = 2,
    control = modifyList(ctrl_ptb, list(adapt_delta = 0.995)),
    init = 0, seed = 2025
  )
}

fits <- lapply(prior_grid, fit_under_prior)

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
  summarize_haps(fits$flat,      "flat")
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

hap_prior_mild <- make_hap_priors(hap_names, sd_hap = 1.0)

ptb_FE <- brm(PTB ~ MainHap + BMI_s + AGE_s + site,
              data=df, family=bernoulli(),
              prior=hap_prior_mild,
              chains=2, iter=3000, warmup=1000,
              control=list(adapt_delta=0.995), init=0, seed=2025)


write_csv(ptb_FE, file.path(OUTDIR, "ptb_brm_sensitivity_SiteFixed.csv"))

ptb_RE <- brm(PTB ~ MainHap + BMI_s + AGE_s + (1|site),
              data=df, family=bernoulli(),
              prior=hap_prior_mild,
              chains=2, iter=3000, warmup=1000,
              control=list(adapt_delta=0.995), init=0, seed=2025)


write_csv(ptb_RE, file.path(OUTDIR, "ptb_brm_sensitivity_SiteRandom.csv"))

