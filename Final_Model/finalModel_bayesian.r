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

# Choose a default reference for the Joint cohort
DEFAULT_REF <- "L3"  # set to "M" if you prefer; script will fall back if absent

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

# Targeted hap priors: skeptical on MainHap only; mild on others; mildly informative RE SDs
make_priors_ptb <- function(dframe) {
  ref <- levels(dframe$MainHap)[1]
  hap_lvls <- setdiff(levels(dframe$MainHap), ref)
  c(
    lapply(hap_lvls, function(h) prior(normal(0, 0.5), class="b", coef=paste0("MainHap", h))),
    prior(normal(0, 1), class="b"),                    # for BMI_s, AGE_s
    prior(student_t(3, 0, 2.5), class="sd")
  )
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
ga_tmb <- glmmTMB(GAGEBRTH ~ MainHap + BMI_s + AGE_s + (1|site),
                  data = df, family = gaussian())
tidy_ga <- broom.mixed::tidy(ga_tmb, effects = "fixed", conf.int = TRUE) %>% bh_on_hap()
write_csv(tidy_ga, file.path(OUTDIR, "ga_glmmtmb_site_fixed.csv"))

# Diagnostics: DHARMa (GA)
png(file.path(OUTDIR, "ga_glmmtmb_DHARMa.png"), width=1200, height=900)
plot(simulateResiduals(ga_tmb))
dev.off()

# PTB (Binomial logit), site random intercept
ptb_tmb <- glmmTMB(PTB ~ MainHap + BMI_s + AGE_s + (1|site),
                   data = df, family = binomial())
fx_ptb  <- broom.mixed::tidy(ptb_tmb, effects = "fixed", conf.int = TRUE) %>%
  bh_on_hap() %>% to_or()
write_csv(fx_ptb, file.path(OUTDIR, "ptb_glmmtmb_site_fixed.csv"))
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
  GAGEBRTH_s ~ MainHap + BMI_s + AGE_s + (1|site),
  data = df, family = student(),
  prior = pri_ga,
  chains = 4, iter = 4000, cores = 4,
  control = ctrl_ga, inits = 0, seed = 2025
)
sink(file.path(OUTDIR, "ga_brm_summary.txt")); print(summary(brm_ga)); sink()

sd_ga <- sd(df$GAGEBRTH, na.rm = TRUE)
fx_brm_ga <- as.data.frame(summary(brm_ga)$fixed) %>%
  tibble::rownames_to_column("term") %>%
  bh_on_hap_wald(term_col="term", mean_col="Estimate", se_col="Est.Error") %>%
  mutate(beta_days = Estimate * sd_ga,
         lo_days   = `l-95% CI` * sd_ga,
         hi_days   = `u-95% CI` * sd_ga)
write_csv(fx_brm_ga, file.path(OUTDIR, "ga_brm_site_fixed.csv"))

png(file.path(OUTDIR, "ga_brm_pp_check.png"), width=1200, height=900)
pp_check(brm_ga, ndraws=200)
dev.off()
capture.output(bayes_R2(brm_ga), file = file.path(OUTDIR, "ga_brm_bayesR2.txt"))

# PTB (Bernoulli) with targeted hap priors
pri_ptb <- make_priors_ptb(df)
brm_ptb <- brm(
  PTB ~ MainHap + BMI_s + AGE_s + (1|site),
  data = df, family = bernoulli(),
  prior = pri_ptb,
  chains = 4, iter = 4000, cores = 4,
  control = ctrl_ptb, inits = 0, seed = 2025
)
sink(file.path(OUTDIR, "ptb_brm_summary.txt")); print(summary(brm_ptb)); sink()

fx_brm_ptb <- as.data.frame(summary(brm_ptb)$fixed) %>%
  tibble::rownames_to_column("term") %>%
  bh_on_hap_wald(term_col="term", mean_col="Estimate", se_col="Est.Error") %>%
  mutate(OR     = exp(Estimate),
         OR_low = exp(`l-95% CI`),
         OR_hi  = exp(`u-95% CI`),
         label  = robust_hap_label(term))
write_csv(fx_brm_ptb, file.path(OUTDIR, "ptb_brm_site_fixed.csv"))
save_forest_ptb(fx_brm_ptb, "PTB brms (Joint/All)", file.path(OUTDIR, "ptb_brm_site_forest.png"))

# Posterior-based probabilities Pr(OR>1) per haplogroup
draws <- as_draws_df(brm_ptb)
keep_cols <- grep("^b_MainHap", names(draws), value = TRUE)
if (length(keep_cols)) {
  prob_tbl <- tibble(
    term = keep_cols,
    label = robust_hap_label(sub("^b_", "", keep_cols)),
    Pr_OR_gt_1 = colMeans(exp(as.matrix(draws[, keep_cols, drop=FALSE])) > 1)
  )
  write_csv(prob_tbl, file.path(OUTDIR, "ptb_brm_PrORgt1.csv"))
}

# Posterior predictive check & LOO for PTB
png(file.path(OUTDIR, "ptb_brm_pp_check.png"), width=1200, height=900)
pp_check(brm_ptb, ndraws = 200)
dev.off()
capture.output(loo(brm_ptb), file = file.path(OUTDIR, "ptb_brm_loo.txt"))

cat(sprintf("\n[Joint/All] ref haplogroup = %s; N = %d; sites = %d\nOutputs written under: %s\n",
            ref_name, nrow(df), dplyr::n_distinct(df$site), normalizePath(OUTDIR)))
