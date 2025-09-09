    
    
    # 7. Quantile regression with PCs and with site for GA
    for q in quantiles:
        model_qr = smf.quantreg('GAGEBRTH ~ MainHap + BMI + PW_AGE + PC1 + PC2 + PC3', data=df).fit(q=q)
        qsum = model_qr.summary2().tables[1]
        qsum['Quantile'] = q
        qsum['Ref'] = ref
        qsum['Model'] = 'PC'
        qr_results.append(qsum)
        model_qr_site = smf.quantreg('GAGEBRTH ~ MainHap + BMI + PW_AGE + site', data=df).fit(q=q)
        qsum_site = model_qr_site.summary2().tables[1]
        qsum_site['Quantile'] = q
        qsum_site['Ref'] = ref
        qsum_site['Model'] = 'site_only'
        qr_results.append(qsum_site)
        model_qr_site = smf.quantreg('GAGEBRTH ~ MainHap + BMI + PW_AGE + site + PC2 + PC3', data=df).fit(q=q)
        qsum_site = model_qr_site.summary2().tables[1]
        qsum_site['Quantile'] = q
        qsum_site['Ref'] = ref
        qsum_site['Model'] = 'site_AND_PC'
        qr_results.append(qsum_site)

# Save quantile regression summary
pd.concat(qr_results).to_csv("quantile_regression_summary.tsv", sep="\t")










# ============================
# Mixed Models for GA & PTB (no PCs), site random intercept
# Frequentist: glmmTMB for GA (Gaussian) and PTB (Binomial)
# Bayesian:    brms mirrors with robust priors/controls
# Outputs: tidy CSVs + PTB forest plots + EMMs
# ============================

# ---- Packages ----
# install.packages(c(
#  "readr","dplyr","stringr","forcats","ggplot2","broom","broom.mixed",
#  "glmmTMB","emmeans","brms"
# ))
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(stringr); library(forcats)
  library(ggplot2); library(broom); library(broom.mixed)
  library(glmmTMB); library(emmeans); library(brms)
})

# ==== CONFIG ====
primary_ref <- "L3"   # change to "M" for sensitivity if you like
outdir <- "model_outputs"
if (!dir.exists(outdir)) dir.create(outdir)

# ---- Load & preprocess ----
df <- read_tsv("Metadata.Final.tsv", show_col_types = FALSE) %>%
  mutate(
    MainHap = factor(MainHap),
    site    = factor(site)
  )

# Set primary reference (use a large group)
if (!primary_ref %in% levels(df$MainHap)) {
  stop(sprintf("Primary ref '%s' not found in MainHap levels.", primary_ref))
}
df$MainHap <- fct_relevel(df$MainHap, primary_ref)

# Center/scale covariates (stability & interpretable intercepts)
df <- df %>%
  mutate(
    BMI_s = as.numeric(scale(BMI)),
    AGE_s = as.numeric(scale(PW_AGE)),
    # For brms GA: standardized outcome helps geometry
    GAGEBRTH_s = as.numeric(scale(GAGEBRTH))
  )

# Optional warning for tiny reference
n_ref <- sum(df$MainHap == levels(df$MainHap)[1])
if (n_ref < 100) message(sprintf("Warning: reference '%s' has n=%d", levels(df$MainHap)[1], n_ref))

# ---- Helpers ----
hap_mask <- function(terms, var = "MainHap") grepl(paste0("^", var), terms)

# BH across haplogroup rows for frequentist (needs p.value present)
bh_on_hap <- function(tbl, term_col = "term", p_col = "p.value", var = "MainHap") {
  if (!p_col %in% names(tbl)) {
    stop("bh_on_hap: table has no 'p.value' column; use bh_on_hap_wald() for brms summaries.")
  }
  m <- hap_mask(tbl[[term_col]], var)
  q <- rep(NA_real_, nrow(tbl))
  if (any(m)) q[m] <- p.adjust(tbl[[p_col]][m], method = "BH")
  tbl$padj <- q
  tbl
}

# For brms summaries: compute a Wald-style p from Estimate/SE, then BH across hap rows
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
  tbl %>% mutate(
    OR     = exp(.data[[est]]),
    OR_low = exp(.data[[lo]]),
    OR_hi  = exp(.data[[hi]])
  )
}

save_forest_ptb <- function(tbl, title, file, label_col = "term", or = "OR", lo = "OR_low", hi = "OR_hi") {
  d <- tbl %>% filter(hap_mask(.data[[label_col]]))
  if (!nrow(d)) return(invisible(NULL))
  # friendly labels: "MainHap[T.L3]" -> "L3"
  d$label <- stringr::str_match(d[[label_col]], "MainHap\\[T\\.(.+)\\]")[,2]
  gg <- ggplot(d, aes(x = reorder(label, .data[[or]]), y = .data[[or]])) +
    geom_point(size = 2.6) +
    geom_errorbar(aes(ymin = .data[[lo]], ymax = .data[[hi]]), width = 0.2) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "grey40") +
    coord_flip() + theme_bw(13) +
    labs(title = title, x = "Haplogroup (vs ref)", y = "Odds ratio (PTB)")
  ggsave(file, gg, width = 7, height = 4.5, dpi = 300)
}

# ============================
# FREQUENTIST: glmmTMB for both outcomes
# ============================

## GA (Gaussian), site random intercept
ga_tmb <- glmmTMB(GAGEBRTH ~ MainHap + BMI_s + AGE_s + (1|site),
                  data = df, family = gaussian())
tidy_ga <- broom.mixed::tidy(ga_tmb, effects = "fixed", conf.int = TRUE) %>% bh_on_hap()
write_csv(tidy_ga, file.path(outdir, "ga_glmmtmb_site_fixed.csv"))

## PTB (Binomial logit), site random intercept
ptb_tmb <- glmmTMB(PTB ~ MainHap + BMI_s + AGE_s + (1|site),
                   data = df, family = binomial())
fx_ptb  <- broom.mixed::tidy(ptb_tmb, effects = "fixed", conf.int = TRUE) %>%
  bh_on_hap() %>% to_or()
write_csv(fx_ptb, file.path(outdir, "ptb_glmmtmb_site_fixed.csv"))
save_forest_ptb(fx_ptb, "PTB GLMM (glmmTMB, site RE)",
                file.path(outdir, "ptb_glmmtmb_site_forest.png"))

# Predicted PTB probability by haplogroup (marginal over random site)
emm_ptb <- emmeans(ptb_tmb, ~ MainHap, type = "response")
write.csv(as.data.frame(emm_ptb), file.path(outdir, "ptb_glmmtmb_emmeans_probs.csv"), row.names = FALSE)

# Pairwise haplogroup differences with BH across pairs
pairs_emm <- pairs(emm_ptb, adjust = "BH")
write.csv(as.data.frame(pairs_emm), file.path(outdir, "ptb_glmmtmb_emmeans_pairs_BH.csv"), row.names = FALSE)

# ============================
# BAYESIAN: brms mirrors with robust priors/controls
# ============================

# Priors
# - Slightly tighter fixed-effects prior to help geometry (we standardized covariates & GA).
# - Random-effect SD: mildly informative.
pri_ga <- c(
  prior(normal(0, 0.5), class = "b"),
  prior(student_t(3, 0, 2.5), class = "sd"),
  prior(student_t(3, 0, 2.5), class = "sigma")   # <-- GA only
)
pri_ptb <- c(
  prior(normal(0, 1), class = "b"),   #normal(0, 0.8) (more shrinkage)     normal(0, 1.0) (recommended default)    Flat (no prior on b) (not ideal, but informative as an upper bound) Then present the ORs/HDIs for L2 and M across these. If M stays elevated across sensible priors, your point is stronger. If a result appears only with flat priors, flag it as sensitive to regularization (likely driven by low M).
  prior(student_t(3, 0, 2.5), class = "sd")
)

# Sampler controls — more conservative to kill divergences
ctrl_ga  <- list(adapt_delta = 0.999, max_treedepth = 15)
ctrl_ptb <- list(adapt_delta = 0.99,  max_treedepth = 13)

# (Optional) deterministic inits can also help
inits0 <- 0

## GA (Gaussian) on standardized outcome
brm_ga <- brm(
  GAGEBRTH_s ~ MainHap + BMI_s + AGE_s + (1|site) + (0 + MainHap | site),
  data = df, family = gaussian(),
  prior = pri_ga,
  chains = 4, iter = 4000, cores = 4,
  control = ctrl_ga, inits = inits0
)
print(summary(brm_ga))  # should show 0 divergences

# Extract fixed effects and compute BH (Wald-style p) on haplogroup terms only
fx_brm_ga <- as.data.frame(summary(brm_ga)$fixed) %>%
  tibble::rownames_to_column("term") %>%
  bh_on_hap_wald(term_col="term", mean_col="Estimate", se_col="Est.Error")
readr::write_csv(fx_brm_ga, file.path(outdir, "ga_brm_site_fixed.csv"))

## PTB (Bernoulli) — NOTE: no sigma prior here
brm_ptb <- brm(
  PTB ~ MainHap + BMI_s + AGE_s + (1|site) + (0 + MainHap || site),
  data = df, family = bernoulli(),
  prior = pri_ptb,
  chains = 4, iter = 4000, cores = 4,
  control = ctrl_ptb, inits = inits0
)
print(summary(brm_ptb))

fx_brm_ptb <- as.data.frame(summary(brm_ptb)$fixed) %>%
  tibble::rownames_to_column("term") %>%
  bh_on_hap_wald(term_col="term", mean_col="Estimate", se_col="Est.Error") %>%
  dplyr::mutate(OR = exp(Estimate),
                OR_low = exp(`l-95% CI`),
                OR_hi  = exp(`u-95% CI`))
readr::write_csv(fx_brm_ptb, file.path(outdir, "ptb_brm_site_fixed.csv"))
save_forest_ptb(fx_brm_ptb, "PTB brms (site RE)",
                file.path(outdir, "ptb_brm_site_forest.png"))
# ---- Notes / footer ----
cat("\nDone.\nPrimary models: glmmTMB with (1|site) for GA & PTB.\n",
    "Bayesian mirrors with brms; GA standardized to aid geometry.\n",
    "BH/FDR applied ONLY to MainHap contrasts (frequentist p, brms Wald p).\n",
    "Reference haplogroup:", levels(df$MainHap)[1], sprintf("(n=%d)\n", n_ref))
















library(brms); library(dplyr); library(tibble)

# Base formula
form_ptb <- bf(PTB ~ MainHap + BMI_s + AGE_s + (1|site))

# Shared sampler controls
ctrl <- list(adapt_delta = 0.99, max_treedepth = 13)

# Three prior settings
priors <- list(
  shrink_08 = c(prior(normal(0, 0.8), class = "b"),
                prior(student_t(3, 0, 2.5), class = "sd")),
  shrink_10 = c(prior(normal(0, 1.0), class = "b"),
                prior(student_t(3, 0, 2.5), class = "sd")),
  flat      = c(prior(student_t(3, 0, 2.5), class = "sd")) # no prior on b
)

# Fit all three
fits <- lapply(priors, function(pr) {
  brm(form_ptb, data = df, family = bernoulli(),
      prior = pr, chains = 4, iter = 2000, cores = 4, control = ctrl)
})

# Extract ORs for haplogroup effects
extract_or <- function(fit) {
  as.data.frame(summary(fit)$fixed) %>%
    rownames_to_column("term") %>%
    filter(grepl("^MainHap", term)) %>%
    mutate(OR = exp(Estimate),
           OR_low = exp(`l-95% CI`),
           OR_hi  = exp(`u-95% CI`))
}

results <- lapply(fits, extract_or)
names(results) <- names(priors)

# Example: print results for M vs L3 under each prior
lapply(results, function(tab) tab[tab$term=="MainHapM", ])
