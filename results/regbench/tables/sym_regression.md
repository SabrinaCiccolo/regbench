# Continuous regression: success ~ sym_score + severity + method

n = 1440 trials (one row per part x method x severity x trial), logistic regression with severity and method as covariates.

## Logistic regression coefficients (full model)

| term | coefficient | direction |
|---|---|---|
| sym_score_z | +0.9861 | higher success |
| severity[medium] | -1.7619 | lower success |
| severity[hard] | -4.1357 | lower success |
| method[icp_pt2pl] | +1.1273 | higher success |
| method[fpfh_ransac] | +0.5175 | higher success |
| method[fgr] | +1.2044 | higher success |

sym_score is z-scored (low = more symmetric); a positive coefficient means less symmetric parts register successfully more often.

## sym_score effect: bootstrap 95% CI (1000 resamples)

coefficient = +0.9861, 95% CI [+0.8405, +1.1759] - excludes zero

## Model fit (McFadden pseudo-R^2, AUC)

| model | pseudo-R^2 | AUC |
|---|---|---|
| full (+ sym_score) | 0.3443 | 0.8744 |
| severity + method only (no sym_score) | 0.2421 | 0.8176 |

sym_score's marginal contribution: delta pseudo-R^2 = +0.1021, delta AUC = +0.0568.

## Part-level rank correlation

Spearman rho (sym_score vs. per-part success rate, n=6 parts) = 1.000
