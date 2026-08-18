"""Trial-level logistic regression: success ~ sym_score + severity + method.

One row per registration of the symmetry sweep (symmetry_sweep.csv), with each
part's continuous symmetry score from symmetry_scores.csv. Reports the
coefficients, a bootstrap 95% CI for the sym_score effect, McFadden pseudo-R^2
and AUC with and without sym_score, and the part-level Spearman rho.

    regbench regression

Output (tables_dir): sym_regression.md.
"""
from __future__ import annotations

import argparse
import os

# single-threaded BLAS is faster for many small fits; set before numpy is imported
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from regbench.config import REPO_ROOT, load_config
from regbench.mdtable import format_table
from regbench.repro import stable_rng

N_BOOT = 1000


def design_matrix(df: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """Columns: sym_score_z, severity dummies (vs easy), method dummies (vs icp_pt2pt)."""
    sym_z = StandardScaler().fit_transform(df[["sym_score"]].to_numpy())
    sev_order = [s for s in cfg["severities"] if s != "easy"]
    sev = np.column_stack([(df["severity"] == s).to_numpy(float) for s in sev_order])
    methods = [m for m in cfg["methods"] if m != "icp_pt2pt"]
    meth = np.column_stack([(df["method"] == m).to_numpy(float) for m in methods])
    X = np.hstack([sym_z, sev, meth])
    names = ["sym_score_z", *[f"severity[{s}]" for s in sev_order],
             *[f"method[{m}]" for m in methods]]
    return X, names


def fit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(C=np.inf, max_iter=5000)  # unpenalized MLE
    clf.fit(X, y)
    return clf


def mcfadden_r2(X: np.ndarray, y: np.ndarray, drop_col: int | None = None) -> float:
    """1 - loglik(model)/loglik(intercept-only null)."""
    Xm = X if drop_col is None else np.delete(X, drop_col, axis=1)
    p_null = np.full(len(y), y.mean())
    ll_null = -log_loss(y, p_null, labels=[0, 1]) * len(y)
    clf = fit(Xm, y)
    p = clf.predict_proba(Xm)[:, 1]
    ll_model = -log_loss(y, p, labels=[0, 1]) * len(y)
    return float(1.0 - ll_model / ll_null)


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser().parse_args(argv)  # no flags; enables `--help`
    cfg = load_config()
    tables = REPO_ROOT / cfg["paths"]["tables_dir"]
    sweep_csv, scores_csv = tables / "symmetry_sweep.csv", tables / "symmetry_scores.csv"
    if not sweep_csv.exists() or not scores_csv.exists():
        raise SystemExit("run `regbench sweep-symmetry` first")
    sweep = pd.read_csv(sweep_csv)
    scores = pd.read_csv(scores_csv)
    df = sweep.merge(scores[["part", "sym_score"]], on="part", how="left")

    X, names = design_matrix(df, cfg)
    y = df["success"].to_numpy(int)
    clf = fit(X, y)
    coef = dict(zip(names, clf.coef_[0]))

    # bootstrap CI for the sym_score coefficient (rows resampled with replacement)
    rng = stable_rng("sym_regression", "bootstrap")
    boot_coefs = np.empty(N_BOOT)
    n = len(y)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot_coefs[b] = fit(X[idx], y[idx]).coef_[0][0]
    lo, hi = np.percentile(boot_coefs, [2.5, 97.5])

    r2_full = mcfadden_r2(X, y)
    r2_no_sym = mcfadden_r2(X, y, drop_col=0)
    auc_full = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    clf_no_sym = fit(np.delete(X, 0, axis=1), y)
    auc_no_sym = roc_auc_score(y, clf_no_sym.predict_proba(np.delete(X, 0, axis=1))[:, 1])

    # part-level Spearman rho
    succ_by_part = df.groupby("part", sort=False)["success"].mean()
    score_by_part = scores.set_index("part")["sym_score"].reindex(succ_by_part.index)
    rho = succ_by_part.rank().corr(score_by_part.rank())

    sym_significant = lo > 0 or hi < 0
    n_parts = len(succ_by_part)
    coef_table = format_table(
        ["term", "coefficient", "direction"],
        [[k, f"{v:+.4f}", "higher success" if v > 0 else "lower success"]
         for k, v in coef.items()])
    fit_table = format_table(
        ["model", "pseudo-R^2", "AUC"],
        [["full (+ sym_score)", f"{r2_full:.4f}", f"{auc_full:.4f}"],
         ["severity + method only (no sym_score)", f"{r2_no_sym:.4f}", f"{auc_no_sym:.4f}"]])
    lines = [
        "# Continuous regression: success ~ sym_score + severity + method", "",
        f"n = {n} trials (one row per part x method x severity x trial), "
        "logistic regression with severity and method as covariates.", "",
        "## Logistic regression coefficients (full model)", "",
        coef_table, "",
        "sym_score is z-scored (low = more symmetric); a positive coefficient "
        "means less symmetric parts register successfully more often.", "",
        f"## sym_score effect: bootstrap 95% CI ({N_BOOT} resamples)", "",
        f"coefficient = {coef['sym_score_z']:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] - "
        f"{'excludes zero' if sym_significant else 'includes zero'}",
        "",
        "## Model fit (McFadden pseudo-R^2, AUC)", "",
        fit_table, "",
        f"sym_score's marginal contribution: "
        f"delta pseudo-R^2 = {r2_full - r2_no_sym:+.4f}, delta AUC = "
        f"{auc_full - auc_no_sym:+.4f}.", "",
        "## Part-level rank correlation", "",
        f"Spearman rho (sym_score vs. per-part success rate, n={n_parts} parts) = "
        f"{rho:.3f}", ""]
    (tables / "sym_regression.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
