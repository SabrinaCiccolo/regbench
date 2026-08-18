import numpy as np
import pandas as pd

from regbench.experiments.regression import design_matrix, fit, mcfadden_r2
from regbench.repro import stable_rng

CFG = {"severities": {"easy": {}, "medium": {}, "hard": {}},
       "methods": ["icp_pt2pt", "icp_pt2pl", "fpfh_ransac", "fgr"]}


def _df(n_per_cell=20):
    rng = stable_rng("sym_regression_test", 0)
    rows = []
    for part, sym_score in [("a", 0.01), ("b", 0.05), ("c", 0.09)]:
        for severity in CFG["severities"]:
            for method in CFG["methods"]:
                for _ in range(n_per_cell):
                    rows.append({"part": part, "sym_score": sym_score,
                                "severity": severity, "method": method})
    df = pd.DataFrame(rows)
    # success probability rises with sym_score, falls with severity - a known
    # ground truth the regression should recover the sign of.
    sev_penalty = df["severity"].map({"easy": 0.0, "medium": 1.5, "hard": 3.5})
    logit = 40 * (df["sym_score"] - 0.05) - sev_penalty
    p = 1 / (1 + np.exp(-logit))
    df["success"] = (rng.uniform(size=len(df)) < p).astype(int)
    return df


def test_design_matrix_shape_and_columns():
    df = _df(n_per_cell=2)
    X, names = design_matrix(df, CFG)
    assert X.shape == (len(df), 6)  # sym_score_z + 2 severity dummies + 3 method dummies
    assert names[0] == "sym_score_z"
    assert "severity[medium]" in names and "severity[hard]" in names
    assert "easy" not in " ".join(names)          # baseline level, not a column
    assert "icp_pt2pt" not in " ".join(names)     # baseline level, not a column


def test_fit_recovers_known_sym_score_direction():
    df = _df(n_per_cell=40)
    X, names = design_matrix(df, CFG)
    y = df["success"].to_numpy(int)
    clf = fit(X, y)
    assert clf.coef_[0][names.index("sym_score_z")] > 0.5  # positive & sizeable


def test_mcfadden_r2_full_model_beats_dropping_the_signal_column():
    df = _df(n_per_cell=40)
    X, names = design_matrix(df, CFG)
    y = df["success"].to_numpy(int)
    r2_full = mcfadden_r2(X, y)
    r2_no_sym = mcfadden_r2(X, y, drop_col=names.index("sym_score_z"))
    assert 0.0 <= r2_no_sym < r2_full <= 1.0
