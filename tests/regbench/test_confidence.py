import numpy as np
import pandas as pd

from regbench.experiments.confidence import signal_auroc, zero_false_alarm_recall


def _toy_grid() -> pd.DataFrame:
    """8 rows: successes at high fitness, failures low, and one failure inside the
    success band."""
    return pd.DataFrame({
        "success": [True, True, True, True, False, False, False, False],
        "fitness": [1.00, 0.999, 0.998, 1.00, 0.10, 0.30, 0.50, 0.998],
        "inlier_rmse": [0.001, 0.001, 0.002, 0.001, 0.05, 0.04, 0.03, 0.0025],
        "rre_deg": [0.1, 0.1, 0.2, 0.1, 40.0, 35.0, 20.0, 1.9],
    })


def test_fitness_auroc_separates_success_from_failure():
    a, n = signal_auroc(_toy_grid(), "fitness", higher_is_worse=False)
    assert n == 8
    assert 0.5 < a <= 1.0


def test_inlier_rmse_auroc_higher_is_worse():
    a, n = signal_auroc(_toy_grid(), "inlier_rmse", higher_is_worse=True)
    assert n == 8
    assert a == 1.0   # rmse cleanly separates in this toy grid


def test_signal_auroc_skips_nan_rows():
    df = _toy_grid()
    df.loc[0, "fitness"] = np.nan
    a, n = signal_auroc(df, "fitness", higher_is_worse=False)
    assert n == 7


def test_zero_false_alarm_recall_catches_all_but_the_overlapping_failure():
    op = zero_false_alarm_recall(_toy_grid())
    assert op["threshold"] == 0.998   # lowest fitness among the 4 successes
    assert op["total_fail"] == 4
    assert op["caught"] == 3          # the fitness=0.997 failure is missed
    assert len(op["missed_rre"]) == 1
    assert op["missed_rre"][0] == 1.9
