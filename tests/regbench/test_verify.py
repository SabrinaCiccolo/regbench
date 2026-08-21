import pandas as pd

from regbench.cli import COMMANDS
from regbench.verify import INVENTORY, RESULTS, check_exact, check_wilson, intervals_overlap


def test_inventory_covers_committed_results():
    committed = {p.relative_to(RESULTS).as_posix()
                 for p in RESULTS.rglob("*") if p.is_file()}
    assert committed == {row[0] for row in INVENTORY}
    for _, command, _, _ in INVENTORY:
        assert command.split()[0] in COMMANDS


def test_intervals_overlap():
    assert intervals_overlap((10, 20), (12, 20))
    assert not intervals_overlap((0, 50), (50, 50))


def _grid(successes: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"method": "fgr", "severity": "easy", "success": successes})


def test_check_wilson(tmp_path):
    base, near, far = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    _grid([True] * 18 + [False] * 2).to_csv(base, index=False)
    _grid([True] * 17 + [False] * 3).to_csv(near, index=False)
    _grid([False] * 20).to_csv(far, index=False)
    assert check_wilson(base, near, ["method", "severity"])[0]
    assert not check_wilson(base, far, ["method", "severity"])[0]


def test_check_exact_csv_tolerates_float_formatting(tmp_path):
    a, b, c = tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"
    a.write_text("x,time_s\n0.01754835077586262,1.0\n")
    b.write_text("x,time_s\n0.0175483507758626,2.0\n")
    c.write_text("x,time_s\n0.0176,1.0\n")
    assert check_exact(a, b)[0]
    assert not check_exact(a, c)[0]
