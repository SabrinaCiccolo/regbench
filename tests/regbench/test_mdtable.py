from regbench.mdtable import format_table


def test_format_table_shape():
    text = format_table(["method", "success"], [["icp", 0.9], ["fgr", 0.8]])
    lines = text.splitlines()
    assert lines[0] == "| method | success |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| icp | 0.9 |"
    assert lines[3] == "| fgr | 0.8 |"
