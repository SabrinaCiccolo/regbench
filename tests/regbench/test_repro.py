from regbench.repro import cpu_info, stable_rng


def test_stable_rng_deterministic():
    a = stable_rng("foo", 1, "bar")
    b = stable_rng("foo", 1, "bar")
    assert a.integers(0, 1_000_000) == b.integers(0, 1_000_000)


def test_stable_rng_distinguishes_keys():
    a = stable_rng("foo", 1)
    b = stable_rng("foo", 2)
    assert a.integers(0, 10**9) != b.integers(0, 10**9)


def test_cpu_info_format():
    info = cpu_info()
    assert "x" in info
    n = info.split("x", 1)[0]
    assert n.isdigit()
