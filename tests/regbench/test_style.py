import matplotlib as mpl

from regbench import style


def test_method_palette_covers_every_method():
    from regbench.config import load_config

    methods = load_config()["methods"]
    assert set(methods) <= set(style.METHOD_COLORS)
    assert set(methods) <= set(style.METHOD_MARKERS)


def test_apply_sets_rcparams():
    style.apply()
    assert mpl.rcParams["font.size"] == 9
    assert mpl.rcParams["savefig.dpi"] == style.DPI
