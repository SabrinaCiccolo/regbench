"""Matplotlib style shared by every figure.

Figure sizes are the printed size in inches (6.3 in = A4 text width), so the
font sizes below are the printed point sizes.
"""
from __future__ import annotations

import logging

import matplotlib as mpl
from matplotlib.axes import Axes

WIDTH_FULL = 6.3
FIGSIZE_1PANEL = (WIDTH_FULL, 3.4)
FIGSIZE_2PANEL = (WIDTH_FULL, 2.9)
DPI = 300

NAVY = "#1F3A5F"
BLUE = "#2E5E9E"
ORANGE = "#ED7D31"
GREEN = "#2E7D32"
RED = "#C0392B"
GREY = "#A6A6A6"
TEXT = "#23314A"
GRID = "#E5E7EB"
CHANCE = "#9AA3AF"

METHOD_COLORS = {"icp_pt2pt": BLUE, "icp_pt2pl": NAVY, "fpfh_ransac": ORANGE, "fgr": GREEN}
METHOD_MARKERS = {"icp_pt2pt": "o", "icp_pt2pl": "s", "fpfh_ransac": "D", "fgr": "^"}

FONT_STACK = ["Roboto", "Montserrat", "Ubuntu Sans", "DejaVu Sans"]


def apply() -> None:
    """Set the shared rcParams; call before creating a figure."""
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    mpl.rcParams.update({
        "font.family": FONT_STACK,
        "font.size": 9,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.labelsize": 9,
        "axes.titlecolor": TEXT,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.edgecolor": "#C7CDD6",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelcolor": TEXT,
        "ytick.labelcolor": TEXT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "savefig.dpi": DPI,
    })


def style_success_axis(ax: Axes, *, xlabel: str, ylabel: str | None = None,
                       title: str | None = None, xticks=None) -> None:
    """Axis dressing for success-rate plots: y-range (-3, 103) plus labels.

    ``xticks`` is either a list of positions or a ``(positions, labels)`` tuple.
    """
    ax.set_xlabel(xlabel)
    ax.set_ylim(-3, 103)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if xticks is not None:
        if isinstance(xticks, tuple):
            ax.set_xticks(xticks[0], xticks[1])
        else:
            ax.set_xticks(xticks)
