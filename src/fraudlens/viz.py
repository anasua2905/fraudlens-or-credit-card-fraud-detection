"""One visual system for every chart in the project (Seaborn/Matplotlib and Plotly).

Colour roles (validated reference palette, light surface):
  * identity  -> categorical slots in fixed order (legit = slot 1 blue, fraud = slot 2 orange;
                 models take slots 1..n in config order)
  * state     -> reserved status colours for alert outcomes, always paired with a text label
  * magnitude -> one blue ramp
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from .palette import *  # noqa: F401,F403  (colour constants)
from .palette import BLUE_RAMP, INK, INK_2, MODEL_ORDER, MUTED, SERIES, SURFACE  # noqa: F401


def model_color(name: str) -> str:
    return SERIES[MODEL_ORDER.index(name) % len(SERIES)] if name in MODEL_ORDER else MUTED


def setup() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "figure.dpi": 110, "lines.linewidth": 2,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "font.family": ["DejaVu Sans"],
    })


def save(fig, path_without_suffix: Path) -> Path:
    out = Path(path_without_suffix).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def baseline(ax, value: float, label: str = "overall") -> None:
    ax.axhline(value, ls="--", lw=1, color=MUTED)
    ax.annotate(f" {label}", xy=(1, value), xycoords=("axes fraction", "data"), va="center",
                fontsize=8, color=MUTED)


def plotly_template():
    """A Plotly layout template matching the static charts."""
    import plotly.graph_objects as go
    return go.layout.Template(layout=go.Layout(
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_2, size=13),
        title=dict(font=dict(color=INK, size=16), x=0, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED)),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED)),
        legend=dict(orientation="h", y=-0.18), hoverlabel=dict(bgcolor="white"),
        margin=dict(t=60, l=60, r=20, b=60)))


setup()
