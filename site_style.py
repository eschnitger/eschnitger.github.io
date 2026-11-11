"""
Shared chart style for eschnitger.github.io.

Every project imports this module so charts share one palette, one typeface
and one set of conventions. Do not set colours or fonts inside a project;
change them here and every chart follows.

In a notebook or script anywhere inside the repo:

    import sys, pathlib
    sys.path.insert(0, str(next(p for p in pathlib.Path.cwd().resolve().parents
                                if (p / "_quarto.yml").exists())))
    import site_style as ss
    ss.apply()

Conventions
    - Brick is the lead series; indigo is the comparison. Everything else is grey.
    - No legends where a direct label will do (ss.end_label, ss.axis_heading).
    - Horizontal gridlines only; no top, right or left spines.
    - Title states the takeaway, left-aligned. A source note sits underneath.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ------------------------------------------------------------------ tokens
# Kept identical to the site's CSS variables.
PAPER = "#faf9f5"      # page background
INK = "#23211b"        # body text, titles
MUTED = "#6d6a60"      # axis labels, tick labels, notes
FAINT = "#a8a49a"      # de-emphasised series, minor annotations
GRID = "#e9e6de"       # gridlines
RULE = "#c9c5ba"       # baseline / bottom spine
BRICK = "#a63d2f"      # accent: the series the chart is about
INDIGO = "#33548c"     # comparison series

# Use in this order. More than two highlighted series is usually a sign the
# chart wants small multiples; grey out the rest with FAINT.
SERIES = [BRICK, INDIGO, "#7a7466", "#b98a2f", "#4f7a6b"]

FONT_FAMILY = "Satoshi"
FONT_FALLBACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"


# ------------------------------------------------------------------ setup
def _register_fonts():
    """Register the bundled fonts and return the font list to use.

    Files are named <Family>-<Weight>.ttf. The family is taken from the file
    name because some builds of Satoshi ship with a broken internal name.
    """
    import dataclasses

    known = {(f.name, f.fname) for f in font_manager.fontManager.ttflist}
    for ttf in sorted(FONT_DIR.glob("*.ttf")) + sorted(FONT_DIR.glob("*.otf")):
        entry = font_manager.ttfFontProperty(font_manager.get_font(str(ttf)))
        entry = dataclasses.replace(entry, name=ttf.stem.split("-")[0])
        if (entry.name, entry.fname) not in known:
            font_manager.fontManager.ttflist.append(entry)
    available = {f.name for f in font_manager.fontManager.ttflist}
    if FONT_FAMILY in available:
        return [FONT_FAMILY]
    return [f for f in FONT_FALLBACK if f in available]


def apply(transparent=True, inline_svg=True):
    """Set the site style as the matplotlib default. Call once, after imports.

    transparent  figures have no background, so they sit on the page colour.
                 Pass False to paint them with PAPER (e.g. for thumbnails).
    inline_svg   in Jupyter, emit SVG instead of PNG so charts stay sharp.
    """
    fonts = _register_fonts()
    bg = "none" if transparent else PAPER
    mpl.rcParams.update({
        "font.family": fonts or ["sans-serif"],
        "font.size": 11,
        "svg.fonttype": "path",            # text as outlines: same on any device
        "text.color": INK,

        "figure.facecolor": bg,
        "figure.dpi": 110,
        "savefig.facecolor": bg,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "savefig.dpi": 220,

        "axes.facecolor": "none",
        "axes.edgecolor": RULE,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "axes.labelcolor": MUTED,
        "axes.labelsize": 11,
        "axes.labelpad": 8,
        "axes.titlesize": 13,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.titlecolor": INK,

        "grid.color": GRID,
        "grid.linewidth": 0.9,

        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,

        "lines.linewidth": 2.2,
        "lines.solid_capstyle": "round",
        "lines.markersize": 5,
        "patch.edgecolor": "none",

        "legend.frameon": False,
        "legend.fontsize": 10.5,
        "legend.labelcolor": MUTED,
    })
    if inline_svg:
        try:
            from matplotlib_inline.backend_inline import set_matplotlib_formats
            set_matplotlib_formats("svg")
        except Exception:
            pass


# ------------------------------------------------------------------ helpers
def _left(fig, x):
    return fig.subplotpars.left if x is None else x


def titles(fig, title, subtitle=None, x=None, y=1.0):
    """Left-aligned figure headline with an optional muted subtitle.

    Sits above the axes, aligned with the plot's left edge. Raise y if the
    axes carry their own titles or headings.
    """
    x = _left(fig, x)
    if subtitle:
        fig.text(x, y, subtitle, ha="left", va="bottom", fontsize=11.5, color=MUTED)
        y += 0.36 / fig.get_figheight()
    fig.text(x, y, title, ha="left", va="bottom", fontsize=17, fontweight="medium",
             color=INK)


def source_note(fig, text, x=None, y=-0.02):
    """Small source / method line under the chart."""
    fig.text(_left(fig, x), y, text, ha="left", va="top", fontsize=9.5, color=FAINT)


def axis_heading(ax, text, color=MUTED, side="left"):
    """Horizontal axis label above the axis, in place of a rotated y-label.

    With two y-axes, give each heading its series colour so no legend is needed.
    """
    x, ha = (0.0, "left") if side == "left" else (1.0, "right")
    ax.text(x, 1.03, text, transform=ax.transAxes, ha=ha, va="bottom",
            fontsize=10.5, fontweight="medium", color=color)


def end_label(ax, x, y, text, color=INK, dx=6, dy=0, **kw):
    """Label a series at its last point instead of using a legend."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                ha="left", va="center", fontsize=10.5, fontweight="medium",
                color=color, annotation_clip=False, **kw)


def callout(ax, x, y, text, color=MUTED, dx=0, dy=14, **kw):
    """Annotate one point that carries the story."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                ha="center", va="bottom" if dy >= 0 else "top",
                fontsize=10, color=color, **kw)


def twin(ax):
    """Second y-axis that follows the style: no extra gridlines or spines."""
    ax2 = ax.twinx()
    ax2.grid(False)
    for side in ("top", "left", "right"):
        ax2.spines[side].set_visible(False)
    return ax2


def color_axis(ax, color):
    """Tint an axis's tick labels to match the series it measures."""
    ax.tick_params(axis="y", labelcolor=color)


def save(fig, path, **kw):
    """Save as SVG (or whatever the suffix says) with the site defaults."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kw)
    return path


# ------------------------------------------------------------------ plotly
def plotly_template():
    """The same look for Plotly, for charts where hover earns its place.

        fig.update_layout(template=ss.plotly_template())

    The page must load Satoshi for the font to apply; otherwise the browser
    falls back to Helvetica/Arial.
    """
    import plotly.graph_objects as go

    font = f"{FONT_FAMILY}, 'Helvetica Neue', Helvetica, Arial, sans-serif"
    axis = dict(
        showgrid=False, zeroline=False, showline=True, linecolor=RULE,
        ticks="", tickfont=dict(color=MUTED, size=13),
        title=dict(font=dict(color=MUTED, size=13)),
    )
    yaxis = dict(axis, showgrid=True, gridcolor=GRID, gridwidth=1, showline=False)
    return go.layout.Template(
        layout=dict(
            font=dict(family=font, color=INK, size=14),
            colorway=SERIES,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(x=0, xanchor="left", font=dict(size=20, color=INK)),
            xaxis=axis,
            yaxis=yaxis,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
            hoverlabel=dict(bgcolor=PAPER, bordercolor=RULE,
                            font=dict(family=font, color=INK)),
            margin=dict(l=10, r=10, t=60, b=10),
        ),
        data=dict(
            scatter=[go.Scatter(line=dict(width=2.4))],
            bar=[go.Bar(marker=dict(line=dict(width=0)))],
        ),
    )
