"""
Draw the project-card thumbnails for projects/index.qmd.

    .venv/bin/python make_thumbnails.py            # all projects
    .venv/bin/python make_thumbnails.py rootbeer   # just one

Each function writes projects/<name>/thumbnail.png in the shared site style.
Cards crop the image to fit, so thumbnails carry no titles and keep the
picture away from the edges.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import site_style as ss

ROOT = Path(__file__).resolve().parent
PROJECTS = ROOT / "projects"
SURFACE = "#f1efe8"   # --color-surface-offset: the card's image well


def _figure():
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.16, right=0.84, top=0.84, bottom=0.18)
    return fig, ax


def _save(fig, project):
    # Fixed 1200x800 canvas with its margins intact (no tight cropping).
    with plt.rc_context({"savefig.bbox": "standard"}):
        out = ss.save(fig, PROJECTS / project / "thumbnail.png",
                      facecolor=SURFACE, dpi=200)
    plt.close(fig)
    print("wrote", out.relative_to(ROOT))


def premier_league():
    df = pd.read_csv(PROJECTS / "premier-league" / "pl_data_cache.csv")
    x = df["Season"].str[:4].astype(int)
    top, bottom = df["Title-Winning Points"], df["Relegation Survival Points"]
    fig, ax = _figure()
    ax.fill_between(x, bottom, top, color=ss.BRICK, alpha=0.07, linewidth=0)
    ax.plot(x, top, color=ss.BRICK, linewidth=2.6)
    ax.plot(x, bottom, color=ss.INDIGO, linewidth=2.6)
    ss.end_label(ax, x.iloc[-1], top.iloc[-1], "Champion", color=ss.BRICK)
    ss.end_label(ax, x.iloc[-1], bottom.iloc[-1], "Survivor", color=ss.INDIGO)
    ax.set_ylim(20, 105)
    ax.set_xticks([1995, 2005, 2015, 2025])
    _save(fig, "premier-league")


def fastest_human():
    df = pd.read_csv(PROJECTS / "fastest-human" / "speed_peak_1700.csv")
    fig, ax = _figure()
    ax.step(df["year"], df["speed_mph"], where="post", color=ss.BRICK, linewidth=2.6)
    ax.set_yscale("log")
    ax.set_yticks([10, 100, 1000, 10000], ["10", "100", "1k", "10k"])
    ax.minorticks_off()
    ax.set_xticks([1700, 1800, 1900, 2000])
    ss.axis_heading(ax, "mph")
    _save(fig, "fastest-human")


def rootbeer():
    # Rankings from RootBeer.ipynb: 1 is best, 6 is worst.
    ranks = pd.DataFrame({
        "A&W": [3, 2, 1, 3, 3, 2, 2],
        "Mug": [5, 4, 2, 4, 4, 3, 5],
        "Summit": [1, 1, 4, 2, 1, 1, 1],
        "Virgil's": [4, 5, 6, 6, 5, 6, 3],
        "Weinhardt": [6, 6, 3, 5, 6, 5, 6],
        "Barq's": [2, 3, 5, 1, 2, 4, 4],
    })
    avg = ranks.mean().sort_values(ascending=False)   # best ends up on top
    fig, ax = _figure()
    fig.subplots_adjust(left=0.27)
    colors = [ss.BRICK if name == avg.index[-1] else ss.FAINT for name in avg.index]
    ax.barh(avg.index, avg.values, color=colors, height=0.62)
    for name, value in avg.items():
        ax.text(value + 0.1, name, f"{value:.1f}", va="center", fontsize=10.5,
                color=ss.MUTED)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_xlim(0, 6.2)
    ax.tick_params(axis="y", labelcolor=ss.INK, labelsize=11.5)
    ss.axis_heading(ax, "Average rank (1 is best)")
    _save(fig, "RootBeer")


def birth_wealth_income():
    # Same FRED series as the notebook, via the keyless CSV endpoint.
    def fred(series):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        s = pd.read_csv(url, index_col=0, parse_dates=True).iloc[:, 0]
        return s.groupby(s.index.year).mean()

    df = pd.concat({"births": fred("SPDYNCBRTINUSA"),
                    "gdp": fred("A939RX0Q048SBEA")}, axis=1).dropna()
    fig, ax = _figure()
    ax.plot(df.index, df["births"], color=ss.BRICK, linewidth=2.6)
    ax.set_ylim(10, 24.5)
    ax.set_xticks([1960, 1980, 2000, 2020])
    ss.color_axis(ax, ss.BRICK)
    ss.axis_heading(ax, "Births per 1,000", color=ss.BRICK)
    ax2 = ss.twin(ax)
    ax2.plot(df.index, df["gdp"] / 1e3, color=ss.INDIGO, linewidth=2.6)
    ax2.set_ylim(0, 75)
    ax2.set_yticks([0, 25, 50, 75], ["$0", "$25k", "$50k", "$75k"])
    ss.color_axis(ax2, ss.INDIGO)
    ss.axis_heading(ax2, "Real GDP per capita", color=ss.INDIGO, side="right")
    _save(fig, "birth-wealth-income")


def road_paving():
    summary = pd.read_csv(PROJECTS / "road-paving" / "data" / "processed"
                          / "national_3r_summary.csv", index_col="year")
    real = summary["three_r_real_bn"]
    fig, ax = _figure()
    ax.bar(real.index, real.values, width=0.7, color=ss.BRICK, alpha=0.85)
    ticks = list(range(0, int(real.max()) + 1, 10))
    ax.set_yticks(ticks, ["$0"] + [f"${t}bn" for t in ticks[1:]])
    ax.set_xticks([real.index[0], 2016, 2020, real.index[-1]])
    ss.axis_heading(ax, f"Road repair spending, {int(summary['base_year'].iloc[0])} prices")
    _save(fig, "road-paving")


def marriage_county():
    # The pipeline has not produced data yet, so this is a mark, not a chart.
    from matplotlib.patches import Circle

    fig, ax = _figure()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_axis_off()
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    for x, color in ((2.45, ss.BRICK), (3.55, ss.INDIGO)):
        ax.add_patch(Circle((x, 2), 0.95, fill=False, edgecolor=color, linewidth=7))
    _save(fig, "marriage-county")


def temperature_animation():
    # Ten years of daily highs and lows for the page's default place (Pasadena),
    # from the same Open-Meteo archive the page itself reads.
    import datetime as dt
    import json
    import urllib.request

    last = dt.date.today().year - 1
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=34.1478&longitude=-118.1445"
           f"&start_date={last - 9}-01-01&end_date={last}-12-31"
           "&daily=temperature_2m_max,temperature_2m_min&temperature_unit=fahrenheit&timezone=auto")
    with urllib.request.urlopen(url, timeout=60) as response:
        daily = pd.DataFrame(json.load(response)["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    daily["doy"] = daily["time"].dt.dayofyear

    fig, ax = _figure()
    for year, d in daily.groupby(daily["time"].dt.year):
        newest = year == last
        style = dict(linewidth=1.6 if newest else 0.7, alpha=1.0 if newest else 0.22)
        ax.plot(d["doy"], d["temperature_2m_max"], color=ss.BRICK, **style)
        ax.plot(d["doy"], d["temperature_2m_min"], color=ss.INDIGO, **style)
    ax.set_xlim(1, 366)
    ax.set_xticks([1, 91, 182, 274], ["Jan", "Apr", "Jul", "Oct"])
    ax.set_yticks([40, 60, 80, 100], ["40", "60", "80", "100°F"])
    ss.axis_heading(ax, "Daily high and low")
    _save(fig, "temperature-animation")


THUMBNAILS = {
    "temperature-animation": temperature_animation,
    "marriage-county": marriage_county,
    "premier-league": premier_league,
    "fastest-human": fastest_human,
    "rootbeer": rootbeer,
    "birth-wealth-income": birth_wealth_income,
    "road-paving": road_paving,
}

if __name__ == "__main__":
    ss.apply(transparent=False, inline_svg=False)
    for name in sys.argv[1:] or THUMBNAILS:
        THUMBNAILS[name]()
