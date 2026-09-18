"""
sheets.py . drawing layer

Reads a Garden from model.py and draws the plan. Nothing here computes
geometry or schedules: if a number looks wrong, fix the model.

   pip install matplotlib
   python sheets.py garden.json
   python sheets.py garden.json --page tabloid --scale-denom 16
   python sheets.py garden.json --year 1 --no-panel
"""

import argparse
import datetime
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow, Rectangle

from model import (CYCLE, FREQ_CLASS, MONTHS, PLANTS, load, place)
from schematic import fig_schematic

# ======================================================================
# 1. SITE-SPECIFIC DATA
#     THIS BLOCK IS CALENDAR-SPECIFIC. A garden outside a Mediterranean
#     zone 10b needs its own SLATE and WINDOWS: a winter salad garden and
#     a continental summer garden are different calendars, not the same
#     one shifted by a few weeks.
# ======================================================================

# Which crops fill a family's slots, in priority order. Lists are
# consumed until the slots run out, so put what you want EVERY year
# first. n is a cap, None means "fill what remains".
SLATE = {
   "mediterranean_10b": {
       "nightshade": [("tomato", 2), ("tomato, cherry", 2),
                      ("tomato, determinate", 2), ("pepper", None),
                      ("basil", None), ("marigold", None)],
       "cucurbit":   [("pickling cucumber", 3), ("mini pumpkin", 2),
                      ("slicing cucumber", 1), ("summer squash", None)],
       "brassica":   [("kale", 4), ("broccoli", 2), ("cauliflower", 1),
                      ("spinach", None), ("lettuce", None),
                      ("asian greens", None)],
       "allium":     [("snap pea", None), ("garlic", None),
                      ("onion", None), ("carrot", None)],
       "open_ground": [("carving pumpkin", 2), ("watermelon", 2),
                       ("cantaloupe", 2), ("sweet corn", 25)],
   },
}

# sow under cover, in ground, harvest, germination. Month-floats, Oct = 0.
WINDOWS = {
   "mediterranean_10b": {
       "tomato": ((3.6, 5.5), (5.5, 5.8), (8.5, 1.0), "5-10 d at 75-85 F"),
       "tomato, cherry": ((3.6, 5.5), (5.5, 5.8), (8.5, 1.0), "5-10 d at 75-85 F"),
       "tomato, determinate": ((3.6, 5.5), (5.5, 5.8), (8.3, 0.7), "5-10 d at 75-85 F"),
       "pepper": ((3.5, 6.3), (6.3, 6.6), (9.0, 2.0), "10-21 d at 80-85 F"),
       "basil": ((5.0, 6.3), (6.3, 6.6), (7.5, 1.0), "5-10 d at 70 F"),
       "marigold": ((4.5, 6.0), (6.0, 6.3), (7.0, 1.9), "5-7 d at 70 F"),
       "pickling cucumber": (None, (7.0, 7.3), (8.6, 10.5), "3-10 d at 70 F+"),
       "slicing cucumber": (None, (7.0, 7.3), (8.6, 10.5), "3-10 d at 70 F+"),
       "summer squash": (None, (7.0, 7.3), (8.3, 9.3), "5-10 d at 70 F+"),
       "mini pumpkin": (None, (9.0, 9.3), (0.3, 0.8), "5-10 d at 75 F"),
       "kale": ((11.1, 0.8), (0.5, 0.8), (2.0, 7.9), "5-8 d at 65-75 F"),
       "broccoli": ((11.1, 0.8), (0.5, 0.8), (2.5, 6.5), "5-10 d at 65-75 F"),
       "cauliflower": ((11.1, 0.8), (0.5, 0.8), (3.0, 5.5), "5-10 d at 65-75 F"),
       "spinach": (None, (0.5, 3.5), (1.5, 5.5), "7-14 d, soil under 70 F"),
       "lettuce": (None, (0.0, 4.5), (1.0, 6.9), "4-10 d at 60-70 F"),
       "asian greens": (None, (0.0, 4.5), (1.0, 6.9), "4-8 d at 60-70 F"),
       "snap pea": (None, (0.8, 1.2), (3.0, 5.6), "7-14 d at 60-70 F"),
       "garlic": (None, (0.8, 1.2), (7.6, 8.3), "shoots in 1-3 wk"),
       "onion": ((11.3, 1.6), (1.3, 1.6), (7.6, 8.3), "7-14 d at 65-75 F"),
       "carrot": (None, (11.5, 4.9), (1.5, 7.5), "14-21 d, the slowest here"),
       "sweet potato": (None, (8.0, 8.3), (11.6, 0.6), "slips, order in Feb"),
       "carving pumpkin": (None, (9.0, 9.3), (0.3, 0.8), "5-10 d at 75 F"),
       "watermelon": ((6.3, 7.0), (7.0, 7.3), (10.0, 11.5), "4-12 d at 75 F+"),
       "cantaloupe": ((6.3, 7.0), (7.0, 7.3), (9.8, 10.9), "4-10 d at 75 F+"),
       "sweet corn": (None, (6.5, 7.3), (9.3, 9.9), "5-10 d at 65 F+"),
   },
}

PALETTE = {
   "nightshade": "#c0504d", "cucurbit": "#e0b13a", "brassica": "#5c8a6a",
   "allium": "#b9a06a", "grass": "#c9a227", "cover": "#9aab7a",
}
CROP_COLOUR = {
   "tomato": "#c0504d", "tomato, cherry": "#d4695f",
   "tomato, determinate": "#a8433f", "pepper": "#c9553d",
   "basil": "#6aa84f", "marigold": "#e8a33d",
   "pickling cucumber": "#4a8db7", "slicing cucumber": "#6aa3c4",
   "summer squash": "#e0b13a", "mini pumpkin": "#e08a3c",
   "carving pumpkin": "#d98a5f", "watermelon": "#c25b5b",
   "cantaloupe": "#d9a15f", "sweet corn": "#c9a227",
   "kale": "#3d6b52", "broccoli": "#5c8a6a", "cauliflower": "#7fa88c",
   "spinach": "#4f8f5c", "lettuce": "#8fbf70", "asian greens": "#a3c98a",
   "snap pea": "#7cb342", "garlic": "#b9a06a", "onion": "#a8577a",
   "carrot": "#d1743c", "sweet potato": "#9c6b4f",
}


def colour(crop):
   return CROP_COLOUR.get(crop, PALETTE.get(
       PLANTS.get(crop, {}).get("family", "cover"), "#888888"))


# ======================================================================
# 2. PAGE PLUMBING
# ======================================================================

PAGE_SIZES = {"letter": (11.0, 8.5), "tabloid": (17.0, 11.0),
             "a4": (11.69, 8.27), "a3": (16.54, 11.69)}
PAGE = PAGE_SIZES["letter"]
MARGIN, SCALE_DENOM, PANEL, DPI = 0.30, None, True, 200
PDF, WRITE_IMAGES = None, True
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]


def fs(pt):
   return pt * (PAGE[0] / 11.0) ** 0.7


def m(inches=None, axis="x"):
   inches = MARGIN if inches is None else inches
   return inches / (PAGE[0] if axis == "x" else PAGE[1])


def page():
   fig = plt.figure(figsize=PAGE)
   fig.patch.set_facecolor("white")
   return fig


def rect(fig, x, y, w, h, **kw):
   """add_artist, since appending to fig.patches was removed in mpl 3.7."""
   fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure, **kw))


def line(fig, xs, ys, **kw):
   fig.add_artist(Line2D(xs, ys, transform=fig.transFigure, **kw))


def header(fig, title, sub=None):
   fig.text(m(), 1 - m(0.34, "y"), title, fontsize=fs(13.5), weight="bold",
            va="top")
   if sub:
       fig.text(m(), 1 - m(0.60, "y"), sub, fontsize=fs(8.5), va="top",
                color="#555")


def footer(fig, lines):
   y = m(0.30, "y") + 0.021 * len(lines)
   for i, t in enumerate(lines):
       fig.text(m(), y - i * 0.021, t, fontsize=fs(7.2), va="top",
                color="#555")


def fit_axes(fig, box, xlim, ylim, denom=None, invert=True):
   fw, fh = fig.get_size_inches()
   dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
   s = min(box[2] * fw / dx, box[3] * fh / dy)
   if denom:
       want = 1.0 / denom
       if want <= s:
           s = want
       else:
           print(f"    note: 1:{denom} needs {dx * want:.1f} x "
                 f"{dy * want:.1f} in, have {box[2] * fw:.1f} x "
                 f"{box[3] * fh:.1f}, using 1:{1 / s:.1f}")
   w, h = dx * s / fw, dy * s / fh
   ax = fig.add_axes([box[0] + (box[2] - w) / 2,
                      box[1] + (box[3] - h) / 2, w, h])
   ax.set_xlim(*xlim)
   ax.set_ylim(*(ylim[::-1] if invert else ylim))
   ax.set_aspect("equal")
   ax.axis("off")
   return ax, s


def save(fig, outdir, name):
   if WRITE_IMAGES:
       fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=DPI,
                   facecolor="white")
       fig.savefig(os.path.join(outdir, f"{name}.svg"), facecolor="white")
   if PDF is not None:
       PDF.savefig(fig, facecolor="white")
   plt.close(fig)
   print("  wrote", name)


def spans(w, min_w=0.0):
   """Split any window that wraps the Oct/Sep boundary. Tomato harvest
   runs Jun to Oct, i.e. 8.5 to 1.0, which otherwise draws backwards."""
   if w is None:
       return []
   a, b = w
   out = [(a, max(b, a + min_w))] if b >= a else [(a, 12.0), (0.0, b)]
   return [(max(0, s), min(12, e)) for s, e in out if e > s]


def fmt_win(w):
   if w is None:
       return "n/a"

   def one(v):
       return f"{MONTHS[int(v) % 12]} {min(31, int(round((v % 1) * 31)) + 1)}"
   a, b = one(w[0]), one(w[1])
   return a if a == b else f"{a} to {b}"


# ======================================================================
# 3. CROP COMPOSITION
#     The seam between geometry and horticulture. The model knows how many
#     slots exist; SLATE decides what fills them.
# ======================================================================


def compose(family, grid, cal_name, prefs):
   """Crop requests for a family, capped by the slots the grid offers and
   filtered by the exclude list."""
   slate = SLATE[cal_name].get(family, [])
   exclude = set(prefs.get("exclude", []))
   n_lat, n_em = len(grid["laterals"]), len(grid["emitters"])
   budget = {"trellis": n_em, "bush": max(1, (n_lat >= 2) + 1) * (n_em // 2),
             "row": max(0, n_lat - 2) * n_em, "band": n_lat, "block": 25}
   out = []
   for name, cap in slate:
       if name in exclude or name not in PLANTS:
           continue
       habit = PLANTS[name]["habit"]
       room = budget.get(habit, n_em)
       if room <= 0:
           continue
       want = room if cap is None else min(cap, room)
       if habit == "bush":
           want = min(want, max(1, room))
       out.append(dict(crop=name, n=want))
       budget[habit] = room - want
   return out


# ======================================================================
# 4. SHEETS
# ======================================================================


def fig_cover(g, outdir):
   fig = page()
   s = g.site
   fig.text(m(), 1 - m(0.5, "y"), "GARDEN PLAN", fontsize=fs(26),
            weight="bold", va="top")
   fig.text(m(), 1 - m(0.95, "y"),
            f'{len(g.plots)} plots on a {len(CYCLE)} year rotation grouped '
            f'by water need' + (f', plus {len(g.open_ground)} open ground '
            f'area' if g.open_ground else ''),
            fontsize=fs(11.5), va="top", color="#555")
   fig.text(m(), 1 - m(1.22, "y"),
            f'{s["name"]} {s.get("zip", "")} . zone {s.get("zone", "?")} . '
            f'generated {datetime.date.today().isoformat()}',
            fontsize=fs(8.5), va="top", color="#888")
   line(fig, [m(), 1 - m()], [0.845, 0.845], color="#6b4f2a", lw=1.8)

   y = 0.79
   fig.text(m(), y, "PLOTS", fontsize=fs(9.5), weight="bold",
            color="#6b4f2a", va="top")
   y -= 0.033
   for p in g.plots:
       gd = p["grid"]
       fig.text(m() + 0.02, y, p["key"], fontsize=fs(9), weight="bold",
                va="top")
       fig.text(m() + 0.20, y,
                f'{p["width"]:.0f} x {p["length"]:.0f} in . '
                f'{g.area_sqft(p):.1f} sq ft . {gd["count"]} emitters . '
                f'{gd["rate_in_hr"]} in/hr . station {g.stations[p["key"]]}',
                fontsize=fs(9), va="top", color="#555")
       y -= 0.029
   for o in g.open_ground:
       gd = o["grid"]
       fig.text(m() + 0.02, y, o["name"], fontsize=fs(9), weight="bold",
                va="top")
       fig.text(m() + 0.20, y,
                f'{gd["count"]} emitters . {gd["rate_in_hr"]} in/hr . '
                f'station {o["station"]} . not rotated',
                fontsize=fs(9), va="top", color="#555")
       y -= 0.029
   y -= 0.014
   fig.text(m(), y, "RATES", fontsize=fs(9.5), weight="bold",
            color="#6b4f2a", va="top")
   y -= 0.033
   for p in g.plots:
       gd = p["grid"]
       fig.text(m() + 0.02, y, p["key"], fontsize=fs(9), weight="bold",
                va="top")
       fig.text(m() + 0.20, y,
                f'{gd["min_quarter"]} min = 0.25 in, {gd["min_one"]} min '
                f'= 1 in, laterals {gd["lateral_spacing"]:.1f} in apart',
                fontsize=fs(9), va="top", color="#555")
       y -= 0.029
   footer(fig, [
       f'Stations run sequentially from a single {g.hw.get("start_time", "04:30")}'
       f' start, which keeps foliage dry through the day.',
       "Rates differ by plot because lateral spacing follows bed width. "
       "Run times are derived per plot, not shared.",
       g.cal["note"]])
   save(fig, outdir, "00_cover")


def fig_feasibility(g, outdir):
   fig = page()
   header(fig, "FEASIBILITY AND WARNINGS",
          "What the model could not honour, and the climate rules it "
          "applied. Read this before buying seed.")
   y = 0.79
   fig.text(m(), y, "CLIMATE RULES IN FORCE", fontsize=fs(9.5),
            weight="bold", color="#6b4f2a", va="top")
   y -= 0.032
   for r in g.climate_notes:
       fig.text(m() + 0.02, y, r["crop"], fontsize=fs(8.5), weight="bold",
                va="top")
       fig.text(m() + 0.13, y, r["verdict"], fontsize=fs(8.5), va="top")
       fig.text(m() + 0.42, y, f'because {r["why"]}', fontsize=fs(8),
                va="top", color="#666")
       y -= 0.028
   y -= 0.016
   fig.text(m(), y, "WARNINGS", fontsize=fs(9.5), weight="bold",
            color="#8a2f2f", va="top")
   y -= 0.032
   if not g.warnings:
       fig.text(m() + 0.02, y, "none", fontsize=fs(8.5), va="top",
                style="italic", color="#666")
   for w in g.warnings:
       fig.text(m() + 0.02, y, f'[{w["kind"]}]', fontsize=fs(8),
                weight="bold", va="top", color="#8a2f2f")
       txt = w["detail"]
       while txt:
           cut = txt[:118].rsplit(" ", 1)[0] if len(txt) > 118 else txt
           fig.text(m() + 0.115, y, cut, fontsize=fs(8), va="top")
           txt = txt[len(cut):].lstrip()
           y -= 0.021
       y -= 0.006
   footer(fig, [
       "A warning is not a failure. Sprawling crops assigned to open "
       "ground, or vines trained out of a plot, are expected behaviour.",
       "An [infeasible] entry does need a decision: either enlarge the "
       "ground, drop the crop, or substitute a trellised miniature."])
   save(fig, outdir, "01_feasibility")


def fig_rotation(g, outdir):
   if not g.rotation:
       return
   fig = page()
   header(fig, f"{len(CYCLE)} YEAR ROTATION",
          "Heavy feeder, restorative, heavy feeder, resting. Each family "
          "visits each plot once, so repeat spacing is maximal.")
   left, top = 0.20, 0.74
   w, h = (1 - m() - left) / len(g.plots), 0.115
   for j, p in enumerate(g.plots):
       fig.text(left + (j + 0.5) * w, top + 0.035,
                f'{p["key"]}\nstation {g.stations[p["key"]]}',
                fontsize=fs(8), weight="bold", ha="center", va="center")
   for i in sorted(g.rotation):
       y = top - i * h
       fig.text(left - 0.012, y + h / 2, f"year {i}", fontsize=fs(8),
                ha="right", va="center")
       for j, p in enumerate(g.plots):
           fam = g.rotation[i][p["key"]]
           rect(fig, left + j * w, y, w, h, facecolor=PALETTE[fam],
                alpha=0.26, edgecolor="white", lw=2)
           fig.text(left + (j + 0.5) * w, y + h / 2,
                    f'{fam}\n{FREQ_CLASS[fam]} rooted', fontsize=fs(7.4),
                    ha="center", va="center", weight="bold")
   notes = ["Frequency class, not volume, is what constrains a shared "
            "valve: rooting depth dictates how often, and volume is "
            "trivially adjusted with run time."]
   for grp in g.groups:
       notes.append(f'{" + ".join(grp)} share one solenoid, so their '
                    f'pairings were chosen to be compatible in every year.')
   if g.open_ground:
       notes.append(f'{", ".join(o["name"] for o in g.open_ground)} sits '
                    f'outside this table and grows the sprawlers every '
                    f'year, which is what makes them annual rather than '
                    f'one year in {len(CYCLE)}.')
   footer(fig, notes)
   save(fig, outdir, "02_rotation")


def draw_grid(ax, plot_w, plot_l, grid, trellis=None):
   ax.add_patch(Rectangle((0, 0), plot_l, plot_w, facecolor="#efe7d8",
                          edgecolor="#6b4f2a", lw=1.8, zorder=0))
   ax.plot([-4, -4], [0, plot_w], color="#2f5d8a", lw=2.6, zorder=2)
   ax.text(-6.6, plot_w / 2, "HEADER", rotation=90, fontsize=fs(6),
           va="center", ha="center", color="#2f5d8a")
   for i, y in enumerate(grid["laterals"], start=1):
       ax.plot([-4, plot_l], [y, y], color="#2f5d8a", lw=0.9,
               ls=(0, (5, 3)), zorder=2)
       ax.text(plot_l + 2.2, y, f"L{i}", fontsize=fs(6.2), va="center",
               color="#2f5d8a")
       for x in grid["emitters"]:
           ax.plot(x, y, "o", ms=fs(2.1), color="#2f5d8a", zorder=6)
   for j, x in enumerate(grid["emitters"], start=1):
       ax.text(x, plot_w + 4, f"E{j}", fontsize=fs(5.8), ha="center",
               va="center", color="#8a8a8a")
   if trellis:
       ax.plot([0, plot_l], [-2, -2], color="#4a4a4a", lw=3.4,
               solid_capstyle="butt", zorder=3)
       ax.text(plot_l / 2, -5.4, trellis, fontsize=fs(6.8), ha="center",
               va="center", color="#4a4a4a")


def draw_placed(ax, plot_w, grid, placed):
   mid = plot_w / 2
   for p in placed:
       x = grid["emitters"][p["em"] - 1]
       y = grid["laterals"][p["lat"] - 1]
       c = colour(p["crop"])
       r = max(4, p["r"])
       ax.add_patch(Circle((x, y), r, facecolor=c, alpha=0.20, edgecolor=c,
                           lw=0.8, zorder=4))
       ax.plot(x, y, "o", ms=fs(4.4) if r > 8 else fs(3.0), color=c,
               mec="white", mew=0.7, zorder=7)
   seen = set()
   for p in placed:
       if p["crop"] in seen:
           continue
       seen.add(p["crop"])
       x = grid["emitters"][p["em"] - 1]
       y = grid["laterals"][p["lat"] - 1]
       up = y < mid
       yy = y - max(4, p["r"]) - 2.2 if up else y + max(4, p["r"]) + 2.2
       ax.text(x, yy, p["crop"], fontsize=fs(6.2), ha="center",
               va="bottom" if up else "top", color="#2b2b2b", zorder=9,
               path_effects=HALO)


def draw_panel(g, fig, placed, cal_name, x0=0.655):
   win = WINDOWS[cal_name]
   y, lh = 0.795, 0.0166
   fig.text(x0, y, "WHAT IS IN EACH SECTION", fontsize=fs(7.4),
            weight="bold", color="#6b4f2a", va="center")
   y -= lh * 1.4
   by_lat = {}
   for p in placed:
       by_lat.setdefault(p["lat"], {}).setdefault(p["crop"], 0)
       by_lat[p["lat"]][p["crop"]] += 1
   for lat in sorted(by_lat):
       fig.text(x0, y, f"L{lat}", fontsize=fs(6.9), weight="bold",
                va="center", color="#222")
       y -= lh
       for crop_name, n in by_lat[lat].items():
           w = win.get(crop_name)
           fig.text(x0 + 0.008, y, f"{n} x {crop_name}", fontsize=fs(6.1),
                    va="center", color="#222", weight="bold")
           y -= lh
           if not w:
               fig.text(x0 + 0.016, y, "no dates for this calendar",
                        fontsize=fs(5.7), va="center", color="#8a2f2f")
               y -= lh
               continue
           indoor, ground, harv, germ = w
           for label, val in (("sow", fmt_win(indoor) + ", cover" if indoor
                               else fmt_win(ground) + ", direct"),
                              ("in ground", fmt_win(ground)),
                              ("germ", germ),
                              ("harvest", fmt_win(harv))):
               fig.text(x0 + 0.016, y, label, fontsize=fs(5.7), va="center",
                        color="#8a8a8a")
               fig.text(x0 + 0.070, y, val, fontsize=fs(5.7), va="center",
                        color="#333")
               y -= lh
       y -= 0.005


def fig_plot_map(g, outdir, year, plot, cal_name):
   fam = g.rotation[year][plot["key"]]
   grid = plot["grid"]
   crops = compose(fam, grid, cal_name, g.cfg.get("preferences", {}))
   placed, unplaced = place(grid, crops)
   st = g.stations[plot["key"]]
   sched = g.schedule_for(st, year)

   fig = page()
   header(fig, f'YEAR {year} . {plot["key"]} . {fam.upper()}',
          f'station {st} . {plot["width"]:.0f} x {plot["length"]:.0f} in . '
          f'{g.area_sqft(plot):.1f} sq ft . {FREQ_CLASS[fam]} rooted')
   pad = 12
   box_w = (0.615 - m()) if PANEL else (1 - 2 * m())
   ax, s = fit_axes(fig, (m(), 0.22, box_w, 0.60),
                    (-pad, plot["length"] + pad), (-pad, plot["width"] + pad),
                    denom=SCALE_DENOM)
   trellised = [p for p in placed if PLANTS[p["crop"]]["habit"] == "trellis"]
   label = ("T R E L L I S  ( " +
            ", ".join(sorted({p["crop"] for p in trellised})) + " )"
            ) if trellised else "north edge"
   draw_grid(ax, plot["width"], plot["length"], grid, label)
   draw_placed(ax, plot["width"], grid, placed)
   if PANEL:
       draw_panel(g, fig, placed, cal_name)

   peak = max(sched, key=lambda t: t[0] * t[1])
   fig.text(m(), 0.845,
            f'{grid["count"]} emitters at {g.hw["emitter_gph"]:g} gph . '
            f'{grid["gph"]:.1f} gph . laterals '
            f'{grid["lateral_spacing"]:.1f} in apart . '
            f'{grid["rate_in_hr"]} in/hr . {grid["min_quarter"]} min = '
            f'0.25 in . peak {peak[0]:g} x {peak[1]} min . '
            f'drawn 1:{1 / s:.0f}', fontsize=fs(7.2), va="top",
            color="#444")
   notes = []
   if unplaced:
       notes.append("Not placed here: " + "; ".join(
           f'{u["n"]} x {u["crop"]} ({u["why"]})' for u in unplaced))
   notes.append("Circles are true mature canopy. Outer laterals are "
                "staggered so diagonal neighbours gain clearance, which is "
                "what permits vertical training rather than caging.")
   for r in g.climate_notes:
       if any(r["crop"] in p["crop"] for p in placed):
           notes.append(f'{r["crop"]}: {r["verdict"]}, because {r["why"]}.')
   footer(fig, notes[:4])
   if placed:
       keys = sorted({p["crop"] for p in placed})
       fig.legend(handles=[Line2D([], [], marker="o", ls="none", ms=fs(6),
                                  mfc=colour(k), mec="white", label=k)
                           for k in keys],
                  loc="lower left", bbox_to_anchor=(m(), m(0.30, "y") + 0.10),
                  ncol=4, frameon=False, fontsize=fs(6.3),
                  handletextpad=0.4, columnspacing=1.1)
   save(fig, outdir, f'y{year}_{plot["key"].replace(" ", "_")}_{fam}')


def fig_open_ground(g, outdir, og, cal_name):
   grid = og["grid"]
   crops = compose("open_ground", grid, cal_name,
                   g.cfg.get("preferences", {}))
   fig = page()
   w, l = og["cfg"]["width_in"], og["cfg"]["length_in"]
   header(fig, f'{og["name"].upper()} . STATION {og["station"]} . '
               f'{w / 12:.0f} x {l / 12:.0f} FT',
          "Outside the rotation, so this sheet is the same every year. "
          "Sprawling crops and any block-pollinated grass live here.")
   pad = 16
   ax, s = fit_axes(fig, (m(), 0.26, 1 - 2 * m(), 0.54),
                    (-pad, l + pad), (-pad, w + pad),
                    denom=SCALE_DENOM, invert=False)
   ax.add_patch(Rectangle((0, 0), l, w, facecolor="#f1ece0",
                          edgecolor="#6b4f2a", lw=1.8, zorder=0))
   ax.plot([-5, -5], [0, w], color="#2f5d8a", lw=2.6, zorder=2)
   for i, y in enumerate(grid["laterals"], start=1):
       ax.plot([-5, l], [y, y], color="#2f5d8a", lw=0.8, ls=(0, (5, 3)),
               zorder=2)
       for x in grid["emitters"]:
           ax.plot(x, y, "o", ms=fs(1.7), color="#2f5d8a", zorder=6)

   # hills spread across the plot, blocks squared off at one end
   hills = [c for c in crops if PLANTS[c["crop"]]["habit"] == "sprawl"]
   blocks = [c for c in crops if PLANTS[c["crop"]]["habit"] == "block"]
   n_hill = sum(h["n"] for h in hills)
   x_end = l - (60 if blocks else 0)
   slot, i = 0, 0
   for hc in hills:
       p = PLANTS[hc["crop"]]
       for _ in range(hc["n"]):
           col_n = max(1, n_hill // 2)
           cx = 30 + (slot % col_n) * max(48, (x_end - 60) / max(1, col_n - 1))
           cy = w * (0.68 if slot < col_n else 0.30)
           c = colour(hc["crop"])
           ax.add_patch(Circle((cx, cy), p["canopy_r"], facecolor=c,
                               alpha=0.17, edgecolor=c, lw=0.8, zorder=4))
           ax.plot(cx, cy, "o", ms=fs(5.6), color=c, mec="white", mew=0.9,
                   zorder=8)
           ax.text(cx, cy + 5.5, hc["crop"], fontsize=fs(6.4), ha="center",
                   va="bottom", weight="bold", zorder=9, path_effects=HALO)
           dx = -1 if slot % col_n == 0 else 0
           dy = 1 if slot < col_n else -1
           nl = (dx ** 2 + dy ** 2) ** 0.5
           ax.annotate("", xytext=(cx, cy),
                       xy=(cx + 30 * dx / nl, cy + 30 * dy / nl),
                       arrowprops=dict(arrowstyle="-|>", lw=1.2, color=c,
                                       alpha=0.75), zorder=5)
           slot += 1
   for bc in blocks:
       side = int(round(bc["n"] ** 0.5))
       step = 12
       x0 = l - side * step - 6
       y0 = w - side * step - 6
       c = colour(bc["crop"])
       ax.add_patch(Rectangle((x0 - 6, y0 - 6), side * step, side * step,
                              facecolor=c, alpha=0.16, edgecolor=c, lw=1.0,
                              zorder=3))
       for a in range(side):
           for b in range(side):
               ax.plot(x0 + a * step, y0 + b * step, "o", ms=fs(3.4),
                       color=c, mec="white", mew=0.6, zorder=7)
       ax.text(x0 + side * step / 2 - 6, y0 + side * step + 4,
               f'{side * side} x {bc["crop"]}, {side} x {side} BLOCK',
               fontsize=fs(6.4), ha="center", va="bottom", weight="bold",
               color="#8a6d1f", zorder=9, path_effects=HALO)
   ax.add_patch(FancyArrow(l + 8, w - 28, 0, 22, width=0.9, head_width=6,
                           head_length=7, color="#333", zorder=9))
   ax.text(l + 8, w + 1, "N", fontsize=fs(10), weight="bold", ha="center",
           va="bottom")
   fig.text(m(), 0.845,
            f'{grid["count"]} emitters . {grid["gph"]:.1f} gph . '
            f'{grid["rate_in_hr"]} in/hr . {grid["min_quarter"]} min = '
            f'0.25 in . 1 in = {w * l / 144 * 0.623:.0f} gal . '
            f'drawn 1:{1 / s:.0f}', fontsize=fs(7.2), va="top", color="#444")
   footer(fig, [
       "Arrows show the trained run direction, out of the plot. Circles are "
       "canopy at about eight weeks and every one will be exceeded, which "
       "is the point: the crown is in the ground and the vine leaves.",
       "A block rather than a row for anything wind pollinated, since a "
       "single row sets scattered kernels.",
       "Not rotated, so it needs its own hygiene: move the hills within the "
       "plot each season, a winter cover crop across the whole area, and "
       "clear all vine debris in autumn."])
   save(fig, outdir, f'og_{og["name"]}')


def fig_schedule(g, outdir, year):
   stations = sorted(set(g.stations.values()))
   data = {st: g.schedule_for(st, year) for st in stations}
   weekly = {st: [e * mn for e, mn in data[st]] for st in stations}
   hi = max(max(v) for v in weekly.values()) or 1
   norm, cmap = Normalize(0, hi), plt.get_cmap("YlGnBu")
   fig = page()
   header(fig, f"DRIP SCHEDULE . YEAR {year}",
          f'one {g.hw.get("start_time", "04:30")} start, stations '
          f'sequential . minutes derived per plot from its own measured '
          f'application rate')
   left, top = 0.165, 0.76
   w = (1 - m() - left) / 12
   h = min(0.098, 0.56 / max(1, len(stations)))
   for k, mon in enumerate(MONTHS):
       fig.text(left + (k + 0.5) * w, top + 0.026, mon, fontsize=fs(8.6),
                weight="bold", ha="center", va="center")
   for i, st in enumerate(stations):
       y = top - (i + 1) * h
       keys = [k for k, v in g.stations.items() if v == st]
       fams = " + ".join(g.rotation[year][k][:1].upper() for k in keys
                         if g.rotation and k in g.rotation[year]) or "fixed"
       fig.text(left - 0.012, y + h / 2,
                f'station {st}\n{", ".join(keys)}\n{fams}',
                fontsize=fs(6.8), ha="right", va="center", weight="bold")
       for k in range(12):
           ev, mn = data[st][k]
           rect(fig, left + k * w, y, w, h,
                facecolor=cmap(norm(weekly[st][k])), alpha=0.88,
                edgecolor="white", lw=1.5)
           col = "white" if norm(weekly[st][k]) > 0.55 else "#222"
           cx = left + (k + 0.5) * w
           if ev == 0:
               fig.text(cx, y + h / 2, "off", fontsize=fs(6.4), ha="center",
                        va="center", color=col, style="italic")
           else:
               lab = ("every 2nd day" if ev == 3.5 else
                      "every 3rd day" if abs(ev - 2.33) < .05 else
                      f"{ev:g} x wk")
               fig.text(cx, y + h * 0.63, lab, fontsize=fs(6.2),
                        ha="center", va="center", color=col)
               fig.text(cx, y + h * 0.32, f"{mn} min", fontsize=fs(7.8),
                        ha="center", va="center", weight="bold", color=col)
   notes = [
       "Rain: half an inch buys five days for shallow rooted plots, an inch "
       "buys a week. Heat: add a quarter above 95 F. Dry wind: add twenty "
       "to thirty percent, because the humidity collapse does the damage.",
       "The test that overrules every number here is a finger three inches "
       "into the soil the morning after. Dry means lengthen events rather "
       "than adding more of them."]
   for grp in g.groups:
       notes.insert(0, f'{" + ".join(grp)} share a solenoid, so that row is '
                       f'the envelope of both: higher frequency, longer '
                       f'event. Any dry-down needs the manual valve.')
   footer(fig, notes[:4])
   save(fig, outdir, f"y{year}_schedule")


def fig_planting(g, outdir, cal_name):
   win = WINDOWS[cal_name]
   rows = []
   for fam in CYCLE + (["open_ground"] if g.open_ground else []):
       for name, _ in SLATE[cal_name].get(fam, []):
           if name in win and name not in g.cfg.get(
                   "preferences", {}).get("exclude", []):
               rows.append((fam, name, win[name]))
   fig = page()
   header(fig, "PLANTING CALENDAR",
          "Rotation year . dashed = under cover, solid = in the ground, "
          "hatched = harvest . germination at right")
   left, top = 0.19, 0.79
   w = (1 - m() - left - 0.17) / 12
   rh = min(0.021, 0.60 / max(1, len(rows) + 6))
   bottom = top - rh * (len(rows) + 6)
   for k, mon in enumerate(MONTHS):
       fig.text(left + (k + 0.5) * w, top + 0.019, mon, fontsize=fs(7.8),
                weight="bold", ha="center", va="center")
       line(fig, [left + k * w] * 2, [top + 0.008, bottom], color="#e6e6e6",
            lw=0.7)
   line(fig, [left + 12 * w] * 2, [top + 0.008, bottom], color="#e6e6e6",
        lw=0.7)
   y, last = top, None
   for fam, name, (indoor, ground, harv, germ) in rows:
       if fam != last:
           y -= 0.011
           fig.text(m(), y, fam.upper(), fontsize=fs(6.8), weight="bold",
                    color=PALETTE.get(fam, "#6b4f2a"), va="center")
           last = fam
           y -= rh
       c = colour(name)
       fig.text(left - 0.008, y, name, fontsize=fs(6.3), ha="right",
                va="center")
       for s0, e0 in spans(indoor):
           rect(fig, left + s0 * w, y - rh * .27, (e0 - s0) * w, rh * .54,
                facecolor=c, alpha=0.18, edgecolor=c, lw=0.6,
                ls=(0, (2, 1.5)))
       for s0, e0 in spans(ground, min_w=0.18):
           rect(fig, left + s0 * w, y - rh * .32, (e0 - s0) * w, rh * .64,
                facecolor=c, alpha=0.88, edgecolor="none")
       for s0, e0 in spans(harv):
           rect(fig, left + s0 * w, y - rh * .27, (e0 - s0) * w, rh * .54,
                facecolor=c, alpha=0.28, edgecolor=c, lw=0.6, hatch="////")
       fig.text(left + 12 * w + 0.006, y, germ, fontsize=fs(5.5),
                va="center", color="#555")
       y -= rh
   footer(fig, [
       "Germination assumes the stated soil temperature. A seed absent "
       "after twice its window has failed, and knowing that in week two "
       "rather than week six is the whole point of the column.",
       "Windows are calendar-specific. A garden outside this climate needs "
       "its own WINDOWS table, not these dates shifted."])
   save(fig, outdir, "03_planting")


# ======================================================================
# 5. MAIN
# ======================================================================


def main():
   global PDF, WRITE_IMAGES, PAGE, DPI, MARGIN, SCALE_DENOM, PANEL
   ap = argparse.ArgumentParser()
   ap.add_argument("config")
   ap.add_argument("--year", type=int)
   ap.add_argument("--outdir", default="garden_out")
   ap.add_argument("--pdf", default="garden_plan.pdf", help="or 'none'")
   ap.add_argument("--page", default="letter", choices=list(PAGE_SIZES))
   ap.add_argument("--margin", type=float, default=0.30)
   ap.add_argument("--scale-denom", type=int)
   ap.add_argument("--no-panel", action="store_true")
   ap.add_argument("--dpi", type=int, default=200)
   ap.add_argument("--no-images", action="store_true")
   args = ap.parse_args()

   PAGE = PAGE_SIZES[args.page]
   MARGIN, SCALE_DENOM = args.margin, args.scale_denom
   PANEL, DPI = not args.no_panel, args.dpi
   WRITE_IMAGES = not args.no_images
   os.makedirs(args.outdir, exist_ok=True)

   g = load(args.config)
   cal_name = g.site.get("calendar", "mediterranean_10b")
   if cal_name not in SLATE:
       raise SystemExit(f"no SLATE for calendar {cal_name!r}. Add one, or "
                        f"set site.calendar to a defined calendar.")
   years = [args.year] if args.year else sorted(g.rotation or {1: None})

   if args.pdf.lower() != "none":
       PDF = PdfPages(os.path.join(args.outdir, args.pdf))
       info = PDF.infodict()
       info["Title"] = f'Garden plan, {g.site["name"]}'
       info["CreationDate"] = datetime.datetime.now()
   try:
       print("fixed sheets:")
       fig_cover(g, args.outdir)
       fig_feasibility(g, args.outdir)
       fig_rotation(g, args.outdir)
       fig_planting(g, args.outdir, cal_name)
       fig_schematic(g, args.outdir)
       for og in g.open_ground:
           fig_open_ground(g, args.outdir, og, cal_name)
       for y in years:
           print(f"year {y}:")
           fig_schedule(g, args.outdir, y)
           if g.rotation:
               for plot in g.plots:
                   fig_plot_map(g, args.outdir, y, plot, cal_name)
   finally:
       if PDF is not None:
           PDF.close()
   print(f'\n{len(os.listdir(args.outdir))} files in {args.outdir}/')


if __name__ == "__main__":
   main()