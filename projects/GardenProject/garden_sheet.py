"""
garden_sheets.py . Altadena CA 91104 . zone 10b

Drawing only. Every number comes from garden_data, so if a figure looks
wrong, fix it there.

   pip install matplotlib
   python garden_sheets.py
   python garden_sheets.py --scale-denom 16      # bed maps at 1:16
   python garden_sheets.py --year 1 --no-panel

Beds: X runs along the length from the header end, Y across the width
with 0 at the north edge, inverted so north is at the top. The annex runs
Y northward and is not inverted.
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

import garden_data as G
from garden_data import (ANNEX, ANNEX_HILLS, CONTAINERS, CORN, EMITTER_GPH,
                        FAMILY_COLOUR, FAMILY_LABEL, FREQ_CLASS, GRIDS,
                        MONTHS, PALETTE, PLOTS, PLOT_ORDER, PULSES,
                        ROTATION, ROTATION_YEARS, STATION_LABEL,
                        SUCCESSIONS, VALVES, VALVE_NOTES, area_sqft,
                        counts, freq_label, layout, plot_of,
                        station_schedule, tasks)

PAGE_SIZES = {"letter": (11.0, 8.5), "tabloid": (17.0, 11.0),
             "a4": (11.69, 8.27), "a3": (16.54, 11.69)}
PAGE = PAGE_SIZES["letter"]
MARGIN, SCALE_DENOM, PANEL, DPI = 0.30, None, True, 200
PDF, WRITE_IMAGES = None, True
HALO = [pe.withStroke(linewidth=2.0, foreground="white")]

# Sow under cover, in ground, harvest bar, germination. Oct = 0.0.
WINDOWS = {
   "tomatoes": ((3.6, 5.5), (5.5, 5.8), (8.5, 1.0), "5-10 d at 75-85 F"),
   "peppers": ((3.5, 6.3), (6.3, 6.6), (9.0, 2.0), "10-21 d at 80-85 F"),
   "basil": ((5.0, 6.3), (6.3, 6.6), (7.5, 1.0), "5-10 d at 70 F"),
   "marigold": ((4.5, 6.0), (6.0, 6.3), (7.0, 1.9), "5-7 d at 70 F"),
   "cucumbers, 1st": (None, (7.0, 7.3), (8.6, 10.5), "3-10 d at 70 F+"),
   "cucumbers, 2nd": (None, (8.5, 8.8), (10.0, 11.9), "3-10 d at 70 F+"),
   "summer squash": (None, (7.0, 7.3), (8.3, 9.3), "5-10 d at 70 F+"),
   "mini pumpkins": (None, (9.0, 9.3), (0.3, 0.8), "5-10 d at 75 F"),
   "kale": ((11.1, 0.8), (0.5, 0.8), (2.0, 7.9), "5-8 d at 65-75 F"),
   "broccoli": ((11.1, 0.8), (0.5, 0.8), (2.5, 6.5), "5-10 d at 65-75 F"),
   "cauliflower": ((11.1, 0.8), (0.5, 0.8), (3.0, 5.5), "5-10 d at 65-75 F"),
   "spinach": (None, (0.5, 3.5), (1.5, 5.5), "7-14 d, soil under 70 F"),
   "lettuce + greens": (None, (0.0, 4.5), (1.0, 6.9), "4-10 d at 60-70 F"),
   "snap peas": (None, (0.8, 1.2), (3.0, 5.6), "7-14 d at 60-70 F"),
   "garlic": (None, (0.8, 1.2), (7.6, 8.3), "shoots in 1-3 wk"),
   "onions": ((11.3, 1.6), (1.3, 1.6), (7.6, 8.3), "7-14 d at 65-75 F"),
   "carrots + radish": (None, (11.5, 4.9), (1.5, 7.5), "14-21 d, slowest here"),
   "bush beans": (None, (5.6, 6.0), (7.5, 9.5), "7-10 d at 70 F"),
   "sweet potatoes": (None, (8.0, 8.3), (11.6, 0.6), "slips, order in Feb"),
   "annex corn": (None, (6.5, 7.3), (9.3, 9.9), "5-10 d at 65 F+"),
   "annex watermelon": ((6.3, 7.0), (7.0, 7.3), (10.0, 11.5), "4-12 d at 75 F+"),
   "annex cantaloupe": ((6.3, 7.0), (7.0, 7.3), (9.8, 10.9), "4-10 d at 75 F+"),
   "annex pumpkins": (None, (9.0, 9.3), (0.3, 0.8), "5-10 d at 75 F"),
   "winter cover": (None, (1.2, 1.5), (4.3, 4.8), "7-14 d on rainfall"),
   "buckwheat": (None, (7.8, 8.2), (8.8, 9.2), "3-5 d at 70 F"),
   "cowpeas": (None, (9.0, 9.5), (0.8, 1.3), "5-10 d at 70 F"),
}
WIN_GROUP = [("nightshade", ["tomatoes", "peppers", "basil", "marigold"]),
            ("cucurbit", ["cucumbers, 1st", "cucumbers, 2nd",
                          "summer squash", "mini pumpkins"]),
            ("brassica", ["kale", "broccoli", "cauliflower", "spinach",
                          "lettuce + greens"]),
            ("allium", ["snap peas", "garlic", "onions", "carrots + radish",
                        "bush beans", "sweet potatoes"]),
            ("annex", ["annex corn", "annex watermelon", "annex cantaloupe",
                       "annex pumpkins"]),
            ("cover", ["winter cover", "buckwheat", "cowpeas"])]

SEED_ORDER = [
   ("tomatoes", "9 to 12 seeds, 4 varieties", "packet lasts 3 to 4 years"),
   ("peppers", "8 to 10 seeds", "packet lasts 4 years"),
   ("kale, broccoli, cauliflower", "8 to 13 total", "packet lasts 3 years"),
   ("spinach", "about 100, in 5 sowings", "REBUY ANNUALLY, poor viability"),
   ("lettuce and Asian greens", "about 90 in 6 sowings", "one packet each"),
   ("carrots", "60 to 92 depending on year", "REBUY ANNUALLY"),
   ("radish", "40 to 50", "one packet per season"),
   ("snap peas", "24 to 38", "one ounce per season"),
   ("bush beans", "24 to 38", "one ounce per season"),
   ("cucumbers", "16 to 20 with the 2nd flush", "packet lasts 2 seasons"),
   ("summer squash", "8, sown in pairs", "packet lasts 3 seasons"),
   ("mini pumpkins", "4", "packet lasts many years"),
   ("annex corn", "30", "one packet per season"),
   ("annex melons", "8", "packet each lasts 3 seasons"),
   ("annex pumpkins", "6", "packet lasts 3 seasons"),
   ("garlic", "24 to 38 cloves", "half a pound, a full pound in year 4"),
   ("onions", "36 to 60 cells sown", "REBUY ANNUALLY, poor viability"),
   ("sweet potato slips", "8", "ORDER IN FEBRUARY, they sell out"),
]

HEAD = [("SUPPLY", "3/4 in bib\nfoothill pressure often 80+ psi", "#8a8a8a"),
       ("BACKFLOW", "type per local code\nask your water purveyor", "#7a6a9a"),
       ("BALL VALVE", "3/4 in isolation\nservice everything below", "#8a2f2f"),
       ("FILTER", "150 to 200 mesh\nUPSTREAM of the solenoids", "#5a7a9a"),
       ("PRV", "ADJUSTABLE spring + diaphragm\nSET TO 25 PSI", "#2f5d8a"),
       ("GAUGE", "0 to 100 psi, liquid filled\nfor setting the PRV", "#4a4a4a"),
       ("METER", "low flow totalizer\nmonthly reading is your log", "#6b8a5a")]

KIND_COLOUR = {"tray": "#7b5aa6", "direct": "#5c8a6a", "plant": "#c0504d",
              "buy": "#2f5d8a", "cultivate": "#9aab7a", "valve": "#8a2f2f"}


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
   # No bbox_inches: the figure IS the page, so nothing is padded twice.
   if WRITE_IMAGES:
       fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=DPI,
                   facecolor="white")
       fig.savefig(os.path.join(outdir, f"{name}.svg"), facecolor="white")
   if PDF is not None:
       PDF.savefig(fig, facecolor="white")
   plt.close(fig)
   print("  wrote", name)


def spans(w, min_w=0.0):
   """Split windows that wrap the Oct/Sep boundary. Tomato harvest runs
   Jun to Oct, 8.5 to 1.0, which otherwise draws backwards."""
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
# BED MAPS
# ======================================================================


def draw_frame(ax, plot, grid, trellis):
   W, L = plot["width"], plot["length"]
   ax.add_patch(Rectangle((0, 0), L, W, facecolor="#efe7d8",
                          edgecolor="#6b4f2a", lw=1.8, zorder=0))
   ax.plot([-4, -4], [0, W], color="#2f5d8a", lw=2.6, zorder=2)
   ax.text(-6.8, W / 2, "HEADER", rotation=90, fontsize=fs(6), va="center",
           ha="center", color="#2f5d8a")
   for i, y in enumerate(grid["laterals"], start=1):
       ax.plot([-4, L], [y, y], color="#2f5d8a", lw=0.9, ls=(0, (5, 3)),
               zorder=2)
       ax.text(L + 2.2, y, f"L{i}", fontsize=fs(6.3), va="center",
               color="#2f5d8a")
       for x in grid["emitters"]:
           ax.plot(x, y, "o", ms=fs(2.1), color="#2f5d8a", zorder=6)
   for j, x in enumerate(grid["emitters"], start=1):
       ax.text(x, W + 4, f"E{j}", fontsize=fs(5.8), ha="center", va="center",
               color="#8a8a8a")
   if trellis:
       ax.plot([0, L], [-2, -2], color="#4a4a4a", lw=3.4,
               solid_capstyle="butt", zorder=3)
       ax.text(L / 2, -5.4, trellis, fontsize=fs(6.8), ha="center",
               va="center", color="#4a4a4a")


def draw_plants(ax, plot, plants):
   mid = plot["width"] / 2
   for p in plants:
       c = PALETTE[p["key"]]
       ax.add_patch(Circle((p["x"], p["y"]), max(4, p["r"]), facecolor=c,
                           alpha=0.20, edgecolor=c, lw=0.8, zorder=4))
       ax.plot(p["x"], p["y"], "o", ms=fs(4.4) if p["r"] > 8 else fs(3.0),
               color=c, mec="white", mew=0.7, zorder=7)
   for p in plants:
       if not p["label"]:
           continue
       up = p["y"] < mid
       r = max(4, p["r"])
       yy = p["y"] - r - 2.2 if up else p["y"] + r + 2.2
       ax.text(p["x"], yy, p["label"], fontsize=fs(6.2), ha="center",
               va="bottom" if up else "top", color="#2b2b2b", zorder=9,
               path_effects=HALO)


def draw_bands(ax, plot, bands):
   L = plot["length"]
   for b in bands:
       c = PALETTE[b["key"]]
       ax.add_patch(Rectangle((0, b["y"] - b["half"]), L, b["half"] * 2,
                              facecolor=c, alpha=0.28, edgecolor=c, lw=0.8,
                              zorder=4))
       ax.text(2.0, b["y"] - 1.0, b["title"], fontsize=fs(6.6),
               weight="bold", va="bottom", zorder=9, path_effects=HALO)
       ax.text(2.0, b["y"] + 1.2, b["detail"], fontsize=fs(5.9), va="top",
               color="#333", zorder=9, path_effects=HALO)
       ax.text(L - 2.0, b["y"], b["after"], fontsize=fs(5.8), ha="right",
               va="center", style="italic", color="#8a2f2f", zorder=9,
               path_effects=HALO)


def panel_sections(key, family, season):
   c = counts(key, family)
   S = []

   def sec(lat, title, plants, win_key, extra=None):
       w = WINDOWS.get(win_key)
       lines = [("plants", plants)]
       if w:
           ind, gnd, hv, germ = w
           lines += [("sow", fmt_win(ind) + ", cover" if ind
                      else fmt_win(gnd) + ", direct"),
                     ("in ground", fmt_win(gnd)), ("germ", germ)]
       if extra:
           lines.append(("note", extra))
       S.append(dict(lat=lat, title=title, lines=lines))

   if family == "nightshade":
       h = G.HINTS[(family, PLOTS[key]["geom"])]
       outer = sorted({t[0] for t in h["tom"]})
       inner = sorted({t[0] for t in h["pep"]})
       sec(f'L{" + L".join(map(str, outer))}', "TOMATOES",
           f'{c["tomato"]} plants, strung vertically', "tomatoes",
           "determinates nearest E1")
       sec(f'L{" + L".join(map(str, inner))}', "PEPPERS",
           f'{c["pepper"]} plants, annual here', "peppers",
           "the collection proper is in pots, station 4")
       sec(f'L{h["herb"][0]}', "BASIL + MARIGOLD",
           f'{c["basil"]} basil, {c["marigold"]} marigold', "basil",
           "Nemagold is nematode suppressive")
   elif family == "cucurbit":
       sec("L1", "CUCUMBERS", f'{c["cucumber"]} trellised',
           "cucumbers, 1st", "2nd flush mid Jun into the gaps")
       sec("L1", "MINI PUMPKINS", f'{c["mini"]} trellised', "mini pumpkins")
       sec("inner", "SUMMER SQUASH", f'{c["squash"]} plants',
           "summer squash", "pulled early Jul, grey with mildew")
   elif family == "brassica":
       h = G.HINTS[(family, PLOTS[key]["geom"])]
       sec("outer", "KALE + BROCCOLI", f'{c["large"]} plants', "kale",
           G.brassica_mix(c["large"]))
       sec(f'L{h["spin"][0]}', "SPINACH", f'{c["spinach"]} positions',
           "spinach", "NOTHING after mid Jan, it bolts")
       sec(f'L{" + L".join(map(str, h["grn"]))}', "LETTUCE + GREENS",
           f'{c["greens"]} positions', "lettuce + greens",
           "resown every 3 weeks")
   elif family == "allium" and season == "summer":
       sec("L2 + L3", "SWEET POTATOES", "8 slips at 18 in",
           "sweet potatoes", "vines may run west into shade")
       sec("L1", "BUSH BEANS", "sown Mar 20", "bush beans")
   else:
       for b in G.allium_bands(key):
           wk = {"pea": "snap peas", "garlic": "garlic", "onion": "onions",
                 "root": "carrots + radish"}[b["key"]]
           sec(f'L{b["lat"]}', b["title"],
               f'{b["n"]} {"cloves" if b["crop"] == "garlic" else "plants"}',
               wk, b["after"])
   return S


def draw_panel(fig, sections, x0=0.655):
   y, lh = 0.795, 0.0164
   fig.text(x0, y, "WHAT IS IN EACH SECTION", fontsize=fs(7.4),
            weight="bold", color="#6b4f2a", va="center")
   y -= lh * 1.4
   for s in sections:
       fig.text(x0, y, f'{s["lat"]} . {s["title"]}', fontsize=fs(6.8),
                weight="bold", va="center", color="#222")
       y -= lh
       for label, val in s["lines"]:
           fig.text(x0 + 0.008, y, label, fontsize=fs(5.8), va="center",
                    color="#8a8a8a")
           fig.text(x0 + 0.066, y, val, fontsize=fs(5.8), va="center",
                    color="#333")
           y -= lh
       y -= 0.006


NOTES = {
   "nightshade": [
       "Transplant tomatoes Mar 15 and string them vertically rather than "
       "caging. Outer rows are staggered so diagonal neighbours gain "
       "clearance, which is what makes vertical training possible.",
       "Peppers on the heat mat mid Jan at 80 to 85 F, out Apr 10. These are "
       "annuals here, pulled each autumn, so they never anchor the bed.",
       "Expect blossom drop above 92 F in July. The cherries and the "
       "determinate heat setters carry you through the gap.",
       "Cover crop broadcast over the whole plot in November, chopped in at "
       "flowering in mid February."],
   "cucurbit": [
       "Cucumbers trellised on the north edge, so they cost almost no floor "
       "area. Direct sow May 1, second flush mid Jun, and pick picklers "
       "daily at 3 to 4 in.",
       "Mini pumpkins share the trellis, sown Jul 1 to 10 for mid Oct. They "
       "are the insurance policy against whatever the annex does.",
       "Squash sown May 1 and pulled in early Jul when it is grey with "
       "mildew. The sprawling melons and carving pumpkins are in the annex.",
       "Water stress makes cucumbers genuinely bitter through cucurbitacin, "
       "so a skipped event here has a flavour cost, not just a yield cost."],
   "brassica": [
       "Transplant Oct 15 to 25 with row cover from the start, since aphids "
       "and cabbage worms do not take a winter off here.",
       "Weighted to kale, which crops from December through April and then "
       "keeps going. Cauliflower is the one single harvest crop, clustered "
       "at E1 so January gaps replant without disturbance.",
       "Spinach in short successions mid Oct to mid Jan only. Only the first "
       "two positions are labelled, since a whole row of labels smears.",
       "Summer: buckwheat from late May, cowpeas from July, or Malabar "
       "spinach on the trellis, which thrives when true spinach cannot."],
   "allium": [
       "Garlic and peas late Oct, onions Nov, carrots from Sep. Softneck "
       "garlic only, since 10b will not vernalise a hardneck properly.",
       "Short day onions only, because long day types grow handsome tops at "
       "this latitude and never bulb.",
       "Drawn as bands because true row spacing merges into a solid stripe. "
       "Radish between the carrot sowings marks the rows while the carrots "
       "are still deliberating.",
       "Every band is a spring finisher, which is exactly why the whole zone "
       "can dry down together in the second week of May."],
}


def plot_map(outdir, year, key, season="winter"):
   plot, grid = PLOTS[key], GRIDS[key]
   fam = ROTATION[year][key]
   plants, bands = layout(key, fam, season)
   st = plot["station"]
   tag = " . JUNE TO OCTOBER" if season == "summer" else ""
   fig = page()
   header(fig, f'YEAR {year} . {plot["name"]} ({key}) . '
               f'{FAMILY_LABEL[fam].upper()}{tag}',
          f'station {st}, {STATION_LABEL[st]} . {ROTATION_YEARS[year]} . '
          f'{area_sqft(plot):.1f} sq ft . {FREQ_CLASS[fam]} rooted')
   pad = 12
   box_w = (0.615 - m()) if PANEL else (1 - 2 * m())
   ax, s = fit_axes(fig, (m(), 0.22, box_w, 0.60),
                    (-pad, plot["length"] + pad), (-pad, plot["width"] + pad),
                    denom=SCALE_DENOM)
   trellis = {"nightshade": "T R E L L I S  ( tomatoes )",
              "cucurbit": "T R E L L I S  ( cucumbers + minis )",
              "allium": "T R E L L I S  ( snap peas )",
              "brassica": "T R E L L I S  ( Malabar, summer )"}[fam]
   draw_frame(ax, plot, grid, trellis if season == "winter" else "north edge")
   draw_plants(ax, plot, plants)
   draw_bands(ax, plot, bands)
   if PANEL:
       draw_panel(fig, panel_sections(key, fam, season))
   sched = station_schedule(st, year)
   peak = max(sched, key=lambda t: t[0] * t[1])
   fig.text(m(), 0.845,
            f'{plot["width"]:.1f} x {plot["length"]:.0f} in . '
            f'{len(grid["laterals"])} laterals at '
            f'{grid["lateral_spacing"]:.2f} in . {grid["count"]} emitters at '
            f'{EMITTER_GPH:g} gph . {grid["gph"]:.1f} gph . {grid["rate"]} '
            f'in/hr . {grid["min_quarter"]} min = 0.25 in . peak '
            f'{peak[0]:g} x {peak[1]} min . drawn 1:{1 / s:.0f}',
            fontsize=fs(7.0), va="top", color="#444")
   footer(fig, NOTES[fam])
   if plants:
       keys = sorted({p["key"] for p in plants})
       fig.legend(handles=[Line2D([], [], marker="o", ls="none", ms=fs(6),
                                  mfc=PALETTE[k], mec="white", label=k)
                           for k in keys], loc="lower left",
                  bbox_to_anchor=(m(), m(0.30, "y") + 0.10), ncol=5,
                  frameon=False, fontsize=fs(6.3), handletextpad=0.4,
                  columnspacing=1.1)
   save(fig, outdir, f'y{year}_{key}_{fam}'
                     f'{"_summer" if season == "summer" else ""}')


def garden_plan(outdir, year):
   fig = page()
   header(fig, f"ROTATION YEAR {year} . PLOT MAP",
          f'{ROTATION_YEARS[year]} . 125.6 sq ft in beds plus a 160 sq ft '
          f'annex . Bed L halves selected by manual ball valve')
   ax, s = fit_axes(fig, (m(), 0.15, 1 - 2 * m(), 0.68),
                    (-16, 176), (-20, 260))
   ax.add_patch(Rectangle((0, -116), 116, 46, facecolor=PALETTE["vine"],
                          alpha=0.14, edgecolor="#6b4f2a", lw=1.2,
                          ls=(0, (4, 3))))
   ax.text(58, -93, "VINE ANNEX . STATION 5 . 8 x 20 ft, shown reduced\n"
           "pumpkins, watermelon, cantaloupe, corn . see its own sheet",
           fontsize=fs(6.6), ha="center", va="center", color="#6b4f2a")
   ax.add_patch(Rectangle((0, 0), 144, 51, facecolor="#efe7d8",
                          edgecolor="#6b4f2a", lw=1.8))
   boxes = [("L1", 0, 0, 72, 51), ("L2", 72, 0, 72, 51),
            ("M", 0, 87, 116, 56.5), ("S", 0, 179, 79, 53)]
   for key, x, y, w, h in boxes:
       fam = ROTATION[year][key]
       ax.add_patch(Rectangle((x, y), w, h, facecolor=FAMILY_COLOUR[fam],
                              alpha=0.22, edgecolor="#6b4f2a", lw=1.8))
       ax.text(x + w / 2, y + h / 2 - 5, f'{key} . {FAMILY_LABEL[fam].upper()}',
               fontsize=fs(8), weight="bold", ha="center", va="center")
       ax.text(x + w / 2, y + h / 2 + 3,
               f'station {PLOTS[key]["station"]} . {w:.0f} x {h:.1f} in . '
               f'{area_sqft(PLOTS[key]):.1f} sq ft', fontsize=fs(6.8),
               ha="center", va="center", color="#444")
       ax.text(x + w / 2, y + h / 2 + 9,
               f'{len(GRIDS[key]["laterals"])} laterals . '
               f'{GRIDS[key]["rate"]} in/hr', fontsize=fs(6.4),
               ha="center", va="center", color="#666")
   ax.plot([72, 72], [0, 51], color="#8a2f2f", lw=2.4)
   ax.text(72, -6, "two 3/4 in ball valves, above grade on the outside face, "
           "tagged by crop", fontsize=fs(6.8), ha="center", va="center",
           color="#8a2f2f")
   for yy in (70, 162):
       ax.text(58, yy, "path", fontsize=fs(7.5), style="italic",
               color="#8a8a8a", ha="center")
   ax.text(158, 208, "shaded:\nsummer\nlettuce", fontsize=fs(6.4),
           ha="center", va="center", color="#5c8a6a")
   ax.add_patch(FancyArrow(166, 20, 0, -26, width=0.9, head_width=6,
                           head_length=7, color="#333"))
   ax.text(166, 28, "N", fontsize=fs(11), weight="bold", ha="center",
           va="center")
   footer(fig, [
       "PVC stubs rise inside each box at the narrow end, turn over the wall "
       "in a U, and the ball valves sit on the outside face where you can "
       "see them from the path.",
       "Bed L: the near valve feeds the header at that end, the far valve "
       "feeds 6 ft of blank poly to the second header at the 72 in line, "
       "tucked along one long side.",
       "Bed M carries FIVE laterals at 11.00 in, so it applies water faster "
       "than the other beds, 1.05 in/hr against 0.96 and 0.92, and takes "
       "correspondingly shorter runs.",
       "Shaded west end of Bed L takes the sweet potato vines in allium "
       "years. Shaded ground north of Bed S grows summer lettuce, the one "
       "thing a 10b July otherwise forbids."])
   save(fig, outdir, f"y{year}_00_plan")


def fig_annex(outdir):
   g = GRIDS["annex"]
   W, L = ANNEX["width"], ANNEX["length"]
   fig = page()
   header(fig, "THE VINE ANNEX . STATION 5 . 8 x 20 FT, 160 SQ FT",
          "Open ground north of Bed M, outside the rotation, so this sheet "
          "is the same every year. Sprawlers and the corn block live here.")
   pad = 16
   ax, s = fit_axes(fig, (m(), 0.26, 1 - 2 * m(), 0.54),
                    (-pad, L + pad), (-pad, W + pad),
                    denom=SCALE_DENOM, invert=False)
   ax.add_patch(Rectangle((0, 0), L, W, facecolor="#f1ece0",
                          edgecolor="#6b4f2a", lw=1.8, zorder=0))
   ax.plot([0, 116], [-3, -3], color="#4a4a4a", lw=3.4,
           solid_capstyle="butt", zorder=3)
   ax.text(58, -10, "BED M north edge . cucumber trellis . ~2 ft of shade in "
           "Aug, ~6 ft in Oct", fontsize=fs(6.4), ha="center", va="center",
           color="#4a4a4a")
   ax.plot([-5, -5], [0, W], color="#2f5d8a", lw=2.6, zorder=2)
   ax.text(-10.5, W / 2, "HEADER", rotation=90, fontsize=fs(6), va="center",
           ha="center", color="#2f5d8a")
   for i, y in enumerate(g["laterals"], start=1):
       ax.plot([-5, L], [y, y], color="#2f5d8a", lw=0.8, ls=(0, (5, 3)),
               zorder=2)
       ax.text(L + 3, y, f"L{i}", fontsize=fs(5.8), va="center",
               color="#2f5d8a")
       for x in g["emitters"]:
           ax.plot(x, y, "o", ms=fs(1.7), color="#2f5d8a", zorder=6)
   c = CORN
   cw, ch = (c["cols"] - 1) * c["step"], (c["rows"] - 1) * c["step"]
   ax.add_patch(Rectangle((c["x0"] - 6, c["y0"] - 6), cw + 12, ch + 12,
                          facecolor=PALETTE["corn"], alpha=0.16,
                          edgecolor=PALETTE["corn"], lw=1.0, zorder=3))
   for i in range(c["cols"]):
       for j in range(c["rows"]):
           ax.plot(c["x0"] + i * c["step"], c["y0"] + j * c["step"], "o",
                   ms=fs(3.5), color=PALETTE["corn"], mec="white", mew=0.6,
                   zorder=7)
   ax.text(c["x0"] + cw / 2, c["y0"] + ch + 13,
           f'SWEET CORN . {c["cols"] * c["rows"]} plants, {c["step"]} in '
           f'grid . {c["variety"]}', fontsize=fs(6.8), ha="center",
           va="bottom", weight="bold", color="#8a6d1f", zorder=9,
           path_effects=HALO)
   for h in ANNEX_HILLS:
       col = PALETTE[h["key"]]
       ax.add_patch(Circle((h["x"], h["y"]), h["r"], facecolor=col,
                           alpha=0.17, edgecolor=col, lw=0.8, zorder=4))
       ax.plot(h["x"], h["y"], "o", ms=fs(5.8), color=col, mec="white",
               mew=0.9, zorder=8)
       ax.text(h["x"], h["y"] + 5.5, h["label"], fontsize=fs(6.6),
               ha="center", va="bottom", weight="bold", zorder=9,
               path_effects=HALO)
       ax.text(h["x"], h["y"] - 6, h["note"], fontsize=fs(5.7), ha="center",
               va="top", style="italic", color="#6b4f2a", zorder=9,
               path_effects=HALO)
       nl = (h["dx"] ** 2 + h["dy"] ** 2) ** 0.5
       ax.annotate("", xytext=(h["x"], h["y"]),
                   xy=(h["x"] + 30 * h["dx"] / nl, h["y"] + 30 * h["dy"] / nl),
                   arrowprops=dict(arrowstyle="-|>", lw=1.2, color=col,
                                   alpha=0.75), zorder=5)
   ax.add_patch(FancyArrow(L + 8, W - 28, 0, 22, width=0.9, head_width=6,
                           head_length=7, color="#333", zorder=9))
   ax.text(L + 8, W + 1, "N", fontsize=fs(10), weight="bold", ha="center",
           va="bottom")
   fig.text(m(), 0.845,
            f'{W:.0f} x {L:.0f} in . 8 laterals at 12 in . {g["count"]} '
            f'emitters at {EMITTER_GPH:g} gph . {g["gph"]:.1f} gph . '
            f'{g["rate"]} in/hr . {g["min_quarter"]} min = 0.25 in . '
            f'1 in = 100 gal . drawn 1:{1 / s:.0f}', fontsize=fs(7.2),
            va="top", color="#444")
   footer(fig, [
       "Six sprawling hills and a corn block in 160 sq ft is crowded, and it "
       "works only because the canopies are meant to leave. Arrows show the "
       "trained run direction, out of the annex.",
       "Corn mid Apr for a Jul harvest, as a 5 x 5 block rather than a row, "
       "since a row pollinates badly. Melon hills May 1, pumpkins direct "
       "sown Jul 1 to 10 for mid to late Oct.",
       "Melons: cut water sharply for the last ten days before harvest, "
       "since that is where the sugar comes from. Skip two events and hand "
       "water the pumpkin seedlings.",
       "Corn is the exception, unforgiving of stress at tassel and silk, so "
       "water through the melon dry down and accept slightly less sweet "
       "melons rather than a poor set.",
       "Not rotated, so: move every hill within the annex each season, "
       "cowpeas across the whole thing each winter, and clear all vine "
       "debris in Nov, because squash bugs overwinter in it."])
   save(fig, outdir, "04_annex")


# ======================================================================
# CALENDARS
# ======================================================================


def fig_cover(outdir):
   fig = page()
   fig.text(m(), 1 - m(0.5, "y"), "GARDEN PLAN", fontsize=fs(26),
            weight="bold", va="top")
   fig.text(m(), 1 - m(0.95, "y"), "Three raised beds on a four year "
            "rotation grouped by water need, plus a fixed vine annex",
            fontsize=fs(11.5), va="top", color="#555")
   fig.text(m(), 1 - m(1.22, "y"), f'Altadena, California 91104 . USDA zone '
            f'10b . generated {datetime.date.today().isoformat()}',
            fontsize=fs(8.5), va="top", color="#888")
   line(fig, [m(), 1 - m()], [0.845, 0.845], color="#6b4f2a", lw=1.8)
   rows = [("GROUND", None)]
   for k in PLOT_ORDER:
       p, g = PLOTS[k], GRIDS[k]
       rows.append((k, f'{p["width"]:.1f} x {p["length"]:.0f} in . '
                       f'{area_sqft(p):.1f} sq ft . '
                       f'{len(g["laterals"])} laterals . {g["count"]} '
                       f'emitters . station {p["station"]}'))
   ga = GRIDS["annex"]
   rows += [("annex", f'96 x 240 in . 160.0 sq ft . 8 laterals . '
                      f'{ga["count"]} emitters . station 5 . not rotated'),
            ("containers", f'{CONTAINERS["count"]} pots at '
                           f'{CONTAINERS["size_gal"]} gal . station 4 . '
                           f'the overwintering pepper collection'),
            ("total", "286 sq ft planted . roughly 6,000 gallons a year"),
            ("RATES, DERIVED PER PLOT", None)]
   for k in PLOT_ORDER + ["annex"]:
       g = GRIDS[k]
       rows.append((k, f'{g["lateral_spacing"]:.2f} in lateral spacing . '
                       f'{g["rate"]} in/hr . {g["min_quarter"]} min = '
                       f'0.25 in . {g["min_one"]} min = 1 in'))
   y = 0.79
   for label, detail in rows:
       if detail is None:
           y -= 0.008
           fig.text(m(), y, label, fontsize=fs(9.5), weight="bold",
                    color="#6b4f2a", va="top")
           y -= 0.031
       else:
           fig.text(m() + 0.02, y, label, fontsize=fs(8.8), weight="bold",
                    va="top", color="#333")
           fig.text(m() + 0.16, y, detail, fontsize=fs(8.8), va="top",
                    color="#555")
           y -= 0.027
   footer(fig, [
       "Bed M carries five laterals at 11.00 in, so it applies water faster "
       "than the others and takes shorter runs. Rates are derived from "
       "lateral spacing rather than assumed uniform.",
       "One 4:30 am start with stations running sequentially, finishing "
       "before seven, which keeps foliage dry and reduces mildew pressure.",
       "The one instruction no script can remember for you: shut the ball "
       "valve on the allium half of Bed L on the tenth of May. Wet bulbs "
       "will not cure and will rot in storage."])
   save(fig, outdir, "00_cover")


def fig_rotation(outdir):
   fig = page()
   header(fig, "FOUR YEAR ROTATION",
          "Heavy feeder, restorative, heavy feeder, resting. Each family "
          "visits each plot once, so repeat spacing is maximal.")
   left, top = 0.185, 0.74
   w, h = (1 - m() - left) / 4, 0.115
   for j, k in enumerate(PLOT_ORDER):
       fig.text(left + (j + 0.5) * w, top + 0.038,
                f'{k}\nstation {PLOTS[k]["station"]} . '
                f'{GRIDS[k]["rate"]} in/hr', fontsize=fs(7.8),
                weight="bold", ha="center", va="center")
   for i in sorted(ROTATION):
       y = top - i * h
       fig.text(left - 0.012, y + h / 2, f'year {i}\n{ROTATION_YEARS[i]}',
                fontsize=fs(7.6), ha="right", va="center")
       for j, k in enumerate(PLOT_ORDER):
           fam = ROTATION[i][k]
           rect(fig, left + j * w, y, w, h, facecolor=FAMILY_COLOUR[fam],
                alpha=0.26, edgecolor="white", lw=2)
           fig.text(left + (j + 0.5) * w, y + h / 2,
                    f'{FAMILY_LABEL[fam]}\n{FREQ_CLASS[fam]} rooted',
                    fontsize=fs(7.2), ha="center", va="center",
                    weight="bold")
   footer(fig, [
       "Frequency class, not volume, is what constrains a shared valve: "
       "rooting depth dictates how often, while volume is trivially "
       "adjusted with run time.",
       "L1 and L2 share one solenoid, so their pairings were chosen to be "
       "compatible in every year: nightshade with allium, allium with "
       "cucurbit, then two years where a winter tenant shares with a summer "
       "tenant and the halves barely overlap.",
       "The annex sits outside this table and grows the sprawlers every "
       "year, which is what makes pumpkins and melons annual rather than "
       "one year in four.",
       "Peppers overwinter as perennials in 10b and would anchor a bed, so "
       "the collection lives in pots on station 4 and only annual peppers go "
       "in the ground."])
   save(fig, outdir, "01_rotation")


def fig_schedule(outdir, year):
   sts = [1, 2, 3, 4, 5]
   data = {st: station_schedule(st, year) for st in sts}
   weekly = {st: [e * mn for e, mn in data[st]] for st in sts}
   hi = max(max(v) for v in weekly.values()) or 1
   norm, cmap = Normalize(0, hi), plt.get_cmap("YlGnBu")
   fig = page()
   header(fig, f"DRIP SCHEDULE . ROTATION YEAR {year}",
          f'{ROTATION_YEARS[year]} . one 4:30 am start, stations sequential '
          f'. minutes derived from each plot\'s own application rate')
   left, top = 0.165, 0.755
   w, h = (1 - m() - left) / 12, 0.082
   for k, mon in enumerate(MONTHS):
       fig.text(left + (k + 0.5) * w, top + 0.024, mon, fontsize=fs(8.6),
                weight="bold", ha="center", va="center")
   for i, st in enumerate(sts):
       y = top - (i + 1) * h
       keys = [k for k in PLOT_ORDER if PLOTS[k]["station"] == st]
       fams = " + ".join(ROTATION[year][k][0].upper() for k in keys) \
           or ("pots" if st == 4 else "fixed")
       rate = (GRIDS[keys[0]]["rate"] if keys else
               GRIDS["annex"]["rate"] if st == 5 else None)
       extra = f'{rate} in/hr' if rate else "volumetric"
       fig.text(left - 0.012, y + h / 2,
                f'station {st}\n{STATION_LABEL[st]}\n{fams} . {extra}',
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
               fig.text(cx, y + h * 0.63, freq_label(ev), fontsize=fs(6.2),
                        ha="center", va="center", color=col)
               fig.text(cx, y + h * 0.32, f"{mn} min", fontsize=fs(7.8),
                        ha="center", va="center", weight="bold", color=col)
   y = top - 6 * h - 0.014
   fig.text(left - 0.012, y + h / 2, "BALL VALVES\nBed L, manual",
            fontsize=fs(6.8), ha="right", va="center", weight="bold",
            color="#8a2f2f")
   for k in range(12):
       state = VALVES[year][k]
       shut = "none" if state == "both" else ("L1" if state == "L2" else "L2")
       rect(fig, left + k * w, y, w, h,
            facecolor="#f6f2ea" if state == "both" else "#f2dcdc",
            edgecolor="white", lw=1.5)
       cx = left + (k + 0.5) * w
       fig.text(cx, y + h * 0.66, f"open {state}", fontsize=fs(6.2),
                ha="center", va="center", color="#333")
       fig.text(cx, y + h * 0.42, f"shut {shut}", fontsize=fs(6.2),
                ha="center", va="center", color="#8a2f2f", weight="bold")
       note = VALVE_NOTES.get((year, k))
       if note:
           fig.text(cx, y + h * 0.16, note, fontsize=fs(5.4), ha="center",
                    va="center", color="#8a2f2f", style="italic")
   footer(fig, [
       "Station 1 is the envelope of whichever Bed L halves are open: higher "
       "frequency, longer event. PC emitters hold their rate regardless, so "
       "closing one half does not change the other.",
       "Station 4 is volumetric, since a pot has no meaningful area. Seven "
       "events a week in July is correct for a 15 gal pot in full sun.",
       "Rain: half an inch buys five days for shallow plots, an inch buys a "
       "week. Heat: add a quarter above 95 F, a 10 min midday pulse for "
       "greens above 105 F. Santa Ana: add twenty to thirty percent.",
       "Turn Rachio wind skip OFF, since drift is irrelevant to buried "
       "emitters and a dry wind is when demand is highest.",
       "The test that overrules every number here is a finger three inches "
       "into the soil the morning after. Dry means lengthen events rather "
       "than adding more of them."])
   save(fig, outdir, f"y{year}_schedule")


def fig_planting(outdir):
   fig = page()
   header(fig, "PLANTING CALENDAR . ALL CROPS",
          "Rotation year, Oct to Sep . dashed = under cover, solid = in the "
          "ground, hatched = cropping . germination at right")
   rows = sum(len(v) for _, v in WIN_GROUP)
   left, top = 0.185, 0.795
   w = (1 - m() - left - 0.155) / 12
   rh = min(0.0205, 0.60 / (rows + 6))
   bottom = top - rh * (rows + 7)
   for k, mon in enumerate(MONTHS):
       fig.text(left + (k + 0.5) * w, top + 0.019, mon, fontsize=fs(7.8),
                weight="bold", ha="center", va="center")
       line(fig, [left + k * w] * 2, [top + 0.008, bottom], color="#e6e6e6",
            lw=0.7)
   line(fig, [left + 12 * w] * 2, [top + 0.008, bottom], color="#e6e6e6",
        lw=0.7)
   y = top
   for grp, names in WIN_GROUP:
       y -= 0.010
       col = FAMILY_COLOUR.get(grp, "#9aab7a")
       fig.text(m(), y, grp.upper(), fontsize=fs(6.8), weight="bold",
                color=col, va="center")
       y -= rh
       for name in names:
           ind, gnd, hv, germ = WINDOWS[name]
           fig.text(left - 0.008, y, name, fontsize=fs(6.2), ha="right",
                    va="center")
           for s0, e0 in spans(ind):
               rect(fig, left + s0 * w, y - rh * .27, (e0 - s0) * w, rh * .54,
                    facecolor=col, alpha=0.18, edgecolor=col, lw=0.6,
                    ls=(0, (2, 1.5)))
           for s0, e0 in spans(gnd, min_w=0.18):
               rect(fig, left + s0 * w, y - rh * .32, (e0 - s0) * w, rh * .64,
                    facecolor=col, alpha=0.88, edgecolor="none")
           for s0, e0 in spans(hv):
               rect(fig, left + s0 * w, y - rh * .27, (e0 - s0) * w, rh * .54,
                    facecolor=col, alpha=0.28, edgecolor=col, lw=0.6,
                    hatch="////")
           fig.text(left + 12 * w + 0.006, y, germ, fontsize=fs(5.5),
                    va="center", color="#555")
           y -= rh
   footer(fig, [
       "Germination assumes the stated soil temperature. A seed absent after "
       "twice its window has failed, and knowing that in week two rather "
       "than week six is the whole point of the column.",
       "Carrots at 14 to 21 days are the slowest here and need the surface "
       "damp throughout, so two five minute pulses daily. Peppers at up to "
       "21 days need 80 to 85 F, not merely a warm room.",
       "Two dates will not move: spinach sown after mid January bolts before "
       "it is worth eating, and garlic still being watered on May 10 will "
       "not cure and will rot in storage.",
       "For cover crops the hatched bar means chop in at flowering rather "
       "than harvest."])
   save(fig, outdir, "02_planting")


def fig_tasks(outdir, year):
   rows = tasks(year)
   fig = page()
   plots = " . ".join(f'{k} {ROTATION[year][k]}' for k in PLOT_ORDER)
   header(fig, f"PLANTING TASKS . YEAR {year}",
          f'{ROTATION_YEARS[year]} . every sowing, transplant and '
          f'cultivation with its unit count . {plots}')
   x_when, x_act, x_units = m(), m() + 0.105, m() + 0.435
   y = 0.795
   lh = min(0.0215, 0.605 / max(1, len(rows)))
   for lbl, x in (("WHEN", x_when), ("ACTION", x_act), ("UNITS", x_units)):
       fig.text(x, y, lbl, fontsize=fs(7), weight="bold", color="#6b4f2a")
   y -= lh * 1.25
   for i, r in enumerate(rows):
       if i % 2 == 0:
           rect(fig, x_when - 0.006, y - lh * 0.42, 1 - 2 * m() + 0.012,
                lh * 0.9, facecolor="#f4f1ea", edgecolor="none")
       emph = r["kind"] == "valve"
       fig.text(x_when, y, r["when"], fontsize=fs(6.5), va="center",
                weight="bold", color="#6b4f2a")
       fig.text(x_act, y, r["action"], fontsize=fs(6.5), va="center",
                color=KIND_COLOUR.get(r["kind"], "#555"),
                weight="bold" if emph else "normal")
       fig.text(x_units, y, r["units"], fontsize=fs(6.3), va="center",
                color="#8a2f2f" if emph else "#333")
       y -= lh
   footer(fig, [
       "Tray sowings carry a surplus of about a third, because germination "
       "is rarely total and you want the best seedlings rather than all of "
       "them. Thin without sentiment.",
       "Successions: " + " . ".join(f'{c}, {w}, every {iv}'
                                    for c, w, iv in SUCCESSIONS),
       "Germination pulses no controller handles: " +
       " . ".join(f'{c}, {p}, {d}' for c, p, d in PULSES),
       "Colour key: purple starts under cover, green direct sown, red "
       "planted out, blue purchased, olive cultivation, dark red irrigation."])
   save(fig, outdir, f"y{year}_tasks")


def fig_seed_order(outdir):
   fig = page()
   header(fig, "SEED AND ORDER LIST . WORST CASE ACROSS FOUR YEARS",
          "What a season actually consumes, taking the largest of the four "
          "rotation years so one order covers any of them")
   y, lh = 0.79, 0.0255
   for lbl, x in (("CROP", m()), ("PER SEASON", m() + 0.30),
                  ("PACKET GUIDANCE", m() + 0.52)):
       fig.text(x, y, lbl, fontsize=fs(7), weight="bold", color="#6b4f2a")
   y -= lh * 1.2
   for i, (name, qty, note) in enumerate(SEED_ORDER):
       if i % 2 == 0:
           rect(fig, m() - 0.006, y - lh * 0.4, 1 - 2 * m() + 0.012,
                lh * 0.85, facecolor="#f4f1ea", edgecolor="none")
       urgent = note.startswith("REBUY") or note.startswith("ORDER")
       fig.text(m(), y, name, fontsize=fs(7.6), va="center", weight="bold")
       fig.text(m() + 0.30, y, qty, fontsize=fs(7.4), va="center",
                color="#333")
       fig.text(m() + 0.52, y, note, fontsize=fs(7.2), va="center",
                color="#8a2f2f" if urgent else "#555",
                weight="bold" if urgent else "normal")
       y -= lh
   footer(fig, [
       "Store the remainder in a sealed jar in the refrigerator with a "
       "desiccant packet. Tomato, brassica and squash seed stays viable four "
       "or five years that way.",
       "Onion and spinach seed does not, so buy those fresh each September. "
       "Carrot is intermediate and worth replacing annually too.",
       "Garlic and sweet potato slips are ordered rather than sown, and both "
       "sell out early: softneck garlic by early October, slips by February.",
       "Year four is the year to order a full pound of garlic, since the "
       "allium plot lands in Bed M and wants 38 cloves rather than 24."])
   save(fig, outdir, "03_seed_order")


def fig_schematic(outdir):
   sts = []
   for st in (1, 2, 3):
       keys = [k for k in PLOT_ORDER if PLOTS[k]["station"] == st]
       n = sum(GRIDS[k]["count"] for k in keys)
       gph = sum(GRIDS[k]["gph"] for k in keys)
       detail = (f'ONE solenoid, {len(keys)} plots\n{len(keys)} manual ball '
                 f'valves at the bed,\ntagged by crop'
                 if len(keys) > 1 else
                 f'{len(GRIDS[keys[0]]["laterals"])} laterals, '
                 f'{len(GRIDS[keys[0]]["emitters"])} emitters each\n'
                 f'{GRIDS[keys[0]]["rate"]} in/hr, '
                 f'{GRIDS[keys[0]]["min_quarter"]} min = 0.25 in')
       sts.append(dict(st=st, label=STATION_LABEL[st], n=n,
                       gph=round(gph, 1), detail=detail,
                       split=keys if len(keys) > 1 else None))
   sts.append(dict(st=4, label="containers", n=None,
                   gph=CONTAINERS["count"] * CONTAINERS["dripper_gph"],
                   detail=f'{CONTAINERS["count"]} pots, 1/4 in microtube\n'
                          f'point emitters at '
                          f'{CONTAINERS["dripper_gph"]:g} gph\n'
                          f'volumetric, daily in high summer', split=None))
   ga = GRIDS["annex"]
   sts.append(dict(st=5, label="annex", n=ga["count"], gph=ga["gph"],
                   detail=f'8 laterals, 20 emitters each\n15 ft buried run, '
                          f'solenoid AT the annex\nso the run sits '
                          f'depressurised when idle', split=None))
   total = sum(s["gph"] for s in sts)
   peak = max(s["gph"] for s in sts)
   fig = page()
   header(fig, "HYDRAULIC SCHEMATIC . NOT TO SCALE",
          f'What feeds what . 5 of 16 stations used . {EMITTER_GPH:g} gph '
          f'emitters at 12 in . Altadena 91104')
   ax = fig.add_axes([m(), 0.17, 1 - 2 * m(), 0.66])
   ax.set_xlim(0, 100)
   ax.set_ylim(0, 64)
   ax.axis("off")

   # Unequal aspect, so every dot is a marker sized in points rather than
   # a Circle patch, which would render as an ellipse.
   def box(x, y, w, h, title, sub, colour, sp=5.4):
       ax.add_patch(Rectangle((x, y), w, h, facecolor=colour, alpha=0.16,
                              edgecolor=colour, lw=1.3, zorder=3))
       ax.text(x + w / 2, y + h * 0.70, title, fontsize=fs(6.6),
               weight="bold", ha="center", va="center", zorder=5,
               linespacing=1.25)
       ax.text(x + w / 2, y + h * 0.30, sub, fontsize=fs(sp), ha="center",
               va="center", color="#444", zorder=5, linespacing=1.35)

   def arrow(x0, y0, x1, y1, colour="#2f5d8a", lw=1.5):
       ax.annotate("", xytext=(x0, y0), xy=(x1, y1),
                   arrowprops=dict(arrowstyle="-|>", lw=lw, color=colour,
                                   shrinkA=0, shrinkB=0), zorder=2)

   gap = 1.5
   bw = (100 - gap * (len(HEAD) - 1)) / len(HEAD)
   bh, ytop = 9.0, 50.0
   for i, (t, s, c) in enumerate(HEAD):
       x = i * (bw + gap)
       box(x, ytop, bw, bh, t, s, c)
       if i:
           arrow(x - gap, ytop + bh / 2, x, ytop + bh / 2)
   ax.text(0, ytop + bh + 2.6, "HEAD ASSEMBLY . order matters . the filter "
           "protects the valve diaphragms as well as the emitters",
           fontsize=fs(6.5), color="#555", va="bottom")
   x_last = (len(HEAD) - 1) * (bw + gap) + bw / 2
   arrow(x_last, ytop, x_last, 43.0)
   ax.plot([72, x_last], [43, 43], color="#2f5d8a", lw=1.5, zorder=2)
   box(28, 34, 44, 9, "MANIFOLD . 16 STATION RACHIO",
       f'5 used, 11 spare . stations run SEQUENTIALLY, so the supply\nnever '
       f'sees more than {peak:.1f} gph . one 4:30 am start\nFlex Daily . '
       f'wind skip OFF . rain skip ON at 0.25 in', "#2f5d8a", sp=5.7)
   sw = (100 - 2.0 * (len(sts) - 1)) / len(sts)
   y_bus, y_sta, sh_box = 30.5, 13.0, 13.5
   ax.plot([sw / 2, 100 - sw / 2], [y_bus, y_bus], color="#2f5d8a", lw=1.5,
           zorder=2)
   ax.plot([50, 50], [34, y_bus], color="#2f5d8a", lw=1.5, zorder=2)
   for i, s in enumerate(sts):
       x = i * (sw + 2.0)
       cx = x + sw / 2
       arrow(cx, y_bus, cx, y_sta + sh_box)
       ax.plot(cx, y_bus, marker="o", ms=fs(4.0), mfc="white", mec="#8a2f2f",
               mew=1.2, zorder=4)
       ax.text(cx + 2.0, y_bus + 1.7, f'S{s["st"]}', fontsize=fs(6.0),
               weight="bold", color="#8a2f2f", zorder=5)
       em = f'{s["n"]} emitters\n' if s["n"] else ""
       lowf = " LOW FLOW" if s["gph"] / 60 < 0.5 else ""
       box(x, y_sta, sw, sh_box, f'STATION {s["st"]}\n{s["label"]}',
           f'{em}{s["gph"]:g} gph, {s["gph"] / 60:.2f} gpm{lowf}\n'
           f'{s["detail"]}', "#6b8a5a", sp=5.1)
       if s["split"]:
           ax.plot([cx, cx], [y_sta, 8.0], color="#8a2f2f", lw=1.5, zorder=2)
           half = sw / 2 - 2.0
           ax.plot([cx - half, cx + half], [8, 8], color="#8a2f2f", lw=1.5,
                   zorder=2)
           for j, key in enumerate(s["split"]):
               xx = cx - half + j * (2 * half / max(1, len(s["split"]) - 1))
               arrow(xx, 8.0, xx, 4.8, colour="#8a2f2f")
               ax.add_patch(Rectangle((xx - 3.2, 0.8), 6.4, 4.0,
                                      facecolor="#8a2f2f", alpha=0.14,
                                      edgecolor="#8a2f2f", lw=1.1, zorder=3))
               ax.text(xx, 2.8, key, fontsize=fs(6.0), weight="bold",
                       ha="center", va="center", zorder=5)
           ax.text(cx + half + 3.5, 4.6,
                   "MANUAL BALL VALVES, above grade on the outside face,\n"
                   "tagged by crop. Open or shut, never partly, since a\n"
                   "throttled valve drops PC emitters below their range.\n"
                   "Shut the allium half MAY 10, reopen in June.",
                   fontsize=fs(5.8), va="center", color="#8a2f2f",
                   linespacing=1.45)
   low = [s for s in sts if s["gph"] / 60 < 0.5]
   fig.text(m(), 0.845,
            f'{sum(s["n"] for s in sts if s["n"])} inline emitters plus '
            f'{CONTAINERS["count"]} pot drippers . {total:.1f} gph if '
            f'everything ran at once, {total / 60:.2f} gpm . sequential, so '
            f'the true peak is {peak:.1f} gph or {peak / 60:.2f} gpm',
            fontsize=fs(7.2), va="top", color="#444")
   footer(fig, [
       f'{len(low)} of {len(sts)} zones draw under 0.5 gpm, the lowest being '
       f'station {min(low, key=lambda s: s["gph"])["st"]} at '
       f'{min(s["gph"] for s in low) / 60:.2f} gpm. A FIXED drip regulator '
       f'needs about 0.5 gpm to regulate at all, so use an ADJUSTABLE spring '
       f'and diaphragm PRV.' if low else
       "Every zone clears 0.5 gpm, so a fixed drip regulator would work.",
       "This is also the argument for pressure compensating tubing: PC "
       "emitters hold their rated flow from roughly 15 to 50 psi and forgive "
       "whatever imprecision remains after regulation.",
       "Diagnostics: 25 psi at a far lateral gauge means all is well, well "
       "above it means the PRV has failed, zero during a run means valve, "
       "filter or shutoff rather than emitters. Time the bleed down and log "
       "it.",
       f'Flush all '
       f'{sum(len(GRIDS[k]["laterals"]) for k in PLOT_ORDER) + 8} lateral '
       f'ends once or twice a year wide open, then an hour of citric acid or '
       f'dilute vinegar if the bleed down has grown slower. Foothill water '
       f'is hard and closes emitters gradually.'])
   save(fig, outdir, "05_schematic")


def main():
   global PDF, WRITE_IMAGES, PAGE, DPI, MARGIN, SCALE_DENOM, PANEL
   ap = argparse.ArgumentParser()
   ap.add_argument("--year", type=int, choices=[1, 2, 3, 4])
   ap.add_argument("--outdir", default="garden_out")
   ap.add_argument("--pdf", default="garden_plan.pdf", help="or 'none'")
   ap.add_argument("--page", default="letter", choices=list(PAGE_SIZES))
   ap.add_argument("--margin", type=float, default=0.30)
   ap.add_argument("--scale-denom", type=int)
   ap.add_argument("--no-panel", action="store_true")
   ap.add_argument("--dpi", type=int, default=200)
   ap.add_argument("--no-images", action="store_true")
   a = ap.parse_args()
   PAGE, MARGIN, SCALE_DENOM = PAGE_SIZES[a.page], a.margin, a.scale_denom
   PANEL, DPI, WRITE_IMAGES = not a.no_panel, a.dpi, not a.no_images
   os.makedirs(a.outdir, exist_ok=True)
   years = [a.year] if a.year else [1, 2, 3, 4]
   if a.pdf.lower() != "none":
       PDF = PdfPages(os.path.join(a.outdir, a.pdf))
       PDF.infodict()["Title"] = "Garden plan, Altadena 91104"
       PDF.infodict()["CreationDate"] = datetime.datetime.now()
   try:
       print(f"page {PAGE[0]} x {PAGE[1]} in, margin {MARGIN} in")
       print("fixed sheets:")
       fig_cover(a.outdir)
       fig_rotation(a.outdir)
       fig_planting(a.outdir)
       fig_seed_order(a.outdir)
       fig_annex(a.outdir)
       fig_schematic(a.outdir)
       for y in years:
           print(f"year {y}:")
           garden_plan(a.outdir, y)
           fig_schedule(a.outdir, y)
           fig_tasks(a.outdir, y)
           for key in PLOT_ORDER:
               plot_map(a.outdir, y, key)
               if ROTATION[y][key] == "allium":
                   plot_map(a.outdir, y, key, season="summer")
   finally:
       if PDF is not None:
           PDF.close()
   print(f'\n{len(os.listdir(a.outdir))} files in {a.outdir}/')


if __name__ == "__main__":
   main()