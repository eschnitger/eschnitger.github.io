"""
Garden plan . Altadena CA 91104 . USDA zone 10b

Three raised beds on a four year rotation grouped by water need, a fixed
8 x 20 ft vine annex north of Bed M for the sprawling cucurbits and sweet
corn, and a container group for the pepper collection.

Bed L is split at 72 in into two plots (L1, L2) fed by ONE solenoid, with
two manual 3/4 in ball valves on the outside face of the box selecting
halves. Stations therefore follow the BED, permanently.

Emitters are 0.598 gph (nominal 0.6, metric 2.26 L/h) on a 12 in grid,
giving 0.96 in/hr gross, so 15 min = 0.25 in and 60 min = 1 in.

   pip install matplotlib
   python garden.py                       # letter landscape, PDF + images
   python garden.py --scale-denom 16      # lock bed maps to 1:16
   python garden.py --page tabloid        # roomier calendars
   python garden.py --no-panel            # bed maps at full width
   python garden.py --year 1 --no-images  # one year, PDF only

Coordinates are inches. In the beds X runs along the length from the
header end, Y across the width with 0 at the north/trellis edge, inverted
so north is at the top. In the annex Y runs northward and is not inverted.
"""

import argparse
import datetime
import os

import matplotlib
matplotlib.use("Agg")                       # headless, works over ssh
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrow, Rectangle

# ======================================================================
# 1. PAGE AND TYPE
# ======================================================================

PAGE_SIZES = {"letter": (11.0, 8.5), "tabloid": (17.0, 11.0),
             "a4": (11.69, 8.27), "a3": (16.54, 11.69)}

PAGE = PAGE_SIZES["letter"]
MARGIN = 0.30
SCALE_DENOM = None          # lock maps to 1:N, or None to fit the page
PANEL = True
PDF = None
WRITE_IMAGES = True
DPI = 200

HALO = [pe.withStroke(linewidth=2.0, foreground="white")]


def fs(pt):
   """Point size, scaled from an 11 in wide reference page."""
   return pt * (PAGE[0] / 11.0) ** 0.7


def m(inches=None, axis="x"):
   """Margin as a figure fraction. Reads MARGIN at call time, not at
   definition time, so --margin actually takes effect."""
   inches = MARGIN if inches is None else inches
   return inches / (PAGE[0] if axis == "x" else PAGE[1])


# ======================================================================
# 2. GEOMETRY AND HARDWARE
# ======================================================================

BEDS = {
   "L1": dict(name="Bed L, north half", width=51.0, length=72.0,
              laterals=[7.5, 19.5, 31.5, 43.5],
              emitters=[6, 18, 30, 42, 54, 66]),
   "L2": dict(name="Bed L, south half", width=51.0, length=72.0,
              laterals=[7.5, 19.5, 31.5, 43.5],
              emitters=[6, 18, 30, 42, 54, 66]),
   "M":  dict(name="Bed M", width=56.5, length=116.0,
              laterals=[8.75, 21.75, 34.75, 47.75],
              emitters=[6, 18, 30, 42, 54, 66, 78, 90, 102, 114]),
   "S":  dict(name="Bed S", width=53.0, length=79.0,
              laterals=[7.75, 20.25, 32.75, 45.25],
              emitters=[6, 18, 30, 42, 54, 66, 78]),
}

# Open ground north of Bed M. Outside the rotation, so it grows the
# sprawlers every year and keeps its own hygiene: hills moved within the
# annex each season, cowpeas each winter, all vine debris cleared in Nov.
ANNEX = dict(name="Vine annex", width=96.0, length=240.0,
            laterals=[6, 18, 30, 42, 54, 66, 78, 90],
            emitters=[6 + 12 * k for k in range(20)])

GEOM = {"L1": "Lhalf", "L2": "Lhalf", "M": "M", "S": "S"}
PLOT_ORDER = ["L1", "L2", "M", "S"]

EMITTER_GPH = 0.598        # nominal 0.6 gph, metric 2.26 L/h dripline
RATE_IN_PER_HR = 0.96      # 0.598 gal/sq ft/hr x 1.604 in per gal/sq ft
MIN_QUARTER_IN = 15
MIN_ONE_IN = 60

STATION = {"L1": 1, "L2": 1, "M": 2, "S": 3, "containers": 4, "annex": 5}
STATION_LABEL = {1: "Bed L (L1 + L2)", 2: "Bed M", 3: "Bed S",
                4: "containers", 5: "annex, 8 x 20 ft"}
STATION_GPH = {1: 28.7, 2: 23.9, 3: 16.7, 4: 12.0, 5: 95.7}
SCHED_STATIONS = (1, 2, 3, 5)

FAMILY = {
   "nightshade": dict(label="Tomatoes + peppers", colour="#c0504d", short="N"),
   "cucurbit":   dict(label="Cucumbers + squash", colour="#e0b13a", short="C"),
   "brassica":   dict(label="Kale, spinach + greens", colour="#5c8a6a",
                      short="B"),
   "allium":     dict(label="Legumes, alliums, roots", colour="#b9a06a",
                      short="A"),
}

ROTATION = {
   1: {"L1": "nightshade", "L2": "allium", "M": "cucurbit", "S": "brassica"},
   2: {"L1": "allium", "L2": "cucurbit", "M": "brassica", "S": "nightshade"},
   3: {"L1": "cucurbit", "L2": "brassica", "M": "nightshade", "S": "allium"},
   4: {"L1": "brassica", "L2": "nightshade", "M": "allium", "S": "cucurbit"},
}
ROTATION_YEARS = {1: "Oct 2026 to Sep 2027", 2: "Oct 2027 to Sep 2028",
                 3: "Oct 2028 to Sep 2029", 4: "Oct 2029 to Sep 2030"}

MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
         "Apr", "May", "Jun", "Jul", "Aug", "Sep"]

VALVES = {                  # which half of Bed L is OPEN, by month
   1: ["both"] * 12,
   2: ["both"] * 12,
   3: ["both", "both", "L2", "L2", "L2", "L2", "L2",
       "L1", "L1", "L1", "L1", "both"],
   4: ["both", "both", "L2", "L2", "L2", "both", "both",
       "L2", "L2", "L2", "L2", "both"],
}
VALVE_NOTES = {(1, 7): "shut L2 May 10", (1, 8): "reopen L2, slips",
              (2, 7): "shut L1 May 10", (2, 8): "reopen L1, slips",
              (3, 7): "clear L2 early May", (4, 5): "N in, B cropping"}

PALETTE = {
   "tomato": "#c0504d", "pepper": "#c9553d", "basil": "#6aa84f",
   "marigold": "#e8a33d", "cucumber": "#4a8db7", "squash": "#e0b13a",
   "mini": "#e08a3c", "vine": "#d98a5f", "melon": "#c25b5b",
   "cantaloupe": "#d9a15f", "corn": "#c9a227", "brassica": "#5c8a6a",
   "kale": "#3d6b52", "spinach": "#4f8f5c", "greens": "#8fbf70",
   "pea": "#7cb342", "garlic": "#b9a06a", "onion": "#a8577a",
   "root": "#d1743c", "sweetpotato": "#9c6b4f", "cover": "#9aab7a",
}

# ======================================================================
# 3a. SEED CATALOGUE
#     Everything you are likely to change lives here. Nothing below needs
#     editing to swap a variety.
#
#     Format is (name, mature canopy radius in inches). The radius sizes
#     the pale circle, so be honest: a bush tomato at 11 draws very
#     differently from a Big Beef at 18. Row crops carry a nominal radius.
#
#     ORDER MATTERS. Lists cycle with a modulo and slot counts differ:
#       tomato 6 / 7 / 9, pepper 3 / 4 / 5, brassica 6 / 8 / 10,
#       cucumber 3 / 4 / 5, for L half / Bed S / Bed M.
#     Longer lists are silently truncated in smaller plots, so put the
#     varieties you want EVERY year at the front.
# ======================================================================

VARIETIES = {
   # Keep a third heat-setting or cherry: that is what carries you
   # through the July blossom drop window above 92 F.
   "tomato": [("Big Beef", 18), ("Sungold", 18), ("Solar Fire", 15),
              ("Black Cherry", 18), ("Heatmaster", 15), ("Momotaro", 17),
              ("Jaune Flamme", 16), ("Black Krim", 17),
              ("Cherokee Purple", 18)],
   # Bed peppers are annuals, pulled each autumn. The collection lives in
   # containers on station 4, where it overwinters and anchors nothing.
   "pepper": [("Jimmy Nardello", 11), ("jalapeno", 10),
              ("Corno di Toro", 11), ("serrano", 10), ("shishito", 11)],
   # Weighted to kale, which crops Dec to Apr then keeps going. Names
   # containing "kale" or "collards" draw in the darker colour.
   "brassica": [("Lacinato kale", 12), ("Green Magic", 11),
                ("Snow Crown", 11), ("Dazzling Blue kale", 12),
                ("De Cicco", 11), ("Red Russian kale", 12),
                ("broccolini", 10), ("Redbor kale", 12), ("collards", 12),
                ("Green Magic", 11)],
   # Mid Oct to mid Jan only. Later sowings bolt. Only the first two are
   # labelled on the map, to keep the row readable.
   "spinach": [("Space", 5), ("Corvair", 5), ("Regiment", 5),
               ("Red Kitten", 5), ("Bloomsdale", 5)],
   "greens": [("lettuce", 5), ("mizuna", 5), ("Tokyo Bekana", 5),
              ("arugula", 5), ("cilantro", 5), ("Joi Choi", 5),
              ("Ruby Streaks", 5)],
   # Trellised, so nearly free on floor area and granted every year.
   # Three picklers and a slicer fill the first four slots, which is all
   # a Bed L half has. Harvest picklers daily at 3 to 4 in.
   "cucumber": [("Calypso", 9), ("Excelsior", 9), ("Boston Pickling", 9),
                ("Marketmore 76", 9), ("Northern Pickling", 9)],
   "squash": [("Dunja", 20), ("Success PM", 20)],
   # Halloween insurance: light enough to trellis, so they fruit even in
   # the tightest year regardless of what the annex does.
   "mini_pumpkin": [("Jack Be Little", 10), ("Baby Boo", 10)],
   # Row crops, drawn as bands. The text prints in the band description
   # and the radius is a fiction nothing reads.
   "garlic": [("California Early and Lorz Italian, softneck", 0)],
   "onion": [("Texas Early Grano 502 and Red Creole, short day", 0)],
   "pea": [("Super Sugar Snap", 0)],
   "carrot": [("Nelson, Napoli and Danvers 126", 0)],
   "radish": [("French Breakfast", 0)],
}


def variety(kind, i=0):
   lst = VARIETIES[kind]
   return lst[i % len(lst)]


def vname(kind, i=0):
   return variety(kind, i)[0]


# ======================================================================
# 3b. DATES AND GERMINATION
#     Month-float: Oct 1 = 0.0, Sep 30 = 11.94. mf(1, 20) is Jan 20.
#     Germination is days at the stated soil temperature, the most useful
#     number here: a seed absent after twice its window has failed, and
#     knowing that in week two rather than week six is the whole point.
# ======================================================================

_MONTH_IDX = {10: 0, 11: 1, 12: 2, 1: 3, 2: 4, 3: 5,
             4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11}


def mf(month, day=1):
   """Divisor 31 so day 31 stays inside its own month."""
   return _MONTH_IDX[month] + (day - 1) / 31.0


def fmt_window(w):
   if w is None:
       return "n/a"

   def one(v):
       i = int(v) % 12
       d = min(31, int(round((v - int(v)) * 31)) + 1)
       return f"{MONTHS[i]} {d}"

   a, b = one(w[0]), one(w[1])
   return a if a == b else f"{a} to {b}"


def spans(w, min_w=0.0):
   """Window to drawable segments, splitting any that wraps the Oct/Sep
   boundary. Tomato harvest is Jun 15 to Oct 31, i.e. 8.45 to 0.97,
   which without this draws backwards across the whole sheet."""
   if w is None:
       return []
   a, b = w
   out = [(a, max(b, a + min_w))] if b >= a else \
         [(a, 12.0), (0.0, max(b, min_w))]
   return [(max(0.0, s), min(12.0, e)) for s, e in out if e > s]


# name, group, indoor window or None, in-ground window, harvest window,
# germination, note
CROPS = [
   dict(name="Tomatoes", grp="N", indoor=(mf(1, 20), mf(3, 15)),
        plant=(mf(3, 15), mf(3, 25)), harv=(mf(6, 15), mf(10, 31)),
        germ="5 to 10 d at 75 to 85 F",
        note="heat mat. Blossom drop above 92 F, cherries carry you"),
   dict(name="Peppers, in bed", grp="N", indoor=(mf(1, 15), mf(4, 10)),
        plant=(mf(4, 10), mf(4, 20)), harv=(mf(7, 1), mf(11, 30)),
        germ="10 to 21 d at 80 to 85 F",
        note="warmer than tomatoes, and slow. Be patient"),
   dict(name="Peppers, pots", grp="N", indoor=(mf(1, 15), mf(4, 10)),
        plant=(mf(4, 10), mf(4, 20)), harv=(mf(7, 1), mf(12, 20)),
        germ="10 to 21 d at 80 to 85 F",
        note="overwinter as perennials, 3 to 4 years. Station 4"),
   dict(name="Basil", grp="N", indoor=(mf(3, 1), mf(4, 10)),
        plant=(mf(4, 10), mf(4, 20)), harv=(mf(5, 15), mf(10, 31)),
        germ="5 to 10 d at 70 F", note="Prospera resists downy mildew"),
   dict(name="Marigold", grp="N", indoor=(mf(2, 15), mf(4, 1)),
        plant=(mf(4, 1), mf(4, 10)), harv=(mf(5, 1), mf(11, 30)),
        germ="5 to 7 d at 70 F",
        note="Nemagold, nematode suppressive. Bar shows bloom"),

   dict(name="Cucumbers, 1st", grp="C", indoor=None,
        plant=(mf(5, 1), mf(5, 10)), harv=(mf(6, 20), mf(8, 15)),
        germ="3 to 10 d at 70 F min",
        note="direct sow, trellised. Pick picklers daily at 3 to 4 in"),
   dict(name="Cucumbers, 2nd", grp="C", indoor=None,
        plant=(mf(6, 15), mf(6, 25)), harv=(mf(8, 1), mf(9, 30)),
        germ="3 to 10 d at 70 F min", note="same trellis, into the gaps"),
   dict(name="Summer squash", grp="C", indoor=None,
        plant=(mf(5, 1), mf(5, 10)), harv=(mf(6, 10), mf(7, 10)),
        germ="5 to 10 d at 70 F min",
        note="pulled early Jul, grey with mildew. Pumpkins take the space"),
   dict(name="Mini pumpkins", grp="C", indoor=None,
        plant=(mf(7, 1), mf(7, 10)), harv=(mf(10, 10), mf(10, 25)),
        germ="5 to 10 d at 75 F", note="trellised, Halloween. Cure 10 days"),

   dict(name="Kale", grp="B", indoor=(mf(9, 5), mf(10, 25)),
        plant=(mf(10, 15), mf(10, 25)), harv=(mf(12, 1), mf(5, 31)),
        germ="5 to 8 d at 65 to 75 F", note="four varieties, crops for months"),
   dict(name="Broccoli", grp="B", indoor=(mf(9, 5), mf(10, 25)),
        plant=(mf(10, 15), mf(10, 25)), harv=(mf(12, 15), mf(4, 15)),
        germ="5 to 10 d at 65 to 75 F", note="De Cicco for side shoots"),
   dict(name="Cauliflower", grp="B", indoor=(mf(9, 5), mf(10, 25)),
        plant=(mf(10, 15), mf(10, 25)), harv=(mf(1, 1), mf(3, 15)),
        germ="5 to 10 d at 65 to 75 F",
        note="single harvest. At E1, replant the gap in Jan"),
   dict(name="Spinach", grp="B", indoor=None,
        plant=(mf(10, 15), mf(1, 15)), harv=(mf(11, 15), mf(3, 15)),
        germ="7 to 14 d, wants soil under 70 F",
        note="NOTHING after mid Jan, it bolts. 2 to 3 wk successions"),
   dict(name="Lettuce", grp="B", indoor=None,
        plant=(mf(10, 1), mf(2, 15)), harv=(mf(11, 1), mf(4, 30)),
        germ="4 to 10 d at 60 to 70 F",
        note="3 wk successions. Summer only in shade north of Bed S"),
   dict(name="Asian greens", grp="B", indoor=None,
        plant=(mf(10, 1), mf(2, 15)), harv=(mf(11, 1), mf(4, 30)),
        germ="4 to 8 d at 60 to 70 F", note="mizuna, Tokyo Bekana, Joi Choi"),
   dict(name="Arugula", grp="B", indoor=None,
        plant=(mf(10, 1), mf(2, 15)), harv=(mf(10, 25), mf(4, 30)),
        germ="4 to 7 d at 60 to 70 F", note="fastest thing in the garden"),
   dict(name="Cilantro", grp="B", indoor=None,
        plant=(mf(10, 1), mf(2, 1)), harv=(mf(11, 1), mf(4, 15)),
        germ="7 to 14 d at 60 to 70 F", note="Santo or Calypso, slow to bolt"),
   dict(name="Malabar spinach", grp="B", indoor=(mf(3, 1), mf(4, 15)),
        plant=(mf(4, 15), mf(5, 1)), harv=(mf(6, 15), mf(10, 15)),
        germ="10 to 21 d at 75 F",
        note="no relation. Trellised, thrives in a 10b July"),

   dict(name="Garlic", grp="A", indoor=None,
        plant=(mf(10, 25), mf(11, 5)), harv=(mf(5, 20), mf(6, 10)),
        germ="shoots in 1 to 3 wk",
        note="SOFTNECK only. WATER OFF May 10 or bulbs will not cure"),
   dict(name="Onions", grp="A", indoor=(mf(9, 10), mf(11, 20)),
        plant=(mf(11, 10), mf(11, 20)), harv=(mf(5, 20), mf(6, 10)),
        germ="7 to 14 d at 65 to 75 F",
        note="SHORT DAY only, long day never bulbs at this latitude"),
   dict(name="Snap peas", grp="A", indoor=None,
        plant=(mf(10, 25), mf(11, 5)), harv=(mf(1, 1), mf(3, 20)),
        germ="7 to 14 d at 60 to 70 F", note="trellised. Out Mar 20 for beans"),
   dict(name="Carrots", grp="A", indoor=None,
        plant=(mf(9, 15), mf(2, 28)), harv=(mf(11, 15), mf(5, 15)),
        germ="14 to 21 d, the slowest here",
        note="TWO 5 min pulses daily for 10 to 14 d. Sep to Feb only"),
   dict(name="Radish", grp="A", indoor=None,
        plant=(mf(9, 15), mf(2, 28)), harv=(mf(10, 15), mf(3, 31)),
        germ="3 to 7 d at 60 to 70 F",
        note="marks the carrot rows while they think it over"),
   dict(name="Bush beans", grp="A", indoor=None,
        plant=(mf(3, 20), mf(4, 1)), harv=(mf(5, 15), mf(7, 15)),
        germ="7 to 10 d at 70 F", note="Jade holds quality in heat"),
   dict(name="Sweet potatoes", grp="A", indoor=None,
        plant=(mf(6, 1), mf(6, 10)), harv=(mf(9, 20), mf(10, 20)),
        germ="slips, no germination. Order in Feb",
        note="5 min daily pulses for week one. Vines may run to shade"),

   dict(name="Sweet corn", grp="X", indoor=None,
        plant=(mf(4, 15), mf(5, 10)), harv=(mf(7, 10), mf(7, 31)),
        germ="5 to 10 d at 65 F min",
        note="5 x 5 BLOCK, not a row. No stress at tassel and silk"),
   dict(name="Watermelon", grp="X", indoor=(mf(4, 10), mf(5, 1)),
        plant=(mf(5, 1), mf(5, 10)), harv=(mf(8, 1), mf(9, 15)),
        germ="4 to 12 d at 75 F min",
        note="icebox types. Dry down hard for the last 10 d"),
   dict(name="Cantaloupe", grp="X", indoor=(mf(4, 10), mf(5, 1)),
        plant=(mf(5, 1), mf(5, 10)), harv=(mf(7, 25), mf(8, 31)),
        germ="4 to 10 d at 75 F min",
        note="south strip, 30 in clear of the Bed M trellis shadow"),
   dict(name="Carving pumpkins", grp="X", indoor=None,
        plant=(mf(7, 1), mf(7, 10)), harv=(mf(10, 10), mf(10, 25)),
        germ="5 to 10 d at 75 F",
        note="95 to 110 d. Jul 1 lands mid Oct. Cure 10 d, 2 plants"),

   dict(name="Bell bean + oat", grp="V", indoor=None,
        plant=(mf(11, 5), mf(11, 15)), harv=(mf(2, 10), mf(2, 25)),
        germ="7 to 14 d on rainfall",
        note="beds. The harvest bar means chop in at flowering"),
   dict(name="Buckwheat", grp="V", indoor=None,
        plant=(mf(5, 25), mf(6, 5)), harv=(mf(6, 25), mf(7, 5)),
        germ="3 to 5 d at 70 F", note="30 days to flowering, then chop"),
   dict(name="Cowpeas", grp="V", indoor=None,
        plant=(mf(7, 1), mf(7, 15)), harv=(mf(10, 25), mf(11, 10)),
        germ="5 to 10 d at 70 F",
        note="beds and the whole annex. Fixes N, suppresses nematodes"),
]

GRP_LABEL = {"N": "Tomatoes + peppers", "C": "Cucurbits, in bed",
            "B": "Kale, spinach + greens", "A": "Legumes, alliums, roots",
            "X": "The annex", "V": "Cover crops"}
GRP_COLOUR = {"N": "#c0504d", "C": "#e0b13a", "B": "#5c8a6a",
             "A": "#b9a06a", "X": "#d98a5f", "V": "#9aab7a"}


def crop(name):
   for c in CROPS:
       if c["name"] == name:
           return c
   raise KeyError(f"no crop named {name!r}")


# ======================================================================
# 3c. PLANT POSITIONS
#     L1 and L4 carry the large plants. L2 carries peppers in the
#     nightshade plot and spinach in the brassica plot. L3 carries basil
#     and marigold, or lettuce and Asian greens.
# ======================================================================

HINTS = {
   ("nightshade", "Lhalf"): dict(
       fruit=[(1, 1), (1, 3), (1, 5), (4, 2), (4, 4), (4, 6)],
       pepper=[(2, 1), (2, 3), (2, 5)]),
   ("nightshade", "S"): dict(
       fruit=[(1, 1), (1, 3), (1, 5), (1, 7), (4, 2), (4, 4), (4, 6)],
       pepper=[(2, 1), (2, 3), (2, 5), (2, 7)]),
   ("nightshade", "M"): dict(
       fruit=[(1, 1), (1, 3), (1, 5), (1, 7), (1, 9),
              (4, 2), (4, 4), (4, 6), (4, 8)],
       pepper=[(2, 1), (2, 3), (2, 5), (2, 7), (2, 9)]),
   ("cucurbit", "Lhalf"): dict(
       cucumber=[(1, 1), (1, 3), (1, 5)], mini=[(1, 2), (1, 4)],
       squash=[(2, 1), (2, 6)]),
   ("cucurbit", "M"): dict(
       cucumber=[(1, 1), (1, 3), (1, 5), (1, 7), (1, 9)],
       mini=[(1, 2), (1, 6)], squash=[(2, 1), (2, 7), (3, 4), (3, 10)]),
   ("cucurbit", "S"): dict(
       cucumber=[(1, 1), (1, 3), (1, 5), (1, 7)], mini=[(1, 2), (1, 4)],
       squash=[(3, 2), (3, 6)]),
   ("brassica", "Lhalf"): dict(
       large=[(1, 1), (1, 3), (1, 5), (4, 1), (4, 3), (4, 5)]),
   ("brassica", "M"): dict(
       large=[(1, 1), (1, 3), (1, 5), (1, 7), (1, 9),
              (4, 1), (4, 3), (4, 5), (4, 7), (4, 9)]),
   ("brassica", "S"): dict(
       large=[(1, 1), (1, 3), (1, 5), (1, 7),
              (4, 1), (4, 3), (4, 5), (4, 7)]),
}

# Variety, position, canopy and run direction are coupled here, so annex
# hills stay together rather than moving to VARIETIES. Every circle will
# be exceeded, which is the point of the run arrows.
ANNEX_HILLS = [
   dict(x=30, y=66, r=42, key="vine", label="Jack O'Lantern",
        dx=-1, dy=1, note="west + north, out of the annex"),
   dict(x=90, y=66, r=42, key="vine", label="Small Sugar",
        dx=0, dy=1, note="north"),
   dict(x=150, y=66, r=36, key="melon", label="Sugar Baby",
        dx=0, dy=1, note="north"),
   dict(x=30, y=30, r=36, key="melon", label="Blacktail Mtn",
        dx=-1, dy=0, note="west"),
   dict(x=90, y=30, r=24, key="cantaloupe", label="Ambrosia",
        dx=0, dy=-1, note="30 in clears the shadow"),
   dict(x=150, y=30, r=24, key="cantaloupe", label="Sugar Cube",
        dx=0, dy=-1, note="south strip"),
]

CORN = dict(x0=186, y0=42, cols=5, rows=5, step=12, variety="Bodacious")


def pos(bed, lat_no, em_no):
   return bed["emitters"][em_no - 1], bed["laterals"][lat_no - 1]


def layout_nightshade(plot, season=None):
   bed, h = BEDS[plot], HINTS[("nightshade", GEOM[plot])]
   plants = []
   for i, cell in enumerate(h["fruit"]):
       x, y = pos(bed, *cell)
       name, r = variety("tomato", i)
       plants.append(dict(x=x, y=y, r=r, key="tomato", label=name))
   for i, cell in enumerate(h["pepper"]):
       x, y = pos(bed, *cell)
       name, r = variety("pepper", i)
       plants.append(dict(x=x, y=y, r=r, key="pepper", label=name))
   for k, x in enumerate(bed["emitters"]):
       basil = (k % 2 == 0)
       label = "basil" if (basil and k == 0) else \
               ("marigold" if (not basil and k == 1) else "")
       plants.append(dict(x=x, y=bed["laterals"][2], r=7 if basil else 5,
                          key="basil" if basil else "marigold", label=label))
   notes = [
       f'{len(h["fruit"])} tomatoes on the outer laterals, 36 in between '
       f'rows and 38 in diagonal, strung vertically rather than caged. '
       f'Transplant Mar 15, determinates nearest E1.',
       f'{len(h["pepper"])} peppers on L2, heat mat mid Jan at 80 to 85 F, '
       f'out Apr 10. Annuals here, pulled each autumn so they never anchor '
       f'the bed.',
       "L3 alternates basil with Nemagold marigold, which is nematode "
       "suppressive and not decoration.",
       "The pepper collection proper is in containers on station 4, where "
       "it overwinters and ripens earlier in the warmer root zone."]
   return plants, [], notes


def layout_cucurbit(plot, season=None):
   bed, h = BEDS[plot], HINTS[("cucurbit", GEOM[plot])]
   plants = []
   for i, cell in enumerate(h["cucumber"]):
       x, y = pos(bed, *cell)
       name, r = variety("cucumber", i)
       plants.append(dict(x=x, y=y, r=r, key="cucumber", label=name))
   for i, cell in enumerate(h.get("mini", [])):
       x, y = pos(bed, *cell)
       name, r = variety("mini_pumpkin", i)
       plants.append(dict(x=x, y=y, r=r, key="mini", label=name))
   for i, cell in enumerate(h["squash"]):
       x, y = pos(bed, *cell)
       name, r = variety("squash", i)
       plants.append(dict(x=x, y=y, r=r, key="squash", label=name))
   notes = [
       f'{len(h["cucumber"])} cucumbers trellised on the north edge, nearly '
       f'free on floor area. Direct sow May 1, second flush mid Jun, harvest '
       f'picklers daily at 3 to 4 in.',
       "Mini pumpkins share the trellis, sown Jul 1 to 10 for mid Oct. "
       "Light enough to hang, and the insurance against whatever the annex "
       "does.",
       "Squash sown May 1, pulled early Jul when it is grey with mildew. L4 "
       "stays open: the 20 in canopies cover it and airflow beats a plant.",
       "Water stress makes cucumbers genuinely bitter through cucurbitacin, "
       "so a skipped event here has a flavour cost, not just a yield cost."]
   return plants, [], notes


def layout_brassica(plot, season=None):
   bed, h = BEDS[plot], HINTS[("brassica", GEOM[plot])]
   plants = []
   for i, cell in enumerate(h["large"]):
       x, y = pos(bed, *cell)
       name, r = variety("brassica", i)
       key = "kale" if ("kale" in name or name == "collards") else "brassica"
       plants.append(dict(x=x, y=y, r=r, key=key, label=name))
   for k, x in enumerate(bed["emitters"]):
       sp, _ = variety("spinach", k)
       gr, _ = variety("greens", k)
       plants.append(dict(x=x, y=bed["laterals"][1], r=5, key="spinach",
                          label=sp if k < 2 else ""))
       plants.append(dict(x=x, y=bed["laterals"][2], r=5, key="greens",
                          label=gr if k < 2 else ""))
   notes = [
       f'{len(h["large"])} large plants on the outer laterals, weighted to '
       f'kale since it crops Dec through Apr then keeps going. Transplant '
       f'Oct 15 to 25, row cover from the start.',
       "L2 is spinach in short successions from mid Oct to mid Jan only. "
       "Nothing after January, because it bolts before it is worth eating.",
       "L3 is lettuce and Asian greens on a three week cycle. Only the first "
       "two positions are labelled: seven adjacent labels would smear.",
       "Summer: buckwheat from late May, cowpeas from July, or two Malabar "
       "spinach on the trellis, which thrives when true spinach cannot."]
   return plants, [], notes


def layout_allium(plot, season="winter"):
   bed = BEDS[plot]
   L, lat = bed["length"], bed["laterals"]
   if season == "summer":
       plants = []
       for lat_no in (2, 3):
           x = 6.0
           while x <= L - 6:
               plants.append(dict(x=x, y=lat[lat_no - 1], r=16,
                                  key="sweetpotato", label=""))
               x += 18
       notes = [
           "Slips planted Jun 1 to 10 into the vacated garlic and onion "
           "bands. Beauregard, plus Murasaki for purple. Order in February.",
           "Five minute daily pulses for the first week. Botanically "
           "unrelated to everything else, which is exactly why they are here.",
           "In Bed L years let the vines run west into shade: tubers form at "
           "the crown, so shaded foliage costs nothing.",
           "Lift late Sep or Oct. Circles are canopy, and the vines will "
           "exceed them."]
       return plants, [], notes

   peas, garlic = int(L / 3), 2 * int(L / 6)
   onions, roots = 2 * int(L / 5), int(L * 0.8)
   bands = [
       dict(y=lat[0], half=3.0, key="pea", title="L1 . SNAP PEAS, trellised",
            detail=f'{vname("pea")}, seed 3 in apart, {peas} plants',
            after="out Mar 20, then beans"),
       dict(y=lat[1], half=4.0, key="garlic", title="L2 . GARLIC",
            detail=f'{vname("garlic")}, double row 3 in either side, '
                   f'cloves 6 in, {garlic} cloves',
            after="WATER OFF May 10"),
       dict(y=lat[2], half=4.0, key="onion", title="L3 . ONIONS",
            detail=f'{vname("onion")}, double row, 5 in apart, '
                   f'{onions} plants',
            after="WATER OFF May 10"),
       dict(y=lat[3], half=4.5, key="root",
            title="L4 . CARROTS, radish interplanted",
            detail=f'{vname("carrot")}, bands 3 in off line, 3 wk '
                   f'successions, ~{roots} roots',
            after="last pull by May"),
   ]
   notes = [
       "Garlic and peas late Oct, onions Nov, carrots from Sep. Softneck "
       "garlic only, since 10b will not vernalise a hardneck.",
       "Short day onions only: long day types grow handsome tops here and "
       "never bulb. Drawn as bands because true spacing merges to a stripe.",
       f'Radish ({vname("radish")}) at 4 in between carrot sowings matures '
       f'in a month and marks the rows while the carrots deliberate.',
       "All four bands are spring finishers, which is exactly why the whole "
       "zone can dry down together in the second week of May."]
   return [], bands, notes


LAYOUT = {"nightshade": layout_nightshade, "cucurbit": layout_cucurbit,
         "brassica": layout_brassica, "allium": layout_allium}


def sections_for(family, plot, season="winter"):
   """Per-lateral contents with dates, for the side panel."""
   h = HINTS.get((family, GEOM[plot]), {})
   out = []

   def ent(lat, title, contents, cropname, extra=None):
       c = crop(cropname)
       lines = [("plants", contents),
                ("sow", (fmt_window(c["indoor"]) + ", cover")
                 if c["indoor"] else (fmt_window(c["plant"]) + ", direct")),
                ("in ground", fmt_window(c["plant"])),
                ("germ", c["germ"]),
                ("harvest", fmt_window(c["harv"]))]
       if extra:
           lines.append(("note", extra))
       out.append(dict(lat=lat, title=title, lines=lines))

   if family == "nightshade":
       n1 = sum(1 for c in h["fruit"] if c[0] == 1)
       n4 = sum(1 for c in h["fruit"] if c[0] == 4)
       ent("L1", "TOMATOES", f"{n1}, strung vertically", "Tomatoes")
       ent("L2", "PEPPERS", f'{len(h["pepper"])}, annual here',
           "Peppers, in bed")
       ent("L3", "BASIL + MARIGOLD", "alternating, 6 total", "Basil")
       ent("L4", "TOMATOES", f"{n4}, determinates at E1", "Tomatoes")
   elif family == "cucurbit":
       ent("L1", "CUCUMBERS", f'{len(h["cucumber"])}, trellised',
           "Cucumbers, 1st", "2nd flush mid Jun into the gaps")
       ent("L1", "MINI PUMPKINS", f'{len(h.get("mini", []))}, trellised',
           "Mini pumpkins")
       ent("L2 / L3", "SUMMER SQUASH", f'{len(h["squash"])} plants',
           "Summer squash")
       out.append(dict(lat="L4", title="OPEN", lines=[
           ("why", "20 in squash canopies cover it"),
           ("note", "airflow beats a plant here")]))
   elif family == "brassica":
       ent("L1 / L4", "KALE + BROCCOLI", f'{len(h["large"])} large', "Kale",
           "cauliflower at E1, replant in Jan")
       ent("L2", "SPINACH", "successions, 2 to 3 wk", "Spinach")
       ent("L3", "LETTUCE + GREENS", "successions, 3 wk", "Lettuce")
       mal = crop("Malabar spinach")
       out.append(dict(lat="summer", title="TRELLIS", lines=[
           ("plants", "2 Malabar spinach"),
           ("in ground", fmt_window(mal["plant"])),
           ("germ", mal["germ"]),
           ("note", "or buckwheat then cowpeas")]))
   elif family == "allium" and season == "summer":
       ent("L2 / L3", "SWEET POTATOES", "8 slips at 18 in", "Sweet potatoes")
       bb = crop("Bush beans")
       out.append(dict(lat="L1 / L4", title="BUSH BEANS", lines=[
           ("in ground", fmt_window(bb["plant"])),
           ("germ", bb["germ"]), ("note", "or bare and mulched")]))
   else:
       ent("L1", "SNAP PEAS", "seed 3 in apart", "Snap peas")
       ent("L2", "GARLIC", "double row, 6 in", "Garlic")
       ent("L3", "ONIONS", "double row, 5 in", "Onions")
       ent("L4", "CARROTS + RADISH", "3 wk successions", "Carrots",
           "radish at 4 in marks the rows")
   return out


# ======================================================================
# 4. SCHEDULE
#     Durations are for 0.598 gph emitters at 0.96 in/hr gross. Change
#     RATE_IN_PER_HR and these together or not at all.
# ======================================================================

SCHEDULE = {   # (events per week, minutes per event) by month, Oct first
   "nightshade": [(2, 25), (1, 15), (1, 15), (1, 15), (1, 15), (3, 12),
                  (2, 20), (2, 30), (3, 28), (3, 35), (3, 35), (2, 40)],
   "cucurbit":   [(2, 20), (1, 15), (1, 15), (1, 15), (1, 15), (1, 15),
                  (3, 10), (3, 23), (3.5, 23), (3.5, 35), (3.5, 30), (3, 28)],
   "brassica":   [(3, 15), (2, 15), (1, 20), (1, 20), (1, 23), (2, 22),
                  (3, 21), (4, 19), (1, 23), (1, 23), (1, 23), (3, 15)],
   "allium":     [(2, 20), (1, 20), (1, 20), (1, 20), (1, 23), (2, 23),
                  (2, 30), (2, 35), (2, 35), (3, 31), (3, 31), (2, 31)],
}

# Open ground, not a raised box, so it does not dry from the sides and
# takes deeper, less frequent soaks.
ANNEX_SCHEDULE = [(1, 30), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0),
                 (0, 0), (2, 40), (2.33, 47), (2.33, 55), (2.33, 55),
                 (2, 47)]


def bed_l_schedule(year):
   """One solenoid, so Bed L runs the envelope of whichever halves are
   open: higher frequency, longer event. Zeroes are filtered first so an
   empty half cannot suppress a growing one."""
   fam = {p: ROTATION[year][p] for p in ("L1", "L2")}
   out = []
   for i in range(12):
       state = VALVES[year][i]
       plots = ["L1", "L2"] if state == "both" else [state]
       cand = [SCHEDULE[fam[p]][i] for p in plots]
       cand = [c for c in cand if c[0] > 0] or [(0, 0)]
       out.append((max(c[0] for c in cand), max(c[1] for c in cand)))
   return out


def zone_schedule(year, station):
   if station == 1:
       return bed_l_schedule(year)
   if station == 5:
       return ANNEX_SCHEDULE
   return SCHEDULE[ROTATION[year][{2: "M", 3: "S"}[station]]]


def freq_label(ev):
   if ev == 3.5:
       return "every 2nd day"
   if abs(ev - 2.33) < 0.01:
       return "every 3rd day"
   return f"{ev:g} x week"


OCCUPANCY = {
   "nightshade": [(0, 1, "compost, resting", "cover"),
                  (1, 4.5, "cover crop", "cover"),
                  (5.5, 12, "tomatoes, peppers, basil", "tomato")],
   "cucurbit":   [(0, 1, "compost", "cover"),
                  (1, 4.5, "cover crop", "cover"),
                  (7, 9.2, "cucumbers and squash", "squash"),
                  (9, 12.5, "mini pumpkins, Halloween", "mini")],
   "brassica":   [(0, 7.3, "kale, broccoli, greens", "brassica"),
                  (0.3, 4.0, "spinach successions", "spinach"),
                  (7.8, 9.1, "buckwheat", "cover"),
                  (9.1, 11.4, "cowpeas or Malabar", "cover")],
   "allium":     [(0.8, 5.6, "snap peas", "pea"),
                  (0.8, 7.8, "garlic and onions", "garlic"),
                  (0, 7.3, "carrots and radish", "root"),
                  (8.1, 12, "sweet potatoes", "sweetpotato")],
}

TASKS = [
   (0.5, "kale, broccoli out"), (0.6, "first spinach"),
   (0.8, "garlic + peas in"), (1.2, "cover crop sown"), (1.4, "onions in"),
   (3.5, "LAST spinach"), (3.6, "peppers on mat"), (4.5, "chop cover"),
   (5.5, "tomatoes out"), (5.6, "beans sown"), (6.3, "peppers, basil out"),
   (7.0, "squash + cukes"), (7.3, "WATER OFF, alliums"),
   (7.8, "lift garlic"), (8.1, "slips planted"), (9.0, "pumpkins sown"),
   (9.1, "cowpeas sown"), (11.2, "next year begins"),
]

SOWING = [
   ("Sep 5 to 12", "sow kale, broccoli, cauliflower, onions in cells"),
   ("Sep 15", "first direct sowing of carrots"),
   ("Sep", "order softneck garlic before it sells out"),
   ("Oct 1", "first lettuce and Asian greens"),
   ("mid Oct", "first spinach, then every 2 to 3 weeks"),
   ("Oct 15 to 25", "transplant kale and broccoli, 5 min pulse for 10 d"),
   ("Oct 25", "plant garlic, direct sow snap peas"),
   ("Nov", "clear ALL annex vine debris, squash bugs overwinter in it"),
   ("Nov 5 to 15", "cover crop in the beds, cowpeas in the annex"),
   ("Nov 10 to 20", "transplant onions"),
   ("Dec", "place the spring order, pepper seed early"),
   ("Jan 15", "sow peppers on the heat mat, 80 to 85 F"),
   ("Jan 20", "sow tomatoes on the heat mat, 75 to 85 F"),
   ("mid Jan", "LAST spinach sowing, later ones bolt"),
   ("Feb", "order sweet potato slips and annex melon seed"),
   ("mid Feb", "chop the cover crop in at flowering"),
   ("Mar 1", "sow basil and Malabar spinach"),
   ("Mar 15", "transplant tomatoes, set the trellis"),
   ("Mar 20", "pull peas, direct sow bush beans"),
   ("mid Apr", "sow the annex corn block, 25 plants"),
   ("Apr 10", "peppers, basil, marigold out"),
   ("May 1", "cucumbers and squash sown, annex melon hills in"),
   ("May 10", "SHUT THE BALL VALVE on the allium half"),
   ("late May", "lift and cure garlic and onions, sow buckwheat"),
   ("Jun 1 to 10", "plant eight sweet potato slips"),
   ("mid Jun", "second cucumber flush"),
   ("Jul 1 to 10", "sow annex pumpkins AND bed mini pumpkins"),
   ("early Jul", "pull mildewed squash, chop buckwheat, sow cowpeas"),
   ("mid Aug", "melon dry down, 10 d, hand water pumpkin seedlings"),
   ("mid Oct", "cut pumpkins, cure 10 d, Halloween secured"),
]

# ======================================================================
# 5. PAGE PLUMBING
# ======================================================================


def page():
   fig = plt.figure(figsize=PAGE)
   fig.patch.set_facecolor("white")
   return fig


def rect(fig, x, y, w, h, **kw):
   """Figure-fraction rectangle via add_artist, since appending to
   fig.patches was deprecated in mpl 3.5 and removed in 3.7."""
   fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure, **kw))


def line(fig, xs, ys, **kw):
   fig.add_artist(Line2D(xs, ys, transform=fig.transFigure, **kw))


def header(fig, title, sub=None):
   fig.text(m(), 1 - m(0.34, "y"), title, fontsize=fs(13.5), weight="bold",
            va="top", ha="left")
   if sub:
       fig.text(m(), 1 - m(0.60, "y"), sub, fontsize=fs(8.5), va="top",
                ha="left", color="#555")


def footer(fig, lines):
   y = m(0.30, "y") + 0.021 * len(lines)
   for i, text in enumerate(lines):
       fig.text(m(), y - i * 0.021, text, fontsize=fs(7.2), va="top",
                ha="left", color="#555")


def fit_axes(fig, box, xlim, ylim, denom=None, invert=True):
   """Axes inside box (figure fractions), data aspect preserved exactly.
   denom locks the scale to 1:denom, warning rather than clipping if it
   will not fit. invert=False for the annex, where Y already runs north."""
   fw, fh = fig.get_size_inches()
   dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]
   s = min(box[2] * fw / dx, box[3] * fh / dy)
   if denom:
       want = 1.0 / denom
       if want <= s:
           s = want
       else:
           print(f"    note: 1:{denom} needs {dx * want:.1f} x "
                 f"{dy * want:.1f} in, only {box[2] * fw:.1f} x "
                 f"{box[3] * fh:.1f} available, using 1:{1 / s:.1f}")
   w, h = dx * s / fw, dy * s / fh
   ax = fig.add_axes([box[0] + (box[2] - w) / 2,
                      box[1] + (box[3] - h) / 2, w, h])
   ax.set_xlim(*xlim)
   ax.set_ylim(*(ylim[::-1] if invert else ylim))
   ax.set_aspect("equal")
   ax.axis("off")
   return ax, s


def save(fig, outdir, name):
   # No bbox_inches: the figure IS the page, so padding is never applied
   # twice and nothing is rescaled after layout.
   if WRITE_IMAGES:
       fig.savefig(os.path.join(outdir, f"{name}.png"), dpi=DPI,
                   facecolor="white")
       fig.savefig(os.path.join(outdir, f"{name}.svg"), facecolor="white")
   if PDF is not None:
       PDF.savefig(fig, facecolor="white")
   plt.close(fig)
   print("  wrote", name)


def legend(fig, keys, loc="lower left", anchor=None, ncol=4):
   handles = [Line2D([], [], marker="o", ls="none", ms=fs(6),
                     mfc=PALETTE[k], mec="white", label=k) for k in keys]
   fig.legend(handles=handles, loc=loc,
              bbox_to_anchor=anchor or (m(), m(0.30, "y") + 0.115),
              ncol=ncol, frameon=False, fontsize=fs(6.5),
              handletextpad=0.4, columnspacing=1.1)


# ======================================================================
# 6. BED MAPS
# ======================================================================


def draw_frame(ax, b, trellis):
   ax.add_patch(Rectangle((0, 0), b["length"], b["width"],
                          facecolor="#efe7d8", edgecolor="#6b4f2a",
                          linewidth=1.8, zorder=0))
   ax.plot([-4, -4], [0, b["width"]], color="#2f5d8a", lw=2.6, zorder=2)
   ax.text(-6.6, b["width"] / 2, "HEADER", rotation=90, fontsize=fs(6),
           va="center", ha="center", color="#2f5d8a")
   for i, y in enumerate(b["laterals"], start=1):
       ax.plot([-4, b["length"]], [y, y], color="#2f5d8a", lw=0.9,
               ls=(0, (5, 3)), zorder=2)
       ax.text(b["length"] + 2.2, y, f"L{i}", fontsize=fs(6.5), va="center",
               color="#2f5d8a")
       for x in b["emitters"]:
           ax.plot(x, y, "o", ms=fs(2.2), color="#2f5d8a", zorder=6)
   for j, x in enumerate(b["emitters"], start=1):
       ax.text(x, b["width"] + 4.0, f"E{j}", fontsize=fs(6), ha="center",
               va="center", color="#8a8a8a")
   if trellis:
       ax.plot([0, b["length"]], [-2.0, -2.0], color="#4a4a4a", lw=3.4,
               solid_capstyle="butt", zorder=3)
       ax.text(b["length"] / 2, -5.4, trellis, fontsize=fs(7), ha="center",
               va="center", color="#4a4a4a")


def draw_plants(ax, b, plants):
   mid = b["width"] / 2
   for p in plants:
       c = PALETTE[p["key"]]
       ax.add_patch(Circle((p["x"], p["y"]), p["r"], facecolor=c, alpha=0.20,
                           edgecolor=c, lw=0.8, zorder=4))
       ax.plot(p["x"], p["y"], "o", ms=fs(4.6) if p["r"] > 8 else fs(3.2),
               color=c, mec="white", mew=0.7, zorder=7)
       if p["label"]:
           up = p["y"] < mid       # push labels away from the bed centre
           yy = p["y"] - p["r"] - 2.2 if up else p["y"] + p["r"] + 2.2
           ax.text(p["x"], yy, p["label"], fontsize=fs(6.4), ha="center",
                   va="bottom" if up else "top", color="#2b2b2b",
                   zorder=9, path_effects=HALO)


def draw_bands(ax, b, bands):
   for bd in bands:
       c = PALETTE[bd["key"]]
       ax.add_patch(Rectangle((0, bd["y"] - bd["half"]), b["length"],
                              bd["half"] * 2, facecolor=c, alpha=0.28,
                              edgecolor=c, lw=0.8, zorder=4))
       ax.text(2.0, bd["y"] - 1.0, bd["title"], fontsize=fs(6.8),
               weight="bold", va="bottom", zorder=9, path_effects=HALO)
       ax.text(2.0, bd["y"] + 1.2, bd["detail"], fontsize=fs(6.1),
               va="top", color="#333", zorder=9, path_effects=HALO)
       ax.text(b["length"] - 2.0, bd["y"], bd["after"], fontsize=fs(5.9),
               ha="right", va="center", style="italic", color="#8a2f2f",
               zorder=9, path_effects=HALO)


def draw_panel(fig, sections, x0=0.655):
   y = 0.795
   lh, gap = 0.0163, 0.0075
   fig.text(x0, y, "WHAT IS IN EACH SECTION", fontsize=fs(7.4),
            weight="bold", color="#6b4f2a", va="center")
   y -= lh * 1.4
   for s in sections:
       fig.text(x0, y, f'{s["lat"]} . {s["title"]}', fontsize=fs(6.9),
                weight="bold", va="center", color="#222")
       y -= lh
       for label, value in s["lines"]:
           fig.text(x0 + 0.008, y, label, fontsize=fs(5.9), va="center",
                    color="#8a8a8a")
           fig.text(x0 + 0.062, y, value, fontsize=fs(5.9), va="center",
                    color="#333")
           y -= lh
       y -= gap


def plot_map(outdir, year, plot, family, season="winter"):
   b, st = BEDS[plot], STATION[plot]
   plants, bands, notes = LAYOUT[family](plot, season)
   tag = " . JUNE TO OCTOBER" if season == "summer" else ""
   fig = page()
   header(fig, f'YEAR {year} . {b["name"]} ({plot}) . '
               f'{FAMILY[family]["label"].upper()}{tag}',
          f'Station {st}, {STATION_LABEL[st]} . {ROTATION_YEARS[year]} . '
          f'Altadena 91104, zone 10b')
   pad = 12
   box_w = (0.615 - m()) if PANEL else (1 - 2 * m())
   ax, s = fit_axes(fig, (m(), 0.22, box_w, 0.60),
                    (-pad, b["length"] + pad), (-pad, b["width"] + pad),
                    denom=SCALE_DENOM)
   trellis = {"nightshade": "T R E L L I S   ( tomatoes )",
              "cucurbit": "T R E L L I S   ( cucumbers + minis )",
              "allium": "T R E L L I S   ( snap peas )",
              "brassica": "T R E L L I S   ( Malabar, summer )"}[family]
   draw_frame(ax, b, trellis if season == "winter" else "north edge")
   draw_plants(ax, b, plants)
   draw_bands(ax, b, bands)
   if PANEL:
       draw_panel(fig, sections_for(family, plot, season))
   n = len(b["laterals"]) * len(b["emitters"])
   fig.text(m(), 0.845,
            f'{b["width"]:.1f} x {b["length"]:.0f} in  .  {n} emitters at '
            f'{EMITTER_GPH:g} gph  .  {n * EMITTER_GPH:.1f} gph  .  '
            f'{RATE_IN_PER_HR} in/hr  .  {MIN_QUARTER_IN} min = 0.25 in  .  '
            f'{MIN_ONE_IN} min = 1 in  .  drawn 1 : {1 / s:.0f}',
            fontsize=fs(7.4), va="top", color="#444")
   footer(fig, notes)
   if plants:
       legend(fig, sorted({p["key"] for p in plants}))
   save(fig, outdir, f"garden_y{year}_{plot}_{family}"
                     f'{"_summer" if season == "summer" else ""}')


def garden_plan(outdir, year):
   fig = page()
   header(fig, f"ROTATION YEAR {year} . PLOT MAP",
          f"{ROTATION_YEARS[year]} . 125.6 sq ft in beds plus a 160 sq ft "
          f"annex . Bed L halves selected by manual ball valve")
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
   for plot, x, y, w, h in [("L1", 0, 0, 72, 51), ("L2", 72, 0, 72, 51),
                            ("M", 0, 87, 116, 56.5), ("S", 0, 179, 79, 53)]:
       fam = FAMILY[ROTATION[year][plot]]
       ax.add_patch(Rectangle((x, y), w, h, facecolor=fam["colour"],
                              alpha=0.22, edgecolor="#6b4f2a", lw=1.8))
       ax.text(x + w / 2, y + h / 2 - 4, f'{plot} . {fam["label"].upper()}',
               fontsize=fs(8), weight="bold", ha="center", va="center")
       ax.text(x + w / 2, y + h / 2 + 4,
               f'Station {STATION[plot]} . {w:.0f} x {h:.1f} in . '
               f'{w * h / 144:.1f} sq ft', fontsize=fs(7), ha="center",
               va="center", color="#444")
   ax.plot([72, 72], [0, 51], color="#8a2f2f", lw=2.4)
   ax.text(72, -6, "two 3/4 in ball valves, above grade on the outside "
           "face, tagged by crop", fontsize=fs(6.8), ha="center",
           va="center", color="#8a2f2f")
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
       "Stations follow the bed permanently, so each October you edit crop "
       "settings in the app rather than moving wire. Station 4 is the "
       "container peppers, station 5 the annex.",
       "Shaded west end of Bed L takes sweet potato vines in allium years. "
       "Shaded ground north of Bed S grows summer lettuce, the one thing a "
       "10b July otherwise forbids."])
   save(fig, outdir, f"garden_y{year}_00_plan")


def fig_annex(outdir):
   a = ANNEX
   n = len(a["laterals"]) * len(a["emitters"])
   fig = page()
   header(fig, "THE VINE ANNEX . STATION 5 . 8 x 20 FT, 160 SQ FT",
          "Open ground north of Bed M . outside the rotation, so this sheet "
          "is the same every year . Altadena 91104, zone 10b")
   pad = 16
   ax, s = fit_axes(fig, (m(), 0.26, 1 - 2 * m(), 0.54),
                    (-pad, a["length"] + pad), (-pad, a["width"] + pad),
                    denom=SCALE_DENOM, invert=False)
   ax.add_patch(Rectangle((0, 0), a["length"], a["width"],
                          facecolor="#f1ece0", edgecolor="#6b4f2a",
                          linewidth=1.8, zorder=0))
   ax.plot([0, 116], [-3.0, -3.0], color="#4a4a4a", lw=3.4,
           solid_capstyle="butt", zorder=3)
   ax.text(58, -10.0, "BED M north edge . cucumber trellis . ~2 ft of shade "
           "in Aug, ~6 ft in Oct", fontsize=fs(6.4), ha="center",
           va="center", color="#4a4a4a")
   ax.plot([-5, -5], [0, a["width"]], color="#2f5d8a", lw=2.6, zorder=2)
   ax.text(-10.5, a["width"] / 2, "HEADER", rotation=90, fontsize=fs(6),
           va="center", ha="center", color="#2f5d8a")
   for i, y in enumerate(a["laterals"], start=1):
       ax.plot([-5, a["length"]], [y, y], color="#2f5d8a", lw=0.8,
               ls=(0, (5, 3)), zorder=2)
       ax.text(a["length"] + 3.0, y, f"L{i}", fontsize=fs(5.8), va="center",
               color="#2f5d8a")
       for x in a["emitters"]:
           ax.plot(x, y, "o", ms=fs(1.8), color="#2f5d8a", zorder=6)
   c = CORN
   cw, ch = (c["cols"] - 1) * c["step"], (c["rows"] - 1) * c["step"]
   ax.add_patch(Rectangle((c["x0"] - 6, c["y0"] - 6), cw + 12, ch + 12,
                          facecolor=PALETTE["corn"], alpha=0.16,
                          edgecolor=PALETTE["corn"], lw=1.0, zorder=3))
   for i in range(c["cols"]):
       for j in range(c["rows"]):
           ax.plot(c["x0"] + i * c["step"], c["y0"] + j * c["step"], "o",
                   ms=fs(3.6), color=PALETTE["corn"], mec="white", mew=0.6,
                   zorder=7)
   ax.text(c["x0"] + cw / 2, c["y0"] + ch + 13,
           f'SWEET CORN . {c["cols"] * c["rows"]} plants, {c["step"]} in '
           f'grid . {c["variety"]}', fontsize=fs(6.8), ha="center",
           va="bottom", weight="bold", color="#8a6d1f", zorder=9,
           path_effects=HALO)
   for hl in ANNEX_HILLS:
       col = PALETTE[hl["key"]]
       ax.add_patch(Circle((hl["x"], hl["y"]), hl["r"], facecolor=col,
                           alpha=0.17, edgecolor=col, lw=0.8, zorder=4))
       ax.plot(hl["x"], hl["y"], "o", ms=fs(6.0), color=col, mec="white",
               mew=0.9, zorder=8)
       ax.text(hl["x"], hl["y"] + 5.5, hl["label"], fontsize=fs(6.8),
               ha="center", va="bottom", weight="bold", color="#2b2b2b",
               zorder=9, path_effects=HALO)
       ax.text(hl["x"], hl["y"] - 6.0, hl["note"], fontsize=fs(5.8),
               ha="center", va="top", style="italic", color="#6b4f2a",
               zorder=9, path_effects=HALO)
       L, nl = 30.0, (hl["dx"] ** 2 + hl["dy"] ** 2) ** 0.5
       ax.annotate("", xytext=(hl["x"], hl["y"]),
                   xy=(hl["x"] + L * hl["dx"] / nl,
                       hl["y"] + L * hl["dy"] / nl),
                   arrowprops=dict(arrowstyle="-|>", lw=1.2, color=col,
                                   alpha=0.75), zorder=5)
   ax.add_patch(FancyArrow(a["length"] + 8, a["width"] - 28, 0, 22,
                           width=0.9, head_width=6, head_length=7,
                           color="#333", zorder=9))
   ax.text(a["length"] + 8, a["width"] + 1, "N", fontsize=fs(10),
           weight="bold", ha="center", va="bottom")
   fig.text(m(), 0.845,
            f'{a["width"]:.0f} x {a["length"]:.0f} in  .  8 laterals at 12 '
            f'in  .  {n} emitters at {EMITTER_GPH:g} gph  .  '
            f'{n * EMITTER_GPH:.1f} gph  .  {RATE_IN_PER_HR} in/hr, same as '
            f'the beds  .  1 in = 100 gal  .  drawn 1 : {1 / s:.0f}',
            fontsize=fs(7.4), va="top", color="#444")
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
       "Not in the rotation, so: move hills within the annex each season, "
       "cowpeas across the whole thing each winter, and clear all vine "
       "debris in Nov, because squash bugs overwinter in it."])
   legend(fig, ["vine", "melon", "cantaloupe", "corn"])
   save(fig, outdir, "annex_00_plan")


# ======================================================================
# 7. CALENDARS AND SCHEMATIC
# ======================================================================


def fig_rotation(outdir):
   fig = page()
   header(fig, "FOUR YEAR ROTATION",
          "Heavy feeder, restorative, heavy feeder, resting. Tomatoes and "
          "peppers stay two years from cucurbits, brassicas four from "
          "themselves.")
   left, top = 0.20, 0.74
   w, h = (1 - m() - left) / 4, 0.115
   for j, plot in enumerate(PLOT_ORDER):
       fig.text(left + (j + 0.5) * w, top + 0.035,
                f'{plot}\nStation {STATION[plot]}', fontsize=fs(8.5),
                weight="bold", ha="center", va="center")
   for i in range(1, 5):
       y = top - i * h
       fig.text(left - 0.012, y + h / 2, f"Year {i}\n{ROTATION_YEARS[i]}",
                fontsize=fs(7.8), ha="right", va="center")
       for j, plot in enumerate(PLOT_ORDER):
           fam = FAMILY[ROTATION[i][plot]]
           rect(fig, left + j * w, y, w, h, facecolor=fam["colour"],
                alpha=0.26, edgecolor="white", lw=2)
           fig.text(left + (j + 0.5) * w, y + h / 2, fam["label"],
                    fontsize=fs(7.6), ha="center", va="center",
                    weight="bold")
   footer(fig, [
       "Bed L holds two plots on one valve, so its pairings were chosen to be "
       "compatible in watering frequency: nightshade with allium, allium with "
       "cucurbit, then two years where a",
       "winter tenant shares with a summer tenant and the halves barely "
       "overlap at all.",
       "The annex, station 5, sits outside this table. It grows the sprawling "
       "cucurbits and corn every year, which is what makes pumpkins and "
       "melons annual rather than one year in four.",
       "Peppers overwinter as perennials in 10b and would anchor a bed, so "
       "the collection lives in containers on station 4 and only annual "
       "peppers go in the ground."])
   save(fig, outdir, "calendar_1_rotation")


def fig_schedule(outdir, year):
   fig = page()
   header(fig, f"DRIP SCHEDULE . ROTATION YEAR {year}",
          f"{ROTATION_YEARS[year]} . one 4:30 am start, stations sequential "
          f". {RATE_IN_PER_HR} in/hr, so {MIN_QUARTER_IN} min = 0.25 in")
   data = {st: zone_schedule(year, st) for st in SCHED_STATIONS}
   weekly = {st: [d[0] * d[1] for d in data[st]] for st in data}
   hi = max(max(v) for v in weekly.values()) or 1
   norm, cmap = Normalize(0, hi), plt.get_cmap("YlGnBu")
   left, top = 0.155, 0.755
   w, h = (1 - m() - left) / 12, 0.088
   for k, mon in enumerate(MONTHS):
       fig.text(left + (k + 0.5) * w, top + 0.026, mon, fontsize=fs(9),
                weight="bold", ha="center", va="center")
   for i, st in enumerate(SCHED_STATIONS):
       y = top - (i + 1) * h
       plots = [p for p in PLOT_ORDER if STATION[p] == st]
       fams = " + ".join(FAMILY[ROTATION[year][p]]["short"]
                         for p in plots) or "cucurbits, fixed"
       fig.text(left - 0.012, y + h / 2,
                f'Station {st}\n{STATION_LABEL[st]}\n{fams} . '
                f'{STATION_GPH[st]:g} gph', fontsize=fs(7.0), ha="right",
                va="center", weight="bold")
       for k in range(12):
           ev, mins = data[st][k]
           rect(fig, left + k * w, y, w, h,
                facecolor=cmap(norm(weekly[st][k])), alpha=0.88,
                edgecolor="white", lw=1.6)
           col = "white" if norm(weekly[st][k]) > 0.55 else "#222"
           cx = left + (k + 0.5) * w
           if ev == 0:
               fig.text(cx, y + h / 2, "empty\nor off", fontsize=fs(6.4),
                        ha="center", va="center", color=col, style="italic")
           else:
               fig.text(cx, y + h * 0.64, freq_label(ev), fontsize=fs(6.6),
                        ha="center", va="center", color=col)
               fig.text(cx, y + h * 0.32, f"{mins} min", fontsize=fs(8.2),
                        ha="center", va="center", weight="bold", color=col)
   y = top - 5 * h - 0.016
   fig.text(left - 0.012, y + h / 2, "BALL VALVES\nBed L, manual\nopen / "
            "shut", fontsize=fs(7.0), ha="right", va="center",
            weight="bold", color="#8a2f2f")
   for k in range(12):
       state = VALVES[year][k]
       shut = "none" if state == "both" else ("L1" if state == "L2" else "L2")
       rect(fig, left + k * w, y, w, h,
            facecolor="#f6f2ea" if state == "both" else "#f2dcdc",
            edgecolor="white", lw=1.6)
       cx = left + (k + 0.5) * w
       fig.text(cx, y + h * 0.68, f"open {state}", fontsize=fs(6.4),
                ha="center", va="center", color="#333")
       fig.text(cx, y + h * 0.44, f"shut {shut}", fontsize=fs(6.4),
                ha="center", va="center", color="#8a2f2f", weight="bold")
       note = VALVE_NOTES.get((year, k))
       if note:
           fig.text(cx, y + h * 0.18, note, fontsize=fs(5.6), ha="center",
                    va="center", color="#8a2f2f", style="italic")
   footer(fig, [
       "Bed L runs on one solenoid, so its row is the envelope of whichever "
       "halves are open. PC emitters hold their rate regardless, so closing "
       "one half does not change the other.",
       "Ball valves are open or shut, never partly. Rain: half an inch buys "
       "five days for shallow plots, an inch buys a week. Heat: add a quarter "
       "above 95 F, a 10 min midday pulse for greens above 105 F.",
       "Santa Ana wind: add twenty to thirty percent regardless of "
       "temperature, because the humidity collapse does the damage. Turn "
       "Rachio wind skip OFF, since drift is irrelevant to buried emitters.",
       "Germination, and any half behind a shut valve: a watering can. 2.5 "
       "gal covers 25 sq ft at 0.25 in. The test that overrules every number "
       "here is a finger three inches down next morning."])
   save(fig, outdir, f"calendar_2_irrigation_y{year}")


def fig_occupancy(outdir, year):
   fig = page()
   header(fig, f"CROP OCCUPANCY AND KEY DATES . YEAR {year}",
          f"{ROTATION_YEARS[year]} . the beds only . the annex runs its own "
          f"fixed calendar on its own sheet")
   left, top = 0.155, 0.735
   w, h = (1 - m() - left) / 12, 0.115
   for k, mon in enumerate(MONTHS):
       fig.text(left + (k + 0.5) * w, top + 0.028, mon, fontsize=fs(9),
                weight="bold", ha="center", va="center")
   for i, plot in enumerate(PLOT_ORDER):
       family = ROTATION[year][plot]
       y = top - (i + 1) * h
       fig.text(left - 0.012, y + h / 2,
                f'{plot} . {FAMILY[family]["label"]}\nStation '
                f'{STATION[plot]} . {BEDS[plot]["name"]}', fontsize=fs(7.2),
                ha="right", va="center", weight="bold")
       sp = OCCUPANCY[family]
       bh = (h - 0.012) / len(sp)     # plain stacking, no division by len-1
       for k, (s0, e0, label, key) in enumerate(sp):
           by = y + 0.006 + k * bh
           x0 = left + max(0.0, s0) * w
           x1 = left + min(12.0, e0) * w      # clamp: minis run past Oct
           rect(fig, x0, by, x1 - x0, bh * 0.86, facecolor=PALETTE[key],
                alpha=0.45, edgecolor=PALETTE[key], lw=0.8)
           fig.text(x0 + 0.004, by + bh * 0.43, label, fontsize=fs(6.2),
                    va="center", color="#2b2b2b")
   base = top - 4 * h - 0.022         # three staggered tiers of task ticks
   for i, (x, label) in enumerate(TASKS):
       yy = base - (i % 3) * 0.026
       cx = left + x * w
       line(fig, [cx, cx], [base + 0.012, yy + 0.004], color="#8a2f2f",
            lw=0.8)
       fig.text(cx, yy, label, fontsize=fs(5.8), ha="center", va="top",
                color="#8a2f2f",
                weight="bold" if "WATER OFF" in label else "normal")
   save(fig, outdir, f"calendar_3_occupancy_y{year}")

def fig_planting_calendar(outdir):
   fig = page()
   header(fig, "PLANTING CALENDAR . ALL CROPS",
          "Rotation year, Oct to Sep . dashed = under cover, solid = in the "
          "ground, hatched = harvest . germination at right")
   left, top = 0.205, 0.795
   w = (1 - m() - left - 0.145) / 12
   rh = 0.0196
   bottom = top - rh * (len(CROPS) + 6) - 0.02
   for k, mon in enumerate(MONTHS):
       fig.text(left + (k + 0.5) * w, top + 0.020, mon, fontsize=fs(8),
                weight="bold", ha="center", va="center")
       line(fig, [left + k * w, left + k * w], [top + 0.008, bottom],
            color="#e6e6e6", lw=0.7)
   line(fig, [left + 12 * w, left + 12 * w], [top + 0.008, bottom],
        color="#e6e6e6", lw=0.7)
   y, last = top, None
   for c in CROPS:
       if c["grp"] != last:
           y -= 0.012
           fig.text(m(), y, GRP_LABEL[c["grp"]].upper(), fontsize=fs(6.8),
                    weight="bold", color=GRP_COLOUR[c["grp"]], va="center")
           last = c["grp"]
           y -= rh
       col = GRP_COLOUR[c["grp"]]
       fig.text(left - 0.008, y, c["name"], fontsize=fs(6.6), ha="right",
                va="center")
       for s0, e0 in spans(c["indoor"]):
           rect(fig, left + s0 * w, y - rh * 0.28, (e0 - s0) * w, rh * 0.56,
                facecolor=col, alpha=0.18, edgecolor=col, lw=0.6,
                ls=(0, (2, 1.5)))
       for s0, e0 in spans(c["plant"], min_w=0.18):
           rect(fig, left + s0 * w, y - rh * 0.33, (e0 - s0) * w, rh * 0.66,
                facecolor=col, alpha=0.88, edgecolor="none")
       for s0, e0 in spans(c["harv"]):
           rect(fig, left + s0 * w, y - rh * 0.28, (e0 - s0) * w, rh * 0.56,
                facecolor=col, alpha=0.28, edgecolor=col, lw=0.6,
                hatch="////")
       fig.text(left + 12 * w + 0.006, y, c["germ"], fontsize=fs(5.6),
                va="center", color="#555")
       y -= rh
   footer(fig, [
       "Germination assumes the stated soil temperature. A seed absent after "
       "twice its window has failed, and knowing that in week two rather "
       "than week six is the whole point of the column.",
       "Carrots at 14 to 21 days are the slowest here and need the surface "
       "damp throughout, so two five minute pulses daily. Peppers at up to "
       "21 days need 80 to 85 F, not merely warm.",
       "Two dates will not move: spinach sown after mid January bolts before "
       "it is worth eating, and garlic still being watered on May 10 will "
       "not cure and will rot in storage."])
   save(fig, outdir, "calendar_5_planting")

def fig_sowing(outdir):
   fig = page()
   header(fig, "SOWING AND TASK CALENDAR . YEAR ONE",
          "Altadena 91104, zone 10b . last frost mid Feb to early Mar at "
          "this elevation . about 19 in of rain, almost all Nov to Mar")
   half = (len(SOWING) + 1) // 2
   for x0, rows in ((m(), SOWING[:half]), (0.52, SOWING[half:])):
       for i, (when, what) in enumerate(rows):
           y = 0.775 - i * 0.0245
           if i % 2 == 0:
               rect(fig, x0 - 0.006, y - 0.010, 0.45, 0.022,
                    facecolor="#f4f1ea", edgecolor="none")
           fig.text(x0, y, when, fontsize=fs(7.4), va="center",
                    weight="bold", color="#6b4f2a")
           emph = any(k in what for k in ("SHUT", "LAST", "ALL"))
           fig.text(x0 + 0.105, y, what, fontsize=fs(7.4), va="center",
                    weight="bold" if emph else "normal")
   footer(fig, [
       "Reference ET near 50 in a year, peaking around 7 in in July. The "
       "garden is watered by the sky from Nov to Mar and by you alone from "
       "Jun to Sep.",
       "Spinach is a genuine winter crop here but a short one. Mid Oct to mid "
       "Jan only, harvested young, with Malabar spinach on the trellis for "
       "summer greens.",
       "Pumpkins sown Jul 1 to 10 land exactly when the mildewed summer "
       "squash is pulled, so one piece of ground grows two crops with no "
       "conflict. Cure ten days before Halloween."])
   save(fig, outdir, "calendar_4_sowing")

# ---- schematic -------------------------------------------------------
# In strict plumbing order. Putting the filter downstream of the
# solenoids is the classic mistake and it quietly ruins valve diaphragms.
HEAD_ASSEMBLY = [
   ("SUPPLY", "3/4 in bib or hard plumb\nfoothill pressure often 80+ psi",
    "#8a8a8a"),
   ("BACKFLOW", "type per local code\ncheck with your purveyor", "#7a6a9a"),
   ("BALL VALVE", "3/4 in isolation\nservice everything downstream",
    "#8a2f2f"),
   ("Y FILTER", "150 to 200 mesh\nUPSTREAM of the solenoids", "#5a7a9a"),
   ("PRV", "adjustable, spring + diaphragm\nSET TO 25 PSI", "#2f5d8a"),
   ("GAUGE", "0 to 100 psi, liquid filled\npermanent, for setting the PRV",
    "#4a4a4a"),
   ("METER", "low flow totalizer, from 0.1 gpm\nmonthly reading = your log",
    "#6b8a5a"),
]

# station, label, inline emitter count or None, gph, note
SCHEM_STATIONS = [
   (1, "BED L\nL1 + L2", 48, 28.7,
    "one solenoid, TWO manual ball valves\nat the bed, tagged by crop"),
   (2, "BED M", 40, 23.9, "4 laterals, 10 emitters each"),
   (3, "BED S", 28, 16.7, "4 laterals, 7 emitters each"),
   (4, "CONTAINERS", None, 12.0,
    "1/4 in microtube + point emitters\nthe overwintering peppers"),
   (5, "ANNEX\n8 x 20 ft", 160, 95.7,
    "15 ft buried run, solenoid AT the annex\n8 laterals, 20 emitters each"),
]

def fig_schematic(outdir):
   total = sum(s[3] for s in SCHEM_STATIONS)
   n_emit = sum(s[2] for s in SCHEM_STATIONS if s[2])
   fig = page()
   header(fig, "HYDRAULIC SCHEMATIC . NOT TO SCALE",
          f"What feeds what . 5 of 16 stations used . Altadena 91104 . "
          f"{EMITTER_GPH:g} gph emitters on a 12 in grid = "
          f"{RATE_IN_PER_HR} in/hr")

   ax = fig.add_axes([m(), 0.17, 1 - 2 * m(), 0.66])
   ax.set_xlim(0, 100)
   ax.set_ylim(0, 64)
   ax.axis("off")

   # No equal aspect: this is a schematic, so data units stretch freely.
   # Every round dot is a marker rather than a Circle patch, which would
   # otherwise render as an ellipse.

   def box(x, y, w, h, title, sub, colour, sub_pt=5.6):
       ax.add_patch(Rectangle((x, y), w, h, facecolor=colour, alpha=0.16,
                              edgecolor=colour, lw=1.3, zorder=3))
       ax.text(x + w / 2, y + h * 0.68, title, fontsize=fs(6.9),
               weight="bold", ha="center", va="center", color="#222",
               zorder=5, linespacing=1.3)
       ax.text(x + w / 2, y + h * 0.28, sub, fontsize=fs(sub_pt),
               ha="center", va="center", color="#444", zorder=5,
               linespacing=1.35)

   def arrow(x0, y0, x1, y1, colour="#2f5d8a", lw=1.6):
       ax.annotate("", xytext=(x0, y0), xy=(x1, y1),
                   arrowprops=dict(arrowstyle="-|>", lw=lw, color=colour,
                                   shrinkA=0, shrinkB=0), zorder=2)

   bw, bh, gap, ytop = 12.2, 9.0, 1.6, 50.0
   for i, (title, sub, colour) in enumerate(HEAD_ASSEMBLY):
       x = i * (bw + gap)
       box(x, ytop, bw, bh, title, sub, colour)
       if i:
           arrow(x - gap, ytop + bh / 2, x, ytop + bh / 2)
   ax.text(0, ytop + bh + 2.6, "HEAD ASSEMBLY . order matters . the filter "
                               "protects the valve diaphragms as well as the emitters",
           fontsize=fs(6.6), color="#555", va="bottom")

   x_last = 6 * (bw + gap) + bw / 2
   y_bus_top = 43.0
   arrow(x_last, ytop, x_last, y_bus_top)
   ax.plot([72, x_last], [y_bus_top, y_bus_top], color="#2f5d8a", lw=1.6,
           zorder=2)
   box(28, 34.0, 44, 9.0, "MANIFOLD . 16 STATION RACHIO",
       "5 used, 11 spare . stations run SEQUENTIALLY, so the supply never "
       "sees more than 96 gph\none 4:30 am start . Flex Daily . wind skip "
       "OFF . rain skip ON at 0.25 in", "#2f5d8a", sub_pt=5.9)

   sw, sh = 17.0, 13.5
   sgap = (100 - 5 * sw) / 4
   y_bus, y_sta = 30.5, 13.0
   ax.plot([sw / 2, 100 - sw / 2], [y_bus, y_bus], color="#2f5d8a", lw=1.6,
           zorder=2)
   ax.plot([50, 50], [34.0, y_bus], color="#2f5d8a", lw=1.6, zorder=2)
   for i, (st, label, em, gph, note) in enumerate(SCHEM_STATIONS):
       x = i * (sw + sgap)
       cx = x + sw / 2
       arrow(cx, y_bus, cx, y_sta + sh)
       ax.plot(cx, y_bus, marker="o", ms=fs(4.2), mfc="white",
               mec="#8a2f2f", mew=1.3, zorder=4)
       ax.text(cx + 2.2, y_bus + 1.8, f"S{st}", fontsize=fs(6.2),
               weight="bold", color="#8a2f2f", zorder=5)
       emtxt = f"{em} emitters at {EMITTER_GPH:g} gph\n" if em else ""
       box(x, y_sta, sw, sh, f"STATION {st}\n{label}",
           f"{emtxt}{gph:g} gph, {gph / 60:.2f} gpm\n{note}", "#6b8a5a",
           sub_pt=5.5)

   cx1 = sw / 2
   ax.plot([cx1, cx1], [y_sta, 8.0], color="#8a2f2f", lw=1.6, zorder=2)
   ax.plot([3.0, sw - 3.0], [8.0, 8.0], color="#8a2f2f", lw=1.6, zorder=2)
   for xx, lab in ((3.0, "L1"), (sw - 3.0, "L2")):
       arrow(xx, 8.0, xx, 4.8, colour="#8a2f2f")
       ax.add_patch(Rectangle((xx - 3.4, 0.8), 6.8, 4.0, facecolor="#8a2f2f",
                              alpha=0.14, edgecolor="#8a2f2f", lw=1.2,
                              zorder=3))
       ax.text(xx, 3.5, lab, fontsize=fs(6.4), weight="bold", ha="center",
               va="center", zorder=5)
       ax.text(xx, 1.8, "24 emitters", fontsize=fs(5.0), ha="center",
               va="center", color="#555", zorder=5)
   ax.text(sw + 2.5, 4.6,
           "MANUAL BALL VALVES, above grade on the outside face of the box, "
           "tagged by crop rather than by plot.\nOpen or shut, never partly, "
           "since a throttled valve drops PC emitters below their range.\n"
           "Closing one half does not change the rate in the other. Shut the "
           "allium half on MAY 10, reopen in June.",
           fontsize=fs(5.9), va="center", color="#8a2f2f", linespacing=1.45)

   fig.text(m(), 0.845,
            f"{n_emit} inline emitters plus 12 container drippers  .  "
            f"{total:.1f} gph if everything ran at once, {total / 60:.1f} gpm"
            f"  .  sequential, so the true peak is 95.7 gph or 1.6 gpm at the "
            f"annex  .  {MIN_QUARTER_IN} min = 0.25 in  .  {MIN_ONE_IN} min "
            f"= 1 in",
            fontsize=fs(7.4), va="top", color="#444")
   footer(fig, [
       "The PRV must be an ADJUSTABLE spring and diaphragm type, not a fixed "
       "25 psi drip regulator. Those need about 0.5 gpm to regulate at all, "
       "and three of these five zones sit below that.",
       "Bed S at 0.28 gpm is barely half the threshold, which is also why the "
       "tubing is pressure compensating: PC emitters hold their rate from "
       "roughly 15 to 50 psi and forgive the rest.",
       "Station 5 puts its solenoid AT the annex rather than at the manifold, "
       "so the 15 ft buried run sits depressurised whenever the annex is not "
       "watering. Wire in the same trench, laid loose beside the pipe.",
       "Diagnostics: 25 psi at a far lateral gauge means all is well, above "
       "40 means the PRV has failed, zero during a run means valve, filter or "
       "shutoff rather than emitters. Time the bleed down and log it.",
       "Flush all 24 lateral ends once or twice a year wide open, then an "
       "hour of citric acid or dilute vinegar if the bleed down has grown "
       "slower. Foothill water is hard and closes emitters gradually."])
   save(fig, outdir, "system_00_schematic")

def fig_cover(outdir, years):
   fig = page()
   fig.text(m(), 1 - m(0.5, "y"), "GARDEN PLAN", fontsize=fs(26),
            weight="bold", va="top")
   fig.text(m(), 1 - m(0.95, "y"), "Three raised beds on a four year rotation "
                                   "grouped by water need, plus a fixed vine annex",
            fontsize=fs(11.5), va="top", color="#555")
   fig.text(m(), 1 - m(1.22, "y"),
            f"Altadena, California 91104  .  USDA zone 10b  .  generated "
            f"{datetime.date.today().isoformat()}", fontsize=fs(8.5),
            va="top", color="#888")
   line(fig, [m(), 1 - m()], [0.845, 0.845], color="#6b4f2a", lw=1.8)

   rows = [
       ("GROUND", None),
       ("Bed L", "51 x 144 in . 51.0 sq ft . split at 72 in into L1 and L2"),
       ("Bed M", "56.5 x 116 in . 45.5 sq ft . one plot"),
       ("Bed S", "53 x 79 in . 29.1 sq ft . one plot"),
       ("annex", "96 x 240 in . 160 sq ft . north of Bed M, not rotated"),
       ("total", "286 sq ft planted . roughly 6,000 gallons a year"),
       ("STATIONS", None),
       ("Station 1", "Bed L, both halves, 28.7 gph, ball valves select halves"),
       ("Station 2", "Bed M, 23.9 gph"),
       ("Station 3", "Bed S, 16.7 gph"),
       ("Station 4", "containers, the overwintering pepper collection"),
       ("Station 5", "annex, 160 emitters, 95.7 gph"),
       ("THE NUMBER EVERYTHING RESTS ON", None),
       ("0.6 gph at 12 in", f"= {RATE_IN_PER_HR} in/hr in the beds AND the "
                            f"annex"),
       ("therefore", f"{MIN_QUARTER_IN} min = 0.25 in, {MIN_ONE_IN} min = "
                     f"1.00 in"),
       ("CONTENTS", None),
       ("6 fixed sheets", "cover, rotation, annex, schematic, sowing, "
                          "planting calendar"),
       ("7 per year", "plot plan, drip schedule, occupancy, four bed maps, "
                      "plus the summer allium map"),
   ]
   y = 0.795
   for label, detail in rows:
       if detail is None:
           y -= 0.010
           fig.text(m(), y, label, fontsize=fs(9.5), weight="bold",
                    color="#6b4f2a", va="top")
           y -= 0.033
       else:
           fig.text(m() + 0.02, y, label, fontsize=fs(9), weight="bold",
                    va="top", color="#333")
           fig.text(m() + 0.20, y, detail, fontsize=fs(9), va="top",
                    color="#555")
           y -= 0.029
   footer(fig, [
       "Set the controller to a single 4:30 am start with stations running "
       "sequentially. The whole garden finishes before seven, which keeps "
       "foliage dry and reduces mildew and blight pressure.",
       "The one instruction no script can remember for you: shut the ball "
       "valve on the allium half of Bed L on the tenth of May. Wet bulbs will "
       "not cure and will rot in storage."])
   save(fig, outdir, "calendar_0_cover")

# ======================================================================
# 8. MAIN
# ======================================================================

def main():
   global PDF, WRITE_IMAGES, PAGE, DPI, MARGIN, SCALE_DENOM, PANEL
   ap = argparse.ArgumentParser()
   ap.add_argument("--year", type=int, choices=[1, 2, 3, 4])
   ap.add_argument("--outdir", default="garden_out")
   ap.add_argument("--pdf", default="garden_plan.pdf",
                   help="single printable PDF, or 'none'")
   ap.add_argument("--page", default="letter", choices=list(PAGE_SIZES))
   ap.add_argument("--margin", type=float, default=0.30,
                   help="page margin in inches")
   ap.add_argument("--scale-denom", type=int, default=None,
                   help="lock maps to 1:N, e.g. 16. Default fits the page")
   ap.add_argument("--no-panel", action="store_true",
                   help="omit the side panel, giving bed maps full width")
   ap.add_argument("--dpi", type=int, default=200)
   ap.add_argument("--no-images", action="store_true")
   args = ap.parse_args()

   PAGE = PAGE_SIZES[args.page]
   MARGIN = args.margin
   SCALE_DENOM = args.scale_denom
   PANEL = not args.no_panel
   DPI = args.dpi
   WRITE_IMAGES = not args.no_images
   os.makedirs(args.outdir, exist_ok=True)
   years = [args.year] if args.year else [1, 2, 3, 4]

   pdf_path = None
   if args.pdf.lower() != "none":
       pdf_path = os.path.join(args.outdir, args.pdf)
       PDF = PdfPages(pdf_path)
       info = PDF.infodict()
       info["Title"] = "Garden plan, Altadena 91104, four year rotation"
       info["Author"] = "Eric"
       info["Subject"] = ("Bed maps, annex plan, schematic, drip schedules "
                          "and planting calendars, USDA zone 10b")
       info["CreationDate"] = datetime.datetime.now()

   try:
       print(f"page {PAGE[0]} x {PAGE[1]} in, margin {MARGIN} in"
             + (f", scale 1:{SCALE_DENOM}" if SCALE_DENOM else "")
             + ("" if PANEL else ", no side panel"))
       print("fixed sheets:")
       fig_cover(args.outdir, years)
       fig_rotation(args.outdir)
       fig_annex(args.outdir)
       fig_schematic(args.outdir)
       fig_sowing(args.outdir)
       fig_planting_calendar(args.outdir)
       for y in years:
           print(f"year {y}:")
           garden_plan(args.outdir, y)
           fig_schedule(args.outdir, y)
           fig_occupancy(args.outdir, y)
           for plot in PLOT_ORDER:
               family = ROTATION[y][plot]
               plot_map(args.outdir, y, plot, family)
               if family == "allium":
                   plot_map(args.outdir, y, plot, family, season="summer")
   finally:
       if PDF is not None:
           PDF.close()
           PDF = None

   print(f"\n{len(os.listdir(args.outdir))} files in {args.outdir}/")
   if pdf_path:
       print("printable PDF:", pdf_path)

if __name__ == "__main__":
   main()