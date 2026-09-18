"""
garden_data.py . Altadena CA 91104 . zone 10b

Everything you would ever edit: bed geometry, varieties, water targets,
the rotation, and the task calendar. No drawing code. Runnable on its
own for a terminal check:

   python garden_data.py            # summary
   python garden_data.py --tasks 1  # year one task list

Beds: L is 51 x 144, split at 72 into L1 and L2 on ONE solenoid with two
manual ball valves. M is 56.5 x 116 with FIVE laterals. S is 53 x 79.
Annex is 96 x 240 of open ground north of M, outside the rotation.
Emitters are 0.598 gph (metric 2.26 L/h) at 12 in along every lateral.
"""

import argparse
import math

GAL_CU_IN = 231.0          # 1 US gallon in cubic inches
EMITTER_GPH = 0.598
EMITTER_SPACING = 12.0
TRAY_YIELD = 0.75          # germination is never total; thin ruthlessly

MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
         "Apr", "May", "Jun", "Jul", "Aug", "Sep"]

# ======================================================================
# 1. GEOMETRY
#     Lateral positions are measured, not assumed, and the application
#     rate is derived from them. Bed M carries five laterals at 11 in,
#     so it applies water FASTER than the others and wants shorter runs.
# ======================================================================

PLOTS = {
   "L1": dict(name="Bed L, north half", bed="Bed L", width=51.0, length=72.0,
              geom="Lhalf", station=1, offset=0,
              laterals=[7.5, 19.5, 31.5, 43.5]),
   "L2": dict(name="Bed L, south half", bed="Bed L", width=51.0, length=72.0,
              geom="Lhalf", station=1, offset=72,
              laterals=[7.5, 19.5, 31.5, 43.5]),
   "M":  dict(name="Bed M", bed="Bed M", width=56.5, length=116.0,
              geom="M", station=2, offset=0,
              laterals=[6.25, 17.25, 28.25, 39.25, 50.25]),
   "S":  dict(name="Bed S", bed="Bed S", width=53.0, length=79.0,
              geom="S", station=3, offset=0,
              laterals=[7.75, 20.25, 32.75, 45.25]),
}
PLOT_ORDER = ["L1", "L2", "M", "S"]

ANNEX = dict(name="Vine annex", width=96.0, length=240.0, station=5,
            laterals=[6 + 12 * i for i in range(8)])

CONTAINERS = dict(station=4, count=8, dripper_gph=1.0, size_gal=15)

STATION_LABEL = {1: "Bed L (L1 + L2)", 2: "Bed M", 3: "Bed S",
                4: "containers", 5: "annex"}


def emitters_for(length):
   n = max(1, int((length - EMITTER_SPACING) // EMITTER_SPACING) + 1)
   return [EMITTER_SPACING / 2 + i * EMITTER_SPACING for i in range(n)]


def grid_for(plot):
   """Emitter positions, flow and application rate from the laterals."""
   lat = plot["laterals"]
   em = emitters_for(plot["length"])
   spacing = (lat[1] - lat[0]) if len(lat) > 1 else plot["width"]
   cell = spacing * EMITTER_SPACING
   rate = EMITTER_GPH * GAL_CU_IN / cell
   n = len(lat) * len(em)
   return dict(laterals=lat, emitters=em, count=n,
               lateral_spacing=round(spacing, 2),
               gph=round(n * EMITTER_GPH, 1), rate=round(rate, 2),
               min_quarter=round(15 / rate), min_one=round(60 / rate))


GRIDS = {k: grid_for(v) for k, v in PLOTS.items()}
GRIDS["annex"] = grid_for(ANNEX)


def area_sqft(plot):
   return plot["width"] * plot["length"] / 144.0


# ======================================================================
# 2. ROTATION AND VALVES
# ======================================================================

ROTATION = {
   1: {"L1": "nightshade", "L2": "allium", "M": "cucurbit", "S": "brassica"},
   2: {"L1": "allium", "L2": "cucurbit", "M": "brassica", "S": "nightshade"},
   3: {"L1": "cucurbit", "L2": "brassica", "M": "nightshade", "S": "allium"},
   4: {"L1": "brassica", "L2": "nightshade", "M": "allium", "S": "cucurbit"},
}
ROTATION_YEARS = {1: "Oct 2026 to Sep 2027", 2: "Oct 2027 to Sep 2028",
                 3: "Oct 2028 to Sep 2029", 4: "Oct 2029 to Sep 2030"}

FAMILY_LABEL = {"nightshade": "Tomatoes + peppers",
               "cucurbit": "Cucumbers + squash",
               "brassica": "Kale, spinach + greens",
               "allium": "Legumes, alliums, roots"}
FAMILY_COLOUR = {"nightshade": "#c0504d", "cucurbit": "#e0b13a",
                "brassica": "#5c8a6a", "allium": "#b9a06a"}
FREQ_CLASS = {"nightshade": "deep", "allium": "deep",
             "cucurbit": "medium", "brassica": "shallow"}

# Which half of Bed L is OPEN each month. Years 3 and 4 pair a winter
# tenant with a summer tenant, so the halves barely overlap and one stays
# shut for months.
VALVES = {
   1: ["both"] * 12,
   2: ["both"] * 12,
   3: ["both", "both", "L2", "L2", "L2", "L2", "L2",
       "L1", "L1", "L1", "L1", "both"],
   4: ["both", "both", "L1", "L1", "L1", "both", "both",
       "L2", "L2", "L2", "L2", "both"],
}
VALVE_NOTES = {(1, 7): "shut L2 May 10", (1, 8): "reopen L2, slips",
              (2, 7): "shut L1 May 10", (2, 8): "reopen L1, slips",
              (3, 7): "clear L2 early May", (4, 5): "N in, B cropping"}


def plot_of(year, family):
   for k in PLOT_ORDER:
       if ROTATION[year][k] == family:
           return k
   return None


# ======================================================================
# 3. WATER
#     Target inches per week, no rain, well mulched. Minutes are DERIVED
#     from each plot's own measured rate, which is why the five-lateral
#     change in Bed M flows through without retyping any table.
# ======================================================================

NEED = {
   "nightshade": [.55, .18, .18, .18, .20, .32, .45, .70, .95, 1.20, 1.20, .85],
   "cucurbit":   [.42, .18, .18, .18, .20, .20, .28, .62, 1.00, 1.55, 1.35, .92],
   "brassica":   [.52, .32, .22, .22, .26, .48, .70, .82, .26, .26, .26, .52],
   "allium":     [.42, .22, .22, .22, .26, .50, .68, .78, .78, 1.05, 1.05, .70],
   "annex":      [.35, 0, 0, 0, 0, 0, 0, .95, 1.35, 1.65, 1.65, 1.25],
}
FREQ = {"deep":    [2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2],
       "medium":  [2, 1, 1, 1, 1, 1, 3, 3, 3.5, 3.5, 3.5, 3],
       "shallow": [3, 2, 1, 1, 1, 2, 3, 4, 1, 1, 1, 3]}

# Pots have no meaningful area, so this zone is volumetric. A 15 gal
# fabric pot in full sun holds about 0.3 gal of readily available water
# in its upper half, so summer events approach that and frequency works.
CONTAINER_GAL = [.25, .15, .10, .10, .12, .20, .30, .40, .50, .60, .60, .45]
CONTAINER_FREQ = [2, 1, 1, 1, 1, 2, 3, 4, 5, 7, 7, 4]


def schedule(need_key, rate):
   freq = FREQ[FREQ_CLASS.get(need_key, "medium")]
   out = []
   for i in range(12):
       inches, ev = NEED[need_key][i], freq[i]
       if inches <= 0 or ev <= 0:
           out.append((0, 0))
       else:
           out.append((ev, max(1, round(inches / ev / rate * 60))))
   return out


def envelope(scheds):
   """Shared valve: higher frequency, longer event, zeroes ignored so an
   empty half cannot silence a growing one."""
   out = []
   for i in range(12):
       c = [s[i] for s in scheds if s[i][0] > 0] or [(0, 0)]
       out.append((max(x[0] for x in c), max(x[1] for x in c)))
   return out


def station_schedule(station, year):
   if station == 4:
       g = CONTAINERS["dripper_gph"]
       return [(CONTAINER_FREQ[i], max(1, round(CONTAINER_GAL[i] / g * 60)))
               for i in range(12)]
   if station == 5:
       return schedule("annex", GRIDS["annex"]["rate"])
   keys = [k for k in PLOT_ORDER if PLOTS[k]["station"] == station]
   open_now = VALVES[year]
   scheds = []
   for k in keys:
       fam = ROTATION[year][k]
       s = schedule(fam, GRIDS[k]["rate"])
       if len(keys) > 1:
           s = [s[i] if open_now[i] in ("both", k) else (0, 0)
                for i in range(12)]
       scheds.append(s)
   return envelope(scheds) if len(scheds) > 1 else scheds[0]


def freq_label(ev):
   if ev == 3.5:
       return "every 2nd day"
   if abs(ev - 2.33) < .05:
       return "every 3rd day"
   return f"{ev:g} x week"


# ======================================================================
# 4. VARIETIES
#     Lists cycle, so slots are filled from the front. Put what you want
#     EVERY year first: a Bed L half offers only 6 tomato slots, so
#     entries 7 to 9 appear solely in the Bed M year.
# ======================================================================

TOMATOES = [("Big Beef", 18), ("Sungold", 18), ("Solar Fire", 15),
           ("Black Cherry", 18), ("Heatmaster", 15), ("Momotaro", 17),
           ("Jaune Flamme", 16), ("Black Krim", 17), ("Cherokee Purple", 18)]
PEPPERS = [("Jimmy Nardello", 11), ("jalapeno", 10), ("Corno di Toro", 11),
          ("serrano", 10), ("shishito", 11), ("padron", 11),
          ("poblano", 12), ("Aji Amarillo", 12)]
BRASSICAS = [("Lacinato kale", 12), ("Green Magic", 11), ("Snow Crown", 11),
            ("Dazzling Blue kale", 12), ("De Cicco", 11),
            ("Red Russian kale", 12), ("broccolini", 10),
            ("Redbor kale", 12), ("collards", 12), ("Green Magic", 11)]
CUCUMBERS = [("Calypso", 9), ("Excelsior", 9), ("Boston Pickling", 9),
            ("Marketmore 76", 9), ("Northern Pickling", 9)]
SQUASH = [("Dunja", 20), ("Success PM", 20)]
MINIS = [("Jack Be Little", 10), ("Baby Boo", 10)]
SPINACH = ["Space", "Corvair", "Regiment", "Red Kitten", "Bloomsdale"]
GREENS = ["lettuce", "mizuna", "Tokyo Bekana", "arugula", "cilantro",
         "Joi Choi", "Ruby Streaks"]

ROW_SEED = dict(garlic="California Early and Lorz Italian, softneck",
               onion="Texas Early Grano 502 and Red Creole, short day",
               pea="Super Sugar Snap", carrot="Nelson, Napoli, Danvers 126",
               radish="French Breakfast")

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
# 5. LAYOUTS, as (lateral, emitter) pairs
#     Bed M now has five laterals, so the nightshade year is symmetric:
#     tomatoes on the outer pair 44 in apart, peppers on the inner pair,
#     herbs down the centre line. The brassica year gains a third
#     succession row.
# ======================================================================

HINTS = {
   ("nightshade", "Lhalf"): dict(
       tom=[(1, 1), (1, 3), (1, 5), (4, 2), (4, 4), (4, 6)],
       pep=[(2, 1), (2, 3), (2, 5)], herb=[3]),
   ("nightshade", "S"): dict(
       tom=[(1, 1), (1, 3), (1, 5), (1, 7), (4, 2), (4, 4), (4, 6)],
       pep=[(2, 1), (2, 3), (2, 5), (2, 7)], herb=[3]),
   ("nightshade", "M"): dict(
       tom=[(1, 1), (1, 3), (1, 5), (1, 7), (1, 9),
            (5, 2), (5, 4), (5, 6), (5, 8)],
       pep=[(2, 2), (2, 4), (2, 6), (2, 8),
            (4, 2), (4, 4), (4, 6), (4, 8)], herb=[3]),

   ("cucurbit", "Lhalf"): dict(
       cuc=[(1, 1), (1, 3), (1, 5)], mini=[(1, 2), (1, 4)],
       sq=[(2, 1), (2, 6)]),
   ("cucurbit", "S"): dict(
       cuc=[(1, 1), (1, 3), (1, 5), (1, 7)], mini=[(1, 2), (1, 4)],
       sq=[(3, 2), (3, 6)]),
   ("cucurbit", "M"): dict(
       cuc=[(1, 1), (1, 3), (1, 5), (1, 7), (1, 9)], mini=[(1, 2), (1, 6)],
       sq=[(3, 2), (3, 6), (3, 10), (5, 4), (5, 8)]),

   ("brassica", "Lhalf"): dict(
       big=[(1, 1), (1, 3), (1, 5), (4, 1), (4, 3), (4, 5)],
       spin=[2], grn=[3]),
   ("brassica", "S"): dict(
       big=[(1, 1), (1, 3), (1, 5), (1, 7), (4, 1), (4, 3), (4, 5), (4, 7)],
       spin=[2], grn=[3]),
   ("brassica", "M"): dict(
       big=[(1, 1), (1, 3), (1, 5), (1, 7), (1, 9),
            (5, 1), (5, 3), (5, 5), (5, 7), (5, 9)],
       spin=[2], grn=[3, 4]),
}

# Band crops: rows, in-row spacing, fraction of length used.
BAND_DENSITY = dict(garlic=(2, 6, 1.0), onion=(2, 5, 1.0), pea=(1, 3, 1.0),
                   bean=(1, 6, 1.0), carrot=(1, 1.0, 0.8), radish=(1, 4, 1.0))


def band_count(crop, length):
   rows, spacing, frac = BAND_DENSITY[crop]
   return int(rows * length * frac / spacing)


def allium_bands(key):
   """Bed M gets five bands, so it carries two staggered root bands. The
   other plots get four. Every band is a spring finisher, which is
   exactly why the whole zone can dry down together in mid May."""
   p = PLOTS[key]
   L, lat = p["length"], p["laterals"]
   b = [dict(lat=1, key="pea", title="SNAP PEAS, trellised",
             n=band_count("pea", L), crop="pea",
             after="out Mar 20, then bush beans"),
        dict(lat=2, key="garlic", title="GARLIC, double row",
             n=band_count("garlic", L), crop="garlic",
             after="WATER OFF May 10"),
        dict(lat=3, key="onion", title="ONIONS, double row",
             n=band_count("onion", L), crop="onion",
             after="WATER OFF May 10"),
        dict(lat=4, key="root", title="CARROTS + radish",
             n=band_count("carrot", L), crop="carrot",
             after="last pull by May")]
   if len(lat) == 5:
       b[3]["n"] = band_count("carrot", L) // 2
       b[3]["title"] = "CARROTS, early successions"
       b.append(dict(lat=5, key="root", title="CARROTS, late successions",
                     n=band_count("carrot", L) // 2, crop="carrot",
                     after="last pull by May"))
   return b


def layout(key, family, season="winter"):
   """Returns (plants, bands). Plants carry x, y, radius, colour key and
   label; bands are drawn as stripes because true row spacing merges
   into a solid smear."""
   p = PLOTS[key]
   g = GRIDS[key]
   lat, em = g["laterals"], g["emitters"]

   def xy(l, e):
       # Clamps or wraps 1-based lateral and emitter indices to available bounds
       safe_l = max(0, min(l - 1, len(lat) - 1))
       safe_e = (e - 1) % len(em) if len(em) > 0 else 0
       return em[safe_e], lat[safe_l]

   plants, bands = [], []
   if family == "allium":
       if season == "summer":
           for l in (2, 3):
               x = 6.0
               while x <= p["length"] - 6:
                   plants.append(dict(x=x, y=lat[l - 1], r=16,
                                      key="sweetpotato", label=""))
                   x += 18
           return plants, []
       for b in allium_bands(key):
           bands.append(dict(y=lat[b["lat"] - 1], half=4.0, key=b["key"],
                             title=f'L{b["lat"]} . {b["title"]}',
                             detail=f'{ROW_SEED[b["crop"]]}, {b["n"]} '
                                    f'{"cloves" if b["crop"] == "garlic" else "plants"}',
                             after=b["after"]))
       return [], bands

   h = HINTS[(family, p["geom"])]
   if family == "nightshade":
       for i, cell in enumerate(h["tom"]):
           x, y = xy(*cell)
           name, r = TOMATOES[i % len(TOMATOES)]
           plants.append(dict(x=x, y=y, r=r, key="tomato", label=name))
       for i, cell in enumerate(h["pep"]):
           x, y = xy(*cell)
           name, r = PEPPERS[i % len(PEPPERS)]
           plants.append(dict(x=x, y=y, r=r, key="pepper", label=name))
       for l in h["herb"]:
           for k, x in enumerate(em):
               basil = k % 2 == 0
               plants.append(dict(x=x, y=lat[l - 1], r=7 if basil else 5,
                                  key="basil" if basil else "marigold",
                                  label=("basil" if k == 0 else
                                         "marigold" if k == 1 else "")))
   elif family == "cucurbit":
       for i, cell in enumerate(h["cuc"]):
           x, y = xy(*cell)
           name, r = CUCUMBERS[i % len(CUCUMBERS)]
           plants.append(dict(x=x, y=y, r=r, key="cucumber", label=name))
       for i, cell in enumerate(h["mini"]):
           x, y = xy(*cell)
           name, r = MINIS[i % len(MINIS)]
           plants.append(dict(x=x, y=y, r=r, key="mini", label=name))
       for i, cell in enumerate(h["sq"]):
           x, y = xy(*cell)
           name, r = SQUASH[i % len(SQUASH)]
           plants.append(dict(x=x, y=y, r=r, key="squash", label=name))
   elif family == "brassica":
       for i, cell in enumerate(h["big"]):
           x, y = xy(*cell)
           name, r = BRASSICAS[i % len(BRASSICAS)]
           k = "kale" if ("kale" in name or name == "collards") else "brassica"
           plants.append(dict(x=x, y=y, r=r, key=k, label=name))
       for l in h["spin"]:
           for k, x in enumerate(em):
               plants.append(dict(x=x, y=lat[l - 1], r=5, key="spinach",
                                  label=SPINACH[k] if k < 2 else ""))
       for l in h["grn"]:
           for k, x in enumerate(em):
               plants.append(dict(x=x, y=lat[l - 1], r=5, key="greens",
                                  label=GREENS[k] if k < 2 and l == h["grn"][0]
                                  else ""))
   return plants, bands


def counts(key, family):
   """Plant tallies for the task list and the side panel."""
   p = PLOTS[key]
   if family == "allium":
       return {b["crop"]: b["n"] for b in allium_bands(key)}
   h = HINTS[(family, p["geom"])]
   n_em = len(GRIDS[key]["emitters"])
   if family == "nightshade":
       return dict(tomato=len(h["tom"]), pepper=len(h["pep"]),
                   basil=len(h["herb"]) * math.ceil(n_em / 2),
                   marigold=len(h["herb"]) * (n_em // 2))
   if family == "cucurbit":
       return dict(cucumber=len(h["cuc"]), mini=len(h["mini"]),
                   squash=len(h["sq"]))
   return dict(large=len(h["big"]), spinach=len(h["spin"]) * n_em,
               greens=len(h["grn"]) * n_em)


def brassica_mix(n):
   kale = math.ceil(n / 2)
   broc = 2 if n < 9 else 3
   extra = n - kale - broc - 1
   parts = [f"{kale} kale", f"{broc} broccoli", "1 cauliflower"]
   if extra > 0:
       parts.append(f"{extra} broccolini or collards")
   return ", ".join(parts)


# ======================================================================
# 6. THE ANNEX, identical every year
# ======================================================================

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
        dx=0, dy=-1, note="30 in clears the trellis shadow"),
   dict(x=150, y=30, r=24, key="cantaloupe", label="Sugar Cube",
        dx=0, dy=-1, note="south strip"),
]
CORN = dict(x0=186, y0=42, cols=5, rows=5, step=12, variety="Bodacious")

# ======================================================================
# 7. TASK CALENDAR
#     Dates are data; every quantity is computed from the layouts above,
#     so the five-lateral Bed M changes the numbers without my retyping
#     them. Harvest is deliberately absent.
# ======================================================================

ORDER = ["Sep", "Sep 5 to 12", "Sep 10", "Sep 15 on", "Oct 1 on",
        "Oct 15 on", "Oct 15 to 25", "Oct 25", "Nov, week 1", "Nov 5 to 15",
        "Nov 10 to 20", "Dec", "Jan 15", "Jan 20", "mid Jan", "Feb",
        "mid Feb", "Mar 1", "Mar 15", "Mar 20", "Apr 10", "mid Apr",
        "May 1", "May 10", "late May", "Jun 1", "Jun 1 to 10", "mid Jun",
        "early Jul", "Jul 1 to 10", "each season"]


def trays(n):
   return math.ceil(n / TRAY_YIELD)


def tasks(year):
   R = []

   def add(when, action, units, kind=""):
       R.append(dict(when=when, action=action, units=units, kind=kind))

   b, a = plot_of(year, "brassica"), plot_of(year, "allium")
   n_, c = plot_of(year, "nightshade"), plot_of(year, "cucurbit")

   # brassicas
   cb = counts(b, "brassica")
   add("Sep 5 to 12", f"start brassicas in trays for {b}",
       f'{cb["large"]} plants: {brassica_mix(cb["large"])}. '
       f'Sow {trays(cb["large"])} cells', "tray")
   add("Oct 1 on", f"direct sow lettuce and Asian greens in {b}",
       f'{cb["greens"]} positions, resown every 3 weeks to mid Feb', "direct")
   add("Oct 15 on", f"direct sow spinach in {b}",
       f'{cb["spinach"]} positions, ~20 seeds each, every 2 to 3 weeks',
       "direct")
   add("Oct 15 to 25", f"transplant brassicas into {b}",
       f'{cb["large"]} plants on the outer laterals, row cover on', "plant")
   add("mid Jan", "last spinach sowing of the season",
       "later sowings bolt before they are worth eating", "direct")
   add("late May", f"clear {b}, broadcast buckwheat",
       f'{area_sqft(PLOTS[b]):.0f} sq ft, about '
       f'{area_sqft(PLOTS[b]) * 0.04:.1f} oz', "direct")
   add("early Jul", f"chop buckwheat, sow cowpeas in {b}",
       f'{area_sqft(PLOTS[b]):.0f} sq ft, about '
       f'{area_sqft(PLOTS[b]) * 0.08:.1f} oz', "direct")

   # alliums and roots
   ca = counts(a, "allium")
   lb = 1.0 if ca["garlic"] > 30 else 0.5
   add("Sep", "order softneck garlic before it sells out",
       f'{ca["garlic"]} cloves from ~{math.ceil(ca["garlic"] / 5)} bulbs, '
       f'so order {lb:g} lb', "buy")
   add("Sep 10", f"start onions in trays for {a}",
       f'{ca["onion"]} plants. Sow {trays(ca["onion"])} cells', "tray")
   add("Sep 15 on", f"direct sow carrots in {a}",
       f'{ca["carrot"]} carrots total, in 6 sowings of '
       f'~{max(1, round(ca["carrot"] / 6))} seeds', "direct")
   add("Sep 15 on", "direct sow radish between the carrot sowings",
       f'{band_count("radish", PLOTS[a]["length"])} positions per round, '
       f'4 in apart, marking the rows', "direct")
   add("Oct 25", f"plant garlic in {a}, double row 3 in either side",
       f'{ca["garlic"]} cloves, 6 in apart', "plant")
   add("Oct 25", f"direct sow snap peas in {a} at the trellis",
       f'{ca["pea"]} seeds, 3 in apart', "direct")
   add("Nov 10 to 20", f"transplant onions into {a}",
       f'{ca["onion"]} plants, 5 in apart in a double row', "plant")
   add("Mar 20", f"pull spent peas, direct sow bush beans in {a}",
       f'{band_count("bean", PLOTS[a]["length"])} seeds, 6 in apart',
       "direct")
   add("Jun 1 to 10", f"plant sweet potato slips in {a}",
       "8 slips at 18 in, into the vacated garlic and onion bands", "plant")

   # nightshades
   cn = counts(n_, "nightshade")
   add("Dec", "place the spring seed order",
       "before the January catalogue rush thins the selection", "buy")
   add("Jan 15", f"start peppers on the heat mat for {n_}",
       f'{cn["pepper"]} plants at 80 to 85 F. Sow '
       f'{trays(cn["pepper"]) + 2} cells, they germinate slowly', "tray")
   add("Jan 20", f"start tomatoes on the heat mat for {n_}",
       f'{cn["tomato"]} plants at 75 to 85 F. Sow {trays(cn["tomato"])} '
       f'cells', "tray")
   add("Mar 1", "start basil and marigold in trays",
       f'{cn["basil"]} basil, {cn["marigold"]} marigold. Sow '
       f'{trays(cn["basil"])} and {trays(cn["marigold"])} cells', "tray")
   add("Mar 15", f"transplant tomatoes into {n_}, set the trellis",
       f'{cn["tomato"]} plants, staggered across the outer laterals',
       "plant")
   add("Apr 10", f"transplant peppers, basil, marigold into {n_}",
       f'{cn["pepper"]} peppers, {cn["basil"]} basil, '
       f'{cn["marigold"]} marigold', "plant")

   # cucurbits
   cc = counts(c, "cucurbit")
   add("May 1", f"direct sow cucumbers in {c} at the trellis",
       f'{cc["cucumber"]} positions, 2 seeds each, thin to 1', "direct")
   add("May 1", f"direct sow summer squash in {c}",
       f'{cc["squash"]} positions, 2 seeds each, thin to 1', "direct")
   add("mid Jun", "direct sow the second cucumber flush",
       f'{cc["cucumber"]} seeds into the trellis gaps', "direct")
   add("Jul 1 to 10", f"direct sow mini pumpkins in {c}",
       f'{cc["mini"] * 2} seeds at {cc["mini"]} trellis positions, thin '
       f'to 1 each', "direct")

   # cover crop on the two resting plots
   rest = [k for k in (n_, c)]
   ra = sum(area_sqft(PLOTS[k]) for k in rest)
   add("Nov 5 to 15", f'broadcast winter cover crop into {" and ".join(rest)}',
       f'{ra:.0f} sq ft, about {ra * 0.17:.0f} oz of bell bean, vetch and '
       f'oat mix', "direct")
   add("mid Feb", "chop the winter cover crop in at flowering",
       f'{ra:.0f} sq ft, at flowering and not before', "cultivate")

   # annex, the same every year
   add("Nov, week 1", "clear ALL annex vine debris, then sow cover crop",
       "160 sq ft, ~1.5 lb cowpeas. Squash bugs overwinter in the debris",
       "cultivate")
   add("Feb", "order melon and pumpkin seed, corn, and sweet potato slips",
       "slips sell out by February", "buy")
   add("Apr 10", "start annex melons in cells",
       "2 watermelon and 2 cantaloupe. Sow 8 cells, 2 per hill", "tray")
   add("mid Apr", "direct sow the annex corn block",
       "30 seeds for 25 plants, 5 x 5 on a 12 in grid, never a row",
       "direct")
   add("May 1", "transplant annex melon hills",
       "4 hills, one plant each, a barrow of compost worked into each",
       "plant")
   add("Jul 1 to 10", "direct sow annex carving pumpkins",
       "2 hills, 3 seeds each, thin to 1. 95 to 110 days lands mid Oct",
       "direct")
   add("each season", "move every annex hill within the plot",
       "rotation hygiene, since the annex sits outside the four year cycle",
       "cultivate")

   # the one date that punishes forgetting
   if PLOTS[a]["station"] == 1:
       add("May 10", f"SHUT THE BALL VALVE on {a}",
           "alliums go bone dry for a fortnight or the bulbs will not cure "
           "and will rot in storage", "valve")
       add("Jun 1", f"reopen {a} for the sweet potato slips",
           "after the garlic and onions are lifted and curing", "valve")
   else:
       add("May 10", f"stop watering {a} at the controller",
           f'{a} has its own solenoid, so no valve to turn', "valve")

   R.sort(key=lambda r: ORDER.index(r["when"]) if r["when"] in ORDER else 99)
   return R


SUCCESSIONS = [("carrots + radish", "Sep 15 to late Feb", "3 weeks"),
              ("lettuce, arugula, Asian greens, cilantro",
               "Oct 1 to mid Feb", "3 weeks"),
              ("spinach", "Oct 15 to mid Jan only", "2 to 3 weeks"),
              ("cucumbers", "May 1 and mid Jun", "two flushes")]
PULSES = [("carrots", "two 5 min pulses daily", "10 to 14 days"),
         ("brassica transplants", "one 5 min pulse daily", "10 days"),
         ("sweet potato slips", "one 5 min pulse daily", "the first week")]


def report():
   L = [f"ALTADENA 91104 . zone 10b . "
        f"{sum(area_sqft(p) for p in PLOTS.values()) + 160:.0f} sq ft", ""]
   L.append(f'{"plot":6}{"size in":>13}{"sq ft":>7}{"lat":>5}{"sp":>7}'
            f'{"emit":>6}{"gph":>7}{"in/hr":>7}{"0.25in":>8}{"1in":>6}'
            f'{"stn":>5}')
   for k in PLOT_ORDER + ["annex"]:
       p = PLOTS.get(k) or ANNEX
       g = GRIDS[k]
       L.append(f'{k:6}{p["width"]:.0f} x {p["length"]:<7.0f}'
                f'{p["width"] * p["length"] / 144:7.1f}'
                f'{len(g["laterals"]):5}{g["lateral_spacing"]:7.2f}'
                f'{g["count"]:6}{g["gph"]:7.1f}{g["rate"]:7.2f}'
                f'{g["min_quarter"]:6} m{g["min_one"]:5}m'
                f'{p["station"]:5}')
   L += ["", "ROTATION"]
   for y in sorted(ROTATION):
       L.append(f'  year {y}  ' + "  ".join(
           f'{k}: {ROTATION[y][k]}' for k in PLOT_ORDER))
   return "\n".join(L)


def main():
   ap = argparse.ArgumentParser()
   ap.add_argument("--tasks", type=int, choices=[1, 2, 3, 4])
   args = ap.parse_args()
   if args.tasks:
       for r in tasks(args.tasks):
           print(f'{r["when"]:16} {r["action"]:52} {r["units"]}')
   else:
       print(report())


if __name__ == "__main__":
   main()