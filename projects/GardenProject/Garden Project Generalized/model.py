"""
model.py . garden solver

Reads a garden.json describing beds, hardware, climate and plant
preferences, then derives the drip grid, the rotation, the plant
placement and the irrigation schedule. Nothing here draws anything.

   python model.py garden.json --report

The design principle: geometry and hydraulics are computed, horticulture
is data. Anything that depends on latitude, chill hours or rainfall
pattern lives in CLIMATE_RULES or a calendar file, never in the solvers,
so that a garden in Vermont gets different advice rather than the same
advice with different numbers.
"""

import argparse
import json
from itertools import permutations

GAL_CU_IN = 231.0          # 1 US gallon in cubic inches

# ======================================================================
# 1. PLANT CATALOGUE
#     Attributes, not positions. habit drives placement, canopy_r sizes
#     the drawn circle and the spacing step, water_class drives which
#     rotation group the plant belongs to.
# ======================================================================

# habit:  trellis | bush | row | sprawl | band
# family: nightshade | cucurbit | brassica | allium | grass | cover
CATALOGUE = [
   # name, family, habit, canopy_r in, water_class, tags
   ("tomato",            "nightshade", "bush",    18, "deep",    ["fruit"]),
   ("tomato, cherry",    "nightshade", "bush",    18, "deep",    ["fruit", "heat"]),
   ("tomato, determinate","nightshade","bush",    15, "deep",    ["fruit", "heat"]),
   ("pepper",            "nightshade", "bush",    11, "deep",    ["fruit", "perennial_warm"]),
   ("eggplant",          "nightshade", "bush",    12, "deep",    ["fruit"]),
   ("basil",             "nightshade", "row",      7, "deep",    ["herb"]),
   ("marigold",          "nightshade", "row",      5, "deep",    ["nematode"]),

   ("pickling cucumber", "cucurbit",   "trellis",  9, "medium",  ["fruit"]),
   ("slicing cucumber",  "cucurbit",   "trellis",  9, "medium",  ["fruit"]),
   ("summer squash",     "cucurbit",   "bush",    20, "medium",  ["fruit"]),
   ("mini pumpkin",      "cucurbit",   "trellis", 10, "medium",  ["fruit", "autumn"]),
   ("carving pumpkin",   "cucurbit",   "sprawl",  42, "medium",  ["fruit", "autumn"]),
   ("watermelon",        "cucurbit",   "sprawl",  36, "medium",  ["fruit", "drydown"]),
   ("cantaloupe",        "cucurbit",   "sprawl",  24, "medium",  ["fruit", "drydown"]),

   ("kale",              "brassica",   "bush",    12, "shallow", ["leaf", "long"]),
   ("broccoli",          "brassica",   "bush",    11, "shallow", ["leaf"]),
   ("cauliflower",       "brassica",   "bush",    11, "shallow", ["leaf", "single"]),
   ("cabbage",           "brassica",   "bush",    10, "shallow", ["leaf", "single"]),
   ("spinach",           "brassica",   "row",      5, "shallow", ["leaf", "succession"]),
   ("lettuce",           "brassica",   "row",      5, "shallow", ["leaf", "succession"]),
   ("asian greens",      "brassica",   "row",      5, "shallow", ["leaf", "succession"]),
   ("arugula",           "brassica",   "row",      5, "shallow", ["leaf", "succession"]),
   ("cilantro",          "brassica",   "row",      5, "shallow", ["herb", "succession"]),
   ("chard",             "brassica",   "row",      8, "shallow", ["leaf", "long"]),
   ("malabar spinach",   "brassica",   "trellis",  8, "medium",  ["leaf", "heat"]),

   ("garlic",            "allium",     "band",     0, "deep",    ["bulb", "drydown"]),
   ("onion",             "allium",     "band",     0, "deep",    ["bulb", "drydown"]),
   ("snap pea",          "allium",     "trellis",  0, "deep",    ["legume"]),
   ("bush bean",         "allium",     "band",     0, "deep",    ["legume"]),
   ("carrot",            "allium",     "band",     0, "deep",    ["root", "slow_germ"]),
   ("beet",              "allium",     "band",     0, "deep",    ["root"]),
   ("radish",            "allium",     "band",     0, "deep",    ["root", "marker"]),
   ("sweet potato",      "allium",     "sprawl",  16, "deep",    ["root", "heat"]),

   ("sweet corn",        "grass",      "block",   12, "medium",  ["grain", "block"]),
   ("bell bean + oat",   "cover",      "band",     0, "shallow", ["winter_cover"]),
   ("buckwheat",         "cover",      "band",     0, "shallow", ["summer_cover"]),
   ("cowpea",            "cover",      "band",     0, "medium",  ["summer_cover", "nematode"]),
]

PLANTS = {c[0]: dict(name=c[0], family=c[1], habit=c[2], canopy_r=c[3],
                    water_class=c[4], tags=c[5]) for c in CATALOGUE}

# Rotation cycle order: heavy feeder, restorative, heavy feeder, resting.
CYCLE = ["nightshade", "allium", "cucurbit", "brassica"]

# Frequency class per family, which is what shared-valve compatibility
# turns on. Rooting depth dictates frequency and cannot be faked; volume
# is trivially adjusted with run time.
FREQ_CLASS = {"nightshade": "deep", "allium": "deep",
             "cucurbit": "medium", "brassica": "shallow"}
COMPATIBLE = {("deep", "deep"), ("deep", "medium"), ("medium", "deep"),
             ("medium", "medium"), ("shallow", "shallow")}

MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
         "Apr", "May", "Jun", "Jul", "Aug", "Sep"]

# ======================================================================
# 2. CLIMATE
#     Target inches per week per family per month, and the rules that
#     encode judgment rather than arithmetic. A new site needs a new
#     calendar entry: a Mediterranean winter garden and a continental
#     summer garden are different calendars, not the same one shifted.
# ======================================================================

CALENDARS = {
   "mediterranean_10b": dict(
       # inches per week, no rain, well mulched. Oct first.
       need={
           "nightshade": [.55, .18, .18, .18, .20, .32, .45, .70, .95,
                          1.20, 1.20, .85],
           "cucurbit":   [.42, .18, .18, .18, .20, .20, .28, .62, 1.00,
                          1.55, 1.35, .92],
           "brassica":   [.52, .32, .22, .22, .26, .48, .70, .82, .26,
                          .26, .26, .52],
           "allium":     [.42, .22, .22, .22, .26, .50, .68, .78, .78,
                          1.05, 1.05, .70],
           "open_ground":[.35, 0, 0, 0, 0, 0, 0, .95, 1.35, 1.65, 1.65,
                          1.25],
       },
       # events per week by frequency class and season index
       freq={"deep":    [2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2],
             "medium":  [2, 1, 1, 1, 1, 1, 3, 3, 3.5, 3.5, 3.5, 3],
             "shallow": [3, 2, 1, 1, 1, 2, 3, 4, 1, 1, 1, 3]},
       frost="mid Feb to early Mar",
       note="Rain and demand are exactly out of phase. The garden is "
            "watered by the sky Nov to Mar and by you alone Jun to Sep.",
   ),
}

CLIMATE_RULES = [
   dict(when=lambda s: s.get("chill_hours", 1000) < 600,
        crop="garlic", verdict="softneck varieties only",
        why="not enough winter chill to vernalise a hardneck properly"),
   dict(when=lambda s: s.get("latitude", 45) < 36,
        crop="onion", verdict="short day varieties only",
        why="long day types grow handsome tops here and never bulb"),
   dict(when=lambda s: s.get("zone", "") in ("10a", "10b", "11a", "11b"),
        crop="pepper", verdict="containerise the collection",
        why="overwinters as a perennial and would anchor a bed, "
            "quietly destroying the rotation"),
   dict(when=lambda s: s.get("zone", "0") >= "9a",
        crop="spinach", verdict="sow mid Oct to mid Jan only",
        why="bolts once day length stretches and February warms"),
   dict(when=lambda s: s.get("zone", "0") >= "9a",
        crop="carrot", verdict="sow Sep to Feb only",
        why="summer roots turn woody and bitter"),
   dict(when=lambda s: not s.get("summer_rain", True)
        and s.get("annual_et_in", 0) > 40,
        crop="all", verdict="irrigation is the sole summer water source",
        why="rainfall and evaporative demand are out of phase"),
   dict(when=lambda s: s.get("zone", "0") >= "10a",
        crop="tomato", verdict="a third heat-setting or cherry types",
        why="blossom drop above about 92 F causes a midsummer fruit gap"),
]

# ======================================================================
# 3. SOLVERS
# ======================================================================


def solve_drip(width_in, length_in, hw):
   """Lateral and emitter positions from bed geometry. The application
   rate derived here is the number every schedule and every controller
   nozzle setting depends on, so it is computed per bed rather than
   assumed to be uniform: a wider bed spreads its laterals further and
   genuinely applies water more slowly."""
   margin = hw.get("edge_margin_in", 6)
   es = hw.get("emitter_spacing_in", 12)
   target = hw.get("target_lateral_spacing_in", es)

   usable = width_in - 2 * margin
   n_lat = max(1, int(round(usable / target)) + 1)
   spacing = usable / (n_lat - 1) if n_lat > 1 else 0.0
   laterals = [margin + i * spacing for i in range(n_lat)]

   n_em = int((length_in - es) // es) + 1
   emitters = [es / 2 + i * es for i in range(n_em)]

   cell = (spacing if n_lat > 1 else width_in) * es
   rate = hw["emitter_gph"] * GAL_CU_IN / cell if cell else 0.0
   total = n_lat * n_em
   return dict(laterals=laterals, emitters=emitters, count=total,
               lateral_spacing=spacing, gph=round(total * hw["emitter_gph"], 1),
               rate_in_hr=round(rate, 2),
               min_quarter=max(1, round(0.25 / rate * 60)) if rate else 0,
               min_one=max(1, round(1.0 / rate * 60)) if rate else 0)


def solve_rotation(plots, groups):
   """A valid rotation is a cyclic assignment: order families once by
   feeding habit, then give each plot an offset into that cycle. Every
   family then visits every plot exactly once and repeat spacing is
   automatically maximal. The search only has to pick offsets that keep
   shared-valve groups compatible in watering frequency."""
   n = len(CYCLE)
   if len(plots) != n:
       return None
   for offs in permutations(range(n)):
       rot = {y: {p: CYCLE[(offs[i] + y - 1) % n]
                  for i, p in enumerate(plots)}
              for y in range(1, n + 1)}
       if all(_group_ok(rot, y, grp) for grp in groups
              for y in range(1, n + 1)):
           return rot
   return None


def _group_ok(rot, year, grp):
   cls = [FREQ_CLASS[rot[year][p]] for p in grp]
   return all((a, b) in COMPATIBLE for a in cls for b in cls)


def place(grid, crops):
   """Assign crops to laterals and emitters from habit and canopy alone.
   Trellised things take the north edge, wide canopies the outer
   laterals so you can reach them from both sides, row crops the inner
   laterals. The two outer rows are staggered so diagonal neighbours
   gain clearance, which is what permits vertical training instead of
   caging."""
   n_lat, n_em = len(grid["laterals"]), len(grid["emitters"])
   es = grid["emitters"][1] - grid["emitters"][0] if n_em > 1 else 12
   slots = {"trellis": [1], "bush": [1, n_lat] if n_lat > 1 else [1],
            "row": list(range(2, n_lat)) or [1],
            "band": list(range(1, n_lat + 1)), "block": [1], "sprawl": []}
   used = {i: set() for i in range(1, n_lat + 1)}
   out, unplaced = [], []

   for c in sorted(crops, key=lambda c: -PLANTS[c["crop"]]["canopy_r"]):
       p = PLANTS[c["crop"]]
       if p["habit"] == "sprawl":
           unplaced.append(dict(crop=c["crop"], n=c.get("n", 1),
                                why="sprawling habit, belongs in open ground"))
           continue
       step = max(1, int(round(2 * p["canopy_r"] / es))) if p["canopy_r"] else 1
       want = c.get("n", 1)
       got = 0
       for li in slots.get(p["habit"], [1]):
           offset = 0 if li == 1 else 1     # stagger the far outer row
           ei = 1 + offset
           while ei <= n_em and got < want:
               if ei not in used[li]:
                   used[li].add(ei)
                   out.append(dict(crop=p["name"], lat=li, em=ei,
                                   r=p["canopy_r"], family=p["family"]))
                   got += 1
               ei += step + (1 if p["habit"] == "bush" else 0)
           if got >= want:
               break
       if got < want:
           unplaced.append(dict(crop=p["name"], n=want - got,
                                why=f"only {got} of {want} slots available"))
   return out, unplaced


def derive_schedule(grid, family, cal, is_open_ground=False):
   """Minutes per event from target inches per week, the frequency class
   and the measured application rate. This is why a wider bed gets
   longer runs automatically rather than by hand."""
   need = cal["need"]["open_ground" if is_open_ground else family]
   freq = cal["freq"][FREQ_CLASS.get(family, "medium")]
   out = []
   for i in range(12):
       inches, ev = need[i], freq[i]
       if inches <= 0 or ev <= 0:
           out.append((0, 0))
           continue
       per_event = inches / ev
       out.append((ev, max(1, int(round(per_event / grid["rate_in_hr"] * 60)))))
   return out


def envelope(schedules):
   """Shared valve: take the higher frequency and the longer event, with
   zeroes filtered first so an empty plot cannot silence a growing one."""
   out = []
   for i in range(12):
       cand = [s[i] for s in schedules if s[i][0] > 0] or [(0, 0)]
       out.append((max(c[0] for c in cand), max(c[1] for c in cand)))
   return out


# ======================================================================
# 4. ASSEMBLY
# ======================================================================


class Garden:
   def __init__(self, cfg):
       self.cfg = cfg
       self.site = cfg["site"]
       self.hw = cfg["hardware"]
       self.cal = CALENDARS[self.site.get("calendar", "mediterranean_10b")]
       self.warnings = []
       self.plots, self.groups, self.stations = [], [], {}
       self._build_plots()
       self.rotation = solve_rotation([p["key"] for p in self.plots],
                                      self.groups)
       if self.rotation is None:
           self.warnings.append(dict(
               kind="rotation", detail=f"{len(self.plots)} plots against "
               f"{len(CYCLE)} families. A clean cycle needs equal numbers, "
               f"so one family sits out each year or a plot takes a cover "
               f"crop year."))
       self._build_open_ground()
       self._check_preferences()
       self._check_climate()

   def _build_plots(self):
       station = 1
       for bed in self.cfg["beds"]:
           splits = bed.get("split_at_in") or []
           bounds = [0] + list(splits) + [bed["length_in"]]
           keys = []
           for i in range(len(bounds) - 1):
               length = bounds[i + 1] - bounds[i]
               key = f'{bed["name"]}-{i + 1}' if len(bounds) > 2 \
                   else bed["name"]
               grid = solve_drip(bed["width_in"], length, self.hw)
               self.plots.append(dict(key=key, bed=bed["name"],
                                      width=bed["width_in"], length=length,
                                      offset=bounds[i], grid=grid,
                                      sun=bed.get("sun_hours", 8)))
               keys.append(key)
           shared = bed.get("shared_solenoid", False) and len(keys) > 1
           if shared:
               self.groups.append(keys)
               for k in keys:
                   self.stations[k] = station
               self.warnings.append(dict(
                   kind="valve", detail=f'{bed["name"]} runs {len(keys)} '
                   f'plots on one solenoid, so its schedule is the envelope '
                   f'of both and any dry-down needs a manual ball valve.'))
               station += 1
           else:
               for k in keys:
                   self.stations[k] = station
                   station += 1
       self.next_station = station

   def _build_open_ground(self):
       self.open_ground = []
       for og in self.cfg.get("open_ground", []):
           grid = solve_drip(og["width_in"], og["length_in"], self.hw)
           st = self.next_station
           self.next_station += 1
           self.open_ground.append(dict(name=og["name"], grid=grid,
                                        station=st, cfg=og))
           self.stations[og["name"]] = st

   def _build_containers(self):
       self.containers = None
       c = self.cfg.get("containers")
       if not c:
           return
       st = c.get("station") or self.next_station
       if st >= self.next_station:
           self.next_station = st + 1
       n = c["count"]
       gph = n * c.get("dripper_gph", 1.0)
       self.containers = dict(name="containers", count=n,
                              dripper_gph=c.get("dripper_gph", 1.0),
                              size_gal=c.get("size_gal"), station=st,
                              gph=round(gph, 1))
       self.stations["containers"] = st
       self.warnings.append(dict(
           kind="containers", detail=f'{n} pots on station {st}. Point '
           f'emitters, not inline tubing, since pots sit at irregular '
           f'spacing and never rotate. Volume is per pot rather than per '
           f'square foot, so this zone is scheduled by pot size, not by '
           f'application rate.'))

   def area_sqft(self, plot):
       return plot["width"] * plot["length"] / 144.0

   def _check_preferences(self):
       prefs = self.cfg.get("preferences", {})
       og_area = sum(o["cfg"]["width_in"] * o["cfg"]["length_in"] / 144.0
                     for o in self.open_ground)
       for req in prefs.get("must_have", []):
           p = PLANTS.get(req["crop"])
           if p is None:
               self.warnings.append(dict(kind="unknown crop",
                                         detail=req["crop"]))
               continue
           if p["habit"] == "sprawl":
               need = req.get("min", 1) * (3.14 * p["canopy_r"] ** 2) / 144.0
               where = "open ground" if og_area else "nowhere suitable"
               self.warnings.append(dict(
                   kind="placement", detail=f'{req["crop"]} needs about '
                   f'{need:.0f} sq ft of run and is assigned to {where}. '
                   f'Vines may be trained out of the plot, so this is a '
                   f'guide rather than a hard requirement.'))
               if not og_area:
                   self.warnings.append(dict(
                       kind="infeasible", detail=f'{req["crop"]} requested '
                       f'but no open ground defined. Either add an open '
                       f'ground area or drop it to a trellised miniature.'))
       for x in prefs.get("exclude", []):
           if x not in PLANTS:
               self.warnings.append(dict(kind="note",
                                         detail=f"excluded '{x}' is not in "
                                         f"the catalogue anyway"))

   def _check_climate(self):
       self.climate_notes = [r for r in CLIMATE_RULES if r["when"](self.site)]

   def schedule_for(self, station, year):
       keys = [k for k, v in self.stations.items() if v == station]
       og = [o for o in self.open_ground if o["name"] in keys]
       if og:
           return derive_schedule(og[0]["grid"], "cucurbit", self.cal,
                                  is_open_ground=True)
       if self.containers and self.containers["station"] == station:
           out = []
           for i in range(12):
               gal, ev = CONTAINER_GAL[i], CONTAINER_FREQ[i]
               mins = max(1, round(gal / self.containers["dripper_gph"] * 60))
               out.append((ev, mins))
           return out

       scheds = []
       for k in keys:
           plot = next(p for p in self.plots if p["key"] == k)
           fam = self.rotation[year][k] if self.rotation else "brassica"
           scheds.append(derive_schedule(plot["grid"], fam, self.cal))
       return envelope(scheds) if len(scheds) > 1 else scheds[0]

   # ---- reporting --------------------------------------------------
   def report(self):
       L = []
       s = self.site
       L.append(f'GARDEN MODEL . {s["name"]} {s.get("zip", "")} . '
                f'zone {s.get("zone", "?")}')
       L.append("")
       L.append("PLOTS AND DRIP GRID")
       L.append(f'{"plot":16} {"size in":>13} {"sq ft":>6} {"lat":>4} '
                f'{"emit":>5} {"gph":>6} {"in/hr":>6} {"0.25in":>7} '
                f'{"stn":>4}')
       for p in self.plots:
           g = p["grid"]
           L.append(f'{p["key"]:16} {p["width"]:.0f} x {p["length"]:<6.0f} '
                    f'{self.area_sqft(p):6.1f} {len(g["laterals"]):4} '
                    f'{g["count"]:5} {g["gph"]:6.1f} {g["rate_in_hr"]:6.2f} '
                    f'{g["min_quarter"]:5} m {self.stations[p["key"]]:4}')
       for o in self.open_ground:
           g = o["grid"]
           a = o["cfg"]["width_in"] * o["cfg"]["length_in"] / 144.0
           L.append(f'{o["name"]:16} {o["cfg"]["width_in"]:.0f} x '
                    f'{o["cfg"]["length_in"]:<6.0f} {a:6.1f} '
                    f'{len(g["laterals"]):4} {g["count"]:5} {g["gph"]:6.1f} '
                    f'{g["rate_in_hr"]:6.2f} {g["min_quarter"]:5} m '
                    f'{o["station"]:4}')
       rates = {round(p["grid"]["rate_in_hr"], 2) for p in self.plots}
       if len(rates) > 1:
           L.append("")
           L.append(f'  note: rates differ across plots ({sorted(rates)}), '
                    f'because lateral spacing follows bed width. Run times '
                    f'are derived per plot rather than shared.')
       L.append("")
       L.append("ROTATION")
       if self.rotation:
           hdr = " ".join(f'{p["key"]:>14}' for p in self.plots)
           L.append(f'{"year":6}{hdr}')
           for y in sorted(self.rotation):
               row = " ".join(f'{self.rotation[y][p["key"]]:>14}'
                              for p in self.plots)
               L.append(f'{y:<6}{row}')
       else:
           L.append("  no clean cycle, see warnings")
       L.append("")
       L.append("CLIMATE RULES IN FORCE")
       for r in self.climate_notes:
           L.append(f'  {r["crop"]:10} {r["verdict"]}')
           L.append(f'  {"":10} because {r["why"]}')
       L.append("")
       L.append("WARNINGS AND FEASIBILITY")
       if not self.warnings:
           L.append("  none")
       for w in self.warnings:
           L.append(f'  [{w["kind"]}] {w["detail"]}')
       return "\n".join(L)


def load(path):
   with open(path) as f:
       return Garden(json.load(f))


def main():
   ap = argparse.ArgumentParser()
   ap.add_argument("config")
   ap.add_argument("--report", action="store_true")
   ap.add_argument("--year", type=int, default=1)
   args = ap.parse_args()
   g = load(args.config)
   if args.report:
       print(g.report())
   else:
       for st in sorted(set(g.stations.values())):
           sched = g.schedule_for(st, args.year)
           row = "  ".join(f'{MONTHS[i]} {e:g}x{m}' for i, (e, m)
                           in enumerate(sched) if e)
           print(f"station {st}: {row}")


if __name__ == "__main__":
   main()