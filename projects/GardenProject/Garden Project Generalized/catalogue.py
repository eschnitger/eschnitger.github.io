#Setting up a drip grid for a generalized garden bed
def solve_drip(width_in, length_in, hw):
   """Lateral and emitter positions from bed geometry. Returns the grid
   plus the application rate, which is the number every schedule and
   every Rachio nozzle setting derives from."""
   margin, es = hw["edge_margin_in"], hw["emitter_spacing_in"]
   target = hw["target_lateral_spacing_in"]

   usable = width_in - 2 * margin
   n_lat = max(1, round(usable / target) + 1)
   spacing = usable / (n_lat - 1) if n_lat > 1 else 0
   laterals = [margin + i * spacing for i in range(n_lat)]

   n_em = int((length_in - 2 * (es / 2)) // es) + 1
   emitters = [es / 2 + i * es for i in range(n_em)]

   # 1 US gallon = 231 cu in, so depth per gallon over the cell area
   # gives inches directly. This holds for ANY spacing combination.
   cell = spacing * es if n_lat > 1 else width_in * es
   rate = hw["emitter_gph"] * 231.0 / cell

   total = n_lat * n_em
   return dict(laterals=laterals, emitters=emitters, count=total,
               lateral_spacing=spacing, gph=total * hw["emitter_gph"],
               rate_in_hr=rate,
               min_quarter=round(15 * (0.96 / rate)),
               min_one=round(60 * (0.96 / rate)))

#Crop rotation:
from itertools import permutations

CYCLE = ["nightshade", "allium", "cucurbit", "brassica"]   # heavy, rest,
#                                                            heavy, resting
FREQ_CLASS = {"nightshade": "deep", "allium": "deep",
             "cucurbit": "medium", "brassica": "shallow"}
COMPATIBLE = {("deep", "deep"), ("deep", "medium"),
             ("medium", "medium"), ("shallow", "shallow")}

def solve_rotation(plots, groups):
   """plots: ordered plot keys. groups: lists of plots sharing a valve.
   Returns {year: {plot: family}} or None if nothing satisfies."""
   n = len(CYCLE)
   if len(plots) != n:
       return None                      # handled separately, see below
   for offs in permutations(range(n)):
       rot = {y: {p: CYCLE[(offs[i] + y - 1) % n]
                  for i, p in enumerate(plots)} for y in range(1, n + 1)}
       ok = True
       for grp in groups:
           for y in range(1, n + 1):
               cls = [FREQ_CLASS[rot[y][p]] for p in grp]
               for a in cls:
                   for b in cls:
                       if (a, b) not in COMPATIBLE and \
                          (b, a) not in COMPATIBLE:
                           ok = False
       if ok:
           return rot
   return None

#Plant Layout:
ROLE_RULES = [
   # (predicate, lateral choice, note)
   (lambda p: p["habit"] == "trellis",  "north_edge",  "vertical"),
   (lambda p: p["canopy_r"] >= 10,      "outer",       "reach from both sides"),
   (lambda p: p["canopy_r"] <= 6,       "inner",       "row crop"),
   (lambda p: True,                     "inner",       "default"),
]

def place(grid, crops):
   """Assign crops to laterals and emitters by role, staggering the two
   outer rows so diagonal neighbours gain clearance."""
   lat = grid["laterals"]
   slots = {"north_edge": [1], "outer": [1, len(lat)],
            "inner": list(range(2, len(lat)))}
   out, used = [], {i: set() for i in range(1, len(lat) + 1)}
   for c in sorted(crops, key=lambda c: -c["canopy_r"]):
       role = next(r for pred, r, _ in ROLE_RULES if pred(c))
       step = max(1, round(2 * c["canopy_r"] / grid["emitters"][0] / 2)) or 1
       for li in slots[role]:
           offset = 0 if li == 1 else 1        # stagger the far outer row
           for ei in range(1 + offset, len(grid["emitters"]) + 1, step + 1):
               if ei in used[li]:
                   continue
               used[li].add(ei)
               out.append(dict(crop=c, lat=li, em=ei))
               break
           else:
               continue
           break
   return out

#Feasability reporting

CLIMATE_RULES = [
   dict(when=lambda s: s["chill_hours"] < 600,
        crop="garlic", verdict="softneck only",
        why="not enough winter chill to vernalise a hardneck"),
   dict(when=lambda s: s["latitude"] < 36,
        crop="onion", verdict="short day only",
        why="long day types never bulb at this latitude"),
   dict(when=lambda s: not s["summer_rain"] and s["annual_et_in"] > 40,
        crop="all", verdict="irrigation is the sole summer water source",
        why="rain and demand are exactly out of phase"),
   dict(when=lambda s: s["zone"] in ("10a", "10b", "11a", "11b"),
        crop="pepper", verdict="containerise the collection",
        why="overwinters as a perennial and would anchor a bed"),
   dict(when=lambda s: s["zone"] >= "9a",
        crop="spinach", verdict="mid Oct to mid Jan sowings only",
        why="bolts once day length stretches and February warms"),
]