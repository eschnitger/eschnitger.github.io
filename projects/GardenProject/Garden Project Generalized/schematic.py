"""
schematic.py . hydraulic schematic sheet

Draws what feeds what, derived from the model rather than hardcoded, so
the station count, flows and regulator advice all follow from garden.json.
Deliberately NOT to scale: this is the sheet you want in your hand when
something is not working. The site plan is a different drawing.

   python schematic.py garden.json
   python schematic.py garden.json --page tabloid

Importable too: sheets.py can call fig_schematic(g, outdir) to append it
to the same PDF.
"""

import argparse
import datetime
import os

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

import sheets as sh
from model import load

# ======================================================================
# 1. HARDWARE THAT IS JUDGMENT RATHER THAN ARITHMETIC
#     The head assembly order matters and cannot be derived from bed
#     dimensions. Putting the filter downstream of the solenoids is the
#     classic mistake and it quietly ruins valve diaphragms.
# ======================================================================

PRV_MIN_GPM = 0.5      # below this a FIXED drip regulator stops regulating
FINE_MESH_GPM = 5.0    # above this, check the filter's own flow rating


def head_assembly(g):
   hw = g.hw
   items = [
       ("SUPPLY", "3/4 in bib or hard plumb\nverify static pressure first",
        "#8a8a8a"),
       ("BACKFLOW", "type per local code\ncheck with your water purveyor",
        "#7a6a9a"),
       ("BALL VALVE", "3/4 in isolation\nservice everything downstream",
        "#8a2f2f"),
       ("FILTER", f'{hw.get("filter_mesh", "150 to 200")} mesh\n'
                  f'UPSTREAM of the solenoids', "#5a7a9a"),
       ("PRV", f'adjustable spring + diaphragm\nSET TO '
               f'{hw.get("target_psi", 25)} PSI', "#2f5d8a"),
       ("GAUGE", "0 to 100 psi, liquid filled\npermanent, for setting the "
                 "PRV", "#4a4a4a"),
   ]
   if hw.get("meter", True):
       items.append(("METER", "low flow totalizer, from 0.1 gpm\nmonthly "
                              "reading is your water log", "#6b8a5a"))
   return items


def stations_from(g):
   """Every station the model knows about, plus containers, which the
   model does not currently own. Returns dicts of station, label, detail,
   emitter count, gph."""
   out = []
   for st in sorted(set(g.stations.values())):
       keys = [k for k, v in g.stations.items() if v == st]
       plots = [p for p in g.plots if p["key"] in keys]
       ogs = [o for o in g.open_ground if o["name"] in keys]
       n = sum(p["grid"]["count"] for p in plots) + \
           sum(o["grid"]["count"] for o in ogs)
       gph = sum(p["grid"]["gph"] for p in plots) + \
           sum(o["grid"]["gph"] for o in ogs)
       shared = len(keys) > 1
       if ogs:
           detail = (f'{len(ogs[0]["grid"]["laterals"])} laterals, '
                     f'{len(ogs[0]["grid"]["emitters"])} emitters each\n'
                     f'solenoid AT the plot, so the buried run\n'
                     f'sits depressurised when idle')
       elif shared:
           detail = (f'ONE solenoid, {len(keys)} plots\n'
                     f'{len(keys)} manual ball valves at the bed,\n'
                     f'tagged by crop rather than by plot')
       else:
           p = plots[0]
           detail = (f'{len(p["grid"]["laterals"])} laterals, '
                     f'{len(p["grid"]["emitters"])} emitters each\n'
                     f'{p["grid"]["rate_in_hr"]} in/hr, '
                     f'{p["grid"]["min_quarter"]} min = 0.25 in')
       out.append(dict(st=st, label="\n".join(keys), detail=detail, n=n,
                       gph=round(gph, 1), keys=keys, shared=shared))

   # Containers are configured but not modelled, so they are read here.
   c = g.cfg.get("containers")
   if c:
       gph = c["count"] * c.get("dripper_gph", 1.0)
       out.append(dict(st=c.get("station", max(
           s["st"] for s in out) + 1 if out else 1),
           label="containers", n=None, gph=round(gph, 1), keys=["containers"],
           shared=False,
           detail=f'{c["count"]} pots, 1/4 in microtube\n'
                  f'point emitters at {c.get("dripper_gph", 1.0):g} gph\n'
                  f'the right place for perennials'))
   return out


# ======================================================================
# 2. THE SHEET
# ======================================================================


def fig_schematic(g, outdir):
   st_rows = stations_from(g)
   total = sum(s["gph"] for s in st_rows)
   peak = max(s["gph"] for s in st_rows) if st_rows else 0.0
   n_emit = sum(s["n"] for s in st_rows if s["n"])
   capacity = g.hw.get("controller_stations", len(st_rows))

   fig = sh.page()
   sh.header(fig, "HYDRAULIC SCHEMATIC . NOT TO SCALE",
             f'What feeds what . {len(st_rows)} of {capacity} stations used '
             f'. {g.site["name"]} {g.site.get("zip", "")} . '
             f'{g.hw["emitter_gph"]:g} gph emitters on a '
             f'{g.hw.get("emitter_spacing_in", 12):g} in grid')

   ax = fig.add_axes([sh.m(), 0.17, 1 - 2 * sh.m(), 0.66])
   ax.set_xlim(0, 100)
   ax.set_ylim(0, 64)
   ax.axis("off")
   # No equal aspect: data units stretch freely, so every round dot is a
   # marker sized in points rather than a Circle patch, which would
   # otherwise render as an ellipse.

   def box(x, y, w, h, title, sub, colour, sub_pt=5.5):
       ax.add_patch(Rectangle((x, y), w, h, facecolor=colour, alpha=0.16,
                              edgecolor=colour, lw=1.3, zorder=3))
       ax.text(x + w / 2, y + h * 0.70, title, fontsize=sh.fs(6.7),
               weight="bold", ha="center", va="center", color="#222",
               zorder=5, linespacing=1.25)
       ax.text(x + w / 2, y + h * 0.30, sub, fontsize=sh.fs(sub_pt),
               ha="center", va="center", color="#444", zorder=5,
               linespacing=1.35)

   def arrow(x0, y0, x1, y1, colour="#2f5d8a", lw=1.5):
       ax.annotate("", xytext=(x0, y0), xy=(x1, y1),
                   arrowprops=dict(arrowstyle="-|>", lw=lw, color=colour,
                                   shrinkA=0, shrinkB=0), zorder=2)

   # ---- head assembly, left to right along the top ------------------
   head = head_assembly(g)
   gap = 1.5
   bw = (100 - gap * (len(head) - 1)) / len(head)
   bh, ytop = 9.0, 50.0
   for i, (title, sub, colour) in enumerate(head):
       x = i * (bw + gap)
       box(x, ytop, bw, bh, title, sub, colour)
       if i:
           arrow(x - gap, ytop + bh / 2, x, ytop + bh / 2)
   ax.text(0, ytop + bh + 2.6, "HEAD ASSEMBLY . order matters . the filter "
           "protects the valve diaphragms as well as the emitters",
           fontsize=sh.fs(6.5), color="#555", va="bottom")

   # ---- manifold ----------------------------------------------------
   x_last = (len(head) - 1) * (bw + gap) + bw / 2
   y_bus_top = 43.0
   arrow(x_last, ytop, x_last, y_bus_top)
   ax.plot([72, x_last], [y_bus_top, y_bus_top], color="#2f5d8a", lw=1.5,
           zorder=2)
   box(28, 34.0, 44, 9.0,
       f'MANIFOLD . {capacity} STATION CONTROLLER',
       f'{len(st_rows)} used, {max(0, capacity - len(st_rows))} spare . '
       f'stations run SEQUENTIALLY, so the supply\nnever sees more than '
       f'{peak:.1f} gph . one {g.hw.get("start_time", "04:30")} start . '
       f'weather based scheduling, wind skip OFF', "#2f5d8a", sub_pt=5.7)

   # ---- station branches, wrapping to a second tier if crowded ------
   per_row = len(st_rows) if len(st_rows) <= 5 else (len(st_rows) + 1) // 2
   tiers = [st_rows[i:i + per_row] for i in range(0, len(st_rows), per_row)]
   sw = (100 - 2.0 * (per_row - 1)) / per_row
   sh_box = 12.5 if len(tiers) == 1 else 10.5
   y_bus = 30.5
   ax.plot([sw / 2, 100 - sw / 2], [y_bus, y_bus], color="#2f5d8a", lw=1.5,
           zorder=2)
   ax.plot([50, 50], [34.0, y_bus], color="#2f5d8a", lw=1.5, zorder=2)

   placed = {}
   for t, tier in enumerate(tiers):
       y_sta = y_bus - 6.0 - t * (sh_box + 4.5) - sh_box
       for i, s in enumerate(tier):
           x = i * (sw + 2.0)
           cx = x + sw / 2
           if t == 0:
               arrow(cx, y_bus, cx, y_sta + sh_box)
           else:
               ax.plot([cx, cx], [y_bus, y_sta + sh_box], color="#2f5d8a",
                       lw=1.2, ls=(0, (3, 2)), zorder=2)
           ax.plot(cx, y_bus, marker="o", ms=sh.fs(4.0), mfc="white",
                   mec="#8a2f2f", mew=1.2, zorder=4)
           ax.text(cx + 2.0, y_bus + 1.7, f'S{s["st"]}', fontsize=sh.fs(6.0),
                   weight="bold", color="#8a2f2f", zorder=5)
           emtxt = f'{s["n"]} emitters\n' if s["n"] else ""
           low = " LOW FLOW" if s["gph"] / 60 < PRV_MIN_GPM else ""
           box(x, y_sta, sw, sh_box, f'STATION {s["st"]}\n{s["label"]}',
               f'{emtxt}{s["gph"]:g} gph, {s["gph"] / 60:.2f} gpm{low}\n'
               f'{s["detail"]}', "#6b8a5a", sub_pt=5.2)
           placed[s["st"]] = (cx, y_sta, sw)

   # ---- shared valve splits, the only manual elements ---------------
   for grp in g.groups:
       st = g.stations[grp[0]]
       if st not in placed:
           continue
       cx, y_sta, w_box = placed[st]
       y_split = max(2.0, y_sta - 6.0)
       ax.plot([cx, cx], [y_sta, y_split + 4.2], color="#8a2f2f", lw=1.5,
               zorder=2)
       half = w_box / 2 - 2.0
       ax.plot([cx - half, cx + half], [y_split + 4.2] * 2, color="#8a2f2f",
               lw=1.5, zorder=2)
       for j, key in enumerate(grp):
           xx = cx - half + j * (2 * half / max(1, len(grp) - 1))
           arrow(xx, y_split + 4.2, xx, y_split + 1.4, colour="#8a2f2f")
           ax.add_patch(Rectangle((xx - 3.2, y_split - 2.6), 6.4, 4.0,
                                  facecolor="#8a2f2f", alpha=0.14,
                                  edgecolor="#8a2f2f", lw=1.1, zorder=3))
           short = key.split("-")[-1] if "-" in key else key[:6]
           ax.text(xx, y_split - 0.6, short, fontsize=sh.fs(6.0),
                   weight="bold", ha="center", va="center", zorder=5)

   # ---- summary line ------------------------------------------------
   fig.text(sh.m(), 0.845,
            f'{n_emit} inline emitters . {total:.1f} gph if everything ran '
            f'at once, {total / 60:.2f} gpm . sequential, so the true peak '
            f'is {peak:.1f} gph or {peak / 60:.2f} gpm . '
            f'1 in over all beds = '
            f'{sum(g.area_sqft(p) for p in g.plots) * 0.623:.0f} gal',
            fontsize=sh.fs(7.2), va="top", color="#444")

   # ---- diagnostics, computed rather than asserted -------------------
   low = [s for s in st_rows if s["gph"] / 60 < PRV_MIN_GPM]
   notes = []
   if low:
       worst = min(low, key=lambda s: s["gph"])
       notes.append(
           f'{len(low)} of {len(st_rows)} zones draw under {PRV_MIN_GPM} gpm, '
           f'the lowest being station {worst["st"]} at '
           f'{worst["gph"] / 60:.2f} gpm. A FIXED drip pressure regulator '
           f'needs roughly {PRV_MIN_GPM} gpm to regulate at all, so it would '
           f'simply pass whatever arrives. Use an ADJUSTABLE spring and '
           f'diaphragm PRV.')
       notes.append(
           "This is also the argument for pressure compensating tubing: PC "
           "emitters hold their rated flow from roughly 15 to 50 psi and "
           "forgive whatever imprecision remains after regulation.")
   else:
       notes.append(
           f'Every zone clears {PRV_MIN_GPM} gpm, so a fixed drip regulator '
           f'would work. An adjustable PRV is still the better buy, since it '
           f'covers any future low flow zone without asterisks.')
   if peak / 60 > FINE_MESH_GPM:
       notes.append(f'Peak flow is {peak / 60:.1f} gpm, so check the '
                    f'filter body is rated for it rather than assuming a '
                    f'small hose thread unit will do.')
   if g.groups:
       notes.append(
           "Ball valves are open or shut, never partly, since a throttled "
           "valve drops PC emitters below their regulating range. Closing "
           "one half does not change the rate in the other.")
   notes.append(
       f'Diagnostics: {g.hw.get("target_psi", 25)} psi at a far lateral '
       f'gauge means all is well, well above it means the PRV has failed, '
       f'zero during a run means valve, filter or shutoff rather than '
       f'emitters. Time the bleed down and log it.')
   notes.append(
       f'Flush all {sum(len(p["grid"]["laterals"]) for p in g.plots) + sum(len(o["grid"]["laterals"]) for o in g.open_ground)} '
       f'lateral ends once or twice a year wide open, then an hour of citric '
       f'acid or dilute vinegar if the bleed down has grown slower. Hard '
       f'water closes emitters gradually.')
   sh.footer(fig, notes[:5])
   sh.save(fig, outdir, "04_schematic")


# ======================================================================
# 3. MAIN
# ======================================================================


def main():
   ap = argparse.ArgumentParser()
   ap.add_argument("config")
   ap.add_argument("--outdir", default="garden_out")
   ap.add_argument("--pdf", default="schematic.pdf", help="or 'none'")
   ap.add_argument("--page", default="letter", choices=list(sh.PAGE_SIZES))
   ap.add_argument("--margin", type=float, default=0.30)
   ap.add_argument("--dpi", type=int, default=200)
   ap.add_argument("--no-images", action="store_true")
   args = ap.parse_args()

   # sheets holds the page state, so set it there rather than shadowing
   # it locally: importing fs() and m() would otherwise read stale values.
   sh.PAGE = sh.PAGE_SIZES[args.page]
   sh.MARGIN = args.margin
   sh.DPI = args.dpi
   sh.WRITE_IMAGES = not args.no_images
   os.makedirs(args.outdir, exist_ok=True)

   g = load(args.config)
   if args.pdf.lower() != "none":
       sh.PDF = PdfPages(os.path.join(args.outdir, args.pdf))
       sh.PDF.infodict()["Title"] = f'Irrigation schematic, {g.site["name"]}'
       sh.PDF.infodict()["CreationDate"] = datetime.datetime.now()
   try:
       print(f'page {sh.PAGE[0]} x {sh.PAGE[1]} in')
       fig_schematic(g, args.outdir)
   finally:
       if sh.PDF is not None:
           sh.PDF.close()
           sh.PDF = None
   print("wrote schematic to", args.outdir + "/")


if __name__ == "__main__":
   main()