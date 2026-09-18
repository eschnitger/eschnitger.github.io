"""
Hydraulic schematic . Altadena CA 91104 . USDA zone 10b

A one page diagram of the irrigation system: what feeds what, in order,
with sizes and settings. Deliberately NOT to scale. This is the sheet you
want in your hand when something is not working. The site plan, with true
positions and the buried run, is a different drawing.

   pip install matplotlib
   python schematics.py                    # letter landscape, PDF + png + svg
   python schematics.py --page tabloid     # roomier
   python schematics.py --no-images        # PDF only
"""

import argparse
import datetime
import os

import matplotlib
matplotlib.use("Agg")                       # headless, works over ssh
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

# ----------------------------------------------------------------------
# page and type
# ----------------------------------------------------------------------

PAGE_SIZES = {                              # landscape, inches
   "letter":  (11.0, 8.5),
   "tabloid": (17.0, 11.0),
   "a4":      (11.69, 8.27),
   "a3":      (16.54, 11.69),
}

PAGE = PAGE_SIZES["letter"]
MARGIN = 0.30
DPI = 200
RATE_IN_PER_HR = 0.75       # 0.5 gph emitters on a 12 in grid


def fs(pt):
   """Font size in points, scaled from an 11 in wide reference page."""
   return pt * (PAGE[0] / 11.0) ** 0.7


def m(inches=None, axis="x"):
   """Margin as a figure fraction. Reads MARGIN at call time, not at
   definition time, so --margin actually takes effect."""
   inches = MARGIN if inches is None else inches
   return inches / (PAGE[0] if axis == "x" else PAGE[1])


# ----------------------------------------------------------------------
# system description. Everything editable lives here.
# ----------------------------------------------------------------------

# In strict plumbing order. Getting the filter downstream of the
# solenoids is the classic mistake and it quietly ruins valve diaphragms.
HEAD_ASSEMBLY = [
   ("SUPPLY", "3/4 in hose bib or hard plumb\nfoothill pressure often 80+ psi",
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

# station, label, emitter count or None, gph, note
STATIONS = [
   (1, "BED L\nL1 + L2", 48, 24,
    "one solenoid, TWO manual ball valves\nat the bed, tagged by crop"),
   (2, "BED M", 40, 20, "4 laterals, 10 emitters each"),
   (3, "BED S", 28, 14, "4 laterals, 7 emitters each"),
   (4, "CONTAINERS", None, 12,
    "1/4 in microtube + point emitters\nthe overwintering peppers"),
   (5, "ANNEX\n8 x 20 ft", 160, 80,
    "15 ft buried run, solenoid AT the annex\n8 laterals, 20 emitters each"),
]

FOOTNOTES = [
   "The PRV must be an adjustable spring and diaphragm type, NOT a fixed "
   "25 psi drip regulator. Those carry a minimum flow near 0.5 gpm and no "
   "zone here exceeds 0.33 gpm, so a fixed unit would simply pass whatever "
   "arrives.",
   "This is also why the tubing is pressure compensating. PC emitters hold "
   "0.5 gph from roughly 15 to 50 psi and forgive whatever imprecision "
   "remains after regulation.",
   "Station 5 puts its solenoid AT the annex rather than at the manifold, "
   "so the 15 ft buried run sits depressurised whenever the annex is not "
   "watering. Wire in the same trench, laid loose beside the pipe.",
   "Diagnostics: 25 psi at a far lateral gauge means all is well, above 40 "
   "means the PRV has failed, zero during a run means valve, filter or "
   "shutoff rather than emitters. Time the bleed down and log it.",
   "Flush all 24 lateral ends once or twice a year, wide open, then an hour "
   "of citric acid or dilute vinegar if the bleed down has grown slower. "
   "Foothill water is hard and closes emitters gradually.",
]


# ----------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------

def page():
   fig = plt.figure(figsize=PAGE)
   fig.patch.set_facecolor("white")
   return fig


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


def draw(fig):
   ax = fig.add_axes([m(), 0.17, 1 - 2 * m(), 0.66])
   ax.set_xlim(0, 100)
   ax.set_ylim(0, 64)
   ax.axis("off")
   # No equal aspect: this is a schematic, so data units stretch freely.
   # Every round dot is therefore drawn with a marker rather than a
   # Circle patch, which would render as an ellipse.

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

   # head assembly, left to right along the top
   bw, bh, gap, ytop = 12.2, 9.0, 1.6, 50.0
   for i, (title, sub, colour) in enumerate(HEAD_ASSEMBLY):
       x = i * (bw + gap)
       box(x, ytop, bw, bh, title, sub, colour)
       if i:
           arrow(x - gap, ytop + bh / 2, x, ytop + bh / 2)
   ax.text(0, ytop + bh + 2.6, "HEAD ASSEMBLY . order matters . the filter "
           "protects the valve diaphragms as well as the emitters",
           fontsize=fs(6.6), color="#555", va="bottom")

   # drop from the last head component to the manifold
   x_last = 6 * (bw + gap) + bw / 2
   y_bus_top = 43.0
   arrow(x_last, ytop, x_last, y_bus_top)
   ax.plot([72, x_last], [y_bus_top, y_bus_top], color="#2f5d8a", lw=1.6,
           zorder=2)
   box(28, 34.0, 44, 9.0, "MANIFOLD . 16 STATION RACHIO",
       "5 stations used, 11 spare . stations run SEQUENTIALLY, so the supply "
       "never sees more than 80 gph\none 4:30 am start . Flex Daily . "
       "wind skip OFF . rain skip ON at 0.25 in", "#2f5d8a", sub_pt=5.9)

   # five station branches off a common bus
   sw, sh = 17.0, 13.5
   sgap = (100 - 5 * sw) / 4
   y_bus, y_sta = 30.5, 13.0
   ax.plot([sw / 2, 100 - sw / 2], [y_bus, y_bus], color="#2f5d8a", lw=1.6,
           zorder=2)
   ax.plot([50, 50], [34.0, y_bus], color="#2f5d8a", lw=1.6, zorder=2)
   for i, (st, label, em, gph, note) in enumerate(STATIONS):
       x = i * (sw + sgap)
       cx = x + sw / 2
       arrow(cx, y_bus, cx, y_sta + sh)
       ax.plot(cx, y_bus, marker="o", ms=fs(4.2), mfc="white",
               mec="#8a2f2f", mew=1.3, zorder=4)
       ax.text(cx + 2.2, y_bus + 1.8, f"S{st}", fontsize=fs(6.2),
               weight="bold", color="#8a2f2f", zorder=5)
       emtxt = f"{em} emitters at 0.5 gph\n" if em else ""
       box(x, y_sta, sw, sh, f"STATION {st}\n{label}",
           f"{emtxt}{gph} gph\n{note}", "#6b8a5a", sub_pt=5.5)

   # the Bed L ball valve split: the only manual element in the system
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
           "since a throttled valve drops PC emitters below their "
           "regulating range.\nClosing one half does not change the rate in "
           "the other. Shut the allium half on MAY 10, reopen in June.",
           fontsize=fs(5.9), va="center", color="#8a2f2f", linespacing=1.45)

   return ax


def make_sheet(outdir, pdf, write_images):
   total_gph = sum(s[3] for s in STATIONS)
   n_emit = sum(s[2] for s in STATIONS if s[2])
   fig = page()
   header(fig, "HYDRAULIC SCHEMATIC . NOT TO SCALE",
          f"What feeds what . 5 of 16 stations used . Altadena 91104, zone "
          f"10b . generated {datetime.date.today().isoformat()}")
   draw(fig)
   fig.text(m(), 0.845,
            f"{n_emit} inline emitters plus 12 container drippers  .  "
            f"{total_gph} gph if everything ran at once, {total_gph / 60:.1f} "
            f"gpm  .  sequential, so the true peak is 80 gph or 1.3 gpm at "
            f"the annex  .  {RATE_IN_PER_HR} in/hr everywhere  .  "
            f"20 min = 0.25 in  .  80 min = 1 in",
            fontsize=fs(7.4), va="top", color="#444")
   footer(fig, FOOTNOTES)

   if write_images:
       fig.savefig(os.path.join(outdir, "schematic.png"), dpi=DPI,
                   facecolor="white")
       fig.savefig(os.path.join(outdir, "schematic.svg"), facecolor="white")
   if pdf is not None:
       pdf.savefig(fig, facecolor="white")
   plt.close(fig)


def main():
   global PAGE, MARGIN, DPI
   ap = argparse.ArgumentParser()
   ap.add_argument("--outdir", default="garden_out")
   ap.add_argument("--pdf", default="schematic.pdf", help="or 'none'")
   ap.add_argument("--page", default="letter", choices=list(PAGE_SIZES))
   ap.add_argument("--margin", type=float, default=0.30)
   ap.add_argument("--dpi", type=int, default=200)
   ap.add_argument("--no-images", action="store_true")
   args = ap.parse_args()

   PAGE = PAGE_SIZES[args.page]
   MARGIN = args.margin
   DPI = args.dpi
   os.makedirs(args.outdir, exist_ok=True)

   pdf, pdf_path = None, None
   if args.pdf.lower() != "none":
       pdf_path = os.path.join(args.outdir, args.pdf)
       pdf = PdfPages(pdf_path)
       info = pdf.infodict()
       info["Title"] = "Irrigation schematic, Altadena 91104"
       info["Author"] = "Eric"
       info["Subject"] = "Five station drip system, three beds and a vine annex"
       info["CreationDate"] = datetime.datetime.now()

   try:
       print(f"page {PAGE[0]} x {PAGE[1]} in, margin {MARGIN} in")
       make_sheet(args.outdir, pdf, not args.no_images)
   finally:
       if pdf is not None:
           pdf.close()

   print("wrote schematic to", args.outdir + "/")
   if pdf_path:
       print("PDF:", pdf_path)


if __name__ == "__main__":
   main()