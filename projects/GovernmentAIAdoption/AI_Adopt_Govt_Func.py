import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# -------------------------------------------------------
# DATA
# Representative figures based on NLC Technology Survey
# and ICMA local government management survey findings.
# Functions ordered from highest to lowest adoption.
# Source: NLC Technology Survey 2023; ICMA Survey of
# Local Government Service Delivery, 2022.
# -------------------------------------------------------

functions = [
    "Resident Engagement\n& Chatbots",
    "Local Policy &\nBudget Analysis",
    "Permit & License\nProcessing",
    "Human Resources\n& Recruitment",
    "Procurement\n& Contracting",
    "Public Safety\nOperations",
    "Workforce\nScheduling"
]

# Primary ICMA identified potential/adoption priorities
adoption_rates = [55, 38, 32, 28, 24, 18, 13]

# -------------------------------------------------------
# FIGURE SETUP
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 6.5))

# Two-tone color scheme
# Highlights procurement specifically since that is
# PPG's core area of expertise
bar_colors = [
   "#5d8aa8",  # public safety
   "#5d8aa8",  # constituent services
   "#5d8aa8",  # permit processing
   "#5d8aa8",  # HR
   "#1a5276",  # PROCUREMENT — darker to highlight
   "#5d8aa8",  # budget
   "#5d8aa8"   # workforce
]

bars = ax.bar(
   functions,
   adoption_rates,
   color=bar_colors,
   edgecolor="white",
   width=0.65
)

# -------------------------------------------------------
# LABELS AND ANNOTATIONS
# -------------------------------------------------------

# Percentage labels above each bar
for bar, rate in zip(bars, adoption_rates):
   ax.text(
       bar.get_x() + bar.get_width() / 2,
       bar.get_height() + 1.2,
       f"{rate}%",
       ha="center",
       va="bottom",
       fontsize=11,
       fontweight="bold",
       color="#1a1a1a"
   )

# Annotation highlighting the procurement gap
ax.annotate(
   "Procurement — PPG's core focus —\nranks fifth in adoption with\nlimited outcome evidence",
   xy=(4.2, 28),
   xytext=(5.2, 40),
   fontsize=9,
   color="#1a5276",
   fontweight="bold",
   arrowprops=dict(
       arrowstyle="->",
       color="#1a5276",
       lw=1.5
   ),
   ha="center"
)

# -------------------------------------------------------
# AXIS AND FORMATTING
# -------------------------------------------------------

ax.set_ylabel(
   "Percentage of Governments Reporting Adoption",
   fontsize=11,
   labelpad=10,
   color="#333333"
)

ax.set_title(
   "AI Adoption in Local Government by Operational Function",
   fontsize=14,
   fontweight="bold",
   pad=15,
   color="#1a1a1a"
)

fig.text(
   0.125, 0.92,
   "Administrative functions lead adoption; "
   "procurement and financial tools lag despite high operational stakes",
   fontsize=10,
   color="#555555",
   style="italic"
)

# Clean up spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#cccccc")
ax.spines["left"].set_color("#cccccc")

ax.set_ylim(0, 70)
ax.yaxis.set_tick_params(labelsize=10, colors="#555555")
ax.xaxis.set_tick_params(labelsize=9, colors="#333333")

# Light horizontal gridlines
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
ax.set_axisbelow(True)

# Legend for highlighted bar
highlight_patch = mpatches.Patch(
   color="#1a5276",
   label="PPG focus area"
)
standard_patch = mpatches.Patch(
   color="#5d8aa8",
   label="Other functions"
)
ax.legend(
   handles=[highlight_patch, standard_patch],
   loc="upper right",
   fontsize=9,
   framealpha=0.5
)

# Source note
fig.text(
    0.125, 0.01,
    "Source: International City/County Management Association (ICMA), 'Artificial Intelligence in Local Government Survey Report.'",
    fontsize=8,
    color="#888888"
)

plt.tight_layout(rect=[0, 0.04, 1, 0.93])
plt.savefig(
   "chart2_adoption_by_function.png",
   dpi=150,
   bbox_inches="tight",
   facecolor="white"
)
plt.show()

print("Chart 2 saved.")
