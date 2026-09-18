import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------
# DATA
# Verified NLC Survey Findings on Municipal AI Readiness
# Source: National League of Cities (NLC) AI in Cities Report
# -------------------------------------------------------

metrics = [
    "Mayors Reporting Active Interest\nin AI Adoption",
    "Tech Leaders Reporting City is\nNot Fully Prepared for AI",
    "Cities with Dedicated AI\nPersonnel or Leads",
    "Cities with Formal AI\nGovernance Policies"
]

percentages = [96, 38, 10, 9]

# -------------------------------------------------------
# FIGURE SETUP
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5.5))

# Palette matching Charts 1 & 2 styling
colors = [
    "#1a5276",  # High mayoral demand
    "#d9534f",  # Unpreparedness warning metric
    "#5dade2",  # Dedicated staffing
    "#85c1e9"   # Policy governance
]

bars = ax.barh(
    metrics,
    percentages,
    color=colors,
    edgecolor="white",
    height=0.55
)

# -------------------------------------------------------
# LABELS AND ANNOTATIONS
# -------------------------------------------------------

# Add percentage labels to the end of each bar
for bar, pct in zip(bars, percentages):
    ax.text(
        bar.get_width() + 1.5,
        bar.get_y() + bar.get_height() / 2,
        f"{pct}%",
        va="center",
        ha="left",
        fontsize=11,
        fontweight="bold",
        color="#1a1a1a"
    )

# Highlight structural gap note
ax.annotate(
    "Structural Bottleneck:\n96% demand vs. single-digit policy readiness",
    xy=(15, 3),
    xytext=(35, 2.7),
    fontsize=9,
    color="#d9534f",
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="->",
        color="#d9534f",
        lw=1.5
    ),
    ha="left"
)

# -------------------------------------------------------
# AXIS AND FORMATTING
# -------------------------------------------------------

ax.set_xlabel(
    "Percentage of Surveyed Local Governments / Leaders",
    fontsize=11,
    labelpad=10,
    color="#333333"
)

ax.set_title(
    "Municipal AI Readiness & Governance Disparities",
    fontsize=14,
    fontweight="bold",
    pad=15,
    color="#1a1a1a"
)

# Subtitle
fig.text(
    0.125, 0.92,
    "Near-universal executive interest contrasts sharply with low operational readiness and policy frameworks",
    fontsize=10,
    color="#555555",
    style="italic"
)

# Clean up spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#cccccc")
ax.spines["left"].set_color("#cccccc")

# X axis range with padding
ax.set_xlim(0, 110)
ax.xaxis.set_tick_params(labelsize=10, colors="#555555")
ax.yaxis.set_tick_params(labelsize=10, colors="#333333")

# Gridlines
ax.xaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
ax.set_axisbelow(True)

# Invert y-axis order
ax.invert_yaxis()

# Source note
fig.text(
    0.125, 0.01,
    "Source: National League of Cities (NLC), 'AI in Cities Report' & 'AI & Emerging Technology Forum Findings.'",
    fontsize=8,
    color="#888888"
)

plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.savefig(
    "chart3_nlc_ai_readiness.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white"
)
plt.show()

print("Chart 3 saved.")