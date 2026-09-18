import matplotlib.pyplot as plt
import numpy as np

# =======================================================
# CHART 1 — AI Adoption Rates by Jurisdiction Size
# =======================================================
jurisdiction_sizes = [
    "Very Large\n(500,000+)",
    "Large\n(100,000–500,000)",
    "Medium\n(50,000–100,000)",
    "Small-Medium\n(25,000–50,000)",
    "Small\n(10,000–25,000)",
    "Very Small\n(Under 10,000)"
]
adoption_rates_c1 = [84, 63, 44, 29, 18, 9]

fig1, ax1 = plt.subplots(figsize=(10, 5.5))
colors_c1 = ["#1a5276", "#2471a3", "#2e86c1", "#5dade2", "#85c1e9", "#aed6f1"]
bars1 = ax1.barh(jurisdiction_sizes, adoption_rates_c1, color=colors_c1, edgecolor="white", height=0.6)

for bar, rate in zip(bars1, adoption_rates_c1):
    ax1.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2, f"{rate}%", va="center", ha="left", fontsize=11, fontweight="bold", color="#1a1a1a")

ax1.set_xlabel("Percentage of Governments Reporting AI Tool Adoption", fontsize=11, labelpad=10, color="#333333")
ax1.set_title("AI Adoption in Local Government by Jurisdiction Size", fontsize=14, fontweight="bold", pad=15, color="#1a1a1a")
fig1.text(0.125, 0.92, "Larger governments adopt at dramatically higher rates, leaving smaller jurisdictions largely behind", fontsize=10, color="#555555", style="italic")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_color("#cccccc")
ax1.spines["left"].set_color("#cccccc")
ax1.set_xlim(0, 105)
ax1.xaxis.set_tick_params(labelsize=10, colors="#555555")
ax1.yaxis.set_tick_params(labelsize=10, colors="#333333")
ax1.xaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
ax1.set_axisbelow(True)
ax1.invert_yaxis()

fig1.text(0.125, 0.01, "Source: National League of Cities (NLC), 'State of the Cities' & ICMA Survey Reports.", fontsize=8, color="#888888")
plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.savefig("chart1_adoption_by_jurisdiction_size.png", dpi=150, bbox_inches="tight", facecolor="white")


# =======================================================
# CHART 2 — AI Adoption by Government Function
# =======================================================
functions = [
    "Resident Engagement\n& Chatbots",
    "Local Policy &\nBudget Analysis",
    "Permit & License\nProcessing",
    "Human Resources\n& Recruitment",
    "Procurement\n& Contracting",
    "Public Safety\nOperations",
    "Workforce\nScheduling"
]
adoption_rates_c2 = [55, 38, 32, 28, 24, 18, 13]

fig2, ax2 = plt.subplots(figsize=(11, 6))
bar_colors_c2 = ["#5d8aa8", "#5d8aa8", "#5d8aa8", "#5d8aa8", "#1a5276", "#5d8aa8", "#5d8aa8"]
bars2 = ax2.bar(functions, adoption_rates_c2, color=bar_colors_c2, edgecolor="white", width=0.65)

for bar, rate in zip(bars2, adoption_rates_c2):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{rate}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1a1a1a")

ax2.annotate(
    "Procurement — PPG's core focus —\nranks fifth in adoption potential\nwith limited outcome evidence",
    xy=(4, 28),
    xytext=(5, 42),
    fontsize=9,
    color="#1a5276",
    fontweight="bold",
    arrowprops = dict(
        arrowstyle="->",
        color="#d9534f",
        lw=1.5,
        connectionstyle="arc3,rad=0.4"
    ),
    ha = "left",
    va = "center",
    #ha="center"
)




ax2.set_ylabel("Percentage of Governments Reporting Adoption / Potential", fontsize=11, labelpad=10, color="#333333")
ax2.set_title("AI Adoption in Local Government by Operational Function", fontsize=14, fontweight="bold", pad=15, color="#1a1a1a")
fig2.text(0.125, 0.92, "Resident-facing & policy tools lead focus; procurement and operational tools lag behind", fontsize=10, color="#555555", style="italic")

ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_color("#cccccc")
ax2.spines["left"].set_color("#cccccc")
ax2.set_ylim(0, 70)
ax2.yaxis.set_tick_params(labelsize=10, colors="#555555")
ax2.xaxis.set_tick_params(labelsize=9, colors="#333333")
ax2.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
ax2.set_axisbelow(True)

fig2.text(0.125, 0.01, "Source: ICMA, 'Artificial Intelligence in Local Government Survey Report.'", fontsize=8, color="#888888")
plt.tight_layout(rect=[0, 0.04, 1, 0.93])
plt.savefig("chart2_adoption_by_function.png", dpi=150, bbox_inches="tight", facecolor="white")


# =======================================================
# CHART 3 — NLC Municipal AI Readiness Benchmarks
# =======================================================
metrics = [
    "Mayors Reporting Active Interest\nin AI Adoption",
    "Tech Leaders Reporting City is\nNot Fully Prepared for AI",
    "Cities with Dedicated AI\nPersonnel or Leads",
    "Cities with Formal AI\nGovernance Policies"
]
percentages = [96, 38, 10, 9]

fig3, ax3 = plt.subplots(figsize=(10, 5.5))
colors_c3 = ["#1a5276", "#d9534f", "#5dade2", "#85c1e9"]
bars3 = ax3.barh(metrics, percentages, color=colors_c3, edgecolor="white", height=0.55)

for bar, pct in zip(bars3, percentages):
    ax3.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2, f"{pct}%", va="center", ha="left", fontsize=11, fontweight="bold", color="#1a1a1a")

ax3.annotate(
    "Structural Bottleneck:\n96% demand vs. single-digit policy readiness",
    xy=(15, 3),
    xytext=(42, 2.2),
    fontsize=9,
    color="#d9534f",
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="->",
        color="#d9534f",
        lw=1.5,
        connectionstyle="arc3,rad=-0.2"
    ),
    ha="left",
    va="center"
)

ax3.set_xlabel("Percentage of Surveyed Local Governments / Leaders", fontsize=11, labelpad=10, color="#333333")
ax3.set_title("Municipal AI Readiness & Governance Disparities", fontsize=14, fontweight="bold", pad=15, color="#1a1a1a")
fig3.text(0.125, 0.92, "Near-universal executive interest contrasts sharply with low operational readiness and policy frameworks", fontsize=10, color="#555555", style="italic")

ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["bottom"].set_color("#cccccc")
ax3.spines["left"].set_color("#cccccc")
ax3.set_xlim(0, 110)
ax3.xaxis.set_tick_params(labelsize=10, colors="#555555")
ax3.yaxis.set_tick_params(labelsize=10, colors="#333333")
ax3.xaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
ax3.set_axisbelow(True)
ax3.invert_yaxis()

fig3.text(0.125, 0.01, "Source: National League of Cities (NLC), 'State of the Cities' & ICMA Survey Reports.", fontsize=8, color="#888888")
plt.tight_layout(rect=[0, 0.04, 1, 0.94])
plt.savefig("chart3_nlc_ai_readiness.png", dpi=150, bbox_inches="tight", facecolor="white")

plt.show()
print("All charts updated and rendered successfully.")