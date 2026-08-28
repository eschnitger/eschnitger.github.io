"""
Generate all six Jupyter notebooks for the European Football Divide project.
Usage:
   python build_notebooks.py          # generate notebooks
   python build_notebooks.py --test   # run smoke tests first
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf


NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def md(text: str):
   return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
   return nbf.v4.new_code_cell(text.strip("\n"))


def write_notebook(path: Path, cells: list):
   nb = nbf.v4.new_notebook()
   nb.cells = cells
   nb.metadata = {
       "kernelspec": {
           "display_name": "Python 3",
           "language": "python",
           "name": "python3",
       },
       "language_info": {"name": "python", "version": "3.11"},
   }
   with open(path, "w", encoding="utf-8") as f:
       nbf.write(nb, f)
   print(f"  wrote {path}")


IMPORTS = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

from _utils import (
   LEAGUES, scrape_league, enrich_dataframe,
   build_league_chart, build_comparison_chart, build_frequency_tables,
)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
"""


def _resolve_outlier_code():
   return """
# Resolve missing y-values from actual data
for o in outliers:
   if o.get("y") is None:
       match = df[df["Season"] == o["season"]]
       if not match.empty:
           o["y"] = int(match["Title-Winning Points"].iloc[0])
"""


def build_premier_league():
   cells = [
       md("""
# Premier League: The Growing Divide
**33 seasons of English football's increasing stratification**

*Eric Schnitger, 2026*

---

## Key Findings

- Title-winning points trend upward at roughly **+0.25 pts/season**
- Relegation survival points are flat or declining at **-0.15 pts/season**
- The gap has nearly **doubled** since 1992
- Sharpest inflection: **2016/17** (Pep + Klopp arms race)
- The **same six clubs** dominate the top 4
       """),
       code(IMPORTS),
       code("""
config = LEAGUES["premier_league"]
df = scrape_league("premier_league", cache_dir="../data")
df = enrich_dataframe(df)

print(f"Loaded {len(df)} seasons")
df[["Season", "Champion", "Title-Winning Points",
   "Survived Relegation", "Relegation Survival Points",
   "Relegated 1", "Relegated 2", "Relegated 3"]].tail(10)
"""),
       md("## Interactive Analysis\n\nUse the **dropdown** to switch eras."),
       code("""
eras = [
   {"label": "Full History (1992-2025)",
    "start": df["Season"].iloc[0], "end": df["Season"].iloc[-1],
    "title": "Premier League: The Growing Divide (1992-2025)"},
   {"label": "Pre-Abramovich (1992-2003)",
    "start": "1992/93", "end": "2002/03",
    "title": "Premier League: Pre-Abramovich Era"},
   {"label": "Oligarch Era (2003-2016)",
    "start": "2003/04", "end": "2015/16",
    "title": "Premier League: The Oligarch Era"},
   {"label": "Pep/Klopp Era (2016-2025)",
    "start": "2016/17", "end": df["Season"].iloc[-1],
    "title": "Premier League: The Pep/Klopp Era"},
]

outliers = [
   {"season": "2003/04", "y": 90,  "text": "Invincibles", "color": "#FFD54F", "ax": -50, "ay": -30},
   {"season": "2015/16", "y": 81,  "text": "Leicester!",  "color": "#66FF66", "ax": -50, "ay": -30},
   {"season": "2017/18", "y": 100, "text": "100 pts",     "color": "#4FC3F7", "ax": 0,   "ay": -25},
   {"season": "2019/20", "y": 99,  "text": "COVID",       "color": "#EF5350", "ax": 40,  "ay": -20},
]
""" + _resolve_outlier_code()),
       code("fig = build_league_chart(df, config, eras=eras, outliers=outliers)\nfig"),
       md("## Champions League Concentration"),
       code("""
top4, relegated = build_frequency_tables(df, config)
top4.head(10).style.format({"Percentage": "{:.1f}%"}).hide(axis="index")
"""),
       md("## Relegated Clubs"),
       code("relegated.head(10).style.hide(axis=\"index\")"),
       md("## Era Comparison"),
       code("""
rows = []
for era in eras:
   if era["start"] not in df["Season"].values:
       continue
   s = df[df["Season"] == era["start"]].index[0]
   e = df[df["Season"] == era["end"]].index[0]
   sub = df.iloc[s:e+1]
   rows.append({
       "Era": era["label"].split("(")[0].strip(),
       "Seasons": len(sub),
       "Avg Title": round(sub["Title-Winning Points"].mean(), 1),
       "Avg Survival": round(sub["Relegation Survival Points"].mean(), 1),
       "Avg Gap": round(sub["Gap"].mean(), 1),
       "Avg Ratio": round(sub["Ratio"].mean(), 2),
   })
pd.DataFrame(rows).style.format({"Avg Ratio": "{:.2f}x"}).hide(axis="index")
"""),
   ]
   write_notebook(NOTEBOOKS_DIR / "01_premier_league.ipynb", cells)


def build_bundesliga():
   cells = [
       md("""
# Bundesliga: Bayern's Iron Grip
**32 seasons of German football's competitive imbalance**

*Eric Schnitger, 2026*

---

## Key Findings

- Bayern have won ~21 of 32 titles (**66%**) since 1992
- Gap trend is flatter than England (dominance already entrenched)
- **Dortmund 2010-12** is the only sustained challenge
       """),
       code(IMPORTS),
       code("""
config = LEAGUES["bundesliga"]
df = scrape_league("bundesliga", cache_dir="../data")
df = enrich_dataframe(df)
print(f"Loaded {len(df)} seasons")
df[["Season", "Champion", "Title-Winning Points",
   "Survived Relegation", "Relegation Survival Points"]].tail(10)
"""),
       md("## Interactive Analysis"),
       code("""
eras = [
   {"label": "Full History (1992-2025)",
    "start": df["Season"].iloc[0], "end": df["Season"].iloc[-1],
    "title": "Bundesliga: Title vs. Relegation (1992-2025)"},
   {"label": "Pre-Guardiola (1992-2013)",
    "start": "1992/93", "end": "2012/13",
    "title": "Bundesliga: Traditional Era"},
   {"label": "Modern Bayern (2013-2025)",
    "start": "2013/14", "end": df["Season"].iloc[-1],
    "title": "Bundesliga: Modern Bayern"},
]

outliers = [
   {"season": "2012/13", "y": None, "text": "Bayern Treble", "color": "#EF5350", "ax": 0, "ay": -25},
]
""" + _resolve_outlier_code()),
       code("fig = build_league_chart(df, config, eras=eras, outliers=outliers)\nfig"),
       md("## Top 4 Concentration"),
       code("""
top4, relegated = build_frequency_tables(df, config)
top4.head(10).style.format({"Percentage": "{:.1f}%"}).hide(axis="index")
"""),
       md("## Relegated Clubs"),
       code("relegated.head(10).style.hide(axis=\"index\")"),
   ]
   write_notebook(NOTEBOOKS_DIR / "02_bundesliga.ipynb", cells)


def build_la_liga():
   cells = [
       md("""
# La Liga: The Duopoly and Beyond
**33 seasons of Real-Barca dominance in Spain**

*Eric Schnitger, 2026*

---

## Key Findings

- Real + Barca account for ~27 of 33 titles
- **Atletico's 2014 title** is the only genuine interruption
- Title points peaked at **100** (2011/12 and 2012/13)
       """),
       code(IMPORTS),
       code("""
config = LEAGUES["la_liga"]
df = scrape_league("la_liga", cache_dir="../data")
df = enrich_dataframe(df)
print(f"Loaded {len(df)} seasons")
df.tail(10)
"""),
       md("## Interactive Analysis"),
       code("""
eras = [
   {"label": "Full History (1992-2025)",
    "start": df["Season"].iloc[0], "end": df["Season"].iloc[-1],
    "title": "La Liga: The Duopoly (1992-2025)"},
   {"label": "Classic Era (1992-2004)",
    "start": "1992/93", "end": "2003/04",
    "title": "La Liga: Classic Duopoly"},
   {"label": "Messi/Ronaldo (2004-2018)",
    "start": "2004/05", "end": "2017/18",
    "title": "La Liga: The Messi/Ronaldo Era"},
   {"label": "Post-Legends (2018-2025)",
    "start": "2018/19", "end": df["Season"].iloc[-1],
    "title": "La Liga: Post-Legends"},
]

outliers = [
   {"season": "2011/12", "y": None, "text": "Real 100 pts",   "color": "#FFD54F", "ax": 40,  "ay": -25},
   {"season": "2013/14", "y": None, "text": "Atletico title", "color": "#26C6DA", "ax": -50, "ay": -30},
]
""" + _resolve_outlier_code()),
       code("fig = build_league_chart(df, config, eras=eras, outliers=outliers)\nfig"),
       md("## Top 4 Concentration"),
       code("""
top4, relegated = build_frequency_tables(df, config)
top4.head(10).style.format({"Percentage": "{:.1f}%"}).hide(axis="index")
"""),
       md("## Relegated Clubs"),
       code("relegated.head(10).style.hide(axis=\"index\")"),
   ]
   write_notebook(NOTEBOOKS_DIR / "03_la_liga.ipynb", cells)


def build_ligue_1():
   cells = [
       md("""
# Ligue 1: The PSG Effect
**French football before and after the Qatari takeover**

*Eric Schnitger, 2026*

---

## Key Findings

- Pre-PSG: **8 different champions** in 20 seasons
- Post-PSG: PSG have won ~10 of 13 titles
- **Montpellier (2012)**, **Monaco (2017)**, **Lille (2021)** are remarkable upsets
       """),
       code(IMPORTS),
       code("""
config = LEAGUES["ligue_1"]
df = scrape_league("ligue_1", cache_dir="../data")
df = enrich_dataframe(df)
print(f"Loaded {len(df)} seasons")
df.tail(10)
"""),
       md("## Interactive Analysis"),
       code("""
eras = [
   {"label": "Full History (1992-2025)",
    "start": df["Season"].iloc[0], "end": df["Season"].iloc[-1],
    "title": "Ligue 1: The PSG Effect (1992-2025)"},
   {"label": "Open Era (1992-2002)",
    "start": "1992/93", "end": "2001/02",
    "title": "Ligue 1: The Open Era"},
   {"label": "Lyon Dynasty (2002-2008)",
    "start": "2002/03", "end": "2007/08",
    "title": "Ligue 1: Lyon's Seven Titles"},
   {"label": "PSG Era (2011-2025)",
    "start": "2011/12", "end": df["Season"].iloc[-1],
    "title": "Ligue 1: The PSG Era"},
]

notable = [
   ("2011/12", "Montpellier!",  "#66FF66", -50, -30),
   ("2015/16", "PSG dominant",  "#4FC3F7",   0, -25),
   ("2020/21", "Lille upset",   "#FFD54F", -50, -30),
]
outliers = []
for season, text, color, ax, ay in notable:
   match = df[df["Season"] == season]
   if not match.empty:
       outliers.append({
           "season": season,
           "y": int(match["Title-Winning Points"].iloc[0]),
           "text": text, "color": color, "ax": ax, "ay": ay,
       })
"""),
       code("fig = build_league_chart(df, config, eras=eras, outliers=outliers)\nfig"),
       md("## Top 4 Concentration"),
       code("""
top4, relegated = build_frequency_tables(df, config)
top4.head(10).style.format({"Percentage": "{:.1f}%"}).hide(axis="index")
"""),
       md("## Relegated Clubs"),
       code("relegated.head(10).style.hide(axis=\"index\")"),
   ]
   write_notebook(NOTEBOOKS_DIR / "04_ligue_1.ipynb", cells)


def build_serie_a():
   cells = [
       md("""
# Serie A: From Seven Sisters to Juve's Nine
**Italian football's collapse and partial recovery**

*Eric Schnitger, 2026*

---

## Key Findings

- **Seven Sisters (1992-2006)**: 5 different champions
- **Calciopoli (2006)** broke the old order
- **Juventus 2012-2020**: nine consecutive titles
- Post-Juve: Inter (2021, 2024), Milan (2022), Napoli (2023)
       """),
       code(IMPORTS),
       code("""
config = LEAGUES["serie_a"]
df = scrape_league("serie_a", cache_dir="../data")
df = enrich_dataframe(df)
print(f"Loaded {len(df)} seasons")
df.tail(10)
"""),
       md("## Interactive Analysis"),
       code("""
eras = [
   {"label": "Full History (1992-2025)",
    "start": df["Season"].iloc[0], "end": df["Season"].iloc[-1],
    "title": "Serie A: Seven Sisters to Monopoly (1992-2025)"},
   {"label": "Seven Sisters (1992-2006)",
    "start": "1992/93", "end": "2005/06",
    "title": "Serie A: The Seven Sisters"},
   {"label": "Juve Dynasty (2011-2020)",
    "start": "2011/12", "end": "2019/20",
    "title": "Serie A: Juventus's Nine"},
   {"label": "Post-Juve (2020-2025)",
    "start": "2020/21", "end": df["Season"].iloc[-1],
    "title": "Serie A: Post-Juve"},
]

notable = [
   ("2013/14", "Juve 102 pts",    "#FFD54F",   0, -25),
   ("2022/23", "Napoli dominant", "#26C6DA", -50, -30),
]
outliers = []
for season, text, color, ax, ay in notable:
   match = df[df["Season"] == season]
   if not match.empty:
       outliers.append({
           "season": season,
           "y": int(match["Title-Winning Points"].iloc[0]),
           "text": text, "color": color, "ax": ax, "ay": ay,
       })
"""),
       code("fig = build_league_chart(df, config, eras=eras, outliers=outliers)\nfig"),
       md("## Top 4 Concentration"),
       code("""
top4, relegated = build_frequency_tables(df, config)
top4.head(10).style.format({"Percentage": "{:.1f}%"}).hide(axis="index")
"""),
       md("## Relegated Clubs"),
       code("relegated.head(10).style.hide(axis=\"index\")"),
   ]
   write_notebook(NOTEBOOKS_DIR / "05_serie_a.ipynb", cells)


def build_comparison():
   cells = [
       md("""
# The European Football Divide
**Cross-league comparison of competitive inequality (1992-2025)**

*Eric Schnitger, 2026*

---

## Key Findings

- **All five leagues show a widening gap** since 1992
- **Bundesliga** has the highest average gap (Bayern entrenched early)
- **Ligue 1** shows the most dramatic inflection (PSG takeover, 2011)
- **Premier League / La Liga** have the steepest rate of increase
- **Serie A** is the most volatile (Calciopoli, bankruptcies)
- **No league has reversed the trend** through regulation alone
       """),
       code(IMPORTS),
       code("""
league_data = {}
for key in ["premier_league", "bundesliga", "la_liga", "ligue_1", "serie_a"]:
   config = LEAGUES[key]
   raw = scrape_league(key, cache_dir="../data")
   league_data[key] = enrich_dataframe(raw)
   print(f"  {config.name}: {len(league_data[key])} seasons")

total = sum(len(v) for v in league_data.values())
print(f"\\nTotal: {total} season-records across 5 leagues")
"""),
       md("## The Gap Over Time (Normalized to 38 games)"),
       code("""
fig = build_comparison_chart(
   league_data,
   metric="Gap (38-game)",
   title="The Growing Divide: Gap Between Champions and Survival (Normalized)",
)
fig
"""),
       md("## Title-Winning Points (Normalized)"),
       code("""
fig = build_comparison_chart(
   league_data,
   metric="Title Pts (38-game)",
   title="Title-Winning Points Across Europe (38-Game Equivalent)",
)
fig
"""),
       md("## Relegation Survival Points (Normalized)"),
       code("""
fig = build_comparison_chart(
   league_data,
   metric="Survival Pts (38-game)",
   title="Relegation Survival Points Across Europe (38-Game Equivalent)",
)
fig
"""),
       md("## Points Ratio (Unit-Free)"),
       code("""
fig = build_comparison_chart(
   league_data, metric="Ratio",
   title="Competitive Inequality: Champion-to-Survival Points Ratio",
)
fig.update_layout(yaxis=dict(title="Ratio (champion pts / survival pts)"))
fig
"""),
       md("## League Summary Statistics"),
       code("""
rows = []
for key, df in league_data.items():
   config = LEAGUES[key]
   row = {
       "League": config.name, "Country": config.country,
       "Seasons": len(df), "Teams": config.num_teams,
       "Avg Title (38g)":    round(df["Title Pts (38-game)"].mean(), 1),
       "Avg Survival (38g)": round(df["Survival Pts (38-game)"].mean(), 1),
       "Avg Gap (38g)":      round(df["Gap (38-game)"].mean(), 1),
       "Avg Ratio": round(df["Ratio"].mean(), 2),
       "Max Ratio": round(df["Ratio"].max(), 2),
   }
   x = np.arange(len(df)).astype(float)
   y = df["Gap (38-game)"].values.astype(float)
   mask = ~np.isnan(y)
   if mask.sum() >= 3:
       row["Gap Trend (pts/yr)"] = round(np.polyfit(x[mask], y[mask], 1)[0], 3)
   rows.append(row)

pd.DataFrame(rows).style.format({
   "Avg Ratio": "{:.2f}x", "Max Ratio": "{:.2f}x",
   "Gap Trend (pts/yr)": "{:+.3f}",
}).hide(axis="index")
"""),
       md("## Top-3 Club Concentration"),
       code("""
conc = []
for key, df in league_data.items():
   config = LEAGUES[key]
   top4, _ = build_frequency_tables(df, config)
   total_slots = len(df) * 4
   share = top4.head(3)["Top 4 Finishes"].sum() / total_slots * 100
   conc.append({"League": config.name,
                "Top 3 Share (%)": round(share, 1),
                "Color": config.color})

conc_df = pd.DataFrame(conc).sort_values("Top 3 Share (%)")

fig = go.Figure(data=[go.Bar(
   x=conc_df["Top 3 Share (%)"], y=conc_df["League"], orientation="h",
   marker=dict(color=conc_df["Color"].tolist()),
   hovertemplate="<b>%{y}</b><br>%{x:.1f}% of all top-4 slots<extra></extra>",
)])
fig.update_layout(
   template="plotly_dark",
   title="Competitive Concentration: Top 3 Clubs' Share of Top-4 Finishes",
   xaxis=dict(title="% of Top-4 Slots", range=[0, 100]),
   height=400, width=900,
)
fig
"""),
       md("## Decade-by-Decade Gap"),
       code("""
rows = []
for key, df in league_data.items():
   config = LEAGUES[key]
   tmp = df.copy()
   tmp["Decade"] = (tmp["Start Year"] // 10) * 10
   tmp["Decade Label"] = tmp["Decade"].astype(str) + "s"
   for decade, group in tmp.groupby("Decade Label"):
       rows.append({
           "League": config.name, "Decade": decade,
           "Avg Gap (38g)": round(group["Gap (38-game)"].mean(), 1),
       })
decade_df = pd.DataFrame(rows)

fig = go.Figure()
for key in league_data:
   config = LEAGUES[key]
   sub = decade_df[decade_df["League"] == config.name].sort_values("Decade")
   fig.add_trace(go.Bar(
       x=sub["Decade"], y=sub["Avg Gap (38g)"],
       name=config.name, marker=dict(color=config.color, opacity=0.85),
   ))
fig.update_layout(
   template="plotly_dark",
   title="Average Gap by Decade (38-Game Normalized)",
   barmode="group",
   xaxis=dict(title="Decade"),
   yaxis=dict(title="Avg Gap (38g equivalent)"),
   height=500, width=1000,
   legend=dict(orientation="h", yanchor="bottom", y=1.02,
               xanchor="center", x=0.5),
)
fig
"""),
       md("""
## Conclusions

1. **The divergence is universal.** Every top-5 European league is more stratified in 2025 than in 1992.
2. **The causes are structural.** TV revenue concentration and Champions League prize money create a feedback loop.
3. **The rate differs by league.** England and Spain accelerated fastest; Germany was already uneven; France had a single shock; Italy has been chaotic.
4. **Survival thresholds are universal.** ~33-37 points (38-game equivalent) across all leagues.
5. **No regulation has worked.** Not 50+1, not collective TV deals, not FFP.
       """),
   ]
   write_notebook(NOTEBOOKS_DIR / "06_comparison.ipynb", cells)


def main():
   if "--test" in sys.argv:
       print("Running test suite first...\n")
       r = subprocess.run([sys.executable, "test_suite.py"])
       if r.returncode != 0:
           print("\nTests failed. Aborting notebook generation.")
           sys.exit(1)
       print()

   print("Generating notebooks in ./notebooks/ ...")
   build_premier_league()
   build_bundesliga()
   build_la_liga()
   build_ligue_1()
   build_serie_a()
   build_comparison()
   print("\nDone. Open notebooks/ in Jupyter.")


if __name__ == "__main__":
   main()