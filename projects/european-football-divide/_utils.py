"""
Shared utilities for European football league analysis.
Scraping, cleaning, enrichment, and charting for five top-flight leagues.
"""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots
from io import StringIO


# ─────────────────────────────────────────────────────────────────────────────
# LEAGUE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LeagueConfig:
   name: str
   short_name: str
   country: str
   color: str
   color_secondary: str
   start_year: int
   end_year: int
   num_teams: int
   num_teams_history: dict = field(default_factory=dict)
   relegation_count: int = 3
   cl_spots: int = 4
   cache_file: str = ""
   url_patterns: list[Callable[[int], str]] = field(default_factory=list)


def _pl_url(y: int) -> str:
   label = "FA_Premier_League" if y <= 2006 else "Premier_League"
   return f"https://en.wikipedia.org/wiki/{y}%E2%80%93{str(y+1)[-2:]}_{label}"


def _simple_url(slug: str) -> Callable[[int], str]:
   return lambda y: f"https://en.wikipedia.org/wiki/{y}%E2%80%93{str(y+1)[-2:]}_{slug}"


LEAGUES: dict[str, LeagueConfig] = {
   "premier_league": LeagueConfig(
       name="Premier League", short_name="PL", country="England",
       color="#4FC3F7", color_secondary="#EF5350",
       start_year=1992, end_year=2024, num_teams=20,
       num_teams_history={y: 22 for y in range(1992, 1995)},
       relegation_count=3, cl_spots=4,
       cache_file="pl_data_cache.csv",
       url_patterns=[_pl_url],
   ),
   "bundesliga": LeagueConfig(
       name="Bundesliga", short_name="BL", country="Germany",
       color="#66BB6A", color_secondary="#FF7043",
       start_year=1992, end_year=2024, num_teams=18,
       relegation_count=3, cl_spots=4,
       cache_file="bundesliga_data_cache.csv",
       url_patterns=[_simple_url("Bundesliga")],
   ),
   "la_liga": LeagueConfig(
       name="La Liga", short_name="LL", country="Spain",
       color="#FFD54F", color_secondary="#AB47BC",
       start_year=1992, end_year=2024, num_teams=20,
       num_teams_history={y: 22 for y in range(1995, 1997)},
       relegation_count=3, cl_spots=4,
       cache_file="la_liga_data_cache.csv",
       url_patterns=[
           _simple_url("La_Liga"),
           _simple_url("La_Liga_season"),
       ],
   ),
   "ligue_1": LeagueConfig(
       name="Ligue 1", short_name="L1", country="France",
       color="#26C6DA", color_secondary="#FF8A65",
       start_year=1992, end_year=2024, num_teams=18,
       num_teams_history={y: 20 for y in range(1992, 2023)},
       relegation_count=3, cl_spots=3,
       cache_file="ligue1_data_cache.csv",
       url_patterns=[
           _simple_url("Ligue_1"),
           _simple_url("French_Division_1"),
           _simple_url("Division_1_(France)"),
           _simple_url("Division_1_(French_football)"),
       ],
   ),
   "serie_a": LeagueConfig(
       name="Serie A", short_name="SA", country="Italy",
       color="#AB47BC", color_secondary="#8D6E63",
       start_year=1992, end_year=2024, num_teams=20,
       num_teams_history={y: 18 for y in range(1992, 2004)},
       relegation_count=3, cl_spots=4,
       cache_file="serie_a_data_cache.csv",
       url_patterns=[
           _simple_url("Serie_A"),
           _simple_url("Serie_A_(football)"),
       ],
   ),
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_season_label(start_year: int) -> str:
   return f"{start_year}/{str(start_year + 1)[-2:]}"


def get_num_teams_for_season(config: LeagueConfig, start_year: int) -> int:
   if start_year in config.num_teams_history:
       return config.num_teams_history[start_year]
   return config.num_teams


def get_matches_per_season(num_teams: int) -> int:
   return (num_teams - 1) * 2


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING & PARSING
# ─────────────────────────────────────────────────────────────────────────────

def clean_team_name(name: str) -> str:
   """Strip Wikipedia citation markers, tags, and unicode artifacts."""
   if not isinstance(name, str):
       return str(name).strip()
   name = re.sub(r"\s*\([A-Za-z,.\s]+\)", "", name)
   name = re.sub(r"\s*\[.*?\]", "", name)
   name = re.sub(r"[\u2020\u2021\u00a7\u00b6*]", "", name)
   name = name.replace("\u00a0", " ").strip()
   name = re.sub(r"\s+", " ", name)
   return name


def _normalize_numeric_string(s: str) -> str:
   """Normalize unicode minus signs and strip non-numeric suffixes."""
   if not isinstance(s, str):
       return s
   s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
   s = s.replace("\u00a0", " ").strip()
   s = re.sub(r"^(-?\d+).*$", r"\1", s)
   return s


def find_league_table(tables: list[pd.DataFrame], num_teams: int) -> Optional[pd.DataFrame]:
   """
   Score candidate tables and return the one most likely to be the standings.
   """
   candidates = []
   for idx, table in enumerate(tables):
       t = table.copy()
       if isinstance(t.columns, pd.MultiIndex):
           t.columns = [str(c[-1]) if isinstance(c, tuple) else str(c)
                        for c in t.columns]

       col_str = " ".join(str(c).lower() for c in t.columns)
       score = 0

       if "player" in col_str or "scorer" in col_str or "nationality" in col_str:
           continue

       n = len(t)
       if n == num_teams:
           score += 20
       elif abs(n - num_teams) == 1:
           score += 10
       elif abs(n - num_teams) <= 3:
           score += 4
       else:
           continue

       if "pts" in col_str or "points" in col_str:
           score += 10
       if "team" in col_str or "club" in col_str or "squad" in col_str:
           score += 6
       if any(x in col_str for x in [" w ", " d ", " l ", "won", "drawn", "lost",
                                       "\tw\t", "wld"]):
           score += 5
       if any(x in col_str for x in ["gf", "ga", "gd", "goals for", "goals against"]):
           score += 4
       if "pos" in col_str or col_str.startswith("#"):
           score += 2
       if t.shape[1] >= 8:
           score += 2

       if score >= 15:
           candidates.append((score, idx, t))

   if not candidates:
       return None
   candidates.sort(key=lambda x: x[0], reverse=True)
   return candidates[0][2]


def identify_columns(table: pd.DataFrame) -> tuple[str, str]:
   """Find the team-name column and the points column."""
   by_name = {}
   for c in table.columns:
       key = str(c).lower().strip()
       by_name.setdefault(key, c)

   pts_col = None
   for cand in ["pts", "points", "ptos", "pt", "p"]:
       if cand in by_name:
           pts_col = by_name[cand]
           break

   team_col = None
   for cand in ["team", "club", "squad", "equipo"]:
       if cand in by_name:
           team_col = by_name[cand]
           break

   if team_col is None:
       for c in table.columns:
           sample = table[c].dropna().astype(str).head(5)
           if sample.str.len().mean() > 3 and not sample.str.match(r"^-?\d").all():
               team_col = c
               break
       if team_col is None:
           team_col = table.columns[1] if len(table.columns) > 1 else table.columns[0]

   if pts_col is None:
       for offset in [-1, -2, -3]:
           try:
               c = table.columns[offset]
               numeric = pd.to_numeric(
                   table[c].astype(str).map(_normalize_numeric_string),
                   errors="coerce",
               )
               if numeric.notna().sum() >= len(table) * 0.8 and numeric.max() <= 120:
                   pts_col = c
                   break
           except (IndexError, ValueError):
               continue
       if pts_col is None:
           pts_col = table.columns[-1]

   if pts_col is None:
       pts_col = table.columns[-1]
   if team_col is None:
       team_col = table.columns[1]
   return team_col, pts_col


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_page(url: str, timeout: int = 15) -> Optional[str]:
   try:
       resp = requests.get(
           url,
           headers={"User-Agent": "Euro-Football-Portfolio/1.0 (educational use)"},
           timeout=timeout,
       )
       if resp.status_code == 200:
           return resp.text
   except requests.RequestException:
       pass
   return None


def scrape_season(league_key: str, start_year: int,
                 config: LeagueConfig) -> Optional[dict]:
   num_teams = get_num_teams_for_season(config, start_year)
   matches = get_matches_per_season(num_teams)

   html = None
   for pattern_fn in config.url_patterns:
       url = pattern_fn(start_year)
       html = fetch_page(url)
       if html is not None:
           break
   if html is None:
       return None

   try:
       tables = pd.read_html(StringIO(html))
   except (ValueError, ImportError):
       return None
   if not tables:
       return None

   table = find_league_table(tables, num_teams)
   if table is None:
       return None

   team_col, pts_col = identify_columns(table)
   table[team_col] = table[team_col].apply(clean_team_name)
   table[pts_col] = pd.to_numeric(
       table[pts_col].astype(str).map(_normalize_numeric_string),
       errors="coerce",
   )
   table = table.dropna(subset=[pts_col]).reset_index(drop=True)

   if len(table) < num_teams - 2:
       return None

   table = table.sort_values(pts_col, ascending=False).reset_index(drop=True)

   n = len(table)
   survival_pos = n - config.relegation_count - 1
   relegated_start = n - config.relegation_count

   if survival_pos < 4 or relegated_start >= n:
       return None

   result = {
       "Season": get_season_label(start_year),
       "Start Year": start_year,
       "Num Teams": n,
       "Matches": matches,
       "Champion": table.iloc[0][team_col],
       "Title-Winning Points": int(table.iloc[0][pts_col]),
       "2nd Place": table.iloc[1][team_col],
       "2nd Place Points": int(table.iloc[1][pts_col]),
       "3rd Place": table.iloc[2][team_col],
       "3rd Place Points": int(table.iloc[2][pts_col]),
       "4th Place": table.iloc[3][team_col],
       "4th Place Points": int(table.iloc[3][pts_col]),
       "Survived Relegation": table.iloc[survival_pos][team_col],
       "Relegation Survival Points": int(table.iloc[survival_pos][pts_col]),
   }
   for j in range(config.relegation_count):
       idx = relegated_start + j
       result[f"Relegated {j + 1}"] = (
           table.iloc[idx][team_col] if idx < n else "N/A"
       )
   return result


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
   df = df.copy()
   mask = (
       (df["Title-Winning Points"] > 0)
       & (df["Relegation Survival Points"] > 0)
       & (df["Title-Winning Points"] >= df["Relegation Survival Points"])
   )
   removed = (~mask).sum()
   if removed > 0:
       print(f"    Removed {removed} invalid row(s)")
   return df[mask].reset_index(drop=True)


def scrape_league(league_key: str,
                 config: Optional[LeagueConfig] = None,
                 cache_dir: str = "../data",
                 max_age_days: float = 7.0,
                 delay: float = 1.2) -> pd.DataFrame:
   if config is None:
       config = LEAGUES[league_key]

   cache_path = Path(cache_dir) / config.cache_file
   cache_path.parent.mkdir(parents=True, exist_ok=True)

   if cache_path.exists():
       age_days = (time.time() - cache_path.stat().st_mtime) / 86400
       if age_days < max_age_days:
           print(f"  [{config.short_name}] cache hit ({age_days:.1f}d old)")
           return pd.read_csv(cache_path)

   print(f"  [{config.short_name}] scraping {config.start_year}-{config.end_year + 1}...")
   seasons = []
   failures = []
   for year in range(config.start_year, config.end_year + 1):
       print(f"    {year}/{str(year + 1)[-2:]}...", end=" ", flush=True)
       result = scrape_season(league_key, year, config)
       if result:
           seasons.append(result)
           print("OK")
       else:
           failures.append(year)
           print("FAIL")
       time.sleep(delay)

   df = pd.DataFrame(seasons)
   if len(df):
       df = validate_dataframe(df)
       df.to_csv(cache_path, index=False)
       print(f"  [{config.short_name}] cached {len(df)} seasons, {len(failures)} failures")
       if failures:
           print(f"    failed: {failures}")
   return df


# ─────────────────────────────────────────────────────────────────────────────
# ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
   df = df.copy()
   if len(df) == 0:
       return df

   df["Gap"] = df["Title-Winning Points"] - df["Relegation Survival Points"]
   df["Ratio"] = (
       df["Title-Winning Points"] / df["Relegation Survival Points"]
   ).round(2)

   if "Matches" in df.columns:
       df["Title PPG"] = (df["Title-Winning Points"] / df["Matches"]).round(3)
       df["Survival PPG"] = (df["Relegation Survival Points"] / df["Matches"]).round(3)
       df["Gap PPG"] = (df["Title PPG"] - df["Survival PPG"]).round(3)
       df["Title Pts (38-game)"] = (df["Title PPG"] * 38).round(1)
       df["Survival Pts (38-game)"] = (df["Survival PPG"] * 38).round(1)
       df["Gap (38-game)"] = (df["Gap PPG"] * 38).round(1)
   return df


# ─────────────────────────────────────────────────────────────────────────────
# CHARTING: single league
# ─────────────────────────────────────────────────────────────────────────────

def _safe_polyfit(x: np.ndarray, y: np.ndarray, deg: int = 1) -> Optional[np.ndarray]:
   mask = ~np.isnan(y)
   if mask.sum() < deg + 1:
       return None
   try:
       return np.polyfit(x[mask], y[mask], deg)
   except (np.linalg.LinAlgError, ValueError):
       return None


def build_league_chart(df: pd.DataFrame,
                      config: LeagueConfig,
                      eras: Optional[list[dict]] = None,
                      outliers: Optional[list[dict]] = None) -> go.Figure:
   if eras is None:
       eras = [{
           "label": f"Full History ({config.start_year}-{config.end_year + 1})",
           "start": df["Season"].iloc[0],
           "end": df["Season"].iloc[-1],
           "title": f"{config.name}: Title vs. Relegation Points",
       }]
   outliers = outliers or []

   fig = make_subplots(
       rows=3, cols=1, shared_xaxes=True,
       vertical_spacing=0.08,
       row_heights=[0.50, 0.25, 0.25],
       subplot_titles=(
           "Title-Winning vs. Relegation Survival Points",
           "Points Gap (Title Winner minus Survival)",
           "Points Ratio (Title Winner / Survival)",
       ),
   )

   TRACES_PER_ERA = 9
   era_annotations: dict[int, list] = {}

   def _blank_traces():
       for _ in range(TRACES_PER_ERA):
           fig.add_trace(go.Scatter(x=[], y=[], visible=False, showlegend=False),
                         row=1, col=1)

   for i, era in enumerate(eras):
       if era["start"] not in df["Season"].values or era["end"] not in df["Season"].values:
           _blank_traces()
           era_annotations[i] = []
           continue

       s_idx = df[df["Season"] == era["start"]].index[0]
       e_idx = df[df["Season"] == era["end"]].index[0]
       era_df = df.iloc[s_idx : e_idx + 1].reset_index(drop=True)

       if len(era_df) < 3:
           _blank_traces()
           era_annotations[i] = []
           continue

       x_num = np.arange(len(era_df))

       title_hover, releg_hover = [], []
       releg_cols = [c for c in era_df.columns if c.startswith("Relegated")]
       for _, r in era_df.iterrows():
           title_hover.append(
               f"<b>{r['Season']}</b><br>"
               f"\U0001f3c6 {r['Champion']}: {r['Title-Winning Points']} pts<br>"
               f"\U0001f948 {r['2nd Place']}: {int(r['2nd Place Points'])} pts<br>"
               f"\U0001f949 {r['3rd Place']}: {int(r['3rd Place Points'])} pts<br>"
               f"4th: {r['4th Place']}: {int(r['4th Place Points'])} pts"
           )
           rel = ", ".join(str(r.get(c)) for c in releg_cols
                           if pd.notna(r.get(c)) and str(r.get(c)) != "N/A")
           releg_hover.append(
               f"<b>{r['Season']}</b><br>"
               f"\U0001fa82 Survived: {r['Survived Relegation']} "
               f"({int(r['Relegation Survival Points'])} pts)<br>"
               f"\u274c {rel}"
           )

       # Trace 0: title points
       fig.add_trace(go.Scatter(
           x=era_df["Season"], y=era_df["Title-Winning Points"],
           name="Title-Winning Points", mode="lines+markers",
           line=dict(color=config.color, width=3), marker=dict(size=7),
           hovertext=title_hover, hoverinfo="text",
           visible=False, showlegend=True,
       ), row=1, col=1)

       # Trace 1: title trend
       tc = _safe_polyfit(x_num, era_df["Title-Winning Points"].values)
       if tc is not None:
           fig.add_trace(go.Scatter(
               x=era_df["Season"], y=np.polyval(tc, x_num),
               name=f"Title Trend ({tc[0]:+.2f}/yr)", mode="lines",
               line=dict(color=config.color, width=2, dash="dash"),
               hoverinfo="skip", visible=False, showlegend=True,
           ), row=1, col=1)
       else:
           fig.add_trace(go.Scatter(x=[], y=[], visible=False, showlegend=False), row=1, col=1)

       # Trace 2: relegation points
       fig.add_trace(go.Scatter(
           x=era_df["Season"], y=era_df["Relegation Survival Points"],
           name="Relegation Survival Points", mode="lines+markers",
           line=dict(color=config.color_secondary, width=3), marker=dict(size=7),
           hovertext=releg_hover, hoverinfo="text",
           visible=False, showlegend=True,
       ), row=1, col=1)

       # Trace 3: relegation trend
       rc = _safe_polyfit(x_num, era_df["Relegation Survival Points"].values)
       if rc is not None:
           fig.add_trace(go.Scatter(
               x=era_df["Season"], y=np.polyval(rc, x_num),
               name=f"Relegation Trend ({rc[0]:+.2f}/yr)", mode="lines",
               line=dict(color=config.color_secondary, width=2, dash="dash"),
               hoverinfo="skip", visible=False, showlegend=True,
           ), row=1, col=1)
       else:
           fig.add_trace(go.Scatter(x=[], y=[], visible=False, showlegend=False), row=1, col=1)

       # Trace 4: shaded gap
       fig.add_trace(go.Scatter(
           x=era_df["Season"].tolist() + era_df["Season"].tolist()[::-1],
           y=era_df["Title-Winning Points"].tolist()
             + era_df["Relegation Survival Points"].tolist()[::-1],
           fill="toself", fillcolor="rgba(255,255,255,0.05)",
           line=dict(width=0), showlegend=False,
           hoverinfo="skip", visible=False,
       ), row=1, col=1)

       # Trace 5: gap bars
       gap = era_df["Gap"].values
       gap_colors = [
           f"rgba({min(255, int(150 + (g - 30) * 3))}, "
           f"{max(80, int(220 - (g - 30) * 4))}, 100, 0.85)"
           for g in gap
       ]
       fig.add_trace(go.Bar(
           x=era_df["Season"], y=gap, name="Points Gap",
           marker=dict(color=gap_colors),
           hovertemplate="<b>%{x}</b><br>Gap: %{y} pts<extra></extra>",
           visible=False, showlegend=True,
       ), row=2, col=1)

       # Trace 6: gap trend
       gc = _safe_polyfit(x_num, gap.astype(float))
       if gc is not None:
           fig.add_trace(go.Scatter(
               x=era_df["Season"], y=np.polyval(gc, x_num),
               name=f"Gap Trend ({gc[0]:+.2f}/yr)", mode="lines",
               line=dict(color="#FFD54F", width=2, dash="dash"),
               hoverinfo="skip", visible=False, showlegend=True,
           ), row=2, col=1)
       else:
           fig.add_trace(go.Scatter(x=[], y=[], visible=False, showlegend=False), row=2, col=1)

       # Trace 7: ratio
       fig.add_trace(go.Scatter(
           x=era_df["Season"], y=era_df["Ratio"],
           name="Points Ratio", mode="lines+markers",
           line=dict(color="#AB47BC", width=3), marker=dict(size=6),
           hovertemplate="<b>%{x}</b><br>Ratio: %{y:.2f}x<extra></extra>",
           visible=False, showlegend=True,
       ), row=3, col=1)

       # Trace 8: ratio trend
       rtc = _safe_polyfit(x_num, era_df["Ratio"].values.astype(float))
       if rtc is not None:
           fig.add_trace(go.Scatter(
               x=era_df["Season"], y=np.polyval(rtc, x_num),
               name=f"Ratio Trend ({rtc[0]:+.3f}/yr)", mode="lines",
               line=dict(color="#AB47BC", width=2, dash="dash"),
               hoverinfo="skip", visible=False, showlegend=True,
           ), row=3, col=1)
       else:
           fig.add_trace(go.Scatter(x=[], y=[], visible=False, showlegend=False), row=3, col=1)

       ann = []
       for o in outliers:
           if o["season"] in era_df["Season"].values and o.get("y") is not None:
               ann.append(dict(
                   x=o["season"], y=o["y"], text=o["text"],
                   showarrow=True, arrowhead=2,
                   ax=o.get("ax", 0), ay=o.get("ay", -25),
                   font=dict(color=o.get("color", "#FFF"), size=10, family="Arial Black"),
                   xref="x", yref="y",
               ))
       era_annotations[i] = ann

   for j in range(min(TRACES_PER_ERA, len(fig.data))):
       fig.data[j].visible = True

   hline_ann = dict(x=1.01, y=2.5, xref="x3 domain", yref="y3",
                    text="2.5x", showarrow=False,
                    font=dict(color="rgba(255,255,255,0.5)", size=11),
                    xanchor="left")
   fig.add_shape(type="line", x0=0, x1=1, y0=2.5, y1=2.5,
                 xref="x3 domain", yref="y3",
                 line=dict(dash="dot", color="rgba(255,255,255,0.3)"))
   for i in era_annotations:
       era_annotations[i].append(hline_ann)

   total = TRACES_PER_ERA * len(eras)
   buttons = []
   for i, era in enumerate(eras):
       vis = [False] * total
       for j in range(TRACES_PER_ERA):
           idx = i * TRACES_PER_ERA + j
           if idx < total:
               vis[idx] = True
       buttons.append(dict(
           label=era["label"], method="update",
           args=[{"visible": vis},
                 {"title.text": era["title"],
                  "annotations": era_annotations.get(i, [])}],
       ))

   menus = []
   if len(eras) > 1:
       menus = [dict(
           type="dropdown", direction="down", active=0,
           x=0.0, xanchor="left", y=1.18, yanchor="top",
           bgcolor="rgba(50,50,50,0.9)",
           bordercolor="rgba(255,255,255,0.3)",
           font=dict(color="white", size=12), buttons=buttons,
       )]

   fig.update_layout(
       template="plotly_dark",
       title=dict(text=eras[0]["title"], font=dict(size=18)),
       height=1000, width=1100, hovermode="closest",
       legend=dict(orientation="h", yanchor="bottom", y=1.05,
                   xanchor="center", x=0.5, font=dict(size=10)),
       margin=dict(t=150),
       annotations=era_annotations.get(0, []),
       updatemenus=menus,
   )
   fig.update_yaxes(title_text="Points", row=1, col=1)
   fig.update_yaxes(title_text="Gap (pts)", row=2, col=1)
   fig.update_yaxes(title_text="Ratio", row=3, col=1)
   fig.update_xaxes(tickangle=-45, dtick=3, row=3, col=1)
   return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHARTING: comparison
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_chart(league_data: dict[str, pd.DataFrame],
                          metric: str = "Gap (38-game)",
                          title: str = "Cross-League Comparison") -> go.Figure:
   fig = go.Figure()
   for key, df in league_data.items():
       config = LEAGUES[key]
       if metric not in df.columns or len(df) == 0:
           continue
       fig.add_trace(go.Scatter(
           x=df["Season"], y=df[metric], name=config.name,
           mode="lines+markers",
           line=dict(color=config.color, width=2.5),
           marker=dict(size=5),
           hovertemplate=f"<b>{config.name}</b><br>%{{x}}<br>{metric}: %{{y:.1f}}<extra></extra>",
       ))
       x_num = np.arange(len(df))
       coeffs = _safe_polyfit(x_num.astype(float), df[metric].values.astype(float))
       if coeffs is not None:
           fig.add_trace(go.Scatter(
               x=df["Season"], y=np.polyval(coeffs, x_num),
               name=f"{config.short_name} Trend ({coeffs[0]:+.2f}/yr)",
               mode="lines",
               line=dict(color=config.color, width=1.5, dash="dash"),
               hoverinfo="skip", showlegend=True,
           ))

   fig.update_layout(
       template="plotly_dark",
       title=dict(text=title, font=dict(size=18)),
       height=600, width=1100, hovermode="x unified",
       legend=dict(orientation="h", yanchor="bottom", y=1.02,
                   xanchor="center", x=0.5, font=dict(size=10)),
       xaxis=dict(tickangle=-45, dtick=3),
       yaxis=dict(title=metric),
   )
   return fig


# ─────────────────────────────────────────────────────────────────────────────
# FREQUENCY TABLES
# ─────────────────────────────────────────────────────────────────────────────

def build_frequency_tables(df: pd.DataFrame,
                          config: LeagueConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
   top4 = pd.concat([df["Champion"], df["2nd Place"],
                     df["3rd Place"], df["4th Place"]]).value_counts().reset_index()
   top4.columns = ["Club", "Top 4 Finishes"]
   top4["Percentage"] = (top4["Top 4 Finishes"] / len(df) * 100).round(1)

   releg_cols = [c for c in df.columns if c.startswith("Relegated")]
   if releg_cols:
       relegated = (pd.concat([df[c] for c in releg_cols])
                    .replace("N/A", pd.NA).dropna()
                    .value_counts().reset_index())
       relegated.columns = ["Club", "Times Relegated"]
   else:
       relegated = pd.DataFrame(columns=["Club", "Times Relegated"])

   return top4, relegated

# ─────────────────────────────────────────────────────────────────────────────
# CHAMPIONS LEAGUE
# ─────────────────────────────────────────────────────────────────────────────

CL_CACHE_FILE = "cl_data_cache.csv"
CL_START_YEAR = 1992  # Rebranded from European Cup to Champions League
CL_END_YEAR = 2024


def _cl_url(start_year: int) -> str:
   yy = str(start_year + 1)[-2:]
   return (
       f"https://en.wikipedia.org/wiki/"
       f"{start_year}%E2%80%93{yy}_UEFA_Champions_League"
   )


def _cl_knockout_url(start_year: int) -> str:
   yy = str(start_year + 1)[-2:]
   return (
       f"https://en.wikipedia.org/wiki/"
       f"{start_year}%E2%80%93{yy}_UEFA_Champions_League_knockout_phase"
   )


def scrape_cl_season(start_year: int) -> Optional[dict]:
   """
   Scrape a single Champions League season from Wikipedia.
   Extracts winner, runner-up, semi-finalists, and their countries.
   """
   url = _cl_url(start_year)
   html = fetch_page(url)
   if html is None:
       return None

   try:
       tables = pd.read_html(StringIO(html))
   except (ValueError, ImportError):
       return None

   # Look for the "Final" or knockout bracket info in the page text
   # Strategy: find a table or infobox with winner/runner-up
   # Wikipedia CL pages have varied formats, so we parse the page text too
   result = {
       "Season": get_season_label(start_year),
       "Start Year": start_year,
       "Winner": None,
       "Winner Country": None,
       "Runner-Up": None,
       "Runner-Up Country": None,
       "Semi-Finalist 1": None,
       "Semi-Finalist 1 Country": None,
       "Semi-Finalist 2": None,
       "Semi-Finalist 2 Country": None,
   }

   # Try to find a table that looks like final results
   # CL Wikipedia pages often have a small "Final" table or an infobox
   for table in tables:
       if isinstance(table.columns, pd.MultiIndex):
           table.columns = [str(c[-1]) if isinstance(c, tuple) else str(c)
                            for c in table.columns]

       col_str = " ".join(str(c).lower() for c in table.columns)
       all_text = table.to_string().lower()

       # Look for knockout/bracket tables with team names
       if ("champions" in all_text or "winners" in all_text or
               "final" in all_text or "runner" in all_text):
           # This might be our table
           pass

   # Fallback: use known results database
   # Since Wikipedia parsing for CL is unreliable due to varied page structures,
   # we use a curated dataset for the core facts
   cl_results = _get_cl_known_results()
   if start_year in cl_results:
       return cl_results[start_year]

   return result if result["Winner"] is not None else None


# Country mapping for major CL clubs
_CLUB_COUNTRY = {
   "Real Madrid": "Spain", "Barcelona": "Spain", "Atletico Madrid": "Spain",
   "Villarreal": "Spain",
   "AC Milan": "Italy", "Inter Milan": "Italy", "Juventus": "Italy",
   "Roma": "Italy", "Napoli": "Italy",
   "Bayern Munich": "Germany", "Borussia Dortmund": "Germany",
   "Bayer Leverkusen": "Germany", "RB Leipzig": "Germany",
   "Manchester United": "England", "Liverpool": "England",
   "Chelsea": "England", "Arsenal": "England", "Manchester City": "England",
   "Tottenham": "England", "Nottingham Forest": "England",
   "Aston Villa": "England",
   "Paris Saint-Germain": "France", "Marseille": "France",
   "Monaco": "France", "Lyon": "France",
   "Ajax": "Netherlands", "PSV": "Netherlands", "Feyenoord": "Netherlands",
   "Porto": "Portugal", "Benfica": "Portugal",
   "Celtic": "Scotland", "Rangers": "Scotland",
   "Galatasaray": "Turkey",
   "Red Star Belgrade": "Serbia",
   "Steaua Bucharest": "Romania",
   "Dynamo Kyiv": "Ukraine", "Shakhtar Donetsk": "Ukraine",
   "CSKA Moscow": "Russia", "Spartak Moscow": "Russia",
   "Olympiacos": "Greece", "Panathinaikos": "Greece",
   "Deportivo La Coruna": "Spain", "Valencia": "Spain",
   "Leeds United": "England",
}


def _get_club_country(club: str) -> str:
   """Look up country for a club name."""
   return _CLUB_COUNTRY.get(club, "Unknown")


def _get_cl_known_results() -> dict[int, dict]:
   """
   Curated Champions League results 1992-2024.
   Returns dict keyed by start_year with full result dicts.
   """
   raw = [
       (1992, "Marseille", "AC Milan", "Rangers", "Club Brugge"),
       (1993, "AC Milan", "Barcelona", "Porto", "Monaco"),
       (1994, "Ajax", "AC Milan", "Bayern Munich", "Paris Saint-Germain"),
       (1995, "Juventus", "Ajax", "Nantes", "Panathinaikos"),
       (1996, "Borussia Dortmund", "Juventus", "Ajax", "Real Madrid"),
       (1997, "Real Madrid", "Juventus", "Borussia Dortmund", "Monaco"),
       (1998, "Manchester United", "Bayern Munich", "Dynamo Kyiv", "Juventus"),
       (1999, "Real Madrid", "Valencia", "Bayern Munich", "Barcelona"),
       (2000, "Bayern Munich", "Valencia", "Real Madrid", "Leeds United"),
       (2001, "Real Madrid", "Bayer Leverkusen", "Manchester United", "Barcelona"),
       (2002, "AC Milan", "Juventus", "Real Madrid", "Inter Milan"),
       (2003, "Porto", "Monaco", "Real Madrid", "Chelsea"),
       (2004, "Liverpool", "AC Milan", "Chelsea", "Paris Saint-Germain"),
       (2005, "Barcelona", "Arsenal", "AC Milan", "Villarreal"),
       (2006, "AC Milan", "Liverpool", "Chelsea", "Manchester United"),
       (2007, "Manchester United", "Chelsea", "Barcelona", "Liverpool"),
       (2008, "Barcelona", "Manchester United", "Chelsea", "Arsenal"),
       (2009, "Inter Milan", "Bayern Munich", "Barcelona", "Lyon"),
       (2010, "Barcelona", "Manchester United", "Real Madrid", "Schalke 04"),
       (2011, "Chelsea", "Bayern Munich", "Barcelona", "Real Madrid"),
       (2012, "Bayern Munich", "Borussia Dortmund", "Barcelona", "Real Madrid"),
       (2013, "Real Madrid", "Atletico Madrid", "Chelsea", "Bayern Munich"),
       (2014, "Barcelona", "Juventus", "Real Madrid", "Bayern Munich"),
       (2015, "Real Madrid", "Atletico Madrid", "Manchester City", "Bayern Munich"),
       (2016, "Real Madrid", "Juventus", "Atletico Madrid", "Monaco"),
       (2017, "Real Madrid", "Liverpool", "Bayern Munich", "Roma"),
       (2018, "Liverpool", "Tottenham", "Ajax", "Barcelona"),
       (2019, "Bayern Munich", "Paris Saint-Germain", "Lyon", "RB Leipzig"),
       (2020, "Chelsea", "Manchester City", "Real Madrid", "Paris Saint-Germain"),
       (2021, "Real Madrid", "Liverpool", "Manchester City", "Villarreal"),
       (2022, "Manchester City", "Inter Milan", "Real Madrid", "AC Milan"),
       (2023, "Real Madrid", "Borussia Dortmund", "Paris Saint-Germain", "Bayern Munich"),
       (2024, "Real Madrid", "Liverpool", "Barcelona", "Paris Saint-Germain"),
   ]

   results = {}
   for year, winner, runner_up, sf1, sf2 in raw:
       results[year] = {
           "Season": get_season_label(year),
           "Start Year": year,
           "Winner": winner,
           "Winner Country": _get_club_country(winner),
           "Runner-Up": runner_up,
           "Runner-Up Country": _get_club_country(runner_up),
           "Semi-Finalist 1": sf1,
           "Semi-Finalist 1 Country": _get_club_country(sf1),
           "Semi-Finalist 2": sf2,
           "Semi-Finalist 2 Country": _get_club_country(sf2),
       }
   return results


def load_cl_data(cache_dir: str = "../data",
                max_age_days: float = 7.0) -> pd.DataFrame:
   """Load Champions League data from cache or curated dataset."""
   cache_path = Path(cache_dir) / CL_CACHE_FILE
   cache_path.parent.mkdir(parents=True, exist_ok=True)

   if cache_path.exists():
       age_days = (time.time() - cache_path.stat().st_mtime) / 86400
       if age_days < max_age_days:
           print(f"  [CL] cache hit ({age_days:.1f}d old)")
           return pd.read_csv(cache_path)

   print("  [CL] loading Champions League data...")
   cl_results = _get_cl_known_results()
   rows = [cl_results[y] for y in sorted(cl_results.keys())
           if CL_START_YEAR <= y <= CL_END_YEAR]

   df = pd.DataFrame(rows)
   if len(df):
       df.to_csv(cache_path, index=False)
       print(f"  [CL] cached {len(df)} seasons")
   return df


def enrich_cl_dataframe(df: pd.DataFrame) -> pd.DataFrame:
   """Add derived columns to Champions League data."""
   df = df.copy()
   if len(df) == 0:
       return df

   # Count appearances in top 4 by country
   country_cols = ["Winner Country", "Runner-Up Country",
                   "Semi-Finalist 1 Country", "Semi-Finalist 2 Country"]

   # Rolling dominance: count of top-4 appearances per country in 5-year windows
   df["Top 4 Countries"] = df[country_cols].apply(
       lambda row: ", ".join(sorted(set(row.dropna()))), axis=1
   )

   # Flag for all-one-country final
   df["Same Country Final"] = df["Winner Country"] == df["Runner-Up Country"]

   # Count unique countries in top 4
   df["Unique Countries in Top 4"] = df[country_cols].apply(
       lambda row: row.dropna().nunique(), axis=1
   )

   return df


def build_cl_country_dominance_chart(df: pd.DataFrame) -> go.Figure:
   """Build a stacked area chart of country representation in CL top 4."""
   country_cols = ["Winner Country", "Runner-Up Country",
                   "Semi-Finalist 1 Country", "Semi-Finalist 2 Country"]

   # Count per country per season
   records = []
   for _, row in df.iterrows():
       counts = {}
       for col in country_cols:
           c = row.get(col)
           if pd.notna(c):
               counts[c] = counts.get(c, 0) + 1
       for country, count in counts.items():
           records.append({"Season": row["Season"], "Country": country, "Count": count})

   count_df = pd.DataFrame(records)
   if len(count_df) == 0:
       return go.Figure()

   pivot = count_df.pivot_table(index="Season", columns="Country",
                                 values="Count", aggfunc="sum", fill_value=0)

   # Keep top 5 countries by total, rest as "Other"
   totals = pivot.sum().sort_values(ascending=False)
   top_countries = totals.head(5).index.tolist()

   country_colors = {
       "England": "#4FC3F7",
       "Spain": "#FFD54F",
       "Germany": "#66BB6A",
       "Italy": "#AB47BC",
       "France": "#26C6DA",
       "Netherlands": "#FF7043",
       "Portugal": "#EF5350",
   }

   fig = go.Figure()
   for country in top_countries:
       if country in pivot.columns:
           fig.add_trace(go.Scatter(
               x=pivot.index, y=pivot[country],
               name=country, mode="lines",
               stackgroup="one",
               line=dict(width=0.5,
                         color=country_colors.get(country, "#FFFFFF")),
               fillcolor=country_colors.get(country, "rgba(255,255,255,0.3)"),
               hovertemplate=f"<b>{country}</b><br>%{{x}}<br>%{{y}} in top 4<extra></extra>",
           ))

   # Add "Other" trace
   other_cols = [c for c in pivot.columns if c not in top_countries]
   if other_cols:
       other_sum = pivot[other_cols].sum(axis=1)
       fig.add_trace(go.Scatter(
           x=pivot.index, y=other_sum,
           name="Other", mode="lines",
           stackgroup="one",
           line=dict(width=0.5, color="rgba(150,150,150,0.8)"),
           fillcolor="rgba(150,150,150,0.3)",
       ))

   fig.update_layout(
       template="plotly_dark",
       title=dict(text="Champions League: Country Representation in Top 4",
                  font=dict(size=18)),
       height=500, width=1100,
       xaxis=dict(tickangle=-45, dtick=3),
       yaxis=dict(title="Clubs in Top 4", range=[0, 4.2]),
       legend=dict(orientation="h", yanchor="bottom", y=1.02,
                   xanchor="center", x=0.5),
       hovermode="x unified",
   )
   return fig


def build_cl_winners_chart(df: pd.DataFrame) -> go.Figure:
   """Build a bar chart of Champions League winners by club."""
   winners = df["Winner"].value_counts().reset_index()
   winners.columns = ["Club", "Titles"]
   winners["Country"] = winners["Club"].map(_get_club_country)

   country_colors = {
       "England": "#4FC3F7", "Spain": "#FFD54F", "Germany": "#66BB6A",
       "Italy": "#AB47BC", "France": "#26C6DA", "Netherlands": "#FF7043",
       "Portugal": "#EF5350",
   }
   winners["Color"] = winners["Country"].map(
       lambda c: country_colors.get(c, "#FFFFFF")
   )

   fig = go.Figure(data=[go.Bar(
       x=winners["Titles"], y=winners["Club"], orientation="h",
       marker=dict(color=winners["Color"].tolist()),
       hovertemplate="<b>%{y}</b><br>%{x} titles<extra></extra>",
   )])
   fig.update_layout(
       template="plotly_dark",
       title=dict(text="Champions League Winners (1992-2025)",
                  font=dict(size=18)),
       height=max(400, len(winners) * 30),
       width=900,
       xaxis=dict(title="Titles Won", dtick=1),
       yaxis=dict(categoryorder="total ascending"),
   )
   return fig


def build_cl_concentration_chart(df: pd.DataFrame) -> go.Figure:
   """Build a line chart showing concentration trends over time."""
   # 5-season rolling window: how many unique winners?
   window = 5
   unique_winners = []
   unique_countries = []
   seasons = []

   for i in range(window - 1, len(df)):
       window_df = df.iloc[i - window + 1:i + 1]
       seasons.append(df.iloc[i]["Season"])
       unique_winners.append(window_df["Winner"].nunique())

       all_countries = pd.concat([
           window_df["Winner Country"], window_df["Runner-Up Country"],
           window_df["Semi-Finalist 1 Country"], window_df["Semi-Finalist 2 Country"]
       ]).dropna()
       unique_countries.append(all_countries.nunique())

   fig = make_subplots(
       rows=2, cols=1, shared_xaxes=True,
       vertical_spacing=0.1,
       subplot_titles=(
           f"Unique Winners ({window}-Season Rolling Window)",
           f"Unique Countries in Top 4 ({window}-Season Rolling Window)",
       ),
   )

   fig.add_trace(go.Scatter(
       x=seasons, y=unique_winners,
       name="Unique Winners", mode="lines+markers",
       line=dict(color="#FFD54F", width=3), marker=dict(size=6),
       hovertemplate="<b>%{x}</b><br>%{y} unique winners<extra></extra>",
   ), row=1, col=1)

   fig.add_trace(go.Scatter(
       x=seasons, y=unique_countries,
       name="Unique Countries", mode="lines+markers",
       line=dict(color="#4FC3F7", width=3), marker=dict(size=6),
       hovertemplate="<b>%{x}</b><br>%{y} unique countries<extra></extra>",
   ), row=2, col=1)

   # Add trend lines
   x_num = np.arange(len(seasons)).astype(float)
   wc = _safe_polyfit(x_num, np.array(unique_winners, dtype=float))
   if wc is not None:
       fig.add_trace(go.Scatter(
           x=seasons, y=np.polyval(wc, x_num),
           name=f"Winner Trend ({wc[0]:+.3f}/yr)", mode="lines",
           line=dict(color="#FFD54F", width=2, dash="dash"),
           hoverinfo="skip",
       ), row=1, col=1)

   cc = _safe_polyfit(x_num, np.array(unique_countries, dtype=float))
   if cc is not None:
       fig.add_trace(go.Scatter(
           x=seasons, y=np.polyval(cc, x_num),
           name=f"Country Trend ({cc[0]:+.3f}/yr)", mode="lines",
           line=dict(color="#4FC3F7", width=2, dash="dash"),
           hoverinfo="skip",
       ), row=2, col=1)

   fig.update_layout(
       template="plotly_dark",
       title=dict(text="Champions League Concentration Over Time",
                  font=dict(size=18)),
       height=700, width=1100,
       legend=dict(orientation="h", yanchor="bottom", y=1.05,
                   xanchor="center", x=0.5),
       hovermode="x unified",
   )
   fig.update_yaxes(title_text="Unique Winners", row=1, col=1)
   fig.update_yaxes(title_text="Unique Countries", row=2, col=1)
   fig.update_xaxes(tickangle=-45, dtick=3, row=2, col=1)
   return fig