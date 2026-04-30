"""
Shared utilities for European football league analysis.

Provides scraping, caching, cleaning, and charting functions
used across all league-specific notebooks and the comparison page.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots
import io


# ─────────────────────────────────────────────────────────────────────────────
# LEAGUE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LeagueConfig:
   """Configuration for a single football league."""

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
   season_format: str = "slash"
   cache_file: str = ""
   url_patterns: list = field(default_factory=list)


LEAGUES = {
   "premier_league": LeagueConfig(
       name="Premier League",
       short_name="PL",
       country="England",
       color="#4FC3F7",
       color_secondary="#EF5350",
       start_year=1992,
       end_year=2024,
       num_teams=20,
       num_teams_history={range(1992, 1995): 22},
       relegation_count=3,
       cl_spots=4,
       cache_file="pl_data_cache.csv",
       url_patterns=[
           # Pre-2007: "FA Premier League"
           lambda y: (
               f"https://en.wikipedia.org/wiki/"
               f"{y}\u2013{str(y+1)[-2:]}_FA_Premier_League"
           )
           if y <= 2006
           else (
               f"https://en.wikipedia.org/wiki/"
               f"{y}\u2013{str(y+1)[-2:]}_Premier_League"
           ),
       ],
   ),
   "bundesliga": LeagueConfig(
       name="Bundesliga",
       short_name="BL",
       country="Germany",
       color="#66BB6A",
       color_secondary="#FF7043",
       start_year=1992,
       end_year=2024,
       num_teams=18,
       num_teams_history={},
       relegation_count=3,  # 2 direct + 1 playoff; track all 3
       cl_spots=4,
       cache_file="bundesliga_data_cache.csv",
       url_patterns=[
           lambda y: (
               f"https://en.wikipedia.org/wiki/"
               f"{y}\u2013{str(y+1)[-2:]}_Bundesliga"
           ),
       ],
   ),
   "la_liga": LeagueConfig(
       name="La Liga",
       short_name="LL",
       country="Spain",
       color="#FFD54F",
       color_secondary="#AB47BC",
       start_year=1992,
       end_year=2024,
       num_teams=20,
       num_teams_history={range(1995, 1997): 22},
       relegation_count=3,
       cl_spots=4,
       cache_file="la_liga_data_cache.csv",
       url_patterns=[
           lambda y: (
               f"https://en.wikipedia.org/wiki/"
               f"{y}\u2013{str(y+1)[-2:]}_La_Liga"
           ),
           # Fallback: older naming
           lambda y: (
               f"https://en.wikipedia.org/wiki/"
               f"{y}\u201399_La_Liga"
               if str(y + 1)[-2:] == "99"
               else f"https://en.wikipedia.org/wiki/"
               f"{y}\u2013{str(y+1)[-2:]}_La_Liga_season"
           ),
       ],
   ),
   "ligue_1": LeagueConfig(
       name="Ligue 1",
       short_name="L1",
       country="France",
       color="#26C6DA",
       color_secondary="#FF8A65",
       start_year=1992,
       end_year=2024,
       num_teams=18,
       num_teams_history={range(1992, 2023): 20},
       relegation_count=3,  # 2 direct + 1 playoff currently; 3 direct historically
       cl_spots=3,
       cache_file="ligue1_data_cache.csv",
       url_patterns=[
           # Post-2002: "Ligue 1"
           lambda y: (
               f"https://en.wikipedia.org/wiki/"
               f"{y}\u2013{str(y+1)[-2:]}_Ligue_1"
           ),
           # Pre-2002: "French Division 1" or "Division 1"
           lambda y: (
               f"https://en.wikipedia.org/wiki/"
               f"{y}\u2013{str(y+1)[-2:]}_French_Division_1"
           ),
           lambda y: (
               f"https://en.wikipedia.org/wiki/"
               f"{y}\u2013{str(y+1)[-2:]}_Division_1_(French_football)"
           ),
       ],
   ),
   "serie_a": LeagueConfig(
       name="Serie A",
       short_name="SA",
       country="Italy",
       color="#AB47BC",
       color_secondary="#8D6E63",
       start_year=1992,
       end_year=2024,
       num_teams=20,
       num_teams_history={range(1992, 2004): 18},
       relegation_count=3,
       cl_spots=4,
       cache_file="serie_a_data_cache.csv",
       url_patterns=[
           lambda y: (
               f"https://en.wikipedia.org/wiki/"
               f"{y}\u2013{str(y+1)[-2:]}_Serie_A"
           ),
           # Some older seasons use "Serie A (football)" or similar
           lambda y: (
               f"https://en.wikipedia.org/wiki/"
               f"{y}\u2013{str(y+1)[-2:]}_Serie_A_(football)"
           ),
       ],
   ),
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def get_season_label(start_year: int, fmt: str = "slash") -> str:
   """Generate a human-readable season label like '2024/25'."""
   end_short = str(start_year + 1)[-2:]
   if fmt == "slash":
       return f"{start_year}/{end_short}"
   return f"{start_year}-{end_short}"


def get_num_teams_for_season(config: LeagueConfig, start_year: int) -> int:
   """Get the number of teams in a league for a specific season."""
   for year_range, num in config.num_teams_history.items():
       if start_year in year_range:
           return num
   return config.num_teams


def get_matches_per_season(num_teams: int) -> int:
   """Calculate matches per team in a round-robin league."""
   return (num_teams - 1) * 2


# ─────────────────────────────────────────────────────────────────────────────
# TABLE PARSING
# ─────────────────────────────────────────────────────────────────────────────


def clean_team_name(name: str) -> str:
   """Remove Wikipedia footnote markers, qualification tags, and artifacts."""
   if not isinstance(name, str):
       return str(name).strip()
   # Remove parenthetical tags: (C), (R), (Q), (champions), etc.
   name = re.sub(r"\s*\([A-Za-z,. ]+\)", "", name)
   # Remove bracketed citations: [a], [1], [note 1]
   name = re.sub(r"\s*\[.*?\]", "", name)
   # Remove dagger, double-dagger, section markers, asterisks
   name = re.sub(r"[\*\u2020\u2021\u00a7\u00b6]", "", name)
   # Remove trailing/leading whitespace and non-breaking spaces
   name = name.replace("\u00a0", " ").strip()
   return name


def find_league_table(
   tables: list[pd.DataFrame], num_teams: int
) -> Optional[pd.DataFrame]:
   """
   Identify the league standings table from a list of HTML tables.

   Uses a scoring system that rewards:
   - Correct row count (highest weight)
   - Presence of 'Pts'/'Points' column
   - Presence of 'Team'/'Club' column
   - Presence of football-specific columns (GF, GA, GD, W, D, L)
   - Sufficient column count (standings tables typically have 8+ columns)
   """
   candidates = []

   for idx, table in enumerate(tables):
       # Flatten MultiIndex columns if present
       if isinstance(table.columns, pd.MultiIndex):
           table = table.copy()
           table.columns = [
               str(c[-1]) if isinstance(c, tuple) else str(c)
               for c in table.columns
           ]

       col_str = " ".join(str(c).lower() for c in table.columns)
       score = 0

       # Row count matching (highest priority)
       if len(table) == num_teams:
           score += 15
       elif abs(len(table) - num_teams) == 1:
           score += 8  # Off by one (header row parsed as data, etc.)
       elif abs(len(table) - num_teams) <= 3:
           score += 3

       # Column name checks
       if "pts" in col_str or "points" in col_str:
           score += 8
       if "team" in col_str or "club" in col_str or "squad" in col_str:
           score += 5
       if any(x in col_str for x in [" w ", " d ", " l ", "won", "drawn", "lost"]):
           score += 4
       if any(x in col_str for x in ["gf", "ga", "gd", "goals for", "goals against"]):
           score += 3
       if "pos" in col_str or "#" in col_str:
           score += 2

       # Sufficient columns for a standings table
       if table.shape[1] >= 8:
           score += 2
       elif table.shape[1] >= 6:
           score += 1

       if score >= 12:
           candidates.append((score, idx, table.copy()))

   if candidates:
       candidates.sort(key=lambda x: x[0], reverse=True)
       return candidates[0][2]

   return None


def identify_columns(table: pd.DataFrame) -> tuple[str, str]:
   """
   Identify the team name column and points column.

   Returns (team_col, pts_col).
   """
   cols_lower = {}
   for c in table.columns:
       key = str(c).lower().strip()
       if key not in cols_lower:  # Take first match to avoid duplicates
           cols_lower[key] = c

   # Find points column
   pts_col = None
   for candidate in ["pts", "points", "pt", "p"]:
       if candidate in cols_lower:
           pts_col = cols_lower[candidate]
           break

   # Find team column
   team_col = None
   for candidate in ["team", "club", "squad"]:
       if candidate in cols_lower:
           team_col = cols_lower[candidate]
           break

   # Fallback: team is usually column 1, pts is last or second-to-last
   if team_col is None:
       team_col = table.columns[1] if len(table.columns) > 1 else table.columns[0]

   if pts_col is None:
       # Try last column, then second-to-last
       for offset in [-1, -2]:
           candidate_col = table.columns[offset]
           if pd.to_numeric(table[candidate_col], errors="coerce").notna().sum() > 0:
               pts_col = candidate_col
               break
       if pts_col is None:
           pts_col = table.columns[-1]

   return team_col, pts_col


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPING
# ─────────────────────────────────────────────────────────────────────────────


def fetch_page(url: str) -> Optional[str]:
   """Fetch a Wikipedia page with proper headers. Returns HTML or None."""
   try:
       resp = requests.get(
           url,
           headers={"User-Agent": "Euro-Football-Portfolio/1.0 (educational use)"},
           timeout=15,
       )
       if resp.status_code == 200:
           return resp.text
       return None
   except requests.RequestException:
       return None


def scrape_season(
   league_key: str, start_year: int, config: LeagueConfig
) -> Optional[dict]:
   """
   Scrape a single season's standings for a given league.

   Tries multiple URL patterns if the first fails.
   Returns a dict with all extracted data, or None on failure.
   """
   num_teams = get_num_teams_for_season(config, start_year)
   matches = get_matches_per_season(num_teams)

   # Try each URL pattern until one works
   html = None
   for pattern_fn in config.url_patterns:
       url = pattern_fn(start_year)
       html = fetch_page(url)
       if html is not None:
           break

   if html is None:
       print(f"    ERROR: All URL patterns failed for {start_year}/{start_year+1}")
       return None

   # Parse HTML tables
    try:
       # Wrap the html string in StringIO to force pandas to treat it as data
       tables = pd.read_html(io.StringIO(html)) 
   except ValueError:
       print(f"    ERROR: No tables found for {start_year}/{start_year+1}")
       return None

   if not tables:
       print(f"    ERROR: Empty table list for {start_year}/{start_year+1}")
       return None

   # Find the standings table
   table = find_league_table(tables, num_teams)
   if table is None:
       print(
           f"    WARNING: Could not identify standings table for "
           f"{start_year}/{start_year+1} (expected {num_teams} teams, "
           f"found tables with rows: {[len(t) for t in tables[:10]]})"
       )
       return None

   # Identify columns
   team_col, pts_col = identify_columns(table)

   # Clean team names
   table[team_col] = table[team_col].apply(clean_team_name)

   # Convert points to numeric, drop non-numeric rows
   table[pts_col] = pd.to_numeric(table[pts_col], errors="coerce")
   table = table.dropna(subset=[pts_col]).reset_index(drop=True)

   # Validate we have enough rows
   if len(table) < num_teams - 2:
       print(
           f"    WARNING: Only {len(table)} valid rows for "
           f"{start_year}/{start_year+1} (expected {num_teams})"
       )
       return None

   # Sort by points descending (should already be sorted, but ensure it)
   table = table.sort_values(pts_col, ascending=False).reset_index(drop=True)

   # Determine positions
   actual_teams = len(table)
   survival_pos = actual_teams - config.relegation_count - 1
   relegated_start = actual_teams - config.relegation_count

   # Safety check
   if survival_pos < 4 or relegated_start >= actual_teams:
       print(
           f"    WARNING: Position calculation error for {start_year}/{start_year+1}"
       )
       return None

   season_label = get_season_label(start_year, config.season_format)

   result = {
       "Season": season_label,
       "Start Year": start_year,
       "Num Teams": actual_teams,
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

   # Add relegated teams
   for j in range(config.relegation_count):
       idx = relegated_start + j
       if idx < actual_teams:
           result[f"Relegated {j + 1}"] = table.iloc[idx][team_col]
       else:
           result[f"Relegated {j + 1}"] = "N/A"

   return result


def validate_dataframe(df: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
   """
   Validate and clean a scraped DataFrame.

   Removes rows with clearly invalid data (e.g., negative points,
   survival points higher than title points).
   """
   df = df.copy()

   # Remove rows where title points <= 0 or survival points <= 0
   mask_valid = (
       (df["Title-Winning Points"] > 0)
       & (df["Relegation Survival Points"] > 0)
       & (df["Title-Winning Points"] > df["Relegation Survival Points"])
   )

   invalid_count = (~mask_valid).sum()
   if invalid_count > 0:
       print(f"    Removed {invalid_count} invalid row(s)")
       df = df[mask_valid].reset_index(drop=True)

   return df


def scrape_league(
   league_key: str,
   config: Optional[LeagueConfig] = None,
   cache_dir: str = ".",
   max_age_days: float = 7.0,
   delay: float = 1.5,
) -> pd.DataFrame:
   """
   Scrape all seasons for a league, with local CSV caching.

   Parameters
   ----------
   league_key : str
       Key into the LEAGUES dict
   config : LeagueConfig, optional
       Override the default config
   cache_dir : str
       Directory for cache files
   max_age_days : float
       Maximum cache age before re-scraping
   delay : float
       Seconds between requests (be polite to Wikipedia)

   Returns
   -------
   pd.DataFrame with one row per season
   """
   if config is None:
       config = LEAGUES[league_key]

   cache_path = Path(cache_dir) / config.cache_file

   # Check cache
   if cache_path.exists():
       age_days = (time.time() - cache_path.stat().st_mtime) / 86400
       if age_days < max_age_days:
           print(f"  [{config.short_name}] Using cached data ({age_days:.1f} days old)")
           return pd.read_csv(cache_path)
       else:
           print(f"  [{config.short_name}] Cache is {age_days:.0f} days old, refreshing...")

   # Scrape
   print(
       f"  [{config.short_name}] Scraping {config.name} "
       f"({config.start_year}/{config.start_year + 1} to "
       f"{config.end_year}/{config.end_year + 1})..."
   )

   seasons = []
   failures = []

   for year in range(config.start_year, config.end_year + 1):
       print(f"    {year}/{year + 1}...", end=" ", flush=True)
       result = scrape_season(league_key, year, config)
       if result:
           seasons.append(result)
           print("OK")
       else:
           failures.append(year)
           print("FAILED")
       time.sleep(delay)

   df = pd.DataFrame(seasons)

   if len(df) > 0:
       df = validate_dataframe(df, config)
       df.to_csv(cache_path, index=False)
       print(
           f"  [{config.short_name}] Done: {len(df)} seasons cached. "
           f"Failures: {len(failures)}"
       )
       if failures:
           print(f"    Failed years: {failures}")
   else:
       print(f"  [{config.short_name}] ERROR: No data scraped!")

   return df


# ─────────────────────────────────────────────────────────────────────────────
# DATA ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
   """Add computed columns to a league DataFrame."""
   df = df.copy()

   # Basic gap and ratio
   df["Gap"] = df["Title-Winning Points"] - df["Relegation Survival Points"]
   df["Ratio"] = (
       df["Title-Winning Points"] / df["Relegation Survival Points"]
   ).round(2)

   # Points per game (for cross-league normalization)
   if "Matches" in df.columns:
       df["Title PPG"] = (df["Title-Winning Points"] / df["Matches"]).round(3)
       df["Survival PPG"] = (
           df["Relegation Survival Points"] / df["Matches"]
       ).round(3)
       df["Gap PPG"] = (df["Title PPG"] - df["Survival PPG"]).round(3)

       # Normalize to 38-game equivalent for comparison
       df["Title Pts (38-game)"] = (df["Title PPG"] * 38).round(1)
       df["Survival Pts (38-game)"] = (df["Survival PPG"] * 38).round(1)
       df["Gap (38-game)"] = (df["Gap PPG"] * 38).round(1)

   return df


# ─────────────────────────────────────────────────────────────────────────────
# CHARTING: SINGLE LEAGUE
# ─────────────────────────────────────────────────────────────────────────────


def build_league_chart(
   df: pd.DataFrame,
   config: LeagueConfig,
   eras: Optional[list[dict]] = None,
   outliers: Optional[list[dict]] = None,
) -> go.Figure:
   """
   Build the full 3-panel interactive chart for a single league.

   Parameters
   ----------
   df : pd.DataFrame
       League data with computed Gap and Ratio columns
   config : LeagueConfig
       League configuration for colors and labels
   eras : list of dicts, optional
       Era definitions with keys: label, start, end, title.
       If None, uses full history only.
   outliers : list of dicts, optional
       Notable season annotations with keys: season, y, text, color, ax, ay.

   Returns
   -------
   plotly.graph_objects.Figure
   """
   if eras is None:
       eras = [
           {
               "label": f"Full History ({config.start_year}-{config.end_year + 1})",
               "start": df["Season"].iloc[0],
               "end": df["Season"].iloc[-1],
               "title": (
                   f"{config.name}: Title vs. Relegation Points "
                   f"({config.start_year}-{config.end_year + 1})"
               ),
           }
       ]

   if outliers is None:
       outliers = []

   fig = make_subplots(
       rows=3,
       cols=1,
       shared_xaxes=True,
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

   for i, era in enumerate(eras):
       # Slice data to this era
       start_mask = df["Season"] == era["start"]
       end_mask = df["Season"] == era["end"]

       if not start_mask.any() or not end_mask.any():
           print(f"    WARNING: Era '{era['label']}' has invalid bounds, skipping")
           # Add empty traces to maintain indexing
           for _ in range(TRACES_PER_ERA):
               fig.add_trace(
                   go.Scatter(x=[], y=[], visible=False, showlegend=False),
                   row=1,
                   col=1,
               )
           era_annotations[i] = []
           continue

       start_idx = df[start_mask].index[0]
       end_idx = df[end_mask].index[0]
       era_df = df.iloc[start_idx : end_idx + 1].reset_index(drop=True)
       x_num = np.arange(len(era_df))

       # Skip eras with too few data points for trend calculation
       if len(era_df) < 3:
           for _ in range(TRACES_PER_ERA):
               fig.add_trace(
                   go.Scatter(x=[], y=[], visible=False, showlegend=False),
                   row=1,
                   col=1,
               )
           era_annotations[i] = []
           continue

       # ── Build hover text ──

       title_hover = []
       for _, row in era_df.iterrows():
           title_hover.append(
               f"<b>{row['Season']}</b><br>"
               f"\U0001F3C6 {row['Champion']}: {row['Title-Winning Points']} pts<br>"
               f"\U0001F948 {row['2nd Place']}: {int(row['2nd Place Points'])} pts<br>"
               f"\U0001F949 {row['3rd Place']}: {int(row['3rd Place Points'])} pts<br>"
               f"4th: {row['4th Place']}: {int(row['4th Place Points'])} pts"
           )

       releg_cols = [c for c in era_df.columns if c.startswith("Relegated")]
       releg_hover = []
       for _, row in era_df.iterrows():
           relegated_names = [
               str(row[c]) for c in releg_cols if pd.notna(row.get(c, None))
           ]
           relegated_str = " | ".join(relegated_names)
           releg_hover.append(
               f"<b>{row['Season']}</b><br>"
               f"\U0001FA82 Survived: {row['Survived Relegation']} "
               f"({int(row['Relegation Survival Points'])} pts)<br>"
               f"\u274C {relegated_str}"
           )

       # ── Trace 0: Title-Winning Points ──
       fig.add_trace(
           go.Scatter(
               x=era_df["Season"],
               y=era_df["Title-Winning Points"],
               name="Title-Winning Points",
               mode="lines+markers",
               line=dict(color=config.color, width=3),
               marker=dict(size=7),
               hovertext=title_hover,
               hoverinfo="text",
               visible=False,
               showlegend=True,
           ),
           row=1,
           col=1,
       )

       # ── Trace 1: Title Trend ──
       t_c = np.polyfit(x_num, era_df["Title-Winning Points"].values, 1)
       t_t = np.polyval(t_c, x_num)
       fig.add_trace(
           go.Scatter(
               x=era_df["Season"],
               y=t_t,
               name=f"Title Trend ({t_c[0]:+.2f} pts/yr)",
               mode="lines",
               line=dict(color=config.color, width=2, dash="dash"),
               hoverinfo="skip",
               visible=False,
               showlegend=True,
           ),
           row=1,
           col=1,
       )

       # ── Trace 2: Relegation Survival Points ──
       fig.add_trace(
           go.Scatter(
               x=era_df["Season"],
               y=era_df["Relegation Survival Points"],
               name="Relegation Survival Points",
               mode="lines+markers",
               line=dict(color=config.color_secondary, width=3),
               marker=dict(size=7),
               hovertext=releg_hover,
               hoverinfo="text",
               visible=False,
               showlegend=True,
           ),
           row=1,
           col=1,
       )

       # ── Trace 3: Relegation Trend ──
       r_c = np.polyfit(x_num, era_df["Relegation Survival Points"].values, 1)
       r_t = np.polyval(r_c, x_num)
       fig.add_trace(
           go.Scatter(
               x=era_df["Season"],
               y=r_t,
               name=f"Relegation Trend ({r_c[0]:+.2f} pts/yr)",
               mode="lines",
               line=dict(color=config.color_secondary, width=2, dash="dash"),
               hoverinfo="skip",
               visible=False,
               showlegend=True,
           ),
           row=1,
           col=1,
       )

       # ── Trace 4: Shaded gap ──
       fig.add_trace(
           go.Scatter(
               x=era_df["Season"].tolist() + era_df["Season"].tolist()[::-1],
               y=era_df["Title-Winning Points"].tolist()
               + era_df["Relegation Survival Points"].tolist()[::-1],
               fill="toself",
               fillcolor="rgba(255, 255, 255, 0.05)",
               line=dict(width=0),
               showlegend=False,
               hoverinfo="skip",
               visible=False,
           ),
           row=1,
           col=1,
       )

       # ── Trace 5: Gap Bars ──
       gap_values = era_df["Gap"].values
       gap_colors = [
           f"rgba("
           f"{min(255, int(150 + (g - 30) * 3))}, "
           f"{max(80, int(220 - (g - 30) * 4))}, "
           f"100, 0.85)"
           for g in gap_values
       ]
       fig.add_trace(
           go.Bar(
               x=era_df["Season"],
               y=era_df["Gap"],
               name="Points Gap",
               marker=dict(color=gap_colors, line=dict(width=0)),
               hovertemplate="<b>%{x}</b><br>Gap: %{y} pts<extra></extra>",
               visible=False,
               showlegend=True,
           ),
           row=2,
           col=1,
       )

       # ── Trace 6: Gap Trend ──
       g_c = np.polyfit(x_num, gap_values, 1)
       g_t = np.polyval(g_c, x_num)
       fig.add_trace(
           go.Scatter(
               x=era_df["Season"],
               y=g_t,
               name=f"Gap Trend ({g_c[0]:+.2f} pts/yr)",
               mode="lines",
               line=dict(color="#FFD54F", width=2, dash="dash"),
               hoverinfo="skip",
               visible=False,
               showlegend=True,
           ),
           row=2,
           col=1,
       )

       # ── Trace 7: Ratio Line ──
       fig.add_trace(
           go.Scatter(
               x=era_df["Season"],
               y=era_df["Ratio"],
               name="Points Ratio",
               mode="lines+markers",
               line=dict(color="#AB47BC", width=3),
               marker=dict(size=6),
               hovertemplate="<b>%{x}</b><br>Ratio: %{y:.2f}x<extra></extra>",
               visible=False,
               showlegend=True,
           ),
           row=3,
           col=1,
       )

       # ── Trace 8: Ratio Trend ──
       rt_c = np.polyfit(x_num, era_df["Ratio"].values, 1)
       rt_t = np.polyval(rt_c, x_num)
       fig.add_trace(
           go.Scatter(
               x=era_df["Season"],
               y=rt_t,
               name=f"Ratio Trend ({rt_c[0]:+.3f}/yr)",
               mode="lines",
               line=dict(color="#AB47BC", width=2, dash="dash"),
               hoverinfo="skip",
               visible=False,
               showlegend=True,
           ),
           row=3,
           col=1,
       )

       # ── Annotations for this era ──
       ann_for_era = []
       for outlier in outliers:
           if outlier.get("y") is not None and outlier["season"] in era_df["Season"].values:
               ann_for_era.append(
                   dict(
                       x=outlier["season"],
                       y=outlier["y"],
                       text=outlier["text"],
                       showarrow=True,
                       arrowhead=2,
                       ax=outlier.get("ax", 0),
                       ay=outlier.get("ay", -25),
                       font=dict(
                           color=outlier.get("color", "#FFFFFF"),
                           size=10,
                           family="Arial Black",
                       ),
                       xref="x",
                       yref="y",
                   )
               )
       era_annotations[i] = ann_for_era

   # ── Set first era visible ──
   for j in range(TRACES_PER_ERA):
       if j < len(fig.data):
           fig.data[j].visible = True

   # ── 2.5x reference line on ratio subplot ──
   fig.add_shape(
       type="line",
       x0=0,
       x1=1,
       y0=2.5,
       y1=2.5,
       xref="x3 domain",
       yref="y3",
       line=dict(dash="dot", color="rgba(255,255,255,0.3)"),
   )

   hline_label = dict(
       x=1.01,
       y=2.5,
       xref="x3 domain",
       yref="y3",
       text="2.5x",
       showarrow=False,
       font=dict(color="rgba(255,255,255,0.5)", size=11),
       xanchor="left",
   )
   for i in era_annotations:
       era_annotations[i].append(hline_label)

   # ── Dropdown buttons ──
   total_traces = TRACES_PER_ERA * len(eras)
   dropdown_buttons = []
   for i, era in enumerate(eras):
       vis = [False] * total_traces
       for j in range(TRACES_PER_ERA):
           trace_idx = i * TRACES_PER_ERA + j
           if trace_idx < total_traces:
               vis[trace_idx] = True
       dropdown_buttons.append(
           dict(
               label=era["label"],
               method="update",
               args=[
                   {"visible": vis},
                   {
                       "title.text": era["title"],
                       "annotations": era_annotations.get(i, []),
                   },
               ],
           )
       )

   # ── Layout ──
   menu_config = []
   if len(eras) > 1:
       menu_config = [
           dict(
               type="dropdown",
               direction="down",
               active=0,
               x=0.0,
               xanchor="left",
               y=1.18,
               yanchor="top",
               bgcolor="rgba(50, 50, 50, 0.9)",
               bordercolor="rgba(255, 255, 255, 0.3)",
               font=dict(color="white", size=12),
               buttons=dropdown_buttons,
           )
       ]

   fig.update_layout(
       template="plotly_dark",
       title=dict(text=eras[0]["title"], font=dict(size=18)),
       height=1000,
       width=1100,
       hovermode="closest",
       legend=dict(
           orientation="h",
           yanchor="bottom",
           y=1.05,
           xanchor="center",
           x=0.5,
           font=dict(size=10),
       ),
       margin=dict(t=150),
       annotations=era_annotations.get(0, []),
       updatemenus=menu_config,
   )

   fig.update_yaxes(title_text="Points", row=1, col=1)
   fig.update_yaxes(title_text="Gap (pts)", row=2, col=1)
   fig.update_yaxes(title_text="Ratio", row=3, col=1)
   fig.update_xaxes(tickangle=-45, dtick=3, row=3, col=1)

   return fig


# ─────────────────────────────────────────────────────────────────────────────
# CHARTING: CROSS-LEAGUE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────


def build_comparison_chart(
   league_data: dict[str, pd.DataFrame],
   metric: str = "Gap (38-game)",
   title: str = "Cross-League Comparison",
) -> go.Figure:
   """
   Build a single-panel chart overlaying a metric across all leagues.

   Parameters
   ----------
   league_data : dict
       Mapping of league_key to enriched DataFrame
   metric : str
       Column name to plot on y-axis
   title : str
       Chart title

   Returns
   -------
   plotly.graph_objects.Figure
   """
   fig = go.Figure()

   for league_key, df in league_data.items():
       config = LEAGUES[league_key]
       if metric not in df.columns:
           continue

       fig.add_trace(
           go.Scatter(
               x=df["Season"],
               y=df[metric],
               name=config.name,
               mode="lines+markers",
               line=dict(color=config.color, width=2.5),
               marker=dict(size=5),
               hovertemplate=(
                   f"<b>{config.name}</b><br>"
                   f"%{{x}}<br>{metric}: %{{y:.1f}}<extra></extra>"
               ),
           )
       )

       # Trend line
       x_num = np.arange(len(df))
       valid_mask = df[metric].notna()
       if valid_mask.sum() >= 3:
           coeffs = np.polyfit(
               x_num[valid_mask], df.loc[valid_mask, metric].values, 1
           )
           trend = np.polyval(coeffs, x_num)
           fig.add_trace(
               go.Scatter(
                   x=df["Season"],
                   y=trend,
                   name=f"{config.short_name} Trend ({coeffs[0]:+.2f}/yr)",
                   mode="lines",
                   line=dict(color=config.color, width=1.5, dash="dash"),
                   hoverinfo="skip",
                   showlegend=True,
               )
           )

   fig.update_layout(
       template="plotly_dark",
       title=dict(text=title, font=dict(size=18)),
       height=600,
       width=1100,
       hovermode="x unified",
       legend=dict(
           orientation="h",
           yanchor="bottom",
           y=1.02,
           xanchor="center",
           x=0.5,
           font=dict(size=10),
       ),
       xaxis=dict(tickangle=-45, dtick=3),
       yaxis=dict(title=metric),
   )

   return fig


# ─────────────────────────────────────────────────────────────────────────────
# FREQUENCY TABLES
# ─────────────────────────────────────────────────────────────────────────────


def build_frequency_tables(
   df: pd.DataFrame, config: LeagueConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
   """
   Build top-4 frequency and relegation frequency DataFrames.

   Returns (top4_df, relegated_df).
   """
   # Top 4 appearances
   top4_clubs = (
       pd.concat(
           [df["Champion"], df["2nd Place"], df["3rd Place"], df["4th Place"]]
       )
       .value_counts()
       .reset_index()
   )
   top4_clubs.columns = ["Club", "Top 4 Finishes"]
   top4_clubs["Percentage"] = (
       top4_clubs["Top 4 Finishes"] / len(df) * 100
   ).round(1)

   # Relegations
   releg_cols = [c for c in df.columns if c.startswith("Relegated")]
   if releg_cols:
       relegated_clubs = (
           pd.concat([df[c] for c in releg_cols])
           .replace("N/A", pd.NA)
           .dropna()
           .value_counts()
           .reset_index()
       )
       relegated_clubs.columns = ["Club", "Times Relegated"]
   else:
       relegated_clubs = pd.DataFrame(columns=["Club", "Times Relegated"])

   return top4_clubs, relegated_clubs