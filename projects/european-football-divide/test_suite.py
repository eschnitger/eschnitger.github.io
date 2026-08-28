"""
Complete test suite for _utils.py.
No network calls: uses synthetic data and mocked HTTP responses.
Run with: python test_suite.py
"""
from __future__ import annotations

import os
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from _utils import (
   LEAGUES, LeagueConfig,
   clean_team_name, find_league_table, identify_columns,
   enrich_dataframe, build_frequency_tables, build_league_chart,
   build_comparison_chart, _safe_polyfit, get_season_label,
   get_num_teams_for_season, get_matches_per_season,
   fetch_page, scrape_season, scrape_league, validate_dataframe,
   _normalize_numeric_string,
)


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

def _fake_standings(n=20, pts_top=90):
   return pd.DataFrame({
       "Pos": list(range(1, n + 1)),
       "Team": [f"Club {i}" for i in range(n)],
       "W": [20 - i for i in range(n)],
       "D": [5] * n,
       "L": [i for i in range(n)],
       "GF": [60 - i for i in range(n)],
       "GA": [20 + i for i in range(n)],
       "GD": [40 - 2 * i for i in range(n)],
       "Pts": [pts_top - i * 3 for i in range(n)],
   })


def _fake_scorers(n=15):
   return pd.DataFrame({
       "Rank": list(range(1, n + 1)),
       "Player": [f"Player {i}" for i in range(n)],
       "Nationality": ["ENG"] * n,
       "Goals": [30 - i for i in range(n)],
   })


def _fake_enriched_seasons(start=1992, end=2025, num_teams=20) -> pd.DataFrame:
   matches = (num_teams - 1) * 2
   rows = []
   for y in range(start, end):
       rows.append({
           "Season": get_season_label(y),
           "Start Year": y,
           "Num Teams": num_teams,
           "Matches": matches,
           "Champion": "A",
           "Title-Winning Points": 85 + (y - start) * 0.25,
           "2nd Place": "B",
           "2nd Place Points": 80,
           "3rd Place": "C",
           "3rd Place Points": 75,
           "4th Place": "D",
           "4th Place Points": 70,
           "Survived Relegation": "E",
           "Relegation Survival Points": 40 - (y - start) * 0.1,
           "Relegated 1": "F",
           "Relegated 2": "G",
           "Relegated 3": "H",
       })
   return enrich_dataframe(pd.DataFrame(rows))


def _build_standings_html(num_teams: int = 20, top_pts: int = 89) -> str:
   rows = []
   for i in range(num_teams):
       pts = top_pts - i * 3
       team = f"Club {chr(65 + i)}" if i < 26 else f"Club {i}"
       rows.append(
           f"<tr><td>{i+1}</td><td>{team}</td>"
           f"<td>38</td><td>{20-i}</td><td>5</td><td>{13+i-5}</td>"
           f"<td>{60-i*2}</td><td>{25+i*2}</td><td>{35-i*4}</td>"
           f"<td>{pts}</td></tr>"
       )
   table_html = (
       "<table class='wikitable'>"
       "<tr><th>Pos</th><th>Team</th><th>Pld</th><th>W</th><th>D</th>"
       "<th>L</th><th>GF</th><th>GA</th><th>GD</th><th>Pts</th></tr>"
       + "".join(rows)
       + "</table>"
   )
   decoy = (
       "<table class='wikitable'>"
       "<tr><th>Rank</th><th>Player</th><th>Nationality</th><th>Goals</th></tr>"
       "<tr><td>1</td><td>Erling Haaland</td><td>NOR</td><td>36</td></tr>"
       "</table>"
   )
   return f"<html><body>{decoy}{table_html}</body></html>"


def _build_bad_html() -> str:
   return (
       "<html><body>"
       "<table><tr><th>Player</th><th>Nationality</th><th>Goals</th></tr>"
       "<tr><td>Someone</td><td>ENG</td><td>20</td></tr></table>"
       "</body></html>"
   )


def _build_standings_with_citations(num_teams: int = 20) -> str:
   rows = []
   for i in range(num_teams):
       pts = 90 - i * 3
       team = f"Club {chr(65 + i)}"
       if i == 0:
           team += " (C)[1]"
       elif i == 1:
           team += "\u2020[a]"
       elif i == num_teams - 1:
           team += " (R)"
       rows.append(
           f"<tr><td>{i+1}</td><td>{team}</td>"
           f"<td>38</td><td>{20-i}</td><td>5</td><td>{13+i-5}</td>"
           f"<td>{60-i*2}</td><td>{25+i*2}</td><td>{35-i*4}</td>"
           f"<td>{pts}</td></tr>"
       )
   return (
       "<html><body>"
       "<table class='wikitable'>"
       "<tr><th>Pos</th><th>Team</th><th>Pld</th><th>W</th><th>D</th>"
       "<th>L</th><th>GF</th><th>GA</th><th>GD</th><th>Pts</th></tr>"
       + "".join(rows)
       + "</table></body></html>"
   )


def _build_standings_unicode_pts(num_teams: int = 20) -> str:
   rows = []
   for i in range(num_teams):
       pts = 90 - i * 3
       gd = 35 - i * 4
       gd_str = f"\u2212{abs(gd)}" if gd < 0 else str(gd)
       rows.append(
           f"<tr><td>{i+1}</td><td>Club {chr(65 + i)}</td>"
           f"<td>38</td><td>{20-i}</td><td>5</td><td>{13+i-5}</td>"
           f"<td>{60-i*2}</td><td>{25+i*2}</td><td>{gd_str}</td>"
           f"<td>{pts}</td></tr>"
       )
   return (
       "<html><body>"
       "<table class='wikitable'>"
       "<tr><th>Pos</th><th>Team</th><th>Pld</th><th>W</th><th>D</th>"
       "<th>L</th><th>GF</th><th>GA</th><th>GD</th><th>Pts</th></tr>"
       + "".join(rows)
       + "</table></body></html>"
   )


def _make_tiny_config(start_year=2023, end_year=2023, num_teams=20,
                     cache_file="test_cache.csv") -> LeagueConfig:
   return LeagueConfig(
       name="Test League", short_name="TL", country="Test",
       color="#4FC3F7", color_secondary="#EF5350",
       start_year=start_year, end_year=end_year, num_teams=num_teams,
       cache_file=cache_file,
       url_patterns=[lambda y: f"http://example.com/{y}"],
   )


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: helpers
# ─────────────────────────────────────────────────────────────────────────────

def test_get_season_label():
   assert get_season_label(2024) == "2024/25"
   assert get_season_label(1999) == "1999/00"
   assert get_season_label(2000) == "2000/01"
   assert get_season_label(1992) == "1992/93"
   print("PASS test_get_season_label")


def test_get_matches_per_season():
   assert get_matches_per_season(20) == 38
   assert get_matches_per_season(18) == 34
   assert get_matches_per_season(22) == 42
   assert get_matches_per_season(16) == 30
   print("PASS test_get_matches_per_season")


def test_get_num_teams_for_season():
   pl = LEAGUES["premier_league"]
   assert get_num_teams_for_season(pl, 1993) == 22
   assert get_num_teams_for_season(pl, 1994) == 22
   assert get_num_teams_for_season(pl, 1995) == 20
   assert get_num_teams_for_season(pl, 2024) == 20
   l1 = LEAGUES["ligue_1"]
   assert get_num_teams_for_season(l1, 2000) == 20
   assert get_num_teams_for_season(l1, 2023) == 18
   sa = LEAGUES["serie_a"]
   assert get_num_teams_for_season(sa, 1995) == 18
   assert get_num_teams_for_season(sa, 2004) == 20
   print("PASS test_get_num_teams_for_season")


def test_league_config_integrity():
   for key, config in LEAGUES.items():
       assert config.name, f"{key}: missing name"
       assert config.short_name, f"{key}: missing short_name"
       assert config.country, f"{key}: missing country"
       assert config.color.startswith("#"), f"{key}: color not hex"
       assert config.color_secondary.startswith("#"), f"{key}: color_secondary not hex"
       assert config.start_year >= 1888, f"{key}: start_year too early"
       assert config.end_year >= config.start_year, f"{key}: end_year < start_year"
       assert config.num_teams >= 10, f"{key}: num_teams too small"
       assert config.relegation_count >= 1, f"{key}: no relegation"
       assert config.cl_spots >= 1, f"{key}: no CL spots"
       assert config.cache_file, f"{key}: no cache_file"
       assert len(config.url_patterns) >= 1, f"{key}: no url_patterns"
   print("PASS test_league_config_integrity")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: cleaning
# ─────────────────────────────────────────────────────────────────────────────

def test_clean_team_name_basic():
   assert clean_team_name("Arsenal (C)") == "Arsenal"
   assert clean_team_name("Man United[1]") == "Man United"
   assert clean_team_name("Chelsea \u2020") == "Chelsea"
   assert clean_team_name("Leicester City (champions)") == "Leicester City"
   assert clean_team_name("Tottenham\u00a0Hotspur") == "Tottenham Hotspur"
   assert clean_team_name("Liverpool [note 1]") == "Liverpool"
   print("PASS test_clean_team_name_basic")


def test_clean_team_name_multiple_markers():
   assert clean_team_name("Bayern Munich (C)[1]\u2020") == "Bayern Munich"
   assert clean_team_name("Real Madrid [a][b] (champions)") == "Real Madrid"
   print("PASS test_clean_team_name_multiple_markers")


def test_clean_team_name_edge_cases():
   assert clean_team_name(123) == "123"
   assert clean_team_name(None) == "None"
   assert clean_team_name("") == ""
   assert clean_team_name("Manchester City") == "Manchester City"
   assert clean_team_name("Parma\u2021") == "Parma"
   assert clean_team_name("Fiorentina\u00a7") == "Fiorentina"
   assert clean_team_name("Juventus*") == "Juventus"
   assert clean_team_name("West   Ham   United") == "West Ham United"
   print("PASS test_clean_team_name_edge_cases")


def test_normalize_numeric_string():
   assert _normalize_numeric_string("\u221212") == "-12"
   assert _normalize_numeric_string("\u201315") == "-15"
   assert _normalize_numeric_string("\u20143") == "-3"
   assert _normalize_numeric_string("42") == "42"
   assert _normalize_numeric_string("39[a]") == "39"
   assert _normalize_numeric_string("-5[1]") == "-5"
   assert _normalize_numeric_string(42) == 42
   assert _normalize_numeric_string("\u00a0 55 ") == "55"
   print("PASS test_normalize_numeric_string")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: find_league_table
# ─────────────────────────────────────────────────────────────────────────────

def test_find_league_table_picks_right_one():
   tables = [_fake_scorers(20), _fake_standings(20), _fake_scorers(10)]
   t = find_league_table(tables, num_teams=20)
   assert t is not None
   assert "Pts" in t.columns
   assert len(t) == 20
   print("PASS test_find_league_table_picks_right_one")


def test_find_league_table_returns_none_when_no_match():
   tables = [_fake_scorers(5), _fake_scorers(12)]
   assert find_league_table(tables, num_teams=20) is None
   print("PASS test_find_league_table_returns_none_when_no_match")


def test_find_league_table_tolerates_off_by_one():
   tables = [_fake_standings(19, pts_top=85)]
   t = find_league_table(tables, num_teams=20)
   assert t is not None
   assert len(t) == 19
   print("PASS test_find_league_table_tolerates_off_by_one")


def test_find_league_table_rejects_too_different_count():
   tables = [_fake_standings(10, pts_top=50)]
   assert find_league_table(tables, num_teams=20) is None
   print("PASS test_find_league_table_rejects_too_different_count")


def test_find_league_table_multi_index_columns():
   t = _fake_standings(20)
   t.columns = pd.MultiIndex.from_tuples(
       [(f"Level0_{i}", c) for i, c in enumerate(t.columns)]
   )
   result = find_league_table([t], num_teams=20)
   assert result is not None
   assert len(result) == 20
   print("PASS test_find_league_table_multi_index_columns")


def test_find_league_table_prefers_higher_score():
   weak = pd.DataFrame({"A": range(20), "B": range(20), "C": range(20)})
   strong = _fake_standings(20)
   result = find_league_table([weak, strong], num_teams=20)
   assert result is not None
   assert "Pts" in result.columns
   print("PASS test_find_league_table_prefers_higher_score")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: identify_columns
# ─────────────────────────────────────────────────────────────────────────────

def test_identify_columns_standard():
   team_col, pts_col = identify_columns(_fake_standings(20))
   assert team_col == "Team"
   assert pts_col == "Pts"
   print("PASS test_identify_columns_standard")


def test_identify_columns_alternative_names():
   t = _fake_standings(20).rename(columns={"Team": "Club", "Pts": "Points"})
   team_col, pts_col = identify_columns(t)
   assert team_col == "Club"
   assert pts_col == "Points"
   print("PASS test_identify_columns_alternative_names")


def test_identify_columns_spanish_names():
   t = _fake_standings(20).rename(columns={"Team": "Equipo", "Pts": "Ptos"})
   team_col, pts_col = identify_columns(t)
   assert team_col == "Equipo"
   assert pts_col == "Ptos"
   print("PASS test_identify_columns_spanish_names")


def test_identify_columns_fallback_no_obvious_names():
   t = pd.DataFrame({
       "Col1": list(range(1, 21)),
       "Col2": [f"Team Name {i}" for i in range(20)],
       "Col3": [30 + i for i in range(20)],
       "Col4": [20 - i for i in range(20)],
       "Col5": [90 - i * 3 for i in range(20)],
   })
   team_col, pts_col = identify_columns(t)
   assert team_col == "Col2"
   assert pts_col == "Col5"
   print("PASS test_identify_columns_fallback_no_obvious_names")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: enrich_dataframe
# ─────────────────────────────────────────────────────────────────────────────

def test_enrich_dataframe_basic():
   df = pd.DataFrame([{
       "Season": "2023/24", "Start Year": 2023, "Num Teams": 20, "Matches": 38,
       "Champion": "A", "Title-Winning Points": 90,
       "2nd Place": "B", "2nd Place Points": 85,
       "3rd Place": "C", "3rd Place Points": 80,
       "4th Place": "D", "4th Place Points": 75,
       "Survived Relegation": "E", "Relegation Survival Points": 36,
       "Relegated 1": "F", "Relegated 2": "G", "Relegated 3": "H",
   }])
   enriched = enrich_dataframe(df)
   assert enriched["Gap"].iloc[0] == 54
   assert enriched["Ratio"].iloc[0] == 2.5
   assert enriched["Title PPG"].iloc[0] == round(90 / 38, 3)
   assert enriched["Survival PPG"].iloc[0] == round(36 / 38, 3)
   assert "Gap (38-game)" in enriched.columns
   print("PASS test_enrich_dataframe_basic")


def test_enrich_dataframe_empty():
   assert len(enrich_dataframe(pd.DataFrame())) == 0
   print("PASS test_enrich_dataframe_empty")


def test_enrich_dataframe_no_matches_column():
   df = pd.DataFrame([{
       "Season": "2023/24", "Start Year": 2023, "Num Teams": 20,
       "Champion": "A", "Title-Winning Points": 90,
       "2nd Place": "B", "2nd Place Points": 85,
       "3rd Place": "C", "3rd Place Points": 80,
       "4th Place": "D", "4th Place Points": 75,
       "Survived Relegation": "E", "Relegation Survival Points": 36,
   }])
   enriched = enrich_dataframe(df)
   assert "Gap" in enriched.columns
   assert "Ratio" in enriched.columns
   assert "Title PPG" not in enriched.columns
   print("PASS test_enrich_dataframe_no_matches_column")


def test_enrich_dataframe_34_match_season():
   df = pd.DataFrame([{
       "Season": "2023/24", "Start Year": 2023, "Num Teams": 18, "Matches": 34,
       "Champion": "A", "Title-Winning Points": 78,
       "2nd Place": "B", "2nd Place Points": 70,
       "3rd Place": "C", "3rd Place Points": 65,
       "4th Place": "D", "4th Place Points": 60,
       "Survived Relegation": "E", "Relegation Survival Points": 30,
   }])
   enriched = enrich_dataframe(df)
   assert abs(enriched["Title Pts (38-game)"].iloc[0] - 87.2) < 0.2
   assert abs(enriched["Survival Pts (38-game)"].iloc[0] - 33.5) < 0.2
   print("PASS test_enrich_dataframe_34_match_season")


def test_enrich_dataframe_does_not_mutate():
   df = pd.DataFrame([{
       "Season": "2023/24", "Start Year": 2023, "Num Teams": 20, "Matches": 38,
       "Champion": "A", "Title-Winning Points": 90,
       "2nd Place": "B", "2nd Place Points": 85,
       "3rd Place": "C", "3rd Place Points": 80,
       "4th Place": "D", "4th Place Points": 75,
       "Survived Relegation": "E", "Relegation Survival Points": 36,
   }])
   original_cols = list(df.columns)
   _ = enrich_dataframe(df)
   assert list(df.columns) == original_cols
   print("PASS test_enrich_dataframe_does_not_mutate")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: _safe_polyfit
# ─────────────────────────────────────────────────────────────────────────────

def test_safe_polyfit_linear():
   x = np.arange(10).astype(float)
   y = 2 * x + 1
   c = _safe_polyfit(x, y)
   assert c is not None
   assert abs(c[0] - 2.0) < 1e-9
   assert abs(c[1] - 1.0) < 1e-9
   print("PASS test_safe_polyfit_linear")


def test_safe_polyfit_with_nans():
   x = np.arange(10).astype(float)
   y = 2.0 * x + 1.0
   y[3] = np.nan
   y[7] = np.nan
   c = _safe_polyfit(x, y)
   assert c is not None
   assert abs(c[0] - 2.0) < 0.1
   print("PASS test_safe_polyfit_with_nans")


def test_safe_polyfit_too_few_points():
   assert _safe_polyfit(np.array([1.0]), np.array([2.0])) is None
   assert _safe_polyfit(np.array([]), np.array([])) is None
   print("PASS test_safe_polyfit_too_few_points")


def test_safe_polyfit_all_nans():
   x = np.arange(5).astype(float)
   y = np.array([np.nan] * 5)
   assert _safe_polyfit(x, y) is None
   print("PASS test_safe_polyfit_all_nans")


def test_safe_polyfit_quadratic():
   x = np.arange(20).astype(float)
   y = 0.5 * x ** 2 + 3 * x + 7
   c = _safe_polyfit(x, y, deg=2)
   assert c is not None
   assert len(c) == 3
   assert abs(c[0] - 0.5) < 1e-6
   print("PASS test_safe_polyfit_quadratic")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: frequency tables
# ─────────────────────────────────────────────────────────────────────────────

def test_frequency_tables_basic():
   df = pd.DataFrame([
       {"Champion": "A", "2nd Place": "B", "3rd Place": "C", "4th Place": "D",
        "Relegated 1": "X", "Relegated 2": "Y", "Relegated 3": "Z"},
       {"Champion": "A", "2nd Place": "C", "3rd Place": "B", "4th Place": "E",
        "Relegated 1": "X", "Relegated 2": "W", "Relegated 3": "Z"},
   ])
   top4, releg = build_frequency_tables(df, LEAGUES["premier_league"])
   assert top4[top4["Club"] == "A"]["Top 4 Finishes"].iloc[0] == 2
   assert releg[releg["Club"] == "X"]["Times Relegated"].iloc[0] == 2
   assert releg[releg["Club"] == "Z"]["Times Relegated"].iloc[0] == 2
   print("PASS test_frequency_tables_basic")


def test_frequency_tables_percentage():
   df = pd.DataFrame([
       {"Champion": "A", "2nd Place": "B", "3rd Place": "C", "4th Place": "D",
        "Relegated 1": "X", "Relegated 2": "Y", "Relegated 3": "Z"},
   ] * 10)
   top4, _ = build_frequency_tables(df, LEAGUES["premier_league"])
   assert top4[top4["Club"] == "A"]["Percentage"].iloc[0] == 100.0
   print("PASS test_frequency_tables_percentage")


def test_frequency_tables_na_handling():
   df = pd.DataFrame([
       {"Champion": "A", "2nd Place": "B", "3rd Place": "C", "4th Place": "D",
        "Relegated 1": "X", "Relegated 2": "N/A", "Relegated 3": "N/A"},
   ])
   _, releg = build_frequency_tables(df, LEAGUES["premier_league"])
   assert len(releg) == 1
   assert releg.iloc[0]["Club"] == "X"
   print("PASS test_frequency_tables_na_handling")


def test_frequency_tables_no_relegated_columns():
   df = pd.DataFrame([
       {"Champion": "A", "2nd Place": "B", "3rd Place": "C", "4th Place": "D"},
   ])
   _, releg = build_frequency_tables(df, LEAGUES["premier_league"])
   assert len(releg) == 0
   print("PASS test_frequency_tables_no_relegated_columns")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: build_league_chart
# ─────────────────────────────────────────────────────────────────────────────

def test_build_league_chart_smoke():
   df = _fake_enriched_seasons(1992, 2025)
   fig = build_league_chart(df, LEAGUES["premier_league"])
   assert fig is not None
   assert len(fig.data) >= 9
   for i in range(9):
       assert fig.data[i].visible is True
   print("PASS test_build_league_chart_smoke")


def test_build_league_chart_with_eras():
   df = _fake_enriched_seasons(1992, 2025)
   eras = [
       {"label": "Era 1", "start": "1992/93", "end": "2005/06", "title": "First Era"},
       {"label": "Era 2", "start": "2006/07", "end": "2024/25", "title": "Second Era"},
   ]
   fig = build_league_chart(df, LEAGUES["premier_league"], eras=eras)
   assert len(fig.data) == 18
   for i in range(9):
       assert fig.data[i].visible is True
   for i in range(9, 18):
       assert fig.data[i].visible is False
   assert len(fig.layout.updatemenus) == 1
   assert len(fig.layout.updatemenus[0].buttons) == 2
   print("PASS test_build_league_chart_with_eras")


def test_build_league_chart_with_outliers():
   df = _fake_enriched_seasons(1992, 2025)
   outliers = [{"season": "2003/04", "y": 88, "text": "Test", "color": "#FFF", "ax": 0, "ay": -25}]
   fig = build_league_chart(df, LEAGUES["premier_league"], outliers=outliers)
   assert len(fig.layout.annotations) >= 2
   print("PASS test_build_league_chart_with_outliers")


def test_build_league_chart_invalid_era_skipped():
   df = _fake_enriched_seasons(2000, 2010)
   eras = [
       {"label": "Valid", "start": "2000/01", "end": "2009/10", "title": "Valid"},
       {"label": "Invalid", "start": "1992/93", "end": "1999/00", "title": "Missing"},
   ]
   fig = build_league_chart(df, LEAGUES["premier_league"], eras=eras)
   assert len(fig.data) == 18
   print("PASS test_build_league_chart_invalid_era_skipped")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: build_comparison_chart
# ─────────────────────────────────────────────────────────────────────────────

def test_build_comparison_chart_smoke():
   data = {}
   for key in ["premier_league", "bundesliga"]:
       data[key] = _fake_enriched_seasons(1992, 2025, num_teams=LEAGUES[key].num_teams)
   fig = build_comparison_chart(data)
   assert fig is not None
   assert len(fig.data) == 4
   print("PASS test_build_comparison_chart_smoke")


def test_build_comparison_chart_missing_metric():
   data = {
       "premier_league": _fake_enriched_seasons(1992, 2025, num_teams=20),
       "bundesliga": pd.DataFrame(),
   }
   fig = build_comparison_chart(data, metric="Gap (38-game)")
   assert len(fig.data) == 2
   print("PASS test_build_comparison_chart_missing_metric")


def test_build_comparison_chart_custom_metric():
   data = {"premier_league": _fake_enriched_seasons(1992, 2025, num_teams=20)}
   fig = build_comparison_chart(data, metric="Ratio", title="Custom Title")
   assert fig.layout.title.text == "Custom Title"
   print("PASS test_build_comparison_chart_custom_metric")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: fetch_page
# ─────────────────────────────────────────────────────────────────────────────

@patch("_utils.requests.get")
def test_fetch_page_success(mock_get):
   mock_resp = MagicMock()
   mock_resp.status_code = 200
   mock_resp.text = "<html>hello</html>"
   mock_get.return_value = mock_resp
   assert fetch_page("http://example.com") == "<html>hello</html>"
   print("PASS test_fetch_page_success")


@patch("_utils.requests.get")
def test_fetch_page_404(mock_get):
   mock_resp = MagicMock()
   mock_resp.status_code = 404
   mock_get.return_value = mock_resp
   assert fetch_page("http://example.com/missing") is None
   print("PASS test_fetch_page_404")


@patch("_utils.requests.get")
def test_fetch_page_500(mock_get):
   mock_resp = MagicMock()
   mock_resp.status_code = 500
   mock_get.return_value = mock_resp
   assert fetch_page("http://example.com/error") is None
   print("PASS test_fetch_page_500")


@patch("_utils.requests.get")
def test_fetch_page_timeout(mock_get):
   import requests as req_lib
   mock_get.side_effect = req_lib.exceptions.Timeout("timed out")
   assert fetch_page("http://example.com/slow") is None
   print("PASS test_fetch_page_timeout")


@patch("_utils.requests.get")
def test_fetch_page_connection_error(mock_get):
   import requests as req_lib
   mock_get.side_effect = req_lib.exceptions.ConnectionError("refused")
   assert fetch_page("http://example.com/down") is None
   print("PASS test_fetch_page_connection_error")


@patch("_utils.requests.get")
def test_fetch_page_sends_user_agent(mock_get):
   mock_resp = MagicMock()
   mock_resp.status_code = 200
   mock_resp.text = "ok"
   mock_get.return_value = mock_resp
   fetch_page("http://example.com")
   call_kwargs = mock_get.call_args[1]
   assert "User-Agent" in call_kwargs["headers"]
   assert "Euro-Football" in call_kwargs["headers"]["User-Agent"]
   print("PASS test_fetch_page_sends_user_agent")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: scrape_season
# ─────────────────────────────────────────────────────────────────────────────

@patch("_utils.fetch_page")
def test_scrape_season_success(mock_fetch):
   mock_fetch.return_value = _build_standings_html(num_teams=20, top_pts=89)
   result = scrape_season("premier_league", 2023, LEAGUES["premier_league"])
   assert result is not None
   assert result["Season"] == "2023/24"
   assert result["Champion"] == "Club A"
   assert result["Title-Winning Points"] == 89
   assert result["2nd Place Points"] == 86
   assert result["Survived Relegation"] == "Club Q"
   assert result["Relegation Survival Points"] == 89 - 16 * 3
   assert result["Relegated 1"] == "Club R"
   assert result["Relegated 2"] == "Club S"
   assert result["Relegated 3"] == "Club T"
   assert result["Matches"] == 38
   print("PASS test_scrape_season_success")


@patch("_utils.fetch_page")
def test_scrape_season_18_teams(mock_fetch):
   mock_fetch.return_value = _build_standings_html(num_teams=18, top_pts=80)
   result = scrape_season("bundesliga", 2023, LEAGUES["bundesliga"])
   assert result is not None
   assert result["Num Teams"] == 18
   assert result["Matches"] == 34
   assert result["Survived Relegation"] == "Club O"
   print("PASS test_scrape_season_18_teams")


@patch("_utils.fetch_page")
def test_scrape_season_no_html(mock_fetch):
   mock_fetch.return_value = None
   assert scrape_season("premier_league", 2023, LEAGUES["premier_league"]) is None
   print("PASS test_scrape_season_no_html")


@patch("_utils.fetch_page")
def test_scrape_season_no_valid_table(mock_fetch):
   mock_fetch.return_value = _build_bad_html()
   assert scrape_season("premier_league", 2023, LEAGUES["premier_league"]) is None
   print("PASS test_scrape_season_no_valid_table")


@patch("_utils.fetch_page")
def test_scrape_season_with_citations(mock_fetch):
   mock_fetch.return_value = _build_standings_with_citations(num_teams=20)
   result = scrape_season("premier_league", 2023, LEAGUES["premier_league"])
   assert result is not None
   assert result["Champion"] == "Club A"
   assert result["2nd Place"] == "Club B"
   assert "(" not in result["Relegated 3"]
   assert "[" not in result["Relegated 3"]
   print("PASS test_scrape_season_with_citations")


@patch("_utils.fetch_page")
def test_scrape_season_unicode_in_table(mock_fetch):
   mock_fetch.return_value = _build_standings_unicode_pts(num_teams=20)
   result = scrape_season("premier_league", 2023, LEAGUES["premier_league"])
   assert result is not None
   assert result["Title-Winning Points"] == 90
   print("PASS test_scrape_season_unicode_in_table")


@patch("_utils.fetch_page")
def test_scrape_season_fallback_url_patterns(mock_fetch):
   call_count = {"n": 0}

   def side_effect(url):
       call_count["n"] += 1
       if call_count["n"] == 1:
           return None
       return _build_standings_html(num_teams=20, top_pts=85)

   mock_fetch.side_effect = side_effect
   result = scrape_season("la_liga", 2023, LEAGUES["la_liga"])
   assert result is not None
   assert result["Title-Winning Points"] == 85
   assert call_count["n"] == 2
   print("PASS test_scrape_season_fallback_url_patterns")


@patch("_utils.fetch_page")
def test_scrape_season_historical_team_count(mock_fetch):
   mock_fetch.return_value = _build_standings_html(num_teams=22, top_pts=84)
   result = scrape_season("premier_league", 1993, LEAGUES["premier_league"])
   assert result is not None
   assert result["Num Teams"] == 22
   assert result["Matches"] == 42
   print("PASS test_scrape_season_historical_team_count")


@patch("_utils.fetch_page")
def test_scrape_season_too_few_rows_returns_none(mock_fetch):
   mock_fetch.return_value = _build_standings_html(num_teams=10, top_pts=70)
   assert scrape_season("premier_league", 2023, LEAGUES["premier_league"]) is None
   print("PASS test_scrape_season_too_few_rows_returns_none")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: validate_dataframe
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_dataframe_removes_invalid():
   df = pd.DataFrame([
       {"Title-Winning Points": 90, "Relegation Survival Points": 35},
       {"Title-Winning Points": 85, "Relegation Survival Points": 40},
       {"Title-Winning Points": 0,  "Relegation Survival Points": 35},
       {"Title-Winning Points": 80, "Relegation Survival Points": 0},
       {"Title-Winning Points": 30, "Relegation Survival Points": 50},
       {"Title-Winning Points": -5, "Relegation Survival Points": 35},
   ])
   result = validate_dataframe(df)
   assert len(result) == 2
   assert result.iloc[0]["Title-Winning Points"] == 90
   print("PASS test_validate_dataframe_removes_invalid")


def test_validate_dataframe_keeps_all_valid():
   df = pd.DataFrame([
       {"Title-Winning Points": 90, "Relegation Survival Points": 35},
       {"Title-Winning Points": 85, "Relegation Survival Points": 40},
       {"Title-Winning Points": 40, "Relegation Survival Points": 40},
   ])
   assert len(validate_dataframe(df)) == 3
   print("PASS test_validate_dataframe_keeps_all_valid")


def test_validate_dataframe_resets_index():
   df = pd.DataFrame([
       {"Title-Winning Points": 0,  "Relegation Survival Points": 35},
       {"Title-Winning Points": 90, "Relegation Survival Points": 35},
   ])
   result = validate_dataframe(df)
   assert result.index[0] == 0
   print("PASS test_validate_dataframe_resets_index")


def test_validate_dataframe_does_not_mutate():
   df = pd.DataFrame([
       {"Title-Winning Points": 0, "Relegation Survival Points": 35},
       {"Title-Winning Points": 90, "Relegation Survival Points": 35},
   ])
   original_len = len(df)
   _ = validate_dataframe(df)
   assert len(df) == original_len
   print("PASS test_validate_dataframe_does_not_mutate")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: scrape_league (caching and integration)
# ─────────────────────────────────────────────────────────────────────────────

def test_scrape_league_cache_hit():
   with tempfile.TemporaryDirectory() as tmpdir:
       cache_path = Path(tmpdir) / "pl_data_cache.csv"
       fake_df = pd.DataFrame([{
           "Season": "2023/24", "Start Year": 2023, "Num Teams": 20,
           "Matches": 38, "Champion": "Cached FC",
           "Title-Winning Points": 91,
           "2nd Place": "B", "2nd Place Points": 80,
           "3rd Place": "C", "3rd Place Points": 75,
           "4th Place": "D", "4th Place Points": 70,
           "Survived Relegation": "E", "Relegation Survival Points": 36,
           "Relegated 1": "X", "Relegated 2": "Y", "Relegated 3": "Z",
       }])
       fake_df.to_csv(cache_path, index=False)
       result = scrape_league("premier_league", cache_dir=tmpdir, max_age_days=7.0)
       assert len(result) == 1
       assert result.iloc[0]["Champion"] == "Cached FC"
       print("PASS test_scrape_league_cache_hit")


def test_scrape_league_cache_expired():
   with tempfile.TemporaryDirectory() as tmpdir:
       cache_path = Path(tmpdir) / "pl_data_cache.csv"
       fake_df = pd.DataFrame([{
           "Season": "2023/24", "Start Year": 2023, "Num Teams": 20,
           "Matches": 38, "Champion": "Old Cached FC",
           "Title-Winning Points": 91,
           "2nd Place": "B", "2nd Place Points": 80,
           "3rd Place": "C", "3rd Place Points": 75,
           "4th Place": "D", "4th Place Points": 70,
           "Survived Relegation": "E", "Relegation Survival Points": 36,
           "Relegated 1": "X", "Relegated 2": "Y", "Relegated 3": "Z",
       }])
       fake_df.to_csv(cache_path, index=False)
       old_time = time.time() - 10 * 86400
       os.utime(cache_path, (old_time, old_time))

       with patch("_utils.fetch_page") as mock_fetch:
           mock_fetch.return_value = _build_standings_html(num_teams=20, top_pts=92)
           tiny_config = _make_tiny_config(cache_file="pl_data_cache.csv")
           result = scrape_league(
               "premier_league", config=tiny_config,
               cache_dir=tmpdir, max_age_days=7.0, delay=0.0,
           )
           assert result.iloc[0]["Title-Winning Points"] == 92
           print("PASS test_scrape_league_cache_expired")


def test_scrape_league_creates_cache_dir():
   with tempfile.TemporaryDirectory() as tmpdir:
       nested = Path(tmpdir) / "deep" / "nested" / "cache"
       assert not nested.exists()
       with patch("_utils.fetch_page") as mock_fetch:
           mock_fetch.return_value = _build_standings_html(num_teams=20, top_pts=88)
           tiny_config = _make_tiny_config(cache_file="pl_data_cache.csv")
           scrape_league(
               "premier_league", config=tiny_config,
               cache_dir=str(nested), max_age_days=7.0, delay=0.0,
           )
           assert nested.exists()
           assert (nested / "pl_data_cache.csv").exists()
           print("PASS test_scrape_league_creates_cache_dir")


@patch("_utils.fetch_page")
def test_scrape_league_handles_all_failures(mock_fetch):
   mock_fetch.return_value = None
   with tempfile.TemporaryDirectory() as tmpdir:
       tiny_config = _make_tiny_config(start_year=2022, end_year=2024, cache_file="fail.csv")
       result = scrape_league("test", config=tiny_config, cache_dir=tmpdir, delay=0.0)
       assert len(result) == 0
       assert not (Path(tmpdir) / "fail.csv").exists()
       print("PASS test_scrape_league_handles_all_failures")


@patch("_utils.fetch_page")
def test_scrape_league_partial_failures(mock_fetch):
   call_count = {"n": 0}

   def side_effect(url):
       call_count["n"] += 1
       if call_count["n"] % 2 == 0:
           return None
       return _build_standings_html(num_teams=20, top_pts=87)

   mock_fetch.side_effect = side_effect
   with tempfile.TemporaryDirectory() as tmpdir:
       tiny_config = _make_tiny_config(start_year=2020, end_year=2024, cache_file="partial.csv")
       result = scrape_league("test", config=tiny_config, cache_dir=tmpdir, delay=0.0)
       assert 2 <= len(result) < 5
       print("PASS test_scrape_league_partial_failures")


@patch("_utils.fetch_page")
def test_scrape_league_writes_cache_file(mock_fetch):
   mock_fetch.return_value = _build_standings_html(num_teams=20, top_pts=90)
   with tempfile.TemporaryDirectory() as tmpdir:
       tiny_config = _make_tiny_config(cache_file="write_test.csv")
       scrape_league("test", config=tiny_config, cache_dir=tmpdir, delay=0.0)
       cached = pd.read_csv(Path(tmpdir) / "write_test.csv")
       assert len(cached) == 1
       assert cached.iloc[0]["Title-Winning Points"] == 90
       print("PASS test_scrape_league_writes_cache_file")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: end-to-end pipeline
# ─────────────────────────────────────────────────────────────────────────────

@patch("_utils.fetch_page")
def test_end_to_end_pipeline(mock_fetch):
   mock_fetch.return_value = _build_standings_html(num_teams=20, top_pts=91)
   with tempfile.TemporaryDirectory() as tmpdir:
       tiny_config = _make_tiny_config(start_year=2020, end_year=2024, cache_file="e2e.csv")
       tiny_config.name = "Premier League"
       tiny_config.short_name = "PL"

       raw = scrape_league("premier_league", config=tiny_config, cache_dir=tmpdir, delay=0.0)
       assert len(raw) == 5

       enriched = enrich_dataframe(raw)
       for col in ["Gap", "Ratio", "Title PPG", "Gap (38-game)"]:
           assert col in enriched.columns, f"Missing: {col}"
       assert (enriched["Gap"] > 0).all()
       assert (enriched["Ratio"] > 1).all()

       fig = build_league_chart(enriched, LEAGUES["premier_league"])
       assert fig is not None and len(fig.data) >= 9

       top4, releg = build_frequency_tables(enriched, LEAGUES["premier_league"])
       assert len(top4) > 0 and len(releg) > 0
       print("PASS test_end_to_end_pipeline")


@patch("_utils.fetch_page")
def test_end_to_end_comparison(mock_fetch):
   mock_fetch.return_value = _build_standings_html(num_teams=20, top_pts=88)
   league_data = {}
   with tempfile.TemporaryDirectory() as tmpdir:
       for key in ["premier_league", "la_liga"]:
           config = _make_tiny_config(start_year=2020, end_year=2024, cache_file=f"{key}.csv")
           config.name = LEAGUES[key].name
           config.short_name = LEAGUES[key].short_name
           config.color = LEAGUES[key].color
           config.color_secondary = LEAGUES[key].color_secondary
           raw = scrape_league(key, config=config, cache_dir=tmpdir, delay=0.0)
           league_data[key] = enrich_dataframe(raw)

       fig = build_comparison_chart(league_data, metric="Gap (38-game)")
       assert fig is not None and len(fig.data) == 4
       print("PASS test_end_to_end_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: Champions League
# ─────────────────────────────────────────────────────────────────────────────

from _utils import (
   _get_cl_known_results, _get_club_country, load_cl_data,
   enrich_cl_dataframe, build_cl_country_dominance_chart,
   build_cl_winners_chart, build_cl_concentration_chart,
   CL_START_YEAR, CL_END_YEAR,
)


def test_cl_known_results_complete():
   """All seasons from 1992-2024 have curated results."""
   results = _get_cl_known_results()
   for year in range(CL_START_YEAR, CL_END_YEAR + 1):
       assert year in results, f"Missing CL season {year}"
       r = results[year]
       assert r["Winner"] is not None, f"No winner for {year}"
       assert r["Runner-Up"] is not None, f"No runner-up for {year}"
       assert r["Semi-Finalist 1"] is not None, f"No SF1 for {year}"
       assert r["Semi-Finalist 2"] is not None, f"No SF2 for {year}"
       assert r["Season"] == get_season_label(year)
   print("PASS test_cl_known_results_complete")


def test_cl_known_results_countries():
   """All clubs in CL results have a country mapping."""
   results = _get_cl_known_results()
   for year, r in results.items():
       for key in ["Winner Country", "Runner-Up Country",
                   "Semi-Finalist 1 Country", "Semi-Finalist 2 Country"]:
           assert r[key] != "Unknown", f"{year}: {key} is Unknown for {r.get(key.replace(' Country', ''))}"
   print("PASS test_cl_known_results_countries")


def test_get_club_country():
   assert _get_club_country("Real Madrid") == "Spain"
   assert _get_club_country("Liverpool") == "England"
   assert _get_club_country("Bayern Munich") == "Germany"
   assert _get_club_country("AC Milan") == "Italy"
   assert _get_club_country("Paris Saint-Germain") == "France"
   assert _get_club_country("Nonexistent FC") == "Unknown"
   print("PASS test_get_club_country")


def test_load_cl_data():
   with tempfile.TemporaryDirectory() as tmpdir:
       df = load_cl_data(cache_dir=tmpdir)
       assert len(df) == CL_END_YEAR - CL_START_YEAR + 1
       assert "Winner" in df.columns
       assert "Runner-Up" in df.columns
       assert "Winner Country" in df.columns
       assert df.iloc[0]["Season"] == "1992/93"
       # Check cache was written
       assert (Path(tmpdir) / "cl_data_cache.csv").exists()
       print("PASS test_load_cl_data")


def test_load_cl_data_cache_hit():
   with tempfile.TemporaryDirectory() as tmpdir:
       # First load creates cache
       df1 = load_cl_data(cache_dir=tmpdir)
       # Second load hits cache
       df2 = load_cl_data(cache_dir=tmpdir)
       assert len(df1) == len(df2)
       assert df1.iloc[0]["Winner"] == df2.iloc[0]["Winner"]
       print("PASS test_load_cl_data_cache_hit")


def test_enrich_cl_dataframe():
   results = _get_cl_known_results()
   df = pd.DataFrame([results[y] for y in sorted(results.keys())])
   enriched = enrich_cl_dataframe(df)

   assert "Same Country Final" in enriched.columns
   assert "Unique Countries in Top 4" in enriched.columns
   assert "Top 4 Countries" in enriched.columns

   # 2007: Man Utd vs Chelsea = same country
   row_2007 = enriched[enriched["Start Year"] == 2007].iloc[0]
   assert row_2007["Same Country Final"] is True or row_2007["Same Country Final"] == True

   # Unique countries should be between 1 and 4
   assert (enriched["Unique Countries in Top 4"] >= 1).all()
   assert (enriched["Unique Countries in Top 4"] <= 4).all()
   print("PASS test_enrich_cl_dataframe")


def test_enrich_cl_dataframe_empty():
   enriched = enrich_cl_dataframe(pd.DataFrame())
   assert len(enriched) == 0
   print("PASS test_enrich_cl_dataframe_empty")


def test_enrich_cl_dataframe_does_not_mutate():
   results = _get_cl_known_results()
   df = pd.DataFrame([results[2020]])
   original_cols = list(df.columns)
   _ = enrich_cl_dataframe(df)
   assert list(df.columns) == original_cols
   print("PASS test_enrich_cl_dataframe_does_not_mutate")


def test_build_cl_country_dominance_chart():
   results = _get_cl_known_results()
   df = pd.DataFrame([results[y] for y in sorted(results.keys())])
   enriched = enrich_cl_dataframe(df)
   fig = build_cl_country_dominance_chart(enriched)
   assert fig is not None
   assert len(fig.data) >= 3  # at least a few countries
   print("PASS test_build_cl_country_dominance_chart")


def test_build_cl_winners_chart():
   results = _get_cl_known_results()
   df = pd.DataFrame([results[y] for y in sorted(results.keys())])
   fig = build_cl_winners_chart(df)
   assert fig is not None
   assert len(fig.data) == 1  # single bar trace
   print("PASS test_build_cl_winners_chart")


def test_build_cl_concentration_chart():
   results = _get_cl_known_results()
   df = pd.DataFrame([results[y] for y in sorted(results.keys())])
   enriched = enrich_cl_dataframe(df)
   fig = build_cl_concentration_chart(enriched)
   assert fig is not None
   assert len(fig.data) >= 2  # at least the two main lines
   print("PASS test_build_cl_concentration_chart")


def test_cl_real_madrid_dominance():
   """Verify Real Madrid's known dominance shows up in data."""
   results = _get_cl_known_results()
   df = pd.DataFrame([results[y] for y in sorted(results.keys())])
   rm_wins = len(df[df["Winner"] == "Real Madrid"])
   assert rm_wins >= 8, f"Real Madrid should have 8+ wins, got {rm_wins}"
   print("PASS test_cl_real_madrid_dominance")


def test_cl_english_clubs_post_2018():
   """English clubs dominated 2018-2022 finals."""
   results = _get_cl_known_results()
   df = pd.DataFrame([results[y] for y in range(2018, 2023)])
   english_winners = df[df["Winner Country"] == "England"]
   assert len(english_winners) >= 2
   print("PASS test_cl_english_clubs_post_2018")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
   tests = [
       test_get_season_label,
       test_get_matches_per_season,
       test_get_num_teams_for_season,
       test_league_config_integrity,
       test_clean_team_name_basic,
       test_clean_team_name_multiple_markers,
       test_clean_team_name_edge_cases,
       test_normalize_numeric_string,
       test_find_league_table_picks_right_one,
       test_find_league_table_returns_none_when_no_match,
       test_find_league_table_tolerates_off_by_one,
       test_find_league_table_rejects_too_different_count,
       test_find_league_table_multi_index_columns,
       test_find_league_table_prefers_higher_score,
       test_identify_columns_standard,
       test_identify_columns_alternative_names,
       test_identify_columns_spanish_names,
       test_identify_columns_fallback_no_obvious_names,
       test_enrich_dataframe_basic,
       test_enrich_dataframe_empty,
       test_enrich_dataframe_no_matches_column,
       test_enrich_dataframe_34_match_season,
       test_enrich_dataframe_does_not_mutate,
       test_safe_polyfit_linear,
       test_safe_polyfit_with_nans,
       test_safe_polyfit_too_few_points,
       test_safe_polyfit_all_nans,
       test_safe_polyfit_quadratic,
       test_frequency_tables_basic,
       test_frequency_tables_percentage,
       test_frequency_tables_na_handling,
       test_frequency_tables_no_relegated_columns,
       test_build_league_chart_smoke,
       test_build_league_chart_with_eras,
       test_build_league_chart_with_outliers,
       test_build_league_chart_invalid_era_skipped,
       test_build_comparison_chart_smoke,
       test_build_comparison_chart_missing_metric,
       test_build_comparison_chart_custom_metric,
       test_fetch_page_success,
       test_fetch_page_404,
       test_fetch_page_500,
       test_fetch_page_timeout,
       test_fetch_page_connection_error,
       test_fetch_page_sends_user_agent,
       test_scrape_season_success,
       test_scrape_season_18_teams,
       test_scrape_season_no_html,
       test_scrape_season_no_valid_table,
       test_scrape_season_with_citations,
       test_scrape_season_unicode_in_table,
       test_scrape_season_fallback_url_patterns,
       test_scrape_season_historical_team_count,
       test_scrape_season_too_few_rows_returns_none,
       test_validate_dataframe_removes_invalid,
       test_validate_dataframe_keeps_all_valid,
       test_validate_dataframe_resets_index,
       test_validate_dataframe_does_not_mutate,
       test_scrape_league_cache_hit,
       test_scrape_league_cache_expired,
       test_scrape_league_creates_cache_dir,
       test_scrape_league_handles_all_failures,
       test_scrape_league_partial_failures,
       test_scrape_league_writes_cache_file,
       test_end_to_end_pipeline,
       test_end_to_end_comparison,
       # Champions League
       test_cl_known_results_complete,
       test_cl_known_results_countries,
       test_get_club_country,
       test_load_cl_data,
       test_load_cl_data_cache_hit,
       test_enrich_cl_dataframe,
       test_enrich_cl_dataframe_empty,
       test_enrich_cl_dataframe_does_not_mutate,
       test_build_cl_country_dominance_chart,
       test_build_cl_winners_chart,
       test_build_cl_concentration_chart,
       test_cl_real_madrid_dominance,
       test_cl_english_clubs_post_2018,
   ]

   print(f"Running {len(tests)} tests...\n")
   fails = 0
   for t in tests:
       try:
           t()
       except Exception as e:
           fails += 1
           import traceback
           print(f"FAIL {t.__name__}: {e}")
           traceback.print_exc()
           print()

   print(f"\n{'='*60}")
   if fails == 0:
       print(f"ALL {len(tests)} TESTS PASSED")
   else:
       print(f"{len(tests) - fails}/{len(tests)} tests passed, {fails} FAILED")
   sys.exit(0 if fails == 0 else 1)