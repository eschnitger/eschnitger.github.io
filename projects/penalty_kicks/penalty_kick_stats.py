"""
EPL Penalty Analysis — Python translation of the R/worldfootballR + ggplot2 workflow
======================================================================================

Original R stack:          Python equivalent used here:
------------------------   ------------------------------------------
worldfootballR (FBref)  -> soccerdata.FBref
tidyverse / dplyr        -> pandas
ggplot2                   -> matplotlib + seaborn
ggimage / ggpath (logos)  -> matplotlib OffsetImage/AnnotationBbox + Pillow
patchwork (combine plots) -> matplotlib subplots / GridSpec

Install:
    pip install soccerdata pandas matplotlib seaborn pillow requests --break-system-packages
"""

import time
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
from io import BytesIO
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import soccerdata as sd

sns.set_theme(style="whitegrid")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Logo lookup table (Tip: add club logos)
# ----------------------------------------------------------------------------
# ESPN's team-logo CDN, same source as the R version.
# Updated for the 2025/26 season roster: Burnley, Leeds United, and Sunderland
# are promoted (replacing Ipswich, Leicester City, and Southampton), and the
# names below match soccerdata's own squad-name spelling ("Newcastle",
# "Nottingham") rather than FBref's longer official names — confirm against
# your own data with `epl_logos_missing_check` below if FBref renames again.
EPL_ESPN_IDS = {
    "Arsenal": 359, "Aston Villa": 362, "Bournemouth": 349, "Brentford": 337,
    "Brighton": 331, "Burnley": 379, "Chelsea": 363, "Crystal Palace": 384,
    "Everton": 368, "Fulham": 370, "Leeds United": 357, "Liverpool": 364,
    "Manchester City": 382, "Manchester Utd": 360, "Newcastle": 361,
    "Nottingham": 393, "Sunderland": 366, "Tottenham": 367,
    "West Ham": 371, "Wolves": 380,
}

epl_logos = pd.DataFrame(
    {"Squad": list(EPL_ESPN_IDS.keys()), "espn_id": list(EPL_ESPN_IDS.values())}
)
epl_logos["logo_url"] = epl_logos["espn_id"].apply(
    lambda i: f"https://a.espncdn.com/combiner/i?img=/i/teamlogos/soccer/500/{i}.png&h=80&w=80"
)

_logo_cache: dict[str, Image.Image] = {}


def get_logo_image(url: str, zoom: float = 0.35):
    """Fetch (and cache) a club logo as a matplotlib OffsetImage."""
    if url not in _logo_cache:
        resp = requests.get(url, timeout=10)
        _logo_cache[url] = Image.open(BytesIO(resp.content)).convert("RGBA")
    return OffsetImage(_logo_cache[url], zoom=zoom)


def add_logos(ax, xs, ys, urls, zoom=0.35):
    """Place a logo image at each (x, y) point on an Axes (like geom_image)."""
    for x, y, url in zip(xs, ys, urls):
        if not isinstance(url, str):
            continue  # missing logo match (e.g. squad-name mismatch) — skip silently
        try:
            ab = AnnotationBbox(get_logo_image(url, zoom), (x, y), frameon=False)
            ax.add_artist(ab)
        except Exception as e:
            print(f"  logo failed for {url}: {e}")


# ----------------------------------------------------------------------------
# 2. Pulling the data
# ----------------------------------------------------------------------------
# soccerdata.FBref wants seasons like "2526" for 2025/26. season_end_year -> "YYZZ".
def season_str(season_end_year: int) -> str:
    start = str(season_end_year - 1)[-2:]
    end = str(season_end_year)[-2:]
    return f"{start}{end}"


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """soccerdata returns MultiIndex columns (e.g. ('Performance','PKwon')).
    Flatten to single strings so downstream code can address them by plain name."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            "_".join(str(p) for p in tup if p not in ("", None)) for tup in df.columns
        ]
    return df


def find_column(df: pd.DataFrame, *needles: str) -> str:
    """Find the single column whose (flattened) name contains ALL of the given
    case-insensitive substrings. Raises with the full column list if 0 or >1 match,
    so you see exactly what's available instead of a bare KeyError."""
    hits = [
        c for c in df.columns
        if all(n.lower() in str(c).lower() for n in needles)
    ]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(
        f"Expected exactly one column matching {needles}, found {hits}.\n"
        f"All columns: {list(df.columns)}"
    )


def pull_season_team_stats(season_end_year: int, stat_type: str = "misc") -> pd.DataFrame:
    """Team-level season stats (equivalent of fb_season_team_stats)."""
    print(f"Pulling team {stat_type} stats for season ending {season_end_year}...")
    fbref = sd.FBref(leagues="ENG-Premier League", seasons=season_str(season_end_year))
    df = flatten_columns(fbref.read_team_season_stats(stat_type=stat_type).reset_index())
    df["season_end"] = season_end_year
    time.sleep(3)  # rate limiting, mirrors Sys.sleep(3) in the R version
    return df


def pull_player_stats(season_end_year: int, stat_type: str = "misc") -> pd.DataFrame:
    """Player-level season stats (equivalent of fb_league_stats(team_or_player='player'))."""
    print(f"Pulling player {stat_type} stats for season ending {season_end_year}...")
    fbref = sd.FBref(leagues="ENG-Premier League", seasons=season_str(season_end_year))
    df = flatten_columns(fbref.read_player_season_stats(stat_type=stat_type).reset_index())
    df["season_end"] = season_end_year
    time.sleep(3)
    return df


def pull_many_seasons(seasons, puller):
    """map_dfr(seasons, possibly(puller, otherwise=NULL)) equivalent."""
    frames = []
    for yr in seasons:
        try:
            frames.append(puller(yr))
        except Exception as e:
            print(f"  season {yr} failed, skipping: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ----------------------------------------------------------------------------
# 3. Pull current-season team + player data
# ----------------------------------------------------------------------------
CURRENT_SEASON = 2026

team_stats = pull_season_team_stats(CURRENT_SEASON, "misc")

# Resolve the real (flattened) column names instead of hardcoding guesses.
# find_column() raises a clear error listing every available column if it
# can't find a unique match, so a mismatch here fails loudly and readably
# rather than as a bare KeyError.
SQUAD_COL = "Squad" if "Squad" in team_stats.columns else find_column(team_stats, "team")
MIN_COL = find_column(team_stats, "90s")

# NOTE: FBref's team-level "misc" table comes back from soccerdata with PKwon/
# PKcon as <NA> for every team (a gap in that particular scraped table, not a
# naming issue — 90s/team come through fine). Player-level misc stats DO carry
# real per-player PKwon/PKcon, so we derive team totals by summing those
# per squad instead of trusting the team-level columns.
print("Team-level misc table has no usable PKwon/PKcon — deriving team totals "
      "from player-level misc stats instead.")
player_misc_current = pull_player_stats(CURRENT_SEASON, "misc")
P_SQUAD_COL = "team" if "team" in player_misc_current.columns else find_column(player_misc_current, "team")
P_PKWON_COL = find_column(player_misc_current, "PKwon")
P_PKCON_COL = find_column(player_misc_current, "PKcon")

player_misc_current[P_PKWON_COL] = pd.to_numeric(player_misc_current[P_PKWON_COL], errors="coerce")
player_misc_current[P_PKCON_COL] = pd.to_numeric(player_misc_current[P_PKCON_COL], errors="coerce")

# Diagnostic: check whether the matched player-level PK columns are actually
# populated, or are <NA>/blank like the team-level table was. groupby().sum()
# silently treats NaN as 0, so an all-<NA> column would otherwise produce this
# exact "all teams show 0" symptom without ever raising an error.
_non_null_won = player_misc_current[P_PKWON_COL].notna().sum()
_non_null_con = player_misc_current[P_PKCON_COL].notna().sum()
_nonzero_won = (player_misc_current[P_PKWON_COL].fillna(0) > 0).sum()
print(f"player_misc_current[{P_PKWON_COL!r}]: {_non_null_won}/{len(player_misc_current)} non-null, "
      f"{_nonzero_won} rows > 0")
print(f"player_misc_current[{P_PKCON_COL!r}]: {_non_null_con}/{len(player_misc_current)} non-null")
if _nonzero_won == 0:
    # Show a known penalty taker explicitly so we can see the raw row, not just
    # an aggregate — helps tell "column is really all blank" apart from
    # "column exists but this player didn't take any."
    _name_col = "player" if "player" in player_misc_current.columns else find_column(player_misc_current, "player")
    _sample = player_misc_current[player_misc_current[_name_col].str.contains("Salah|Palmer|Haaland", na=False, case=False)]
    print("Sample known penalty-takers, all columns matching 'PK':")
    _pk_cols = [c for c in player_misc_current.columns if "pk" in c.lower()]
    print(_sample[[_name_col] + _pk_cols].to_string())
    print("All columns available in player_misc_current:")
    print(list(player_misc_current.columns))

team_pk_totals = (
    player_misc_current
    .groupby(P_SQUAD_COL)[[P_PKWON_COL, P_PKCON_COL]]
    .sum()
    .reset_index()
    .rename(columns={P_SQUAD_COL: "Squad", P_PKWON_COL: "PKwon", P_PKCON_COL: "PKcon"})
)

team_drawn = (
    team_stats[[SQUAD_COL, MIN_COL]]
    .rename(columns={SQUAD_COL: "Squad", MIN_COL: "Mins_Per_90"})
    .merge(team_pk_totals, on="Squad", how="left")
)

# soccerdata returns nullable extension dtypes (e.g. "Int64") that carry pd.NA
# instead of NaN. matplotlib can't handle pd.NA, so coerce to plain float64 and
# drop any team missing a stat rather than crashing deep inside a plotting call.
for col in ("PKwon", "PKcon", "Mins_Per_90"):
    team_drawn[col] = pd.to_numeric(team_drawn[col], errors="coerce").astype("float64")

_before = len(team_drawn)
team_drawn = team_drawn.dropna(subset=["PKwon", "PKcon", "Mins_Per_90"])
if len(team_drawn) < _before:
    print(f"Dropped {_before - len(team_drawn)} team(s) with missing PK/90s data.")

team_drawn = (
    team_drawn
    .assign(
        PK_drawn_per90=lambda d: d.PKwon / d.Mins_Per_90,
        PK_conceded_per90=lambda d: d.PKcon / d.Mins_Per_90,
    )
    .merge(epl_logos, on="Squad", how="left")
)

print(f"team_drawn PKwon range: {team_drawn['PKwon'].min()}-{team_drawn['PKwon'].max()}, "
      f"PKcon range: {team_drawn['PKcon'].min()}-{team_drawn['PKcon'].max()}")
if team_drawn["PKwon"].max() == 0 and team_drawn["PKcon"].max() == 0:
    print("WARNING: all team PKwon/PKcon totals are 0 — the player-level "
          "aggregation likely isn't finding real values. Sample rows:")
    print(team_drawn[["Squad", "PKwon", "PKcon"]].head(10).to_string())

# check for squad-name mismatches, same idea as anti_join() in the R script
missing = team_drawn[team_drawn["logo_url"].isna()]["Squad"].tolist()
if missing:
    print("No logo match for:", missing)

# ----------------------------------------------------------------------------
# 4. Chart 1 — Team scatter: penalties drawn vs conceded, with logos as points
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7))

med_won = team_drawn["PKwon"].median()
med_con = team_drawn["PKcon"].median()

ax.axhspan(med_won, team_drawn["PKwon"].max() * 1.1, color="#1a9850", alpha=0.05)
ax.axvspan(med_con, team_drawn["PKcon"].max() * 1.1, color="#d73027", alpha=0.05)

lims = [0, max(team_drawn["PKwon"].max(), team_drawn["PKcon"].max()) * 1.1]
ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1)

add_logos(ax, team_drawn["PKcon"], team_drawn["PKwon"], team_drawn["logo_url"], zoom=0.3)

ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Penalties Conceded")
ax.set_ylabel("Penalties Drawn")
ax.set_title("EPL 2025/26: Which Teams Draw vs Concede the Most Penalties?", fontweight="bold")
ax.text(0.02, 0.95, "Draw many,\nconcede few", color="#1a9850", transform=ax.transAxes,
        fontsize=9, style="italic", va="top")
ax.text(0.75, 0.05, "Draw few,\nconcede many", color="#d73027", transform=ax.transAxes,
        fontsize=9, style="italic")
fig.text(0.99, 0.01, "Data: FBref via soccerdata | Logos: ESPN", ha="right", fontsize=8, color="gray")
fig.tight_layout()
fig.savefig("p_scatter.png", dpi=300)
plt.close(fig)

# ----------------------------------------------------------------------------
# 5. Chart 2 — Team bar chart, drawn vs conceded, with logos next to bars
# ----------------------------------------------------------------------------
team_long = team_drawn.melt(
    id_vars=["Squad", "logo_url"],
    value_vars=["PKwon", "PKcon"],
    var_name="type",
    value_name="count",
)
team_long["type"] = team_long["type"].map({"PKwon": "Penalties Drawn", "PKcon": "Penalties Conceded"})

order = (
    team_drawn.sort_values("PKwon", ascending=True)["Squad"].tolist()
)

fig, ax = plt.subplots(figsize=(9, 10))
sns.barplot(
    data=team_long, y="Squad", x="count", hue="type", order=order,
    palette={"Penalties Drawn": "#2196F3", "Penalties Conceded": "#F44336"},
    ax=ax,
)
ax.set_xlim(left=-2)
logo_x = {sq: -1.5 for sq in order}
add_logos(
    ax,
    [logo_x[sq] for sq in order],
    range(len(order)),
    [team_drawn.set_index("Squad").loc[sq, "logo_url"] for sq in order],
    zoom=0.22,
)
ax.set_title("EPL 2025/26: Penalties Drawn vs Conceded by Team", fontweight="bold")
ax.set_xlabel("Count")
ax.set_ylabel(None)
ax.legend(title=None, loc="lower right")
fig.text(0.99, 0.01, "Data: FBref via soccerdata | Logos: ESPN", ha="right", fontsize=8, color="gray")
fig.tight_layout()
fig.savefig("p_team_bar.png", dpi=300)
plt.close(fig)

# ----------------------------------------------------------------------------
# 6. Chart 3 — Net penalty balance lollipop chart, with logos
# ----------------------------------------------------------------------------
team_drawn["net_pk"] = team_drawn["PKwon"] - team_drawn["PKcon"]
team_drawn_sorted = team_drawn.sort_values("net_pk")

fig, ax = plt.subplots(figsize=(8, 10))
colors = ["#1a9850" if v > 0 else "#d73027" for v in team_drawn_sorted["net_pk"]]
ax.hlines(
    y=range(len(team_drawn_sorted)), xmin=0, xmax=team_drawn_sorted["net_pk"],
    color=colors, linewidth=2, alpha=0.7,
)
add_logos(
    ax,
    team_drawn_sorted["net_pk"],
    range(len(team_drawn_sorted)),
    team_drawn_sorted["logo_url"],
    zoom=0.25,
)
ax.axvline(0, color="gray", linewidth=1)
ax.set_yticks(range(len(team_drawn_sorted)))
ax.set_yticklabels(team_drawn_sorted["Squad"])
ax.set_title("EPL 2025/26: Net Penalty Balance by Team", fontweight="bold")
ax.set_xlabel("Net Penalties (Drawn minus Conceded)")
fig.text(0.99, 0.01, "Data: FBref via soccerdata | Logos: ESPN", ha="right", fontsize=8, color="gray")
fig.tight_layout()
fig.savefig("p_lollipop.png", dpi=300)
plt.close(fig)

# ----------------------------------------------------------------------------
# 7. Combine charts with GridSpec (patchwork equivalent)
# ----------------------------------------------------------------------------
def combine_pngs(paths, layout, out_path, title):
    """Simple combiner: lay out saved PNGs into a grid figure."""
    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(*layout, figure=fig)
    for i, p in enumerate(paths):
        ax = fig.add_subplot(gs[i])
        ax.imshow(plt.imread(p))
        ax.axis("off")
    fig.suptitle(title, fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


combine_pngs(
    ["p_scatter.png", "p_team_bar.png", "p_lollipop.png"],
    (2, 2),
    "epl_penalty_analysis.png",
    "EPL 2025/26 — Penalty Drawing & Conceding Analysis",
)

# ----------------------------------------------------------------------------
# 8. Multi-season pull with rate limiting + caching
# ----------------------------------------------------------------------------
SEASONS = range(2022, 2027)

# Team-level misc stats have the same <NA> PKwon/PKcon gap across all seasons
# (see note in section 3) — the multi-season trend chart derives team totals
# from player-level data instead (see section 10), so we only need to pull
# player-level stats here, not a separate team-level multi-season scrape.
player_cache = DATA_DIR / "epl_player_misc.pkl"
if player_cache.exists():
    all_players = pickle.load(open(player_cache, "rb"))
else:
    all_players = pull_many_seasons(SEASONS, lambda y: pull_player_stats(y, "misc"))
    pickle.dump(all_players, open(player_cache, "wb"))

# ----------------------------------------------------------------------------
# 9. Per-90 normalization (player stats)
# ----------------------------------------------------------------------------
P_MIN_COL = find_column(all_players, "90s")
P_PKWON_COL = find_column(all_players, "PKwon")
P_PKCON_COL = find_column(all_players, "PKcon")
PLAYER_COL = "player" if "player" in all_players.columns else find_column(all_players, "player")
SQUAD_COL_PLAYERS = "Squad" if "Squad" in all_players.columns else find_column(all_players, "team")

all_players = all_players.rename(columns={
    P_MIN_COL: "90s", P_PKWON_COL: "PKwon", P_PKCON_COL: "PKcon",
    PLAYER_COL: "player", SQUAD_COL_PLAYERS: "Squad",
})
for col in ("90s", "PKwon", "PKcon"):
    all_players[col] = pd.to_numeric(all_players[col], errors="coerce").astype("float64")

player_per90 = all_players[all_players["90s"] >= 5].copy()
player_per90["PKwon_per90"] = (player_per90["PKwon"] / player_per90["90s"]).round(3)
player_per90["PKcon_per90"] = (player_per90["PKcon"] / player_per90["90s"]).round(3)

# ----------------------------------------------------------------------------
# 10. Chart 4 — multi-season team trend lines, with logos on each point
# ----------------------------------------------------------------------------
# Same team-level <NA> gap applies across all past seasons (see note in section
# 3), so derive per-season team totals by summing player-level PKwon/PKcon,
# grouped by squad and season — using all_players, already normalized above
# (Squad is already renamed above, so this groups directly on it).
team_trends = (
    all_players
    .groupby(["Squad", "season_end"])[["PKwon", "PKcon"]]
    .sum()
    .reset_index()
    .merge(epl_logos, on="Squad", how="left")
    .dropna(subset=["logo_url"])
)

top_drawers = (
    team_trends.groupby("Squad")["PKwon"].sum().nlargest(8).index.tolist()
)

fig, ax = plt.subplots(figsize=(11, 7))
palette = sns.color_palette("hsv", len(top_drawers))
for color, squad in zip(palette, top_drawers):
    sub = team_trends[team_trends["Squad"] == squad].sort_values("season_end")
    ax.plot(sub["season_end"], sub["PKwon"], color=color, linewidth=1.5, alpha=0.85)
    add_logos(ax, sub["season_end"], sub["PKwon"], sub["logo_url"], zoom=0.2)

ax.set_xticks(list(SEASONS))
ax.set_xticklabels([f"{y-1}/{str(y)[-2:]}" for y in SEASONS])
ax.set_title("Penalties Drawn by Top EPL Teams (2021/22 – 2025/26)", fontweight="bold")
ax.set_xlabel("Season")
ax.set_ylabel("Penalties Drawn")
fig.text(0.99, 0.01, "Data: FBref via soccerdata | Logos: ESPN", ha="right", fontsize=8, color="gray")
fig.tight_layout()
fig.savefig("p_trend.png", dpi=300)
plt.close(fig)

# ----------------------------------------------------------------------------
# 11. Charts 5 & 6 — per-90 leaderboards (drawn / conceded)
# ----------------------------------------------------------------------------
def per90_leaderboard(df, col, title, xlabel, cmap, out_path):
    top = (
        df[df["season_end"] == CURRENT_SEASON]
        .nlargest(20, col)
        .merge(epl_logos, on="Squad", how="left")
        .sort_values(col)
    )
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = sns.color_palette(cmap, len(top))
    ax.barh(top["player"], top[col], color=colors, alpha=0.9)
    for i, (val, url) in enumerate(zip(top[col], top["logo_url"])):
        ax.text(val, i, f" {val:.3f}", va="center", fontweight="bold", fontsize=8)
    add_logos(ax, [-top[col].max() * 0.03] * len(top), range(len(top)), top["logo_url"], zoom=0.18)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlabel)
    fig.text(0.99, 0.01, "Data: FBref via soccerdata | Logos: ESPN", ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# Note: rename "player" below to match soccerdata's actual player-name column.
per90_leaderboard(
    player_per90, "PKwon_per90",
    "Top 20 EPL Players: Penalties Drawn Per 90 Minutes",
    "Penalties Drawn per 90", "plasma", "p_player_p90.png",
)
per90_leaderboard(
    player_per90, "PKcon_per90",
    "Top 20 EPL Players: Penalties Conceded Per 90 Minutes",
    "Penalties Conceded per 90", "magma", "p_conceder_p90.png",
)

# ----------------------------------------------------------------------------
# 12. Historic deep dive — one player's career penalty trend
# ----------------------------------------------------------------------------
# all_players is already normalized (flattened, renamed, coerced to float) and
# covers every season already, so just filter it rather than re-scraping.
def player_career_trend(player_name: str, players_df: pd.DataFrame = all_players) -> pd.DataFrame:
    return (
        players_df[players_df["player"] == player_name][["season_end", "PKwon", "PKcon"]]
        .sort_values("season_end")
        .reset_index(drop=True)
    )


salah_career = player_career_trend("Mohamed Salah")

if not salah_career.empty:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(salah_career["season_end"], salah_career["PKwon"], color="#C8102E",
            marker="o", linewidth=2, markersize=8)
    for x, y in zip(salah_career["season_end"], salah_career["PKwon"]):
        ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontweight="bold")
    ax.set_title("Mohamed Salah: Career Penalties Drawn (Premier League)", fontweight="bold")
    ax.set_xlabel("Season")
    ax.set_ylabel("Penalties Drawn")
    fig.text(0.99, 0.01, "Data: FBref via soccerdata", ha="right", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig("p_career.png", dpi=300)
    plt.close(fig)
else:
    print("No career data found for 'Mohamed Salah' — check the exact name in all_players['player'].unique().")

# ----------------------------------------------------------------------------
# 13. Final combined dashboard
# ----------------------------------------------------------------------------
_dashboard_candidates = ["p_trend.png", "p_player_p90.png", "p_conceder_p90.png", "p_career.png"]
_dashboard_pngs = [p for p in _dashboard_candidates if Path(p).exists()]
if len(_dashboard_pngs) < len(_dashboard_candidates):
    _missing = set(_dashboard_candidates) - set(_dashboard_pngs)
    print(f"Skipping missing chart(s) in final dashboard: {sorted(_missing)}")

combine_pngs(
    _dashboard_pngs,
    (2, 2),
    "epl_penalty_dashboard.png",
    "EPL Penalty Analysis Dashboard (2021/22 – 2025/26)",
)

print("Done. See generated PNGs: p_scatter, p_team_bar, p_lollipop, "
      "epl_penalty_analysis, p_trend, p_player_p90, p_conceder_p90, "
      "p_career, epl_penalty_dashboard.")