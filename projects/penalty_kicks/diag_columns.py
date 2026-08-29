import soccerdata as sd
fbref = sd.FBref(leagues="ENG-Premier League", seasons="2526")
df = fbref.read_team_season_stats(stat_type="misc")
print("INDEX NAMES:", df.index.names)
print("COLUMNS:")
for c in df.columns:
    print("  ", c)