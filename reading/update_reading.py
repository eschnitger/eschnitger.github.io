"""
Turn a StoryGraph library export into the public data for the Reading page.

    .venv/bin/python reading/update_reading.py [path/to/export.csv]

With no path, it uses the newest .csv in the folder that holds this repo
(~/githubrepo), so the raw export never has to sit inside the public repo.

Only books read or being read are kept, and only the columns the page shows.
Reviews, to-read and did-not-finish books are never copied.
"""

import sys
from pathlib import Path

import pandas as pd

KEEP_STATUS = ["read", "currently-reading"]
KEEP_COLUMNS = ["Title", "Authors", "Read Status", "Dates Read",
                "Last Date Read", "Read Count", "Star Rating"]

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "reading.csv"


def newest_export():
    exports = sorted(HERE.parents[1].glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not exports:
        raise SystemExit(f"No .csv export found in {HERE.parents[1]}")
    return exports[-1]


def main(source):
    export = pd.read_csv(source, dtype=str)
    missing = [c for c in KEEP_COLUMNS if c not in export.columns]
    if missing:
        raise SystemExit(f"This export is missing expected columns: {missing}")

    public = export.loc[export["Read Status"].isin(KEEP_STATUS), KEEP_COLUMNS]
    OUT.parent.mkdir(exist_ok=True)
    public.to_csv(OUT, index=False)

    counts = public["Read Status"].value_counts()
    print(f"From {Path(source).name}: wrote {len(public)} books to {OUT.relative_to(HERE.parent)} "
          f"({counts.get('read', 0)} read, {counts.get('currently-reading', 0)} currently reading).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else newest_export())
