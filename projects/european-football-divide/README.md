# The European Football Divide

Interactive analysis of 33 seasons across Europe's top 5 leagues.

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests
python test_suite.py

# 4. Generate notebooks
python build_notebooks.py

# 5. Launch Jupyter
jupyter lab notebooks/
```

## Structure

- `_utils.py` - shared scraping, cleaning, charting code
- `build_notebooks.py` - regenerates all six notebooks
- `test_suite.py` - full offline test suite (no network needed)
- `notebooks/` - the six generated notebooks
- `data/` - cache directory (auto-created)

## Regenerating

Any change to notebook content goes through `build_notebooks.py`. After editing:

```bash
python build_notebooks.py --test    # runs tests, then regenerates
```

## First Run vs Cached

- **First run** of each notebook: scrapes ~33 Wikipedia pages, takes 1-2 minutes per league
- **Subsequent runs**: reads cached CSV, instant

Cache expires after 7 days. To force refresh, delete files in `data/`.