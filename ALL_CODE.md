# All project code (copy-paste version)
Create the folders shown in each heading, paste the code into a file with that exact name, then follow README.md.
Also create empty files `src/__init__.py` and `api/__init__.py`. The notebook is delivered separately as `flight_delay_end_to_end.ipynb` (put it in `notebooks/`).

---
## `README.md`

````markdown
# Flight Delay Predictor (US DOT data → LightGBM → FastAPI → web page)

Estimate the probability that a US domestic flight **arrives 15+ minutes late**, using Bureau of Transportation Statistics (BTS) on-time data.

```
BTS monthly zips ──download_data.py──▶ Parquet (small, fast)
                                          │  DuckDB (SQL on 100M+ rows)
                                          ▼
                      features.py (history + calendar features, no leakage)
                                          ▼
                      train.py ──▶ artifacts/ (model, lookups, metrics)
                                          ▼
 Browser (web/) ◀── JSON ──▶ FastAPI (api/main.py) ── predictor.py loads artifacts
                                          ▼
                              Docker ▶ Hugging Face Spaces / Cloud Run / Render
```

## 0. Dataset (official source, free, no login)

* Download page (choose **"Reporting Carrier On-Time Performance (1987-present)"** → *Download*):
  **https://www.transtats.bts.gov/Tables.asp?QO_VQ=EFD**
* Dataset description: https://transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EFD
* Direct monthly zip pattern used by `src/download_data.py` (replace `{year}` and `{month}`, month has no leading zero):
  `https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip`
  e.g. January 2023 → `..._1987_present_2023_1.zip`

If the script cannot download a month (site slow, certificate or format change), download the CSV by hand from the page above
(select all fields, or at least the ones listed in `src/download_data.py`) and drop it in `data/raw/` with a name ending in `_YEAR_MONTH.csv`
(e.g. `ontime_2023_1.csv`). The script will convert it.

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get data

**Option A: try everything in 5 minutes with fake data (no download):**
```bash
python scripts/make_sample_data.py --years 2021 2022 2023 2024 --rows-per-year 150000
```
**Option B: real data (recommended for your portfolio):** about 7 years ≈ 40M+ rows, ~2 GB as Parquet, takes a while.
```bash
python -m src.download_data --start-year 2018 --end-year 2024
# smoke test first:  python -m src.download_data --start-year 2023 --end-year 2023 --months 1 2 3
```
You need **at least 4 distinct years** (1 for history, ≥1 train, 1 validation, 1 test).
Do not mix fake and real files in `data/parquet/`.

## 3. Learn the data (notebook)

```bash
jupyter notebook notebooks/flight_delay_end_to_end.ipynb
```

## 4. Train

```bash
python -m src.train --quick                               # ~10 seconds, checks the pipeline
python -m src.train --max-train-rows 5000000 --memory-limit 8GB   # real run
```
This writes `artifacts/model.joblib`, `artifacts/lookups.joblib`, `artifacts/metrics.json`.
Read `metrics.json`: compare the model to the baselines. Real data with schedule + history features typically gives a modest ROC-AUC
(roughly 0.65 to 0.72). That is normal: weather and the inbound aircraft's delay are the missing signals. Report your own numbers honestly.

## 5. Run the app locally (backend + frontend together)

```bash
uvicorn api.main:app --reload --port 8000
```
Open http://localhost:8000 (web page) and http://localhost:8000/docs (interactive API docs).
Tests: `pytest -q`.

### How the frontend and backend are connected

* `api/main.py` creates the FastAPI app and defines JSON routes under `/api/...`.
* The last line, `app.mount("/", StaticFiles(directory="web", html=True))`, makes the **same server** also serve `web/index.html`, `style.css` and `app.js`.
* `web/app.js` calls the API with relative URLs, e.g. `fetch("/api/predict", {method: "POST", body: JSON.stringify({...})})`.
  Same origin → no CORS configuration needed, one deployment, one URL.
* Flow: page loads → `GET /api/meta` fills the airport list → choosing an airport calls `GET /api/destinations?origin=ORD` →
  choosing a destination calls `GET /api/carriers?origin=ORD&dest=LAX` → pressing the button calls `POST /api/predict` → the response
  (`probability`, `risk`, `reasons`, `context`) is drawn on the boarding-pass result and the chart.
* **If you host the web page on a different domain** (e.g. Netlify/Vercel) and the API elsewhere: in `web/app.js` prefix every
  path with your API URL (`const API = "https://your-api.example.com"`), and on the server set the env var
  `ALLOWED_ORIGINS=https://your-site.netlify.app` (this turns on CORS).

## 6. Deploy

The image only contains code + `artifacts/` (not the data), so it is small. **Commit `artifacts/`** (it is not git-ignored).

### Option 1: Hugging Face Spaces (free, easiest)
1. Create an account → **New Space** → SDK: **Docker** → blank template → public.
2. In your project folder:
   ```bash
   git lfs install
   git lfs track "*.joblib"          # model files are > 10 MB
   git add .gitattributes
   git init && git add . && git commit -m "flight delay app"
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push --force space HEAD:main
   ```
3. Add this to the top of `README.md` (Spaces reads it) if the Space asks for it:
   ```
   ---
   title: Flight Delay Predictor
   sdk: docker
   app_port: 7860
   ---
   ```
4. Wait for the build; your app is at `https://<your-username>-<space-name>.hf.space`.

### Option 2: Google Cloud Run (real cloud skills, generous free tier)
```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud run deploy flight-delay --source . --region us-central1 --allow-unauthenticated --memory 1Gi
```
Cloud Run builds the Dockerfile, injects `$PORT`, and prints a public URL.

### Option 3: Render / Railway / Fly.io
Push the repo to GitHub → create a **Web Service** from the repo → environment *Docker* → deploy. Nothing else to configure.

### Test the Docker image locally first
```bash
docker build -t flight-delay .
docker run --rm -p 7860:7860 flight-delay      # http://localhost:7860
```

### CI
`.github/workflows/ci.yml` runs on every push: builds sample data, trains a tiny model, runs the tests.

## 7. Project layout

```
src/config.py          paths, feature lists, constants
src/download_data.py   BTS zip → Parquet
src/features.py        SQL feature engineering + calendar features + time split
src/train.py           baselines, LightGBM, calibration, saves artifacts
src/lookups.py         small lookup tables for serving
src/predictor.py       inputs → features → probability + reasons
api/main.py            FastAPI routes + serves web/
web/                   index.html, style.css, app.js
notebooks/             step-by-step learning notebook
scripts/               fake data generator, local run helper
tests/test_api.py      API tests
Dockerfile             production image
```

## 8. Concepts to be able to explain in an interview

* **Leakage**: only pre-departure information is used; history features use the previous year only.
* **Time-based split**: train → validation → test are consecutive years, never random.
* **Metrics**: ROC-AUC, PR-AUC, Brier score, calibration. Accuracy is misleading with ~20% positives.
* **Training/serving parity**: `features.py` and `predictor.py` share the same calendar and encoding code.
* **Honest limits**: no live weather, no inbound-aircraft status; a "30%" is a probability, not a promise.

## 9. Upgrade ideas
Weather features (Meteostat/NOAA), aircraft-rotation delay (previous flight of the same tail number), Optuna tuning, MLflow tracking,
PySpark/Dask feature pipeline, monthly retraining (Airflow/Prefect), a drift-monitoring page, a second page with airline/airport dashboards.
````

---
## `requirements.txt`

```text
# --- data + modelling (training machine) ---
duckdb>=1.0
pandas>=2.0
pyarrow>=14
numpy>=1.26
scikit-learn>=1.4
lightgbm>=4.3
holidays>=0.50
requests>=2.31
joblib>=1.3
matplotlib>=3.8
jupyter>=1.0
# --- serving (also used inside Docker) ---
fastapi>=0.110
uvicorn[standard]>=0.29
# --- tests ---
pytest>=8.0
httpx>=0.27
```

---
## `requirements-serve.txt`

```text
# Minimal packages needed to SERVE the model (keeps the Docker image small)
fastapi>=0.110
uvicorn[standard]>=0.29
pandas>=2.0
pyarrow>=14
numpy>=1.26
scikit-learn>=1.4
lightgbm>=4.3
holidays>=0.50
joblib>=1.3
```

---
## `.gitignore`

```
__pycache__/
*.pyc
.venv/
.ipynb_checkpoints/
.pytest_cache/
mlruns/
# raw + converted data is big: never commit it
data/
# NOTE: artifacts/ (trained model + lookups) IS committed so deployment can use it
```

---
## `.dockerignore`

```
.git
.venv
data
notebooks
tests
mlruns
__pycache__
*.pyc
.ipynb_checkpoints
```

---
## `Dockerfile`

```
# Small image that serves the API + web page. Training happens on your own machine, not here.
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only code + trained artifacts go into the image (see .dockerignore: data/ is excluded)
COPY src ./src
COPY api ./api
COPY web ./web
COPY artifacts ./artifacts

# Hugging Face Spaces expects 7860, Cloud Run injects $PORT. This works for both.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
```

---
## `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Install
        run: pip install -r requirements.txt
      - name: Make sample data and train a tiny model
        run: |
          python scripts/make_sample_data.py --rows-per-year 60000
          python -m src.train --quick
      - name: Run tests
        run: pytest -q
```

---
## `src/config.py`

```python
"""Central place for paths and constants used by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"
ARTIFACT_DIR = ROOT / "artifacts"

# Official monthly file pattern from the Bureau of Transportation Statistics (BTS).
BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# "Delayed" = arrived 15+ minutes after the scheduled time (the official DOT definition).
DELAY_THRESHOLD_MIN = 15

# Smoothing strength for historical delay rates (acts like m "pseudo-flights" at the global rate).
SMOOTHING_M = 20

# Feature lists: these are the ONLY columns the model sees.
CAT_FEATURES = ["carrier", "origin", "dest"]
NUM_FEATURES = [
    "month", "dow", "day", "dep_hour", "is_weekend",
    "is_holiday", "days_to_holiday",
    "distance", "crs_elapsed",
    "carrier_rate", "origin_rate", "dest_rate", "route_rate",
    "origin_hour_rate", "origin_hour_flights",
]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "y"
```

---
## `src/download_data.py`

```python
"""
Step 1: download BTS monthly On-Time Performance files and convert each one to a small Parquet file.

Why convert right away?
  * One month of raw CSV is ~250 MB; 7 years is ~20 GB.
  * We only need ~20 of the 100+ columns, and Parquet is compressed + columnar,
    so the same data shrinks to ~1-2 GB and DuckDB can query it in seconds.

Usage:
    python -m src.download_data --start-year 2018 --end-year 2024
    python -m src.download_data --start-year 2024 --end-year 2024 --months 1 2 3   # tiny test
"""
import argparse
import io
import time
import zipfile

import duckdb
import requests

from src.config import BTS_URL, PARQUET_DIR, RAW_DIR

# Keep only the columns we need. Some are ONLY used to build history (DepDelay/ArrDelay),
# never as inputs to the model for the flight being predicted.
CONVERT_SQL = """
COPY (
  SELECT
    CAST(TRY_CAST(FlightDate AS DATE) AS DATE)                    AS flight_date,
    TRY_CAST(Year AS INTEGER)                                     AS Year,
    TRY_CAST(Month AS INTEGER)                                    AS month,
    TRY_CAST(DayofMonth AS INTEGER)                               AS day,
    TRY_CAST(DayOfWeek AS INTEGER)                                AS dow,
    Reporting_Airline                                             AS carrier,
    Tail_Number                                                   AS tail_number,
    TRY_CAST(Flight_Number_Reporting_Airline AS INTEGER)          AS flight_number,
    Origin                                                        AS origin,
    OriginCityName                                                AS origin_city,
    Dest                                                          AS dest,
    DestCityName                                                  AS dest_city,
    TRY_CAST(CRSDepTime AS INTEGER)                               AS crs_dep,
    TRY_CAST(CRSArrTime AS INTEGER)                               AS crs_arr,
    TRY_CAST(CRSElapsedTime AS DOUBLE)                            AS crs_elapsed,
    TRY_CAST(Distance AS DOUBLE)                                  AS distance,
    TRY_CAST(Cancelled AS DOUBLE)                                 AS cancelled,
    TRY_CAST(Diverted AS DOUBLE)                                  AS diverted,
    TRY_CAST(DepDelay AS DOUBLE)                                  AS dep_delay,
    TRY_CAST(ArrDelay AS DOUBLE)                                  AS arr_delay,
    TRY_CAST(ArrDel15 AS DOUBLE)                                  AS arr_del15
  FROM read_csv('{csv_path}', header=true, all_varchar=true, ignore_errors=true)
) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
"""


def download_month(year: int, month: int, retries: int = 3) -> bytes:
    url = BTS_URL.format(year=year, month=month)
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            return r.content
        except requests.RequestException as e:
            print(f"  attempt {attempt}/{retries} failed: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(
        f"Could not download {url}. Download it manually from "
        "https://www.transtats.bts.gov (Reporting Carrier On-Time Performance) and put the "
        f"CSV into {RAW_DIR}/ - the script will pick up any CSV named *_{year}_{month}.csv"
    )


def process_month(year: int, month: int) -> None:
    out_path = PARQUET_DIR / f"flights_{year}_{month:02d}.parquet"
    if out_path.exists():
        print(f"[skip] {out_path.name} already exists")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    # Use a manually downloaded CSV if present, otherwise download the zip.
    manual = list(RAW_DIR.glob(f"*_{year}_{month}.csv"))
    if manual:
        csv_path = manual[0]
        cleanup = False
    else:
        print(f"[download] {year}-{month:02d}")
        data = download_month(year, month)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
            zf.extract(name, RAW_DIR)
            csv_path = RAW_DIR / name
        cleanup = True

    print(f"[convert] {csv_path.name} -> {out_path.name}")
    con = duckdb.connect()
    con.execute(CONVERT_SQL.format(csv_path=csv_path.as_posix(), out_path=out_path.as_posix()))
    con.close()
    if cleanup:
        csv_path.unlink()  # delete the big CSV, keep only the Parquet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--end-year", type=int, required=True)
    ap.add_argument("--months", type=int, nargs="*", default=list(range(1, 13)))
    args = ap.parse_args()

    for year in range(args.start_year, args.end_year + 1):
        for month in args.months:
            try:
                process_month(year, month)
            except Exception as e:  # keep going if a not-yet-published month is missing
                print(f"[warn] {year}-{month:02d} failed: {e}")


if __name__ == "__main__":
    main()
```

---
## `src/features.py`

```python
"""
Feature engineering - used by BOTH training and the API, so there is no training/serving skew.

Key ideas
---------
1. Only information known BEFORE departure is used (no DepDelay, TaxiOut, ArrTime ... = no leakage).
2. "History" features (delay rates, congestion) for a flight in year Y are computed from year Y-1 only.
   That way the model never sees its own label inside a feature, and at serving time we simply use
   the most recent full year as "last year".
"""
from __future__ import annotations

import datetime as dt

import duckdb
import holidays
import pandas as pd

from src.config import CAT_FEATURES, FEATURES, NUM_FEATURES, PARQUET_DIR, SMOOTHING_M, TARGET

HOLIDAY_CAP = 14  # days_to_holiday is capped so far-away holidays don't matter


# --------------------------------------------------------------------------- calendar features
def build_calendar(start_year: int, end_year: int) -> pd.DataFrame:
    """One row per date with is_holiday and days_to_holiday (distance to nearest US federal holiday)."""
    us = holidays.US(years=range(start_year - 1, end_year + 2))
    hol = sorted(pd.Timestamp(d) for d in us.keys())
    hol_idx = pd.DatetimeIndex(hol)
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31")
    pos = hol_idx.searchsorted(dates)
    prev_d = hol_idx[(pos - 1).clip(0, len(hol_idx) - 1)]
    next_d = hol_idx[pos.clip(0, len(hol_idx) - 1)]
    dist = pd.concat([pd.Series((dates - prev_d).days), pd.Series((next_d - dates).days)], axis=1).abs().min(axis=1)
    return pd.DataFrame({
        "flight_date": dates.date,
        "is_holiday": [int(d.date() in us) for d in dates],
        "days_to_holiday": dist.clip(upper=HOLIDAY_CAP).astype(int).values,
    })


def calendar_row(d: dt.date) -> dict:
    row = build_calendar(d.year, d.year)
    r = row[row["flight_date"] == d].iloc[0]
    return {"is_holiday": int(r["is_holiday"]), "days_to_holiday": int(r["days_to_holiday"])}


# --------------------------------------------------------------------------- DuckDB helpers
def connect(memory_limit: str | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    if memory_limit:
        con.execute(f"PRAGMA memory_limit='{memory_limit}'")
    glob = (PARQUET_DIR / "flights_*.parquet").as_posix()
    con.execute(f"CREATE VIEW flights AS SELECT * FROM read_parquet('{glob}')")
    return con


def available_years(con) -> list[int]:
    return [r[0] for r in con.execute("SELECT DISTINCT Year FROM flights ORDER BY 1").fetchall()]


def create_base(con) -> None:
    """Completed, non-cancelled flights with a known label. This is the table everything builds on."""
    con.execute("""
        CREATE OR REPLACE TEMP TABLE base AS
        SELECT flight_date, Year, month, day, dow, carrier, origin, origin_city, dest, dest_city,
               (crs_dep // 100) % 24 AS dep_hour, distance, crs_elapsed,
               arr_del15 AS y
        FROM flights
        WHERE cancelled = 0 AND diverted = 0 AND arr_del15 IS NOT NULL
          AND distance IS NOT NULL AND crs_elapsed IS NOT NULL
    """)


def create_stats(con) -> None:
    """Per-year aggregates ('how late were flights in this group that year?')."""
    con.execute("CREATE OR REPLACE TEMP TABLE st_global AS SELECT Year, avg(y) AS g FROM base GROUP BY Year")
    groups = {
        "st_carrier": "carrier",
        "st_origin": "origin",
        "st_dest": "dest",
        "st_route": "origin, dest",
        "st_oh": "origin, dep_hour",
    }
    for name, keys in groups.items():
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE {name} AS
            SELECT Year, {keys}, sum(y) AS s, count(*) AS n, count(DISTINCT flight_date) AS days
            FROM base GROUP BY Year, {keys}
        """)


def _rate(alias: str) -> str:
    """Smoothed delay rate: (delays + m * global) / (flights + m). Small groups shrink toward the global rate."""
    return f"(({alias}.s + {SMOOTHING_M} * g.g) / ({alias}.n + {SMOOTHING_M}))"


def build_dataset(con, years: list[int], max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Return a model-ready DataFrame for the given years (optionally randomly sampled)."""
    y_min, y_max = min(years), max(years)
    cal = build_calendar(y_min, y_max)
    con.register("cal", cal)
    sample = f"USING SAMPLE reservoir({int(max_rows)} ROWS) REPEATABLE ({seed})" if max_rows else ""
    ylist = ",".join(str(int(y)) for y in years)
    sql = f"""
        SELECT
            b.flight_date, b.Year, b.month, b.day, b.dow, b.dep_hour,
            CAST(b.dow IN (6, 7) AS INTEGER) AS is_weekend,
            cal.is_holiday, cal.days_to_holiday,
            b.carrier, b.origin, b.dest, b.distance, b.crs_elapsed,
            {_rate('c')}  AS carrier_rate,
            {_rate('o')}  AS origin_rate,
            {_rate('d')}  AS dest_rate,
            {_rate('r')}  AS route_rate,
            {_rate('oh')} AS origin_hour_rate,
            oh.n * 1.0 / oh.days AS origin_hour_flights,
            b.y AS {TARGET}
        FROM base b
        JOIN cal ON cal.flight_date = b.flight_date
        JOIN st_global g ON g.Year = b.Year - 1
        LEFT JOIN st_carrier c ON c.Year = b.Year - 1 AND c.carrier = b.carrier
        LEFT JOIN st_origin  o ON o.Year = b.Year - 1 AND o.origin  = b.origin
        LEFT JOIN st_dest    d ON d.Year = b.Year - 1 AND d.dest    = b.dest
        LEFT JOIN st_route   r ON r.Year = b.Year - 1 AND r.origin  = b.origin AND r.dest = b.dest
        LEFT JOIN st_oh     oh ON oh.Year = b.Year - 1 AND oh.origin = b.origin AND oh.dep_hour = b.dep_hour
        WHERE b.Year IN ({ylist})
        {sample}
    """
    df = con.execute(sql).df()
    return df


def make_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_FEATURES}


def apply_categories(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Give train/val/test/serving the exact same categorical encoding. Unknown values become NaN."""
    out = df.copy()
    for c in CAT_FEATURES:
        out[c] = pd.Categorical(out[c], categories=categories[c])
    return out


def split_years(years: list[int]) -> dict[str, list[int]]:
    """Time-based split. First year is only used to build history; last = test; second last = validation."""
    if len(years) < 4:
        raise ValueError(f"Need at least 4 distinct years of data (found {years}).")
    return {"history_only": [years[0]], "train": years[1:-2], "val": [years[-2]], "test": [years[-1]]}


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES]
```

---
## `src/lookups.py`

```python
"""
Small lookup tables the API needs at prediction time (the API never touches the 100M-row dataset).
Built from the most recent full year, which plays the role of 'last year' for future flights.
"""
import joblib

from src.config import SMOOTHING_M


def build_lookups(con, stats_year: int, out_path) -> dict:
    g = con.execute("SELECT g FROM st_global WHERE Year = ?", [stats_year]).fetchone()[0]

    def rates(table, keys):
        cols = ", ".join(keys)
        rows = con.execute(
            f"SELECT {cols}, (s + {SMOOTHING_M} * {g}) / (n + {SMOOTHING_M}) AS r, n * 1.0 / days AS per_day FROM {table} WHERE Year = ?",
            [stats_year]).fetchall()
        k = len(keys)
        rate = {(r[0] if k == 1 else tuple(r[:k])): r[k] for r in rows}
        per_day = {(r[0] if k == 1 else tuple(r[:k])): r[k + 1] for r in rows}
        return rate, per_day

    carrier_rate, _ = rates("st_carrier", ["carrier"])
    origin_rate, _ = rates("st_origin", ["origin"])
    dest_rate, _ = rates("st_dest", ["dest"])
    route_rate, _ = rates("st_route", ["origin", "dest"])
    oh_rate, oh_flights = rates("st_oh", ["origin", "dep_hour"])

    # Valid (origin, dest, carrier) combinations with typical distance / scheduled duration
    routes = con.execute("""
        SELECT origin, dest, carrier, count(*) AS n,
               median(distance) AS distance, median(crs_elapsed) AS crs_elapsed
        FROM base WHERE Year = ? GROUP BY 1, 2, 3 HAVING count(*) >= 30
    """, [stats_year]).df()

    airports = {}
    for col_code, col_city in (("origin", "origin_city"), ("dest", "dest_city")):
        df = con.execute(f"SELECT DISTINCT {col_code} AS code, {col_city} AS city FROM base WHERE Year = ?",
                         [stats_year]).df()
        airports.update(dict(zip(df["code"], df["city"])))

    lookups = {
        "stats_year": int(stats_year),
        "global_rate": float(g),
        "carrier_rate": carrier_rate, "origin_rate": origin_rate, "dest_rate": dest_rate,
        "route_rate": route_rate, "origin_hour_rate": oh_rate, "origin_hour_flights": oh_flights,
        "routes": routes, "airports": airports,
    }
    joblib.dump(lookups, out_path, compress=3)
    return lookups
```

---
## `src/train.py`

```python
"""
Step 2: build features, train LightGBM, evaluate on a time-based test set, calibrate, and save artifacts.

    python -m src.train                        # defaults (5M training rows)
    python -m src.train --max-train-rows 2000000 --memory-limit 6GB
    python -m src.train --quick                # tiny/fast run for testing
"""
import argparse
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src import features as F
from src.config import ARTIFACT_DIR, CAT_FEATURES, FEATURES, TARGET
from src.lookups import build_lookups


def evaluate(y, p) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train-rows", type=int, default=5_000_000)
    ap.add_argument("--max-eval-rows", type=int, default=2_000_000)
    ap.add_argument("--memory-limit", type=str, default=None, help="e.g. 6GB (DuckDB)")
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--quick", action="store_true", help="fast run for testing the pipeline")
    a = ap.parse_args()
    if a.quick:
        a.max_train_rows, a.max_eval_rows, a.n_estimators, a.num_leaves = 200_000, 100_000, 300, 31

    t0 = time.time()
    con = F.connect(a.memory_limit)
    years = F.available_years(con)
    split = F.split_years(years)
    print("Years found:", years)
    print("Split      :", split)

    F.create_base(con)
    F.create_stats(con)

    print("Building datasets ...")
    train = F.build_dataset(con, split["train"], a.max_train_rows)
    val = F.build_dataset(con, split["val"], a.max_eval_rows)
    test = F.build_dataset(con, split["test"], a.max_eval_rows)
    print(f"rows -> train {len(train):,}  val {len(val):,}  test {len(test):,}")

    categories = F.make_categories(train)
    X_train, y_train = F.apply_categories(train, categories)[FEATURES], train[TARGET]
    X_val, y_val = F.apply_categories(val, categories)[FEATURES], val[TARGET]
    X_test, y_test = F.apply_categories(test, categories)[FEATURES], test[TARGET]

    # ------------------------------------------------------------------ baselines
    base_rate = float(y_train.mean())
    baselines = {
        "always_predict_base_rate": evaluate(y_test, np.full(len(y_test), base_rate)),
        "route_rate_only": evaluate(y_test, X_test["route_rate"].fillna(base_rate)),
        "origin_hour_rate_only": evaluate(y_test, X_test["origin_hour_rate"].fillna(base_rate)),
    }

    # ------------------------------------------------------------------ model
    print("Training LightGBM ...")
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=a.n_estimators,
        learning_rate=a.learning_rate,
        num_leaves=a.num_leaves,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        cat_smooth=50,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        categorical_feature=CAT_FEATURES,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    # ------------------------------------------------------------------ calibration (fit on validation)
    raw_val = model.predict_proba(X_val)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(raw_val, y_val)

    raw_test = model.predict_proba(X_test)[:, 1]
    cal_test = calibrator.predict(raw_test)
    metrics = {
        "test_raw": evaluate(y_test, raw_test),
        "test_calibrated": evaluate(y_test, cal_test),
        "baselines_test": baselines,
        "base_rate_train": base_rate,
        "base_rate_test": float(y_test.mean()),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "years": split,
        "trained_at_seconds": round(time.time() - t0, 1),
    }
    importance = dict(sorted(
        zip(FEATURES, model.booster_.feature_importance(importance_type="gain").tolist()),
        key=lambda kv: -kv[1]))
    metrics["feature_importance_gain"] = importance

    # ------------------------------------------------------------------ save
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "calibrator": calibrator, "categories": categories,
         "features": FEATURES, "base_rate": float(y_test.mean())},
        ARTIFACT_DIR / "model.joblib", compress=3,
    )
    build_lookups(con, stats_year=years[-1], out_path=ARTIFACT_DIR / "lookups.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n================ RESULTS (test year %s) ================" % split["test"])
    print(json.dumps({k: metrics[k] for k in ("test_raw", "test_calibrated", "baselines_test")}, indent=2))
    print("Top features:", list(importance)[:8])
    print(f"Done in {time.time() - t0:.0f}s. Artifacts saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
```

---
## `src/predictor.py`

```python
"""
Serving-time logic: turn a few user inputs into the same feature row used in training,
then return probability + plain-English reasons. No web code here, so it is easy to test.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, FEATURES
from src.features import apply_categories, calendar_row

CARRIER_NAMES = {
    "AA": "American Airlines", "DL": "Delta Air Lines", "UA": "United Airlines", "WN": "Southwest Airlines",
    "B6": "JetBlue", "AS": "Alaska Airlines", "NK": "Spirit Airlines", "F9": "Frontier Airlines",
    "G4": "Allegiant Air", "HA": "Hawaiian Airlines", "YX": "Republic Airways", "OO": "SkyWest Airlines",
    "MQ": "Envoy Air", "9E": "Endeavor Air", "OH": "PSA Airlines", "YV": "Mesa Airlines",
    "QX": "Horizon Air", "EV": "ExpressJet", "CP": "Compass Airlines", "VX": "Virgin America",
    "US": "US Airways", "NW": "Northwest Airlines", "FL": "AirTran Airways", "XE": "ExpressJet",
    "SY": "Sun Country Airlines", "G7": "GoJet Airlines", "ZW": "Air Wisconsin", "C5": "CommutAir",
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def _hour_label(h: int) -> str:
    return f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}"


class RouteNotFound(ValueError):
    pass


class Predictor:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        bundle = joblib.load(artifact_dir / "model.joblib")
        self.model = bundle["model"]
        self.booster = self.model.booster_
        self.calibrator = bundle["calibrator"]
        self.categories = bundle["categories"]
        self.base_rate = bundle["base_rate"]
        self.lk = joblib.load(artifact_dir / "lookups.joblib")
        mpath = artifact_dir / "metrics.json"
        self.metrics = json.loads(mpath.read_text()) if mpath.exists() else {}
        routes = self.lk["routes"]
        self.routes = routes.set_index(["origin", "dest", "carrier"]).sort_index()
        self._dests = routes.groupby("origin")["dest"].unique().apply(lambda a: sorted(a.tolist())).to_dict()
        self._carriers = routes.groupby(["origin", "dest"])["carrier"].unique().apply(
            lambda a: sorted(a.tolist())).to_dict()

    # ------------------------------------------------------------------ metadata for the UI
    def carrier_name(self, code: str) -> str:
        return CARRIER_NAMES.get(code, code)

    def origins(self) -> list[dict]:
        return [{"code": o, "city": self.lk["airports"].get(o, o)} for o in sorted(self._dests)]

    def destinations(self, origin: str) -> list[dict]:
        return [{"code": d, "city": self.lk["airports"].get(d, d)} for d in self._dests.get(origin, [])]

    def carriers(self, origin: str, dest: str) -> list[dict]:
        return [{"code": c, "name": self.carrier_name(c)} for c in self._carriers.get((origin, dest), [])]

    def meta(self) -> dict:
        m = self.metrics.get("test_calibrated", {})
        return {
            "base_rate": self.base_rate,
            "stats_year": self.lk["stats_year"],
            "test_years": self.metrics.get("years", {}).get("test"),
            "roc_auc": m.get("roc_auc"),
            "pr_auc": m.get("pr_auc"),
            "rows_train": self.metrics.get("rows", {}).get("train"),
        }

    # ------------------------------------------------------------------ prediction
    def _row(self, date: dt.date, hour: int, carrier: str, origin: str, dest: str) -> dict:
        key = (origin, dest, carrier)
        if key not in self.routes.index:
            raise RouteNotFound(f"No flights found for {carrier} from {origin} to {dest} in our data.")
        r = self.routes.loc[key]
        lk = self.lk
        g = lk["global_rate"]
        cal = calendar_row(date)
        return {
            "carrier": carrier, "origin": origin, "dest": dest,
            "month": date.month, "dow": date.isoweekday(), "day": date.day, "dep_hour": hour,
            "is_weekend": int(date.isoweekday() >= 6),
            "is_holiday": cal["is_holiday"], "days_to_holiday": cal["days_to_holiday"],
            "distance": float(r["distance"]), "crs_elapsed": float(r["crs_elapsed"]),
            "carrier_rate": lk["carrier_rate"].get(carrier, g),
            "origin_rate": lk["origin_rate"].get(origin, g),
            "dest_rate": lk["dest_rate"].get(dest, g),
            "route_rate": lk["route_rate"].get((origin, dest), np.nan),
            "origin_hour_rate": lk["origin_hour_rate"].get((origin, hour), np.nan),
            "origin_hour_flights": lk["origin_hour_flights"].get((origin, hour), np.nan),
        }

    def predict(self, date: dt.date, dep_time: str, carrier: str, origin: str, dest: str) -> dict:
        hour = int(dep_time.split(":")[0])
        row = self._row(date, hour, carrier, origin, dest)
        X = apply_categories(pd.DataFrame([row]), self.categories)[FEATURES]

        raw = float(self.model.predict_proba(X)[0, 1])
        prob = float(self.calibrator.predict([raw])[0])

        contrib = self.booster.predict(X, pred_contrib=True)[0][:-1]  # last value is the bias term
        reasons = self._reasons(row, contrib)

        hourly = [self.lk["origin_hour_rate"].get((origin, h)) for h in range(24)]
        return {
            "probability": prob,
            "base_rate": self.base_rate,
            "risk": self._risk(prob),
            "reasons": reasons,
            "context": {
                "hourly_origin_rates": hourly,
                "selected_hour": hour,
                "route_rate": row["route_rate"] if not np.isnan(row["route_rate"]) else None,
                "origin_rate": row["origin_rate"], "dest_rate": row["dest_rate"],
                "carrier_rate": row["carrier_rate"], "stats_year": self.lk["stats_year"],
                "distance": row["distance"], "scheduled_minutes": row["crs_elapsed"],
            },
        }

    def _risk(self, p: float) -> str:
        b = self.base_rate
        if p < b * 0.8:
            return "low"
        if p < b * 1.4:
            return "medium"
        return "high"

    # ------------------------------------------------------------------ explanations
    def _reasons(self, row: dict, contrib: np.ndarray, top_k: int = 5) -> list[dict]:
        yr = self.lk["stats_year"]
        pct = lambda v: "n/a" if v is None or np.isnan(v) else f"{v:.0%}"
        text = {
            "dep_hour": f"Scheduled to leave around {_hour_label(row['dep_hour'])}",
            "month": f"Travelling in {MONTHS[row['month'] - 1]}",
            "dow": f"Flying on a {DAYS[row['dow'] - 1]}",
            "day": f"Day {row['day']} of the month",
            "is_weekend": "Weekend travel" if row["is_weekend"] else "Weekday travel",
            "is_holiday": "Flying on a public holiday" if row["is_holiday"] else "Not a public holiday",
            "days_to_holiday": f"{row['days_to_holiday']} day(s) from the nearest public holiday",
            "distance": f"Flight distance of {row['distance']:,.0f} miles",
            "crs_elapsed": f"Scheduled flight time of {row['crs_elapsed']:.0f} minutes",
            "carrier_rate": f"{self.carrier_name(row['carrier'])} flights were late {pct(row['carrier_rate'])} of the time in {yr}",
            "origin_rate": f"Flights leaving {row['origin']} were late {pct(row['origin_rate'])} of the time in {yr}",
            "dest_rate": f"Flights arriving at {row['dest']} were late {pct(row['dest_rate'])} of the time in {yr}",
            "route_rate": f"{row['origin']} to {row['dest']} flights were late {pct(row['route_rate'])} of the time in {yr}",
            "origin_hour_rate": f"{row['origin']} departures around {_hour_label(row['dep_hour'])} were late {pct(row['origin_hour_rate'])} of the time in {yr}",
            "origin_hour_flights": f"About {0 if np.isnan(row['origin_hour_flights']) else row['origin_hour_flights']:.0f} flights per day leave {row['origin']} at this hour",
            "carrier": f"Airline: {self.carrier_name(row['carrier'])}",
            "origin": f"Departure airport: {row['origin']}",
            "dest": f"Arrival airport: {row['dest']}",
        }
        total = float(np.abs(contrib).sum()) or 1.0
        order = np.argsort(-np.abs(contrib))[:top_k]
        return [{
            "feature": FEATURES[i],
            "text": text.get(FEATURES[i], FEATURES[i]),
            "direction": "up" if contrib[i] > 0 else "down",
            "weight": round(float(abs(contrib[i]) / total), 3),
        } for i in order]
```

---
## `api/main.py`

```python
"""
FastAPI backend. Serves:
    /api/*   -> JSON endpoints used by the web page
    /        -> the web page itself (the ./web folder)
Because both come from the same server, the browser needs no CORS setup.

Run locally:  uvicorn api.main:app --reload --port 8000
"""
import datetime as dt
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predictor import Predictor, RouteNotFound

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = Predictor()  # load the model ONCE at startup, not per request
    yield
    state.clear()


app = FastAPI(title="Flight Delay Predictor", version="1.0.0", lifespan=lifespan)

# Only needed if you host the web page on a DIFFERENT domain than the API.
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


class FlightRequest(BaseModel):
    flight_date: dt.date
    dep_time: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$", examples=["18:45"])
    carrier: str = Field(min_length=2, max_length=3)
    origin: str = Field(min_length=3, max_length=3)
    dest: str = Field(min_length=3, max_length=3)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    p: Predictor = state["predictor"]
    return {**p.meta(), "airports": p.origins()}


@app.get("/api/destinations")
def destinations(origin: str = Query(min_length=3, max_length=3)):
    return state["predictor"].destinations(origin.upper())


@app.get("/api/carriers")
def carriers(origin: str = Query(min_length=3, max_length=3), dest: str = Query(min_length=3, max_length=3)):
    return state["predictor"].carriers(origin.upper(), dest.upper())


@app.post("/api/predict")
def predict(req: FlightRequest):
    try:
        return state["predictor"].predict(
            req.flight_date, req.dep_time, req.carrier.upper(), req.origin.upper(), req.dest.upper())
    except RouteNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))


# Must be mounted LAST so it doesn't shadow the /api routes.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
```

---
## `web/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Will my flight be late?</title>
  <meta name="description" content="Estimate the chance a US domestic flight arrives 15 or more minutes late, based on years of Department of Transportation data.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/style.css">
</head>
<body>
  <header class="site-head">
    <h1>Will my flight be late?</h1>
    <p>Pick a US domestic flight and see how likely it is to land 15 minutes or more behind schedule. The estimate comes from a model trained on millions of Department of Transportation flight records.</p>
  </header>

  <main>
    <section class="pass" aria-labelledby="pass-title">
      <form id="flight-form" class="pass-main" novalidate>
        <h2 id="pass-title" class="pass-title">Your flight</h2>

        <div class="field">
          <label for="origin">From</label>
          <select id="origin" required><option value="">Loading airports...</option></select>
        </div>

        <div class="field">
          <label for="dest">To</label>
          <select id="dest" required disabled><option value="">Choose a departure airport first</option></select>
        </div>

        <div class="field">
          <label for="carrier">Airline</label>
          <select id="carrier" required disabled><option value="">Choose a destination first</option></select>
        </div>

        <div class="field-row">
          <div class="field">
            <label for="date">Date</label>
            <input id="date" type="date" required>
          </div>
          <div class="field">
            <label for="time">Scheduled departure</label>
            <input id="time" type="time" value="09:00" required>
          </div>
        </div>

        <button id="submit" type="submit" disabled>Check delay risk</button>
        <p id="form-error" class="error" role="alert" hidden></p>
      </form>

      <div class="perforation" aria-hidden="true"></div>

      <aside class="pass-stub" aria-live="polite" aria-labelledby="stub-title">
        <h2 id="stub-title" class="visually-hidden">Result</h2>

        <div id="stub-empty" class="stub-empty">
          <p>Your result appears here.</p>
          <p class="muted">Choose where you are flying and when, then check the risk.</p>
        </div>

        <div id="stub-result" hidden>
          <div class="route">
            <span id="r-origin" class="code">---</span>
            <svg class="plane" viewBox="0 0 24 24" width="28" height="28" aria-hidden="true"><path fill="currentColor" d="M21 16v-2l-8-5V3.5a1.5 1.5 0 0 0-3 0V9l-8 5v2l8-2.5V19l-2 1.5V22l3.5-1 3.5 1v-1.5L13 19v-5.5z"/></svg>
            <span id="r-dest" class="code">---</span>
          </div>
          <p id="r-meta" class="muted route-meta"></p>

          <p class="prob"><span id="r-prob">--</span><span class="prob-label">chance of arriving 15+ minutes late</span></p>

          <div class="meter" role="img" id="meter" aria-label="Delay risk compared with a typical flight">
            <div class="meter-track">
              <div id="meter-fill" class="meter-fill"></div>
              <div id="meter-typical" class="meter-typical"><span>typical flight</span></div>
            </div>
          </div>

          <p id="r-verdict" class="verdict"></p>
        </div>
      </aside>
    </section>

    <section id="details" class="details" hidden>
      <div class="panel">
        <h2>What drives this estimate</h2>
        <p class="muted">Bars show how strongly each factor pushed this flight's risk up or down.</p>
        <ul id="reasons" class="reasons"></ul>
      </div>

      <div class="panel">
        <h2 id="chart-title">Late arrivals by departure hour</h2>
        <p class="muted" id="chart-sub"></p>
        <div id="chart" class="chart"></div>
      </div>
    </section>
  </main>

  <footer class="site-foot">
    <p id="model-note" class="muted"></p>
    <p class="muted">A flight counts as late if it arrives 15 or more minutes after its scheduled time, the same definition the Department of Transportation uses. This is a statistical estimate, not a guarantee. It does not know about today's weather or air traffic, so check your airline for live status.</p>
  </footer>

  <script src="/app.js"></script>
</body>
</html>
```

---
## `web/style.css`

```css
/* Design notes: the page is one boarding pass. Barlow is the typeface family used on US highway
   signs, so it feels like airport wayfinding. Colour is reserved for risk (green / amber / red). */
:root {
  --bg: #dde4ea;
  --paper: #fbfbf9;
  --ink: #0f2b46;
  --ink-soft: #4b6075;
  --line: #c3ced8;
  --good: #1c8c6e;
  --warn: #c98a00;
  --bad: #c4372a;
  --focus: #0b5fff;
  --r: 14px;
  font-family: "Barlow", system-ui, -apple-system, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
html { background: var(--bg); color: var(--ink); }
body { margin: 0; padding: 32px 20px 56px; font-size: 17px; line-height: 1.5; max-width: 1040px; margin-inline: auto; }

h1, h2, .code, .prob { font-family: "Barlow Condensed", "Barlow", system-ui, sans-serif; letter-spacing: 0.01em; }
h1 { font-size: clamp(2.4rem, 6vw, 3.6rem); line-height: 1.02; margin: 0 0 10px; font-weight: 700; }
h2 { font-size: 1.5rem; margin: 0 0 4px; font-weight: 600; }
.site-head p { max-width: 62ch; margin: 0 0 28px; color: var(--ink-soft); }
.muted { color: var(--ink-soft); font-size: 0.95rem; margin: 0; }
.visually-hidden { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }

/* ---------- the boarding pass ---------- */
.pass { display: grid; grid-template-columns: minmax(0, 1.15fr) 0 minmax(0, 1fr); background: var(--paper);
  border-radius: var(--r); box-shadow: 0 1px 0 #fff inset, 0 18px 40px -22px rgba(15, 43, 70, .45); overflow: hidden; position: relative; }
.pass-main { padding: 28px 30px 30px; }
.pass-title { margin-bottom: 18px; }
.pass-stub { padding: 28px 30px; background: #f2f4f1; display: flex; flex-direction: column; justify-content: center; min-height: 360px; }
.perforation { border-left: 2px dashed var(--line); position: relative; }
.perforation::before, .perforation::after { content: ""; position: absolute; left: -13px; width: 24px; height: 24px; border-radius: 50%; background: var(--bg); }
.perforation::before { top: -12px; } .perforation::after { bottom: -12px; }

.field { display: flex; flex-direction: column; gap: 5px; margin-bottom: 16px; min-width: 0; }
.field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
label { font-weight: 600; font-size: 0.95rem; }
select, input { font: inherit; color: var(--ink); background: #fff; border: 1.5px solid var(--line); border-radius: 8px; padding: 10px 12px; min-height: 46px; width: 100%; }
select:disabled { background: #eef1f3; color: #7b8c9b; }
select:focus-visible, input:focus-visible, button:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }

button { font: inherit; font-weight: 600; width: 100%; min-height: 50px; margin-top: 6px; border: 0; border-radius: 8px; cursor: pointer;
  background: var(--ink); color: #fff; transition: background .15s; }
button:hover:not(:disabled) { background: #173f65; }
button:disabled { background: #9fb0bf; cursor: not-allowed; }
button.loading { cursor: progress; }
.error { color: var(--bad); margin: 12px 0 0; font-weight: 500; }

/* ---------- result stub ---------- */
.stub-empty p:first-child { font-family: "Barlow Condensed", sans-serif; font-size: 1.6rem; font-weight: 600; margin: 0 0 4px; }
.route { display: flex; align-items: center; gap: 12px; }
.code { font-size: 3rem; font-weight: 700; line-height: 1; }
.plane { transform: rotate(90deg); color: var(--ink-soft); flex: none; }
.route-meta { margin: 6px 0 18px; }
.prob { display: flex; flex-direction: column; margin: 0 0 14px; line-height: 1; }
.prob #r-prob { font-size: 4.6rem; font-weight: 700; }
.prob-label { font-family: "Barlow", sans-serif; font-size: 0.98rem; color: var(--ink-soft); margin-top: 4px; letter-spacing: 0; }

.meter-track { position: relative; height: 14px; border-radius: 7px; background: #dfe5e9; margin: 18px 0 32px; }
.meter-fill { height: 100%; width: 0; border-radius: 7px; background: var(--good); transition: width .7s cubic-bezier(.2, .8, .2, 1), background .3s; }
.meter-typical { position: absolute; top: -8px; bottom: -8px; width: 2px; background: var(--ink); }
.meter-typical span { position: absolute; top: 100%; left: 50%; transform: translateX(-50%); font-size: 0.78rem; white-space: nowrap; color: var(--ink-soft); padding-top: 2px; }
.verdict { margin: 0; font-weight: 500; }
.risk-low .meter-fill { background: var(--good); } .risk-medium .meter-fill { background: var(--warn); } .risk-high .meter-fill { background: var(--bad); }
.risk-low #r-prob { color: var(--good); } .risk-medium #r-prob { color: #9c6a00; } .risk-high #r-prob { color: var(--bad); }

/* ---------- detail panels ---------- */
.details { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 22px; }
.panel { background: var(--paper); border-radius: var(--r); padding: 24px 26px; }
.reasons { list-style: none; padding: 0; margin: 18px 0 0; display: grid; gap: 14px; }
.reasons li { display: grid; gap: 5px; }
.reasons .txt { font-weight: 500; }
.reasons .bar { height: 8px; border-radius: 4px; background: #e6ebee; overflow: hidden; }
.reasons .bar i { display: block; height: 100%; border-radius: 4px; }
.reasons .up i { background: var(--bad); } .reasons .down i { background: var(--good); }
.reasons .tag { font-size: 0.85rem; color: var(--ink-soft); }
.chart { margin-top: 14px; }
.chart svg { width: 100%; height: auto; display: block; }
.chart text { font-family: "Barlow", sans-serif; fill: var(--ink-soft); font-size: 11px; }

.site-foot { margin-top: 30px; display: grid; gap: 8px; max-width: 80ch; }

@media (max-width: 820px) {
  .pass { grid-template-columns: 1fr; }
  .perforation { border-left: 0; border-top: 2px dashed var(--line); height: 0; }
  .perforation::before, .perforation::after { top: -12px; }
  .perforation::before { left: -12px; } .perforation::after { left: auto; right: -12px; bottom: auto; }
  .pass-stub { min-height: 0; }
  .details { grid-template-columns: 1fr; }
}
@media (prefers-reduced-motion: reduce) { .meter-fill { transition: none; } }

@media (max-width: 480px) {
  body { padding: 22px 14px 44px; }
  .field-row { grid-template-columns: 1fr; gap: 0; }
  .pass-main, .pass-stub { padding: 22px 20px; }
  .prob #r-prob { font-size: 3.8rem; }
  .code { font-size: 2.4rem; }
}
```

---
## `web/app.js`

```javascript
// The browser talks to the FastAPI backend through these JSON endpoints (same origin, so no CORS):
//   GET  /api/meta            -> airports, model info
//   GET  /api/destinations    -> where can you fly from an airport
//   GET  /api/carriers        -> which airlines fly that route
//   POST /api/predict         -> probability + reasons

const $ = (id) => document.getElementById(id);
const el = {
  origin: $("origin"), dest: $("dest"), carrier: $("carrier"), date: $("date"), time: $("time"),
  submit: $("submit"), form: $("flight-form"), error: $("form-error"),
};
let airports = {};

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(body.detail) ? "Please check the date and time you entered." : body.detail;
    throw new Error(detail || "Something went wrong. Please try again.");
  }
  return body;
}

function fill(select, items, placeholder, label) {
  select.innerHTML = "";
  select.append(new Option(placeholder, ""));
  items.forEach((i) => select.append(new Option(label(i), i.code)));
  select.disabled = items.length === 0;
}

function showError(msg) { el.error.textContent = msg; el.error.hidden = !msg; }
function updateSubmit() { el.submit.disabled = !(el.origin.value && el.dest.value && el.carrier.value && el.date.value && el.time.value); }

// ---------- initial load ----------
async function init() {
  const today = new Date();
  const iso = (d) => new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  el.date.min = iso(today);
  const t = new Date(today.getTime() + 7 * 864e5);
  el.date.value = iso(t);
  try {
    const meta = await api("/api/meta");
    meta.airports.forEach((a) => (airports[a.code] = a.city));
    fill(el.origin, meta.airports, "Choose departure airport", (a) => `${a.code} · ${a.city}`);
    const bits = [];
    if (meta.roc_auc) bits.push(`ROC-AUC ${meta.roc_auc.toFixed(2)} on ${meta.test_years ? meta.test_years.join(", ") : "held-out"} flights the model never saw`);
    bits.push(`Typical flight: ${(meta.base_rate * 100).toFixed(0)}% arrive late`);
    bits.push(`History from ${meta.stats_year}`);
    $("model-note").textContent = bits.join(". ") + ".";
    window.__baseRate = meta.base_rate;
  } catch (e) {
    showError("The prediction service is not reachable right now. Refresh the page to try again.");
  }
}

el.origin.addEventListener("change", async () => {
  fill(el.dest, [], "Choose a departure airport first", () => "");
  fill(el.carrier, [], "Choose a destination first", () => "");
  updateSubmit();
  if (!el.origin.value) return;
  try {
    const dests = await api(`/api/destinations?origin=${el.origin.value}`);
    fill(el.dest, dests, "Choose destination", (a) => `${a.code} · ${a.city}`);
  } catch (e) { showError(e.message); }
});

el.dest.addEventListener("change", async () => {
  fill(el.carrier, [], "Choose a destination first", () => "");
  updateSubmit();
  if (!el.dest.value) return;
  try {
    const cs = await api(`/api/carriers?origin=${el.origin.value}&dest=${el.dest.value}`);
    fill(el.carrier, cs, "Choose airline", (c) => c.name);
    if (cs.length === 1) { el.carrier.value = cs[0].code; }
    updateSubmit();
  } catch (e) { showError(e.message); }
});

[el.carrier, el.date, el.time].forEach((n) => n.addEventListener("input", updateSubmit));

// ---------- predict ----------
el.form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  showError("");
  el.submit.classList.add("loading");
  el.submit.disabled = true;
  el.submit.textContent = "Checking...";
  try {
    const result = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        flight_date: el.date.value, dep_time: el.time.value,
        carrier: el.carrier.value, origin: el.origin.value, dest: el.dest.value,
      }),
    });
    render(result);
  } catch (e) {
    showError(e.message);
  } finally {
    el.submit.classList.remove("loading");
    el.submit.textContent = "Check delay risk";
    updateSubmit();
  }
});

const VERDICT = {
  low: "Less likely to be late than a typical flight.",
  medium: "About as likely to be late as a typical flight.",
  high: "More likely to be late than a typical flight.",
};

function render(r) {
  $("stub-empty").hidden = true;
  $("stub-result").hidden = false;
  $("details").hidden = false;

  $("r-origin").textContent = el.origin.value;
  $("r-dest").textContent = el.dest.value;
  const carrierName = el.carrier.options[el.carrier.selectedIndex].text;
  $("r-meta").textContent = `${carrierName}, ${el.date.value} at ${el.time.value}`;

  const pct = Math.round(r.probability * 100);
  $("r-prob").textContent = `${pct}%`;
  $("r-verdict").textContent = `${VERDICT[r.risk]} About ${Math.round(r.base_rate * 100)}% of flights arrive late.`;

  const stub = $("stub-result");
  stub.className = `risk-${r.risk}`;
  // Scale the meter to 0-60% so differences are visible (most flights sit between 10% and 40%).
  const MAX = 0.6;
  $("meter-typical").style.left = `${Math.min(r.base_rate / MAX, 1) * 100}%`;
  $("meter").setAttribute("aria-label", `${pct}% delay risk, typical flight is ${Math.round(r.base_rate * 100)}%`);
  requestAnimationFrame(() => { $("meter-fill").style.width = `${Math.min(r.probability / MAX, 1) * 100}%`; });

  // reasons
  const ul = $("reasons");
  ul.innerHTML = "";
  const maxW = Math.max(...r.reasons.map((x) => x.weight), 0.01);
  r.reasons.forEach((x) => {
    const li = document.createElement("li");
    li.className = x.direction;
    li.innerHTML = `<span class="txt"></span><span class="bar"><i></i></span><span class="tag"></span>`;
    li.querySelector(".txt").textContent = x.text;
    li.querySelector("i").style.width = `${(x.weight / maxW) * 100}%`;
    li.querySelector(".tag").textContent = x.direction === "up" ? "Raises the risk" : "Lowers the risk";
    ul.append(li);
  });

  drawChart(r);
  $("details").scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "nearest" });
}

function drawChart(r) {
  const rates = r.context.hourly_origin_rates;
  const sel = r.context.selected_hour;
  $("chart-title").textContent = `Late arrivals by departure hour at ${el.origin.value}`;
  $("chart-sub").textContent = `Share of ${el.origin.value} departures that arrived 15+ minutes late in ${r.context.stats_year}. Your departure hour is highlighted.`;

  const W = 520, H = 220, padL = 34, padB = 26, padT = 10;
  const vals = rates.map((v) => (v == null ? 0 : v));
  const top = Math.max(0.4, Math.ceil(Math.max(...vals) * 10) / 10);
  const bw = (W - padL) / 24;
  const y = (v) => padT + (H - padT - padB) * (1 - v / top);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Bar chart of late-arrival rate for each departure hour">`;
  for (let t = 0; t <= top + 1e-9; t += 0.1) {
    svg += `<line x1="${padL}" x2="${W}" y1="${y(t)}" y2="${y(t)}" stroke="#dde4ea"/><text x="${padL - 6}" y="${y(t) + 4}" text-anchor="end">${Math.round(t * 100)}%</text>`;
  }
  rates.forEach((v, h) => {
    if (v == null) return;
    const x = padL + h * bw + 2;
    svg += `<rect x="${x}" y="${y(v)}" width="${bw - 4}" height="${H - padB - y(v)}" rx="2" fill="${h === sel ? "#c4372a" : "#9fb4c7"}"><title>${h}:00 - ${(v * 100).toFixed(0)}% late</title></rect>`;
  });
  [0, 6, 12, 18, 23].forEach((h) => { svg += `<text x="${padL + h * bw + bw / 2}" y="${H - 8}" text-anchor="${h === 23 ? "end" : "middle"}">${h}:00</text>`; });
  const base = r.base_rate;
  svg += `<line x1="${padL}" x2="${W}" y1="${y(base)}" y2="${y(base)}" stroke="#0f2b46" stroke-dasharray="4 4"/>`;
  svg += `</svg>`;
  $("chart").innerHTML = svg;
}

init();
```

---
## `scripts/make_sample_data.py`

```python
"""
Generate SYNTHETIC flight data with the same columns as the converted BTS Parquet files.
Use it to test the whole pipeline (train -> API -> web UI) in minutes, without downloading ~20 GB.
The results are NOT real - swap in real BTS data for the actual project.

    python scripts/make_sample_data.py --years 2021 2022 2023 2024 --rows-per-year 150000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "parquet"

AIRPORTS = {
    "ATL": "Atlanta, GA", "ORD": "Chicago, IL", "DFW": "Dallas/Fort Worth, TX", "DEN": "Denver, CO",
    "LAX": "Los Angeles, CA", "JFK": "New York, NY", "LGA": "New York, NY", "SFO": "San Francisco, CA",
    "SEA": "Seattle, WA", "MIA": "Miami, FL", "BOS": "Boston, MA", "PHX": "Phoenix, AZ",
    "LAS": "Las Vegas, NV", "MCO": "Orlando, FL", "EWR": "Newark, NJ", "CLT": "Charlotte, NC",
    "MSP": "Minneapolis, MN", "DTW": "Detroit, MI", "PHL": "Philadelphia, PA", "SLC": "Salt Lake City, UT",
}
CARRIERS = ["AA", "DL", "UA", "WN", "B6", "AS", "NK", "F9"]
rng = np.random.default_rng(7)

codes = list(AIRPORTS)
coords = {c: rng.uniform(0, 1, 2) for c in codes}
origin_bias = {c: rng.normal(0, 0.35) for c in codes}   # some airports are worse than others
carrier_bias = {c: rng.normal(0, 0.3) for c in CARRIERS}


def make_year(year: int, n: int) -> pd.DataFrame:
    dates = pd.to_datetime(rng.choice(pd.date_range(f"{year}-01-01", f"{year}-12-31"), n))
    origin = rng.choice(codes, n)
    dest = rng.choice(codes, n)
    same = origin == dest
    dest[same] = rng.choice(codes, same.sum())
    carrier = rng.choice(CARRIERS, n)
    hour = np.clip(rng.normal(13, 4.5, n).round(), 5, 23).astype(int)
    minute = rng.choice([0, 15, 30, 45, 55], n)
    dist = np.array([300 + 2500 * np.linalg.norm(coords[o] - coords[d]) for o, d in zip(origin, dest)])

    logit = (
        -2.0
        + 0.07 * (hour - 5)                               # later in the day -> more delays
        + np.where(np.isin(dates.month, [6, 7, 12]), 0.35, 0.0)
        + np.where(dates.dayofweek == 4, 0.15, 0.0)
        + np.array([origin_bias[o] for o in origin])
        + np.array([carrier_bias[c] for c in carrier])
        + rng.normal(0, 0.8, n)
    )
    p = 1 / (1 + np.exp(-logit))
    arr_delay = np.where(rng.random(n) < p, rng.gamma(2.0, 30, n) + 15, rng.normal(-5, 8, n).clip(-30, 14))
    cancelled = (rng.random(n) < 0.015).astype(float)
    df = pd.DataFrame({
        "flight_date": dates.date, "Year": year, "month": dates.month, "day": dates.day,
        "dow": dates.dayofweek + 1, "carrier": carrier, "tail_number": "N" + pd.Series(rng.integers(100, 999, n)).astype(str),
        "flight_number": rng.integers(1, 2500, n), "origin": origin,
        "origin_city": [AIRPORTS[o] for o in origin], "dest": dest, "dest_city": [AIRPORTS[d] for d in dest],
        "crs_dep": hour * 100 + minute, "crs_arr": ((hour + 2) % 24) * 100 + minute,
        "crs_elapsed": (dist / 7.5 + 30).round(), "distance": dist.round(),
        "cancelled": cancelled, "diverted": 0.0,
        "dep_delay": arr_delay * 0.8 + rng.normal(0, 4, n), "arr_delay": arr_delay,
    })
    df["arr_del15"] = (df["arr_delay"] >= 15).astype(float)
    df.loc[df["cancelled"] == 1, ["arr_delay", "dep_delay", "arr_del15"]] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024])
    ap.add_argument("--rows-per-year", type=int, default=150_000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for y in a.years:
        df = make_year(y, a.rows_per_year)
        for m, part in df.groupby("month"):
            part.to_parquet(OUT / f"flights_{y}_{m:02d}.parquet", index=False)
        print("wrote", y, len(df))
```

---
## `scripts/run_local.sh`

```bash
#!/usr/bin/env bash
# Start the app locally: http://localhost:8000
uvicorn api.main:app --reload --port 8000
```

---
## `tests/test_api.py`

```python
"""Run with:  pytest -q   (needs artifacts/ to exist, i.e. run `python -m src.train` first)."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # `with` triggers the startup event that loads the model
        yield c


def _first_route(client):
    origin = client.get("/api/meta").json()["airports"][0]["code"]
    dest = client.get("/api/destinations", params={"origin": origin}).json()[0]["code"]
    carrier = client.get("/api/carriers", params={"origin": origin, "dest": dest}).json()[0]["code"]
    return origin, dest, carrier


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_predict_ok(client):
    o, d, c = _first_route(client)
    r = client.post("/api/predict", json={"flight_date": "2025-07-03", "dep_time": "18:30",
                                          "carrier": c, "origin": o, "dest": d})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["risk"] in {"low", "medium", "high"}
    assert len(body["reasons"]) > 0
    assert len(body["context"]["hourly_origin_rates"]) == 24


def test_unknown_route_404(client):
    r = client.post("/api/predict", json={"flight_date": "2025-07-03", "dep_time": "18:30",
                                          "carrier": "ZZ", "origin": "ORD", "dest": "ATL"})
    assert r.status_code == 404


def test_bad_time_422(client):
    r = client.post("/api/predict", json={"flight_date": "2025-07-03", "dep_time": "25:99",
                                          "carrier": "AA", "origin": "ORD", "dest": "ATL"})
    assert r.status_code == 422
```
