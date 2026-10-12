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
def connect(memory_limit: str | None = None):
    import duckdb  # imported here so the web app does not need duckdb
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
