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
