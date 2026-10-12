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
