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
