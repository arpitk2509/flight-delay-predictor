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
