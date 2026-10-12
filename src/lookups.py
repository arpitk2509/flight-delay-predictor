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
