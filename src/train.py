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
