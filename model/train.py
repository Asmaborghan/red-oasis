"""
Train, evaluate and save the Red Oasis forecast model.

Usage (from the repository root):

    # 1. Look at the CSV first: columns, units, gaps, date range
    python -m model.train --inspect

    # 2. Train on the real data in data/raw/red_oasis_daily.csv
    #    (writes model/artifacts/, including model_rows_audit.csv)
    python -m model.train

    # Interface test only: train on SYNTHETIC data into a separate,
    # git-ignored folder. Never present its scores as real results.
    python -m model.train --synthetic-demo

Task: regression. Forecast observed SMAP soil moisture (%) `horizon`
days ahead from today's observed soil moisture, NDVI, 7-day rainfall,
air temperature and reference evapotranspiration (ET0).

Evaluation: the most recent 20% of usable dates are held out as a test
set, with a gap of `horizon` days so no training target overlaps the
test period. The model is compared with simple baselines. After
evaluation, the saved model is refit on all usable rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, data, explain


def build_pipeline() -> Pipeline:
    return Pipeline([
        # keep_empty_features keeps column positions stable for SHAP.
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            # Single process: bit-identical results run to run (parallel tree
            # averaging differed by ~1e-15) and no joblib worker warnings.
            n_jobs=1,
            random_state=config.RANDOM_STATE,
        )),
    ])


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _persistence(df: pd.DataFrame, target: str) -> pd.Series:
    """Baseline: 'the value in `horizon` days equals the value today'."""
    return df[target]


def chronological_split(df: pd.DataFrame, horizon_days: int):
    """Hold out the latest TEST_FRACTION of dates, with a `horizon` gap."""
    dates = np.sort(df["date"].unique())
    cutoff = pd.Timestamp(dates[int(len(dates) * (1 - config.TEST_FRACTION))])
    train = df[df["date"] < cutoff - pd.Timedelta(days=horizon_days)]
    test = df[df["date"] >= cutoff]
    return train, test, cutoff


def _period(df: pd.DataFrame) -> list[str]:
    return [str(df["date"].min().date()), str(df["date"].max().date())] if len(df) else []


def evaluate(df: pd.DataFrame, target: str, horizon_days: int) -> dict:
    """Honest held-out evaluation. Returns a dict saved as evaluation.json."""
    train, test, cutoff = chronological_split(df, horizon_days)
    result = {
        "task": "regression (forecast)",
        "target": f"observed {target} in {horizon_days} days",
        "target_unit": config.TARGETS[target]["unit"],
        "features": config.FEATURE_ORDER,
        "validation": (
            f"Chronological hold-out (no shuffling): train on feature dates up to "
            f"{(cutoff - pd.Timedelta(days=horizon_days + 1)).date()}, test on feature dates "
            f"from {cutoff.date()}. A {horizon_days}-day gap prevents training targets "
            f"from overlapping the test period."
        ),
        "train_period": _period(train),
        "test_period": _period(test),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }
    if len(train) < config.MIN_TRAIN_ROWS or len(test) < config.MIN_TEST_ROWS:
        result["status"] = "insufficient_data"
        result["message"] = (
            f"Need at least {config.MIN_TRAIN_ROWS} training and {config.MIN_TEST_ROWS} "
            f"test rows with an observed soil-moisture value and an observed target."
        )
        return result

    X_cols = config.FEATURE_ORDER
    unit = config.TARGETS[target]["unit"]
    result["metric_units"] = f"MAE and RMSE in {unit}; R2 unitless"

    # --- 1. Time-ordered CV on the training period only (model comparison).
    # Rows are sorted by date but not every day is usable, so the row gap
    # is backed up by a calendar check on target dates.
    n_splits = min(config.CV_SPLITS, max(2, len(train) // 50))
    gap_rows = horizon_days * train["site"].nunique()
    folds, pooled = [], []
    for i, (tr_idx, va_idx) in enumerate(TimeSeriesSplit(n_splits=n_splits, gap=gap_rows)
                                         .split(train)):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        tr = tr[tr["target_date"] < va["date"].min()]
        va = va[_persistence(va, target).notna()]
        if len(va) == 0 or len(tr) < 20:
            continue
        preds = _predict_all(tr, va, target)
        y = va["target_future"]
        folds.append({
            "fold": i + 1, "validation_period": _period(va),
            "n_train": int(len(tr)), "n_validation": int(len(va)),
            "metrics": {name: _metrics(y, p) for name, p in preds.items()},
        })
        pooled.append(pd.DataFrame({"y": y.to_numpy(), "wet": va["rainfall_7d_mm"].to_numpy() > 0,
                                    **{n: np.asarray(p) for n, p in preds.items()}}))

    names = list(folds[0]["metrics"]) if folds else []
    cv_mean = {n: {k: float(np.mean([f["metrics"][n][k] for f in folds]))
                   for k in ("mae", "rmse", "r2")} for n in names}
    wins = {n: int(sum(f["metrics"][n]["mae"] < f["metrics"]["baseline_persistence"]["mae"]
                       for f in folds)) for n in names if n != "baseline_persistence"}
    result["cv_training_period"] = {
        "method": f"TimeSeriesSplit(n_splits={len(folds)}), training rows only; "
                  f"calendar gap of {horizon_days} days between fold train and validation",
        "mean_over_folds": cv_mean,
        "folds_beating_persistence": wins,
        "folds": folds,
        "by_condition_mae": _by_condition(pd.concat(pooled) if pooled else None, names),
    }

    # --- 2. Final test period, used once (no tuning against it).
    preds = _predict_all(train, test, target)
    y = test["target_future"]
    result["n_test_compared"] = int(len(test))
    result["test_metrics"] = {n: _metrics(y, p) for n, p in preds.items()}
    result["test_by_condition_mae"] = _by_condition(
        pd.DataFrame({"y": y.to_numpy(), "wet": test["rainfall_7d_mm"].to_numpy() > 0,
                      **{n: np.asarray(p) for n, p in preds.items()}}), list(preds))
    # Deployment check: Amna's agreed schema has no ET0, so the website will
    # usually call the model with ET0 imputed. Measure that case too.
    rf = build_pipeline().fit(train[X_cols], train["target_future"])
    no_et0 = test[X_cols].assign(et0_mm=np.nan)
    result["test_metrics_random_forest_without_et0"] = _metrics(y, rf.predict(no_et0))

    # --- 3. Verdict, using the rule fixed in config before testing.
    m, p = result["test_metrics"]["random_forest"], result["test_metrics"]["baseline_persistence"]
    cv_ok = bool(folds) and (
        cv_mean["random_forest"]["mae"] < cv_mean["baseline_persistence"]["mae"]
        and wins["random_forest"] > len(folds) / 2)
    result["skill_vs_persistence"] = float(1 - m["mae"] / p["mae"]) if p["mae"] > 0 else None
    result["model_beats_persistence_on_test"] = bool(m["mae"] < p["mae"])
    result["model_beats_persistence_in_cv"] = bool(cv_ok)
    result["reliable"] = bool(cv_ok and m["mae"] < p["mae"])
    result["reliability_rule"] = config.RELIABILITY_RULE
    result["model_status_label"] = "reliable" if result["reliable"] else "experimental"
    result["model_selection"] = (
        "The saved model is the Random Forest (kept for exact SHAP explanations). "
        "Ridge and a soil-moisture-only linear model are reported for comparison only; "
        "no model was chosen or tuned on the final test period."
    )
    result["status"] = "ok"
    return result


def _candidates() -> dict:
    """The models compared in CV. All fit imputers/scalers on training rows only."""
    return {
        "random_forest": (build_pipeline(), config.FEATURE_ORDER),
        "ridge": (Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                            ("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
                  config.FEATURE_ORDER),
        # Persistence with mean reversion: y = a * soil_today + b.
        "linear_soil_only": (LinearRegression(), ["soil_moisture_pct"]),
    }


def _predict_all(tr: pd.DataFrame, va: pd.DataFrame, target: str) -> dict:
    out = {}
    for name, (model, cols) in _candidates().items():
        out[name] = model.fit(tr[cols], tr["target_future"]).predict(va[cols])
    out["baseline_persistence"] = _persistence(va, target).to_numpy()
    out["baseline_training_mean"] = np.full(len(va), tr["target_future"].mean())
    return out


def _by_condition(pooled, names) -> dict | None:
    """MAE on days with and without rain in the previous 7 days."""
    if pooled is None or not len(pooled):
        return None
    res = {}
    for label, mask in (("rain_in_last_7_days", pooled["wet"]), ("no_rain_in_last_7_days", ~pooled["wet"])):
        sub = pooled[mask]
        res[label] = {"n": int(len(sub)),
                      **({n: float(mean_absolute_error(sub["y"], sub[n])) for n in names}
                         if len(sub) else {})}
    return res


def train(raw: pd.DataFrame, target: str, horizon_days: int, out_dir: Path,
          data_source: str, is_synthetic: bool, audit_path: Path | None = None) -> dict:
    out_dir = Path(out_dir)
    if is_synthetic and out_dir.resolve() == config.DEFAULT_MODEL_DIR.resolve():
        raise data.DataError(
            "Refusing to save a SYNTHETIC model into model/artifacts/, which is "
            "reserved for the model trained on real data."
        )

    feats = data.build_features(raw)
    df, audit = data.select_rows(data.add_target(feats, target, horizon_days))
    summary = data.exclusion_summary(audit)
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_path, index=False, date_format="%Y-%m-%d")

    missing_share = {c: float(df[c].isna().mean()) for c in config.FEATURE_ORDER} if len(df) else {}
    # An optional input absent from the file (e.g. ET0 in the agreed schema)
    # is recorded as unavailable instead of stopping training. It then
    # carries no information for the model, and predict() says so.
    unavailable = [c for c, share in missing_share.items() if share == 1.0]
    if config.REQUIRED_FEATURES & set(unavailable):
        raise data.DataError(f"Required input has no values: {unavailable}.")

    evaluation = evaluate(df, target, horizon_days) if len(df) else {
        "status": "insufficient_data", "message": "No usable rows."
    }
    evaluation.update({
        "data_source": data_source,
        "data_location": data.site_info(raw),
        "agreed_site": config.AGREED_SITE,
        "is_synthetic": is_synthetic,
        "row_selection": summary,
        "feature_missing_share": missing_share,
        "unavailable_features": unavailable,
        "validation_warnings": feats.attrs.get("validation_warnings", []),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    if is_synthetic:
        evaluation["warning"] = (
            "Computed on SYNTHETIC data to test the software. "
            "This is NOT a measure of real predictive performance."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / config.EVALUATION_FILE).write_text(json.dumps(evaluation, indent=2, allow_nan=False))
    if evaluation["status"] != "ok":
        # No evaluated model -> no saved model. Remove any stale one.
        (out_dir / config.MODEL_FILE).unlink(missing_ok=True)
        (out_dir / config.METADATA_FILE).unlink(missing_ok=True)
        return evaluation

    X, y = df[config.FEATURE_ORDER], df["target_future"]
    pipeline = build_pipeline().fit(X, y)

    import platform

    import shap
    import sklearn

    def _num(v):
        return None if pd.isna(v) else float(v)

    medians = dict(zip(config.FEATURE_ORDER, pipeline.named_steps["imputer"].statistics_))
    metadata = {
        "model_type": "RandomForestRegressor (scikit-learn) with median imputation",
        "status_label": evaluation["model_status_label"],
        "reliable": evaluation["reliable"],
        "recommendations_validated": config.RECOMMENDATIONS_VALIDATED,
        "target": target,
        "target_label": config.TARGETS[target]["label"],
        "target_unit": config.TARGETS[target]["unit"],
        "target_description": (f"Observed SMAP soil moisture {horizon_days} days after the input "
                               "date (a regional ~9 km signal). Not crop stress, irrigation "
                               "demand or yield."),
        "horizon_days": horizon_days,
        "features": {k: dict(v) for k, v in config.FEATURES.items()},
        "feature_order": config.FEATURE_ORDER,
        "unavailable_features": unavailable,
        "training_range": {
            c: {"min": _num(X[c].min()), "max": _num(X[c].max())} for c in config.FEATURE_ORDER
        },
        "training_median": {c: (None if c in unavailable else _num(v)) for c, v in medians.items()},
        "target_distribution": {f"p{q}": float(np.percentile(y, q)) for q in (5, 50, 95)},
        "n_rows": int(len(df)),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "data_source": data_source,
        "data_location": evaluation["data_location"],
        "agreed_site": config.AGREED_SITE,
        "is_synthetic": is_synthetic,
        "global_importance": explain.global_importance(pipeline, X),
        "trained_at": evaluation["created_at"],
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                             "shap": shap.__version__, "numpy": np.__version__,
                             "pandas": pd.__version__, "joblib": joblib.__version__},
        "note": "Saved model is refit on all usable rows after the held-out evaluation.",
    }
    joblib.dump(pipeline, out_dir / config.MODEL_FILE, compress=3)
    (out_dir / config.METADATA_FILE).write_text(json.dumps(metadata, indent=2, allow_nan=False))
    return evaluation


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the Red Oasis forecast model.")
    p.add_argument("--data", type=Path, default=config.DEFAULT_DATA_FILE,
                   help="Daily CSV (default: data/raw/red_oasis_daily.csv)")
    p.add_argument("--inspect", action="store_true", help="Only report what the CSV contains")
    p.add_argument("--target", default=config.DEFAULT_TARGET, choices=list(config.TARGETS))
    p.add_argument("--horizon", type=int, help="Days ahead to forecast")
    p.add_argument("--out", type=Path, help="Output folder (default: model/artifacts)")
    p.add_argument("--synthetic-demo", action="store_true",
                   help="Train on SYNTHETIC data for interface testing only")
    args = p.parse_args(argv)

    horizon = args.horizon or config.TARGETS[args.target]["default_horizon_days"]

    if args.synthetic_demo:
        from .synthetic import SYNTHETIC_NOTICE, make_synthetic_raw

        raw = make_synthetic_raw()
        out = args.out or config.SYNTHETIC_MODEL_DIR
        source, synthetic, audit_path = SYNTHETIC_NOTICE, True, None
    else:
        if not args.data.exists():
            p.error(f"Data file not found: {args.data}")
        if args.inspect:
            print(data.inspect(args.data))
            return 0
        raw = data.load_raw(args.data)
        out = args.out or config.DEFAULT_MODEL_DIR
        source, synthetic = f"NASA-derived daily CSV: {args.data.name}", False
        audit_path = config.ROW_AUDIT_FILE

    try:
        evaluation = train(raw, args.target, horizon, out, source, synthetic, audit_path)
    except data.DataError as exc:
        print(f"Data problem: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(evaluation, indent=2))
    print(f"\nWrote results to {out}")
    return 0 if evaluation["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
