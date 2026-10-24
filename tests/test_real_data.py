"""Tests on the real NASA-derived dataset (data/raw/red_oasis_daily.csv).

They check the preprocessing rules (no leakage, suspect rain excluded,
observed values only), the chronological split, and that the saved real
model and its evaluation are consistent with the data.
"""

import json

import numpy as np
import pandas as pd
import pytest

from model import config, data, train
from model.predict import model_status, predict

H = config.TARGETS["soil_moisture_pct"]["default_horizon_days"]


@pytest.fixture(scope="module")
def original(real_csv):
    return pd.read_csv(real_csv, encoding="utf-8-sig", parse_dates=["date"])


@pytest.fixture(scope="module")
def prepared(real_csv):
    raw = data.load_raw(real_csv)
    df = data.add_target(data.build_features(raw), "soil_moisture_pct", H)
    clean, audit = data.select_rows(df)
    return raw, df, clean, audit


def suspect_dates(original):
    return set(original.loc[original["rain_suspect"] == 1, "date"])


# ------------------------------------------------------------ raw file

def test_raw_file_shape_and_dates(original):
    assert len(original) == 1004
    assert original["date"].is_monotonic_increasing and original["date"].is_unique
    assert (original["date"].diff().dropna() == pd.Timedelta(days=1)).all()
    assert len(suspect_dates(original)) == int(original["rain_suspect"].sum())


def test_raw_columns_are_used_not_gap_filled(real_csv):
    mapping = data._match_columns(pd.read_csv(real_csv, nrows=1, encoding="utf-8-sig").columns)
    assert mapping["soil_moisture"] == "soil_moisture_raw"
    assert mapping["ndvi"] == "ndvi_raw"
    assert mapping["rainfall"] == "rain_gpm_mm"


# ----------------------------------------------------- suspect rainfall

def test_no_used_row_touches_a_suspect_day(original, prepared):
    _, _, clean, _ = prepared
    bad = suspect_dates(original)
    for _, row in clean.iterrows():
        window = {row["date"] - pd.Timedelta(days=k) for k in range(config.RAIN_WINDOW_DAYS)}
        assert not (window & bad), row["date"]
        assert row["target_date"] not in bad


def test_rain_feature_matches_original_valid_rain(original, prepared):
    _, _, clean, _ = prepared
    rain = original.set_index("date")["rain_gpm_mm"]
    for _, row in clean.sample(40, random_state=1).iterrows():
        if pd.isna(row["rainfall_7d_mm"]):
            continue
        start = row["date"] - pd.Timedelta(days=config.RAIN_WINDOW_DAYS - 1)
        assert row["rainfall_7d_mm"] == pytest.approx(rain[start:row["date"]].sum())


def test_audit_counts_add_up(prepared, original):
    _, _, clean, audit = prepared
    summary = data.exclusion_summary(audit)
    assert summary["rows_total"] == len(original)
    assert summary["rows_used"] == len(clean)
    assert summary["rows_used"] + sum(summary["rows_excluded"].values()) == summary["rows_total"]
    assert summary["rows_excluded"]["rain_suspect = 1 on the feature date"] == int(original["rain_suspect"].sum())


# ---------------------------------------------------------- no leakage

def test_soil_feature_and_target_are_observed_values(original, prepared):
    _, _, clean, _ = prepared
    obs = original.set_index("date")["soil_moisture_raw"] * 100
    assert np.allclose(clean["soil_moisture_pct"], obs[clean["date"]].to_numpy())
    assert np.allclose(clean["target_future"], obs[clean["target_date"]].to_numpy())
    filled = set(original.loc[original["soil_moisture_is_filled"] == 1, "date"])
    assert not (set(clean["date"]) & filled) and not (set(clean["target_date"]) & filled)


def test_ndvi_uses_only_composites_already_available(original, prepared):
    _, df, _, _ = prepared
    comps = original.dropna(subset=["ndvi_raw"]).set_index("date")["ndvi_raw"]
    for _, row in df.dropna(subset=["ndvi"]).sample(40, random_state=2).iterrows():
        latest_known = row["date"] - pd.Timedelta(days=config.NDVI_AVAILABILITY_LAG_DAYS)
        available = comps[:latest_known]
        assert row["ndvi"] == pytest.approx(available.iloc[-1])


def test_chronological_split(prepared):
    _, _, clean, _ = prepared
    tr, te, _ = train.chronological_split(clean, H)
    assert tr["date"].max() < te["date"].min()
    assert tr["target_date"].max() < te["date"].min()
    # Test rows are the most recent ones, and enough of them for metrics.
    assert te["date"].min() > clean["date"].quantile(0.7)
    assert len(te) >= config.MIN_TEST_ROWS


# ----------------------------------------------------- saved real model

@pytest.fixture(scope="module")
def real_evaluation():
    path = config.DEFAULT_MODEL_DIR / config.EVALUATION_FILE
    if not path.exists():
        pytest.skip("Real model not trained yet (python -m model.train)")
    return json.loads(path.read_text())


def test_saved_evaluation_is_real_and_consistent(real_evaluation, prepared):
    ev = real_evaluation
    _, _, clean, audit = prepared
    assert ev["is_synthetic"] is False and "warning" not in ev
    assert ev["row_selection"] == data.exclusion_summary(audit)
    assert ev["data_location"]["lat"] == [23.308639] and ev["data_location"]["lon"] == [56.613083]
    # The persistence baseline is deterministic, so recompute it.
    _, te, _ = train.chronological_split(clean, H)
    persist = np.abs(te["target_future"] - te["soil_moisture_pct"]).mean()
    assert ev["test_metrics"]["baseline_persistence"]["mae"] == pytest.approx(persist)
    assert ev["n_test"] == len(te)


def test_real_model_predicts(real_evaluation):
    status = model_status()
    assert status["available"] and status["is_synthetic"] is False
    r = predict({"soil_moisture_pct": 11.5, "ndvi": 0.23, "rainfall_7d_mm": 0.0,
                 "temperature_c": 34.8, "et0_mm": 7.3})
    assert r["status"] == "ok", r["errors"]
    assert r["model"]["is_synthetic"] is False
    assert not any("SYNTHETIC" in w for w in r["warnings"])
    assert 0 < r["prediction"]["value"] < 100
    e = r["explanation"]
    total = e["base_value"] + sum(c["contribution"] for c in e["contributions"])
    assert total == pytest.approx(r["prediction"]["value"], abs=1e-6)
