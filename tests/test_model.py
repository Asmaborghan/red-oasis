"""Software tests for data handling, training/evaluation, predict() and SHAP.

These use SYNTHETIC data to check that the code works. They say nothing
about real-world accuracy; see test_real_data.py for the real dataset.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from model import config, data, train
from model.predict import load_model, model_status, predict, validate_inputs
from model.synthetic import make_synthetic_raw

# Most tests here use the synthetic model, which is blocked by default.
pytestmark = pytest.mark.usefixtures("allow_synthetic")

OPTIONAL = set(config.FEATURE_ORDER) - config.REQUIRED_FEATURES


# ---------------------------------------------------------------- data

def test_build_features_units_and_columns():
    feats = data.build_features(make_synthetic_raw(n_days=60))
    assert set(config.FEATURE_ORDER) <= set(feats.columns)
    # SMAP m3/m3 converted to percent
    assert feats["soil_moisture_pct"].max() > 1
    # Daily rows, one site, no duplicate dates
    assert feats["date"].is_unique


def test_missing_rain_days_are_not_treated_as_zero():
    raw = make_synthetic_raw(n_days=30)
    raw.loc[15, "rainfall_mm"] = np.nan
    feats = data.build_features(raw)
    # Every 7-day window that contains the missing day is NaN, not a smaller sum.
    assert feats.loc[15:21, "rainfall_7d_mm"].isna().all()
    assert feats.loc[22, "rainfall_7d_mm"] == pytest.approx(raw.loc[16:22, "rainfall_mm"].sum())


def test_target_is_future_value_and_never_filled():
    feats = data.build_features(make_synthetic_raw(n_days=120))
    df = data.add_target(feats, "soil_moisture_pct", 3)
    lookup = feats.set_index("date")["soil_moisture_pct"]
    for _, row in df.dropna(subset=["target_future"]).sample(10, random_state=0).iterrows():
        assert row["target_future"] == lookup[row["date"] + pd.Timedelta(days=3)]
    clean, _ = data.select_rows(df)
    assert clean["target_future"].notna().all() and clean["soil_moisture_pct"].notna().all()


def test_csv_without_soil_moisture_is_rejected():
    raw = make_synthetic_raw(n_days=30).drop(columns="soil_moisture")
    with pytest.raises(data.DataError):
        data.build_features(raw)


def test_load_raw_and_inspect(tmp_path):
    path = tmp_path / "merged.csv"
    make_synthetic_raw(n_days=40).to_csv(path, index=False)
    assert "soil_moisture" in data.load_raw(path).columns
    assert "-> soil_moisture" in data.inspect(path)


# ---------------------------------------------------- training / evaluation

def test_split_has_no_leakage():
    feats = data.build_features(make_synthetic_raw())
    df, _ = data.select_rows(data.add_target(feats, "soil_moisture_pct", 3))
    tr, te, _ = train.chronological_split(df, 3)
    # The latest training target date must be before the first test date.
    assert tr["target_date"].max() < te["date"].min()


def test_evaluation_file_contents(synthetic_model_dir):
    ev = json.loads((synthetic_model_dir / config.EVALUATION_FILE).read_text())
    assert ev["status"] == "ok"
    assert ev["is_synthetic"] is True and "NOT a measure" in ev["warning"]
    for name in ("random_forest", "ridge", "linear_soil_only",
                 "baseline_persistence", "baseline_training_mean"):
        m = ev["test_metrics"][name]
        assert m["mae"] >= 0 and m["rmse"] >= m["mae"] and math.isfinite(m["r2"])
    assert ev["n_test"] >= config.MIN_TEST_ROWS
    assert isinstance(ev["model_beats_persistence_on_test"], bool)
    assert ev["model_status_label"] in {"reliable", "experimental"}


def test_insufficient_data_saves_no_model(tmp_path):
    ev = train.train(make_synthetic_raw(n_days=60), "soil_moisture_pct", 3, tmp_path, "test", True)
    assert ev["status"] == "insufficient_data"
    assert not (tmp_path / config.MODEL_FILE).exists()
    assert (tmp_path / config.EVALUATION_FILE).exists()


def test_ndvi_target_trains(tmp_path):
    ev = train.train(make_synthetic_raw(), "ndvi", 16, tmp_path, "test", True)
    assert ev["status"] in {"ok", "insufficient_data"}


# ------------------------------------------------- synthetic-model safety

def test_synthetic_model_cannot_be_saved_as_the_real_model():
    with pytest.raises(data.DataError, match="SYNTHETIC"):
        train.train(make_synthetic_raw(), "soil_moisture_pct", 3,
                    config.DEFAULT_MODEL_DIR, "test", is_synthetic=True)


def test_synthetic_model_is_blocked_by_default(monkeypatch, synthetic_model_dir, good_inputs):
    monkeypatch.delenv(config.ALLOW_SYNTHETIC_ENV, raising=False)
    assert load_model(synthetic_model_dir) is None
    status = model_status(synthetic_model_dir)
    assert status["available"] is False and "SYNTHETIC" in status["reason"]
    r = predict(good_inputs, model_dir=synthetic_model_dir)
    assert r["status"] == "model_unavailable" and "SYNTHETIC" in r["errors"][0]


# ---------------------------------------------------------------- loading

def test_model_unavailable_when_not_trained(tmp_path, good_inputs):
    assert load_model(tmp_path) is None
    assert model_status(tmp_path)["available"] is False
    result = predict(good_inputs, model_dir=tmp_path)
    assert result["status"] == "model_unavailable" and result["prediction"] is None


def test_model_loads_and_is_cached(synthetic_model_dir):
    a, b = load_model(synthetic_model_dir), load_model(synthetic_model_dir)
    assert a is b
    status = model_status(synthetic_model_dir)
    assert status["available"] and status["is_synthetic"]


# ------------------------------------------------------------- validation

@pytest.mark.parametrize("bad, fragment", [
    ({"soil_moisture_pct": -5}, "outside the valid range"),
    ({"soil_moisture_pct": 20, "ndvi": 1.5}, "outside the valid range"),
    ({"soil_moisture_pct": 20, "et0_mm": -1}, "outside the valid range"),
    ({"soil_moisture_pct": "wet"}, "must be a number"),
    ({"soil_moisture_pct": float("inf")}, "must be finite"),
    ({"soil_moisture_pct": True}, "not a boolean"),
    ({"ndvi": 0.3}, "soil_moisture_pct is required"),
    ({"soil_moisture_pct": 20, "humidity": 40}, "Unknown input"),
])
def test_invalid_inputs(bad, fragment, synthetic_model_dir):
    result = predict(bad, model_dir=synthetic_model_dir)
    assert result["status"] == "invalid_input"
    assert any(fragment in e for e in result["errors"])


def test_non_dict_input(synthetic_model_dir):
    assert predict([1, 2, 3], model_dir=synthetic_model_dir)["status"] == "invalid_input"


def test_numeric_strings_are_accepted():
    clean, missing, errors = validate_inputs({"soil_moisture_pct": "12.5"})
    assert clean["soil_moisture_pct"] == 12.5 and not errors
    assert set(missing) == OPTIONAL


# ---------------------------------------------------------------- predict

def test_predict_output_structure(synthetic_model_dir, good_inputs):
    r = predict(good_inputs, model_dir=synthetic_model_dir)
    assert r["status"] == "ok", r["errors"]
    assert set(r) == {"status", "prediction", "recommendation", "explanation",
                      "inputs_used", "imputed", "warnings", "errors", "model"}
    p = r["prediction"]
    assert p["target"] == "soil_moisture_pct" and p["horizon_days"] == 3
    assert math.isfinite(p["value"])
    assert r["recommendation"]["level"] in {"irrigate", "monitor", "none"}
    assert r["model"]["is_synthetic"] is True
    assert any("SYNTHETIC" in w for w in r["warnings"])


def test_missing_optional_inputs_are_imputed(synthetic_model_dir):
    r = predict({"soil_moisture_pct": 12.0}, model_dir=synthetic_model_dir)
    assert r["status"] == "ok"
    assert set(r["imputed"]) == OPTIONAL
    assert sum("training median" in w for w in r["warnings"]) == len(OPTIONAL)


def test_extrapolation_warning(synthetic_model_dir, good_inputs):
    r = predict({**good_inputs, "soil_moisture_pct": 90.0}, model_dir=synthetic_model_dir)
    assert any("outside the training data range" in w for w in r["warnings"])


def test_thresholds_from_website_are_used(synthetic_model_dir, good_inputs):
    always_dry = predict(good_inputs, thresholds={"soil_dry": 99, "soil_moderate": 100},
                         model_dir=synthetic_model_dir)
    never_dry = predict(good_inputs, thresholds={"soil_dry": 0, "soil_moderate": 0},
                        model_dir=synthetic_model_dir)
    assert always_dry["recommendation"]["level"] == "irrigate"
    assert never_dry["recommendation"]["level"] == "none"


def test_predict_never_raises_on_corrupt_model(tmp_path, good_inputs):
    (tmp_path / config.MODEL_FILE).write_text("not a model")
    (tmp_path / config.METADATA_FILE).write_text("{}")
    r = predict(good_inputs, model_dir=tmp_path)
    assert r["status"] == "error" and r["errors"]


# ------------------------------------------------------------------- SHAP

def test_shap_contributions_add_up(synthetic_model_dir, good_inputs):
    r = predict(good_inputs, model_dir=synthetic_model_dir)
    e = r["explanation"]
    total = e["base_value"] + sum(c["contribution"] for c in e["contributions"])
    assert total == pytest.approx(r["prediction"]["value"], abs=1e-6)
    assert e["prediction"] == pytest.approx(r["prediction"]["value"], abs=1e-6)
    assert {c["feature"] for c in e["contributions"]} == set(config.FEATURE_ORDER)
    sizes = [abs(c["contribution"]) for c in e["contributions"]]
    assert sizes == sorted(sizes, reverse=True)
    assert "not proof" in e["summary"]


def test_shap_flags_imputed_features(synthetic_model_dir):
    e = predict({"soil_moisture_pct": 12.0}, model_dir=synthetic_model_dir)["explanation"]
    flagged = {c["feature"] for c in e["contributions"] if c["imputed"]}
    assert flagged == OPTIONAL


def test_explain_can_be_turned_off(synthetic_model_dir, good_inputs):
    assert predict(good_inputs, explain=False, model_dir=synthetic_model_dir)["explanation"] is None


def test_global_importance_saved(synthetic_model_dir):
    gi = model_status(synthetic_model_dir)["global_importance"]
    assert [g["feature"] for g in gi] and len(gi) == len(config.FEATURE_ORDER)
    values = [g["mean_abs_shap"] for g in gi]
    assert values == sorted(values, reverse=True)
