"""Tests added in the October 2026 audit (SYNTHETIC data; software checks only).

Covers: data validation and units, column mapping with the actual CSV
names (agreed-schema spellings deliberately not mapped yet), leakage rules
(rain window, suspect rain, observed soil moisture, NDVI lag, 3-day
target), training-only preprocessing, baseline comparison, artifact
loading, synthetic-model blocking, SHAP failure handling, the unchanged
predict()/model_status() return formats, the daily-data conversion layer,
and reproducibility.
"""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from model import config, data, inputs, predict as predict_module, train
from model.predict import model_status, predict
from model.synthetic import make_synthetic_raw

AGREED_PDF = {"temperature_c": "temp_c", "rainfall_mm": "rain_mm", "soil_moisture": "soil_moisture"}
AGREED_BRIEF = {"temperature_c": "c_temp", "rainfall_mm": "mm_rain", "soil_moisture": "moisture_soil"}
# Column names of Amna's delivered file (the actual CSV).
ACTUAL_CSV = {"temperature_c": "temp_mean_c", "rainfall_mm": "rain_gpm_mm",
              "soil_moisture": "soil_moisture_raw"}

# Return formats before the audit (from red-oasis-work-backup-20261021-201635.zip).
PRE_AUDIT_MODEL_STATUS_KEYS = {"available", "data_location", "data_source", "date_range",
                               "evaluation", "global_importance", "horizon_days",
                               "is_synthetic", "target", "target_label", "target_unit"}
PRE_AUDIT_PREDICT_KEYS = {"errors", "explanation", "imputed", "inputs_used", "model",
                          "prediction", "recommendation", "status", "warnings"}
PRE_AUDIT_MODEL_KEYS = {"data_location", "data_source", "is_synthetic", "n_rows",
                        "trained_at", "type"}
PRE_AUDIT_PREDICTION_KEYS = {"horizon_days", "label", "target", "unit", "value"}
PRE_AUDIT_RECOMMENDATION_KEYS = {"basis", "caveat", "level", "thresholds_used"}


def agreed_frame(names=AGREED_PDF, n_days=120):
    """Synthetic data with the given column spelling and no ET0 column."""
    raw = make_synthetic_raw(n_days=n_days).drop(columns="et0_mm").rename(columns=names)
    if names is AGREED_PDF:
        raw["solar_radiation"] = 22.0
    elif names is AGREED_BRIEF:
        raw["radiation_solar"] = 22.0
    return raw


# ------------------------------------------------------- validation & units

@pytest.mark.parametrize("change, fragment", [
    (lambda r: r.assign(soil_moisture=r["soil_moisture"] * 100), "soil_moisture"),   # already %
    (lambda r: r.assign(temperature_c=r["temperature_c"] + 273.15), "temperature"),  # Kelvin
    (lambda r: r.assign(rainfall_mm=r["rainfall_mm"] - 5), "rainfall"),              # negative
    (lambda r: r.assign(ndvi=r["ndvi"] * 10000), "ndvi"),                            # MODIS scale
    (lambda r: r.assign(et0_mm=r["et0_mm"] + 40), "et0"),
])
def test_values_that_do_not_fit_the_declared_unit_are_rejected(change, fragment):
    with pytest.raises(data.DataError, match=fragment):
        data.build_features(change(make_synthetic_raw(n_days=60)))


def test_soil_moisture_is_multiplied_by_100_only_once():
    raw = make_synthetic_raw(n_days=60)
    feats = data.build_features(raw)
    obs = raw["soil_moisture"].dropna()
    assert np.allclose(feats.loc[obs.index, "soil_moisture_pct"], obs * 100)


def test_percent_declared_but_fraction_supplied_is_rejected(monkeypatch):
    monkeypatch.setitem(config.RAW_UNITS, "soil_moisture", "pct")
    with pytest.raises(data.DataError, match="declared as %"):
        data.build_features(make_synthetic_raw(n_days=60))


def test_duplicate_dates_are_rejected():
    raw = make_synthetic_raw(n_days=60)
    raw = pd.concat([raw, raw.iloc[[5]]])
    with pytest.raises(data.DataError, match="duplicate"):
        data.build_features(raw)


def test_actual_csv_names_are_mapped():
    raw = agreed_frame(ACTUAL_CSV)
    mapping = data.column_mapping(raw.columns)
    assert mapping["temperature"] == "temp_mean_c"
    assert mapping["rainfall"] == "rain_gpm_mm"
    assert mapping["soil_moisture"] == "soil_moisture_raw"
    assert mapping["et0"] is None
    feats = data.build_features(raw)
    assert feats["et0_mm"].isna().all()
    assert any("ET0" in w for w in feats.attrs["validation_warnings"])


@pytest.mark.parametrize("names", [AGREED_PDF, AGREED_BRIEF])
def test_unconfirmed_agreed_spellings_are_reported_not_renamed(names):
    raw = agreed_frame(names)
    mapping = data.column_mapping(raw.columns)
    assert mapping["temperature"] is None and mapping["rainfall"] is None
    expected = [c for c in names.values() if c in config.UNCONFIRMED_AGREED_COLUMNS]
    assert set(expected) <= set(data.unconfirmed_agreed_columns(raw.columns))
    if names is AGREED_BRIEF:   # required soil moisture is not found -> loud error
        with pytest.raises(data.DataError, match="Soil moisture"):
            data.build_features(raw)
    else:                       # "soil_moisture" is a pre-audit alias
        feats = data.build_features(raw)
        assert any("NOT used" in w for w in feats.attrs["validation_warnings"])
        assert feats["temperature_c"].isna().all()


def test_raw_columns_win_over_gap_filled_columns():
    raw = make_synthetic_raw(n_days=60)
    raw["soil_moisture_raw"] = raw["soil_moisture"]
    raw["soil_moisture"] = raw["soil_moisture"].interpolate()
    assert data.column_mapping(raw.columns)["soil_moisture"] == "soil_moisture_raw"


def test_location_other_than_dariz_is_reported():
    raw = make_synthetic_raw(n_days=60).assign(lat=22.93, lon=57.53)
    feats = data.build_features(raw)
    assert any("not the agreed site" in w for w in feats.attrs["validation_warnings"])


# ------------------------------------------------------------- leakage rules

def rows(raw, horizon=3):
    return data.select_rows(data.add_target(data.build_features(raw), "soil_moisture_pct", horizon))


def test_missing_rain_day_excludes_rows_instead_of_imputing_zero():
    raw = make_synthetic_raw(n_days=120)
    raw.loc[60, "rainfall_mm"] = np.nan
    clean, audit = rows(raw)
    window = pd.date_range(raw.loc[60, "date"], periods=7)
    assert not set(clean["date"]) & set(window)
    reasons = audit.set_index("date").loc[window, "exclusion_reason"]
    assert reasons.str.contains("incomplete 7-day rainfall|no observed").all()
    assert clean["rainfall_7d_mm"].notna().all()


def test_suspect_rain_is_removed_and_its_window_excluded():
    raw = make_synthetic_raw(n_days=120)
    raw["rain_suspect"] = 0
    raw.loc[50, ["rain_suspect", "rainfall_mm"]] = [1, 250.0]
    clean, audit = rows(raw)
    bad = set(pd.date_range(raw.loc[50, "date"], periods=7))
    assert not set(clean["date"]) & bad
    assert not set(clean["target_date"]) & {raw.loc[50, "date"]}
    assert (clean["rainfall_7d_mm"] < 250).all()


def test_rows_need_observed_soil_moisture_today_and_in_three_days():
    raw = make_synthetic_raw(n_days=120)
    clean, _ = rows(raw)
    obs = raw.set_index("date")["soil_moisture"] * 100
    assert np.allclose(clean["soil_moisture_pct"], obs[clean["date"]].to_numpy())
    assert np.allclose(clean["target_future"], obs[clean["date"] + pd.Timedelta(days=3)].to_numpy())
    assert (clean["target_date"] - clean["date"] == pd.Timedelta(days=3)).all()


def test_ndvi_is_used_only_after_its_16_day_availability_lag():
    raw = make_synthetic_raw(n_days=120)
    feats = data.build_features(raw)
    lag = config.NDVI_AVAILABILITY_LAG_DAYS
    assert feats.loc[:lag - 1, "ndvi"].isna().all()
    assert feats.loc[lag, "ndvi"] == pytest.approx(raw.loc[0, "ndvi"])
    # A composite is never used before it was available.
    for t in (40, 77, 100):
        known = raw.loc[: t - lag, "ndvi"].dropna()
        assert feats.loc[t, "ndvi"] == pytest.approx(known.iloc[-1])


def test_imputer_is_fitted_on_training_rows_only():
    clean, _ = rows(make_synthetic_raw())
    tr, te, _ = train.chronological_split(clean, 3)
    model = train.build_pipeline().fit(tr[config.FEATURE_ORDER], tr["target_future"])
    assert np.allclose(model.named_steps["imputer"].statistics_,
                       tr[config.FEATURE_ORDER].median().fillna(0).to_numpy())
    assert tr["target_date"].max() < te["date"].min()


# ---------------------------------------------------------- evaluation

def test_evaluation_compares_models_with_baselines(synthetic_model_dir):
    ev = json.loads((synthetic_model_dir / config.EVALUATION_FILE).read_text())
    names = {"random_forest", "ridge", "linear_soil_only", "baseline_persistence",
             "baseline_training_mean"}
    assert set(ev["test_metrics"]) == names
    cv = ev["cv_training_period"]
    assert set(cv["mean_over_folds"]) == names and len(cv["folds"]) >= 2
    for f in cv["folds"]:
        assert f["n_train"] > 0 and f["n_validation"] > 0 and set(f["metrics"]) == names
    assert set(cv["by_condition_mae"]) == {"rain_in_last_7_days", "no_rain_in_last_7_days"}
    assert "test_metrics_random_forest_without_et0" in ev
    assert "MAE" in ev["metric_units"]


def test_reliability_verdict_follows_the_rule(synthetic_model_dir):
    ev = json.loads((synthetic_model_dir / config.EVALUATION_FILE).read_text())
    cv = ev["cv_training_period"]
    cv_ok = (cv["mean_over_folds"]["random_forest"]["mae"]
             < cv["mean_over_folds"]["baseline_persistence"]["mae"]
             and cv["folds_beating_persistence"]["random_forest"] > len(cv["folds"]) / 2)
    test_ok = (ev["test_metrics"]["random_forest"]["mae"]
               < ev["test_metrics"]["baseline_persistence"]["mae"])
    assert ev["reliable"] == (cv_ok and test_ok)
    assert ev["model_status_label"] == ("reliable" if ev["reliable"] else "experimental")


def test_training_is_reproducible(tmp_path):
    raw = make_synthetic_raw()
    a = train.train(raw, "soil_moisture_pct", 3, tmp_path / "a", "t", True)
    b = train.train(raw, "soil_moisture_pct", 3, tmp_path / "b", "t", True)
    a.pop("created_at"), b.pop("created_at")
    assert a == b


def test_training_works_without_et0_column(tmp_path, allow_synthetic, good_inputs):
    ev = train.train(agreed_frame(ACTUAL_CSV, n_days=730), "soil_moisture_pct", 3, tmp_path, "t", True)
    assert ev["status"] == "ok" and ev["unavailable_features"] == ["et0_mm"]
    meta = json.loads((tmp_path / config.METADATA_FILE).read_text())
    assert meta["training_median"]["et0_mm"] is None
    r = predict(good_inputs, model_dir=tmp_path)
    assert r["status"] == "ok"
    assert "et0_mm" not in r["inputs_used"] and "et0_mm" not in r["imputed"]
    assert any("et0_mm was not available" in w for w in r["warnings"])


# ------------------------------------------------------ artifact loading

def test_model_status_never_raises_on_a_corrupt_artifact(tmp_path):
    (tmp_path / config.MODEL_FILE).write_text("garbage")
    (tmp_path / config.METADATA_FILE).write_text(json.dumps(
        {"library_versions": {"scikit-learn": __import__("sklearn").__version__}}))
    status = model_status(tmp_path)
    assert status["available"] is False and "could not be loaded" in status["reason"]


def test_incompatible_sklearn_version_is_refused(tmp_path, synthetic_model_dir, allow_synthetic,
                                                 good_inputs):
    shutil.copytree(synthetic_model_dir, tmp_path / "m")
    meta_path = tmp_path / "m" / config.METADATA_FILE
    meta = json.loads(meta_path.read_text())
    meta["library_versions"]["scikit-learn"] = "0.24.2"
    meta_path.write_text(json.dumps(meta))
    status = model_status(tmp_path / "m")
    assert status["available"] is False and "0.24.2" in status["reason"]
    # A saved model that cannot be used is an "error", as before the audit.
    assert predict(good_inputs, model_dir=tmp_path / "m")["status"] == "error"


def test_incomplete_metadata_is_refused(tmp_path, synthetic_model_dir, allow_synthetic):
    shutil.copytree(synthetic_model_dir, tmp_path / "m")
    meta_path = tmp_path / "m" / config.METADATA_FILE
    meta = json.loads(meta_path.read_text())
    meta.pop("feature_order")
    meta_path.write_text(json.dumps(meta))
    assert "incomplete" in model_status(tmp_path / "m")["reason"]


# -------------------------------------------------- synthetic protection

def test_synthetic_model_in_production_folder_is_refused_even_with_override(
        tmp_path, synthetic_model_dir, allow_synthetic, monkeypatch, good_inputs):
    prod = tmp_path / "artifacts"
    shutil.copytree(synthetic_model_dir, prod)
    monkeypatch.setattr(config, "DEFAULT_MODEL_DIR", prod)
    assert model_status(prod)["available"] is False
    assert predict(good_inputs, model_dir=prod)["status"] == "model_unavailable"


# ------------------------------------------------- predict() behaviour

def test_shap_failure_does_not_break_the_forecast(monkeypatch, synthetic_model_dir,
                                                  allow_synthetic, good_inputs):
    bundle = predict_module.load_model(synthetic_model_dir)
    monkeypatch.setitem(bundle, "explainer", None)

    def boom(_):
        raise RuntimeError("shap broken")

    monkeypatch.setattr("model.explain.make_explainer", boom)
    r = predict(good_inputs, model_dir=synthetic_model_dir)
    assert r["status"] == "ok" and r["explanation"] is None
    assert any("SHAP explanation unavailable" in w for w in r["warnings"])


def test_required_soil_moisture_is_never_imputed(synthetic_model_dir, allow_synthetic):
    r = predict({"ndvi": 0.3, "rainfall_7d_mm": 0, "temperature_c": 30},
                model_dir=synthetic_model_dir)
    assert r["status"] == "invalid_input" and r["prediction"] is None


def test_return_formats_are_unchanged_from_before_the_audit(synthetic_model_dir, allow_synthetic,
                                                            good_inputs, tmp_path):
    status = model_status(synthetic_model_dir)
    assert set(status) == PRE_AUDIT_MODEL_STATUS_KEYS
    assert set(model_status(tmp_path)) == {"available", "reason"}
    r = predict(good_inputs, model_dir=synthetic_model_dir)
    assert set(r) == PRE_AUDIT_PREDICT_KEYS
    assert set(r["model"]) == PRE_AUDIT_MODEL_KEYS
    assert set(r["prediction"]) == PRE_AUDIT_PREDICTION_KEYS
    assert set(r["recommendation"]) == PRE_AUDIT_RECOMMENDATION_KEYS
    assert predict(good_inputs, model_dir=tmp_path)["status"] == "model_unavailable"


def test_recommendation_caveat_says_not_validated(synthetic_model_dir, allow_synthetic, good_inputs):
    r = predict(good_inputs, model_dir=synthetic_model_dir)
    assert "not validated" in r["recommendation"]["caveat"]
    assert config.RECOMMENDATIONS_VALIDATED is False


def test_threshold_above_observed_range_is_flagged(synthetic_model_dir, allow_synthetic,
                                                   good_inputs):
    p95 = predict_module.load_model(synthetic_model_dir)["metadata"]["target_distribution"]["p95"]
    r = predict(good_inputs, thresholds={"soil_dry": p95 + 10, "soil_moderate": p95 + 20},
                model_dir=synthetic_model_dir)
    assert any("almost every day" in w for w in r["warnings"])


def test_experimental_verdict_is_reported(synthetic_model_dir, allow_synthetic, good_inputs):
    ev = model_status(synthetic_model_dir)["evaluation"]
    assert ev["model_status_label"] in {"reliable", "experimental"}
    r = predict(good_inputs, model_dir=synthetic_model_dir)
    if not ev["reliable"]:
        assert any("Experimental model" in w for w in r["warnings"])


# ----------------------------------------------------- conversion layer

def test_inputs_from_daily_actual_csv_names():
    raw = agreed_frame(ACTUAL_CSV, n_days=90)
    day = raw.loc[raw["soil_moisture_raw"].notna()].iloc[-1]["date"]
    out = inputs.inputs_from_daily(raw, as_of=day)
    assert set(out["inputs"]) == set(config.FEATURE_ORDER)
    i = raw.index[raw["date"] == day][0]
    assert out["inputs"]["rainfall_7d_mm"] == pytest.approx(raw.loc[i - 6:i, "rain_gpm_mm"].sum())
    assert out["inputs"]["soil_moisture_pct"] == pytest.approx(raw.loc[i, "soil_moisture_raw"] * 100)
    assert out["inputs"]["et0_mm"] is None
    assert any("ET0" in n for n in out["notes"])
    assert out["column_mapping"]["temperature"] == "temp_mean_c"


def test_inputs_from_daily_reports_missing_soil_and_suspect_rain():
    raw = agreed_frame(ACTUAL_CSV, n_days=90)
    day = raw.loc[raw["soil_moisture_raw"].isna()].iloc[-1]["date"]
    out = inputs.inputs_from_daily(raw, as_of=day)
    assert out["inputs"]["soil_moisture_pct"] is None
    assert any("soil moisture" in n for n in out["notes"])

    raw["rain_suspect"] = 0
    raw.loc[len(raw) - 3, "rain_suspect"] = 1
    out = inputs.inputs_from_daily(raw)
    assert out["inputs"]["rainfall_7d_mm"] is None
    assert any("suspect rainfall" in n for n in out["notes"])


def test_inputs_from_daily_feed_predict(synthetic_model_dir, allow_synthetic):
    raw = agreed_frame(ACTUAL_CSV, n_days=90)
    day = raw.loc[raw["soil_moisture_raw"].notna()].iloc[-1]["date"]
    r = predict(inputs.inputs_from_daily(raw, as_of=day)["inputs"], model_dir=synthetic_model_dir)
    assert r["status"] == "ok" and "et0_mm" in r["imputed"]
