"""Week 3 tests: irrigation rule, figures, training on a new data path, warning fix."""

import hashlib
import importlib.util
import json

import pytest

from model import config, data, figures, irrigation, train
from model.synthetic import make_synthetic_raw

REF = {"soil_median_pct": 9.0, "et0_median_mm": 6.0, "source": "test"}


# ------------------------------------------------------------ irrigation rule

def test_all_four_checks_met_gives_irrigate():
    r = irrigation.irrigation_rule(8.0, 8.5, 0.0, 7.0, REF)
    assert r["level"] == "irrigate" and r["score"] == 4 and r["checks_scored"] == 4
    assert [c["met"] for c in r["checks"]] == [True, True, True, True]


def test_wet_soil_and_rain_gives_none():
    r = irrigation.irrigation_rule(12.0, 15.0, 40.0, 4.0, REF)
    assert r["level"] == "none" and r["score"] == 0


@pytest.mark.parametrize("args, level", [
    ((8.0, 8.5, 40.0, 4.0), "monitor"),   # dry now + dry forecast only -> 2 points
    ((10.0, 8.5, 0.0, 4.0), "monitor"),   # dry now + rain below demand -> 2 points
    ((10.0, 9.5, 0.0, 4.0), "none"),      # rain below demand only -> 1 point
    ((8.0, 8.5, 0.0, 4.0), "irrigate"),   # 3 points
])
def test_score_to_level(args, level):
    assert irrigation.irrigation_rule(*args, REF)["level"] == level


def test_rain_vs_demand_uses_three_days_of_et0():
    r = irrigation.irrigation_rule(10.0, 10.0, 17.9, 6.0, REF)
    check = next(c for c in r["checks"] if c["name"] == "rain_below_demand")
    assert check["reference"] == pytest.approx(18.0) and check["met"] is True
    r = irrigation.irrigation_rule(10.0, 10.0, 18.0, 6.0, REF)
    assert next(c for c in r["checks"] if c["name"] == "rain_below_demand")["met"] is False


def test_missing_rain_or_et0_is_not_invented_and_caps_level():
    r = irrigation.irrigation_rule(8.0, 8.5, None, None, REF)
    assert r["checks_scored"] == 2 and r["score"] == 2 and r["level"] == "monitor"
    assert r["notes"]
    r = irrigation.irrigation_rule(8.0, 8.5, 0.0, float("nan"), REF)
    assert r["level"] in {"monitor", "none"}


def test_required_inputs():
    with pytest.raises(ValueError):
        irrigation.irrigation_rule(None, 8.0, 0.0, 5.0, REF)


def test_site_reference_comes_from_metadata():
    meta = {"training_median": {"soil_moisture_pct": 8.79, "et0_mm": 5.9},
            "data_source": "x", "date_range": ["2024-01-08", "2026-09-27"]}
    ref = irrigation.site_reference(meta)
    assert ref["soil_median_pct"] == 8.79 and ref["et0_median_mm"] == 5.9


def test_advise_with_synthetic_model(synthetic_model_dir, allow_synthetic, good_inputs):
    out = irrigation.advise(good_inputs, model_dir=synthetic_model_dir)
    assert out["status"] == "ok" and out["level"] in {"irrigate", "monitor", "none"}
    assert out["rule"]["max_score"] == 4 and out["forecast_pct"] is not None
    assert "not validated" in out["rule"]["caveat"]


def test_advise_falls_back_when_no_model(tmp_path, good_inputs):
    out = irrigation.advise(good_inputs, model_dir=tmp_path)
    assert out["status"] == "model_unavailable" and out["level"] is None and out["errors"]


def test_advise_with_saved_real_model(good_inputs):
    if not (config.DEFAULT_MODEL_DIR / config.MODEL_FILE).exists():
        pytest.skip("No saved model")
    out = irrigation.advise({**good_inputs, "soil_moisture_pct": 8.5})
    assert out["status"] == "ok", out["errors"]
    assert out["rule"]["reference"]["soil_median_pct"] > 0


# ------------------------------------------------------------------ figures

def _csv(tmp_path):
    path = tmp_path / "daily.csv"
    make_synthetic_raw().to_csv(path, index=False)
    return path


def test_figure_numbers_must_match_evaluation(tmp_path):
    raw = data.load_raw(_csv(tmp_path))
    summary = figures.summarise(figures.test_period_results(raw))
    fake = {"n_test": summary["n_test"],
            "test_metrics": {"random_forest": {"mae": summary["mae"]["random_forest"] + 0.5},
                             "baseline_persistence": {"mae": summary["mae"]["baseline_persistence"]}}}
    with pytest.raises(figures.FigureCheckError):
        figures.check_against_evaluation(summary, fake)


@pytest.mark.skipif(importlib.util.find_spec("matplotlib") is None,
                    reason="matplotlib not installed (only needed to draw the figures)")
def test_make_figures_writes_three_images(tmp_path):
    csv = _csv(tmp_path)
    model_dir = tmp_path / "m"
    train.train(data.load_raw(csv), "soil_moisture_pct", 3, model_dir, "t", True)
    out = tmp_path / "fig"
    s = figures.make_figures(csv, out, model_dir / config.EVALUATION_FILE)
    names = {"fig1_forecast_vs_actual.png", "fig2_error_rain_vs_no_rain.png",
             "fig3_shap_summary.png"}
    assert set(s["figures"]) == names
    for n in names:
        assert (out / n).stat().st_size > 10_000
    saved = json.loads((out / "figures_data.json").read_text())
    assert saved["mae"]["random_forest"] == pytest.approx(
        json.loads((model_dir / config.EVALUATION_FILE).read_text())
        ["test_metrics"]["random_forest"]["mae"])


def test_committed_figures_match_saved_evaluation():
    fig_json = figures.DEFAULT_OUT / "figures_data.json"
    if not fig_json.exists():
        pytest.skip("Figures not generated")
    s = json.loads(fig_json.read_text())
    ev = json.loads((config.DEFAULT_MODEL_DIR / config.EVALUATION_FILE).read_text())
    figures.check_against_evaluation(s, ev)


# ------------------------------------------------- training on a new data path

def _hashes(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.iterdir() if p.is_file()}


def test_train_accepts_a_new_data_path_and_keeps_production_untouched(tmp_path):
    csv = tmp_path / "red_oasis_daily_2015_2026.csv"   # name of Amna's coming file
    make_synthetic_raw().to_csv(csv, index=False)
    before = _hashes(config.DEFAULT_MODEL_DIR)
    out = tmp_path / "out"
    assert train.main(["--data", str(csv), "--out", str(out)]) == 0
    assert (out / config.MODEL_FILE).exists()
    assert (out / config.ROW_AUDIT_FILE.name).exists()      # audit follows --out
    meta = json.loads((out / config.METADATA_FILE).read_text())
    assert csv.name in meta["data_source"]
    assert _hashes(config.DEFAULT_MODEL_DIR) == before      # production files unchanged


def test_train_reports_missing_data_file(tmp_path):
    with pytest.raises(SystemExit):
        train.main(["--data", str(tmp_path / "missing.csv"), "--out", str(tmp_path / "o")])


# ------------------------------------------------------------- warning fix

def test_no_false_unconfirmed_column_warning_after_load_raw(tmp_path):
    raw = make_synthetic_raw(n_days=60)
    raw["solar_mj_m2"] = 20.0                    # actual CSV name
    path = tmp_path / "f.csv"
    raw.to_csv(path, index=False)
    feats = data.build_features(data.load_raw(path))
    assert not any("NOT used" in w for w in feats.attrs["validation_warnings"])


def test_real_unconfirmed_spelling_is_still_reported():
    raw = make_synthetic_raw(n_days=60).rename(columns={"temperature_c": "c_temp"})
    assert data.unconfirmed_agreed_columns(raw.columns) == ["c_temp"]
