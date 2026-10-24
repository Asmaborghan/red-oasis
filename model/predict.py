"""
predict(): the single entry point the website calls.

Example (from the repository root):

    from model.predict import predict
    result = predict({
        "soil_moisture_pct": 11.5,   # % volumetric (SMAP 0.115 m3/m3)
        "ndvi": 0.23,
        "rainfall_7d_mm": 0.0,
        "temperature_c": 34.8,
        "et0_mm": 7.3,
    })
    if result["status"] == "ok":
        print(result["prediction"]["value"], result["recommendation"]["level"])

Inputs (all numbers; see config.FEATURES for limits):
    soil_moisture_pct  required  % volumetric soil moisture observed today (SMAP x 100)
    ndvi               optional  latest available NDVI composite, -1 to 1
    rainfall_7d_mm     optional  rainfall total over the 7 days up to today, mm
    temperature_c      optional  daily mean air temperature, deg C
    et0_mm             optional  reference evapotranspiration today, mm/day
Missing optional inputs are filled with the training median and listed in
result["imputed"].

Output: a dict. result["status"] is one of
    "ok"                 prediction made
    "model_unavailable"  no trained model saved yet (website should fall back)
    "invalid_input"      result["errors"] says what is wrong
    "error"              unexpected failure; result["errors"] has the message
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import config


def _empty_result(status: str) -> dict:
    return {
        "status": status,
        "prediction": None,
        "recommendation": None,
        "explanation": None,
        "inputs_used": {},
        "imputed": [],
        "warnings": [],
        "errors": [],
        "model": None,
    }


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime: float):
    import joblib

    folder = Path(path)
    pipeline = joblib.load(folder / config.MODEL_FILE)
    metadata = json.loads((folder / config.METADATA_FILE).read_text())
    eval_path = folder / config.EVALUATION_FILE
    evaluation = json.loads(eval_path.read_text()) if eval_path.exists() else None
    return {"pipeline": pipeline, "metadata": metadata, "evaluation": evaluation, "explainer": None}


def _version_problem(meta: dict) -> str | None:
    """A saved scikit-learn model must be loaded with the same minor version."""
    import sklearn

    saved = (meta.get("library_versions") or {}).get("scikit-learn")
    if not saved:
        return "The model metadata does not record its scikit-learn version."
    if saved.split(".")[:2] != sklearn.__version__.split(".")[:2]:
        return (f"The model was saved with scikit-learn {saved} but {sklearn.__version__} is "
                f"installed. Install scikit-learn=={saved} or retrain.")
    return None


def _check_model(model_dir) -> tuple[dict | None, str | None, str | None]:
    """Return (bundle, None, None) or (None, reason, predict-status). Never raises.

    The predict-status keeps the pre-audit meaning: "model_unavailable" when
    there is no usable model by design (not trained, synthetic blocked),
    "error" when a saved model exists but cannot be used (corrupt,
    incomplete or incompatible artifact).
    """
    folder = Path(model_dir) if model_dir else config.model_dir()
    model_file = folder / config.MODEL_FILE
    if not model_file.exists() or not (folder / config.METADATA_FILE).exists():
        return None, "No trained model found. Train one with: python -m model.train", \
            "model_unavailable"
    try:
        meta = json.loads((folder / config.METADATA_FILE).read_text())
        problem = _version_problem(meta)
        if problem:
            return None, problem, "error"
        bundle = _load_cached(str(folder.resolve()), model_file.stat().st_mtime)
        for key in ("feature_order", "target", "horizon_days", "training_median"):
            if key not in bundle["metadata"]:
                return None, f"The model metadata is incomplete (missing '{key}').", "error"
    except Exception as exc:  # corrupt, truncated or incompatible artifact
        return None, (f"The saved model could not be loaded "
                      f"({type(exc).__name__}: {exc})."), "error"

    if bundle["metadata"].get("is_synthetic"):
        # Defence in depth: model/artifacts/ is the production location, so a
        # synthetic model found there is refused even if the override is set.
        if folder.resolve() == config.DEFAULT_MODEL_DIR.resolve():
            return None, "A SYNTHETIC model was found in model/artifacts/ and is refused.", \
                "model_unavailable"
        if not config.synthetic_allowed():
            return None, (
                "The saved model was trained on SYNTHETIC data and is blocked. "
                f"Set {config.ALLOW_SYNTHETIC_ENV}=1 only for a local interface test."
            ), "model_unavailable"
    return bundle, None, None


def load_model(model_dir: str | Path | None = None):
    """Load the saved model once and reuse it. Returns None if not trained
    yet, or if it is a synthetic model and synthetic models are not allowed."""
    return _check_model(model_dir)[0]


def model_status(model_dir: str | Path | None = None) -> dict:
    """Small summary for the website: is a model available, and how good is it?

    The experimental/reliability verdict is in status["evaluation"]
    ("model_status_label", "reliable"); the return keys are unchanged.
    """
    bundle, reason, _ = _check_model(model_dir)
    if bundle is None:
        return {"available": False, "reason": reason}
    meta, ev = bundle["metadata"], bundle["evaluation"] or {}
    return {
        "available": True,
        "is_synthetic": meta.get("is_synthetic", False),
        "data_location": meta.get("data_location", {}),
        "target": meta["target"],
        "target_label": meta["target_label"],
        "target_unit": meta["target_unit"],
        "horizon_days": meta["horizon_days"],
        "data_source": meta.get("data_source"),
        "date_range": meta.get("date_range"),
        "global_importance": meta.get("global_importance", []),
        "evaluation": ev,
    }


def validate_inputs(inputs) -> tuple[dict, list[str], list[str]]:
    """Return (clean values, missing feature names, error messages)."""
    if not isinstance(inputs, dict):
        return {}, [], ["Inputs must be a dict of feature name -> number."]
    errors = []
    unknown = sorted(set(inputs) - set(config.FEATURES))
    if unknown:
        errors.append(f"Unknown input(s): {unknown}. Expected: {config.FEATURE_ORDER}.")

    clean, missing = {}, []
    for name, spec in config.FEATURES.items():
        raw = inputs.get(name)
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            missing.append(name)
            continue
        if isinstance(raw, bool):
            errors.append(f"{name} must be a number, not a boolean.")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{name} must be a number, got {raw!r}.")
            continue
        if not math.isfinite(value):
            errors.append(f"{name} must be finite, got {raw!r}.")
            continue
        if not spec["min"] <= value <= spec["max"]:
            errors.append(
                f"{name}={value} is outside the valid range "
                f"{spec['min']} to {spec['max']} ({spec['unit']})."
            )
            continue
        clean[name] = value

    for name in sorted(config.REQUIRED_FEATURES & set(missing)):
        errors.append(f"{name} is required ({config.FEATURES[name]['unit']}).")
    return clean, missing, errors


def _recommend(meta: dict, value: float, thresholds: dict) -> tuple[dict | None, list[str]]:
    """Turn the forecast into a level, using the website's thresholds.

    Returns (recommendation, notes). The recommendation keeps the pre-audit
    keys (level, basis, thresholds_used, caveat); notes go to warnings.
    """
    spec = config.TARGETS[meta["target"]]
    dry = thresholds.get(spec["dry_key"])
    moderate = thresholds.get(spec["moderate_key"]) if spec["moderate_key"] else None
    if dry is None:
        return None, []
    if value < dry:
        level = "irrigate"
    elif moderate is not None and value < moderate:
        level = "monitor"
    else:
        level = "none"
    notes = []
    dist = meta.get("target_distribution")
    if dist and dry > dist["p95"]:
        notes.append(
            f"The dry threshold ({dry}) is above 95% of observed values in the training data "
            f"(p95 = {dist['p95']:.1f}), so this level is 'irrigate' almost every day."
        )
    recommendation = {
        "level": level,
        "basis": (
            f"Forecast {meta['target_label'].lower()} in {meta['horizon_days']} days "
            f"({value:.2f}) compared with thresholds: dry < {dry}"
            + (f", moderate < {moderate}" if moderate is not None else "")
            + "."
        ),
        "thresholds_used": {spec["dry_key"]: dry, **({spec["moderate_key"]: moderate} if moderate is not None else {})},
        # Not validated agronomic advice (config.RECOMMENDATIONS_VALIDATED is False).
        "caveat": (
            "Experimental level, not validated irrigation advice: the thresholds are not "
            "crop- or site-specific, no crop-stress measurements exist, and SMAP soil "
            "moisture is a ~9 km regional signal that does not see water inside a "
            "falaj-irrigated orchard. Confirm with the grower before acting."
        ),
    }
    return recommendation, notes


def predict(inputs: dict, *, explain: bool = True, thresholds: dict | None = None,
            model_dir: str | Path | None = None) -> dict:
    """Forecast the target and, if possible, explain it. Never raises."""
    try:
        bundle, reason, status = _check_model(model_dir)
        if bundle is None:
            result = _empty_result(status)
            result["errors"].append(reason)
            return result

        clean, missing, errors = validate_inputs(inputs)
        if errors:
            result = _empty_result("invalid_input")
            result["errors"] = errors
            return result

        meta = bundle["metadata"]
        unavailable = set(meta.get("unavailable_features", []))
        result = _empty_result("ok")
        for name in sorted(unavailable & set(clean)):
            # The model never saw this input vary, so a value cannot change the forecast.
            clean.pop(name)
            result["warnings"].append(f"{name} was not available in the training data and is ignored.")
        result["imputed"] = [m for m in missing
                             if m not in config.REQUIRED_FEATURES and m not in unavailable]
        result["inputs_used"] = clean
        result["model"] = {
            "type": meta["model_type"],
            "trained_at": meta["trained_at"],
            "data_source": meta["data_source"],
            "is_synthetic": meta.get("is_synthetic", False),
            "n_rows": meta["n_rows"],
            "data_location": meta.get("data_location", {}),
        }
        if meta.get("is_synthetic"):
            result["warnings"].append(
                "This model was trained on SYNTHETIC data for interface testing. "
                "Its output is not a real forecast."
            )
        if not meta.get("reliable", False):
            result["warnings"].append(
                "Experimental model: it has not reliably beaten the simple 'no change' "
                "(persistence) forecast in evaluation. Do not use it as a validated forecast."
            )
        for name in result["imputed"]:
            result["warnings"].append(
                f"{name} was missing and was filled with the training median "
                f"({meta['training_median'][name]:.3g})."
            )
        for name, value in clean.items():
            rng = meta["training_range"][name]
            if rng["min"] is not None and not rng["min"] <= value <= rng["max"]:
                result["warnings"].append(
                    f"{name}={value:g} is outside the training data range "
                    f"({rng['min']:.3g} to {rng['max']:.3g}); the forecast is less reliable."
                )

        X = pd.DataFrame([{f: clean.get(f, float("nan")) for f in config.FEATURE_ORDER}])
        value = float(bundle["pipeline"].predict(X)[0])
        result["prediction"] = {
            "target": meta["target"],
            "label": meta["target_label"],
            "value": value,
            "unit": meta["target_unit"],
            "horizon_days": meta["horizon_days"],
        }
        result["recommendation"], notes = _recommend(
            meta, value, {**config.DEFAULT_THRESHOLDS, **(thresholds or {})}
        )
        result["warnings"].extend(notes)

        if explain:
            try:
                from . import explain as shap_explain

                if bundle["explainer"] is None:
                    bundle["explainer"] = shap_explain.make_explainer(bundle["pipeline"])
                exp = shap_explain.explain_row(
                    bundle["pipeline"], X, bundle["explainer"], imputed=result["imputed"]
                )
                exp["summary"] = shap_explain.summary_text(
                    exp, meta["target_label"], meta["target_unit"], meta["horizon_days"]
                )
                result["explanation"] = exp
            except Exception as exc:  # SHAP is optional; the forecast still stands
                result["warnings"].append(f"SHAP explanation unavailable: {exc}")
        return result

    except Exception as exc:
        result = _empty_result("error")
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        return result


if __name__ == "__main__":
    # Illustrative values in the range of the Nizwa-area training data.
    example = {"soil_moisture_pct": 11.5, "ndvi": 0.23, "rainfall_7d_mm": 0.0,
               "temperature_c": 34.8, "et0_mm": 7.3}
    print(json.dumps(predict(example), indent=2))
