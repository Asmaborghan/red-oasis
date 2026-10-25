"""
Irrigation rule (Week 3, provisional): combines the 3-day soil-moisture
forecast with today's soil moisture, recent rain and ET0.

Why: the forecast alone is nearly flat (on the Dariz test period its
spread is about 1.1 percentage points against 2.8 for the observed
values), so a level based on the forecast alone says little. The rule
adds three observed signals. See model/IRRIGATION_RULE.md.

Four checks, each worth 1 point when it points towards irrigation:
  1. Soil now is dry for this site:   soil_now <= site median soil moisture
  2. Forecast stays dry:              forecast <= site median soil moisture
  3. Recent rain does not cover the
     next 3 days of demand:           rain_7d < 3 x ET0
  4. Evaporative demand is high:      ET0 >= site median ET0
Level: 3-4 points -> "irrigate", 2 -> "monitor", 0-1 -> "none".
A missing input (rain or ET0) means that check cannot score; the result
says so and the level can then be at most "monitor".

The site medians come from the training rows recorded in
model/artifacts/metadata.json, so they follow the data the model was
trained on (currently Dariz, 2024-2026). They describe what is usual for
this SMAP pixel, NOT crop water requirements: this is a transparent
decision-support heuristic, not validated agronomic advice.

predict() is not changed; this module calls it.
"""

from __future__ import annotations

import math

from .predict import load_model, predict

RULE_VERSION = "1.0 (provisional, 25 Oct 2026)"
HORIZON_DEMAND_DAYS = 3          # demand window = forecast horizon
IRRIGATE_MIN_SCORE = 3
MONITOR_MIN_SCORE = 2
CAVEAT = (
    "Provisional rule, not validated irrigation advice. Reference values are this site's "
    "satellite record (SMAP ~9 km), not crop thresholds, and the model is experimental. "
    "Confirm with the grower before acting."
)


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def site_reference(metadata: dict) -> dict:
    """Reference values for the rule, taken from the model's training rows."""
    medians = metadata["training_median"]
    return {
        "soil_median_pct": float(medians["soil_moisture_pct"]),
        "et0_median_mm": (None if medians.get("et0_mm") is None else float(medians["et0_mm"])),
        "source": f"training rows of the saved model ({metadata.get('data_source', 'unknown')}, "
                  f"{' to '.join(metadata.get('date_range', []))})",
    }


def irrigation_rule(forecast_pct: float, soil_now_pct: float, rain_7d_mm: float | None,
                    et0_mm: float | None, reference: dict) -> dict:
    """Apply the four checks. Pure function: no model or file access."""
    if _missing(forecast_pct) or _missing(soil_now_pct):
        raise ValueError("forecast_pct and soil_now_pct are required.")
    soil_ref = reference["soil_median_pct"]
    et0_ref = reference.get("et0_median_mm")
    checks = [
        {"name": "soil_now_dry", "met": soil_now_pct <= soil_ref, "value": soil_now_pct,
         "reference": soil_ref,
         "text": f"Soil moisture now {soil_now_pct:.1f}% vs site median {soil_ref:.1f}%"},
        {"name": "forecast_dry", "met": forecast_pct <= soil_ref, "value": forecast_pct,
         "reference": soil_ref,
         "text": f"Forecast in 3 days {forecast_pct:.1f}% vs site median {soil_ref:.1f}%"},
    ]
    notes = []
    if _missing(rain_7d_mm) or _missing(et0_mm):
        checks.append({"name": "rain_below_demand", "met": None, "value": None, "reference": None,
                       "text": "Rain vs demand not checked (rain or ET0 missing)"})
        notes.append("Rain or ET0 missing: the rain-vs-demand check was not scored.")
    else:
        demand = HORIZON_DEMAND_DAYS * et0_mm
        checks.append({"name": "rain_below_demand", "met": rain_7d_mm < demand, "value": rain_7d_mm,
                       "reference": demand,
                       "text": f"Rain in last 7 days {rain_7d_mm:.1f} mm vs 3-day demand "
                               f"{demand:.1f} mm (3 x ET0)"})
    if _missing(et0_mm) or et0_ref is None:
        checks.append({"name": "high_demand", "met": None, "value": None, "reference": et0_ref,
                       "text": "Evaporative demand not checked (ET0 missing)"})
        notes.append("ET0 missing: the demand check was not scored.")
    else:
        checks.append({"name": "high_demand", "met": et0_mm >= et0_ref, "value": et0_mm,
                       "reference": et0_ref,
                       "text": f"ET0 today {et0_mm:.1f} mm/day vs site median {et0_ref:.1f} mm/day"})

    score = sum(1 for c in checks if c["met"] is True)
    scored = sum(1 for c in checks if c["met"] is not None)
    level = ("irrigate" if score >= IRRIGATE_MIN_SCORE
             else "monitor" if score >= MONITOR_MIN_SCORE else "none")
    if scored < len(checks) and level == "irrigate":
        level = "monitor"
        notes.append("Not all checks could be scored, so the level is capped at 'monitor'.")
    return {"level": level, "score": score, "checks_scored": scored, "max_score": len(checks),
            "checks": checks, "notes": notes, "rule_version": RULE_VERSION, "caveat": CAVEAT}


def advise(inputs: dict, *, model_dir=None) -> dict:
    """Forecast with predict(), then apply the rule. Never raises.

    Returns {"status", "level", "forecast_pct", "rule", "warnings", "errors"}.
    "status" uses predict()'s values; "level" is None unless status is "ok".
    """
    result = predict(inputs, explain=False, model_dir=model_dir)
    out = {"status": result["status"], "level": None, "forecast_pct": None, "rule": None,
           "warnings": list(result["warnings"]), "errors": list(result["errors"])}
    if result["status"] != "ok":
        return out
    try:
        bundle = load_model(model_dir)
        reference = site_reference(bundle["metadata"])
        used = result["inputs_used"]
        rule = irrigation_rule(
            result["prediction"]["value"], used["soil_moisture_pct"],
            used.get("rainfall_7d_mm"), used.get("et0_mm"), reference)
        rule["reference"] = reference
        out.update({"level": rule["level"], "forecast_pct": result["prediction"]["value"],
                    "rule": rule})
    except Exception as exc:  # keep the website's fallback path simple
        out.update({"status": "error", "errors": out["errors"] + [f"{type(exc).__name__}: {exc}"]})
    return out


__all__ = ["irrigation_rule", "site_reference", "advise", "RULE_VERSION"]
