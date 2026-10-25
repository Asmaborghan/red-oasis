# Irrigation rule (provisional, v1.0, 25 Oct 2026)

Code: `model/irrigation.py`. Owner: Yaqeen.

## Why a rule is needed

The model's 3-day soil-moisture forecast is nearly flat. On the Dariz final
test period (18 Mar to 27 Sep 2026, 122 days), its spread was **1.1**
percentage points, while the observed values spread over **2.8**.

With the website's current thresholds (`THRESHOLDS` in `app/app.py`: dry
< 8.4%, moderate < 9.3%), the forecast alone **never** says "irrigate" on
the 122 test days: the lowest forecast was 8.53%. The earlier 30% / 45%
thresholds said "irrigate" on all 122 days.

Either way the forecast alone hardly separates days, so the rule combines
it with three observed signals.

## The rule

Each check scores **1 point** when it points towards irrigation:

| # | Check | Scores when |
|---|---|---|
| 1 | Soil is dry now | today's soil moisture ≤ site median soil moisture |
| 2 | Forecast stays dry | 3-day forecast ≤ site median soil moisture |
| 3 | Recent rain doesn't cover demand | rain in the last 7 days < 3 × ET0 (3-day reference demand) |
| 4 | Demand is high | today's ET0 ≥ site median ET0 |

| Points | Level |
|---|---|
| 3–4 | `irrigate` |
| 2 | `monitor` |
| 0–1 | `none` |

If rain or ET0 is missing, the checks that need it are not scored and the
level is capped at `monitor`. Nothing is invented for missing inputs.

**Site medians.** These come from the training rows of the saved model
(`model/artifacts/metadata.json`). For the current Dariz model:

| Reference | Value |
|---|---|
| Soil moisture median | 8.79% |
| ET0 median | 5.92 mm/day |

They update automatically when the model is retrained. They describe what is
usual for this SMAP pixel. **They are not crop water requirements.**

## How to call it

```python
from model.irrigation import advise
out = advise({"soil_moisture_pct": 8.5, "ndvi": 0.43, "rainfall_7d_mm": 0.0,
              "temperature_c": 33.0, "et0_mm": 7.0})
out["level"]            # "irrigate" / "monitor" / "none" (None if no forecast)
out["rule"]["checks"]   # each check, its value, reference and a plain-English text
```

`predict()` and `model_status()` are unchanged; `advise()` calls `predict()`.
It never raises: if `predict()` fails, `advise()` returns predict's status
with `level = None`, so the website can fall back to its own rules.

## What it did on the real Dariz test period

This is descriptive only: there are no irrigation labels to validate against.

| | irrigate | monitor | none |
|---|---|---|---|
| Forecast alone, current website thresholds (8.4 / 9.3) | 0 | 61 | 61 |
| Forecast alone, earlier thresholds (30 / 45) | 122 | 0 | 0 |
| This rule | 59 | 49 | 14 |
| Mean **observed** soil moisture 3 days later, by rule level | 8.77% | 9.31% | 13.05% |

The rule does not replace or change `THRESHOLDS`; that belongs to the website.

- "none" appears mainly in March–April, after the late-March storm.
- "irrigate" is most common in May (18 of 19 days).
- The site medians used in this check come from the saved model, which was
  refit on all usable rows, including the test period.

## Limits

- **Not validated advice.** There are no field measurements, crop-stress
  labels or grower irrigation logs to check it against.
- **SMAP is regional.** It is about a 9 km signal and doesn't see falaj water
  inside the orchard. High "observed soil moisture" means wet regional surface
  soil, not a wet root zone.
- **Thresholds are relative to the satellite record.** Crop-based
  soil-moisture limits for date palm (Shareefa's Week 3 research) should
  replace the site medians when they are available.
- **The model itself is experimental.** It doesn't reliably beat persistence
  in cross-validation (see `model/README.md`).
