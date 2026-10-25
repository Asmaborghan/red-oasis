# Red Oasis — AI component (`model/`)

Owner: Yaqeen (AI/ML). Last audit: 22 October 2026; team-lead revisions 23 October 2026.

## Current state (read this first)

| Item | State |
|---|---|
| Saved model | `model/artifacts/model.joblib`, retrained 24 Oct 2026 on the **Dariz** data from Amna |
| Label | **EXPERIMENTAL**: it does not reliably beat the simple "no change" (persistence) forecast |
| Data location | **Dariz farm point** (`Farm_23.3086_56.6131`, 23.308639° N, 56.613083° E), 2024-01-01 to 2026-09-30 |
| Irrigation levels | **Not validated advice.** They are always marked experimental |
| Website | Not connected yet. Do not integrate until Dariz data and credible thresholds exist |
| Production-ready | **No** |

## Update 24 October 2026: retrained on Dariz data

Amna delivered the Dariz dataset (same 21 columns, 1,004 days, no gaps or
duplicates). `python -m model.train` was run on it with no code change.
Rows used: 608 of 1,004 (484 train, 122 final test).

| Result (percentage points of soil moisture) | Random Forest | Persistence ("same as today") |
|---|---|---|
| Final test MAE | 1.204 | 1.307 |
| Cross-validation MAE (training period) | 1.178 | 1.030 |

- The Random Forest is better than persistence on the final test (about 8%)
  but worse in cross-validation, so the verdict is still **experimental**
  (`reliable = false`).
- Dariz soil moisture is drier than Nizwa (median about 8.8%), so the website's
  thresholds (30% / 45%) are still far above the observed range.
- The tables further down were produced on the earlier **Nizwa** run (22 Oct)
  and are kept only for comparison. The saved model and
  `model/artifacts/evaluation.json` are the Dariz run.
- Tests: the Nizwa-specific expected values in `tests/test_real_data.py` were
  replaced by values read from the data file. 79 passed, 1 skipped.

## Week 3 additions (25–29 Oct 2026, Yaqeen)

| File | What it is |
|---|---|
| `MODEL_CARD.md` | One-page model card in plain language |
| `IRRIGATION_RULE.md` + `irrigation.py` | Provisional irrigation rule: forecast + soil now + rain + ET0 → irrigate / monitor / none (`advise()`) |
| `figures/fig1_forecast_vs_actual.png` | 3-day forecast vs observed soil moisture, final test period |
| `figures/fig2_error_rain_vs_no_rain.png` | Error with and without recent rain |
| `figures/fig3_shap_summary.png` | SHAP summary, test rows |
| `figures/figures_data.json` | The numbers behind the three figures |
| `figures.py` | Redraws the figures: `python -m model.figures` (needs `pip install matplotlib`) |
| `CHALLENGE_CLAIMS_CHECKLIST.md` | Current model claims, to check against the official challenge text (pending) |

**Changes:**
- `python -m model.train --data <file>` trains on any daily CSV with the same
  columns, such as Amna's coming `red_oasis_daily_2015_2026.csv`.
- With `--out`, the row audit is now written next to that model. Before, it
  was always written to `model/artifacts/`.
- `data.unconfirmed_agreed_columns()` no longer reports the internal column
  name `solar_radiation`. This removes the false warning stored in the
  24 Oct `evaluation.json`.

**Checks run:**
- A rerun on the current Dariz file reproduced every saved metric exactly.
- The website's thresholds are now 8.4% / 9.3% (Dariz site percentiles), so
  the "30% / 45%" sentence above is out of date. The forecast alone never fell
  below 8.4% on the test days.
- Tests (24 Oct): **101 passed** with the data file.

## What the model predicts

The model forecasts **observed SMAP surface soil moisture (% volumetric) 3 days ahead**.

- SMAP is a regional signal about 9 km across. It is not a measurement
  inside an orchard, and it does not see falaj irrigation water.
- The model does **not** predict date-palm water stress, crop health,
  irrigation demand or yield. No crop-stress labels or field measurements
  exist to train or check such a prediction.
- SHAP values explain how this model uses its inputs. They are not proof
  of physical cause.

## Data

The file is `data/raw/red_oasis_daily.csv` (from Amna, unchanged), with
sources in `data/raw/red_oasis_sources.md`.

- 1,004 consecutive days, 2024-01-01 → 2026-09-30.
- No duplicate dates and no missing days. 4 days are flagged `rain_suspect = 1`.
- Units were checked against the values in the file:

| Variable | Unit | Range in file |
|---|---|---|
| Soil moisture | m³/m³ | 0.093–0.311 |
| Temperature | °C | 13.5–39.2 |
| Rain | mm/day | 0–247 |
| NDVI | — | 0.10–0.36 |
| Solar radiation | MJ/m²/day | 6.7–29.8 |

### Column mapping (explicit)

`data.column_mapping()` reports exactly which column is used for each
variable, and `python -m model.train --inspect` prints it.

Only the **actual CSV column names** (Amna's delivered file) and the
pre-audit aliases are mapped.

| Model variable | Actual CSV (used) | Agreed schema, roadmap PDF (not mapped) | Agreed schema, audit brief (not mapped) |
|---|---|---|---|
| soil moisture | `soil_moisture_raw` | `soil_moisture`* | `moisture_soil` |
| temperature | `temp_mean_c` | `temp_c` | `c_temp` |
| rainfall | `rain_gpm_mm` | `rain_mm` | `mm_rain` |
| NDVI | `ndvi_raw` | `ndvi`* | `ndvi`* |
| ET0 | `et0_mm` | — (not in schema) | — (not in schema) |
| solar radiation (not a model input) | `solar_mj_m2` | `solar_radiation` | `radiation_solar` |

\* Already a pre-audit alias, so it is recognised.

**Open conflict:** the two "agreed" spellings differ; one looks like the
other with the words reversed around the underscore. Neither is mapped
until Amna confirms which one the final file uses.
- A file using them gets a warning naming the columns that were **not used**.
- If the required soil-moisture column isn't found, training stops with an error.
- Columns are never renamed silently.

### Validation (`data.validate_raw`)

The file is rejected if:
- a date appears twice for the same site
- a value is not a number
- a value doesn't fit its declared unit. For example, soil moisture
  already in % while m³/m³ is declared, temperature in Kelvin, negative
  rain, or NDVI scaled by 10,000.

Soil moisture is multiplied by 100 only when it is declared as m³/m³ and
the values fit that unit.

The file is accepted with a warning when:
- the location isn't Dariz
- there's no `rain_suspect` column
- there's no ET0 column
- the series looks gap-filled

## Leakage-safe preprocessing (`data.py`)

| Rule | Why |
|---|---|
| Only observed soil moisture is used (`soil_moisture_raw`; filled values are discarded) | Gap-filled values were interpolated from later days |
| NDVI composites are used only 16 days after their composite date, then carried forward up to 16 days | A 16-day composite uses images from its whole period |
| 7-day rainfall needs all 7 daily totals | A missing day is never treated as 0 mm |
| A row is excluded if its 7-day rain record is incomplete (**new in audit**) | Before the audit, 4 such rows were filled with the training median, which is 0 mm |
| `rain_suspect = 1` days are removed from the rain record, and rows touching them (feature date, 7-day window, target date) are excluded | They are satellite artefacts |
| The target is the observed value exactly 3 days later | Checked by tests |
| Imputer, scaler and model are fitted on training rows only | Checked by tests |

Every row and the reason it was used or excluded is in
`model/artifacts/model_rows_audit.csv`. It is kept with the model outputs
because `data/` belongs to the data team.

**Current data:** 597 of 1,004 rows are used.

| Exclusion reason | Rows |
|---|---|
| Suspect rain on the feature date | 4 |
| Suspect rain in the 7-day window | 16 |
| Suspect rain on the target date | 3 |
| Incomplete rain window | 6 |
| No soil-moisture observation today | 252 |
| No observation 3 days later | 126 |

## Evaluation (final run, 22 Oct 2026, `model/artifacts/evaluation.json`)

- **Inputs:** soil moisture, NDVI, 7-day rainfall, mean temperature, ET0.
- **Split:** chronological, with no shuffling.
  - Training: feature dates 2024-01-08 → 2026-03-16 (475 rows).
  - Final test: 2026-03-21 → 2026-09-27 (120 rows). It was used once and
    nothing was tuned on it.
  - Training targets end before the test period starts.
- **Units:** MAE and RMSE are in percentage points of soil moisture
  (median value 11.5%); R² has no unit.

### Time-ordered cross-validation on the training period (5 folds, 79 validation rows each)

| Model | MAE | RMSE | R² | Folds beating persistence |
|---|---|---|---|---|
| Random Forest (saved model) | 1.117 | 1.454 | −0.91 | 2 / 5 |
| Ridge regression | 1.093 | 1.413 | −0.68 | 1 / 5 |
| Linear, soil moisture only | **0.969** | **1.274** | −0.35 | 2 / 5 |
| Persistence ("same as today") | 1.030 | 1.443 | −0.68 | — |
| Training mean | 1.179 | 1.447 | −0.90 | 2 / 5 |

### Final test period (120 rows)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest (saved model) | 1.201 | 2.362 | 0.32 |
| Random Forest with ET0 missing (how the website will usually call it) | 1.261 | 2.387 | 0.31 |
| Ridge regression | 1.302 | 2.571 | 0.20 |
| Linear, soil moisture only | **1.175** | 2.387 | 0.31 |
| Persistence | 1.478 | 2.607 | 0.17 |
| Training mean | 1.359 | 2.934 | −0.05 |

### By weather condition (MAE)

| Days | RF | Soil-only linear | Persistence |
|---|---|---|---|
| Test, rain in last 7 days (n=74) | 1.46 | 1.46 | 1.85 |
| Test, no rain (n=46) | 0.78 | 0.71 | 0.88 |
| CV, rain in last 7 days (n=162) | 1.36 | 1.18 | 1.35 |
| CV, no rain (n=233) | 0.95 | 0.82 | **0.81** |

### Conclusion

- The Random Forest beats persistence on the final test period, but **not**
  in cross-validation (worse on average, 2 of 5 folds).
- Under the rule fixed in advance (`config.RELIABILITY_RULE`), it is **not
  reliable**, so it is labelled **experimental**.
- The simplest model (soil moisture only) is at least as good as the
  Random Forest. The extra inputs add no demonstrated value on this data.
- The Random Forest is kept only as an experimental baseline, so that
  SHAP and the `predict()` interface can be tested.

SHAP global importance (mean |SHAP|, percentage points): soil moisture
0.38, 7-day rainfall 0.37, temperature 0.24, NDVI 0.09, ET0 0.08.

## Retraining decision

The model was **retrained on 22 Oct 2026** because the old artifact no
longer matched the corrected preprocessing. It had been trained on 4 rows
where missing rainfall was filled with 0 mm.

The old results reproduced exactly from the files before the change:
- hold-out MAE 1.23 vs persistence 1.52
- CV 1.12 vs 1.03
- the artifact matched a refit on the same rows

Retraining on the **same Nizwa data does not fix the location mismatch.**
A reliable model for Dariz needs Dariz data. See "Missing inputs" below.

## Irrigation levels

`predict()` compares the forecast with the website's thresholds (30% dry,
45% moderate). These are not crop- or site-specific: 95% of observed
targets are below 15.2%, so the level would be "irrigate" almost every day.

The `THRESHOLDS` themselves are not changed (they belong to the website).
Instead:
- every recommendation's `caveat` says it is an experimental level, not
  validated irrigation advice
- a warning is added when a threshold lies above the observed range

No crop thresholds have been invented.

## Interface (signatures and return formats unchanged)

```python
from model.predict import predict, model_status
from model.inputs import inputs_from_daily      # new conversion layer (additive)

status = model_status()       # never raises; same keys as before the audit
prepared = inputs_from_daily(daily_df)          # daily rows (actual CSV names) -> model inputs
result = predict(prepared["inputs"], thresholds=THRESHOLDS)
```

- `predict()` and `model_status()` return exactly the same keys as before
  the audit. A test checks this.
- `predict()`'s `status` is one of:
  - `ok`
  - `invalid_input`
  - `model_unavailable`: no model by design (not trained, or a synthetic model blocked)
  - `error`: a saved model exists but can't be used (corrupt, incomplete or incompatible)
- The **experimental verdict** is in `model_status()["evaluation"]`
  (`model_status_label`, `reliable`). `predict()` also adds a warning
  while the model is experimental.
- Missing optional inputs are filled with the **training** median and
  listed in `imputed` and `warnings`.
- `soil_moisture_pct` is required and is never imputed.
- `inputs_from_daily()` uses the same feature code as training. It
  returns `None`, and never an invented value, for:
  - soil moisture with no SMAP pass that day
  - an incomplete or suspect rain week
  - NDVI not yet available
  - ET0 when the file has none
- **ET0** isn't in the agreed schema. Radiation-only formulas were tested
  against the Penman–Monteith ET0 in the current file: Makkink r = 0.86,
  but about 1.7 mm/day too low. So ET0 is not computed from the agreed
  columns; it is imputed and reported. If a training file has no ET0 at
  all, training still works and ET0 is recorded as unavailable.
- **Website fallback:** use the existing rule-based advisor whenever
  `model_status()["available"]` is False. While
  `model_status()["evaluation"]["model_status_label"]` is "experimental",
  show any forecast only as an experimental research output.

## Safety

- **Synthetic models:**
  - They can't be saved into `model/artifacts/`.
  - They are blocked unless `RED_OASIS_ALLOW_SYNTHETIC=1` is set, which is
    for local interface tests only. Never set it on Streamlit Cloud.
  - A synthetic model found in `model/artifacts/` is refused even if the
    variable is set.
- **Loading:** `model_status()` never raises.
  - A missing artifact gives `available: False` with a reason, and
    `predict()` returns `model_unavailable`.
  - A corrupt or incomplete artifact, or one saved with a different
    scikit-learn minor version, also gives `available: False` with a
    reason. `predict()` then returns `error`.

## Dependencies and deployment

- The artifact was saved with **scikit-learn 1.9.1**, numpy 2.5.4, shap
  0.54.1, Python 3.14.
- `scikit-learn==1.9.1` is available on Linux for Python 3.11, 3.12 and
  3.13. This was checked with `pip --dry-run`.
  - On **3.12 / 3.13** the same numpy (2.5.4) and shap (0.54.1) versions resolve.
  - On 3.11, numpy falls back to 2.4 and shap to 0.51. This wasn't tested
    with the artifact, so **Python 3.12 is recommended on Streamlit Cloud.**
- `requirements.txt` on `main` lists only streamlit and pandas.
  - Before the website imports the model, `numpy`, `scikit-learn==1.9.1`,
    `shap` and `joblib` must be added. That's Asma's file, so it wasn't
    changed in this audit.
  - `requirements-dev.txt` allows any scikit-learn from 1.4; tests that
    load the artifact need 1.9.1.
- Paths are relative to the repository, and the commands run from the repository root.

## Commands (from the repository root)

```bash
pip install -r requirements-dev.txt
pip install scikit-learn==1.9.1
python -m model.train --inspect      # mapping, units, validation
python -m model.train                # train + evaluate; writes model/artifacts/* (incl. model_rows_audit.csv)
python -m model.predict              # example prediction
python -m pytest -q -rs              # tests
```

Training is bit-for-bit reproducible: two runs produced an identical
`model.joblib` and identical evaluation.

**Tests (23 Oct 2026, 80 tests):**

| Situation | Result |
|---|---|
| With `data/raw/red_oasis_daily.csv` present | 79 passed, 1 skipped |
| Without it | 70 passed, 10 skipped |

The skip in both runs is the website-integration check, which waits until
`app.py` is connected. Without the data file, the 9 real-data tests also skip.

## Missing inputs before the model can be trusted

1. ~~Dariz data (Amna)~~ delivered 24 Oct 2026 (from 2024). A longer record
   (ideally from 2015) would still make the model more reliable.
2. Confirmation of which agreed column spelling the final file uses.
3. Site-appropriate soil-moisture thresholds from credible agronomic
   sources (Shareefa), or a decision to show levels only as experimental.
4. Any ground truth (field soil moisture, ECOSTRESS ESI, grower irrigation
   logs) if the project wants to claim anything about crop water stress.
5. `requirements.txt` updated and Python 3.12 selected before deployment (Asma).
