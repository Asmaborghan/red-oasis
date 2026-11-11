# Red Oasis — AI/ML component (`model/`)

## Current status

| Item | Current state |
|---|---|
| Active model | `model/artifacts/model.joblib` |
| Model type | Random Forest regressor with median imputation |
| Study location | Ad Dariz palm oasis, Ibri, Oman (23.308639° N, 56.613083° E) |
| Forecast horizon | 3 days |
| Model status | **EXPERIMENTAL — not considered reliable** |
| Irrigation recommendations | Provisional rule only; **not validated agronomic advice** |
| Website integration | The Streamlit app imports the model and provisional irrigation rule |
| Production readiness | **Research prototype only** |

The active model directory is `model/artifacts/`, which is the default model directory in `model/config.py`. The longer-record experiment in `model/artifacts_long/` is kept separately for comparison and does not replace the active model.

## What the model predicts

The model forecasts **observed NASA SMAP surface soil moisture (% volumetric) three days after the input date**.

It uses these five inputs:

| Model input | Meaning | Data source |
|---|---|---|
| `soil_moisture_pct` | Observed soil moisture today | NASA SMAP SPL3SMP_E |
| `ndvi` | Latest available vegetation index | MODIS MOD13Q1 / VIIRS VNP13A1 |
| `rainfall_7d_mm` | Total rainfall over the previous 7 days | NASA GPM IMERG Late V07 |
| `temperature_c` | Daily mean air temperature | NASA POWER T2M |
| `et0_mm` | Reference evapotranspiration (ET0) | NASA POWER; FAO-56 Penman–Monteith derivation |

**Important interpretation limits:** SMAP represents a regional signal at approximately 9 km resolution. It is not a measurement inside an individual orchard and does not detect falaj irrigation water in the root zone. The model does not predict date-palm water stress, crop health, irrigation demand or yield. Those outcomes have no field labels or measurements in the current training data.

SHAP explanations describe how the model uses its inputs; they do not establish physical causes.

## Active model: Dariz 2024–2026

The active artifact was trained on the Dariz daily dataset covering 1 January 2024 to 30 September 2026 (1,004 daily records). The evaluation retained **608 usable rows** after preprocessing and quality checks.

The evaluation used a chronological split without shuffling:

- Training feature dates: 8 January 2024 to 13 March 2026 (**484 rows**).
- Final test feature dates: 18 March 2026 to 27 September 2026 (**122 rows**).
- A three-day gap prevents training targets from overlapping the test period.

### Evaluation results

MAE and RMSE are in percentage points of volumetric soil moisture. Lower MAE is better.

| Evaluation | Random Forest | Persistence (same as today) |
|---|---:|---:|
| Cross-validation MAE, 5 time-ordered folds | 1.178 | 1.030 |
| Final test MAE | 1.204 | 1.307 |

The Random Forest performed better than persistence on the final test set, but worse on average in cross-validation and in only 2 of the 5 folds. The pre-defined reliability rule requires the model to beat persistence in cross-validation, in a majority of folds, and on the final test. It therefore remains **experimental** (`reliable: false`).

The active evaluation is recorded in `model/artifacts/evaluation.json`; model details and training medians are in `model/artifacts/metadata.json`.

### SHAP global importance

The active model's mean absolute SHAP values indicate the relative influence of inputs on its predictions (in percentage points):

| Input | Mean absolute SHAP value |
|---|---:|
| Soil moisture today | 0.367 |
| Rainfall over the previous 7 days | 0.270 |
| NDVI | 0.143 |
| Air temperature | 0.143 |
| ET0 | 0.108 |

These values explain model behaviour; they are not evidence of causation.

## Longer-record experiment: 2015–2026

A separate experiment was trained from the extended Dariz dataset and saved in `model/artifacts_long/`. It did **not** overwrite `model/artifacts/`.

- Input record: 4,314 daily dates, 1 January 2015 to 23 October 2026.
- Chronological training period: 2 April 2015 to 4 August 2024 (**1,951 rows**).
- Final test period: 8 August 2024 to 17 October 2026 (**489 rows**).

| Evaluation | Random Forest | Persistence (same as today) |
|---|---:|---:|
| Cross-validation MAE, 5 time-ordered folds | 1.010 | 1.155 |
| Final test MAE | 1.086 | 1.043 |

The Random Forest beat persistence in all five cross-validation folds, but its final-test MAE was higher than persistence by approximately 4.2%. It therefore still fails the reliability rule and remains **experimental** (`reliable: false`).

**Decision:** keep `model/artifacts/` as the active model. Keep `model/artifacts_long/` as a comparison experiment only. Its files include `evaluation.json`, `metadata.json`, `model_rows_audit.csv` and `model.joblib`.

## Provisional irrigation rule

The separate rule in `model/irrigation.py` combines the three-day forecast with today's soil moisture, recent rainfall and ET0. Each of four checks contributes one point:

| Check | Scores when |
|---|---|
| Soil moisture today | Today's soil moisture is at or below the site's training median |
| Three-day forecast | Forecast soil moisture is at or below the site's training median |
| Recent rain versus demand | Seven-day rainfall is less than `3 × ET0` |
| Evaporative demand | Today's ET0 is at or above the site's training median |

| Score | Output level |
|---:|---|
| 3–4 | `irrigate` |
| 2 | `monitor` |
| 0–1 | `none` |

For the active model, the reference medians are **8.79% soil moisture** and **5.915 mm/day ET0**, taken from `model/artifacts/metadata.json`. They describe typical values in this satellite record; they are **not date-palm water requirements or crop-specific thresholds**.

If rainfall or ET0 is missing, the affected checks are skipped. The level is capped at `monitor` when not all checks can be scored. The rule does not invent missing inputs.

The rule is descriptive and provisional. There are no field measurements, crop-stress labels or grower irrigation logs to validate its recommendations. See `model/IRRIGATION_RULE.md` for the full rule, rationale and limits.

## Streamlit app integration

`app/app.py` imports:

- `model.predict` for the three-day forecast and model status;
- `model.irrigation.advise()` for the provisional four-check irrigation rule.

When the trained model returns a usable result, the app displays the forecast and uses the provisional rule when available. If the model cannot provide a usable result, the app falls back to its simpler demonstration rules.

The advisor values are entered manually in the app. They are **not fetched live from NASA services**. The model itself was trained using the NASA-derived Dariz dataset described above.

The website's illustrative soil-moisture cut-offs are 8.4% and 9.3%. These are based on the Dariz satellite record, not on validated date-palm agronomy. Do not present them, or the provisional rule, as validated irrigation advice.

## Data quality and preprocessing

The training pipeline is designed to avoid time leakage:

- Only observed soil-moisture measurements are used; gap-filled values are excluded.
- NDVI composites are used only after their 16-day composite period has completed, then carried forward for up to 16 days.
- A seven-day rainfall total is used only when all seven daily totals are available.
- Days flagged `rain_suspect = 1`, and rows whose feature window or target touches those days, are excluded.
- The target is the observed soil-moisture value exactly three days after the feature date.
- Imputation and model fitting are performed on the training data only during evaluation.
- Evaluation uses chronological splits and time-ordered cross-validation, not random shuffling.

The row-level inclusion and exclusion records for each model run are stored alongside its artifacts as `model_rows_audit.csv`.

The data validation code checks date uniqueness, numeric values and declared units. It does not silently rename unknown columns. Use `python -m model.train --inspect` to review the CSV columns, units and validation before training.

## Main files

| Path | Purpose |
|---|---|
| `model/config.py` | Model settings, features, target, paths and reliability rule |
| `model/data.py` | Input validation, column mapping and leakage-safe feature preparation |
| `model/train.py` | Training, chronological evaluation and artifact creation |
| `model/predict.py` | Model loading, status checks, prediction and explanation interface |
| `model/irrigation.py` | Provisional four-check irrigation rule and `advise()` interface |
| `model/IRRIGATION_RULE.md` | Full explanation and limitations of the irrigation rule |
| `model/MODEL_CARD.md` | Plain-language model summary |
| `model/artifacts/` | Active Dariz 2024–2026 model and evaluation files |
| `model/artifacts_long/` | Separate 2015–2026 comparison experiment |

## Dependencies and environment

The repository's current `requirements.txt` includes:

```text
streamlit>=1.40
pandas
numpy
scikit-learn==1.9.1
shap>=0.46
joblib
```

The active model metadata records scikit-learn 1.9.1, SHAP 0.54.1, NumPy 2.4.4, pandas 3.0.2 and joblib 1.5.3 (training Python 3.12.3). The long-record experiment records scikit-learn 1.9.1, SHAP 0.54.1, NumPy 2.5.4, pandas 3.0.6 and joblib 1.6.0 (training Python 3.14.4). Keep the scikit-learn version aligned with the saved artifacts when loading them; successful deployment should be verified in the target runtime.

## Commands

Run commands from the repository root.

Install application/model dependencies:

```bash
pip install -r requirements.txt
```

Inspect the default training CSV before training:

```bash
python -m model.train --inspect
```

Train and evaluate the active model using the default data file (`data/raw/red_oasis_daily.csv`). This writes to `model/artifacts/`, so do not run it when you intend to preserve the current saved model without replacing it:

```bash
python -m model.train
```

Train a separate long-record experiment without overwriting the active artifacts:

```bash
python -m model.train --data data/raw/red_oasis_daily_2015_2026.csv --out model/artifacts_long
```

Run the test suite (install `pytest` separately if it is not already available):

```bash
pip install pytest
python -m pytest -q -rs
```

## Limitations and responsible use

1. **Regional measurement:** SMAP is approximately 9 km resolution and does not measure the palm root zone or falaj water inside an orchard.
2. **Narrow prediction target:** the model predicts observed satellite soil moisture, not crop stress, crop health, irrigation demand or yield.
3. **Experimental performance:** neither the active model nor the longer-record experiment satisfies the pre-defined reliability rule.
4. **Unvalidated recommendations:** the provisional rule and website cut-offs have not been checked against field measurements, date-palm stress labels or grower irrigation records.
5. **Single study point:** the model was trained for the Dariz study point; it has not been validated across other farms or regions.
6. **No live satellite feed in the advisor:** the current Streamlit advisor uses manually entered values, not live NASA API data.
7. **SHAP is explanatory, not causal:** feature contributions describe model behaviour and do not prove physical relationships.

Red Oasis is a research and decision-support prototype. Forecasts and irrigation levels should not be treated as operational or agronomic instructions.
