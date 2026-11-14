# Red Oasis

**From the Falaj to Mars.** A NASA-data-informed irrigation decision-support prototype for date-palm farming in Oman, with an illustrative Mars water-management mode. Built for the NASA Space Apps Challenge 2026 (theme: The Next Frontier).

**Challenge:** Field Shift: Adapting Farms with NASA Data. Red Oasis combines NASA Earth observations, local soil and crop information from published studies, and a farmer-priority control (how much yield loss is acceptable to save water). Date palm is a perennial tree crop, so crop rotation does not apply here; the project addresses water conservation and adaptation through irrigation decisions.

**Live app:** https://red-oasis.streamlit.app/ (English and Arabic)

**Demo video (30 seconds):** [Watch on Google Drive](https://drive.google.com/file/d/1ptpcXlE1VFtOKTqoa7zulfonW-ExMkkc/view?usp=sharing)

## What the app shows

- **Challenge alignment:** how each Field Shift requirement is answered, with its status.
- **Study site:** Ad Dariz agricultural area, Ibri, Oman (23.308639° N, 56.613083° E), date palm.
- **Site evidence:** published Falaj al-Dariz water balance (Al-Ghafri, 2006; observations 2002–2003) and its seasonal shortage.
- **Strategy explorer:** published irrigation trial on Khalas palms at Al Kamil and Al Wafi (2015–2016; a different site), filtered by the yield loss the farmer would accept.
- **Irrigation advisor:** a 3-day soil-moisture forecast, a provisional four-check irrigation rule and a SHAP explanation of each forecast.
- **Model performance:** forecast vs observed, error with and without rain, and a SHAP summary on the held-out test period.
- **Mars (concept):** a water-reuse what-if calculator based on NASA reference values and team assumptions.
- **Project limits:** shown in the app and listed below.

## Data

NASA Earth observation data for the Ad Dariz study point: SMAP soil moisture, MODIS/VIIRS NDVI, GPM IMERG rainfall and NASA POWER weather (used to calculate FAO-56 reference evapotranspiration).

- `data/raw/red_oasis_daily.csv`: 1 January 2024 to 30 September 2026 (used by the app's model).
- `data/raw/red_oasis_daily_2015_2026.csv`: 1 January 2015 to 23 October 2026 (longer-record comparison).
- Sources and attribution: `data/raw/red_oasis_sources.md` and `data/raw/red_oasis_sources_2015_2026.md`.

The app uses prepared historical datasets; it does not query NASA services live.

## Model status

An experimental Random Forest forecasts observed SMAP soil moisture three days ahead. On 122 held-out days its error is 1.20 percentage points, against 1.31 for the simple "no change" forecast, but it did not beat that baseline in cross-validation (1.18 vs 1.03), so it is not a validated predictor. A longer 2015–2026 record was also tested (`model/artifacts_long/`): it did better in cross-validation but not on its final test, so the app keeps the 2024–2026 model and the longer run is a comparison only. The model does not predict crop stress, irrigation need or yield.

Details: `model/README.md`, `model/MODEL_CARD.md` and `model/IRRIGATION_RULE.md`.

## Repository

| Folder | Contents |
|---|---|
| `app/` | Streamlit application (`app.py`) and logo |
| `model/` | Data preparation, training, prediction, SHAP, irrigation rule, saved models and figures |
| `data/raw/` | NASA-derived daily datasets and source documentation |
| `docs/field-research/` | Site research brief, dossier, irrigation comparison and project point (KML) |
| `docs/pitch/` | Abstract, presentation, script, judge Q&A and delivery checklist |
| `tests/` | Automated tests |

## Run locally

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py
```

Tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Limitations

- SMAP soil moisture is a regional signal of about 9 km, not a measurement of the orchard root zone, and it cannot see falaj water inside the orchard.
- The model is experimental and forecasts soil moisture only; it does not measure crop stress, irrigation need or yield.
- The irrigation levels come from a provisional rule relative to this site's satellite record; no field measurements or grower logs exist to validate it.
- The Ad Dariz water-balance figures are historical (2002–2003); the irrigation trial is from a different site.
- SHAP shows how the model uses its inputs, not physical cause.
- The Mars mode is a concept based on NASA reference values and team assumptions, not a simulation.
- This is an independent student prototype, not an official NASA product. Data credit: NASA.

## Team

Asma (App Builder), Amna (NASA Data), Yaqeen (ML Model), Shareefa (Field Research), Jumana (Storytelling and Presentation).
