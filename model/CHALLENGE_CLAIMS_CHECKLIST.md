# Model claims vs official challenge text: checklist

Owner: Yaqeen, with Jumana. Prepared 24 Oct 2026; completed 11 Nov 2026 against the official text.

Status: **completed**. Labels: Supported (backed by project evidence), Partially supported
(true only with the stated limits), Not supported (the project does not do this; do not claim it).

## Official challenge text

Field Shift: Adapting Farms with NASA Data (Red Oasis team page, NASA Space Apps 2026):

> Farmers around the world navigate shifting temperatures, changing rainfall patterns, water
> shortages, extreme weather, and declining soil health. These pressures make it difficult to
> choose crop rotations that protect fields, conserve water, and support long-term resilience.
> Your challenge is to create a decision-support tool that uses NASA Earth observations along with
> local soil information, crop characteristics, and farmer priorities to help farmers explore
> rotation strategies that could strengthen soil health and adapt their farms to changing conditions.

## 1. What the challenge asks for, and what Red Oasis does

| # | Challenge asks for | What Red Oasis actually does | Status | Evidence |
|---|---|---|---|---|
| R1 | A decision-support tool | Streamlit app with a 3-day soil-moisture forecast, a provisional irrigation level with its reasons, replay of real test days and a strategy explorer | Supported (experimental) | `app/app.py`, `model/irrigation.py`, `model/replay.py` |
| R2 | NASA Earth observations | SMAP soil moisture, MODIS/VIIRS NDVI, GPM IMERG rain, NASA POWER weather | Supported | `data/DATA_CARD.md`, `data/raw/red_oasis_sources*.md` |
| R3 | Local soil information | Ad Dariz soil shown as context: clay loam, field capacity about 22% by volume (Al-Ghafri, 2006). Not used by the model or the rule | Partially supported | Dossier section 3; app "Context only" |
| R4 | Crop characteristics | Date-palm annual demand 3,620 mm and the monthly shortage pattern from the Falaj al-Dariz study (2002-2003), shown as evidence. Not used by the model | Partially supported | Dossier section 2; app Site evidence |
| R5 | Farmer priorities | Slider: the farmer sets the yield loss they accept to save water. Numbers come from a published trial at Al Kamil and Al Wafi (2015-2016), not from Ad Dariz | Partially supported | App Strategy explorer; Dhehibi et al. (2022) |
| R6 | Explore rotation strategies | No crop rotation. Date palm is a perennial tree crop; the app explores irrigation timing and irrigation methods instead | Not supported (out of scope, stated openly) | App `chal_note`, root `README.md` |
| R7 | Strengthen soil health | No soil-health measure (salinity, organic matter, structure). Soil moisture is not soil health | Not supported | No soil-health data in `data/` |
| R8 | Conserve water | Shows a published result (subsurface drip at 60% ETc used 35.6% less water with similar yield) from another site. No water saving was measured at Ad Dariz | Partially supported | Dossier section 6; app Strategy explorer |
| R9 | Adapt to changing conditions | Daily level uses current soil moisture, forecast, 7-day rain and ET0; a 2015-2026 record was tested. No climate projections | Partially supported | `model/IRRIGATION_RULE.md`, `model/README.md` |

## 2. Claims made about the AI model

| # | Claim | Where it appears | Status | Evidence |
|---|---|---|---|---|
| 1 | Forecasts SMAP surface soil moisture 3 days ahead | `model/README.md`, `model/MODEL_CARD.md`, app forecast line | Supported | Final test MAE 1.20 vs 1.31 for "same as today" (`model/artifacts/evaluation.json`) |
| 2 | The model is experimental, not reliable | `MODEL_CARD.md`, `README.md`, app evaluation box and limits | Supported | Cross-validation MAE 1.18 vs 1.03: not better than "same as today" |
| 3 | Uses NASA data: SMAP, MODIS/VIIRS NDVI, GPM, NASA POWER | `MODEL_CARD.md`, app data table | Supported | `data/DATA_CARD.md` |
| 4 | Data are for the Ad Dariz point | `MODEL_CARD.md`, app location note | Partially supported | Point 23.308639 N, 56.613083 E in the Ad Dariz agricultural area; the exact orchard is not confirmed (dossier section 1); SMAP covers about 9 km |
| 5 | Explains each forecast with SHAP, not as physical cause | App "Why the model gave this forecast", `figures/fig3_shap_summary.png` | Supported | `model/explain.py`; captions in `figures/CAPTIONS.md` |
| 6 | Irrigation levels (irrigate / monitor / none) with reasons | App recommendation, `IRRIGATION_RULE.md` | Partially supported | Provisional rule relative to the site's satellite record; no crop thresholds (dossier section 3); not validated |
| 7 | "estimate crop water stress" (earlier app wording) | Was in `app/app.py` `approach` text | Resolved | The app now says "forecast soil moisture and suggest when irrigation may be needed"; limit `lim2` says it does not measure crop stress |
| 8 | "Supports irrigation decisions for farms in Oman" | App `earth_text` | Partially supported | One point at Ad Dariz; a regional ~9 km signal that does not see falaj water |
| 9 | Mars mode | App Mars tab | Supported as a labelled concept | Not an AI-model claim; limit `lim7` calls it a concept, not a simulation |
| 10 | The longer 2015-2026 record did not make the model reliable | `model/README.md`, app limit `lim4` | Supported | Final test 1.086 vs 1.043 for "same as today"; cross-validation 1.010 vs 1.155 (`model/artifacts_long/evaluation.json`) |

## 3. Draft project description (Jumana), claim by claim

| # | Sentence in the draft | Status | Note |
|---|---|---|---|
| D1 | "for date-palm farms at Ad Dariz in Ibri, Oman" | Partially supported | One satellite point in the Ad Dariz area; exact orchard not confirmed |
| D2 | "combines NASA SMAP, MODIS/VIIRS, GPM and POWER data" | Supported | `data/DATA_CARD.md` |
| D3 | "with published local soil and crop evidence" | Supported | Al-Ghafri (2006) for Ad Dariz; the drip trial (Dhehibi et al., 2022) is another site |
| D4 | "to forecast soil moisture" | Supported if called experimental | Add "experimental" next to the forecast |
| D5 | "explain each irrigation suggestion" | Supported | Rule checks and SHAP shown in the app; the suggestion itself is provisional |
| D6 | "let the farmer choose how much yield they'd trade for water savings" | Partially supported | The trade-off uses published trial numbers from Al Kamil and Al Wafi, not Ad Dariz forecasts |
| D7 | "Date palms are perennial trees that can't be rotated" | Supported (general fact) | No reference for it in the repository yet; cite the FAO date-palm chapter listed in the dossier references |
| D8 | "meets the challenge's goals of water conservation and adaptation through smarter irrigation decisions" | Partially supported | Water saving was not measured at Ad Dariz and the irrigation level is not validated; "aims to support" is safer than "meets" |

## 4. Answers to the earlier questions

- Expected outcome: a decision-support tool. Red Oasis gives a forecast, a provisional irrigation level with reasons, and a strategy explorer.
- Required data: NASA Earth observations plus local soil, crop and farmer information. No resolution or time span is stated.
- Validation: the text does not ask for accuracy. The project reports it anyway: final test 1.20 vs 1.31, cross-validation 1.18 vs 1.03.
- Out of scope or overstated: crop rotation (R6) and soil health (R7) must not be claimed. Water saving (R8) is a published result from another site.
- Wording for the abstract and slides: "an experimental 3-day soil-moisture forecast and a provisional irrigation guide for the Ad Dariz area, built on NASA SMAP, MODIS/VIIRS, GPM and POWER data".
