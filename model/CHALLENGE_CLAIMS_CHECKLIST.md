# Model claims vs official challenge text: checklist

Owner: Yaqeen · Week 3, Wednesday 28 Oct 2026 task.

> **STATUS: WAITING FOR THE OFFICIAL CHALLENGE TEXT** (expected 28 Oct 2026).
> It was not available on 24 Oct 2026, when this list was prepared. Nothing
> below assumes what the challenge asks. On 28 Oct, read the text and fill
> the last column. The "evidence" column records what the project can
> actually support today.

## Claims currently made about the AI model

| # | Claim | Where it appears | Evidence today | Change needed after reading the challenge text |
|---|---|---|---|---|
| 1 | Forecasts SMAP surface soil moisture 3 days ahead | `model/README.md`, `model/MODEL_CARD.md`, app forecast line | Yes. Dariz test MAE 1.20 vs persistence 1.31 (`evaluation.json`) | *pending* |
| 2 | The model is experimental, not reliable | `MODEL_CARD.md`, `README.md`, app evaluation box | Yes. CV MAE 1.18 vs persistence 1.03 | *pending* |
| 3 | Uses NASA data: SMAP, MODIS/VIIRS NDVI, GPM, NASA POWER | `MODEL_CARD.md`, app data table | Yes. `data/raw/red_oasis_sources.md` | *pending* |
| 4 | Data are for the Dariz farm point | `MODEL_CARD.md`, app location note | Yes. 23.308639 N, 56.613083 E; 2024–2026 | *pending* |
| 5 | Explains each forecast with SHAP, not as physical cause | app "Why the model gave this forecast", `figures/fig3_shap_summary.png` | Yes | *pending* |
| 6 | Irrigation levels (irrigate / monitor / none) | app recommendation, `IRRIGATION_RULE.md` | Partly. Provisional rule relative to the site record; **not validated** | *pending* |
| 7 | "estimate crop water stress" | `app/app.py`, `approach` text (English and Arabic) | **No.** The model predicts SMAP soil moisture; no stress labels or field data exist | *pending* (likely reword; Asma's file) |
| 8 | Supports irrigation decisions for Omani farms | app `earth_text` | Partly. Regional ~9 km signal; falaj water not visible | *pending* |
| 9 | Mars mode | app Mars tab | Not an AI-model claim; the model is not used there | *pending* |

## Questions to answer from the official text (fill in on 28 Oct)

- [ ] Which outcome does the challenge expect (forecast, recommendation, risk map, other)?
- [ ] Does it require specific NASA datasets, spatial resolution or time span?
- [ ] Does it ask for validation or accuracy reporting? Which kind?
- [ ] Are any claims above out of scope or overstated for the challenge?
- [ ] What wording should the abstract and slides use for the model (send to Jumana)?
