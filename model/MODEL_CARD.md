# Red Oasis model card (one page)

Owner: Yaqeen (AI Brain) · 27 Oct 2026 · Status: **experimental**

## What it predicts

It predicts **how wet the surface soil will be in 3 days** at the Dariz palm
oasis (Ibri, Oman), as measured by NASA's SMAP satellite.

- The answer is a soil-moisture percentage.
- It does **not** measure palm water stress, crop health, irrigation need or
  yield. Nothing in the data measures those.
- A separate, simple irrigation rule (`IRRIGATION_RULE.md`) turns the forecast
  plus today's soil moisture, rain and ET0 into `irrigate` / `monitor` / `none`.
  That rule is a provisional guide, not validated advice.

## The data

NASA data for the Dariz farm point (23.3086° N, 56.6131° E), collected by
Amna, for every day from 1 Jan 2024 to 30 Sep 2026 (1,004 days).

| Input | Source |
|---|---|
| Soil moisture today | SMAP |
| Vegetation index (NDVI) | MODIS / VIIRS |
| Rain in the last 7 days | GPM |
| Air temperature | NASA POWER |
| Reference evapotranspiration (ET0) | NASA POWER |

- **608 days** could be used. The satellite doesn't pass every day, and days
  with suspect satellite rain were removed.
- The model learned from the earlier days and was tested **once** on the
  latest 122 days (18 Mar to 27 Sep 2026).
- It never saw those test days while learning.

## The results

The error is the average gap between the forecast and what SMAP measured, in
percentage points of soil moisture. Typical soil moisture here is about 8.8%.

| Method | Error on the 122 test days |
|---|---|
| **Our model** (Random Forest) | **1.20** |
| "Same as today" (persistence) | 1.31 |
| "Always the average" | 1.29 |

- On the test days, the model's error is about 8% smaller than "same as today".
- **But** on earlier periods (cross-validation) it was **not** better:
  1.18 vs 1.03. So we call it experimental, not reliable.
- After rain, errors are larger for every method: 1.57 for the model vs 1.72
  for "same as today". On days without recent rain they are 0.63 vs 0.67.
- The inputs that move the forecast most are today's soil moisture, then
  recent rain and temperature.

Figures: `figures/fig1_forecast_vs_actual.png`,
`figures/fig2_error_rain_vs_no_rain.png`, `figures/fig3_shap_summary.png`.

## The limits (please read)

1. **Regional, not field-level.** One SMAP value covers about 9 km. It shows
   the surrounding area's surface soil, not the root zone of irrigated palms,
   and falaj water inside the orchard is invisible to it.
2. **The forecast is nearly flat.** It moves much less than the real values
   (spread 1.1 vs 2.8 points) and misses sudden rain events.
3. **Short record.** Less than 3 years of data, with one big storm in the test
   period. A longer record (2015–2026) is being prepared.
4. **No ground truth for stress.** There are no field measurements or grower
   logs, so nothing about crop stress can be claimed.
5. **Thresholds are not agronomic.** The website's levels (dry < 8.4%,
   moderate < 9.3%) and the irrigation rule's references (median 8.8%) are
   this site's typical satellite values, not date-palm requirements. Because
   the forecast barely moves, the forecast alone never fell below 8.4% on the
   test days. Crop-based limits are still needed.
6. **SHAP is not cause.** It shows how the model uses each input, not what
   physically changes the soil.

Technical details: `model/README.md` · Numbers: `artifacts/evaluation.json`.
