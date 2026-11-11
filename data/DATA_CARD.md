# Red Oasis data card

Owner: Amna (NASA data). Prepared 11 Nov 2026 from the files in `data/raw/`. Every number below
is counted from the CSV files and the model's row audits.

Site: Ad Dariz agricultural area, Ibri, Oman. One point, 23.308639 N, 56.613083 E (elevation
about 607 m). The satellite values describe the area around this point, not one orchard.

## Files

| File | Period | Days | AppEEARS task ID | Used by |
|---|---|---|---|---|
| `red_oasis_daily.csv` | 1 Jan 2024 to 30 Sep 2026 | 1,004 | `cd1bbb38-96bd-4a8e-97bd-8788409325e4` (24 Oct 2026) | Production model, `model/artifacts/` |
| `red_oasis_daily_2015_2026.csv` | 1 Jan 2015 to 23 Oct 2026 | 4,314 | `bcc42ced-cf94-4125-85d2-d9e0984742f2` (30 Oct 2026) | Comparison model only, `model/artifacts_long/` |

Both files have one row per day, 21 columns, no missing dates and no duplicate dates. Column
meanings and units are in `red_oasis_sources.md` and `red_oasis_sources_2015_2026.md`.

## NASA datasets

| Dataset | Product | Columns | Resolution | Access |
|---|---|---|---|---|
| SMAP soil moisture | SPL3SMP_E.006 (mean of AM and PM passes) | `soil_moisture_raw` | 9 km | AppEEARS |
| MODIS / VIIRS NDVI | MOD13Q1.061, VNP13A1.002 (16-day composites) | `ndvi_raw` | 250 m / 500 m | AppEEARS |
| GPM IMERG rain | GPM_3IMERGDL V07 (Late) | `rain_gpm_mm` | 0.1° (about 10 km) | GES DISC |
| NASA POWER weather | T2M, T2M_MAX, T2M_MIN, RH2M, WS2M, ALLSKY_SFC_SW_DWN, PS | temperature, humidity, wind, solar, pressure | 0.5° x 0.625° (solar 1°) | POWER API |
| ET0 (derived) | FAO-56 Penman-Monteith from NASA POWER | `et0_mm` | daily | calculated |

## Coverage and quality

| | 2024-2026 file | 2015-2026 file |
|---|---|---|
| Days with an observed SMAP value | 749 (74.6%) | 3,037 (70.4%; SMAP starts in 2015, first pass here 2 Apr 2015) |
| NDVI composites | 127 | 541 |
| Other missing values | none | rain: 1 day (20 May 2015); POWER temperature, humidity, wind and pressure: 22-23 Oct 2026; solar and ET0: 20-23 Oct 2026 |
| Suspect-rain days (`rain_suspect = 1`) | 2: 25 and 26 May 2025 | 5: 9 Nov 2016, 25 Apr 2021, 3 May 2021, 25 and 26 May 2025 |
| Rows usable by the model | 608 (484 train, 122 final test) | 2,441 (1,951 train, 489 final test) |

NDVI composites are a median 8 days apart. SMAP passes every 2-3 days.
Yearly GPM rain in the long file runs from 13.9 mm (2018) to 410.2 mm (2021, of which 57.6 mm
fell on the two suspect days). NASA POWER rain (`PRECTOTCORR`) was checked and not used: it was
about 24 times too high at this point.

A day is usable when SMAP observed soil moisture on that day and 3 days later, the full 7-day rain
record exists and none of those days is a suspect-rain day. The model uses the raw columns
(`soil_moisture_raw`, `ndvi_raw`), not the gap-filled `soil_moisture` and `ndvi`, because the
filled values are interpolated from later days and would leak the future into training.

## 2024-2026 vs 2015-2026

EN:
Both files have the same 21 columns, the same point and identical values on every shared day (1 Jan 2024 to 30 Sep 2026); the 2015-2026 file adds 3,287 earlier days and 23 later days from a separate AppEEARS task.
The app and the production model use the 2024-2026 file; the 2015-2026 file feeds only the comparison model in `model/artifacts_long/`.

AR:
الملفان لهما نفس الأعمدة الـ 21 ونفس النقطة، وقيمهما متطابقة في كل يوم مشترك (1 يناير 2024 إلى 30 سبتمبر 2026)، ويضيف ملف 2015-2026 عدد 3,287 يوماً أقدم و23 يوماً لاحقاً من مهمة AppEEARS مستقلة.
التطبيق ونموذج الإنتاج يستخدمان ملف 2024-2026، أما ملف 2015-2026 فيُستخدم فقط لنموذج المقارنة في `model/artifacts_long/`.

## Limits

SMAP measures surface soil moisture over about 9 km, so it does not see falaj water inside one
orchard. NASA POWER and GPM are gridded estimates, not station readings. NDVI between
composites is interpolated in the filled column only.

Data credit: NASA SMAP (NSIDC DAAC), NASA MODIS and VIIRS (LP DAAC), NASA GPM IMERG (GES DISC) and
the NASA POWER project, accessed through NASA Earthdata AppEEARS, GES DISC and the POWER API.
