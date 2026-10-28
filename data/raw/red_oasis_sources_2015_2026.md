# Red Oasis · Data Sources · `red_oasis_daily_2015_2026.csv`
Site: Al Duraiz farm (23.308639, 56.613083) · Period: 2015-01-01 → 2026-10-23 · 4314 days

| Column | المعنى | Unit | Source |
|---|---|---|---|
| `date` | التاريخ | YYYY-MM-DD | — |
| `temp_c` | متوسط حرارة الهواء على ارتفاع 2 م | °C | NASA POWER · T2M |
| `rain_mm` | الأمطار اليومية | mm/day | NASA GPM IMERG Late Daily V07 (GPM_3IMERGDL), 0.1° grid cell nearest the farm |
| `solar_radiation` | الإشعاع الشمسي الواصل للسطح | MJ/m²/day | NASA POWER · ALLSKY_SFC_SW_DWN |
| `soil_moisture` | رطوبة التربة السطحية (فراغات ≤3 أيام معبأة خطياً) | m³/m³ | NASA SMAP · SPL3SMP_E.006 (9 km) via AppEEARS, mean of AM+PM passes |
| `ndvi` | مؤشر الغطاء النباتي (يومي بالاستيفاء بين صور كل 16 يوم) | unitless | NASA MODIS MOD13Q1.061 (250 m) + VIIRS VNP13A1.002 (500 m) via AppEEARS |

Missing share: `temp_c` 0.0%, `rain_mm` 0.1%, `solar_radiation` 0.1%, `soil_moisture` 4.0%, `ndvi` 0.5%

## Notes
- Same columns as `data/red_oasis_daily.csv`. The current file was not changed or deleted.
- **Rain comes from GPM, not NASA POWER.** POWER `PRECTOTCORR` for this point was checked and found unusable (values ≈ 24× too high, e.g. 8,279 mm on 2024-08-18; yearly totals of 1,000–51,000 mm). GPM yearly totals here are 14–410 mm, which matches the region.
- GPM check: the March 2026 storm (23–27 March) appears in GPM and SMAP soil moisture rose from 0.10 to 0.30 m³/m³ at the same time.
- Suspicious rain removed (left empty): 2016-11-09, 2021-04-25, 2021-05-03, 2025-05-25, 2025-05-26. Rule: GPM > 20 mm but SMAP did not rise (< 0.03) and air humidity < 50%.
- SMAP starts 2015-03-31, so `soil_moisture` is empty before that date.
- SMAP pixel ≈ 9 km describes the area around the farm, not one field.