# Red Oasis · Data Sources Sheet
Site: **Farm_23.3086_56.6131** (23.308639, 56.613083) · Period: 2024-01-01 → 2026-09-30 · Elevation ≈ 607.04 m
AppEEARS task ID: `cd1bbb38-96bd-4a8e-97bd-8788409325e4` · Generated: 2026-10-24

| Column | المعنى | Unit | Source / Product | Resolution |
|---|---|---|---|---|
| `date` | التاريخ | YYYY-MM-DD | — · — | daily |
| `rain_gpm_mm` | الأمطار اليومية | mm/day | NASA GPM IMERG (GES DISC) · GPM_3IMERGDL V07 (Late) | 0.1° (~10 km) |
| `rain_suspect` | علامة مطر مشكوك فيه | 0/1 | Quality check · GPM rain >20 mm with no SMAP rise and humidity <50% | daily |
| `temp_mean_c` | متوسط الحرارة | °C | NASA POWER · T2M | 0.5°×0.625° |
| `temp_max_c` | الحرارة العظمى | °C | NASA POWER · T2M_MAX | 0.5°×0.625° |
| `temp_min_c` | الحرارة الصغرى | °C | NASA POWER · T2M_MIN | 0.5°×0.625° |
| `humidity_pct` | رطوبة الجو | % | NASA POWER · RH2M | 0.5°×0.625° |
| `wind_2m_ms` | سرعة الرياح | m/s | NASA POWER · WS2M | 0.5°×0.625° |
| `solar_mj_m2` | الإشعاع الشمسي | MJ/m²/day | NASA POWER · ALLSKY_SFC_SW_DWN | 1° |
| `pressure_kpa` | الضغط الجوي | kPa | NASA POWER · PS | 0.5°×0.625° |
| `et0_mm` | التبخر-نتح المرجعي (احتياج الماء الأساسي) | mm/day | Calculated from NASA POWER · FAO-56 Penman-Monteith | daily |
| `soil_moisture_raw` | رطوبة التربة (القراءات الأصلية) | m³/m³ | NASA SMAP via AppEEARS · SPL3SMP_E.006: Soil_Moisture_Retrieval_Data_AM_soil_moisture, Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm | 9 km |
| `soil_moisture` | رطوبة التربة (معبأة) | m³/m³ | Derived · linear interpolation ≤3 days | 9 km |
| `ndvi_raw` | مؤشر الغطاء النباتي (الأصلي) | unitless (-1..1) | NASA MODIS / VIIRS via AppEEARS · MOD13Q1.061: _250m_16_days_NDVI; VNP13A1.002: 500_m_16_days_NDVI | 250 m (MODIS) / 500 m (VIIRS) |
| `ndvi` | مؤشر الغطاء النباتي (يومي) | unitless | Derived · linear interpolation ≤20 days | — |

## Important notes for the team
- SMAP pixel ≈ 9 km: it describes the **area** around the farm, not one field.
- NDVI is a 16-day composite; daily values between composites are interpolated (see flag column).
- `*_raw` columns are the untouched satellite values. Use them to check the model, use filled columns to train.
- Data accessed through NASA Earthdata (AppEEARS, GES DISC) and the NASA POWER API.
- `rain_suspect = 1`: GPM shows >20 mm but SMAP soil moisture did not rise and air humidity stayed <50%. Likely a satellite rain artifact; exclude these days when training.