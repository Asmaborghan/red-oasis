# How to reproduce the Red Oasis daily dataset

Owner: Amna (NASA data). Written 11 Nov 2026 from the source sheets in `data/raw/` and checked
against `red_oasis_daily_2015_2026.csv`. The original download script is not in the repository,
so this file records the steps and states which ones were re-run and which were not.

Point: 23.308639 N, 56.613083 E (Ad Dariz area, Ibri). One row per day, 21 columns. Column
meanings and units: `red_oasis_sources_2015_2026.md`.

## 1. Download

SMAP soil moisture and MODIS/VIIRS NDVI come from NASA AppEEARS (https://appeears.earthdatacloud.nasa.gov),
point sample request at the coordinates above. A free NASA Earthdata login is needed.

| Product | Layer | Becomes |
|---|---|---|
| SPL3SMP_E.006 | `Soil_Moisture_Retrieval_Data_AM_soil_moisture`, `Soil_Moisture_Retrieval_Data_PM_soil_moisture_pm` | `soil_moisture_raw` = mean of the AM and PM values |
| MOD13Q1.061 | `_250m_16_days_NDVI` | `ndvi_raw` (16-day composites) |
| VNP13A1.002 | `500_m_16_days_NDVI` | `ndvi_raw` where MODIS has no value |

Task IDs used: `cd1bbb38-96bd-4a8e-97bd-8788409325e4` (2024-2026 file) and
`bcc42ced-cf94-4125-85d2-d9e0984742f2` (2015-2026 file).

Rain comes from GPM IMERG daily, product GPM_3IMERGDL V07 (Late), 0.1°, from NASA GES DISC
(https://gpm.nasa.gov/data/imerg), extracted for the point (method to confirm, see section 4).
Earthdata login needed. NASA POWER rain (`PRECTOTCORR`) is not used (about 24 times too high here).

Weather comes from the NASA POWER daily point API, community AG (no login):

```
https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PS&community=AG&longitude=56.613083&latitude=23.308639&start=20150101&end=20261023&format=JSON
```

T2M, T2M_MAX, T2M_MIN, RH2M, WS2M, ALLSKY_SFC_SW_DWN (MJ/m²/day) and PS (kPa) become
`temp_mean_c`, `temp_max_c`, `temp_min_c`, `humidity_pct`, `wind_2m_ms`, `solar_mj_m2` and
`pressure_kpa`. POWER reports the elevation of this point as 607.04 m.

## 2. Merge and derive

1. Join all sources on the date, one row per calendar day. Keep missing values empty, never 0.
2. `rain_source` is `GPM_IMERG`, or `missing` when GPM has no value for that day.
3. `soil_moisture` is `soil_moisture_raw` with gaps of at most 3 days filled by linear
   interpolation; `soil_moisture_is_filled` = 1 on those days.
4. `ndvi` is `ndvi_raw` interpolated linearly across gaps of at most 20 days;
   `ndvi_is_interpolated` = 1 on those days.
5. `et0_mm`: FAO-56 Penman-Monteith reference evapotranspiration from the POWER columns.
6. `rain_suspect` = 1 when GPM rain is above 20 mm, SMAP soil moisture does not rise and humidity
   stays below 50%.

The model uses only the raw columns (`soil_moisture_raw`, `ndvi_raw`). The filled columns are
built from later days, so the model must not train on them.

## 3. What was checked on 11 Nov 2026

| Step | Re-run? | Result |
|---|---|---|
| NASA POWER download (1-7 Jun 2025, 7 variables) | Yes | Identical to the CSV on every value; elevation 607.04 m |
| `soil_moisture_is_filled`, `ndvi_is_interpolated` | Yes, from the raw columns | 100% identical |
| `soil_moisture` (fill gaps up to 3 days) | Yes | 99.7% of values identical; the rest are not explained by the recorded rule |
| `ndvi` (fill gaps up to 20 days) | Yes | 93.8% of values identical; the exact interpolation setting is not recorded |
| `et0_mm` (FAO-56) | Yes, standard daily form | Within 0.01 mm on 33% of days, at most 0.2 mm/day apart; the exact settings are not recorded |
| `rain_suspect` | Partly | The 5 flagged days are consistent with the rule, but the size of "no SMAP rise" is not written down, so the flags cannot be rebuilt exactly |
| AppEEARS and GPM downloads | No | Need a NASA Earthdata login |

## 4. Still needed from Amna

1. The download and merge script, or the exact settings for steps 2.4 to 2.6: NDVI interpolation
   method, ET0 settings (soil heat flux, clear-sky radiation) and the SMAP rise threshold with its
   time window for `rain_suspect`.
2. Confirmation of how GPM was extracted for the point (nearest grid cell or another method).

Data credit: NASA SMAP (NSIDC DAAC), NASA MODIS and VIIRS (LP DAAC), NASA GPM IMERG (GES DISC) and
the NASA POWER Project, accessed through NASA Earthdata AppEEARS, GES DISC and the POWER API.
