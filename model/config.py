"""
Shared settings for the Red Oasis AI component.

Everything that training, prediction and the website must agree on lives
here: feature names, units, valid ranges, the forecast target, and where
the saved model is stored.
"""

from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------
# Paths (relative to this file, so they work locally and on Streamlit Cloud)
# ----------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent
DEFAULT_MODEL_DIR = PACKAGE_DIR / "artifacts"
SYNTHETIC_MODEL_DIR = PACKAGE_DIR / "artifacts_synthetic_demo"
DEFAULT_DATA_FILE = REPO_DIR / "data" / "raw" / "red_oasis_daily.csv"
# Kept with the other model outputs: data/ belongs to the data team.
ROW_AUDIT_FILE = DEFAULT_MODEL_DIR / "model_rows_audit.csv"

# Lets tests or a local demo point at another model folder.
MODEL_DIR_ENV = "RED_OASIS_MODEL_DIR"
# A model trained on synthetic data is refused unless this is set to "1".
ALLOW_SYNTHETIC_ENV = "RED_OASIS_ALLOW_SYNTHETIC"


def model_dir() -> Path:
    override = os.environ.get(MODEL_DIR_ENV)
    return Path(override) if override else DEFAULT_MODEL_DIR


def synthetic_allowed() -> bool:
    return os.environ.get(ALLOW_SYNTHETIC_ENV) == "1"


MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
EVALUATION_FILE = "evaluation.json"

# ----------------------------------------------------------------------
# Model input features (canonical names used by predict())
# ----------------------------------------------------------------------
# min/max are hard validity limits: values outside them are rejected.

FEATURES = {
    "soil_moisture_pct": {
        "label": "Soil moisture",
        "unit": "% volumetric (SMAP m3/m3 x 100), observed today",
        "min": 0.0,
        "max": 100.0,
        "source": "NASA SMAP SPL3SMP_E (9 km)",
    },
    "ndvi": {
        "label": "Vegetation index (NDVI)",
        "unit": "unitless, -1 to 1 (latest composite already available)",
        "min": -1.0,
        "max": 1.0,
        "source": "MODIS MOD13Q1 / VIIRS VNP13A1",
    },
    "rainfall_7d_mm": {
        "label": "Rainfall, last 7 days",
        "unit": "mm (sum of the 7 daily totals up to today)",
        "min": 0.0,
        "max": 2000.0,
        "source": "NASA GPM IMERG Late V07",
    },
    "temperature_c": {
        "label": "Air temperature",
        "unit": "deg C (daily mean at 2 m)",
        "min": -30.0,
        "max": 65.0,
        "source": "NASA POWER T2M",
    },
    "et0_mm": {
        "label": "Reference evapotranspiration (ET0)",
        "unit": "mm/day (FAO-56 Penman-Monteith from NASA POWER)",
        "min": 0.0,
        "max": 25.0,
        "source": "Derived from NASA POWER (temperature, humidity, wind, radiation)",
        # Not re-computed from the agreed columns: radiation-only methods
        # (Makkink, Priestley-Taylor) were checked against this column on
        # the Nizwa data and underestimate it by ~1.7 mm/day (r = 0.86).
        # When ET0 is not supplied it is imputed with the training median
        # and reported in `imputed` / `warnings`.
    },
}
FEATURE_ORDER = list(FEATURES)

# Without current soil moisture a soil-moisture forecast is meaningless,
# so it may never be filled in automatically. Other inputs may be
# imputed with the training median, and the result says so.
REQUIRED_FEATURES = {"soil_moisture_pct"}

# ----------------------------------------------------------------------
# Forecast targets
# ----------------------------------------------------------------------
# The model forecasts the OBSERVED value of a variable `horizon_days`
# ahead. Gap-filled or interpolated values are never used as targets.

TARGETS = {
    "soil_moisture_pct": {
        "label": "Soil moisture",
        "unit": "% volumetric",
        "default_horizon_days": 3,
        # Threshold names in the website's THRESHOLDS dict.
        "dry_key": "soil_dry",
        "moderate_key": "soil_moderate",
    },
    "ndvi": {
        "label": "Vegetation index (NDVI)",
        "unit": "unitless",
        "default_horizon_days": 16,
        "dry_key": "ndvi_low",
        "moderate_key": None,
    },
}
DEFAULT_TARGET = "soil_moisture_pct"

# Fallback thresholds, copied from app/app.py. They are illustrative and
# NOT agronomically validated; the website passes its own values in.
DEFAULT_THRESHOLDS = {
    "soil_dry": 30.0,
    "soil_moderate": 45.0,
    "ndvi_low": 0.30,
}

# ----------------------------------------------------------------------
# Raw CSV handling
# ----------------------------------------------------------------------
# Canonical raw variable -> accepted column names, in priority order
# (case-insensitive). Raw satellite columns come first on purpose: the
# gap-filled "soil_moisture" and "ndvi" columns in Amna's file were
# interpolated using LATER observations, which would leak the future.

#
# Only the actual CSV column names (Amna's delivered file) and the
# pre-audit aliases are mapped. The team agreement exists in two
# conflicting spellings and is NOT mapped until Amna confirms one:
#   Week 2 roadmap PDF : date, temp_c, rain_mm, solar_radiation, soil_moisture, ndvi
#   audit brief        : date, c_temp, mm_rain, radiation_solar, moisture_soil, ndvi
# ("soil_moisture" was already a pre-audit alias.) Unrecognised columns
# are reported by data.validate_raw(), never renamed silently.
# data.column_mapping() reports exactly which column feeds each variable.

COLUMN_ALIASES = {
    "date": ["date", "time", "datetime", "day"],
    "site": ["site", "site_id", "location", "pixel_id"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "longitude"],
    "soil_moisture": ["soil_moisture_raw", "smap_sm", "smap", "sm", "soil_moisture"],
    "ndvi": ["ndvi_raw", "modis_ndvi", "viirs_ndvi", "ndvi"],
    "rainfall": ["rain_gpm_mm", "rainfall", "rainfall_mm", "precipitation", "precip"],
    "temperature": ["temp_mean_c", "temperature", "temperature_c", "t2m", "temp"],
    "et0": ["et0_mm", "et0", "reference_et"],
    # Not a model input; mapped only so its range can be checked.
    "solar_radiation": ["solar_mj_m2"],
    # (the remaining entries are flag/QA columns)
    "rain_suspect": ["rain_suspect"],
    "soil_moisture_is_filled": ["soil_moisture_is_filled"],
    "ndvi_is_interpolated": ["ndvi_is_interpolated"],
}

# Agreed-schema spellings that are deliberately NOT mapped (see above).
# A file using them gets a warning naming the columns that were not used.
UNCONFIRMED_AGREED_COLUMNS = ["temp_c", "rain_mm", "solar_radiation",
                              "c_temp", "mm_rain", "radiation_solar", "moisture_soil"]

# Units of the raw CSV columns, declared explicitly. data.validate_raw()
# rejects a file whose values do not fit the declared unit (e.g. soil
# moisture already in % while "m3/m3" is declared), so the x100
# conversion is never applied twice.
RAW_UNITS = {
    "soil_moisture": "m3/m3",  # or "pct"
    "temperature": "C",        # or "K"
}

# Plausible ranges in the declared units. Values outside are errors.
RAW_RANGES = {
    "soil_moisture": {"m3/m3": (0.0, 1.0), "pct": (0.0, 100.0)},
    "temperature": {"C": (-30.0, 60.0), "K": (243.0, 333.0)},
    "rainfall": (0.0, 500.0),       # mm/day
    "ndvi": (-1.0, 1.0),            # MODIS/VIIRS scaled x10000 would fail here
    "et0": (0.0, 25.0),             # mm/day
    "solar_radiation": (0.0, 45.0), # MJ/m2/day
}

# Study site in the team agreement (Shareefa's brief). Data from another
# location are accepted but always reported as such.
AGREED_SITE = {"name": "Dariz palm oasis, Ibri", "lat": 23.308639, "lon": 56.613083}
SITE_TOLERANCE_DEG = 0.05

# A 16-day NDVI composite is labelled with its FIRST day but uses images
# from the whole period, so it only becomes usable 16 days later.
NDVI_AVAILABILITY_LAG_DAYS = 16
NDVI_FFILL_DAYS = 16          # carry the latest available composite forward
RAIN_WINDOW_DAYS = 7          # a full 7-day window is required (no zero-filling)

# ----------------------------------------------------------------------
# Evaluation rules
# ----------------------------------------------------------------------

TEST_FRACTION = 0.2           # final, most recent 20% of dates
MIN_TRAIN_ROWS = 100
MIN_TEST_ROWS = 30
CV_SPLITS = 5

RANDOM_STATE = 42

# The model is called "reliable" only if, in the time-ordered CV on the
# training period, its mean MAE is lower than persistence AND it beats
# persistence in a majority of folds, AND it also beats persistence on
# the untouched final test period. Otherwise it stays "experimental".
# Decided before looking at the final test period.
RELIABILITY_RULE = (
    "mean CV MAE < persistence, wins in a majority of CV folds, "
    "and lower MAE than persistence on the final test period"
)

# Irrigation levels are NOT validated agronomic advice: no crop-stress
# labels or field measurements exist, and SMAP (~9 km) does not see
# water inside a falaj-irrigated orchard.
RECOMMENDATIONS_VALIDATED = False
