"""
SYNTHETIC data for software testing only.

This generates a toy daily series with a simple made-up water balance so
the training, prediction, SHAP and website code can be exercised before
the real NASA dataset exists. Its numbers are NOT observations, and any
score obtained on it says nothing about real-world performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SYNTHETIC_NOTICE = (
    "SYNTHETIC interface-test data. Not NASA observations. "
    "Scores and explanations from it have no scientific meaning."
)


def make_synthetic_raw(n_days: int = 730, seed: int = 0, start: str = "2023-01-01") -> pd.DataFrame:
    """Return a raw-format CSV frame using the same columns Amna's file is expected to have."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy()
    season = np.sin(2 * np.pi * (doy - 110) / 365.25)

    temperature = 30 + 9 * season + rng.normal(0, 1.5, n_days)
    rain = np.where(rng.random(n_days) < 0.04, rng.gamma(1.5, 8.0, n_days), 0.0)

    sm = np.empty(n_days)
    sm[0] = 0.12
    for i in range(1, n_days):
        drying = 0.0015 * max(temperature[i] - 15, 0) / 15
        sm[i] = np.clip(sm[i - 1] - drying * sm[i - 1] * 4 + rain[i] * 0.004 + rng.normal(0, 0.003), 0.02, 0.45)

    ndvi = np.clip(0.35 + 0.4 * pd.Series(sm).rolling(20, min_periods=1).mean().to_numpy()
                   - 0.004 * (temperature - 30) + rng.normal(0, 0.01, n_days), 0.05, 0.9)

    df = pd.DataFrame({
        "date": dates,
        "soil_moisture": sm,          # m3/m3, like SMAP
        "ndvi": ndvi,
        "rainfall_mm": rain,
        "temperature_c": temperature,
        "et0_mm": np.clip(0.25 * temperature - 2 + rng.normal(0, 0.5, n_days), 0.5, None),
    })
    # SMAP revisits every 2-3 days and MODIS NDVI is a 16-day composite.
    df.loc[rng.random(n_days) < 0.4, "soil_moisture"] = np.nan
    df.loc[(np.arange(n_days) % 16) != 0, "ndvi"] = np.nan
    return df
