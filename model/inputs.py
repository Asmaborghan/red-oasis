"""
Conversion layer: daily data (Amna's agreed CSV format) -> predict() inputs.

The website should not assemble model inputs by hand. This module uses
the exact feature code used for training (data.build_features), so units,
the 7-day rainfall sum, the NDVI availability lag and the suspect-rain
rule are identical at training and prediction time.

    from model.inputs import inputs_from_daily
    from model.predict import predict

    prepared = inputs_from_daily(daily_df)          # latest date by default
    result = predict(prepared["inputs"])            # None values are allowed

Values that genuinely do not exist are returned as None, never invented:
- soil_moisture_pct: None if SMAP did not pass on that date (predict()
  then returns "invalid_input", because it is required)
- rainfall_7d_mm: None if any of the 7 daily totals is missing or suspect
- ndvi: None if no composite is available yet
- et0_mm: None if the file has no ET0 column (the agreed schema has none);
  predict() then imputes the training median and says so
"""

from __future__ import annotations

import math

import pandas as pd

from . import config, data


def inputs_from_daily(raw: pd.DataFrame, as_of=None) -> dict:
    """Return {"as_of", "inputs", "notes", "column_mapping", "validation_warnings"}."""
    feats = data.build_features(raw)  # raises data.DataError on invalid files
    as_of = pd.Timestamp(as_of) if as_of is not None else feats["date"].max()
    row = feats[feats["date"] == as_of]
    result = {
        "as_of": str(as_of.date()),
        "inputs": None,
        "notes": [],
        "column_mapping": data.column_mapping(raw.columns),
        "validation_warnings": feats.attrs.get("validation_warnings", []),
    }
    if row.empty:
        result["notes"].append(f"No data for {as_of.date()}.")
        return result
    if row["site"].nunique() > 1:
        result["notes"].append("Several sites in the file; pass one site at a time.")
        return result
    row = row.iloc[0]

    def value(name):
        v = row[name]
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

    inputs = {name: value(name) for name in config.FEATURE_ORDER}
    if inputs["soil_moisture_pct"] is None:
        result["notes"].append("No observed SMAP soil moisture on this date (required input).")
    if inputs["rainfall_7d_mm"] is None:
        reason = ("suspect rainfall in the last 7 days" if row["suspect_in_rain_window"] == 1
                  else "a missing daily total in the last 7 days")
        result["notes"].append(f"7-day rainfall unavailable: {reason}.")
    if inputs["ndvi"] is None:
        result["notes"].append("No NDVI composite available yet for this date.")
    if inputs["et0_mm"] is None:
        result["notes"].append("ET0 not in the data; it will be imputed by predict().")
    result["inputs"] = inputs
    return result
