"""
Load the merged environmental CSV and turn it into model features and a
forecast target, without leaking future information.

Rules applied here (see model/README.md for the reasons):
- Only OBSERVED soil moisture is used, as a feature and as the target.
  Gap-filled values (interpolated from later passes) are discarded.
- NDVI composites are used only from the day they could actually have
  been known (composite start date + 16 days), then carried forward.
- Rainfall windows must be complete; missing days are never treated as 0.
- Days flagged rain_suspect = 1 are removed from the rainfall record, and
  every training row whose feature window or target date touches one is
  excluded, with the reason recorded in an audit table.

The same feature definitions are used at prediction time (see
config.FEATURES), so the website must pass values in the same units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

RAW_VARIABLES = ("soil_moisture", "ndvi", "rainfall", "temperature", "et0")


class DataError(ValueError):
    """The CSV cannot be used for training as it is."""


def _match_columns(columns) -> dict[str, str]:
    """Map canonical raw names to the actual CSV column names."""
    lower = {c.strip().lower(): c for c in columns}
    found = {}
    for canonical, aliases in config.COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                found[canonical] = lower[alias]
                break
    return found


def column_mapping(columns) -> dict[str, str | None]:
    """Which CSV column feeds each canonical variable (None = not present).

    This is the explicit, documented mapping between the file and the
    model: e.g. {"temperature": "c_temp", "rainfall": "mm_rain", ...}.
    """
    found = _match_columns(columns)
    return {name: found.get(name) for name in config.COLUMN_ALIASES}


def unconfirmed_agreed_columns(columns) -> list[str]:
    """Columns using an agreed-schema spelling that is not mapped yet."""
    lower = {str(c).strip().lower(): c for c in columns}
    return [lower[c] for c in config.UNCONFIRMED_AGREED_COLUMNS if c in lower]


def validate_raw(df: pd.DataFrame) -> dict:
    """Check a standardised frame. Returns {"errors": [...], "warnings": [...]}.

    Errors make the file unusable for training: duplicate dates, values
    that do not fit the declared unit, or values outside plausible ranges.
    """
    errors, warnings = [], []
    sites = df["site"] if "site" in df else pd.Series("site_1", index=df.index)
    dup = int(pd.DataFrame({"s": sites, "d": df["date"]}).duplicated().sum())
    if dup:
        errors.append(f"{dup} duplicate date(s) for the same site.")

    def check(col, lo, hi, unit_text):
        if col not in df:
            return
        v = pd.to_numeric(df[col], errors="coerce")
        bad_text = int(v.isna().sum() - df[col].isna().sum())
        if bad_text:
            errors.append(f"{col}: {bad_text} value(s) are not numbers.")
        out = int(((v < lo) | (v > hi)).sum())
        if out:
            errors.append(f"{col}: {out} value(s) outside {lo}..{hi} {unit_text} "
                          f"(observed {v.min():.4g}..{v.max():.4g}). Check the unit.")

    sm_unit = config.RAW_UNITS["soil_moisture"]
    check("soil_moisture", *config.RAW_RANGES["soil_moisture"][sm_unit], sm_unit)
    if "soil_moisture" in df and sm_unit == "pct":
        if pd.to_numeric(df["soil_moisture"], errors="coerce").max() <= 1:
            errors.append("soil_moisture is declared as % but all values are <= 1 "
                          "(looks like m3/m3). Set RAW_UNITS['soil_moisture'] = 'm3/m3'.")
    t_unit = config.RAW_UNITS["temperature"]
    check("temperature", *config.RAW_RANGES["temperature"][t_unit], f"deg {t_unit}")
    for col, unit in (("rainfall", "mm/day"), ("ndvi", ""), ("et0", "mm/day"),
                      ("solar_radiation", "MJ/m2/day")):
        check(col, *config.RAW_RANGES[col], unit)

    # Gap-filled series leak the future. Raw SMAP passes are every 2-3 days
    # and NDVI composites every 8-16 days, so near-complete series are suspicious.
    if "soil_moisture" in df and "soil_moisture_is_filled" not in df:
        if df["soil_moisture"].notna().mean() > 0.95:
            warnings.append("soil_moisture has a value on almost every day; if it is gap-filled "
                            "it uses later observations. Supply raw passes or a fill flag.")
    if "ndvi" in df and "ndvi_is_interpolated" not in df:
        if df["ndvi"].notna().mean() > 0.5:
            warnings.append("ndvi has a value on most days; if it is interpolated it uses later "
                            "composites. Supply raw composites or an interpolation flag.")
    if "rain_suspect" not in df:
        warnings.append("No rain_suspect column: suspect satellite rainfall cannot be excluded.")
    if "et0" not in df:
        warnings.append("No ET0 column: et0_mm will be imputed with the training median.")
    if "lat" in df and "lon" in df:
        lat, lon = df["lat"].dropna().unique(), df["lon"].dropna().unique()
        if len(lat) and len(lon) and (
            abs(lat[0] - config.AGREED_SITE["lat"]) > config.SITE_TOLERANCE_DEG
            or abs(lon[0] - config.AGREED_SITE["lon"]) > config.SITE_TOLERANCE_DEG
        ):
            warnings.append(f"Data location ({lat[0]}, {lon[0]}) is not the agreed site "
                            f"{config.AGREED_SITE['name']} ({config.AGREED_SITE['lat']}, "
                            f"{config.AGREED_SITE['lon']}).")
    else:
        warnings.append("No lat/lon columns: the data location cannot be checked.")
    return {"errors": errors, "warnings": warnings}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep recognised columns, rename them to canonical names, parse dates."""
    mapping = _match_columns(df.columns)
    if "date" not in mapping:
        raise DataError(
            f"No date column found. Columns are: {list(df.columns)}. "
            "Add the name to config.COLUMN_ALIASES['date']."
        )
    df = df[list(mapping.values())].rename(columns={v: k for k, v in mapping.items()})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise DataError(f"{int(df['date'].isna().sum())} rows have an unreadable date.")
    return df


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV (UTF-8 with or without BOM) and standardise its columns."""
    return standardize_columns(pd.read_csv(path, encoding="utf-8-sig"))


def site_info(df: pd.DataFrame) -> dict:
    """Name and coordinates of the data location(s), as stated in the CSV."""
    df = standardize_columns(df)
    info = {}
    if "site" in df:
        info["sites"] = sorted(map(str, df["site"].dropna().unique()))
    for col in ("lat", "lon"):
        if col in df:
            info[col] = sorted(map(float, df[col].dropna().unique()))
    return info


def inspect(path: str | Path) -> str:
    """Return a plain-text report on what the CSV actually contains."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    mapping = _match_columns(raw.columns)
    lines = [f"File: {path}", f"Rows: {len(raw)}", "", "Columns:"]
    for col in raw.columns:
        s = raw[col]
        desc = f"  {col!r}: dtype={s.dtype}, missing={s.isna().mean():.1%}"
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            desc += f", min={s.min():.4g}, max={s.max():.4g}, mean={s.mean():.4g}"
        lines.append(desc)
    lines += ["", "Used by the model as:"]
    for canonical in config.COLUMN_ALIASES:
        lines.append(f"  {canonical:<24} -> {mapping.get(canonical, 'NOT FOUND')}")
    if "date" in mapping:
        d = pd.to_datetime(raw[mapping["date"]], errors="coerce")
        lines += [
            "",
            f"Dates: {d.min().date()} to {d.max().date()}, unreadable={int(d.isna().sum())}, "
            f"duplicates={int(d.duplicated().sum())}, sorted={bool(d.is_monotonic_increasing)}",
        ]
    if "rain_suspect" in mapping:
        lines.append(f"rain_suspect = 1 on {int((raw[mapping['rain_suspect']] == 1).sum())} days")
    if "soil_moisture" in mapping:
        mx = pd.to_numeric(raw[mapping["soil_moisture"]], errors="coerce").max()
        unit = config.RAW_UNITS["soil_moisture"]
        if unit == "m3/m3" and mx > 1:
            lines.append("WARNING: soil moisture > 1 but RAW_UNITS says m3/m3. Is it already %?")
        if unit == "pct" and mx <= 1:
            lines.append("WARNING: soil moisture <= 1 but RAW_UNITS says pct. Is it m3/m3?")
    if "date" in mapping:
        report = validate_raw(standardize_columns(raw))
        lines += ["", f"Valid for training: {not report['errors']}"]
        lines += [f"ERROR: {e}" for e in report["errors"]]
        lines += [f"WARNING: {w}" for w in report["warnings"]]
    unconfirmed = unconfirmed_agreed_columns(raw.columns)
    if unconfirmed:
        lines.append(f"WARNING: {unconfirmed} use an agreed-schema spelling that is not mapped "
                     "until Amna confirms the final column names.")
    return "\n".join(lines)


def _daily_site_frame(g: pd.DataFrame) -> pd.DataFrame:
    """Resample one site's records to one row per calendar day."""
    g = g.set_index("date").sort_index()
    agg = {}
    for col in ("soil_moisture", "ndvi", "temperature", "et0"):
        agg[col] = g[col].resample("D").mean()
    # min_count=1 keeps a missing day as NaN instead of 0 mm.
    agg["rainfall"] = g["rainfall"].resample("D").sum(min_count=1)
    agg["rain_suspect"] = g["rain_suspect"].resample("D").max()
    return pd.DataFrame(agg)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per site-day with the canonical model features.

    Every feature on day t uses only information available on day t.
    Missing values stay NaN; nothing is filled with made-up values.
    """
    unconfirmed = unconfirmed_agreed_columns(df.columns)
    df = standardize_columns(df)
    if "soil_moisture" not in df:
        raise DataError("Soil moisture column is required and was not found.")
    missing = [c for c in RAW_VARIABLES if c not in df]

    report = validate_raw(df)
    if unconfirmed:
        report["warnings"].append(
            f"Column(s) {unconfirmed} use an agreed-schema spelling that is not mapped until "
            "Amna confirms the final column names; they were NOT used.")
    if report["errors"]:
        raise DataError("; ".join(report["errors"]))

    df = df.copy()
    for col in RAW_VARIABLES:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan

    # Discard values that were filled in from neighbouring (later) days.
    if "soil_moisture_is_filled" in df:
        df.loc[df["soil_moisture_is_filled"] == 1, "soil_moisture"] = np.nan
    if "ndvi_is_interpolated" in df:
        df.loc[df["ndvi_is_interpolated"] == 1, "ndvi"] = np.nan

    df["rain_suspect"] = (
        pd.to_numeric(df["rain_suspect"], errors="coerce").fillna(0)
        if "rain_suspect" in df else 0
    )
    # Suspect rainfall is not a measurement: remove it from the record.
    df.loc[df["rain_suspect"] == 1, "rainfall"] = np.nan

    if config.RAW_UNITS["soil_moisture"] == "m3/m3":
        df["soil_moisture"] = df["soil_moisture"] * 100.0
    if config.RAW_UNITS["temperature"] == "K":
        df["temperature"] = df["temperature"] - 273.15

    if "site" not in df:
        df["site"] = "site_1"

    w = config.RAIN_WINDOW_DAYS
    frames = []
    for site, g in df.groupby("site", sort=True):
        d = _daily_site_frame(g)
        out = pd.DataFrame(index=d.index)
        out["soil_moisture_pct"] = d["soil_moisture"]
        out["ndvi"] = (
            d["ndvi"]
            .shift(config.NDVI_AVAILABILITY_LAG_DAYS)
            .ffill(limit=config.NDVI_FFILL_DAYS)
        )
        # A full window is required, so a missing or suspect day gives NaN.
        out["rainfall_7d_mm"] = d["rainfall"].rolling(w, min_periods=w).sum()
        out["temperature_c"] = d["temperature"]
        out["et0_mm"] = d["et0"]
        out["rain_suspect"] = d["rain_suspect"].fillna(0).astype(int)
        out["suspect_in_rain_window"] = (
            d["rain_suspect"].fillna(0).rolling(w, min_periods=1).max().astype(int)
        )
        out["site"] = site
        frames.append(out.reset_index())
    feats = pd.concat(frames, ignore_index=True)
    feats.attrs["missing_raw_columns"] = missing
    feats.attrs["validation_warnings"] = report["warnings"]
    return feats


def add_target(feats: pd.DataFrame, target: str, horizon_days: int) -> pd.DataFrame:
    """Add `target_future` (the observed target `horizon_days` later) and
    `target_rain_suspect` (whether that later day is flagged). Nothing is
    dropped here; select_rows() decides which rows are usable."""
    if target not in config.TARGETS:
        raise DataError(f"Unknown target {target!r}. Choose from {list(config.TARGETS)}.")
    if horizon_days < 1:
        raise DataError("horizon_days must be at least 1.")
    parts = []
    for _, g in feats.groupby("site", sort=True):
        g = g.sort_values("date").copy()
        # Rows are daily and contiguous per site, so shift = days ahead.
        g["target_future"] = g[target].shift(-horizon_days)
        g["target_date"] = g["date"] + pd.Timedelta(days=horizon_days)
        g["target_rain_suspect"] = g["rain_suspect"].shift(-horizon_days).fillna(0).astype(int)
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(["date", "site"]).reset_index(drop=True)


EXCLUSION_REASONS = [
    ("rain_suspect", "rain_suspect = 1 on the feature date"),
    ("suspect_in_rain_window", "suspect rain inside the 7-day rainfall window"),
    ("target_rain_suspect", "rain_suspect = 1 on the target date"),
    ("incomplete_rain", "incomplete 7-day rainfall record (a day is missing)"),
    ("no_soil_obs", "no observed soil moisture on the feature date"),
    ("no_target_obs", "no observed target value on the target date"),
]


def select_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into usable training rows and an audit of every row.

    Returns (clean rows, audit table). The audit lists each date, whether
    it is used, and the first reason it was excluded.
    """
    checks = {
        "rain_suspect": df["rain_suspect"] == 1,
        "suspect_in_rain_window": (df["suspect_in_rain_window"] == 1) & (df["rain_suspect"] != 1),
        "target_rain_suspect": df["target_rain_suspect"] == 1,
        # Missing rain must not be imputed as the (zero) training median.
        "incomplete_rain": df["rainfall_7d_mm"].isna(),
        "no_soil_obs": df["soil_moisture_pct"].isna(),
        "no_target_obs": df["target_future"].isna(),
    }
    reason = pd.Series("", index=df.index, dtype=object)
    for key, text in reversed(EXCLUSION_REASONS):  # first listed reason wins
        reason[checks[key]] = text
    audit = df[["date", "site", "target_date"]].copy()
    audit["used_for_model"] = reason == ""
    audit["exclusion_reason"] = reason
    clean = df[reason == ""].reset_index(drop=True)
    return clean, audit


def exclusion_summary(audit: pd.DataFrame) -> dict:
    """Counts of rows per exclusion reason, plus totals."""
    counts = audit.loc[~audit["used_for_model"], "exclusion_reason"].value_counts()
    return {
        "rows_total": int(len(audit)),
        "rows_used": int(audit["used_for_model"].sum()),
        "rows_excluded": {text: int(counts.get(text, 0)) for _, text in EXCLUSION_REASONS},
    }
