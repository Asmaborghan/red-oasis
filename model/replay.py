"""
"Replay a real day": precomputed forecasts for every day of the final test period.

    python -m model.replay            # writes model/figures/replay_days.json

For each test day the file holds the inputs that were available on that date,
the 3-day forecast, the provisional irrigation level and what SMAP actually
observed three days later.

No leakage: the forecasts come from the evaluation-period model, fitted only on
training rows that end before the test period (the same model as figures.py and
evaluation.json), and the rule's site medians are taken from those training rows
only. The saved production model in model/artifacts/ is NOT used, because it was
refitted on all rows, including the test days.

The script checks its errors against model/artifacts/evaluation.json and stops
if they disagree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, data, figures, irrigation

DEFAULT_OUT = config.PACKAGE_DIR / "figures" / "replay_days.json"


def _num(v, digits=4):
    return None if v is None or v != v else round(float(v), digits)


def build_replay(raw, horizon_days: int = 3) -> dict:
    res = figures.test_period_results(raw, horizon_days)
    te, tr = res["test"], res["train"]
    reference = {
        "soil_median_pct": float(tr["soil_moisture_pct"].median()),
        "et0_median_mm": float(tr["et0_mm"].median()),
    }
    days = []
    for _, row in te.iterrows():
        rule = irrigation.irrigation_rule(
            float(row["forecast"]), float(row["soil_moisture_pct"]),
            None if row["rainfall_7d_mm"] != row["rainfall_7d_mm"] else float(row["rainfall_7d_mm"]),
            None if row["et0_mm"] != row["et0_mm"] else float(row["et0_mm"]),
            reference,
        )
        days.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "target_date": row["target_date"].strftime("%Y-%m-%d"),
            "inputs": {name: _num(row[name]) for name in config.FEATURE_ORDER},
            "forecast_pct": _num(row["forecast"]),
            "observed_pct": _num(row["target_future"]),
            "persistence_pct": _num(row["persistence"]),
            "level": rule["level"],
            "checks": [{"name": c["name"], "met": c["met"], "value": _num(c["value"]),
                        "reference": _num(c["reference"])} for c in rule["checks"]],
        })
    return {
        "description": ("Final test period replay. Forecasts from the evaluation-period model "
                        "(trained only on days before the test period); rule medians from "
                        "the same training rows."),
        "train_period": [str(tr["date"].min().date()), str(tr["date"].max().date())],
        "test_period": [days[0]["date"], days[-1]["date"]] if days else [],
        "horizon_days": horizon_days,
        "rule_reference": reference,
        "rule_version": irrigation.RULE_VERSION,
        "summary": figures.summarise(res)["mae"],
        "days": days,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Precompute the 'Replay a real day' file.")
    p.add_argument("--data", type=Path, default=config.DEFAULT_DATA_FILE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    replay = build_replay(data.load_raw(args.data))
    evaluation = json.loads((config.DEFAULT_MODEL_DIR / config.EVALUATION_FILE).read_text())
    got = replay["summary"]["random_forest"]
    want = evaluation["test_metrics"]["random_forest"]["mae"]
    if len(replay["days"]) != evaluation["n_test"] or abs(got - want) > 1e-6:
        print(f"Replay does not match evaluation.json (MAE {got:.6f} vs {want:.6f}).", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(replay, indent=1, allow_nan=False))
    print(f"Wrote {len(replay['days'])} days to {args.out} (MAE {got:.3f}, matches evaluation.json).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
