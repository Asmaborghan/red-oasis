"""
Three figures from the final test period (Week 3):

  1. fig1_forecast_vs_actual.png   3-day forecast vs observed soil moisture
  2. fig2_error_rain_vs_no_rain.png  forecast error with and without recent rain
  3. fig3_shap_summary.png          how much each input moves the forecast (SHAP)

    python -m model.figures                 # uses data/raw/red_oasis_daily.csv
    python -m model.figures --data FILE.csv --out model/figures

The figures use the same evaluation-period model as `train.evaluate()`
(fitted on training rows only, scored once on the final test period). The
script checks its numbers against model/artifacts/evaluation.json and stops
if they disagree, so the figures always match the reported results. The
numbers behind each figure are saved in figures_data.json (table view).

Needs matplotlib (not required by the website): pip install matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, data, train

DEFAULT_OUT = config.PACKAGE_DIR / "figures"

# Reference palette (validated: light surface, CVD-safe pair). Model = blue,
# persistence = orange in every figure; observed values in neutral ink.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL, PERSIST = "#2a78d6", "#eb6834"
LABELS = {
    "soil_moisture_pct": "Soil moisture today",
    "rainfall_7d_mm": "Rain, last 7 days",
    "ndvi": "Vegetation index (NDVI)",
    "temperature_c": "Air temperature",
    "et0_mm": "Reference evapotranspiration (ET0)",
}


class FigureCheckError(RuntimeError):
    """The recomputed test results do not match evaluation.json."""


def test_period_results(raw: pd.DataFrame, horizon_days: int = 3) -> dict:
    """Refit the evaluation-period model and return test-period predictions + SHAP."""
    feats = data.build_features(raw)
    df, _ = data.select_rows(data.add_target(feats, "soil_moisture_pct", horizon_days))
    tr, te, cutoff = train.chronological_split(df, horizon_days)
    X = config.FEATURE_ORDER
    model = train.build_pipeline().fit(tr[X], tr["target_future"])
    te = te.assign(forecast=model.predict(te[X]), persistence=te["soil_moisture_pct"])

    from . import explain

    explainer = explain.make_explainer(model)
    shap_values = np.asarray(explainer.shap_values(explain._transform(model, te)))
    return {"test": te.reset_index(drop=True), "train": tr, "shap": shap_values,
            "cutoff": cutoff}


def summarise(res: dict) -> dict:
    te = res["test"]
    err_m = (te["forecast"] - te["target_future"]).abs()
    err_p = (te["persistence"] - te["target_future"]).abs()
    wet = te["rainfall_7d_mm"] > 0
    mean_abs = np.abs(res["shap"]).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return {
        "test_period": [str(te["date"].min().date()), str(te["date"].max().date())],
        "n_test": int(len(te)),
        "mae": {"random_forest": float(err_m.mean()), "baseline_persistence": float(err_p.mean())},
        "mae_by_condition": {
            "rain_in_last_7_days": {"n": int(wet.sum()), "random_forest": float(err_m[wet].mean()),
                                    "baseline_persistence": float(err_p[wet].mean())},
            "no_rain_in_last_7_days": {"n": int((~wet).sum()),
                                       "random_forest": float(err_m[~wet].mean()),
                                       "baseline_persistence": float(err_p[~wet].mean())},
        },
        "spread_std": {"forecast": float(te["forecast"].std()),
                       "observed": float(te["target_future"].std())},
        "shap_mean_abs_test_rows": [
            {"feature": config.FEATURE_ORDER[i], "label": LABELS[config.FEATURE_ORDER[i]],
             "mean_abs_shap": float(mean_abs[i])} for i in order],
        "series": {
            "target_date": te["target_date"].dt.strftime("%Y-%m-%d").tolist(),
            "observed_pct": te["target_future"].round(4).tolist(),
            "forecast_pct": te["forecast"].round(4).tolist(),
        },
    }


def check_against_evaluation(summary: dict, evaluation: dict, tol: float = 1e-6) -> None:
    """Stop if the figures would not match the saved evaluation."""
    pairs = [
        (summary["mae"]["random_forest"], evaluation["test_metrics"]["random_forest"]["mae"]),
        (summary["mae"]["baseline_persistence"],
         evaluation["test_metrics"]["baseline_persistence"]["mae"]),
    ]
    by = evaluation.get("test_by_condition_mae") or {}
    for cond in ("rain_in_last_7_days", "no_rain_in_last_7_days"):
        if cond in by:
            for name in ("random_forest", "baseline_persistence"):
                pairs.append((summary["mae_by_condition"][cond][name], by[cond][name]))
    if summary["n_test"] != evaluation["n_test"]:
        raise FigureCheckError(f"n_test {summary['n_test']} != {evaluation['n_test']}")
    for got, want in pairs:
        if abs(got - want) > tol:
            raise FigureCheckError(f"Recomputed {got:.6f} != evaluation.json {want:.6f}. "
                                   "Retrain (python -m model.train) or pass the same data file.")


def _style(ax, title, subtitle):
    """Recessive axes; title and subtitle anchored to the figure's left edge."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig = ax.figure
    fig.text(0.015, 0.965, title, fontsize=13, color=INK, fontweight="bold", va="top")
    fig.text(0.015, 0.905, subtitle, fontsize=9, color=INK_2, va="top")


LAYOUT_RECT = (0, 0.03, 1, 0.87)   # leave room for the figure-level title and note


def draw(summary: dict, out_dir: Path, site_label: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    s = summary["series"]
    period = f"{summary['test_period'][0]} to {summary['test_period'][1]}"
    note = f"{site_label}. Experimental model; SMAP is a ~9 km regional signal."

    # 1. Forecast vs observed over time
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200, facecolor=SURFACE)
    dates = pd.to_datetime(s["target_date"])
    ax.plot(dates, s["observed_pct"], color=INK_2, linewidth=1, alpha=0.6)
    ax.scatter(dates, s["observed_pct"], color=INK_2, s=14, zorder=3,
               edgecolors=SURFACE, linewidths=0.8, label="Observed (SMAP)")
    ax.plot(dates, s["forecast_pct"], color=MODEL, linewidth=2, zorder=4,
            label="Model forecast (made 3 days earlier)")
    ax.set_ylabel("Soil moisture (%)", color=INK_2, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    sp = summary["spread_std"]
    _style(ax, "3-day forecast vs observed soil moisture",
           f"Final test period {period}, {summary['n_test']} days. Forecast spread "
           f"{sp['forecast']:.1f} vs observed {sp['observed']:.1f} percentage points (std).")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.text(0.01, 0.01, note, fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=LAYOUT_RECT)
    paths.append(out_dir / "fig1_forecast_vs_actual.png")
    fig.savefig(paths[-1], facecolor=SURFACE)
    plt.close(fig)

    # 2. Error with and without rain in the previous 7 days
    by = summary["mae_by_condition"]
    conds = [("rain_in_last_7_days", "Rain in last 7 days"),
             ("no_rain_in_last_7_days", "No rain in last 7 days")]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200, facecolor=SURFACE)
    x = np.arange(len(conds))
    w = 0.36
    for i, (name, color, label) in enumerate([("random_forest", MODEL, "Model (Random Forest)"),
                                              ("baseline_persistence", PERSIST,
                                               "Persistence (same as today)")]):
        vals = [by[c][name] for c, _ in conds]
        bars = ax.bar(x + (i - 0.5) * (w + 0.02), vals, width=w, color=color, label=label,
                      edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{lab}\n(n = {by[c]['n']})" for c, lab in conds], color=INK)
    ax.set_ylabel("Mean absolute error (percentage points)", color=INK_2, fontsize=9)
    ax.set_ylim(0, max(max(by[c]["random_forest"], by[c]["baseline_persistence"])
                       for c, _ in conds) * 1.25)
    _style(ax, "Forecast error with and without recent rain",
           f"Final test period {period}. Lower is better.")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.text(0.01, 0.01, note, fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=LAYOUT_RECT)
    paths.append(out_dir / "fig2_error_rain_vs_no_rain.png")
    fig.savefig(paths[-1], facecolor=SURFACE)
    plt.close(fig)

    # 3. SHAP summary (mean absolute contribution on test rows)
    rows = summary["shap_mean_abs_test_rows"][::-1]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200, facecolor=SURFACE)
    vals = [r["mean_abs_shap"] for r in rows]
    bars = ax.barh([r["label"] for r in rows], vals, color=MODEL, height=0.6,
                   edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=9, color=INK)
    ax.set_xlabel("Average size of the input's effect on the forecast "
                  "(percentage points, mean |SHAP|)", color=INK_2, fontsize=9)
    ax.tick_params(axis="y", colors=INK)
    _style(ax, "What drives the forecast (SHAP summary)",
           f"Final test period {period}. How the model uses each input; not physical cause.")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.text(0.01, 0.01, note, fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=LAYOUT_RECT)
    paths.append(out_dir / "fig3_shap_summary.png")
    fig.savefig(paths[-1], facecolor=SURFACE)
    plt.close(fig)
    return paths


def make_figures(data_path: Path, out_dir: Path, evaluation_path: Path | None = None) -> dict:
    raw = data.load_raw(data_path)
    summary = summarise(test_period_results(raw))
    evaluation_path = evaluation_path or (config.DEFAULT_MODEL_DIR / config.EVALUATION_FILE)
    if evaluation_path.exists():
        check_against_evaluation(summary, json.loads(evaluation_path.read_text()))
        summary["checked_against"] = str(evaluation_path.relative_to(config.REPO_DIR)
                                         if evaluation_path.is_relative_to(config.REPO_DIR)
                                         else evaluation_path.name)
    info = data.site_info(raw)
    site = ", ".join(info.get("sites", ["site"]))
    if info.get("lat") and info.get("lon"):
        site += f" ({info['lat'][0]:.4f} N, {info['lon'][0]:.4f} E)"
    summary["site"] = site
    summary["data_file"] = data_path.name
    paths = draw(summary, out_dir, site)
    summary["figures"] = [p.name for p in paths]
    (out_dir / "figures_data.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Draw the three Red Oasis test-period figures.")
    p.add_argument("--data", type=Path, default=config.DEFAULT_DATA_FILE)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--evaluation", type=Path, default=None,
                   help="evaluation.json to check against (default: model/artifacts)")
    args = p.parse_args(argv)
    try:
        s = make_figures(args.data, args.out, args.evaluation)
    except (FigureCheckError, data.DataError) as exc:
        print(f"Figures not drawn: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {', '.join(s['figures'])} and figures_data.json to {args.out}")
    print(f"Test MAE: model {s['mae']['random_forest']:.3f}, "
          f"persistence {s['mae']['baseline_persistence']:.3f} (n = {s['n_test']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
