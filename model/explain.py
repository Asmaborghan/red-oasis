"""
SHAP explanations for the Red Oasis forecast model.

SHAP splits one prediction into a starting point (the model's average
prediction on the training data) plus one contribution per input. The
contributions add up exactly to the prediction. They describe how THIS
model uses the inputs; they are associations, not proof of cause.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def make_explainer(pipeline):
    """Exact TreeSHAP explainer for the fitted tree model in the pipeline."""
    import shap  # imported lazily so the website still loads if shap is missing

    return shap.TreeExplainer(pipeline.named_steps["model"])


def _transform(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Apply the same preprocessing the model saw during training."""
    return pipeline.named_steps["imputer"].transform(X[config.FEATURE_ORDER])


def _base_value(explainer) -> float:
    return float(np.ravel(explainer.expected_value)[0])


def explain_row(pipeline, X_row: pd.DataFrame, explainer=None, imputed=()) -> dict:
    """Explain a single prediction.

    Returns the base value, the prediction, and contributions ranked by
    absolute size. `imputed` lists features that were filled with the
    training median, so the display can flag them.
    """
    explainer = explainer or make_explainer(pipeline)
    Xt = _transform(pipeline, X_row)
    values = np.asarray(explainer.shap_values(Xt))[0]
    base = _base_value(explainer)

    contributions = []
    for name, value, contrib in zip(config.FEATURE_ORDER, Xt[0], values):
        contributions.append({
            "feature": name,
            "label": config.FEATURES[name]["label"],
            "value": float(value),
            "contribution": float(contrib),
            "imputed": name in imputed,
        })
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    return {
        "method": "SHAP TreeExplainer (exact, tree-path-dependent)",
        "base_value": base,
        "prediction": base + float(values.sum()),
        "contributions": contributions,
    }


def global_importance(pipeline, X: pd.DataFrame, explainer=None, max_rows: int = 500,
                      random_state: int = config.RANDOM_STATE) -> list[dict]:
    """Mean absolute SHAP value per feature over a sample of rows, ranked."""
    explainer = explainer or make_explainer(pipeline)
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=random_state)
    values = np.asarray(explainer.shap_values(_transform(pipeline, X)))
    mean_abs = np.abs(values).mean(axis=0)
    ranked = sorted(zip(config.FEATURE_ORDER, mean_abs), key=lambda p: p[1], reverse=True)
    return [
        {"feature": f, "label": config.FEATURES[f]["label"], "mean_abs_shap": float(v)}
        for f, v in ranked
    ]


def summary_text(explanation: dict, target_label: str, unit: str, horizon_days: int) -> str:
    """A short, plain-English explanation suitable for Streamlit."""
    pred = explanation["prediction"]
    base = explanation["base_value"]
    lines = [
        f"Forecast {target_label.lower()} in {horizon_days} days: {pred:.2f} {unit}. "
        f"The model's average forecast is {base:.2f}; the inputs moved it as follows:"
    ]
    for c in explanation["contributions"]:
        direction = "raised" if c["contribution"] >= 0 else "lowered"
        note = " (missing, filled with training median)" if c["imputed"] else ""
        lines.append(
            f"- {c['label']} = {c['value']:.2f}{note} {direction} it by {abs(c['contribution']):.2f}"
        )
    lines.append(
        "These are the model's learned associations, not proof that a factor causes the change."
    )
    return "\n".join(lines)
