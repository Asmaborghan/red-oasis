"""The 'Replay a real day' file must match the saved evaluation and avoid leakage."""

import json
from datetime import date, timedelta

import pytest

from model import config

REPLAY = config.PACKAGE_DIR / "figures" / "replay_days.json"
EVALUATION = config.DEFAULT_MODEL_DIR / config.EVALUATION_FILE


@pytest.fixture(scope="module")
def replay():
    if not REPLAY.exists():
        pytest.skip("replay_days.json not generated (python -m model.replay)")
    return json.loads(REPLAY.read_text(encoding="utf-8"))


def test_replay_matches_saved_evaluation(replay):
    evaluation = json.loads(EVALUATION.read_text())
    assert len(replay["days"]) == evaluation["n_test"]
    errors = [abs(d["forecast_pct"] - d["observed_pct"]) for d in replay["days"]]
    assert abs(sum(errors) / len(errors) - evaluation["test_metrics"]["random_forest"]["mae"]) < 1e-3


def test_replay_days_are_after_training_and_three_days_ahead(replay):
    train_end = date.fromisoformat(replay["train_period"][1])
    for d in replay["days"]:
        day, target = date.fromisoformat(d["date"]), date.fromisoformat(d["target_date"])
        assert day > train_end
        assert target - day == timedelta(days=replay["horizon_days"])
        assert d["level"] in ("irrigate", "monitor", "none")
