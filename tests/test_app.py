"""The website (app/app.py on main) must keep working, with or without a model.

Connecting app.py to the model is Asma's task and is deliberately not done
until the audit is complete. Until then, only start-up and fallback checks
run; the integration checks skip.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from model import config

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"
CONNECTED = "from model." in APP.read_text(encoding="utf-8")


def _run(monkeypatch, model_dir):
    monkeypatch.setenv(config.MODEL_DIR_ENV, str(model_dir))
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_starts_and_shows_rule_based_advice_without_a_model(monkeypatch, tmp_path):
    at = _run(monkeypatch, tmp_path)
    text = " ".join(m.value for m in at.markdown)
    assert "Recommendation" in text and "Contributing factors" in text


def test_app_starts_when_only_a_blocked_synthetic_model_exists(monkeypatch, synthetic_model_dir):
    monkeypatch.delenv(config.ALLOW_SYNTHETIC_ENV, raising=False)
    at = _run(monkeypatch, synthetic_model_dir)
    assert "synthetic" not in " ".join(m.value for m in at.markdown).lower()


@pytest.mark.skipif(not CONNECTED, reason="app.py is not connected to model/ yet (after the audit)")
def test_connected_app_falls_back_when_model_unavailable(monkeypatch, tmp_path):
    at = _run(monkeypatch, tmp_path)
    assert "Contributing factors" in " ".join(m.value for m in at.markdown)


@pytest.mark.skipif(not CONNECTED, reason="app.py is not connected to model/ yet (after the audit)")
def test_connected_app_runs_with_the_saved_model_in_both_languages():
    for language in ("English", "العربية"):
        at = AppTest.from_file(str(APP), default_timeout=120)
        at.session_state["lang_choice"] = language
        at.run()
        assert not at.exception, (language, at.exception)
        assert any("Dariz" in i.value or "الدريز" in i.value for i in at.info)
