"""Shared fixtures.

Models trained here on SYNTHETIC data only test the software. Synthetic
models are blocked by default, so tests that use one opt in explicitly
through the `allow_synthetic` fixture.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import config, train  # noqa: E402
from model.synthetic import SYNTHETIC_NOTICE, make_synthetic_raw  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_model_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic_model")
    evaluation = train.train(
        make_synthetic_raw(), "soil_moisture_pct", 3, out, SYNTHETIC_NOTICE, is_synthetic=True
    )
    assert evaluation["status"] == "ok"
    return out


@pytest.fixture
def allow_synthetic(monkeypatch):
    monkeypatch.setenv(config.ALLOW_SYNTHETIC_ENV, "1")


@pytest.fixture
def good_inputs():
    return {"soil_moisture_pct": 12.0, "ndvi": 0.4, "rainfall_7d_mm": 0.0,
            "temperature_c": 35.0, "et0_mm": 6.5}


@pytest.fixture(scope="session")
def real_csv():
    if not config.DEFAULT_DATA_FILE.exists():
        pytest.skip("data/raw/red_oasis_daily.csv is not present")
    return config.DEFAULT_DATA_FILE
