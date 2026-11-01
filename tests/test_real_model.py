"""Real Timestral inference (no test doubles). Downloads Timestral-28M (~111 MB) on first run.

Run with: uv run pytest -m model
"""

import numpy as np
import pandas as pd
import pytest

from timestral_mcp import model as model_mod
from timestral_mcp import server
from timestral_mcp.forecasting import NATIVE_QUANTILES, run_forecast, run_forecast_joint, validate_values

pytestmark = pytest.mark.model


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """One year of daily sales ending 2026-10-31, without and with a promo column plus 14 future rows."""
    rng = np.random.default_rng(42)
    weekly = np.array([-12.0, -8.0, -4.0, 0.0, 6.0, 14.0, 4.0])
    history = pd.date_range("2025-11-01", "2026-10-31", freq="D")
    dates = history.append(pd.date_range("2026-11-01", periods=14, freq="D"))
    promo = (rng.random(len(dates)) < 0.08).astype(int)
    sales = 120 + 0.08 * np.arange(len(dates)) + weekly[dates.day_of_week] + rng.normal(0, 4, len(dates))
    sales += 25 * promo
    sales[len(history) :] = np.nan
    day = dates.strftime("%Y-%m-%d")
    pd.DataFrame({"date": day[: len(history)], "sales": sales[: len(history)].round(1)}).to_csv(
        tmp_path / "sales.csv", index=False
    )
    pd.DataFrame({"date": day, "sales": sales.round(1), "promo": promo}).to_csv(
        tmp_path / "sales_promo.csv", index=False
    )
    cfg = model_mod.Settings(
        model=model_mod.DEFAULT_MODEL, device="cpu", allowed_dirs=(tmp_path.resolve(),), max_file_bytes=10 * 1024 * 1024
    )
    monkeypatch.setattr(server, "_settings", cfg)
    return tmp_path


@pytest.fixture(scope="module")
def series() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(200)
    return 50 + 0.2 * t + 8 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, t.size)


def test_real_forecast_shape_validity_and_ordering(series):
    result = run_forecast(validate_values(series), 24, model_mod.DEFAULT_MODEL, "cpu", NATIVE_QUANTILES)
    stacked = np.stack([result.quantiles[q] for q in NATIVE_QUANTILES], axis=-1)
    assert stacked.shape == (24, 21)
    assert np.isfinite(stacked).all()
    assert (np.diff(stacked, axis=-1) >= 0).all(), "quantiles must be non-decreasing in the level"
    # A sanity bound, not an accuracy claim: the median stays near the series' range.
    assert series.min() - 50 < result.quantiles[0.5].min() and result.quantiles[0.5].max() < series.max() + 50


def test_real_model_is_cached():
    first = model_mod.get_forecaster(model_mod.DEFAULT_MODEL, "cpu")
    assert model_mod.get_forecaster(model_mod.DEFAULT_MODEL, "cpu") is first


def test_real_tools_on_a_file(data_dir):
    out = server.forecast_csv("sales.csv", horizon=14)
    assert len(out.forecast) == 14 and out.forecast[0]["date"] == "2026-11-01"
    for row in out.forecast:
        assert row["p10"] <= row["p50"] <= row["p90"]

    bt = server.backtest("sales.csv", test_size=30, seasonal_period=7)
    ev = bt.evaluation
    assert ev.test_size == 30 and bt.windows[0].context_size == 335
    assert np.isfinite([ev.mae, ev.rmse]).all() and 0.0 <= ev.coverage_80 <= 1.0


def test_real_rolling_backtest_and_known_covariates(data_dir):
    out = server.forecast_csv("sales_promo.csv", horizon=14, known_covariates=["promo"])
    assert out.covariates.known == ["promo"] and out.covariates.future_rows == 14
    assert out.forecast[0]["date"] == "2026-11-01" and out.last_observation == "2026-10-31"
    assert all(np.isfinite([r["p10"], r["p50"], r["p90"]]).all() for r in out.forecast)
    assert all(r["p10"] <= r["p50"] <= r["p90"] for r in out.forecast)

    for cov in (None, ["promo"]):
        bt = server.backtest("sales_promo.csv", test_size=14, windows=4, known_covariates=cov, target_column="sales")
        assert bt.evaluation.method == "rolling_origin" and len(bt.windows) == 4
        assert [w.context_size for w in bt.windows] == [309, 323, 337, 351]
        assert len(bt.predictions) == 56
        assert np.isfinite([w.mae for w in bt.windows]).all()


def test_real_joint_forecast(series):
    related = np.stack([series, 2 * series + 5])
    results = run_forecast_joint(related, 12, model_mod.DEFAULT_MODEL, "cpu", NATIVE_QUANTILES)
    assert len(results) == 2
    for result in results:
        stacked = np.stack([result.quantiles[q] for q in NATIVE_QUANTILES], axis=-1)
        assert stacked.shape == (12, 21) and np.isfinite(stacked).all()
        assert (np.diff(stacked, axis=-1) >= 0).all()
    # the second variate is about twice the first: a sanity check on the variate order
    assert results[1].quantiles[0.5].mean() > 1.5 * results[0].quantiles[0.5].mean()
