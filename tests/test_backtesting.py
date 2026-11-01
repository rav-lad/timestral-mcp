import math

import numpy as np
import pytest

from timestral_mcp.backtesting import (
    baseline_forecasters,
    drift_forecast,
    improvement_percent,
    interval_coverage,
    mae,
    mean_forecast,
    moving_average_forecast,
    naive_forecast,
    origins,
    rmse,
    run_backtest,
    seasonal_growth_forecast,
    seasonal_naive_forecast,
    split,
)
from timestral_mcp.forecasting import Covariates, InputError

A = np.array([1.0, 2.0, 3.0, 4.0])
P = np.array([2.0, 2.0, 5.0, 0.0])


def test_mae():
    # |1-2| + 0 + |3-5| + |4-0| = 7
    assert mae(A, P) == pytest.approx(7 / 4)


def test_rmse():
    # 1 + 0 + 4 + 16 = 21
    assert rmse(A, P) == pytest.approx(math.sqrt(21 / 4))


def test_perfect_forecast_has_zero_error():
    assert mae(A, A) == 0.0 and rmse(A, A) == 0.0


def test_interval_coverage_is_inclusive():
    lower = np.array([0.0, 2.0, 3.5, 0.0])
    upper = np.array([2.0, 2.0, 4.0, 3.0])
    # inside, on both bounds, below lower, above upper
    assert interval_coverage(A, lower, upper) == pytest.approx(0.5)


def test_improvement_percent():
    assert improvement_percent(10.0, 7.5) == pytest.approx(25.0)
    assert improvement_percent(10.0, 12.0) == pytest.approx(-20.0)
    assert improvement_percent(0.0, 0.0) is None
    assert improvement_percent(0.0, 3.0) is None


def test_naive_forecast_uses_last_observed_value():
    assert naive_forecast(np.array([1.0, 5.0, np.nan]), 3).tolist() == [5.0, 5.0, 5.0]


def test_seasonal_naive_forecast():
    context = np.array([10.0, 11.0, 12.0, 1.0, 2.0, 3.0])
    # Period 3: repeat the last season, wrapping over a horizon longer than the period.
    assert seasonal_naive_forecast(context, 7, 3).tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0]
    # Period 1 is the naive forecast.
    assert seasonal_naive_forecast(context, 2, 1).tolist() == [3.0, 3.0]
    # Period equal to the context length.
    assert seasonal_naive_forecast(context, 2, 6).tolist() == [10.0, 11.0]


def test_seasonal_naive_validation():
    with pytest.raises(InputError, match="longer than"):
        seasonal_naive_forecast(np.ones(5), 2, 6)
    with pytest.raises(InputError, match="at least 1"):
        seasonal_naive_forecast(np.ones(5), 2, 0)
    with pytest.raises(InputError, match="missing values"):
        seasonal_naive_forecast(np.array([1.0, np.nan, 3.0, 4.0]), 1, 3)


def test_split_is_chronological():
    context, test = split(np.arange(10.0), 7, 3)
    assert context.tolist() == list(range(7))
    assert test.tolist() == [7.0, 8.0, 9.0]


def test_split_rejects_missing_values_in_test_window():
    with pytest.raises(InputError, match="test window"):
        split(np.array([1.0, 2.0, 3.0, 4.0, np.nan]), 3, 2)


def test_single_origin_is_the_last_test_size_observations():
    assert origins(10, 3) == [7]


def test_rolling_origins_end_at_the_last_observation():
    # 3 non-overlapping windows of 4 steps in 20 observations: [8,12), [12,16), [16,20)
    assert origins(20, 4, windows=3) == [8, 12, 16]
    # Overlapping windows one step apart
    assert origins(20, 4, windows=3, step=1) == [14, 15, 16]
    # Gaps between windows when step > test_size
    assert origins(20, 2, windows=2, step=5) == [13, 18]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"test_size": 0}, "test_size"),
        ({"test_size": -1}, "test_size"),
        ({"test_size": True}, "test_size"),
        ({"test_size": 8}, "leave 2 observation"),
        ({"test_size": 11}, "leave 0 observation"),
        ({"test_size": 2, "windows": 0}, "windows"),
        ({"test_size": 2, "windows": 101}, "windows"),
        ({"test_size": 2, "windows": 2, "step": 0}, "step"),
        ({"test_size": 3, "windows": 3}, "before the first window"),
    ],
)
def test_origins_validation(kwargs, message):
    with pytest.raises(InputError, match=message):
        origins(10, **kwargs)


def test_total_forecast_points_are_bounded():
    with pytest.raises(InputError, match="forecast points"):
        origins(100_000, 4000, windows=2)


def test_backtest_never_shows_test_window_to_model(fake_forecaster):
    values = np.arange(1.0, 21.0)
    result = run_backtest(values, 5, "m", "cpu")
    context, horizon = fake_forecaster.calls[0]
    assert horizon == 5
    assert context.tolist() == list(range(1, 16))  # only observations before the window
    (window,) = result.windows
    assert window.actual.tolist() == [16.0, 17.0, 18.0, 19.0, 20.0]
    assert window.origin == 15


def test_rolling_backtest_contexts_never_include_their_window(fake_forecaster):
    values = np.arange(1.0, 31.0)
    result = run_backtest(values, 5, "m", "cpu", windows=3, step=4)
    assert [w.origin for w in result.windows] == [17, 21, 25]
    for (context, horizon), w in zip(fake_forecaster.calls, result.windows, strict=True):
        assert horizon == 5
        assert context.tolist() == values[: w.origin].tolist()
        assert w.actual.tolist() == values[w.origin : w.origin + 5].tolist()
        assert context.max() < w.actual.min()  # strictly earlier observations only


def test_backtest_metrics_against_hand_computation(fake_forecaster):
    values = np.arange(1.0, 21.0)
    result = run_backtest(values, 5, "m", "cpu", seasonal_period=5)
    # Fake median = last context value 15; actual 16..20 -> errors 1..5.
    assert result.mae == pytest.approx(3.0)
    assert result.rmse == pytest.approx(math.sqrt(11.0))
    # Fake p10/p90 = 15 -/+ 4 -> [11, 19] covers 16..19, not 20.
    assert result.coverage_80 == pytest.approx(0.8)
    naive, seasonal = result.baselines
    assert naive.method == "naive" and naive.mae == pytest.approx(3.0)
    assert naive.mae_improvement_percent == pytest.approx(0.0)
    # Seasonal naive with period 5 predicts 11..15 -> errors all 5.
    assert seasonal.method == "seasonal_naive" and seasonal.mae == pytest.approx(5.0)
    assert seasonal.mae_improvement_percent == pytest.approx(40.0)


def test_rolling_metrics_are_pooled_over_all_points(fake_forecaster):
    # Window 1: context 1..10, actual [20, 10] vs fake median 10 -> errors 10, 0
    # Window 2: context ..10,20,10, actual [11, 14] vs fake median 10 -> errors 1, 4
    values = np.array([*range(1, 11), 20, 10, 11, 14], dtype=float)
    result = run_backtest(values, 2, "m", "cpu", windows=2)
    w1, w2 = result.windows
    assert (w1.mae, w2.mae) == (pytest.approx(5.0), pytest.approx(2.5))
    assert result.mae == pytest.approx((10 + 0 + 1 + 4) / 4)
    assert result.rmse == pytest.approx(math.sqrt((100 + 0 + 1 + 16) / 4))
    # Fake interval = median -/+ 4: covers 10 (w1) and 11, 14 (w2), not 20.
    assert result.coverage_80 == pytest.approx(0.75)
    # Naive baseline equals the fake median here, so both share the pooled errors.
    assert result.baselines[0].mae == pytest.approx(result.mae)
    assert result.step == 2


def test_backtest_constant_series_has_no_improvement_percent(fake_forecaster):
    result = run_backtest(np.full(20, 7.0), 4, "m", "cpu")
    assert result.mae == 0.0 and result.rmse == 0.0
    assert result.baselines[0].mae_improvement_percent is None
    assert result.baselines[0].rmse_improvement_percent is None


def test_backtest_validates_everything_before_running_model(fake_forecaster):
    with pytest.raises(InputError, match="longer than"):
        run_backtest(np.arange(20.0), 5, "m", "cpu", seasonal_period=16)
    values = np.arange(20.0)
    values[12] = np.nan  # inside the oldest of three windows of 3
    with pytest.raises(InputError, match="test window starting at position 11"):
        run_backtest(values, 3, "m", "cpu", windows=3)
    assert fake_forecaster.calls == []


def test_backtest_slices_covariates_per_window(fake_forecaster):
    values = np.arange(1.0, 21.0)
    cov = Covariates(
        past={"temp": np.arange(100.0, 120.0)},
        known={"promo": np.arange(200.0, 220.0)},
        categorical=frozenset(),
        labels=tuple(str(i) for i in range(20)),
    )
    run_backtest(values, 3, "m", "cpu", windows=2, covariates=cov)
    first, second = fake_forecaster.covariate_calls
    # Past covariates stop at the origin; known covariates extend over the window.
    assert first["past_covariates"].shape == (1, 1, 14)
    assert first["known_covariates"].shape == (1, 1, 17)
    assert first["known_covariates"][0, 0, -1] == 216.0
    assert second["past_covariates"][0, 0, -1] == 116.0
    assert second["known_covariates"][0, 0, -1] == 219.0


# ---------------------------------------------------------------- extra baselines


def test_extra_baseline_formulas():
    ctx = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    assert mean_forecast(ctx, 2).tolist() == [3.0, 3.0]
    assert moving_average_forecast(ctx, 1, 2).tolist() == [4.5]  # last two observed values
    # line through (0, 1) and (4, 5): slope 1, continued after the last position
    assert drift_forecast(ctx, 3).tolist() == [6.0, 7.0, 8.0]
    # two seasons of 2: [1, 2] then [3, 4] -> growth 7/3, seasonal naive [3, 4]
    np.testing.assert_allclose(seasonal_growth_forecast(np.array([1.0, 2.0, 3.0, 4.0]), 2, 2), [7.0, 28 / 3])


def test_drift_counts_trailing_missing_steps():
    # last observed value at position 2, context ends at position 3: the first forecast is 2 steps after it
    assert drift_forecast(np.array([0.0, 1.0, 2.0, np.nan]), 1).tolist() == [4.0]


@pytest.mark.parametrize(
    "specs, period, message",
    [
        (["arima"], None, "unknown baseline 'arima'"),
        (["moving_average"], None, "moving_average:N"),
        (["moving_average:0"], None, "moving_average:N"),
        (["seasonal_growth"], None, "needs seasonal_period"),
        (["seasonal_naive"], None, "needs seasonal_period"),
    ],
)
def test_bad_baselines(specs, period, message):
    with pytest.raises(InputError, match=message):
        baseline_forecasters(specs, period)


def test_baseline_names_are_normalized_and_deduplicated():
    names = list(baseline_forecasters(["Mean", "moving_average: 4", "naive", "seasonal_naive", "drift"], 7))
    assert names == ["naive", "seasonal_naive", "mean", "moving_average:4", "drift"]


def test_requested_baselines_are_scored_on_the_same_windows(fake_forecaster):
    values = np.arange(1.0, 21.0)
    result = run_backtest(values, 2, "m", "cpu", windows=2, baselines=["mean", "moving_average:3"])
    assert [b.method for b in result.baselines] == ["naive", "mean", "moving_average:3"]
    # window origins 16 and 18; mean of 1..16 = 8.5 vs actual 17, 18; mean of 1..18 = 9.5 vs 19, 20
    mean = result.baselines[1]
    assert mean.mae == pytest.approx(np.mean([8.5, 9.5, 9.5, 10.5]))


def test_benchmark_is_scored_like_a_baseline(fake_forecaster):
    values = np.arange(1.0, 21.0)
    user = values + 1.0  # off by one everywhere
    result = run_backtest(values, 2, "m", "cpu", windows=2, benchmark=("benchmark:plan", user))
    bench = result.baselines[-1]
    assert bench.method == "benchmark:plan" and bench.mae == 1.0


def test_benchmark_must_cover_the_test_windows(fake_forecaster):
    values = np.arange(1.0, 21.0)
    user = values.copy()
    user[-1] = np.nan
    with pytest.raises(InputError, match="must cover the last 4 periods"):
        run_backtest(values, 2, "m", "cpu", windows=2, benchmark=("benchmark:plan", user))
