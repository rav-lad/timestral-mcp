"""Chronological backtesting (single holdout or rolling origin) against naive baselines.

Each window w has a forecast origin: the model sees only the observations before it and
forecasts the next H steps, which are compared with what actually happened. With W
windows spaced `step` apart, the most recent window ends at the last observation:

    window w (oldest first): context = y[:o_w], test = y[o_w : o_w + H],
    o_w = N - H - (W - 1 - w) * step

W = 1 is a single holdout. Metrics are pooled over every forecast point of every window
(equal to the mean of per-window MAE, since all windows have H points). Neither form
establishes general forecasting reliability; more windows only make the estimate less
dependent on one particular stretch of data.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from timestral_mcp.forecasting import (
    MAX_HORIZON,
    MIN_OBSERVATIONS,
    Covariates,
    ForecastResult,
    InputError,
    covariate_signature,
    run_forecast_batch,
)

BACKTEST_QUANTILES = (0.1, 0.5, 0.9)
MAX_WINDOWS = 100
# Every forecast point is returned to the client, so their total is bounded.
MAX_BACKTEST_POINTS = 5000


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actuals inside the closed interval [lower, upper]."""
    return float(np.mean((lower <= actual) & (actual <= upper)))


def wape(mae_value: float, mean_abs_actual: float) -> float | None:
    """Weighted absolute percentage error: sum |error| / sum |actual|; None when every actual is 0."""
    return None if mean_abs_actual == 0 else mae_value / mean_abs_actual


def improvement_percent(baseline_error: float, model_error: float) -> float | None:
    """Relative error reduction versus a baseline; None when the baseline error is 0."""
    if baseline_error == 0 or not math.isfinite(baseline_error):
        return None
    return 100.0 * (baseline_error - model_error) / baseline_error


def naive_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    """y_hat[T+h] = y[T], using the last observed (non-missing) value."""
    observed = context[~np.isnan(context)]
    if observed.size == 0:
        raise InputError("the context has no observed values for the naive baseline")
    return np.full(horizon, observed[-1])


def seasonal_naive_forecast(context: np.ndarray, horizon: int, period: int) -> np.ndarray:
    """y_hat[T+h] = y[T+h-m*(floor((h-1)/m)+1)]: the value one season earlier."""
    if period < 1:
        raise InputError(f"seasonal_period must be at least 1, got {period}")
    if period > context.size:
        raise InputError(f"seasonal_period {period} is longer than the {context.size}-step context")
    t = context.size
    h = np.arange(1, horizon + 1)
    forecast = context[t + h - period * ((h - 1) // period + 1) - 1]
    if np.isnan(forecast).any():
        raise InputError("the seasonal naive baseline refers to missing values in the last season of the context")
    return forecast


def _observed(context: np.ndarray, needed: int, method: str) -> np.ndarray:
    observed = context[~np.isnan(context)]
    if observed.size < needed:
        raise InputError(f"the {method} baseline needs {needed} observed value(s) in the context, got {observed.size}")
    return observed


def mean_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    """y_hat[T+h] = mean of every observed value of the context."""
    return np.full(horizon, float(np.mean(_observed(context, 1, "mean"))))


def moving_average_forecast(context: np.ndarray, horizon: int, window: int) -> np.ndarray:
    """y_hat[T+h] = mean of the last `window` observed values."""
    return np.full(horizon, float(np.mean(_observed(context, window, f"moving_average:{window}")[-window:])))


def drift_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    """The line through the first and last observed values, extended: y[T] + h * slope."""
    positions = np.flatnonzero(~np.isnan(context))
    if positions.size < 2:
        raise InputError("the drift baseline needs 2 observed values in the context")
    first, last = positions[0], positions[-1]
    slope = (context[last] - context[first]) / (last - first)
    return context[last] + slope * (np.arange(1, horizon + 1) + (context.size - 1 - last))


def seasonal_growth_forecast(context: np.ndarray, horizon: int, period: int) -> np.ndarray:
    """The seasonal naive forecast scaled by the growth of the last season over the one before.

    E.g. weekly data, period 52: same week last year x (last 52 weeks / the 52 weeks before).
    """
    if context.size < 2 * period:
        raise InputError(
            f"the seasonal_growth baseline needs two seasons ({2 * period} steps) of context, got {context.size}"
        )
    last, previous = np.nansum(context[-period:]), np.nansum(context[-2 * period : -period])
    if previous == 0:
        raise InputError("the seasonal_growth baseline cannot scale by growth: the season before the last sums to 0")
    return seasonal_naive_forecast(context, horizon, period) * (last / previous)


# Extra baselines a backtest can be asked for, with whether they need seasonal_period.
EXTRA_BASELINES = ("mean", "moving_average:N", "drift", "seasonal_growth")


def baseline_forecasters(
    specs: list[str] | None, seasonal_period: int | None
) -> dict[str, Callable[[np.ndarray, int], np.ndarray]]:
    """The baselines of a backtest, by name: naive always, seasonal_naive with a seasonal_period, then `specs`."""
    out: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {"naive": naive_forecast}
    if seasonal_period is not None:
        out["seasonal_naive"] = lambda c, h: seasonal_naive_forecast(c, h, seasonal_period)
    for raw in specs or []:
        spec = raw.strip().lower().replace(" ", "")
        name, _, arg = spec.partition(":")
        if spec in out:
            continue
        if name in ("naive", "mean", "drift") and not arg:
            out[spec] = {"naive": naive_forecast, "mean": mean_forecast, "drift": drift_forecast}[name]
        elif name in ("seasonal_naive", "seasonal_growth") and not arg:
            if seasonal_period is None:
                raise InputError(f"the {name} baseline needs seasonal_period (e.g. 52 for weekly data)")
            if name == "seasonal_growth":
                out[spec] = lambda c, h: seasonal_growth_forecast(c, h, seasonal_period)
        elif name == "moving_average":
            if not arg.isdigit() or int(arg) < 1:
                raise InputError(f"write the moving average as 'moving_average:N' with N >= 1 periods, got {raw!r}")
            n = int(arg)
            out[f"moving_average:{n}"] = lambda c, h, n=n: moving_average_forecast(c, h, n)
        else:
            raise InputError(
                f"unknown baseline {raw!r}; available: naive, seasonal_naive, {', '.join(EXTRA_BASELINES)}. Other "
                "methods cannot be computed by this tool: compare with the user's own forecasts via benchmark_column."
            )
    return out


@dataclass(frozen=True)
class Baseline:
    method: str
    mae: float
    rmse: float
    wape: float | None
    mae_improvement_percent: float | None
    rmse_improvement_percent: float | None


@dataclass(frozen=True)
class Window:
    origin: int  # observations in the context; the test window is [origin, origin + H)
    forecast: ForecastResult
    actual: np.ndarray
    mae: float
    rmse: float
    coverage_80: float
    baselines: list[Baseline]


@dataclass(frozen=True)
class BacktestResult:
    test_size: int
    step: int
    windows: list[Window]  # oldest first
    mae: float
    rmse: float
    coverage_80: float
    baselines: list[Baseline]
    mean_abs_actual: float  # over every test point; WAPE = MAE / mean_abs_actual

    @property
    def wape(self) -> float | None:
        return wape(self.mae, self.mean_abs_actual)


def _is_int(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def origins(n: int, test_size: int, windows: int = 1, step: int | None = None) -> list[int]:
    """Forecast origins (context lengths) of each window, oldest first."""
    if not _is_int(test_size) or test_size < 1:
        raise InputError(f"test_size must be a positive integer, got {test_size!r}")
    if test_size > MAX_HORIZON:
        raise InputError(f"test_size must be at most {MAX_HORIZON}, got {test_size}")
    if not _is_int(windows) or not 1 <= windows <= MAX_WINDOWS:
        raise InputError(f"windows must be an integer from 1 to {MAX_WINDOWS}, got {windows!r}")
    step = test_size if step is None else step
    if not _is_int(step) or step < 1:
        raise InputError(f"step must be a positive integer, got {step!r}")
    if windows * test_size > MAX_BACKTEST_POINTS:
        raise InputError(
            f"windows x test_size = {windows * test_size} forecast points; at most {MAX_BACKTEST_POINTS} are supported"
        )
    first = n - test_size - (windows - 1) * step
    if first < MIN_OBSERVATIONS:
        raise InputError(
            f"{windows} window(s) of {test_size} steps spaced {step} apart leave {max(first, 0)} observation(s) "
            f"of context before the first window, out of {n}; at least {MIN_OBSERVATIONS} are needed. "
            "Reduce test_size, windows or step."
        )
    return [first + w * step for w in range(windows)]


def split(values: np.ndarray, origin: int, test_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split into context `values[:origin]` and test window `values[origin:origin+H]`."""
    context, test = values[:origin], values[origin : origin + test_size]
    observed = int((~np.isnan(context)).sum())
    if observed < MIN_OBSERVATIONS:
        raise InputError(
            f"the context before position {origin} has {observed} observed value(s); at least {MIN_OBSERVATIONS} "
            "are needed"
        )
    if np.isnan(test).any():
        raise InputError(
            f"the test window starting at position {origin} contains missing target values; "
            "choose windows with complete data"
        )
    return context, test


def _baselines(
    actual_by_method: dict[str, tuple[np.ndarray, np.ndarray]], model_mae: float, model_rmse: float
) -> list[Baseline]:
    out = []
    for method, (actual, predicted) in actual_by_method.items():
        b_mae, b_rmse = mae(actual, predicted), rmse(actual, predicted)
        out.append(
            Baseline(
                method=method,
                mae=b_mae,
                rmse=b_rmse,
                wape=wape(b_mae, float(np.mean(np.abs(actual)))),
                mae_improvement_percent=improvement_percent(b_mae, model_mae),
                rmse_improvement_percent=improvement_percent(b_rmse, model_rmse),
            )
        )
    return out


def run_backtest(
    values: np.ndarray,
    test_size: int,
    model_id: str,
    device: str,
    seasonal_period: int | None = None,
    windows: int = 1,
    step: int | None = None,
    covariates: Covariates | None = None,
    baselines: list[str] | None = None,
    benchmark: tuple[str, np.ndarray] | None = None,
) -> BacktestResult:
    """Backtest over `windows` origins. `covariates` are aligned with `values`.

    Past covariates are cut at each origin like the target. Known covariates are given over
    the test window too: they are assumed to be known in advance (e.g. a promotion calendar).
    `benchmark` is `(name, forecasts)`: forecasts made by other means (e.g. the user's own),
    aligned with `values`, scored like a baseline on the same points.
    """
    result = run_backtests(
        [(values, covariates)],
        test_size,
        model_id,
        device,
        seasonal_period,
        windows,
        step,
        baselines,
        None if benchmark is None else [benchmark],
    )[0]
    if isinstance(result, InputError):
        raise result
    return result


@dataclass(frozen=True)
class _Plan:
    values: np.ndarray
    covariates: Covariates | None
    starts: list[int]
    splits: list[tuple[np.ndarray, np.ndarray]]
    baselines: list[dict[str, np.ndarray]]


def _plan(
    values: np.ndarray,
    covariates: Covariates | None,
    test_size: int,
    forecasters: dict[str, Callable[[np.ndarray, int], np.ndarray]],
    windows: int,
    step: int | None,
    benchmark: tuple[str, np.ndarray] | None = None,
) -> _Plan:
    """Validate every split and baseline of one series before running the model, so that a bad argument fails fast."""
    starts = origins(values.size, test_size, windows, step)
    splits = [split(values, o, test_size) for o in starts]
    baseline_forecasts = []
    for o, (context, _) in zip(starts, splits, strict=True):
        forecasts = {name: f(context, test_size) for name, f in forecasters.items()}
        if benchmark is not None:
            name, row = benchmark
            given = np.asarray(row[o : o + test_size], dtype=np.float64)
            if given.size < test_size or np.isnan(given).any():
                raise InputError(
                    f"{name!r} has no value for some dates of the test windows: the forecasts to compare with must "
                    f"cover the last {values.size - starts[0]} periods of the history (windows x test_size)"
                )
            forecasts[name] = given
        baseline_forecasts.append(forecasts)
    if covariates is not None:
        for o in starts:
            covariates.window(o, test_size)
    return _Plan(values, covariates, starts, splits, baseline_forecasts)


def run_backtests(
    series: list[tuple[np.ndarray, Covariates | None]],
    test_size: int,
    model_id: str,
    device: str,
    seasonal_period: int | None = None,
    windows: int = 1,
    step: int | None = None,
    baselines: list[str] | None = None,
    benchmarks: list[tuple[str, np.ndarray] | None] | None = None,
) -> list[BacktestResult | InputError]:
    """Backtest several series; each entry is a result or the error that kept the series out.

    Series of the same length share their forecast origins, so each window is forecast for
    all of them in one batch; the results are those of separate calls. An unknown baseline
    name fails the whole call; `benchmarks` (one per series, or None) are scored as baselines.
    """
    forecasters = baseline_forecasters(baselines, seasonal_period)
    plans: list[_Plan | InputError] = []
    for i, (values, covariates) in enumerate(series):
        benchmark = None if benchmarks is None else benchmarks[i]
        try:
            plans.append(_plan(values, covariates, test_size, forecasters, windows, step, benchmark))
        except InputError as e:
            plans.append(e)

    forecasts: dict[int, list[ForecastResult]] = {}
    groups: dict[tuple[int, tuple[tuple[str, ...], tuple[str, ...]]], list[int]] = {}
    for i, plan in enumerate(plans):
        if isinstance(plan, _Plan):
            groups.setdefault((plan.values.size, covariate_signature(plan.covariates)), []).append(i)
    for members in groups.values():
        per_window = []
        for w in range(len(plans[members[0]].starts)):
            items = [(plans[i].splits[w][0], plans[i].covariates) for i in members]
            per_window.append(run_forecast_batch(items, test_size, model_id, device, BACKTEST_QUANTILES))
        for k, i in enumerate(members):
            forecasts[i] = [per_window[w][k] for w in range(len(per_window))]

    return [
        plan if isinstance(plan, InputError) else _score(plan, forecasts[i], test_size, step)
        for i, plan in enumerate(plans)
    ]


def _score(plan: _Plan, forecasts: list[ForecastResult], test_size: int, step: int | None) -> BacktestResult:
    results = []
    for o, (_, actual), baseline, forecast in zip(plan.starts, plan.splits, plan.baselines, forecasts, strict=True):
        median = forecast.quantiles[0.5]
        w_mae, w_rmse = mae(actual, median), rmse(actual, median)
        results.append(
            Window(
                origin=o,
                forecast=forecast,
                actual=actual,
                mae=w_mae,
                rmse=w_rmse,
                coverage_80=interval_coverage(actual, forecast.quantiles[0.1], forecast.quantiles[0.9]),
                baselines=_baselines({m: (actual, p) for m, p in baseline.items()}, w_mae, w_rmse),
            )
        )

    actual = np.concatenate([w.actual for w in results])
    median = np.concatenate([w.forecast.quantiles[0.5] for w in results])
    lower = np.concatenate([w.forecast.quantiles[0.1] for w in results])
    upper = np.concatenate([w.forecast.quantiles[0.9] for w in results])
    pooled_mae, pooled_rmse = mae(actual, median), rmse(actual, median)
    pooled_baselines = {
        method: (actual, np.concatenate([f[method] for f in plan.baselines])) for method in plan.baselines[0]
    }
    return BacktestResult(
        test_size=test_size,
        step=test_size if step is None else step,
        windows=results,
        mae=pooled_mae,
        rmse=pooled_rmse,
        coverage_80=interval_coverage(actual, lower, upper),
        baselines=_baselines(pooled_baselines, pooled_mae, pooled_rmse),
        mean_abs_actual=float(np.mean(np.abs(actual))),
    )


def pool(results: list[BacktestResult]) -> BacktestResult:
    """Pool backtests of several series run with the same test_size, windows and step.

    Every series contributes the same number of points, so pooled MAE and coverage are the
    means of the per-series values and pooled RMSE is the root of the mean squared RMSE.
    Errors are in the units of each series: series on larger scales weigh more. WAPE is
    pooled as total absolute error over total absolute actual values.
    """
    if len(results) == 1:
        return results[0]
    model_mae = float(np.mean([r.mae for r in results]))
    model_rmse = float(np.sqrt(np.mean([r.rmse**2 for r in results])))
    mean_abs_actual = float(np.mean([r.mean_abs_actual for r in results]))
    baselines = []
    for i, method in enumerate(b.method for b in results[0].baselines):
        b_mae = float(np.mean([r.baselines[i].mae for r in results]))
        b_rmse = float(np.sqrt(np.mean([r.baselines[i].rmse ** 2 for r in results])))
        baselines.append(
            Baseline(
                method=method,
                mae=b_mae,
                rmse=b_rmse,
                wape=wape(b_mae, mean_abs_actual),
                mae_improvement_percent=improvement_percent(b_mae, model_mae),
                rmse_improvement_percent=improvement_percent(b_rmse, model_rmse),
            )
        )
    return BacktestResult(
        test_size=results[0].test_size,
        step=results[0].step,
        windows=[w for r in results for w in r.windows],
        mae=model_mae,
        rmse=model_rmse,
        coverage_80=float(np.mean([r.coverage_80 for r in results])),
        baselines=baselines,
        mean_abs_actual=mean_abs_actual,
    )
