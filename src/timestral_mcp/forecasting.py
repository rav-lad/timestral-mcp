"""Univariate Timestral inference: input validation, covariates, the model call and quantile extraction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from timestral_mcp import model as model_mod

# Timestral's trained context length; older observations are dropped by the forecaster.
MODEL_CONTEXT_LENGTH = 32768
MIN_OBSERVATIONS = 3
MAX_INPUT_LENGTH = 100_000
# Horizons past one decode block (1024 steps) are decoded block by block, feeding back
# each block's median; beyond this cap a forecast is mostly the model's own feedback.
MAX_HORIZON = 4096
# Series forecast together in one model call; bounds memory, not results.
MAX_BATCH = 256
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
# The 21 quantile levels produced by every released Timestral checkpoint (config.json
# `quantile_knots`). The levels of the loaded checkpoint are re-checked at inference.
NATIVE_QUANTILES = (
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99,
)  # fmt: skip


class InputError(ValueError):
    """Invalid user input; the message explains how to fix it."""


@dataclass(frozen=True)
class Covariates:
    """Covariate rows aligned with a series of T observations.

    Past covariates are known over the history only and hold T values. Known (future)
    covariates also hold values for the F rows after the history, F >= 0. NaN marks a
    missing value. Categorical columns hold numeric category codes (NaN = missing);
    Timestral target-encodes them against the series.
    """

    past: dict[str, np.ndarray]
    known: dict[str, np.ndarray]
    categorical: frozenset[str]
    labels: tuple[str, ...]  # timestamp of each of the T + F rows, for error messages

    def window(self, context_end: int, horizon: int) -> dict[str, np.ndarray]:
        """`TimestralForecaster.forecast` covariate arguments for the context `[0, context_end)`.

        Known covariates must have a value for every one of the `horizon` steps after it.
        """
        for name, row in self.known.items():
            if row.size < context_end + horizon:
                available = row.size - context_end
                raise InputError(
                    f"known covariate {name!r} has values for {available} step(s) after the history, but the "
                    f"horizon is {horizon}; add future rows (empty target, filled covariates) to the file"
                )
            gaps = np.flatnonzero(np.isnan(row[context_end : context_end + horizon]))
            if gaps.size:
                raise InputError(
                    f"known covariate {name!r} has no value for {self.labels[context_end + gaps[0]]}; "
                    "known covariates must be filled for every forecast step (dates inserted by fill_missing_dates "
                    "only get values from a covariates file that has them)"
                )

        def stack(rows: dict[str, np.ndarray], length: int, dtype: type) -> np.ndarray | None:
            if not rows:
                return None
            return np.stack([row[:length] for row in rows.values()]).astype(dtype)[None]

        def split(rows: dict[str, np.ndarray]) -> tuple[dict, dict]:
            return (
                {k: v for k, v in rows.items() if k not in self.categorical},
                {k: v for k, v in rows.items() if k in self.categorical},
            )

        num_past, cat_past = split(self.past)
        num_known, cat_known = split(self.known)
        arrays = {
            "past_covariates": stack(num_past, context_end, np.float32),
            "known_covariates": stack(num_known, context_end + horizon, np.float32),
            # float64 so that distinct codes stay distinct, as upstream does
            "categorical_past_covariates": stack(cat_past, context_end, np.float64),
            "categorical_known_covariates": stack(cat_known, context_end + horizon, np.float64),
        }
        return {k: v for k, v in arrays.items() if v is not None}

    def with_future(self, overrides: dict[str, np.ndarray], context_end: int) -> Covariates:
        """A copy where known covariates take the `overrides` values over the steps after `context_end`."""
        known = dict(self.known)
        for name, future in overrides.items():
            if name not in known:
                raise InputError(f"scenario column {name!r} is not a known covariate")
            if name in self.categorical:
                raise InputError(f"scenario column {name!r} is categorical; only numeric columns can be changed")
            row = known[name]
            if row.size < context_end + future.size:
                raise InputError(
                    f"known covariate {name!r} has {row.size - context_end} future row(s) in the file but the "
                    f"scenario covers {future.size}; add future rows (empty target, filled covariates) to the file"
                )
            row = row.copy()
            row[context_end : context_end + future.size] = future
            known[name] = row
        return Covariates(past=self.past, known=known, categorical=self.categorical, labels=self.labels)


def covariate_signature(covariates: Covariates | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """What series must share to be forecast in one batch: the covariate columns and which are categorical.

    A column can be numeric in one series and categorical in another (text in some rows only), which
    changes the model arguments it goes to.
    """
    if covariates is None:
        return (), ()
    return tuple(sorted({*covariates.past, *covariates.known})), tuple(sorted(covariates.categorical))


@dataclass(frozen=True)
class ForecastResult:
    model: str
    device: str
    horizon: int
    quantiles: dict[float, np.ndarray]  # level -> (horizon,) values
    input_length: int
    context_length: int  # observations the model actually read
    missing_values: int


def quantile_key(level: float) -> str:
    return f"{level:g}"


def validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise InputError(f"horizon must be an integer, got {horizon!r}")
    if horizon < 1:
        raise InputError(f"horizon must be at least 1, got {horizon}")
    if horizon > MAX_HORIZON:
        raise InputError(f"horizon must be at most {MAX_HORIZON}, got {horizon}")
    return horizon


def validate_quantiles(quantiles: Sequence[float] | None) -> tuple[float, ...]:
    """Map requested levels onto Timestral's native levels; no interpolation is done."""
    if quantiles is None:
        return DEFAULT_QUANTILES
    if len(quantiles) == 0:
        raise InputError("quantiles must not be empty")
    levels = []
    for q in quantiles:
        match = next((n for n in NATIVE_QUANTILES if math.isclose(q, n, abs_tol=1e-9)), None)
        if match is None:
            supported = ", ".join(quantile_key(n) for n in NATIVE_QUANTILES)
            raise InputError(f"quantile {q} is not produced by Timestral; supported levels: {supported}")
        levels.append(match)
    return tuple(sorted(set(levels)))


def validate_values(values: Sequence[float] | np.ndarray, allow_missing: bool = False) -> np.ndarray:
    """Return the series as float64; NaN marks a missing value only when `allow_missing`."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise InputError("values must be a list of numbers") from None
    if arr.ndim != 1:
        raise InputError(f"values must be a flat list of numbers, got an array of shape {arr.shape}")
    if arr.size == 0:
        raise InputError("values must not be empty")
    if arr.size > MAX_INPUT_LENGTH:
        raise InputError(f"at most {MAX_INPUT_LENGTH} values are accepted, got {arr.size}")
    if np.isinf(arr).any():
        raise InputError(f"values must be finite; found infinity at position {int(np.flatnonzero(np.isinf(arr))[0])}")
    nan = np.isnan(arr)
    if nan.any() and not allow_missing:
        raise InputError(f"values must be finite numbers; found NaN at position {int(np.flatnonzero(nan)[0])}")
    observed = int((~nan).sum())
    if observed < MIN_OBSERVATIONS:
        raise InputError(f"at least {MIN_OBSERVATIONS} observed values are required, got {observed}")
    return arr


def run_forecast(
    values: np.ndarray,
    horizon: int,
    model_id: str,
    device: str,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    covariates: Covariates | None = None,
) -> ForecastResult:
    """Forecast `horizon` steps after a validated univariate series (NaN = missing).

    `covariates`, when given, are aligned with `values` (see :class:`Covariates`). Uses the
    forecaster defaults: symmetric averaging, and clipping at 0 when every observed value
    is non-negative.
    """
    return run_forecast_batch([(values, covariates)], horizon, model_id, device, quantiles)[0]


def run_forecast_batch(
    items: Sequence[tuple[np.ndarray, Covariates | None]],
    horizon: int,
    model_id: str,
    device: str,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> list[ForecastResult]:
    """Forecast several validated series of the same length and covariate columns, `MAX_BATCH` at a time.

    Each series gets the forecast it would get alone: Timestral treats the series of a batch
    independently (clipping at 0 is decided per series), and equal lengths mean no padding.
    """
    if not items:
        return []
    if len({values.size for values, _ in items}) != 1:
        raise ValueError("a batch must hold series of the same length")
    forecaster = model_mod.get_forecaster(model_id, device)
    cov_args = [cov.window(values.size, horizon) if cov is not None else {} for values, cov in items]
    if len({tuple(sorted(args)) for args in cov_args}) != 1:
        raise ValueError("a batch must hold series with the same covariate columns")
    if cov_args[0] and not forecaster.supports_covariates:
        raise InputError(f"checkpoint {model_id} was not trained with covariates")
    if any(cov is not None and cov.known for _, cov in items) and not forecaster.supports_known_covariates:
        raise InputError(f"checkpoint {model_id} cannot condition on known-future covariates")
    levels = tuple(forecaster.quantile_levels)
    indices = []
    for q in quantiles:
        idx = next((i for i, n in enumerate(levels) if math.isclose(q, n, abs_tol=1e-9)), None)
        if idx is None:
            raise InputError(f"quantile {q} is not produced by checkpoint {model_id}; it provides {levels}")
        indices.append(idx)

    results = []
    for start in range(0, len(items), MAX_BATCH):
        chunk = items[start : start + MAX_BATCH]
        args = cov_args[start : start + MAX_BATCH]
        context = np.stack([values for values, _ in chunk]).astype(np.float32)
        stacked = {name: np.concatenate([a[name] for a in args], axis=0) for name in args[0]}
        with model_mod.inference_lock:
            out = forecaster.forecast(context, prediction_length=horizon, **stacked)
        # (series, variates=1, horizon, levels) -> (series, horizon, levels)
        q = out.quantiles[:, 0].detach().to("cpu").double().numpy()
        if q.shape != (len(chunk), horizon, len(levels)) or not np.isfinite(q).all():
            raise RuntimeError(f"Timestral returned an invalid forecast of shape {q.shape}")
        for row, (values, _) in zip(q, chunk, strict=True):
            results.append(
                ForecastResult(
                    model=model_id,
                    device=device,
                    horizon=horizon,
                    quantiles={level: row[:, i] for level, i in zip(quantiles, indices, strict=True)},
                    input_length=int(values.size),
                    context_length=int(min(values.size, MODEL_CONTEXT_LENGTH)),
                    missing_values=int(np.isnan(values).sum()),
                )
            )
    return results
