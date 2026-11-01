import math

import numpy as np
import pytest

from timestral_mcp.forecasting import (
    DEFAULT_QUANTILES,
    MAX_HORIZON,
    MAX_INPUT_LENGTH,
    MIN_OBSERVATIONS,
    Covariates,
    InputError,
    quantile_key,
    run_forecast,
    validate_horizon,
    validate_quantiles,
    validate_values,
)


def test_valid_series_is_float64():
    arr = validate_values([1, 2.5, 3, 4])
    assert arr.dtype == np.float64
    assert arr.tolist() == [1.0, 2.5, 3.0, 4.0]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "must not be empty"),
        ([1.0, float("nan"), 3.0, 4.0], "NaN at position 1"),
        ([1.0, 2.0, float("inf")], "infinity at position 2"),
        ([1.0, 2.0, float("-inf")], "infinity at position 2"),
        ([1.0, 2.0], f"at least {MIN_OBSERVATIONS}"),
        ([[1.0, 2.0], [3.0, 4.0]], "flat list"),
        (["a", "b", "c"], "list of numbers"),
    ],
)
def test_invalid_values(values, message):
    with pytest.raises(InputError, match=message):
        validate_values(values)


def test_too_many_values():
    with pytest.raises(InputError, match="at most"):
        validate_values(np.ones(MAX_INPUT_LENGTH + 1))


def test_missing_values_allowed_only_when_requested():
    arr = validate_values([1.0, float("nan"), 3.0, 4.0], allow_missing=True)
    assert np.isnan(arr[1])
    with pytest.raises(InputError, match="observed values"):
        validate_values([float("nan")] * 5 + [1.0], allow_missing=True)


@pytest.mark.parametrize("horizon", [0, -3, MAX_HORIZON + 1, 2.5, True, "7"])
def test_invalid_horizons(horizon):
    with pytest.raises(InputError):
        validate_horizon(horizon)


def test_valid_horizon():
    assert validate_horizon(1) == 1
    assert validate_horizon(MAX_HORIZON) == MAX_HORIZON


def test_quantiles_default_sorted_and_deduplicated():
    assert validate_quantiles(None) == DEFAULT_QUANTILES
    assert validate_quantiles([0.9, 0.1, 0.5, 0.9]) == (0.1, 0.5, 0.9)
    assert validate_quantiles([0.05, 0.95]) == (0.05, 0.95)


@pytest.mark.parametrize("q", [0.33, 0.0, 1.0, 0.975])
def test_unsupported_quantiles_rejected_not_interpolated(q):
    with pytest.raises(InputError, match="not produced by Timestral"):
        validate_quantiles([q])


def test_empty_quantiles_rejected():
    with pytest.raises(InputError):
        validate_quantiles([])


def test_quantile_key():
    assert [quantile_key(q) for q in (0.01, 0.1, 0.5, 0.95)] == ["0.01", "0.1", "0.5", "0.95"]


def test_run_forecast_maps_quantile_columns(fake_forecaster):
    values = validate_values([10.0, 11.0, 12.0, 20.0])
    result = run_forecast(values, 5, "m", "cpu", (0.1, 0.5, 0.9))
    assert result.horizon == 5
    assert set(result.quantiles) == {0.1, 0.5, 0.9}
    # The fake predicts last + 10 * (level - 0.5): each level must come from its own column.
    for level, vals in result.quantiles.items():
        assert vals.shape == (5,)
        assert np.allclose(vals, 20.0 + 10 * (level - 0.5), atol=1e-5)
    assert result.input_length == 4 and result.context_length == 4 and result.missing_values == 0
    context, horizon = fake_forecaster.calls[0]
    assert horizon == 5 and context.dtype == np.float32


def test_run_forecast_counts_missing(fake_forecaster):
    values = validate_values([1.0, float("nan"), 3.0, 4.0], allow_missing=True)
    assert run_forecast(values, 2, "m", "cpu").missing_values == 1


def test_run_forecast_rejects_levels_missing_from_checkpoint(fake_forecaster):
    fake_forecaster.quantile_levels = (0.1, 0.5, 0.9)
    with pytest.raises(InputError, match="not produced by checkpoint"):
        run_forecast(validate_values([1.0, 2.0, 3.0]), 2, "m", "cpu", (0.05,))


def test_run_forecast_rejects_non_finite_model_output(fake_forecaster, monkeypatch):
    original = fake_forecaster.forecast

    def broken(context, prediction_length):
        out = original(context, prediction_length)
        out.quantiles[0, 0, 0, 0] = math.inf
        return out

    monkeypatch.setattr(fake_forecaster, "forecast", broken)
    with pytest.raises(RuntimeError, match="invalid forecast"):
        run_forecast(validate_values([1.0, 2.0, 3.0]), 2, "m", "cpu")


def covariates(**overrides) -> Covariates:
    fields = {
        "past": {"temp": np.array([1.0, 2.0, 3.0, 4.0])},
        "known": {"promo": np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]), "day": np.array([0.0, 1, 2, 0, 1, 2])},
        "categorical": frozenset({"day"}),
        "labels": ("d1", "d2", "d3", "d4", "d5", "d6"),
    }
    return Covariates(**{**fields, **overrides})


def test_covariates_window_shapes_and_kinds():
    args = covariates().window(4, 2)
    assert set(args) == {"past_covariates", "known_covariates", "categorical_known_covariates"}
    assert args["past_covariates"].shape == (1, 1, 4) and args["past_covariates"].dtype == np.float32
    assert args["known_covariates"].tolist() == [[[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]]
    assert args["categorical_known_covariates"].dtype == np.float64
    # A shorter horizon uses only the rows it needs.
    assert covariates().window(4, 1)["known_covariates"].shape == (1, 1, 5)


def test_known_covariates_must_cover_the_horizon():
    with pytest.raises(InputError, match="values for 2 step"):
        covariates().window(4, 3)


def test_known_covariates_must_be_filled_over_the_horizon():
    cov = covariates(known={"promo": np.array([0.0, 1.0, np.nan, 0.0, 1.0, np.nan])}, categorical=frozenset())
    # A missing value in the history is fine; one in the horizon is not.
    with pytest.raises(InputError, match="no value for d6"):
        cov.window(4, 2)
    assert cov.window(4, 1)["known_covariates"].shape == (1, 1, 5)


def test_run_forecast_passes_covariates(fake_forecaster):
    run_forecast(validate_values([1.0, 2.0, 3.0, 4.0]), 2, "m", "cpu", covariates=covariates())
    (args,) = fake_forecaster.covariate_calls
    assert set(args) == {"past_covariates", "known_covariates", "categorical_known_covariates"}


def test_run_forecast_without_covariates_passes_none(fake_forecaster):
    run_forecast(validate_values([1.0, 2.0, 3.0]), 2, "m", "cpu")
    assert fake_forecaster.covariate_calls == [{}]


def test_checkpoint_without_covariate_support(fake_forecaster):
    fake_forecaster.supports_known_covariates = False
    with pytest.raises(InputError, match="known-future covariates"):
        run_forecast(validate_values([1.0, 2.0, 3.0, 4.0]), 2, "m", "cpu", covariates=covariates())
    fake_forecaster.supports_covariates = False
    with pytest.raises(InputError, match="not trained with covariates"):
        run_forecast(validate_values([1.0, 2.0, 3.0, 4.0]), 2, "m", "cpu", covariates=covariates(known={}))
