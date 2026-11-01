"""Joint (multivariate) forecasts of several target columns of one series."""

import numpy as np
import pytest
from mcp import Client

from tests.conftest import write_csv
from timestral_mcp import server
from timestral_mcp.forecasting import InputError, run_forecast_joint

pytestmark = pytest.mark.anyio


def error_text(result) -> str:
    assert result.is_error
    return " ".join(c.text for c in result.content)


def two_columns(path, n: int = 10, extra: str = ""):
    rows = "\n".join(f"2026-01-{d:02d},{d},{100 + d}{extra}" for d in range(1, n + 1))
    return write_csv(path, "date,sales,visits\n" + rows)


def test_joint_forecast_sends_every_variate_in_one_call(fake_forecaster):
    values = np.array([[1.0, 2, 3, 4], [10.0, 20, np.nan, 40]])
    results = run_forecast_joint(values, 2, "m", "cpu")
    assert len(fake_forecaster.calls) == 1
    context, horizon = fake_forecaster.calls[0]
    assert context.shape == (2, 4) and horizon == 2
    assert [r.quantiles[0.5].tolist() for r in results] == [[4.0, 4.0], [40.0, 40.0]]
    assert [r.missing_values for r in results] == [0, 1]


def test_joint_forecast_needs_known_covariate_support(fake_forecaster):
    from timestral_mcp.forecasting import Covariates

    fake_forecaster.supports_known_covariates = False
    cov = Covariates(past={}, known={"promo": np.zeros(6)}, categorical=frozenset(), labels=tuple("abcdef"))
    with pytest.raises(InputError, match="known-future"):
        run_forecast_joint(np.ones((2, 4)), 2, "m", "cpu", covariates=cov)


async def test_forecast_multivariate_call(fake_forecaster, data_dir):
    two_columns(data_dir / "shop.csv")
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "forecast_multivariate", {"file_path": "shop.csv", "target_columns": ["sales", "visits"], "horizon": 2}
        )
    assert not result.is_error, result.content
    out = result.structured_content
    assert out["target_columns"] == ["sales", "visits"]
    assert out["last_observation"] == "2026-01-10"
    assert [(r["variable"], r["date"], r["p50"]) for r in out["forecast"]] == [
        ("sales", "2026-01-11", 10.0),
        ("sales", "2026-01-12", 10.0),
        ("visits", "2026-01-11", 110.0),
        ("visits", "2026-01-12", 110.0),
    ]
    assert out["totals"]["visits"]["p50"] == 220.0
    assert out["holdout"] is None
    assert len(fake_forecaster.calls) == 1  # one joint call, not one per column
    assert fake_forecaster.calls[0][0].shape == (2, 10)


async def test_holdout_compares_joint_and_independent(fake_forecaster, data_dir):
    two_columns(data_dir / "shop.csv")
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "forecast_multivariate",
            {"file_path": "shop.csv", "target_columns": ["sales", "visits"], "horizon": 1, "compare_test_size": 3},
        )
    assert not result.is_error, result.content
    holdout = result.structured_content["holdout"]
    assert (holdout["test_start"], holdout["test_end"]) == ("2026-01-08", "2026-01-10")
    sales = holdout["variables"][0]
    # the test double predicts the last value (7) for days 8-10: errors 1, 2, 3
    assert sales["variable"] == "sales"
    assert sales["joint_mae"] == sales["independent_mae"] == 2.0
    assert sales["mae_improvement_percent"] == 0.0


async def test_known_covariates_are_shared_by_the_targets(fake_forecaster, data_dir):
    rows = "\n".join(f"2026-01-{d:02d},{d},{100 + d},{d % 2}" for d in range(1, 9))
    future = "\n2026-01-09,,,1\n2026-01-10,,,0"
    write_csv(data_dir / "shop.csv", "date,sales,visits,promo\n" + rows + future)
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "forecast_multivariate",
            {"file_path": "shop.csv", "target_columns": ["sales", "visits"], "known_covariates": ["promo"]},
        )
    assert not result.is_error, result.content
    out = result.structured_content
    assert out["horizon"] == 2
    assert fake_forecaster.covariate_calls[0]["known_covariates"].shape == (1, 1, 10)


async def test_series_column_picks_one_series(fake_forecaster, data_dir):
    rows = "\n".join(f"{s},2026-01-{d:02d},{d * s},{d}" for s in (1, 2) for d in range(1, 6))
    write_csv(data_dir / "stores.csv", "store,date,sales,visits\n" + rows)
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "forecast_multivariate",
            {
                "file_path": "stores.csv",
                "target_columns": ["sales", "visits"],
                "series_column": "store",
                "series_id": "2",
                "horizon": 1,
            },
        )
    assert not result.is_error, result.content
    out = result.structured_content
    assert out["series"] == "2"
    assert out["totals"]["sales"]["p50"] == 10.0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"target_columns": ["sales", "sales"]}, "more than once"),
        ({"target_columns": ["sales", "visits"], "past_covariates": ["visits"]}, "both a target and a covariate"),
        ({"target_columns": ["sales", "visits"], "series_id": "1"}, "series_id needs series_column"),
        ({"target_columns": ["sales", "visits"], "compare_test_size": 9}, "compare_test_size"),
    ],
)
async def test_invalid_calls(fake_forecaster, data_dir, args, message):
    two_columns(data_dir / "shop.csv")
    async with Client(server.mcp) as client:
        result = await client.call_tool("forecast_multivariate", {"file_path": "shop.csv", "horizon": 1, **args})
    assert message in error_text(result)


async def test_columns_must_end_on_the_same_date(fake_forecaster, data_dir):
    rows = "2026-01-01,1,10\n2026-01-02,2,20\n2026-01-03,3,30\n2026-01-04,4,"
    write_csv(data_dir / "shop.csv", "date,sales,visits\n" + rows)
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "forecast_multivariate", {"file_path": "shop.csv", "target_columns": ["sales", "visits"], "horizon": 1}
        )
    assert "do not cover the same dates" in error_text(result)
