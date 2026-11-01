"""Protocol-level tests through the official MCP client (in memory and over STDIO)."""

import json
import os
import subprocess
import sys

import pytest
from mcp import Client, StdioServerParameters

from tests.conftest import daily_csv
from timestral_mcp import server

pytestmark = pytest.mark.anyio

TOOLS = {"forecast", "forecast_csv", "backtest", "compare_scenarios", "forecast_multivariate"}


def error_text(result) -> str:
    assert result.is_error
    return " ".join(c.text for c in result.content)


async def test_tools_are_discoverable():
    async with Client(server.mcp) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == TOOLS
    for tool in tools.values():
        assert tool.description
        assert tool.output_schema is not None
        assert tool.annotations.destructive_hint is not True
    # the tools that can write an output_file are not read-only; the others are
    assert {name for name, tool in tools.items() if tool.annotations.read_only_hint} == {
        "forecast",
        "compare_scenarios",
        "forecast_multivariate",
    }
    assert tools["forecast"].input_schema["required"] == ["values", "horizon"]
    # file_path or csv_text: neither is required on its own
    assert "required" not in tools["forecast_csv"].input_schema
    assert tools["backtest"].input_schema["required"] == ["test_size"]
    assert tools["compare_scenarios"].input_schema["required"] == ["scenario"]
    for name in ("forecast_csv", "backtest", "compare_scenarios"):
        properties = tools[name].input_schema["properties"]
        assert {"file_path", "csv_text", "aggregate_by"} <= set(properties)
        assert "ctx" not in properties


async def test_forecast_call(fake_forecaster, data_dir):
    async with Client(server.mcp) as client:
        result = await client.call_tool("forecast", {"values": [1, 2, 3, 4], "horizon": 2, "quantiles": [0.05, 0.5]})
    assert not result.is_error
    out = result.structured_content
    assert out["model"] == "mistralai/Timestral-28M"
    assert out["horizon"] == 2
    assert set(out["quantiles"]) == {"0.05", "0.5"}
    assert out["quantiles"]["0.5"] == [4.0, 4.0]
    assert out["quantiles"]["0.05"] == pytest.approx([-0.5, -0.5])
    assert out["total"] == {"0.05": pytest.approx(-1.0), "0.5": 8.0}
    assert out["metadata"] == {
        "input_length": 4, "context_length": 4, "missing_values": 0, "filled_timestamps": 0, "device": "cpu",
    }  # fmt: skip
    json.dumps(out)  # serializable


async def test_forecast_csv_call(fake_forecaster, data_dir):
    daily_csv(data_dir / "sales.csv", [10.0, 11.0, 12.0, 13.0], start="2026-10-28")
    async with Client(server.mcp) as client:
        result = await client.call_tool("forecast_csv", {"file_path": "sales.csv", "horizon": 3})
    assert not result.is_error, result.content
    out = result.structured_content
    assert (out["date_column"], out["target_column"], out["frequency"]) == ("date", "sales", "daily")
    assert out["last_observation"] == "2026-10-31"
    assert [r["date"] for r in out["forecast"]] == ["2026-11-01", "2026-11-02", "2026-11-03"]
    assert set(out["forecast"][0]) == {"date", "p10", "p50", "p90"}
    assert out["forecast"][0]["p50"] == 13.0


async def test_backtest_call(fake_forecaster, data_dir):
    daily_csv(data_dir / "sales.csv", [float(v) for v in range(1, 21)])
    async with Client(server.mcp) as client:
        result = await client.call_tool("backtest", {"file_path": "sales.csv", "test_size": 5, "seasonal_period": 7})
    assert not result.is_error, result.content
    out = result.structured_content
    ev = out["evaluation"]
    assert ev["method"] == "single_chronological_holdout"
    assert (ev["test_size"], ev["windows"], ev["step"]) == (5, 1, 5)
    assert (ev["test_start"], ev["test_end"]) == ("2026-01-16", "2026-01-20")
    assert ev["mae"] == pytest.approx(3.0)
    assert [b["method"] for b in out["baselines"]] == ["naive", "seasonal_naive"]
    assert out["windows"][0]["context_size"] == 15
    assert len(out["predictions"]) == 5
    assert out["predictions"][0] == {
        "window": 1, "date": "2026-01-16", "actual": 16.0, "p10": 11.0, "p50": 15.0, "p90": 19.0,
    }  # fmt: skip
    assert out["covariates"] == {"past": [], "known": [], "categorical": [], "files": [], "future_rows": 0}
    assert out["assumed_known_in_advance"] == []


async def test_rolling_backtest_call(fake_forecaster, data_dir):
    daily_csv(data_dir / "sales.csv", [float(v) for v in range(1, 31)])
    async with Client(server.mcp) as client:
        result = await client.call_tool("backtest", {"file_path": "sales.csv", "test_size": 5, "windows": 3})
    assert not result.is_error, result.content
    out = result.structured_content
    assert out["evaluation"]["method"] == "rolling_origin"
    assert (out["evaluation"]["test_start"], out["evaluation"]["test_end"]) == ("2026-01-16", "2026-01-30")
    assert [w["context_size"] for w in out["windows"]] == [15, 20, 25]
    assert [w["test_start"] for w in out["windows"]] == ["2026-01-16", "2026-01-21", "2026-01-26"]
    assert len(out["predictions"]) == 15
    assert [r["window"] for r in out["predictions"]] == [1] * 5 + [2] * 5 + [3] * 5
    assert out["metadata"]["input_length"] == 25  # most recent window


async def test_forecast_csv_with_covariates(fake_forecaster, data_dir):
    (data_dir / "promo.csv").write_text(
        "date,sales,promo,weather\n"
        "2026-01-01,10,0,3.5\n2026-01-02,12,1,4.0\n2026-01-03,11,0,\n2026-01-04,,1,\n2026-01-05,,0,\n"
    )
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "forecast_csv",
            {"file_path": "promo.csv", "horizon": 2, "known_covariates": ["promo"], "past_covariates": ["weather"]},
        )
        assert not result.is_error, result.content
        out = result.structured_content
        assert out["covariates"] == {
            "past": ["weather"],
            "known": ["promo"],
            "categorical": [],
            "files": [],
            "future_rows": 2,
        }
        assert out["last_observation"] == "2026-01-03"
        assert [r["date"] for r in out["forecast"]] == ["2026-01-04", "2026-01-05"]
        (args,) = fake_forecaster.covariate_calls
        assert args["known_covariates"].tolist() == [[[0.0, 1.0, 0.0, 1.0, 0.0]]]
        assert args["past_covariates"].shape == (1, 1, 3)

        result = await client.call_tool(
            "forecast_csv", {"file_path": "promo.csv", "horizon": 3, "known_covariates": ["promo"]}
        )
        assert "values for 2 step(s) after the history, but the horizon is 3" in error_text(result)


@pytest.mark.parametrize(
    ("tool", "args", "message"),
    [
        ("forecast", {"values": [], "horizon": 3}, "must not be empty"),
        ("forecast", {"values": [1, 2, 3], "horizon": 3, "quantiles": [0.42]}, "not produced by Timestral"),
        ("forecast", {"values": [1, 2, 3], "horizon": 3, "model": "evil/repo"}, "not allowed"),
        ("forecast_csv", {"file_path": "missing.csv", "horizon": 3}, "file not found"),
        ("forecast_csv", {"file_path": "../outside.csv", "horizon": 3}, "access denied"),
        ("backtest", {"file_path": "sales.csv", "test_size": 30}, "leave 0 observation"),
        ("backtest", {"file_path": "sales.csv", "test_size": 1, "windows": 3}, "before the first window"),
        ("forecast_csv", {"file_path": "sales.csv", "horizon": 1, "known_covariates": ["promo"]}, "not found"),
    ],
)
async def test_tool_errors_are_readable(fake_forecaster, data_dir, tool, args, message):
    daily_csv(data_dir / "sales.csv", [1.0, 2.0, 3.0, 4.0])
    async with Client(server.mcp) as client:
        result = await client.call_tool(tool, args)
    assert message in error_text(result)
    assert result.structured_content is None


@pytest.mark.parametrize(
    "args", [{"values": [1, 2, 3], "horizon": 0}, {"values": [1, 2, 3]}, {"values": "1,2,3", "horizon": 2}]
)
async def test_schema_validation_rejects_bad_arguments(fake_forecaster, args):
    async with Client(server.mcp) as client:
        result = await client.call_tool("forecast", args)
    assert result.is_error
    assert fake_forecaster.calls == []


async def test_unexpected_errors_do_not_leak_internals(data_dir, monkeypatch):
    def boom(model_id, device):
        raise KeyError("internal secret detail")

    monkeypatch.setattr(server.model_mod, "get_forecaster", boom)
    async with Client(server.mcp) as client:
        result = await client.call_tool("forecast", {"values": [1, 2, 3], "horizon": 2})
    assert "internal secret detail" not in error_text(result)


async def test_stdio_server_starts_lists_tools_and_keeps_stdout_clean(tmp_path):
    # Runs the real entry point in a subprocess; no tool call here needs the model.
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "timestral_mcp.server"],
        env={"TIMESTRAL_ALLOWED_DIRS": str(tmp_path)},
    )
    async with Client(params) as client:
        assert {t.name for t in (await client.list_tools()).tools} == TOOLS
        result = await client.call_tool("forecast", {"values": [1, float("nan")], "horizon": 1})
        assert result.is_error
        result = await client.call_tool("forecast_csv", {"file_path": "none.csv", "horizon": 1})
        assert "file not found" in error_text(result)


def test_invalid_configuration_exits_with_message():
    env = {**os.environ, "TIMESTRAL_DEVICE": "tpu"}
    proc = subprocess.run(
        [sys.executable, "-m", "timestral_mcp.server"], env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "TIMESTRAL_DEVICE" in proc.stderr


async def test_backtest_reports_known_covariates_as_an_assumption(fake_forecaster, data_dir):
    rows = "".join(f"2026-01-{d:02d},{d},{d % 2},{d * 1.5}\n" for d in range(1, 21))
    (data_dir / "promo.csv").write_text("date,sales,promo,weather\n" + rows)
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "backtest",
            {"file_path": "promo.csv", "test_size": 3, "known_covariates": ["promo"], "past_covariates": ["weather"]},
        )
    assert not result.is_error, result.content
    # Only the known covariates are an assumption; past covariates are cut at each origin.
    assert result.structured_content["assumed_known_in_advance"] == ["promo"]


async def test_covariate_guidance_reaches_the_client():
    async with Client(server.mcp) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        instructions = client.instructions
    assert "ask the user" in instructions and "assumed_known_in_advance" in instructions
    for name in ("forecast_csv", "backtest"):
        props = tools[name].input_schema["properties"]
        known = props["known_covariates"]["description"]
        assert "decided or published in advance" in known and "ask the user" in known
        assert "when in doubt" in props["past_covariates"]["description"].lower()
