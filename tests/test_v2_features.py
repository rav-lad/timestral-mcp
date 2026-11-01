"""File formats, extra frequencies, gap filling, several series per file, totals and scenarios."""

import numpy as np
import pandas as pd
import pytest
from mcp import Client

from tests.conftest import daily_csv, write_csv
from timestral_mcp import server
from timestral_mcp.backtesting import pool, run_backtest
from timestral_mcp.forecasting import Covariates, InputError
from timestral_mcp.preprocessing import (
    future_dates,
    infer_frequency,
    prepare_series,
    read_table,
    split_series,
)

# ---------------------------------------------------------------- file formats


def test_semicolon_csv_with_decimal_commas_and_day_first_dates(tmp_path):
    path = tmp_path / "fr.csv"
    path.write_bytes(
        "date;ventes;météo\n30/10/2026;1 234,5;doux\n31/10/2026;-3,25;froid\n01/11/2026;7;doux\n".encode("cp1252")
    )
    df = read_table(path)
    assert df["ventes"].tolist() == [1234.5, -3.25, 7.0]
    s = prepare_series(df)
    assert s.target_column == "ventes"
    assert s.dates.strftime("%Y-%m-%d").tolist() == ["2026-10-30", "2026-10-31", "2026-11-01"]


def test_ambiguous_slashed_dates_follow_the_file_convention(tmp_path):
    rows = "\n".join(f"0{d}/11/2026{sep}{v}" for d, v in ((1, 1), (2, 2), (3, 3)) for sep in ";")
    path = write_csv(tmp_path / "fr.csv", "date;sales\n" + rows)
    assert prepare_series(read_table(path)).dates[0] == pd.Timestamp("2026-11-01")  # day first, not January 11
    path = write_csv(tmp_path / "us.csv", "date,sales\n11/01/2026,1\n11/02/2026,2\n11/03/2026,3")
    assert prepare_series(read_table(path)).dates[0] == pd.Timestamp("2026-11-01")  # month first


def test_iso_dates_are_never_reordered(tmp_path):
    path = write_csv(tmp_path / "iso.csv", "date;sales\n2026-11-01;1\n2026-11-02;2\n2026-11-03;3")
    assert prepare_series(read_table(path)).dates[0] == pd.Timestamp("2026-11-01")


def test_thousands_with_commas_are_not_reinterpreted_in_comma_files(tmp_path):
    path = write_csv(tmp_path / "s.csv", 'date,sales\n2026-01-01,"1,000"\n2026-01-02,"2,000"')
    assert read_table(path)["sales"].tolist() == ["1,000", "2,000"]


def test_excel_first_sheet_and_named_sheet(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3), "sales": [1.0, 2, 3]}).to_excel(
            writer, sheet_name="first", index=False
        )
        pd.DataFrame({"date": pd.date_range("2026-02-01", periods=3), "sales": [4.0, 5, 6]}).to_excel(
            writer, sheet_name="second", index=False
        )
    assert prepare_series(read_table(path)).values.tolist() == [1, 2, 3]
    assert prepare_series(read_table(path, "second")).values.tolist() == [4, 5, 6]
    with pytest.raises(InputError, match=r"sheet 'nope' not found.*\['first', 'second'\]"):
        read_table(path, "nope")
    with pytest.raises(InputError, match="only used for Excel"):
        read_table(daily_csv(tmp_path / "s.csv", [1.0, 2, 3]), "first")


# ---------------------------------------------------------------- frequencies


def idx(*values: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(values)))


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (("2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 00:30"), "15min"),
        (("2026-01-01 00:00", "2026-01-01 00:30", "2026-01-01 01:00"), "30min"),
        (("2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03"), "business_daily"),
        (("2026-01-01", "2026-04-01", "2026-07-01"), "quarterly"),
        (("2026-03-31", "2026-06-30", "2026-09-30"), "quarterly"),
        (("2024-01-01", "2025-01-01", "2026-01-01"), "yearly"),
    ],
)
def test_infers_new_frequencies(values, expected):
    assert infer_frequency(idx(*values)) == expected


def test_contiguous_weekdays_are_daily():
    assert infer_frequency(idx("2026-10-26", "2026-10-27", "2026-10-28")) == "daily"


def test_business_daily_future_dates_skip_weekends(tmp_path):
    path = write_csv(tmp_path / "b.csv", "date,sales\n2026-10-28,1\n2026-10-29,2\n2026-10-30,3\n2026-11-02,4")
    s = prepare_series(read_table(path))
    assert s.frequency == "business_daily"
    assert future_dates(s, 3) == ["2026-11-03", "2026-11-04", "2026-11-05"]


# ---------------------------------------------------------------- gap filling


def gappy(tmp_path):
    return write_csv(
        tmp_path / "g.csv", "date,sales,promo\n2026-01-01,1,0\n2026-01-02,2,1\n2026-01-05,5,0\n2026-01-06,6,0"
    )


def test_gaps_are_rejected_with_a_hint(tmp_path):
    with pytest.raises(InputError, match=r"2 missing timestamp.*fill_missing_dates=true"):
        prepare_series(read_table(gappy(tmp_path)), frequency="daily")


def test_gaps_are_filled_as_missing_values(tmp_path):
    s = prepare_series(
        read_table(gappy(tmp_path)), frequency="daily", past_covariates=["promo"], fill_missing_dates=True
    )
    assert s.dates.strftime("%m-%d").tolist() == ["01-01", "01-02", "01-03", "01-04", "01-05", "01-06"]
    np.testing.assert_array_equal(s.values, [1, 2, np.nan, np.nan, 5, 6])
    np.testing.assert_array_equal(s.covariates.past["promo"], [0, 1, np.nan, np.nan, 0, 0])
    assert s.filled_timestamps == 2


def test_fill_infers_the_frequency_from_the_median_step(tmp_path):
    rows = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05", "2026-01-06"]
    path = write_csv(tmp_path / "s.csv", "date,sales\n" + "\n".join(f"{d},{i}" for i, d in enumerate(rows)))
    s = prepare_series(read_table(path), fill_missing_dates=True)
    assert (s.frequency, s.filled_timestamps, len(s.dates)) == ("daily", 1, 6)


def test_fill_cannot_repair_off_grid_timestamps(tmp_path):
    rows = "2026-01-01 00:00,1\n2026-01-02 00:00,2\n2026-01-02 12:00,3\n2026-01-04 00:00,4"
    path = write_csv(tmp_path / "s.csv", "date,sales\n" + rows)
    with pytest.raises(InputError, match="off the daily grid"):
        prepare_series(read_table(path), frequency="daily", fill_missing_dates=True)


def test_complete_series_are_unchanged_by_fill(tmp_path):
    df = read_table(daily_csv(tmp_path / "s.csv", [1.0, 2, 3, 4]))
    assert prepare_series(df, fill_missing_dates=True).values.tolist() == prepare_series(df).values.tolist()


# ---------------------------------------------------------------- several series


LONG = """
date,store,sales,promo
2026-01-01,12,10,0
2026-01-01,B,100,0
2026-01-02,12,11,1
2026-01-02,B,101,0
2026-01-03,12,12,0
2026-01-03,B,102,1
2026-01-04,12,,0
2026-01-04,B,,1
"""


def test_split_series_keeps_order_ids_and_file_lines(tmp_path):
    df = read_table(write_csv(tmp_path / "long.csv", LONG))
    groups = split_series(df, ["store"])
    assert [g.label for g in groups] == ["12", "B"]
    assert groups[0].keys == {"store": "12"}
    s = prepare_series(groups[1].frame, exclude=["store"])
    assert (s.target_column, s.values.tolist(), s.future_rows) == ("sales", [100, 101, 102], 1)
    assert [g.label for g in split_series(df, ["store"], ["B"])] == ["B"]
    with pytest.raises(InputError, match=r"series \['Z'\] not found.*\['12', 'B'\]"):
        split_series(df, ["store"], ["Z"])
    bad = read_table(write_csv(tmp_path / "bad.csv", LONG.replace("2026-01-02,B,101", "2026-01-02,B,oops")))
    with pytest.raises(InputError, match=r"lines \[5\]"):
        prepare_series(split_series(bad, ["store"])[1].frame, target_column="sales", exclude=["store"])


def test_pool_averages_equal_sized_backtests(fake_forecaster):
    a = run_backtest(np.arange(1.0, 21.0), 5, "m", "cpu", windows=2)
    b = run_backtest(np.arange(1.0, 21.0) * 3, 5, "m", "cpu", windows=2)
    pooled = pool([a, b])
    assert pooled.mae == pytest.approx((a.mae + b.mae) / 2)
    assert pooled.rmse == pytest.approx(np.sqrt((a.rmse**2 + b.rmse**2) / 2))
    assert len(pooled.windows) == 4
    assert pooled.baselines[0].mae == pytest.approx((a.baselines[0].mae + b.baselines[0].mae) / 2)


def test_with_future_replaces_only_the_horizon():
    cov = Covariates(past={}, known={"promo": np.array([1.0, 1, 1, 1, 1])}, categorical=frozenset(), labels=())
    changed = cov.with_future({"promo": np.array([0.0, 0])}, 3)
    assert changed.known["promo"].tolist() == [1, 1, 1, 0, 0]
    assert cov.known["promo"].tolist() == [1, 1, 1, 1, 1]
    with pytest.raises(InputError, match="has 2 future row"):
        cov.with_future({"promo": np.zeros(3)}, 3)
    with pytest.raises(InputError, match="not a known covariate"):
        cov.with_future({"price": np.zeros(1)}, 3)


# ---------------------------------------------------------------- tools


@pytest.mark.anyio
async def test_forecast_csv_several_series(fake_forecaster, data_dir):
    write_csv(data_dir / "long.csv", LONG)
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "forecast_csv",
            {"file_path": "long.csv", "horizon": 1, "series_column": "store", "known_covariates": ["promo"]},
        )
    assert not result.is_error, result.content
    out = result.structured_content
    assert out["series_columns"] == ["store"]
    assert out["forecast"] == [
        {"series": "12", "date": "2026-01-04", "p10": 8.0, "p50": 12.0, "p90": 16.0},
        {"series": "B", "date": "2026-01-04", "p10": 98.0, "p50": 102.0, "p90": 106.0},
    ]
    assert [s["series"] for s in out["series"]] == ["12", "B"]
    assert out["series"][1]["total"] == {"p10": 98.0, "p50": 102.0, "p90": 106.0}
    assert out["total"] is None and out["metadata"]["input_length"] == 3  # shared by both series, given once
    assert all(s["metadata"] is None for s in out["series"])
    assert out["covariates"]["known"] == ["promo"]


@pytest.mark.anyio
async def test_forecast_csv_total_and_filled_gaps(fake_forecaster, data_dir):
    write_csv(data_dir / "g.csv", "date,sales\n2026-01-01,1\n2026-01-02,2\n2026-01-04,4\n2026-01-05,5")
    async with Client(server.mcp) as client:
        rejected = await client.call_tool("forecast_csv", {"file_path": "g.csv", "horizon": 3})
        result = await client.call_tool(
            "forecast_csv", {"file_path": "g.csv", "horizon": 3, "fill_missing_dates": True}
        )
    assert rejected.is_error and "fill_missing_dates" in rejected.content[0].text
    out = result.structured_content
    assert out["total"] == {"p10": 3.0, "p50": 15.0, "p90": 27.0}
    assert (out["metadata"]["filled_timestamps"], out["metadata"]["missing_values"]) == (1, 1)


@pytest.mark.anyio
async def test_backtest_several_series(fake_forecaster, data_dir):
    rows = [f"2026-01-{d:02d},{s},{d * (1 if s == 'A' else 10)}" for d in range(1, 21) for s in "AB"]
    write_csv(data_dir / "long.csv", "date,store,sales\n" + "\n".join(rows))
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "backtest", {"file_path": "long.csv", "test_size": 5, "windows": 2, "series_column": "store"}
        )
    assert not result.is_error, result.content
    out = result.structured_content
    a, b = out["series_results"]
    assert (a["series"], b["series"]) == ("A", "B")
    assert out["evaluation"]["mae"] == pytest.approx((a["evaluation"]["mae"] + b["evaluation"]["mae"]) / 2)
    assert out["evaluation"]["windows"] == 2
    assert len(out["windows"]) == 4 and out["windows"][2]["series"] == "B"
    assert out["predictions"][0]["series"] == "A" and len(out["predictions"]) == 20
    assert out["metadata"] is None


@pytest.mark.anyio
async def test_compare_scenarios(fake_forecaster, data_dir):
    write_csv(
        data_dir / "p.csv",
        "date,sales,promo\n2026-01-01,1,0\n2026-01-02,2,1\n2026-01-03,3,0\n2026-01-04,,1\n2026-01-05,,0",
    )
    async with Client(server.mcp) as client:
        result = await client.call_tool(
            "compare_scenarios", {"file_path": "p.csv", "horizon": 2, "scenario": {"promo": 0}}
        )
        too_short = await client.call_tool(
            "compare_scenarios", {"file_path": "p.csv", "horizon": 2, "scenario": {"promo": [0]}}
        )
        clash = await client.call_tool(
            "compare_scenarios",
            {"file_path": "p.csv", "horizon": 2, "scenario": {"promo": 0}, "past_covariates": ["promo"]},
        )
    assert not result.is_error, result.content
    out = result.structured_content
    assert out["scenario"] == {"promo": [0.0, 0.0]}
    assert out["covariates"]["known"] == ["promo"]
    assert set(out["forecast"][0]) == {
        "date", "base_p10", "base_p50", "base_p90", "scenario_p10", "scenario_p50", "scenario_p90", "delta_p50",
    }  # fmt: skip
    assert out["delta_total_p50"] == 0.0  # the test double ignores covariates
    base_call, scenario_call = fake_forecaster.covariate_calls
    assert base_call["known_covariates"][0, 0].tolist() == [0, 1, 0, 1, 0]
    assert scenario_call["known_covariates"][0, 0].tolist() == [0, 1, 0, 0, 0]
    assert too_short.is_error and "needs one per step (2)" in too_short.content[0].text
    assert clash.is_error and "are past covariates" in clash.content[0].text


@pytest.mark.anyio
async def test_compare_scenarios_needs_one_series(fake_forecaster, data_dir):
    write_csv(data_dir / "long.csv", LONG)
    async with Client(server.mcp) as client:
        missing = await client.call_tool(
            "compare_scenarios",
            {"file_path": "long.csv", "horizon": 1, "scenario": {"promo": 0}, "series_column": "store"},
        )
        ok = await client.call_tool(
            "compare_scenarios",
            {
                "file_path": "long.csv",
                "horizon": 1,
                "scenario": {"promo": 0},
                "series_column": "store",
                "series_id": "B",
            },
        )
    assert missing.is_error and "pass series_id" in missing.content[0].text
    assert not ok.is_error, ok.content
    assert ok.structured_content["series"] == "B"
