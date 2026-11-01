"""Several series columns, covariate and future files, output files, skipped series and batching."""

import numpy as np
import pandas as pd
import pytest
from mcp import Client

from tests.conftest import write_csv
from timestral_mcp import server
from timestral_mcp.backtesting import run_backtest, run_backtests
from timestral_mcp.forecasting import InputError, run_forecast, run_forecast_batch

pytestmark = pytest.mark.anyio

WEEKS = pd.date_range("2026-01-02", periods=8, freq="7D").strftime("%Y-%m-%d")


def long_file(path, stores=("1", "2"), depts=("10", "20"), weeks=WEEKS, skip=()):
    rows = [
        f"{s},{d},{w},{100 * int(s) + int(d) + i},{int(i == 3)}"
        for i, w in enumerate(weeks)
        for s in stores
        for d in depts
        if (s, d, w) not in skip
    ]
    return write_csv(path, "Store,Dept,Date,Sales,Holiday\n" + "\n".join(rows))


async def call(name, args):
    async with Client(server.mcp) as client:
        return await client.call_tool(name, args)


def text(result) -> str:
    return " ".join(c.text for c in result.content)


# ---------------------------------------------------------------- several series columns


async def test_two_series_columns_and_totals_by_first_column(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    result = await call("forecast_csv", {"file_path": "train.csv", "horizon": 2, "series_column": ["Store", "Dept"]})
    assert not result.is_error, text(result)
    out = result.structured_content
    assert out["series_columns"] == ["Store", "Dept"]
    assert [s["series"] for s in out["series"]] == ["1/10", "1/20", "2/10", "2/20"]
    assert out["forecast"][0] == {"series": "1/10", "date": "2026-02-27", "p10": 113.0, "p50": 117.0, "p90": 121.0}
    assert out["totals_by"]["column"] == "Store"
    assert out["totals_by"]["totals"]["1"]["p50"] == 2 * (117 + 127)
    assert out["overall_total"]["p50"] == 2 * (117 + 127 + 217 + 227)


async def test_series_ids_with_two_columns(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    ok = await call(
        "forecast_csv",
        {"file_path": "train.csv", "horizon": 1, "series_column": ["Store", "Dept"], "series_ids": ["2/20"]},
    )
    bad = await call(
        "forecast_csv",
        {"file_path": "train.csv", "horizon": 1, "series_column": ["Store", "Dept"], "series_ids": ["2-20"]},
    )
    assert [s["series"] for s in ok.structured_content["series"]] == ["2/20"]
    assert bad.is_error and "ids join Store, Dept with '/'" in text(bad)


async def test_unusable_series_are_skipped_with_their_reason(fake_forecaster, data_dir):
    skip = {("2", "20", w) for w in WEEKS[1:]}  # one observation left
    long_file(data_dir / "train.csv", skip=skip)
    result = await call("forecast_csv", {"file_path": "train.csv", "horizon": 1, "series_column": ["Store", "Dept"]})
    out = result.structured_content
    assert out["series_count"] == 3 and out["skipped_count"] == 1
    assert out["skipped"][0]["series"] == "2/20"
    assert "at least" in out["skipped"][0]["reason"]


async def test_every_series_unusable_is_an_error(fake_forecaster, data_dir):
    write_csv(data_dir / "s.csv", "store,date,sales\nA,2026-01-01,1\nB,2026-01-01,2")
    result = await call(
        "forecast_csv", {"file_path": "s.csv", "horizon": 1, "series_column": "store", "frequency": "daily"}
    )
    assert result.is_error and "none of the 2 series could be used" in text(result)


async def test_frequency_is_inferred_for_the_whole_file(fake_forecaster, data_dir):
    # store 2 has three rows four weeks apart: on its own it would look monthly
    weeks = pd.date_range("2026-01-02", periods=9, freq="7D").strftime("%Y-%m-%d")
    skip = {("2", "10", w) for i, w in enumerate(weeks) if i % 4}
    long_file(data_dir / "train.csv", depts=("10",), weeks=weeks, skip=skip)
    result = await call(
        "forecast_csv",
        {
            "file_path": "train.csv",
            "horizon": 1,
            "series_column": "Store",
            "target_column": "Sales",
            "fill_missing_dates": True,
        },
    )
    out = result.structured_content
    assert out["frequency"] == "weekly" and out["skipped_count"] == 0
    assert [s["metadata"]["filled_timestamps"] for s in out["series"]] == [0, 6]


# ---------------------------------------------------------------- covariate and future files


async def test_covariates_file_joined_on_shared_series_column_and_date(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    later = pd.date_range("2026-01-02", periods=10, freq="7D").strftime("%Y-%m-%d")
    rows = [f"{s},{w},{20 + i + int(s)},{int(i == 8)}" for i, w in enumerate(later) for s in ("1", "2")]
    write_csv(data_dir / "features.csv", "Store,Date,Temp,Holiday\n" + "\n".join(rows))
    result = await call(
        "forecast_csv",
        {
            "file_path": "train.csv",
            "series_column": ["Store", "Dept"],
            "target_column": "Sales",
            "past_covariates": ["Temp"],
            "known_covariates": ["Holiday"],
            "covariates_files": ["features.csv"],
        },
    )
    assert not result.is_error, text(result)
    out = result.structured_content
    assert out["covariates"]["files"] == ["features.csv"]
    assert out["horizon"] is None and out["series"][0]["horizon"] == 2  # the two later dates of features.csv
    first = fake_forecaster.covariate_calls[0]
    # Holiday comes from train.csv where it has values, from features.csv after
    assert first["known_covariates"][0, 0].tolist() == [0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    assert first["past_covariates"][0, 0].tolist() == [21, 22, 23, 24, 25, 26, 27, 28]


async def test_covariates_file_without_series_columns_applies_to_every_series(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    write_csv(data_dir / "calendar.csv", "Date,Event\n" + "\n".join(f"{w},{i % 2}" for i, w in enumerate(WEEKS)))
    result = await call(
        "forecast_csv",
        {"file_path": "train.csv", "horizon": 1, "series_column": ["Store", "Dept"], "target_column": "Sales",
         "past_covariates": ["Event"], "covariates_files": ["calendar.csv"]},
    )  # fmt: skip
    assert not result.is_error, text(result)
    assert all(c["past_covariates"][0, 0].tolist() == [0, 1] * 4 for c in fake_forecaster.covariate_calls)


async def test_covariates_file_errors(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    write_csv(data_dir / "stores.csv", "Store,Size\n1,100\n2,200")
    write_csv(data_dir / "dupes.csv", "Store,Date,Temp\n1,2026-01-02,1\n1,2026-01-02,2")
    base = {"file_path": "train.csv", "horizon": 1, "series_column": ["Store", "Dept"], "target_column": "Sales"}
    static = await call("forecast_csv", {**base, "past_covariates": ["Size"], "covariates_files": ["stores.csv"]})
    dupes = await call("forecast_csv", {**base, "past_covariates": ["Temp"], "covariates_files": ["dupes.csv"]})
    unnamed = await call("forecast_csv", {**base, "covariates_files": ["dupes.csv"]})
    # a file without dates is ignored with a warning instead of failing the forecast
    assert not static.is_error, text(static)
    out = static.structured_content
    assert out["covariates"]["past"] == [] and out["covariates"]["files"] == []
    assert len(out["warnings"]) == 1 and "stores.csv was ignored" in out["warnings"][0]
    assert "['Size']" in out["warnings"][0]
    assert dupes.is_error and "several rows on 2026-01-02 for Store = 1" in text(dupes)
    assert unnamed.is_error and "needs the columns to use" in text(unnamed)


async def test_skipped_series_are_flagged_in_warnings(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    # a known covariate filled in the future for store 1 only: store 2 is skipped
    rows = [f"{s},{w},{int(s == '1' or w != '2026-02-27')}" for s in ("1", "2") for w in [*WEEKS, "2026-02-27"]]
    rows = [r for r in rows if not (r.startswith("2,2026-02-27"))]
    write_csv(data_dir / "promo.csv", "Store,Date,Promo\n" + "\n".join(rows))
    result = await call(
        "forecast_csv",
        {"file_path": "train.csv", "aggregate_by": "Store", "target_column": "Sales", "horizon": 1,
         "known_covariates": ["Promo"], "covariates_files": ["promo.csv"]},
    )  # fmt: skip
    assert not result.is_error, text(result)
    out = result.structured_content
    assert out["series_count"] == 1 and out["skipped_count"] == 1
    (warning,) = out["warnings"]
    assert "1 of 2 series were skipped" in warning and "overall_total" in warning
    assert "retrying without that covariate" in warning


async def test_any_file_and_column_names_work(fake_forecaster, data_dir):
    """Nothing depends on the example files: names, language and layout are the user's."""
    quarters = pd.date_range("2021-01-01", periods=12, freq="QS")
    rows = [f"{q:%d/%m/%Y};{port};{10 * i + len(port)},5" for i, q in enumerate(quarters) for port in ("Brest", "Nice")]
    main = write_csv(data_dir / "ventes_bateau_trimestres2026.csv", "trimestre;port;ventes_bateaux\n" + "\n".join(rows))
    shows = [
        f"{q:%d/%m/%Y};{i % 2}" for i, q in enumerate([*quarters, *pd.date_range("2024-01-01", periods=2, freq="QS")])
    ]
    write_csv(data_dir / "calendrier salons nautiques.csv", "date;salon\n" + "\n".join(shows))
    write_csv(data_dir / "infos_ports.csv", "port;longueur_quai\nBrest;500\nNice;300")
    result = await call(
        "forecast_csv",
        {"file_path": str(main), "aggregate_by": "port", "horizon": 2, "known_covariates": ["salon"],
         "covariates_files": [str(data_dir / "calendrier salons nautiques.csv"), str(data_dir / "infos_ports.csv")],
         "output_file": "prévisions bateaux.csv"},
    )  # fmt: skip
    assert not result.is_error, text(result)
    out = result.structured_content
    assert (out["date_column"], out["target_column"], out["frequency"]) == ("trimestre", "ventes_bateaux", "quarterly")
    assert out["series_count"] == 2 and out["covariates"]["known"] == ["salon"]
    assert any("infos_ports.csv was ignored" in w for w in out["warnings"])
    written = pd.read_csv(data_dir / "prévisions bateaux.csv")
    assert written["trimestre"].tolist() == ["2024-01-01", "2024-04-01"] * 2
    assert fake_forecaster.calls[0][0][:2].tolist() == [5.5, 15.5]  # "5,5" read as 5.5 for Brest


async def test_future_file_rows_are_forecast_and_only_they_are_reported(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    # store 1 / dept 10 asks for weeks 10 and 11 only (week 9 is forecast but not reported); 1/20 is not asked for
    future = ["1,10,2026-03-06,0", "1,10,2026-03-13,1", "2,10,2026-02-27,0", "2,20,2026-02-27,0"]
    write_csv(data_dir / "test.csv", "Store,Dept,Date,Holiday\n" + "\n".join(future))
    result = await call(
        "forecast_csv",
        {"file_path": "train.csv", "target_column": "Sales", "series_column": ["Store", "Dept"],
         "future_file": "test.csv", "fill_missing_dates": True},
    )  # fmt: skip
    assert not result.is_error, text(result)
    out = result.structured_content
    assert [(r["series"], r["date"]) for r in out["forecast"]] == [
        ("1/10", "2026-03-06"), ("1/10", "2026-03-13"), ("2/10", "2026-02-27"), ("2/20", "2026-02-27"),
    ]  # fmt: skip
    assert out["series_count"] == 3 and out["skipped_count"] == 0
    assert out["series"][0]["horizon"] == 3
    assert out["series"][0]["total"]["p50"] == 2 * 117


async def test_future_file_must_not_hold_targets(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    write_csv(data_dir / "test.csv", "Store,Dept,Date,Sales\n1,10,2026-02-27,5")
    result = await call(
        "forecast_csv",
        {
            "file_path": "train.csv",
            "target_column": "Sales",
            "series_column": ["Store", "Dept"],
            "future_file": "test.csv",
        },
    )
    assert result.is_error and "has values in the target column" in text(result)


# ---------------------------------------------------------------- output files


async def test_output_file_is_written_and_never_overwritten(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    args = {"file_path": "train.csv", "horizon": 2, "series_column": ["Store", "Dept"], "output_file": "out.csv"}
    first = await call("forecast_csv", args)
    again = await call("forecast_csv", args)
    outside = await call("forecast_csv", {**args, "output_file": str(data_dir.parent / "x.csv")})
    assert not first.is_error, text(first)
    out = first.structured_content
    assert out["forecast"] == [] and out["rows_written"] == 8
    written = pd.read_csv(data_dir / "out.csv")
    assert list(written.columns) == ["Store", "Dept", "Date", "p10", "p50", "p90"]
    assert written.iloc[0].tolist() == [1, 10, "2026-02-27", 113.0, 117.0, 121.0]
    # a second run picks a free name instead of overwriting
    assert not again.is_error, text(again)
    assert again.structured_content["output_file"] == str((data_dir / "out (2).csv").resolve())
    assert pd.read_csv(data_dir / "out.csv").equals(written)
    assert outside.is_error and "outside the allowed directories" in text(outside)


async def test_output_file_lands_next_to_the_input_file(fake_forecaster, data_dir):
    sub = data_dir / "client" / "2026"
    sub.mkdir(parents=True)
    long_file(sub / "train.csv")
    result = await call(
        "forecast_csv",
        {
            "file_path": str(sub / "train.csv"),
            "horizon": 2,
            "series_column": ["Store", "Dept"],
            "output_file": "fc.csv",
        },
    )
    assert not result.is_error, text(result)
    assert result.structured_content["output_file"] == str((sub / "fc.csv").resolve())
    assert (sub / "fc.csv").exists() and not (data_dir / "fc.csv").exists()


async def test_many_series_need_an_output_file(fake_forecaster, data_dir, monkeypatch):
    import timestral_mcp.preprocessing as pre

    monkeypatch.setattr(server, "MAX_SERIES", 2)
    monkeypatch.setattr(pre, "MAX_SERIES", 2)
    long_file(data_dir / "train.csv")
    args = {"file_path": "train.csv", "horizon": 1, "series_column": ["Store", "Dept"]}
    inline = await call("forecast_csv", args)
    to_file = await call("forecast_csv", {**args, "output_file": "out.csv"})
    assert inline.is_error and "Pass output_file" in text(inline)
    assert not to_file.is_error, text(to_file)
    assert to_file.structured_content["series"] is None  # too many to detail
    assert to_file.structured_content["series_count"] == 4


async def test_backtest_many_series_with_output_file_and_wape(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    result = await call(
        "backtest",
        {"file_path": "train.csv", "test_size": 2, "windows": 2, "series_column": ["Store", "Dept"],
         "target_column": "Sales", "output_file": "bt.csv"},
    )  # fmt: skip
    assert not result.is_error, text(result)
    out = result.structured_content
    assert out["predictions"] == [] and out["rows_written"] == 16
    assert list(pd.read_csv(data_dir / "bt.csv").columns) == [
        "Store",
        "Dept",
        "window",
        "date",
        "actual",
        "p10",
        "p50",
        "p90",
    ]
    ev = out["evaluation"]
    assert ev["wape"] == pytest.approx(
        ev["mae"] / np.mean([100 * s + d + i for s in (1, 2) for d in (10, 20) for i in (4, 5, 6, 7)]), rel=1e-3
    )
    assert out["baselines"][0]["wape"] is not None


# ---------------------------------------------------------------- batching


def test_batches_match_separate_calls(fake_forecaster):
    series = [np.arange(10.0) * k for k in (1, 2, 3)]
    batch = run_forecast_batch([(s, None) for s in series], 3, "m", "cpu")
    for s, b in zip(series, batch, strict=True):
        alone = run_forecast(s, 3, "m", "cpu")
        assert all(np.array_equal(alone.quantiles[q], b.quantiles[q]) for q in alone.quantiles)
    with pytest.raises(ValueError, match="same length"):
        run_forecast_batch([(np.arange(5.0), None), (np.arange(6.0), None)], 1, "m", "cpu")


def test_run_backtests_batches_and_reports_errors_per_series(fake_forecaster):
    ok = np.arange(1.0, 21.0)
    results = run_backtests([(ok, None), (ok * 2, None), (np.arange(1.0, 5.0), None)], 5, "m", "cpu", windows=2)
    assert isinstance(results[2], InputError)
    alone = run_backtest(ok * 2, 5, "m", "cpu", windows=2)
    assert results[1].mae == alone.mae and results[1].wape == alone.wape
    # two series of length 20 -> one model call per window
    assert len(fake_forecaster.calls) == 2 * 2 + 2


# ---------------------------------------------------------------- aggregate_by


async def test_aggregate_by_sums_rows_into_one_series_per_group(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    result = await call(
        "forecast_csv",
        {"file_path": "train.csv", "horizon": 2, "aggregate_by": "Store", "target_column": "Sales"},
    )
    assert not result.is_error, text(result)
    out = result.structured_content
    assert out["series_columns"] == ["Store"] and out["series_count"] == 2
    # store s sums departments 10 and 20: (100s + 10 + i) + (100s + 20 + i)
    contexts = [row.tolist() for row, _ in fake_forecaster.calls]
    n = len(WEEKS)
    assert contexts == [[200.0 * s + 30 + 2 * i for i in range(n)] for s in (1, 2)]
    first = out["forecast"][0]
    assert (first["series"], first["date"], first["p50"]) == ("1", "2026-02-27", 230.0 + 2 * (n - 1))


async def test_aggregate_by_empty_list_forecasts_the_total(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    result = await call(
        "forecast_csv", {"file_path": "train.csv", "horizon": 1, "aggregate_by": [], "target_column": "Sales"}
    )
    assert not result.is_error, text(result)
    out = result.structured_content
    assert out["series_columns"] is None and out["total"] is not None
    n = len(WEEKS)
    # 4 series: (100s + d + i) over s in {1, 2}, d in {10, 20}
    assert fake_forecaster.calls[0][0].tolist() == [660.0 + 4 * i for i in range(n)]


async def test_aggregate_by_keeps_covariates_and_future_rows(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    future = "\n".join(f"{s},{d},2026-02-27,0" for s in ("1", "2") for d in ("10", "20"))
    write_csv(data_dir / "test.csv", "Store,Dept,Date,Holiday\n" + future)
    result = await call(
        "forecast_csv",
        {
            "file_path": "train.csv",
            "future_file": "test.csv",
            "aggregate_by": "Store",
            "target_column": "Sales",
            "known_covariates": ["Holiday"],
        },
    )
    assert not result.is_error, text(result)
    out = result.structured_content
    assert out["series_count"] == 2 and [s["horizon"] for s in out["series"]] == [1, 1]
    holiday = fake_forecaster.covariate_calls[0]
    assert holiday, "the known covariate reaches the model"


async def test_aggregate_by_conflicts_with_a_different_series_column(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    result = await call(
        "forecast_csv",
        {"file_path": "train.csv", "horizon": 1, "aggregate_by": "Store", "series_column": "Dept"},
    )
    assert result.is_error and "aggregate_by already sets the series" in text(result)


async def test_aggregate_by_never_turns_empty_cells_into_zero(fake_forecaster, data_dir):
    write_csv(
        data_dir / "f.csv",
        "Store,Dept,Date,Sales\n1,1,2026-01-01,5\n1,2,2026-01-01,6\n1,1,2026-01-02,\n1,2,2026-01-02,\n"
        "1,1,2026-01-03,7\n1,2,2026-01-03,8\n1,1,2026-01-04,1\n1,2,2026-01-04,1",
    )
    result = await call(
        "forecast_csv", {"file_path": "f.csv", "horizon": 1, "aggregate_by": "Store", "target_column": "Sales"}
    )
    assert not result.is_error, text(result)
    row = fake_forecaster.calls[0][0]
    assert row[0] == 11.0 and np.isnan(row[1]) and row[2:].tolist() == [15.0, 2.0]


# ---------------------------------------------------------------- csv_text


async def test_csv_text_is_forecast_without_a_file(fake_forecaster, data_dir):
    result = await call(
        "forecast_csv", {"csv_text": "date,sales\n2026-01-01,1\n2026-01-02,2\n2026-01-03,3", "horizon": 2}
    )
    assert not result.is_error, text(result)
    out = result.structured_content
    assert out["file"] == "<csv_text>" and out["frequency"] == "daily"
    assert [r["date"] for r in out["forecast"]] == ["2026-01-04", "2026-01-05"]


async def test_csv_text_reads_european_files_and_writes_to_the_first_allowed_dir(fake_forecaster, data_dir):
    pasted = "﻿date;sales\n01/02/2026;1,5\n02/02/2026;2,5\n03/02/2026;3,5"
    result = await call("forecast_csv", {"csv_text": pasted, "horizon": 1, "output_file": "fc.csv"})
    assert not result.is_error, text(result)
    assert result.structured_content["output_file"] == str((data_dir / "fc.csv").resolve())
    assert fake_forecaster.calls[0][0].tolist() == [1.5, 2.5, 3.5]
    assert pd.read_csv(data_dir / "fc.csv")["date"].tolist() == ["2026-02-04"]


@pytest.mark.parametrize(
    "args, message",
    [
        ({}, "pass file_path"),
        ({"file_path": "a.csv", "csv_text": "date,sales\n2026-01-01,1"}, "not both"),
        ({"csv_text": "  "}, "csv_text must not be empty"),
        ({"csv_text": "date,sales\n"}, "no data rows"),
        ({"csv_text": "date,sales\n2026-01-01,1", "sheet": "S"}, "only used for Excel"),
    ],
)
async def test_csv_text_errors(fake_forecaster, data_dir, args, message):
    result = await call("forecast_csv", {"horizon": 1, **args})
    assert result.is_error and message in text(result), text(result)


async def test_backtest_accepts_csv_text_and_aggregate_by(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    pasted = (data_dir / "train.csv").read_text(encoding="utf-8")
    result = await call(
        "backtest", {"csv_text": pasted, "test_size": 2, "aggregate_by": "Store", "target_column": "Sales"}
    )
    assert not result.is_error, text(result)
    assert result.structured_content["series_count"] == 2


async def test_covariates_per_store_are_averaged_for_the_total(fake_forecaster, data_dir):
    long_file(data_dir / "sales history.csv")
    weeks = [*WEEKS, "2026-02-27"]
    rows = [f"{s},{w},{10 * int(s) + i}" for i, w in enumerate(weeks) for s in ("1", "2")]
    write_csv(data_dir / "weather.csv", "Store,Date,Temp\n" + "\n".join(rows))
    result = await call(
        "forecast_csv",
        {"file_path": "sales history.csv", "aggregate_by": [], "target_column": "Sales", "horizon": 1,
         "known_covariates": ["Temp"], "covariates_files": ["weather.csv"]},
    )  # fmt: skip
    assert not result.is_error, text(result)
    out = result.structured_content
    (warning,) = out["warnings"]
    assert "weather.csv has one row per Store" in warning and "averaged" in warning
    # mean of stores 1 and 2: (10 + i + 20 + i) / 2 = 15 + i, history then the forecast step
    temp = fake_forecaster.covariate_calls[0]["known_covariates"][0, 0].tolist()
    assert temp == [15.0 + i for i in range(len(weeks))]


async def test_backtest_of_many_series_stays_compact(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv", stores=tuple(str(s) for s in range(1, 7)))  # 12 series
    result = await call(
        "backtest",
        {"file_path": "train.csv", "series_column": ["Store", "Dept"], "target_column": "Sales", "test_size": 2,
         "windows": 2},
    )  # fmt: skip
    assert not result.is_error, text(result)
    out = result.structured_content
    assert out["series_count"] == 12 and out["evaluation"]["windows"] == 2
    assert out["series_results"] is None and out["windows"] == []
    assert len(out["series_wape"]) == 12 and "1/10" in out["series_wape"]
    assert len(out["predictions"]) == 48  # 12 series x 2 windows x 2 steps, under the inline limit


# ---------------------------------------------------------------- backtest comparisons


def plan_file(path, offset=1.0, blank_last=False):
    """Weekly sales with the user's own forecasts beside them (off by `offset`)."""
    rows = [f"{w},{10 + i},{10 + i + offset}" for i, w in enumerate(WEEKS)]
    if blank_last:
        rows[-1] = rows[-1].rsplit(",", 1)[0] + ","
    return write_csv(path, "Week,Sales,Our forecast\n" + "\n".join(rows))


async def test_backtest_compares_with_requested_baselines_and_user_forecasts(fake_forecaster, data_dir):
    plan_file(data_dir / "sales and plan.csv")
    result = await call(
        "backtest",
        {"file_path": "sales and plan.csv", "target_column": "Sales", "test_size": 2, "windows": 2,
         "baselines": ["moving_average:2", "drift"], "benchmark_column": "Our forecast"},
    )  # fmt: skip
    assert not result.is_error, text(result)
    out = result.structured_content
    methods = {b["method"]: b for b in out["baselines"]}
    assert list(methods) == ["naive", "moving_average:2", "drift", "benchmark:Our forecast"]
    assert methods["benchmark:Our forecast"]["mae"] == 1.0
    assert methods["drift"]["mae"] == 0.0  # the series is a straight line
    # the user's forecasts are not passed to the model as a covariate
    assert out["covariates"]["past"] == [] and fake_forecaster.covariate_calls[0] == {}


async def test_user_forecasts_must_cover_the_test_windows(fake_forecaster, data_dir):
    plan_file(data_dir / "p.csv", blank_last=True)
    result = await call(
        "backtest",
        {"file_path": "p.csv", "target_column": "Sales", "test_size": 2, "benchmark_column": "Our forecast"},
    )
    assert result.is_error and "must cover the last 2 periods" in text(result)


async def test_user_forecasts_from_another_file_are_added_up_with_aggregate_by(fake_forecaster, data_dir):
    long_file(data_dir / "train.csv")
    # forecasts per store and department, each 2 above the actual sales
    rows = [
        f"{s},{d},{w},{100 * int(s) + int(d) + i + 2}" for i, w in enumerate(WEEKS) for s in "12" for d in ("10", "20")
    ]
    write_csv(data_dir / "planner forecasts.csv", "Store,Dept,Date,Plan\n" + "\n".join(rows))
    result = await call(
        "backtest",
        {"file_path": "train.csv", "aggregate_by": [], "target_column": "Sales", "test_size": 2,
         "covariates_files": ["planner forecasts.csv"], "benchmark_column": "Plan"},
    )  # fmt: skip
    assert not result.is_error, text(result)
    out = result.structured_content
    bench = out["baselines"][-1]
    assert bench["method"] == "benchmark:Plan" and bench["mae"] == 8.0  # 4 series x 2, summed like the target
    assert any("'Plan' from planner forecasts.csv was added up over" in w for w in out["warnings"]), out["warnings"]


async def test_unknown_baseline_is_refused_with_the_alternatives(fake_forecaster, data_dir):
    plan_file(data_dir / "p.csv")
    result = await call(
        "backtest", {"file_path": "p.csv", "target_column": "Sales", "test_size": 2, "baselines": ["prophet"]}
    )
    assert result.is_error and "unknown baseline 'prophet'" in text(result) and "benchmark_column" in text(result)


def test_covariate_numeric_in_one_series_and_categorical_in_another(fake_forecaster, data_dir):
    # "kind" holds numbers for store A and text for store B: the two series cannot share a batch.
    days = pd.date_range("2026-01-01", periods=10, freq="D").strftime("%Y-%m-%d")
    rows = [f"{d},A,{10 + i},1" for i, d in enumerate(days)] + [f"{d},B,{20 + i},big" for i, d in enumerate(days)]
    write_csv(data_dir / "mixed.csv", "date,store,sales,kind\n" + "\n".join(rows))

    bt = server.backtest(file_path="mixed.csv", test_size=2, series_column="store", past_covariates=["kind"])
    assert bt.series_count == 2 and bt.skipped_count == 0

    out = server.forecast_csv(file_path="mixed.csv", horizon=2, series_column="store", past_covariates=["kind"])
    assert out.series_count == 2
