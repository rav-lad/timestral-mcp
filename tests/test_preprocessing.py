import os

import numpy as np
import pandas as pd
import pytest

from tests.conftest import daily_csv, write_csv
from timestral_mcp.forecasting import InputError
from timestral_mcp.preprocessing import (
    future_dates,
    infer_frequency,
    prepare_series,
    read_table,
    regular_grid,
    resolve_data_path,
)

MB = 1024 * 1024


def frame(text: str) -> pd.DataFrame:
    from io import StringIO

    return pd.read_csv(StringIO(text.strip()))


def dates(*values: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(values)))


# ---------------------------------------------------------------- file access


def test_resolves_relative_path_against_first_allowed_dir(tmp_path):
    csv = daily_csv(tmp_path / "s.csv", [1.0, 2.0, 3.0])
    assert resolve_data_path("s.csv", [tmp_path], MB) == csv.resolve()


def test_rejects_paths_outside_allowed_dirs(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = daily_csv(tmp_path / "secret.csv", [1.0, 2.0, 3.0])
    with pytest.raises(InputError, match="access denied"):
        resolve_data_path(str(outside), [allowed], MB)
    with pytest.raises(InputError, match="access denied"):
        resolve_data_path("../secret.csv", [allowed], MB)
    # A sibling directory whose name merely starts with the allowed name is outside too.
    sibling = tmp_path / "allowed-but-not"
    sibling.mkdir()
    daily_csv(sibling / "x.csv", [1.0, 2.0, 3.0])
    with pytest.raises(InputError, match="access denied"):
        resolve_data_path(str(sibling / "x.csv"), [allowed], MB)


def test_rejects_symlink_escaping_allowed_dir(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = daily_csv(tmp_path / "secret.csv", [1.0, 2.0, 3.0])
    link = allowed / "link.csv"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks is not permitted on this system")
    with pytest.raises(InputError, match="access denied"):
        resolve_data_path(str(link), [allowed], MB)


@pytest.mark.parametrize("path", ["", "   ", "https://example.com/data.csv", "file:///etc/passwd"])
def test_rejects_empty_paths_and_urls(tmp_path, path):
    with pytest.raises(InputError):
        resolve_data_path(path, [tmp_path], MB)


def test_missing_file_directory_and_extension(tmp_path):
    with pytest.raises(InputError, match="not found"):
        resolve_data_path("nope.csv", [tmp_path], MB)
    (tmp_path / "dir.csv").mkdir()
    with pytest.raises(InputError, match="not a regular file"):
        resolve_data_path("dir.csv", [tmp_path], MB)
    (tmp_path / "data.json").write_bytes(b"x")
    with pytest.raises(InputError, match="only .csv, .xlsx"):
        resolve_data_path("data.json", [tmp_path], MB)
    (tmp_path / "old.xls").write_bytes(b"x")
    with pytest.raises(InputError, match="save .xls files as .xlsx"):
        resolve_data_path("old.xls", [tmp_path], MB)


def test_file_size_limit(tmp_path):
    daily_csv(tmp_path / "big.csv", list(np.arange(200.0)))
    with pytest.raises(InputError, match="above the"):
        resolve_data_path("big.csv", [tmp_path], 100)


def test_read_csv_errors(tmp_path):
    (tmp_path / "empty.csv").write_text("")
    with pytest.raises(InputError, match="is empty"):
        read_table(tmp_path / "empty.csv")
    (tmp_path / "header.csv").write_text("date,sales\n")
    with pytest.raises(InputError, match="no data rows"):
        read_table(tmp_path / "header.csv")


# ---------------------------------------------------------------- columns


def test_detects_named_columns_and_sorts_chronologically():
    df = frame("ds,y\n2026-01-03,3\n2026-01-01,1\n2026-01-02,2")
    s = prepare_series(df)
    assert (s.date_column, s.target_column, s.frequency) == ("ds", "y", "daily")
    assert s.values.tolist() == [1.0, 2.0, 3.0]
    assert s.dates[0] == pd.Timestamp("2026-01-01")


def test_detects_date_column_by_content_not_only_name():
    df = frame("day,units\n2026-01-01,1\n2026-01-02,2\n2026-01-03,3")
    s = prepare_series(df)
    assert (s.date_column, s.target_column) == ("day", "units")


def test_numeric_target_among_text_columns():
    df = frame("date,store,amount\n2026-01-01,a,1\n2026-01-02,a,2\n2026-01-03,a,3")
    assert prepare_series(df).target_column == "amount"


def test_target_name_breaks_tie_between_numeric_columns():
    df = frame("date,store_id,sales\n2026-01-01,7,1\n2026-01-02,7,2\n2026-01-03,7,3")
    assert prepare_series(df).target_column == "sales"


def test_ambiguous_target_lists_candidates():
    df = frame("date,price,units\n2026-01-01,1,4\n2026-01-02,2,5\n2026-01-03,3,6")
    with pytest.raises(InputError, match=r"several numeric columns.*'price', 'units'"):
        prepare_series(df)
    assert prepare_series(df, target_column="units").values.tolist() == [4.0, 5.0, 6.0]


def test_ambiguous_date_column():
    df = frame("start,end,v\n2026-01-01,2026-01-02,1\n2026-01-02,2026-01-03,2\n2026-01-03,2026-01-04,3")
    with pytest.raises(InputError, match="several columns could be the date"):
        prepare_series(df)
    assert prepare_series(df, date_column="end").dates[0] == pd.Timestamp("2026-01-02")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_column": "revenue"}, "target column 'revenue' not found"),
        ({"date_column": "when"}, "date column 'when' not found"),
    ],
)
def test_missing_columns(kwargs, message):
    df = frame("date,sales\n2026-01-01,1\n2026-01-02,2\n2026-01-03,3")
    with pytest.raises(InputError, match=message):
        prepare_series(df, **kwargs)


def test_no_date_or_no_numeric_column():
    with pytest.raises(InputError, match="no column of dates"):
        prepare_series(frame("a,b\nx,1\ny,2"))
    with pytest.raises(InputError, match="no numeric column"):
        prepare_series(frame("date,label\n2026-01-01,a\n2026-01-02,b"))


def test_numeric_date_column_rejected():
    with pytest.raises(InputError, match="holds numbers"):
        prepare_series(frame("year,sales\n2020,1\n2021,2\n2022,3"), date_column="year")


def test_unparseable_dates_reported_with_line_numbers():
    df = frame("date,sales\n2026-01-01,1\nnot a date,2\n2026-01-03,3")
    with pytest.raises(InputError, match="line 3"):
        prepare_series(df, date_column="date")


def test_non_numeric_target_values_reported():
    df = frame("date,sales\n2026-01-01,1\n2026-01-02,n/a!\n2026-01-03,3")
    with pytest.raises(InputError, match=r"non-numeric values on lines \[3\]"):
        prepare_series(df, target_column="sales")


def test_empty_target_cells_become_missing_not_zero(tmp_path):
    csv = daily_csv(tmp_path / "s.csv", [1.0, float("nan"), 3.0, 4.0])
    s = prepare_series(read_table(csv))
    assert np.isnan(s.values[1])
    assert np.nansum(s.values) == 8.0


def test_duplicate_timestamps():
    df = frame("date,sales\n2026-01-01,1\n2026-01-02,2\n2026-01-02,3\n2026-01-03,4")
    with pytest.raises(InputError, match="1 duplicate timestamp.*2026-01-02"):
        prepare_series(df)


# ---------------------------------------------------------------- frequency


def test_gap_in_daily_series_is_rejected_with_missing_dates():
    idx = dates("2026-01-01", "2026-01-02", "2026-01-03", "2026-01-05", "2026-01-06")
    with pytest.raises(InputError, match=r"looks daily.*1 missing timestamp\(s\) \(first: 2026-01-04\)"):
        infer_frequency(idx)


def test_irregular_series_rejected():
    idx = dates("2026-01-01", "2026-01-04", "2026-01-15", "2026-03-01")
    with pytest.raises(InputError, match="could not infer"):
        infer_frequency(idx)


def test_explicit_frequency_must_match():
    idx = dates("2026-01-01", "2026-01-02", "2026-01-03")
    assert infer_frequency(idx, "Daily") == "daily"
    with pytest.raises(InputError, match="not regular at weekly"):
        infer_frequency(idx, "weekly")
    with pytest.raises(InputError, match="must be one of"):
        infer_frequency(idx, "fortnightly")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (("2026-01-01 00:00", "2026-01-01 01:00", "2026-01-01 02:00"), "hourly"),
        (("2026-01-05", "2026-01-12", "2026-01-19"), "weekly"),
        (("2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01"), "monthly"),
        (("2026-01-31", "2026-02-28", "2026-03-31", "2026-04-30"), "monthly"),
        (("2026-01-15", "2026-02-15", "2026-03-15"), "monthly"),
    ],
)
def test_infers_supported_frequencies(values, expected):
    assert infer_frequency(dates(*values)) == expected


def test_month_end_series_is_not_mistaken_for_fixed_day():
    # 30th of each month, except February: not a regular monthly grid.
    with pytest.raises(InputError):
        infer_frequency(dates("2026-01-30", "2026-02-27", "2026-03-30"))


def test_monthly_grid_uses_calendar_months():
    assert regular_grid(pd.Timestamp("2026-01-31"), 4, "monthly").strftime("%Y-%m-%d").tolist() == [
        "2026-01-31",
        "2026-02-28",
        "2026-03-31",
        "2026-04-30",
    ]
    # Day 30 must not drift to the 28th after February.
    assert regular_grid(pd.Timestamp("2026-01-30"), 4, "monthly").strftime("%Y-%m-%d").tolist() == [
        "2026-01-30",
        "2026-02-28",
        "2026-03-30",
        "2026-04-30",
    ]
    # Leap year
    assert regular_grid(pd.Timestamp("2027-12-31"), 3, "monthly")[2] == pd.Timestamp("2028-02-29")


def test_future_monthly_dates():
    df = frame("month,revenue\n2026-10-01,1\n2026-11-01,2\n2026-12-01,3")
    s = prepare_series(df)
    assert s.frequency == "monthly"
    assert future_dates(s, 3) == ["2027-01-01", "2027-02-01", "2027-03-01"]


def test_future_month_end_dates():
    df = frame("date,v\n2026-11-30,1\n2026-12-31,2\n2027-01-31,3")
    assert future_dates(prepare_series(df), 2) == ["2027-02-28", "2027-03-31"]


def test_future_hourly_dates_keep_time():
    df = frame("timestamp,load\n2026-03-01 22:00,1\n2026-03-01 23:00,2\n2026-03-02 00:00,3")
    assert future_dates(prepare_series(df), 2) == ["2026-03-02T01:00:00", "2026-03-02T02:00:00"]


def test_timezone_aware_daily_index_across_dst_keeps_wall_clock():
    idx = pd.date_range("2026-03-27", periods=5, freq=pd.DateOffset(days=1), tz="Europe/Paris")
    assert infer_frequency(pd.DatetimeIndex(idx)) == "daily"


def test_single_offset_timestamps():
    df = frame("time,v\n2026-01-01T00:00Z,1\n2026-01-01T01:00Z,2\n2026-01-01T02:00Z,3")
    s = prepare_series(df)
    assert s.frequency == "hourly"
    assert future_dates(s, 1) == ["2026-01-01T03:00:00+00:00"]


def test_mixed_offsets_across_dst_are_compared_in_utc():
    # Hourly local time over the 2026-03-29 Paris DST change: regular as absolute instants.
    idx = pd.date_range("2026-03-29 00:00", periods=4, freq="h", tz="Europe/Paris")
    df = pd.DataFrame({"time": [t.isoformat() for t in idx], "v": [1.0, 2.0, 3.0, 4.0]})
    s = prepare_series(df)
    assert s.frequency == "hourly"
    assert future_dates(s, 1) == ["2026-03-29T03:00:00+00:00"]  # 00:00+01:00 is 23:00Z


def test_csv_round_trip_of_unsorted_file(tmp_path):
    csv = write_csv(tmp_path / "u.csv", "date,sales\n2026-01-03,3\n2026-01-01,1\n2026-01-02,2")
    s = prepare_series(read_table(csv))
    assert s.values.tolist() == [1.0, 2.0, 3.0]
    assert future_dates(s, 1) == ["2026-01-04"]


# ---------------------------------------------------------------- covariates

PROMO_CSV = """
date,sales,promo,weather,store_type
2026-01-03,13,1,5.5,big
2026-01-01,11,0,4.0,small
2026-01-02,12,0,,big
2026-01-04,,1,,small
2026-01-05,,0,,big
"""


def test_known_covariates_split_history_and_future_rows():
    s = prepare_series(frame(PROMO_CSV), known_covariates=["promo"])
    assert s.target_column == "sales"
    assert s.values.tolist() == [11.0, 12.0, 13.0]
    assert len(s.dates) == 3 and s.future_rows == 2
    assert s.covariates.known["promo"].tolist() == [0.0, 0.0, 1.0, 1.0, 0.0]  # sorted by date
    assert s.covariates.labels[-1] == "2026-01-05"
    assert future_dates(s, 2) == ["2026-01-04", "2026-01-05"]


def test_past_covariates_cover_history_only_and_keep_missing_values():
    s = prepare_series(frame(PROMO_CSV), past_covariates=["weather"], known_covariates=["promo"])
    weather = s.covariates.past["weather"]
    assert weather.shape == (3,)
    assert weather[0] == 4.0 and np.isnan(weather[1]) and weather[2] == 5.5


def test_trailing_empty_targets_are_future_rows_even_without_covariates():
    s = prepare_series(frame(PROMO_CSV), target_column="sales")
    assert s.values.tolist() == [11.0, 12.0, 13.0]
    assert s.future_rows == 2 and s.covariates is None
    # The forecast starts right after the last observation, not after the empty rows.
    assert future_dates(s, 1) == ["2026-01-04"]


def test_empty_targets_inside_the_history_stay_missing_values():
    df = frame("date,sales\n2026-01-01,1\n2026-01-02,\n2026-01-03,3\n2026-01-04,")
    s = prepare_series(df)
    assert len(s.values) == 3 and np.isnan(s.values[1]) and s.future_rows == 1


def test_non_numeric_covariates_become_category_codes():
    s = prepare_series(frame(PROMO_CSV), known_covariates=["store_type", "promo"])
    cov = s.covariates
    assert cov.categorical == {"store_type"}
    # Codes in order of first appearance in the file (big on its first line = 0), then sorted by date
    assert cov.known["store_type"].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]


def test_numbers_stored_as_text_are_numeric_covariates():
    df = frame("date,sales,price\n2026-01-01,1,9.5\n2026-01-02,2,10\n2026-01-03,3,10.5")
    df["price"] = df["price"].astype(str)
    s = prepare_series(df, past_covariates=["price"])
    assert s.covariates.categorical == frozenset()
    assert s.covariates.past["price"].tolist() == [9.5, 10.0, 10.5]


def test_covariate_columns_are_excluded_from_target_detection():
    df = frame("date,units,price\n2026-01-01,1,4\n2026-01-02,2,5\n2026-01-03,3,6")
    assert prepare_series(df, past_covariates=["price"]).target_column == "units"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"known_covariates": ["nope"]}, "covariate column 'nope' not found"),
        ({"known_covariates": ["date"]}, "date column 'date' cannot be a covariate"),
        ({"known_covariates": ["promo"], "past_covariates": ["promo"]}, "listed more than once"),
        ({"target_column": "sales", "past_covariates": ["sales"]}, "cannot also be a covariate"),
        ({"past_covariates": [f"c{i}" for i in range(21)]}, "covariate column 'c0' not found"),
    ],
)
def test_invalid_covariates(kwargs, message):
    with pytest.raises(InputError, match=message):
        prepare_series(frame(PROMO_CSV), **kwargs)


def test_too_many_covariates():
    cols = [f"c{i}" for i in range(21)]
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=3).astype(str), "sales": [1, 2, 3]})
    for c in cols:
        df[c] = 1.0
    with pytest.raises(InputError, match="at most 20"):
        prepare_series(df, target_column="sales", past_covariates=cols)


def test_future_rows_must_stay_on_the_grid():
    text = PROMO_CSV.replace("2026-01-05", "2026-01-07")
    with pytest.raises(InputError, match="missing timestamp"):
        prepare_series(frame(text), known_covariates=["promo"])
