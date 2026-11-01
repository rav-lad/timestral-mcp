"""File access, column detection, timestamp validation and frequency handling.

Files: CSV (delimiter sniffed; semicolon files are read with decimal commas and day-first
dates, as exported by European spreadsheets) and Excel workbooks (.xlsx, .xlsm).

Missing-data policy: by default the timestamps must form a complete regular grid (no gaps,
no duplicates); otherwise the series is rejected with the offending timestamps listed.
With `fill_missing_dates`, missing grid timestamps are inserted as rows with an empty
target (a missing value for Timestral) and counted; values are never invented. The history
ends at the last row with a target value; the rows after it (empty target cells) are future
rows, which carry known covariates when there are any and are counted in the response.
Empty target cells inside the history are passed to Timestral as missing values (Timestral
masks them natively) and are counted too; they are never filled with zero or interpolated.

Covariates: columns named as past covariates are known over the history only; columns
named as known (future) covariates must also be filled for the forecast horizon, in the
future rows. Numeric columns are passed as numbers, any other column as categories (codes
in order of first appearance in the file).

Several series: with one or more series columns (e.g. store, or store and department), the
file holds one series per distinct combination, stacked in long format; each is prepared as
its own series.

Other files: covariate files (e.g. weather or prices per date, optionally per store) are
joined on the date and on the series columns they share with the main file; they fill only
cells the main file leaves empty, and their dates past the end of a series become its future
rows. A future file holds the rows to forecast (target absent or empty) and is appended to the
main file.
"""

from __future__ import annotations

import csv
import io
import os
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from timestral_mcp.forecasting import MAX_INPUT_LENGTH, Covariates, InputError

DATE_NAMES = ("date", "timestamp", "datetime", "ds", "time", "jour", "mois", "periode", "période", "datum", "fecha")
TARGET_NAMES = (
    *("sales", "revenue", "demand", "value", "target", "y", "quantity", "volume", "amount"),
    # French, the language of many users of this server's first deployments
    *("ventes", "vente", "ca", "chiffre_affaires", "chiffre d'affaires", "quantite", "quantité", "montant"),
)
FREQUENCIES = ("15min", "30min", "hourly", "daily", "business_daily", "weekly", "monthly", "quarterly", "yearly")
DATA_SUFFIXES = (".csv", ".xlsx", ".xlsm")
MAX_COVARIATES = 20
# Series per call whose details are returned to the client; more need an output file.
MAX_SERIES = 100
MAX_SERIES_TO_FILE = 20_000
# Rows of one data file (all series together); a single series is capped at MAX_INPUT_LENGTH.
MAX_FILE_ROWS = 2_000_000
_MAX_LISTED = 5
_MAX_GRID = 1_000_000
# Fixed-length steps; business days and calendar months are handled separately.
_FIXED_STEPS = {
    "15min": pd.Timedelta(minutes=15),
    "30min": pd.Timedelta(minutes=30),
    "hourly": pd.Timedelta(hours=1),
    "daily": pd.DateOffset(days=1),
    "weekly": pd.DateOffset(weeks=1),
}
_MONTH_STEPS = {"monthly": 1, "quarterly": 3, "yearly": 12}
# Shortest real duration of one step, to size a grid (DST and weekends only make steps longer).
_APPROX_STEPS = {
    "15min": pd.Timedelta(minutes=15),
    "30min": pd.Timedelta(minutes=30),
    "hourly": pd.Timedelta(hours=1),
    "daily": pd.Timedelta(hours=23),
    "business_daily": pd.Timedelta(hours=23),
    "weekly": pd.Timedelta(days=7) - pd.Timedelta(hours=1),
}
# Dates such as 31/10/2026 or 10.31.26: day-first or month-first, never ISO (year first).
_SLASHED_DATE = re.compile(r"\s*\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}(\D.*)?")
# Decimal-comma numbers, optionally with a space or dot as thousands separator: 1 234,5 / 1.234,5 / -3,25
_COMMA_NUMBER = re.compile(r"[+-]?(\d{1,3}(?:[.\s]\d{3})+|\d+)(,\d+)?")


@dataclass(frozen=True)
class PreparedSeries:
    dates: pd.DatetimeIndex  # the history: rows up to the last one with a target value
    values: np.ndarray  # float64, NaN = missing target value
    date_column: str
    target_column: str
    frequency: str
    covariates: Covariates | None = None
    future_rows: int = 0  # rows after the history (empty target cells)
    filled_timestamps: int = 0  # grid timestamps absent from the file, inserted as missing values


# ---------------------------------------------------------------- file access


def _within(path: Path, directory: Path) -> bool:
    return Path(os.path.normcase(path)).is_relative_to(Path(os.path.normcase(directory)))


def resolve_data_path(file_path: str, allowed_dirs: Sequence[Path], max_bytes: int) -> Path:
    """Canonicalize `file_path` and check it is a CSV or Excel file inside an allowed directory.

    Relative paths are resolved against the first allowed directory. Symlinks are resolved
    before the check, so a link cannot point outside the allowed directories.
    """
    if not file_path or not file_path.strip():
        raise InputError("file_path must not be empty")
    if "://" in file_path:
        raise InputError("file_path must be a local file path; URLs are not supported")
    path = Path(file_path.strip()).expanduser()
    if not path.is_absolute():
        path = allowed_dirs[0] / path
    try:
        real = path.resolve()
    except (OSError, RuntimeError) as e:
        raise InputError(f"cannot resolve file path {file_path!r}: {e}") from None

    if not any(_within(real, d) for d in allowed_dirs):
        dirs = ", ".join(str(d) for d in allowed_dirs)
        raise InputError(
            f"access denied: {str(real)!r} is outside the allowed directories ({dirs}). "
            "Ask the user to move the file to one of them, or to add its folder to TIMESTRAL_ALLOWED_DIRS; "
            "a small file can also be passed as csv_text."
        )
    if not real.exists():
        raise InputError(f"file not found: {str(real)!r}")
    if not real.is_file():
        raise InputError(f"not a regular file: {str(real)!r}")
    if real.suffix.lower() not in DATA_SUFFIXES:
        hint = " (save .xls files as .xlsx or .csv)" if real.suffix.lower() == ".xls" else ""
        raise InputError(f"only {', '.join(DATA_SUFFIXES)} files are supported, got {real.name!r}{hint}")
    size = real.stat().st_size
    if size > max_bytes:
        raise InputError(
            f"{real.name} is {size / 1e6:.1f} MB, above the {max_bytes / 1e6:.1f} MB limit (TIMESTRAL_MAX_FILE_MB)"
        )
    return real


def _free_name(path: Path) -> Path:
    """`path`, or `name (2).csv`, `name (3).csv`, ... if it is taken."""
    candidate, number = path, 1
    while candidate.exists() or candidate.is_symlink():
        number += 1
        candidate = path.with_name(f"{path.stem} ({number}){path.suffix}")
    return candidate


def resolve_output_path(file_path: str, allowed_dirs: Sequence[Path], base_dir: Path | None = None) -> Path:
    """Check that `file_path` is a .csv file in an existing directory inside an allowed directory.

    Relative paths resolve against `base_dir` (the folder of the input file) or else the first
    allowed directory. Existing files are never overwritten: a free name such as
    `forecast (2).csv` is used instead, and reported back by the tools.
    """
    if not file_path or not file_path.strip():
        raise InputError("output_file must not be empty")
    path = Path(file_path.strip()).expanduser()
    if not path.is_absolute():
        path = (base_dir or allowed_dirs[0]) / path
    try:
        real = path.parent.resolve() / path.name
    except (OSError, RuntimeError) as e:
        raise InputError(f"cannot resolve output path {file_path!r}: {e}") from None
    if not any(_within(real, d) for d in allowed_dirs):
        dirs = ", ".join(str(d) for d in allowed_dirs)
        raise InputError(f"access denied: {str(real)!r} is outside the allowed directories ({dirs})")
    if real.suffix.lower() != ".csv":
        raise InputError(f"output_file must be a .csv file, got {real.name!r}")
    if not real.parent.is_dir():
        raise InputError(f"directory {str(real.parent)!r} does not exist")
    return _free_name(real)


def _decode(path: Path) -> tuple[str, str]:
    """Return `(encoding, sample)`: UTF-8 (with or without BOM), else Windows-1252."""
    raw = path.read_bytes()[:65536]
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return encoding, raw.decode(encoding)
        except UnicodeDecodeError as e:
            # a multi-byte character cut by the sample boundary is not a decoding failure
            if encoding == "utf-8-sig" and e.start >= len(raw) - 3:
                return encoding, raw[: e.start].decode(encoding)
    return "latin-1", raw.decode("latin-1")


def _sniff_delimiter(sample: str) -> str:
    lines = "\n".join(sample.splitlines()[:50])
    try:
        return csv.Sniffer().sniff(lines, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _decimal_comma_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns written with decimal commas (1 234,5) to numbers; others are left alone."""
    for column in df.columns:
        if not (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])):
            continue
        text = df[column].dropna().astype(str).str.strip()
        text = text[text != ""]
        if text.empty or not text.map(lambda s: _COMMA_NUMBER.fullmatch(s) is not None).all():
            continue
        cleaned = df[column].astype(str).str.strip().str.replace(r"[.\s]", "", regex=True).str.replace(",", ".")
        df[column] = pd.to_numeric(cleaned.where(df[column].notna()), errors="coerce")
    return df


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        encoding, sample = _decode(path)
        delimiter = _sniff_delimiter(sample)
        df = pd.read_csv(path, sep=delimiter, encoding=encoding)
    except pd.errors.EmptyDataError:
        raise InputError(f"{path.name} is empty") from None
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as e:
        raise InputError(f"could not read {path.name} as CSV: {e}") from None
    return _european(df, delimiter)


def _european(df: pd.DataFrame, delimiter: str) -> pd.DataFrame:
    if delimiter == ";":
        df = _decimal_comma_columns(df)
        df.attrs["dayfirst"] = True
    return df


def read_csv_text(text: str, max_bytes: int) -> pd.DataFrame:
    """Read CSV data passed as text (e.g. a file the user attached to the conversation)."""
    if not text or not text.strip():
        raise InputError("csv_text must not be empty")
    size = len(text.encode("utf-8"))
    if size > max_bytes:
        raise InputError(
            f"csv_text is {size / 1e6:.1f} MB, above the {max_bytes / 1e6:.1f} MB limit (TIMESTRAL_MAX_FILE_MB)"
        )
    text = text.lstrip("﻿")
    delimiter = _sniff_delimiter(text[:65536])
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except pd.errors.EmptyDataError:
        raise InputError("csv_text is empty") from None
    except pd.errors.ParserError as e:
        raise InputError(f"could not read csv_text as CSV: {e}") from None
    df = _european(df, delimiter)
    if df.empty:
        raise InputError("csv_text has a header but no data rows")
    if len(df) > MAX_FILE_ROWS:
        raise InputError(f"csv_text has {len(df)} rows; at most {MAX_FILE_ROWS} are supported")
    return df


def aggregate_rows(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    by: Sequence[str],
    covariates: Sequence[str] = (),
    summed: Sequence[str] = (),
) -> pd.DataFrame:
    """Sum `target_col` over the rows sharing a date and the `by` columns (e.g. store sales over departments).

    A group whose target cells are all empty stays empty (a missing value or a future row), never
    zero. Covariate columns keep their first non-empty value in each group: they must describe the
    group as a whole (a holiday flag per date, a promotion per store). `summed` columns (e.g. other
    forecasts of the target) are added up like the target. Other columns are dropped.
    """
    covariates = [c for c in covariates if c not in summed]
    for column in summed:
        if column not in df.columns:
            raise InputError(f"column {column!r} not found; available columns: {list(map(str, df.columns))}")
        df = df.assign(**{column: pd.to_numeric(df[column], errors="coerce")})
    for column in [*by, *covariates]:
        if column not in df.columns:
            raise InputError(f"column {column!r} not found; available columns: {list(map(str, df.columns))}")
    target = pd.to_numeric(df[target_col], errors="coerce")
    bad = target.isna() & df[target_col].notna() & (df[target_col].astype(str).str.strip() != "")
    if bad.any():
        rows = [_line(i) for i in df.index[bad.to_numpy()][:_MAX_LISTED]]
        raise InputError(f"target column {target_col!r} has non-numeric values on lines {rows}")
    keys = [*by, date_col]
    frame = df[[*keys, *summed]].assign(**{target_col: target})
    grouped = frame.groupby(keys, sort=False, dropna=False)
    out = grouped[[target_col, *summed]].sum(min_count=1).reset_index()
    if covariates:
        firsts = df[[*keys, *covariates]].groupby(keys, sort=False, dropna=False).first().reset_index()
        out = out.merge(firsts, on=keys, how="left")
    out.attrs.update(df.attrs)
    return out


def _read_excel(path: Path, sheet: str | None) -> pd.DataFrame:
    try:
        with pd.ExcelFile(path, engine="openpyxl") as book:
            if sheet is not None and sheet not in book.sheet_names:
                raise InputError(f"sheet {sheet!r} not found in {path.name}; sheets: {book.sheet_names}")
            return pd.read_excel(book, sheet_name=sheet if sheet is not None else 0)
    except InputError:
        raise
    except Exception as e:  # openpyxl raises a variety of errors on corrupt or protected files
        raise InputError(f"could not read {path.name} as an Excel workbook: {e}") from None


def read_table(path: Path, sheet: str | None = None) -> pd.DataFrame:
    """Read a CSV file or one sheet of an Excel workbook (the first sheet by default)."""
    if path.suffix.lower() == ".csv":
        if sheet is not None:
            raise InputError("sheet is only used for Excel files")
        df = _read_csv(path)
    else:
        df = _read_excel(path, sheet)
    if df.empty:
        raise InputError(f"{path.name} has a header but no data rows")
    if len(df) > MAX_FILE_ROWS:
        raise InputError(f"{path.name} has {len(df)} rows; at most {MAX_FILE_ROWS} are supported")
    return df


# ---------------------------------------------------------------- column detection


def _parse_dates(series: pd.Series, dayfirst: bool = False) -> pd.Series:
    """Parse to datetimes (NaT where unparseable).

    Timestamps with mixed UTC offsets (e.g. local time across a DST change) are converted
    to UTC, so they are compared as absolute instants. Dates such as 01/11/2026 are read
    month first unless `dayfirst`; whichever order parses more values wins. ISO dates
    (year first) are never reordered.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        # pandas warns when it falls back to per-element format inference
        warnings.simplefilter("ignore", UserWarning)
        text = series.dropna().astype(str)
        if not text.empty and text.map(lambda s: _SLASHED_DATE.fullmatch(s) is not None).all():
            first = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
            other = pd.to_datetime(series, errors="coerce", dayfirst=not dayfirst)
            return other if other.notna().sum() > first.notna().sum() else first
        try:
            return pd.to_datetime(series, errors="coerce")
        except ValueError:
            return pd.to_datetime(series, errors="coerce", utc=True)


def _looks_like_dates(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    try:
        return bool(_parse_dates(series).notna().all())
    except (ValueError, TypeError, OverflowError):
        return False


def _missing_column(name: str, df: pd.DataFrame, role: str) -> InputError:
    return InputError(f"{role} column {name!r} not found; available columns: {list(map(str, df.columns))}")


def detect_date_column(df: pd.DataFrame, date_column: str | None, exclude: Sequence[str] = ()) -> str:
    if date_column is not None:
        if date_column not in df.columns:
            raise _missing_column(date_column, df, "date")
        return date_column
    parseable = [c for c in df.columns if c not in exclude and _looks_like_dates(df[c])]
    named = [c for c in parseable if str(c).strip().lower() in DATE_NAMES]
    for candidates in (named, parseable):
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise InputError(f"several columns could be the date column: {candidates}; pass date_column")
    raise InputError(f"no column of dates found in {list(map(str, df.columns))}; pass date_column")


def detect_target_column(
    df: pd.DataFrame, date_column: str, target_column: str | None, exclude: Sequence[str] = ()
) -> str:
    if target_column is not None:
        if target_column not in df.columns:
            raise _missing_column(target_column, df, "target")
        if target_column == date_column:
            raise InputError("target_column and date_column must be different columns")
        return target_column
    numeric = [
        c
        for c in df.columns
        if c != date_column
        and c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    if len(numeric) == 1:
        return numeric[0]
    if not numeric:
        raise InputError(f"no numeric column found besides {date_column!r}; pass target_column")
    named = [c for c in numeric if str(c).strip().lower() in TARGET_NAMES]
    if len(named) == 1:
        return named[0]
    raise InputError(f"several numeric columns could be the target: {numeric}; pass target_column")


def parse_date_column(df: pd.DataFrame, date_col: str) -> pd.Series:
    """The parsed `date_col`; empty or unparseable cells are reported with their file lines."""
    if pd.api.types.is_numeric_dtype(df[date_col]) and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise InputError(f"date column {date_col!r} holds numbers; expected dates such as 2026-01-31")
    parsed = _parse_dates(df[date_col], dayfirst=bool(df.attrs.get("dayfirst")))
    if parsed.isna().any():
        bad = parsed.index[parsed.isna().to_numpy()][:_MAX_LISTED]
        samples = [f"line {_line(i)}: {df[date_col].loc[i]!r}" for i in bad]
        raise InputError(f"date column {date_col!r} has empty or unparseable values ({'; '.join(samples)})")
    return parsed


def _line(index_label: object) -> int:
    """File line of a data row: +1 for the header, +1 because lines are 1-based."""
    return int(index_label) + 2


def _target_values(df: pd.DataFrame, column: str) -> np.ndarray:
    raw = df[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    bad = numeric.isna() & raw.notna() & (raw.astype(str).str.strip() != "")
    if bad.any():
        rows = [_line(i) for i in raw.index[bad.to_numpy()][:_MAX_LISTED]]
        raise InputError(f"target column {column!r} has non-numeric values on lines {rows}")
    values = numeric.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise InputError(f"target column {column!r} contains infinite values")
    return values


def _covariate_names(
    df: pd.DataFrame,
    past: Sequence[str] | None,
    known: Sequence[str] | None,
    date_col: str,
    other_columns: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    past, known = list(past or []), list(known or [])
    names = past + known
    available = [*map(str, df.columns), *(c for c in other_columns if c not in df.columns)]
    for name in names:
        if name not in available:
            raise InputError(f"covariate column {name!r} not found; available columns: {available}")
        if name == date_col:
            raise InputError(f"the date column {name!r} cannot be a covariate")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise InputError(f"covariate column(s) {duplicates} are listed more than once; a column is past or known")
    if len(names) > MAX_COVARIATES:
        raise InputError(f"at most {MAX_COVARIATES} covariate columns are supported, got {len(names)}")
    return past, known


def _covariate_values(
    df: pd.DataFrame, column: str, categories_from: pd.Series | None = None
) -> tuple[np.ndarray, bool]:
    """Return `(values, categorical)`: numbers as float64, other columns as category codes (NaN = missing).

    Codes follow the order of first appearance in `categories_from` (default: the column itself).
    """
    raw = df[column]
    if pd.api.types.is_bool_dtype(raw):
        return raw.to_numpy(dtype=np.float64), False
    if not pd.api.types.is_numeric_dtype(raw):
        numeric = pd.to_numeric(raw, errors="coerce")
        if (numeric.notna() | raw.isna()).all():
            raw = numeric
        else:
            source = raw if categories_from is None else categories_from
            codes = pd.Categorical(raw, categories=pd.unique(source.dropna())).codes
            return np.where(codes < 0, np.nan, codes).astype(np.float64), True
    values = raw.to_numpy(dtype=np.float64)
    if np.isinf(values).any():
        raise InputError(f"covariate column {column!r} contains infinite values")
    return values, False


# ---------------------------------------------------------------- frequency


def _check_frequency(frequency: str) -> str:
    frequency = frequency.strip().lower()
    if frequency not in FREQUENCIES:
        raise InputError(f"frequency must be one of {list(FREQUENCIES)}, got {frequency!r}")
    return frequency


def regular_grid(start: pd.Timestamp, periods: int, frequency: str) -> pd.DatetimeIndex:
    """`periods` timestamps from `start` at a calendar-aware regular frequency.

    Daily and weekly steps are calendar days (wall-clock time is kept across DST changes in
    time-zone-aware data); sub-daily steps are absolute durations. Business days skip
    Saturdays and Sundays (not public holidays). Monthly, quarterly and yearly steps are
    calendar months, computed from `start` rather than step by step so that e.g. the 30th
    never drifts to the 28th after February; a month-end start stays on month ends.
    """
    if frequency in _FIXED_STEPS:
        step = _FIXED_STEPS[frequency]
        if start.tz is None and isinstance(step, pd.DateOffset):
            # without a time zone a calendar day is 24 h: same grid, built without per-step date arithmetic
            step = pd.Timedelta(days=step.n * (7 if frequency == "weekly" else 1))
        return pd.date_range(start, periods=periods, freq=step)
    if frequency == "business_daily":
        # a weekend start rolls to Monday, so a series starting on a weekend never matches
        return pd.date_range(start, periods=periods, freq="B")
    if frequency in _MONTH_STEPS:
        months = _MONTH_STEPS[frequency]
        if start.is_month_end:
            return pd.DatetimeIndex([start + pd.offsets.MonthEnd(months * k) for k in range(periods)])
        return pd.DatetimeIndex([start + pd.DateOffset(months=months * k) for k in range(periods)])
    raise InputError(f"frequency must be one of {list(FREQUENCIES)}, got {frequency!r}")


def _grid_until(start: pd.Timestamp, end: pd.Timestamp, frequency: str) -> pd.DatetimeIndex:
    """The regular grid from `start` up to `end` included."""
    if frequency in _MONTH_STEPS:
        months = (end.year - start.year) * 12 + end.month - start.month
        count = months // _MONTH_STEPS[frequency] + 2
    else:
        count = int((end - start) / _APPROX_STEPS[frequency]) + 2
    if count > _MAX_GRID:
        raise InputError(f"the timestamps span far more than {_MAX_GRID} {frequency} steps")
    grid = regular_grid(start, count, frequency)
    return grid[grid <= end]


def _guess_frequency(dates: pd.DatetimeIndex) -> str | None:
    step = pd.Series(dates).diff().dropna().median()
    hours = step / pd.Timedelta(hours=1)
    for freq, lo, hi in (
        ("15min", 0.25, 0.25),
        ("30min", 0.5, 0.5),
        ("hourly", 1, 1),
        ("daily", 24, 24),
        ("weekly", 168, 168),
        ("monthly", 28 * 24, 31 * 24),
        ("quarterly", 89 * 24, 92 * 24),
        ("yearly", 365 * 24, 366 * 24),
    ):
        tolerance = 0.1 * lo if lo < 1 else 1
        if lo - tolerance <= hours <= hi + tolerance:
            if freq == "daily" and len(dates) >= 7 and bool((dates.dayofweek < 5).all()):
                return "business_daily"
            return freq
    return None


def _grid_mismatch(dates: pd.DatetimeIndex, frequency: str) -> str:
    """Describe how `dates` departs from the regular `frequency` grid starting at its first date."""
    try:
        grid = _grid_until(dates[0], dates[-1], frequency)
    except InputError:
        return f"the timestamps are far denser or sparser than {frequency}"
    missing = grid.difference(dates)
    extra = dates.difference(grid)
    parts = []
    if len(missing):
        listed = ", ".join(format_timestamp(t, _date_only(dates)) for t in missing[:_MAX_LISTED])
        parts.append(f"{len(missing)} missing timestamp(s) (first: {listed})")
    if len(extra):
        listed = ", ".join(format_timestamp(t, _date_only(dates)) for t in extra[:_MAX_LISTED])
        parts.append(f"{len(extra)} timestamp(s) off the {frequency} grid (first: {listed})")
    return " and ".join(parts) or f"the timestamps do not follow a {frequency} grid"


_GAP_HINT = (
    "Fill or remove the gaps in the file, or pass fill_missing_dates=true to insert the missing "
    "timestamps as missing values (no value is ever invented)."
)


# Shortest and longest real duration of one step of each frequency (DST, weekends, month lengths).
_STEP_RANGE = {
    "15min": (pd.Timedelta(minutes=15), pd.Timedelta(minutes=15)),
    "30min": (pd.Timedelta(minutes=30), pd.Timedelta(minutes=30)),
    "hourly": (pd.Timedelta(hours=1), pd.Timedelta(hours=1)),
    "daily": (pd.Timedelta(hours=23), pd.Timedelta(hours=25)),
    "business_daily": (pd.Timedelta(hours=23), pd.Timedelta(days=3, hours=1)),
    "weekly": (pd.Timedelta(days=7, hours=-1), pd.Timedelta(days=7, hours=1)),
    "monthly": (pd.Timedelta(days=28, hours=-1), pd.Timedelta(days=31, hours=1)),
    "quarterly": (pd.Timedelta(days=89, hours=-1), pd.Timedelta(days=92, hours=1)),
    "yearly": (pd.Timedelta(days=365, hours=-1), pd.Timedelta(days=366, hours=1)),
}


def infer_frequency(dates: pd.DatetimeIndex, frequency: str | None = None) -> str:
    """Return the frequency of sorted unique `dates`, or check an explicit one.

    A frequency is accepted only when every timestamp lies exactly on its regular grid.
    When several frequencies fit (e.g. a few weekdays in a row), the finest one wins.
    """
    if frequency is not None:
        frequency = _check_frequency(frequency)
        if not regular_grid(dates[0], len(dates), frequency).equals(dates):
            raise InputError(
                f"the series is not regular at {frequency} frequency: {_grid_mismatch(dates, frequency)}. {_GAP_HINT}"
            )
        return frequency

    if len(dates) < 2:
        raise InputError("at least two timestamps are needed to infer the frequency")
    shortest = (dates[1:] - dates[:-1]).min()
    candidates = [f for f in FREQUENCIES if _STEP_RANGE[f][0] <= shortest <= _STEP_RANGE[f][1]]
    matches = [f for f in candidates if regular_grid(dates[0], len(dates), f).equals(dates)]
    if matches:
        return matches[0]
    guess = _guess_frequency(dates)
    if guess is None:
        raise InputError(
            f"could not infer a regular frequency (supported: {', '.join(FREQUENCIES)}); "
            "the median spacing is " + str(pd.Series(dates).diff().dropna().median())
        )
    raise InputError(f"the series looks {guess} but is irregular: {_grid_mismatch(dates, guess)}. {_GAP_HINT}")


# ---------------------------------------------------------------- series assembly


def _date_only(dates: pd.DatetimeIndex) -> bool:
    return dates.tz is None and bool((dates == dates.normalize()).all())


def format_timestamp(ts: pd.Timestamp, date_only: bool) -> str:
    return ts.strftime("%Y-%m-%d") if date_only else ts.isoformat()


def _take(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """`values[rows]`, with NaN where `rows` is -1 (an inserted timestamp)."""
    out = np.full(rows.size, np.nan)
    present = rows >= 0
    out[present] = values[rows[present]]
    return out


def _fill_grid(dates: pd.DatetimeIndex, frequency: str | None) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """The complete grid covering sorted unique `dates`, and for each grid slot the position in `dates` (-1 = gap)."""
    freq = _check_frequency(frequency) if frequency is not None else _guess_frequency(dates)
    if freq is None:
        raise InputError(
            f"could not infer a regular frequency to fill the gaps (supported: {', '.join(FREQUENCIES)}); "
            "pass frequency"
        )
    grid = _grid_until(dates[0], dates[-1], freq)
    extra = dates.difference(grid)
    if len(extra):
        listed = ", ".join(format_timestamp(t, _date_only(dates)) for t in extra[:_MAX_LISTED])
        raise InputError(
            f"{len(extra)} timestamp(s) lie off the {freq} grid (first: {listed}); filling gaps cannot repair "
            "them. Pass another frequency or fix these rows."
        )
    if len(grid) > MAX_INPUT_LENGTH:
        raise InputError(f"filling the gaps would give {len(grid)} rows; at most {MAX_INPUT_LENGTH} are supported")
    return grid, dates.get_indexer(grid)


def prepare_series(
    df: pd.DataFrame,
    date_column: str | None = None,
    target_column: str | None = None,
    frequency: str | None = None,
    past_covariates: Sequence[str] | None = None,
    known_covariates: Sequence[str] | None = None,
    fill_missing_dates: bool = False,
    exclude: Sequence[str] = (),
    extra: pd.DataFrame | None = None,
    extend: bool = True,
) -> PreparedSeries:
    """Select, validate and chronologically sort the date, target and covariate columns.

    `exclude` lists columns that are neither date nor target candidates (e.g. a series id column).
    `extra` holds covariate values from other files, indexed by timestamp: they fill the cells
    the file leaves empty, and (if `extend`) its timestamps after the last row extend the series
    with future rows.
    """
    date_col = detect_date_column(df, date_column, exclude)
    other = list(extra.columns) if extra is not None else []
    past_cols, known_cols = _covariate_names(df, past_covariates, known_covariates, date_col, other)
    target_col = detect_target_column(df, date_col, target_column, exclude=[*past_cols, *known_cols, *exclude])
    if target_col in past_cols + known_cols:
        raise InputError(f"the target column {target_col!r} cannot also be a covariate")
    parsed = parse_date_column(df, date_col)

    values = _target_values(df, target_col)
    order = np.argsort(parsed.to_numpy(), kind="stable")
    dates = pd.DatetimeIndex(parsed.iloc[order])

    duplicated = dates.duplicated()
    if duplicated.any():
        dupes = sorted(set(dates[duplicated]))
        listed = ", ".join(format_timestamp(t, _date_only(dates)) for t in dupes[:_MAX_LISTED])
        raise InputError(f"{len(dupes)} duplicate timestamp(s) in {date_col!r} (first: {listed}); keep one row each")

    # rows[i] = position in the file of the i-th timestamp of the series, -1 for an inserted one
    rows = order
    if fill_missing_dates and len(dates) >= 2:
        dates, slots = _fill_grid(dates, frequency)
        rows = np.where(slots >= 0, order[np.maximum(slots, 0)], -1)
    freq = infer_frequency(dates, frequency)
    names = past_cols + known_cols
    if extend and extra is not None and names and len(extra) and extra.index.max() > dates[-1]:
        later = _grid_until(dates[-1], extra.index.max(), freq)[1:]
        dates = dates.append(later)
        rows = np.concatenate([rows, np.full(len(later), -1)])
    values = _take(values, rows)

    observed = np.flatnonzero(~np.isnan(values))
    history = int(observed[-1]) + 1 if observed.size else 0
    covariates = None
    if names:
        frame = df[[n for n in names if n in df.columns]].reset_index(drop=True).reindex(rows)
        frame.index = dates
        if extra is not None:
            frame = frame.combine_first(extra[[n for n in names if n in extra.columns]].reindex(dates))
        sources = [df] if extra is None else [df, extra]
        parsed_cols = {
            name: _covariate_values(frame, name, pd.concat([f[name] for f in sources if name in f.columns]))
            for name in names
        }
        cols = {name: vals for name, (vals, _) in parsed_cols.items()}
        covariates = Covariates(
            past={name: cols[name][:history] for name in past_cols},
            known={name: cols[name] for name in known_cols},
            categorical=frozenset(name for name, (_, cat) in parsed_cols.items() if cat),
            labels=tuple(dates_as_strings(dates)),
        )
    return PreparedSeries(
        dates=dates[:history],
        values=values[:history],
        date_column=str(date_col),
        target_column=str(target_col),
        frequency=freq,
        covariates=covariates,
        future_rows=len(dates) - history,
        filled_timestamps=int((rows[:history] < 0).sum()),
    )


SERIES_ID_SEPARATOR = "/"


@dataclass(frozen=True)
class SeriesGroup:
    label: str  # the key values joined by SERIES_ID_SEPARATOR, e.g. "1/12"
    keys: dict[str, str]  # series column -> value
    frame: pd.DataFrame


def series_labels(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """The series label of every row: its values of `columns` joined by SERIES_ID_SEPARATOR."""
    for column in columns:
        if column not in df.columns:
            raise _missing_column(column, df, "series")
        empty = df[column].isna()
        if empty.any():
            rows = [_line(i) for i in df.index[empty.to_numpy()][:_MAX_LISTED]]
            raise InputError(f"series column {column!r} is empty on lines {rows}")
    parts = [df[c].map(_series_label) for c in columns]
    labels = parts[0]
    for part in parts[1:]:
        labels = labels + SERIES_ID_SEPARATOR + part
    return labels


def split_series(
    df: pd.DataFrame,
    series_columns: Sequence[str],
    series_ids: Sequence[str] | None = None,
    max_series: int = MAX_SERIES,
) -> list[SeriesGroup]:
    """Split a long-format table into one table per combination of `series_columns`, in order of first appearance.

    Rows keep their original index, so error messages still point at file lines.
    """
    labels = series_labels(df, series_columns)
    groups = {label: frame for label, frame in df.groupby(labels, sort=False)}
    available = list(groups)
    if series_ids is not None:
        wanted = list(dict.fromkeys(str(s) for s in series_ids))
        unknown = [s for s in wanted if s not in groups]
        if unknown:
            shown = available[:20]
            more = f" (and {len(available) - 20} more)" if len(available) > 20 else ""
            hint = (
                f" (ids join {', '.join(series_columns)} with {SERIES_ID_SEPARATOR!r})"
                if len(series_columns) > 1
                else ""
            )
            raise InputError(f"series {unknown[:20]} not found{hint}; available: {shown}{more}")
        available = wanted
    if not available:
        raise InputError("series_ids must not be empty")
    if len(available) > max_series:
        raise InputError(
            f"the file holds {len(available)} series; at most {max_series} per call are supported here. "
            + ("Pass output_file to write the forecasts to a file, or " if max_series == MAX_SERIES else "")
            + "pass series_ids to choose some."
        )
    out = []
    for label in available:
        frame = groups[label]
        first = frame.iloc[0]
        out.append(SeriesGroup(label, {c: _series_label(first[c]) for c in series_columns}, frame))
    return out


@dataclass(frozen=True)
class CovariateTable:
    """Covariate rows of another file, by series key (the shared series columns) and timestamp."""

    name: str
    keys: tuple[str, ...]
    columns: tuple[str, ...]
    by_key: dict[str, pd.DataFrame]  # key label ("" when the file has no series column) -> rows by timestamp
    averaged_over: tuple[str, ...] = ()  # columns whose rows were averaged to match aggregated series

    def lookup(self, series_keys: dict[str, str]) -> pd.DataFrame | None:
        label = SERIES_ID_SEPARATOR.join(series_keys[k] for k in self.keys)
        return self.by_key.get(label)


class StaticCovariateFile(InputError):
    """A covariate file without dates: attributes that never change for a series (e.g. a store's size)."""

    def __init__(self, name: str, columns: Sequence[str]):
        self.name, self.columns = name, list(columns)
        super().__init__(
            f"{name} has no date column. Covariates must vary over time: attributes that are constant per "
            "series (e.g. store size) cannot inform a forecast made from each series' own history."
        )


def covariate_table(
    df: pd.DataFrame,
    name: str,
    date_column: str,
    series_columns: Sequence[str],
    wanted: Sequence[str],
    average_over: Sequence[str] = (),
    summed: Sequence[str] = (),
) -> CovariateTable:
    """Index the `wanted` columns of a covariate file by the series columns it shares with the main file and by date.

    Its date column is the one named like the main file's, or else the only date column.
    `average_over` lists the columns the main file's series were added up over (aggregate_by):
    rows of this file that differ only by those columns are averaged (numbers) or take their
    first value (categories), so a file per store can condition the total of all stores.
    """
    keys = tuple(c for c in series_columns if c in df.columns)
    if date_column in df.columns:
        date_col = date_column
    else:
        try:
            date_col = detect_date_column(df, None, exclude=keys)
        except InputError:
            columns = [c for c in wanted if c in df.columns and c not in keys]
            raise StaticCovariateFile(name, columns) from None
    columns = tuple(c for c in wanted if c in df.columns and c != date_col and c not in keys)
    averaged = tuple(
        c for c in average_over if c in df.columns and c not in keys and c != date_col and c not in columns
    )
    frame = df[[*keys, *columns]].copy()
    frame.index = pd.DatetimeIndex(parse_date_column(df, date_col))
    if averaged and columns:
        frame.index.name = "__date__"
        numeric = {c: "mean" for c in columns if pd.api.types.is_numeric_dtype(frame[c])}
        how = {c: numeric.get(c, "first") for c in columns}
        # other forecasts of the target add up like the target: sum, but keep all-empty groups empty
        how.update({c: (lambda s: s.sum(min_count=1)) for c in summed if c in columns})
        frame = frame.groupby([*keys, "__date__"], sort=False, dropna=False).agg(how).reset_index(list(keys))
        frame.index.name = None
    labels = series_labels(frame, keys) if keys else pd.Series("", index=frame.index)
    by_key: dict[str, pd.DataFrame] = {}
    for label, rows in frame.groupby(labels.to_numpy(), sort=False):
        rows = rows[list(columns)]
        if rows.index.duplicated().any():
            when = format_timestamp(rows.index[rows.index.duplicated()][0], _date_only(rows.index))
            where = f" for {', '.join(keys)} = {label}" if keys else ""
            raise InputError(f"{name} has several rows on {when}{where}; keep one row per date")
        by_key[str(label)] = rows.sort_index()
    return CovariateTable(name, keys, columns, by_key, averaged if columns else ())


def _series_label(value: object) -> str:
    # 12.0 read from a numeric column is the id "12", as the user would write it
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def future_dates(series: PreparedSeries, horizon: int) -> list[str]:
    """Timestamps of the `horizon` steps after the series, formatted like the input."""
    n = len(series.dates)
    grid = regular_grid(series.dates[0], n + horizon, series.frequency)[n:]
    date_only = _date_only(series.dates)
    return [format_timestamp(t, date_only) for t in grid]


def dates_as_strings(dates: pd.DatetimeIndex) -> list[str]:
    if _date_only(dates):
        return list(dates.strftime("%Y-%m-%d"))
    return [t.isoformat() for t in dates]


def file_frequency(dates: pd.Series, frequency: str | None) -> str | None:
    """The frequency of a file holding several series, from all its timestamps together.

    Inferring it once is faster and more reliable than per series: a sparse series (a few
    rows months apart) would otherwise look monthly in a weekly file. None when the
    timestamps do not tell; each series is then inferred on its own.
    """
    if frequency is not None:
        return _check_frequency(frequency)
    unique = pd.DatetimeIndex(dates.dropna().unique()).sort_values()
    if len(unique) < 2:
        return None
    try:
        return infer_frequency(unique)
    except InputError:
        return _guess_frequency(unique)
