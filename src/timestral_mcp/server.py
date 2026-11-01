"""MCP server exposing Timestral forecasting tools over STDIO.

STDOUT carries MCP protocol messages only; all logging goes to STDERR.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from mcp.server import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from timestral_mcp import __version__
from timestral_mcp import model as model_mod
from timestral_mcp.backtesting import (
    MAX_BACKTEST_POINTS,
    MAX_WINDOWS,
    BacktestResult,
    Baseline,
    baseline_forecasters,
    pool,
    run_backtests,
)
from timestral_mcp.forecasting import (
    MAX_HORIZON,
    NATIVE_QUANTILES,
    Covariates,
    ForecastResult,
    InputError,
    covariate_signature,
    quantile_key,
    run_forecast,
    run_forecast_batch,
    validate_horizon,
    validate_quantiles,
    validate_values,
)
from timestral_mcp.preprocessing import (
    FREQUENCIES,
    MAX_SERIES,
    MAX_SERIES_TO_FILE,
    SERIES_ID_SEPARATOR,
    CovariateTable,
    PreparedSeries,
    SeriesGroup,
    StaticCovariateFile,
    aggregate_rows,
    covariate_table,
    dates_as_strings,
    detect_date_column,
    detect_target_column,
    file_frequency,
    future_dates,
    parse_date_column,
    prepare_series,
    read_csv_text,
    read_table,
    regular_grid,
    resolve_data_path,
    resolve_output_path,
    series_labels,
    split_series,
)

logger = logging.getLogger("timestral_mcp")

INSTRUCTIONS = """\
Local time-series forecasting with Mistral AI's Timestral foundation models. The file \
tools read CSV and Excel files; a file can hold several series in long format (pass \
series_column, e.g. "store" or ["store", "dept"]). Results are quantile forecasts: p50 is \
the median forecast and [p10, p90] is an 80% prediction interval of the model, not a \
statistical confidence interval. Use `backtest` (ideally with several rolling windows) to \
check how the model would have done on recent data before relying on its forecasts.

The user's files: pass the absolute path of the file the user attached or named as \
file_path, and read it where it is: do not copy, move or convert it, and do not \
preprocess it with your own scripts (the tools read Excel and European CSV files and add up \
rows with aggregate_by). With no path (a small file attached to a chat), pass its content \
as csv_text. Write results with a bare output_file name such as forecast.csv: it is created \
next to the input file. If a path is outside the allowed folders, tell the user which \
folders are allowed instead of moving their file.

Comparisons in a backtest: the tool scores naive and seasonal_naive, plus mean, \
moving_average:N, drift and seasonal_growth when asked (`baselines`), and the user's own \
past forecasts when a column holds them (`benchmark_column`). Never compute another \
method yourself and compare it with the backtest: it would not use the same windows. If \
the user asks for a method the tool lacks, say so and offer the closest one.

Level of the forecast: when a file is more detailed than the question (sales per store and \
department, but the user asks per store or in total), pass aggregate_by (e.g. "Store", or [] \
for one total) rather than forecasting every detailed series and adding them up.

Several files: covariates kept in other files (e.g. weather or prices per date and store) \
go in covariates_files; they are joined on the date and on the series columns they share \
with the main file. The rows to forecast (e.g. a test.csv without the target) go in \
future_file; the horizon then defaults to those rows. Large files: with more than 100 \
series, or to fill a test file, pass output_file. The forecasts are written there and the \
response only summarizes them; tell the user where the file is.

Totals: report the `total` returned by the tools instead of adding up forecasts yourself, \
and mention that its p10/p90 range is conservative. Always tell the user every entry of \
`warnings`; when series were skipped, say that the totals leave them out.

Several files from the user: any file names work. Use the file with the history of the \
quantity to forecast as file_path; files with other columns by date (and optionally by the \
same ids) as covariates_files; a file of future rows to fill as future_file. Files without \
dates (attributes per store or product) cannot be used and are reported in warnings.

Effects of a covariate (e.g. how much a promotion adds): use `compare_scenarios` on one \
file, never the difference between forecasts of two different files or series.

Gaps: if a file is rejected for missing timestamps, tell the user, and retry with \
fill_missing_dates=true only if leaving those dates as missing values is acceptable. With \
several series, series that cannot be used are skipped and listed with the reason; report \
them.

Covariates: put a column in known_covariates only if its future values are decided or \
published in advance (planned promotions, holidays, opening hours, a price list). Anything \
measured as it happens (actual weather, traffic, observed prices) belongs in \
past_covariates; giving its real future values in a backtest would leak information and \
make the results optimistic. If you cannot tell from the column name and the conversation \
whether a column is known in advance, ask the user before calling the tool instead of \
guessing. When reporting a backtest, mention the columns listed in \
assumed_known_in_advance."""

mcp = MCPServer("timestral", instructions=INSTRUCTIONS, version=__version__)
# Forecast rows returned in one response, over all series; more need an output file.
MAX_OUTPUT_POINTS = 20_000
# Skipped series listed in a response (all are counted).
MAX_SKIPPED_LISTED = 50
# Groups of totals_by returned in a response.
MAX_TOTAL_GROUPS = 200
# Backtests: series whose windows and full results are returned; above, only a WAPE per series,
# so that a backtest of every store fits in the client's context.
MAX_DETAILED_SERIES = 10
# Backtest predictions returned in a response; more need an output file.
MAX_INLINE_PREDICTIONS = 300
_READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
# Writes only a new file named by output_file, never overwriting one.
_MAY_WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)

_settings: model_mod.Settings | None = None


def settings() -> model_mod.Settings:
    global _settings
    if _settings is None:
        _settings = model_mod.load_settings()
    return _settings


_roots_logged = False


def _log_client_roots(ctx: Context | None) -> None:
    """Log once whether the client declares MCP roots (folders it lets servers use).

    Diagnostic only: it tells whether a future version can take its allowed folders from
    the client instead of TIMESTRAL_ALLOWED_DIRS.
    """
    global _roots_logged
    if _roots_logged or ctx is None:
        return
    _roots_logged = True
    try:
        session = ctx.request_context.session
        caps = session.client_capabilities
        info = session.client_params.client_info if session.client_params else None
        logger.info(
            "Client %s declares roots capability: %s",
            f"{info.name} {info.version}" if info else "(unknown)",
            None if caps is None else caps.roots,
        )
    except Exception as e:  # never let a diagnostic break a tool call
        logger.info("Could not read the client's capabilities: %s", e)


@contextlib.contextmanager
def _tool_errors() -> Iterator[None]:
    """Turn expected failures into tool errors the calling model can read and act on."""
    try:
        yield
    except (InputError, model_mod.ConfigurationError, model_mod.ModelLoadError) as e:
        raise ToolError(str(e)) from e
    except Exception as e:
        # A bug, not a user error: the details go to the server log (stderr), not to the client.
        logger.exception("Unexpected error in a tool call")
        raise ToolError(
            f"internal error in timestral-mcp ({type(e).__name__}; details in the server log). This is a bug in "
            "the server, not a problem with the user's data: tell the user and suggest reporting it."
        ) from e


def _num(x: float) -> float:
    """Round to float32 precision (the model computes in float32)."""
    return float(f"{x:.7g}")


def _column_key(level: float) -> str:
    return f"p{round(level * 100):g}"


# ---------------------------------------------------------------- output schemas

QuantileLevels = Annotated[
    list[float] | None,
    Field(
        description="Quantile levels to return; each must be one of Timestral's native levels "
        f"({', '.join(quantile_key(q) for q in NATIVE_QUANTILES)}). Default: 0.1, 0.5, 0.9."
    ),
]
ModelId = Annotated[
    str | None,
    Field(description="Checkpoint, e.g. mistralai/Timestral-28M. Default: the server's configured model."),
]
Horizon = Annotated[int, Field(ge=1, le=MAX_HORIZON, description="Number of future steps to forecast.")]

TOTAL_DESCRIPTION = (
    "Sum over the horizon of each quantile column (pNN -> total). The p50 total is the sum of the medians. "
    "The p10-p90 totals are not the 80% interval of the total: they assume errors move together across "
    "steps, so the range is normally wider than the true one (conservative)."
)
WARNINGS_DESCRIPTION = (
    "Things the user must be told, in plain words, alongside the results: e.g. series left out of the totals "
    "or a file that was ignored."
)


class ForecastMetadata(BaseModel):
    input_length: int = Field(description="Observations supplied.")
    context_length: int = Field(description="Most recent observations the model read (capped at 32768).")
    missing_values: int = Field(description="Observations passed to the model as missing (empty cells or filled gaps).")
    filled_timestamps: int = Field(
        default=0, description="Timestamps absent from the file, inserted as missing values (fill_missing_dates)."
    )
    device: str


class ForecastOutput(BaseModel):
    model: str
    horizon: int
    quantiles: dict[str, list[float]] = Field(description="Quantile level -> forecast for each future step.")
    total: dict[str, float] = Field(description=TOTAL_DESCRIPTION)
    metadata: ForecastMetadata


class CovariateInfo(BaseModel):
    past: list[str] = Field(description="Columns used as past-only covariates.")
    known: list[str] = Field(description="Columns used as known-future covariates.")
    categorical: list[str] = Field(description="Covariates treated as categories (non-numeric columns).")
    files: list[str] = Field(default_factory=list, description="Covariate files joined to the main file.")
    future_rows: int = Field(
        description="Rows after the last target value (empty target cells, future_file rows or later dates of "
        "covariate files); they carry the known covariates. With several series: the fewest of any series."
    )


class SkippedSeries(BaseModel):
    series: str
    reason: str


class SeriesForecast(BaseModel):
    series: str
    horizon: int
    last_observation: str
    total: dict[str, float] = Field(description=TOTAL_DESCRIPTION)
    metadata: ForecastMetadata | None = Field(
        default=None, description="Null when every series has the same metadata: see the top-level `metadata`."
    )


class TotalsBy(BaseModel):
    column: str = Field(description="The first series column.")
    totals: dict[str, dict[str, float]] = Field(
        description="Value of that column -> sum of the totals of its series (same caveat as `total`)."
    )


class CsvForecastOutput(BaseModel):
    model: str
    file: str
    date_column: str
    target_column: str
    series_columns: list[str] | None = Field(description="Columns identifying the series, when the file holds several.")
    frequency: str
    covariates: CovariateInfo
    horizon: int | None = Field(description="Steps forecast; null when each series is forecast over its future rows.")
    last_observation: str = Field(
        description="Timestamp of the last observed target value (the latest of all series when there are several)."
    )
    forecast: list[dict[str, str | float]] = Field(
        description="One row per future step: `date` and one `pNN` column per quantile level (p50 = median). "
        "With several series, each row starts with `series`. Empty when written to output_file."
    )
    output_file: str | None = Field(
        description="File the forecasts were written to: the series columns, the date column and one pNN column "
        "per quantile, one row per series and step."
    )
    rows_written: int | None = None
    total: dict[str, float] | None = Field(description=TOTAL_DESCRIPTION + " Null with several series: see `series`.")
    series_count: int = Field(description="Series forecast.")
    series: list[SeriesForecast] | None = Field(
        description=f"Per-series details, with several series (omitted above {MAX_SERIES} series)."
    )
    overall_total: dict[str, float] | None = Field(
        description="Sum of the totals of every series forecast, with several series (same caveat as `total`)."
    )
    totals_by: TotalsBy | None = Field(
        description="Totals summed by the first series column, when there are several series columns."
    )
    skipped_count: int = Field(default=0, description="Series that could not be forecast.")
    skipped: list[SkippedSeries] = Field(
        default_factory=list, description=f"Skipped series with the reason (the first {MAX_SKIPPED_LISTED})."
    )
    warnings: list[str] = Field(default_factory=list, description=WARNINGS_DESCRIPTION)
    metadata: ForecastMetadata | None = Field(
        description="With several series: their common metadata, or null when it differs (see `series`)."
    )


class ScenarioOutput(BaseModel):
    model: str
    file: str
    date_column: str
    target_column: str
    series: str | None
    frequency: str
    covariates: CovariateInfo
    horizon: int
    last_observation: str
    scenario: dict[str, list[float]] = Field(description="Values the scenario gives each changed column, per step.")
    forecast: list[dict[str, str | float]] = Field(
        description="One row per future step: `date`, `base_pNN` (covariates as in the file), `scenario_pNN` "
        "(with the scenario values) and `delta_p50` = scenario_p50 - base_p50."
    )
    base_total: dict[str, float] = Field(description=TOTAL_DESCRIPTION)
    scenario_total: dict[str, float] = Field(description=TOTAL_DESCRIPTION)
    delta_total_p50: float = Field(
        description="scenario_total.p50 - base_total.p50: the change in expected volume over the horizon. A "
        "difference of medians, not a quantile of the effect; no interval is given for the effect."
    )
    warnings: list[str] = Field(default_factory=list, description=WARNINGS_DESCRIPTION)
    metadata: ForecastMetadata


class Evaluation(BaseModel):
    method: str = Field(description="single_chronological_holdout (1 window) or rolling_origin.")
    test_size: int = Field(description="Steps forecast in each window.")
    windows: int = Field(description="Windows per series.")
    step: int = Field(description="Steps between consecutive forecast origins.")
    test_start: str = Field(description="First date of the oldest window.")
    test_end: str = Field(description="Last date of the most recent window (the last observation).")
    mae: float = Field(description="Mean absolute error of the median (p50) forecast, over all windows.")
    rmse: float = Field(description="Root mean squared error of the median (p50) forecast, over all windows.")
    wape: float | None = Field(
        description="Sum of absolute errors / sum of absolute actual values (0.08 = 8%); scale-free, so it can be "
        "compared across series. Null when every actual value is 0."
    )
    coverage_80: float = Field(description="Share of actual values inside [p10, p90] over all windows; 0.8 is nominal.")


class BaselineOutput(BaseModel):
    method: str = Field(
        description="naive (last value), seasonal_naive (value one season earlier), one of the requested "
        "baselines (mean, moving_average:N, drift, seasonal_growth), or benchmark:<column> for the forecasts "
        "given in benchmark_column."
    )
    mae: float
    rmse: float
    wape: float | None
    mae_improvement_percent: float | None = Field(
        description="(baseline MAE - model MAE) / baseline MAE * 100; null when the baseline MAE is 0."
    )
    rmse_improvement_percent: float | None = Field(
        description="(baseline RMSE - model RMSE) / baseline RMSE * 100; null when the baseline RMSE is 0."
    )


class WindowOutput(BaseModel):
    series: str | None = Field(default=None, description="Series of the window, when there are several.")
    window: int = Field(description="1 = oldest window.")
    context_size: int = Field(description="Observations before the window given to the model.")
    test_start: str
    test_end: str
    mae: float
    rmse: float
    coverage_80: float
    baselines: list[BaselineOutput]


class SeriesEvaluation(BaseModel):
    series: str
    evaluation: Evaluation
    baselines: list[BaselineOutput]


class BacktestOutput(BaseModel):
    model: str
    file: str
    date_column: str
    target_column: str
    series_columns: list[str] | None
    frequency: str
    covariates: CovariateInfo
    assumed_known_in_advance: list[str] = Field(
        description="Known covariates whose actual values over the test windows were given to the model, on "
        "the assumption that they were known in advance. If any was only observed after the fact, the "
        "backtest is optimistic; it should then be passed as a past covariate."
    )
    evaluation: Evaluation = Field(
        description="Pooled over every window (and every series: MAE and RMSE are then in each series' own units, "
        "so larger series weigh more; WAPE is scale-free)."
    )
    baselines: list[BaselineOutput] = Field(description="Baselines scored on the same points as the model.")
    series_count: int = Field(description="Series backtested.")
    series_results: list[SeriesEvaluation] | None = Field(
        description=f"Per-series results, with several series (omitted above {MAX_DETAILED_SERIES} series: see "
        "`series_wape`)."
    )
    series_wape: dict[str, float | None] | None = Field(
        default=None,
        description=f"WAPE of each series, above {MAX_DETAILED_SERIES} series (up to {MAX_TOTAL_GROUPS}), to find "
        "the series the model forecasts worst. The full details go to output_file.",
    )
    windows: list[WindowOutput] = Field(description=f"Per-window results (empty above {MAX_DETAILED_SERIES} series).")
    predictions: list[dict[str, str | int | float]] = Field(
        description="One row per forecast point: window, date, actual, p10, p50, p90 (and series, with several). "
        f"Empty when written to output_file or above {MAX_BACKTEST_POINTS} points."
    )
    output_file: str | None = Field(description="File the predictions were written to.")
    rows_written: int | None = None
    skipped_count: int = Field(default=0, description="Series that could not be backtested.")
    skipped: list[SkippedSeries] = Field(
        default_factory=list, description=f"Skipped series with the reason (the first {MAX_SKIPPED_LISTED})."
    )
    warnings: list[str] = Field(default_factory=list, description=WARNINGS_DESCRIPTION)
    metadata: ForecastMetadata | None = Field(description="Of the most recent window; null with several series.")


def _metadata(result: ForecastResult, series: PreparedSeries | None = None) -> ForecastMetadata:
    return ForecastMetadata(
        input_length=result.input_length,
        context_length=result.context_length,
        missing_values=result.missing_values,
        filled_timestamps=series.filled_timestamps if series is not None else 0,
        device=result.device,
    )


def _covariate_info(loaded: Loaded) -> CovariateInfo:
    series = loaded.series[0].prepared
    rows = min(s.prepared.future_rows for s in loaded.series)
    cov = series.covariates
    if cov is None:
        return CovariateInfo(past=[], known=[], categorical=[], files=loaded.covariate_files, future_rows=rows)
    return CovariateInfo(
        past=list(cov.past),
        known=list(cov.known),
        categorical=[c for c in (*cov.past, *cov.known) if c in cov.categorical],
        files=loaded.covariate_files,
        future_rows=rows,
    )


def _pct(x: float | None) -> float | None:
    return None if x is None else round(x, 2)


def _ratio(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def _baseline_output(b: Baseline) -> BaselineOutput:
    return BaselineOutput(
        method=b.method,
        mae=_num(b.mae),
        rmse=_num(b.rmse),
        wape=_ratio(b.wape),
        mae_improvement_percent=_pct(b.mae_improvement_percent),
        rmse_improvement_percent=_pct(b.rmse_improvement_percent),
    )


def _total(result: ForecastResult, keep: np.ndarray | None = None) -> dict[str, float]:
    """Totals over the horizon, or over the steps in `keep`."""
    return {_column_key(q): _num(float(np.sum(v if keep is None else v[keep]))) for q, v in result.quantiles.items()}


def _sum_totals(totals: list[dict[str, float]]) -> dict[str, float]:
    return {key: _num(sum(t[key] for t in totals)) for key in totals[0]}


# ---------------------------------------------------------------- loading


@dataclass(frozen=True)
class LoadedSeries:
    group: SeriesGroup | None  # None for a file holding one series
    prepared: PreparedSeries

    @property
    def label(self) -> str | None:
        return None if self.group is None else self.group.label


@dataclass(frozen=True)
class Loaded:
    path: str
    series_columns: list[str]
    series: list[LoadedSeries]
    skipped: list[SkippedSeries]
    covariate_files: list[str]
    # With a future file: the timestamps it asks for, by series label ("" for a single series).
    requested: dict[str, pd.DatetimeIndex] | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def single(self) -> bool:
        return not self.series_columns


def _series_columns(series_column: str | list[str] | None) -> list[str]:
    if series_column is None:
        return []
    columns = [series_column] if isinstance(series_column, str) else list(series_column)
    if not columns:
        raise InputError("series_column must name at least one column")
    if len(set(columns)) != len(columns):
        raise InputError(f"series_column lists a column more than once: {columns}")
    return columns


def _skip(skipped: list[SkippedSeries], label: str | None, error: Exception) -> None:
    if label is None:
        raise error if isinstance(error, InputError) else InputError(str(error))
    skipped.append(SkippedSeries(series=label, reason=str(error)))


def _none_left(skipped: list[SkippedSeries]) -> InputError:
    reasons = "; ".join(f"{s.series}: {s.reason}" for s in skipped[:3])
    return InputError(f"none of the {len(skipped)} series could be used ({reasons})")


PASTED = "<csv_text>"


def _source_path(file_path: str | None, csv_text: str | None) -> Path | None:
    """The checked input file, or None for pasted data; exactly one of the two must be given."""
    if (file_path is None) == (csv_text is None):
        raise InputError(
            "pass either file_path (a file on the user's computer) or csv_text (the CSV content itself), not both"
            if file_path is not None
            else "pass file_path (a file on the user's computer) or csv_text (the CSV content itself)"
        )
    if file_path is None:
        return None
    cfg = settings()
    return resolve_data_path(file_path, cfg.allowed_dirs, cfg.max_file_bytes)


def _aggregate_columns(aggregate_by: str | list[str] | None, series_column: str | list[str] | None) -> list[str] | None:
    if aggregate_by is None:
        return None
    columns = [aggregate_by] if isinstance(aggregate_by, str) else list(aggregate_by)
    if len(set(columns)) != len(columns):
        raise InputError(f"aggregate_by lists a column more than once: {columns}")
    if series_column is not None and _series_columns(series_column) != columns:
        raise InputError("aggregate_by already sets the series; leave series_column out")
    return columns


def _load_file(
    file_path: str | None,
    *,
    csv_text: str | None = None,
    aggregate_by: str | list[str] | None = None,
    sheet: str | None = None,
    date_column: str | None = None,
    target_column: str | None = None,
    frequency: str | None = None,
    past_covariates: list[str] | None = None,
    known_covariates: list[str] | None = None,
    fill_missing_dates: bool = False,
    series_column: str | list[str] | None = None,
    series_ids: list[str] | None = None,
    covariates_files: list[str] | None = None,
    future_file: str | None = None,
    max_series: int = MAX_SERIES,
    sum_columns: tuple[str, ...] = (),
) -> Loaded:
    """Read the main file (plus its future and covariate files) and prepare its series.

    With series columns, a series that cannot be prepared is skipped with its reason; a
    file holding one series raises instead.
    """
    cfg = settings()
    path = _source_path(file_path, csv_text)
    if path is None:
        if sheet is not None:
            raise InputError("sheet is only used for Excel files")
        assert csv_text is not None
        df = read_csv_text(csv_text, cfg.max_file_bytes)
    else:
        df = read_table(path, sheet)
    aggregated = _aggregate_columns(aggregate_by, series_column)
    columns = aggregated if aggregated is not None else _series_columns(series_column)
    past, known = list(past_covariates or []), list(known_covariates or [])
    if set(columns) & {date_column, target_column, *past, *known}:
        raise InputError(f"a series column {columns} cannot also be the date, target or a covariate")
    if not columns and series_ids is not None:
        raise InputError("series_ids needs series_column")
    for column in columns:
        if column not in df.columns:
            raise InputError(f"series column {column!r} not found; available columns: {list(map(str, df.columns))}")

    # Parse the dates once for the whole file rather than once per series.
    date_col = detect_date_column(df, date_column, exclude=columns)
    df[date_col] = parse_date_column(df, date_col)

    if future_file is not None:
        future_path = resolve_data_path(future_file, cfg.allowed_dirs, cfg.max_file_bytes)
        future = read_table(future_path)
        missing = [c for c in (date_col, *columns) if c not in future.columns]
        if missing:
            raise InputError(f"{future_path.name} lacks the column(s) {missing} of the main file")
        if target_column is not None and target_column in future.columns and future[target_column].notna().any():
            raise InputError(
                f"{future_path.name} has values in the target column {target_column!r}; "
                "a future file holds the rows to forecast"
            )
        future[date_col] = parse_date_column(future, date_col)
        labels = series_labels(future, columns) if columns else pd.Series("", index=future.index)
        requested = {
            str(label): pd.DatetimeIndex(rows)
            for label, rows in future[date_col].groupby(labels.to_numpy(), sort=False)
        }
        df = pd.concat([df, future], ignore_index=True)

    else:
        requested = None

    summed_over: list[str] = []
    if aggregated is not None:
        # Sum the rows of each group (e.g. every department of a store) into one series per group.
        target_column = detect_target_column(df, date_col, target_column, exclude=[*columns, *past, *known])
        in_file = [c for c in (*past, *known) if c in df.columns]
        summed_over = [str(c) for c in df.columns if c not in {*columns, date_col, target_column, *in_file}]
        df = aggregate_rows(
            df, date_col, target_column, columns, in_file, summed=[c for c in sum_columns if c in df.columns]
        )

    tables: list[CovariateTable] = []
    warnings: list[str] = []
    if covariates_files and not past and not known:
        raise InputError("covariates_files needs the columns to use, in past_covariates or known_covariates")
    for name in covariates_files or []:
        cov_path = resolve_data_path(name, cfg.allowed_dirs, cfg.max_file_bytes)
        try:
            table = covariate_table(
                read_table(cov_path),
                cov_path.name,
                date_col,
                columns,
                [*past, *known],
                average_over=summed_over,
                summed=sum_columns,
            )
        except StaticCovariateFile as e:
            # Attributes such as a store's size or type: not usable, but no reason to fail the forecast.
            dropped = [c for c in e.columns if c not in df.columns]
            past = [c for c in past if c not in dropped]
            known = [c for c in known if c not in dropped]
            listed = f" (columns {dropped} were not used)" if dropped else ""
            warnings.append(
                f"{e.name} was ignored: it has no dates, so its values never change for a series{listed}. Such "
                "attributes cannot inform a forecast made from each series' own history."
            )
            continue
        if not table.columns:
            raise InputError(f"{cov_path.name} has none of the covariate columns {[*past, *known]}")
        averaged = [c for c in table.columns if c not in sum_columns]
        if table.averaged_over:
            level = f"per {', '.join(columns)}" if columns else "for the total"
            over = ", ".join(table.averaged_over)
            if averaged:
                warnings.append(
                    f"{cov_path.name} has one row per {over}; its covariates {averaged} were averaged over {over} "
                    f"to match the series forecast {level}."
                )
            for name in (c for c in table.columns if c in sum_columns):
                warnings.append(f"{name!r} from {cov_path.name} was added up over {over}, like the target.")
        tables.append(table)

    if columns:
        frequency = file_frequency(df[date_col], frequency)
    groups: list[SeriesGroup | None] = list(split_series(df, columns, series_ids, max_series)) if columns else [None]
    loaded: list[LoadedSeries] = []
    skipped: list[SkippedSeries] = []
    for group in groups:
        extra = None
        for table in tables:
            rows = table.lookup(group.keys if group is not None else {})
            if rows is not None:
                extra = rows if extra is None else extra.combine_first(rows)
        try:
            prepared = prepare_series(
                group.frame if group is not None else df,
                date_col,
                target_column,
                frequency,
                past,
                known,
                fill_missing_dates,
                exclude=columns,
                extra=extra,
                # the future file says which rows to forecast; covariate files only fill them
                extend=future_file is None,
            )
        except InputError as e:
            _skip(skipped, group.label if group is not None else None, e)
            continue
        loaded.append(LoadedSeries(group, prepared))
    if not loaded:
        raise _none_left(skipped)
    kinds = {(s.prepared.date_column, s.prepared.target_column, s.prepared.frequency) for s in loaded}
    if len(kinds) > 1:
        raise InputError(f"the series differ in date column, target or frequency: {sorted(kinds)}; pass them")
    source = PASTED if path is None else str(path)
    return Loaded(source, columns, loaded, skipped, [Path(t.name).name for t in tables], requested, warnings)


def _skipped_warning(skipped: list[SkippedSeries], used: int, what: str) -> list[str]:
    """Say plainly that totals or pooled metrics cover only part of the file."""
    if not skipped:
        return []
    message = (
        f"{len(skipped)} of {len(skipped) + used} series were skipped (see skipped): the {what} cover only the "
        f"{used} series that could be used, not the whole file. Tell the user before quoting any total."
    )
    if sum("covariate" in s.reason for s in skipped) > len(skipped) / 2:
        message += (
            " Most were skipped because of a covariate (see the reasons): retrying without that covariate "
            "would keep them."
        )
    return [message]


def _write_csv(path: Path, frame: pd.DataFrame) -> int:
    # "x": fails rather than overwrite a file created since the path was checked
    with open(path, "x", newline="", encoding="utf-8") as f:
        frame.to_csv(f, index=False)
    return len(frame)


def _output_path(output_file: str | None, file_path: str | None, csv_text: str | None) -> Path | None:
    """The output file; a bare name lands next to the input file (pasted data: the first allowed folder)."""
    if output_file is None:
        return None
    source = _source_path(file_path, csv_text)
    return resolve_output_path(output_file, settings().allowed_dirs, None if source is None else source.parent)


# ---------------------------------------------------------------- tools

Frequency = Annotated[
    str | None,
    Field(
        description=f"One of {', '.join(FREQUENCIES)} (business_daily = Monday to Friday). "
        "Default: inferred from the timestamps."
    ),
]
FilePath = Annotated[
    str | None,
    Field(
        description="Absolute path of a .csv, .xlsx or .xlsm file on the user's computer, e.g. the path of a "
        "file they attached or named. Read where it is: never copy or move it first. Allowed folders: by "
        "default the user's Downloads, Desktop and Documents. CSV delimiters are detected; semicolon files "
        "are read with decimal commas and day-first dates. Give file_path or csv_text."
    ),
]
CsvText = Annotated[
    str | None,
    Field(
        description="The CSV content itself, header line first, when there is no file path (e.g. a small file "
        "attached to a chat whose content you can see). Pass the data exactly as given, every row: never "
        "summarize, round or retype it. For large files ask for the file's location and use file_path."
    ),
]
AggregateBy = Annotated[
    str | list[str] | None,
    Field(
        description="Add up the target over the rows that share a date and these columns, and forecast one "
        'series per group: e.g. "Store" turns per-store-and-department sales into per-store sales. An empty '
        "list [] adds up every row of a date into one total series. Replaces series_column. Covariate "
        "columns keep their first value per group, so they must describe the group as a whole (a holiday "
        "flag per date). Only use it when the user wants the forecast at that level."
    ),
]
Sheet = Annotated[str | None, Field(description="Excel sheet to read. Default: the first sheet.")]
DateColumn = Annotated[str | None, Field(description="Timestamp column. Default: detected.")]
TargetColumn = Annotated[str | None, Field(description="Numeric column to forecast. Default: detected.")]
SeriesColumn = Annotated[
    str | list[str] | None,
    Field(
        description="Column(s) identifying each series when the file stacks several (long format), e.g. "
        '"store" or ["store", "dept"]. Each series is forecast separately, from its own history. Series ids '
        f'join the values with {SERIES_ID_SEPARATOR!r}, e.g. "1{SERIES_ID_SEPARATOR}12".'
    ),
]
SeriesIds = Annotated[
    list[str] | None,
    Field(
        description=f"Only these series ids. Default: every series (at most {MAX_SERIES}, or "
        f"{MAX_SERIES_TO_FILE} with output_file)."
    ),
]
FillMissingDates = Annotated[
    bool,
    Field(
        description="Insert timestamps missing from the regular grid as missing values (reported as "
        "metadata.filled_timestamps) instead of rejecting the series. No value is invented: Timestral masks "
        "them. Use it only when the user accepts that those dates are unknown, not zero."
    ),
]
PastCovariates = Annotated[
    list[str] | None,
    Field(
        description="Columns measured as time goes by, so known only up to each forecast origin (e.g. "
        "actual weather, traffic, observed prices). Use this when in doubt. Numeric columns are used as "
        "numbers, others as categories. They may come from covariates_files."
    ),
]
KnownCovariates = Annotated[
    list[str] | None,
    Field(
        description="ONLY columns whose future values are decided or published in advance (planned "
        "promotions, holidays, opening hours, a price list). Never measured values such as actual "
        "weather: those go in past_covariates. If unsure which kind a column is, ask the user before "
        "calling. For forecasts they need a value for every future step: future rows of the file (empty "
        "target), a future_file, or later dates in covariates_files."
    ),
]
CovariatesFiles = Annotated[
    list[str] | None,
    Field(
        description="Other files holding covariate columns, by date and optionally by some series columns (e.g. "
        "features.csv with store and date). Joined on the date and the series columns they share with the "
        "main file; they fill cells the main file leaves empty, and their dates after a series' last row "
        "become its future rows. Name the columns to use in past_covariates / known_covariates."
    ),
]
FutureFile = Annotated[
    str | None,
    Field(
        description="File with the rows to forecast (e.g. test.csv): the date and series columns, no target "
        "values, plus any known covariates. Appended to the main file; without horizon, each series is "
        "forecast over its future rows."
    ),
]
OutputFile = Annotated[
    str | None,
    Field(
        description="Name of a new .csv file to write the results to instead of returning them; needed above "
        "100 series or 20,000 forecast rows. A bare name such as forecast.csv is created next to the input "
        "file. Existing files are never overwritten: a free name such as 'forecast (2).csv' is used, so tell "
        "the user the returned output_file."
    ),
]


@mcp.tool(annotations=_READ_ONLY)
def forecast(
    values: Annotated[list[float], Field(description="Ordered, evenly spaced observations, oldest first.")],
    horizon: Horizon,
    quantiles: QuantileLevels = None,
    model: ModelId = None,
) -> ForecastOutput:
    """Forecast the next `horizon` values of a univariate time series with Timestral.

    Returns quantile forecasts per future step (p50 = median; p10-p90 = 80% prediction interval)
    and their totals over the horizon.
    """
    with _tool_errors():
        cfg = settings()
        model_id = model_mod.resolve_model_id(model, cfg.model)
        levels = validate_quantiles(quantiles)
        result = run_forecast(validate_values(values), validate_horizon(horizon), model_id, cfg.device, levels)
        return ForecastOutput(
            model=result.model,
            horizon=result.horizon,
            quantiles={quantile_key(q): [_num(v) for v in vals] for q, vals in result.quantiles.items()},
            total={quantile_key(q): _num(float(np.sum(v))) for q, v in result.quantiles.items()},
            metadata=_metadata(result),
        )


@dataclass(frozen=True)
class _Job:
    series: LoadedSeries
    values: np.ndarray
    horizon: int
    keep: np.ndarray  # steps to report: all, or those the future file asks for


def _forecast_jobs(jobs: list[_Job], model_id: str, device: str, levels: tuple[float, ...]) -> list[ForecastResult]:
    """Forecast every job, batching those of equal length, horizon and covariate columns."""
    results: list[ForecastResult | None] = [None] * len(jobs)
    batches: dict[tuple[int, int, tuple[tuple[str, ...], tuple[str, ...]]], list[int]] = {}
    for i, job in enumerate(jobs):
        key = (job.values.size, job.horizon, covariate_signature(job.series.prepared.covariates))
        batches.setdefault(key, []).append(i)
    for (_, horizon, _), members in batches.items():
        items = [(jobs[i].values, jobs[i].series.prepared.covariates) for i in members]
        for i, result in zip(members, run_forecast_batch(items, horizon, model_id, device, levels), strict=True):
            results[i] = result
    return [r for r in results if r is not None]


def _plan_jobs(loaded: Loaded, horizon: int | None, skipped: list[SkippedSeries]) -> list[_Job]:
    """Validate every series and its horizon before running the model; unusable series are skipped.

    With a future file, series it does not mention are not forecast, and only the steps it asks
    for are reported (the steps between the last observation and those are forecast too).
    """
    jobs = []
    for series in loaded.series:
        prepared = series.prepared
        wanted = None
        if loaded.requested is not None:
            wanted = loaded.requested.get(series.label or "")
            if wanted is None:
                continue
        try:
            steps = horizon if horizon is not None else prepared.future_rows
            if steps == 0:
                raise InputError("no future rows to forecast; pass horizon or add future rows (future_file)")
            validate_horizon(steps)
            if prepared.values.size == 0:
                raise InputError("no history: the series only has future rows")
            values = validate_values(prepared.values, allow_missing=True)
            if prepared.covariates is not None:
                prepared.covariates.window(values.size, steps)
        except InputError as e:
            _skip(skipped, series.label, e)
            continue
        n = len(prepared.dates)
        keep = np.ones(steps, dtype=bool)
        if wanted is not None:
            keep = regular_grid(prepared.dates[0], n + steps, prepared.frequency)[n:].isin(wanted)
        jobs.append(_Job(series, values, steps, keep))
    if not jobs:
        raise _none_left(skipped)
    return jobs


@mcp.tool(annotations=_MAY_WRITE)
def forecast_csv(
    file_path: FilePath = None,
    *,
    csv_text: CsvText = None,
    horizon: Annotated[
        int | None,
        Field(
            ge=1,
            le=MAX_HORIZON,
            description="Number of future steps to forecast. Default: each series' future rows (future_file, "
            "empty target cells at the end of the file or later dates of covariates_files).",
        ),
    ] = None,
    target_column: TargetColumn = None,
    date_column: DateColumn = None,
    series_column: SeriesColumn = None,
    aggregate_by: AggregateBy = None,
    series_ids: SeriesIds = None,
    frequency: Frequency = None,
    fill_missing_dates: FillMissingDates = False,
    past_covariates: PastCovariates = None,
    known_covariates: KnownCovariates = None,
    covariates_files: CovariatesFiles = None,
    future_file: FutureFile = None,
    output_file: OutputFile = None,
    sheet: Sheet = None,
    quantiles: QuantileLevels = None,
    model: ModelId = None,
    ctx: Context | None = None,
) -> CsvForecastOutput:
    """Forecast a column of a local CSV or Excel file with Timestral, returning dated quantile forecasts.

    The timestamps must be a complete regular series (no gaps or duplicates, unless
    fill_missing_dates); rows are sorted chronologically. Empty target cells in the history are
    passed to the model as missing values and counted in metadata.missing_values. Optional
    covariate columns condition the forecast, from the file itself or from covariates_files;
    they are not forecast themselves. With series_column, every series of the file is
    forecast separately (series that cannot be are skipped and listed). Totals over the
    horizon are returned: use them rather than adding up the rows. With output_file, the
    forecasts are written to that file and only summarized in the response.
    """
    _log_client_roots(ctx)
    with _tool_errors():
        cfg = settings()
        model_id = model_mod.resolve_model_id(model, cfg.model)
        levels = validate_quantiles(quantiles)
        if horizon is not None:
            validate_horizon(horizon)
        out_path = _output_path(output_file, file_path, csv_text)
        loaded = _load_file(
            file_path,
            csv_text=csv_text,
            aggregate_by=aggregate_by,
            sheet=sheet,
            date_column=date_column,
            target_column=target_column,
            frequency=frequency,
            past_covariates=past_covariates,
            known_covariates=known_covariates,
            fill_missing_dates=fill_missing_dates,
            series_column=series_column,
            series_ids=series_ids,
            covariates_files=covariates_files,
            future_file=future_file,
            max_series=MAX_SERIES if out_path is None else MAX_SERIES_TO_FILE,
        )
        skipped = list(loaded.skipped)
        jobs = _plan_jobs(loaded, horizon, skipped)
        points = sum(int(job.keep.sum()) for job in jobs)
        if out_path is None and points > MAX_OUTPUT_POINTS:
            raise InputError(
                f"{len(jobs)} series give {points} forecast rows; at most {MAX_OUTPUT_POINTS} are returned. "
                "Pass output_file to write them to a file, or use fewer series_ids or a shorter horizon."
            )
        results = _forecast_jobs(jobs, model_id, cfg.device, levels)

        first = jobs[0].series.prepared
        date_col = first.date_column
        rows: list[dict[str, str | float]] = []
        summaries: list[SeriesForecast] = []
        for job, result in zip(jobs, results, strict=True):
            label, group = job.series.label, job.series.group
            for i, date in enumerate(future_dates(job.series.prepared, job.horizon)):
                if not job.keep[i]:
                    continue
                if out_path is not None:
                    row: dict[str, str | float] = dict(group.keys) if group is not None else {}
                    row[date_col] = date
                else:
                    row = {"series": label} if label is not None else {}
                    row["date"] = date
                row.update({_column_key(q): _num(v[i]) for q, v in result.quantiles.items()})
                rows.append(row)
            summaries.append(
                SeriesForecast(
                    series=label or "",
                    horizon=job.horizon,
                    last_observation=dates_as_strings(job.series.prepared.dates[-1:])[0],
                    total=_total(result, job.keep),
                    metadata=_metadata(result, job.series.prepared),
                )
            )
        written = _write_csv(out_path, pd.DataFrame(rows)) if out_path is not None else None

        # Metadata repeated identically for every series only bloats the response: give it once.
        common = summaries[0].metadata
        if loaded.single:
            pass
        elif any(s.metadata != common for s in summaries):
            common = None
        else:
            for summary in summaries:
                summary.metadata = None

        totals_by = None
        if len(loaded.series_columns) > 1:
            by: dict[str, list[dict[str, float]]] = {}
            for job, summary in zip(jobs, summaries, strict=True):
                assert job.series.group is not None
                by.setdefault(job.series.group.keys[loaded.series_columns[0]], []).append(summary.total)
            if len(by) <= MAX_TOTAL_GROUPS:
                totals_by = TotalsBy(column=loaded.series_columns[0], totals={k: _sum_totals(v) for k, v in by.items()})
        single = loaded.single
        return CsvForecastOutput(
            model=model_id,
            file=loaded.path,
            date_column=date_col,
            target_column=first.target_column,
            series_columns=None if single else loaded.series_columns,
            frequency=first.frequency,
            covariates=_covariate_info(loaded),
            horizon=horizon if horizon is not None else (jobs[0].horizon if single else None),
            last_observation=max(s.last_observation for s in summaries),
            forecast=[] if out_path is not None else rows,
            output_file=None if out_path is None else str(out_path),
            rows_written=written,
            total=summaries[0].total if single else None,
            series_count=len(summaries),
            series=None if single or len(summaries) > MAX_SERIES else summaries,
            overall_total=None if single else _sum_totals([s.total for s in summaries]),
            totals_by=totals_by,
            skipped_count=len(skipped),
            skipped=skipped[:MAX_SKIPPED_LISTED],
            warnings=[*loaded.warnings, *_skipped_warning(skipped, len(jobs), "overall_total and totals_by")],
            metadata=common,
        )


ScenarioValues = Annotated[
    dict[str, float | list[float]],
    Field(
        description="New future values for known-covariate columns, e.g. {\"promo\": 0} for 'no promotion' or "
        '{"price": [9.9, 9.9, 8.5]} step by step. A single number applies to every step of the horizon; a '
        "list must have one value per step. These columns are used as known covariates."
    ),
]


@mcp.tool(annotations=_READ_ONLY)
def compare_scenarios(
    file_path: FilePath = None,
    *,
    scenario: ScenarioValues,
    csv_text: CsvText = None,
    horizon: Annotated[
        int | None,
        Field(ge=1, le=MAX_HORIZON, description="Number of future steps. Default: the series' future rows."),
    ] = None,
    target_column: TargetColumn = None,
    date_column: DateColumn = None,
    series_column: SeriesColumn = None,
    aggregate_by: AggregateBy = None,
    series_id: Annotated[
        str | None, Field(description="The series to use, when series_column or aggregate_by is given.")
    ] = None,
    frequency: Frequency = None,
    fill_missing_dates: FillMissingDates = False,
    past_covariates: PastCovariates = None,
    known_covariates: KnownCovariates = None,
    covariates_files: CovariatesFiles = None,
    future_file: FutureFile = None,
    sheet: Sheet = None,
    quantiles: QuantileLevels = None,
    model: ModelId = None,
) -> ScenarioOutput:
    """Forecast one series twice: with the known covariates as planned in the file (base) and
    with some of them changed (scenario), and report the difference.

    This is how to estimate the effect of a covariate, e.g. what a planned promotion adds:
    scenario {"promo": 0} removes it. Comparing forecasts of two different files or series
    does not measure an effect. The difference is the model's response to the covariate,
    learned from this history; it is not a causal estimate.
    """
    with _tool_errors():
        cfg = settings()
        model_id = model_mod.resolve_model_id(model, cfg.model)
        levels = validate_quantiles(quantiles)
        if 0.5 not in levels:
            levels = tuple(sorted({*levels, 0.5}))
        if not scenario:
            raise InputError("scenario must change at least one column")
        several = series_column is not None or bool(aggregate_by)
        if not several and series_id is not None:
            raise InputError("series_id needs series_column or aggregate_by")
        if several and series_id is None:
            raise InputError("compare_scenarios uses one series: pass series_id")
        clash = sorted(set(scenario) & set(past_covariates or []))
        if clash:
            raise InputError(f"scenario columns {clash} are past covariates; a scenario changes known covariates")
        loaded = _load_file(
            file_path,
            csv_text=csv_text,
            aggregate_by=aggregate_by,
            sheet=sheet,
            date_column=date_column,
            target_column=target_column,
            frequency=frequency,
            past_covariates=past_covariates,
            known_covariates=list(dict.fromkeys([*(known_covariates or []), *scenario])),
            fill_missing_dates=fill_missing_dates,
            series_column=series_column,
            series_ids=None if series_id is None else [series_id],
            covariates_files=covariates_files,
            future_file=future_file,
        )
        if loaded.skipped:
            raise InputError(f"series {loaded.skipped[0].series!r}: {loaded.skipped[0].reason}")
        series = loaded.series[0].prepared
        steps = horizon if horizon is not None else series.future_rows
        if steps == 0:
            raise InputError("no future rows to forecast; pass horizon or add future rows (future_file)")
        validate_horizon(steps)
        covariates: Covariates | None = series.covariates
        assert covariates is not None
        future: dict[str, np.ndarray] = {}
        for name, raw in scenario.items():
            arr = np.full(steps, float(raw)) if isinstance(raw, int | float) else np.asarray(raw, dtype=np.float64)
            if arr.size != steps:
                raise InputError(f"scenario {name!r} has {arr.size} values; it needs one per step ({steps})")
            if not np.isfinite(arr).all():
                raise InputError(f"scenario {name!r} must contain finite numbers")
            future[name] = arr
        values = validate_values(series.values, allow_missing=True)
        changed = covariates.with_future(future, values.size)
        base, alt = run_forecast_batch([(values, covariates), (values, changed)], steps, model_id, cfg.device, levels)

        rows: list[dict[str, str | float]] = []
        for i, date in enumerate(future_dates(series, steps)):
            row: dict[str, str | float] = {"date": date}
            row.update({"base_" + _column_key(q): _num(v[i]) for q, v in base.quantiles.items()})
            row.update({"scenario_" + _column_key(q): _num(v[i]) for q, v in alt.quantiles.items()})
            row["delta_p50"] = _num(alt.quantiles[0.5][i] - base.quantiles[0.5][i])
            rows.append(row)
        base_total, alt_total = _total(base), _total(alt)
        return ScenarioOutput(
            model=model_id,
            file=loaded.path,
            date_column=series.date_column,
            target_column=series.target_column,
            series=loaded.series[0].label,
            frequency=series.frequency,
            covariates=_covariate_info(loaded),
            horizon=steps,
            last_observation=dates_as_strings(series.dates[-1:])[0],
            scenario={name: [_num(v) for v in arr] for name, arr in future.items()},
            forecast=rows,
            base_total=base_total,
            scenario_total=alt_total,
            delta_total_p50=_num(alt_total["p50"] - base_total["p50"]),
            warnings=loaded.warnings,
            metadata=_metadata(base, series),
        )


def _take_benchmark(loaded: Loaded, column: str) -> tuple[Loaded, list[tuple[str, np.ndarray] | None]]:
    """Take the benchmark column out of each series' covariates, as (name, values aligned with the dates)."""
    name = f"benchmark:{column}"
    series: list[LoadedSeries] = []
    benchmarks: list[tuple[str, np.ndarray] | None] = []
    for s in loaded.series:
        cov = s.prepared.covariates
        assert cov is not None and column in cov.past
        if column in cov.categorical:
            raise InputError(f"benchmark_column {column!r} must hold numbers (forecasts), not text")
        past = {k: v for k, v in cov.past.items() if k != column}
        rest = None
        if past or cov.known:
            rest = dataclasses.replace(cov, past=past, categorical=cov.categorical - {column})
        series.append(LoadedSeries(s.group, dataclasses.replace(s.prepared, covariates=rest)))
        benchmarks.append((name, cov.past[column]))
    return dataclasses.replace(loaded, series=series), benchmarks


def _evaluation(result: BacktestResult, starts: list[str], ends: list[str]) -> Evaluation:
    per_series = len(result.windows) // len(starts)
    return Evaluation(
        method="single_chronological_holdout" if per_series == 1 else "rolling_origin",
        test_size=result.test_size,
        windows=per_series,
        step=result.step,
        test_start=min(starts),
        test_end=max(ends),
        mae=_num(result.mae),
        rmse=_num(result.rmse),
        wape=_ratio(result.wape),
        coverage_80=_num(result.coverage_80),
    )


@mcp.tool(annotations=_MAY_WRITE)
def backtest(
    file_path: FilePath = None,
    *,
    test_size: Annotated[int, Field(ge=1, le=MAX_HORIZON, description="Steps forecast in each window.")],
    csv_text: CsvText = None,
    target_column: TargetColumn = None,
    date_column: DateColumn = None,
    series_column: SeriesColumn = None,
    aggregate_by: AggregateBy = None,
    series_ids: SeriesIds = None,
    frequency: Frequency = None,
    fill_missing_dates: FillMissingDates = False,
    seasonal_period: Annotated[
        int | None,
        Field(
            ge=1,
            description="Season length in steps (e.g. 7 for daily data with a weekly cycle). "
            "When given, a seasonal naive baseline is added. Never assumed.",
        ),
    ] = None,
    windows: Annotated[
        int,
        Field(
            ge=1,
            le=MAX_WINDOWS,
            description="Number of forecast origins (rolling-origin evaluation). 1 = a single holdout of the "
            "last test_size observations; each extra window steps further back into the history.",
        ),
    ] = 1,
    step: Annotated[
        int | None,
        Field(ge=1, description="Steps between consecutive origins. Default: test_size (windows do not overlap)."),
    ] = None,
    past_covariates: PastCovariates = None,
    known_covariates: KnownCovariates = None,
    covariates_files: CovariatesFiles = None,
    output_file: Annotated[
        str | None,
        Field(
            description="Name of a new .csv file to write every prediction to (actual, p10, p50, p90 per series, "
            "window, date). A bare name is created next to the input file; existing files are never overwritten."
        ),
    ] = None,
    baselines: Annotated[
        list[str] | None,
        Field(
            description="Extra methods to compare with, computed on exactly the same windows: 'mean' (average of "
            "the whole history), 'moving_average:N' (average of the last N periods, e.g. 'moving_average:4'), "
            "'drift' (the trend from the first to the last value, extended), 'seasonal_growth' (same period one "
            "season earlier x the growth of the last season; needs seasonal_period). naive is always included, "
            "seasonal_naive whenever seasonal_period is given. Any other method cannot be computed here: if the "
            "user has such forecasts, use benchmark_column."
        ),
    ] = None,
    benchmark_column: Annotated[
        str | None,
        Field(
            description="Column holding forecasts made by other means for each past date, e.g. the user's own "
            "forecasts or those of their current tool, in the main file or in a covariates_files file. They are "
            "scored like a baseline, on the same points as Timestral, and must cover every date of the test "
            "windows. With aggregate_by they are added up like the target."
        ),
    ] = None,
    sheet: Sheet = None,
    model: ModelId = None,
) -> BacktestOutput:
    """Evaluate Timestral on the most recent observations of a CSV or Excel series.

    For each window the model sees only the observations before its origin (no leakage),
    forecasts the next test_size steps, and is scored by MAE/RMSE/WAPE of its median and by 80%
    interval coverage, alongside a naive last-value baseline (and a seasonal naive one when
    seasonal_period is given, plus any requested `baselines` and the user's own forecasts in
    `benchmark_column`). The most recent window ends at the last observation. Metrics
    are reported per window and pooled over all windows. Known covariates are used over the
    test windows as if known in advance. With series_column, every series is backtested
    (series too short for the windows are skipped and listed) and results are pooled; WAPE
    compares series of different sizes. Backtests do not establish general reliability.
    """
    with _tool_errors():
        cfg = settings()
        model_id = model_mod.resolve_model_id(model, cfg.model)
        baseline_forecasters(baselines, seasonal_period)  # fail fast on an unknown baseline
        not_benchmarks = {*(past_covariates or []), *(known_covariates or []), target_column}
        if benchmark_column is not None and benchmark_column in not_benchmarks:
            raise InputError("benchmark_column must be a column of other forecasts, not the target or a covariate")
        out_path = _output_path(output_file, file_path, csv_text)
        loaded = _load_file(
            file_path,
            csv_text=csv_text,
            aggregate_by=aggregate_by,
            sheet=sheet,
            date_column=date_column,
            target_column=target_column,
            frequency=frequency,
            # the benchmark rides along as a past covariate to be aligned with the dates, then is taken out
            past_covariates=[*(past_covariates or []), *([benchmark_column] if benchmark_column else [])],
            known_covariates=known_covariates,
            fill_missing_dates=fill_missing_dates,
            series_column=series_column,
            series_ids=series_ids,
            covariates_files=covariates_files,
            max_series=MAX_SERIES_TO_FILE,
            sum_columns=(benchmark_column,) if benchmark_column else (),
        )
        benchmarks = None
        if benchmark_column is not None:
            loaded, benchmarks = _take_benchmark(loaded, benchmark_column)
        skipped = list(loaded.skipped)
        outcomes = run_backtests(
            [(s.prepared.values, s.prepared.covariates) for s in loaded.series],
            test_size,
            model_id,
            cfg.device,
            seasonal_period,
            windows,
            step,
            baselines,
            benchmarks,
        )
        kept: list[tuple[LoadedSeries, BacktestResult]] = []
        for series, outcome in zip(loaded.series, outcomes, strict=True):
            if isinstance(outcome, InputError):
                _skip(skipped, series.label, outcome)
            else:
                kept.append((series, outcome))
        if not kept:
            raise _none_left(skipped)

        detailed = len(kept) <= MAX_DETAILED_SERIES
        points = sum(len(r.windows) * test_size for _, r in kept)
        inline = out_path is None and points <= min(MAX_INLINE_PREDICTIONS, MAX_BACKTEST_POINTS)
        series_results: list[SeriesEvaluation] = []
        window_outputs: list[WindowOutput] = []
        predictions: list[dict[str, str | int | float]] = []
        starts: list[str] = []
        ends: list[str] = []
        for series, result in kept:
            label, group = series.label, series.group
            s_starts, s_ends = [], []
            for number, w in enumerate(result.windows, start=1):
                dates = dates_as_strings(series.prepared.dates[w.origin : w.origin + test_size])
                s_starts.append(dates[0])
                s_ends.append(dates[-1])
                if detailed:
                    window_outputs.append(
                        WindowOutput(
                            series=label,
                            window=number,
                            context_size=w.origin,
                            test_start=dates[0],
                            test_end=dates[-1],
                            mae=_num(w.mae),
                            rmse=_num(w.rmse),
                            coverage_80=_num(w.coverage_80),
                            baselines=[_baseline_output(b) for b in w.baselines],
                        )
                    )
                if inline or out_path is not None:
                    q = w.forecast.quantiles
                    for d, a, lo, mid, hi in zip(dates, w.actual, q[0.1], q[0.5], q[0.9], strict=True):
                        if out_path is not None:
                            row: dict[str, str | int | float] = dict(group.keys) if group is not None else {}
                        else:
                            row = {"series": label} if label is not None else {}
                        row.update({"window": number, "date": d, "actual": _num(a)})
                        row.update({"p10": _num(lo), "p50": _num(mid), "p90": _num(hi)})
                        predictions.append(row)
            starts.append(s_starts[0])
            ends.append(s_ends[-1])
            if label is not None and detailed:
                series_results.append(
                    SeriesEvaluation(
                        series=label,
                        evaluation=_evaluation(result, s_starts[:1], s_ends[-1:]),
                        baselines=[_baseline_output(b) for b in result.baselines],
                    )
                )
        written = _write_csv(out_path, pd.DataFrame(predictions)) if out_path is not None else None

        pooled = pool([r for _, r in kept])
        first = kept[0][0].prepared
        single = loaded.single
        return BacktestOutput(
            model=model_id,
            file=loaded.path,
            date_column=first.date_column,
            target_column=first.target_column,
            series_columns=None if single else loaded.series_columns,
            frequency=first.frequency,
            covariates=_covariate_info(loaded),
            assumed_known_in_advance=list(first.covariates.known) if first.covariates else [],
            evaluation=_evaluation(pooled, starts, ends),
            baselines=[_baseline_output(b) for b in pooled.baselines],
            series_count=len(kept),
            series_results=None if single or not detailed else series_results,
            series_wape=(
                {s.label or "": _ratio(r.wape) for s, r in kept}
                if not single and not detailed and len(kept) <= MAX_TOTAL_GROUPS
                else None
            ),
            windows=window_outputs,
            predictions=predictions if inline else [],
            output_file=None if out_path is None else str(out_path),
            rows_written=written,
            skipped_count=len(skipped),
            skipped=skipped[:MAX_SKIPPED_LISTED],
            warnings=[*loaded.warnings, *_skipped_warning(skipped, len(kept), "pooled evaluation and baselines")],
            metadata=_metadata(kept[0][1].windows[-1].forecast, first) if single else None,
        )


# ---------------------------------------------------------------- entry point


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="timestral-mcp", description="MCP server (STDIO) for local Timestral time-series forecasting."
    )
    parser.add_argument("--version", action="version", version=f"timestral-mcp {__version__}")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the configured checkpoint (TIMESTRAL_MODEL) to the Hugging Face cache and exit",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    try:
        cfg = settings()
    except model_mod.ConfigurationError as e:
        logger.error("Invalid configuration: %s", e)
        sys.exit(2)

    if args.download:
        try:
            path = model_mod.download(cfg.model)
        except Exception as e:
            logger.error("Download of %s failed: %s", cfg.model, e)
            sys.exit(1)
        print(f"{cfg.model} is available at {path}", file=sys.stderr)
        return

    logger.info(
        "Starting timestral-mcp %s (model=%s, device=%s, allowed_dirs=%s)",
        __version__,
        cfg.model,
        cfg.device,
        [str(d) for d in cfg.allowed_dirs],
    )
    mcp.run()


if __name__ == "__main__":
    main()
