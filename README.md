# timestral-mcp

An [MCP](https://modelcontextprotocol.io/) server that lets Claude (Claude Code, Claude Desktop) and
other MCP clients forecast time series locally with Mistral AI's
[Timestral](https://github.com/mistralai/timestral) models.

Ask *"Forecast the next 30 days of sales from sales.csv"* and the assistant runs Timestral on your
machine and returns quantile forecasts. No Python, no tensors, no data sent anywhere.

> An independent project, not affiliated with or endorsed by Mistral AI.

## Tools

| Tool | What it does |
|---|---|
| `forecast` | Forecasts a list of numbers. |
| `forecast_csv` | Forecasts a column of a CSV or Excel file: detects the date and target columns and the frequency, and returns dated forecasts with their total. Handles covariates, several series per file and writing results to a CSV file. |
| `backtest` | Measures how the model would have done on the most recent data (one or several rolling windows) with MAE, RMSE, WAPE and interval coverage, compared with simple baselines. |
| `compare_scenarios` | What-if analysis: forecasts a series with a covariate as planned and as changed (e.g. `{"promo": 0}`), and reports the difference. |

Results are **quantile forecasts**: `p50` is the median, `[p10, p90]` an 80% prediction interval.
Any of Timestral's 21 native levels (0.01, 0.05, 0.1, …, 0.95, 0.99) can be requested.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/).
- About 1 GB of disk for PyTorch (several GB on Linux, where the default wheel includes CUDA).
- No GPU needed. The weights are downloaded from Hugging Face on first use (111 MB for the default
  `Timestral-28M`). On a CPU laptop a forecast takes a fraction of a second once the model is loaded.

## Installation

### Claude Code

```bash
claude mcp add timestral -- uvx --from git+https://github.com/rav-lad/timestral-mcp timestral-mcp
```

Add `--scope user` to make it available in every project. Check it with `claude mcp list`.

### Claude Desktop

In **Settings → Developer → Edit Config** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "timestral": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/rav-lad/timestral-mcp", "timestral-mcp"]
    }
  }
}
```

Restart Claude Desktop afterwards. If it cannot find `uvx`, use its absolute path (`where uvx` on
Windows, `which uvx` elsewhere).

### From a clone

```bash
git clone https://github.com/rav-lad/timestral-mcp
cd timestral-mcp
uv sync
uv run timestral-mcp --download   # optional: fetch the weights now
```

Then use `uv run --directory /absolute/path/to/timestral-mcp timestral-mcp` as the command.

## Configuration

Environment variables, all optional:

| Variable | Default | Meaning |
|---|---|---|
| `TIMESTRAL_MODEL` | `mistralai/Timestral-28M` | Hugging Face repo id or local checkpoint directory. Official sizes: `5M`, `28M`, `123M`, `209M`. |
| `TIMESTRAL_DEVICE` | `cpu` | `cpu`, `cuda` or `cuda:<index>`. Never falls back to CPU silently. |
| `TIMESTRAL_ALLOWED_DIRS` | Downloads, Desktop, Documents | Folders the server may read from and write to (`;`-separated on Windows, `:` elsewhere). |
| `TIMESTRAL_MAX_FILE_MB` | `50` | Maximum file size. |

The server only reads `.csv`, `.xlsx` and `.xlsm` files inside the allowed folders, and never
overwrites a file: `forecast.csv` becomes `forecast (2).csv` if it already exists.

## Example prompts

> Forecast the next 14 days of sales from sales.csv.

> Backtest the model on the last 30 days of sales.csv over 4 rolling windows. The data has a weekly cycle.

> Forecast next week's sales for every store in stores.xlsx, with the total per store.

> How many extra sales do the promotions planned in sales.csv bring over the next two weeks?

> Forecast the next 6 values of this series: 112, 118, 132, 129, 121, 135, 148, 148, 136, 119.

## How data is handled

- **Files:** CSV (delimiter and encoding detected; semicolon files read with decimal commas and
  day-first dates) and Excel. Rows are sorted by date; duplicate dates are rejected.
- **Frequency:** from 15 minutes to yearly, inferred from the dates or passed explicitly. The dates
  must form a regular grid.
- **Missing data:** gaps are rejected unless `fill_missing_dates=true`, which passes them to the
  model as missing values. Values are never invented.
- **Covariates:** `past_covariates` are known only up to today (e.g. actual weather);
  `known_covariates` are known in advance (e.g. a promotion calendar) and need a value for every
  future step, given as rows with an empty target at the end of the file.
- **Several series:** `series_column="store"` forecasts each series of a long-format file
  separately; `aggregate_by` adds them up first. Series that can't be used are skipped and listed.
- **Totals:** the `p10`/`p90` totals over the horizon are conservative (wider than the true interval).

## Limitations

- Forecasts can be wrong. Run `backtest` on your own data before relying on them.
- Only the CPU path has been tested.
- Large files (thousands of series) can take minutes on a CPU, and some clients may time out.
- Series are forecast independently; Timestral's multivariate mode is not exposed.

## Development

```bash
uv sync
uv run pytest              # unit and protocol tests, with a test double instead of the model
uv run pytest -m model     # real Timestral inference (downloads the weights once)
uv run ruff check . && uv run ruff format --check .
```

| Module | Role |
|---|---|
| `server.py` | MCP tools, output schemas, CLI entry point |
| `model.py` | configuration, lazy cached model loading |
| `forecasting.py` | input validation, inference, quantiles |
| `preprocessing.py` | file access, CSV/Excel parsing, column and frequency detection |
| `backtesting.py` | rolling windows, metrics, baselines |

The package cannot be published on PyPI yet, because Timestral itself is only available from Git.

## License

- **This server:** [Apache 2.0](LICENSE).
- **Timestral source code:** Apache 2.0, by Mistral AI.
- **Timestral model weights:** not included. They are downloaded from Hugging Face under Mistral
  AI's *Modified MIT* license (see
  [Timestral-28M](https://huggingface.co/mistralai/Timestral-28M/blob/main/LICENSE)), which
  excludes companies with more than $20 million in monthly revenue unless they get a commercial
  license. Review it before use.
