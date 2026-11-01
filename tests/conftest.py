"""Shared fixtures.

`fake_forecaster` replaces the Timestral model with a deterministic test double so that
unit and protocol tests run without downloading weights. Its outputs are NOT forecasts;
real inference is covered by tests/test_real_model.py (`pytest -m model`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch

from timestral_mcp import model as model_mod
from timestral_mcp import server
from timestral_mcp.forecasting import NATIVE_QUANTILES


@dataclass
class FakeOutput:
    quantiles: torch.Tensor


@dataclass
class FakeForecaster:
    """Predicts last observed value + 10 * (level - 0.5) for every step; records its inputs.

    A `(B, V, T)` context (joint forecast) is recorded as one `(V, T)` call per series, and each
    variate gets its own last value.
    """

    quantile_levels: tuple[float, ...] = NATIVE_QUANTILES
    supports_covariates: bool = True
    supports_known_covariates: bool = True
    calls: list[tuple[np.ndarray, int]] = field(default_factory=list)
    covariate_calls: list[dict[str, np.ndarray]] = field(default_factory=list)

    def forecast(self, context: np.ndarray, prediction_length: int, **covariates: np.ndarray) -> FakeOutput:
        arr = np.array(context)
        rows = arr if arr.ndim == 3 else np.atleast_2d(arr)[:, None]  # (B, V, T)
        for i, row in enumerate(rows):
            self.calls.append((row if arr.ndim == 3 else row[0], prediction_length))
            self.covariate_calls.append({name: a[i : i + 1] for name, a in covariates.items()})
        offsets = torch.tensor([10 * (q - 0.5) for q in self.quantile_levels])
        last = torch.tensor([[float(v[~np.isnan(v)][-1]) for v in row] for row in rows])  # (B, V)
        q = last[:, :, None, None] + offsets  # (B, V, 1, levels)
        return FakeOutput(quantiles=q.expand(*last.shape, prediction_length, len(self.quantile_levels)).clone())


@pytest.fixture
def fake_forecaster(monkeypatch: pytest.MonkeyPatch) -> FakeForecaster:
    fake = FakeForecaster()
    monkeypatch.setattr(model_mod, "get_forecaster", lambda model_id, device: fake)
    return fake


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An allowed directory for the server's file tools."""
    allowed = tmp_path / "data"
    allowed.mkdir()
    cfg = model_mod.Settings(
        model=model_mod.DEFAULT_MODEL, device="cpu", allowed_dirs=(allowed.resolve(),), max_file_bytes=1024 * 1024
    )
    monkeypatch.setattr(server, "_settings", cfg)
    return allowed


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def daily_csv(path: Path, values: list[float], start: str = "2026-01-01") -> Path:
    import pandas as pd

    dates = pd.date_range(start, periods=len(values), freq="D").strftime("%Y-%m-%d")
    rows = "\n".join(f"{d},{'' if np.isnan(v) else v}" for d, v in zip(dates, values, strict=True))
    return write_csv(path, "date,sales\n" + rows)
