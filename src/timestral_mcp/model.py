"""Configuration, device selection and the lazily loaded, cached Timestral forecaster.

torch and timestral are imported inside functions so that the MCP server starts (and
lists its tools) without paying the import cost of PyTorch.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistralai/Timestral-28M"
OFFICIAL_MODELS = (
    "mistralai/Timestral-5M",
    "mistralai/Timestral-28M",
    "mistralai/Timestral-123M",
    "mistralai/Timestral-209M",
)
DEFAULT_MAX_FILE_MB = 50.0


class ConfigurationError(ValueError):
    """Invalid server configuration (environment variables or tool arguments)."""


class ModelLoadError(RuntimeError):
    """A checkpoint could not be loaded."""


@dataclass(frozen=True)
class Settings:
    model: str
    device: str
    allowed_dirs: tuple[Path, ...]
    max_file_bytes: int


def validate_device(device: str) -> str:
    device = device.strip().lower()
    if device == "cpu":
        return device
    if device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit()):
        return device
    raise ConfigurationError(f"TIMESTRAL_DEVICE must be 'cpu', 'cuda' or 'cuda:<index>', got {device!r}")


def default_data_dirs(home: Path | None = None) -> list[Path]:
    """The folders where people keep the files they hand to an assistant.

    Downloads, Desktop and Documents of the user's home (and their OneDrive copies, which
    Windows uses when folder backup is on), in that order: relative paths and outputs of
    pasted data go to the first one. The home folder itself is not included, so hidden
    folders such as AppData or .ssh stay out of reach. Falls back to the working directory
    when none exists.
    """
    home = Path.home() if home is None else home
    candidates = [home / "Downloads", home / "Desktop", home / "Documents"]
    candidates += [home / "OneDrive" / "Desktop", home / "OneDrive" / "Documents"]
    dirs: list[Path] = []
    for path in candidates:
        if path.is_dir() and path.resolve() not in dirs:
            dirs.append(path.resolve())
    return dirs or [Path.cwd().resolve()]


def load_settings(env: dict[str, str] | None = None) -> Settings:
    """Read and validate configuration from environment variables.

    TIMESTRAL_MODEL         checkpoint: a Hugging Face repo id or a local directory
    TIMESTRAL_DEVICE        cpu (default), cuda or cuda:<index>
    TIMESTRAL_ALLOWED_DIRS  directories the file tools may read, separated by os.pathsep
                            (';' on Windows, ':' elsewhere); defaults to the user's Downloads,
                            Desktop and Documents folders (see default_data_dirs)
    TIMESTRAL_MAX_FILE_MB   maximum CSV size in megabytes (default 50)

    Entries that do not exist are skipped with a warning, so a deleted folder never stops the
    server from starting; if none is left, the default folders are used.
    """
    env = dict(os.environ) if env is None else env
    model = env.get("TIMESTRAL_MODEL", "").strip() or DEFAULT_MODEL
    device = validate_device(env.get("TIMESTRAL_DEVICE", "").strip() or "cpu")

    raw_dirs = [d.strip() for d in env.get("TIMESTRAL_ALLOWED_DIRS", "").split(os.pathsep) if d.strip()]
    dirs: list[Path] = []
    for raw in raw_dirs:
        path = Path(raw).expanduser()
        if not path.is_dir():
            logger.warning("TIMESTRAL_ALLOWED_DIRS entry %r is not an existing directory; ignored", raw)
            continue
        if path.resolve() not in dirs:
            dirs.append(path.resolve())
    if not dirs:
        if raw_dirs:
            logger.warning("no TIMESTRAL_ALLOWED_DIRS entry exists; using the default folders")
        dirs = default_data_dirs()

    raw_mb = env.get("TIMESTRAL_MAX_FILE_MB", "").strip()
    try:
        max_mb = float(raw_mb) if raw_mb else DEFAULT_MAX_FILE_MB
    except ValueError:
        raise ConfigurationError(f"TIMESTRAL_MAX_FILE_MB must be a number, got {raw_mb!r}") from None
    if max_mb <= 0:
        raise ConfigurationError(f"TIMESTRAL_MAX_FILE_MB must be positive, got {max_mb}")

    return Settings(model=model, device=device, allowed_dirs=tuple(dirs), max_file_bytes=int(max_mb * 1024 * 1024))


def resolve_model_id(requested: str | None, configured: str) -> str:
    """The checkpoint a tool call should use.

    Tool callers may pick the configured checkpoint or one of the official Timestral
    checkpoints; arbitrary repositories or paths can only be set by the user through
    TIMESTRAL_MODEL.
    """
    if requested is None or requested.strip() == "" or requested == configured:
        return configured
    if requested in OFFICIAL_MODELS:
        return requested
    allowed = ", ".join(dict.fromkeys((configured, *OFFICIAL_MODELS)))
    raise ConfigurationError(f"model {requested!r} is not allowed; choose one of: {allowed}")


_cache: dict[tuple[str, str], Any] = {}
_cache_lock = threading.Lock()
# One inference at a time: concurrent tool calls would otherwise compete for the same
# CPU cores (or GPU memory) and all finish later.
inference_lock = threading.Lock()


def get_forecaster(model_id: str, device: str) -> Any:
    """Return a cached `timestral.TimestralForecaster`, loading it on first use."""
    key = (model_id, device)
    with _cache_lock:
        if key not in _cache:
            _cache[key] = _load(model_id, device)
        return _cache[key]


def _load(model_id: str, device: str) -> Any:
    import torch
    from timestral import TimestralForecaster

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ModelLoadError(
                f"TIMESTRAL_DEVICE={device} but CUDA is not available to PyTorch "
                f"(torch {torch.__version__}). Install a CUDA build of PyTorch or set TIMESTRAL_DEVICE=cpu."
            )
        index = int(device.split(":")[1]) if ":" in device else 0
        if index >= torch.cuda.device_count():
            raise ModelLoadError(f"{device} requested but only {torch.cuda.device_count()} CUDA device(s) found")

    logger.info("Loading Timestral checkpoint %s on %s", model_id, device)
    try:
        forecaster = TimestralForecaster.from_pretrained(model_id, device=device)
    except Exception as e:
        raise ModelLoadError(
            f"could not load checkpoint {model_id!r}: {e}. The first use downloads the weights from the "
            "Hugging Face Hub, so check the network connection, or run `timestral-mcp --download` "
            "beforehand; a local directory must contain config.json and model.safetensors."
        ) from e
    logger.info("Loaded %s", model_id)
    return forecaster


def download(model_id: str) -> Path:
    """Download (or find in the Hub cache) a checkpoint's files without loading the model."""
    from timestral.loading import resolve_checkpoint

    return resolve_checkpoint(model_id)
