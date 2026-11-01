import os

import pytest

from timestral_mcp import model as model_mod
from timestral_mcp.model import (
    DEFAULT_MODEL,
    ConfigurationError,
    ModelLoadError,
    load_settings,
    resolve_model_id,
    validate_device,
)


def test_default_settings(tmp_path, monkeypatch):
    for name in ("Desktop", "Documents", "Downloads"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(model_mod.Path, "home", classmethod(lambda cls: tmp_path))
    cfg = load_settings({})
    assert cfg.model == DEFAULT_MODEL
    assert cfg.device == "cpu"
    # Downloads first: relative paths and outputs of pasted data go there
    assert cfg.allowed_dirs == tuple((tmp_path / n).resolve() for n in ("Downloads", "Desktop", "Documents"))
    assert cfg.max_file_bytes == 50 * 1024 * 1024


def test_default_dirs_skip_missing_folders_and_never_include_home(tmp_path, monkeypatch):
    (tmp_path / "Desktop").mkdir()
    (tmp_path / "OneDrive" / "Documents").mkdir(parents=True)
    assert model_mod.default_data_dirs(tmp_path) == [
        (tmp_path / "Desktop").resolve(),
        (tmp_path / "OneDrive" / "Documents").resolve(),
    ]
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert model_mod.default_data_dirs(empty) == [empty.resolve()]


def test_missing_allowed_dirs_do_not_stop_the_server(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.mkdir()
    gone = tmp_path / "deleted"
    cfg = load_settings({"TIMESTRAL_ALLOWED_DIRS": f"{gone}{os.pathsep}{a}"})
    assert cfg.allowed_dirs == (a.resolve(),)
    (tmp_path / "Desktop").mkdir()
    monkeypatch.setattr(model_mod.Path, "home", classmethod(lambda cls: tmp_path))
    cfg = load_settings({"TIMESTRAL_ALLOWED_DIRS": str(gone)})
    assert cfg.allowed_dirs == ((tmp_path / "Desktop").resolve(),)


def test_settings_from_environment(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    cfg = load_settings(
        {
            "TIMESTRAL_MODEL": "mistralai/Timestral-5M",
            "TIMESTRAL_DEVICE": "CUDA:1",
            "TIMESTRAL_ALLOWED_DIRS": f"{a}{os.pathsep}{b}",
            "TIMESTRAL_MAX_FILE_MB": "2",
        }
    )
    assert cfg.model == "mistralai/Timestral-5M"
    assert cfg.device == "cuda:1"
    assert cfg.allowed_dirs == (a.resolve(), b.resolve())
    assert cfg.max_file_bytes == 2 * 1024 * 1024


@pytest.mark.parametrize(
    "env",
    [
        {"TIMESTRAL_DEVICE": "gpu"},
        {"TIMESTRAL_DEVICE": "cuda:x"},
        {"TIMESTRAL_MAX_FILE_MB": "lots"},
        {"TIMESTRAL_MAX_FILE_MB": "0"},
    ],
)
def test_invalid_settings(env):
    with pytest.raises(ConfigurationError):
        load_settings(env)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", " Cuda:3 "])
def test_valid_devices(device):
    assert validate_device(device) == device.strip().lower()


def test_resolve_model_id():
    configured = "/models/my-local-timestral"
    assert resolve_model_id(None, configured) == configured
    assert resolve_model_id("", configured) == configured
    assert resolve_model_id(configured, configured) == configured
    assert resolve_model_id("mistralai/Timestral-209M", configured) == "mistralai/Timestral-209M"
    with pytest.raises(ConfigurationError, match="not allowed"):
        resolve_model_id("someone/else", configured)
    with pytest.raises(ConfigurationError, match="not allowed"):
        resolve_model_id("C:/other/dir", configured)


def test_forecaster_is_loaded_once_per_checkpoint(monkeypatch):
    loads = []
    monkeypatch.setattr(model_mod, "_cache", {})
    monkeypatch.setattr(model_mod, "_load", lambda m, d: loads.append((m, d)) or object())
    first = model_mod.get_forecaster("m", "cpu")
    assert model_mod.get_forecaster("m", "cpu") is first
    assert model_mod.get_forecaster("other", "cpu") is not first
    assert loads == [("m", "cpu"), ("other", "cpu")]


def test_cuda_unavailable_is_a_clear_error(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ModelLoadError, match="CUDA is not available"):
        model_mod._load(DEFAULT_MODEL, "cuda")


def test_load_failure_is_actionable(tmp_path):
    with pytest.raises(ModelLoadError, match="config.json and model.safetensors"):
        model_mod._load(str(tmp_path), "cpu")  # an empty local directory
