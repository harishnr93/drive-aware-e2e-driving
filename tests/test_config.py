from dataclasses import dataclass, field

import pytest

from driveaware.core.config import ConfigError, from_dict, load_config, save_config, to_dict


@dataclass
class Optim:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class Cfg:
    name: str
    batch_size: int = 4
    optim: Optim = field(default_factory=Optim)
    note: str | None = None


def test_nested_and_int_to_float():
    cfg = from_dict(Cfg, {"name": "a", "optim": {"lr": 1, "betas": [0.8, 0.9]}})
    assert cfg.optim.lr == 1.0 and isinstance(cfg.optim.lr, float)
    assert cfg.optim.betas == (0.8, 0.9)


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="optim"):
        from_dict(Cfg, {"name": "a", "optim": {"lrr": 0.1}})


def test_type_mismatch_rejected():
    with pytest.raises(ConfigError):
        from_dict(Cfg, {"name": "a", "batch_size": "4"})


def test_missing_required():
    with pytest.raises(ConfigError):
        from_dict(Cfg, {})


def test_roundtrip_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    save_config(Cfg(name="x", batch_size=8), path)
    cfg = load_config(Cfg, path, {"optim.lr": 0.5, "batch_size": 2})
    assert cfg.batch_size == 2 and cfg.optim.lr == 0.5
    assert to_dict(cfg)["optim"]["betas"] == [0.9, 0.999]
