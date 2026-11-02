"""Strict YAML <-> dataclass configuration.

Configs are plain YAML mapped onto (possibly nested) dataclasses. Unknown keys raise, so a typo
in an experiment config cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


class ConfigError(ValueError):
    pass


def from_dict(cls: type[T], data: dict[str, Any] | None, _path: str = "") -> T:
    """Build dataclass ``cls`` from a mapping, recursing into nested dataclasses."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    data = dict(data or {})
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(f"unknown config key(s) at '{_path or '<root>'}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        kwargs[name] = _convert(hints[name], value, f"{_path}.{name}" if _path else name)
    try:
        return cls(**kwargs)
    except TypeError as exc:  # missing required fields
        raise ConfigError(f"at '{_path or '<root>'}': {exc}") from exc


def _convert(hint: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(hint)
    if dataclasses.is_dataclass(hint) and isinstance(hint, type):
        if not isinstance(value, dict):
            raise ConfigError(f"'{path}' expects a mapping")
        return from_dict(hint, value, path)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if value is None:
            return None
        if len(args) == 1:
            return _convert(args[0], value, path)
        return value
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"'{path}' expects a mapping")
        _, val_t = typing.get_args(hint) or (Any, Any)
        return {k: _convert(val_t, v, f"{path}.{k}") for k, v in value.items()}
    if origin in (tuple,) and isinstance(value, list):
        return tuple(value)
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if hint in (int, float, str, bool) and not isinstance(value, hint):
        raise ConfigError(f"'{path}' expects {hint.__name__}, got {type(value).__name__}")
    return value


def to_dict(cfg: Any) -> dict[str, Any]:
    if not dataclasses.is_dataclass(cfg) or isinstance(cfg, type):
        raise TypeError("to_dict expects a dataclass instance")
    return _plain(dataclasses.asdict(cfg))


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_config(cls: type[T], path: str | Path, overrides: dict[str, Any] | None = None) -> T:
    """Load YAML into ``cls``. ``overrides`` uses dotted keys, e.g. {"optim.lr": 1e-4}."""
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    for dotted, value in (overrides or {}).items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    return from_dict(cls, data)


def save_config(cfg: Any, path: str | Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(to_dict(cfg), fh, sort_keys=False)
