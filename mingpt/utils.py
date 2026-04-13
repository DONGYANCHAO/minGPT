"""
Utility functions and configuration management for minGPT.

This module provides:
- Seed setting for reproducibility
- Logging setup
- Type-safe configuration node (CfgNode)
"""

from __future__ import annotations

import json
import os
import random
import sys
from ast import literal_eval
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar, Union

import numpy as np
import torch

T = TypeVar('T')

DEFAULT_SEED: int = 42
ARGS_FILE_NAME: str = 'args.txt'
CONFIG_FILE_NAME: str = 'config.json'


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set random seed for reproducibility across all random libraries.

    Args:
        seed: Random seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(config: CfgNode) -> Path:
    """
    Set up logging directory and save configuration.

    Args:
        config: Configuration node containing system.work_dir.

    Returns:
        Path to the work directory.
    """
    work_dir = Path(config.system.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    with open(work_dir / ARGS_FILE_NAME, 'w', encoding='utf-8') as f:
        f.write(' '.join(sys.argv))

    with open(work_dir / CONFIG_FILE_NAME, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=4)

    return work_dir


class CfgNode:
    """
    A lightweight, type-safe configuration class inspired by yacs.

    This class provides:
    - Nested attribute access
    - Freezing to prevent accidental modification
    - Validation of required keys
    - Import/export to dict and JSON

    Attributes:
        _frozen: Whether the config is frozen (read-only).
        _required_keys: Set of keys that must be present.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize configuration node with given key-value pairs.

        Args:
            **kwargs: Key-value pairs to set as attributes.
        """
        self._frozen: bool = False
        self._required_keys: set = set()
        self.__dict__.update(kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set attribute with frozen check.

        Args:
            name: Attribute name.
            value: Attribute value.

        Raises:
            RuntimeError: If config is frozen.
        """
        if name in ('_frozen', '_required_keys'):
            object.__setattr__(self, name, value)
        elif hasattr(self, '_frozen') and self._frozen:
            raise RuntimeError(
                f"Cannot modify frozen config. Attempted to set '{name}'."
            )
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        """
        Get attribute with helpful error message for missing keys.

        Args:
            name: Attribute name.

        Returns:
            Attribute value.

        Raises:
            AttributeError: If attribute does not exist.
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        raise AttributeError(
            f"Config has no attribute '{name}'. "
            f"Available keys: {list(self.__dict__.keys())}"
        )

    def __contains__(self, key: str) -> bool:
        """
        Check if key exists in config.

        Args:
            key: Key to check.

        Returns:
            True if key exists.
        """
        return key in self.__dict__

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over config keys.

        Yields:
            Configuration keys.
        """
        for key in self.__dict__:
            if not key.startswith('_'):
                yield key

    def freeze(self) -> None:
        """
        Freeze the config to prevent further modifications.

        This recursively freezes all nested CfgNode instances.
        """
        self._frozen = True
        for value in self.__dict__.values():
            if isinstance(value, CfgNode):
                value.freeze()

    def unfreeze(self) -> None:
        """
        Unfreeze the config to allow modifications.

        This recursively unfreezes all nested CfgNode instances.
        """
        self._frozen = False
        for value in self.__dict__.values():
            if isinstance(value, CfgNode):
                value.unfreeze()

    def is_frozen(self) -> bool:
        """
        Check if the config is frozen.

        Returns:
            True if frozen, False otherwise.
        """
        return self._frozen

    def set_required(self, *keys: str) -> None:
        """
        Mark keys as required for validation.

        Args:
            *keys: Keys that must be present.
        """
        self._required_keys.update(keys)

    def validate(self) -> List[str]:
        """
        Validate that all required keys are present and non-None.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []
        for key in self._required_keys:
            if key not in self.__dict__:
                errors.append(f"Required key '{key}' is missing")
            elif self.__dict__[key] is None:
                errors.append(f"Required key '{key}' is None")
        return errors

    def assert_valid(self) -> None:
        """
        Assert that config is valid, raising error if not.

        Raises:
            ValueError: If validation fails.
        """
        errors = self.validate()
        if errors:
            raise ValueError("Config validation failed:\n" + "\n".join(errors))

    def __str__(self) -> str:
        """
        Return pretty-printed string representation.

        Returns:
            Formatted string of config contents.
        """
        return self._str_helper(0)

    def _str_helper(self, indent: int) -> str:
        """
        Helper for nested indentation in string representation.

        Args:
            indent: Current indentation level.

        Returns:
            Formatted string.
        """
        parts: List[str] = []
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if isinstance(v, CfgNode):
                parts.append(f"{k}:\n")
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append(f"{k}: {v}\n")
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to a dictionary.

        Returns:
            Dictionary representation of config.
        """
        result: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            result[k] = v.to_dict() if isinstance(v, CfgNode) else v
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CfgNode:
        """
        Create a CfgNode from a dictionary.

        Args:
            d: Dictionary to convert.

        Returns:
            New CfgNode instance.
        """
        kwargs: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                kwargs[k] = cls.from_dict(v)
            else:
                kwargs[k] = v
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> CfgNode:
        """
        Load config from a JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            New CfgNode instance.
        """
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: Union[str, Path]) -> None:
        """
        Save config to a JSON file.

        Args:
            path: Path to save JSON file.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)

    def merge_from_dict(self, d: Dict[str, Any]) -> None:
        """
        Merge values from a dictionary into this config.

        Args:
            d: Dictionary with values to merge.
        """
        for k, v in d.items():
            if isinstance(v, dict) and hasattr(self, k):
                existing = getattr(self, k)
                if isinstance(existing, CfgNode):
                    existing.merge_from_dict(v)
                    continue
            setattr(self, k, v)

    def merge_from_args(self, args: List[str]) -> None:
        """
        Update configuration from command-line arguments.

        Arguments are expected in the form `--arg=value`, where arg can use
        . to denote nested sub-attributes.

        Example:
            --model.n_layer=10 --trainer.batch_size=32

        Args:
            args: List of command-line argument strings.

        Raises:
            AssertionError: If argument format is invalid or key doesn't exist.
        """
        for arg in args:
            keyval = arg.split('=')
            assert len(keyval) == 2, f"Expected --arg=value format, got '{arg}'"
            key, val = keyval

            try:
                val = literal_eval(val)
            except ValueError:
                pass

            assert key.startswith('--'), f"Expected '--' prefix, got '{key}'"
            key = key[2:]
            keys = key.split('.')
            obj: CfgNode = self

            for k in keys[:-1]:
                obj = getattr(obj, k)

            leaf_key = keys[-1]
            assert hasattr(obj, leaf_key), (
                f"'{key}' is not a valid config attribute. "
                f"Available: {list(obj.__dict__.keys())}"
            )

            print(f"Overriding config: {key} = {val}")
            setattr(obj, leaf_key, val)

    def get(self, key: str, default: Optional[T] = None) -> Union[Any, T, None]:
        """
        Get a config value with a default fallback.

        Args:
            key: Key to look up.
            default: Default value if key doesn't exist.

        Returns:
            Config value or default.
        """
        return self.__dict__.get(key, default)

    def keys(self) -> List[str]:
        """
        Get all config keys.

        Returns:
            List of keys.
        """
        return [k for k in self.__dict__.keys() if not k.startswith('_')]

    def values(self) -> List[Any]:
        """
        Get all config values.

        Returns:
            List of values.
        """
        return [v for k, v in self.__dict__.items() if not k.startswith('_')]

    def items(self) -> List[tuple]:
        """
        Get all config key-value pairs.

        Returns:
            List of (key, value) tuples.
        """
        return [(k, v) for k, v in self.__dict__.items() if not k.startswith('_')]

    def copy(self) -> CfgNode:
        """
        Create a deep copy of this config.

        Returns:
            New CfgNode with same values.
        """
        return CfgNode.from_dict(self.to_dict())

    def update(self, other: Union[Dict[str, Any], CfgNode]) -> None:
        """
        Update config with values from another config or dict.

        Args:
            other: Config or dict with values to merge.
        """
        if isinstance(other, CfgNode):
            self.merge_from_dict(other.to_dict())
        else:
            self.merge_from_dict(other)
