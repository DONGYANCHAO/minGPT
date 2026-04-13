"""
Utility functions and configuration classes for minGPT.
"""

import os
import sys
import json
import random
from ast import literal_eval
from typing import (
    Any, Callable, Dict, Generic, List, Optional,
    Tuple, TypeVar, Union, cast
)

import numpy as np
import torch


T = TypeVar('T')

RANDOM_SEED = 42
DEFAULT_WORK_DIR = './work'
ARGS_FILENAME = 'args.txt'
CONFIG_FILENAME = 'config.json'
DEFAULT_INDENT = 4


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(config: 'CfgNode') -> None:
    """Setup logging directory and configuration.

    Args:
        config: Configuration object containing system.work_dir.
    """
    work_dir = config.system.work_dir
    os.makedirs(work_dir, exist_ok=True)

    with open(os.path.join(work_dir, ARGS_FILENAME), 'w') as f:
        f.write(' '.join(sys.argv))

    with open(os.path.join(work_dir, CONFIG_FILENAME), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=DEFAULT_INDENT))


class FrozenConfigError(Exception):
    """Raised when attempting to modify a frozen configuration."""
    pass


class ConfigKeyError(KeyError):
    """Raised when configuration key is invalid."""
    pass


class CfgNode:
    """A lightweight and type-safe configuration class inspired by yacs.

    Attributes:
        _frozen: Whether the configuration is frozen and immutable.
    """

    _frozen: bool

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__['_frozen'] = False
        self.__dict__.update(kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get('_frozen', False):
            raise FrozenConfigError(
                f"Cannot modify frozen config. Attempted to set '{name}'."
            )
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)
        if name not in self.__dict__:
            raise ConfigKeyError(f"Config key '{name}' does not exist.")
        return self.__dict__[name]

    def __str__(self) -> str:
        return self._str_helper(0)

    def _str_helper(self, indent: int) -> str:
        """Helper for nested indentation in pretty printing.

        Args:
            indent: Current indentation level.

        Returns:
            Formatted string representation.
        """
        parts: List[str] = []
        for k, v in self.__dict__.items():
            if k == '_frozen':
                continue
            if isinstance(v, CfgNode):
                parts.append(f"{k}:\n")
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append(f"{k}: {v}\n")
        prefix = ' ' * (indent * DEFAULT_INDENT)
        parts = [prefix + p for p in parts]
        return "".join(parts)

    def freeze(self) -> 'CfgNode':
        """Freeze the configuration to prevent modification.

        Returns:
            Self for method chaining.
        """
        self.__dict__['_frozen'] = True
        for v in self.__dict__.values():
            if isinstance(v, CfgNode):
                v.freeze()
        return self

    def defrost(self) -> 'CfgNode':
        """Unfreeze the configuration to allow modification.

        Returns:
            Self for method chaining.
        """
        self.__dict__['_frozen'] = False
        for v in self.__dict__.values():
            if isinstance(v, CfgNode):
                v.defrost()
        return self

    def is_frozen(self) -> bool:
        """Check if configuration is frozen."""
        return self.__dict__.get('_frozen', False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to nested dictionary.

        Returns:
            Dictionary representation of config.
        """
        result: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if k == '_frozen':
                continue
            result[k] = v.to_dict() if isinstance(v, CfgNode) else v
        return result

    def merge_from_dict(self, config_dict: Dict[str, Any]) -> 'CfgNode':
        """Merge configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration values.

        Returns:
            Self for method chaining.
        """
        for k, v in config_dict.items():
            if hasattr(self, k) and isinstance(getattr(self, k), CfgNode) and isinstance(v, dict):
                getattr(self, k).merge_from_dict(v)
            else:
                setattr(self, k, v)
        return self

    def merge_from_args(self, args: List[str]) -> 'CfgNode':
        """Update configuration from command line arguments.

        Args:
            args: List of command line arguments in form --arg=value.

        Returns:
            Self for method chaining.

        Raises:
            AssertionError: If argument format is invalid.
        """
        for arg in args:
            keyval = arg.split('=')
            assert len(keyval) == 2, (
                f"Expected each override arg to be of form --arg=value, got: {arg}"
            )
            key, val = keyval

            try:
                val = literal_eval(val)
            except ValueError:
                pass

            assert key[:2] == '--', f"Argument must start with '--': {key}"
            key = key[2:]
            keys = key.split('.')
            obj = self

            for k in keys[:-1]:
                obj = getattr(obj, k)

            leaf_key = keys[-1]
            assert hasattr(obj, leaf_key), f"{key} is not a valid config attribute"

            print(f"Command line overwriting config attribute {key} with {val}")
            setattr(obj, leaf_key, val)

        return self

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CfgNode':
        """Create configuration from dictionary.

        Args:
            config_dict: Nested dictionary of configuration values.

        Returns:
            New CfgNode instance.
        """
        config = cls()
        for k, v in config_dict.items():
            if isinstance(v, dict):
                setattr(config, k, cls.from_dict(v))
            else:
                setattr(config, k, v)
        return config

    @classmethod
    def from_json(cls, filepath: str) -> 'CfgNode':
        """Load configuration from JSON file.

        Args:
            filepath: Path to JSON configuration file.

        Returns:
            New CfgNode instance.
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def validate(self, schema: Dict[str, Tuple[type, bool]]) -> List[str]:
        """Validate configuration against a schema.

        Args:
            schema: Dictionary mapping keys to (type, required) tuples.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []
        for key, (expected_type, required) in schema.items():
            if required and not hasattr(self, key):
                errors.append(f"Missing required config key: {key}")
            elif hasattr(self, key):
                value = getattr(self, key)
                if not isinstance(value, expected_type) and value is not None:
                    errors.append(
                        f"Config key '{key}' has type {type(value).__name__}, "
                        f"expected {expected_type.__name__}"
                    )
        return errors
