"""Utility functions and configuration management for minGPT.

This module provides common utilities including:
    - Random seed setting for reproducibility
    - Logging setup for experiment tracking
    - CfgNode: A lightweight configuration class inspired by YACS
"""

import os
import sys
import json
import random
from ast import literal_eval
from typing import Any, Dict, Optional, Union, List, Iterator
from pathlib import Path

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SEED_MAX: int = 2**32 - 1
ARGS_FILENAME: str = "args.txt"
CONFIG_FILENAME: str = "config.json"

# -----------------------------------------------------------------------------
# Type Aliases
# -----------------------------------------------------------------------------

ConfigValue = Union[str, int, float, bool, None, List[Any], Dict[str, Any]]
ConfigDict = Dict[str, Any]

# -----------------------------------------------------------------------------
# Seed and Logging Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all random number generators.

    Args:
        seed: The random seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(config: "CfgNode") -> None:
    """Setup logging directory and save configuration files.

    Creates the working directory if it doesn't exist and saves:
    - Command line arguments to args.txt
    - Configuration to config.json

    Args:
        config: Configuration object containing system.work_dir path.
    """
    work_dir = Path(config.system.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Log the args (if any)
    args_path = work_dir / ARGS_FILENAME
    args_path.write_text(" ".join(sys.argv), encoding="utf-8")

    # Log the config itself
    config_path = work_dir / CONFIG_FILENAME
    config_path.write_text(json.dumps(config.to_dict(), indent=4), encoding="utf-8")


# -----------------------------------------------------------------------------
# Configuration Node
# -----------------------------------------------------------------------------

class FrozenConfigError(Exception):
    """Raised when attempting to modify a frozen configuration."""
    pass


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class CfgNode:
    """A lightweight configuration class inspired by YACS.

    Supports nested configurations, attribute-style and dict-style access,
    freezing to prevent accidental modifications, and validation.

    Example:
        >>> cfg = CfgNode()
        >>> cfg.model = CfgNode()
        >>> cfg.model.n_layer = 12
        >>> cfg.freeze()
        >>> cfg.model.n_layer = 24  # Raises FrozenConfigError

    Attributes:
        _is_frozen: Whether this config node is frozen (read-only).
    """

    _is_frozen: bool = False

    def __init__(self, **kwargs: ConfigValue) -> None:
        """Initialize a CfgNode with optional keyword arguments.

        Args:
            **kwargs: Initial configuration values.
        """
        super().__setattr__("_is_frozen", False)
        self.__dict__.update(kwargs)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute, respecting the frozen state.

        Args:
            name: Attribute name.
            value: Attribute value.

        Raises:
            FrozenConfigError: If the config is frozen.
        """
        if self._is_frozen and not name.startswith("_"):
            raise FrozenConfigError(
                f"Cannot modify frozen config: attempted to set '{name}' = {value}. "
                "Use unfreeze() first if you need to make changes."
            )
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Delete an attribute, respecting the frozen state.

        Args:
            name: Attribute name to delete.

        Raises:
            FrozenConfigError: If the config is frozen.
        """
        if self._is_frozen:
            raise FrozenConfigError(
                f"Cannot delete attribute '{name}' from frozen config. "
                "Use unfreeze() first if you need to make changes."
            )
        super().__delattr__(name)

    def __str__(self) -> str:
        """Return a string representation of the config."""
        return self._str_helper(0)

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return f"CfgNode({self.to_dict()})"

    def _str_helper(self, indent: int) -> str:
        """Helper for nested indentation in pretty printing.

        Args:
            indent: Current indentation level.

        Returns:
            Formatted string representation.
        """
        parts: List[str] = []
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, CfgNode):
                parts.append(f"{k}:\n")
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append(f"{k}: {v}\n")
        parts = [" " * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self) -> ConfigDict:
        """Convert the configuration to a nested dictionary.

        Returns:
            Dictionary representation of the config.
        """
        result: ConfigDict = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, CfgNode):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, d: ConfigDict) -> "CfgNode":
        """Create a CfgNode from a dictionary.

        Args:
            d: Dictionary to convert.

        Returns:
            New CfgNode with nested structure preserved.
        """
        node = cls()
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(node, k, cls.from_dict(v))
            else:
                setattr(node, k, v)
        return node

    def merge_from_dict(self, d: ConfigDict) -> None:
        """Merge values from a dictionary into this config.

        Args:
            d: Dictionary with values to merge.

        Raises:
            FrozenConfigError: If the config is frozen.
        """
        self._check_frozen()
        for k, v in d.items():
            if hasattr(self, k) and isinstance(getattr(self, k), CfgNode) and isinstance(v, dict):
                getattr(self, k).merge_from_dict(v)
            else:
                setattr(self, k, v)

    def merge_from_args(self, args: List[str]) -> None:
        """Update configuration from command line arguments.

        Arguments should be in the form `--arg=value`, where arg can use
        dots to denote nested sub-attributes.

        Example:
            --model.n_layer=10 --trainer.batch_size=32

        Args:
            args: List of command line argument strings.

        Raises:
            ValueError: If an argument is malformed.
            AttributeError: If an attribute doesn't exist in the config.
            FrozenConfigError: If the config is frozen.
        """
        self._check_frozen()
        for arg in args:
            if not arg.startswith("--"):
                raise ValueError(
                    f"Expected argument to start with '--', got: {arg}"
                )

            keyval = arg.split("=", 1)
            if len(keyval) != 2:
                raise ValueError(
                    f"Expected argument of form --arg=value, got: {arg}"
                )
            key, val = keyval

            # Parse the value
            try:
                parsed_val = literal_eval(val)
            except (ValueError, SyntaxError):
                parsed_val = val  # Keep as string if parsing fails

            # Strip the '--' and navigate to the target attribute
            key = key[2:]
            keys = key.split(".")
            obj: Any = self
            for k in keys[:-1]:
                if not hasattr(obj, k):
                    raise AttributeError(
                        f"Config has no attribute '{k}' in path '{key}'"
                    )
                obj = getattr(obj, k)
                if not isinstance(obj, CfgNode):
                    raise AttributeError(
                        f"Cannot set nested attribute on non-CfgNode: {k}"
                    )

            leaf_key = keys[-1]
            if not hasattr(obj, leaf_key):
                raise AttributeError(
                    f"'{key}' is not an existing attribute in the config"
                )

            print(f"Command line overwriting config attribute {key} with {parsed_val}")
            setattr(obj, leaf_key, parsed_val)

    def freeze(self) -> "CfgNode":
        """Freeze this config to prevent modifications.

        Returns:
            Self for method chaining.
        """
        self._is_frozen = True
        for v in self.__dict__.values():
            if isinstance(v, CfgNode):
                v.freeze()
        return self

    def unfreeze(self) -> "CfgNode":
        """Unfreeze this config to allow modifications.

        Returns:
            Self for method chaining.
        """
        self._is_frozen = False
        for v in self.__dict__.values():
            if isinstance(v, CfgNode):
                v.unfreeze()
        return self

    def is_frozen(self) -> bool:
        """Check if this config is frozen.

        Returns:
            True if frozen, False otherwise.
        """
        return self._is_frozen

    def _check_frozen(self) -> None:
        """Internal helper to check frozen state and raise if needed."""
        if self._is_frozen:
            raise FrozenConfigError("Cannot modify frozen configuration")

    def clone(self) -> "CfgNode":
        """Create a deep copy of this config.

        Returns:
            New CfgNode with copied values.
        """
        return self.from_dict(self.to_dict())

    def validate_required(self, required_keys: List[str]) -> None:
        """Validate that required keys are present and not None.

        Args:
            required_keys: List of required attribute names (supports dot notation).

        Raises:
            ConfigValidationError: If any required key is missing or None.
        """
        for key in required_keys:
            keys = key.split(".")
            obj: Any = self
            for k in keys:
                if not hasattr(obj, k):
                    raise ConfigValidationError(f"Missing required config key: {key}")
                obj = getattr(obj, k)
            if obj is None:
                raise ConfigValidationError(f"Required config key is None: {key}")

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to a JSON file.

        Args:
            filepath: Path to save the configuration.
        """
        path = Path(filepath)
        path.write_text(json.dumps(self.to_dict(), indent=4), encoding="utf-8")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CfgNode":
        """Load configuration from a JSON file.

        Args:
            filepath: Path to the configuration file.

        Returns:
            New CfgNode loaded from file.
        """
        path = Path(filepath)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over config items (excluding private attributes).

        Yields:
            Tuples of (key, value).
        """
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                yield k, v

    def keys(self) -> Iterator[str]:
        """Iterate over config keys (excluding private attributes).

        Yields:
            Config keys.
        """
        for k in self.__dict__.keys():
            if not k.startswith("_"):
                yield k

    def values(self) -> Iterator[Any]:
        """Iterate over config values (excluding private attributes).

        Yields:
            Config values.
        """
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                yield v
