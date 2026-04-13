"""
minGPT - A minimal PyTorch implementation of GPT.

This package provides:
- GPT: Transformer-based language model
- Trainer: Generic training loop
- CfgNode: Type-safe configuration management
- Utilities: Seed setting and logging setup
"""

from __future__ import annotations

from mingpt.model import (
    GPT,
    Block,
    CausalSelfAttention,
    MLP,
    NewGELU,
    MODEL_CONFIGS,
    PRETRAINED_MODEL_TYPES,
)
from mingpt.trainer import (
    CallbackEvent,
    CallbackFunction,
    Trainer,
)
from mingpt.utils import (
    CfgNode,
    set_seed,
    setup_logging,
)

__all__: list = [
    'GPT',
    'Block',
    'CausalSelfAttention',
    'MLP',
    'NewGELU',
    'MODEL_CONFIGS',
    'PRETRAINED_MODEL_TYPES',
    'CallbackEvent',
    'CallbackFunction',
    'Trainer',
    'CfgNode',
    'set_seed',
    'setup_logging',
]

__version__: str = '1.0.0'
