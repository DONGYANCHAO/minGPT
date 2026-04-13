"""
minGPT: A minimal implementation of GPT in PyTorch.

Type-safe and performance-optimized fork of Andrej Karpathy's minGPT.
"""

from mingpt.model import (
    CausalSelfAttention,
    Block,
    MLP,
    NewGELU,
    GPT,
    MODEL_CONFIGS,
    InvalidConfigError,
)
from mingpt.trainer import (
    Trainer,
    CallbackEvent,
    TrainerState,
)
from mingpt.bpe import (
    Encoder,
    BPETokenizer,
    get_encoder,
)
from mingpt.utils import (
    CfgNode,
    FrozenConfigError,
    ConfigKeyError,
    set_seed,
    setup_logging,
)

__all__ = [
    'CausalSelfAttention',
    'Block',
    'MLP',
    'NewGELU',
    'GPT',
    'MODEL_CONFIGS',
    'InvalidConfigError',
    'Trainer',
    'CallbackEvent',
    'TrainerState',
    'Encoder',
    'BPETokenizer',
    'get_encoder',
    'CfgNode',
    'FrozenConfigError',
    'ConfigKeyError',
    'set_seed',
    'setup_logging',
]
