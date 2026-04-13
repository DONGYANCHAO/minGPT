"""minGPT: A minimal PyTorch implementation of GPT.

This package provides a clean, readable implementation of the GPT
(Generative Pre-trained Transformer) architecture for educational
purposes and research.

Example:
    >>> from mingpt.model import GPT
    >>> from mingpt.bpe import BPETokenizer
    >>> from mingpt.trainer import Trainer
    >>>
    >>> # Create a GPT model
    >>> config = GPT.get_default_config()
    >>> config.model_type = 'gpt2'
    >>> config.vocab_size = 50257
    >>> config.block_size = 1024
    >>> model = GPT(config)
    >>>
    >>> # Tokenize text
    >>> tokenizer = BPETokenizer()
    >>> tokens = tokenizer("Hello, world!")
"""

__version__ = "0.1.0"

# Core exports
from mingpt.model import GPT, Block, MLP, CausalSelfAttention, NewGELU
from mingpt.bpe import BPETokenizer, Encoder, get_encoder
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import CfgNode, set_seed, setup_logging

__all__ = [
    # Model components
    "GPT",
    "Block",
    "MLP",
    "CausalSelfAttention",
    "NewGELU",
    # Tokenization
    "BPETokenizer",
    "Encoder",
    "get_encoder",
    # Training
    "Trainer",
    "TrainerConfig",
    # Utilities
    "CfgNode",
    "set_seed",
    "setup_logging",
]
