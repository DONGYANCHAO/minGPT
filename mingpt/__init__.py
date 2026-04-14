"""
minGPT - A minimal PyTorch implementation of OpenAI's GPT architecture.

This package provides:
- GPT model implementation (model.py)
- Training utilities (trainer.py)
- Checkpoint management (checkpoint.py)
- Training visualization and logging (logger.py)
- Model evaluation and metrics (evaluator.py)
- Command line tools (cli.py)
"""

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode, set_seed, setup_logging
from mingpt.checkpoint import CheckpointManager
from mingpt.logger import TrainingLogger, WebMonitor
from mingpt.evaluator import Evaluator, ModelComparator, HuggingFaceBenchmark

__version__ = '1.0.0'
__all__ = [
    'GPT',
    'Trainer',
    'CfgNode',
    'set_seed',
    'setup_logging',
    'CheckpointManager',
    'TrainingLogger',
    'WebMonitor',
    'Evaluator',
    'ModelComparator',
    'HuggingFaceBenchmark',
]
