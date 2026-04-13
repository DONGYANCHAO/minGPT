"""Simple training loop for neural networks.

This module provides a generic Trainer class that can be used with any
PyTorch model, with specific optimizations for GPT-style language models.
"""

import time
import itertools
from typing import Optional, Callable, List, Dict, Any, Protocol, runtime_checkable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_DEVICE: str = "auto"
DEFAULT_NUM_WORKERS: int = 4
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LEARNING_RATE: float = 3e-4
DEFAULT_WEIGHT_DECAY: float = 0.1
DEFAULT_GRAD_NORM_CLIP: float = 1.0
DEFAULT_BETAS: tuple[float, float] = (0.9, 0.95)

SAMPLER_NUM_SAMPLES: int = int(1e10)

# -----------------------------------------------------------------------------
# Callback Types
# -----------------------------------------------------------------------------


@runtime_checkable
class TrainerCallback(Protocol):
    """Protocol for trainer callbacks.

    Callbacks can implement any subset of these methods to hook into
the training loop at various points.
    """

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at the beginning of training."""
        ...

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        ...

    def on_batch_begin(self, trainer: "Trainer") -> None:
        """Called at the beginning of each batch."""
        ...

    def on_batch_end(self, trainer: "Trainer") -> None:
        """Called at the end of each batch."""
        ...


CallbackType = Callable[["Trainer"], None]


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainerConfig:
    """Immutable configuration for the Trainer.

    Attributes:
        device: Device to train on ('auto', 'cuda', 'cpu', etc.).
        num_workers: Number of DataLoader workers.
        max_iters: Maximum number of training iterations (None for infinite).
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        betas: Adam optimizer beta parameters.
        weight_decay: Weight decay coefficient.
        grad_norm_clip: Gradient clipping threshold.
    """

    device: str = DEFAULT_DEVICE
    num_workers: int = DEFAULT_NUM_WORKERS
    max_iters: Optional[int] = None
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    betas: tuple[float, float] = DEFAULT_BETAS
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    grad_norm_clip: float = DEFAULT_GRAD_NORM_CLIP

    @classmethod
    def from_cfgnode(cls, config: CN) -> "TrainerConfig":
        """Create TrainerConfig from a CfgNode.

        Args:
            config: Configuration node with trainer settings.

        Returns:
            TrainerConfig instance.
        """
        return cls(
            device=getattr(config, "device", DEFAULT_DEVICE),
            num_workers=getattr(config, "num_workers", DEFAULT_NUM_WORKERS),
            max_iters=getattr(config, "max_iters", None),
            batch_size=getattr(config, "batch_size", DEFAULT_BATCH_SIZE),
            learning_rate=getattr(config, "learning_rate", DEFAULT_LEARNING_RATE),
            betas=getattr(config, "betas", DEFAULT_BETAS),
            weight_decay=getattr(config, "weight_decay", DEFAULT_WEIGHT_DECAY),
            grad_norm_clip=getattr(config, "grad_norm_clip", DEFAULT_GRAD_NORM_CLIP),
        )


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


class Trainer:
    """Generic trainer for PyTorch models.

    Handles training loop, optimization, gradient clipping, and callbacks.
    Designed to work with any model that implements `configure_optimizers`.

    Attributes:
        config: Training configuration.
        model: The model being trained.
        optimizer: The optimizer (initialized in run()).
        train_dataset: Training dataset.
        device: Device to train on.
        callbacks: Registered callbacks organized by event.
        iter_num: Current iteration number.
        iter_time: Timestamp of last iteration.
        iter_dt: Time delta between iterations.
        loss: Current loss value.
    """

    # Callback event names
    EVENT_TRAIN_BEGIN: str = "on_train_begin"
    EVENT_TRAIN_END: str = "on_train_end"
    EVENT_BATCH_BEGIN: str = "on_batch_begin"
    EVENT_BATCH_END: str = "on_batch_end"

    def __init__(
        self,
        config: CN,
        model: nn.Module,
        train_dataset: Dataset,
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Training configuration (CfgNode).
            model: Model to train (must have configure_optimizers method).
            train_dataset: Training dataset.
        """
        self.config: TrainerConfig = TrainerConfig.from_cfgnode(config)
        self.model: nn.Module = model
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.train_dataset: Dataset = train_dataset
        self.callbacks: Dict[str, List[CallbackType]] = {
            self.EVENT_TRAIN_BEGIN: [],
            self.EVENT_TRAIN_END: [],
            self.EVENT_BATCH_BEGIN: [],
            self.EVENT_BATCH_END: [],
        }

        # Determine device
        if self.config.device == "auto":
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        self.model = self.model.to(self.device)
        print(f"Running on device: {self.device}")

        # Training state
        self.iter_num: int = 0
        self.iter_time: float = 0.0
        self.iter_dt: float = 0.0
        self.loss: Optional[torch.Tensor] = None

    def add_callback(self, event: str, callback: CallbackType) -> None:
        """Add a callback for an event.

        Args:
            event: Event name (one of the EVENT_* constants).
            callback: Callback function to add.

        Raises:
            ValueError: If event is not a valid callback event.
        """
        if event not in self.callbacks:
            valid_events = ", ".join(self.callbacks.keys())
            raise ValueError(
                f"Invalid event '{event}'. Valid events: {valid_events}"
            )
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback: CallbackType) -> None:
        """Set a single callback for an event, replacing any existing ones.

        Args:
            event: Event name.
            callback: Callback function to set.

        Raises:
            ValueError: If event is not a valid callback event.
        """
        if event not in self.callbacks:
            valid_events = ", ".join(self.callbacks.keys())
            raise ValueError(
                f"Invalid event '{event}'. Valid events: {valid_events}"
            )
        self.callbacks[event] = [callback]

    def trigger_callbacks(self, event: str) -> None:
        """Trigger all callbacks for an event.

        Args:
            event: Event name to trigger.
        """
        for callback in self.callbacks.get(event, []):
            callback(self)

    def _create_dataloader(self) -> DataLoader:
        """Create the training DataLoader with infinite sampling.

        Returns:
            Configured DataLoader.
        """
        # Use RandomSampler with replacement for effectively infinite sampling
        sampler: Sampler = torch.utils.data.RandomSampler(
            self.train_dataset,
            replacement=True,
            num_samples=SAMPLER_NUM_SAMPLES,
        )

        return DataLoader(
            self.train_dataset,
            sampler=sampler,
            shuffle=False,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup the optimizer using the model's configure_optimizers method.

        Returns:
            Configured optimizer.

        Raises:
            AttributeError: If model doesn't have configure_optimizers method.
        """
        if not hasattr(self.model, "configure_optimizers"):
            raise AttributeError(
                "Model must implement 'configure_optimizers' method"
            )
        return self.model.configure_optimizers(self.config)

    def run(self) -> None:
        """Run the training loop.

        This method runs until max_iters is reached (if specified).
        """
        # Setup
        self.optimizer = self._setup_optimizer()
        train_loader = self._create_dataloader()

        self.model.train()
        self.iter_num = 0
        self.iter_time = time.time()

        # Use cycle for infinite iteration over the loader
        data_iter = itertools.cycle(train_loader)

        self.trigger_callbacks(self.EVENT_TRAIN_BEGIN)

        while True:
            self.trigger_callbacks(self.EVENT_BATCH_BEGIN)

            # Fetch next batch
            batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # Forward pass
            _, self.loss = self.model(x, y)

            # Backward pass and optimization
            self.model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_norm_clip
            )
            self.optimizer.step()

            self.trigger_callbacks(self.EVENT_BATCH_END)

            # Update iteration tracking
            self.iter_num += 1
            t_now = time.time()
            self.iter_dt = t_now - self.iter_time
            self.iter_time = t_now

            # Check termination condition
            if (
                self.config.max_iters is not None
                and self.iter_num >= self.config.max_iters
            ):
                break

        self.trigger_callbacks(self.EVENT_TRAIN_END)


# -----------------------------------------------------------------------------
# Utility Callbacks
# -----------------------------------------------------------------------------


class PrintLossCallback:
    """Callback that prints loss at regular intervals."""

    def __init__(self, print_every: int = 100) -> None:
        """Initialize the callback.

        Args:
            print_every: Print loss every N iterations.
        """
        self.print_every: int = print_every

    def __call__(self, trainer: Trainer) -> None:
        """Print loss if at the right interval.

        Args:
            trainer: The trainer instance.
        """
        if trainer.iter_num % self.print_every == 0 and trainer.loss is not None:
            print(
                f"Iter {trainer.iter_num}: loss = {trainer.loss.item():.4f}, "
                f"dt = {trainer.iter_dt * 1000:.2f}ms"
            )


class LearningRateSchedulerCallback:
    """Callback for learning rate scheduling with warmup and cosine decay."""

    def __init__(
        self,
        warmup_iters: int = 0,
        lr_decay_iters: int = 0,
        min_lr: float = 0.0,
    ) -> None:
        """Initialize the scheduler.

        Args:
            warmup_iters: Number of warmup iterations.
            lr_decay_iters: Total iterations for decay.
            min_lr: Minimum learning rate.
        """
        self.warmup_iters: int = warmup_iters
        self.lr_decay_iters: int = lr_decay_iters
        self.min_lr: float = min_lr

    def get_lr(self, trainer: Trainer) -> float:
        """Calculate learning rate for current iteration.

        Args:
            trainer: The trainer instance.

        Returns:
            Current learning rate.
        """
        it = trainer.iter_num
        config = trainer.config

        # Linear warmup
        if it < self.warmup_iters:
            return config.learning_rate * it / self.warmup_iters

        # Cosine decay
        if it > self.lr_decay_iters:
            return self.min_lr

        decay_ratio = (it - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (config.learning_rate - self.min_lr)

    def __call__(self, trainer: Trainer) -> None:
        """Update learning rate.

        Args:
            trainer: The trainer instance.
        """
        if trainer.optimizer is None:
            return

        lr = self.get_lr(trainer)
        for param_group in trainer.optimizer.param_groups:
            param_group["lr"] = lr


# Import math for LearningRateSchedulerCallback
import math
