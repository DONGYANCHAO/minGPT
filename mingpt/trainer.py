"""
Training loop implementation for neural networks.

This module provides a generic Trainer class that handles:
- Device management
- Data loading with infinite iteration
- Optimization with gradient clipping
- Callback system for custom behavior
"""

from __future__ import annotations

import time
from collections import defaultdict
from enum import Enum, auto
from itertools import cycle
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import torch
from torch.utils.data import DataLoader, Dataset

from mingpt.utils import CfgNode as CN

T = TypeVar('T')

DEFAULT_NUM_WORKERS: int = 4
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_LEARNING_RATE: float = 3e-4
DEFAULT_BETAS: Tuple[float, float] = (0.9, 0.95)
DEFAULT_WEIGHT_DECAY: float = 0.1
DEFAULT_GRAD_NORM_CLIP: float = 1.0
INFINITE_SAMPLE_COUNT: int = int(1e10)


class CallbackEvent(Enum):
    """Enumeration of callback event types."""
    ON_BATCH_END = auto()
    ON_EPOCH_END = auto()
    ON_TRAIN_START = auto()
    ON_TRAIN_END = auto()


CallbackFunction = Callable[['Trainer'], None]


class Trainer:
    """
    Generic trainer for neural network models.

    This class provides a complete training loop with:
    - Automatic device selection (CUDA/CPU)
    - Infinite data iteration
    - Gradient clipping
    - Callback system for extensibility

    Attributes:
        config: Training configuration.
        model: The neural network model to train.
        optimizer: The optimizer instance (created on run).
        train_dataset: Dataset for training.
        device: Device to train on.
        callbacks: Dictionary of callback functions by event.
        iter_num: Current iteration number.
        iter_time: Time of last iteration.
        iter_dt: Time delta since last iteration.
        loss: Most recent loss value.
    """

    @staticmethod
    def get_default_config() -> CN:
        """
        Get default training configuration.

        Returns:
            Default configuration with standard hyperparameters.
        """
        C = CN()
        C.device = 'auto'
        C.num_workers = DEFAULT_NUM_WORKERS
        C.max_iters = None
        C.batch_size = DEFAULT_BATCH_SIZE
        C.learning_rate = DEFAULT_LEARNING_RATE
        C.betas = DEFAULT_BETAS
        C.weight_decay = DEFAULT_WEIGHT_DECAY
        C.grad_norm_clip = DEFAULT_GRAD_NORM_CLIP
        return C

    def __init__(
        self,
        config: CN,
        model: torch.nn.Module,
        train_dataset: Dataset
    ) -> None:
        """
        Initialize the trainer.

        Args:
            config: Training configuration containing:
                - device: 'auto', 'cuda', or 'cpu'
                - num_workers: DataLoader worker count
                - batch_size: Training batch size
                - learning_rate: Optimizer learning rate
                - betas: Adam beta parameters
                - weight_decay: Weight decay coefficient
                - grad_norm_clip: Maximum gradient norm
            model: Neural network model to train.
            train_dataset: Dataset for training data.

        Raises:
            ValueError: If device is invalid.
        """
        self.config: CN = config
        self.model: torch.nn.Module = model
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.train_dataset: Dataset = train_dataset
        self.callbacks: Dict[CallbackEvent, List[CallbackFunction]] = defaultdict(list)

        self.device: str = self._resolve_device(config.device)
        self.model = self.model.to(self.device)
        print(f"running on device {self.device}")

        self.iter_num: int = 0
        self.iter_time: float = 0.0
        self.iter_dt: float = 0.0
        self.loss: Optional[torch.Tensor] = None

    def _resolve_device(self, device_config: str) -> str:
        """
        Resolve device string from configuration.

        Args:
            device_config: Device specification ('auto', 'cuda', or 'cpu').

        Returns:
            Resolved device string.

        Raises:
            ValueError: If device specification is invalid.
        """
        if device_config == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device_config in ('cuda', 'cpu'):
            return device_config
        else:
            raise ValueError(
                f"Invalid device '{device_config}'. Must be 'auto', 'cuda', or 'cpu'."
            )

    def add_callback(
        self,
        event: CallbackEvent,
        callback: CallbackFunction
    ) -> None:
        """
        Add a callback function for a specific event.

        Args:
            event: The callback event type.
            callback: Function to call, receives Trainer instance.
        """
        self.callbacks[event].append(callback)

    def set_callback(
        self,
        event: CallbackEvent,
        callback: CallbackFunction
    ) -> None:
        """
        Set a single callback for an event, replacing any existing.

        Args:
            event: The callback event type.
            callback: Function to call, receives Trainer instance.
        """
        self.callbacks[event] = [callback]

    def remove_callbacks(self, event: CallbackEvent) -> None:
        """
        Remove all callbacks for a specific event.

        Args:
            event: The callback event type.
        """
        self.callbacks[event] = []

    def trigger_callbacks(self, event: CallbackEvent) -> None:
        """
        Trigger all callbacks registered for an event.

        Args:
            event: The callback event type.
        """
        for callback in self.callbacks.get(event, []):
            callback(self)

    def _create_dataloader(self) -> DataLoader:
        """
        Create the training DataLoader.

        Returns:
            Configured DataLoader with infinite sampling.
        """
        return DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset,
                replacement=True,
                num_samples=INFINITE_SAMPLE_COUNT
            ),
            shuffle=False,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def _prepare_batch(
        self,
        batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a batch for training.

        Args:
            batch: List of tensors from DataLoader.

        Returns:
            Tuple of (input, target) tensors on device.
        """
        batch = [t.to(self.device) for t in batch]
        return batch[0], batch[1]

    def _training_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute a single training step.

        Args:
            x: Input tensor.
            y: Target tensor.

        Returns:
            Loss tensor.
        """
        logits, loss = self.model(x, y)

        self.model.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_norm_clip
        )
        self.optimizer.step()

        return loss

    def run(self) -> None:
        """
        Execute the training loop.

        This method:
        1. Creates the optimizer
        2. Creates the DataLoader with infinite iteration
        3. Runs training steps until max_iters is reached
        4. Triggers callbacks after each batch
        """
        self.optimizer = self.model.configure_optimizers(self.config)

        train_loader = self._create_dataloader()

        self.model.train()
        self.iter_num = 0
        self.iter_time = time.time()

        data_iter = cycle(train_loader)

        while True:
            batch = next(data_iter)
            x, y = self._prepare_batch(batch)

            self.loss = self._training_step(x, y)

            self.trigger_callbacks(CallbackEvent.ON_BATCH_END)

            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if self.config.max_iters is not None and self.iter_num >= self.config.max_iters:
                break
