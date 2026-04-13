"""
Generic training loop for neural networks, decoupled from GPT specifics.
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, unique
from itertools import cycle
from typing import (
    Any, Callable, DefaultDict, Dict, List,
    Optional, Tuple, TypeVar, Union, cast
)

from torch.utils.data.sampler import Sampler

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler

from mingpt.utils import CfgNode as CN


Callback = Callable[['Trainer'], None]
T = TypeVar('T')

INFINITE_SAMPLES = int(1e10)
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_NORM_CLIP = 1.0
DEFAULT_NUM_WORKERS = 4
DEFAULT_DEVICE = 'auto'
DEVICE_AUTO = 'auto'
DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'
BETAS_DEFAULT = (0.9, 0.95)


@unique
class CallbackEvent(Enum):
    """Callback event types for training lifecycle."""

    BATCH_END = 'on_batch_end'
    EPOCH_END = 'on_epoch_end'
    TRAIN_START = 'on_train_start'
    TRAIN_END = 'on_train_end'


@dataclass(frozen=True)
class TrainerState:
    """Immutable trainer state for callback context."""

    iter_num: int
    iter_dt: float
    loss: float
    device: str


class Trainer:
    """Generic trainer for PyTorch models.

    Attributes:
        config: Training configuration.
        model: PyTorch model to train.
        optimizer: Optimizer instance.
        train_dataset: Dataset for training.
        callbacks: Dictionary mapping events to callback lists.
        device: Device to run training on.
        iter_num: Current iteration number.
        iter_time: Timestamp of previous iteration.
        iter_dt: Time delta of previous iteration.
        loss: Current loss value.
    """

    config: CN
    model: Any
    optimizer: Optional[torch.optim.Optimizer]
    train_dataset: Dataset
    callbacks: DefaultDict[str, List[Callback]]
    device: str
    iter_num: int
    iter_time: float
    iter_dt: float
    loss: torch.Tensor

    @staticmethod
    def get_default_config() -> CN:
        """Get default training configuration.

        Returns:
            CfgNode with default training hyperparameters.
        """
        C = CN()
        C.device = DEFAULT_DEVICE
        C.num_workers = DEFAULT_NUM_WORKERS
        C.max_iters = None
        C.batch_size = DEFAULT_BATCH_SIZE
        C.learning_rate = DEFAULT_LEARNING_RATE
        C.betas = BETAS_DEFAULT
        C.weight_decay = DEFAULT_WEIGHT_DECAY
        C.grad_norm_clip = DEFAULT_GRAD_NORM_CLIP
        return C

    def __init__(self, config: CN, model: Any, train_dataset: Dataset) -> None:
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        if config.device == DEVICE_AUTO:
            self.device = DEVICE_CUDA if torch.cuda.is_available() else DEVICE_CPU
        else:
            self.device = config.device

        self.model = self.model.to(self.device)
        print(f"Running on device: {self.device}")

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.loss = torch.tensor(0.0)

    def get_state(self) -> TrainerState:
        """Get immutable snapshot of trainer state.

        Returns:
            TrainerState with current training metrics.
        """
        return TrainerState(
            iter_num=self.iter_num,
            iter_dt=self.iter_dt,
            loss=float(self.loss.item()),
            device=self.device
        )

    @staticmethod
    def _resolve_event(event: Union[CallbackEvent, str]) -> str:
        """Resolve event to string key for backward compatibility.

        Args:
            event: CallbackEvent enum or string event name.

        Returns:
            String event key.
        """
        if isinstance(event, str):
            return event
        return event.value

    def add_callback(self, event: Union[CallbackEvent, str], callback: Callback) -> None:
        """Register a callback for an event.

        Args:
            event: Event to trigger callback on (CallbackEvent enum or string).
            callback: Callable taking Trainer as argument.
        """
        self.callbacks[self._resolve_event(event)].append(callback)

    def set_callback(self, event: Union[CallbackEvent, str], callback: Callback) -> None:
        """Set exclusive callback for an event, replacing existing.

        Args:
            event: Event to trigger callback on (CallbackEvent enum or string).
            callback: Callable taking Trainer as argument.
        """
        self.callbacks[self._resolve_event(event)] = [callback]

    def trigger_callbacks(self, event: Union[CallbackEvent, str]) -> None:
        """Execute all callbacks registered for an event.

        Args:
            event: Event to trigger callbacks for (CallbackEvent enum or string).
        """
        for callback in self.callbacks.get(self._resolve_event(event), []):
            callback(self)

    @staticmethod
    def _infinite_dataloader(
        dataset: Dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = True,
        shuffle: bool = False
    ) -> DataLoader:
        """Create infinite cyclic dataloader.

        Args:
            dataset: Dataset to load from.
            batch_size: Batch size.
            num_workers: Number of worker processes.
            pin_memory: Whether to pin memory for CUDA transfer.
            shuffle: Whether to shuffle data.

        Returns:
            DataLoader with infinite sampling.
        """
        sampler = RandomSampler(
            cast(Any, dataset),
            replacement=True,
            num_samples=INFINITE_SAMPLES
        )
        loader = DataLoader(
            dataset,
            sampler=cast(Sampler[int], sampler),
            shuffle=shuffle,
            pin_memory=pin_memory,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        return loader

    def run(self) -> None:
        """Main training loop."""
        model, config = self.model, self.config

        self.optimizer = model.configure_optimizers(config)

        train_loader = self._infinite_dataloader(
            self.train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=True,
            shuffle=False
        )

        data_iter = cycle(train_loader)

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()

        while True:
            batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            logits, self.loss = model(x, y)

            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks(CallbackEvent.BATCH_END)
            self.iter_num += 1

            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
