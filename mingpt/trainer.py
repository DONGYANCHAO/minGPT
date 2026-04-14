"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict
from typing import Optional, Dict, Any

import torch
from torch.utils.data.dataloader import DataLoader

from mingpt.utils import CfgNode as CN
from mingpt.checkpoint import CheckpointManager
from mingpt.logger import TrainingLogger
from mingpt.evaluator import Evaluator


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        C.device = 'auto'
        C.num_workers = 4
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1
        C.grad_norm_clip = 1.0
        C.checkpoint = CheckpointManager.get_default_config()
        C.logging = TrainingLogger.get_default_config()
        C.evaluation = Evaluator.get_default_config()
        C.enable_checkpoint = True
        C.enable_logging = True
        C.enable_evaluation = False
        C.eval_interval = 1000
        C.eval_dataset = None
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.logger: Optional[TrainingLogger] = None
        self.evaluator: Optional[Evaluator] = None

        self._setup_checkpoint()
        self._setup_logging()
        self._setup_evaluation()

    def _setup_checkpoint(self):
        if not self.config.enable_checkpoint:
            return
        
        checkpoint_config = self.config.checkpoint
        self.checkpoint_manager = CheckpointManager(
            checkpoint_config,
            self.model,
            optimizer=None,
            device=self.device
        )
        print(f"Checkpoint enabled: {checkpoint_config.checkpoint_dir}")

    def _setup_logging(self):
        if not self.config.enable_logging:
            return
        
        logging_config = self.config.logging
        model_config = getattr(self.model, 'config', None)
        self.logger = TrainingLogger(logging_config, model_config)
        
        if self.logger:
            self.logger.log_model_summary(self.model)
        
        print(f"Logging enabled: {logging_config.log_dir}")

    def _setup_evaluation(self):
        if not self.config.enable_evaluation:
            return
        
        eval_config = self.config.evaluation
        self.evaluator = Evaluator(eval_config, self.model, self.device)
        print("Evaluation enabled")

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def restore_from_checkpoint(self, checkpoint_path: Optional[str] = None,
                                load_best: bool = False) -> bool:
        if self.checkpoint_manager is None:
            print("Checkpoint manager not enabled")
            return False
        
        try:
            checkpoint = self.checkpoint_manager.load(
                checkpoint_path=checkpoint_path,
                load_best=load_best
            )
            self.iter_num = checkpoint.get('iter_num', 0)
            print(f"Restored from checkpoint at iteration {self.iter_num}")
            return True
        except (FileNotFoundError, ValueError) as e:
            print(f"Could not restore checkpoint: {e}")
            return False

    def run(self):
        model, config = self.model, self.config

        self.optimizer = model.configure_optimizers(config)
        
        if self.checkpoint_manager:
            self.checkpoint_manager.optimizer = self.optimizer

        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, replacement=True, num_samples=int(1e10)
            ),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        if self.iter_num == 0:
            self.iter_time = time.time()
        data_iter = iter(train_loader)

        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            logits, self.loss = model(x, y)

            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            self._log_iteration()
            self._save_checkpoint_if_needed()
            self._evaluate_if_needed()

            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

        self._finish_training()

    def _log_iteration(self):
        if self.logger is None:
            return
        
        log_interval = self.config.logging.log_interval
        if self.iter_num % log_interval != 0:
            return
        
        metrics = {'loss': self.loss.item()}
        
        extra = {
            'learning_rate': self._get_current_lr(),
            'iter_time': self.iter_dt,
        }
        
        if self.iter_num % (log_interval * 10) == 0:
            grad_norm = self.logger.log_gradients(self.model, self.iter_num)
            if grad_norm:
                extra['grad_norm'] = grad_norm
        
        self.logger.log(self.iter_num, metrics, extra)

    def _get_current_lr(self) -> float:
        if self.optimizer is None:
            return 0.0
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def _save_checkpoint_if_needed(self):
        if self.checkpoint_manager is None:
            return
        
        save_interval = self.config.checkpoint.save_interval
        if self.iter_num % save_interval != 0:
            return
        
        metrics = {'loss': self.loss.item()}
        self.checkpoint_manager.save(self.iter_num, metrics)
        print(f"Checkpoint saved at iteration {self.iter_num}")

    def _evaluate_if_needed(self):
        if self.evaluator is None:
            return
        
        if self.config.eval_dataset is None:
            return
        
        eval_interval = self.config.eval_interval
        if self.iter_num % eval_interval != 0:
            return
        
        self.model.eval()
        results = self.evaluator.evaluate(self.config.eval_dataset)
        self.model.train()
        
        if self.logger:
            self.logger.log(self.iter_num, results, {'phase': 'evaluation'})
        
        print(f"Evaluation at iteration {self.iter_num}: {results}")

    def _finish_training(self):
        if self.checkpoint_manager:
            metrics = {'loss': self.loss.item() if hasattr(self, 'loss') else 0.0}
            self.checkpoint_manager.save(self.iter_num, metrics)
            print(f"Final checkpoint saved at iteration {self.iter_num}")
        
        if self.logger:
            self.logger.close()
            print("Training report generated")

    def get_training_state(self) -> Dict[str, Any]:
        return {
            'iter_num': self.iter_num,
            'loss': self.loss.item() if hasattr(self, 'loss') else None,
            'learning_rate': self._get_current_lr(),
            'device': self.device,
        }
