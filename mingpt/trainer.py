"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN
from mingpt.checkpoint import CheckpointManager

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        # checkpoint parameters
        C.checkpoint = CheckpointManager.get_default_config()
        C.resume = False
        C.resume_from = None
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0
        self.loss = None

        # initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(config.checkpoint)

        # resume from checkpoint if enabled
        if config.resume:
            self._resume_checkpoint()

    def _resume_checkpoint(self):
        resume_from = self.config.resume_from
        if resume_from == 'latest' or resume_from is None:
            print(f"resuming from latest checkpoint in {self.checkpoint_manager.checkpoint_dir}")
            resume_info = self.checkpoint_manager.load_latest(
                self.model, self.optimizer, self.device
            )
        elif resume_from == 'best':
            print(f"resuming from best checkpoint in {self.checkpoint_manager.checkpoint_dir}")
            resume_info = self.checkpoint_manager.load_best(
                self.model, self.optimizer, self.device
            )
        else:
            print(f"resuming from {resume_from}")
            resume_info = self.checkpoint_manager.load(
                resume_from, self.model, self.optimizer, self.device
            )

        if resume_info:
            self.iter_num = resume_info['iter_num']
            print(f"resumed from iteration {self.iter_num}")
        else:
            print("no checkpoint found, starting from scratch")

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # re-apply optimizer state if we are resuming
        if config.resume and self.config.resume_from:
            if self.config.resume_from == 'latest':
                self.checkpoint_manager.load_latest(model, self.optimizer, self.device)
            elif self.config.resume_from == 'best' and self.checkpoint_manager.best_checkpoint:
                self.checkpoint_manager.load_best(model, self.optimizer, self.device)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        if not config.resume:
            self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            logits, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')

            # auto-save checkpoint
            if config.checkpoint.enabled and self.iter_num % config.checkpoint.save_interval == 0:
                metrics = {'loss': self.loss.item()}
                self.checkpoint_manager.save(
                    model, self.optimizer, self.iter_num,
                    config, metrics
                )

            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
