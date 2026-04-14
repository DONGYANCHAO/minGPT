"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
import os
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN


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
        # checkpoint configuration
        C.checkpoint = CN()
        C.checkpoint.enabled = False
        C.checkpoint.checkpoint_dir = './checkpoints'
        C.checkpoint.save_interval = 1000
        C.checkpoint.max_checkpoints = 5
        C.checkpoint.save_best = True
        C.checkpoint.best_metric = 'loss'
        C.checkpoint.best_mode = 'min'
        C.checkpoint.resume_from = None  # path to checkpoint to resume from
        # logging configuration
        C.logging = CN()
        C.logging.enabled = False
        C.logging.log_dir = './logs'
        C.logging.console_log = True
        C.logging.file_log = True
        C.logging.json_log = True
        C.logging.tensorboard = False
        C.logging.wandb = False
        C.logging.wandb_project = 'mingpt'
        C.logging.log_interval = 10
        C.logging.generate_report = True
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
        
        # checkpoint manager (initialized in run())
        self.checkpoint_manager = None
        
        # training logger (initialized in run())
        self.logger = None
        
        # track learning rate for logging
        self.current_lr = config.learning_rate

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)
    
    def _setup_checkpoint_manager(self):
        """Setup checkpoint manager if enabled."""
        if not self.config.checkpoint.enabled:
            return None
        
        from mingpt.checkpoint import CheckpointManager
        
        checkpoint_manager = CheckpointManager(
            self.config.checkpoint,
            self.model,
            self.optimizer
        )
        
        # resume from checkpoint if specified
        if self.config.checkpoint.resume_from is not None:
            try:
                checkpoint_data = checkpoint_manager.load(
                    self.config.checkpoint.resume_from,
                    load_optimizer=True
                )
                self.iter_num = checkpoint_data.get('step', 0)
                print(f"Resumed training from step {self.iter_num}")
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
        
        return checkpoint_manager
    
    def _setup_logger(self):
        """Setup training logger if enabled."""
        if not self.config.logging.enabled:
            return None
        
        from mingpt.logger import TrainingLogger
        
        return TrainingLogger(self.config.logging, self.model)
    
    def _log_metrics(self, loss_value: float):
        """Log training metrics."""
        if self.logger is None:
            return
        
        # log basic metrics
        self.logger.log_metrics(self.iter_num, {
            'loss': loss_value,
            'iter_time_ms': self.iter_dt * 1000,
        }, prefix='train/')
        
        # log learning rate
        self.logger.log_learning_rate(self.iter_num, self.current_lr)
        
        # log gradients periodically
        if self.iter_num % 100 == 0:
            self.logger.log_model_gradients(self.iter_num)
    
    def _save_checkpoint(self, loss_value: float):
        """Save checkpoint if needed."""
        if self.checkpoint_manager is None:
            return
        
        if self.checkpoint_manager.should_save(self.iter_num):
            extra_data = {
                'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config),
            }
            self.checkpoint_manager.save(self.iter_num, loss_value, extra_data)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup checkpoint manager
        self.checkpoint_manager = self._setup_checkpoint_manager()
        
        # setup logger
        self.logger = self._setup_logger()
        
        if self.logger:
            self.logger.log("Starting training...")
            self.logger.log(f"Config: {config}")

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
            
            # get current learning rate
            self.current_lr = self.optimizer.param_groups[0]['lr']

            self.trigger_callbacks('on_batch_end')
            
            # logging
            loss_value = self.loss.item()
            self._log_metrics(loss_value)
            
            # checkpointing
            self._save_checkpoint(loss_value)
            
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
        
        # final cleanup
        if self.logger:
            self.logger.log("Training completed!")
            self.logger.close()


class EnhancedTrainer(Trainer):
    """
    Enhanced trainer with additional features for evaluation and generation.
    """
    
    @staticmethod
    def get_default_config():
        C = Trainer.get_default_config()
        # evaluation configuration
        C.evaluation = CN()
        C.evaluation.enabled = False
        C.evaluation.eval_interval = 500
        C.evaluation.eval_dataset = None
        C.evaluation.metrics = ['perplexity', 'accuracy']
        # generation configuration
        C.generation = CN()
        C.generation.enabled = False
        C.generation.sample_interval = 500
        C.generation.num_samples = 1
        C.generation.max_new_tokens = 100
        C.generation.temperature = 1.0
        C.generation.top_k = 10
        C.generation.prompt = None
        return C
    
    def __init__(self, config, model, train_dataset, eval_dataset=None, tokenizer=None):
        super().__init__(config, model, train_dataset)
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.evaluator = None
    
    def _setup_evaluator(self):
        """Setup evaluator if evaluation is enabled."""
        if not self.config.evaluation.enabled or self.eval_dataset is None:
            return None
        
        from mingpt.evaluator import GPTEvaluator
        
        eval_config = GPTEvaluator.get_default_config()
        eval_config.metrics = self.config.evaluation.metrics
        eval_config.device = self.device
        
        return GPTEvaluator(eval_config, self.model, self.tokenizer)
    
    def _run_evaluation(self):
        """Run evaluation on the evaluation dataset."""
        if self.evaluator is None or self.eval_dataset is None:
            return
        
        self.model.eval()
        
        try:
            results = self.evaluator.evaluate_language_modeling(self.eval_dataset)
            
            # log results
            if self.logger:
                metrics = {name: result.value for name, result in results.items()}
                self.logger.log_metrics(self.iter_num, metrics, prefix='eval/')
            
            # print results
            print("\n--- Evaluation Results ---")
            for name, result in results.items():
                print(f"  {name}: {result.value:.4f}")
            print("--------------------------\n")
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
        
        self.model.train()
    
    def _generate_samples(self):
        """Generate text samples from the model."""
        if not self.config.generation.enabled:
            return
        
        self.model.eval()
        
        gen_config = self.config.generation
        
        for i in range(gen_config.num_samples):
            # prepare prompt
            if gen_config.prompt is not None:
                if self.tokenizer:
                    prompt_ids = self.tokenizer.encode(gen_config.prompt)
                    x = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)
                else:
                    x = torch.tensor([[0]], dtype=torch.long).to(self.device)
            else:
                x = torch.tensor([[0]], dtype=torch.long).to(self.device)
            
            # generate
            with torch.no_grad():
                y = self.model.generate(
                    x,
                    max_new_tokens=gen_config.max_new_tokens,
                    temperature=gen_config.temperature,
                    do_sample=True,
                    top_k=gen_config.top_k
                )[0]
            
            # decode and log
            if self.tokenizer:
                text = self.tokenizer.decode(y.cpu().tolist())
            else:
                text = str(y.cpu().tolist())
            
            print(f"\n--- Generated Sample {i+1} ---")
            print(text)
            print("-----------------------------\n")
            
            if self.logger:
                self.logger.log_text_sample(self.iter_num, f'generation/sample_{i+1}', text)
        
        self.model.train()
    
    def run(self):
        """Extended training loop with evaluation and generation."""
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup checkpoint manager
        self.checkpoint_manager = self._setup_checkpoint_manager()
        
        # setup logger
        self.logger = self._setup_logger()
        
        # setup evaluator
        self.evaluator = self._setup_evaluator()
        
        if self.logger:
            self.logger.log("Starting enhanced training...")
            self.logger.log(f"Config: {config}")

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
            
            # get current learning rate
            self.current_lr = self.optimizer.param_groups[0]['lr']

            self.trigger_callbacks('on_batch_end')
            
            # logging
            loss_value = self.loss.item()
            self._log_metrics(loss_value)
            
            # checkpointing
            self._save_checkpoint(loss_value)
            
            # evaluation
            if config.evaluation.enabled and self.iter_num % config.evaluation.eval_interval == 0:
                self._run_evaluation()
            
            # generation
            if config.generation.enabled and self.iter_num % config.generation.sample_interval == 0:
                self._generate_samples()
            
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
        
        # final evaluation
        if config.evaluation.enabled:
            self._run_evaluation()
        
        # final cleanup
        if self.logger:
            self.logger.log("Training completed!")
            self.logger.close()
