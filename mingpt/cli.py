#!/usr/bin/env python
"""
minGPT Command Line Tools
Provides unified CLI for checkpoint management, logging, and evaluation.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def checkpoint_command(args):
    from mingpt.checkpoint import (
        list_checkpoints, restore_checkpoint, 
        export_checkpoint, cleanup_checkpoints
    )
    
    if args.action == 'list':
        list_checkpoints(args.dir)
    elif args.action == 'restore':
        restore_checkpoint(args.checkpoint, args.output)
    elif args.action == 'export':
        export_checkpoint(args.checkpoint, args.output)
    elif args.action == 'cleanup':
        cleanup_checkpoints(args.dir, args.keep)


def logger_command(args):
    from mingpt.logger import generate_dashboard_html, WebMonitor
    import time
    
    if args.action == 'dashboard':
        generate_dashboard_html(args.log_dir, args.output)
    elif args.action == 'monitor':
        monitor = WebMonitor(args.log_dir, args.port)
        if monitor.start():
            try:
                print("Press Ctrl+C to stop the monitor")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                monitor.stop()


def eval_command(args):
    from mingpt.evaluator import evaluate_model, compare_models
    from mingpt.utils import CfgNode as CN
    
    if args.action == 'evaluate':
        config = CN()
        config.batch_size = args.batch_size
        config.num_workers = args.num_workers
        config.device = args.device
        config.compute_perplexity = True
        config.compute_accuracy = True
        config.compute_bleu = args.bleu
        config.compute_rouge = args.rouge
        config.max_samples = args.max_samples
        
        print(f"Evaluating model: {args.model}")
        print("Note: This requires a dataset. Please use programmatically with your dataset.")
        
    elif args.action == 'compare':
        if len(args.models) != len(args.names):
            print("Error: Number of models and names must match")
            return
        
        print(f"Comparing {len(args.models)} models:")
        for name, path in zip(args.names, args.models):
            print(f"  - {name}: {path}")
        print("Note: This requires a dataset. Please use programmatically with your dataset.")


def train_command(args):
    import torch
    from mingpt.model import GPT
    from mingpt.trainer import Trainer
    from mingpt.utils import CfgNode as CN
    
    model_config = GPT.get_default_config()
    model_config.model_type = args.model_type
    model_config.vocab_size = args.vocab_size
    model_config.block_size = args.block_size
    
    trainer_config = Trainer.get_default_config()
    trainer_config.max_iters = args.max_iters
    trainer_config.batch_size = args.batch_size
    trainer_config.learning_rate = args.learning_rate
    
    trainer_config.checkpoint.checkpoint_dir = args.checkpoint_dir
    trainer_config.checkpoint.save_interval = args.save_interval
    trainer_config.checkpoint.max_checkpoints = args.max_checkpoints
    
    trainer_config.logging.log_dir = args.log_dir
    trainer_config.logging.experiment_name = args.experiment_name
    trainer_config.logging.log_interval = args.log_interval
    trainer_config.logging.use_tensorboard = args.tensorboard
    trainer_config.logging.use_wandb = args.wandb
    
    trainer_config.enable_checkpoint = not args.no_checkpoint
    trainer_config.enable_logging = not args.no_logging
    
    print("Training configuration:")
    print(f"  Model type: {args.model_type}")
    print(f"  Max iterations: {args.max_iters}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Log dir: {args.log_dir}")
    print("\nNote: This requires a dataset. Please use programmatically with your dataset.")


def main():
    parser = argparse.ArgumentParser(
        description='minGPT Command Line Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all checkpoints
  python mingpt/cli.py checkpoint list --dir checkpoints
  
  # Restore a checkpoint
  python mingpt/cli.py checkpoint restore checkpoint.pt --output restored/
  
  # Generate monitoring dashboard
  python mingpt/cli.py logger dashboard --log-dir logs/experiment
  
  # Start web monitor
  python mingpt/cli.py logger monitor --log-dir logs/experiment --port 6007
  
  # Evaluate a model
  python mingpt/cli.py eval evaluate --model checkpoint.pt --batch-size 32
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    checkpoint_parser = subparsers.add_parser('checkpoint', help='Checkpoint management')
    checkpoint_parser.add_argument('action', choices=['list', 'restore', 'export', 'cleanup'],
                                  help='Checkpoint action')
    checkpoint_parser.add_argument('--dir', type=str, default='checkpoints',
                                  help='Checkpoint directory')
    checkpoint_parser.add_argument('--checkpoint', type=str,
                                  help='Checkpoint file path')
    checkpoint_parser.add_argument('--output', type=str,
                                  help='Output path')
    checkpoint_parser.add_argument('--keep', type=int, default=1,
                                  help='Number of checkpoints to keep (for cleanup)')
    
    logger_parser = subparsers.add_parser('logger', help='Logging and monitoring')
    logger_parser.add_argument('action', choices=['dashboard', 'monitor'],
                              help='Logger action')
    logger_parser.add_argument('--log-dir', type=str, required=True,
                              help='Log directory')
    logger_parser.add_argument('--output', type=str,
                              help='Output file path (for dashboard)')
    logger_parser.add_argument('--port', type=int, default=6007,
                              help='Port for web monitor')
    
    eval_parser = subparsers.add_parser('eval', help='Model evaluation')
    eval_parser.add_argument('action', choices=['evaluate', 'compare'],
                            help='Evaluation action')
    eval_parser.add_argument('--model', type=str,
                            help='Model checkpoint path')
    eval_parser.add_argument('--models', type=str, nargs='+',
                            help='Model checkpoint paths (for compare)')
    eval_parser.add_argument('--names', type=str, nargs='+',
                            help='Model names (for compare)')
    eval_parser.add_argument('--output', type=str, default='eval_results',
                            help='Output directory')
    eval_parser.add_argument('--batch-size', type=int, default=64,
                            help='Evaluation batch size')
    eval_parser.add_argument('--num-workers', type=int, default=4,
                            help='Number of data loading workers')
    eval_parser.add_argument('--device', type=str, default='auto',
                            help='Device to use')
    eval_parser.add_argument('--max-samples', type=int,
                            help='Maximum samples to evaluate')
    eval_parser.add_argument('--bleu', action='store_true',
                            help='Compute BLEU score')
    eval_parser.add_argument('--rouge', action='store_true',
                            help='Compute ROUGE score')
    
    train_parser = subparsers.add_parser('train', help='Training utilities')
    train_parser.add_argument('--model-type', type=str, default='gpt-nano',
                             help='Model type')
    train_parser.add_argument('--vocab-size', type=int, default=50257,
                             help='Vocabulary size')
    train_parser.add_argument('--block-size', type=int, default=128,
                             help='Block size')
    train_parser.add_argument('--max-iters', type=int, default=10000,
                             help='Maximum iterations')
    train_parser.add_argument('--batch-size', type=int, default=64,
                             help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, default=3e-4,
                             help='Learning rate')
    train_parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                             help='Checkpoint directory')
    train_parser.add_argument('--save-interval', type=int, default=1000,
                             help='Checkpoint save interval')
    train_parser.add_argument('--max-checkpoints', type=int, default=5,
                             help='Maximum checkpoints to keep')
    train_parser.add_argument('--log-dir', type=str, default='logs',
                             help='Log directory')
    train_parser.add_argument('--experiment-name', type=str, default='experiment',
                             help='Experiment name')
    train_parser.add_argument('--log-interval', type=int, default=100,
                             help='Logging interval')
    train_parser.add_argument('--tensorboard', action='store_true',
                             help='Enable TensorBoard')
    train_parser.add_argument('--wandb', action='store_true',
                             help='Enable wandb')
    train_parser.add_argument('--no-checkpoint', action='store_true',
                             help='Disable checkpointing')
    train_parser.add_argument('--no-logging', action='store_true',
                             help='Disable logging')
    
    args = parser.parse_args()
    
    if args.command == 'checkpoint':
        checkpoint_command(args)
    elif args.command == 'logger':
        logger_command(args)
    elif args.command == 'eval':
        eval_command(args)
    elif args.command == 'train':
        train_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
