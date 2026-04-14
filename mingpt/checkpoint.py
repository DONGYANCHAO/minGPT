"""
Checkpoint management for minGPT training.
Supports automatic saving, best model tracking, and recovery from interruptions.
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import torch

from mingpt.utils import CfgNode as CN


class CheckpointManager:
    """
    Manages model checkpoints with support for:
    - Periodic automatic saving
    - Keeping only the latest N checkpoints
    - Saving the best model based on a metric
    - Recovery from training interruptions
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.checkpoint_dir = 'checkpoints'
        C.save_interval = 1000
        C.max_checkpoints = 5
        C.save_best = True
        C.best_metric = 'loss'
        C.best_mode = 'min'
        return C

    def __init__(self, config, model, optimizer=None, device='cpu'):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_file = self.checkpoint_dir / 'checkpoint_history.json'
        self.best_model_file = self.checkpoint_dir / 'best_model.pt'
        
        self.history: List[Dict[str, Any]] = self._load_history()
        self.best_metric_value: Optional[float] = None
        self.iter_num = 0

    def _load_history(self) -> List[Dict[str, Any]]:
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        return []

    def _save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def save(self, iter_num: int, metrics: Optional[Dict[str, float]] = None, 
             extra_data: Optional[Dict[str, Any]] = None):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'checkpoint_iter_{iter_num}_{timestamp}.pt'
        filepath = self.checkpoint_dir / filename
        
        checkpoint = {
            'iter_num': iter_num,
            'model_state_dict': self.model.state_dict(),
            'config': self._get_model_config(),
            'timestamp': timestamp,
        }
        
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if extra_data is not None:
            checkpoint['extra_data'] = extra_data
        
        torch.save(checkpoint, filepath)
        
        self.history.append({
            'filename': filename,
            'iter_num': iter_num,
            'timestamp': timestamp,
            'metrics': metrics or {},
        })
        
        self._cleanup_old_checkpoints()
        self._save_history()
        
        if self.config.save_best and metrics:
            self._update_best_model(iter_num, metrics, filepath)
        
        return filepath

    def _get_model_config(self) -> Dict[str, Any]:
        if hasattr(self.model, 'config'):
            if hasattr(self.model.config, 'to_dict'):
                return self.model.config.to_dict()
            return vars(self.model.config)
        return {}

    def _update_best_model(self, iter_num: int, metrics: Dict[str, float], 
                           source_path: Path):
        metric_name = self.config.best_metric
        if metric_name not in metrics:
            return
        
        current_value = metrics[metric_name]
        
        is_best = False
        if self.best_metric_value is None:
            is_best = True
        elif self.config.best_mode == 'min':
            is_best = current_value < self.best_metric_value
        else:
            is_best = current_value > self.best_metric_value
        
        if is_best:
            self.best_metric_value = current_value
            checkpoint = torch.load(source_path, map_location='cpu')
            checkpoint['best_metric_value'] = current_value
            torch.save(checkpoint, self.best_model_file)

    def _cleanup_old_checkpoints(self):
        while len(self.history) > self.config.max_checkpoints:
            old_entry = self.history.pop(0)
            old_file = self.checkpoint_dir / old_entry['filename']
            if old_file.exists():
                os.remove(old_file)

    def load(self, checkpoint_path: Optional[str] = None, 
             load_best: bool = False) -> Dict[str, Any]:
        if load_best:
            if not self.best_model_file.exists():
                raise FileNotFoundError(f"Best model not found at {self.best_model_file}")
            filepath = self.best_model_file
        elif checkpoint_path:
            filepath = Path(checkpoint_path)
        else:
            if not self.history:
                raise ValueError("No checkpoints available")
            latest = self.history[-1]
            filepath = self.checkpoint_dir / latest['filename']
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.iter_num = checkpoint.get('iter_num', 0)
        
        return checkpoint

    def get_latest_checkpoint(self) -> Optional[Path]:
        if not self.history:
            return None
        latest = self.history[-1]
        return self.checkpoint_dir / latest['filename']

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        return [
            {
                'path': str(self.checkpoint_dir / entry['filename']),
                'iter_num': entry['iter_num'],
                'timestamp': entry['timestamp'],
                'metrics': entry['metrics'],
            }
            for entry in self.history
        ]

    def export_checkpoint(self, checkpoint_path: str, export_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        export_data = {
            'model_state_dict': checkpoint['model_state_dict'],
            'config': checkpoint.get('config', {}),
            'iter_num': checkpoint.get('iter_num', 0),
            'metrics': checkpoint.get('metrics', {}),
        }
        torch.save(export_data, export_path)

    def cleanup(self, keep_last: int = 1):
        while len(self.history) > keep_last:
            old_entry = self.history.pop(0)
            old_file = self.checkpoint_dir / old_entry['filename']
            if old_file.exists():
                os.remove(old_file)
        self._save_history()


def list_checkpoints(checkpoint_dir: str):
    manager = CheckpointManager(
        CheckpointManager.get_default_config().__class__(checkpoint_dir=checkpoint_dir),
        model=torch.nn.Identity()
    )
    checkpoints = manager.list_checkpoints()
    
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoint(s):\n")
    print(f"{'Iter':<10} {'Timestamp':<20} {'Metrics'}")
    print("-" * 60)
    for cp in checkpoints:
        metrics_str = ', '.join(f"{k}: {v:.4f}" for k, v in cp['metrics'].items())
        print(f"{cp['iter_num']:<10} {cp['timestamp']:<20} {metrics_str}")


def restore_checkpoint(checkpoint_path: str, output_dir: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, 'restored_config.json')
    with open(config_path, 'w') as f:
        json.dump(checkpoint.get('config', {}), f, indent=2)
    
    state_dict_path = os.path.join(output_dir, 'model_state.pt')
    torch.save(checkpoint['model_state_dict'], state_dict_path)
    
    if 'optimizer_state_dict' in checkpoint:
        optimizer_path = os.path.join(output_dir, 'optimizer_state.pt')
        torch.save(checkpoint['optimizer_state_dict'], optimizer_path)
    
    info = {
        'iter_num': checkpoint.get('iter_num', 0),
        'timestamp': checkpoint.get('timestamp', ''),
        'metrics': checkpoint.get('metrics', {}),
    }
    info_path = os.path.join(output_dir, 'checkpoint_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Checkpoint restored to {output_dir}")
    print(f"  - Iteration: {info['iter_num']}")
    print(f"  - Metrics: {info['metrics']}")


def export_checkpoint(checkpoint_path: str, export_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    export_data = {
        'model_state_dict': checkpoint['model_state_dict'],
        'config': checkpoint.get('config', {}),
        'iter_num': checkpoint.get('iter_num', 0),
        'metrics': checkpoint.get('metrics', {}),
    }
    
    torch.save(export_data, export_path)
    print(f"Checkpoint exported to {export_path}")


def cleanup_checkpoints(checkpoint_dir: str, keep_last: int = 1):
    history_file = os.path.join(checkpoint_dir, 'checkpoint_history.json')
    
    if not os.path.exists(history_file):
        print("No checkpoint history found.")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    removed_count = 0
    while len(history) > keep_last:
        old_entry = history.pop(0)
        old_file = os.path.join(checkpoint_dir, old_entry['filename'])
        if os.path.exists(old_file):
            os.remove(old_file)
            removed_count += 1
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Cleaned up {removed_count} checkpoint(s), keeping {keep_last}")


def main():
    parser = argparse.ArgumentParser(description='Checkpoint management tools for minGPT')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    list_parser = subparsers.add_parser('list', help='List all checkpoints')
    list_parser.add_argument('--dir', type=str, default='checkpoints',
                            help='Checkpoint directory')
    
    restore_parser = subparsers.add_parser('restore', help='Restore a checkpoint')
    restore_parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
    restore_parser.add_argument('--output', type=str, default='restored_checkpoint',
                               help='Output directory')
    
    export_parser = subparsers.add_parser('export', help='Export a checkpoint')
    export_parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
    export_parser.add_argument('--output', type=str, required=True,
                              help='Output file path')
    
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup old checkpoints')
    cleanup_parser.add_argument('--dir', type=str, default='checkpoints',
                               help='Checkpoint directory')
    cleanup_parser.add_argument('--keep', type=int, default=1,
                               help='Number of checkpoints to keep')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_checkpoints(args.dir)
    elif args.command == 'restore':
        restore_checkpoint(args.checkpoint, args.output)
    elif args.command == 'export':
        export_checkpoint(args.checkpoint, args.output)
    elif args.command == 'cleanup':
        cleanup_checkpoints(args.dir, args.keep)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
