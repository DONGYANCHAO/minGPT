"""
Checkpoint manager for saving and restoring training state.
Supports periodic auto-save, keeping latest N checkpoints, and saving best model.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any

import torch

from mingpt.utils import CfgNode as CN

class CheckpointManager:

    @staticmethod
    def get_default_config():
        C = CN()
        C.enabled = True
        C.checkpoint_dir = './checkpoints'
        C.save_interval = 100
        C.keep_latest_n = 5
        C.save_best = True
        C.best_metric = 'loss'
        C.best_mode = 'min'
        C.create_run_dir = True
        C.run_name = None
        return C

    def __init__(self, config):
        self.config = config
        self.checkpoints: List[Dict] = []
        self.best_checkpoint: Optional[Dict] = None
        self.best_metric_value = float('inf') if config.best_mode == 'min' else float('-inf')

        if config.create_run_dir:
            run_name = config.run_name or f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            self.checkpoint_dir = os.path.join(config.checkpoint_dir, run_name)
        else:
            self.checkpoint_dir = config.checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.index_path = os.path.join(self.checkpoint_dir, 'checkpoint_index.json')
        self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'r') as f:
                data = json.load(f)
                self.checkpoints = data.get('checkpoints', [])
                self.best_checkpoint = data.get('best_checkpoint')
                if self.best_checkpoint:
                    self.best_metric_value = self.best_checkpoint.get('metric_value', self.best_metric_value)

    def _save_index(self):
        data = {
            'checkpoints': self.checkpoints,
            'best_checkpoint': self.best_checkpoint,
            'updated_at': datetime.now().isoformat()
        }
        with open(self.index_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _get_checkpoint_path(self, iter_num: int) -> str:
        return os.path.join(self.checkpoint_dir, f'checkpoint_iter_{iter_num}.pt')

    def _get_best_model_path(self) -> str:
        return os.path.join(self.checkpoint_dir, 'best_model.pt')

    def save(self, model, optimizer, iter_num: int, config=None, metrics: Optional[Dict] = None) -> str:
        if not self.config.enabled:
            return ''

        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'iter_num': iter_num,
            'config': config.to_dict() if config else None,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }

        ckpt_path = self._get_checkpoint_path(iter_num)
        torch.save(checkpoint_data, ckpt_path)

        checkpoint_info = {
            'path': ckpt_path,
            'iter_num': iter_num,
            'metrics': metrics or {},
            'timestamp': checkpoint_data['timestamp']
        }
        self.checkpoints.append(checkpoint_info)

        if self.config.keep_latest_n > 0 and len(self.checkpoints) > self.config.keep_latest_n:
            old_ckpt = self.checkpoints.pop(0)
            if os.path.exists(old_ckpt['path']):
                os.remove(old_ckpt['path'])

        if self.config.save_best and metrics:
            current_metric = metrics.get(self.config.best_metric)
            if current_metric is not None:
                is_better = False
                if self.config.best_mode == 'min' and current_metric < self.best_metric_value:
                    is_better = True
                elif self.config.best_mode == 'max' and current_metric > self.best_metric_value:
                    is_better = True

                if is_better:
                    self.best_metric_value = current_metric
                    best_path = self._get_best_model_path()
                    shutil.copy2(ckpt_path, best_path)
                    self.best_checkpoint = {
                        'path': best_path,
                        'iter_num': iter_num,
                        'metrics': metrics,
                        'metric_value': current_metric,
                        'timestamp': checkpoint_data['timestamp']
                    }

        self._save_index()
        return ckpt_path

    def load(self, ckpt_path: str, model, optimizer=None, device='cpu') -> Dict:
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return {
            'iter_num': checkpoint.get('iter_num', 0),
            'config': checkpoint.get('config'),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp')
        }

    def load_latest(self, model, optimizer=None, device='cpu') -> Optional[Dict]:
        if not self.checkpoints:
            return None
        latest_ckpt = self.checkpoints[-1]
        return self.load(latest_ckpt['path'], model, optimizer, device)

    def load_best(self, model, optimizer=None, device='cpu') -> Optional[Dict]:
        if not self.best_checkpoint:
            return None
        return self.load(self.best_checkpoint['path'], model, optimizer, device)

    def list_checkpoints(self) -> List[Dict]:
        return self.checkpoints.copy()

    def get_best_checkpoint_info(self) -> Optional[Dict]:
        return self.best_checkpoint.copy() if self.best_checkpoint else None

    def export_model(self, ckpt_path: str, export_path: str, only_weights: bool = True) -> str:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if only_weights:
            torch.save(checkpoint['model_state_dict'], export_path)
        else:
            torch.save(checkpoint, export_path)
        return export_path

    def clean_old_checkpoints(self, keep_n: Optional[int] = None) -> int:
        keep_n = keep_n or self.config.keep_latest_n
        removed = 0
        while len(self.checkpoints) > keep_n:
            old_ckpt = self.checkpoints.pop(0)
            if os.path.exists(old_ckpt['path']):
                os.remove(old_ckpt['path'])
                removed += 1
        self._save_index()
        return removed

    def cleanup_all(self) -> None:
        shutil.rmtree(self.checkpoint_dir, ignore_errors=True)
        self.checkpoints = []
        self.best_checkpoint = None
