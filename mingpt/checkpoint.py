"""
Checkpoint management for minGPT training.
Supports automatic saving, keeping top N checkpoints, and saving best model.
"""

import os
import json
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

from mingpt.utils import CfgNode as CN


@dataclass
class CheckpointInfo:
    """Metadata for a checkpoint."""
    step: int
    loss: float
    timestamp: float
    path: str
    is_best: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CheckpointInfo':
        return cls(**d)


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Features:
    - Automatic periodic saving
    - Keep only the latest N checkpoints
    - Save best model based on metric
    - Resume training from checkpoint
    """
    
    @staticmethod
    def get_default_config():
        C = CN()
        # checkpoint directory
        C.checkpoint_dir = './checkpoints'
        # save checkpoint every N steps (None to disable)
        C.save_interval = 1000
        # keep only the latest N checkpoints (None to keep all)
        C.max_checkpoints = 5
        # save best model based on metric
        C.save_best = True
        # metric to track for best model ('loss' or custom)
        C.best_metric = 'loss'
        # mode for best metric: 'min' or 'max'
        C.best_mode = 'min'
        return C
    
    def __init__(self, config, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        
        # create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # track checkpoints
        self.checkpoints: List[CheckpointInfo] = []
        self.best_metric_value = float('inf') if config.best_mode == 'min' else float('-inf')
        self.metadata_path = self.checkpoint_dir / 'checkpoint_metadata.json'
        
        # load existing metadata if available
        self._load_metadata()
    
    def _load_metadata(self):
        """Load checkpoint metadata from disk."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    self.checkpoints = [CheckpointInfo.from_dict(c) for c in data.get('checkpoints', [])]
                    self.best_metric_value = data.get('best_metric_value', self.best_metric_value)
            except Exception as e:
                print(f"Warning: Failed to load checkpoint metadata: {e}")
    
    def _save_metadata(self):
        """Save checkpoint metadata to disk."""
        data = {
            'checkpoints': [c.to_dict() for c in self.checkpoints],
            'best_metric_value': self.best_metric_value,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config)
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_checkpoint_path(self, step: int, is_best: bool = False) -> Path:
        """Generate checkpoint file path."""
        if is_best:
            return self.checkpoint_dir / 'best_model.pt'
        return self.checkpoint_dir / f'checkpoint_step_{step}.pt'
    
    def _is_better_metric(self, value: float) -> bool:
        """Check if the new metric value is better than the current best."""
        if self.config.best_mode == 'min':
            return value < self.best_metric_value
        else:
            return value > self.best_metric_value
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the latest N."""
        if self.config.max_checkpoints is None:
            return
        
        # sort by step number
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x.step, reverse=True)
        
        # keep only non-best checkpoints beyond max_checkpoints
        to_remove = sorted_checkpoints[self.config.max_checkpoints:]
        for ckpt_info in to_remove:
            if not ckpt_info.is_best:  # never remove best checkpoint
                ckpt_path = Path(ckpt_info.path)
                if ckpt_path.exists():
                    ckpt_path.unlink()
                    print(f"Removed old checkpoint: {ckpt_path}")
        
        # update list
        self.checkpoints = [c for c in self.checkpoints if c.step in [ckpt.step for ckpt in sorted_checkpoints[:self.config.max_checkpoints]] or c.is_best]
    
    def save(self, step: int, loss: float, extra_data: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """
        Save a checkpoint.
        
        Args:
            step: Current training step
            loss: Current loss value
            extra_data: Additional data to save (e.g., scheduler state, training config)
        
        Returns:
            Path to saved checkpoint, or None if save failed
        """
        checkpoint_path = self._get_checkpoint_path(step)
        
        # prepare checkpoint data
        checkpoint = {
            'step': step,
            'loss': loss,
            'timestamp': time.time(),
            'model_state_dict': self.model.state_dict(),
        }
        
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if extra_data is not None:
            checkpoint['extra_data'] = extra_data
        
        # save checkpoint
        try:
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path} (step={step}, loss={loss:.4f})")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return None
        
        # track checkpoint info
        ckpt_info = CheckpointInfo(
            step=step,
            loss=loss,
            timestamp=time.time(),
            path=str(checkpoint_path),
            is_best=False
        )
        self.checkpoints.append(ckpt_info)
        
        # save best model if applicable
        if self.config.save_best and self._is_better_metric(loss):
            self.best_metric_value = loss
            best_path = self._get_checkpoint_path(step, is_best=True)
            shutil.copy(checkpoint_path, best_path)
            print(f"Saved best model: {best_path} ({self.config.best_metric}={loss:.4f})")
            
            # update best flag for metadata
            for c in self.checkpoints:
                c.is_best = (c.step == step)
        
        # cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # save metadata
        self._save_metadata()
        
        return checkpoint_path
    
    def load(self, checkpoint_path: Optional[str] = None, load_optimizer: bool = True) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint. If None, loads the latest checkpoint.
            load_optimizer: Whether to load optimizer state
        
        Returns:
            Dictionary containing checkpoint data
        """
        if checkpoint_path is None:
            # find latest checkpoint
            if not self.checkpoints:
                raise ValueError("No checkpoints found")
            latest = max(self.checkpoints, key=lambda x: x.step)
            checkpoint_path = latest.path
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model state from step {checkpoint['step']}")
        
        # load optimizer state
        if load_optimizer and self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state")
        
        return checkpoint
    
    def load_best(self, load_optimizer: bool = True) -> Dict[str, Any]:
        """Load the best checkpoint."""
        best_ckpt = None
        for c in self.checkpoints:
            if c.is_best:
                best_ckpt = c
                break
        
        if best_ckpt is None:
            # fallback to checkpoint with best metric
            if self.config.best_mode == 'min':
                best_ckpt = min(self.checkpoints, key=lambda x: x.loss)
            else:
                best_ckpt = max(self.checkpoints, key=lambda x: x.loss)
        
        return self.load(best_ckpt.path, load_optimizer)
    
    def should_save(self, step: int) -> bool:
        """Check if checkpoint should be saved at this step."""
        if self.config.save_interval is None:
            return False
        return step > 0 and step % self.config.save_interval == 0
    
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """Return list of all checkpoints."""
        return sorted(self.checkpoints, key=lambda x: x.step)
    
    def get_latest_checkpoint(self) -> Optional[CheckpointInfo]:
        """Get the latest checkpoint info."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda x: x.step)
    
    def delete_checkpoint(self, step: int) -> bool:
        """Delete a specific checkpoint."""
        for ckpt_info in self.checkpoints:
            if ckpt_info.step == step and not ckpt_info.is_best:
                ckpt_path = Path(ckpt_info.path)
                if ckpt_path.exists():
                    ckpt_path.unlink()
                self.checkpoints.remove(ckpt_info)
                self._save_metadata()
                print(f"Deleted checkpoint at step {step}")
                return True
        return False
    
    def export_checkpoint(self, step: int, export_path: str, include_optimizer: bool = False):
        """
        Export a checkpoint to a new location with optional filtering.
        
        Args:
            step: Checkpoint step to export
            export_path: Destination path
            include_optimizer: Whether to include optimizer state
        """
        # find checkpoint
        ckpt_info = None
        for c in self.checkpoints:
            if c.step == step:
                ckpt_info = c
                break
        
        if ckpt_info is None:
            raise ValueError(f"Checkpoint at step {step} not found")
        
        # load checkpoint
        checkpoint = torch.load(ckpt_info.path, map_location='cpu')
        
        # filter if needed
        if not include_optimizer and 'optimizer_state_dict' in checkpoint:
            del checkpoint['optimizer_state_dict']
        
        # save to export path
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, export_path)
        print(f"Exported checkpoint to: {export_path}")


def list_checkpoints(checkpoint_dir: str) -> List[Dict[str, Any]]:
    """Utility function to list all checkpoints in a directory."""
    manager = CheckpointManager(
        CN({'checkpoint_dir': checkpoint_dir, 'save_interval': None, 'max_checkpoints': None, 'save_best': True, 'best_metric': 'loss', 'best_mode': 'min'}),
        model=None
    )
    return [c.to_dict() for c in manager.list_checkpoints()]


def cleanup_checkpoints(checkpoint_dir: str, keep: int = 3):
    """Utility function to clean up old checkpoints, keeping only the latest N."""
    config = CN({
        'checkpoint_dir': checkpoint_dir,
        'save_interval': None,
        'max_checkpoints': keep,
        'save_best': True,
        'best_metric': 'loss',
        'best_mode': 'min'
    })
    manager = CheckpointManager(config, model=None)
    manager._cleanup_old_checkpoints()
    manager._save_metadata()
    print(f"Cleaned up checkpoints in {checkpoint_dir}, keeping latest {keep}")
