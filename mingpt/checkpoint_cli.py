#!/usr/bin/env python
"""
Command line tool for managing checkpoints.
Usage:
  python -m mingpt.checkpoint_cli list --checkpoint-dir ./checkpoints/adder
  python -m mingpt.checkpoint_cli export --checkpoint ./checkpoints/adder/best_model.pt --output ./exported_model.pt
  python -m mingpt.checkpoint_cli clean --checkpoint-dir ./checkpoints/adder --keep 3
  python -m mingpt.checkpoint_cli info --checkpoint ./checkpoints/adder/checkpoint_iter_500.pt
"""

import argparse
import os
import json
import torch


def list_checkpoints(args):
    checkpoint_dir = args.checkpoint_dir
    index_path = os.path.join(checkpoint_dir, 'checkpoint_index.json')
    
    if not os.path.exists(index_path):
        print(f"No checkpoint index found in {checkpoint_dir}")
        return
    
    with open(index_path, 'r') as f:
        data = json.load(f)
    
    print(f"\nCheckpoints in {checkpoint_dir}:")
    print("-" * 80)
    
    checkpoints = data.get('checkpoints', [])
    for i, ckpt in enumerate(checkpoints):
        path = ckpt['path']
        iter_num = ckpt['iter_num']
        metrics = ckpt.get('metrics', {})
        timestamp = ckpt.get('timestamp', '')
        size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
        
        metrics_str = ' '.join([f"{k}={v:.4f}" for k, v in metrics.items()])
        print(f"{i+1:2d}. Iter {iter_num:4d} | {size_mb:.1f}MB | {timestamp} | {metrics_str}")
    
    best = data.get('best_checkpoint')
    if best:
        print("-" * 80)
        metrics_str = ' '.join([f"{k}={v:.4f}" for k, v in best.get('metrics', {}).items()])
        print(f"\nBest Checkpoint: Iter {best['iter_num']} | {metrics_str}")
        print(f"  Path: {best['path']}")
    
    print(f"\nTotal: {len(checkpoints)} checkpoints")


def checkpoint_info(args):
    ckpt_path = args.checkpoint
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Size: {size_mb:.2f} MB")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print(f"\nIteration: {ckpt.get('iter_num', 'N/A')}")
    print(f"Timestamp: {ckpt.get('timestamp', 'N/A')}")
    
    metrics = ckpt.get('metrics', {})
    if metrics:
        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    
    config = ckpt.get('config')
    if config:
        print("\nConfig keys:")
        for k in config.keys():
            print(f"  - {k}")
    
    model_state = ckpt.get('model_state_dict', {})
    print(f"\nModel parameters: {len(model_state)} keys")
    
    if ckpt.get('optimizer_state_dict'):
        print("Optimizer state: Included")
    else:
        print("Optimizer state: Not included")


def export_checkpoint(args):
    ckpt_path = args.checkpoint
    output_path = args.output
    only_weights = not args.full
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    from mingpt.checkpoint import CheckpointManager
    
    dummy_config = CheckpointManager.get_default_config()
    dummy_config.checkpoint_dir = os.path.dirname(ckpt_path)
    manager = CheckpointManager(dummy_config)
    
    exported_path = manager.export_model(ckpt_path, output_path, only_weights)
    
    size_mb = os.path.getsize(exported_path) / (1024 * 1024)
    mode = "weights only" if only_weights else "full checkpoint"
    print(f"Exported {mode} to {exported_path} ({size_mb:.2f} MB)")


def clean_checkpoints(args):
    checkpoint_dir = args.checkpoint_dir
    keep = args.keep
    
    from mingpt.checkpoint import CheckpointManager
    
    config = CheckpointManager.get_default_config()
    config.checkpoint_dir = checkpoint_dir
    config.create_run_dir = False
    config.keep_latest_n = keep
    
    manager = CheckpointManager(config)
    removed = manager.clean_old_checkpoints(keep)
    
    print(f"Removed {removed} old checkpoints, kept latest {keep}")


def main():
    parser = argparse.ArgumentParser(description='Checkpoint Management CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    list_parser = subparsers.add_parser('list', help='List all checkpoints')
    list_parser.add_argument('--checkpoint-dir', required=True, help='Checkpoint directory')

    info_parser = subparsers.add_parser('info', help='Show checkpoint info')
    info_parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')

    export_parser = subparsers.add_parser('export', help='Export model from checkpoint')
    export_parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    export_parser.add_argument('--output', required=True, help='Output path')
    export_parser.add_argument('--full', action='store_true', help='Export full checkpoint (not just weights)')

    clean_parser = subparsers.add_parser('clean', help='Clean old checkpoints')
    clean_parser.add_argument('--checkpoint-dir', required=True, help='Checkpoint directory')
    clean_parser.add_argument('--keep', type=int, default=5, help='Number of latest checkpoints to keep')

    args = parser.parse_args()

    if args.command == 'list':
        list_checkpoints(args)
    elif args.command == 'info':
        checkpoint_info(args)
    elif args.command == 'export':
        export_checkpoint(args)
    elif args.command == 'clean':
        clean_checkpoints(args)


if __name__ == '__main__':
    main()
