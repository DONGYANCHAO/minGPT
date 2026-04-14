"""
Command-line tool for managing minGPT checkpoints.

Usage:
    python tools/checkpoint_cli.py list --checkpoint-dir ./checkpoints
    python tools/checkpoint_cli.py restore --checkpoint-dir ./checkpoints --step 1000
    python tools/checkpoint_cli.py export --checkpoint-dir ./checkpoints --step 1000 --output model.pt
    python tools/checkpoint_cli.py cleanup --checkpoint-dir ./checkpoints --keep 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mingpt.checkpoint import CheckpointManager, list_checkpoints, cleanup_checkpoints
from mingpt.utils import CfgNode as CN


def cmd_list(args):
    """List all checkpoints."""
    checkpoints = list_checkpoints(args.checkpoint_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return
    
    print(f"\nCheckpoints in {args.checkpoint_dir}:")
    print("-" * 80)
    print(f"{'Step':<10} {'Loss':<12} {'Best':<6} {'Timestamp':<20} {'Path'}")
    print("-" * 80)
    
    for ckpt in sorted(checkpoints, key=lambda x: x['step']):
        best_marker = "*" if ckpt.get('is_best') else ""
        import datetime
        ts = datetime.datetime.fromtimestamp(ckpt['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{ckpt['step']:<10} {ckpt['loss']:<12.4f} {best_marker:<6} {ts:<20} {ckpt['path']}")
    
    print("-" * 80)
    print(f"Total: {len(checkpoints)} checkpoints")
    print()


def cmd_restore(args):
    """Restore/resume from a checkpoint."""
    print(f"Restoring checkpoint from step {args.step}...")
    
    config = CN({
        'checkpoint_dir': args.checkpoint_dir,
        'save_interval': None,
        'max_checkpoints': None,
        'save_best': True,
        'best_metric': 'loss',
        'best_mode': 'min'
    })
    
    # Note: This requires a model to be loaded
    # For now, just print instructions
    print("\nTo resume training from this checkpoint, use the following code:")
    print(f"""
    from mingpt.checkpoint import CheckpointManager
    from mingpt.model import GPT
    
    # Load your model
    model = GPT(config)
    
    # Create checkpoint manager
    ckpt_manager = CheckpointManager(config, model, optimizer)
    
    # Load checkpoint
    checkpoint = ckpt_manager.load('{args.checkpoint_dir}/checkpoint_step_{args.step}.pt')
    
    # Resume training
    trainer.iter_num = checkpoint['step']
    """)


def cmd_export(args):
    """Export a checkpoint."""
    print(f"Exporting checkpoint at step {args.step} to {args.output}...")
    
    config = CN({
        'checkpoint_dir': args.checkpoint_dir,
        'save_interval': None,
        'max_checkpoints': None,
        'save_best': True,
        'best_metric': 'loss',
        'best_mode': 'min'
    })
    
    # Create a dummy checkpoint manager (no model needed for export)
    manager = CheckpointManager(config, model=None)
    
    try:
        manager.export_checkpoint(args.step, args.output, include_optimizer=args.include_optimizer)
        print(f"Successfully exported to {args.output}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def cmd_cleanup(args):
    """Clean up old checkpoints."""
    print(f"Cleaning up checkpoints in {args.checkpoint_dir}, keeping {args.keep} most recent...")
    
    cleanup_checkpoints(args.checkpoint_dir, keep=args.keep)
    print("Cleanup complete!")


def main():
    parser = argparse.ArgumentParser(
        description='minGPT Checkpoint Manager CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all checkpoints
  python tools/checkpoint_cli.py list --checkpoint-dir ./checkpoints
  
  # Export checkpoint at step 1000 (without optimizer state)
  python tools/checkpoint_cli.py export --step 1000 --output model.pt
  
  # Clean up old checkpoints, keeping only the 3 most recent
  python tools/checkpoint_cli.py cleanup --keep 3
        """
    )
    
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory containing checkpoints (default: ./checkpoints)')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # list command
    list_parser = subparsers.add_parser('list', help='List all checkpoints')
    list_parser.set_defaults(func=cmd_list)
    
    # restore command
    restore_parser = subparsers.add_parser('restore', help='Show restore instructions')
    restore_parser.add_argument('--step', type=int, required=True, help='Step to restore')
    restore_parser.set_defaults(func=cmd_restore)
    
    # export command
    export_parser = subparsers.add_parser('export', help='Export a checkpoint')
    export_parser.add_argument('--step', type=int, required=True, help='Step to export')
    export_parser.add_argument('--output', type=str, required=True, help='Output file path')
    export_parser.add_argument('--include-optimizer', action='store_true',
                               help='Include optimizer state in export')
    export_parser.set_defaults(func=cmd_export)
    
    # cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old checkpoints')
    cleanup_parser.add_argument('--keep', type=int, default=3, help='Number of checkpoints to keep')
    cleanup_parser.set_defaults(func=cmd_cleanup)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
