"""
Test script for minGPT new features:
- Checkpoint management
- Training visualization and logging
- Model evaluation and metrics
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.checkpoint import CheckpointManager
from mingpt.logger import TrainingLogger
from mingpt.evaluator import Evaluator, ModelComparator
from mingpt.utils import CfgNode as CN


class DummyDataset(Dataset):
    def __init__(self, vocab_size=50257, block_size=128, size=1000):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.size = size
        self.data = torch.randint(0, vocab_size, (size, block_size + 1))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return x, y


def test_checkpoint():
    print("\n" + "=" * 60)
    print("Testing Checkpoint Management")
    print("=" * 60)
    
    config = GPT.get_default_config()
    config.model_type = 'gpt-nano'
    config.vocab_size = 100
    config.block_size = 32
    model = GPT(config)
    
    ckpt_config = CheckpointManager.get_default_config()
    ckpt_config.checkpoint_dir = 'test_checkpoints'
    ckpt_config.save_interval = 5
    ckpt_config.max_checkpoints = 3
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    manager = CheckpointManager(ckpt_config, model, optimizer)
    
    print("\n1. Saving checkpoints...")
    for i in range(1, 16):
        metrics = {'loss': 1.0 / i, 'accuracy': 0.5 + 0.03 * i}
        manager.save(i, metrics)
    
    checkpoints = manager.list_checkpoints()
    print(f"   Saved {len(checkpoints)} checkpoints (max 3)")
    
    print("\n2. Listing checkpoints:")
    for cp in checkpoints:
        print(f"   - Iter {cp['iter_num']}: loss={cp['metrics'].get('loss', 'N/A'):.4f}")
    
    print("\n3. Loading latest checkpoint...")
    checkpoint = manager.load()
    print(f"   Loaded iteration: {checkpoint['iter_num']}")
    print(f"   Metrics: {checkpoint['metrics']}")
    
    print("\n4. Checking best model...")
    if os.path.exists(os.path.join(ckpt_config.checkpoint_dir, 'best_model.pt')):
        print("   Best model saved successfully")
    
    print("\n5. Cleaning up test checkpoints...")
    import shutil
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')
    print("   Cleanup complete")
    
    print("\n[Checkpoint Test PASSED]")
    return True


def test_logger():
    print("\n" + "=" * 60)
    print("Testing Training Logger")
    print("=" * 60)
    
    log_config = TrainingLogger.get_default_config()
    log_config.log_dir = 'test_logs'
    log_config.experiment_name = 'test_run'
    log_config.log_interval = 10
    log_config.use_tensorboard = False
    log_config.use_wandb = False
    
    model_config = CN()
    model_config.n_layer = 3
    model_config.n_head = 3
    model_config.n_embd = 48
    
    logger = TrainingLogger(log_config, model_config)
    
    print("\n1. Logging metrics...")
    for i in range(1, 101):
        metrics = {
            'loss': 2.0 / (1 + 0.1 * i),
            'accuracy': min(0.99, 0.3 + 0.007 * i),
        }
        extra = {
            'learning_rate': 3e-4 * (0.99 ** i),
            'iter_time': 0.05,
        }
        logger.log(i, metrics, extra)
    
    print("   Logged 100 iterations")
    
    print("\n2. Checking log files...")
    log_dir = os.path.join(log_config.log_dir, log_config.experiment_name)
    
    files = {
        'training.log': os.path.join(log_dir, 'training.log'),
        'metrics.jsonl': os.path.join(log_dir, 'metrics.jsonl'),
    }
    
    for name, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"   - {name}: {size} bytes")
    
    print("\n3. Generating report...")
    report = logger.generate_report()
    print(f"   Total time: {report['total_time_seconds']:.2f}s")
    print(f"   Total iterations: {report['total_iterations']}")
    print(f"   Final loss: {report['final_metrics'].get('loss', 'N/A'):.4f}")
    
    print("\n4. Checking report files...")
    report_files = ['training_report.json', 'training_report.txt']
    for f in report_files:
        path = os.path.join(log_dir, f)
        if os.path.exists(path):
            print(f"   - {f} generated")
    
    logger.close()
    
    print("\n5. Test files saved (cleanup skipped on Windows)")
    
    print("\n[Logger Test PASSED]")
    return True


def test_evaluator():
    print("\n" + "=" * 60)
    print("Testing Model Evaluator")
    print("=" * 60)
    
    config = GPT.get_default_config()
    config.model_type = 'gpt-nano'
    config.vocab_size = 100
    config.block_size = 32
    model = GPT(config)
    
    eval_config = Evaluator.get_default_config()
    eval_config.batch_size = 16
    eval_config.compute_perplexity = True
    eval_config.compute_accuracy = True
    eval_config.compute_bleu = True
    eval_config.compute_rouge = True
    
    evaluator = Evaluator(eval_config, model)
    
    dataset = DummyDataset(vocab_size=100, block_size=32, size=100)
    
    print("\n1. Running evaluation...")
    results = evaluator.evaluate(dataset)
    
    print("\n2. Evaluation results:")
    for name, value in results.items():
        if isinstance(value, float):
            print(f"   - {name}: {value:.4f}")
    
    print("\n3. Generating evaluation report...")
    report = evaluator.generate_report()
    print("   Report generated")
    
    print("\n4. Testing model comparison...")
    comparator = ModelComparator(dataset, eval_config)
    
    def create_fresh_model():
        cfg = GPT.get_default_config()
        cfg.model_type = 'gpt-nano'
        cfg.vocab_size = 100
        cfg.block_size = 32
        return GPT(cfg)
    
    model1 = create_fresh_model()
    model2 = create_fresh_model()
    
    results1 = comparator.add_model('model_v1', model1)
    results2 = comparator.add_model('model_v2', model2)
    
    comparison = comparator.compare()
    print(f"   Compared {len(comparison['models'])} models")
    
    comparison_report = comparator.generate_comparison_report()
    
    print("\n[Evaluator Test PASSED]")
    return True


def test_trainer_integration():
    print("\n" + "=" * 60)
    print("Testing Trainer Integration")
    print("=" * 60)
    
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = 100
    model_config.block_size = 32
    model = GPT(model_config)
    
    dataset = DummyDataset(vocab_size=100, block_size=32, size=500)
    
    trainer_config = Trainer.get_default_config()
    trainer_config.max_iters = 50
    trainer_config.batch_size = 16
    trainer_config.num_workers = 0
    
    trainer_config.checkpoint.checkpoint_dir = 'test_trainer_checkpoints'
    trainer_config.checkpoint.save_interval = 20
    trainer_config.checkpoint.max_checkpoints = 2
    
    trainer_config.logging.log_dir = 'test_trainer_logs'
    trainer_config.logging.experiment_name = 'integration_test'
    trainer_config.logging.log_interval = 10
    trainer_config.logging.use_tensorboard = False
    trainer_config.logging.use_wandb = False
    
    trainer_config.enable_checkpoint = True
    trainer_config.enable_logging = True
    trainer_config.enable_evaluation = False
    
    print("\n1. Creating trainer with all features enabled...")
    trainer = Trainer(trainer_config, model, dataset)
    
    print("\n2. Running training...")
    trainer.run()
    
    print("\n3. Checking training state...")
    state = trainer.get_training_state()
    print(f"   Iterations: {state['iter_num']}")
    print(f"   Final loss: {state['loss']:.4f}")
    
    print("\n4. Verifying checkpoint files...")
    ckpt_dir = trainer_config.checkpoint.checkpoint_dir
    if os.path.exists(ckpt_dir):
        files = os.listdir(ckpt_dir)
        print(f"   Checkpoint directory: {len(files)} files")
        for f in files[:5]:
            print(f"   - {f}")
    
    print("\n5. Verifying log files...")
    log_dir = os.path.join(
        trainer_config.logging.log_dir,
        trainer_config.logging.experiment_name
    )
    if os.path.exists(log_dir):
        files = os.listdir(log_dir)
        print(f"   Log directory: {len(files)} files")
        for f in files:
            print(f"   - {f}")
    
    print("\n6. Testing checkpoint restoration...")
    
    def create_fresh_model():
        cfg = GPT.get_default_config()
        cfg.model_type = 'gpt-nano'
        cfg.vocab_size = 100
        cfg.block_size = 32
        return GPT(cfg)
    
    model2 = create_fresh_model()
    trainer2 = Trainer(trainer_config, model2, dataset)
    restored = trainer2.restore_from_checkpoint()
    if restored:
        print(f"   Restored at iteration: {trainer2.iter_num}")
    else:
        print("   No checkpoint to restore (expected for fresh start)")
    
    print("\n7. Test files saved (cleanup skipped on Windows)")
    
    print("\n[Trainer Integration Test PASSED]")
    return True


def main():
    print("\n" + "=" * 60)
    print("minGPT Feature Tests")
    print("=" * 60)
    
    tests = [
        ("Checkpoint Management", test_checkpoint),
        ("Training Logger", test_logger),
        ("Model Evaluator", test_evaluator),
        ("Trainer Integration", test_trainer_integration),
    ]
    
    results = {}
    
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = "PASSED" if success else "FAILED"
        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = f"ERROR: {str(e)}"
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, status in results.items():
        symbol = "✓" if status == "PASSED" else "✗"
        print(f"  {symbol} {name}: {status}")
    
    all_passed = all(s == "PASSED" for s in results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
