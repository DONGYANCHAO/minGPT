"""
Command-line tool for evaluating minGPT models.

Usage:
    python tools/eval_cli.py --model-path model.pt --dataset input.txt --task language_modeling
    python tools/eval_cli.py --model-path model.pt --dataset data.json --task classification --metrics accuracy f1
    python tools/eval_cli.py --compare model1.pt model2.pt --dataset input.txt --output report.md
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from mingpt.model import GPT
from mingpt.evaluator import GPTEvaluator, compare_models, EvaluationReport
from mingpt.utils import CfgNode as CN, set_seed


class TextDataset:
    """Simple text dataset for evaluation."""
    
    def __init__(self, data, block_size=128):
        self.data = data
        self.block_size = block_size
        self.chars = sorted(list(set(data)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


def load_model(model_path, device='cpu'):
    """Load a model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to get config from checkpoint
    if 'extra_data' in checkpoint and 'config' in checkpoint['extra_data']:
        config_dict = checkpoint['extra_data']['config']
        # Handle nested config structure
        if 'model' in config_dict:
            model_config_dict = config_dict['model']
        else:
            model_config_dict = config_dict
    else:
        # Use default config
        model_config_dict = {}
    
    # Create config
    config = GPT.get_default_config()
    for key, value in model_config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Ensure required fields
    if config.vocab_size is None:
        config.vocab_size = 256  # default
    if config.block_size is None:
        config.block_size = 128  # default
    
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def load_dataset(dataset_path, block_size=128):
    """Load dataset from file."""
    path = Path(dataset_path)
    
    if path.suffix == '.json':
        with open(path, 'r') as f:
            data = json.load(f)
        # Handle different JSON formats
        if isinstance(data, list):
            texts = [item['text'] if isinstance(item, dict) else str(item) for item in data]
            text = '\n'.join(texts)
        elif isinstance(data, dict):
            text = data.get('text', str(data))
        else:
            text = str(data)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    return TextDataset(text, block_size)


def cmd_eval(args):
    """Evaluate a single model."""
    print(f"Evaluating model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Metrics: {', '.join(args.metrics)}")
    print()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    model, model_config = load_model(args.model_path, device)
    print(f"Model loaded successfully")
    
    # Load dataset
    dataset = load_dataset(args.dataset, block_size=model_config.block_size)
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Setup evaluator
    eval_config = GPTEvaluator.get_default_config()
    eval_config.metrics = args.metrics
    eval_config.device = device
    eval_config.batch_size = args.batch_size
    
    evaluator = GPTEvaluator(eval_config, model)
    
    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluator.evaluate(dataset, args.task)
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for name, result in results.items():
        print(f"{name:20s}: {result.value:.4f}")
        if result.details:
            for key, value in result.details.items():
                if isinstance(value, float):
                    print(f"  {key:18s}: {value:.4f}")
                else:
                    print(f"  {key:18s}: {value}")
    print("=" * 50)
    
    # Save report if requested
    if args.output:
        report = EvaluationReport({args.model_path: results}, [args.model_path])
        report.save(args.output, format='both')
        print(f"\nReport saved to: {args.output}")


def cmd_compare(args):
    """Compare multiple models."""
    print(f"Comparing {len(args.model_paths)} models")
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print()
    
    # Set seed
    set_seed(42)
    
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset (use first model's config)
    first_model, first_config = load_model(args.model_paths[0], device)
    dataset = load_dataset(args.dataset, block_size=first_config.block_size)
    print(f"Dataset loaded: {len(dataset)} samples\n")
    
    # Load all models
    models = {}
    for model_path in args.model_paths:
        name = Path(model_path).stem
        print(f"Loading {name}...")
        model, _ = load_model(model_path, device)
        models[name] = model
    
    # Setup evaluator config
    eval_config = GPTEvaluator.get_default_config()
    eval_config.metrics = args.metrics
    eval_config.device = device
    eval_config.batch_size = args.batch_size
    
    # Compare models
    print("\nRunning evaluation...")
    report = compare_models(models, dataset, eval_config, args.task, args.output)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    # Get all unique metrics
    all_metrics = set()
    for model_results in report.results.values():
        all_metrics.update(model_results.keys())
    
    # Print header
    header = f"{'Model':<20}"
    for metric in sorted(all_metrics):
        header += f"{metric:>12}"
    print(header)
    print("-" * 70)
    
    # Print results for each model
    for model_name in report.model_names:
        line = f"{model_name:<20}"
        for metric in sorted(all_metrics):
            if metric in report.results[model_name]:
                value = report.results[model_name][metric].value
                line += f"{value:>12.4f}"
            else:
                line += f"{'N/A':>12}"
        print(line)
    
    print("=" * 70)
    
    if args.output:
        print(f"\nReport saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description='minGPT Model Evaluation CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single model on language modeling
  python tools/eval_cli.py eval --model-path model.pt --dataset input.txt
  
  # Evaluate with specific metrics
  python tools/eval_cli.py eval --model-path model.pt --dataset input.txt \\
      --metrics perplexity accuracy
  
  # Compare multiple models
  python tools/eval_cli.py compare model1.pt model2.pt --dataset input.txt --output report.md
        """
    )
    
    parser.add_argument('--cpu', action='store_true', help='Use CPU even if CUDA is available')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate a single model')
    eval_parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--dataset', type=str, required=True, help='Path to dataset file')
    eval_parser.add_argument('--task', type=str, default='language_modeling',
                            choices=['language_modeling', 'classification', 'generation'],
                            help='Evaluation task type')
    eval_parser.add_argument('--metrics', type=str, nargs='+', default=['perplexity', 'accuracy'],
                            help='Metrics to compute')
    eval_parser.add_argument('--output', type=str, help='Output path for report')
    eval_parser.set_defaults(func=cmd_eval)
    
    # compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('model_paths', type=str, nargs='+', help='Paths to model checkpoints')
    compare_parser.add_argument('--dataset', type=str, required=True, help='Path to dataset file')
    compare_parser.add_argument('--task', type=str, default='language_modeling',
                               choices=['language_modeling', 'classification', 'generation'],
                               help='Evaluation task type')
    compare_parser.add_argument('--metrics', type=str, nargs='+', default=['perplexity', 'accuracy'],
                               help='Metrics to compute')
    compare_parser.add_argument('--output', type=str, help='Output path for comparison report')
    compare_parser.set_defaults(func=cmd_compare)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()