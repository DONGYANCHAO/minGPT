#!/usr/bin/env python
"""
Command line tool for model evaluation.
Usage:
  python -m mingpt.evaluator_cli evaluate --model ./checkpoints/adder/best_model.pt --dataset adder
  python -m mingpt.evaluator_cli compare --models "model1=./ckpt1.pt,model2=./ckpt2.pt" --dataset adder
"""

import argparse
import os
import sys
import json

import torch

from mingpt.model import GPT
from mingpt.evaluator import ModelEvaluator


def get_dataset(dataset_name, ndigit=2):
    if dataset_name == 'adder':
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../projects/adder'))
        from adder import AdditionDataset
        from mingpt.utils import CfgNode as CN
        data_config = CN()
        data_config.ndigit = ndigit
        return AdditionDataset(data_config, split='test')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def run_evaluation(args):
    print(f"\nEvaluating model: {args.model}")
    print(f"Dataset: {args.dataset}")

    evaluator_config = ModelEvaluator.get_default_config()
    evaluator_config.batch_size = args.batch_size
    evaluator = ModelEvaluator(evaluator_config)

    model_config = GPT.get_default_config()
    model_config.model_type = args.model_type
    model_config.vocab_size = 10
    model_config.block_size = 6

    model = GPT(model_config)

    checkpoint = torch.load(args.model, map_location=evaluator.device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from iteration {checkpoint.get('iter_num', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")

    dataset = get_dataset(args.dataset)
    results = evaluator.full_evaluation(
        model, 
        dataset,
        task_type='language_modeling',
        num_gen_samples=args.num_samples
    )

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Perplexity: {results.get('perplexity', 'N/A'):.2f}")
    print(f"Token Accuracy: {results.get('accuracy', {}).get('overall', 0):.4f}")
    
    if 'bleu' in results:
        print(f"BLEU: {results['bleu'].get('bleu', 0):.4f}")
    if 'rouge' in results:
        print(f"ROUGE-1: {results['rouge'].get('rouge1', 0):.4f}")
        print(f"ROUGE-2: {results['rouge'].get('rouge2', 0):.4f}")
    print("="*60)

    if args.output:
        evaluator.save_report(results, args.output)


def run_comparison(args):
    print(f"\nComparing models on dataset: {args.dataset}")

    model_dict = {}
    for item in args.models.split(','):
        name, path = item.split('=')
        model_dict[name.strip()] = path.strip()
        print(f"  {name.strip()}: {path.strip()}")

    evaluator_config = ModelEvaluator.get_default_config()
    evaluator_config.batch_size = args.batch_size
    evaluator = ModelEvaluator(evaluator_config)

    model_config = GPT.get_default_config()
    model_config.model_type = args.model_type
    model_config.vocab_size = 10
    model_config.block_size = 6

    dataset = get_dataset(args.dataset)
    comparison = evaluator.compare_models(model_dict, GPT, model_config, dataset)

    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    metrics = ['perplexity', 'accuracy', 'bleu', 'rouge1', 'rouge2']
    header = f"{'Model':<20}" + "".join([f"{m:<15}" for m in metrics])
    print(header)
    print("-" * 80)

    for model_name, results in comparison.items():
        row = f"{model_name:<20}"
        row += f"{results.get('perplexity', 0):<15.2f}"
        row += f"{results.get('accuracy', {}).get('overall', 0):<15.4f}"
        row += f"{results.get('bleu', {}).get('bleu', 0):<15.4f}"
        row += f"{results.get('rouge', {}).get('rouge1', 0):<15.4f}"
        row += f"{results.get('rouge', {}).get('rouge2', 0):<15.4f}"
        print(row)

    print("="*80)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison report saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description='Model Evaluation CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a single model')
    eval_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--dataset', default='adder', help='Dataset to use')
    eval_parser.add_argument('--model-type', default='gpt-nano', help='GPT model type')
    eval_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    eval_parser.add_argument('--num-samples', type=int, default=100, help='Number of generation samples')
    eval_parser.add_argument('--output', help='Output report path')

    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--models', required=True, help='Comma-separated name=path pairs')
    compare_parser.add_argument('--dataset', default='adder', help='Dataset to use')
    compare_parser.add_argument('--model-type', default='gpt-nano', help='GPT model type')
    compare_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    compare_parser.add_argument('--output', help='Output report path')

    args = parser.parse_args()

    if args.command == 'evaluate':
        run_evaluation(args)
    elif args.command == 'compare':
        run_comparison(args)


if __name__ == '__main__':
    main()
