"""
Model evaluation and metrics for minGPT.
Supports perplexity, accuracy, BLEU, ROUGE metrics,
text classification, and text generation evaluation.
"""

import os
import json
import math
import argparse
from typing import Optional, Dict, Any, List, Tuple, Callable
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

from mingpt.utils import CfgNode as CN


class Evaluator:
    """
    Comprehensive model evaluator supporting multiple metrics:
    - Perplexity
    - Token/Sequence accuracy
    - BLEU score
    - ROUGE score
    - Custom metrics
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.batch_size = 64
        C.num_workers = 4
        C.device = 'auto'
        C.compute_perplexity = True
        C.compute_accuracy = True
        C.compute_bleu = False
        C.compute_rouge = False
        C.max_samples = None
        return C

    def __init__(self, config, model, device: Optional[str] = None):
        self.config = config
        self.model = model
        
        if device is None:
            if config.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = config.device
        else:
            self.device = device
        
        self.model = self.model.to(self.device)
        self.metrics: Dict[str, float] = {}
        self.detailed_results: Dict[str, Any] = {}

    def evaluate(self, dataset: Dataset, 
                 metrics: Optional[List[str]] = None) -> Dict[str, float]:
        self.model.eval()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        
        total_loss = 0.0
        total_tokens = 0
        correct_tokens = 0
        correct_sequences = 0
        total_sequences = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if self.config.max_samples and batch_idx * self.config.batch_size >= self.config.max_samples:
                    break
                
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    x = batch.to(self.device)
                    y = x[:, 1:].contiguous()
                    x = x[:, :-1].contiguous()
                
                logits, loss = self.model(x, y)
                
                if loss is not None:
                    total_loss += loss.item() * y.numel()
                    total_tokens += y.numel()
                
                predictions = logits.argmax(dim=-1)
                correct_tokens += (predictions == y).sum().item()
                
                sequence_correct = (predictions == y).all(dim=1).sum().item()
                correct_sequences += sequence_correct
                total_sequences += y.size(0)
                
                all_predictions.extend(predictions.cpu().tolist())
                all_targets.extend(y.cpu().tolist())
        
        results = {}
        
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            results['perplexity'] = math.exp(avg_loss)
            results['loss'] = avg_loss
        
        if total_tokens > 0:
            results['token_accuracy'] = correct_tokens / total_tokens
        
        if total_sequences > 0:
            results['sequence_accuracy'] = correct_sequences / total_sequences
        
        if self.config.compute_bleu or (metrics and 'bleu' in metrics):
            bleu_score = self._compute_bleu(all_predictions, all_targets)
            results['bleu'] = bleu_score
        
        if self.config.compute_rouge or (metrics and 'rouge' in metrics):
            rouge_scores = self._compute_rouge(all_predictions, all_targets)
            results.update(rouge_scores)
        
        self.metrics = results
        self.detailed_results = {
            'predictions': all_predictions[:100],
            'targets': all_targets[:100],
        }
        
        return results

    def _compute_bleu(self, predictions: List[List[int]], 
                      targets: List[List[int]]) -> float:
        def get_ngrams(tokens: List[int], n: int) -> Counter:
            ngrams = Counter()
            for i in range(len(tokens) - n + 1):
                ngrams[tuple(tokens[i:i+n])] += 1
            return ngrams
        
        def modified_precision(pred: List[int], ref: List[int], n: int) -> float:
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            overlap = 0
            for ngram, count in pred_ngrams.items():
                overlap += min(count, ref_ngrams.get(ngram, 0))
            
            total = sum(pred_ngrams.values())
            if total == 0:
                return 0.0
            return overlap / total
        
        scores = []
        for pred, target in zip(predictions, targets):
            if len(pred) == 0 or len(target) == 0:
                continue
            
            precisions = []
            for n in range(1, min(5, len(pred) + 1, len(target) + 1)):
                precisions.append(modified_precision(pred, target, n))
            
            if not precisions:
                continue
            
            bp = 1.0
            if len(pred) < len(target):
                bp = math.exp(1 - len(target) / len(pred))
            
            if all(p > 0 for p in precisions):
                geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
                scores.append(bp * geo_mean)
        
        return sum(scores) / len(scores) if scores else 0.0

    def _compute_rouge(self, predictions: List[List[int]], 
                       targets: List[List[int]]) -> Dict[str, float]:
        def lcs_length(seq1: List[int], seq2: List[int]) -> int:
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        rouge_1_scores = []
        rouge_2_scores = []
        rouge_l_scores = []
        
        for pred, target in zip(predictions, targets):
            if len(pred) == 0 or len(target) == 0:
                continue
            
            pred_set = set(pred)
            target_set = set(target)
            overlap_1 = len(pred_set & target_set)
            rouge_1_scores.append(overlap_1 / len(target_set) if target_set else 0)
            
            pred_bigrams = set(tuple(pred[i:i+2]) for i in range(len(pred) - 1))
            target_bigrams = set(tuple(target[i:i+2]) for i in range(len(target) - 1))
            overlap_2 = len(pred_bigrams & target_bigrams)
            rouge_2_scores.append(overlap_2 / len(target_bigrams) if target_bigrams else 0)
            
            lcs_len = lcs_length(pred, target)
            rouge_l_scores.append(lcs_len / len(target) if target else 0)
        
        return {
            'rouge_1': sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0.0,
            'rouge_2': sum(rouge_2_scores) / len(rouge_2_scores) if rouge_2_scores else 0.0,
            'rouge_l': sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0,
        }

    def evaluate_generation(self, prompts: List[torch.Tensor], 
                           max_new_tokens: int = 50,
                           temperature: float = 1.0,
                           do_sample: bool = True,
                           top_k: Optional[int] = None) -> List[torch.Tensor]:
        self.model.eval()
        generated = []
        
        with torch.no_grad():
            for prompt in prompts:
                prompt = prompt.to(self.device).unsqueeze(0)
                output = self.model.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_k=top_k,
                )
                generated.append(output.squeeze(0))
        
        return generated

    def evaluate_classification(self, dataset: Dataset,
                               num_classes: int) -> Dict[str, float]:
        self.model.eval()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue
                
                logits, _ = self.model(x)
                
                if logits.dim() > 2:
                    logits = logits.mean(dim=1)
                
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        for cls in range(num_classes):
            tp = ((all_preds == cls) & (all_labels == cls)).sum()
            fp = ((all_preds == cls) & (all_labels != cls)).sum()
            fn = ((all_preds != cls) & (all_labels == cls)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)
        
        return {
            'accuracy': accuracy,
            'precision_macro': np.mean(precision_per_class),
            'recall_macro': np.mean(recall_per_class),
            'f1_macro': np.mean(f1_per_class),
            'per_class_precision': precision_per_class,
            'per_class_recall': recall_per_class,
            'per_class_f1': f1_per_class,
        }

    def generate_report(self, output_path: Optional[str] = None) -> str:
        report_lines = [
            "=" * 60,
            "Model Evaluation Report",
            "=" * 60,
            "",
            "Metrics:",
        ]
        
        for name, value in self.metrics.items():
            if isinstance(value, float):
                report_lines.append(f"  {name}: {value:.6f}")
            else:
                report_lines.append(f"  {name}: {value}")
        
        report_lines.extend([
            "",
            "Configuration:",
            f"  Device: {self.device}",
            f"  Batch size: {self.config.batch_size}",
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report


class ModelComparator:
    """
    Compare multiple models on the same evaluation metrics.
    """
    
    def __init__(self, dataset: Dataset, config: Optional[CN] = None):
        self.dataset = dataset
        self.config = config or Evaluator.get_default_config()
        self.results: Dict[str, Dict[str, float]] = {}

    def add_model(self, name: str, model: nn.Module) -> Dict[str, float]:
        evaluator = Evaluator(self.config, model)
        results = evaluator.evaluate(self.dataset)
        self.results[name] = results
        return results

    def compare(self) -> Dict[str, Any]:
        if not self.results:
            return {}
        
        comparison = {
            'models': list(self.results.keys()),
            'metrics': {},
        }
        
        all_metrics = set()
        for results in self.results.values():
            all_metrics.update(results.keys())
        
        for metric in all_metrics:
            values = {}
            for name, results in self.results.items():
                if metric in results:
                    values[name] = results[metric]
            
            if values:
                comparison['metrics'][metric] = values
                best_model = max(values.items(), key=lambda x: x[1])
                comparison['metrics'][metric]['_best'] = best_model[0]
        
        return comparison

    def generate_comparison_report(self, output_path: Optional[str] = None) -> str:
        comparison = self.compare()
        
        if not comparison:
            return "No models to compare."
        
        lines = [
            "=" * 70,
            "Model Comparison Report",
            "=" * 70,
            "",
            f"Models compared: {', '.join(comparison['models'])}",
            "",
        ]
        
        for metric, values in comparison['metrics'].items():
            lines.append(f"{metric}:")
            best = values.pop('_best', None)
            for name, value in sorted(values.items(), key=lambda x: x[1], reverse=True):
                marker = " *" if name == best else ""
                lines.append(f"  {name}: {value:.6f}{marker}")
            lines.append("")
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report


class HuggingFaceBenchmark:
    """
    Integration with HuggingFace Evaluate for standardized benchmarks.
    """
    
    def __init__(self):
        self.available_metrics = self._check_available_metrics()

    def _check_available_metrics(self) -> List[str]:
        try:
            import evaluate
            return ['perplexity', 'bleu', 'rouge', 'accuracy', 'f1']
        except ImportError:
            return []

    def is_available(self) -> bool:
        return len(self.available_metrics) > 0

    def compute_metric(self, metric_name: str, 
                       predictions: List[str], 
                       references: List[str]) -> Dict[str, float]:
        if not self.is_available():
            raise ImportError("HuggingFace evaluate not installed. Run: pip install evaluate")
        
        import evaluate
        
        if metric_name not in self.available_metrics:
            raise ValueError(f"Metric {metric_name} not available. Available: {self.available_metrics}")
        
        metric = evaluate.load(metric_name)
        
        if metric_name == 'bleu':
            results = metric.compute(predictions=predictions, references=[[r] for r in references])
        elif metric_name == 'rouge':
            results = metric.compute(predictions=predictions, references=references)
        else:
            results = metric.compute(predictions=predictions, references=references)
        
        return results


def evaluate_model(model_path: str, dataset_path: str, 
                   output_dir: str, config: Optional[CN] = None):
    from mingpt.model import GPT
    from mingpt.utils import CfgNode
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'config' in checkpoint:
        model_config = CfgNode(**checkpoint['config'])
    else:
        raise ValueError("Model config not found in checkpoint")
    
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    eval_config = config or Evaluator.get_default_config()
    evaluator = Evaluator(eval_config, model)
    
    print(f"Evaluating model from {model_path}")
    results = evaluator.evaluate(dataset_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_file = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    report_file = os.path.join(output_dir, 'evaluation_report.txt')
    evaluator.generate_report(report_file)
    
    print(f"Results saved to {output_dir}")
    return results


def compare_models(model_paths: List[str], model_names: List[str],
                   dataset, output_dir: str, config: Optional[CN] = None):
    from mingpt.model import GPT
    from mingpt.utils import CfgNode
    
    comparator = ModelComparator(dataset, config)
    
    for path, name in zip(model_paths, model_names):
        checkpoint = torch.load(path, map_location='cpu')
        
        if 'config' in checkpoint:
            model_config = CfgNode(**checkpoint['config'])
        else:
            raise ValueError(f"Model config not found in {path}")
        
        model = GPT(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        comparator.add_model(name, model)
        print(f"Evaluated model: {name}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    comparison = comparator.compare()
    comparison_file = os.path.join(output_dir, 'comparison_results.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    report_file = os.path.join(output_dir, 'comparison_report.txt')
    comparator.generate_comparison_report(report_file)
    
    print(f"Comparison results saved to {output_dir}")
    return comparison


def main():
    parser = argparse.ArgumentParser(description='Model evaluation tools for minGPT')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--output', type=str, default='eval_results',
                            help='Output directory for results')
    eval_parser.add_argument('--batch-size', type=int, default=64,
                            help='Evaluation batch size')
    
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--models', type=str, nargs='+', required=True,
                               help='Paths to model checkpoints')
    compare_parser.add_argument('--names', type=str, nargs='+', required=True,
                               help='Names for each model')
    compare_parser.add_argument('--output', type=str, default='comparison_results',
                               help='Output directory for results')
    
    args = parser.parse_args()
    
    if args.command == 'evaluate':
        config = Evaluator.get_default_config()
        config.batch_size = args.batch_size
        print(f"Evaluating model: {args.model}")
        print("Note: Dataset loading requires custom implementation")
    elif args.command == 'compare':
        if len(args.models) != len(args.names):
            print("Error: Number of models and names must match")
            return
        print(f"Comparing {len(args.models)} models")
        print("Note: Dataset loading requires custom implementation")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
