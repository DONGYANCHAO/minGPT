"""
Model evaluation utilities for minGPT.
Supports perplexity, accuracy, BLEU, ROUGE metrics and HuggingFace Evaluate integration.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from mingpt.utils import CfgNode as CN


# -----------------------------------------------------------------------------
# Metric Classes

@dataclass
class MetricResult:
    """Container for metric computation results."""
    name: str
    value: float
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'details': self.details or {}
        }


class PerplexityMetric:
    """Compute perplexity from cross-entropy loss."""
    
    def compute(self, total_loss: float, num_tokens: int) -> MetricResult:
        """
        Compute perplexity.
        
        Args:
            total_loss: Sum of cross-entropy losses
            num_tokens: Number of tokens evaluated
        
        Returns:
            MetricResult with perplexity value
        """
        avg_loss = total_loss / num_tokens if num_tokens > 0 else float('inf')
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return MetricResult(
            name='perplexity',
            value=perplexity,
            details={'avg_loss': avg_loss, 'num_tokens': num_tokens}
        )


class AccuracyMetric:
    """Compute token-level accuracy."""
    
    def compute(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> MetricResult:
        """
        Compute accuracy.
        
        Args:
            predictions: Predicted token IDs [batch_size, seq_len]
            targets: Target token IDs [batch_size, seq_len]
            mask: Optional mask for valid positions
        
        Returns:
            MetricResult with accuracy value
        """
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)
        
        correct = (predictions == targets).float() * mask.float()
        accuracy = correct.sum() / mask.sum()
        
        return MetricResult(
            name='accuracy',
            value=accuracy.item(),
            details={'correct': int(correct.sum()), 'total': int(mask.sum())}
        )


class BLEUMetric:
    """Compute BLEU score for text generation."""
    
    def __init__(self, max_n: int = 4):
        self.max_n = max_n
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            self.sentence_bleu = sentence_bleu
            self.smoothing = SmoothingFunction().method1
            self.available = True
        except ImportError:
            self.available = False
    
    def compute(self, references: List[List[str]], hypotheses: List[List[str]]) -> MetricResult:
        """
        Compute corpus-level BLEU score.
        
        Args:
            references: List of reference token lists
            hypotheses: List of hypothesis token lists
        
        Returns:
            MetricResult with BLEU score
        """
        if not self.available:
            return MetricResult(name='bleu', value=0.0, details={'error': 'nltk not installed'})
        
        scores = []
        for ref, hyp in zip(references, hypotheses):
            try:
                score = self.sentence_bleu([ref], hyp, smoothing_function=self.smoothing)
                scores.append(score)
            except:
                scores.append(0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return MetricResult(
            name='bleu',
            value=avg_score,
            details={'num_samples': len(scores), 'individual_scores': scores}
        )


class ROUGEMetric:
    """Compute ROUGE score for text generation."""
    
    def __init__(self):
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.available = True
        except ImportError:
            self.available = False
    
    def compute(self, references: List[str], hypotheses: List[str]) -> MetricResult:
        """
        Compute ROUGE scores.
        
        Args:
            references: List of reference strings
            hypotheses: List of hypothesis strings
        
        Returns:
            MetricResult with ROUGE scores
        """
        if not self.available:
            return MetricResult(name='rouge', value=0.0, details={'error': 'rouge-score not installed'})
        
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for ref, hyp in zip(references, hypotheses):
            score = self.scorer.score(ref, hyp)
            for key in scores:
                scores[key].append(score[key].fmeasure)
        
        avg_scores = {key: sum(vals) / len(vals) if vals else 0.0 for key, vals in scores.items()}
        
        return MetricResult(
            name='rouge',
            value=avg_scores['rougeL'],  # use ROUGE-L as main metric
            details=avg_scores
        )


# -----------------------------------------------------------------------------
# Evaluator Classes

class GPTEvaluator:
    """
    Evaluator for GPT language models.
    
    Supports:
    - Language modeling evaluation (perplexity)
    - Text classification evaluation (accuracy)
    - Text generation evaluation (BLEU, ROUGE)
    - HuggingFace Evaluate integration
    """
    
    @staticmethod
    def get_default_config():
        C = CN()
        # batch size for evaluation
        C.batch_size = 32
        # max sequence length
        C.max_length = 512
        # number of samples for generation evaluation
        C.num_samples = 100
        # max new tokens for generation
        C.max_new_tokens = 100
        # temperature for generation
        C.temperature = 1.0
        # top-k sampling
        C.top_k = None
        # metrics to compute
        C.metrics = ['perplexity', 'accuracy']
        # device for evaluation
        C.device = 'auto'
        return C
    
    def __init__(self, config, model: nn.Module, tokenizer: Optional[Any] = None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        # setup device
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        
        # initialize metrics
        self.perplexity_metric = PerplexityMetric()
        self.accuracy_metric = AccuracyMetric()
        self.bleu_metric = BLEUMetric()
        self.rouge_metric = ROUGEMetric()
        
        # HuggingFace evaluate
        self.hf_evaluate = None
        try:
            import evaluate
            self.hf_evaluate = evaluate
        except ImportError:
            pass
    
    @torch.no_grad()
    def evaluate_language_modeling(self, dataset: Dataset) -> Dict[str, MetricResult]:
        """
        Evaluate on language modeling task.
        
        Args:
            dataset: Dataset with (input_ids, target_ids) pairs
        
        Returns:
            Dictionary of metric results
        """
        self.model.eval()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        total_loss = 0.0
        total_tokens = 0
        total_correct = 0
        total_predictions = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x, y = batch['input_ids'], batch['labels']
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            # forward pass
            logits, loss = self.model(x, y)
            
            # accumulate metrics
            if loss is not None:
                # compute number of valid tokens (not -1)
                valid_mask = (y != -1)
                num_valid = valid_mask.sum().item()
                
                total_loss += loss.item() * num_valid
                total_tokens += num_valid
            
            # compute accuracy
            predictions = logits.argmax(dim=-1)
            if 'accuracy' in self.config.metrics:
                acc_result = self.accuracy_metric.compute(predictions, y, valid_mask)
                total_correct += acc_result.details['correct']
                total_predictions += acc_result.details['total']
        
        results = {}
        
        # perplexity
        if 'perplexity' in self.config.metrics:
            results['perplexity'] = self.perplexity_metric.compute(total_loss, total_tokens)
        
        # accuracy
        if 'accuracy' in self.config.metrics and total_predictions > 0:
            results['accuracy'] = MetricResult(
                name='accuracy',
                value=total_correct / total_predictions,
                details={'correct': total_correct, 'total': total_predictions}
            )
        
        return results
    
    @torch.no_grad()
    def evaluate_classification(self, dataset: Dataset) -> Dict[str, MetricResult]:
        """
        Evaluate on text classification task.
        
        Args:
            dataset: Dataset with (input_ids, labels) pairs
        
        Returns:
            Dictionary of metric results
        """
        self.model.eval()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x, y = batch['input_ids'], batch['labels']
            
            x = x.to(self.device)
            y = y.to(self.device)
            
            # forward pass
            logits, _ = self.model(x)
            
            # get predictions (assume last token is for classification)
            predictions = logits[:, -1, :].argmax(dim=-1)
            
            total_correct += (predictions == y).sum().item()
            total_samples += y.size(0)
            
            all_preds.extend(predictions.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
        
        results = {
            'accuracy': MetricResult(
                name='accuracy',
                value=total_correct / total_samples if total_samples > 0 else 0.0,
                details={'correct': total_correct, 'total': total_samples}
            )
        }
        
        # use HuggingFace evaluate for additional metrics if available
        if self.hf_evaluate:
            try:
                f1_metric = self.hf_evaluate.load('f1')
                f1_score = f1_metric.compute(predictions=all_preds, references=all_labels, average='weighted')
                results['f1'] = MetricResult(name='f1', value=f1_score['f1'])
            except:
                pass
        
        return results
    
    @torch.no_grad()
    def evaluate_generation(self, dataset: Dataset, num_samples: Optional[int] = None) -> Dict[str, MetricResult]:
        """
        Evaluate text generation quality.
        
        Args:
            dataset: Dataset with prompt/target pairs
            num_samples: Number of samples to evaluate (None for all)
        
        Returns:
            Dictionary of metric results
        """
        self.model.eval()
        
        if num_samples is None:
            num_samples = min(self.config.num_samples, len(dataset))
        
        references = []
        hypotheses = []
        
        for i in range(num_samples):
            sample = dataset[i]
            
            if isinstance(sample, (list, tuple)):
                prompt, target = sample
            else:
                prompt, target = sample['prompt'], sample['target']
            
            # generate
            if isinstance(prompt, str) and self.tokenizer:
                prompt_ids = self.tokenizer.encode(prompt)
                prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long).to(self.device)
            else:
                prompt_tensor = prompt.unsqueeze(0).to(self.device) if prompt.dim() == 1 else prompt.to(self.device)
            
            generated = self.model.generate(
                prompt_tensor,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                top_k=self.config.top_k
            )[0]
            
            # decode
            if self.tokenizer:
                generated_text = self.tokenizer.decode(generated.cpu().tolist())
                target_text = target if isinstance(target, str) else self.tokenizer.decode(target.tolist())
            else:
                generated_text = ' '.join(map(str, generated.cpu().tolist()))
                target_text = ' '.join(map(str, target.tolist() if hasattr(target, 'tolist') else target))
            
            references.append(target_text)
            hypotheses.append(generated_text)
        
        results = {}
        
        # BLEU
        if 'bleu' in self.config.metrics:
            ref_tokens = [ref.split() for ref in references]
            hyp_tokens = [hyp.split() for hyp in hypotheses]
            results['bleu'] = self.bleu_metric.compute(ref_tokens, hyp_tokens)
        
        # ROUGE
        if 'rouge' in self.config.metrics:
            results['rouge'] = self.rouge_metric.compute(references, hypotheses)
        
        return results
    
    def evaluate(self, dataset: Dataset, task_type: str = 'language_modeling') -> Dict[str, MetricResult]:
        """
        Main evaluation entry point.
        
        Args:
            dataset: Evaluation dataset
            task_type: One of 'language_modeling', 'classification', 'generation'
        
        Returns:
            Dictionary of metric results
        """
        if task_type == 'language_modeling':
            return self.evaluate_language_modeling(dataset)
        elif task_type == 'classification':
            return self.evaluate_classification(dataset)
        elif task_type == 'generation':
            return self.evaluate_generation(dataset)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def evaluate_hf_benchmark(self, benchmark_name: str, split: str = 'validation') -> Dict[str, MetricResult]:
        """
        Evaluate on a HuggingFace benchmark dataset.
        
        Args:
            benchmark_name: Name of the benchmark (e.g., 'wikitext', 'glue')
            split: Dataset split to use
        
        Returns:
            Dictionary of metric results
        """
        if self.hf_evaluate is None:
            raise ImportError("HuggingFace datasets and evaluate libraries required")
        
        from datasets import load_dataset
        
        # load dataset
        dataset = load_dataset(benchmark_name, split=split)
        
        # evaluate based on benchmark type
        if benchmark_name.startswith('glue'):
            return self.evaluate_classification(dataset)
        else:
            return self.evaluate_language_modeling(dataset)


# -----------------------------------------------------------------------------
# Evaluation Report

class EvaluationReport:
    """Generate and save evaluation reports."""
    
    def __init__(self, results: Dict[str, Dict[str, MetricResult]], model_names: Optional[List[str]] = None):
        self.results = results
        self.model_names = model_names or list(results.keys())
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'timestamp': self.timestamp,
            'models': {
                name: {metric: result.to_dict() for metric, result in model_results.items()}
                for name, model_results in self.results.items()
            }
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Model Evaluation Report",
            f"",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
            f"",
        ]
        
        # comparison table if multiple models
        if len(self.model_names) > 1:
            lines.extend([
                "## Model Comparison",
                "",
                "| Model | Metric | Value |",
                "|-------|--------|-------|"
            ])
            
            for model_name in self.model_names:
                for metric_name, result in self.results[model_name].items():
                    lines.append(f"| {model_name} | {metric_name} | {result.value:.4f} |")
            
            lines.append("")
        
        # detailed results for each model
        for model_name in self.model_names:
            lines.extend([
                f"## {model_name}",
                "",
            ])
            
            for metric_name, result in self.results[model_name].items():
                lines.extend([
                    f"### {metric_name}",
                    f"- **Value:** {result.value:.4f}",
                ])
                
                if result.details:
                    lines.append("- **Details:**")
                    for key, value in result.details.items():
                        if isinstance(value, float):
                            lines.append(f"  - {key}: {value:.4f}")
                        else:
                            lines.append(f"  - {key}: {value}")
                
                lines.append("")
        
        return '\n'.join(lines)
    
    def save(self, output_path: str, format: str = 'both'):
        """
        Save report to file.
        
        Args:
            output_path: Base path for output files
            format: 'json', 'markdown', or 'both'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format in ('json', 'both'):
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            print(f"Saved JSON report: {json_path}")
        
        if format in ('markdown', 'both'):
            md_path = output_path.with_suffix('.md')
            with open(md_path, 'w') as f:
                f.write(self.to_markdown())
            print(f"Saved Markdown report: {md_path}")


# -----------------------------------------------------------------------------
# Utility Functions

def evaluate_model(model: nn.Module, dataset: Dataset, config: Optional[CN] = None, 
                   task_type: str = 'language_modeling') -> Dict[str, MetricResult]:
    """
    Quick evaluation function.
    
    Args:
        model: Model to evaluate
        dataset: Evaluation dataset
        config: Evaluation configuration
        task_type: Type of evaluation task
    
    Returns:
        Dictionary of metric results
    """
    if config is None:
        config = GPTEvaluator.get_default_config()
    
    evaluator = GPTEvaluator(config, model)
    return evaluator.evaluate(dataset, task_type)


def compare_models(models: Dict[str, nn.Module], dataset: Dataset, 
                   config: Optional[CN] = None, task_type: str = 'language_modeling',
                   output_path: Optional[str] = None) -> EvaluationReport:
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary mapping model names to models
        dataset: Evaluation dataset
        config: Evaluation configuration
        task_type: Type of evaluation task
        output_path: Path to save comparison report
    
    Returns:
        EvaluationReport with comparison results
    """
    if config is None:
        config = GPTEvaluator.get_default_config()
    
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        evaluator = GPTEvaluator(config, model)
        results[name] = evaluator.evaluate(dataset, task_type)
    
    report = EvaluationReport(results, list(models.keys()))
    
    if output_path:
        report.save(output_path)
    
    return report
