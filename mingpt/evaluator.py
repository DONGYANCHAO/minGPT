"""
Model evaluation with perplexity, accuracy, BLEU, ROUGE metrics
and HuggingFace Evaluate integration.
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mingpt.utils import CfgNode as CN


class ModelEvaluator:

    @staticmethod
    def get_default_config():
        C = CN()
        C.batch_size = 32
        C.num_workers = 2
        C.device = 'auto'
        C.verbose = True
        
        C.perplexity = True
        C.accuracy = True
        C.bleu = True
        C.rouge = True
        
        C.max_gen_length = 100
        C.temperature = 1.0
        C.top_k = None
        
        C.use_hf_evaluate = True
        return C

    def __init__(self, config):
        self.config = config
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device

        self._hf_evaluate = None
        if config.use_hf_evaluate:
            try:
                import evaluate
                self._hf_evaluate = evaluate
                if config.verbose:
                    print("HuggingFace Evaluate available")
            except ImportError:
                if config.verbose:
                    print("HuggingFace Evaluate not available, install with: pip install evaluate")

    def calculate_perplexity(self, model, dataset) -> float:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        loader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            num_workers=self.config.num_workers,
            shuffle=False
        )

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                logits, loss = model(x, y)
                
                mask = (y != -1)
                n_tokens = mask.sum().item()
                
                if n_tokens > 0:
                    total_loss += loss.item() * n_tokens
                    total_tokens += n_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

    def calculate_accuracy(self, model, dataset) -> Dict[str, float]:
        model.eval()
        correct = 0
        total = 0
        correct_per_position = defaultdict(lambda: [0, 0])
        
        loader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            num_workers=self.config.num_workers,
            shuffle=False
        )

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                logits, _ = model(x, y)
                predictions = torch.argmax(logits, dim=-1)
                
                mask = (y != -1)
                correct_predictions = (predictions == y) & mask
                
                correct += correct_predictions.sum().item()
                total += mask.sum().item()
                
                for pos in range(y.size(1)):
                    pos_mask = mask[:, pos]
                    pos_correct = correct_predictions[:, pos] & pos_mask
                    correct_per_position[pos][0] += pos_correct.sum().item()
                    correct_per_position[pos][1] += pos_mask.sum().item()

        overall = correct / max(total, 1)
        position_acc = {
            f'position_{pos}': vals[0] / max(vals[1], 1)
            for pos, vals in correct_per_position.items()
        }
        
        return {
            'overall': overall,
            **position_acc
        }

    def calculate_bleu(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        if not self._hf_evaluate:
            predictions_split = [p.split() for p in predictions]
            references_split = [[r.split() for r in ref] for ref in references]
            return self._simple_bleu(predictions_split, references_split)

        bleu = self._hf_evaluate.load('bleu')
        results = bleu.compute(predictions=predictions, references=references)
        return {
            'bleu': results['bleu'],
            'precisions': results['precisions'],
            'brevity_penalty': results['brevity_penalty'],
            'length_ratio': results['length_ratio']
        }

    def _simple_bleu(self, predictions, references) -> Dict[str, float]:
        from collections import Counter
        import math

        def ngrams(seq, n):
            return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]

        total_score = 0.0
        for pred, refs in zip(predictions, references):
            score = 0.0
            for n in range(1, 5):
                pred_ngrams = Counter(ngrams(pred, n))
                max_ref_counts = Counter()
                for ref in refs:
                    ref_ngrams = Counter(ngrams(ref, n))
                    for ng, count in ref_ngrams.items():
                        max_ref_counts[ng] = max(max_ref_counts[ng], count)
                
                clipped = sum(min(pred_ngrams[ng], max_ref_counts[ng]) for ng in pred_ngrams)
                total = max(sum(pred_ngrams.values()), 1)
                score += math.log(max(clipped / total, 1e-10)) / 4
            
            total_score += math.exp(score)

        return {'bleu': total_score / max(len(predictions), 1)}

    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        if not self._hf_evaluate:
            return self._simple_rouge(predictions, references)

        rouge = self._hf_evaluate.load('rouge')
        results = rouge.compute(predictions=predictions, references=references)
        return {
            'rouge1': results['rouge1'],
            'rouge2': results['rouge2'],
            'rougeL': results['rougeL'],
        }

    def _simple_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        from collections import Counter

        def rouge_n(pred, ref, n=1):
            pred_ngrams = Counter([tuple(pred.split()[i:i+n]) for i in range(len(pred.split())-n+1)])
            ref_ngrams = Counter([tuple(ref.split()[i:i+n]) for i in range(len(ref.split())-n+1)])
            
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total = max(sum(ref_ngrams.values()), 1)
            return overlap / total

        scores = defaultdict(list)
        for pred, ref in zip(predictions, references):
            scores['rouge1'].append(rouge_n(pred, ref, n=1))
            scores['rouge2'].append(rouge_n(pred, ref, n=2))

        return {k: sum(v) / max(len(v), 1) for k, v in scores.items()}

    def evaluate_generation(
        self, 
        model, 
        dataset, 
        tokenizer=None,
        num_samples: int = 100
    ) -> Tuple[List[str], List[str]]:
        model.eval()
        predictions = []
        references = []
        
        loader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            num_workers=self.config.num_workers,
            shuffle=False
        )

        samples_processed = 0
        with torch.no_grad():
            for x, y in loader:
                if samples_processed >= num_samples:
                    break
                    
                x = x.to(self.device)
                batch_size = x.size(0)
                context_len = min(x.size(1) // 2, 10)
                
                context = x[:, :context_len]
                generated = model.generate(
                    context,
                    max_new_tokens=self.config.max_gen_length,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    do_sample=False
                )

                for i in range(batch_size):
                    if samples_processed >= num_samples:
                        break
                        
                    if tokenizer:
                        pred_str = tokenizer.decode(generated[i].cpu().tolist())
                        ref_str = tokenizer.decode(x[i].cpu().tolist())
                    else:
                        pred_str = ' '.join(map(str, generated[i].cpu().tolist()))
                        ref_str = ' '.join(map(str, x[i].cpu().tolist()))
                    
                    predictions.append(pred_str)
                    references.append(ref_str)
                    samples_processed += 1

        return predictions, references

    def evaluate_classification(
        self,
        model,
        dataset,
        num_classes: int
    ) -> Dict[str, float]:
        model.eval()
        all_preds = []
        all_labels = []
        
        loader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            num_workers=self.config.num_workers,
            shuffle=False
        )

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                logits, _ = model(x)
                
                last_logits = logits[:, -1, :num_classes]
                preds = torch.argmax(last_logits, dim=-1)
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())

        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / max(len(all_preds), 1)

        return {
            'accuracy': accuracy,
            'total_samples': len(all_preds),
            'correct': correct
        }

    def full_evaluation(
        self,
        model,
        dataset,
        task_type: str = 'language_modeling',
        tokenizer=None,
        num_gen_samples: int = 100
    ) -> Dict[str, Any]:
        model = model.to(self.device)
        results = {}

        if self.config.verbose:
            print("\n" + "="*60)
            print("Starting Model Evaluation")
            print("="*60)

        if self.config.perplexity:
            if self.config.verbose:
                print("\nCalculating Perplexity...")
            ppl = self.calculate_perplexity(model, dataset)
            results['perplexity'] = ppl
            if self.config.verbose:
                print(f"  Perplexity: {ppl:.2f}")

        if self.config.accuracy:
            if self.config.verbose:
                print("\nCalculating Token Accuracy...")
            acc = self.calculate_accuracy(model, dataset)
            results['accuracy'] = acc
            if self.config.verbose:
                print(f"  Overall Accuracy: {acc['overall']:.4f}")

        if task_type == 'language_modeling' and (self.config.bleu or self.config.rouge):
            if self.config.verbose:
                print("\nGenerating samples for BLEU/ROUGE...")
            preds, refs = self.evaluate_generation(model, dataset, tokenizer, num_gen_samples)
            
            if self.config.bleu:
                if self.config.verbose:
                    print("Calculating BLEU...")
                refs_list = [[r] for r in refs]
                bleu_scores = self.calculate_bleu(preds, refs_list)
                results['bleu'] = bleu_scores
                if self.config.verbose:
                    print(f"  BLEU: {bleu_scores.get('bleu', 0):.4f}")

            if self.config.rouge:
                if self.config.verbose:
                    print("Calculating ROUGE...")
                rouge_scores = self.calculate_rouge(preds, refs)
                results['rouge'] = rouge_scores
                if self.config.verbose:
                    print(f"  ROUGE-1: {rouge_scores.get('rouge1', 0):.4f}")
                    print(f"  ROUGE-2: {rouge_scores.get('rouge2', 0):.4f}")
                    print(f"  ROUGE-L: {rouge_scores.get('rougeL', 0):.4f}")

        if self.config.verbose:
            print("\n" + "="*60)
            print("Evaluation Complete!")
            print("="*60)

        return results

    def compare_models(
        self,
        model_paths: List[str],
        model_class,
        model_config,
        dataset
    ) -> Dict[str, Dict]:
        comparison = {}

        for name, path in model_paths.items():
            if self.config.verbose:
                print(f"\nEvaluating: {name}")
            
            model = model_class(model_config)
            checkpoint = torch.load(path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            results = self.full_evaluation(model, dataset)
            comparison[name] = results

        return comparison

    def save_report(self, results: Dict, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.config.verbose:
            print(f"\nEvaluation report saved to: {output_path}")
