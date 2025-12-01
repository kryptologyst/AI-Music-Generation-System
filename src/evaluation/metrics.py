"""Evaluation metrics for music generation models."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from collections import Counter
import math

logger = logging.getLogger(__name__)


class MusicMetrics:
    """Collection of music-specific evaluation metrics."""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
    
    def perplexity(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute perplexity from logits and targets."""
        # Flatten tensors
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, targets, reduction='mean')
        
        # Perplexity is exp(loss)
        return torch.exp(loss).item()
    
    def negative_log_likelihood(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute negative log-likelihood."""
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        loss = F.cross_entropy(logits, targets, reduction='mean')
        return loss.item()
    
    def diversity_metrics(self, generated_sequences: List[List[int]]) -> Dict[str, float]:
        """Compute diversity metrics for generated sequences."""
        if not generated_sequences:
            return {"distinct_1": 0.0, "distinct_2": 0.0, "self_bleu": 0.0}
        
        # Flatten all sequences
        all_tokens = []
        for seq in generated_sequences:
            all_tokens.extend(seq)
        
        # Distinct-1: unique unigrams / total unigrams
        unique_unigrams = len(set(all_tokens))
        total_unigrams = len(all_tokens)
        distinct_1 = unique_unigrams / total_unigrams if total_unigrams > 0 else 0.0
        
        # Distinct-2: unique bigrams / total bigrams
        bigrams = []
        for seq in generated_sequences:
            for i in range(len(seq) - 1):
                bigrams.append((seq[i], seq[i + 1]))
        
        unique_bigrams = len(set(bigrams))
        total_bigrams = len(bigrams)
        distinct_2 = unique_bigrams / total_bigrams if total_bigrams > 0 else 0.0
        
        # Self-BLEU (simplified version)
        self_bleu = self._compute_self_bleu(generated_sequences)
        
        return {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
            "self_bleu": self_bleu
        }
    
    def _compute_self_bleu(self, sequences: List[List[int]], n: int = 2) -> float:
        """Compute self-BLEU score."""
        if len(sequences) < 2:
            return 0.0
        
        bleu_scores = []
        
        for i, seq in enumerate(sequences):
            other_seqs = sequences[:i] + sequences[i+1:]
            if not other_seqs:
                continue
            
            # Compute n-gram precision
            seq_ngrams = self._get_ngrams(seq, n)
            if not seq_ngrams:
                continue
            
            max_precision = 0.0
            for other_seq in other_seqs:
                other_ngrams = self._get_ngrams(other_seq, n)
                if not other_ngrams:
                    continue
                
                overlap = len(seq_ngrams & other_ngrams)
                precision = overlap / len(seq_ngrams) if seq_ngrams else 0.0
                max_precision = max(max_precision, precision)
            
            bleu_scores.append(max_precision)
        
        return np.mean(bleu_scores) if bleu_scores else 0.0
    
    def _get_ngrams(self, sequence: List[int], n: int) -> set:
        """Get n-grams from a sequence."""
        ngrams = set()
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i + n])
            ngrams.add(ngram)
        return ngrams
    
    def rhythm_consistency(self, sequences: List[List[int]], int_to_note: Dict[int, str]) -> float:
        """Compute rhythm consistency metric."""
        if not sequences:
            return 0.0
        
        rhythm_patterns = []
        
        for seq in sequences:
            # Extract rhythm patterns (simplified)
            durations = []
            for token in seq:
                if token in int_to_note:
                    note_str = int_to_note[token]
                    if '_' in note_str:
                        duration = float(note_str.split('_')[-1])
                        durations.append(duration)
            
            if durations:
                # Normalize durations to create pattern
                pattern = tuple(np.round(np.array(durations) * 4).astype(int))
                rhythm_patterns.append(pattern)
        
        if not rhythm_patterns:
            return 0.0
        
        # Compute consistency as the most common pattern frequency
        pattern_counts = Counter(rhythm_patterns)
        most_common_count = max(pattern_counts.values())
        consistency = most_common_count / len(rhythm_patterns)
        
        return consistency
    
    def coherence_score(self, sequences: List[List[int]], int_to_note: Dict[int, str]) -> float:
        """Compute coherence score based on musical intervals."""
        if not sequences:
            return 0.0
        
        coherence_scores = []
        
        for seq in sequences:
            pitches = []
            for token in seq:
                if token in int_to_note:
                    note_str = int_to_note[token]
                    if '_' in note_str:
                        pitch = int(note_str.split('_')[0])
                        pitches.append(pitch)
            
            if len(pitches) < 2:
                continue
            
            # Compute interval consistency
            intervals = []
            for i in range(len(pitches) - 1):
                interval = abs(pitches[i + 1] - pitches[i])
                intervals.append(interval)
            
            if intervals:
                # Coherence based on interval distribution
                interval_counts = Counter(intervals)
                most_common_interval = max(interval_counts.values())
                coherence = most_common_interval / len(intervals)
                coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        test_loader,
        device: torch.device = torch.device("cpu"),
        int_to_note: Optional[Dict[int, str]] = None
    ) -> Dict[str, float]:
        """Comprehensive model evaluation."""
        model.eval()
        
        all_losses = []
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_seq, target = batch
                input_seq = input_seq.to(device)
                target = target.to(device)
                
                # Forward pass
                if hasattr(model, 'model') and hasattr(model.model, 'init_hidden'):
                    hidden = model.model.init_hidden(input_seq.size(0), device)
                    output, _ = model(input_seq, hidden)
                else:
                    output = model(input_seq)
                
                # Compute loss
                logits = output.view(-1, output.size(-1))
                targets = target.view(-1)
                loss = F.cross_entropy(logits, targets)
                
                all_losses.append(loss.item())
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all results
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = {
            "perplexity": self.perplexity(all_logits, all_targets),
            "nll": self.negative_log_likelihood(all_logits, all_targets),
            "loss": np.mean(all_losses)
        }
        
        # Generate samples for diversity evaluation
        if int_to_note is not None:
            generated_sequences = self._generate_samples(model, device, num_samples=50)
            diversity_metrics = self.diversity_metrics(generated_sequences)
            metrics.update(diversity_metrics)
            
            # Additional music-specific metrics
            metrics["rhythm_consistency"] = self.rhythm_consistency(generated_sequences, int_to_note)
            metrics["coherence"] = self.coherence_score(generated_sequences, int_to_note)
        
        return metrics
    
    def _generate_samples(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_samples: int = 50,
        max_length: int = 100
    ) -> List[List[int]]:
        """Generate samples for evaluation."""
        model.eval()
        generated_sequences = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Start with random seed
                seed_length = 10
                current_seq = torch.randint(0, self.vocab_size, (1, seed_length), device=device)
                
                generated_seq = current_seq[0].tolist()
                
                for _ in range(max_length - seed_length):
                    # Get model prediction
                    if hasattr(model, 'model') and hasattr(model.model, 'init_hidden'):
                        hidden = model.model.init_hidden(1, device)
                        output, _ = model(current_seq, hidden)
                    else:
                        output = model(current_seq)
                    
                    # Sample next token
                    logits = output[:, -1, :]
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    generated_seq.append(next_token.item())
                    
                    # Update current sequence
                    current_seq = torch.cat([current_seq[:, 1:], next_token], dim=1)
                
                generated_sequences.append(generated_seq)
        
        return generated_sequences


def create_evaluation_report(
    metrics: Dict[str, float],
    model_name: str = "Music Generation Model"
) -> str:
    """Create a formatted evaluation report."""
    report = f"\n{'='*50}\n"
    report += f"EVALUATION REPORT: {model_name}\n"
    report += f"{'='*50}\n\n"
    
    # Basic metrics
    report += "BASIC METRICS:\n"
    report += f"  Perplexity: {metrics.get('perplexity', 0.0):.4f}\n"
    report += f"  Negative Log-Likelihood: {metrics.get('nll', 0.0):.4f}\n"
    report += f"  Loss: {metrics.get('loss', 0.0):.4f}\n\n"
    
    # Diversity metrics
    report += "DIVERSITY METRICS:\n"
    report += f"  Distinct-1: {metrics.get('distinct_1', 0.0):.4f}\n"
    report += f"  Distinct-2: {metrics.get('distinct_2', 0.0):.4f}\n"
    report += f"  Self-BLEU: {metrics.get('self_bleu', 0.0):.4f}\n\n"
    
    # Music-specific metrics
    report += "MUSIC-SPECIFIC METRICS:\n"
    report += f"  Rhythm Consistency: {metrics.get('rhythm_consistency', 0.0):.4f}\n"
    report += f"  Coherence: {metrics.get('coherence', 0.0):.4f}\n\n"
    
    # Overall score
    overall_score = (
        metrics.get('distinct_1', 0.0) * 0.3 +
        metrics.get('distinct_2', 0.0) * 0.3 +
        metrics.get('rhythm_consistency', 0.0) * 0.2 +
        metrics.get('coherence', 0.0) * 0.2
    )
    report += f"OVERALL SCORE: {overall_score:.4f}\n"
    report += f"{'='*50}\n"
    
    return report
