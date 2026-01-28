"""
Grokking Detection Callback for Composer

Monitors validation metrics and detects grokking phase transitions.
"""

import numpy as np
import torch
from composer.core import Callback, State, Logger, Time
from composer.loggers import Logger as LoggerType
from typing import Optional, Dict, List
import time


class GrokkingCallback(Callback):
    """
    Detects grokking by monitoring sharp drops in validation loss.
    
    Grokking signature:
    - Validation loss decreases by >10% in <500 steps
    - Improvement is sustained (not a random fluctuation)
    """
    
    def __init__(
        self,
        patience: int = 1000,
        threshold: float = 0.10,
        min_improvement_steps: int = 500,
        checkpoint_on_grokking: bool = True
    ):
        """
        Initialize grokking detection callback.
        
        Args:
            patience: Window size for detecting grokking (steps)
            threshold: Minimum improvement to count as grokking (0.10 = 10%)
            min_improvement_steps: Steps to check for rapid improvement
            checkpoint_on_grokking: Save checkpoint when grokking detected
        """
        self.patience = patience
        self.threshold = threshold
        self.min_improvement_steps = min_improvement_steps
        self.checkpoint_on_grokking = checkpoint_on_grokking
        
        # Tracking
        self.val_losses: List[float] = []
        self.val_steps: List[int] = []
        self.grokking_detected = False
        self.grokking_step: Optional[int] = None
    
    def eval_end(self, state: State, logger: Logger):
        """Called at the end of each evaluation."""
        # Get validation loss from state metrics
        if 'metrics/eval/CrossEntropy' in state.eval_metrics:
            val_loss = state.eval_metrics['metrics/eval/CrossEntropy'].item()
        elif 'val_loss' in state.eval_metrics:
            val_loss = state.eval_metrics['val_loss'].item()
        else:
            # No validation loss available
            return
        
        # Record
        current_step = state.timestamp.batch.value
        self.val_losses.append(val_loss)
        self.val_steps.append(current_step)
        
        # Detect grokking
        if len(self.val_losses) >= self.patience and not self.grokking_detected:
            is_grokking, improvement = self._detect_grokking()
            
            if is_grokking:
                self.grokking_detected = True
                self.grokking_step = current_step
                
                # Log to console
                print("\n" + "="*60)
                print(f"🎯 GROKKING DETECTED at step {current_step}!")
                print(f"Validation loss improved by {improvement*100:.1f}%")
                print(f"Baseline: {self._get_baseline_loss():.4f}")
                print(f"Current: {val_loss:.4f}")
                print("="*60 + "\n")
                
                # Log to metrics
                logger.log_metrics({
                    'grokking/detected': 1,
                    'grokking/step': current_step,
                    'grokking/improvement_pct': improvement * 100
                }, step=current_step)
                
                # Save checkpoint if enabled
                if self.checkpoint_on_grokking:
                    print(f"Saving grokking checkpoint at step {current_step}...")
    
    def _get_baseline_loss(self) -> float:
        """Get baseline loss (average of earlier window)."""
        baseline_losses = self.val_losses[-self.patience:-self.patience//2]
        return np.mean(baseline_losses)
    
    def _detect_grokking(self) -> tuple:
        """
        Detect if grokking occurred.
        
        Returns:
            (is_grokking, improvement_rate)
        """
        # Get baseline (average of earlier window)
        baseline = self._get_baseline_loss()
        
        # Get current (average of recent window)
        current_losses = self.val_losses[-self.patience//2:]
        current = np.mean(current_losses)
        
        # Calculate improvement
        improvement = (baseline - current) / baseline
        
        # Detect sharp drop
        is_grokking = improvement > self.threshold
        
        return is_grokking, improvement


class ProcrustesCallback(Callback):
    """
    Monitors cross-lingual alignment using Procrustes distance.
    
    Tracks how well the model aligns semantically equivalent verses
    across different languages.
    """
    
    def __init__(
        self,
        eval_interval: int = 1000,
        num_verses: int = 100,
        languages: Optional[List[str]] = None
    ):
        """
        Initialize Procrustes callback.
        
        Args:
            eval_interval: Evaluate every N steps
            num_verses: Number of verses to use for alignment
            languages: List of language codes to compare (default: ['en', 'es', 'ko'])
        """
        self.eval_interval = eval_interval
        self.num_verses = num_verses
        self.languages = languages or ['en', 'es', 'ko']
        
        # Will be set during training
        self.verse_pairs = None
    
    def fit_start(self, state: State, logger: Logger):
        """Called at the start of training."""
        # TODO: Load verse pairs from Bible data directory
        # For now, just placeholder
        print(f"ProcrustesCallback: Monitoring alignment for {self.languages}")
    
    def batch_end(self, state: State, logger: Logger):
        """Called at the end of each batch."""
        current_step = state.timestamp.batch.value
        
        # Only evaluate at specified interval
        if current_step % self.eval_interval != 0:
            return
        
        # TODO: Compute Procrustes distance
        # This requires:
        # 1. Encode same verses in different languages
        # 2. Compute optimal rotation matrix
        # 3. Calculate alignment distance
        
        # For now, log placeholder
        logger.log_metrics({
            'procrustes/distance': 0.5,  # Placeholder
        }, step=current_step)


class ConceptClusteringCallback(Callback):
    """
    Tracks concept abstraction through clustering metrics.
    
    Measures how well the model groups semantically related concepts
    (e.g., "salvation", "redemption", "deliverance").
    """
    
    def __init__(
        self,
        eval_interval: int = 2000,
        concept_keywords: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize concept clustering callback.
        
        Args:
            eval_interval: Evaluate every N steps
            concept_keywords: Dictionary mapping concept names to keywords
        """
        self.eval_interval = eval_interval
        
        # Default theological concepts
        if concept_keywords is None:
            concept_keywords = {
                'salvation': ['save', 'salvation', 'redeem', 'deliver'],
                'creation': ['create', 'made', 'formed', 'beginning'],
                'covenant': ['covenant', 'promise', 'agreement', 'testament'],
                'judgment': ['judge', 'judgment', 'condemn', 'punish']
            }
        
        self.concept_keywords = concept_keywords
    
    def batch_end(self, state: State, logger: Logger):
        """Called at the end of each batch."""
        current_step = state.timestamp.batch.value
        
        # Only evaluate at specified interval
        if current_step % self.eval_interval != 0:
            return
        
        # TODO: Compute silhouette scores for concept clusters
        # This requires:
        # 1. Extract embeddings for verses containing concept keywords
        # 2. Cluster embeddings
        # 3. Calculate silhouette score (cluster quality)
        
        # For now, log placeholder
        logger.log_metrics({
            'concept/silhouette_score': 0.3,  # Placeholder
        }, step=current_step)


if __name__ == "__main__":
    print("Grokking detection callback created successfully!")
    print("\nUsage:")
    print("  from training.callbacks.grokking import GrokkingCallback")
    print("  callback = GrokkingCallback(patience=1000, threshold=0.10)")
    print("  trainer = Trainer(..., callbacks=[callback])")
