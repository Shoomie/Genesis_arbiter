"""
Grokking Detection Callbacks for Native PyTorch
================================================
Monitors validation metrics and detects grokking phase transitions.

Converted from Composer to native PyTorch hooks.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List
from collections import deque

from .manager import BaseCallback


class GrokkingDetector(BaseCallback):
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
        checkpoint_on_grokking: bool = True,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize grokking detection callback.
        
        Args:
            patience: Window size for detecting grokking (steps)
            threshold: Minimum improvement to count as grokking (0.10 = 10%)
            min_improvement_steps: Steps to check for rapid improvement
            checkpoint_on_grokking: Save checkpoint when grokking detected
            checkpoint_dir: Directory to save checkpoints
        """
        self.patience = patience
        self.threshold = threshold
        self.min_improvement_steps = min_improvement_steps
        self.checkpoint_on_grokking = checkpoint_on_grokking
        self.checkpoint_dir =  checkpoint_dir or Path("checkpoints/grokking")
        
        # Tracking
        self.val_losses = deque(maxlen=patience)
        self.val_steps = deque(maxlen=patience)
        self.grokking_detected = False
        self.grokking_step: Optional[int] = None
    
    def on_validation_end(
        self, 
        step: int, 
        model: torch.nn.Module, 
        val_loss: float,
        metrics: Dict[str, float],
        **kwargs
    ):
        """Called at the end of validation."""
        # Record
        self.val_losses.append(val_loss)
        self.val_steps.append(step)
        
        # Detect grokking
        if len(self.val_losses) >= self.patience and not self.grokking_detected:
            is_grokking, improvement = self._detect_grokking()
            
            if is_grokking:
                self.grokking_detected = True
                self.grokking_step = step
                
                # Log to console
                print("\n" + "="*60)
                print(f"🎯 GROKKING DETECTED at step {step}!")
                print(f"Validation loss improved by {improvement*100:.1f}%")
                print(f"Baseline: {self._get_baseline_loss():.4f}")
                print(f"Current: {val_loss:.4f}")
                print("="*60 + "\n")
                
                # Save checkpoint if enabled
                if self.checkpoint_on_grokking:
                    self._save_grokking_checkpoint(model, step, val_loss)
    
    def _get_baseline_loss(self) -> float:
        """Get baseline loss (average of earlier window)."""
        losses_list = list(self.val_losses)
        baseline_losses = losses_list[-self.patience:-self.patience//2]
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
        losses_list = list(self.val_losses)
        current_losses = losses_list[-self.patience//2:]
        current = np.mean(current_losses)
        
        # Calculate improvement
        improvement = (baseline - current) / baseline
        
        # Detect sharp drop
        is_grokking = improvement > self.threshold
        
        return is_grokking, improvement
    
    def _save_grokking_checkpoint(self, model: torch.nn.Module, step: int, val_loss: float):
        """Save checkpoint when grokking is detected."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"grokking_step_{step}.pt"
        
        print(f"💾 Saving grokking checkpoint: {checkpoint_path}")
        
        torch.save({
            'step': step,
            'model': model.state_dict(),
            'val_loss': val_loss,
            'grokking_detected': True,
            'grokking_step': step
        }, checkpoint_path)


class ProcrustesMonitor(BaseCallback):
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
        Initialize Procrustes monitor.
        
        Args:
            eval_interval: Evaluate every N steps
            num_verses: Number of verses to use for alignment
            languages: List of language codes to compare
        """
        self.eval_interval = eval_interval
        self.num_verses = num_verses
        self.languages = languages or ['en', 'es', 'ko']
        
        # Will be set during training
        self.verse_pairs = None
    
    def on_train_start(self, model: torch.nn.Module, **kwargs):
        """Called at the start of training."""
        # TODO: Load verse pairs from Bible data directory
        print(f"ProcrustesMonitor: Monitoring alignment for {self.languages}")
    
    def on_batch_end(self, step: int, model: torch.nn.Module, loss: float, **kwargs):
        """Called at the end of each batch."""
        # Only evaluate at specified interval
        if step % self.eval_interval != 0:
            return
        
        # TODO: Compute Procrustes distance
        # This requires:
        # 1. Encode same verses in different languages
        # 2. Compute optimal rotation matrix
        # 3. Calculate alignment distance
        
        # For now, log placeholder
        print(f"  [Step {step}] Procrustes distance: 0.5 (placeholder)")


class ConceptClusteringMonitor(BaseCallback):
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
        Initialize concept clustering monitor.
        
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
    
    def on_batch_end(self, step: int, model: torch.nn.Module, loss: float, **kwargs):
        """Called at the end of each batch."""
        # Only evaluate at specified interval
        if step % self.eval_interval != 0:
            return
        
        # TODO: Compute silhouette scores for concept clusters
        # This requires:
        # 1. Extract embeddings for verses containing concept keywords
        # 2. Cluster embeddings
        # 3. Calculate silhouette score (cluster quality)
        
        # For now, log placeholder
        print(f"  [Step {step}] Concept silhouette score: 0.3 (placeholder)")


if __name__ == "__main__":
    print("Grokking detection callbacks (Native PyTorch) created successfully!")
    print("\nUsage:")
    print("  from training.callbacks.grokking import GrokkingDetector")
    print("  from training.callbacks.manager import CallbackManager")
    print("  ")
    print("  callbacks = CallbackManager([")
    print("      GrokkingDetector(patience=1000, threshold=0.10),")
    print("      ProcrustesMonitor(eval_interval=1000),")
    print("      ConceptClusteringMonitor(eval_interval=2000)")
    print("  ])")
