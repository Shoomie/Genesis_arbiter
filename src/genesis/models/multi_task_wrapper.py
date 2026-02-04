"""
Multi-Task Wrapper for Genesis Llama Model

Adds task-specific heads for:
1. Language Modeling (70% of training)
2. Coherence Detection (15% of training)
3. Cross-Reference Prediction (7.5% of training)
4. Cross-Lingual Paraphrase (7.5% of training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class MultiTaskLlama(nn.Module):
    """
    Wrapper around base Llama model that adds multiple task-specific heads.
    
    The model shares a common transformer backbone but has different output heads
    for each task. This enables the model to learn more robust representations
    by training on diverse objectives simultaneously.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        dim: int = None,
        vocab_size: int = None,
        coherence_hidden_dim: int = 1024,
    ):
        """
        Initialize multi-task wrapper.
        
        Args:
            base_model: The base Llama transformer model
            dim: Model dimension (required if base_model doesn't have .dim attribute)
            vocab_size: Vocabulary size (required if base_model doesn't have .vocab_size attribute)
            coherence_hidden_dim: Hidden dimension for coherence detection MLP
        """
        super().__init__()
        self.base = base_model
        
        # Get dim and vocab_size from base_model or parameters
        self.dim = dim if dim is not None else getattr(base_model, 'dim', None)
        self.vocab_size = vocab_size if vocab_size is not None else getattr(base_model, 'vocab_size', None)
        
        if self.dim is None:
            raise ValueError("dim must be provided either via parameter or base_model.dim")
        if self.vocab_size is None:
            raise ValueError("vocab_size must be provided either via parameter or base_model.vocab_size")
        
        # ===== Task Head 1: Language Modeling (Standard Causal LM) =====
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)
        # Tie weights with input embeddings (common practice)
        self.lm_head.weight = base_model.tok_embeddings.weight
        
        # Also ensure base model's own output head is tied if it exists
        if hasattr(base_model, 'output'):
            base_model.output.weight = base_model.tok_embeddings.weight
    
    def forward_lm(self, tokens: torch.Tensor, labels: torch.Tensor, reduction: str = 'mean') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for language modeling task.
        
        Args:
            tokens: Input token IDs, shape (batch, seq_len)
            labels: Target token IDs, shape (batch, seq_len)
            reduction: 'mean' (scalar) or 'none' (per-sample [batch])
        
        Returns:
            (loss, logits)
        """
        # Get hidden states from base model
        h, _ = self.base(tokens, return_hiddens=True)  # Shape: (batch, seq_len, dim)
        
        # Project to vocabulary
        logits = self.lm_head(h)  # Shape: (batch, seq_len, vocab_size)
        
        # Compute cross-entropy loss
        if reduction == 'mean':
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        else:
            # Per-token loss
            raw_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction='none'
            )
            # Reshape to [batch, seq_len]
            B, L = tokens.shape
            raw_loss = raw_loss.view(B, L)
            
            # Mask out ignored tokens for averaging
            mask = (labels != -100).float()
            # Mean per sequence
            loss = (raw_loss * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        
        return loss, logits
    
    
    def forward(self, batch: Dict[str, torch.Tensor], reduction: str = 'mean') -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Simplified forward pass for single-task LM.
        """
        loss, outputs = self.forward_lm(batch['tokens'], batch['labels'], reduction=reduction)
        return outputs, loss, 'lm'
    
    def get_num_params(self) -> int:
        """Return total number of parameters across all heads."""
        return sum(p.numel() for p in self.parameters())


def create_multi_task_model(
    base_model: nn.Module,
    config: Optional[Dict] = None
) -> MultiTaskLlama:
    """
    Factory function to create model wrapper.
    """
    return MultiTaskLlama(
        base_model=base_model
    )


if __name__ == "__main__":
    # Simple test
    from .llama.model import Llama
    
    print("Creating test model...")
    base = Llama(
        vocab_size=1000,
        n_layers=2,
        dim=128,
        n_heads=4,
        intermediate_size=256,
        max_seq_len=128
    )
    
    multi_task = create_multi_task_model(base)
    
    print(f"Base model parameters: {base.get_num_params():,}")
    print(f"Multi-task model parameters: {multi_task.get_num_params():,}")
    print(f"Additional parameters: {multi_task.get_num_params() - base.get_num_params():,}")
    
    print("\n✓ Multi-task wrapper created successfully!")
