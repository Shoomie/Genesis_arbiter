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
        enable_cross_ref: bool = True,
        enable_paraphrase: bool = True,
        cross_ref_margin: float = 0.5
    ):
        """
        Initialize multi-task wrapper.
        
        Args:
            base_model: The base Llama transformer model
            dim: Model dimension (required if base_model doesn't have .dim attribute)
            vocab_size: Vocabulary size (required if base_model doesn't have .vocab_size attribute)
            coherence_hidden_dim: Hidden dimension for coherence detection MLP
            enable_cross_ref: Whether to enable cross-reference task
            enable_paraphrase: Whether to enable paraphrase task
            cross_ref_margin: Margin for triplet loss in cross-reference task
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
        
        # Task flags
        self.enable_cross_ref = enable_cross_ref
        self.enable_paraphrase = enable_paraphrase
        
        # Cross-reference margin
        self.cross_ref_margin = cross_ref_margin
        
        # ===== Task Head 1: Language Modeling (Standard Causal LM) =====
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)
        # Tie weights with input embeddings (common practice)
        self.lm_head.weight = base_model.tok_embeddings.weight
        
        # Also ensure base model's own output head is tied if it exists
        if hasattr(base_model, 'output'):
            base_model.output.weight = base_model.tok_embeddings.weight
        
        # ===== Task Head 2: Coherence Detection (Binary Classification) =====
        # Given two verse embeddings, predict if they form a coherent sequence
        self.coherence_head = nn.Sequential(
            nn.Linear(self.dim * 2, coherence_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(coherence_hidden_dim, coherence_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(coherence_hidden_dim // 2, 1)  # Binary classification
        )
        
        # ===== Task Head 3: Cross-Reference Projection =====
        # Project verse embeddings to a space where semantic similarity = vector similarity
        if self.enable_cross_ref:
            self.cross_ref_proj = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim)
            )
        
        # ===== Task Head 4: Cross-Lingual Projection =====
        # Project to language-invariant space for paraphrase detection
        if self.enable_paraphrase:
            self.paraphrase_proj = nn.Sequential(
                nn.Linear(self.dim, self.dim),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim)
            )
    
    def encode_verse(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode a verse (or any text) into a fixed-size representation.
        
        Args:
            tokens: Token IDs, shape (batch_size, seq_len)
        
        Returns:
            Verse embedding, shape (batch_size, dim)
        """
        # Pass through transformer (request hidden states)
        h, _ = self.base(tokens, return_hiddens=True)  # Shape: (batch, seq_len, dim)
        
        # Pool to sentence-level (mean pooling over sequence)
        verse_emb = h.mean(dim=1)  # Shape: (batch, dim)
        
        return verse_emb
    
    def forward_lm(self, tokens: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for language modeling task.
        
        Args:
            tokens: Input token IDs, shape (batch, seq_len)
            labels: Target token IDs, shape (batch, seq_len)
        
        Returns:
            (logits, loss)
        """
        # Get hidden states from base model
        h, _ = self.base(tokens, return_hiddens=True)  # Shape: (batch, seq_len, dim)
        
        # Project to vocabulary
        logits = self.lm_head(h)  # Shape: (batch, seq_len, vocab_size)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100  # Ignore padding tokens
        )
        
        return loss, logits
    
    def forward_coherence(
        self, 
        verse1_tokens: torch.Tensor, 
        verse2_tokens: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for coherence detection task.
        
        Args:
            verse1_tokens: First verse tokens, shape (batch, seq_len)
            verse2_tokens: Second verse tokens, shape (batch, seq_len)
            labels: Binary labels (1 = coherent, 0 = incoherent), shape (batch,)
        
        Returns:
            (logits, loss)
        """
        # Encode both verses
        verse1_emb = self.encode_verse(verse1_tokens)  # (batch, dim)
        verse2_emb = self.encode_verse(verse2_tokens)  # (batch, dim)
        
        # Concatenate embeddings
        combined = torch.cat([verse1_emb, verse2_emb], dim=-1)  # (batch, dim*2)
        
        # Pass through coherence head
        logits = self.coherence_head(combined).squeeze(-1)  # (batch,)
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        return loss, logits
    
    def forward_cross_ref(
        self,
        anchor_tokens: torch.Tensor,
        positive_tokens: torch.Tensor,
        negative_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-reference prediction task using triplet loss.
        
        Args:
            anchor_tokens: Anchor verse, shape (batch, seq_len)
            positive_tokens: Related verse (cross-reference), shape (batch, seq_len)
            negative_tokens: Unrelated verse, shape (batch, seq_len)
        
        Returns:
            (anchor_emb, positive_emb, negative_emb, loss)
        """
        if not self.enable_cross_ref:
            raise ValueError("Cross-reference task not enabled")
        
        # Encode all three verses
        anchor_emb = self.cross_ref_proj(self.encode_verse(anchor_tokens))
        positive_emb = self.cross_ref_proj(self.encode_verse(positive_tokens))
        negative_emb = self.cross_ref_proj(self.encode_verse(negative_tokens))
        
        # Compute distances
        d_pos = F.cosine_similarity(anchor_emb, positive_emb, dim=-1)
        d_neg = F.cosine_similarity(anchor_emb, negative_emb, dim=-1)
        
        # Triplet loss: encourage positive closer than negative
        # We want d_pos > d_neg + margin
        loss = F.relu(d_neg - d_pos + self.cross_ref_margin).mean()
        
        return loss, (anchor_emb, positive_emb, negative_emb)
    
    def forward_paraphrase(
        self,
        verse1_tokens: torch.Tensor,
        verse2_tokens: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-lingual paraphrase detection.
        
        Same verse in different languages should have similar embeddings.
        
        Args:
            verse1_tokens: Verse in language 1, shape (batch, seq_len)
            verse2_tokens: Verse in language 2, shape (batch, seq_len)
            labels: Binary labels (1 = paraphrase, 0 = different), shape (batch,)
        
        Returns:
            (similarity, loss)
        """
        if not self.enable_paraphrase:
            raise ValueError("Paraphrase task not enabled")
        
        # Encode both verses and project to language-invariant space
        verse1_emb = self.paraphrase_proj(self.encode_verse(verse1_tokens))
        verse2_emb = self.paraphrase_proj(self.encode_verse(verse2_tokens))
        
        # Cosine similarity
        similarity = F.cosine_similarity(verse1_emb, verse2_emb, dim=-1)
        
        # Binary classification: similar (1) or different (0)
        # Map cosine similarity [-1, 1] to probability [0, 1]
        # Use sigmoid on scaled similarity
        logits = similarity * 5.0  # Scale for sharper distinction
        
        # BCE loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        return loss, similarity
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Unified forward pass that routes to appropriate task head.
        
        Args:
            batch: Dictionary with keys:
                - 'task': Task name ('lm', 'coherence', 'cross_ref', 'paraphrase')
                - ... task-specific inputs
        
        Returns:
            (outputs, loss, task_name)
        """
        task = batch['task']
        
        if task == 'lm':
            loss, outputs = self.forward_lm(batch['tokens'], batch['labels'])
            return outputs, loss, 'lm'
        
        elif task == 'coherence':
            loss, outputs = self.forward_coherence(
                batch['verse1_tokens'],
                batch['verse2_tokens'],
                batch['labels']
            )
            return outputs, loss, 'coherence'
        
        elif task == 'cross_ref':
            loss, outputs = self.forward_cross_ref(
                batch['anchor_tokens'],
                batch['positive_tokens'],
                batch['negative_tokens']
            )
            # Note: outputs is a tuple (anchor_emb, pos_emb, neg_emb)
            return outputs, loss, 'cross_ref'
        
        elif task == 'paraphrase':
            loss, outputs = self.forward_paraphrase(
                batch['verse1_tokens'],
                batch['verse2_tokens'],
                batch['labels']
            )
            return outputs, loss, 'paraphrase'
        
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def get_num_params(self) -> int:
        """Return total number of parameters across all heads."""
        return sum(p.numel() for p in self.parameters())


def create_multi_task_model(
    base_model: nn.Module,
    config: Optional[Dict] = None
) -> MultiTaskLlama:
    """
    Factory function to create multi-task model with sensible defaults.
    
    Args:
        base_model: Base Llama transformer
        config: Optional configuration dictionary
    
    Returns:
        MultiTaskLlama model
    """
    if config is None:
        config = {}
    
    return MultiTaskLlama(
        base_model=base_model,
        coherence_hidden_dim=config.get('coherence_hidden_dim', 1024),
        enable_cross_ref=config.get('enable_cross_ref', True),
        enable_paraphrase=config.get('enable_paraphrase', True),
        cross_ref_margin=config.get('cross_ref_margin', 0.5)
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
