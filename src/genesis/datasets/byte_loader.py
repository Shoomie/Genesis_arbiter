import torch
import numpy as np
from typing import Dict, List, Optional
import time

class InfiniteGPULoader:
    """
    High-throughput, GPU-resident data loader.
    Loads the entire tokenized dataset into VRAM once and samples infinitely.
    """
    def __init__(
        self,
        data_tensor: torch.Tensor,
        verse_indices: List[tuple],
        locale_map: Dict[str, List[int]],
        batch_size: int,
        max_seq_len: int,
        task_distribution: Dict[str, float],
        device: torch.device,
        seed: int = 42
    ):
        self.data_tensor = data_tensor.to(device, non_blocking=True)
        self.verse_indices = verse_indices
        self.locale_map = locale_map
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.task_distribution = task_distribution
        self.tasks = list(task_distribution.keys())
        self.task_probs = np.array([task_distribution[t] for t in self.tasks])
        self.device = device
        self.rng = np.random.RandomState(seed)
        
        # Pre-convert verse_indices to tensor for faster access on GPU if possible
        # verse_indices is list of (start, len)
        self.verse_starts = torch.tensor([v[0] for v in verse_indices], device=device, dtype=torch.long)
        self.verse_lens = torch.tensor([v[1] for v in verse_indices], device=device, dtype=torch.long)
        
        print(f"[OK] GPU DataLoader initialized on {device}")
        print(f"     Dataset size: {len(data_tensor)} tokens, {len(verse_indices)} verses")
        
        # Compatibility aliases for ProceduralEvaluator
        self.locale_verse_map = self.locale_map

        # Pre-compute grid for vectorization
        self.grid = torch.arange(max_seq_len, device=device).unsqueeze(0) # [1, max_seq_len]

    def _sample_lm(self):
        """Vectorized sample for Language Modeling."""
        # 1. Randomly pick batch_size verse indices on GPU
        verse_idxs = torch.randint(0, len(self.verse_starts), (self.batch_size,), device=self.device)
        starts = self.verse_starts[verse_idxs]
        lens = self.verse_lens[verse_idxs]
        
        # 2. Generate full index matrix [B, L]
        # Clamp to prevent OOB at the very end of data_tensor
        max_idx = len(self.data_tensor) - 1
        indices = torch.clamp(starts.unsqueeze(1) + self.grid, 0, max_idx)
        
        # 3. Gather tokens
        batch_tokens = self.data_tensor[indices]
        
        # 4. Mask labels (shifted)
        # mask is True where we are inside the verse bounds
        mask = self.grid < lens.unsqueeze(1)
        
        batch_labels = torch.full_like(batch_tokens, -100)
        # Shift tokens for labels: labels[i] = tokens[i+1]
        # Only valid if the next token is ALSO within the verse mask
        valid_label_mask = mask[:, 1:] 
        batch_labels[:, :-1] = torch.where(valid_label_mask, batch_tokens[:, 1:], -100)
        
        # Final cleanup: zerout tokens outside verse (optional but cleaner)
        batch_tokens = torch.where(mask, batch_tokens, 0)
        
        return {"task": "lm", "tokens": batch_tokens, "labels": batch_labels}

    def __iter__(self):
        while True:
            yield self._sample_lm()
