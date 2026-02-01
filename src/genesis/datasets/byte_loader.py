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
        self._get_coherence_sample = self._sample_coherence
        self._get_cross_ref_sample = self._sample_cross_ref
        self._get_paraphrase_sample = self._sample_paraphrase

    def _sample_lm(self):
        """Sample batch for Language Modeling."""
        # Standard causal LM: tokens and labels (shifted)
        indices = self.rng.randint(0, len(self.verse_indices), size=self.batch_size)
        
        batch_tokens = torch.zeros((self.batch_size, self.max_seq_len), dtype=torch.long, device=self.device)
        batch_labels = torch.full((self.batch_size, self.max_seq_len), -100, dtype=torch.long, device=self.device)
        
        for i, idx in enumerate(indices):
            start = self.verse_starts[idx]
            length = min(self.verse_lens[idx], self.max_seq_len)
            
            chunk = self.data_tensor[start : start + length]
            batch_tokens[i, :length] = chunk
            
            if length > 1:
                batch_labels[i, :length-1] = chunk[1:]
                # We could set the last token label to something, but -100 is safe
                
        return {"task": "lm", "tokens": batch_tokens, "labels": batch_labels}

    def _sample_coherence(self):
        """Sample batch for Coherence detection."""
        is_coherent = self.rng.rand(self.batch_size) < 0.5
        
        v1_batch = torch.zeros((self.batch_size, self.max_seq_len), dtype=torch.long, device=self.device)
        v2_batch = torch.zeros((self.batch_size, self.max_seq_len), dtype=torch.long, device=self.device)
        labels = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        
        for i in range(self.batch_size):
            if is_coherent[i]:
                idx1 = self.rng.randint(0, len(self.verse_indices) - 1)
                idx2 = idx1 + 1
                labels[i] = 1.0
            else:
                idx1, idx2 = self.rng.choice(len(self.verse_indices), size=2, replace=False)
                labels[i] = 0.0
                
            l1 = min(self.verse_lens[idx1], self.max_seq_len)
            l2 = min(self.verse_lens[idx2], self.max_seq_len)
            
            v1_batch[i, :l1] = self.data_tensor[self.verse_starts[idx1] : self.verse_starts[idx1] + l1]
            v2_batch[i, :l2] = self.data_tensor[self.verse_starts[idx2] : self.verse_starts[idx2] + l2]
            
        return {"task": "coherence", "verse1_tokens": v1_batch, "verse2_tokens": v2_batch, "labels": labels}

    def _sample_cross_ref(self):
        """Sample batch for Cross-Reference prediction."""
        anchor_batch = torch.zeros((self.batch_size, self.max_seq_len), dtype=torch.long, device=self.device)
        pos_batch = torch.zeros((self.batch_size, self.max_seq_len), dtype=torch.long, device=self.device)
        neg_batch = torch.zeros((self.batch_size, self.max_seq_len), dtype=torch.long, device=self.device)
        
        for i in range(self.batch_size):
            a_idx = self.rng.randint(0, len(self.verse_indices))
            p_idx = max(0, min(len(self.verse_indices) - 1, a_idx + self.rng.randint(-5, 6)))
            n_idx = self.rng.randint(0, len(self.verse_indices))
            while abs(n_idx - a_idx) < 100:
                n_idx = self.rng.randint(0, len(self.verse_indices))
                
            la = min(self.verse_lens[a_idx], self.max_seq_len)
            lp = min(self.verse_lens[p_idx], self.max_seq_len)
            ln = min(self.verse_lens[n_idx], self.max_seq_len)
            
            anchor_batch[i, :la] = self.data_tensor[self.verse_starts[a_idx] : self.verse_starts[a_idx] + la]
            pos_batch[i, :lp] = self.data_tensor[self.verse_starts[p_idx] : self.verse_starts[p_idx] + lp]
            neg_batch[i, :ln] = self.data_tensor[self.verse_starts[n_idx] : self.verse_starts[n_idx] + ln]
            
        return {"task": "cross_ref", "anchor_tokens": anchor_batch, "positive_tokens": pos_batch, "negative_tokens": neg_batch}

    def _sample_paraphrase(self):
        """Sample batch for Cross-Lingual Paraphrase."""
        locales = list(self.locale_map.keys())
        if len(locales) < 2:
            return self._sample_coherence()
            
        is_para = self.rng.rand(self.batch_size) < 0.5
        v1_batch = torch.zeros((self.batch_size, self.max_seq_len), dtype=torch.long, device=self.device)
        v2_batch = torch.zeros((self.batch_size, self.max_seq_len), dtype=torch.long, device=self.device)
        labels = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        
        for i in range(self.batch_size):
            lang1, lang2 = self.rng.choice(locales, size=2, replace=False)
            idx_list1 = self.locale_map[lang1]
            idx_list2 = self.locale_map[lang2]
            
            if is_para[i]:
                common_len = min(len(idx_list1), len(idx_list2))
                k = self.rng.randint(0, common_len)
                idx1, idx2 = idx_list1[k], idx_list2[k]
                labels[i] = 1.0
            else:
                k1, k2 = self.rng.randint(0, len(idx_list1)), self.rng.randint(0, len(idx_list2))
                while k1 == k2: k2 = self.rng.randint(0, len(idx_list2))
                idx1, idx2 = idx_list1[k1], idx_list2[k2]
                labels[i] = 0.0
                
            l1, l2 = min(self.verse_lens[idx1], self.max_seq_len), min(self.verse_lens[idx2], self.max_seq_len)
            v1_batch[i, :l1] = self.data_tensor[self.verse_starts[idx1] : self.verse_starts[idx1] + l1]
            v2_batch[i, :l2] = self.data_tensor[self.verse_starts[idx2] : self.verse_starts[idx2] + l2]
            
        return {"task": "paraphrase", "verse1_tokens": v1_batch, "verse2_tokens": v2_batch, "labels": labels}

    def __iter__(self):
        while True:
            task = self.rng.choice(self.tasks, p=self.task_probs)
            if task == 'lm': yield self._sample_lm()
            elif task == 'coherence': yield self._sample_coherence()
            elif task == 'cross_ref': yield self._sample_cross_ref()
            elif task == 'paraphrase': yield self._sample_paraphrase()
