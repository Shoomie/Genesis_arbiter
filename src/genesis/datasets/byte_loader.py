import torch
import numpy as np
from typing import Dict, List, Optional
import time
import threading
import queue

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
        boundary_tensor: Optional[torch.Tensor] = None,
        seed: int = 42
    ):
        self.device = device
        self.data_tensor = data_tensor.to(device, non_blocking=True)
        self.boundary_tensor = boundary_tensor.to(device, non_blocking=True) if boundary_tensor is not None else None
        self.verse_indices = verse_indices
        self.locale_map = locale_map # locale -> List[verse_indices]
        
        # Pre-compute locale ID tensor for telemetry
        # This maps every token index to a language index
        self.locales = sorted(list(locale_map.keys()))
        self.locale_to_id = {lang: i for i, lang in enumerate(self.locales)}
        
        token_locales = torch.zeros(len(data_tensor), dtype=torch.uint8, device=device)
        for lang, v_indices in locale_map.items():
            l_id = self.locale_to_id[lang]
            for v_idx in v_indices:
                start, length = verse_indices[v_idx]
                token_locales[start : start + length] = l_id
        self.locale_tensor = token_locales

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.task_distribution = task_distribution
        self.tasks = list(task_distribution.keys())
        self.task_probs = np.array([task_distribution[t] for t in self.tasks])
        self.rng = np.random.RandomState(seed)
        
        # Pre-convert verse_indices to tensor for faster access on GPU
        self.verse_starts = torch.tensor([v[0] for v in verse_indices], device=device, dtype=torch.long)
        self.verse_lens = torch.tensor([v[1] for v in verse_indices], device=device, dtype=torch.long)
        
        print(f"[OK] GPU DataLoader initialized on {device}")
        if self.boundary_tensor is not None:
            print(f"     Whole-Word Masking (WWM) enabled via boundary map.")
        
        # Compatibility aliases for ProceduralEvaluator
        self.locale_verse_map = self.locale_map

        # Pre-compute grid for vectorization
        self.grid = torch.arange(max_seq_len, device=device).unsqueeze(0) # [1, max_seq_len]
        
        # Masking State (Controlled by Trainer)
        self.mask_prob = 0.0
        self.use_wwm = False
        self.use_span = False
        self.span_range = (1, 1) # (min, max)

    def _sample_lm(self, mask_prob: Optional[float] = None, use_wwm: Optional[bool] = None, starts: Optional[torch.Tensor] = None):
        """Vectorized sample for Language Modeling with optional Masking."""
        # Use internal state if not provided
        m_prob = mask_prob if mask_prob is not None else self.mask_prob
        u_wwm = use_wwm if use_wwm is not None else self.use_wwm
        
        # 1. pick batch_size start points (if not provided)
        if starts is None:
            max_start = len(self.data_tensor) - self.max_seq_len - 1
            starts = torch.randint(0, max_start, (self.batch_size,), device=self.device)
        
        # 2. Generate full index matrix [B, L]
        indices = starts.unsqueeze(1) + self.grid
        
        # 3. Gather tokens
        batch_tokens = self.data_tensor[indices]
        
        # 4. Generate labels (standard shift)
        label_indices = indices + 1
        batch_labels = self.data_tensor[label_indices]
        
        # 5. Optional Masking (for WWM feasibility)
        # In a causal model, this "scrambles" the input to force semantic recovery.
        if m_prob > 0:
            if u_wwm and self.boundary_tensor is not None:
                # Vectorized Whole-Word Masking
                mask_boundaries = self.boundary_tensor[indices]
                # word_ids: [Batch, SeqLen] (0 to ~SeqLen/3)
                word_ids = mask_boundaries.cumsum(dim=1)
                
                # Each word gets a random value.
                word_rand = torch.rand(self.batch_size, self.max_seq_len + 1, device=self.device)
                
                # Optional Phase 3: Span Masking (Word Dilation)
                if self.use_span:
                    min_s, max_s = self.span_range
                    # Sample a span length for this specific batch
                    current_span = int(torch.randint(min_s, max_s + 1, (1,)).item()) if min_s < max_s else min_s
                    
                    # Normalize probability so total density matches m_prob (Intuitive Density)
                    seed_prob = m_prob / max(current_span, 1)
                    word_mask = word_rand < seed_prob
                    
                    if current_span > 1:
                        # Simple dilation: if a word is masked, mask the next N-1 words too.
                        combined = word_mask.clone()
                        for s in range(1, current_span):
                            # Shift word mask to the right and OR
                            shifted = torch.zeros_like(word_mask)
                            shifted[:, s:] = word_mask[:, :-s]
                            combined = combined | shifted
                        word_mask = combined
                else:
                    # Phase 2: Standard Whole-Word Masking (1:1 Density)
                    word_mask = word_rand < m_prob

                # Distribute random values / masks to tokens based on word_ids
                mask = torch.gather(word_mask, 1, word_ids)
            else:
                # Random Byte Masking
                mask = torch.rand(self.batch_size, self.max_seq_len, device=self.device) < m_prob
            
            # Apply mask to input (using pad_id as mask)
            # We skip masking the first token sometimes or just mask everything
            batch_tokens = torch.where(mask, 0, batch_tokens) 
            
        # TELEMETRY: Get most common locale for each sequence in batch
        # shape [B, L] -> [B, L]
        batch_locales = self.locale_tensor[indices]
        # Just take the first token's locale as representative for the whole sequence
        # (Usually accurate enough for Bible verses, which are 50-200 tokens)
        seq_locales = batch_locales[:, 0].tolist()
        
        return {
            "task": "lm", 
            "tokens": batch_tokens, 
            "labels": batch_labels,
            "locales": seq_locales # List of lang IDs [B]
        }

    def __iter__(self):
        while True:
            yield self._sample_lm()

    def sample_locale(self, locale: str, mask_prob: float = 0.0):
        """Sample a batch exclusively from a specific locale."""
        if locale not in self.locale_map:
            return None
        
        # Pick random verses from this locale
        indices = self.locale_map[locale]
        if not indices: return None
        
        chosen_idx = self.rng.choice(indices, size=self.batch_size)
        chosen_idx_t = torch.tensor(chosen_idx, device=self.device)
        
        # Get start positions (with slight random offset to avoid always starting at verse start)
        starts = self.verse_starts[chosen_idx_t]
        v_lens = self.verse_lens[chosen_idx_t]
        
        # For validation, we can just use starts, or add jitter
        # jit = torch.randint(0, (v_lens - self.max_seq_len).clamp(min=1), (self.batch_size,))
        # but let's just keep it simple for now. 
        
        return self._sample_lm(mask_prob=mask_prob, starts=starts)

class BackgroundPrefetcher:
    """
    Overlaps data sampling with model computation using a background thread.
    """
    def __init__(self, loader, buffer_size: int = 4):
        self.loader = loader
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        """Worker thread to pre-sample batches."""
        # Use a new random state per thread if randomization is done in Python,
        # but InfiniteGPULoader uses torch.randint on GPU.
        while not self.stop_event.is_set():
            try:
                # Sample on the background thread
                batch = self.loader._sample_lm()
                # This will block if the queue is full
                self.queue.put(batch, timeout=1.0)
            except queue.Full:
                continue
            except Exception as e:
                print(f"[Prefetcher] Error: {e}")
                time.sleep(0.1)

    def __iter__(self):
        return self

    def __next__(self):
        # Get from queue (blocks if empty)
        return self.queue.get()

    def clear(self):
        """Purge the current queue to avoid phase leakage."""
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
            print(f"  [DATA] Prefetcher queue purged.")
        except Exception:
            pass

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
