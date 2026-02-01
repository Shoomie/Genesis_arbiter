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
        """Vectorized sample for Language Modeling with Dense Packing."""
        # 1. Randomly pick batch_size start points anywhere in the tensor
        # Subtract max_seq_len + 1 to ensure we have a full sequence + 1 for label shift
        max_start = len(self.data_tensor) - self.max_seq_len - 1
        starts = torch.randint(0, max_start, (self.batch_size,), device=self.device)
        
        # 2. Generate full index matrix [B, L]
        indices = starts.unsqueeze(1) + self.grid
        
        # 3. Gather tokens
        batch_tokens = self.data_tensor[indices]
        
        # 4. Generate labels (shifted tokens)
        label_indices = indices + 1
        batch_labels = self.data_tensor[label_indices]
        
        return {"task": "lm", "tokens": batch_tokens, "labels": batch_labels}

    def __iter__(self):
        while True:
            yield self._sample_lm()

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

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
