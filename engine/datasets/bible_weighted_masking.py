import torch
from torch.utils.data import Dataset
import random
import numpy as np
import os

class WeightedMaskingDataset(Dataset):
    """BibleDataset with difficulty-weighted masking of logical connectives."""
    
    # Difficulty weights based on reasoning depth required
    CONNECTIVE_WEIGHTS = {
        # High difficulty: Require understanding multi-verse arguments
        "therefore": 1.0,
        "Therefore": 1.0,
        "thus": 0.95,
        "Thus": 0.95,
        "consequently": 0.9,
        "Consequently": 0.9,
        "accordingly": 0.9,
        "Accordingly": 0.9,
        
        # Medium difficulty: Local causal reasoning
        "because": 0.7,
        "Because": 0.7,
        "for": 0.6,
        "For": 0.6,
        "since": 0.65,
        "Since": 0.65,
        "so": 0.5,
        "So": 0.5,
        
        # Lower difficulty: Local contrast/addition
        "but": 0.3,
        "But": 0.3,
        "however": 0.35,
        "However": 0.35,
        "yet": 0.3,
        "Yet": 0.3,
        "moreover": 0.4,
        "Moreover": 0.4,
        "furthermore": 0.4,
        "Furthermore": 0.4,
        "although": 0.45,
        "Although": 0.45,
    }
    
    def __init__(self, corpus_path, tokenizer, max_seq_len=1024, base_prob=0.4):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.base_prob = base_prob
        
        # Build token ID → weight mapping
        self.token_weights = {}
        self.word_to_id = {}  # For debugging
        
        for word, weight in self.CONNECTIVE_WEIGHTS.items():
            token_id = tokenizer.tokenizer.token_to_id(word)
            if token_id is not None:
                self.token_weights[token_id] = weight
                self.word_to_id[word] = token_id
        
        print(f"[WeightedMasking] Loaded {len(self.token_weights)} connective token IDs")
        if len(self.token_weights) > 0:
            print(f"[WeightedMasking] Weight range: {min(self.token_weights.values()):.2f} - {max(self.token_weights.values()):.2f}")
            print(f"[WeightedMasking] Sample mappings:")
            for word in ["therefore", "because", "but", "so"]:
                if word in self.word_to_id:
                    tid = self.word_to_id[word]
                    print(f"  '{word}' -> ID {tid}, weight {self.token_weights[tid]:.2f}")
        
        # Load corpus
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus not found at {corpus_path}")
        
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        print(f"[WeightedMasking] Tokenizing corpus from {corpus_path}...")
        self.tokens = np.array(tokenizer.tokenizer.encode(text).ids, dtype=np.int32)
        print(f"[WeightedMasking] Tokenization complete. Total tokens: {len(self.tokens)}")
        
        # Count connectives for statistics
        connective_count = sum(1 for t in self.tokens if t in self.token_weights)
        pct = 100 * connective_count / len(self.tokens) if len(self.tokens) > 0 else 0
        print(f"[WeightedMasking] Connectives in corpus: {connective_count} (~{pct:.2f}%)")
        
        # Calculate number of samples
        self.num_samples = (len(self.tokens) - 1) // max_seq_len
        
        # Get mask token ID (or use 0 as fallback)
        self.mask_token_id = tokenizer.tokenizer.token_to_id("[MASK]")
        if self.mask_token_id is None:
            # If no [MASK] token, use token ID 0 (usually padding/unknown)
            self.mask_token_id = 0
            print(f"[WeightedMasking] Warning: No [MASK] token found, using ID 0")
        else:
            print(f"[WeightedMasking] Using [MASK] token ID: {self.mask_token_id}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.max_seq_len
        end_idx = start_idx + self.max_seq_len
        
        # Get sequence (ensure we have enough tokens)
        if end_idx >= len(self.tokens):
            end_idx = len(self.tokens) - 1
        
        x = self.tokens[start_idx:end_idx].copy()
        y = self.tokens[start_idx+1:end_idx+1].copy()
        
        # Apply weighted masking
        masked_count = 0
        for i in range(len(x)):
            if x[i] in self.token_weights:
                weight = self.token_weights[x[i]]
                mask_prob = self.base_prob * weight
                
                if random.random() < mask_prob:
                    x[i] = self.mask_token_id
                    masked_count += 1
        
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def get_bible_weighted_dataloader(corpus_path, tokenizer, batch_size, max_seq_len, 
                                   base_prob=0.4, world_size=1, rank=0):
    """Create DataLoader with weighted masking."""
    dataset = WeightedMaskingDataset(corpus_path, tokenizer, max_seq_len, base_prob)
    
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
