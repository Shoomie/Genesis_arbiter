import torch
from torch.utils.data import Dataset, DataLoader
import os

class BibleDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus not found at {corpus_path}")
            
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        if hasattr(tokenizer, 'is_character_level') and tokenizer.is_character_level():
            # Character-level
            self.tokens = tokenizer.encode(text)
        else:
            # BPE
            self.tokens = tokenizer.tokenizer.encode(text).ids
        
        print(f"Tokenization complete. Total tokens: {len(self.tokens)}")
        
        # Calculate number of samples
        self.num_samples = (len(self.tokens) - 1) // max_seq_len
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.max_seq_len
        end_idx = start_idx + self.max_seq_len
        
        # Input tokens
        x = torch.tensor(self.tokens[start_idx:end_idx], dtype=torch.long)
        # Target tokens (shifted by 1)
        y = torch.tensor(self.tokens[start_idx+1:end_idx+1], dtype=torch.long)
        
        return {
            "input_ids": x,
            "labels": y
        }

def get_bible_dataloader(corpus_path, tokenizer, batch_size, max_seq_len, world_size=1, rank=0):
    dataset = BibleDataset(corpus_path, tokenizer, max_seq_len)
    
    sampler = None
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
