from tokenizers import Tokenizer
import torch

class GenesisTokenizer:
    def __init__(self, tokenizer_path):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        
    def encode(self, text, add_special_tokens=True):
        # Placeholder for complex encoding logic if needed
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def __call__(self, text, truncation=True, max_length=1024):
        tokens = self.tokenizer.encode(text).ids
        if truncation:
            tokens = tokens[:max_length]
        return torch.tensor([tokens])
