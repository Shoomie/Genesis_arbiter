from tokenizers import Tokenizer
import torch
import json
import unicodedata


class ByteTokenizer:
    """
    Byte-level UTF-8 tokenizer.
    Uses exactly 256 raw bytes (0-255) + 4 special tokens.
    """
    def __init__(self):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        
        # Bytes are offset by 4
        self.byte_offset = 4
        self.vocab_size = 256 + self.byte_offset
        self.mask_id = self.pad_id # Use pad for masking in byte-level
        
    def encode(self, text, add_special_tokens=False):
        """Encode text to byte-level token IDs."""
        byte_data = text.encode('utf-8')
        token_ids = [b + self.byte_offset for b in byte_data]
        
        if add_special_tokens:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
            
        return token_ids

    def decode(self, ids, skip_special_tokens=True):
        """Decode token IDs back to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        
        byte_list = []
        for tid in ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid >= self.byte_offset:
                byte_list.append(tid - self.byte_offset)
            elif tid == self.unk_id and not skip_special_tokens:
                 byte_list.append(0)
                 
        try:
            return bytes(byte_list).decode('utf-8', errors='replace')
        except Exception:
            return "[Decode Error]"

    def __call__(self, text, truncation=True, max_length=1024, add_special_tokens=False):
        if isinstance(text, str):
            tokens = self.encode(text, add_special_tokens=add_special_tokens)
            if truncation:
                tokens = tokens[:max_length]
            return torch.tensor([tokens])
        else:
            all_tokens = []
            for t in text:
                tokens = self.encode(t, add_special_tokens=add_special_tokens)
                if truncation:
                    tokens = tokens[:max_length]
                all_tokens.append(tokens)
            
            max_len = max(len(t) for t in all_tokens)
            padded = []
            for tokens in all_tokens:
                padded.append(tokens + [self.pad_id] * (max_len - len(tokens)))
            
            return torch.tensor(padded)


class GenesisTokenizer:
    """
    Simplified tokenizer wrapper that enforces pure byte-level tokenization.
    Vocabulary is fixed at 260 (256 bytes + 4 special tokens).
    """
    
    def __init__(self, tokenizer_path=None, type='byte'):
        # We now ignore the tokenizer_path and force byte-level
        self.tokenizer = ByteTokenizer()
        self.tokenizer_type = 'byte'
        self.vocab_size = self.tokenizer.vocab_size
        self.mask_id = self.tokenizer.mask_id
        self.bos_id = self.tokenizer.bos_id
        self.eos_id = self.tokenizer.eos_id
        self.pad_id = self.tokenizer.pad_id
        self.unk_id = self.tokenizer.unk_id
        
        if tokenizer_path:
            print(f"[INFO] Ignoring tokenizer_path '{tokenizer_path}'. Using pure byte-level mode.")

    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, ids, skip_special_tokens=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
    
    def __call__(self, text, truncation=True, max_length=1024, add_special_tokens=False):
        return self.tokenizer(text, truncation=truncation, max_length=max_length, add_special_tokens=add_special_tokens)
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def is_character_level(self):
        return False

    def is_byte_level(self):
        return True
