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
        
        print(f"[OK] Initialized Byte-Level UTF-8 Tokenizer (vocab size: {self.vocab_size})")

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
            # UNK or other specials (if not skipped) are handled as 0 byte or ignored
            elif tid == self.unk_id and not skip_special_tokens:
                 byte_list.append(0) # Minimal fallback
                 
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

    def get_vocab_size(self):
        return self.vocab_size

    def is_character_level(self):
        return False

    def is_byte_level(self):
        return True


class GenesisTokenizer:
    """
    Unified tokenizer wrapper supporting BPE, character-level, and byte-level tokenization.
    Automatically detects tokenizer type.
    """
    
    def __init__(self, tokenizer_path=None, type=None):
        self.tokenizer_path = tokenizer_path
        self.tokenizer_type = type
        if type == 'byte':
            self.tokenizer = ByteTokenizer()
            self.tokenizer_type = 'byte'
            self.vocab_size = self.tokenizer.vocab_size
            self.mask_id = self.tokenizer.mask_id
            self.bos_id = self.tokenizer.bos_id
            self.eos_id = self.tokenizer.eos_id
            self.pad_id = self.tokenizer.pad_id
            self.unk_id = self.tokenizer.unk_id
            return

        if not tokenizer_path:
            # Default to byte if no path provided
            self.tokenizer = ByteTokenizer()
            self.tokenizer_type = 'byte'
            self.vocab_size = self.tokenizer.vocab_size
            self.mask_id = self.tokenizer.mask_id
            self.bos_id = self.tokenizer.bos_id
            self.eos_id = self.tokenizer.eos_id
            self.pad_id = self.tokenizer.pad_id
            self.unk_id = self.tokenizer.unk_id
            return
            
        # Load and detect tokenizer type from file
        try:
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if config.get('type') == 'ByteLevel' or config.get('model', {}).get('type') == 'ByteLevel':
                self.tokenizer_type = 'byte'
                self.tokenizer = ByteTokenizer()
                self.vocab_size = self.tokenizer.vocab_size
                self.mask_id = self.tokenizer.mask_id
                self.bos_id = self.tokenizer.bos_id
                self.eos_id = self.tokenizer.eos_id
                self.pad_id = self.tokenizer.pad_id
                self.unk_id = self.tokenizer.unk_id
            elif config.get('type') == 'CharacterLevel' or config.get('model', {}).get('type') == 'CharacterLevel':
                self.tokenizer_type = 'character'
                self._init_char_tokenizer(config)
            else:
                self.tokenizer_type = 'bpe'
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
                self.vocab_size = self.tokenizer.get_vocab_size()
        except Exception as e:
            print(f"[WARN] Failed to load tokenizer from {tokenizer_path}, defaulting to ByteTokenizer: {e}")
            self.tokenizer = ByteTokenizer()
            self.tokenizer_type = 'byte'
            self.vocab_size = self.tokenizer.vocab_size
            self.mask_id = self.tokenizer.mask_id
    
    def _init_char_tokenizer(self, config):
        """Initialize character-level tokenizer from config."""
        vocab_data = config.get('model', {}).get('vocab', {})
        self.char_to_id = vocab_data
        self.id_to_char = {v: k for k, v in vocab_data.items()}
        self.vocab_size = len(vocab_data)
        
        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = config.get('model', {}).get('unk_token', '<unk>')
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        
        self.pad_id = self.char_to_id.get(self.pad_token,0)
        self.unk_id = self.char_to_id.get(self.unk_token, 1)
        self.bos_id = self.char_to_id.get(self.bos_token, 2)
        self.eos_id = self.char_to_id.get(self.eos_token, 3)
        
        print(f"[OK] Loaded character-level tokenizer (vocab size: {self.vocab_size})")
    
    def encode(self, text, add_special_tokens=False, normalize=True):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Add BOS/EOS tokens
            normalize: Apply Unicode normalization (char-level only)
            
        Returns:
            List of token IDs
        """
        if self.tokenizer_type == 'byte':
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        elif self.tokenizer_type == 'bpe':
            return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
        else:
            # Character-level encoding
            if normalize:
                text = unicodedata.normalize('NFKC', text)
            
            token_ids = []
            
            if add_special_tokens:
                token_ids.append(self.bos_id)
            
            for char in text:
                token_id = self.char_to_id.get(char, self.unk_id)
                token_ids.append(token_id)
            
            if add_special_tokens:
                token_ids.append(self.eos_id)
            
            return token_ids
    
    def decode(self, ids, skip_special_tokens=True):
        """
        Decode token IDs to text.
        
        Args:
            ids: List or tensor of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text string
        """
        if self.tokenizer_type == 'byte':
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        elif self.tokenizer_type == 'bpe':
            # Convert tensor to list if needed
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        else:
            # Character-level decoding
            # Convert tensor to list if needed
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            special_token_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            
            chars = []
            for token_id in ids:
                if skip_special_tokens and token_id in special_token_ids:
                    continue
                chars.append(self.id_to_char.get(token_id, self.unk_token))
            
            return ''.join(chars)
    
    def __call__(self, text, truncation=True, max_length=1024, add_special_tokens=False):
        """
        Encode text and return as tensor (for model compatibility).
        
        Args:
            text: Input text string or list of strings
            truncation: Truncate to max_length
            max_length: Maximum sequence length
            add_special_tokens: Add BOS/EOS tokens
            
        Returns:
            Tensor of token IDs
        """
        if self.tokenizer_type == 'byte':
            return self.tokenizer(text, truncation=truncation, max_length=max_length, add_special_tokens=add_special_tokens)
        
        if isinstance(text, str):
            tokens = self.encode(text, add_special_tokens=add_special_tokens)
            if truncation:
                tokens = tokens[:max_length]
            return torch.tensor([tokens])
        else:
            # Batch encoding
            all_tokens = []
            for t in text:
                tokens = self.encode(t, add_special_tokens=add_special_tokens)
                if truncation:
                    tokens = tokens[:max_length]
                all_tokens.append(tokens)
            
            # Pad to same length
            max_len = max(len(t) for t in all_tokens)
            padded = []
            for tokens in all_tokens:
                # Use getattr for pad_id as it might not be directly on GenesisTokenizer for BPE
                padded.append(tokens + [getattr(self, 'pad_id', 0)] * (max_len - len(tokens)))
            
            return torch.tensor(padded)
    
    def get_vocab_size(self):
        """Get vocabulary size."""
        return self.vocab_size
    
    def is_character_level(self):
        """Check if tokenizer is character-level."""
        return self.tokenizer_type == 'character'

    def is_byte_level(self):
        """Check if tokenizer is byte-level."""
        return self.tokenizer_type == 'byte'

    def compress_vocab(self, active_token_ids):
        """
        Compress vocabulary to only include active tokens.
        
        Args:
            active_token_ids (set/list/tensor): IDs of tokens present in the dataset.
            
        Returns:
            mapping_table (torch.Tensor): A tensor for efficient remapping (old_id -> new_id).
                                        Size is [original_vocab_size].
        """
        if self.tokenizer_type != 'character':
            print("[WARN] Vocab compression only supported for character-level tokenizers.")
            return None
        
        # Ensure special tokens are always included
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        
        # Convert input to set of ints
        if isinstance(active_token_ids, torch.Tensor):
            active_set = set(active_token_ids.tolist())
        else:
            active_set = set(active_token_ids)
            
        # Merge
        active_set.update(special_ids)
        
        # Create new mapping
        # We want to preserve special tokens at 0, 1, 2, 3 if possible, or at least deterministic
        sorted_active = sorted(list(active_set))
        
        # Priority: special tokens first (0,1,2,3) to match established conventions if possible
        # Actually, self.pad_id etc are fixed in _init_char_tokenizer.
        # If we change IDs, we MUST update self.pad_id etc.
        
        new_id_to_char = {}
        new_char_to_id = {}
        old_to_new = {}
        
        current_new_id = 0
        
        # 1. Map Special Tokens (force them to be 0,1,2,3 for sanity)
        # Assuming original special IDs are small. If not, we just re-assign.
        specials_ordered = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        for token in specials_ordered:
            old_id = self.char_to_id.get(token)
            if old_id is not None:
                new_char_to_id[token] = current_new_id
                new_id_to_char[current_new_id] = token
                old_to_new[old_id] = current_new_id
                current_new_id += 1
                
        # 2. Map remaining active tokens
        for old_id in sorted_active:
            if old_id in old_to_new:
                continue # Already handled (special)
                
            char = self.id_to_char.get(old_id)
            if char:
                new_char_to_id[char] = current_new_id
                new_id_to_char[current_new_id] = char
                old_to_new[old_id] = current_new_id
                current_new_id += 1
                
        # Update internal state
        print(f"  [Tokenizer] Compressing vocab: {self.vocab_size} -> {len(new_char_to_id)}")
        self.char_to_id = new_char_to_id
        self.id_to_char = new_id_to_char
        self.vocab_size = len(new_char_to_id)
        
        # Update special token IDs
        self.pad_id = self.char_to_id.get(self.pad_token, 0)
        self.unk_id = self.char_to_id.get(self.unk_token, 1)
        self.bos_id = self.char_to_id.get(self.bos_token, 2)
        self.eos_id = self.char_to_id.get(self.eos_token, 3)
        
        # Create lookup tensor for fast remapping
        # Max old ID could be anything.
        max_old_id = max(active_set) if active_set else 0
        # Safe upper bound: max(original vocab size, max_old_id)
        table_size = max(len(old_to_new), max_old_id + 1)
        
        mapping_table = torch.zeros(table_size, dtype=torch.long)
        # Default unmapped to UNK? Or keep as is?
        # If dataset says token X exists but we mapped it to nothing, that's error.
        # But we constructed valid map for ALL active tokens.
        # Any token NOT in active_set should effectively map to UNK.
        mapping_table.fill_(self.unk_id) 
        
        for old, new in old_to_new.items():
            mapping_table[old] = new
            
        return mapping_table
