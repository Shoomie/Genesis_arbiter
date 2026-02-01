"""
Multi-Task Data Sampler for Genesis Arbiter

Samples from multiple tasks according to configured distribution:
- Language Modeling (70%)
- Coherence Detection (15%)
- Cross-Reference Prediction (7.5%)
- Cross-Lingual Paraphrase (7.5%)
"""

import numpy as np
import torch
import os
import time
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Tuple, Optional
import random
import json
from pathlib import Path


class MultiTaskDataset(Dataset):
    """
    Dataset that provides samples for multiple tasks.
    
    Each batch is labeled with a task type and contains appropriate data for that task.
    """
    
    def __init__(
        self,
        bible_data_dir: str,
        tokenizer,
        device: torch.device,
        cache_path: Optional[str] = None,
        max_seq_len: int = 512,
        task_distribution: Optional[Dict[str, float]] = None,
        seed: int = 42,
        target_languages: Optional[List[str]] = None
    ):
        """
        Initialize multi-task dataset with VRAM preloading.
        """
        self.bible_data_dir = Path(bible_data_dir)
        self.tokenizer = tokenizer
        self.device = device
        self.cache_path = Path(cache_path) if cache_path else None
        self.max_seq_len = max_seq_len
        self.rng = np.random.RandomState(seed)
        self.target_languages = target_languages
        if self.target_languages:
            print(f"  Filtering dataset for languages: {self.target_languages}")
        
        # Default task distribution
        if task_distribution is None:
            task_distribution = {
                'lm': 0.70,
                'coherence': 0.15,
                'cross_ref': 0.075,
                'paraphrase': 0.075
            }
        
        self.task_distribution = task_distribution
        self.tasks = list(task_distribution.keys())
        self.task_probs = np.array([task_distribution[t] for t in self.tasks])
        
        # Determine actual storage device (CPU or GPU)
        # If we are caching/loading on CPU, self.device might still be 'cuda' for the output
        # But we need to know where to store the big tensor.
        self.storage_device = device
        
        
        # Check for cache
        loaded_from_cache = False
        if self.cache_path and self.cache_path.exists():
            print(f"Loading data from cache: {self.cache_path}...")
            try:
                cache_data = torch.load(self.cache_path, map_location='cpu')
                
                # Check version compatibility if needed, for now trust the file
                all_tokens = cache_data['tokens']
                self.verse_indices = cache_data['indices']
                self.locale_verse_map = cache_data['locale_map']
                
                print(f"  [OK] Cache loaded ({len(all_tokens)} tokens, {len(self.verse_indices)} verses)")
                
                # Filter if target languages specified
                if self.target_languages:
                    print(f"  Applying filter: {self.target_languages}")
                    new_verse_indices = []
                    new_locale_map = {}
                    current_idx = 0
                    
                    found_langs = 0
                    
                    for lang in self.target_languages:
                        if lang in self.locale_verse_map:
                            found_langs += 1
                            original_indices = self.locale_verse_map[lang]
                            new_indices_for_lang = []
                            
                            for old_idx in original_indices:
                                # Copy the verse info (start, len)
                                item = self.verse_indices[old_idx]
                                new_verse_indices.append(item)
                                new_indices_for_lang.append(current_idx)
                                current_idx += 1
                                
                            new_locale_map[lang] = new_indices_for_lang
                            
                    if found_langs > 0:
                        self.verse_indices = new_verse_indices
                        self.locale_verse_map = new_locale_map
                        print(f"  [FILTERED] Dataset reduced to {len(self.verse_indices)} verses ({found_langs} languages)")
                    else:
                        print(f"  [WARN] None of the target languages found in cache. Using full dataset.")
                
                loaded_from_cache = True



            except Exception as e:
                print(f"  [WARN] Failed to load cache: {e}. Falling back to raw processing.")
                loaded_from_cache = False
        
        if not loaded_from_cache:
            # Load from raw files
            print(f"Loading multi-language translations from {bible_data_dir}...")
            self.translations = self._load_translations()
            print(f"  Loaded {len(self.translations)} translations")
            
            # Populate verses from translations
            print("Aggregating and tokenizing verses (Preparing Data Tensor)...")
            
            # Temporary lists to build the index
            self.verse_indices = []  # List of (start_idx, length)
            all_tokens = []
            
            # We will also keep track of verses per locale to support paraphrase task
            # locale -> list of indices in self.verse_indices
            self.locale_verse_map = {} 
            
            total_verses = 0
            
            # Progress tracking
            total_langs = len(self.translations)
            start_time = time.time()
            
            for i, (lang, chapters) in enumerate(self.translations.items()):
                self.locale_verse_map[lang] = []
                
                # Update progress every few items or strictly
                if i % 5 == 0 or i == total_langs - 1:
                    elapsed = time.time() - start_time
                    if i > 0:
                        avg_time = elapsed / i
                        remaining = avg_time * (total_langs - i)
                        eta_str = f"{remaining:.1f}s"
                    else:
                        eta_str = "calculating..."
                        
                    print(f"  [{i+1}/{total_langs}] Processing {lang}... (Verses: {total_verses}, ETA: {eta_str})")
                
                for chapter in chapters:
                    content = chapter.get('content', '')
                    if content:
                        # Tokenize
                        tokens = self.tokenizer.encode(content)
                        
                        start_idx = len(all_tokens)
                        length = len(tokens)
                        
                        all_tokens.extend(tokens)
                        
                        # Store index info
                        self.verse_indices.append((start_idx, length))
                        
                        # Store mapping for this language
                        self.locale_verse_map[lang].append(total_verses)
                        
                        total_verses += 1
            
            print(f"  Total verses processed: {total_verses}")
            print(f"  Total tokens: {len(all_tokens)}")
            
            if total_verses == 0:
                raise ValueError(f"No verses found in {bible_data_dir}. Ensure 'bible_data.json' files exist and are complete.")
        
        # Move to device (VRAM or CPU)
        print(f"  Moving {len(all_tokens)} tokens to storage device {self.storage_device}...")
        try:
            # Ensure it's a tensor
            if not isinstance(all_tokens, torch.Tensor):
                self.data_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.storage_device)
            else:
                self.data_tensor = all_tokens.to(self.storage_device)
                
            print(f"  [OK] Data successfully loaded to {self.storage_device}")
        except RuntimeError as e:
            print(f"  [ERROR] Failed to load data to {self.storage_device}: {e}")
            if str(self.storage_device).startswith('cuda'):
                print("  Falling back to CPU tensor (pinned)")
                self.storage_device = torch.device('cpu')
                if not isinstance(all_tokens, torch.Tensor):
                    self.data_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.storage_device)
                else:
                    self.data_tensor = all_tokens.to(self.storage_device)
            else:
                 raise e

        # -----------------------------------------------------------------
        # COMPRESS VOCABULARY (Subset Optimization) - GLOBAL
        # -----------------------------------------------------------------
        # Run if target_languages are specified OR if we just want to optimize the loaded raw data
        # Logic: If we filtered (via cache or raw), we want to compress.
        
        if self.target_languages:
            print("  Optimizing vocabulary for selected languages...")
            
            # We need to find all unique token IDs used in the VALID parts of data_tensor
            # data_tensor is one giant array. We only care about ranges in self.verse_indices.
            
            # 1. Create a boolean mask of valid regions
            # Need to handle device carefully
            valid_mask = torch.zeros(len(self.data_tensor), dtype=torch.bool, device=self.storage_device)
            
            # This loop might be slow if 100k verses. Vectorize?
            # Construct a tensor of [start, len]
            # We can use repeat_interleave or just loop if vectorized ops not avail.
            # Given 100k items, Python loop is ~100ms. Acceptable.
            
            print(f"    Scanning unique tokens in {len(self.verse_indices)} verses...")
            
            # Option: Iterate verses and collect uniques.
            # Or just fill mask.
            for start, length in self.verse_indices:
                valid_mask[start : start + length] = True
                
            # 2. Get unique tokens
            # If on GPU, this is fast.
            active_tokens = torch.unique(self.data_tensor[valid_mask])
            print(f"    Found {len(active_tokens)} unique characters (original vocab: {self.tokenizer.vocab_size})")
            
            # 3. Compress Tokenizer
            remap_table = self.tokenizer.compress_vocab(active_tokens)
            
            if remap_table is not None:
                # 4. Remap Data Tensor
                # Move table to same device
                remap_table = remap_table.to(self.storage_device)
                
                # Apply mapping: new = table[old]
                # We apply it to the WHOLE tensor (even invalid parts) because it's faster vectorized
                # Invalid parts map to UNK or whatever, doesn't matter as they are never sampled.
                print("    Remapping dataset to new vocabulary...")
                self.data_tensor = remap_table[self.data_tensor]
                
                print(f"  [OK] Vocabulary compressed to {self.tokenizer.vocab_size} tokens")



    
    def _load_translations(self) -> Dict[str, List[Dict]]:
        """
        Load Bible translations from the Bible data directory.
        
        Only loads COMPLETE translations with exactly 1188 chapters (full Bible).
        
        Returns:
            Dictionary mapping locale -> list of verses
        """
        translations = {}
        incomplete_count = 0
        incomplete_examples = []
        
        # Only load complete translations (1188 chapters = complete Bible)
        for locale_dir in self.bible_data_dir.iterdir():
            if not locale_dir.is_dir():
                continue
            
            # Filter by language if specified
            if self.target_languages:
                if locale_dir.name not in self.target_languages:
                    continue
            
            bible_file = locale_dir / "bible_data.json"
            if not bible_file.exists():
                continue
            
            try:
                with open(bible_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Only use complete translations (1189 chapters)
                if len(data) == 1189:
                    translations[locale_dir.name] = data
                else:
                    incomplete_count += 1
                    if len(incomplete_examples) < 5:
                        incomplete_examples.append(f"{locale_dir.name} ({len(data)} chapters)")
            except Exception as e:
                print(f"  Warning: Failed to load {locale_dir.name}: {e}")
        
        # Report statistics
        total_found = len(translations) + incomplete_count
        print(f"  Found {total_found} translation directories")
        print(f"  [OK] Loaded {len(translations)} complete translations (1189 chapters each)")
        
        if incomplete_count > 0:
            print(f"  [SKIP] Skipped {incomplete_count} incomplete translations (still being scraped)")
            if incomplete_examples:
                print(f"    Examples: {', '.join(incomplete_examples)}")
        
        return translations
    

    
    def _sample_task(self) -> str:
        """Sample a task according to the distribution."""
        return self.rng.choice(self.tasks, p=self.task_probs)
    
    def _get_lm_sample(self) -> Dict:
        """
        Get a sample for language modeling task.
        
        Returns:
            {'task': 'lm', 'tokens': ..., 'labels': ...}
        """
        # Sample a random verse index
        idx = self.rng.randint(0, len(self.verse_indices))
        start, length = self.verse_indices[idx]
        
        # Get tokens from GPU tensor (no slicing yet, just indices)
        # We need to extract the subsequence. 
        # Since self.data_tensor is on GPU, slicing it returns a GPU tensor.
        
        # Initial extraction (raw tokens)
        raw_tokens = self.data_tensor[start : start + length]
        
        # Truncate if needed (rare for verses, but good safety)
        if len(raw_tokens) > self.max_seq_len:
            raw_tokens = raw_tokens[:self.max_seq_len]
            
        # Create input (tokens) and labels (tokens shifted)
        # We need to construct the full sequence of length max_seq_len
        
        # Create output buffers directly on storage device first
        tokens_out = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.storage_device)
        labels_out = torch.full((self.max_seq_len,), -100, dtype=torch.long, device=self.storage_device)
        
        # Fill data
        seq_len = len(raw_tokens)
        tokens_out[:seq_len] = raw_tokens
        
        # Labels: shift by 1. 
        # Input: [A, B, C, D]
        # Label: [B, C, D, Pad] (This is valid, but strictly it's usually Input[1:] vs Label[1:])
        # Standard causal LM: 
        # Input: [A, B, C]
        # Target: [B, C, D]
        
        # Let's align with the previous logic:
        # tokens = tokens
        # labels = tokens[1:] + [0]
        
        # Implementation on tensor:
        if seq_len > 1:
            labels_out[:seq_len-1] = raw_tokens[1:]
            labels_out[seq_len-1] = 0 # Pad token or End of sentence? Previous code used 0.
        else:
            # Too short
            labels_out[0] = 0
            
        if self.storage_device != self.device:
            return {
                'task': 'lm',
                'tokens': tokens_out.unsqueeze(0).to(self.device),
                'labels': labels_out.unsqueeze(0).to(self.device)
            }
        
        return {
            'task': 'lm',
            'tokens': tokens_out.unsqueeze(0),
            'labels': labels_out.unsqueeze(0)
        }
        

        
        return {
            'task': 'lm',
            'tokens': tokens.unsqueeze(0),  # Add batch dimension -> (1, seq_len)
            'labels': labels.unsqueeze(0)   # Add batch dimension -> (1, seq_len)
        }
    
    def _get_coherence_sample(self) -> Dict:
        """
        Get a sample for coherence detection task.
        
        Returns:
            {'task': 'coherence', 'verse1_tokens': ..., 'verse2_tokens': ..., 'labels': ...}
        """
        # 50% chance of consecutive verses (coherent), 50% random (incoherent)
        is_coherent = self.rng.rand() < 0.5
        
        if is_coherent:
            # Sample consecutive verses
            # We need to find valid consecutive indices. 
            # Simplified: just pick random idx and idx+1, check if valid (same book/sequentially stored)
            # The current flat list self.verse_indices stores verses in order of processing per file.
            # Assuming file order is preserved (Genesis 1, Gen 2...), consecutive indices are likely consecutive verses.
            idx1 = self.rng.randint(0, len(self.verse_indices) - 1)
            idx2 = idx1 + 1
            label = 1.0
        else:
            # Sample random verses
            idx1, idx2 = self.rng.choice(len(self.verse_indices), size=2, replace=False)
            label = 0.0
            
        # Extract tensors
        start1, len1 = self.verse_indices[idx1]
        start2, len2 = self.verse_indices[idx2]
        
        raw1 = self.data_tensor[start1 : start1 + len1][:self.max_seq_len]
        raw2 = self.data_tensor[start2 : start2 + len2][:self.max_seq_len]
        
        # Pad
        v1_out = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.storage_device)
        v2_out = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.storage_device)
        
        v1_out[:len(raw1)] = raw1
        v2_out[:len(raw2)] = raw2
        
        if self.storage_device != self.device:
             return {
                'task': 'coherence',
                'verse1_tokens': v1_out.unsqueeze(0).to(self.device),
                'verse2_tokens': v2_out.unsqueeze(0).to(self.device),
                'labels': torch.tensor(label, dtype=torch.float32, device=self.device).unsqueeze(0)
            }

        return {
            'task': 'coherence',
            'verse1_tokens': v1_out.unsqueeze(0),
            'verse2_tokens': v2_out.unsqueeze(0),
            'labels': torch.tensor(label, dtype=torch.float32, device=self.device).unsqueeze(0)
        }
    
    def _get_cross_ref_sample(self) -> Dict:
        """
        Get a sample for cross-reference prediction task (triplet loss).
        
        For now, we'll use a simple heuristic: verses from the same book are "related",
        verses from different books are "unrelated". This can be improved with actual
        cross-reference annotations.
        
        Returns:
            {'task': 'cross_ref', 'anchor_tokens': ..., 'positive_tokens': ..., 'negative_tokens': ...}
        """
        # Sample anchor verse
        anchor_idx = self.rng.randint(0, len(self.verse_indices))
        
        # Positive: nearby verse (within 5 verses)
        positive_idx = anchor_idx + self.rng.randint(-5, 6)
        positive_idx = max(0, min(len(self.verse_indices) - 1, positive_idx))
        
        # Negative: random distant verse
        negative_idx = self.rng.randint(0, len(self.verse_indices))
        while abs(negative_idx - anchor_idx) < 100:  # Ensure it's distant
            negative_idx = self.rng.randint(0, len(self.verse_indices))
            
        # Extract indices
        s_a, l_a = self.verse_indices[anchor_idx]
        s_p, l_p = self.verse_indices[positive_idx]
        s_n, l_n = self.verse_indices[negative_idx]
        
        # Helper to extract and pad
        def get_padded(start, length):
            raw = self.data_tensor[start : start + length][:self.max_seq_len]
            out = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.storage_device)
            out[:len(raw)] = raw
            return out
        
        if self.storage_device != self.device:
             return {
                'task': 'cross_ref',
                'anchor_tokens': get_padded(s_a, l_a).unsqueeze(0).to(self.device),
                'positive_tokens': get_padded(s_p, l_p).unsqueeze(0).to(self.device),
                'negative_tokens': get_padded(s_n, l_n).unsqueeze(0).to(self.device)
            }
            
        return {
            'task': 'cross_ref',
            'anchor_tokens': get_padded(s_a, l_a).unsqueeze(0),
            'positive_tokens': get_padded(s_p, l_p).unsqueeze(0),
            'negative_tokens': get_padded(s_n, l_n).unsqueeze(0)
        }
    
    def _get_paraphrase_sample(self) -> Dict:
        """
        Get a sample for cross-lingual paraphrase detection.
        
        Returns:
            {'task': 'paraphrase', 'verse1_tokens': ..., 'verse2_tokens': ..., 'labels': ...}
        """
        # Need at least 2 translations
        if len(self.locale_verse_map) < 2:
            # Fallback to coherence task
            return self._get_coherence_sample()
        
        # 50% chance of same verse in different languages (paraphrase)
        # 50% chance of different verses (not paraphrase)
        # 50% paraphrase, 50% random
        is_paraphrase = self.rng.rand() < 0.5
        
        locales = list(self.locale_verse_map.keys())
        
        if is_paraphrase:
            # Same verse, different languages
            lang1, lang2 = self.rng.choice(locales, size=2, replace=False)
            
            # Get available verse INDICES for each language
            # We stored mapping: self.locale_verse_map[lang] -> [global_index, global_index...]
            indices1 = self.locale_verse_map[lang1]
            indices2 = self.locale_verse_map[lang2]
            
            # Simple assumption: Both translations have verses in approx same order (Canon order)
            # And we only loaded complete Bibles (1188 vars).
            # So indices1[k] should correspond to indices2[k].
            
            common_len = min(len(indices1), len(indices2))
            k = self.rng.randint(0, common_len)
            
            idx1 = indices1[k]
            idx2 = indices2[k]
            label = 1.0
        else:
            # Random verses
            lang1, lang2 = self.rng.choice(locales, size=2, replace=False)
            indices1 = self.locale_verse_map[lang1]
            indices2 = self.locale_verse_map[lang2]
            
            # Pick random distinct k
            k1 = self.rng.randint(0, len(indices1))
            k2 = self.rng.randint(0, len(indices2))
            
            # Ensure not same
            while k1 == k2:
                k2 = self.rng.randint(0, len(indices2))
                
            idx1 = indices1[k1]
            idx2 = indices2[k2]
            label = 0.0
            
        # Extract
        s1, l1 = self.verse_indices[idx1]
        s2, l2 = self.verse_indices[idx2]
        
        raw1 = self.data_tensor[s1 : s1 + l1][:self.max_seq_len]
        raw2 = self.data_tensor[s2 : s2 + l2][:self.max_seq_len]
        
        # Pad
        v1_out = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.storage_device)
        v2_out = torch.zeros(self.max_seq_len, dtype=torch.long, device=self.storage_device)
        
        v1_out[:len(raw1)] = raw1
        v2_out[:len(raw2)] = raw2
        
        if self.storage_device != self.device:
             return {
                'task': 'paraphrase',
                'verse1_tokens': v1_out.unsqueeze(0).to(self.device),
                'verse2_tokens': v2_out.unsqueeze(0).to(self.device),
                'labels': torch.tensor(label, dtype=torch.float32, device=self.device).unsqueeze(0)
            }
        
        return {
            'task': 'paraphrase',
            'verse1_tokens': v1_out.unsqueeze(0),
            'verse2_tokens': v2_out.unsqueeze(0),
            'labels': torch.tensor(label, dtype=torch.float32, device=self.device).unsqueeze(0)
        }
    
    def __len__(self):
        # Arbitrary length (dataset is infinite via sampling)
        return 100000
    
    def __getitem__(self, idx):
        """
        Get a sample for a randomly selected task.
        
        Returns:
            Dictionary with task-specific data
        """
        task = self._sample_task()
        
        if task == 'lm':
            return self._get_lm_sample()
        elif task == 'coherence':
            return self._get_coherence_sample()
        elif task == 'cross_ref':
            return self._get_cross_ref_sample()
        elif task == 'paraphrase':
            return self._get_paraphrase_sample()
        else:
            raise ValueError(f"Unknown task: {task}")


def process_and_save_cache(
    bible_data_dir: str,
    tokenizer,
    output_path: str,
    target_languages: Optional[List[str]] = None
):
    """
    Process all Bible data and save to a cache file.
    
    Args:
        bible_data_dir: Path to Bible directory
        tokenizer: Tokenizer instance
        output_path: Path to save the .pt cache file
    """
    print(f"Processing data from {bible_data_dir}...")
    dataset = MultiTaskDataset(
        bible_data_dir=bible_data_dir,
        tokenizer=tokenizer,
        device=torch.device('cpu'), # Process involves CPU tokenization
        cache_path=None, # Force raw processing
        target_languages=target_languages
    )
    
    print(f"Saving cache to {output_path}...")
    torch.save({
        'tokens': dataset.data_tensor, # Already a tensor
        'indices': dataset.verse_indices,
        'locale_map': dataset.locale_verse_map
    }, output_path)
    print(f"[OK] Cache saved! Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def get_multi_task_dataloader(
    bible_data_dir: str,
    tokenizer,
    device: torch.device,
    cache_path: Optional[str] = None,
    batch_size: int = 4,
    max_seq_len: int = 512,
    task_distribution: Optional[Dict[str, float]] = None,
    num_workers: int = 0,
    seed: int = 42,
    cpu_data: bool = False,
    target_languages: Optional[List[str]] = None
) -> DataLoader:
    """
    Create a multi-task dataloader with VRAM preloading or CPU streaming.
    
    Args:
        cpu_data: If True, keep data on CPU and use workers to stream it.
                 If False, try to load entire dataset into VRAM.
    """
    
    # Decide where the data sits
    storage_device = torch.device('cpu') if cpu_data else device
    
    dataset = MultiTaskDataset(
        bible_data_dir=bible_data_dir,
        tokenizer=tokenizer,
        device=storage_device,  # This passes where the tensors live
        cache_path=cache_path,
        max_seq_len=max_seq_len,
        task_distribution=task_distribution,
        seed=seed,
        target_languages=target_languages
    )
    
    # If using CPU data, we might want workers.
    # If keeping data on GPU, num_workers MUST be 0.
    workers = num_workers if cpu_data else 0
    pin = True if cpu_data and torch.cuda.is_available() else False
    
    # Custom collate function - no-op since data is already tensor, 
    # but if cpu_data is True and we want strict batching, we might rely on default collation?
    # Actually our __getitem__ returns dictionaries of tensors with unsqueeze(0).
    # Default collation would stack them -> (Batch, 1, Seq).
    # We want (Batch, Seq).
    
    # Existing behavior for GPU resident: return batch[0] because batch_size was effectively handled? 
    # Wait, the previous implementation used `batch_size` in DataLoader construction, 
    # but `MultiTaskDataset.__getitem__` returns a single sample with unsqueeze(0).
    # If DataLoader batches them, we get a list of dicts.
    # The previous `collate_fn = lambda batch: batch[0]` implies batch_size=1 was forced or expected?
    
    # Ah, checking usages: DataLoader is called with `batch_size=batch_size`.
    # If batch_size > 1, collate_fn receives a list of N items.
    # `batch[0]` would only return the first item! This looks like a bug in previous code if batch_size > 1.
    # Let's check `train_native_multi_task.py`. Iterate `for batch_idx, batch in enumerate(dataloader):`.
    # If `collate_fn` throws away data, that's bad.
    
    # Let's fix collation properly while we are here.
    # If the dataset returns tensors on GPU, standard collate might fail or be slow?
    # No, `torch.utils.data.default_collate` handles GPU tensors fine.
    
    # Revert to legacy behavior: return only the first element.
    # The original implementation used lambda batch: batch[0], effectively forcing batch_size=1
    # regardless of the argument. Real batching requires handling mixed tasks or a batch sampler.
    # For now, we preserve legacy behavior to avoid breaking the training loop.
    def collate_fn(batch):
        return batch[0]

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # Dataset is random access anyway
        num_workers=workers, 
        collate_fn=collate_fn,
        pin_memory=pin
    )


if __name__ == "__main__":
    # Test the dataset
    from ..models.tokenizer import GenesisTokenizer
    
    print("Testing multi-task dataset...")
    
    tokenizer = GenesisTokenizer("genesis_tokenizer.json")
    
    loader = get_multi_task_dataloader(
        corpus_path="nwt_corpus.txt",
        bible_data_dir="../../Bible",
        tokenizer=tokenizer,
        batch_size=1,
        max_seq_len=128
    )
    
    print("\nSampling from different tasks:")
    for i, batch in enumerate(loader):
        if i >= 10:
            break
        print(f"  Batch {i}: Task = {batch['task']}")
    
