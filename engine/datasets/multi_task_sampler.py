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
        corpus_path: str,
        bible_data_dir: str,
        tokenizer,
        max_seq_len: int = 512,
        task_distribution: Optional[Dict[str, float]] = None,
        seed: int = 42
    ):
        """
        Initialize multi-task dataset.
        
        Args:
            corpus_path: Path to main corpus (nwt_corpus.txt)
            bible_data_dir: Path to Bible directory with translations
            tokenizer: Tokenizer instance
            max_seq_len: Maximum sequence length
            task_distribution: Distribution of tasks {'lm': 0.70, 'coherence': 0.15, ...}
            seed: Random seed
        """
        self.corpus_path = corpus_path
        self.bible_data_dir = Path(bible_data_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.rng = np.random.RandomState(seed)
        
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
        
        # Load corpus
        print(f"Loading corpus from {corpus_path}...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = f.read()
        
        # Parse verses from corpus (simple splitting by newlines for now)
        self.verses = [line.strip() for line in self.corpus.split('\n') if line.strip()]
        print(f"  Found {len(self.verses)} verses")
        
        # Load multi-language data
        print(f"Loading multi-language translations from {bible_data_dir}...")
        self.translations = self._load_translations()
        print(f"  Loaded {len(self.translations)} translations")
        
        # Build verse index (book-chapter-verse mapping)
        self.verse_index = self._build_verse_index()
    
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
            
            bible_file = locale_dir / "bible_data.json"
            if not bible_file.exists():
                continue
            
            try:
                with open(bible_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Only use complete translations (1188 chapters = 66 books complete)
                if len(data) == 1188:
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
        print(f"  [OK] Loaded {len(translations)} complete translations (1188 chapters each)")
        
        if incomplete_count > 0:
            print(f"  [SKIP] Skipped {incomplete_count} incomplete translations (still being scraped)")
            if incomplete_examples:
                print(f"    Examples: {', '.join(incomplete_examples)}")
        
        return translations
    
    def _build_verse_index(self) -> Dict[Tuple[int, int, int], str]:
        """
        Build index mapping (book, chapter, verse) -> verse text.
        
        For simplicity, we'll parse verse markers from the corpus.
        Format expected: "Genesis 1:1 In the beginning..."
        """
        index = {}
        
        # Simple regex-based parsing (can be improved)
        import re
        book_pattern = r'^([A-Za-z0-9 ]+) (\d+):(\d+)'
        
        for verse_line in self.verses:
            match = re.match(book_pattern, verse_line)
            if match:
                book_name = match.group(1).strip()
                chapter = int(match.group(2))
                verse = int(match.group(3))
                
                # Map book name to number (simplified)
                book_num = hash(book_name) % 66 + 1  # Placeholder
                
                # Extract verse text
                verse_text = verse_line[match.end():].strip()
                
                index[(book_num, chapter, verse)] = verse_text
        
        return index
    
    def _sample_task(self) -> str:
        """Sample a task according to the distribution."""
        return self.rng.choice(self.tasks, p=self.task_probs)
    
    def _get_lm_sample(self) -> Dict:
        """
        Get a sample for language modeling task.
        
        Returns:
            {'task': 'lm', 'tokens': ..., 'labels': ...}
        """
        # Sample a random verse
        verse = self.rng.choice(self.verses)
        
        # Tokenize
        tokens = self.tokenizer.encode(verse)[:self.max_seq_len]  # Truncate manually
        
        # Create labels (shifted by 1 for autoregressive prediction)
        labels = tokens[1:] + [0]  # Use 0 as pad token
        
        # Convert to tensors
        tokens = torch.tensor(tokens[:self.max_seq_len], dtype=torch.long)
        labels = torch.tensor(labels[:self.max_seq_len], dtype=torch.long)
        
        # Pad if necessary
        if len(tokens) < self.max_seq_len:
            padding = self.max_seq_len - len(tokens)
            tokens = torch.cat([tokens, torch.zeros((padding,), dtype=torch.long)])  # Pad with 0
            labels = torch.cat([labels, torch.full((padding,), -100, dtype=torch.long)])  # -100 = ignore
        
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
            idx = self.rng.randint(0, len(self.verses) - 1)
            verse1 = self.verses[idx]
            verse2 = self.verses[idx + 1]
            label = 1.0
        else:
            # Sample random verses
            idx1, idx2 = self.rng.choice(len(self.verses), size=2, replace=False)
            verse1 = self.verses[idx1]
            verse2 = self.verses[idx2]
            label = 0.0
        
        # Tokenize
        verse1_tokens = torch.tensor(
            self.tokenizer.encode(verse1)[:self.max_seq_len],  # Truncate manually
            dtype=torch.long
        )
        verse2_tokens = torch.tensor(
            self.tokenizer.encode(verse2)[:self.max_seq_len],  # Truncate manually
            dtype=torch.long
        )
        
        # Pad
        if len(verse1_tokens) < self.max_seq_len:
            verse1_tokens = torch.cat([
                verse1_tokens,
                torch.zeros((self.max_seq_len - len(verse1_tokens),), dtype=torch.long)  # Pad with 0
            ])
        if len(verse2_tokens) < self.max_seq_len:
            verse2_tokens = torch.cat([
                verse2_tokens,
                torch.zeros((self.max_seq_len - len(verse2_tokens),), dtype=torch.long)  # Pad with 0
            ])
        
        return {
            'task': 'coherence',
            'verse1_tokens': verse1_tokens[:self.max_seq_len].unsqueeze(0),  # Add batch dim
            'verse2_tokens': verse2_tokens[:self.max_seq_len].unsqueeze(0),  # Add batch dim
            'labels': torch.tensor(label, dtype=torch.float32).unsqueeze(0)   # Add batch dim
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
        anchor_idx = self.rng.randint(0, len(self.verses))
        anchor = self.verses[anchor_idx]
        
        # Positive: nearby verse (within 5 verses)
        positive_idx = anchor_idx + self.rng.randint(-5, 6)
        positive_idx = max(0, min(len(self.verses) - 1, positive_idx))
        positive = self.verses[positive_idx]
        
        # Negative: random distant verse
        negative_idx = self.rng.randint(0, len(self.verses))
        while abs(negative_idx - anchor_idx) < 100:  # Ensure it's distant
            negative_idx = self.rng.randint(0, len(self.verses))
        negative = self.verses[negative_idx]
        
        # Tokenize all three
        anchor_tokens = torch.tensor(
            self.tokenizer.encode(anchor)[:self.max_seq_len],  # Truncate manually
            dtype=torch.long
        )
        positive_tokens = torch.tensor(
            self.tokenizer.encode(positive)[:self.max_seq_len],  # Truncate manually
            dtype=torch.long
        )
        negative_tokens = torch.tensor(
            self.tokenizer.encode(negative)[:self.max_seq_len],  # Truncate manually
            dtype=torch.long
        )
        
        # Pad all three tensors
        if len(anchor_tokens) < self.max_seq_len:
            padding = self.max_seq_len - len(anchor_tokens)
            anchor_tokens = torch.cat([anchor_tokens, torch.zeros((padding,), dtype=torch.long)])
        if len(positive_tokens) < self.max_seq_len:
            padding = self.max_seq_len - len(positive_tokens)
            positive_tokens = torch.cat([positive_tokens, torch.zeros((padding,), dtype=torch.long)])
        if len(negative_tokens) < self.max_seq_len:
            padding = self.max_seq_len - len(negative_tokens)
            negative_tokens = torch.cat([negative_tokens, torch.zeros((padding,), dtype=torch.long)])
        
        return {
            'task': 'cross_ref',
            'anchor_tokens': anchor_tokens[:self.max_seq_len].unsqueeze(0),    # Add batch dim
            'positive_tokens': positive_tokens[:self.max_seq_len].unsqueeze(0),  # Add batch dim
            'negative_tokens': negative_tokens[:self.max_seq_len].unsqueeze(0)   # Add batch dim
        }
    
    def _get_paraphrase_sample(self) -> Dict:
        """
        Get a sample for cross-lingual paraphrase detection.
        
        Returns:
            {'task': 'paraphrase', 'verse1_tokens': ..., 'verse2_tokens': ..., 'labels': ...}
        """
        # Need at least 2 translations
        if len(self.translations) < 2:
            # Fallback to coherence task
            return self._get_coherence_sample()
        
        # 50% chance of same verse in different languages (paraphrase)
        # 50% chance of different verses (not paraphrase)
        is_paraphrase = self.rng.rand() < 0.5
        
        if is_paraphrase:
            # Sample same verse from two different languages
            locales = list(self.translations.keys())
            lang1, lang2 = self.rng.choice(locales, size=2, replace=False)
            
            # Find a verse that exists in both
            verses1 = self.translations[lang1]
            verses2 = self.translations[lang2]
            
            # Sample random chapter index (both have 1188)
            idx = self.rng.randint(0, min(len(verses1), len(verses2)))
            
            verse1_text = verses1[idx].get('content', '')
            verse2_text = verses2[idx].get('content', '')
            
            label = 1.0
        else:
            # Sample different verses from different languages
            locales = list(self.translations.keys())
            lang1, lang2 = self.rng.choice(locales, size=2, replace=False)
            
            verses1 = self.translations[lang1]
            verses2 = self.translations[lang2]
            
            idx1 = self.rng.randint(0, len(verses1))
            idx2 = self.rng.randint(0, len(verses2))
            
            # Ensure different chapters
            while idx1 == idx2:
                idx2 = self.rng.randint(0, len(verses2))
            
            verse1_text = verses1[idx1].get('content', '')
            verse2_text = verses2[idx2].get('content', '')
            
            label = 0.0
        
        # Tokenize
        verse1_tokens = torch.tensor(
            self.tokenizer.encode(verse1_text)[:self.max_seq_len],  # Truncate manually
            dtype=torch.long
        )
        verse2_tokens = torch.tensor(
            self.tokenizer.encode(verse2_text)[:self.max_seq_len],  # Truncate manually
            dtype=torch.long
        )
        
        # Pad
        if len(verse1_tokens) < self.max_seq_len:
            verse1_tokens = torch.cat([
                verse1_tokens,
                torch.zeros((self.max_seq_len - len(verse1_tokens),), dtype=torch.long)  # Pad with 0
            ])
        if len(verse2_tokens) < self.max_seq_len:
            verse2_tokens = torch.cat([
                verse2_tokens,
                torch.zeros((self.max_seq_len - len(verse2_tokens),), dtype=torch.long)  # Pad with 0
            ])
        
        return {
            'task': 'paraphrase',
            'verse1_tokens': verse1_tokens[:self.max_seq_len].unsqueeze(0),  # Add batch dim
            'verse2_tokens': verse2_tokens[:self.max_seq_len].unsqueeze(0),  # Add batch dim
            'labels': torch.tensor(label, dtype=torch.float32).unsqueeze(0)   # Add batch dim
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


def get_multi_task_dataloader(
    corpus_path: str,
    bible_data_dir: str,
    tokenizer,
    batch_size: int = 4,
    max_seq_len: int = 512,
    task_distribution: Optional[Dict[str, float]] = None,
    num_workers: int = 0,
    seed: int = 42
) -> DataLoader:
    """
    Create a multi-task dataloader.
    
    Args:
        corpus_path: Path to main corpus
        bible_data_dir: Path to Bible directory with translations
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        task_distribution: Task distribution
        num_workers: Number of dataloader workers
        seed: Random seed
    
    Returns:
        DataLoader instance
    """
    dataset = MultiTaskDataset(
        corpus_path=corpus_path,
        bible_data_dir=bible_data_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        task_distribution=task_distribution,
        seed=seed
    )
    
    # Custom collate function  - always return first element
    def collate_fn(batch):
        return batch[0]
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffling handled by random sampling
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test the dataset
    from models.tokenizer import GenesisTokenizer
    
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
    
    print("\n✓ Multi-task dataloader working!")
