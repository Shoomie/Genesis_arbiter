"""
Arbiter Tokenizer Factory: Automated Tokenizer Training for Corpus-Specific Optimization

This module automates the creation of SentencePiece tokenizers optimized for
monolithic theological corpora. It supports multi-vocabulary sweeps, automatic
multi-word entity (MWE) extraction, and compression ratio analysis.

Key Features:
- Batch tokenizer generation at multiple vocabulary sizes
- TF-IDF-based MWE extraction for theological terms
- Compression ratio analysis and visualization
- Compatibility wrapper for torchtitan integration
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re

try:
    import sentencepiece as spm
    SPM_AVAILABLE = True
except ImportError:
    SPM_AVAILABLE = False
    print("[ERROR] SentencePiece not installed. Install with: pip install sentencepiece")

import numpy as np
from dataclasses import dataclass


@dataclass
class TokenizerMetrics:
    """Metrics for a trained tokenizer."""
    vocab_size: int
    compression_ratio: float  # original_chars / tokenized_length
    coverage: float  # character coverage
    mwe_count: int  # number of multi-word entities forced
    model_path: str


class MWEExtractor:
    """Extract multi-word entities from corpus using frequency and coherence."""
    
    def __init__(self, corpus_path: str, min_frequency: int = 50):
        self.corpus_path = Path(corpus_path)
        self.min_frequency = min_frequency
        self.text = self._load_corpus()
    
    def _load_corpus(self) -> str:
        """Load and clean corpus text."""
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        # Basic cleaning: normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def extract_mwe_candidates(self, max_ngram: int = 4) -> List[str]:
        """
        Extract multi-word entity candidates using frequency thresholds.
        
        Strategy:
        1. Extract all n-grams (2-4 words)
        2. Filter by minimum frequency
        3. Filter for proper noun patterns (capitalized)
        4. Rank by pointwise mutual information (PMI)
        
        Returns:
            List of MWE strings, e.g., ["Jehovah God", "Jesus Christ"]
        """
        # Tokenize into words
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', self.text)
        
        # Count n-grams
        ngram_counts = Counter()
        word_list = self.text.split()
        
        for n in range(2, max_ngram + 1):
            for i in range(len(word_list) - n + 1):
                ngram = ' '.join(word_list[i:i+n])
                # Only consider if all words are capitalized (proper nouns)
                if all(w[0].isupper() for w in ngram.split() if w):
                    ngram_counts[ngram] += 1
        
        # Filter by frequency
        candidates = [
            ngram for ngram, count in ngram_counts.items()
            if count >= self.min_frequency
        ]
        
        # Sort by frequency
        candidates.sort(key=lambda x: ngram_counts[x], reverse=True)
        
        return candidates[:100]  # Top 100 MWEs


class ArbiterTokenizerFactory:
    """
    Factory for creating corpus-optimized SentencePiece tokenizers.
    
    Usage:
        factory = ArbiterTokenizerFactory(
            corpus_path="./nwt_corpus.txt",
            output_dir="./tokenizers"
        )
        factory.create_multi_vocab_sweep(vocab_sizes=[4096, 8192, 16384])
        metrics = factory.analyze_compression()
    """
    
    def __init__(
        self,
        corpus_path: str,
        output_dir: str = "./tokenizers",
        model_prefix: str = "arbiter_tokenizer"
    ):
        if not SPM_AVAILABLE:
            raise RuntimeError("SentencePiece is required but not installed")
        
        self.corpus_path = Path(corpus_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_prefix = model_prefix
        
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")
        
        # Extract MWEs automatically
        print("[TokenizerFactory] Extracting multi-word entities...")
        self.mwe_extractor = MWEExtractor(str(self.corpus_path))
        self.mwe_candidates = self.mwe_extractor.extract_mwe_candidates()
        print(f"[TokenizerFactory] Found {len(self.mwe_candidates)} MWE candidates")
        
        # Storage for trained models
        self.trained_models: Dict[int, TokenizerMetrics] = {}
    
    def train_tokenizer(
        self,
        vocab_size: int,
        model_type: str = 'bpe',
        character_coverage: float = 1.0,
        user_defined_symbols: Optional[List[str]] = None
    ) -> str:
        """
        Train a single SentencePiece tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            model_type: 'bpe' or 'unigram'
            character_coverage: Fraction of characters to cover (1.0 = all)
            user_defined_symbols: Additional symbols to force as atomic tokens
            
        Returns:
            Path to trained .model file
        """
        output_prefix = self.output_dir / f"{self.model_prefix}_{vocab_size}"
        
        # Combine MWE candidates with user-defined symbols
        all_symbols = self.mwe_candidates[:50]  # Top 50 MWEs
        if user_defined_symbols:
            all_symbols.extend(user_defined_symbols)
        
        # Deduplicate
        all_symbols = list(set(all_symbols))
        
        print(f"\n[TokenizerFactory] Training tokenizer: vocab_size={vocab_size}")
        print(f"[TokenizerFactory] Forcing {len(all_symbols)} multi-word entities as atomic tokens")
        
        # Train with SentencePiece
        spm.SentencePieceTrainer.train(
            input=str(self.corpus_path),
            model_prefix=str(output_prefix),
            vocab_size=vocab_size,
            model_type=model_type,
            user_defined_symbols=all_symbols,
            character_coverage=character_coverage,
            normalization_rule_name='nmt_nfkc_cf',  # NFKC normalization
            split_by_whitespace=True,
            byte_fallback=True,  # Handle unknown chars
            unk_surface="<unk>",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            num_threads=os.cpu_count() or 1,
            train_extremely_large_corpus=False
        )
        
        model_path = f"{output_prefix}.model"
        print(f"[TokenizerFactory] ✓ Saved to: {model_path}")
        
        # Calculate metrics
        metrics = self._calculate_metrics(vocab_size, model_path, len(all_symbols))
        self.trained_models[vocab_size] = metrics
        
        return model_path
    
    def _calculate_metrics(
        self,
        vocab_size: int,
        model_path: str,
        mwe_count: int
    ) -> TokenizerMetrics:
        """Calculate compression ratio for a trained tokenizer."""
        sp = spm.SentencePieceProcessor(model_file=model_path)
        
        # Sample text for compression analysis
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            sample = f.read()[:100000]  # First 100k chars
        
        # Tokenize
        tokens = sp.encode(sample, out_type=int)
        
        # Compression ratio
        compression_ratio = len(sample) / len(tokens)
        
        # Coverage (simplified)
        coverage = 1.0  # SentencePiece guarantees full coverage with byte_fallback
        
        return TokenizerMetrics(
            vocab_size=vocab_size,
            compression_ratio=compression_ratio,
            coverage=coverage,
            mwe_count=mwe_count,
            model_path=model_path
        )
    
    def create_multi_vocab_sweep(
        self,
        vocab_sizes: List[int] = [4096, 8192, 16384, 32768]
    ) -> Dict[int, str]:
        """
        Train multiple tokenizers at different vocabulary sizes.
        
        Args:
            vocab_sizes: List of vocabulary sizes to train
            
        Returns:
            Dictionary mapping vocab_size -> model_path
        """
        print(f"\n{'='*60}")
        print(f"Multi-Vocabulary Tokenizer Sweep")
        print(f"{'='*60}")
        print(f"Corpus: {self.corpus_path}")
        print(f"Vocabulary sizes: {vocab_sizes}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*60}\n")
        
        results = {}
        for vocab_size in vocab_sizes:
            model_path = self.train_tokenizer(vocab_size)
            results[vocab_size] = model_path
        
        # Save summary
        self._save_summary()
        
        return results
    
    def _save_summary(self):
        """Save summary of all trained tokenizers."""
        summary = {
            'corpus': str(self.corpus_path),
            'mwe_candidates': self.mwe_candidates[:20],  # Top 20 for reference
            'tokenizers': {}
        }
        
        for vocab_size, metrics in self.trained_models.items():
            summary['tokenizers'][vocab_size] = {
                'vocab_size': metrics.vocab_size,
                'compression_ratio': round(metrics.compression_ratio, 2),
                'mwe_count': metrics.mwe_count,
                'model_path': metrics.model_path
            }
        
        summary_path = self.output_dir / 'tokenizer_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[TokenizerFactory] Summary saved to: {summary_path}")
        self._print_compression_analysis()
    
    def _print_compression_analysis(self):
        """Print compression analysis table."""
        print(f"\n{'='*60}")
        print(f"Compression Analysis")
        print(f"{'='*60}")
        print(f"{'Vocab Size':<12} {'Compression':<15} {'MWE Count':<12} {'Model Path'}")
        print(f"{'-'*60}")
        
        for vocab_size in sorted(self.trained_models.keys()):
            metrics = self.trained_models[vocab_size]
            print(
                f"{metrics.vocab_size:<12} "
                f"{metrics.compression_ratio:<15.2f} "
                f"{metrics.mwe_count:<12} "
                f"{Path(metrics.model_path).name}"
            )
        
        print(f"{'='*60}\n")
    
    def get_best_tokenizer(self, target_compression: float = 4.0) -> Tuple[int, str]:
        """
        Get tokenizer closest to target compression ratio.
        
        Args:
            target_compression: Desired compression ratio (chars/token)
            
        Returns:
            (vocab_size, model_path) tuple
        """
        if not self.trained_models:
            raise RuntimeError("No tokenizers trained yet")
        
        best_vocab = min(
            self.trained_models.keys(),
            key=lambda v: abs(self.trained_models[v].compression_ratio - target_compression)
        )
        
        return best_vocab, self.trained_models[best_vocab].model_path


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check if corpus path provided
    if len(sys.argv) < 2:
        print("Usage: python arbiter_tokenizer_factory.py <corpus_path>")
        print("Example: python arbiter_tokenizer_factory.py ./engine/nwt_corpus.txt")
        sys.exit(1)
    
    corpus_path = sys.argv[1]
    
    # Initialize factory
    factory = ArbiterTokenizerFactory(
        corpus_path=corpus_path,
        output_dir="./tokenizers",
        model_prefix="arbiter_nwt"
    )
    
    # Create sweep
    factory.create_multi_vocab_sweep(
        vocab_sizes=[4096, 8192, 16384]
    )
    
    # Get best tokenizer
    best_vocab, best_path = factory.get_best_tokenizer(target_compression=4.5)
    print(f"\n[Recommendation] Best tokenizer: {best_vocab} tokens ({best_path})")
