"""
Character-Level Tokenizer Training Script
==========================================
Creates a pure character-level tokenizer for multilingual Bible corpus.

Features:
- Automatic character vocabulary extraction from corpus
- Unicode normalization (NFKC)
- Simple character-to-ID mapping
- No word-specific special tokens (no "Jehovah" token)
- Supports all languages in multilingual Bible database

Usage:
    python scripts/train_char_tokenizer.py
    python scripts/train_char_tokenizer.py --corpus path/to/corpus.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
import unicodedata

# Ensure UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class CharacterTokenizer:
    """Pure character-level tokenizer."""
    
    # Special tokens (minimal set)
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        
        # Initialize with special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN
        ]
        
        for token in special_tokens:
            self.char_to_id[token] = len(self.char_to_id)
            self.id_to_char[len(self.id_to_char)] = token
        
        self.vocab_size = len(self.char_to_id)
    
    def build_vocab_from_corpus(self, corpus_path, normalize=True):
        """
        Build character vocabulary from a single corpus file.
        
        Args:
            corpus_path: Path to text corpus
            normalize: Apply Unicode NFKC normalization
        """
        print(f"Reading corpus from {corpus_path}...")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if normalize:
            print("Applying Unicode NFKC normalization...")
            text = unicodedata.normalize('NFKC', text)
        
        # Extract unique characters
        print("Extracting unique characters...")
        char_counts = Counter(text)
        self._add_chars_to_vocab(char_counts)

    def build_vocab_from_dataset(self, bible_dir, normalize=True):
        """
        Build character vocabulary from the entire Bible dataset.
        
        Args:
            bible_dir: Path to directory containing language subdirectories
            normalize: Apply Unicode NFKC normalization
        """
        bible_path = Path(bible_dir)
        print(f"Scanning for complete translations in {bible_path}...")
        
        all_char_counts = Counter()
        processed_count = 0
        
        # Discover all bible_data.json files
        for locale_dir in bible_path.iterdir():
            if not locale_dir.is_dir():
                continue
            
            bible_file = locale_dir / "bible_data.json"
            if not bible_file.exists():
                continue
                
            try:
                with open(bible_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"  Processing {locale_dir.name} ({len(data)} chapters)...")
                full_text = []
                for chapter_data in data:
                    if isinstance(chapter_data, dict):
                        content = chapter_data.get('content', '')
                        full_text.append(content)
                
                lang_text = " ".join(full_text)
                if normalize:
                    lang_text = unicodedata.normalize('NFKC', lang_text)
                
                all_char_counts.update(lang_text)
                processed_count += 1
            except Exception as e:
                print(f"    Warning: Failed to process {locale_dir.name}: {e}")

        print(f"\n[OK] Processed {processed_count} translations.")
        self._add_chars_to_vocab(all_char_counts)

    def _add_chars_to_vocab(self, char_counts):
        """Add characters from counts to vocabulary."""
        unique_chars = sorted(char_counts.keys())
        print(f"Found {len(unique_chars)} unique characters")
        
        # Add characters to vocabulary (excluding special tokens)
        for char in unique_chars:
            if char not in self.char_to_id:
                char_id = len(self.char_to_id)
                self.char_to_id[char] = char_id
                self.id_to_char[char_id] = char
        
        self.vocab_size = len(self.char_to_id)
        
        print(f"\nVocabulary Statistics:")
        print(f"  Total vocabulary size: {self.vocab_size}")
        print(f"  Special tokens: 4")
        print(f"  Character tokens: {self.vocab_size - 4}")
        
        # Show character frequency distribution
        print(f"\nTop 20 Most Frequent Characters:")
        for char, count in char_counts.most_common(20):
            char_display = repr(char) if char in [' ', '\n', '\t'] else char
            print(f"  {char_display}: {count:,}")
    
    def encode(self, text, normalize=True, add_special_tokens=False):
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
            normalize: Apply Unicode normalization
            add_special_tokens: Add BOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        if normalize:
            text = unicodedata.normalize('NFKC', text)
        
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.char_to_id[self.BOS_TOKEN])
        
        for char in text:
            token_id = self.char_to_id.get(char, self.char_to_id[self.UNK_TOKEN])
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.char_to_id[self.EOS_TOKEN])
        
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            Decoded text string
        """
        special_tokens = {
            self.char_to_id[self.PAD_TOKEN],
            self.char_to_id[self.UNK_TOKEN],
            self.char_to_id[self.BOS_TOKEN],
            self.char_to_id[self.EOS_TOKEN]
        }
        
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_tokens:
                continue
            chars.append(self.id_to_char.get(token_id, self.UNK_TOKEN))
        
        return ''.join(chars)
    
    def save(self, output_path):
        """Save tokenizer to JSON file (HuggingFace tokenizers format)."""
        tokenizer_config = {
            "version": "1.0",
            "type": "CharacterLevel",
            "model": {
                "type": "CharacterLevel",
                "vocab": self.char_to_id,
                "unk_token": self.UNK_TOKEN
            },
            "normalizer": {
                "type": "NFKC"
            },
            "pre_tokenizer": None,
            "post_processor": None,
            "decoder": None,
            "added_tokens": [
                {"id": self.char_to_id[self.PAD_TOKEN], "content": self.PAD_TOKEN, "special": True},
                {"id": self.char_to_id[self.UNK_TOKEN], "content": self.UNK_TOKEN, "special": True},
                {"id": self.char_to_id[self.BOS_TOKEN], "content": self.BOS_TOKEN, "special": True},
                {"id": self.char_to_id[self.EOS_TOKEN], "content": self.EOS_TOKEN, "special": True}
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
        
        print(f"\n[OK] Tokenizer saved to: {output_path}")
        
        # Also save vocabulary mapping for reference
        vocab_path = output_path.replace('.json', '_vocab.json')
        vocab_info = {
            'vocab_size': self.vocab_size,
            'char_to_id': self.char_to_id,
            'id_to_char': {str(k): v for k, v in self.id_to_char.items()}
        }
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_info, f, ensure_ascii=False, indent=2)
        
        print(f"[OK] Vocabulary mapping saved to: {vocab_path}")


def find_bible_dir():
    """Find the Bible data directory using robust resolution."""
    script_path = Path(__file__).parent.absolute()
    
    # Try common locations
    fallbacks = [
        script_path.parent / "Bible",           # root/Bible
        script_path.parent.parent / "Bible",    # root/../Bible
        Path("Bible").absolute(),               # cwd/Bible
        Path("E:/AI/Research/Bible"),           # Absolute path
    ]
    
    for fb in fallbacks:
        if fb.exists() and fb.is_dir():
            return fb
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Train character-level tokenizer for Genesis Arbiter",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--bible-dir',
        type=str,
        default=None,
        help='Path to Bible dataset directory (containing language subdirs)'
    )
    parser.add_argument(
        '--corpus',
        type=str,
        default=None,
        help='Path to a single corpus file (overrides --bible-dir)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='genesis_char_tokenizer.json',
        help='Output tokenizer file path (default: genesis_char_tokenizer.json)'
    )
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable Unicode NFKC normalization'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run encoding/decoding tests after training'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Character-Level Tokenizer Training")
    print("=" * 70)
    print()
    
    # Initialize tokenizer
    tokenizer = CharacterTokenizer()
    
    if args.corpus:
        # Build from single file
        if not os.path.exists(args.corpus):
            print(f"Error: Corpus file not found: {args.corpus}")
            sys.exit(1)
        tokenizer.build_vocab_from_corpus(args.corpus, normalize=not args.no_normalize)
    else:
        # Build from dataset
        bible_dir = args.bible_dir or find_bible_dir()
        if not bible_dir:
            print("Error: Bible directory not found. Please provide --bible-dir.")
            sys.exit(1)
        tokenizer.build_vocab_from_dataset(bible_dir, normalize=not args.no_normalize)
    
    # Save tokenizer
    tokenizer.save(args.output)
    
    # Run tests if requested
    if args.test:
        print("\n" + "=" * 70)
        print("Running Encoding/Decoding Tests")
        print("=" * 70)
        
        test_cases = [
            "In the beginning God created the heavens and the earth.",
            "Jehovah is the name of God.",
            "The quick brown fox jumps over the lazy dog.",
            "Θεός",  # Greek
            "אלהים",  # Hebrew
            "粵語",    # Cantonese
            "தமிழ்",   # Tamil
            "한국어",   # Korean
            "١٢٣",     # Arabic numerals
        ]
        
        for i, text in enumerate(test_cases, 1):
            try:
                print(f"\nTest {i}: {repr(text)}")
            except UnicodeEncodeError:
                print(f"\nTest {i}: [Non-ASCII text]")
            
            # Encode
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            print(f"  Tokens: {token_ids[:50]}{'...' if len(token_ids) > 50 else ''}")
            print(f"  Length: {len(token_ids)} characters")
            
            # Decode
            decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
            
            # Verify
            if decoded == text:
                print(f"  [OK] Decode successful")
            else:
                print(f"  [ERR] Decode mismatch!")
                print(f"    Expected: {repr(text)}")
                print(f"    Got: {repr(decoded)}")
    
    print("\n" + "=" * 70)
    print("[OK] Character-Level Tokenizer Training Complete!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. Test tokenizer: python scripts/count_unique_words.py")
    print(f"  2. Update training scripts to use: {args.output}")
    print()


if __name__ == "__main__":
    main()
