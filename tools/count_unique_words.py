import os
import sys
import json
from pathlib import Path
from collections import Counter
import unicodedata

# Add src directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from genesis.models.tokenizer import GenesisTokenizer

# Ensure UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

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

def analyze_dataset(tokenizer_path):
    """Analyze the entire multilingual Bible dataset."""
    bible_dir = find_bible_dir()
    if not bible_dir:
        print("[ERROR] Bible directory not found. Please ensure it exists in the project root or at E:/AI/Research/Bible.")
        return

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = GenesisTokenizer(tokenizer_path)
    
    print(f"\n[SCAN] Searching for complete translations in: {bible_dir}")
    
    total_data_size = 0
    total_chapters = 0
    all_unique_chars = Counter()
    all_unique_words = set()
    language_stats = []
    
    complete_translations = []
    for locale_dir in bible_dir.iterdir():
        if not locale_dir.is_dir():
            continue
        
        bible_file = locale_dir / "bible_data.json"
        if not bible_file.exists():
            continue
            
        try:
            file_size = bible_file.stat().st_size
            with open(bible_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Only use complete translations (>= 1188 chapters)
            if len(data) >= 1188:
                complete_translations.append((locale_dir.name, bible_file, data, file_size))
        except Exception as e:
            print(f"  Warning: Failed to load {locale_dir.name}: {e}")

    print(f"[OK] Found {len(complete_translations)} complete translations.\n")
    print(f"{'Language':<15} | {'Size (MB)':<10} | {'Unique Chars':<12} | {'Approx Words'}")
    print("-" * 60)

    for lang, path, chapters, size in sorted(complete_translations):
        lang_chars = Counter()
        lang_words = set()
        
        # Combine all text for this language
        full_text = []
        for chapter_data in chapters:
            if isinstance(chapter_data, dict):
                content = chapter_data.get('content', '')
                full_text.append(content)
        
        lang_text = " ".join(full_text)
        
        # Unique characters
        normalized_text = unicodedata.normalize('NFKC', lang_text)
        lang_chars.update(normalized_text)
        all_unique_chars.update(normalized_text)
        
        # Unique words (simple split)
        words = lang_text.lower().split()
        lang_words.update(words)
        all_unique_words.update(words)
        
        total_data_size += size
        total_chapters += len(chapters)
        
        size_mb = size / (1024 * 1024)
        print(f"{lang:<15} | {size_mb:>10.2f} | {len(lang_chars):>12} | {len(lang_words):>12,}")

    # Global Statistics
    print(f"\n{'='*70}")
    print("GLOBAL MULTILINGUAL CORPUS STATISTICS")
    print(f"{'='*70}")
    print(f"Total Complete Translations: {len(complete_translations)}")
    print(f"Total Data Size:            {total_data_size / (1024*1024):.2f} MB")
    print(f"Total Chapters:             {total_chapters:,}")
    print(f"Global Unique Characters:    {len(all_unique_chars):,}")
    print(f"Global Unique Words:         {len(all_unique_words):,}")
    
    # Tokenizer specific info
    print(f"\nTokenizer Status:")
    print(f"  Type: {'Character-Level' if tokenizer.is_character_level() else 'BPE'}")
    print(f"  Vocab Size: {tokenizer.vocab_size}")
    
    if tokenizer.is_character_level():
        # Check coverage
        tokenizer_chars = set(tokenizer.char_to_id.keys())
        dataset_chars = set(all_unique_chars.keys())
        missing = dataset_chars - tokenizer_chars
        print(f"  Vocab Coverage: {len(dataset_chars & tokenizer_chars)} / {len(dataset_chars)}")
        if missing:
            print(f"  Missing Characters from Vocab: {len(missing)} (first 20: {list(missing)[:20]})")

    print(f"{'='*70}\n")

if __name__ == "__main__":
    # Ensure we use the latest tokenizer
    tokenizer_file = os.path.join(parent_dir, "data", "genesis_char_tokenizer.json")
    if not os.path.exists(tokenizer_file):
        # Fallback to standard location
        alt_path = os.path.join(parent_dir, "genesis_tokenizer.json")
        if os.path.exists(alt_path):
            tokenizer_file = alt_path
        elif os.path.exists("genesis_char_tokenizer.json"):
            tokenizer_file = "genesis_char_tokenizer.json"

    analyze_dataset(tokenizer_file)
