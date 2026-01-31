#!/usr/bin/env python3
"""
Update VRAM Data Cache
======================
Pre-processes the Bible dataset into a tokenized tensor and saves it to disk
for fast loading into VRAM during training.
"""

import os
import sys
from pathlib import Path
import torch
import toml

# Genesis Imports
from genesis.config import get_data_path, find_project_root

# Add src to path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

from genesis.models.tokenizer import GenesisTokenizer
from genesis.datasets.multi_task_sampler import process_and_save_cache

def main():
    print(">>> Genesis Data Cache Updater\n")
    
    # 1. Resolve paths using central config/defaults
    try:
        project_root = find_project_root()
        bible_dir = get_data_path("bible_dir", "Bible", project_root)
        cache_path = get_data_path("cache_path", "data/genesis_data_cache.pt", project_root)
        tokenizer_path = get_data_path("tokenizer_path", "data/genesis_char_tokenizer.json", project_root)
        print(f"[CONFIG] Integrated paths from global configuration")
    except Exception as e:
        print(f"[ERROR] Path resolution failed: {e}")
        return
    
    print(f"Bible Directory: {bible_dir}")
    print(f"Cache Output:    {cache_path}")
    print(f"Tokenizer:       {tokenizer_path}")
    print("-" * 50)
    
    if not bible_dir.exists():
        print(f"[ERROR] Bible directory not found at {bible_dir}")
        return
        
    if not tokenizer_path.exists():
        print(f"[ERROR] Tokenizer not found at {tokenizer_path}")
        return
        
    # Initialize tokenizer
    tokenizer = GenesisTokenizer(str(tokenizer_path))
    
    # Interactive Language Selection
    target_languages = None
    print("\n[?] Do you want to process ALL languages?")
    choice = input("    Press Enter for ALL, or type 'n' to select specific languages: ").strip().lower()
    
    if choice == 'n':
        lang_input = input("    Enter language codes (comma-separated, e.g., 'en, sv, de'): ").strip()
        if lang_input:
            target_languages = [l.strip() for l in lang_input.split(',') if l.strip()]
            print(f"    Selected languages: {target_languages}")
        else:
            print("    No languages entered. Defaulting to ALL.")
            
    # Run process
    try:
        process_and_save_cache(
            bible_data_dir=str(bible_dir),
            tokenizer=tokenizer,
            output_path=str(cache_path),
            target_languages=target_languages
        )
        print("\n[SUCCESS] Cache created successfully.")
        print("You can now run training using this cache file.")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to update cache: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
