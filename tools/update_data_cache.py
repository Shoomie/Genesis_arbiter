#!/usr/bin/env python3
"""
Update VRAM Data Cache
======================
Pre-processes the Bible dataset into a byte-level tokenized tensor and saves it to disk.
"""

import os
import sys
from pathlib import Path
import torch
import time

# Add src to path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

from genesis.models.tokenizer import GenesisTokenizer
from genesis.datasets.multi_task_sampler import process_and_save_cache
from genesis.utils.config_loader import get_data_path

def main():
    print(">>> Genesis Byte-Level Data Cache Updater\n")
    
    try:
        bible_dir = get_data_path("bible_dir")
        cache_path = get_data_path("cache_path")
        print(f"[CONFIG] Using paths from global configuration")
    except Exception as e:
        print(f"[ERROR] Path resolution failed: {e}")
        return
    
    print(f"Bible Directory: {bible_dir}")
    print(f"Cache Output:    {cache_path}")
    print("-" * 50)
    
    if not bible_dir.exists():
        print(f"[ERROR] Bible directory not found at {bible_dir}")
        return
        
    # Initialize Byte Tokenizer
    tokenizer = GenesisTokenizer(type='byte')
    
    # Interactive Language Selection
    target_languages = None
    print("\n[?] Which languages should be included?")
    print("    1. ALL (Full Multilingual Corpus)")
    print("    2. Specific Languages (e.g., 'en, sv')")
    choice = input("    Selection [1/2, default 1]: ").strip()
    
    if choice == '2':
        lang_input = input("    Enter language codes (comma-separated): ").strip()
        if lang_input:
            target_languages = [l.strip() for l in lang_input.split(',') if l.strip()]
            print(f"    Selected: {target_languages}")
        else:
            print("    No languages entered. Defaulting to ALL.")
            
    # Always overwrite
    if cache_path.exists():
        print(f"[INFO] Overwriting existing cache at {cache_path}")
        
    # Run process
    try:
        start_time = time.time()
        process_and_save_cache(
            bible_data_dir=str(bible_dir),
            tokenizer=tokenizer,
            output_path=str(cache_path),
            target_languages=target_languages
        )
        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] Cache created in {elapsed:.1f}s.")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to update cache: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
