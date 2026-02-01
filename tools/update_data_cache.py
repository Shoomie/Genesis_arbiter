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
import json

# Add src to path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

from genesis.models.tokenizer import GenesisTokenizer
from genesis.datasets.multi_task_sampler import process_and_save_cache
from genesis.utils.config_loader import get_data_path

def scan_complete_languages(bible_dir):
    """Scan the Bible directory for languages with exactly 1189 chapters."""
    complete_langs = []
    
    if not bible_dir.exists():
        return []
        
    print(f"Scanning {bible_dir} for complete translations...")
    for lang_dir in sorted(bible_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
            
        bible_file = lang_dir / "bible_data.json"
        if bible_file.exists():
            try:
                with open(bible_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if len(data) == 1189:
                    complete_langs.append(lang_dir.name)
            except Exception:
                continue
    return complete_langs

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
        
    # Scan for complete languages
    complete_langs = scan_complete_languages(Path(bible_dir))
    
    if not complete_langs:
        print(f"[ERROR] No complete translations (1189 chapters) found in {bible_dir}")
        return
        
    print(f"\nFound {len(complete_langs)} complete translations:")
    # Print in columns
    cols = 5
    for i in range(0, len(complete_langs), cols):
        print("  " + "  ".join(f"{l:6}" for l in complete_langs[i:i+cols]))
        
    # Initialize Byte Tokenizer
    tokenizer = GenesisTokenizer(type='byte')
    
    # Interactive Language Selection
    target_languages = None
    print("\n[?] Which languages should be included?")
    print("    - Enter 'all' for all complete translations")
    print("    - Enter short codes comma-separated (e.g., 'en, sv, de')")
    
    choice = input("\n    Selection: ").strip().lower()
    
    if choice != 'all' and choice != '':
        target_languages = [l.strip() for l in choice.split(',') if l.strip()]
        # Validate selection
        valid_selected = [l for l in target_languages if l in complete_langs]
        invalid_selected = [l for l in target_languages if l not in complete_langs]
        
        if invalid_selected:
            print(f"    [WARN] Skipping invalid/incomplete languages: {invalid_selected}")
            
        if not valid_selected:
            print("    [ERROR] No valid languages selected. Defaulting to ALL.")
            target_languages = complete_langs
        else:
            target_languages = valid_selected
            print(f"    Selected: {target_languages}")
    else:
        target_languages = complete_langs
        print("    Processing ALL complete translations.")
            
    # Always overwrite
    if cache_path.exists():
        print(f"\n[INFO] Overwriting existing cache at {cache_path}")
        
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
