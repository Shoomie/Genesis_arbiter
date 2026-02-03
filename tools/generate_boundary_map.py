import os
import json
import torch
import re
from tqdm import tqdm
import sys
from pathlib import Path

# Add src to path for internal imports
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
if str(project_root / "src") not in sys.path:
    sys.path.append(str(project_root / "src"))

from genesis.utils.config_loader import get_data_path, find_project_root

def get_word_starts(text):
    """Identify starting indices of words in a string."""
    # Pattern includes alphanumeric, underscores, and single quotes
    pattern = re.compile(r"[\w']+")
    return [match.start() for match in pattern.finditer(text)]

def generate_boundaries(text):
    """Generate a byte-level mask for word starts."""
    byte_data = text.encode('utf-8')
    mask = torch.zeros(len(byte_data), dtype=torch.uint8)
    
    if not text:
        return mask
        
    char_to_byte_offset = []
    current_byte = 0
    for char in text:
        char_to_byte_offset.append(current_byte)
        current_byte += len(char.encode('utf-8'))
        
    word_starts = get_word_starts(text)
    for char_idx in word_starts:
        byte_idx = char_to_byte_offset[char_idx]
        mask[byte_idx] = 1
        
    return mask

import concurrent.futures

def process_language_worker(lang, bible_path):
    """Worker function to process a single language directory."""
    bible_file = Path(bible_path) / lang / "bible_data.json"
    if not bible_file.exists():
        return []
        
    try:
        with open(bible_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if len(data) != 1189:
            return []
            
        lang_masks = []
        for chapter in data:
            content = chapter.get('content', '')
            if content:
                mask = generate_boundaries(content)
                lang_masks.append(mask)
        return lang_masks
    except Exception as e:
        print(f"Error processing {lang}: {e}")
        return []

def process_bible_dir(bible_dir, target_languages=None):
    """Processes the same directories as the main data pipeline in parallel."""
    bible_path = Path(bible_dir)
    
    # We follow the exact same alphabetical ordering as MultiTaskDataset
    langs = sorted([d.name for d in bible_path.iterdir() if d.is_dir()])
    if target_languages:
        langs = [l for l in langs if l in target_languages]
        
    print(f"Generating boundaries for {len(langs)} languages using multiprocessing...")
    
    all_masks = []
    # Use ProcessPoolExecutor for CPU-bound string processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a list of futures to maintain order
        future_to_lang = {executor.submit(process_language_worker, lang, bible_dir): lang for lang in langs}
        
        # Collect results in original sorted order
        for lang in tqdm(langs):
            # Find the future for this lang
            # (We could just use executor.map, but submitting manually handles exceptions/filtering better)
            for future, l in future_to_lang.items():
                if l == lang:
                    res = future.result()
                    all_masks.extend(res)
                    break
                
    if not all_masks:
        return torch.tensor([], dtype=torch.uint8)
        
    return torch.cat(all_masks)

if __name__ == "__main__":
    try:
        # Use centralized config loader to resolve paths
        BIBLE_DIR = get_data_path("bible_dir")
        CACHE_PATH = get_data_path("cache_path")
        OUTPUT_PATH = CACHE_PATH.parent / "genesis_boundaries.pt"
        
        print(f"Project Root: {find_project_root()}")
        print(f"Bible Dir:    {BIBLE_DIR}")
        print(f"Data Cache:   {CACHE_PATH}")
        print(f"Output File:  {OUTPUT_PATH}")
        print("-" * 50)
        
    except Exception as e:
        print(f"[ERROR] Configuration failed: {e}")
        sys.exit(1)
    
    # Auto-detect target languages from existing cache to ensure perfect alignment
    target_langs = None
    if CACHE_PATH.exists():
        print(f"Loading cache to detect target languages...")
        try:
            cache_data = torch.load(CACHE_PATH, map_location='cpu', weights_only=False)
            target_langs = list(cache_data['locale_map'].keys())
            print(f"[OK] Detected active languages in cache: {target_langs}")
        except Exception as e:
            print(f"[ERROR] Failed to read cache file: {e}")
            sys.exit(1)
    else:
        print(f"[CAUTION] No data cache found at {CACHE_PATH}.")
        print("          It is highly recommended to run Option 3c first to define your training subset.")
        print("          Fallback: Processing ALL languages found in Bible directory...")
    
    if not OUTPUT_PATH.parent.exists():
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        
    boundaries = process_bible_dir(BIBLE_DIR, target_languages=target_langs)
    
    if boundaries.numel() > 0:
        print(f"Final boundary tensor size: {len(boundaries)} bytes")
        torch.save(boundaries, OUTPUT_PATH)
        print(f"[SUCCESS] Boundaries saved to {OUTPUT_PATH}")
    else:
        print("[ERROR] No boundaries generated. Check your data and cache status.")
