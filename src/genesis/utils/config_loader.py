import os
import json
import toml
from pathlib import Path
from typing import Any, Dict, Optional

def find_project_root(start_path: Optional[Path] = None) -> Path:
    """Find the project root by looking for genesis_config.toml."""
    curr = Path(start_path or __file__).resolve()
    # Search up to 5 levels
    for _ in range(5):
        if (curr / "genesis_config.toml").exists():
            return curr
        if curr.parent == curr:
            break
        curr = curr.parent
    # Fallback to current working directory or two levels up from this file
    return Path(__file__).resolve().parent.parent.parent

def load_global_config(root: Optional[Path] = None) -> Dict[str, Any]:
    """Load the central genesis_config.toml."""
    project_root = root or find_project_root()
    config_path = project_root / "genesis_config.toml"
    
    if not config_path.exists():
        return {}
        
    try:
        return toml.load(config_path)
    except Exception as e:
        print(f"[WARN] Failed to load genesis_config.toml: {e}")
        return {}

def get_config_section(section: str, root: Optional[Path] = None) -> Dict[str, Any]:
    """Get a specific section from the global config."""
    config = load_global_config(root)
    return config.get(section, {})

def get_data_path(key: str, default: Optional[str] = None, root: Optional[Path] = None) -> Path:
    """Get a data path from the config, resolving it relative to project root."""
    project_root = root or find_project_root()
    data_cfg = get_config_section("data", project_root)
    
    path_str = data_cfg.get(key, default)
    if not path_str:
        raise ValueError(f"Data path key '{key}' not found in config and no default provided.")
        

def resolve_vocab_size(config: Dict[str, Any], root: Optional[Path] = None) -> Dict[str, Any]:
    """
    If vocab_size is 'auto' (or None), load it from the tokenizer file.
    Updates the config in-place and returns it.
    """
    model_cfg = config.get("model", {})
    # Check if we need to resolve
    if model_cfg.get("vocab_size") == "auto":
        project_root = root or find_project_root()
        
        # 1. Try to find tokenizer path in the config itself
        data_cfg = config.get("data", {})
        tokenizer_rel_path = data_cfg.get("tokenizer_path")
        
        # 2. If not in config, check if we can find the default genesis_char_tokenizer.json
        if not tokenizer_rel_path:
             # Look for it in data/
             default_tok = project_root / "data" / "genesis_char_tokenizer.json"
             if default_tok.exists():
                 tokenizer_rel_path = "data/genesis_char_tokenizer.json"
        
        if tokenizer_rel_path:
            tokenizer_path = project_root / tokenizer_rel_path
            if tokenizer_path.exists():
                try:
                    with open(tokenizer_path, 'r', encoding='utf-8') as f:
                        tokenizer_data = json.load(f)
                    
                    # Try to find vocab dict
                    vocab = None
                    if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
                        vocab = tokenizer_data["model"]["vocab"]
                        
                    if vocab:
                        size = len(vocab)
                        print(f"[Config] Auto-detected vocab size from {tokenizer_rel_path}: {size}")
                        model_cfg["vocab_size"] = size
                        return config
                except Exception as e:
                    print(f"[Config] Warning: Failed to parse tokenizer {tokenizer_path}: {e}")
        
        # Fallback
        print("[Config] Warning: Could not auto-detect vocab size. Using default 8192.")
        model_cfg["vocab_size"] = 8192
        
    return config
