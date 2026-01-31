import os
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
        
    p = Path(path_str)
    return p if p.is_absolute() else project_root / p
