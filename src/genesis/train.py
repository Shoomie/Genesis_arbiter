"""
Genesis Multi-Task Master Trainer
=================================
Consolidated training engine for the Genesis Arbiter project.
Integrates Multi-Task Learning, FlashAttention, and Grokking Detection.
Now refactored to use the modular 'GenesisTrainer' architecture.
"""

import os
from typing import Tuple, Optional
import argparse
import torch
import torch.nn as nn
from pathlib import Path

# Config
from .training.config import TrainingConfig, ModelConfig
from .utils.config_loader import get_config_section, resolve_vocab_size, get_data_path

# Trainer
from .training.trainer import GenesisTrainer

# Models & Data
from .models.llama.model import Transformer, ModelArgs
from .models.multi_task_wrapper import MultiTaskLlama
from .models.tokenizer import GenesisTokenizer
from .datasets.multi_task_sampler import get_multi_task_dataloader
from .training.flash_attention_config import print_flash_attention_status

def load_global_config_into_training_config(args) -> Tuple[TrainingConfig, ModelConfig]:
    """Load settings from genesis_config.toml and override with args."""
    
    # 1. Load TOML defaults
    train_cfg_dict = get_config_section("training")
    sys_cfg_dict = get_config_section("system")
    eval_cfg_dict = get_config_section("evaluation")
    data_cfg_dict = get_config_section("data")
    
    # Merge into a single dict for TrainingConfig
    merged_config = {}
    merged_config.update(train_cfg_dict)
    merged_config.update(sys_cfg_dict)
    merged_config.update(eval_cfg_dict)
    merged_config.update(data_cfg_dict)

    # Create config object
    # We call resolve_vocab_size on the merged dict (mimicking structure expected by resolver)
    # Resolver expects {'model': {...}, 'data': {...}}
    # But merged_config is flat.
    # Re-construct a structured dict for resolution
    structured_config = {
        "model": {"vocab_size": merged_config.get("vocab_size", "auto") if "vocab_size" in merged_config else "auto"}, # Default to auto if not present?
        "data": data_cfg_dict, # Use original dicts
        "training": train_cfg_dict
    }
    
    # Actually, resolve_vocab_size reads 'vocab_size' from model section.
    # If the TOML 'model' section was merged into merged_config, we might have lost the structure.
    # We should load 'model' section separately.
    model_cfg_dict = get_config_section("model")
    structured_config["model"].update(model_cfg_dict)
    
    # Resolve
    resolved = resolve_vocab_size(structured_config)
    resolved_vocab_size = resolved["model"].get("vocab_size")
    
    # Merge resolved back into flat config if needed, or just set it on ModelConfig
    if resolved_vocab_size:
        merged_config["vocab_size"] = resolved_vocab_size
        model_cfg_dict["vocab_size"] = resolved_vocab_size

    # Create Objects
    train_config = TrainingConfig.from_dict(merged_config)
    
    # For ModelConfig, merge CLI overrides if any? (CLI doesn't usually set dims)
    # But we should rely on TOML [model] + dynamic logical
    model_config = ModelConfig.from_dict(model_cfg_dict)
    
    return train_config, model_config

def get_model(mode: str, device: str, vocab_size: Optional[int] = None) -> MultiTaskLlama:
    """Initialize the model architecture."""
    # Define architectures
    archs = {
        "microscope": ModelConfig(dim=144, n_layers=144, n_heads=12, intermediate_size=1536, vocab_size=8000),
        "deep_narrow_40": ModelConfig(dim=384, n_layers=40, n_heads=12, intermediate_size=1536, vocab_size=16000),
        "standard": ModelConfig(dim=512, n_layers=12, n_heads=8, intermediate_size=2048, vocab_size=16000)
    }
    
    if mode not in archs:
        print(f"Warning: Unknown mode '{mode}', defaulting to 'standard'")
        mode = "standard"
        
    cfg = archs[mode]
    
    if vocab_size is not None:
        print(f"[Model] Overriding vocab_size {cfg.vocab_size} -> {vocab_size}")
        cfg.vocab_size = vocab_size
    
    
    # Base Llama
    model_args = ModelArgs(
        dim=cfg.dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        vocab_size=cfg.vocab_size,
        multiple_of=256,
        norm_eps=1e-5,
        max_seq_len=1024
    )
    
    base_model = Transformer(model_args)
    
    # Multi-Task Wrapper
    model = MultiTaskLlama(
        base_model,
        dim=cfg.dim,
        vocab_size=cfg.vocab_size
    )
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Genesis Master Trainer")
    parser.add_argument("--mode", type=str, default="standard", help="Model architecture mode")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--compile", action="store_true", help="Compile model")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Use gradient checkpointing")
    
    # Hyperparam overrides
    parser.add_argument("--lr", type=float, dest="learning_rate")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--val-interval", type=int)
    parser.add_argument("--eval-interval", type=int)
    
    args = parser.parse_args()
    
    # 1. Load Configuration
    train_config, model_config = load_global_config_into_training_config(args)
    print(f"Loaded Configuration:\n{train_config}")
    print(f"Model Configuration:\n{model_config}")
    
    # 2. Tokenizer
    print("Loading Tokenizer...")
    try:
        # Use get_data_path to correctly resolve from config or default
        tokenizer_path = get_data_path("tokenizer_path")
        if not tokenizer_path.exists():
            print(f"Warning: Tokenizer not found at {tokenizer_path}")
    except Exception as e:
        # Fallback logic if needed, or crash
        print(f"Error resolving tokenizer path: {e}")
        # Legacy fallback
        tokenizer_path = Path(train_config.bible_dir) / "tokenizers" / "genesis_char_tokenizer.json"
        
    print(f"Using tokenizer: {tokenizer_path}")
    tokenizer = GenesisTokenizer(str(tokenizer_path))
    
    # 3. Data Loader
    print("Initializing Data Loader...")
    train_loader = get_multi_task_dataloader(
        bible_data_dir=train_config.bible_dir,
        tokenizer=tokenizer,
        batch_size=train_config.batch_size,
        max_seq_len=1024, # Could be in config
        device=torch.device(train_config.device)
    )
    
    # 4. Model
    print(f"Initializing Model ({args.mode})...")
    # Pass dynamically resolved vocab size
    model = get_model(args.mode, train_config.device, vocab_size=model_config.vocab_size)
    
    # 5. Trainer
    trainer = GenesisTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        config=train_config
    )
    
    # Resume?
    if args.resume:
        # Find latest
        cp_dir = Path("checkpoints")
        if cp_dir.exists():
            checkpoints = sorted(list(cp_dir.glob("step_*.pt")), key=os.path.getmtime)
            if checkpoints:
                latest = checkpoints[-1]
                trainer.load_checkpoint(latest)
            else:
                print("No checkpoints found to resume from.")
    
    # Start
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving emergency checkpoint...")
        trainer.save_checkpoint("checkpoints/interrupted.pt")

if __name__ == "__main__":
    main()
