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
from genesis.training.config import TrainingConfig, ModelConfig
from genesis.utils.config_loader import get_config_section, resolve_vocab_size, get_data_path

# Trainer
from genesis.training.trainer import GenesisTrainer

# Models & Data
from genesis.models.llama.model import Llama
from genesis.models.multi_task_wrapper import MultiTaskLlama
from genesis.models.tokenizer import GenesisTokenizer
from genesis.datasets.multi_task_sampler import get_multi_task_dataloader
from genesis.training.flash_attention_config import print_flash_attention_status

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

    # 1b. Load model section for specific resolution
    model_cfg_dict = get_config_section("model")

    # Create config object for resolution
    structured_config = {
        "model": model_cfg_dict,
        "data": data_cfg_dict,
        "training": train_cfg_dict
    }
    
    # Ensure vocab_size is present and set to 'auto' if missing
    if "vocab_size" not in structured_config["model"]:
        structured_config["model"]["vocab_size"] = "auto"
    
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
        "microscope": ModelConfig(dim=144, n_layers=144, n_heads=12, intermediate_size=1536, vocab_size=0),
        "deep_narrow_40": ModelConfig(dim=384, n_layers=40, n_heads=12, intermediate_size=1536, vocab_size=0),
        "standard": ModelConfig(dim=512, n_layers=12, n_heads=8, intermediate_size=2048, vocab_size=0)
    }
    
    if mode not in archs:
        print(f"Warning: Unknown mode '{mode}', defaulting to 'standard'")
        mode = "standard"
        
    cfg = archs[mode]
    
    if vocab_size is not None:
        print(f"[Model] Overriding vocab_size {cfg.vocab_size} -> {vocab_size}")
        cfg.vocab_size = vocab_size
    
    
    # Base Llama
    base_model = Llama(
        vocab_size=cfg.vocab_size,
        n_layers=cfg.n_layers,
        dim=cfg.dim,
        n_heads=cfg.n_heads,
        intermediate_size=cfg.intermediate_size,
        max_seq_len=1024,
        norm_type=cfg.norm_type
    )
    
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
    
    # 2. Tokenizer & Data Loader (VRAM Resident)
    print("Initializing Byte-Level GPU Data Pipeline...")
    
    try:
        cache_path = get_data_path("cache_path")
        if not cache_path.exists():
            print(f"[ERROR] Data cache not found at {cache_path}. Please run option 3c first.")
            return
            
        print(f"Loading pre-tokenized data from {cache_path}...")
        cache_data = torch.load(cache_path, map_location='cpu')
        
        # Auto-detect languages
        languages = list(cache_data['locale_map'].keys())
        print(f"[DATA] Detected languages in cache: {', '.join(languages)}")
        
        # Initialize Tokenizer (Always Byte-Level for this pipeline)
        tokenizer = GenesisTokenizer(type='byte')
        
        # Override vocab size in model config based on tokenizer
        model_config.vocab_size = tokenizer.vocab_size
        print(f"[TOKENIZER] Byte-level vocab size: {model_config.vocab_size}")
        
        # Load task distribution
        task_dist = get_config_section("tasks") or {
            'lm': 0.70, 'coherence': 0.15, 'cross_ref': 0.075, 'paraphrase': 0.075
        }
        
        # Initialize GPU-Resident Loader
        from genesis.datasets.byte_loader import InfiniteGPULoader
        train_loader = InfiniteGPULoader(
            data_tensor=cache_data['tokens'],
            verse_indices=cache_data['indices'],
            locale_map=cache_data['locale_map'],
            batch_size=train_config.batch_size,
            max_seq_len=train_config.max_seq_len,
            task_distribution=task_dist,
            device=torch.device(train_config.device)
        )
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize data pipeline: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Model
    print(f"Initializing Model ({args.mode})...")
    model = get_model(args.mode, train_config.device, vocab_size=model_config.vocab_size)
    
    # 4. Trainer
    trainer = GenesisTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        config=train_config
    )
    
    # Resume?
    if args.resume:
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
