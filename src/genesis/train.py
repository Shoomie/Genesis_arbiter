"""
Genesis Multi-Task Master Trainer
=================================
Consolidated training engine for the Genesis Arbiter project.
Integrates Multi-Task Learning, FlashAttention, and Grokking Detection.
Now refactored to use the modular 'GenesisTrainer' architecture.
"""

import os
import argparse
import torch
import torch.nn as nn
from pathlib import Path

# Config
from .training.config import TrainingConfig, ModelConfig
from .utils.config_loader import get_config_section

# Trainer
from .training.trainer import GenesisTrainer

# Models & Data
from .models.llama.model import Transformer, ModelArgs
from .models.multi_task_wrapper import MultiTaskLlama
from .models.tokenizer import GenesisTokenizer
from .datasets.multi_task_sampler import get_multi_task_dataloader
from .training.flash_attention_config import print_flash_attention_status

def load_global_config_into_training_config(args) -> TrainingConfig:
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
    
    # Normalize keys (TOML naming to Python attr naming if needed)
    # The dataclass uses lowercase, our TOML uses lowercase mostly.
    
    # 2. Override with CLI args
    if args.learning_rate: merged_config['learning_rate'] = args.learning_rate
    if args.batch_size: merged_config['batch_size'] = args.batch_size
    if args.steps: merged_config['max_steps'] = args.steps
    if args.val_interval: merged_config['val_interval'] = args.val_interval
    if args.eval_interval: merged_config['eval_interval'] = args.eval_interval
    if args.compile: merged_config['compile_model'] = True
    if args.gradient_checkpointing: merged_config['gradient_checkpointing'] = True
    
    # Create config object (safely ignoring unknown keys from TOML)
    config = TrainingConfig.from_dict(merged_config)
    
    return config

def get_model(mode: str, device: str) -> MultiTaskLlama:
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
    config = load_global_config_into_training_config(args)
    print(f"Loaded Configuration:\n{config}")
    
    # 2. Tokenizer
    print("Loading Tokenizer...")
    tokenizer_path = Path(config.bible_dir) / "tokenizers" / "genesis_8k.json"
    if not tokenizer_path.exists():
         print(f"Warning: Tokenizer not found at {tokenizer_path}, checking default location...")
         # Fallback to absolute path or other logic if needed, but for now just warn
    
    tokenizer = GenesisTokenizer(str(tokenizer_path))
    
    # 3. Data Loader
    print("Initializing Data Loader...")
    train_loader = get_multi_task_dataloader(
        bible_data_dir=config.bible_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_seq_len=1024, # Could be in config
        device=torch.device(config.device)
    )
    
    # 4. Model
    print(f"Initializing Model ({args.mode})...")
    model = get_model(args.mode, config.device)
    
    # 5. Trainer
    trainer = GenesisTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        config=config
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
