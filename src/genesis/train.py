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
from genesis.utils.config_loader import get_config_section, get_data_path

# Trainer
from genesis.training.trainer import GenesisTrainer

# Models & Data
from genesis.models.llama.model import Llama
from genesis.models.multi_task_wrapper import MultiTaskLlama
from genesis.models.tokenizer import GenesisTokenizer
from genesis.datasets.multi_task_sampler import get_multi_task_dataloader
from genesis.training.flash_attention_config import print_flash_attention_status

# Define global architectural presets
ARCH_PRESETS = {
    "microscope": ModelConfig(dim=288, n_layers=144, n_heads=12, intermediate_size=768, vocab_size=260, norm_type="deepnorm"),
    "deep_narrow_40": ModelConfig(dim=384, n_layers=40, n_heads=12, intermediate_size=1536, vocab_size=260, norm_type="deepnorm"),
    "standard": ModelConfig(dim=512, n_layers=12, n_heads=8, intermediate_size=2048, vocab_size=260, norm_type="deepnorm")
}

def load_global_config_into_training_config(args) -> Tuple[TrainingConfig, ModelConfig]:
    """Load settings from genesis_config.toml and override with args."""
    
    # 1. Load TOML sections
    train_cfg_dict = get_config_section("training")
    sys_cfg_dict = get_config_section("system")
    eval_cfg_dict = get_config_section("evaluation")
    data_cfg_dict = get_config_section("data")
    model_cfg_dict = get_config_section("model")
    
    # Merge into a single dict for TrainingConfig
    merged_config = {}
    merged_config.update(train_cfg_dict)
    merged_config.update(sys_cfg_dict)
    merged_config.update(eval_cfg_dict)
    merged_config.update(data_cfg_dict)

    # Create Objects
    train_config = TrainingConfig.from_dict(merged_config)
    
    # Merge CLI overrides for training
    if args.learning_rate: train_config.learning_rate = args.learning_rate
    if args.weight_decay: train_config.weight_decay = args.weight_decay
    if args.batch_size: train_config.batch_size = args.batch_size
    if args.steps: train_config.max_steps = args.steps
    if args.val_interval: train_config.val_interval = args.val_interval
    if args.eval_interval: train_config.eval_interval = args.eval_interval
    if args.use_cuda_graph: train_config.use_cuda_graph = True
    
    # CLI Overrides for WWM
    if args.wwm_trigger: train_config.wwm_trigger_steps = args.wwm_trigger
    if args.wwm_window: train_config.wwm_window = args.wwm_window
    if args.wwm_threshold: train_config.wwm_threshold = args.wwm_threshold
    
    # --- Model Resolution ---
    # Start with standard fallback
    mode = args.mode or train_cfg_dict.get("mode", "standard")
    base_cfg = ARCH_PRESETS.get(mode, ARCH_PRESETS["standard"])
    
    # Override with [model] section from TOML
    model_config = base_cfg.merge(model_cfg_dict)
    
    # Handle auto settings
    if model_cfg_dict.get("intermediate_size") == "auto":
        model_config.intermediate_size = model_config.dim * 4
        
    # Sync max_seq_len (model vs data)
    # Default to data's max_seq_len if not explicitly set in [model]
    if "max_seq_len" not in model_cfg_dict:
        model_config.max_seq_len = train_config.max_seq_len
    else:
        # If set in [model], ensure training loop matches it
        train_config.max_seq_len = model_config.max_seq_len

    return train_config, model_config

def get_model(config: ModelConfig, device: str) -> MultiTaskLlama:
    """Initialize the model architecture from a ModelConfig."""
    
    # Base Llama
    base_model = Llama(
        vocab_size=config.vocab_size,
        n_layers=config.n_layers,
        dim=config.dim,
        n_heads=config.n_heads,
        intermediate_size=config.intermediate_size,
        max_seq_len=config.max_seq_len,
        norm_type=config.norm_type
    )
    
    # Multi-Task Wrapper
    model = MultiTaskLlama(
        base_model,
        dim=config.dim,
        vocab_size=config.vocab_size
    )
    
    # Attach config for checkpointing
    model.config = config
    
    return model

def main():
    parser = argparse.ArgumentParser(description="Genesis Master Trainer")
    parser.add_argument("--mode", type=str, default="standard", help="Model architecture mode")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--compile", action="store_true", help="Compile model")
    parser.add_argument("--use-cuda-graph", action="store_true", help="Use CUDA Graphs")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Use gradient checkpointing")
    
    # Hyperparam overrides
    parser.add_argument("--lr", type=float, dest="learning_rate")
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--val-interval", type=int)
    parser.add_argument("--eval-interval", type=int)
    
    # WWM Trigger Overrides
    parser.add_argument("--wwm-trigger", type=int, help="Steps to wait before checking for plateau")
    parser.add_argument("--wwm-window", type=int, help="Window size for loss comparison")
    parser.add_argument("--wwm-threshold", type=float, help="Improvement threshold (0.005 = 0.5%%)")
    
    args = parser.parse_args()
    
    # 1. Load Configuration
    train_config, model_config = load_global_config_into_training_config(args)
    # Silent config load, only error if something is wrong
    
    # 2. Tokenizer & Data Loader (VRAM Resident)
    print("  [PIPELINE] Initializing Byte-Level GPU Data Pipeline...")
    
    try:
        cache_path = get_data_path("cache_path")
        if not cache_path.exists():
            print(f"  [ERROR] Data cache not found at {cache_path}. Run option 3c first.")
            return
            
        cache_data = torch.load(cache_path, map_location='cpu')
        
        # Auto-detect languages
        languages = list(cache_data['locale_map'].keys())
        print(f"  [DATA] Detected languages: {', '.join(languages)}")
        
        # Initialize Tokenizer (Always Byte-Level for this pipeline)
        tokenizer = GenesisTokenizer(type='byte')
        
        # Override vocab size in model config based on tokenizer
        model_config.vocab_size = tokenizer.vocab_size
        
        # Load task distribution
        task_dist = get_config_section("tasks") or {
            'lm': 0.70, 'coherence': 0.15, 'cross_ref': 0.075, 'paraphrase': 0.075
        }
        
        # Initialize GPU-Resident Loader
        from genesis.datasets.byte_loader import InfiniteGPULoader, BackgroundPrefetcher
        raw_loader = InfiniteGPULoader(
            data_tensor=cache_data['tokens'],
            verse_indices=cache_data['indices'],
            locale_map=cache_data['locale_map'],
            batch_size=train_config.batch_size,
            max_seq_len=train_config.max_seq_len,
            task_distribution=task_dist,
            device=torch.device(train_config.device)
        )
        train_loader = BackgroundPrefetcher(raw_loader)
        
    except Exception as e:
        print(f"  [ERROR] Data pipeline failure: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. Model
    print(f"  [MODEL] Initializing {args.mode.upper() if args.mode else 'TOML'} architecture...")
    model = get_model(model_config, train_config.device)
    
    # Enable gradient checkpointing if configured
    if train_config.gradient_checkpointing or args.gradient_checkpointing:
        print("  [SYSTEM] Gradient Checkpointing enabled.")
        model.base.gradient_checkpointing_enable()
    
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
