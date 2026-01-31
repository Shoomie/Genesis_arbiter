"""
Composer-based Training Script for Genesis Arbiter

Integrates FlashAttention, multi-task learning, and grokking detection
while maintaining backward compatibility with existing configurations.
"""

import os
os.environ["USE_LIBUV"] = "0"

import torch
import torch.distributed as dist
import toml
import argparse
from pathlib import Path
import time
from typing import Optional

# Composer imports
try:
    from composer import Trainer, algorithms
    from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
    from composer.loggers import TensorBoardLogger
    from composer.models import ComposerModel
    from composer.core import DataSpec
    COMPOSER_AVAILABLE = True
except ImportError:
    COMPOSER_AVAILABLE = False
    print("[ERROR] Composer not installed. Please run: pip install mosaicml")
    exit(1)

# Core imports
from .models.llama.model import Llama
from .models.tokenizer import GenesisTokenizer
from .datasets.bible import get_bible_dataloader
from .components.checkpoint import save_checkpoint
from .training.flash_attention_config import FlashAttentionConfig


class GenesisComposerModel(ComposerModel):
    """
    Wrapper around Genesis Llama model to make it Composer-compatible.
    
    This minimal wrapper allows us to use Composer's training infrastructure
    while keeping our custom model architecture unchanged.
    """
    
    def __init__(self, model: Llama):
        super().__init__()
        self.model = model
    
    def forward(self, batch):
        """
        Forward pass for Composer training.
        
        Args:
            batch: Dictionary with 'tokens' and 'labels' keys
        
        Returns:
            Tuple of (logits, loss)
        """
        tokens = batch['tokens']
        labels = batch['labels']
        logits, loss = self.model(tokens, labels)
        return logits, loss
    
    def loss(self, outputs, batch):
        """
        Extract loss from forward pass outputs.
        
        Composer expects this method to return the loss value.
        """
        logits, loss = outputs
        return loss
    
    def eval_forward(self, batch, outputs=None):
        """
        Forward pass for evaluation (same as training for now).
        """
        return self.forward(batch)
    
    def get_metrics(self, is_train: bool = False):
        """
        Return metrics to track during training/evaluation.
        """
        return {}


def get_protocol_config(mode: str):
    """
    Returns model and training hyperparameters for the selected protocol mode.
    
    This is the same as train.py to maintain compatibility.
    """
    modes = {
        # Phase 3: Deep & Narrow Topologies (500M-1B range)
        "deep_narrow_40": {
            "model": {
                "dim": 768,
                "n_layers": 40,
                "n_heads": 12,
                "intermediate_size": 3072,
                "vocab_size": 8192,
                "norm_type": "deepnorm"
            },
            "training": {
                "learning_rate": 2e-4,
                "weight_decay": 0.08
            },
            "params": "~800M",
            "purpose": "Budget deep model for quick experiments"
        },
        "deep_narrow_32": {
            "model": {
                "dim": 640,
                "n_layers": 32,
                "n_heads": 10,
                "intermediate_size": 2560,
                "vocab_size": 8192,
                "norm_type": "deepnorm"
            },
            "training": {
                "learning_rate": 2e-4,
                "weight_decay": 0.08
            },
            "params": "~550M",
            "purpose": "Lower-end deep architecture for faster iteration"
        },
        "deep_narrow_48": {
            "model": {
                "dim": 896,
                "n_layers": 48,
                "n_heads": 14,
                "intermediate_size": 3584,
                "vocab_size": 8192,
                "norm_type": "deepnorm"
            },
            "training": {
                "learning_rate": 2e-4,
                "weight_decay": 0.1
            },
            "params": "~1.0B",
            "purpose": "1B parameter sweet spot for grokking"
        },
        
        # Phase 3: Deep & Narrow Topologies (1B+ range)
        "theos_small": {
            "model": {
                "dim": 1024, 
                "n_layers": 80, 
                "n_heads": 16, 
                "intermediate_size": 4096,
                "vocab_size": 8192,
                "norm_type": "deepnorm"
            },
            "training": {
                "learning_rate": 3e-4,
                "weight_decay": 0.1
            },
            "params": "~1.8B",
            "purpose": "Grokking experiments with extended training"
        },
        "deep_narrow_60": {
            "model": {
                "dim": 768,
                "n_layers": 60,
                "n_heads": 12,
                "intermediate_size": 3072,
                "vocab_size": 8192,
                "norm_type": "deepnorm"
            },
            "training": {
                "learning_rate": 2e-4,
                "weight_decay": 0.1
            },
            "params": "~1.2B",
            "purpose": "Lighter deep architecture"
        },
        "deep_narrow_100": {
            "model": {
                "dim": 1024,
                "n_layers": 100,
                "n_heads": 16,
                "intermediate_size": 4096,
                "vocab_size": 8192,
                "norm_type": "deepnorm"
            },
            "training": {
                "learning_rate": 1e-4,
                "weight_decay": 0.15
            },
            "params": "~2.3B",
            "purpose": "Extreme depth for reasoning"
        },
        
        # Legacy Phase 1-2 Architectures
        "microscope": {
            "model": {
                "dim": 768, 
                "n_layers": 12, 
                "n_heads": 12, 
                "intermediate_size": 3072, 
                "vocab_size": 8000,
                "norm_type": "rmsnorm"
            },
            "training": {"learning_rate": 3e-4},
            "params": "125.5M",
            "purpose": "Baseline comparisons"
        },
        "tower_of_truth": {
            "model": {
                "dim": 288, 
                "n_layers": 144, 
                "n_heads": 12, 
                "intermediate_size": 576, 
                "vocab_size": 12000,
                "norm_type": "rmsnorm"
            },
            "training": {"learning_rate": 1e-4},
            "params": "~5-8M",
            "purpose": "Extreme depth experiment (legacy)"
        },
        "high_res_arbiter": {
            "model": {
                "dim": 1024, 
                "n_layers": 24, 
                "n_heads": 16, 
                "intermediate_size": 4096, 
                "vocab_size": 8000,
                "norm_type": "rmsnorm"
            },
            "training": {"learning_rate": 2e-4},
            "params": "~180M",
            "purpose": "Semantic resolution (legacy)"
        }
    }
    return modes.get(mode.lower())


def find_tokenizer(vocab_size: int = 8192):
    """Find appropriate tokenizer from arbiter_tokenizer_factory output."""
    tokenizer_dir = Path("../tokenizers")
    
    # Try to find arbiter tokenizer matching vocab size
    if tokenizer_dir.exists():
        model_file = tokenizer_dir / f"arbiter_nwt_{vocab_size}.model"
        if model_file.exists():
            print(f"✓ Found custom tokenizer: {model_file}")
            # For now, fall back to GenesisTokenizer
    
    # Fallback to genesis tokenizer
    if Path("data/genesis_char_tokenizer.json").exists():
        print(f"Using data/genesis_char_tokenizer.json")
        return GenesisTokenizer("data/genesis_char_tokenizer.json")
    
    raise FileNotFoundError("No tokenizer found. Run 'python ../tools/arbiter_tokenizer_factory.py nwt_corpus.txt'")


def train():
    parser = argparse.ArgumentParser(description="Genesis Arbiter Training (Composer)")
    parser.add_argument("--mode", type=str, 
                       choices=["deep_narrow_32", "deep_narrow_40", "deep_narrow_48",
                               "deep_narrow_60", "theos_small", "deep_narrow_100", 
                               "microscope", "tower_of_truth", "high_res_arbiter"], 
                       help="Select model architecture")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                       help="Directory for checkpoints and logs")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Experiment name for logging")
    parser.add_argument("--steps", type=int, default=10000,
                       help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--log-interval", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                       help="Save checkpoint every N steps")
    args = parser.parse_args()
    
    # Verify FlashAttention status
    print("\n" + "="*60)
    print("   GENESIS ARBITER TRAINING - Composer Edition")
    print("="*60)
    
    fa_config = FlashAttentionConfig()
    fa_config.print_status()
    
    # Interactive architecture selection if not provided
    selected_mode = args.mode
    if selected_mode is None:
        print("\n=== Architecture Selection ===")
        print("\n--- Phase 3: Deep & Narrow (500M-1B Range) ---")
        print("[1] Deep Narrow 32 - 32L×640D (550M params) - Budget quick experiments")
        print("[2] Deep Narrow 40 - 40L×768D (800M params) - Mid-range deep")
        print("[3] Deep Narrow 48 - 48L×896D (1.0B params) - 1B sweet spot")
        print("\n--- Phase 3: Deep & Narrow (1B+ Range) ---")
        print("[4] Deep Narrow 60 - 60L×768D (1.2B params) - Lighter deep model")
        print("[5] Theos-Small - 80L×1024D (1.8B params) - Grokking experiments")
        print("[6] Deep Narrow 100 - 100L×1024D (2.3B params) - Extreme depth")
        print("\n--- Legacy Phase 1-2 Architectures ---")
        print("[7] Microscope - 12L×768D (125M params) - Baseline")
        print("[8] Tower of Truth - 144L×288D (8M params) - Legacy deep")
        print("[9] High-Res Arbiter - 24L×1024D (180M params) - Legacy resolution")
        
        m_choice = input("\nSelect architecture [1-9]: ").strip()
        m_mapping = {
            "1": "deep_narrow_32",
            "2": "deep_narrow_40",
            "3": "deep_narrow_48",
            "4": "deep_narrow_60",
            "5": "theos_small", 
            "6": "deep_narrow_100",
            "7": "microscope",
            "8": "tower_of_truth",
            "9": "high_res_arbiter"
        }
        selected_mode = m_mapping.get(m_choice, "deep_narrow_48")
    
    mode_cfg = get_protocol_config(selected_mode)
    if not mode_cfg:
        raise ValueError(f"Unknown mode: {selected_mode}")
    
    print(f"\n✓ Selected: {selected_mode.upper()}")
    print(f"  Parameters: {mode_cfg['params']}")
    print(f"  Purpose: {mode_cfg['purpose']}")
    print(f"  Norm Type: {mode_cfg['model']['norm_type']}")
    
    # Initialize model
    model_cfg = mode_cfg["model"]
    
    # Safety: Auto-correct dim % n_heads == 0
    dim = model_cfg["dim"]
    n_heads = model_cfg["n_heads"]
    if dim % n_heads != 0:
        old_dim = dim
        dim = (dim // n_heads + 1) * n_heads
        model_cfg["dim"] = dim
        print(f"⚠ Safety Auto-Correct: Adjusting dim from {old_dim} to {dim}")
    
    print(f"\n>>> Initializing {selected_mode.upper()} with {model_cfg['norm_type']} normalization...")
    
    base_model = Llama(
        vocab_size=model_cfg["vocab_size"],
        n_layers=model_cfg["n_layers"],
        dim=model_cfg["dim"],
        n_heads=model_cfg["n_heads"],
        intermediate_size=model_cfg["intermediate_size"],
        max_seq_len=model_cfg.get("max_seq_len", 1024),
        norm_type=model_cfg.get("norm_type", "rmsnorm")
    )
    
    print(f"✓ Model initialized: {base_model.get_num_params():,} parameters")
    
    # Load tokenizer
    print(f"\nLoading tokenizer...")
    tokenizer = find_tokenizer(model_cfg["vocab_size"])
    
    # Wrap model for Composer
    composer_model = GenesisComposerModel(base_model)
    
    # Create dataloader
    print(f"\n>>> Loading dataset...")
    train_dataloader = get_bible_dataloader(
        corpus_path="data/nwt_corpus.txt",
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=model_cfg.get("max_seq_len", 1024)
    )
    
    # Wrap dataloader for Composer
    dataspec = DataSpec(
        dataloader=train_dataloader,
        split_batch=lambda batch, num_microbatches: [
            {'tokens': batch[0][i::num_microbatches], 
             'labels': batch[1][i::num_microbatches]}
            for i in range(num_microbatches)
        ],
        get_num_samples_in_batch=lambda batch: batch[0].shape[0]
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        composer_model.parameters(),
        lr=mode_cfg["training"].get("learning_rate", 3e-4),
        weight_decay=mode_cfg["training"].get("weight_decay", 0.01)
    )
    
    # Setup experiment name
    experiment_name = args.experiment_name or f"{selected_mode}_{int(time.time())}"
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create Composer Trainer
    print(f"\n>>> Initializing Composer Trainer...")
    print(f"  Experiment: {experiment_name}")
    print(f"  Total steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_accum}")
    
    trainer = Trainer(
        model=composer_model,
        train_dataloader=dataspec,
        max_duration=f"{args.steps}ba",  # ba = batches
        optimizers=optimizer,
        
        # Algorithms for optimization
        algorithms=[
            # FlashAttention is automatically used via F.scaled_dot_product_attention
            # No need to add it explicitly as an algorithm
            algorithms.GradientClipping(clipping_threshold=1.0)
        ],
        
        # Callbacks for monitoring
        callbacks=[
            LRMonitor(),
            MemoryMonitor(),
            SpeedMonitor(window_size=100)
        ],
        
        # Logging
        loggers=[
            TensorBoardLogger(log_dir=str(log_dir / "tensorboard"))
        ],
        
        # Checkpointing
        save_interval=f"{args.checkpoint_interval}ba",
        save_folder=str(Path(args.output_dir) / experiment_name),
        save_filename="checkpoint_ep{epoch}_ba{batch}.pt",
        
        # Precision and compilation
        precision="amp_bf16" if torch.cuda.is_available() else "fp32",
        device_train_microbatch_size=args.batch_size // args.grad_accum,
        
        # Display
        log_to_console=True,
        console_log_interval=f"{args.log_interval}ba",
        
        # Progress bar
        progress_bar=True
    )
    
    print(f"\n{'='*60}")
    print(f"  TRAINING START")
    print(f"{'='*60}\n")
    
    # Train!
    trainer.fit()
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
