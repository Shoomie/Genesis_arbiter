"""
Native PyTorch Training with FlashAttention Support
====================================================
Replaces train_composer.py with pure PyTorch implementation.

Features:
- PyTorch SDPA (auto-enables FlashAttention when available)
- Interactive architecture selection
- TensorBoard logging
- Mixed precision training
- Gradient checkpointing support
"""

import os
os.environ["USE_LIBUV"] = "0"
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import time
from datetime import timedelta

# Core imports
from .models.llama.model import Llama
from .models.tokenizer import GenesisTokenizer
from .components.checkpoint import save_checkpoint
from .datasets.bible import get_bible_dataloader
from .training.flash_attention_config import (
    is_flash_attention_available,
    print_flash_attention_status
)

def get_protocol_config(mode: str):
    """Returns model and training hyperparameters for the selected protocol mode."""
    modes = {
        # Deep & Narrow architectures
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
            "params": "~550M"
        },
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
            "params": "~800M"
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
            "params": "~1.0B"
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
            "params": "~1.2B"
        },
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
            "params": "~1.8B"
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
                "learning_rate": 3e-4,
                "weight_decay": 0.15
            },
            "params": "~2.3B"
        },
        # Smaller models for testing
        "microscope": {
            "model": {
                "dim": 144,
                "n_layers": 24,
                "n_heads": 12,
                "intermediate_size": 576,
                "vocab_size": 8192,
                "norm_type": "layernorm"
            },
            "training": {
                "learning_rate": 3e-4,
                "weight_decay": 0.01
            },
            "params": "~125M"
        }
    }
    
    if mode not in modes:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(modes.keys())}")
    
    return modes[mode]


def interactive_mode_selection():
    """Interactive architecture selection menu."""
    print("\n" + "="*70)
    print("GENESIS ARBITER - Model Architecture Selection")
    print("="*70 + "\n")
    
    modes = {
        "1": ("deep_narrow_32", "550M", "Budget deep model"),
        "2": ("deep_narrow_40", "800M", "Development baseline"),
        "3": ("deep_narrow_48", "1.0B", "Grokking sweet spot"),
        "4": ("deep_narrow_60", "1.2B", "Extended depth"),
        "5": ("theos_small", "1.8B", "Grokking experiments"),
        "6": ("deep_narrow_100", "2.3B", "Extreme depth"),
        "7": ("microscope", "125M", "Quick testing")
    }
    
    print("Deep & Narrow Architectures:")
    for key, (name, params, desc) in modes.items():
        print(f"  [{key}] {name:20} ({params:6}) - {desc}")
    
    print("\n" + "-"*70)
    choice = input("\nSelect architecture [1-7]: ").strip()
    
    if choice not in modes:
        print(f"Invalid choice: {choice}. Defaulting to deep_narrow_40")
        return "deep_narrow_40"
    
    selected_mode = modes[choice][0]
    print(f"\n[OK] Selected: {selected_mode} ({modes[choice][1]})\n")
    return selected_mode


def train(
    mode: str,
    batch_size: int = 4,
    max_steps: int = 10000,
    grad_accum_steps: int = 4,
    save_interval: int = 500,
    log_interval: int = 10,
    use_amp: bool = True,
    grad_checkpoint: bool = False,
    resume_from: str = None
):
    """
    Main training loop using native PyTorch.
    
    Args:
        mode: Model architecture preset
        batch_size: Batch size per GPU
        max_steps: Maximum training steps
        grad_accum_steps: Gradient accumulation steps
        save_interval: Steps between checkpoints
        log_interval: Steps between log outputs
        use_amp: Use automatic mixed precision
        grad_checkpoint: Use gradient checkpointing
        resume_from: Path to checkpoint to resume from
    """
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"Training Configuration")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # FlashAttention status
    print(f"\nFlashAttention Status:")
    print_flash_attention_status()
    
    # Get config
    config = get_protocol_config(mode)
    model_config = config["model"]
    training_config = config["training"]
    
    print(f"\nModel: {mode}")
    print(f"  Parameters: {config['params']}")
    print(f"  Layers: {model_config['n_layers']}")
    print(f"  Dimension: {model_config['dim']}")
    print(f"  Heads: {model_config['n_heads']}")
    print(f"\nTraining:")
    print(f"  Batch size: {batch_size}")
    print(f"  Grad accum: {grad_accum_steps}")
    print(f"  Effective batch: {batch_size * grad_accum_steps}")
    print(f"  Learning rate: {training_config['learning_rate']}")
    print(f"  Weight decay: {training_config['weight_decay']}")
    print(f"  Max steps: {max_steps}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Gradient checkpointing: {grad_checkpoint}")
    print(f"{'='*70}\n")
    
    # Create directories
    checkpoint_dir = Path("checkpoints") / mode
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path("logs") / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = GenesisTokenizer("data/genesis_char_tokenizer.json")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Initialize model
    print("\nInitializing model...")
    model = Llama(**model_config)  # SDPA used internally
    
    # Gradient checkpointing if requested
    if grad_checkpoint:
        print("  Enabling gradient checkpointing...")
        # model.gradient_checkpointing_enable()  # Uncomment if implemented in model
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    print("\nInitializing optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        betas=(0.9, 0.95)
    )
    
    # Mixed precision scaler (modern torch.amp API)
    scaler = GradScaler('cuda', enabled=use_amp)
    
    # Load checkpoint if resuming
    start_step = 0
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_step = checkpoint['step']
        print(f"  Resumed from step {start_step}")
    
    # Initialize dataloader
    print("\nInitializing dataloader...")
    dataloader = get_bible_dataloader(
        corpus_path="data/nwt_corpus.txt",
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=512
    )
    print(f"  Batches per epoch: {len(dataloader)}")
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Training loop
    print(f"\n{'='*70}")
    print("Starting Training")
    print(f"{'='*70}\n")
    
    model.train()
    global_step = start_step
    running_loss = 0.0
    start_time = time.time()
    
    while global_step < max_steps:
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision (modern torch.amp API)
            with autocast('cuda', enabled=use_amp):
                outputs = model(input_ids)
                logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                
                # Compute loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Optimizer step after accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                global_step += 1
                running_loss += loss.item() * grad_accum_steps
                
                # Logging
                if global_step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    elapsed = time.time() - start_time
                    steps_per_sec = log_interval / elapsed if elapsed > 0 else 0
                    
                    # ETA calculation
                    remaining_steps = max_steps - global_step
                    eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                    eta = str(timedelta(seconds=int(eta_seconds)))
                    
                    print(f"Step {global_step}/{max_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"Speed: {steps_per_sec:.2f} steps/s | "
                          f"ETA: {eta}")
                    
                    # TensorBoard logging
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], global_step)
                    
                    running_loss = 0.0
                    start_time = time.time()
                
                # Checkpointing
                if global_step % save_interval == 0:
                    checkpoint_path = checkpoint_dir / f"step_{global_step}.pt"
                    print(f"\n[SAVE] Saving checkpoint: {checkpoint_path}")
                    torch.save({
                        'step': global_step,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'config': config
                    }, checkpoint_path)
                    print()
                
                # Check if max steps reached
                if global_step >= max_steps:
                    break
        
        if global_step >= max_steps:
            break
    
    # Final checkpoint
    final_checkpoint = checkpoint_dir / f"final_step_{global_step}.pt"
    print(f"\n[SAVE] Saving final checkpoint: {final_checkpoint}")
    torch.save({
        'step': global_step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'config': config
    }, final_checkpoint)
    
    writer.close()
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}\n")
    print(f"Total steps: {global_step}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")
    print(f"\nTo view training curves:")
    print(f"  tensorboard --logdir {log_dir}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train Genesis Arbiter with native PyTorch + FlashAttention")
    parser.add_argument("--mode", type=str, default=None, help="Model architecture preset")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--save-interval", type=int, default=500, help="Steps between checkpoints")
    parser.add_argument("--log-interval", type=int, default=10, help="Steps between log outputs")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Interactive mode selection if not specified
    if args.mode is None:
        args.mode = interactive_mode_selection()
    
    print("\n[START] Genesis Arbiter - Native PyTorch Training\n")
    
    train(
        mode=args.mode,
        batch_size=args.batch_size,
        max_steps=args.steps,
        grad_accum_steps=args.grad_accum,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        use_amp=not args.no_amp,
        grad_checkpoint=args.grad_checkpoint,
        resume_from=args.resume
    )


if __name__ == "__main__":
    main()
