"""
Native PyTorch Multi-Task Training with Grokking Detection
===========================================================
Replaces train_multi_task.py with pure PyTorch implementation.

Features:
- Multi-task learning (LM, coherence, cross-ref, paraphrase)
- Grokking detection callbacks
- PyTorch SDPA (FlashAttention when available)
- Cross-lingual validation
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
from models.multi_task_wrapper import MultiTaskLlama
from models.tokenizer import GenesisTokenizer
from datasets.multi_task_sampler import get_multi_task_dataloader
from training.flash_attention_config import print_flash_attention_status
from training.callbacks.grokking import GrokkingDetector, ProcrustesMonitor, ConceptClusteringMonitor
from training.callbacks.manager import CallbackManager


def get_model_config(mode: str):
    """Get model configuration for the specified mode."""
    configs = {
        "microscope": {
            "dim": 384,
            "n_layers": 77,
            "n_heads": 6,
            "intermediate_size": 1536,
            "vocab_size": 8192,
            "norm_type": "layernorm",
            "params": "~125M"
        },
        "deep_narrow_40": {
            "dim": 768,
            "n_layers": 40,
            "n_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 8192,
            "norm_type": "deepnorm",
            "params": "~800M"
        },
        "deep_narrow_48": {
            "dim": 896,
            "n_layers": 48,
            "n_heads": 14,
            "intermediate_size": 3584,
            "vocab_size": 8192,
            "norm_type": "deepnorm",
            "params": "~1.0B"
        }
    }
    
    if mode not in configs:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(configs.keys())}")
    
    return configs[mode]


def train_multi_task(
    mode: str = "microscope",
    batch_size: int = 4,
    max_steps: int = 10000,
    grad_accum_steps: int = 4,
    save_interval: int = 500,
    log_interval: int = 10,
    val_interval: int = 100,
    precision: str = "fp16",
    detect_grokking: bool = True,
    bible_dir: str = "../../Bible",
    learning_rate: float = 2e-4,
    weight_decay: float = 0.08
):
    """
    Multi-task training with native PyTorch.
    
    Args:
        mode: Model architecture
        batch_size: Batch size
        max_steps: Maximum training steps
        grad_accum_steps: Gradient accumulation steps
        save_interval: Checkpoint interval
        log_interval: Logging interval
        val_interval: Validation interval
        precision: Training precision (fp32, fp16, bf16, int8, int4)
        detect_grokking: Enable grokking detection
        bible_dir: Directory containing Bible translations
        learning_rate: Learning rate
        weight_decay: Weight decay
    """
    
    # Validate precision mode
    valid_precision = ["fp32", "fp16", "bf16", "int8", "int4"]
    if precision not in valid_precision:
        raise ValueError(f"Invalid precision: {precision}. Choose from {valid_precision}")
    
    # Check for quantization library
    use_quantization = precision in ["int8", "int4"]
    if use_quantization:
        try:
            import bitsandbytes as bnb
            print(f"✓ bitsandbytes detected (version {bnb.__version__})")
        except ImportError:
            raise ImportError(
                f"Precision '{precision}' requires bitsandbytes library.\n"
                "Install with: pip install bitsandbytes"
            )
    
    # Determine AMP settings
    use_amp = precision in ["fp16", "bf16"]
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16 if precision == "bf16" else None
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print("Multi-Task Training Configuration")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # FlashAttention status
    print(f"\nFlashAttention Status:")
    print_flash_attention_status()
    
    # Get config
    config = get_model_config(mode)
    
    print(f"\nModel: {mode}")
    print(f"  Estimated Parameters: {config['params']}")
    print(f"  Layers: {config['n_layers']}")
    print(f"  Dimension: {config['dim']}")
    print(f"  Heads: {config['n_heads']}")
    print(f"  Vocab Size: {config['vocab_size']}")
    
    print(f"\nMulti-Task Distribution:")
    print(f"  Language Modeling: 70%")
    print(f"  Coherence Detection: 15%")
    print(f"  Cross-Reference: 7.5%")
    print(f"  Paraphrase Detection: 7.5%")
    
    print(f"\nTraining:")
    print(f"  Precision: {precision.upper()}")
    print(f"  Batch size: {batch_size}")
    print(f"  Grad accum: {grad_accum_steps}")
    print(f"  Effective batch: {batch_size * grad_accum_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Max steps: {max_steps}")
    print(f"  Grokking detection: {detect_grokking}")
    print(f"{'='*70}\n")
    
    # Create directories
    checkpoint_dir = Path("checkpoints") / f"multi_task_{mode}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path("logs") / f"multi_task_{mode}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = GenesisTokenizer("genesis_tokenizer.json")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Initialize multi-task model
    print("\nInitializing multi-task model...")
    
    # Create base Llama model first (filter out 'params' key used for display)
    from models.llama.model import Llama
    model_config = {k: v for k, v in config.items() if k != 'params'}
    base_model = Llama(**model_config)
    
    # Wrap with multi-task heads (pass dim and vocab_size explicitly)
    model = MultiTaskLlama(
        base_model=base_model,
        dim=config['dim'],
        vocab_size=config['vocab_size']
    )
    model = model.to(device)
    
    # Count parameters and calculate memory
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate memory footprint based on precision
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5
    }
    model_size_gb = (total_params * bytes_per_param[precision]) / 1e9
    
    # Estimate VRAM usage (model + optimizer states + gradients + activations)
    # Rule of thumb: 4x model size for Adam optimizer in FP32 training
    optimizer_multiplier = 3 if precision in ["int8", "int4"] else 4
    estimated_vram_gb = model_size_gb * optimizer_multiplier
    
    print(f"\n{'='*70}")
    print("Model Statistics")
    print(f"{'='*70}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size ({precision}): {model_size_gb:.2f} GB")
    print(f"  Estimated VRAM usage: {estimated_vram_gb:.2f} GB")
    print(f"  (includes model + optimizer + gradients + activations)")
    print(f"{'='*70}")
    
    # User confirmation prompt
    print(f"\n⚠️  Please review the configuration above.")
    try:
        response = input("\nProceed with training? [Y/n]: ").strip().lower()
        if response in ['n', 'no']:
            print("\n❌ Training aborted by user.\n")
            return
        print()
    except EOFError:
        # Handle non-interactive environments (e.g., scripts)
        print("Non-interactive mode detected. Proceeding automatically...\n")
    
    # Initialize optimizer (with quantization support if needed)
    print("Initializing optimizer...")
    if precision == "int8":
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        print("  Using 8-bit AdamW optimizer")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        print("  Using standard AdamW optimizer")
    
    # Mixed precision scaler (modern torch.amp API)
    scaler = GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)
    print(f"  Mixed precision: {precision}")
    
    # Initialize multi-task dataloader
    print("\nInitializing multi-task dataloader...")
    dataloader = get_multi_task_dataloader(
        corpus_path="nwt_corpus.txt",
        bible_data_dir=bible_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=512
    )
    print(f"  Dataset initialized")
    
    # Initialize callbacks
    callbacks = None
    if detect_grokking:
        print("\nInitializing grokking detection callbacks...")
        callbacks = CallbackManager([
            GrokkingDetector(
                patience=1000,
                threshold=0.10,
                checkpoint_dir=checkpoint_dir / "grokking"
            ),
            ProcrustesMonitor(eval_interval=1000),
            ConceptClusteringMonitor(eval_interval=2000)
        ])
        callbacks.on_train_start(model)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Training loop
    print(f"\n{'='*70}")
    print("Starting Multi-Task Training")
    print(f"{'='*70}\n")
    
    model.train()
    global_step = 0
    running_loss = 0.0
    running_lm_loss = 0.0
    running_coherence_loss = 0.0
    start_time = time.time()
    
    while global_step < max_steps:
        for batch_idx, batch in enumerate(dataloader):
            task = batch['task']
            
            # Trigger batch start callbacks
            if callbacks:
                callbacks.on_batch_start(global_step, model, batch)
            
            # Forward pass with mixed precision (modern torch.amp API)
            with autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                if task == 'lm':
                    # Language modeling
                    tokens = batch['tokens'].to(device)
                    labels = batch['labels'].to(device)
                    
                    loss, _ = model.forward_lm(tokens, labels)
                    running_lm_loss += loss.item()
                    
                elif task == 'coherence':
                    # Coherence detection
                    verse1 = batch['verse1_tokens'].to(device)
                    verse2 = batch['verse2_tokens'].to(device)
                    labels = batch['labels'].to(device)
                    
                    loss, _ = model.forward_coherence(verse1, verse2, labels)
                    running_coherence_loss += loss.item()
                    
                elif task == 'cross_ref':
                    # Cross-reference prediction (triplet loss)
                    anchor = batch['anchor_tokens'].to(device)
                    positive = batch['positive_tokens'].to(device)
                    negative = batch['negative_tokens'].to(device)
                    
                    loss, _ = model.forward_cross_ref(anchor, positive, negative)
                    
                elif task == 'paraphrase':
                    # Cross-lingual paraphrase
                    verse1 = batch['verse1_tokens'].to(device)
                    verse2 = batch['verse2_tokens'].to(device)
                    labels = batch['labels'].to(device)
                    
                    loss, _ = model.forward_paraphrase(verse1, verse2, labels)
                
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
                
                # Trigger batch end callbacks
                if callbacks:
                    callbacks.on_batch_end(global_step, model, loss.item())
                
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
                          f"Task: {task} | "
                          f"Speed: {steps_per_sec:.2f} steps/s | "
                          f"ETA: {eta}")
                    
                    # TensorBoard logging
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar(f"train/loss_{task}", avg_loss, global_step)
                    writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], global_step)
                    
                    running_loss = 0.0
                    start_time = time.time()
                
                # Validation
                if global_step % val_interval == 0:
                    print(f"\n[VAL] Running validation at step {global_step}...")
                    val_loss = run_validation(model, dataloader, device, use_amp=use_amp)
                    print(f"  Validation loss: {val_loss:.4f}\n")
                    
                    writer.add_scalar("val/loss", val_loss, global_step)
                    
                    # Trigger validation end callbacks
                    if callbacks:
                        callbacks.on_validation_end(global_step, model, val_loss, {})
                    
                    model.train()
                
                # Checkpointing
                if global_step % save_interval == 0:
                    checkpoint_path = checkpoint_dir / f"step_{global_step}.pt"
                    print(f"\n💾 Saving checkpoint: {checkpoint_path}")
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
    
    # Trigger train end callbacks
    if callbacks:
        callbacks.on_train_end(model)
    
    print(f"\n{'='*70}")
    print("Multi-Task Training Complete!")
    print(f"{'='*70}\n")
    print(f"Total steps: {global_step}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")
    print(f"\nTo view training curves:")
    print(f"  tensorboard --logdir {log_dir}")
    print()


def run_validation(model, dataloader, device, use_amp=True, num_batches=10):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            task = batch['task']
            
            with autocast('cuda', enabled=use_amp, dtype=amp_dtype if use_amp else None):
                if task == 'lm':
                    tokens = batch['tokens'].to(device)
                    labels = batch['labels'].to(device)
                    loss, _ = model.forward_lm(tokens, labels)
                elif task == 'coherence':
                    verse1 = batch['verse1_tokens'].to(device)
                    verse2 = batch['verse2_tokens'].to(device)
                    labels = batch['labels'].to(device)
                    loss, _ = model.forward_coherence(verse1, verse2, labels)
                # Add other tasks as needed
                else:
                    continue
            
            total_loss += loss.item()
            count += 1
    
    return total_loss / count if count > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Multi-task training with native PyTorch + grokking detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Precision Modes:
  fp32   - Full precision (highest accuracy, most VRAM)
  fp16   - Half precision with automatic mixed precision (default)
  bf16   - BFloat16 precision (better for training stability)
  int8   - 8-bit training (requires bitsandbytes, lower VRAM)
  int4   - 4-bit training (requires bitsandbytes, experimental)

Example usage:
  python train_native_multi_task.py --mode microscope --precision fp16 --steps 10000
  python train_native_multi_task.py --mode deep_narrow_40 --precision int8 --batch-size 8
        """
    )
    parser.add_argument("--mode", type=str, default="microscope", 
                        choices=["microscope", "deep_narrow_40", "deep_narrow_48"],
                        help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--steps", type=int, default=10000, help="Maximum training steps")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--save-interval", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument("--val-interval", type=int, default=100, help="Validation interval")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp32", "fp16", "bf16", "int8", "int4"],
                        help="Training precision mode")
    parser.add_argument("--no-grokking", action="store_true", help="Disable grokking detection")
    parser.add_argument("--bible-dir", type=str, default="../../Bible", help="Bible data directory")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.08, help="Weight decay")
    
    args = parser.parse_args()
    
    print("\n>>> Genesis Arbiter - Multi-Task Training (Native PyTorch)\n")
    
    train_multi_task(
        mode=args.mode,
        batch_size=args.batch_size,
        max_steps=args.steps,
        grad_accum_steps=args.grad_accum,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        precision=args.precision,
        detect_grokking=not args.no_grokking,
        bible_dir=args.bible_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )


if __name__ == "__main__":
    main()
