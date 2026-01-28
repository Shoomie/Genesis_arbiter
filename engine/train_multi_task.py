"""
Multi-Task Training Script with FlashAttention and Grokking Detection

Integrates all Phase 1, 2, and 3 components:
- FlashAttention (Phase 1)
- Multi-task learning (Phase 2)
- Grokking detection (Phase 3)
"""

import os
os.environ["USE_LIBUV"] = "0"

import torch
import argparse
from pathlib import Path
import time

# Composer
from composer import Trainer, algorithms
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.loggers import TensorBoardLogger
from composer.core import DataSpec

# Genesis Arbiter
from models.llama.model import Llama, logos_init_hook
from models.tokenizer import GenesisTokenizer
from models.multi_task_wrapper import MultiTaskLlama
from datasets.multi_task_sampler import get_multi_task_dataloader
from training.flash_attention_config import FlashAttentionConfig
from training.callbacks.grokking import GrokkingCallback, ProcrustesCallback, ConceptClusteringCallback
from train_composer import GenesisComposerModel, get_protocol_config, find_tokenizer


class MultiTaskComposerModel(GenesisComposerModel):
    """
    Composer wrapper for multi-task model.
    """
    
    def __init__(self, model: MultiTaskLlama):
        super().__init__(model)
    
    def forward(self, batch):
        """Forward pass that handles multi-task batches."""
        outputs, loss, task = self.model(batch)
        return outputs, loss
    
    def loss(self, outputs, batch):
        """Extract loss from multi-task forward."""
        outputs_data, loss = outputs
        return loss


def train_multi_task():
    parser = argparse.ArgumentParser(description="Genesis Arbiter Multi-Task Training")
    parser.add_argument("--mode", type=str, 
                       choices=["deep_narrow_32", "deep_narrow_40", "deep_narrow_48",
                               "deep_narrow_60", "theos_small", "deep_narrow_100"], 
                       help="Select model architecture")
    parser.add_argument("--corpus-path", type=str, default="nwt_corpus.txt",
                       help="Path to NWT corpus")
    parser.add_argument("--bible-dir", type=str, default="../../Bible",
                       help="Path to Bible directory with translations")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                       help="Directory for checkpoints")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Experiment name")
    parser.add_argument("--steps", type=int, default=10000,
                       help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                       help="Gradient accumulation")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate (override config)")
    parser.add_argument("--weight-decay", type=float, default=None,
                       help="Weight decay (override config)")
    
    # Multi-task options
    parser.add_argument("--lm-weight", type=float, default=0.70,
                       help="Language modeling task weight")
    parser.add_argument("--coherence-weight", type=float, default=0.15,
                       help="Coherence detection task weight")
    parser.add_argument("--cross-ref-weight", type=float, default=0.075,
                       help="Cross-reference task weight")
    parser.add_argument("--paraphrase-weight", type=float, default=0.075,
                       help="Paraphrase task weight")
    
    # Grokking detection
    parser.add_argument("--detect-grokking", action="store_true",
                       help="Enable grokking detection")
    parser.add_argument("--grokking-threshold", type=float, default=0.10,
                       help="Grokking detection threshold (10% = 0.10)")
    
    args = parser.parse_args()
    
    # Print startup banner
    print("\n" + "="*60)
    print("   GENESIS ARBITER MULTI-TASK TRAINING")
    print("   Phase 2: Multi-Task Learning + Phase 3: Grokking Detection")
    print("="*60 + "\n")
    
    # Verify FlashAttention
    fa_config = FlashAttentionConfig()
    fa_config.print_status()
    
    # Interactive architecture selection if not provided
    selected_mode = args.mode
    if selected_mode is None:
        print("\n=== Architecture Selection ===")
        print("[1] Deep Narrow 32 - 32L×640D (550M params)")
        print("[2] Deep Narrow 40 - 40L×768D (800M params)")
        print("[3] Deep Narrow 48 - 48L×896D (1.0B params)")
        print("[4] Deep Narrow 60 - 60L×768D (1.2B params)")
        print("[5] Theos-Small - 80L×1024D (1.8B params)")
        print("[6] Deep Narrow 100 - 100L×1024D (2.3B params)")
        
        m_choice = input("\nSelect architecture [1-6]: ").strip()
        m_mapping = {
            "1": "deep_narrow_32",
            "2": "deep_narrow_40",
            "3": "deep_narrow_48",
            "4": "deep_narrow_60",
            "5": "theos_small",
            "6": "deep_narrow_100"
        }
        selected_mode = m_mapping.get(m_choice, "deep_narrow_48")
    
    mode_cfg = get_protocol_config(selected_mode)
    model_cfg = mode_cfg["model"]
    
    print(f"\n✓ Selected: {selected_mode.upper()}")
    print(f"  Parameters: {mode_cfg['params']}")
    print(f"  Norm Type: {model_cfg['norm_type']}")
    
    # Create base model
    print(f"\n>>> Initializing base model...")
    base_model = Llama(
        vocab_size=model_cfg["vocab_size"],
        n_layers=model_cfg["n_layers"],
        dim=model_cfg["dim"],
        n_heads=model_cfg["n_heads"],
        intermediate_size=model_cfg["intermediate_size"],
        max_seq_len=model_cfg.get("max_seq_len", 1024),
        norm_type=model_cfg.get("norm_type", "deepnorm")
    )
    
    print(f"✓ Base model: {base_model.get_num_params():,} parameters")
    
    # Apply Jehovah token initialization
    tokenizer = find_tokenizer(model_cfg["vocab_size"])
    jhvh_id = 5
    if hasattr(tokenizer, 'tokenizer'):
        jhvh_id = tokenizer.tokenizer.token_to_id("Jehovah") or 5
    
    logos_init_hook(base_model, jehovah_token_id=jhvh_id, multiplier=1.0)
    print(f"✓ Jehovah token initialization applied (ID={jhvh_id})")
    
    # Wrap in multi-task model
    print(f"\n>>> Creating multi-task wrapper...")
    multi_task_model = MultiTaskLlama(
        base_model=base_model,
        coherence_hidden_dim=1024,
        enable_cross_ref=True,
        enable_paraphrase=True
    )
    
    print(f"✓ Multi-task model: {multi_task_model.get_num_params():,} parameters")
    print(f"  Additional task heads: {multi_task_model.get_num_params() - base_model.get_num_params():,} parameters")
    
    # Wrap for Composer
    composer_model = MultiTaskComposerModel(multi_task_model)
    
    # Create multi-task dataloader
    print(f"\n>>> Loading multi-task dataset...")
    task_distribution = {
        'lm': args.lm_weight,
        'coherence': args.coherence_weight,
        'cross_ref': args.cross_ref_weight,
        'paraphrase': args.paraphrase_weight
    }
    
    print(f"Task distribution:")
    for task, weight in task_distribution.items():
        print(f"  {task}: {weight*100:.1f}%")
    
    train_dataloader = get_multi_task_dataloader(
        corpus_path=args.corpus_path,
        bible_data_dir=args.bible_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=model_cfg.get("max_seq_len", 1024),
        task_distribution=task_distribution
    )
    
    # Wrap for Composer
    dataspec = DataSpec(dataloader=train_dataloader)
    
    # Create optimizer
    lr = args.learning_rate or mode_cfg["training"].get("learning_rate", 3e-4)
    wd = args.weight_decay or mode_cfg["training"].get("weight_decay", 0.01)
    
    print(f"\n>>> Optimizer configuration:")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {wd}")
    
    optimizer = torch.optim.AdamW(
        composer_model.parameters(),
        lr=lr,
        weight_decay=wd
    )
    
    # Setup callbacks
    callbacks = [
        LRMonitor(),
        MemoryMonitor(),
        SpeedMonitor(window_size=100)
    ]
    
    # Add grokking detection if enabled
    if args.detect_grokking:
        print(f"\n✓ Grokking detection enabled (threshold={args.grokking_threshold*100:.0f}%)")
        callbacks.extend([
            GrokkingCallback(
                patience=1000,
                threshold=args.grokking_threshold,
                checkpoint_on_grokking=True
            ),
            ProcrustesCallback(eval_interval=1000),
            ConceptClusteringCallback(eval_interval=2000)
        ])
    
    # Setup experiment
    experiment_name = args.experiment_name or f"multitask_{selected_mode}_{int(time.time())}"
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n>>> Creating Composer Trainer...")
    print(f"  Experiment: {experiment_name}")
    print(f"  Total steps: {args.steps}")
    
    trainer = Trainer(
        model=composer_model,
        train_dataloader=dataspec,
        max_duration=f"{args.steps}ba",
        optimizers=optimizer,
        
        algorithms=[
            algorithms.GradientClipping(clipping_threshold=1.0)
        ],
        
        callbacks=callbacks,
        
        loggers=[
            TensorBoardLogger(log_dir=str(log_dir / "tensorboard"))
        ],
        
        save_interval=f"1000ba",
        save_folder=str(Path(args.output_dir) / experiment_name),
        save_filename="checkpoint_ep{epoch}_ba{batch}.pt",
        
        precision="amp_bf16" if torch.cuda.is_available() else "fp32",
        device_train_microbatch_size=args.batch_size // args.grad_accum,
        
        log_to_console=True,
        console_log_interval="10ba",
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
    train_multi_task()
