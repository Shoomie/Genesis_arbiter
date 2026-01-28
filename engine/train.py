import os
os.environ["USE_LIBUV"] = "0"
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
import toml
import argparse
from pathlib import Path
import math
import time

# Core imports
from models.llama.model import Llama, logos_init_hook
from models.tokenizer import GenesisTokenizer
from components.checkpoint import save_checkpoint
from datasets.bible import get_bible_dataloader

# Phase 3 Arbiter imports
try:
    from arbiter_logger import ArbiterLogger
    ARBITER_LOGGER_AVAILABLE = True
except ImportError:
    ARBITER_LOGGER_AVAILABLE = False
    print("[Warning] arbiter_logger not available - using basic logging")

def setup():
    if not dist.is_initialized():
        # Fallbacks for direct python execution (single node)
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
            
        world_size = int(os.environ["WORLD_SIZE"])
        
        # If world_size > 1, we MUST initialize distributed
        # If world_size == 1, we can skip it on Windows to avoid Gloo networking bugs
        if world_size > 1:
            # Windows specific distributed setup
            backend = "gloo" # NCCL is not supported on Windows
            os.environ["GLOO_SOCKET_IFNAME"] = "127.0.0.1" # Force loopback for local
            
            print(f"Initializing distributed process group (Backend: {backend})...")
            dist.init_process_group(
                backend=backend,
                init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                rank=int(os.environ["RANK"]),
                world_size=world_size
            )
        else:
            print("Single-node detected: Skipping distributed initialization to maximize performance.")

        if torch.cuda.is_available():
            try:
                device_id = int(os.environ.get("LOCAL_RANK", 0))
                torch.cuda.set_device(device_id)
                print(f"CUDA initialized on device: {torch.cuda.get_device_name(device_id)}")
            except Exception as e:
                print(f"[{os.environ.get('RANK', 0)}] Warning: Failed to set CUDA device: {e}")

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_protocol_config(mode: str):
    """Returns model and training hyperparameters for the selected protocol mode.
    
    Updated for Phase 3 with Deep & Narrow architectures and DeepNorm support.
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
    """
    Find appropriate tokenizer from arbiter_tokenizer_factory output.
    Falls back to genesis_tokenizer.json if not found.
    """
    tokenizer_dir = Path("../tokenizers")
    
    # Try to find arbiter tokenizer matching vocab size
    if tokenizer_dir.exists():
        model_file = tokenizer_dir / f"arbiter_nwt_{vocab_size}.model"
        if model_file.exists():
            print(f"✓ Found custom tokenizer: {model_file}")
            # For now, return None and we'll handle SentencePiece in future update
            # Fall back to GenesisTokenizer
    
    # Fallback to genesis tokenizer
    if Path("genesis_tokenizer.json").exists():
        print(f"Using genesis_tokenizer.json (legacy)")
        return GenesisTokenizer("genesis_tokenizer.json")
    
    raise FileNotFoundError("No tokenizer found. Run 'python ../scripts/arbiter_tokenizer_factory.py nwt_corpus.txt'")

def train():
    parser = argparse.ArgumentParser(description="Genesis Arbiter Training (Single Model)")
    parser.add_argument("--config", type=str, help="Path to TOML config file (optional, will prompt if missing)")
    parser.add_argument("--mode", type=str, 
                       choices=["deep_narrow_32", "deep_narrow_40", "deep_narrow_48",
                               "deep_narrow_60", "theos_small", "deep_narrow_100", 
                               "microscope", "tower_of_truth", "high_res_arbiter"], 
                       help="Select model architecture")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                       help="Directory for checkpoints and logs")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Experiment name for logging")
    args = parser.parse_args()
    
    setup()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 1. Hardware Selection Menu
    selected_config_path = args.config
    if selected_config_path is None and local_rank == 0:
        print("\n" + "="*60)
        print("   GENESIS ARBITER TRAINING - Phase 3")
        print("="*60)
        print("\n=== Hardware Configuration ===")
        print("[1] High VRAM (12+ GB) - Performance Mode")
        print("[2] Low VRAM (4-6 GB) - Compatibility Mode with Gradient Checkpointing")
        h_choice = input("Select hardware tier [1-2]: ").strip()
        h_mapping = {"1": "train_configs/high_vram.toml", "2": "train_configs/low_vram.toml"}
        selected_config_path = h_mapping.get(h_choice, "train_configs/high_vram.toml")
        print(f"✓ Using Config: {selected_config_path}\n")

    # Fallback/Broadcast for other ranks (simplified)
    if selected_config_path is None:
        selected_config_path = "train_configs/high_vram.toml"
        
    config = toml.load(selected_config_path)
    
    # 2. Architecture Selection Menu
    selected_mode = args.mode
    if selected_mode is None and local_rank == 0:
        print("=== Architecture Selection ===")
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
        
    if selected_mode is None:
        selected_mode = "deep_narrow_48"  # Default to 1B model
    
    mode_cfg = get_protocol_config(selected_mode)
    if not mode_cfg:
        raise ValueError(f"Unknown mode: {selected_mode}")
    
    # Override config with mode-specific values
    for k, v in mode_cfg["model"].items():
        config["model"][k] = v
    for k, v in mode_cfg["training"].items():
        config["training"][k] = v
    
    if local_rank == 0:
        print(f"\n✓ Selected: {selected_mode.upper()}")
        print(f"  Parameters: {mode_cfg['params']}")
        print(f"  Purpose: {mode_cfg['purpose']}")
        print(f"  Norm Type: {mode_cfg['model']['norm_type']}")

    # Safety: Auto-correct dim % n_heads == 0
    dim = config["model"]["dim"]
    n_heads = config["model"]["n_heads"]
    if dim % n_heads != 0:
        old_dim = dim
        dim = (dim // n_heads + 1) * n_heads
        config["model"]["dim"] = dim
        if local_rank == 0:
            print(f"⚠ Safety Auto-Correct: Adjusting dim from {old_dim} to {dim} to align with {n_heads} heads.")

    # 3. Initialize Arbiter Logger
    logger = None
    if ARBITER_LOGGER_AVAILABLE and local_rank == 0:
        experiment_name = args.experiment_name or f"{selected_mode}_{int(time.time())}"
        log_dir = Path(args.output_dir) / "logs"
        
        try:
            logger = ArbiterLogger(
                log_dir=str(log_dir),
                experiment_name=experiment_name,
                enable_tensorboard=True
            )
            
            # Start experiment
            run_id = logger.start_experiment(config)
            print(f"\n✓ Arbiter Logger initialized")
            print(f"  Experiment: {experiment_name}")
            print(f"  Run ID: {run_id}")
            print(f"  TensorBoard: tensorboard --logdir {log_dir / 'tensorboard'}")
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize Arbiter Logger: {e}")
            logger = None

    # 4. Device Setup
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[{local_rank}] Using device: {device}", flush=True)
    
    # 5. Initialize Model
    print(f"[{local_rank}] Initializing {selected_mode.upper()} with {mode_cfg['model']['norm_type']} normalization...", flush=True)
    model_cfg = config["model"]
    
    model = Llama(
        vocab_size=model_cfg["vocab_size"],
        n_layers=model_cfg["n_layers"],
        dim=model_cfg["dim"],
        n_heads=model_cfg["n_heads"],
        intermediate_size=model_cfg["intermediate_size"],
        max_seq_len=model_cfg["max_seq_len"],
        norm_type=model_cfg.get("norm_type", "rmsnorm")  # DeepNorm or RMSNorm
    ).to(device)
    
    if local_rank == 0:
        print(f"✓ Model initialized: {model.get_num_params():,} parameters")
    
    # 6. Apply Jehovah Token Initialization
    if local_rank == 0:
        print(f"\n[Logos Initialization] Loading tokenizer...", flush=True)
        jhvh_id = 5 # Default
        tokenizer = None
        
        try:
            tokenizer = find_tokenizer(model_cfg["vocab_size"])
            if hasattr(tokenizer, 'tokenizer'):
                jhvh_id = tokenizer.tokenizer.token_to_id("Jehovah") or 5
                print(f"✓ Found 'Jehovah' token ID: {jhvh_id}", flush=True)
        except Exception as e:
            print(f"⚠ Warning: Tokenizer error: {e}")
            try:
                tokenizer = GenesisTokenizer("genesis_tokenizer.json")
                jhvh_id = tokenizer.tokenizer.token_to_id("Jehovah") or 5
            except:
                print(f"⚠ Using default Jehovah ID: {jhvh_id}")
        
        # Apply initialization (1x multiplier)
        logos_init_hook(model, jehovah_token_id=jhvh_id, multiplier=1.0)
        print(f"✓ Jehovah token initialization applied (ID={jhvh_id}, multiplier=1.0)")
    
    # 7. Compilation
    if config["training"].get("compile", False):
        if local_rank == 0:
            print("\n>>> Initiating torch.compile... (This can take 2-5 minutes on the first run)", flush=True)
        model = torch.compile(model)
        if local_rank == 0:
            print(">>> Model compilation complete.", flush=True)
        
    # 8. Parallelism setup
    if dist.is_initialized():
        if config["parallelism"].get("fsdp", False):
            from torch.distributed.fsdp import ShardingStrategy
            model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
            if local_rank == 0:
                print("✓ FSDP enabled (Full Shard)")
        else:
            model = DDP(model, device_ids=[local_rank])
            if local_rank == 0:
                print("✓ DDP enabled")
    else:
        if local_rank == 0:
            print("✓ Local training mode (no distributed overhead)")
    
    # 9. Gradient Checkpointing
    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if local_rank == 0:
            print("✓ Gradient checkpointing enabled")

    # 10. Optimizer with weight decay
    weight_decay = config["training"].get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["training"]["learning_rate"],
        weight_decay=weight_decay
    )
    
    if local_rank == 0:
        print(f"\n✓ Optimizer: AdamW (LR={config['training']['learning_rate']:.2e}, WD={weight_decay})")
    
    # 11. Learning Rate Schedule
    total_steps = config["training"]["steps"]
    warmup_steps = config["training"].get("warmup_steps", 100)
    
    def lr_lambda(s):
        if s < warmup_steps:
            return s / warmup_steps
        else:
            progress = (s - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 12. Data loading
    masking_config = config.get("masking", {})
    masking_enabled = masking_config.get("enabled", False)
    
    if masking_enabled:
        from datasets.bible_weighted_masking import get_bible_weighted_dataloader
        if local_rank == 0:
            print(f"\n✓ Using WEIGHTED MASKING strategy", flush=True)
            print(f"  Base probability: {masking_config.get('base_probability', 0.4)}", flush=True)
        
        dataloader = get_bible_weighted_dataloader(
            corpus_path="nwt_corpus.txt",
            tokenizer=tokenizer if tokenizer else GenesisTokenizer("genesis_tokenizer.json"),
            batch_size=config["training"]["batch_size"],
            max_seq_len=model_cfg["max_seq_len"],
            base_prob=masking_config.get("base_probability", 0.4),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0))
        )
    else:
        if local_rank == 0:
            print(f"\n✓ Standard dataset (no special masking)")
        
        dataloader = get_bible_dataloader(
            corpus_path="nwt_corpus.txt",
            tokenizer=tokenizer if tokenizer else GenesisTokenizer("genesis_tokenizer.json"),
            batch_size=config["training"]["batch_size"],
            max_seq_len=model_cfg["max_seq_len"]
        )

    # 13. Training Loop with ETA Tracking
    print(f"\n{'='*60}")
    print(f"  TRAINING START")
    print(f"{'='*60}")
    print(f"  Total steps: {total_steps}")
    print(f"  Checkpoint interval: {config['checkpoint']['interval']}")
    print(f"  Log interval: {config.get('logging', {}).get('log_interval', 10)}")
    print(f"{'='*60}\n")
    
    # ETA tracking variables
    start_time = time.time()
    step_times = []  # Rolling window of recent step times
    
    step = 0
    epoch = 0
    done = False
    last_log_time = start_time
    
    while not done:
        epoch += 1
        for tokens, labels in dataloader:
            step_start_time = time.time()
            step += 1
            if step > total_steps:
                done = True
                break
            
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            
            autocast_dtype = torch.bfloat16 if config["training"]["precision"] == "bf16" else torch.float16
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", 
                              enabled=(device.type != "cpu"), 
                              dtype=autocast_dtype):
                logits, loss = model(tokens, labels)
            
            loss.backward()
            
            # Gradient clipping (important for deep networks)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config["training"].get("grad_clip", 1.0)
            )
            
            optimizer.step()
            scheduler.step()
            
            # Track step time for ETA calculation
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            step_times.append(step_duration)
            if len(step_times) > 100:  # Keep rolling window of last 100 steps
                step_times.pop(0)
            
            # Logging with ETA
            log_interval = config.get("logging", {}).get("log_interval", 10)
            if step % log_interval == 0 and local_rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                
                # Calculate ETA
                avg_step_time = sum(step_times) / len(step_times)
                remaining_steps = total_steps - step
                eta_seconds = avg_step_time * remaining_steps
                eta_hours = eta_seconds / 3600
                eta_minutes = (eta_seconds % 3600) / 60
                
                # Elapsed time
                elapsed_seconds = step_end_time - start_time
                elapsed_hours = elapsed_seconds / 3600
                
                # Format ETA string
                if eta_hours >= 1:
                    eta_str = f"{int(eta_hours)}h {int(eta_minutes)}m"
                else:
                    eta_str = f"{int(eta_minutes)}m {int(eta_seconds % 60)}s"
                
                print(f"[Epoch {epoch}] Step {step}/{total_steps} | Loss={loss.item():.4f} | LR={current_lr:.2e} | GradNorm={grad_norm:.3f} | "
                      f"Elapsed={elapsed_hours:.1f}h | ETA={eta_str} | {avg_step_time:.2f}s/step", flush=True)
                
                # Log to Arbiter Logger
                if logger:
                    logger.log_training_step(
                        step=step,
                        loss=loss.item(),
                        grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        learning_rate=current_lr
                    )
                
                last_log_time = step_end_time
            
            # Checkpointing
            if step % config["checkpoint"]["interval"] == 0 and local_rank == 0:
                checkpoint_dir = Path(args.output_dir) / "checkpoints"
                save_checkpoint(
                    model.module if hasattr(model, "module") else model, 
                    optimizer, 
                    step, 
                    str(checkpoint_dir)
                )
                print(f"✓ Checkpoint saved: step_{step}")

    # 14. Finalization
    if logger and local_rank == 0:
        logger.finalize_experiment(
            final_train_loss=loss.item(),
            final_val_loss=None,  # Would need validation set
            status="completed"
        )
        print(f"\n✓ Experiment finalized in logger")
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Final loss: {loss.item():.4f}")
    print(f"  Total steps: {step}")
    print(f"{'='*60}\n")
    
    cleanup()

if __name__ == "__main__":
    train()
