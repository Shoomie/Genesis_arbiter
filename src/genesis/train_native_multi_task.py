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
import numpy as np
os.environ["USE_LIBUV"] = "0"
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import time
from datetime import timedelta
import toml

# Core imports
from .models.multi_task_wrapper import MultiTaskLlama
from .models.tokenizer import GenesisTokenizer
from .datasets.multi_task_sampler import get_multi_task_dataloader
from .training.flash_attention_config import print_flash_attention_status
from .training.callbacks.grokking import GrokkingDetector, ProcrustesMonitor, ConceptClusteringMonitor
from .training.callbacks.manager import CallbackManager
from .evaluation.procedural_evaluator import ProceduralEvaluator


# ============================================================================
# TRAINING CONFIGURATION - Edit these settings before running
# ============================================================================

CONFIG = {
    # Model Architecture
    "MODE": "microscope",  # Options: "microscope", "deep_narrow_40", "deep_narrow_48"
    
    # Model Architecture Overrides (Set to apply custom values, None to use mode defaults)
    "MODEL_OVERRIDES": {
        "dim": 512,                # Hidden dimension size (e.g., 768)
        "n_layers": 24,           # Number of transformer layers (e.g., 12)
        "n_heads": 8,            # Number of attention heads (e.g., 12)
        "intermediate_size": 2048,  # Feed-forward layer size (typically 4 * dim)
        "vocab_size": None,         # Vocabulary size (usually determined by tokenizer)
    },

    # Training Hyperparameters
    "LEARNING_RATE": 1e-4,           # Base learning rate
    "WEIGHT_DECAY": 0.08,            # L2 regularization strength
    "MAX_STEPS": 300000,              # Total training steps
    "LR_SCHEDULE": "cosine",         # Options: "constant", "cosine"
    "WARMUP_STEPS": 2000,            # Number of steps for linear LR warmup
    "MIN_LR_RATIO": 0.1,             # Minimum LR as a fraction of base LR (for cosine)
    
    # Precision & Performance
    "PRECISION": "bf16",             # Options: "fp32", "fp16", "bf16", "int8", "int4"
    "BATCH_SIZE": 1024,                 # Micro-batch size per optimization step (per GPU)
    "GRAD_ACCUM_STEPS": 1,           # Number of steps to accumulate gradients before updating
    # Effective Batch Size = BATCH_SIZE * GRAD_ACCUM_STEPS
    # For Grokking: Recommended effective batch >= 16
    
    # Checkpoint & Logging Intervals
    "SAVE_INTERVAL": 2000,            # Steps between checkpoints
    "LOG_INTERVAL": 100,              # Steps between console logs
    "VAL_INTERVAL": 500,             # Steps between validation runs
    "EVAL_INTERVAL": 1000,            # Steps between extensive evaluation (BERT, Span, Verse)
    "EVAL_RECON_SAMPLES": 50,         # Number of samples for reconstruction eval
    "EVAL_AUX_SAMPLES": 20,           # Number of samples for auxiliary tasks
    
    # Advanced Features
    "DETECT_GROKKING": True,         # Enable grokking detection callbacks
    "VERBOSE_LOGGING": True,         # Extended runtime metrics (throughput etc.)
    
# Data Configuration
    "BIBLE_DIR": "../../../Bible",      # Path to Bible data directory (relative to script)
}

def load_central_config():
    """Load settings from genesis_config.toml at project root."""
    try:
        # Find project root (3 levels up from this file)
        root = Path(__file__).parent.parent.parent
        config_path = root / "genesis_config.toml"
        if config_path.exists():
            central_cfg = toml.load(config_path)
            train_cfg = central_cfg.get("training", {})
            
            # Map central to local keys
            mapping = {
                "mode": "MODE",
                "learning_rate": "LEARNING_RATE",
                "weight_decay": "WEIGHT_DECAY",
                "max_steps": "MAX_STEPS",
                "batch_size": "BATCH_SIZE",
                "grad_accum_steps": "GRAD_ACCUM_STEPS",
                "save_interval": "SAVE_INTERVAL",
                "log_interval": "LOG_INTERVAL",
                "val_interval": "VAL_INTERVAL",
                "eval_interval": "EVAL_INTERVAL",
                "detect_grokking": "DETECT_GROKKING",
                "verbose_logging": "VERBOSE_LOGGING",
                "bible_dir": "BIBLE_DIR"
            }
            
            for central_key, local_key in mapping.items():
                if central_key in train_cfg:
                    CONFIG[local_key] = train_cfg[central_key]
            
            return True
    except Exception as e:
        print(f"[WARN] Failed to load central config: {e}")
    return False

# Load central config if available
load_central_config()

# ============================================================================
# Effective batch size = BATCH_SIZE × GRAD_ACCUM_STEPS
# For grokking: Recommended effective batch >= 16
# For GPU usage: Increase BATCH_SIZE, decrease GRAD_ACCUM_STEPS
# ============================================================================


def get_model_config(mode: str):
    """Get model configuration for the specified mode."""
    configs = {
        "microscope": {
            "dim": 144,
            "n_layers": 144,
            "n_heads": 12,
            "intermediate_size": 1536,
            "vocab_size": 8000,
            "norm_type": "layernorm",
            "params": "~15M"
        },
        "deep_narrow_40": {
            "dim": 768,
            "n_layers": 40,
            "n_heads": 12,
            "intermediate_size": 3072,
            "vocab_size": 8000,
            "norm_type": "deepnorm",
            "params": "~800M"
        },
        "deep_narrow_48": {
            "dim": 896,
            "n_layers": 48,
            "n_heads": 14,
            "intermediate_size": 3584,
            "vocab_size": 8000,
            "norm_type": "deepnorm",
            "params": "~1.0B"
        }
    }
    
    if mode not in configs:
        raise ValueError(f"Unknown mode: {mode}. Available: {list(configs.keys())}")
    
    # Start with default config for the mode
    config = configs[mode].copy()
    
    # Apply overrides from CONFIG if present
    overrides = CONFIG.get("MODEL_OVERRIDES", {})
    if overrides:
        print("\n[CONFIG] Checking for architecture overrides...")
        applied_override = False
        for key, value in overrides.items():
            if value is not None and key in config:
                print(f"  Overriding {key}: {config[key]} -> {value}")
                config[key] = value
                applied_override = True
        if not applied_override:
            print("  No active overrides found.")
            
    return config


def train_multi_task(
    mode: str = None,
    batch_size: int = None,
    max_steps: int = None,
    grad_accum_steps: int = None,
    save_interval: int = None,
    log_interval: int = None,
    val_interval: int = None,
    precision: str = None,
    detect_grokking: bool = None,
    bible_dir: str = None,
    learning_rate: float = None,
    weight_decay: float = None,
    lr_schedule: str = None,
    warmup_steps: int = None,
    min_lr_ratio: float = None,
    verbose_logging: bool = None,
    resume_from_checkpoint: bool = False,
    compile_model: bool = False,
    gradient_checkpointing: bool = False,
    cpu_data: bool = False,
    target_languages: list = None,
    prepare_cache: bool = False,
    eval_interval: int = None,
    eval_recon_samples: int = None,
    eval_aux_samples: int = None
):
    """Load defaults from CONFIG if not provided via command line."""
    mode = mode or CONFIG["MODE"]
    batch_size = batch_size or CONFIG["BATCH_SIZE"]
    max_steps = max_steps or CONFIG["MAX_STEPS"]
    grad_accum_steps = grad_accum_steps or CONFIG["GRAD_ACCUM_STEPS"]
    save_interval = save_interval or CONFIG["SAVE_INTERVAL"]
    log_interval = log_interval or CONFIG["LOG_INTERVAL"]
    val_interval = val_interval or CONFIG["VAL_INTERVAL"]
    eval_interval = eval_interval or CONFIG.get("EVAL_INTERVAL", 1000)
    eval_recon_samples = eval_recon_samples or CONFIG.get("EVAL_RECON_SAMPLES", 50)
    eval_aux_samples = eval_aux_samples or CONFIG.get("EVAL_AUX_SAMPLES", 20)
    precision = precision or CONFIG["PRECISION"]
    detect_grokking = detect_grokking if detect_grokking is not None else CONFIG["DETECT_GROKKING"]
    bible_dir = bible_dir or CONFIG["BIBLE_DIR"]
    learning_rate = learning_rate or CONFIG["LEARNING_RATE"]
    weight_decay = weight_decay or CONFIG["WEIGHT_DECAY"]
    lr_schedule = lr_schedule or CONFIG.get("LR_SCHEDULE", "cosine")
    warmup_steps = warmup_steps if warmup_steps is not None else CONFIG.get("WARMUP_STEPS", 2000)
    min_lr_ratio = min_lr_ratio if min_lr_ratio is not None else CONFIG.get("MIN_LR_RATIO", 0.1)
    verbose_logging = verbose_logging if verbose_logging is not None else CONFIG["VERBOSE_LOGGING"]
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
        compile_model: Use torch.compile()
        gradient_checkpointing: Enable gradient checkpointing
        cpu_data: Keep dataset in system RAM (CPU) instead of VRAM
        target_languages: List of languages to train on (e.g., ["en", "es"])
        prepare_cache: If True, generate the cache file and exit
    """
    
    # Resolve script root (dir where this script lives)
    script_dir = Path(__file__).parent.absolute()
    
    # Resolve bible_dir path (anchored to script location or absolute)
    bible_path = Path(bible_dir)
    if not bible_path.is_absolute():
        # Try relative to script
        bible_path = (script_dir / bible_dir).resolve()
    
    if not bible_path.exists():
        # Fallback search strategies
        fallbacks = [
            script_dir.parent / "Bible",           # Genesis_arbiter/Bible
            script_dir.parent.parent / "Bible",    # Research/Bible
            Path("Bible").absolute(),              # Current dir/Bible
        ]
        for fb in fallbacks:
            if fb.exists():
                bible_path = fb
                break
    
    bible_dir = str(bible_path)
    print(f"  Using Bible data from: {bible_dir}")
    
    # Determine cache path based on languages
    cache_filename = "genesis_data_cache.pt"
    if target_languages and len(target_languages) == 1:
        # Special case: single language cache
        cache_filename = f"genesis_data_cache_{target_languages[0]}.pt"
    elif target_languages:
        # For multiple specific languages, use a generic filtered cache name
        cache_filename = "genesis_data_cache_filtered.pt"

    cache_path = script_dir.parent.parent / "data" / cache_filename
    
    # Initialize tokenizer (anchored to script location)
    print("Loading tokenizer...")
    tokenizer_path = script_dir.parent.parent / "data" / "genesis_char_tokenizer.json"
    
    tokenizer = GenesisTokenizer(str(tokenizer_path))
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Handle --prepare-cache
    if prepare_cache:
        print(f"\n{'='*70}")
        print(f"CACHE PREPARATION MODE")
        print(f"{'='*70}")
        print(f"Target file: {cache_path}")
        print(f"Languages: {target_languages if target_languages else 'ALL'}")
        
        from .datasets.multi_task_sampler import process_and_save_cache
        process_and_save_cache(
            bible_data_dir=bible_dir,
            tokenizer=tokenizer,
            output_path=str(cache_path),
            target_languages=target_languages
        )
        print("\n[DONE] Cache preparation complete. Exiting.")
        return
    
    # Validate precision mode
    valid_precision = ["fp32", "fp16", "bf16", "int8", "int4"]
    if precision not in valid_precision:
        raise ValueError(f"Invalid precision: {precision}. Choose from {valid_precision}")
    
    # Check for quantization library
    use_quantization = precision in ["int8", "int4"]
    if use_quantization:
        try:
            import bitsandbytes as bnb
            print(f"[OK] bitsandbytes detected (version {bnb.__version__})")
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
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints") / f"multi_task_{mode}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log directory
    log_dir = Path(f"logs/{mode}_{int(time.time())}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # DATA LOADER INIT (Before showing config, so vocab size is accurate)
    # -------------------------------------------------------------------------
    print("\nInitializing multi-task dataloader...")
    
    dataloader = get_multi_task_dataloader(
        bible_data_dir=bible_dir,
        tokenizer=tokenizer,
        device=device,
        cache_path=str(cache_path) if cache_path.exists() else None,
        batch_size=batch_size,
        max_seq_len=512, # Using 512 as a common default for sequence length
        task_distribution=None, # Will let the logic below determine it, or default
        seed=42,
        cpu_data=cpu_data,
        target_languages=target_languages
    )
    print(f"  Dataset initialized")
    
    # -------------------------------------------------------------------------
    # TASK DISTRIBUTION ADJUSTMENT (Single Language)
    # -------------------------------------------------------------------------
    if target_languages and len(target_languages) == 1:
        print(f"  [AUTO] Single language detected ({target_languages[0]}). Adjusting task distribution...")
        # Disable cross-lingual tasks
        dataloader.dataset.task_distribution = {
            'lm': 0.85,
            'coherence': 0.15,
            'cross_ref': 0.0,
            'paraphrase': 0.0
        }
        # Re-init probabilities
        dataloader.dataset.tasks = list(dataloader.dataset.task_distribution.keys())
        dataloader.dataset.task_probs = np.array([dataloader.dataset.task_distribution[t] for t in dataloader.dataset.tasks])
        print(f"  [AUTO] New distribution: {dataloader.dataset.task_distribution}")
    
    # Get config
    config = get_model_config(mode)
    
    # Override vocab size with actual tokenizer size immediately
    # (Dataloader might have compressed tokenizer to fit dataset)
    original_vocab_size = config.get('vocab_size', 'unknown')
    config['vocab_size'] = tokenizer.vocab_size
    if original_vocab_size != config['vocab_size']:
        print(f"\n[CONFIG] Auto-adjusted vocab_size: {original_vocab_size} -> {config['vocab_size']}")
    
    # Save target languages and LR settings for easier restoration
    config['target_languages'] = target_languages
    config['lr_schedule'] = lr_schedule
    config['warmup_steps'] = warmup_steps
    config['min_lr_ratio'] = min_lr_ratio
    
    # FlashAttention status
    print(f"\nFlashAttention Status:")
    print_flash_attention_status()
    
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
    print(f"  Compile Model: {compile_model}")
    print(f"  Gradient Checkpointing: {gradient_checkpointing}")
    print(f"  CPU Data Loading: {cpu_data}")
    print(f"  Languages: {target_languages if target_languages else 'ALL'}")
    print(f"  Cache Path: {cache_path}")
    print(f"{'='*70}\n")
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Resume Logic
    initial_step = 0
    resume_checkpoint = None
    
    if resume_from_checkpoint:
        print("\n[RESUME] Checking for existing checkpoints...")
        try:
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            if checkpoints:
                # Sort by step number (assuming step_X.pt or final_step_X.pt)
                # Helper to extract step
                def get_step(p):
                    import re
                    match = re.search(r'step_(\d+)', p.name)
                    return int(match.group(1)) if match else -1
                
                checkpoints.sort(key=get_step, reverse=True)
                latest_cp = checkpoints[0]
                
                if get_step(latest_cp) >= 0:
                    print(f"  Found latest checkpoint: {latest_cp.name}")
                    
                    # ASK USER
                    print(f"\n[?] Found compatible checkpoint from step {get_step(latest_cp)}.")
                    choice = input("    Do you want to RESUME from this checkpoint? [Y/n]: ").strip().lower()
                    
                    if choice in ('', 'y', 'yes'):
                        resume_checkpoint = latest_cp
                    else:
                        print("    Starting FRESH training (ignoring checkpoint).")
                        resume_checkpoint = None
                else:
                    print("  No valid step-labeled checkpoints found.")
            else:
                 print("  No checkpoints found to resume from.")
        except Exception as e:
            print(f"  Error checking for checkpoints: {e}")

    # Initialize multi-task model
    print("\nInitializing multi-task model...")
    
    # Create base Llama model first (filter out non-architecture keys)
    from .models.llama.model import Llama
    # We only pass keys that Llama.__init__ actually accepts
    llama_keys = {'vocab_size', 'n_layers', 'dim', 'n_heads', 'intermediate_size', 'max_seq_len', 'norm_type'}
    model_init_config = {k: v for k, v in config.items() if k in llama_keys}
    
    # Ensure vocab_size is correct
    model_init_config['vocab_size'] = tokenizer.vocab_size
    
    base_model = Llama(**model_init_config)
    
    # Wrap with multi-task heads (pass dim and vocab_size explicitly)
    model = MultiTaskLlama(
        base_model=base_model,
        dim=config['dim'],
        vocab_size=tokenizer.vocab_size
    )
    model = model.to(device)
    
    # Enable Gradient Checkpointing
    if gradient_checkpointing:
        print("  [OPT] Enabling gradient checkpointing...")
        # Access the inner base model
        if hasattr(model.base, 'gradient_checkpointing_enable'):
            model.base.gradient_checkpointing_enable()
        else:
            print("  [WARN] underlying model does not support gradient_checkpointing_enable()")

    # Compile Model
    if compile_model:
        print("  [OPT] Compiling model with torch.compile()...")
        try:
            # backend='inductor' is standard, 'cudagraphs' is faster but finicky
            # mode='reduce-overhead' is good for small batches
            model = torch.compile(model) 
            print("  [OK] Model compiled scheduled")
        except Exception as e:
            print(f"  [WARN] Compilation failed: {e}")
    
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
    print(f"\n[!] Please review the configuration above.")
    try:
        response = input("\nProceed with training? [Y/n]: ").strip().lower()
        if response in ['n', 'no']:
            print("\n[X] Training aborted by user.\n")
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
    
    # Scaler for AMP
    scaler = GradScaler('cuda', enabled=(precision == "fp16"))
    
    # Scheduler Initialization
    print(f"Initializing scheduler: {lr_schedule}")
    def get_lr_multiplier(step):
        # Linear Warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        # After warmup
        if lr_schedule == "constant":
            return 1.0
        
        # Cosine Decay
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        progress = min(1.0, progress)
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        
        # Min LR scale
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    
    # Load State if Resuming
    if resume_checkpoint:
        print(f"\n[RESUME] Loading state from {resume_checkpoint}...")
        try:
            checkpoint_data = torch.load(resume_checkpoint, map_location=device)
            
            # Check Config Compatibility (roughly)
            saved_config = checkpoint_data.get('config', {})
            # We care about structural params: dim, n_layers, n_heads, vocab_size
            structural_keys = ['dim', 'n_layers', 'n_heads', 'vocab_size']
            mismatch = False
            for k in structural_keys:
                if saved_config.get(k) != config.get(k):
                    print(f"  [WARN] Config mismatch for {k}: saved={saved_config.get(k)}, current={config.get(k)}")
                    mismatch = True
            
            if mismatch:
                print("  [ERROR] Critical config mismatch. Cannot resume safely. Starting fresh.")
            else:
                model.load_state_dict(checkpoint_data['model'])
                optimizer.load_state_dict(checkpoint_data['optimizer'])
                if 'scaler' in checkpoint_data:
                    scaler.load_state_dict(checkpoint_data['scaler'])
                if 'scheduler' in checkpoint_data:
                    scheduler.load_state_dict(checkpoint_data['scheduler'])
                
                initial_step = checkpoint_data.get('step', 0)
                print(f"  [OK] Resumed training from step {initial_step}")
                
        except Exception as e:
            print(f"  [ERROR] Failed to load checkpoint: {e}. Starting fresh.")
    

    
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

            ProcrustesMonitor(
                eval_interval=1000,
                languages=target_languages # Pass actual languages
            ),
            ConceptClusteringMonitor(eval_interval=2000)
        ])
        callbacks.on_train_start(model)
    
    # Initialize Procedural Evaluator
    evaluator = ProceduralEvaluator(
        model=model,
        dataset=dataloader.dataset,
        tokenizer=tokenizer,
        device=device,
        verbose=True
    )
    
    # Training Loop
    print(f"\n{'='*70}")
    print(f"Starting Multi-Task Training")
    print(f"Target Steps: {max_steps}")
    print(f"Start Step: {initial_step}")
    print(f"{'='*70}\n")
    
    model.train()
    global_step = initial_step
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
                
                # Scheduler step
                scheduler.step()
                
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
                    
                    # Single-line compact logging
                    lr = optimizer.param_groups[0]['lr']
                    print(f"Step {global_step}/{max_steps} ({100*global_step/max_steps:.1f}%) | "
                          f"Loss: {avg_loss:.4f} | Task: {task} | "
                          f"Speed: {steps_per_sec:.2f} st/s | ETA: {eta} | LR: {lr:.2e}")
                    
                    # TensorBoard logging
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar(f"train/loss_{task}", avg_loss, global_step)
                    writer.add_scalar("train/learning_rate", optimizer.param_groups[0]['lr'], global_step)
                    
                    if verbose_logging:
                        # Calculate throughput metrics
                        # Effective batch size = batch_size * grad_accum_steps
                        effective_batch_size = batch_size * grad_accum_steps
                        samples_per_sec = steps_per_sec * effective_batch_size
                        # Estimate tokens per sec (assuming full context usage)
                        tokens_per_sec = samples_per_sec * 512
                        
                        writer.add_scalar("throughput/samples_per_sec", samples_per_sec, global_step)
                        writer.add_scalar("throughput/tokens_per_sec", tokens_per_sec, global_step)
                    
                    running_loss = 0.0
                    start_time = time.time()
                
                # Extensive Evaluation
                if global_step % eval_interval == 0:
                    print(f"\n[EVAL] Running extensive procedural evaluation at step {global_step}...")
                    eval_results = evaluator.run_suite(
                        use_amp=use_amp, 
                        amp_dtype=amp_dtype,
                        num_recon_samples=eval_recon_samples,
                        num_aux_samples=eval_aux_samples
                    )
                    
                    # Log all metrics to TensorBoard
                    for key, val in eval_results.items():
                        writer.add_scalar(f"val/{key}", val, global_step)
                    
                    # Main evaluation loss
                    val_loss = eval_results.get('recon_bert_loss', 0.0)
                    print(f"  BERT-Loss: {val_loss:.4f} | CER: {eval_results.get('recon_bert_cer', 0.0):.4f}\n")
                    
                    # Trigger validation end callbacks
                    if callbacks:
                        callbacks.on_validation_end(global_step, model, val_loss, eval_results)
                    
                    model.train()
                
                # Simple Validation (if different from extensive eval)
                elif global_step % val_interval == 0:
                    print(f"\n[VAL] Running quick validation at step {global_step}...")
                    # For now just run a subset of reconstruction or something simple 
                    # but since the user wants to control extensive eval specifically:
                    # We'll just run a smaller/faster version if they are separate.
                    pass
                
                # Checkpointing
                if global_step % save_interval == 0:
                    checkpoint_path = checkpoint_dir / f"step_{global_step}.pt"
                    print(f"\n[SAVE] Saving checkpoint: {checkpoint_path}")
                    torch.save({
                        'step': global_step,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                        'scheduler': scheduler.state_dict(),
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
        'scheduler': scheduler.state_dict(),
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
    parser.add_argument("--steps", type=int, default=50000, help="Maximum training steps")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--save-interval", type=int, default=500, help="Checkpoint interval")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument("--val-interval", type=int, default=100, help="Validation interval")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp32", "fp16", "bf16", "int8", "int4"],
                        help="Training precision mode")
    parser.add_argument("--eval-interval", type=int, help="Extensive evaluation interval")
    parser.add_argument("--eval-recon-samples", type=int, help="Number of samples for reconstruction eval")
    parser.add_argument("--eval-aux-samples", type=int, help="Number of samples for auxiliary tasks")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--no-grokking", action="store_true", help="Disable grokking detection")
    parser.add_argument("--bible-dir", type=str, help="Bible data directory")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, help="Weight decay")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (GPU, memory, throughput)")
    parser.add_argument("--no-verbose", action="store_true", help="Disable verbose logging")
    
    # Scheduler options
    parser.add_argument("--schedule", type=str, choices=["constant", "cosine"], help="LR schedule type")
    parser.add_argument("--warmup", type=int, help="Number of warmup steps")
    parser.add_argument("--min-lr-ratio", type=float, help="Minimum LR ratio for cosine decay")
    
    # Optimization flags
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile()")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to save VRAM")
    parser.add_argument("--cpu-data", action="store_true", help="Keep dataset on CPU to save VRAM (allows larger datasets)")
    
    # Data filtering and caching
    parser.add_argument("--language", type=str, help="Comma-separated list of languages to train on (e.g., 'en')")
    parser.add_argument("--prepare-cache", action="store_true", help="Prepare dataset cache and exit")
    
    args = parser.parse_args()
    
    print("\n>>> Genesis Arbiter - Multi-Task Training (Native PyTorch)\n")
    print("[CONFIG] Using configuration from CONFIG section (override with command-line args)\n")
    
    # Determine verbose_logging
    verbose = None
    if args.verbose:
        verbose = True
    elif args.no_verbose:
        verbose = False
    
    # Interactive Language Selection
    target_languages = None
    # Only ask if not passed via args (which we don't have yet, but good practice)
    # We'll validatethe resume flag here too
    
    print("\n[?] Do you want to train on ALL languages?")
    choice = input("    Press Enter for ALL, or type 'n' to select specific languages: ").strip().lower()
    
    if choice == 'n':
        lang_input = input("    Enter language codes (comma-separated, e.g., 'en, sv, de'): ").strip()
        if lang_input:
            target_languages = [l.strip() for l in lang_input.split(',') if l.strip()]
            print(f"    Selected languages: {target_languages}")
        else:
            print("    No languages entered. Defaulting to ALL.")

    # Call training function
    train_multi_task(
        mode=args.mode if args.mode != "microscope" or "--mode" in " ".join(__import__('sys').argv) else None,
        batch_size=args.batch_size if "--batch-size" in " ".join(__import__('sys').argv) else None,
        max_steps=args.steps if "--steps" in " ".join(__import__('sys').argv) else None,
        grad_accum_steps=args.grad_accum if "--grad-accum" in " ".join(__import__('sys').argv) else None,
        save_interval=args.save_interval if "--save-interval" in " ".join(__import__('sys').argv) else None,
        log_interval=args.log_interval if "--log-interval" in " ".join(__import__('sys').argv) else None,
        val_interval=args.val_interval if "--val-interval" in " ".join(__import__('sys').argv) else None,
        eval_interval=args.eval_interval,
        eval_recon_samples=args.eval_recon_samples,
        eval_aux_samples=args.eval_aux_samples,
        precision=args.precision if "--precision" in " ".join(__import__('sys').argv) else None,
        detect_grokking=not args.no_grokking if args.no_grokking else None,
        bible_dir=args.bible_dir,
        learning_rate=args.lr if args.lr else None,
        weight_decay=args.weight_decay if args.weight_decay else None,
        lr_schedule=args.schedule if args.schedule else None,
        warmup_steps=args.warmup if args.warmup is not None else None,
        min_lr_ratio=args.min_lr_ratio if args.min_lr_ratio is not None else None,
        verbose_logging=verbose,
        resume_from_checkpoint=args.resume,
        compile_model=args.compile,
        gradient_checkpointing=args.gradient_checkpointing,
        cpu_data=args.cpu_data,
        target_languages=target_languages,
        prepare_cache=args.prepare_cache
    )


if __name__ == "__main__":
    main()
