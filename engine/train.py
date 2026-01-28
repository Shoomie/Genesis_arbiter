import os
os.environ["USE_LIBUV"] = "0"
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import toml
import argparse
from models.llama.model import Llama, logos_init_hook
from models.tokenizer import GenesisTokenizer
from components.checkpoint import save_checkpoint
from datasets.bible import get_bible_dataloader
import time
import math

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
    """Returns model and training hyperparameters for the selected protocol mode."""
    modes = {
        "microscope": {
            "model": {"dim": 768, "n_layers": 12, "n_heads": 12, "intermediate_size": 3072, "vocab_size": 8000},
            "training": {"learning_rate": 3e-4},
            "params": "125.5M",
            "vram_fp16": "~2-3 GB",
            "vram_bf16": "~2-3 GB"
        },
        "tower_of_truth": {
            "model": {"dim": 288, "n_layers": 144, "n_heads": 12, "intermediate_size": 576, "vocab_size": 12000},
            "training": {"learning_rate": 1e-4},
            "params": "~5-8M",
            "vram_fp16": "~1-2 GB",
            "vram_bf16": "~1-2 GB"
        },
        "high_res_arbiter": {
            "model": {"dim": 1024, "n_layers": 16, "n_heads": 16, "intermediate_size": 4096, "vocab_size": 8000},
            "training": {"learning_rate": 2e-4},
            "params": "~180M",
            "vram_fp16": "~3-4 GB",
            "vram_bf16": "~3-4 GB"
        }
    }
    return modes.get(mode.lower())

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to TOML config file (optional, will prompt if missing)")
    parser.add_argument("--mode", type=str, choices=["microscope", "tower_of_truth", "high_res_arbiter"], help="Select model mode")
    args = parser.parse_args()
    
    setup()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 1. Hardware Selection Menu
    selected_config_path = args.config
    if selected_config_path is None and local_rank == 0:
        print("\n=== Genesis Hardware Configuration ===")
        print("[1] High VRAM (12+ GB) - Performance Mode")
        print("[2] Low VRAM (4-6 GB) - Compatibility Mode with Gradient Checkpointing")
        h_choice = input("Select hardware tier [1-2]: ").strip()
        h_mapping = {"1": "train_configs/high_vram.toml", "2": "train_configs/low_vram.toml"}
        selected_config_path = h_mapping.get(h_choice, "train_configs/high_vram.toml")
        print(f"Using Config: {selected_config_path}")

    # Fallback/Broadcast for other ranks (simplified)
    if selected_config_path is None:
        selected_config_path = "train_configs/high_vram.toml"
        
    config = toml.load(selected_config_path)
    
    # 2. Protocol Selection Menu
    selected_mode = args.mode
    if selected_mode is None and local_rank == 0:
        print("\n=== Genesis Protocol Selection ===")
        print("[1] Microscope - Baseline (125M params, ~2-3GB VRAM)")
        print("[2] Tower of Truth - Deep Compression (5-8M params, ~1-2GB VRAM, 144 Layers)")
        print("[3] High-Res Arbiter - Maximum Resolution (180M params, ~3-4GB VRAM)")
        m_choice = input("Select protocol [1-3]: ").strip()
        m_mapping = {"1": "microscope", "2": "tower_of_truth", "3": "high_res_arbiter"}
        selected_mode = m_mapping.get(m_choice, "microscope")
        print(f"Selected Protocol: {selected_mode}\n")
    
    if selected_mode is None:
        selected_mode = "microscope"
        
    mode_cfg = get_protocol_config(selected_mode)
    if mode_cfg:
        # Override config with mode-specific values
        for k, v in mode_cfg["model"].items():
            config["model"][k] = v
        for k, v in mode_cfg["training"].items():
            config["training"][k] = v

    # Safety: Auto-correct dim % n_heads == 0
    dim = config["model"]["dim"]
    n_heads = config["model"]["n_heads"]
    if dim % n_heads != 0:
        old_dim = dim
        dim = (dim // n_heads + 1) * n_heads
        config["model"]["dim"] = dim
        if local_rank == 0:
            print(f"Safety Auto-Correct: Adjusting dim from {old_dim} to {dim} to align with {n_heads} heads.")

    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[{local_rank}] Using device: {device}", flush=True)
    
    print(f"[{local_rank}] Initializing {selected_mode.upper()} mode...", flush=True)
    model_cfg = config["model"]
    model = Llama(
        vocab_size=model_cfg["vocab_size"],
        n_layers=model_cfg["n_layers"],
        dim=model_cfg["dim"],
        n_heads=model_cfg["n_heads"],
        intermediate_size=model_cfg["intermediate_size"],
        max_seq_len=model_cfg["max_seq_len"]
    ).to(device)
    
    if local_rank == 0:
        print(f"[{local_rank}] Total Parameters: {model.get_num_params():,}")
    
    # Apply Logos initialization
    if local_rank == 0:
        print(f"[{local_rank}] Applying Jehovah token initialization...", flush=True)
        jhvh_id = 5 # Default
        tokenizer = None
        try:
            tokenizer = GenesisTokenizer("genesis_tokenizer.json")
            jhvh_id = tokenizer.tokenizer.token_to_id("Jehovah") or 5
            print(f"[{local_rank}] Found Jehovah token ID: {jhvh_id}", flush=True)
        except Exception as e:
            print(f"[{local_rank}] Warning: Could not load tokenizer: {e}")
        
        # User requested 1x multiplier for now
        logos_init_hook(model, jehovah_token_id=jhvh_id, multiplier=1.0)
    
    # Enable torch.compile if requested
    if config["training"].get("compile", False):
        if local_rank == 0:
            print(">>> Initiating torch.compile... (This can take 2-5 minutes on the first run)", flush=True)
        model = torch.compile(model)
        if local_rank == 0:
            print(">>> Model compilation complete.", flush=True)
        
    # Parallelism setup
    if dist.is_initialized():
        if config["parallelism"].get("fsdp", False):
            from torch.distributed.fsdp import ShardingStrategy
            model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)
        else:
            model = DDP(model, device_ids=[local_rank])
    else:
        if local_rank == 0:
            print("Parallelism: Running in pure Local mode (No FSDP/DDP overhead)", flush=True)
    
    # Gradient Checkpointing
    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    
    # Cosine schedule
    total_steps = config["training"]["steps"]
    warmup_steps = config["training"].get("warmup_steps", 100)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, s/warmup_steps) * 0.5 * (1.0 + math.cos(math.pi * max(0, s-warmup_steps)/(total_steps-warmup_steps))))
    
    # Data loading - Dynamic dataset selection based on masking config
    masking_config = config.get("masking", {})
    masking_enabled = masking_config.get("enabled", False)
    
    if masking_enabled:
        from datasets.bible_weighted_masking import get_bible_weighted_dataloader
        if local_rank == 0:
            print(f"[{local_rank}] ✨ Using WEIGHTED MASKING strategy", flush=True)
            print(f"[{local_rank}]    Base probability: {masking_config.get('base_probability', 0.4)}", flush=True)
        
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
            print(f"[{local_rank}] Using standard dataset (no masking)", flush=True)
        
        dataloader = get_bible_dataloader(
            corpus_path="nwt_corpus.txt",
            tokenizer=tokenizer if tokenizer else GenesisTokenizer("genesis_tokenizer.json"),
            batch_size=config["training"]["batch_size"],
            max_seq_len=model_cfg["max_seq_len"]
        )

    
    print(f"[{local_rank}] Starting training loop...", flush=True)
    step = 0
    done = False
    while not done:
        for tokens, labels in dataloader:
            step += 1
            if step > total_steps:
                done = True
                break
            
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            
            autocast_dtype = torch.bfloat16 if config["training"]["precision"] == "bf16" else torch.float16
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=(device.type != "cpu"), dtype=autocast_dtype):
                logits, loss = model(tokens, labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if step % 10 == 0 and local_rank == 0:
                print(f"Step {step}/{total_steps}: Loss = {loss.item():.4f}, LR = {optimizer.param_groups[0]['lr']:.2e}", flush=True)
                
            if step % config["checkpoint"]["interval"] == 0 and local_rank == 0:
                save_checkpoint(model.module if hasattr(model, "module") else model, optimizer, step, config["checkpoint"]["dir"])

    cleanup()

if __name__ == "__main__":
    train()
