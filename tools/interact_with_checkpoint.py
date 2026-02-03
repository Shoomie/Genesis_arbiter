#!/usr/bin/env python3
"""
Interactive Checkpoint Tester
=============================
Allows users to chat with trained model checkpoints found in engine/checkpoints/test.
"""

import os
import sys
import torch
import argparse
import torch.nn.functional as F
from pathlib import Path
import io
import numpy as np
import toml

# Add safe globals for PyTorch 2.6+ to allow loading older/custom checkpoints
if hasattr(torch, 'serialization'):
    try:
        # Handle the specific numpy scalar issue in PyTorch 2.6+
        if hasattr(np, '_core'): # Newer numpy
            torch.serialization.add_safe_globals([np._core.multiarray.scalar])
        else: # Older numpy
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except:
        pass

# Force UTF-8 stdout for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add src to path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

# Genesis Imports
from genesis.utils.config_loader import get_config_section
from genesis.models.tokenizer import GenesisTokenizer
from genesis.models.llama.model import Llama
from genesis.models.multi_task_wrapper import MultiTaskLlama

def print_header():
    """Print the interactive chat header."""
    print("\n" + "="*60)
    print("  G E N E S I S   I N T E R A C T I V E   C H A T")
    print("="*60)
    print("  Byte-Level UTF-8 Inference Pipeline | PyTorch 2.6 Ready")
    print("="*60 + "\n")

def get_checkpoints():
    """List valid checkpoints in various common directories."""
    search_paths = [
        project_root / "src" / "genesis" / "checkpoints" / "test",
    ]
    
    files = []
    for base_dir in search_paths:
        if base_dir.exists():
            files.extend(list(base_dir.glob("*.pt")))
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files

def load_model(checkpoint_path, device):
    """Load model and tokenizer from checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_path.name}...")
    
    try:
        # Use weights_only=False for researchers loading their own checkpoints 
        # to avoid PyTorch 2.6+ security restrictions on local files.
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            # Fallback for older torch versions < 2.4
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 1. Load Config
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"  Loaded config from checkpoint: {config}")
        else:
            print("[WARN] No config found in checkpoint. Using default small config.")
            config = {
                "vocab_size": 32000,
                "dim": 512,
                "n_layers": 8,
                "n_heads": 8,
                "max_seq_len": 512
            }

        # 2. Initialize Tokenizer (Always Byte-Level in the new regime)
        sd = checkpoint.get('model_state_dict', checkpoint.get('model', {}))
        if not sd:
            print("[ERROR] No model state dict found in checkpoint!")
            return None, None
        
        tokenizer = GenesisTokenizer(type='byte')
        checkpoint_vocab_size = sd.get('base.tok_embeddings.weight', 
                                       sd.get('tok_embeddings.weight', 
                                              torch.tensor([0]*260))).shape[0]
        
        if checkpoint_vocab_size != 260:
            print(f"[WARN] Checkpoint vocab size ({checkpoint_vocab_size}) does not match current Byte-Level regime (260).")

        print(f"  Model Config: Dim={config.get('dim')}, Layers={config.get('n_layers')}, Vocab={checkpoint_vocab_size}")
        
        # 3. Initialize Model
        # CRITICAL: If config is missing architecture, infer from state_dict
        sd = checkpoint.get('model_state_dict', checkpoint.get('model', {}))
        
        # Infer layers
        if 'n_layers' not in config:
            layer_keys = [k for k in sd.keys() if 'layers.' in k]
            if layer_keys:
                max_layer = max([int(k.split('layers.')[1].split('.')[0]) for k in layer_keys])
                config['n_layers'] = max_layer + 1
                print(f"  [AUTO] Inferred n_layers: {config['n_layers']}")
        
        # Infer dim
        if 'dim' not in config:
            if 'base.tok_embeddings.weight' in sd:
                config['dim'] = sd['base.tok_embeddings.weight'].shape[1]
            elif 'tok_embeddings.weight' in sd:
                config['dim'] = sd['tok_embeddings.weight'].shape[1]
            if 'dim' in config:
                print(f"  [AUTO] Inferred dim: {config['dim']}")
        
        # Infer intermediate_size
        if 'intermediate_size' not in config:
            ffn_keys = [k for k in sd.keys() if 'feed_forward.w1.weight' in k]
            if ffn_keys:
                config['intermediate_size'] = sd[ffn_keys[0]].shape[0]
                print(f"  [AUTO] Inferred intermediate_size: {config['intermediate_size']}")
        
        # Infer norm_type
        if 'norm_type' not in config:
            if any('.attention_norm.norm.' in k for k in sd.keys()):
                config['norm_type'] = "deepnorm"
            else:
                config['norm_type'] = "rmsnorm"
            print(f"  [AUTO] Inferred norm_type: {config['norm_type']}")

        # Ensure essential params are present with defaults if still missing
        if 'dim' not in config: config['dim'] = 512
        if 'n_layers' not in config: config['n_layers'] = 8
        if 'n_heads' not in config: config['n_heads'] = 8
        if 'intermediate_size' not in config: config['intermediate_size'] = 3072

        print(f"  Final Model Config: Dim={config.get('dim')}, Layers={config.get('n_layers')}, Vocab={checkpoint_vocab_size}, Norm={config.get('norm_type')}")
        
        # Filter for Llama.__init__ keys
        llama_keys = {'vocab_size', 'n_layers', 'dim', 'n_heads', 'intermediate_size', 'max_seq_len', 'norm_type'}
        model_init_config = {k: v for k, v in config.items() if k in llama_keys}
        model_init_config['vocab_size'] = checkpoint_vocab_size
        
        # Initialize Base Llama
        base_model = Llama(**model_init_config)
        
        # Wrap with MultiTaskLlama to match checkpoint keys (base.layers...)
        model = MultiTaskLlama(
            base_model=base_model,
            dim=model_init_config['dim'],
            vocab_size=model_init_config['vocab_size']
        )
        
        # Load state dict
        sd = checkpoint.get('model_state_dict', checkpoint.get('model'))
        
        # CRITICAL FIX: The base model's 'output' layer might not have been trained 
        # (MultiTaskLlama uses 'lm_head'). If weights are not tied in the checkpoint,
        # we must ensure the trained 'lm_head' weights are used for the base model during inference.
        if 'lm_head.weight' in sd:
            if 'base.output.weight' in sd:
                # print("  [DEBUG] Syncing base.output.weight with lm_head.weight")
                sd['base.output.weight'] = sd['lm_head.weight']
            if 'output.weight' in sd:
                # print("  [DEBUG] Syncing output.weight with lm_head.weight")
                sd['output.weight'] = sd['lm_head.weight']

        model.load_state_dict(sd) # This fits the wrapped model
        
        # For generation, we only need the base model
        # (Llama forward returns logits, MultiTaskLlama wrapper expects dict batch)
        # So we extract the base.
        inference_model = model.base
        inference_model.to(device)
        inference_model.eval()

        return inference_model, tokenizer
        
    except Exception as e:
        print(f"\n[ERROR] Failed to load checkpoint: {e}")
        return None, None

def generate_response(model, tokenizer, prompt, device, max_new_tokens=300, temperature=0.7):
    """Generate text from the model."""
    
    # Tokenize
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        generated = input_tensor
        
        for _ in range(max_new_tokens):
            # Forward pass
            # Handle context window (Llama expects tokens)
            ctx = generated[:, -model.max_seq_len:] if hasattr(model, 'max_seq_len') else generated[:, -512:]
            
            # Llama forward returns (logits, loss=None)
            logits, _ = model(ctx)
            next_token_logits = logits[:, -1, :]
            
            if temperature > 0:
                # Sample with temperature
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy sampling
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append
            generated = torch.cat((generated, next_token), dim=1)
            
            # Stop if EOS
            eos_id = getattr(tokenizer, 'eos_id', 2) # GenesisTokenizer usually has .eos_id
            if next_token.item() == eos_id:
                break
                
    # Decode
    output_ids = generated[0].tolist()
    
    # DEBUG: Print first few output IDs and their chars
    # print(f"DEBUG: IDs={output_ids[:20]}...")
    
    return tokenizer.decode(output_ids)

def load_central_config():
    """Load settings from genesis_config.toml at project root."""
    return get_config_section("inference")

def main(initial_temp=None, initial_max_tokens=None, force_device=None):
    """Main interaction loop."""
    print_header()
    
    # Load central config for defaults
    interact_cfg = load_central_config()
    
    # 1. Device Setup
    if force_device:
        device = torch.device(force_device)
    elif "device" in interact_cfg and interact_cfg["device"] != "auto" and interact_cfg["device"] is not None:
        device = torch.device(interact_cfg["device"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set initial parameters from args or config defaults
    if initial_temp is None:
        initial_temp = interact_cfg.get("temperature", 0.7)
    if initial_max_tokens is None:
        initial_max_tokens = interact_cfg.get("max_tokens", 100)
    
    # List checkpoints
    checkpoints = get_checkpoints()
    
    if not checkpoints:
        print("[!] No checkpoints found in common search paths (src/genesis/checkpoints/test, checkpoints/)")
        return
        
    print("Available Checkpoints:")
    for i, cp in enumerate(checkpoints):
        size_mb = cp.stat().st_size / (1024 * 1024)
        print(f"  [{i+1}] {cp.name} ({size_mb:.1f} MB)")
        
    print("\n  [0] Exit")
    
    # Selection
    while True:
        try:
            choice = input("\nSelect checkpoint number: ").strip()
            if choice == '0':
                return
            idx = int(choice) - 1
            if 0 <= idx < len(checkpoints):
                target_cp = checkpoints[idx]
                break
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a number.")
            
    # Load
    model, tokenizer = load_model(target_cp, device)
    if not model:
        return
        
    print(f"\n[OK] Model loaded! Entering chat mode.")
    print(f"{'='*60}")
    print("  TIP: You can use flags at the end of your prompt:")
    print("       'Your prompt --temp 0.8 --tokens 50'")
    print("       Defaults: Temperature=0.0, Tokens=100")
    print(f"{'='*60}")
    print("\n(Type 'exit' or 'quit' to stop)")
    print("-" * 60)
    
    # Chat Loop
    while True:
        try:
            print("\n" + "="*50)
            print("NEW SESSION (Context Cleared)")
            print("="*50)
            
            user_input = input("Enter Prompt: ")
            if user_input.lower() in ('exit', 'quit'):
                break
                
            # Parse flags
            temp = initial_temp
            max_tokens = initial_max_tokens
            
            # Simple regex-less parsing for flags
            words = user_input.split()
            clean_prompt_words = []
            
            skip_next = False
            for i, word in enumerate(words):
                if skip_next:
                    skip_next = False
                    continue
                    
                if word == "--temp" and i + 1 < len(words):
                    try:
                        temp = float(words[i+1])
                        skip_next = True
                    except:
                        pass
                elif word == "--tokens" and i + 1 < len(words):
                    try:
                        max_tokens = int(words[i+1])
                        skip_next = True
                    except:
                        pass
                else:
                    clean_prompt_words.append(word)
            
            user_input = " ".join(clean_prompt_words)
            
            if not user_input.strip():
                continue
                
            # Display current settings
            print(f"\n[SETTINGS] Sampling Mode: {'Greedy' if temp == 0 else f'Temp {temp}'} | Length: {max_tokens} tokens")
            print(f"Model Response:")
            print("-" * 20)
            
            # Generate full sequence including prompt
            full_response = generate_response(model, tokenizer, user_input, device, max_new_tokens=max_tokens, temperature=temp)
            
            # Extract just the new part for cleaner display
            # We use a more robust way to find the generated part
            # by looking for the prompt in the decoded string
            prompt_normalized = tokenizer.decode(tokenizer.encode(user_input))
            
            if full_response.startswith(prompt_normalized):
                new_content = full_response[len(prompt_normalized):]
                print(new_content)
            elif user_input in full_response:
                new_content = full_response.split(user_input, 1)[1]
                print(new_content)
            else:
                # Fallback if tokenization changed spacing drastically
                print("\n[WARNING] Could not isolate response from prompt. Full output:")
                print(full_response)
                
            print("-" * 20)
            
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"\n[Error] Generation failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis Interactive Checkpoint Tester")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--temp", type=float, default=0.7, help="Default temperature")
    parser.add_argument("--max-tokens", type=int, default=100, help="Default max tokens")
    args = parser.parse_args()
    
    # Override defaults with args
    # Note: main() needs to be updated to accept these or we set them globally
    # For simplicity, let's update main() to accept them.
    main(initial_temp=args.temp, initial_max_tokens=args.max_tokens, force_device=args.device)
