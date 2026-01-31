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

from genesis.models.tokenizer import GenesisTokenizer
from genesis.models.llama.model import Llama
from genesis.models.multi_task_wrapper import MultiTaskLlama

def get_checkpoints():
    """List valid checkpoints in checkpoints/."""
    base_dir = project_root / "checkpoints"
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        return []
        
    # Get all .pt files
    files = list(base_dir.glob("*.pt"))
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

        # 2. Load Tokenizer (REQUIRED for vocab_size)
        # Try to find tokenizer in data/ directory first
        tokenizer_path = project_root / "data" / "genesis_char_tokenizer.json"
        if not tokenizer_path.exists():
             # Fallback to root
             tokenizer_path = project_root / "genesis_tokenizer.json"
             
        if tokenizer_path.exists():
            print(f"  Loading tokenizer from {tokenizer_path.name}...")
            tokenizer = GenesisTokenizer(str(tokenizer_path))
        else:
            print("[ERROR] Tokenizer file not found!")
            return None, None
            
        print(f"  Model Config: Dim={config.get('dim')}, Layers={config.get('n_layers')}")
        
        # 3. Initialize Model
        # We only pass keys that Llama.__init__ actually accepts
        llama_keys = {'vocab_size', 'n_layers', 'dim', 'n_heads', 'intermediate_size', 'max_seq_len', 'norm_type'}
        model_init_config = {k: v for k, v in config.items() if k in llama_keys}
        
        # Determine vocab_size strictly from state_dict to avoid mismatches
        sd = checkpoint['model']
        if 'base.tok_embeddings.weight' in sd:
            checkpoint_vocab_size = sd['base.tok_embeddings.weight'].shape[0]
        elif 'tok_embeddings.weight' in sd:
            checkpoint_vocab_size = sd['tok_embeddings.weight'].shape[0]
        else:
            checkpoint_vocab_size = config.get('vocab_size', tokenizer.vocab_size)
            
        model_init_config['vocab_size'] = checkpoint_vocab_size
        
        # Synchronize tokenizer if vocab_size is compressed
        if model_init_config['vocab_size'] < tokenizer.vocab_size:
            print(f"  [!] Checkpoint has compressed vocab ({model_init_config['vocab_size']})")
            print(f"      Attempting to synchronize tokenizer...")
            
            # Try to determine language from config or filename
            target_langs = config.get('target_languages')
            if not target_langs:
                # Guess from filename (e.g., step_500en.pt -> en)
                import re
                match = re.search(r'_([a-z]{2,3})\.pt$', checkpoint_path.name)
                if match:
                    target_langs = [match.group(1)]
                    print(f"      Inferred language from filename: {target_langs}")
            
            if target_langs:
                # Check root/Bible first, then engine/Bible, then parent/Bible
                bible_dir = project_root / "Bible"
                if not bible_dir.exists():
                    bible_dir = project_root / "engine" / "Bible"
                if not bible_dir.exists():
                    bible_dir = project_root.parent / "Bible"
                
                if bible_dir.exists():
                    print(f"      Reconstructing vocabulary mapping for {target_langs}...")
                    try:
                        from genesis.datasets.multi_task_sampler import MultiTaskDataset
                        # Create a dummy dataset to trigger compression logic
                        ds = MultiTaskDataset(
                            bible_data_dir=str(bible_dir),
                            tokenizer=tokenizer,
                            device=torch.device('cpu'),
                            target_languages=target_langs
                        )
                        print(f"      [OK] Tokenizer synchronized (New vocab: {tokenizer.vocab_size})")
                    except Exception as e:
                        print(f"      [WARN] Failed to synchronize tokenizer: {e}")
                else:
                    print(f"      [WARN] Bible data not found at {bible_dir}. Generation might fail.")
            else:
                 print(f"      [WARN] Could not determine language for compression. Generation might fail.")
        
        # Ensure other essential params are present
        if 'dim' not in model_init_config: model_init_config['dim'] = 512
        if 'n_layers' not in model_init_config: model_init_config['n_layers'] = 8
        if 'n_heads' not in model_init_config: model_init_config['n_heads'] = 8
        
        print(f"  Final Model Config: Dim={model_init_config.get('dim')}, Layers={model_init_config.get('n_layers')}, Vocab={model_init_config['vocab_size']}")
        
        # Initialize Base Llama
        base_model = Llama(**model_init_config)
        
        # Wrap with MultiTaskLlama to match checkpoint keys (base.layers...)
        model = MultiTaskLlama(
            base_model=base_model,
            dim=model_init_config['dim'],
            vocab_size=model_init_config['vocab_size']
        )
        
        # Load state dict
        sd = checkpoint['model']
        
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
            
            # Stop if EOS (assuming 0 is PAD/EOS for now, or check tokenizer)
            if next_token.item() == 0: # 1 is usually EOS, 0 PAD. GenesisTokenizer likely uses 2 for EOS? 
                # Let's check genesis_tokenizer.json manually if this persists.
                # Usually standard sentencepiece is 1=BOS, 2=EOS.
                break
                
    # Decode
    output_ids = generated[0].tolist()
    
    # DEBUG: Print first few output IDs and their chars
    # print(f"DEBUG: IDs={output_ids[:20]}...")
    
    return tokenizer.decode(output_ids)

def load_central_config():
    """Load settings from genesis_config.toml at project root."""
    try:
        # Find project root (1 level up from this file's folder tools/)
        root = Path(__file__).parent.parent
        config_path = root / "genesis_config.toml"
        if config_path.exists():
            return toml.load(config_path)
    except Exception as e:
        print(f"[WARN] Failed to load central config: {e}")
    return {}

def main(initial_temp=None, initial_max_tokens=None, force_device=None):
    """Main interaction loop."""
    print_header()
    
    # Load central config for defaults
    central_cfg = load_central_config()
    interact_cfg = central_cfg.get("interaction", {})
    
    # 1. Device Setup
    if force_device:
        device = torch.device(force_device)
    elif "device" in interact_cfg and interact_cfg["device"] != "auto":
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
        print("[!] No checkpoints found in engine/checkpoints/test/")
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
