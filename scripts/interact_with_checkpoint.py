#!/usr/bin/env python3
"""
Interactive Checkpoint Tester
=============================
Allows users to chat with trained model checkpoints found in engine/checkpoints/test.
"""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import io

# Force UTF-8 stdout for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add engine to path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.append(str(project_root))

from engine.models.tokenizer import GenesisTokenizer
from engine.models.llama.model import Llama
from engine.models.multi_task_wrapper import MultiTaskLlama

def get_checkpoints():
    """List valid checkpoints in engine/checkpoints/test."""
    base_dir = project_root / "engine" / "checkpoints" / "test"
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
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 1. Load Config
        if 'config' in checkpoint:
            config = checkpoint['config']
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
        # Try to find tokenizer in project root first
        tokenizer_path = project_root / "engine" / "genesis_tokenizer.json"
        if not tokenizer_path.exists():
             # Fallback to standard location
             tokenizer_path = project_root / "genesis_tokenizer.json"
             
        if tokenizer_path.exists():
            print(f"  Loading tokenizer from {tokenizer_path.name}...")
            tokenizer = GenesisTokenizer(str(tokenizer_path))
        else:
            print("[ERROR] Tokenizer file not found!")
            return None, None
            
        print(f"  Model Config: Dim={config.get('dim')}, Layers={config.get('n_layers')}")
        
        # 3. Initialize Model
        # Remove 'params' if present (display only key)
        model_config = {k: v for k, v in config.items() if k != 'params'}
        
        # Ensure essential params are present
        model_config['vocab_size'] = tokenizer.vocab_size # TRUST TOKENIZER
        if 'dim' not in model_config: model_config['dim'] = 512
        if 'n_layers' not in model_config: model_config['n_layers'] = 8
        if 'n_heads' not in model_config: model_config['n_heads'] = 8
        
        # Initialize Base Llama
        base_model = Llama(**model_config)
        
        # Wrap with MultiTaskLlama to match checkpoint keys (base.layers...)
        # We need to pass required args to wrapper
        model = MultiTaskLlama(
            base_model=base_model,
            dim=model_config['dim'],
            vocab_size=model_config['vocab_size']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model']) # This fits the wrapped model
        
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

def generate_response(model, tokenizer, prompt, device, max_new_tokens=100, temperature=0.7):
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
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat((generated, next_token), dim=1)
            
            # Stop if EOS (assuming 0 is PAD/EOS for now, or check tokenizer)
            if next_token.item() == 0: # 1 is usually EOS, 0 PAD. GenesisTokenizer likely uses 2 for EOS? 
                # Let's check genesis_tokenizer.json manually if this persists.
                # Usually standard sentencepiece is 1=BOS, 2=EOS.
                break
                
    # Decode
    output_ids = generated[0].tolist()
    return tokenizer.decode(output_ids)

def main():
    print(">>> Genesis Checkpoint Tester\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}\n")
    
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
        
    print(f"\n[OK] Model loaded! Entering chat mode (Type 'exit' or 'quit' to stop).")
    print("-" * 50)
    
    # Chat Loop
    while True:
        try:
            print("\n" + "="*50)
            print("NEW SESSION (Context Cleared)")
            print("="*50)
            
            user_input = input("Enter Prompt: ")
            if user_input.lower() in ('exit', 'quit'):
                break
                
            if not user_input.strip():
                continue
                
            print("\nModel Response:")
            print("-" * 20)
            
            # Determine generation params from input flags if present (advanced usage)
            temp = 0.7
            if " --temp " in user_input:
                parts = user_input.split(" --temp ")
                user_input = parts[0]
                try:
                    temp = float(parts[1])
                except:
                    pass
            
            # Generate full sequence including prompt
            full_response = generate_response(model, tokenizer, user_input, device, temperature=temp)
            
            # Extract just the new part for cleaner display
            # We can re-encode the prompt to find its length in tokens, then decode the rest?
            # A simple string check:
            if full_response.startswith(user_input):
                new_content = full_response[len(user_input):]
                print(new_content)
            else:
                # Fallback if tokenization changed spacing
                print(full_response)
                
            print("-" * 20)
            
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"\n[Error] Generation failed: {e}")

if __name__ == "__main__":
    main()
