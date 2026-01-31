import torch
import torch.nn.functional as F
from safetensors.torch import load_file
import os
import sys
import math

# Add src directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from genesis.models.llama.model import Llama
from genesis.models.tokenizer import GenesisTokenizer

def calculate_perplexity(model, tokenizer, text, device):
    inputs = tokenizer(text).to(device)
    with torch.no_grad():
        logits, _ = model(inputs)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return math.exp(loss.item())

def run_perplexity_test():
    checkpoint_path = os.path.join(parent_dir, "checkpoints", "step_2000", "model.safetensors")
    tokenizer_path = os.path.join(parent_dir, "data", "genesis_char_tokenizer.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Axiomatic Stability Check ===")
    
    # 1. Load Tokenizer
    tokenizer = GenesisTokenizer(tokenizer_path)
    print(f"Tokenizer loaded: vocab size = {tokenizer.vocab_size}")
    
    # 2. Initialize Model (Resonant Blade 144-Layer)
    # Corrected based on checkpoint mismatch:
    # vocab_size: 12000, intermediate_size: 576 (2 * dim)
    vocab_size = 12000
    print(f"Using architecture: Dim 288, Layers 144, Heads 12, Vocab {vocab_size}, Intermediate {576}")

    model = Llama(
        vocab_size=vocab_size,
        n_layers=144,
        dim=288,
        n_heads=12,
        intermediate_size=576,
        max_seq_len=1024
    ).to(device)

    # 3. Load Checkpoint
    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded from {checkpoint_path}")

    # Set TF32 for performance if available
    torch.set_float32_matmul_precision('high')
    
    # 4. Define Test Sets
    test_cases = {
        "Scriptural": "In the beginning God created the heavens and the earth.",
        "Secular Noise": "The stock market fluctuations were influenced by quarterly reports.",
        "Logical Axiom": "The area of a circle is calculated by multiplying pi by the radius squared."
    }

    print("\nResult Table:")
    print("-" * 50)
    print(f"{'Category':<20} | {'Perplexity':<15}")
    print("-" * 50)

    for category, text in test_cases.items():
        ppl = calculate_perplexity(model, tokenizer, text, device)
        print(f"{category:<20} | {ppl:<15.4f}")
    print("-" * 50)

if __name__ == "__main__":
    run_perplexity_test()
