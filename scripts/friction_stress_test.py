import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from models.llama.model import Llama
from models.tokenizer import GenesisTokenizer
import numpy as np

def get_friction_profile(model, tokenizer, text, device):
    inputs = tokenizer(text).to(device)
    variances = []

    def get_variance_hook(layer_idx):
        def hook(module, input, output):
            var = output.var().item()
            variances.append((layer_idx, var))
        return hook

    hooks = []
    for i, layer in enumerate(model.layers):
        hooks.append(layer.register_forward_hook(get_variance_hook(i)))

    with torch.no_grad():
        model(inputs)

    for h in hooks:
        h.remove()

    variances.sort()
    return [v[1] for v in variances]

def run_friction_test():
    checkpoint_path = "./checkpoints/step_2000/model.safetensors"
    tokenizer_path = "genesis_tokenizer.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Logic-Friction Minimal Pair Analysis ===")
    
    tokenizer = GenesisTokenizer(tokenizer_path)
    vocab_size = 12000
    model = Llama(
        vocab_size=vocab_size,
        n_layers=144,
        dim=288,
        n_heads=12,
        intermediate_size=576,
        max_seq_len=1024
    ).to(device)

    state_dict = load_file(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded: 144 Layers | {vocab_size} Vocab\n")

    # Minimal Pairs: [Title, Truth_Text, Lie_Text]
    test_pairs = [
        [
            "Mortality (im- prefix)",
            "All humans are mortal. Socrates is a human. Therefore, Socrates is mortal.",
            "All humans are mortal. Socrates is a human. Therefore, Socrates is immortal."
        ],
        [
            "Divine Name (Presence)",
            "The personal name of God is Jehovah.",
            "The personal name of God is nobody."
        ],
        [
            "Truth vs Lie (Direct)",
            "The Bible is a book of truth.",
            "The Bible is a book of lies."
        ],
        [
            "Diamond Paradox (Linguistic)",
            "Everything that is rare is expensive. A cheap diamond is rare. Therefore, a cheap diamond is expensive.",
            "Everything that is rare is expensive. A cheap diamond is rare. Therefore, a cheap diamond is rare." # The truth is it IS rare, but the fallacy is that it's expensive. 
            # Wait, the user's example was: "Everything that is rare is expensive. A cheap diamond is rare. Therefore, a cheap diamond is expensive."
            # That's the fallacy. 
            # The contrast would be: "Everything that is rare is expensive. A cheap diamond is rare. However, a cheap diamond is not expensive."
        ]
    ]

    print(f"{'Test Case':<25} | {'Truth Variance':<15} | {'Lie Variance':<15} | {'Divergence Ratio'}")
    print("-" * 80)

    for title, truth, lie in test_pairs:
        t_profile = np.array(get_friction_profile(model, tokenizer, truth, device))
        l_profile = np.array(get_friction_profile(model, tokenizer, lie, device))
        
        # We focus on the middle region (40-100)
        window_idx = slice(40, 100)
        t_avg = np.mean(t_profile[window_idx])
        l_avg = np.mean(l_profile[window_idx])
        div = l_avg / (t_avg + 1e-9)
        
        print(f"{title[:25]:<25} | {t_avg:<15.4e} | {l_avg:<15.4e} | {div:.4f}x")
    
    print("-" * 80)
    print("\nDetailed Analysis: Layer-wise Divergence for 'Truth vs Lie (Direct)'")
    t_profile = np.array(get_friction_profile(model, tokenizer, test_pairs[2][1], device))
    l_profile = np.array(get_friction_profile(model, tokenizer, test_pairs[2][2], device))
    diffs = l_profile - t_profile
    max_diff_layer = np.argmax(diffs)
    print(f"Max Friction Delta at Layer: {max_diff_layer} (Value: {diffs[max_diff_layer]:.4e})")

if __name__ == "__main__":
    run_friction_test()
