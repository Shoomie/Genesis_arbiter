import torch
import os
from safetensors.torch import save_file

def save_checkpoint(model, optimizer, step, checkpoint_dir, export_format="safetensors"):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    # Save standard torch checkpoint
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoint_path, "training_state.pt"))
    
    # Export for downstream use (Steering Project compatible)
    if export_format == "safetensors":
        state_dict = model.state_dict()
        metadata = {
            "project": "Genesis",
            "architecture": "Micro-Llama-124M",
            "logos_init": "True",
            "step": str(step)
        }
        save_file(state_dict, os.path.join(checkpoint_path, "model.safetensors"), metadata=metadata)
        print(f"Exported Steering-compatible Safetensors checkpoint to {checkpoint_path}")
    
    print(f"Checkpoint saved at step {step}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0
    
    state = torch.load(os.path.join(checkpoint_path, "training_state.pt"))
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    return state['step']
