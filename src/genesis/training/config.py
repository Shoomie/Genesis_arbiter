"""
Training Configuration Modules
==============================
Defines structured configuration for the Genesis training pipeline.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List

@dataclass
class ModelConfig:
    """Configuration for the model architecture."""
    dim: int
    n_layers: int
    n_heads: int
    vocab_size: int
    intermediate_size: int
    norm_type: str = "layernorm"
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        # Filter for known keys
        known_keys = cls.__annotations__.keys()
        return cls(**{k: v for k, v in d.items() if k in known_keys})

@dataclass
class TrainingConfig:
    """Configuration for the training loop."""
    # Hyperparameters
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_steps: int
    grad_accum_steps: int = 1
    warmup_steps: int = 2000
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.1
    
    # System
    precision: str = "bf16"
    compile_model: bool = False
    gradient_checkpointing: bool = False
    device: str = "cuda"
    
    # Data
    bible_dir: str = "../../../Bible"
    
    # Checkpointing & Logging
    save_interval: int = 2000
    log_interval: int = 100
    val_interval: int = 500
    eval_interval: int = 1000
    
    # Evaluation
    eval_recon_samples: int = 50
    eval_aux_samples: int = 20
    detect_grokking: bool = True
    verbose_logging: bool = True
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TrainingConfig':
        known_keys = cls.__annotations__.keys()
        valid_kwargs = {k: v for k, v in d.items() if k in known_keys}
        return cls(**valid_kwargs)

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps
