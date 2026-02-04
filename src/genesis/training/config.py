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
    dim: int = 512
    n_layers: int = 12
    n_heads: int = 8
    vocab_size: int = 260
    intermediate_size: int = 2048
    norm_type: str = "deepnorm"
    max_seq_len: int = 1024
    
    def merge(self, other_dict: Dict[str, Any]) -> 'ModelConfig':
        """Merge values from a dictionary into a new ModelConfig instance."""
        base_dict = asdict(self)
        for k, v in other_dict.items():
            if k in base_dict and v is not None:
                base_dict[k] = v
        return ModelConfig(**base_dict)
    
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
    
    # Plateau Recovery
    plateau_lr_drop: float = 0.5
    plateau_lr_patience: int = 1000
    
    # System
    precision: str = "bf16"
    compile_model: bool = False
    use_cuda_graph: bool = False
    gradient_checkpointing: bool = False
    device: str = "cuda"
    
    # Data
    bible_dir: str = "../../../Bible"
    max_seq_len: int = 512
    
    # Checkpointing & Logging
    save_interval: int = 2000
    log_interval: int = 100
    val_interval: int = 500
    eval_interval: int = 1000
    multilingual_log_interval: int = 25
    
    # Dynamic WWM Trigger
    wwm_trigger_steps: int = 5000
    wwm_window: int = 2000
    wwm_threshold: float = 0.005
    wwm_mask_prob: float = 0.15
    wwm_ramp_steps: int = 1000
    
    # Span Masking
    span_trigger_steps: int = 5000
    span_window: int = 2000
    span_threshold: float = 0.005
    span_mask_prob: float = 0.15
    span_min_len: int = 3
    span_max_len: int = 5
    span_ramp_steps: int = 500
    
    # Evaluation
    enable_validation: bool = True
    enable_extensive_eval: bool = True
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
