"""
Learning Rate Scheduler
=======================
Handles learning rate adjustments during training.
"""

import math

class LRScheduler:
    """
    Manages learning rate decay schedules.
    """
    
    def __init__(
        self,
        optimizer,
        base_lr: float,
        warmup_steps: int,
        max_steps: int,
        schedule: str = "cosine",
        min_lr_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.schedule = schedule
        self.min_lr_ratio = min_lr_ratio
        
    def step(self, current_step: int):
        """Update optimizer learning rate for the current step."""
        lr = self.get_lr(current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
            
    def get_lr(self, step: int) -> float:
        """Calculate learning rate multiplier for the given step."""
        # 1. Linear Warmup
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / (self.warmup_steps + 1)
        
        # 2. Main Schedule
        if step > self.max_steps:
            return self.base_lr * self.min_lr_ratio
            
        if self.schedule == "constant":
            return self.base_lr
            
        elif self.schedule == "cosine":
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decayed = (1 - self.min_lr_ratio) * cosine_decay + self.min_lr_ratio
            return self.base_lr * decayed
            
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
