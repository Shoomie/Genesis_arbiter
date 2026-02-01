"""
Genesis Trainer
===============
Modular training engine for Genesis Arbiter.
"""

import os
import time
import torch
import torch.nn as nn
from datetime import timedelta
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict

# Local imports
from .config import TrainingConfig
from .scheduler import LRScheduler
from .flash_attention_config import print_flash_attention_status
from .callbacks.manager import CallbackManager
from .callbacks.grokking import GrokkingDetector, ProcrustesMonitor, ConceptClusteringMonitor
from ..models.multi_task_wrapper import MultiTaskLlama
from ..evaluation.procedural_evaluator import ProceduralEvaluator
from ..utils.logger import ArbiterLogger

class GenesisTrainer:
    """
    Main trainer class for Genesis models.
    Handles loop, checkpointing, logging, and callbacks.
    """
    
    def __init__(
        self,
        model: MultiTaskLlama,
        tokenizer,
        train_loader,
        config: TrainingConfig,
        val_loader=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        # Scheduler
        self.scheduler = LRScheduler(
            self.optimizer,
            base_lr=config.learning_rate,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            schedule=config.lr_schedule,
            min_lr_ratio=config.min_lr_ratio
        )
        
        # Scaler for mixed precision
        self.scaler = GradScaler(enabled=(config.precision == "bf16" or config.precision == "fp16"))
        
        # Compilation
        if config.compile_model:
            print(f"  [SYSTEM] Compiling model (torch.compile)...")
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"  [WARN] Compilation failed: {e}")
                
        # Logging
        self.logger = ArbiterLogger(
            log_dir=f"logs/run_{int(time.time())}",
            experiment_name=f"genesis_{config.max_steps}_steps"
        )
        
        # Prepare structured config for ArbiterLogger
        logger_config = {
            "training": asdict(self.config),
            "model": {
                "n_layers": getattr(self.model.base, 'n_layers', 0),
                "dim": getattr(self.model.base, 'dim', 0),
                "n_heads": getattr(self.model.base, 'n_heads', 0),
                "vocab_size": getattr(self.model, 'vocab_size', 0),
            },
            "system": {
                "precision": config.precision,
                "device": config.device
            }
        }
        # Start logger silently
        self.logger.start_experiment(logger_config)
        
        # Callbacks
        self.callbacks = CallbackManager()
        if config.detect_grokking:
            self.callbacks.add_callback(GrokkingDetector(patience=100, threshold=0.08))
            self.callbacks.add_callback(ProcrustesMonitor(eval_interval=500))
            self.callbacks.add_callback(ConceptClusteringMonitor(eval_interval=1000))
            
        # State
        self.current_step = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Display model info on init if requested or standard
        self._print_startup_banner()

    def _print_startup_banner(self):
        """Print a clean summary of the training session."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate size based on precision
        param_bytes = 4 if self.config.precision == "fp32" else 2
        model_size_mb = total_params * param_bytes / (1024**2)
        
        print("\n" + "="*60)
        print(f"  G E N E S I S   T R A I N E R")
        print("="*60)
        print(f"  [ARCH]  {self.config.precision.upper()} | {getattr(self.model.base, 'n_layers', 0)} Layers | {getattr(self.model.base, 'dim', 0)} Dim")
        print(f"  [MODEL] Parameters: {total_params/1e6:.2f}M ({trainable_params/1e6:.2f}M trainable)")
        print(f"  [MODEL] Estimated Weight Size: {model_size_mb:.2f} MB")
        print(f"  [DATA]  Batch Size: {self.config.batch_size} (Accum: {self.config.grad_accum_steps})")
        print(f"  [DATA]  Seq Length: {self.config.max_seq_len}")
        print(f"  [TRAIN] Device: {self.device} | Steps: {self.config.max_steps}")
        print("="*60 + "\n")

    def train(self):
        """Execute the training loop."""
        print_flash_attention_status()
        
        self.model.train()
        t0 = time.time()
        
        # Infinite iterator
        data_iter = iter(self.train_loader)
        
        print(f"  [RUN] Training loop engaged. Monitoring diagnostics...")
        
        while self.global_step < self.config.max_steps:
            # Training step
            step_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)
            
            for _ in range(self.config.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                
                # Move all tensors in batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device, non_blocking=True)
                
                # Context for precision
                with autocast(device_type=self.device.type, dtype=self._get_dtype()):
                    # MultiTaskLlama returns (outputs, loss, task_name)
                    # We only need loss for backward, but we could log task_name
                    outputs, loss, task_name = self.model(batch)
                    scaled_loss = loss / self.config.grad_accum_steps
                    
                self.scaler.scale(scaled_loss).backward()
                step_loss += scaled_loss.item()
            
            # Optimization
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Scheduler
            lr = self.scheduler.step(self.global_step)
            
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                dt = time.time() - t0
                tokens_per_sec = (self.config.effective_batch_size * 512) / dt
                print(f"Step {self.global_step}: Loss {step_loss:.4f} | LR {lr:.2e} | {tokens_per_sec:.0f} tok/s")
                self.logger.log_training_step(
                    step=self.global_step,
                    loss=step_loss,
                    learning_rate=lr,
                    tokens_per_sec=tokens_per_sec
                )
                t0 = time.time()
                
            # 5. Validation
            if self.config.enable_validation and self.global_step % self.config.val_interval == 0:
                self.validate()
                
            # Evaluation
            # 4. Extensive Evaluation
            if self.config.enable_extensive_eval and self.global_step % self.config.eval_interval == 0:
                self.evaluate_extensive()
                
            # Checkpointing
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint()
                
            # Callbacks
            self.callbacks.on_batch_end(self.global_step, self.model, step_loss)
            
    def validate(self):
        """Run validation loop."""
        self.model.eval()
        total_loss = 0
        steps = 0
        
        print("\nRunning Validation...")
        with torch.no_grad():
            for batch in self.val_loader:
                if steps >= 50: break # Partial validation for speed
                
                # Move all tensors in batch to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device, non_blocking=True)
                
                with autocast(device_type=self.device.type, dtype=self._get_dtype()):
                    outputs, loss, task_name = self.model(batch)
                
                total_loss += loss.item()
                steps += 1
                
        avg_loss = total_loss / steps
        print(f"Validation Loss: {avg_loss:.4f}\n")
        self.logger.log_evaluation(
            step=self.global_step,
            val_perplexity=avg_loss
        )
        
        self.model.train()
        
    def evaluate_extensive(self):
        """Run extensive procedural evaluation."""
        print(f"\n[Step {self.global_step}] Running Extensive Evaluation...")
        evaluator = ProceduralEvaluator(self.model, self.train_loader, self.tokenizer, self.device)
        
        # Use run_suite to get all metrics (reconstruction, coherence, etc.)
        metrics = evaluator.run_suite(
            use_amp=(self.config.precision in ["bf16", "fp16"]),
            amp_dtype=self._get_dtype(),
            num_recon_samples=self.config.eval_recon_samples,
            num_aux_samples=self.config.eval_aux_samples
        )
        
        # Log to ArbiterLogger
        self.logger.log_metrics(metrics, step=self.global_step)
        
        # Already prints summary inside run_suite
        
    def save_checkpoint(self, path=None):
        """Save model checkpoint."""
        if path is None:
            path = f"checkpoints/step_{self.global_step}.pt"
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': asdict(self.config),
            'step': self.global_step
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path):
        """Load checkpoint."""
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['step']
        print(f"Resumed from step {self.global_step}")

    def _get_dtype(self):
        if self.config.precision == "bf16": return torch.bfloat16
        if self.config.precision == "fp16": return torch.float16
        return torch.float32
