"""
Genesis Trainer
===============
Modular training engine for Genesis Arbiter.
"""

import os
import time
import json
import numpy as np
import torch
import sys
import threading
import queue
import torch.nn as nn
from datetime import timedelta, datetime
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict

# Local imports
from .config import TrainingConfig, ModelConfig
from .scheduler import LRScheduler
from .flash_attention_config import print_flash_attention_status, FlashAttentionConfig
from .analytics import LossAnalytics
from .callbacks.manager import CallbackManager
from .callbacks.grokking import GrokkingDetector, ProcrustesMonitor, ConceptClusteringMonitor
from ..models.multi_task_wrapper import MultiTaskLlama
from ..evaluation.procedural_evaluator import ProceduralEvaluator
from ..utils.logger import ArbiterLogger, enable_windows_ansi
from ..utils.visualizer import GenesisVisualizer

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
        val_loader=None,
        boundary_tensor: Optional[torch.Tensor] = None
    ):
        enable_windows_ansi() # Ensure fast console updates work on Windows
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.arch_config = getattr(model, 'config', None)
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizer
        use_fused = self.device.type == 'cuda'
        optimizer_kwargs = {
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "fused": use_fused
        }
        if config.use_cuda_graph:
            optimizer_kwargs["capturable"] = True
            
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            **optimizer_kwargs
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
                "max_seq_len": self.config.max_seq_len,
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
        self.loss_ema = None
        self.wwm_active = False
        self.wwm_activated_step = None
        self.span_active = False
        self.span_activated_step = None
        
        self.lr_stun_executed = False # Track if automated stun happened
        
        # Multilingual Telemetry
        loader_obj = getattr(self.train_loader, 'loader', self.train_loader)
        self.locales = getattr(loader_obj, 'locales', [])
        self.locale_to_id = getattr(loader_obj, 'locale_to_id', {})
        self.lang_stats = {lang: {"ema": None, "steps": 0} for lang in self.locales}
        
        # Interactive Dashboard
        self.visualizer = GenesisVisualizer(self.logger, self.logger.current_run_id)
        self.visualizer.start()
        
        # Plateau Detection State (Persisted)
        self.last_ema_check = None
        self.ema_at_last_check = None
        
        # Research Performance Tracking
        self.ema_at_wwm = None
        self.ema_at_span = None
        self.wwm_final_improvement = 0.0
        self.span_improvement = 0.0
        
        self.analytics = LossAnalytics(window_size=100)
        self.wwm_analytics = LossAnalytics(window_size=100)
        self.span_analytics = LossAnalytics(window_size=100)
        
        # Buffers & Stats
        self.static_loss = torch.tensor(0.0, device=self.device)
        self.interval_tokens = 0
        self.interval_time = 0.0
        self.smoothed_s_per_step = None
        self.vram_warned = False
        
        # Boundary Map
        self.boundary_tensor = boundary_tensor
        self.wwm_loaded = self.boundary_tensor is not None
        
        # Inject into loader if it's InfiniteGPULoader and wasn't set
        if self.boundary_tensor is not None and hasattr(self.train_loader, 'loader'):
            if getattr(self.train_loader.loader, 'boundary_tensor', None) is None:
                self.train_loader.loader.boundary_tensor = self.boundary_tensor.to(self.device)
            
        # Performance Tracking
        self.interval_tokens = 0
        self.interval_time = 0.0
        self.smoothed_s_per_step = None
        self.vram_warned = False
        
        # CUDA Graph State
        self.cuda_graph = None
        self.static_tokens = None
        self.static_labels = None
        self.graph_warmup_steps = 10
        
        # Display model info on init if requested or standard
        self.is_compiled = config.compile_model
        # self.wwm_loaded already set above
        self._print_startup_banner()
        
        # Session Logging
        self.session_log = []
        self.last_stats_display = ""
        
        # Async Stats Engine
        self.stats_queue = queue.Queue(maxsize=1)
        self.stats_stop_event = threading.Event()
        self.stats_thread = threading.Thread(target=self._stats_worker_loop, daemon=True)
        self.stats_thread.start()

    def _print_startup_banner(self):
        """Print a comprehensive high-fidelity dashboard (Wide Layout)."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"
        
        param_bytes = 4 if self.config.precision == "fp32" else 2
        model_size_mb = total_params * param_bytes / (1024**2)

        width = 110
        print("\033[1m" + "="*width + "\033[0m")
        print(f"G E N E S I S   A R B I T E R   |   T R A I N I N G   D A S H B O A R D".center(width))
        print(f"DASHBOARD: http://localhost:{self.visualizer.port}".center(width))
        print("\033[1m" + "="*width + "\033[0m")
        
        # Row 1: Hardware vs Topology
        sys_info = f"[SYSTEM] Hardware: {gpu_name[:25]} | CUDA {cuda_ver}"
        mod_info = f"[MODEL]  L: {getattr(self.model.base, 'n_layers', 0)} | D: {getattr(self.model.base, 'dim', 0)} | H: {getattr(self.model.base, 'n_heads', 0)}"
        print(f" {sys_info:<54} | {mod_info}")
        
        # Row 2: Precision vs Params
        prec_info = f"[SYSTEM] Precision: {self.config.precision} | VRAM Weights: {model_size_mb:.1f}MB"
        param_info = f"[MODEL]  Params: {total_params/1e6:.1f}M ({trainable_params/1e6:.1f}M train) | Vocab: {getattr(self.model, 'vocab_size', 0)}"
        print(f" {prec_info:<54} | {param_info}")
        
        print("-" * width)
        
        # Row 3: Training Config vs Research Phase
        total_tokens = 0
        if hasattr(self.train_loader, 'loader'):
            total_tokens = len(self.train_loader.loader.data_tensor)
        elif hasattr(self.train_loader, 'data_tensor'):
            total_tokens = len(self.train_loader.data_tensor)
        token_str = f"{total_tokens/1e6:.2f}M" if total_tokens >= 1e6 else f"{total_tokens/1e3:.1f}K"
        
        train_info = f"[TRAIN]  Steps: {self.config.max_steps} | Batch: {self.config.batch_size} (x{self.config.grad_accum_steps})"
        
        loader = self.train_loader.loader if hasattr(self.train_loader, 'loader') else self.train_loader
        m_prob = getattr(loader, 'mask_prob', 0.0)
        
        if self.span_active:
            r_phase = f"PH3 (SPAN) @ {m_prob*100:.1f}%"
            r_imp = f"Improv: {self.span_improvement:.2f}%"
        elif self.wwm_active:
            r_phase = f"PH2 (WWM) @ {m_prob*100:.1f}%"
            wwm_imp = (self.ema_at_wwm - self.loss_ema) / self.ema_at_wwm * 100 if self.ema_at_wwm and self.loss_ema else 0.0
            r_imp = f"Improv: {wwm_imp:.2f}%"
        else:
            r_phase = "PH1 (BYTE) @ 0.0%"
            r_imp = "Improv: 0.00%"
            
        res_info = f"[RESEARCH] Phase: {r_phase:<20} | {r_imp}"
        print(f" {train_info:<54} | {res_info}")
        
        # Row 4: Sequence vs Dynamics
        seq_info = f"[TRAIN]  SeqLen: {self.config.max_seq_len} | Corpus: {token_str} tokens"
        lr_status = "STUNNED" if self.lr_stun_executed else "COSINE"
        stag_count = f"{self.analytics.stagnation_steps}/{self.config.plateau_lr_patience}"
        res_dyn = f"[RESEARCH] LR: {lr_status:<10} | Stagnation: {stag_count}"
        print(f" {seq_info:<54} | {res_dyn}")
        
        # Language Stats
        if self.lang_stats:
            print("-" * width)
            lang_lines = []
            for lang, stats in self.lang_stats.items():
                if stats['ema'] is not None:
                    lang_lines.append(f"{lang}: {stats['ema']:.4f}")
            
            for i in range(0, len(lang_lines), 8):
                print(" [LANGS] " + " | ".join(lang_lines[i:i+8]))
        
        print("\033[1m" + "="*width + "\033[0m" + "\n")

    def _refresh_console(self, current_info=None):
        """Clear console and reprint banner + current info (Wide Layout)."""
        os.system('cls' if os.name == 'nt' else 'clear')
        self._print_startup_banner()
        
        width = 110
        if current_info:
            print(f" \033[1m{current_info}\033[0m")
        if self.last_stats_display:
            print(self.last_stats_display)
            
        print("\n" + "-"*width)
        print(" [CTRL+C] to safely abort and save session log.".center(width))
        print("-"*width + "\n")

    def _stats_worker_loop(self):
        """Background thread for heavy NumPy statistics calculation."""
        while not self.stats_stop_event.is_set():
            try:
                data = self.stats_queue.get(timeout=1.0)
                if data is None: break
                
                step = data['step']
                progress = data['progress']
                
                output_sections = []
                
                # 1. Lifetime Stats
                stats_life = LossAnalytics.calculate_from_data(data['lifetime'], progress)
                if stats_life:
                    sec = f"\n--- LOSS DYNAMICS: LIFETIME (Step {step}) ---\n"
                    sec += json.dumps(stats_life, indent=2)
                    output_sections.append(sec)
                
                # 2. Phase 2 Stats (only if active)
                if data['wwm'] and data['wwm']['total_steps'] > 5:
                    stats_wwm = LossAnalytics.calculate_from_data(data['wwm'], progress)
                    sec = f"\n--- LOSS DYNAMICS: PHASE 2 (WWM) ---\n"
                    sec += json.dumps(stats_wwm, indent=2)
                    output_sections.append(sec)
                    
                # 3. Phase 3 Stats (only if active)
                if data['span'] and data['span']['total_steps'] > 5:
                    stats_span = LossAnalytics.calculate_from_data(data['span'], progress)
                    sec = f"\n--- LOSS DYNAMICS: PHASE 3 (SPAN) ---\n"
                    sec += json.dumps(stats_span, indent=2)
                    output_sections.append(sec)

                if output_sections:
                    self.last_stats_display = "\n".join(output_sections) + "\n" + "-" * 40
                
                self.stats_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                pass

    def _train_step(self, tokens, labels, is_capture=False):
        """Single training step (Forward + Backward + Optimizer)"""
        self.optimizer.zero_grad(set_to_none=True)
        
        # Note: grad_accum_steps is handled outside for simplicity in CUDA Graphs
        with autocast(device_type=self.device.type, dtype=self._get_dtype()):
            outputs, loss, task_name = self.model({"tokens": tokens, "labels": labels})
            scaled_loss = loss / self.config.grad_accum_steps
            self.scaler.scale(scaled_loss).backward()
        
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Write to static buffer for analytics/logging (recorded in graph if capturing)
        self.static_loss.copy_(scaled_loss)
        
        return scaled_loss.detach()

    def train(self):
        """Execute the training loop."""
        self.model.train()
        t0 = time.time()
        
        # Infinite iterator
        data_iter = iter(self.train_loader)
        
        print(f"  [RUN] Training loop engaged. Monitoring diagnostics...")
        
        try:
            while self.global_step < self.config.max_steps:
                step_start = time.time()
                # --- Suggestion 3: Automated LR Stun (Stagnation Break) ---
                if not self.lr_stun_executed and self.analytics.stagnation_steps >= self.config.plateau_lr_patience:
                    print(f"\n  [DYNAMIC] Automated LR Stun: Stagnation reached {self.config.plateau_lr_patience} steps.")
                    
                    old_multiplier = self.scheduler.multiplier
                    new_multiplier = old_multiplier * self.config.plateau_lr_drop
                    
                    # Apply stun through scheduler multiplier (fixes persistence bug)
                    self.scheduler.multiplier = new_multiplier
                    self.lr_stun_executed = True
                    
                    # Momentum Mitigation: Scale down AdamW momentum (exp_avg) to prevent "gradient drag"
                    # from the high-LR regime pushing the model too hard in the new lower-LR regime.
                    self._scale_optimizer_momentum(self.config.plateau_lr_drop)
                    
                    print(f"  [DYNAMIC] Scaling LR Multiplier: {old_multiplier:.2f} -> {new_multiplier:.2f}")
                    print(f"  [DYNAMIC] AdamW Momentum Scaled by {self.config.plateau_lr_drop:.2f} to mitigate velocity drag.")
                    
                    self.logger.log_grokking_signal(
                        self.global_step, 
                        "lr_stun", 
                        self.config.plateau_lr_drop, 
                        f"Plateau recovery triggered at step {self.global_step}"
                    )

                # --- Suggestion 4: Ramped Masking Progression ---
                if self.wwm_active or self.span_active:
                    loader = self.train_loader.loader if hasattr(self.train_loader, 'loader') else self.train_loader
                    if self.span_active:
                        steps_into_phase = self.global_step - self.span_activated_step
                        ramp = min(steps_into_phase / max(self.config.span_ramp_steps, 1), 1.0)
                        loader.mask_prob = 0.05 + (self.config.span_mask_prob - 0.05) * ramp
                        self.mask_prob_active = loader.mask_prob
                        
                        # Peak Anchoring: Capture baseline once ramp is finished
                        if steps_into_phase == self.config.span_ramp_steps:
                            self.ema_at_span = self.loss_ema
                            print(f"\n  [CHECKPOINT] Phase 3 Calibration Complete. Baseline anchored at {self.loss_ema:.4f}")
                            
                    elif self.wwm_active:
                        steps_into_phase = self.global_step - self.wwm_activated_step
                        ramp = min(steps_into_phase / max(self.config.wwm_ramp_steps, 1), 1.0)
                        loader.mask_prob = 0.05 + (self.config.wwm_mask_prob - 0.05) * ramp
                        self.mask_prob_active = loader.mask_prob

                        # Peak Anchoring: Capture baseline once ramp is finished
                        if steps_into_phase == self.config.wwm_ramp_steps:
                            self.ema_at_wwm = self.loss_ema
                            print(f"\n  [CHECKPOINT] Phase 2 Calibration Complete. Baseline anchored at {self.loss_ema:.4f}")
                else:
                    self.mask_prob_active = 0.0

                # 2. Forward and Backward
                if self.config.use_cuda_graph and self.cuda_graph is not None:
                    # REPLAY MODE
                    try: batch = next(data_iter)
                    except StopIteration: batch = next(iter(self.train_loader))
                    
                    self.static_tokens.copy_(batch['tokens'], non_blocking=True)
                    self.static_labels.copy_(batch['labels'], non_blocking=True)
                    self.cuda_graph.replay()
                    curr_loss = self.static_loss.item()
                    lr = self.scheduler.step(self.global_step)
                else:
                    # EAGER/CAPTURE MODE
                    if self.config.use_cuda_graph and self.global_step == self.graph_warmup_steps:
                        try: batch = next(data_iter)
                        except StopIteration: batch = next(iter(self.train_loader))
                        
                        if self.static_tokens is None:
                            self.static_tokens = batch['tokens'].clone().detach()
                            self.static_labels = batch['labels'].clone().detach()
                        
                        print(f"  [SYSTEM] Capturing CUDA Graph at step {self.global_step}...")
                        
                        s = torch.cuda.Stream()
                        s.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(s):
                            for _ in range(3):
                                self._train_step(self.static_tokens, self.static_labels)
                        torch.cuda.current_stream().wait_stream(s)
                        
                        self.cuda_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(self.cuda_graph):
                            self._train_step(self.static_tokens, self.static_labels, is_capture=True)
                        
                        print("  [SYSTEM] CUDA Graph captured successfully.")
                        curr_loss = self.static_loss.item()
                    else:
                        # Standard Eager Step - ACCUMULATE ON GPU
                        accum_loss = torch.tensor(0.0, device=self.device)
                        for _ in range(self.config.grad_accum_steps):
                            try: batch = next(data_iter)
                            except StopIteration: batch = next(iter(self.train_loader))
                            # We pass the sum to keep track without syncing yet
                            accum_loss += self._train_step(batch['tokens'], batch['labels'])
                        
                        # SINGLE SYNC POINT PER FULL STEP
                        curr_loss = accum_loss.item()
                    
                    lr = self.scheduler.step(self.global_step)
                
                # Update Analytics (CPU)
                # Redundant check removed.
                
                # --- Performance Precision ---
                self.interval_time += time.time() - step_start
                tokens_this_step = self.config.effective_batch_size * self.config.max_seq_len
                self.interval_tokens += tokens_this_step
                # ------------------------------

                # 3. Step Administrative
                self.global_step += 1
                
                # High-Resolution EMA (Per-step update for smooth triggering)
                if self.loss_ema is None:
                    self.loss_ema = curr_loss
                else:
                    # Use a very slow alpha (0.99) for per-step EMA to act as a noise filter
                    # This is roughly equivalent to a 100-step average but updated live
                    self.loss_ema = 0.99 * self.loss_ema + 0.01 * curr_loss
                
                # Analytics Updates
                self.analytics.update(curr_loss)
                if self.wwm_active and not self.span_active:
                    self.wwm_analytics.update(curr_loss)
                if self.span_active:
                    self.span_analytics.update(curr_loss)

                # Buffer log for session file
                self.session_log.append({
                    "step": self.global_step,
                    "loss": curr_loss,
                    "lr": lr,
                    "wwm_imp": self.wwm_final_improvement if self.span_active else (
                    0.0 if self.ema_at_wwm is None else (self.ema_at_wwm - self.loss_ema) / self.ema_at_wwm * 100
                ),
                "span_imp": self.span_improvement if self.span_active else (
                    0.0 if self.ema_at_span is None else (self.ema_at_span - self.loss_ema) / self.ema_at_span * 100
                ),
                    "time": time.time()
                })

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    tokens_per_sec = self.interval_tokens / max(self.interval_time, 1e-6)
                    
                    s_per_step = self.interval_time / self.config.log_interval
                    if self.smoothed_s_per_step is None:
                        self.smoothed_s_per_step = s_per_step
                    else:
                        self.smoothed_s_per_step = 0.2 * s_per_step + 0.8 * self.smoothed_s_per_step
                    
                    steps_remaining = self.config.max_steps - self.global_step
                    eta_s = steps_remaining * self.smoothed_s_per_step
                    
                    from datetime import timedelta
                    eta_str = str(timedelta(seconds=int(eta_s)))
                    
                    log_line = f"Step {self.global_step}/{self.config.max_steps} | Loss: {curr_loss:.4f} | EMA: {self.loss_ema:.4f} | LR: {lr:.2e} | {tokens_per_sec:.0f} tok/s | ETA: {eta_str}"
                    self._refresh_console(log_line)
                    
                    # --- Plateau Detection (Dynamic WWM Trigger) ---
                    if not self.wwm_active and self.global_step > self.config.wwm_trigger_steps and self.boundary_tensor is not None:
                        if self.last_ema_check is not None:
                            steps_since_check = self.global_step - self.last_ema_check
                            if steps_since_check >= self.config.wwm_window:
                                improvement = (self.ema_at_last_check - self.loss_ema) / max(self.ema_at_last_check, 1e-6)
                                if improvement < self.config.wwm_threshold: 
                                    print(f"\n  [DYNAMIC] Plateau detected (Improvement: {improvement*100:.3f}%).")
                                    print("  [DYNAMIC] Activating Whole-Word Masking (WWM) to stimulate Grokking...")
                                    self.wwm_active = True
                                    self.wwm_activated_step = self.global_step
                                    # self.ema_at_wwm set later via Peak Anchoring
                                    
                                    # Permanent Logging
                                    self.logger.log_grokking_signal(
                                        step=self.global_step,
                                        signal_type="wwm_activation",
                                        value=1.0,
                                        description=f"Dynamic Whole-Word Masking activated (Prob: {self.config.wwm_mask_prob})."
                                    )
                                    self.logger.log_metrics({"Research/WWM_Active": 1.0}, self.global_step)
                                    
                                    if hasattr(self.train_loader, 'loader'):
                                        self.train_loader.loader.mask_prob = self.config.wwm_mask_prob
                                        self.train_loader.loader.use_wwm = True
                                        
                                    # Precision Refinement: Purge queue and reset phase analytics
                                    if hasattr(self.train_loader, 'clear'):
                                        self.train_loader.clear()
                                    self.wwm_analytics = LossAnalytics(window_size=100)
                                
                                self.last_ema_check = self.global_step
                                self.ema_at_last_check = self.loss_ema
                        else:
                            self.last_ema_check = self.global_step
                            self.ema_at_last_check = self.loss_ema
                            
                    # --- Phase 3: Span Masking Trigger (Contiguous Grokking) ---
                    elif not self.span_active and self.wwm_active and (self.global_step - self.wwm_activated_step) > self.config.span_trigger_steps:
                        if self.last_ema_check is not None:
                            steps_since_check = self.global_step - self.last_ema_check
                            if steps_since_check >= self.config.span_window:
                                improvement = (self.ema_at_last_check - self.loss_ema) / max(self.ema_at_last_check, 1e-6)
                                if improvement < self.config.span_threshold:
                                    print(f"\n  [DYNAMIC] Phase 3 Plateau detected (Improvement: {improvement*100:.3f}%).")
                                    print(f"  [DYNAMIC] Activating Word Sequence (Span) Masking (Len: {self.config.span_min_len}-{self.config.span_max_len})...")
                                    self.span_active = True
                                    self.span_activated_step = self.global_step
                                    # self.ema_at_span set later via Peak Anchoring
                                    
                                    # Lock WWM Stats
                                    if self.ema_at_wwm and self.loss_ema:
                                        self.wwm_final_improvement = (self.ema_at_wwm - self.loss_ema) / self.ema_at_wwm * 100
                                    
                                    self.ema_at_span = self.loss_ema # Capture New Baseline
                                    
                                    # Permanent Logging
                                    self.logger.log_grokking_signal(
                                        step=self.global_step,
                                        signal_type="span_activation",
                                        value=1.0,
                                        description=f"Phase 3 Word Sequence (Span) Masking activated (Prob: {self.config.span_mask_prob}, Len: {self.config.span_min_len}-{self.config.span_max_len})."
                                    )
                                    self.logger.log_metrics({"Research/Span_Active": 1.0}, self.global_step)
                                    
                                    if hasattr(self.train_loader, 'loader'):
                                        self.train_loader.loader.mask_prob = self.config.span_mask_prob
                                        self.train_loader.loader.use_span = True
                                        self.train_loader.loader.span_range = (self.config.span_min_len, self.config.span_max_len)
                                        # m_prob set from config
                                        
                                    # Precision Refinement: Purge queue and reset phase analytics
                                    if hasattr(self.train_loader, 'clear'):
                                        self.train_loader.clear()
                                    self.span_analytics = LossAnalytics(window_size=100)
                                
                                self.last_ema_check = self.global_step
                                self.ema_at_last_check = self.loss_ema
                        else:
                            self.last_ema_check = self.global_step
                            self.ema_at_last_check = self.loss_ema
                    # -----------------------------------------------

                    # Monitor VRAM
                    gpu_mem_gb = 0.0
                    if self.device.type == 'cuda':
                        gpu_mem_gb = torch.cuda.max_memory_reserved() / (1024**3)
                        if not self.vram_warned and gpu_mem_gb > 11.4:
                            print(f"\n  [CAUTION] VRAM usage is critical ({gpu_mem_gb:.2f} GB).")
                            self.vram_warned = True

                    current_wwm_imp = self.wwm_final_improvement if self.span_active else (
                        (self.ema_at_wwm - self.loss_ema) / self.ema_at_wwm * 100 if self.ema_at_wwm and self.loss_ema else 0.0
                    )

                    self.logger.log_training_step(
                        step=self.global_step,
                        loss=curr_loss,
                        learning_rate=lr,
                        wwm_improvement=current_wwm_imp,
                        span_improvement=self.span_improvement,
                        tokens_per_sec=tokens_per_sec,
                        gpu_memory_gb=gpu_mem_gb
                    )
                    
                    self.logger.log_language_performance(self.global_step, self.lang_stats)
                    
                    self.interval_tokens = 0
                    self.interval_time = 0.0
                    
                    # Update live research stats for Span
                    if self.span_active and self.ema_at_span:
                        self.span_improvement = (self.ema_at_span - self.loss_ema) / self.ema_at_span * 100

                # --- Asynchronous Statistics Update ---
                if self.stats_queue.empty():
                    try:
                        progress = (self.global_step / self.config.max_steps) * 100
                        
                        def get_snap(an):
                            return {
                                "y": np.array(an.loss_buffer),
                                "ema_100": an.ema_100,
                                "global_min": an.global_min,
                                "stagnation_steps": an.stagnation_steps,
                                "spikes": list(an.spikes),
                                "total_steps": an.total_steps,
                                "window_size": an.window_size
                            }

                        snapshot = {
                            "step": self.global_step,
                            "progress": progress,
                            "lifetime": get_snap(self.analytics),
                            "wwm": get_snap(self.wwm_analytics) if self.wwm_active else None,
                            "span": get_snap(self.span_analytics) if self.span_active else None
                        }
                        self.stats_queue.put_nowait(snapshot)
                    except (queue.Full, RuntimeError):
                        pass
                
                # 5. Validation
                if self.config.enable_validation and self.global_step % self.config.val_interval == 0:
                    self.validate()
                    
                # 4. Extensive Evaluation
                if self.config.enable_extensive_eval and self.global_step % self.config.eval_interval == 0:
                    self.evaluate_extensive()
                    
                # Checkpointing
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint()
                    
                # Callbacks
                self.callbacks.on_batch_end(self.global_step, self.model, curr_loss)
            
            # Successful Completion
            self._save_session_log()
            self.logger.finalize_experiment(status="completed")
            
        except KeyboardInterrupt:
            print(f"\n\n  [SYSTEM] Safe abort triggered by user.")
            self.stats_stop_event.set()
            if self.stats_queue.empty(): self.stats_queue.put(None)
            self._save_session_log()
            self.save_checkpoint(f"checkpoints/abort_step_{self.global_step}.pt")
            self.logger.finalize_experiment(status="aborted")
            print("  [SYSTEM] Cleanup complete. Shutdown successful.")
            sys.exit(0)
            
    def validate(self):
        """Run validation loop with per-language diagnostics."""
        self.model.eval()
        
        # Get locale map from the actual loader (through the prefixer)
        loader = self.train_loader.loader if hasattr(self.train_loader, 'loader') else self.train_loader
        locale_map = getattr(loader, 'locale_map', {})
        
        print("\n" + "-"*30)
        print(f"  V A L I D A T I O N   (Step {self.global_step})")
        print("-"*30)
        
        all_metrics = {}
        total_val_loss = 0
        total_langs = 0
        
        with torch.no_grad():
            # 1. Global Performance (Mixed)
            val_steps = 30 # Reduced for total latency
            for i in range(val_steps):
                batch = next(iter(self.val_loader)) if self.val_loader else None
                if not batch: break
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): batch[k] = v.to(self.device, non_blocking=True)
                with autocast(device_type=self.device.type, dtype=self._get_dtype()):
                    _, loss, _ = self.model(batch)
                total_val_loss += loss.item()
            
            avg_loss = total_val_loss / val_steps
            print(f"  [GLOBAL] Avg Val Loss: {avg_loss:.4f}")
            self.logger.log_evaluation(step=self.global_step, val_perplexity=avg_loss)
            
            # 2. Per-Language targeted evaluation
            if locale_map:
                print("  [LANGS] Individual Perplexity:")
                for lang in locale_map:
                    lang_total = 0
                    l_steps = 10
                    
                    # Direct access to loader since Prefetcher doesn't have sample_locale
                    loader_obj = self.train_loader.loader if hasattr(self.train_loader, 'loader') else self.train_loader
                    if not hasattr(loader_obj, 'sample_locale'): break
                    
                    for _ in range(l_steps):
                        batch = loader_obj.sample_locale(lang)
                        if batch is None: break
                        with autocast(device_type=self.device.type, dtype=self._get_dtype()):
                            _, loss, _ = self.model(batch)
                        lang_total += loss.item()
                    
                    if l_steps > 0:
                        l_avg = lang_total / l_steps
                        print(f"    - {lang:5}: {l_avg:.4f}")
                        self.logger.log_metrics({f"Eval/Perplexity_{lang}": l_avg}, self.global_step)

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
            'step': self.global_step,
            # Research State Persistence
            'research_state': {
                'wwm_active': self.wwm_active,
                'wwm_activated_step': self.wwm_activated_step,
                'span_active': self.span_active,
                'span_activated_step': self.span_activated_step,
                'loss_ema': self.loss_ema,
                'last_ema_check': self.last_ema_check,
                'ema_at_last_check': self.ema_at_last_check,
                'ema_at_wwm': self.ema_at_wwm,
                'ema_at_span': self.ema_at_span,
                'wwm_final_improvement': self.wwm_final_improvement,
                'span_improvement': self.span_improvement,
                'lr_stun_executed': self.lr_stun_executed,
                'lr_multiplier': self.scheduler.multiplier,
                'lang_stats': self.lang_stats,
                # Analytics Persistence
                'analytics_state': self.analytics.to_dict(),
                'wwm_analytics_state': self.wwm_analytics.to_dict(),
                'span_analytics_state': self.span_analytics.to_dict()
            }
        }
        
        # Add model architecture if available
        if self.arch_config:
            checkpoint['config'].update(asdict(self.arch_config))
        elif hasattr(self.model, 'config') and self.model.config:
            checkpoint['config'].update(asdict(self.model.config))
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def _save_session_log(self):
        """Save the in-memory session log to a text file."""
        if not self.session_log:
            return
            
        log_dir = Path("logs/sessions")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = int(time.time())
        log_path = log_dir / f"session_{timestamp}_step_{self.global_step}.txt"
        
        try:
            with open(log_path, "w") as f:
                f.write(f"GENESIS SESSION LOG - Step {self.global_step}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write("-" * 60 + "\n")
                f.write("Step\tLoss\tLR\tWWM_Imp\tSpan_Imp\tTime\n")
                for entry in self.session_log:
                    f.write(f"{entry['step']}\t{entry['loss']:.6f}\t{entry['lr']:.2e}\t{entry.get('wwm_imp',0):.2f}%\t{entry.get('span_imp',0):.2f}%\t{entry['time']}\n")
            print(f"  [DATA] Session log saved to: {log_path}")
        except Exception as e:
            print(f"  [WARN] Failed to save session log: {e}")

    def load_checkpoint(self, path):
        """Load checkpoint."""
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.global_step = checkpoint['step']
        
        # Restore Research State
        if 'research_state' in checkpoint:
            rs = checkpoint['research_state']
            self.wwm_active = rs.get('wwm_active', False)
            self.wwm_activated_step = rs.get('wwm_activated_step')
            self.span_active = rs.get('span_active', False)
            self.span_activated_step = rs.get('span_activated_step')
            self.loss_ema = rs.get('loss_ema')
            self.last_ema_check = rs.get('last_ema_check')
            self.ema_at_last_check = rs.get('ema_at_last_check')
            self.ema_at_wwm = rs.get('ema_at_wwm')
            self.ema_at_span = rs.get('ema_at_span')
            self.wwm_final_improvement = rs.get('wwm_final_improvement', 0.0)
            self.span_improvement = rs.get('span_improvement', 0.0)
            self.lr_stun_executed = rs.get('lr_stun_executed', False)
            self.scheduler.multiplier = rs.get('lr_multiplier', 1.0)
            self.lang_stats = rs.get('lang_stats', self.lang_stats)
            
            # Restore Analytics History
            if 'analytics_state' in rs:
                self.analytics.from_dict(rs['analytics_state'])
            if 'wwm_analytics_state' in rs:
                self.wwm_analytics.from_dict(rs['wwm_analytics_state'])
            if 'span_analytics_state' in rs:
                self.span_analytics.from_dict(rs['span_analytics_state'])
                
            print(f"  [SYTEM] Research state restored (WWM: {'Active' if self.wwm_active else 'OFF'}, Span: {'Active' if self.span_active else 'OFF'})")
            
            # --- Loader Synchronization ---
            # Ensure the restored masking flags are pushed into the current data loader
            if hasattr(self.train_loader, 'loader'):
                loader = self.train_loader.loader
                if self.span_active:
                    loader.use_span = True
                    loader.mask_prob = self.config.span_mask_prob
                    loader.span_range = (self.config.span_min_len, self.config.span_max_len)
                elif self.wwm_active:
                    loader.use_wwm = True
                    loader.mask_prob = self.config.wwm_mask_prob
                
                if self.wwm_active or self.span_active:
                    print(f"  [DATA] Masking configuration re-injected into GPU Loader.")
        
        print(f"Resumed from step {self.global_step}")

    def _get_dtype(self):
        if self.config.precision == "bf16": return torch.bfloat16
        if self.config.precision == "fp16": return torch.float16
        return torch.float32

    def _scale_optimizer_momentum(self, factor: float):
        """Scale down AdamW first moment (exp_avg) to mitigate momentum drag."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                if 'exp_avg' in state:
                    state['exp_avg'].mul_(factor)
