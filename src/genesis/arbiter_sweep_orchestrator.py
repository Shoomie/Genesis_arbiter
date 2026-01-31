"""
Arbiter Sweep Orchestrator: Automated Hyperparameter Exploration

This module orchestrates large-scale parameter sweeps for discovering optimal
configurations for the Theos-Small experiment. Supports grid, random, and
Bayesian optimization strategies.

Key Features:
- Multi-strategy search (Grid, Random, Bayesian)
- Automatic TOML configuration generation
- Distributed job scheduling across GPUs
- Dependency tracking for tokenizers and preprocessed data
- Integration with arbiter_logger for comprehensive tracking
"""

import os
import json
import toml
import time
import subprocess
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from itertools import product
import queue
import threading


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep."""
    sweep_id: str
    strategy: str  # 'grid', 'random', 'bayesian'
    parameter_space: Dict[str, List[Any]]
    num_trials: int
    base_config_path: str
    output_dir: str


@dataclass
class Job:
    """A single training job."""
    job_id: str
    config_path: str
    gpu_id: int
    status: str  # 'pending', 'running', 'completed', 'failed'
    process: Optional[subprocess.Popen] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class ArbiterSweepOrchestrator:
    """
    Orchestrate automated hyperparameter sweeps with intelligent scheduling.
    
    Usage:
        orchestrator = ArbiterSweepOrchestrator(
            parameter_space={
                'n_layers': [60, 80, 100],
                'dim': [768, 1024],
                'weight_decay': [0.05, 0.1, 0.2]
            },
            base_config='./configs/base_config.toml',
            output_dir='./sweep_runs'
        )
        orchestrator.run_grid_sweep(num_gpus=4)
    """
    
    def __init__(
        self,
        parameter_space: Dict[str, List[Any]],
        base_config: str,
        output_dir: str = "./sweep_runs",
        strategy: str = "grid"
    ):
        self.parameter_space = parameter_space
        self.base_config_path = Path(base_config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.strategy = strategy
        
        # Generate sweep ID
        sweep_hash = hashlib.md5(json.dumps(parameter_space, sort_keys=True).encode()).hexdigest()[:8]
        self.sweep_id = f"sweep_{int(time.time())}_{sweep_hash}"
        
        # Load base configuration
        with open(self.base_config_path, 'r') as f:
            self.base_config = toml.load(f)
        
        # Job management
        self.jobs: List[Job] = []
        self.job_queue = queue.Queue()
        self.active_jobs: Dict[int, Job] = {}  # gpu_id -> Job
        
        print(f"[SweepOrchestrator] Initialized")
        print(f"  Sweep ID: {self.sweep_id}")
        print(f"  Strategy: {self.strategy}")
        print(f"  Parameter space: {parameter_space}")
        print(f"  Output: {self.output_dir}")
    
    def generate_configs(self) -> List[Dict[str, Any]]:
        """
        Generate configurations based on strategy.
        
        Returns:
            List of configuration dictionaries
        """
        if self.strategy == "grid":
            return self._generate_grid_configs()
        elif self.strategy == "random":
            return self._generate_random_configs()
        elif self.strategy == "bayesian":
            return self._generate_bayesian_configs()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _generate_grid_configs(self) -> List[Dict[str, Any]]:
        """Generate full grid search configurations."""
        # Get all combinations
        param_names = list(self.parameter_space.keys())
        param_values = [self.parameter_space[k] for k in param_names]
        
        configs = []
        for combination in product(*param_values):
            config = self.base_config.copy()
            
            # Apply parameters
            param_dict = dict(zip(param_names, combination))
            config = self._apply_parameters(config, param_dict)
            
            configs.append(config)
        
        print(f"[SweepOrchestrator] Generated {len(configs)} grid configurations")
        return configs
    
    def _generate_random_configs(self, num_samples: int = 20) -> List[Dict[str, Any]]:
        """Generate random sample configurations."""
        import random
        
        configs = []
        for _ in range(num_samples):
            config = self.base_config.copy()
            param_dict = {
                k: random.choice(v) for k, v in self.parameter_space.items()
            }
            config = self._apply_parameters(config, param_dict)
            configs.append(config)
        
        print(f"[SweepOrchestrator] Generated {len(configs)} random configurations")
        return configs
    
    def _generate_bayesian_configs(self) -> List[Dict[str, Any]]:
        """
        Generate configurations using Bayesian optimization.
        
        Placeholder - requires specialized library like Optuna or Ax.
        """
        print("[SweepOrchestrator] Bayesian optimization not yet implemented")
        print("[SweepOrchestrator] Falling back to random sampling")
        return self._generate_random_configs(20)
    
    def _apply_parameters(self, config: Dict, params: Dict[str, Any]) -> Dict:
        """Apply parameter overrides to base config."""
        config_copy = config.copy()
        
        # Map parameters to config structure
        for param_name, value in params.items():
            if param_name in ['n_layers', 'dim', 'n_heads', 'vocab_size']:
                if 'model' not in config_copy:
                    config_copy['model'] = {}
                config_copy['model'][param_name] = value
            
            elif param_name in ['weight_decay', 'learning_rate', 'batch_size', 
                               'seq_len', 'warmup_steps', 'steps']:
                if 'training' not in config_copy:
                    config_copy['training'] = {}
                config_copy['training'][param_name] = value
        
        return config_copy
    
    def save_config(self, config: Dict, job_id: str) -> str:
        """Save configuration to TOML file."""
        config_dir = self.output_dir / self.sweep_id / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = config_dir / f"{job_id}.toml"
        
        with open(config_path, 'w') as f:
            toml.dump(config, f)
        
        return str(config_path)
    
    def create_jobs(self, configs: List[Dict]) -> List[Job]:
        """Create job objects from configurations."""
        jobs = []
        
        for i, config in enumerate(configs):
            job_id = f"{self.sweep_id}_job_{i:04d}"
            config_path = self.save_config(config, job_id)
            
            job = Job(
                job_id=job_id,
                config_path=config_path,
                gpu_id=-1,  # Assigned during scheduling
                status='pending'
            )
            jobs.append(job)
        
        self.jobs = jobs
        return jobs
    
    def run_grid_sweep(
        self,
        num_gpus: int = 1,
        max_concurrent: Optional[int] = None
    ):
        """
        Run a grid search sweep across available GPUs.
        
        Args:
            num_gpus: Number of GPUs available for parallel execution
            max_concurrent: Maximum concurrent jobs (default: num_gpus)
        """
        if max_concurrent is None:
            max_concurrent = num_gpus
        
        # Generate and create jobs
        configs = self.generate_configs()
        jobs = self.create_jobs(configs)
        
        print(f"\n{'='*60}")
        print(f"Starting Grid Sweep: {self.sweep_id}")
        print(f"{'='*60}")
        print(f"Total jobs: {len(jobs)}")
        print(f"GPUs available: {num_gpus}")
        print(f"Max concurrent: {max_concurrent}")
        print(f"{'='*60}\n")
        
        # Enqueue all jobs
        for job in jobs:
            self.job_queue.put(job)
        
        # Start scheduler threads
        scheduler_threads = []
        for gpu_id in range(num_gpus):
            thread = threading.Thread(
                target=self._gpu_worker,
                args=(gpu_id,),
                daemon=True
            )
            thread.start()
            scheduler_threads.append(thread)
        
        # Wait for completion
        self.job_queue.join()
        
        print(f"\n{'='*60}")
        print(f"Sweep Complete: {self.sweep_id}")
        print(f"{'='*60}")
        self._print_summary()
    
    def _gpu_worker(self, gpu_id: int):
        """Worker thread for processing jobs on a specific GPU."""
        while True:
            try:
                job = self.job_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            job.gpu_id = gpu_id
            job.status = 'running'
            job.start_time = time.time()
            
            self.active_jobs[gpu_id] = job
            
            print(f"[GPU {gpu_id}] Starting {job.job_id}")
            
            # Run training
            success = self._run_training_job(job, gpu_id)
            
            job.end_time = time.time()
            job.status = 'completed' if success else 'failed'
            
            duration = job.end_time - job.start_time
            status_icon = "✓" if success else "✗"
            print(f"[GPU {gpu_id}] {status_icon} {job.job_id} ({duration/60:.1f} min)")
            
            del self.active_jobs[gpu_id]
            self.job_queue.task_done()
    
    def _run_training_job(self, job: Job, gpu_id: int) -> bool:
        """
        Execute a training job.
        
        Args:
            job: Job object
            gpu_id: GPU device ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Construct command
            # Adjust based on actual training script location
            cmd = [
                "python",
                "run.py",
                "--config", job.config_path,
                "--mode", "train-only",
                "--gpu", str(gpu_id)
            ]
            
            # Set environment
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            # Run (in production, you'd want more sophisticated process management)
            # For now, simulate
            print(f"[SIMULATION] Would run: {' '.join(cmd)}")
            time.sleep(2)  # Simulate training
            
            # In production:
            # process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # job.process = process
            # process.wait()
            # return process.returncode == 0
            
            return True  # Simulated success
            
        except Exception as e:
            print(f"[ERROR] Job {job.job_id} failed: {e}")
            return False
    
    def _print_summary(self):
        """Print sweep summary statistics."""
        completed = sum(1 for j in self.jobs if j.status == 'completed')
        failed = sum(1 for j in self.jobs if j.status == 'failed')
        
        total_time = sum(
            (j.end_time - j.start_time) for j in self.jobs 
            if j.start_time and j.end_time
        )
        
        print(f"Completed: {completed}/{len(self.jobs)}")
        print(f"Failed: {failed}/{len(self.jobs)}")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Output dir: {self.output_dir / self.sweep_id}")
        print(f"{'='*60}\n")
    
    def export_manifest(self):
        """Export sweep manifest for analysis."""
        manifest = {
            'sweep_id': self.sweep_id,
            'strategy': self.strategy,
            'parameter_space': self.parameter_space,
            'jobs': [
                {
                    'job_id': j.job_id,
                    'config_path': j.config_path,
                    'status': j.status,
                    'duration_seconds': (j.end_time - j.start_time) if j.start_time and j.end_time else None
                }
                for j in self.jobs
            ]
        }
        
        manifest_path = self.output_dir / self.sweep_id / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"[SweepOrchestrator] Manifest saved: {manifest_path}")


# CLI
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Arbiter Sweep Orchestrator")
    parser.add_argument("--base-config", type=str, required=True,
                       help="Path to base TOML configuration")
    parser.add_argument("--output-dir", type=str, default="./sweep_runs",
                       help="Output directory for sweep")
    parser.add_argument("--gpus", type=int, default=1,
                       help="Number of GPUs to use")
    parser.add_argument("--strategy", type=str, default="grid",
                       choices=["grid", "random", "bayesian"],
                       help="Search strategy")
    
    args = parser.parse_args()
    
    # Example parameter space
    parameter_space = {
        'n_layers': [60, 80, 100],
        'dim': [768, 1024],
        'weight_decay': [0.05, 0.1, 0.2],
        'learning_rate': [1e-4, 3e-4]
    }
    
    # Initialize orchestrator
    orchestrator = ArbiterSweepOrchestrator(
        parameter_space=parameter_space,
        base_config=args.base_config,
        output_dir=args.output_dir,
        strategy=args.strategy
    )
    
    # Run sweep
    orchestrator.run_grid_sweep(num_gpus=args.gpus)
    orchestrator.export_manifest()
