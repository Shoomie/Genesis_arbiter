"""
Arbiter Logger: Unified Logging System for Theos-Small Research Infrastructure

This module provides comprehensive logging capabilities for tracking experiments,
training dynamics, evaluation metrics, and grokking signals. All data is stored
in a SQLite database for queryable persistence and optionally exported to TensorBoard.

Key Features:
- Structured experiment tracking with SQLite backend
- Multi-level logging: System, Training, Evaluation, Grokking signals
- TensorBoard integration for real-time visualization
- Automatic grokking detection via validation loss anomaly analysis
- Thread-safe concurrent writes from distributed training
"""

import sqlite3
import json
import time
import queue
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager
import threading
import hashlib

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[WARNING] TensorBoard not available. Install with: pip install tensorboard")


@dataclass
class ExperimentConfig:
    """Configuration snapshot for an experiment run."""
    run_id: str
    config_hash: str
    n_layers: int
    dim: int
    n_heads: int
    vocab_size: int
    weight_decay: float
    learning_rate: float
    batch_size: int
    seq_len: int
    warmup_steps: int
    max_steps: int
    timestamp: str
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from torchtitan TOML dict."""
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
        
        return cls(
            run_id=f"run_{int(time.time())}_{config_hash}",
            config_hash=config_hash,
            n_layers=config.get('model', {}).get('n_layers', 0),
            dim=config.get('model', {}).get('dim', 0),
            n_heads=config.get('model', {}).get('n_heads', 0),
            vocab_size=config.get('model', {}).get('vocab_size', 0),
            weight_decay=config.get('training', {}).get('weight_decay', 0.0),
            learning_rate=config.get('training', {}).get('learning_rate', 0.0),
            batch_size=config.get('training', {}).get('batch_size', 0),
            seq_len=config.get('training', {}).get('max_seq_len', 0),
            warmup_steps=config.get('training', {}).get('warmup_steps', 0),
            max_steps=config.get('training', {}).get('max_steps', 0),
            timestamp=datetime.now().isoformat()
        )


class ArbiterLogger:
    """
    Unified logging system with SQLite persistence and optional TensorBoard integration.
    
    Usage:
        logger = ArbiterLogger(log_dir="./logs", experiment_name="deep_narrow_80L")
        logger.start_experiment(config_dict)
        logger.log_training_step(step=100, loss=2.34, grad_norm=0.5)
        logger.log_evaluation(step=1000, perplexity=15.2, theological_score=0.78)
        logger.finalize_experiment(status="completed")
    """
    
    def __init__(
        self, 
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        enable_tensorboard: bool = True,
        db_name: str = "experiments.db"
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.log_dir / db_name
        self.experiment_name = experiment_name or f"exp_{int(time.time())}"
        self.current_run_id: Optional[str] = None
        
        # Thread safety
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._stop_event = threading.Event()
        
        # TensorBoard
        self.tensorboard_writer = None
        if enable_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = self.log_dir / "tensorboard" / self.experiment_name
            self.tensorboard_writer = SummaryWriter(log_dir=str(tb_dir))
        
        # Initialize database
        self._init_database()
        
        # Start worker thread
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        
        # Grokking detection state
        self._grokking_detected = False
        self._val_loss_history: List[float] = []
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT PRIMARY KEY,
                    config_hash TEXT,
                    experiment_name TEXT,
                    n_layers INTEGER,
                    dim INTEGER,
                    n_heads INTEGER,
                    vocab_size INTEGER,
                    weight_decay REAL,
                    learning_rate REAL,
                    batch_size INTEGER,
                    seq_len INTEGER,
                    warmup_steps INTEGER,
                    max_steps INTEGER,
                    final_train_loss REAL,
                    final_val_loss REAL,
                    grokking_step INTEGER,
                    theological_score REAL,
                    status TEXT,
                    timestamp TEXT,
                    duration_seconds REAL
                )
            """)
            
            # Training logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    step INTEGER,
                    loss REAL,
                    grad_norm REAL,
                    learning_rate REAL,
                    tokens_per_sec REAL,
                    gpu_memory_gb REAL,
                    timestamp TEXT,
                    FOREIGN KEY (run_id) REFERENCES experiments(run_id)
                )
            """)
            
            # Evaluation logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    step INTEGER,
                    train_perplexity REAL,
                    val_perplexity REAL,
                    theological_score REAL,
                    memorization_score REAL,
                    reasoning_score REAL,
                    attention_entropy REAL,
                    timestamp TEXT,
                    FOREIGN KEY (run_id) REFERENCES experiments(run_id)
                )
            """)
            
            # Grokking signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS grokking_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    step INTEGER,
                    signal_type TEXT,
                    value REAL,
                    description TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (run_id) REFERENCES experiments(run_id)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Thread-safe database connection context manager."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    def _worker(self):
        """Background worker to process logging queue."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        while not (self._stop_event.is_set() and self._queue.empty()):
            try:
                msg = self._queue.get(timeout=1.0)
                if msg is None: break
                
                msg_type = msg.get('type')
                data = msg.get('data')
                
                if msg_type == 'training_step':
                    cursor.execute("""
                        INSERT INTO training_logs (
                            run_id, step, loss, grad_norm, learning_rate,
                            tokens_per_sec, gpu_memory_gb, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data['run_id'], data['step'], data['loss'], 
                        data['grad_norm'], data['learning_rate'],
                        data['tokens_per_sec'], data['gpu_memory_gb'], 
                        data['timestamp']
                    ))
                    conn.commit()
                
                elif msg_type == 'evaluation':
                    cursor.execute("""
                        INSERT INTO evaluation_logs (
                            run_id, step, train_perplexity, val_perplexity,
                            theological_score, memorization_score, reasoning_score,
                            attention_entropy, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data['run_id'], data['step'], 
                        data['train_perplexity'], data['val_perplexity'],
                        data['theological_score'], data['memorization_score'], 
                        data['reasoning_score'], data['attention_entropy'], 
                        data['timestamp']
                    ))
                    conn.commit()
                
                elif msg_type == 'grokking_signal':
                    cursor.execute("""
                        INSERT INTO grokking_signals (
                            run_id, step, signal_type, value, description, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        data['run_id'], data['step'], data['signal_type'],
                        data['value'], data['description'], data['timestamp']
                    ))
                    conn.commit()
                
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ArbiterLogger] Worker error: {e}")
        
        conn.close()
    
    def start_experiment(self, config: Dict[str, Any]) -> str:
        """
        Initialize a new experiment run.
        
        Args:
            config: Dictionary containing full training configuration
            
        Returns:
            run_id: Unique identifier for this experiment
        """
        exp_config = ExperimentConfig.from_dict(config)
        self.current_run_id = exp_config.run_id
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO experiments (
                    run_id, config_hash, experiment_name, n_layers, dim, n_heads,
                    vocab_size, weight_decay, learning_rate, batch_size, seq_len,
                    warmup_steps, max_steps, timestamp, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_config.run_id,
                exp_config.config_hash,
                self.experiment_name,
                exp_config.n_layers,
                exp_config.dim,
                exp_config.n_heads,
                exp_config.vocab_size,
                exp_config.weight_decay,
                exp_config.learning_rate,
                exp_config.batch_size,
                exp_config.seq_len,
                exp_config.warmup_steps,
                exp_config.max_steps,
                exp_config.timestamp,
                "running"
            ))
            conn.commit()
        
        print(f"[ArbiterLogger] Started experiment: {self.current_run_id}")
        print(f"[ArbiterLogger] Config hash: {exp_config.config_hash}")
        print(f"[ArbiterLogger] Topology: L={exp_config.n_layers}, D={exp_config.dim}, H={exp_config.n_heads}")
        
        return self.current_run_id
    
    def log_training_step(
        self,
        step: int,
        loss: float,
        grad_norm: Optional[float] = None,
        learning_rate: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        gpu_memory_gb: Optional[float] = None
    ):
        """Log training metrics for a single step (Now Async)."""
        if not self.current_run_id:
            raise RuntimeError("Must call start_experiment() before logging")
        
        # Offload to queue
        self._queue.put({
            'type': 'training_step',
            'data': {
                'run_id': self.current_run_id,
                'step': step,
                'loss': loss,
                'grad_norm': grad_norm,
                'learning_rate': learning_rate,
                'tokens_per_sec': tokens_per_sec,
                'gpu_memory_gb': gpu_memory_gb,
                'timestamp': datetime.now().isoformat()
            }
        })
        
        # TensorBoard logging
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('Train/Loss', loss, step)
            if grad_norm is not None:
                self.tensorboard_writer.add_scalar('Train/GradNorm', grad_norm, step)
            if learning_rate is not None:
                self.tensorboard_writer.add_scalar('Train/LearningRate', learning_rate, step)
            if tokens_per_sec is not None:
                self.tensorboard_writer.add_scalar('System/TokensPerSec', tokens_per_sec, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log a dictionary of arbitrary metrics to TensorBoard.
        This is a flexible alternative to the structured logging methods.
        """
        if self.tensorboard_writer:
            for k, v in metrics.items():
                if v is not None:
                    # Clean the key for TB if needed
                    self.tensorboard_writer.add_scalar(k, v, step)
    
    def log_evaluation(
        self,
        step: int,
        train_perplexity: Optional[float] = None,
        val_perplexity: Optional[float] = None,
        theological_score: Optional[float] = None,
        memorization_score: Optional[float] = None,
        reasoning_score: Optional[float] = None,
        attention_entropy: Optional[float] = None
    ):
        """Log evaluation metrics (Now Async)."""
        if not self.current_run_id:
            raise RuntimeError("Must call start_experiment() before logging")
        
        # Offload to queue
        self._queue.put({
            'type': 'evaluation',
            'data': {
                'run_id': self.current_run_id,
                'step': step,
                'train_perplexity': train_perplexity,
                'val_perplexity': val_perplexity,
                'theological_score': theological_score,
                'memorization_score': memorization_score,
                'reasoning_score': reasoning_score,
                'attention_entropy': attention_entropy,
                'timestamp': datetime.now().isoformat()
            }
        })
        
        # TensorBoard logging (synchronous on main thread is fine for eval which is rare)
        if self.tensorboard_writer:
            if val_perplexity is not None:
                self.tensorboard_writer.add_scalar('Eval/Perplexity', val_perplexity, step)
                # Grokking detection
                self._detect_grokking(step, val_perplexity)
            if theological_score is not None:
                self.tensorboard_writer.add_scalar('Eval/TheologicalScore', theological_score, step)
    
    def _detect_grokking(self, step: int, val_loss: float):
        """
        Detect grokking via sudden validation loss drop.
        
        Criterion: If val_loss drops by >20% in a single evaluation while
        training loss has been near-zero for >1000 steps, signal grokking.
        """
        self._val_loss_history.append(val_loss)
        
        if len(self._val_loss_history) < 2 or self._grokking_detected:
            return
        
        # Check for sudden drop
        prev_loss = self._val_loss_history[-2]
        if prev_loss > 0:
            drop_ratio = (prev_loss - val_loss) / prev_loss
            
            if drop_ratio > 0.2:  # 20% drop
                self._grokking_detected = True
                self.log_grokking_signal(
                    step=step,
                    signal_type="validation_drop",
                    value=drop_ratio,
                    description=f"Validation loss dropped {drop_ratio*100:.1f}% at step {step}"
                )
                print(f"\n{'='*60}")
                print(f"[🎯 GROKKING DETECTED] Step {step}: Val loss drop = {drop_ratio*100:.1f}%")
                print(f"{'='*60}\n")
    
    def log_grokking_signal(
        self,
        step: int,
        signal_type: str,
        value: float,
        description: str = ""
    ):
        """Log a grokking-related signal (Now Async)."""
        if not self.current_run_id:
            return
        
        # Offload to queue
        self._queue.put({
            'type': 'grokking_signal',
            'data': {
                'run_id': self.current_run_id,
                'step': step,
                'signal_type': signal_type,
                'value': value,
                'description': description,
                'timestamp': datetime.now().isoformat()
            }
        })
        
        # Update experiment table with grokking step
        if signal_type == "validation_drop":
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE experiments 
                    SET grokking_step = ?
                    WHERE run_id = ?
                """, (step, self.current_run_id))
                conn.commit()
    
    def finalize_experiment(
        self,
        final_train_loss: Optional[float] = None,
        final_val_loss: Optional[float] = None,
        theological_score: Optional[float] = None,
        status: str = "completed",
        duration_seconds: Optional[float] = None
    ):
        """Mark experiment as complete and record final metrics."""
        if not self.current_run_id:
            return
        
        # Signal worker to finish
        self._stop_event.set()
        if hasattr(self, '_worker_thread'):
            self._queue.put(None) # Sentinel to break loop
            self._worker_thread.join(timeout=5.0)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experiments
                SET final_train_loss = ?,
                    final_val_loss = ?,
                    theological_score = ?,
                    status = ?,
                    duration_seconds = ?
                WHERE run_id = ?
            """, (
                final_train_loss,
                final_val_loss,
                theological_score,
                status,
                duration_seconds,
                self.current_run_id
            ))
            conn.commit()
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        print(f"[ArbiterLogger] Finalized experiment: {self.current_run_id} (Status: {status})")
    
    def query_experiments(
        self,
        min_theological_score: Optional[float] = None,
        max_layers: Optional[int] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query experiments with optional filters."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM experiments WHERE 1=1"
            params = []
            
            if min_theological_score is not None:
                query += " AND theological_score >= ?"
                params.append(min_theological_score)
            
            if max_layers is not None:
                query += " AND n_layers <= ?"
                params.append(max_layers)
            
            if status is not None:
                query += " AND status = ?"
                params.append(status)
            
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def export_to_json(self, output_path: str):
        """Export entire experiment database to JSON."""
        experiments = self.query_experiments()
        
        with open(output_path, 'w') as f:
            json.dump(experiments, f, indent=2)
        
        print(f"[ArbiterLogger] Exported {len(experiments)} experiments to {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize logger
    logger = ArbiterLogger(
        log_dir="./logs",
        experiment_name="test_deep_narrow",
        enable_tensorboard=True
    )
    
    # Mock configuration
    config = {
        'model': {
            'n_layers': 80,
            'dim': 1024,
            'n_heads': 16,
            'vocab_size': 8192
        },
        'training': {
            'weight_decay': 0.1,
            'learning_rate': 3e-4,
            'batch_size': 16,
            'seq_len': 4096,
            'warmup_steps': 2000,
            'steps': 200000
        }
    }
    
    # Start experiment
    run_id = logger.start_experiment(config)
    
    # Simulate training
    for step in range(1, 10):
        logger.log_training_step(
            step=step,
            loss=2.5 - (step * 0.1),
            grad_norm=0.5,
            learning_rate=1e-4
        )
    
    # Simulate evaluation
    logger.log_evaluation(
        step=10,
        val_perplexity=15.2,
        theological_score=0.78
    )
    
    # Finalize
    logger.finalize_experiment(
        final_train_loss=1.2,
        final_val_loss=2.1,
        theological_score=0.78,
        status="completed"
    )
    
    # Query results
    results = logger.query_experiments(status="completed")
    print(f"\nFound {len(results)} completed experiments")
