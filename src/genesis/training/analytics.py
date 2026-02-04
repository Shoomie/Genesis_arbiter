import numpy as np
import collections
from typing import Dict, List, Optional

class LossAnalytics:
    """
    Research-grade loss analytics engine.
    Calculates landscape, physics, and anomalies on the CPU.
    """
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loss_buffer = collections.deque(maxlen=window_size)
        self.ema_100 = None
        self.global_min = float('inf')
        self.stagnation_steps = 0
        self.total_steps = 0
        self.spikes = []
        self.spike_threshold = 2.0 # 2x the EMA
        
    def update(self, raw_loss: float):
        self.total_steps += 1
        self.loss_buffer.append(raw_loss)
        
        # 1. Landscape Metrics
        if self.ema_100 is None:
            self.ema_100 = raw_loss
        else:
            alpha = 2 / (self.window_size + 1)
            self.ema_100 = (raw_loss * alpha) + (self.ema_100 * (1 - alpha))
            
        if raw_loss < self.global_min:
            self.global_min = raw_loss
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1
            
        # 2. Anomaly Detection
        if raw_loss > self.ema_100 * self.spike_threshold:
            self.spikes.append(raw_loss)
            if len(self.spikes) > 3:
                self.spikes.pop(0)

    def to_dict(self) -> Dict:
        """Serialize state for checkpointing."""
        return {
            "loss_buffer": list(self.loss_buffer),
            "ema_100": self.ema_100,
            "global_min": self.global_min,
            "stagnation_steps": self.stagnation_steps,
            "total_steps": self.total_steps,
            "spikes": self.spikes
        }
    
    def from_dict(self, d: Dict):
        """Restore state from checkpoint dictionary."""
        self.loss_buffer = collections.deque(d.get("loss_buffer", []), maxlen=self.window_size)
        self.ema_100 = d.get("ema_100")
        self.global_min = d.get("global_min", float('inf'))
        self.stagnation_steps = d.get("stagnation_steps", 0)
        self.total_steps = d.get("total_steps", 0)
        self.spikes = d.get("spikes", [])
                
    def get_stats(self, progress_pct: float) -> Dict:
        if len(self.loss_buffer) < 2:
            return {}
            
        data = {
            "y": np.array(list(self.loss_buffer)),
            "ema_100": self.ema_100,
            "global_min": self.global_min,
            "stagnation_steps": self.stagnation_steps,
            "spikes": list(self.spikes),
            "total_steps": self.total_steps,
            "window_size": self.window_size
        }
        return self.calculate_from_data(data, progress_pct)

    @staticmethod
    def calculate_from_data(data: Dict, progress_pct: float) -> Dict:
        y = data["y"]
        if len(y) < 2: return {}
        
        x = np.arange(len(y))
        
        # Linear Regression for Trend Physics
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Signal to Noise Ratio (Turbulence)
        preds = m * x + c
        residuals = y - preds
        res_var = np.var(residuals)
        total_var = np.var(y)
        snr = res_var / (total_var + 1e-8)
        snr = min(1.0, snr) 
        
        # Oscillation Frequency
        diffs = np.diff(y)
        flips = np.sum(diffs[:-1] * diffs[1:] < 0)
        osc_freq = flips / (len(y) - 1)
        
        return {
            "meta": {
                "steps": f"{max(0, data['total_steps'] - data['window_size'])} - {data['total_steps']}",
                "progress": f"{progress_pct:.2f}%"
            },
            "loss_landscape": {
                "current_raw": float(y[-1]),
                "current_ema_100": float(data["ema_100"]),
                "global_min": float(data["global_min"]),
                "drift_from_min": float(y[-1] - data["global_min"]),
                "stagnation_duration": f"{data['stagnation_steps']} steps"
            },
            "trend_physics_window_100": {
                "slope_regression": f"{m:.6f}",
                "vector_direction": "CLIMBING" if m > 0 else "DESCENDING",
                "turbulence_snr": f"{snr:.2f} (0=Pure Signal, 1=Pure Noise)",
                "oscillation_freq": f"{osc_freq:.2f} (flips/step)"
            },
            "anomalies": {
                "spike_count": len(data["spikes"]),
                "last_3_spikes": [float(s) for s in data["spikes"]]
            }
        }

    def get_summary(self) -> Dict:
        """Get a compact summary for the terminal dashboard."""
        if len(self.loss_buffer) < 2:
            return None
            
        y = np.array(list(self.loss_buffer))
        x = np.arange(len(y))
        
        # Linear Regression for Trend
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # SNR
        preds = m * x + c
        residuals = y - preds
        res_var = np.var(residuals)
        total_var = np.var(y)
        snr = res_var / (total_var + 1e-8)
        
        return {
            "ema": self.ema_100 if self.ema_100 else y[-1],
            "min": self.global_min,
            "slope": m,
            "snr": min(1.0, snr),
            "steps": self.total_steps
        }
