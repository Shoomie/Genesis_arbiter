"""
FlashAttention Configuration and Verification Utilities

Provides centralized configuration for FlashAttention-2/3 integration
with PyTorch SDPA (Scaled Dot Product Attention).
"""

import torch
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Optional, Tuple
import warnings


class FlashAttentionConfig:
    """Configuration manager for FlashAttention integration."""
    
    def __init__(self):
        self.available = self._check_availability()
        self.backend = self._detect_backend()
        
    def _check_availability(self) -> bool:
        """Check if FlashAttention is available via SDPA."""
        if not torch.cuda.is_available():
            return False
        
        try:
            # Check if flash_sdp is enabled
            return torch.backends.cuda.flash_sdp_enabled()
        except Exception as e:
            warnings.warn(f"Could not verify FlashAttention availability: {e}")
            return False
    
    def _detect_backend(self) -> str:
        """Detect which SDPA backend will be used."""
        if not torch.cuda.is_available():
            return "cpu"
        
        backends = []
        if torch.backends.cuda.flash_sdp_enabled():
            backends.append("flash")
        if torch.backends.cuda.mem_efficient_sdp_enabled():
            backends.append("mem_efficient")
        if torch.backends.cuda.math_sdp_enabled():
            backends.append("math")
        
        return backends[0] if backends else "unknown"
    
    def print_status(self):
        """Print FlashAttention status information."""
        print("=" * 60)
        print("FlashAttention Configuration Status")
        print("=" * 60)
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Device: {torch.cuda.get_device_name(0)}")
            print(f"\nSDPA Backends Enabled:")
            print(f"  FlashAttention: {torch.backends.cuda.flash_sdp_enabled()}")
            print(f"  Memory Efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
            print(f"  Math (fallback): {torch.backends.cuda.math_sdp_enabled()}")
            print(f"\nActive Backend: {self.backend}")
        else:
            print("CUDA not available - running on CPU")
        
        print("=" * 60)


def print_flash_attention_status():
    """Convenience function to print FlashAttention status."""
    config = FlashAttentionConfig()
    config.print_status()


def is_flash_attention_available() -> bool:
    """Check if FlashAttention is available."""
    config = FlashAttentionConfig()
    return config.available


@contextmanager
def force_flash_attention(enabled: bool = True):
    """
    Context manager to force FlashAttention usage (or disable it for testing).
    
    Args:
        enabled: If True, only allow FlashAttention. If False, disable FlashAttention.
    
    Example:
        >>> with force_flash_attention(True):
        ...     output = model(input)  # Will use FA or raise error
    """
    if enabled:
        # Force FlashAttention only
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False
        ):
            yield
    else:
        # Disable FlashAttention (for benchmarking baseline)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=True
        ):
            yield


def benchmark_attention(
    batch_size: int = 4,
    seq_len: int = 2048,
    dim: int = 768,
    n_heads: int = 12,
    warmup_steps: int = 10,
    benchmark_steps: int = 50
) -> Tuple[float, float]:
    """
    Benchmark FlashAttention vs baseline SDPA.
    
    Returns:
        (baseline_time, flash_time) in seconds
    """
    import time
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_dim = dim // n_heads
    
    # Create dummy inputs
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    
    # Baseline (no FlashAttention)
    print("Benchmarking baseline SDPA...")
    with force_flash_attention(False):
        # Warmup
        for _ in range(warmup_steps):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(benchmark_steps):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
        baseline_time = time.time() - start
    
    # FlashAttention
    print("Benchmarking FlashAttention...")
    try:
        with force_flash_attention(True):
            # Warmup
            for _ in range(warmup_steps):
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(benchmark_steps):
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch.cuda.synchronize()
            flash_time = time.time() - start
    except Exception as e:
        print(f"FlashAttention failed: {e}")
        flash_time = float('inf')
    
    return baseline_time, flash_time


def get_memory_stats() -> dict:
    """Get current CUDA memory statistics."""
    if not torch.cuda.is_available():
        return {}
    
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
    }


if __name__ == "__main__":
    # Print configuration status
    config = FlashAttentionConfig()
    config.print_status()
    
    # Run benchmark if CUDA is available
    if torch.cuda.is_available():
        print("\nRunning attention benchmark...")
        baseline_time, flash_time = benchmark_attention()
        
        print(f"\nResults:")
        print(f"Baseline time: {baseline_time:.3f}s")
        print(f"FlashAttention time: {flash_time:.3f}s")
        
        if flash_time < float('inf'):
            speedup = baseline_time / flash_time
            print(f"Speedup: {speedup:.2f}x")
        else:
            print("FlashAttention not available")
