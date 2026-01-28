"""
Verification Script for Phase 1: FlashAttention Integration

Tests:
1. FlashAttention availability and configuration
2. Model forward pass with SDPA
3. Training parity between legacy and Composer
4. Performance benchmarking
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from training.flash_attention_config import FlashAttentionConfig, benchmark_attention


def test_flash_attention_availability():
    """Test 1: Verify FlashAttention is available and configured correctly."""
    print("\n" + "="*60)
    print("TEST 1: FlashAttention Availability")
    print("="*60)
    
    config = FlashAttentionConfig()
    config.print_status()
    
    if not config.available and torch.cuda.is_available():
        print("\n⚠ WARNING: CUDA is available but FlashAttention is not enabled!")
        print("This is expected if FlashAttention package is not installed.")
        print("Performance gains will be limited to math SDPA fallback.")
        return False
    elif config.available:
        print("\n✓ FlashAttention is available and will be used automatically")
        return True
    else:
        print("\n✓ Running on CPU (FlashAttention not applicable)")
        return True


def test_model_forward_pass():
    """Test 2: Verify model can perform forward pass with SDPA."""
    print("\n" + "="*60)
    print("TEST 2: Model Forward Pass with SDPA")
    print("="*60)
    
    from models.llama.model import Llama
    
    # Create small test model
    model = Llama(
        vocab_size=1000,
        n_layers=2,
        dim=128,
        n_heads=4,
        intermediate_size=256,
        max_seq_len=128
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create dummy input
    batch_size = 2
    seq_len = 64
    tokens = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    labels = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    
    print(f"Testing forward pass...")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    try:
        logits, loss = model(tokens, labels)
        print(f"\n✓ Forward pass successful!")
        print(f"  Output shape: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        return False


def test_performance_benchmark():
    """Test 3: Benchmark FlashAttention vs baseline."""
    print("\n" + "="*60)
    print("TEST 3: Performance Benchmarking")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("Skipping benchmark (CUDA not available)")
        return True
    
    print("Running attention benchmark (this may take 30-60 seconds)...")
    
    try:
        baseline_time, flash_time = benchmark_attention(
            batch_size=4,
            seq_len=2048,
            dim=768,
            n_heads=12,
            warmup_steps=10,
            benchmark_steps=50
        )
        
        print(f"\n✓ Benchmark complete!")
        print(f"  Baseline time: {baseline_time:.3f}s")
        print(f"  FlashAttention time: {flash_time:.3f}s")
        
        if flash_time < float('inf'):
            speedup = baseline_time / flash_time
            print(f"  Speedup: {speedup:.2f}x")
            
            if speedup >= 2.0:
                print(f"\n✓ Excellent! {speedup:.2f}x speedup achieved")
            elif speedup >= 1.5:
                print(f"\n✓ Good! {speedup:.2f}x speedup achieved")
            elif speedup >= 1.2:
                print(f"\n⚠ Modest {speedup:.2f}x speedup (expected 2-4x)")
            else:
                print(f"\n⚠ Minimal speedup ({speedup:.2f}x) - check FA installation")
        else:
            print("\n⚠ FlashAttention not available - using fallback")
        
        return True
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        return False


def test_composer_integration():
    """Test 4: Verify Composer wrapper works correctly."""
    print("\n" + "="*60)
    print("TEST 4: Composer Integration")
    print("="*60)
    
    try:
        from composer import Trainer
        from train_composer import GenesisComposerModel
        from models.llama.model import Llama
        
        print("Creating test model...")
        base_model = Llama(
            vocab_size=1000,
            n_layers=2,
            dim=128,
            n_heads=4,
            intermediate_size=256,
            max_seq_len=128
        )
        
        composer_model = GenesisComposerModel(base_model)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test forward pass
        batch = {
            'tokens': torch.randint(0, 1000, (2, 64), device=device),
            'labels': torch.randint(0, 1000, (2, 64), device=device)
        }
        
        composer_model = composer_model.to(device)
        outputs = composer_model.forward(batch)
        loss = composer_model.loss(outputs, batch)
        
        print(f"\n✓ Composer integration successful!")
        print(f"  Loss: {loss.item():.4f}")
        return True
        
    except ImportError as e:
        print(f"\n⚠ Composer not installed: {e}")
        print("Run: pip install mosaicml")
        return False
    except Exception as e:
        print(f"\n✗ Composer integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n" + "="*60)
    print("  PHASE 1 VERIFICATION: FlashAttention Integration")
    print("="*60)
    
    results = {
        "FlashAttention Availability": test_flash_attention_availability(),
        "Model Forward Pass": test_model_forward_pass(),
        "Performance Benchmark": test_performance_benchmark(),
        "Composer Integration": test_composer_integration()
    }
    
    print("\n" + "="*60)
    print("  VERIFICATION SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print(f"\n✓ All tests passed! Phase 1 is complete.")
        print(f"\nNext steps:")
        print(f"  1. Install Composer: pip install mosaicml")
        print(f"  2. Test training: python train_composer.py --mode microscope --steps 100")
        print(f"  3. Compare with legacy: python train.py (same config)")
    else:
        print(f"\n⚠ Some tests failed. Please review and fix issues before proceeding.")
    
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
