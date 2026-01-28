# Phase 1 Installation \u0026 Verification Guide

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (for FlashAttention support)
- PyTorch 2.0+
- At least 12GB VRAM recommended (RTX 4070 or better)

## Installation Steps

### 1. Install Composer Framework

```powershell
pip install mosaicml
```

### 2. Install FlashAttention (Optional but Recommended)

> [!WARNING]
> FlashAttention requires CUDA 11.8+ and may take 10-15 minutes to compile. If you encounter issues, the system will gracefully fall back to PyTorch's optimized SDPA implementation which still provides significant speedups.

```powershell
pip install flash-attn --no-build-isolation
```

**Common Issues:**
- **CUDA version mismatch**: Verify with `nvcc --version` or `nvidia-smi`
- **Compiler not found**: Install Visual Studio Build Tools on Windows
- **Out of memory during build**: Close other applications

### 3. Install Other Dependencies

```powershell
cd e:\AI\Research\Genesis_arbiter
pip install -r requirements_composer.txt
```

## Verification

### Run Verification Script

```powershell
cd e:\AI\Research\Genesis_arbiter\engine
python verify_phase1.py
```

**Expected Output:**
```
==========================================================
TEST 1: FlashAttention Availability
==========================================================
CUDA Available: True
CUDA Version: 11.8
Device: NVIDIA GeForce RTX 4070
...
✓ FlashAttention is available and will be used automatically

==========================================================
TEST 2: Model Forward Pass with SDPA
==========================================================
✓ Forward pass successful!

==========================================================
TEST 3: Performance Benchmarking
==========================================================
Baseline time: 2.450s
FlashAttention time: 0.612s
Speedup: 4.00x
✓ Excellent! 4.00x speedup achieved

==========================================================
TEST 4: Composer Integration
==========================================================
✓ Composer integration successful!

==========================================================
VERIFICATION SUMMARY
==========================================================
✓ PASS  FlashAttention Availability
✓ PASS  Model Forward Pass
✓ PASS  Performance Benchmark
✓ PASS  Composer Integration

✓ All tests passed! Phase 1 is complete.
```

### Manual Testing

If you want to test training manually:

```powershell
cd e:\AI\Research\Genesis_arbiter\engine

# Test with small model (Microscope - 125M params)
python train_composer.py --mode microscope --steps 100 --batch-size 4

# Compare with legacy training (for validation)
python train.py --mode microscope
```

**Expected Behavior:**
- Training should start without errors
- FlashAttention status should show in initial output
- Training speed should be 2-4x faster than legacy `train.py`
- Memory usage should be similar or lower
- Loss curves should match between Composer and legacy (within 5%)

## Troubleshooting

### Issue: "Composer not installed"
**Solution:**
```powershell
pip install mosaicml
```

### Issue: "FlashAttention not available"
**Diagnosis:** Run the verification script to check CUDA availability
```powershell
python verify_phase1.py
```

**Solution:**
- Verify CUDA 11.8+ is installed
- Try installing FlashAttention again with verbose output:
  ```powershell
  pip install flash-attn --no-build-isolation --verbose
  ```
- If it still fails, you can proceed without FlashAttention. PyTorch SDPA will provide ~2x speedup instead of ~4x

### Issue: "CUDA out of memory"
**Solution:**
- Reduce batch size: `--batch-size 2` or `--batch-size 1`
- Increase gradient accumulation: `--grad-accum 8`
- Use a smaller model: `--mode microscope` or `--mode deep_narrow_32`

### Issue: "Training loss doesn't match legacy"
**Diagnosis:** Run both training scripts side-by-side with same config
```powershell
# Terminal 1
python train_composer.py --mode microscope --steps 100

# Terminal 2
python train.py --mode microscope
```

**Expected:** Losses should be within 5% of each other

## Next Steps

Once Phase 1 verification passes:

1. **Proceed to Phase 2**: Multi-task learning infrastructure
   - Implement multi-task heads (coherence, cross-reference, paraphrase)
   - Create multi-task data sampling
   
2. **Or run initial experiments**: Test FlashAttention speedup on your target model
   ```powershell
   python train_composer.py --mode deep_narrow_40 --steps 10000
   ```

3. **Monitor with TensorBoard**:
   ```powershell
   tensorboard --logdir ./checkpoints/logs/tensorboard
   ```

## Files Created in Phase 1

- `engine/training/flash_attention_config.py` - FlashAttention utilities
- `engine/train_composer.py` - Composer training script
- `engine/verify_phase1.py` - Verification script
- `requirements_composer.txt` - Dependencies

## Performance Expectations

With FlashAttention enabled on RTX 4070 (12GB VRAM):

| Model | Params | Batch Size | Tokens/sec (Before) | Tokens/sec (After) | Speedup |
|-------|--------|------------|---------------------|--------------------| |
| Microscope | 125M | 4 | ~1000 | ~3500 | 3.5x |
| Deep Narrow 40 | 800M | 4 | ~400 | ~1400 | 3.5x |
| Deep Narrow 48 | 1B | 2 | ~300 | ~1000 | 3.3x |

Memory usage should remain roughly the same or decrease slightly due to SDPA's memory-efficient implementation.
