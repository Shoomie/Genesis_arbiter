# Training Parameters Guide: train_native_multi_task.py

**Complete Reference for Genesis Arbiter Multi-Task Training Configuration**

This guide provides an extensive explanation of every parameter in the `train_native_multi_task.py` training script, helping you understand how to configure your training runs for optimal results.

---

## Table of Contents

1. [Model Architecture Parameters](#model-architecture-parameters)
2. [Training Hyperparameters](#training-hyperparameters)
3. [Precision & Quantization Modes](#precision--quantization-modes)
4. [Batch Processing Parameters](#batch-processing-parameters)
5. [Checkpoint & Logging Parameters](#checkpoint--logging-parameters)
6. [Advanced Features](#advanced-features)
7. [Memory Optimization Strategies](#memory-optimization-strategies)
8. [Example Configurations](#example-configurations)

---

## Model Architecture Parameters

### `--mode` (Model Architecture)

**Type**: String  
**Default**: `"microscope"`  
**Choices**: `microscope`, `deep_narrow_40`, `deep_narrow_48`  
**Location**: Line 79, argparse line 429

**What it does:**  
Selects the neural network architecture configuration. Each mode defines a complete model specification including layer count, dimensions, and parameter count.

#### Available Architectures:

##### **1. Microscope** (~125M parameters)
```python
{
    "dim": 384,              # Hidden dimension size
    "n_layers": 77,          # Number of transformer layers
    "n_heads": 6,            # Number of attention heads
    "intermediate_size": 1536,  # FFN intermediate dimension
    "vocab_size": 8192,      # Vocabulary size
    "norm_type": "layernorm" # Normalization type
}
```

**Use case:**
- Fast experimentation and prototyping
- Limited VRAM (can run on 8GB GPUs)
- Quick iteration cycles
- Debugging and testing new features
- Educational purposes

**Memory footprint:**
- FP32: ~0.5 GB
- FP16: ~0.25 GB
- INT8: ~0.125 GB

---

##### **2. Deep Narrow 40** (~800M parameters)
```python
{
    "dim": 768,              # 2x wider than microscope
    "n_layers": 40,          # Moderate depth
    "n_heads": 12,           # More attention heads
    "intermediate_size": 3072,  # 4x hidden dim (standard ratio)
    "vocab_size": 8192,
    "norm_type": "deepnorm"  # Better for deep networks
}
```

**Use case:**
- Production-quality models
- Balance between speed and capacity
- Suitable for most downstream tasks
- 16-24GB VRAM GPUs (RTX 4090, A6000)

**Memory footprint:**
- FP32: ~3.2 GB
- FP16: ~1.6 GB
- INT8: ~0.8 GB

**Why DeepNorm?**  
DeepNorm (introduced in Microsoft's research) provides better gradient flow in deep networks by adjusting residual connection scaling. Critical for 40+ layer models to avoid vanishing gradients.

---

##### **3. Deep Narrow 48** (~1.0B parameters)
```python
{
    "dim": 896,              # Largest hidden dimension
    "n_layers": 48,          # Deepest architecture
    "n_heads": 14,           # Maximum attention heads
    "intermediate_size": 3584,
    "vocab_size": 8192,
    "norm_type": "deepnorm"  # Essential for this depth
}
```

**Use case:**
- Research-grade models
- Maximum representation capacity
- Complex theological reasoning tasks
- 24GB+ VRAM (A100, H100)

**Memory footprint:**
- FP32: ~4.0 GB
- FP16: ~2.0 GB
- INT8: ~1.0 GB

**Architecture philosophy:**  
"Deep and narrow" architectures prioritize depth (layers) over width (dimensions). This design:
- Enables hierarchical feature learning
- Better captures long-range dependencies
- Potentially exhibits grokking behavior more clearly
- Requires careful normalization (hence DeepNorm)

---

## Training Hyperparameters

### `--lr` (Learning Rate)

**Type**: Float  
**Default**: `2e-4` (0.0002)  
**Range**: `1e-5` to `5e-4` (typical)  
**Location**: Line 82, argparse line 442

**What it does:**  
Controls the step size during gradient descent optimization. This is arguably the **most critical hyperparameter** in deep learning.

**How it works:**
```
new_weight = old_weight - learning_rate × gradient
```

#### Learning Rate Guidelines:

| Model Size | Recommended LR | Reasoning |
|------------|----------------|-----------|
| Microscope (125M) | 3e-4 to 5e-4 | Smaller models can handle larger updates |
| Deep Narrow 40 (800M) | 2e-4 to 3e-4 | Standard LLM learning rate |
| Deep Narrow 48 (1B) | 1e-4 to 2e-4 | Large models need gentler updates |

**Precision adjustments:**
- **FP32**: Use default
- **FP16**: Can increase by 1.5-2x due to better gradient flow
- **INT8**: Reduce by 0.5-0.75x due to limited precision
- **INT4**: Reduce by 0.3-0.5x (experimental)

**Symptoms of incorrect LR:**

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Too high** | Loss explodes to NaN<br>Unstable training<br>Divergence | Reduce by 2-10x |
| **Too low** | Training plateaus early<br>Loss decreases too slowly<br>Never reaches good performance | Increase by 2-5x |

**Advanced tip:**  
Consider learning rate schedules (not implemented in current script):
- Warmup: Start low (1e-7), increase to target over 1000-5000 steps
- Cosine decay: Gradually reduce to 10% of initial LR
- Inverse square root: `lr = base_lr / sqrt(step)`

---

### `--weight-decay` (L2 Regularization)

**Type**: Float  
**Default**: `0.08`  
**Range**: `0.0` to `0.2` (typical)  
**Location**: Line 83, argparse line 443

**What it does:**  
Applies L2 regularization to prevent overfitting by penalizing large weights. Part of the AdamW optimizer.

**Mathematics:**
```
loss_total = loss_task + weight_decay × ||weights||²
```

**How to tune:**

| Dataset Size | Recommended Decay | Reasoning |
|--------------|-------------------|-----------|
| Small (<10M tokens) | 0.1 - 0.2 | Strong regularization needed |
| Medium (10M-100M) | 0.05 - 0.1 | Balanced approach |
| Large (>100M tokens) | 0.01 - 0.08 | Data provides regularization |

**Default 0.08 rationale:**  
- Based on GPT-3 and LLaMA training recipes
- Balances model capacity with generalization
- Works well for multi-task learning (prevents task-specific overfitting)

**Interaction with other parameters:**
- Higher LR → Increase weight decay (more aggressive updates need more regularization)
- Smaller batch size → Increase weight decay (more noise needs more regularization)
- Longer training → Increase weight decay (prevents late-stage overfitting)

**When to adjust:**
- **Overfitting** (val loss > train loss): Increase to 0.1-0.15
- **Underfitting** (both losses high): Decrease to 0.01-0.05
- **Grokking experiments**: Try 0.0 (zero decay) to study pure optimization

---

### `--batch-size` (Micro Batch Size)

**Type**: Integer  
**Default**: `4`  
**Range**: `1` to `64` (hardware dependent)  
**Location**: Line 73, argparse line 431

**What it does:**  
Number of training examples processed in a single forward/backward pass before gradient updates. This is the **physical batch size** that fits in GPU memory.

**Critical distinction:**
```
Micro batch size (--batch-size) = Per-GPU samples per iteration
Effective batch size = batch_size × grad_accum_steps × num_gpus
```

#### Batch Size Selection Guide:

| Model | Precision | 8GB VRAM | 16GB VRAM | 24GB VRAM | 40GB VRAM |
|-------|-----------|----------|-----------|-----------|-----------|
| Microscope | FP16 | 16 | 32 | 64 | 128 |
| Deep Narrow 40 | FP16 | 2 | 8 | 16 | 32 |
| Deep Narrow 48 | FP16 | 1 | 4 | 12 | 24 |
| Deep Narrow 48 | INT8 | 2 | 8 | 24 | 48 |

**Impact on training:**

1. **Gradient Noise**
   - Small batches (1-4): High variance, more exploration
   - Large batches (16+): Low variance, more exploitation
   
2. **Training Speed**
   - Larger batches = Better GPU utilization
   - But: Diminishing returns after certain point
   
3. **Generalization**
   - Research shows smaller batches often generalize better
   - But: Use gradient accumulation to get effective large batch

4. **Memory Usage**
   - Linear relationship: 2x batch size ≈ 2x memory
   - Accounts for ~40% of total VRAM usage

**Practical recommendations:**
- Start with largest batch size that fits in memory
- Monitor GPU utilization (aim for >80%)
- If memory constrained, reduce batch size and increase grad accumulation

---

### `--grad-accum` (Gradient Accumulation Steps)

**Type**: Integer  
**Default**: `4`  
**Range**: `1` to `64`  
**Location**: Line 75, argparse line 432

**What it does:**  
Accumulates gradients over multiple micro-batches before performing an optimizer step. This simulates larger batch sizes without additional memory.

**How it works:**
```python
for i in range(grad_accum_steps):
    loss = forward_pass(micro_batch[i])
    (loss / grad_accum_steps).backward()  # Accumulate gradients
optimizer.step()  # Single update using accumulated gradients
optimizer.zero_grad()
```

**Effective batch size formula:**
```
Effective Batch = batch_size × grad_accum_steps
```

**Examples:**
- `--batch-size 4 --grad-accum 4` → Effective batch = 16
- `--batch-size 2 --grad-accum 16` → Effective batch = 32
- `--batch-size 1 --grad-accum 64` → Effective batch = 64

#### Why Use Gradient Accumulation?

| Scenario | Configuration | Benefit |
|----------|---------------|---------|
| **Limited VRAM** | Small batch + Large accum | Fit large models on consumer GPUs |
| **Large batch training** | Medium batch + Medium accum | Match research paper configurations |
| **Memory efficiency** | Maximize batch, adjust accum | Optimal VRAM usage |

**Trade-offs:**

✅ **Advantages:**
- Enables training with limited memory
- Achieves large effective batch sizes
- Maintains gradient stability
- No accuracy loss vs. true large batches

❌ **Disadvantages:**
- Slower training (more forward passes per update)
- BatchNorm statistics less accurate (use LayerNorm instead)
- Slightly slower convergence in wall-clock time

**Recommended configurations:**

| Goal | Batch Size | Grad Accum | Effective Batch |
|------|------------|------------|-----------------|
| **Fast iteration** | 8-16 | 1-2 | 8-32 |
| **Stable training** | 4-8 | 4-8 | 16-64 |
| **Large batch (research)** | 2-4 | 16-32 | 32-128 |
| **Memory constrained** | 1-2 | 32-64 | 32-128 |

**Interaction with learning rate:**
- Larger effective batch → Can use slightly higher LR
- Rule of thumb: `lr_adjusted = lr_base × sqrt(effective_batch / base_batch)`

**Current default (4 × 4 = 16 effective):**  
Balanced choice for most scenarios. Provides stable gradients while maintaining reasonable training speed.

---

## Precision & Quantization Modes

### `--precision` (Training Precision)

**Type**: String  
**Default**: `"fp16"`  
**Choices**: `fp32`, `fp16`, `bf16`, `int8`, `int4`  
**Location**: Line 79, argparse line 433

**What it does:**  
Controls the numerical precision used for model weights, activations, and gradients during training. This is the **single most impactful parameter** for memory usage.

---

#### **FP32 (Full Precision)**

**Format**: 32-bit floating point  
**Range**: ±3.4 × 10³⁸  
**Precision**: ~7 decimal digits

**Memory**: 4 bytes per parameter  
**VRAM multiplier**: 1.0x (baseline)

**When to use:**
- ✅ Debugging numerical instabilities
- ✅ Small models where memory isn't a concern
- ✅ Research requiring maximum precision
- ✅ Final validation runs

**When to avoid:**
- ❌ Large models (>500M parameters)
- ❌ Limited VRAM (<16GB)
- ❌ Production training (too slow)

**Example:**
```bash
python train_native_multi_task.py --precision fp32 --mode microscope
```

**Performance characteristics:**
- **Speed**: Slowest (baseline 1.0x)
- **Stability**: Most stable
- **Memory**: Highest usage
- **Accuracy**: Maximum precision

---

#### **FP16 (Half Precision)** ⭐ DEFAULT

**Format**: 16-bit floating point  
**Range**: ±65,504  
**Precision**: ~3 decimal digits

**Memory**: 2 bytes per parameter  
**VRAM multiplier**: 0.5x

**How it works:**
- Forward/backward passes in FP16
- Gradient accumulation in FP16
- Optimizer states in FP32 (mixed precision)
- Loss scaling to prevent underflow

**When to use:**
- ✅ **Default choice for most training**
- ✅ All NVIDIA GPUs with Tensor Cores (2000 series+)
- ✅ Production training
- ✅ Balance of speed and memory

**Limitations:**
- Small gradients can underflow (< 6 × 10⁻⁸)
- Requires gradient scaling
- Some operations fallback to FP32

**Example:**
```bash
python train_native_multi_task.py --precision fp16  # Default
```

**Performance characteristics:**
- **Speed**: 2-3x faster than FP32 (with Tensor Cores)
- **Stability**: Good (with proper scaling)
- **Memory**: 50% of FP32
- **Accuracy**: Minimal degradation (<1%)

**Gradient scaling:**
The script automatically uses PyTorch's `GradScaler`:
```python
scaler = GradScaler('cuda', enabled=True)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

This multiplies losses by 2¹⁶ before backward pass to prevent gradient underflow.

---

#### **BF16 (Brain Float 16)**

**Format**: 16-bit floating point (Google's format)  
**Range**: ±3.4 × 10³⁸ (same as FP32!)  
**Precision**: ~2 decimal digits

**Memory**: 2 bytes per parameter  
**VRAM multiplier**: 0.5x

**Key difference from FP16:**
- **FP16**: 5 exponent bits, 10 mantissa bits (limited range, good precision)
- **BF16**: 8 exponent bits, 7 mantissa bits (same range as FP32, less precision)

**When to use:**
- ✅ Training very deep networks (>40 layers)
- ✅ Models prone to overflow/underflow
- ✅ Modern GPUs (Ampere/Ada/Hopper: RTX 3000+, A100, H100)
- ✅ When FP16 shows instability

**When to avoid:**
- ❌ Older GPUs without BF16 support (pre-Ampere)
- ❌ Inference (FP16 usually better)

**Example:**
```bash
python train_native_multi_task.py --precision bf16 --mode deep_narrow_48
```

**Performance characteristics:**
- **Speed**: Similar to FP16 (2-3x faster than FP32)
- **Stability**: Better than FP16 (no gradient scaling needed)
- **Memory**: Same as FP16 (50% of FP32)
- **Accuracy**: Comparable to FP16

**Why BF16 for deep models:**
- Wider dynamic range prevents intermediate overflow
- No gradient scaling required (simpler training)
- Matches FP32 exponent range exactly
- Used in TPU training (Google's default)

---

#### **INT8 (8-bit Quantization)**

**Format**: 8-bit integer  
**Range**: 0 to 255 (unsigned) or -128 to 127 (signed)  
**Precision**: Integer values only

**Memory**: 1 byte per parameter  
**VRAM multiplier**: 0.25x

**Requirements:**
```bash
pip install bitsandbytes
```

**How it works (LLM.int8() method):**
1. Weights stored in INT8
2. Activations in FP16/FP32
3. Outlier features handled separately in FP16
4. 8-bit matrix multiplication
5. Optimizer also quantized (AdamW8bit)

**When to use:**
- ✅ Limited VRAM (8-12GB GPUs)
- ✅ Large models that don't fit in FP16
- ✅ Longer training runs (memory matters more than speed)
- ✅ Experimenting with >1B parameter models

**When to avoid:**
- ❌ Small models (overhead not worth it)
- ❌ Maximum accuracy required
- ❌ Short experiments (setup overhead)

**Example:**
```bash
python train_native_multi_task.py --precision int8 --mode deep_narrow_48 --batch-size 8
```

**Performance characteristics:**
- **Speed**: 0.7-0.9x of FP16 (slight slowdown)
- **Stability**: Good (requires tuning)
- **Memory**: 25% of FP32, 50% of FP16
- **Accuracy**: Minor degradation (1-3%)

**Script integration:**
```python
if precision == "int8":
    import bitsandbytes as bnb
    optimizer = bnb.optim.AdamW8bit(...)  # Lines 208-215
```

**Tuning recommendations for INT8:**
- Reduce learning rate by 20-30%
- Use slightly higher weight decay
- Monitor training curves closely
- May require more steps to converge

---

#### **INT4 (4-bit Quantization)** ⚠️ EXPERIMENTAL

**Format**: 4-bit integer  
**Range**: 0 to 15 (unsigned) or -8 to 7 (signed)  
**Precision**: Very limited

**Memory**: 0.5 bytes per parameter  
**VRAM multiplier**: 0.125x

**Status**: Experimental (research use only)

**When to use:**
- ✅ Extreme memory constraints
- ✅ Research into quantization effects
- ✅ Prototyping massive models (>10B parameters)

**When to avoid:**
- ❌ **Most production scenarios**
- ❌ Critical accuracy requirements
- ❌ Stable training needed

**Example:**
```bash
python train_native_multi_task.py --precision int4 --mode deep_narrow_48 --lr 1e-4
```

**Performance characteristics:**
- **Speed**: Variable (0.5-0.8x of FP16)
- **Stability**: Poor (requires careful tuning)
- **Memory**: 12.5% of FP32
- **Accuracy**: Significant degradation (5-10%)

**Recommendations:**
- Start with pretrained model in higher precision
- Use very low learning rates (1e-5 to 5e-5)
- Increase training steps by 2-3x
- Consider mixed precision (4-bit weights, 16-bit activations)

---

### Precision Comparison Table

| Precision | Memory | Speed | Stability | Accuracy | Best For |
|-----------|--------|-------|-----------|----------|----------|
| **FP32** | 4 bytes | 1.0x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Debugging, validation |
| **FP16** ⭐ | 2 bytes | 2.5x | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **Default choice** |
| **BF16** | 2 bytes | 2.5x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Deep networks (40+ layers) |
| **INT8** | 1 byte | 2.0x | ⭐⭐⭐ | ⭐⭐⭐ | Memory-constrained GPUs |
| **INT4** | 0.5 bytes | 1.5x | ⭐⭐ | ⭐⭐ | Research only |

---

## Batch Processing Parameters

### `--steps` (Maximum Training Steps)

**Type**: Integer  
**Default**: `50000` (updated from 10000)  
**Range**: `100` to `1,000,000+`  
**Location**: Line 74, argparse line 432

**What it does:**  
Total number of optimizer updates (not batches or epochs). This determines training duration.

**Step vs. Batch vs. Epoch:**
```
1 Step = 1 optimizer update
       = grad_accum_steps × forward/backward passes
       = processing grad_accum_steps × batch_size examples

1 Epoch = Processing entire dataset once
        = dataset_size / (batch_size × grad_accum_steps) steps
```

**How to choose:**

| Model Size | Dataset Size | Recommended Steps | Training Time (estimate) |
|------------|--------------|-------------------|--------------------------|
| Microscope | 10M tokens | 5,000 - 10,000 | 2-4 hours (single GPU) |
| Deep Narrow 40 | 50M tokens | 20,000 - 50,000 | 12-24 hours |
| Deep Narrow 48 | 100M+ tokens | 50,000 - 200,000 | 2-7 days |

**Grokking considerations:**
- Grokking often occurs at 10,000-100,000 steps
- Need patience: Performance may plateau before sudden jump
- Current default (50,000) allows observation of grokking

**Early stopping:**
- Monitor validation loss
- Stop if no improvement for 5,000-10,000 steps
- This script doesn't implement automatic early stopping

**Conversion to epochs:**
```python
# Example: 100M token dataset, batch_size=4, grad_accum=4, seq_len=512
tokens_per_step = 4 × 4 × 512 = 8,192 tokens
steps_per_epoch = 100M / 8,192 ≈ 12,200 steps

# 50,000 steps ≈ 4.1 epochs
```

**Step interval relationships:**
- `save_interval` should be divisible into `steps`
- `val_interval` should be < `save_interval`
- `log_interval` should be much smaller (1% of steps)

---

## Checkpoint & Logging Parameters

### `--save-interval` (Checkpoint Frequency)

**Type**: Integer  
**Default**: `500`  
**Range**: `100` to `10,000`  
**Location**: Line 76, argparse line 434

**What it does:**  
Number of steps between saving model checkpoints to disk.

**Checkpoint contents** (see line 336-342):
```python
{
    'step': global_step,
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'config': config
}
```

**Storage calculation:**
```
Checkpoint size ≈ Model parameters × precision × 3
                  (model + optimizer states [2x])

Example: Deep Narrow 40 (800M params) in FP16
= 800M × 2 bytes × 3 = 4.8 GB per checkpoint
```

**How to choose:**

| Training Duration | Save Interval | Reasoning |
|-------------------|---------------|-----------|
| <10,000 steps | 500-1,000 | Frequent saves for short runs |
| 10,000-50,000 | 1,000-2,000 | Balance storage and safety |
| >50,000 steps | 2,000-5,000 | Reduce disk usage |

**Storage planning:**
```
Total checkpoints = steps / save_interval
Disk usage = checkpoint_size × total_checkpoints

Example: 50,000 steps, interval 500, 4.8GB checkpoints
= 100 checkpoints × 4.8GB = 480GB required
```

**Recommendations:**
- Save more frequently early (when learning is rapid)
- Reduce frequency later (when converged)
- Keep last 3-5 checkpoints + best validation checkpoint
- For long runs: Implement checkpoint rotation (not in current script)

**Current locations:**
```
checkpoints/multi_task_{mode}/step_500.pt
checkpoints/multi_task_{mode}/step_1000.pt
...
checkpoints/multi_task_{mode}/final_step_50000.pt
```

---

### `--log-interval` (Console Logging Frequency)

**Type**: Integer  
**Default**: `10`  
**Range**: `1` to `1,000`  
**Location**: Line 77, argparse line 435

**What it does:**  
Number of steps between printing training metrics to console.

**Logged metrics** (see lines 304-308):
```
Step 1000/50000 | Loss: 3.8452 | Task: lm | Speed: 2.34 steps/s | ETA: 5:48:12
```

**Display includes:**
- Current step / total steps
- Average loss over interval
- Current task type
- Training speed (steps/second)
- Estimated time to completion (ETA)

**How to choose:**

| Preference | Interval | Use Case |
|------------|----------|----------|
| **Detailed monitoring** | 1-5 | Debugging, short runs |
| **Standard** | 10-50 | Normal training |
| **Minimal** | 100-500 | Long runs, reduce clutter |

**Performance impact:**
- Almost negligible (<0.1% overhead)
- TensorBoard logging has more overhead

**Relationship to other intervals:**
```
log_interval << val_interval << save_interval
Typical: 10 << 100 << 500
```

**ETA calculation** (lines 299-302):
```python
remaining_steps = max_steps - global_step
eta_seconds = remaining_steps / steps_per_sec
```
Provides real-time estimate of training completion.

---

### `--val-interval` (Validation Frequency)

**Type**: Integer  
**Default**: `100`  
**Range**: `10` to `5,000`  
**Location**: Line 78, argparse line 436

**What it does:**  
Number of steps between running validation passes to check model performance on held-out data.

**Validation process** (see lines 380-410):
1. Switch model to eval mode
2. Disable gradient computation
3. Run forward passes on validation data (10 batches default)
4. Calculate average validation loss
5. Log to TensorBoard
6. Trigger grokking detection callbacks
7. Return to training mode

**Why validate:**
- Detect overfitting (train loss ↓, val loss ↑)
- Monitor generalization
- Detect grokking events (sudden val loss drop)
- Guide hyperparameter tuning

**Validation cost:**
```
Time cost = num_val_batches × batch_forward_time
          ≈ 10 batches × 0.5s = 5s every 100 steps
          ≈ 0.5% total training time overhead
```

**How to choose:**

| Training Phase | Interval | Reasoning |
|----------------|----------|-----------|
| **Early (<5K steps)** | 50-100 | Rapid changes, frequent monitoring |
| **Middle** | 100-500 | Stable phase |
| **Late (convergence)** | 500-1000 | Infrequent changes |

**Grokking detection:**
The `GrokkingDetector` callback (lines 201-204) monitors validation loss for sudden improvements:
```python
GrokkingDetector(
    patience=1000,      # Wait 1000 steps before declaring grokking
    threshold=0.10,     # >10% improvement triggers detection
    checkpoint_dir=...
)
```

**TensorBoard logging:**
```python
writer.add_scalar("val/loss", val_loss, global_step)  # Line 324
```

**Best practices:**
- Validation should be frequent enough to catch grokking
- But not so frequent it slows training
- 1-2% of training steps is ideal
- Always validate before saving checkpoints

---

## Advanced Features

### `--no-grokking` (Disable Grokking Detection)

**Type**: Boolean flag  
**Default**: `False` (grokking detection enabled)  
**Location**: Line 80, argparse line 441

**What it does:**  
Disables the callback system that monitors for grokking behavior during training.

**Grokking callbacks** (lines 196-209):

1. **GrokkingDetector**
   - Monitors validation loss for sudden improvements
   - Patience: 1000 steps
   - Threshold: 10% improvement
   - Saves special checkpoint when grokking detected

2. **ProcrustesMonitor**
   - Tracks representation alignment across layers
   - Evaluation interval: 1000 steps
   - Detects when representations crystallize

3. **ConceptClusteringMonitor**
   - Analyzes hidden state clustering
   - Evaluation interval: 2000 steps
   - Identifies concept formation

**When to disable:**
- ✅ Production training (faster, less overhead)
- ✅ Memory constrained (saves ~500MB)
- ✅ Not studying grokking phenomena
- ✅ Shorter training runs (<10K steps)

**When to enable:**
- Research into grokking
- Studying generalization dynamics
- Multi-task learning experiments
- Understanding representation learning

**Performance overhead:**
- ~2-5% training time increase
- ~500MB additional memory
- Extra checkpoints saved

**Example usage:**
```bash
# Disable for production
python train_native_multi_task.py --no-grokking --steps 100000

# Enable for research (default)
python train_native_multi_task.py --steps 50000
```

---

### `--bible-dir` (Data Directory)

**Type**: String  
**Default**: `"../../Bible"`  
**Location**: Line 81, argparse line 442

**What it does:**  
Path to directory containing Bible translation data for multi-task learning.

**Expected directory structure:**
```
Bible/
├── complete/
│   ├── english_nwt.json
│   ├── spanish_nwt.json
│   └── ...
├── partial/
│   └── ...
└── metadata.json
```

**Used by multi-task dataloader** (lines 186-193):
```python
dataloader = get_multi_task_dataloader(
    corpus_path="nwt_corpus.txt",       # Primary LM data
    bible_data_dir=bible_dir,           # Cross-lingual data
    tokenizer=tokenizer,
    batch_size=batch_size,
    max_seq_len=512
)
```

**Multi-task breakdown:**
- **70%** Language Modeling (corpus_path)
- **15%** Coherence Detection (bible_dir verses)
- **7.5%** Cross-Reference (bible_dir references)
- **7.5%** Paraphrase Detection (bible_dir translations)

**Data requirements:**
- Minimum: 10MB of text data
- Recommended: 100MB+ for deep models
- Cross-lingual: Multiple translations for same verses

**Path resolution:**
- Relative to script execution directory
- Use absolute path if running from different directories

**Example configurations:**
```bash
# Default (relative path)
python train_native_multi_task.py --bible-dir ../../Bible

# Absolute path
python train_native_multi_task.py --bible-dir E:/Data/Genesis_Bible_Data

# Different dataset
python train_native_multi_task.py --bible-dir /mnt/data/multilingual_bible
```

---

## Memory Optimization Strategies

### Understanding VRAM Usage

**Total VRAM breakdown:**
```
Total = Model + Optimizer + Gradients + Activations + Framework Overhead

Example: Deep Narrow 40 (800M params), FP16, batch_size=4
- Model parameters:    1.6 GB (800M × 2 bytes)
- Optimizer states:    3.2 GB (AdamW: 2× parameters in FP32)
- Gradients:           1.6 GB (same size as model)
- Activations:         2.4 GB (depends on batch size & seq length)
- PyTorch overhead:    0.8 GB
─────────────────────────────
Total:                 9.6 GB
```

### Memory-Saving Techniques

#### 1. **Reduce Batch Size**
```bash
# Instead of: --batch-size 8
python train_native_multi_task.py --batch-size 2 --grad-accum 16
# Same effective batch (32), 4x less VRAM
```

#### 2. **Use Quantization**
```bash
# FP16 → INT8: 50% memory reduction
python train_native_multi_task.py --precision int8
```

#### 3. **Gradient Checkpointing** (not yet implemented)
Recompute activations during backward pass instead of storing them.
Trade: 30% slower for 50% less memory

#### 4. **Disable Grokking Callbacks**
```bash
python train_native_multi_task.py --no-grokking
# Saves ~500MB
```

#### 5. **Reduce Sequence Length** (edit dataloader config)
```python
max_seq_len=256  # Instead of 512
# Memory ∝ sequence_length, so 50% reduction
```

### Memory Budget Examples

**8GB GPU (RTX 3070, RTX 4060 Ti):**
```bash
# Microscope model
python train_native_multi_task.py \
    --mode microscope \
    --precision fp16 \
    --batch-size 8 \
    --grad-accum 4

# Deep Narrow 40 (tight fit)
python train_native_multi_task.py \
    --mode deep_narrow_40 \
    --precision int8 \
    --batch-size 1 \
    --grad-accum 32 \
    --no-grokking
```

**16GB GPU (RTX 4080, RTX 4090):**
```bash
# Deep Narrow 40 (comfortable)
python train_native_multi_task.py \
    --mode deep_narrow_40 \
    --precision fp16 \
    --batch-size 4 \
    --grad-accum 4

# Deep Narrow 48 (tight fit)
python train_native_multi_task.py \
    --mode deep_narrow_48 \
    --precision int8 \
    --batch-size 2 \
    --grad-accum 8
```

**24GB GPU (RTX 4090, RTX 6000 Ada, A5000):**
```bash
# Deep Narrow 48 (optimal)
python train_native_multi_task.py \
    --mode deep_narrow_48 \
    --precision fp16 \
    --batch-size 6 \
    --grad-accum 4

# Or with higher batch for speed
python train_native_multi_task.py \
    --mode deep_narrow_48 \
    --precision bf16 \
    --batch-size 12 \
    --grad-accum 2
```

**40GB+ GPU (A100, H100):**
```bash
# Maximum throughput
python train_native_multi_task.py \
    --mode deep_narrow_48 \
    --precision bf16 \
    --batch-size 16 \
    --grad-accum 2 \
    --steps 100000
```

---

## Example Configurations

### 1. Quick Prototype (Development)
**Goal**: Fast iteration, test code changes
```bash
python train_native_multi_task.py \
    --mode microscope \
    --precision fp16 \
    --batch-size 16 \
    --grad-accum 1 \
    --steps 1000 \
    --save-interval 200 \
    --log-interval 5 \
    --val-interval 50 \
    --no-grokking \
    --lr 5e-4
```
**Expected**: 15-30 minutes, 2-4GB VRAM

---

### 2. Grokking Research (Small Model)
**Goal**: Study grokking phenomena
```bash
python train_native_multi_task.py \
    --mode microscope \
    --precision fp16 \
    --batch-size 8 \
    --grad-accum 4 \
    --steps 50000 \
    --save-interval 1000 \
    --val-interval 100 \
    --lr 3e-4 \
    --weight-decay 0.08
```
**Expected**: 8-12 hours, 4-6GB VRAM, grokking at 10K-30K steps

---

### 3. Production Training (Medium Model)
**Goal**: Best model quality, reasonable time
```bash
python train_native_multi_task.py \
    --mode deep_narrow_40 \
    --precision bf16 \
    --batch-size 6 \
    --grad-accum 6 \
    --steps 100000 \
    --save-interval 2000 \
    --log-interval 20 \
    --val-interval 500 \
    --lr 2e-4 \
    --weight-decay 0.08 \
    --no-grokking
```
**Expected**: 2-3 days, 14-18GB VRAM

---

### 4. Large Model (Limited VRAM)
**Goal**: Train 1B model on 16GB GPU
```bash
python train_native_multi_task.py \
    --mode deep_narrow_48 \
    --precision int8 \
    --batch-size 2 \
    --grad-accum 16 \
    --steps 150000 \
    --save-interval 3000 \
    --log-interval 25 \
    --val-interval 1000 \
    --lr 1.5e-4 \
    --weight-decay 0.1 \
    --no-grokking
```
**Expected**: 5-7 days, 15GB VRAM

---

### 5. Maximum Performance (High-End GPU)
**Goal**: Fastest training on A100/H100
```bash
python train_native_multi_task.py \
    --mode deep_narrow_48 \
    --precision bf16 \
    --batch-size 24 \
    --grad-accum 2 \
    --steps 200000 \
    --save-interval 5000 \
    --log-interval 50 \
    --val-interval 1000 \
    --lr 2.5e-4 \
    --weight-decay 0.08
```
**Expected**: 3-4 days, 35-40GB VRAM

---

### 6. Debugging Run
**Goal**: Find and fix issues quickly
```bash
python train_native_multi_task.py \
    --mode microscope \
    --precision fp32 \
    --batch-size 2 \
    --grad-accum 1 \
    --steps 100 \
    --save-interval 50 \
    --log-interval 1 \
    --val-interval 10 \
    --lr 1e-4 \
    --no-grokking
```
**Expected**: 5 minutes, shows any errors immediately

---

## Parameter Interaction Matrix

Understanding how parameters interact is crucial for effective tuning:

| Change | Affects | Recommended Adjustment |
|--------|---------|------------------------|
| Increase `--batch-size` | VRAM ↑, Speed ↑ | May increase `--lr` by √(ratio) |
| Increase `--grad-accum` | Time ↑, Stability ↑ | May increase `--lr` by √(ratio) |
| Change to `--precision int8` | VRAM ↓, Speed ↓ | Decrease `--lr` by 0.7x |
| Increase `--steps` | Training time ↑ | Increase `--save-interval` |
| Deeper `--mode` | VRAM ↑, Capacity ↑ | Decrease `--lr`, increase steps |
| Decrease `--lr` | Convergence slower | Increase `--steps` |
| Increase `--weight-decay` | Regularization ↑ | May need higher `--lr` |
| Enable `--no-grokking` | VRAM ↓, Speed ↑ | N/A |

---

## Quick Reference: Default Configuration

When you run:
```bash
python train_native_multi_task.py
```

You get:
```yaml
Model:
  mode: microscope
  parameters: ~125M
  layers: 77
  dimension: 384
  heads: 6

Training:
  precision: fp16
  batch_size: 4
  grad_accum: 4
  effective_batch: 16
  learning_rate: 2e-4
  weight_decay: 0.08
  max_steps: 50000

Intervals:
  log_interval: 10     (every 10 steps)
  val_interval: 100    (every 100 steps)
  save_interval: 500   (every 500 steps)

Features:
  grokking_detection: enabled
  mixed_precision: enabled (FP16)
  flash_attention: auto-detect

Resources:
  estimated_vram: ~2 GB
  estimated_time: ~6-10 hours (single GPU)
  estimated_storage: ~50 checkpoints × 500MB = 25GB
```

---

## Troubleshooting Guide

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions (try in order):**
1. Reduce `--batch-size` by 50%
2. Increase `--grad-accum` to maintain effective batch
3. Change `--precision` to int8
4. Add `--no-grokking`
5. Use smaller `--mode`

---

### Loss is NaN

**Symptoms:**
```
Step 850/50000 | Loss: nan | Task: lm
```

**Solutions:**
1. **Reduce learning rate**: Try 0.1x of current
2. **Check precision**: FP16 may underflow, try BF16 or FP32
3. **Gradient clipping**: Already at 1.0 (line 279), try 0.5
4. **Check data**: Ensure no inf/nan in dataset
5. **Warmup**: Start with very low LR (1e-7), increase gradually

---

### Training Too Slow

**Symptoms:**
- <1 step/second on modern GPU
- GPU utilization <50%

**Solutions:**
1. **Increase batch size** (GPU underutilized)
2. **Check precision**: FP32 is 2-3x slower than FP16
3. **Verify FlashAttention**: Script prints status at startup
4. **Reduce validation**: Increase `--val-interval`
5. **Disable callbacks**: Use `--no-grokking`

---

### Not Learning / Plateau

**Symptoms:**
- Loss decreases then plateaus
- Validation loss not improving

**Solutions:**
1. **Increase learning rate** (underfitting)
2. **Reduce weight decay** (over-regularization)
3. **Train longer** (may be pre-grokking)
4. **Check data diversity** (overfitting to limited data)
5. **Increase model capacity** (model too small)

---

## Advanced Topics

### Learning Rate Scheduling (Future Enhancement)

Currently, the script uses constant LR. Consider implementing:

```python
# Warmup + Cosine Decay
def get_lr(step, base_lr, warmup_steps, max_steps):
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
```

### Gradient Checkpointing (Future Enhancement)

Trade compute for memory:
```python
from torch.utils.checkpoint import checkpoint

# In transformer layer:
output = checkpoint(self.attention, hidden_states)
```
Effect: ~30% slower, ~50% less memory

### Multi-GPU Training (Not Yet Implemented)

For distributed training, would need:
```bash
torchrun --nproc_per_node=4 train_native_multi_task.py ...
```

### Custom Task Weights (Not Configurable)

Currently hardcoded (lines 125-129):
- LM: 70%
- Coherence: 15%
- Cross-ref: 7.5%
- Paraphrase: 7.5%

Future: Add `--task-weights` argument

---

## Conclusion

This guide covered all parameters in `train_native_multi_task.py`. Key takeaways:

1. **Start simple**: Use defaults, then adjust based on observations
2. **Monitor closely**: Watch loss curves and resource usage
3. **Iterate carefully**: Change one parameter at a time
4. **Understand trade-offs**: Memory vs speed vs accuracy
5. **Document experiments**: Track parameter combinations and results

**Recommended first run:**
```bash
python train_native_multi_task.py \
    --mode microscope \
    --precision fp16 \
    --steps 10000 \
    --save-interval 500
```

Then scale up based on results and available resources.

Happy training! 🚀
