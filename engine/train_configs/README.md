# Training Configurations

This directory contains hardware-tier specific TOML configuration files for the Genesis training system.

> **Note**: These are **legacy configurations** from Phase 1-2. For Phase 3 Arbiter automation, see [`configs/sweep_templates/`](../../configs/sweep_templates/) for parameter sweep configurations.

---

## Legacy Hardware Tiers

### `high_vram.toml`
**For GPUs with 12+ GB VRAM**

Compatible with: RTX 3080, RTX 4070, RTX 4080, RTX 4090, A5000, A6000

**Configuration**:
- Batch Size: 4
- Gradient Accumulation: 4
- Precision: BF16
- Flash Attention: Enabled
- Effective Batch Size: 16

**Usage**:
```bash
cd engine
python train.py
# Select: [1] High VRAM tier
```

---

### `low_vram.toml`
**For GPUs with 4-6 GB VRAM**

Compatible with: GTX 1660, RTX 3050, Quadro T2000, RTX 2060

**Configuration**:
- Batch Size: 1
- Gradient Accumulation: 32
- Precision: FP16
- Gradient Checkpointing: Enabled
- Effective Batch Size: 32

**Usage**:
```bash
cd engine
python train.py
# Select: [2] Low VRAM tier
```

---

## Phase 3: Arbiter Automation Configs

For automated parameter sweeps and grokking experiments, use the new configuration templates:

### Location
[`configs/sweep_templates/`](../../configs/sweep_templates/)

### Available Templates

1. **`base_config.toml`**: Shared settings (checkpointing, logging, precision)
2. **`deep_narrow_template.toml`**: Parameterized Deep & Narrow architecture
3. **`grokking_regime.toml`**: Extreme weight decay (0.2) + extended training (500k steps)
4. **`baseline_standard.toml`**: Wide & shallow comparison baseline

### Usage with Sweep Orchestrator
```bash
cd engine
python arbiter_sweep_orchestrator.py \
    --base-config ../configs/sweep_templates/base_config.toml \
    --strategy grid \
    --gpus 4
```

---

## Configuration Structure

Both legacy and modern configs follow this TOML structure:

```toml
[model]
name = "theos_small"
dim = 1024
n_layers = 80
n_heads = 16
vocab_size = 8192

[training]
batch_size = 16
seq_len = 4096
learning_rate = 3e-4
weight_decay = 0.1
steps = 200000
warmup_steps = 2000

[checkpoint]
enable = true
interval = 5000

[logging]
log_interval = 10
enable_tensorboard = true
```

---

## Migration Guide

**Moving from Phase 1-2 to Phase 3?**

1. **For simple training**: Continue using `train.py` with `high_vram.toml` / `low_vram.toml`

2. **For automated research**:
   - Use `arbiter_long_pipeline.py` instead of `train.py`
   - Create custom configs in `configs/sweep_templates/`
   - Leverage parameter sweeps via `arbiter_sweep_orchestrator.py`

3. **For quick experiments**:
   - Use `configs/sweep_templates/grokking_regime.toml` as a starting point
   - Modify `weight_decay`, `steps`, `n_layers` for specific hypotheses

---

## Related Documentation

- **[Implementation Plan](../../../brain/.../implementation_plan.md)**: Full automation architecture
- **[Main README](../../README.md)**: Project overview
- **[Research Foundation](../../docs/research/Architecting_Emergent_Reasoning_in_Data-Constrained_Regimes.md)**: Deep & Narrow topology theory

---

**Last Updated**: 2026-01-28  
**Status**: Legacy configurations maintained for backward compatibility
