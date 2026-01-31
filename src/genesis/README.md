# Genesis Engine

This folder contains the core training and inference system for the Genesis Arbiter project, now featuring **FlashAttention**, **multi-task learning**, and **grokking detection**.

---

## 📂 Contents

### Training Scripts

- **`train.py`** - Legacy training script (preserved for backward compatibility)
- **`train_composer.py`** - ⚡ **NEW**: Composer-based training with FlashAttention (Phase 1)
- **`train_multi_task.py`** - ⚡ **NEW**: Full multi-task training with grokking detection (Phase 2-3)
- **`verify_phase1.py`** - Verification script for FlashAttention integration

### Core Data
- **`nwt_corpus.txt`**: New World Translation corpus (1,004,926 tokens)
- **`genesis_tokenizer.json`**: BPE tokenizer with specialized "Jehovah" token

### Directories

#### `models/` - Model Architectures
- `llama/model.py` - Micro-Llama with RoPE, SwiGLU, DeepNorm, **FlashAttention via SDPA**
- `multi_task_wrapper.py` - ⚡ **NEW**: Multi-task wrapper with 4 task heads
- `tokenizer.py` - GenesisTokenizer wrapper

#### `datasets/` - Data Loading
- `bible.py` - BibleDataset with blocked loading
- `multi_task_sampler.py` - ⚡ **NEW**: Multi-task sampling from 142 Bible translations

#### `training/` - ⚡ **NEW** Training Infrastructure
- `flash_attention_config.py` - FlashAttention utilities and benchmarking
- `callbacks/grokking.py` - Grokking detection, Procrustes alignment, concept clustering

#### `components/` - Training Utilities
- `checkpoint.py` - Model saving/loading
- `optimizer.py` - AdamW configuration

#### `train_configs/` - Hardware Configurations
- `local_4070.toml` - RTX 4070 (12GB VRAM)
- `t2000_low_vram.toml` - Quadro T2000 (4GB VRAM)

---

## 🚀 Quick Start

### Option 1: Legacy Training (Backward Compatible)
```powershell
python train.py
```

### Option 2: FlashAttention Training (3-4x Faster)
```powershell
python train_composer.py --mode deep_narrow_40 --steps 10000
```

### Option 3: Multi-Task Training with Grokking Detection
```powershell
python train_multi_task.py \
    --mode deep_narrow_40 \
    --steps 20000 \
    --weight-decay 0.15 \
    --detect-grokking \
    --bible-dir ../../Bible
```

---

## ⚡ New Features (Jan 2026)

### FlashAttention Integration (Phase 1)
- **3-4x training speedup** on character-level sequences
- Automatic backend selection via PyTorch SDPA
- Zero code changes to existing checkpoints
- Reduced memory usage for long sequences

**Verification:**
```powershell
python verify_phase1.py
```

### Multi-Task Learning (Phase 2)
Four task heads trained simultaneously:
1. **Language Modeling (70%)** - Causal next-token prediction
2. **Coherence Detection (15%)** - Binary verse coherence classification
3. **Cross-Reference Prediction (7.5%)** - Triplet loss for semantic similarity
4. **Cross-Lingual Paraphrase (7.5%)** - Language-invariant embeddings

**Benefits:**
- More robust representations
- Leverages 142 Bible translations
- <10% parameter overhead

### Grokking Detection (Phase 3)
Automated detection of phase transitions from memorization → generalization:
- Monitors validation loss over 1000-step windows
- Detects >10% improvement in <500 steps
- Auto-saves checkpoints on grokking events
- Cross-lingual alignment tracking (Procrustes distance)
- Theological concept clustering analysis

**Documentation:** See `docs/research/grokking_detection_methodology.md`

---

## 🏗️ Model Architectures

### Phase 3: Deep & Narrow (Current Focus)

| Architecture | Layers | Dim | Heads | Params | Use Case |
|-------------|--------|-----|-------|--------|----------|
| **Deep Narrow 32** | 32 | 640 | 10 | 550M | Budget quick experiments |
| **Deep Narrow 40** | 40 | 768 | 12 | 800M | Mid-range deep model |
| **Deep Narrow 48** | 48 | 896 | 14 | 1.0B | 1B sweet spot for grokking |
| **Deep Narrow 60** | 60 | 768 | 12 | 1.2B | Lighter deep architecture |
| **Theos-Small** | 80 | 1024 | 16 | 1.8B | Grokking experiments |
| **Deep Narrow 100** | 100 | 1024 | 16 | 2.3B | Extreme depth reasoning |

All use **DeepNorm** for training stability and **FlashAttention** for efficiency.

### Legacy Architectures (Phase 1-2)

| Architecture | Layers | Dim | Heads | Params | Use Case |
|-------------|--------|-----|-------|--------|----------|
| **Microscope** | 12 | 768 | 12 | 125M | Baseline comparisons |
| **Tower of Truth** | 144 | 288 | 12 | 5-8M | Extreme depth (legacy) |
| **High-Res Arbiter** | 24 | 1024 | 16 | 180M | Semantic resolution (legacy) |

---

## 🔧 Configuration

### Hardware Requirements

| GPU | VRAM | Recommended Model | Batch Size | Notes |
|-----|------|------------------|------------|-------|
| RTX 4070 | 12GB | Deep Narrow 40 | 4 | Optimal for development |
| RTX 4090 | 24GB | Deep Narrow 100 | 8 | Full scale training |
| Quadro T2000 | 4GB | Microscope | 1 | Gradient checkpointing required |

### Task Distribution (Multi-Task Training)

Default distribution:
```python
## 📖 Related Documentation

- **[Quick Reference](../docs/reference/QUICK_REFERENCE.md)**: Project overview
- **[Theoretical Foundations](../docs/research/theoretical_foundations.md)**: Research motivation
- **[Implementation Roadmap](../docs/roadmap/README.md)**: Phased development plan

---

**Last Updated**: 2026-01-28
