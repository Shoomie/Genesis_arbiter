# Genesis Engine

This folder contains the core training and inference system for the Genesis prototype.

## 📂 Contents

### Core Files
- **`train.py`**: Main training script with interactive hardware/protocol selection
- **`nwt_corpus.txt`**: New World Translation biblical corpus (1,004,926 tokens)
- **`genesis_tokenizer.json`**: BPE tokenizer with specialized "Jehovah" token

### Directories
- **`models/`**: Model architectures
  - `llama/`: Micro-Llama implementation with RoPE, SwiGLU, RMSNorm
  - `tokenizer.py`: GenesisTokenizer wrapper
  
- **`datasets/`**: Data loading and preprocessing
  - `bible.py`: BibleDataset with blocked loading and verse boundaries
  
- **`components/`**: Training utilities
  - `checkpoint.py`: Model saving/loading
  - `optimizer.py`: AdamW configuration
  
- **`train_configs/`**: Hardware-specific TOML configurations
  - `local_4070.toml`: RTX 4070 (12GB VRAM)
  - `t2000_low_vram.toml`: Quadro T2000 (4GB VRAM)

---

## 🚀 Usage

### Basic Training
```powershell
python train.py
```

Interactive menus will prompt for:
1. Hardware configuration
2. Model architecture (Microscope, Tower of Truth, High-Res Arbiter)

### Distributed Training
```powershell
$env:USE_LIBUV=0
torchrun --nproc_per_node=2 train.py
```

### Custom Configuration
```powershell
python train.py --config train_configs/local_4070.toml --mode microscope
```

---

## 🏗️ Model Architectures

### Microscope (Baseline)
- **Dimensions**: 768
- **Layers**: 12
- **Parameters**: 125,553,408
- **Use Case**: Rapid prototyping and baseline comparisons

### Tower of Truth (Deep Reasoning)
- **Dimensions**: 144
- **Layers**: 144 (12² for apostolic resonance)
- **Parameters**: ~3-5M
- **Use Case**: Testing depth-based logical abstraction

### High-Res Arbiter (Semantic Resolution)
- **Dimensions**: 1024
- **Layers**: 24
- **Use Case**: Maximum embedding bandwidth for judgment tasks

---

## 🔧 Configuration Notes

### Hardware Requirements
- **Minimum**: 4GB VRAM (Quadro T2000 with gradient checkpointing)
- **Recommended**: 12GB VRAM (RTX 4070)
- **Optimal**: 24GB+ VRAM (RTX 4090, A100)

### Key Hyperparameters
```toml
[training]
batch_size = 4              # Per-device
learning_rate = 3e-4        # Microscope default
max_seq_len = 512           # Context window
gradient_accumulation = 4   # Effective batch = 16
```

### Logos Initialization
The "Jehovah" token (ID: 5) receives specialized initialization:
- **Current setting**: 1.0x variance (baseline)
- **Configurable** in `models/llama/model.py::logos_init_hook()`
- ~7,000 corpus occurrences provide natural semantic weight

---

## 📊 Training Data

The `nwt_corpus.txt` contains the complete New World Translation:
- **Format**: Plain text with verse markers
- **Preprocessing**: Tokenized on-the-fly by BibleDataset
- **Blocking**: 512-token chunks with overlap for context continuity

---

## 🔍 File Paths

**Important**: Update script import paths if moving files. Key dependencies:

```python
# In train.py
from models.llama.model import Llama, logos_init_hook
from models.tokenizer import GenesisTokenizer
from datasets.bible import get_bible_dataloader
from components.checkpoint import save_checkpoint
```

---

## 🧪 Development Notes

### Adding New Architectures
1. Define mode in `train.py::get_logos_config()`
2. Add configuration section with hyperparameters
3. Update interactive menu

### Modifying Dataset
1. Edit `datasets/bible.py::BibleDataset`
2. Implement custom preprocessing in `__getitem__()`
3. Update dataloader in `get_bible_dataloader()`

### Custom Training Techniques
See [Logical Refinement Strategies](../docs/research/logical_refinement_strategies.md) for prioritized techniques like:
- Dynamic masking on logical connectives
- Logos bridge supervision
- End-of-verse penalty weighting

---

## 🐛 Troubleshooting

### CUDA Out of Memory
1. Use `t2000_low_vram.toml` configuration
2. Reduce batch size
3. Enable gradient checkpointing
4. Reduce `max_seq_len`

### DistStoreError on Windows
```powershell
$env:USE_LIBUV=0  # Disable libuv before torchrun
```

### Slow Training
1. Verify FlashAttention is installed
2. Use `torch.compile` (requires Triton)
3. Ensure CUDA version matches PyTorch build

---

## 📖 Related Documentation

- **[Quick Reference](../docs/reference/QUICK_REFERENCE.md)**: Project overview
- **[Theoretical Foundations](../docs/research/theoretical_foundations.md)**: Research motivation
- **[Implementation Roadmap](../docs/roadmap/README.md)**: Phased development plan

---

**Last Updated**: 2026-01-28
