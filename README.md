# Genesis: Bible-Trained Language Model for Causal Reasoning

Genesis is a research prototype exploring whether large language models trained exclusively on biblical text can develop logical reasoning capabilities through **semantic coherence** rather than **dataset scale**.

## 🎯 Project Mission

Train specialized LLMs on the Bible alone to investigate:
- Whether deep compression of coherent data enables reasoning without internet-scale corpora
- How architectural choices (depth vs. width) affect logical abstraction
- If special initialization of high-frequency tokens improves semantic structure learning

---

## 📁 Project Structure

```
Genesis_prototype/
├── run.py                      # 🎮 Central menu system (START HERE!)
├── README.md                   # Project overview
├── CONTRIBUTING.md             # Contribution guidelines
├── engine/                     # Core training and inference system
│   ├── models/                 # Model architectures (Llama implementation)
│   ├── datasets/               # Data loading and preprocessing
│   ├── components/             # Checkpointing, optimization utilities
│   ├── train_configs/          # Hardware-agnostic TOML configurations
│   │   ├── high_vram.toml      # 12+ GB VRAM (Performance mode)
│   │   └── low_vram.toml       # 4-6 GB VRAM (Compatibility mode)
│   ├── train.py                # Main training script
│   ├── nwt_corpus.txt          # New World Translation corpus (1M tokens)
│   └── genesis_tokenizer.json  # BPE tokenizer with Jehovah token
├── docs/
│   ├── research/               # Technical and research documentation
│   │   ├── theoretical_foundations.md
│   │   ├── dynamic_masking_assessment.md
│   │   └── logical_refinement_strategies.md
│   ├── reference/              # Quick reference and walkthroughs
│   │   ├── QUICK_REFERENCE.md
│   │   └── walkthrough1.md
│   └── roadmap/                # Implementation planning
│       └── README.md
├── scripts/                    # Utility and analysis scripts
│   ├── count_unique_words.py
│   ├── count_logical_connectives.py
│   ├── train_tokenizer.py
│   ├── arbiter_perplexity.py
│   └── friction_stress_test.py
└── checkpoints/                # 💾 Model snapshots (Git-ignored)
```

---

## 🚀 Quick Start

### Recommended: Central Menu System

Run the unified interface from the project root:
```powershell
python run.py
```

The interactive menu provides:
- **[1] Train Model** - Interactive training with hardware tier selection
- **[2] Corpus Analysis** - Word counts and logical connective analysis
- **[3] Evaluation & Testing** - Perplexity calculation and stress tests
- **[4] Documentation** - Open reference guides in your default viewer
- **[5] Project Information** - View statistics and project status

### Advanced: Direct Training

For developers who want direct access:
```powershell
cd engine
python train.py
```

Interactive prompts will guide you through:
1. **Hardware Tier**: High VRAM (12+ GB) or Low VRAM (4-6 GB)
2. **Protocol Selection**: Microscope, Tower of Truth, or High-Res Arbiter

### Distributed Training
```powershell
cd engine
$env:USE_LIBUV=0
torchrun --nproc_per_node=2 train.py
```

---

## 🏗️ Model Architectures

| Protocol | Dimensions | Layers | Parameters | VRAM (FP16/BF16) | Purpose | Learning Rate |
|----------|-----------|--------|-----------|------------------|---------|---------------|
| **Microscope** | 768 | 12 | 125.5M | ~2-3 GB | Baseline comparisons | 3e-4 |
| **Tower of Truth** | 144 | 144 | ~5-8M | ~1-2 GB | Deep logical abstraction | 1e-4 |
| **High-Res Arbiter** | 1024 | 24 | ~180M | ~3-4 GB | Maximum semantic resolution | 2e-4 |

### Key Features
- **Jehovah Token**: Special initialization for this high-frequency term (~7,000 occurrences)
- **Multi-mode training**: Dynamic configuration injection based on selected architecture
- **Hardware-agnostic configs**: High VRAM (12+ GB) and Low VRAM (4-6 GB) tiers
- **Optimized training**: FSDP/DDP for distributed setups or single-node fallback

---

## � Hardware Requirements

### High VRAM Tier (12+ GB)
**Compatible GPUs**: RTX 3080, RTX 4070, RTX 4080, RTX 4090, A5000, A6000, etc.

**Configuration** (`train_configs/high_vram.toml`):
- Batch Size: 4
- Gradient Accumulation: 4
- Precision: BF16
- Flash Attention: Enabled

### Low VRAM Tier (4-6 GB)
**Compatible GPUs**: GTX 1660, RTX 3050, Quadro T2000, RTX 2060, etc.

**Configuration** (`train_configs/low_vram.toml`):
- Batch Size: 1
- Gradient Accumulation: 32
- Precision: FP16
- Gradient Checkpointing: Enabled

---

## �📚 Documentation

### New to the Project?
Start here: **[Quick Reference Guide](docs/reference/QUICK_REFERENCE.md)**

### Understanding the Vision
- **[Theoretical Foundations](docs/research/theoretical_foundations.md)**: Why train on scripture alone?
- **[Research Roadmap](docs/roadmap/README.md)**: Phased implementation plan

### Technical Deep Dives
- **[Dynamic Masking Assessment](docs/research/dynamic_masking_assessment.md)**: Training logical reasoning through connective prediction
- **[Logical Refinement Strategies](docs/research/logical_refinement_strategies.md)**: Prioritized training techniques

### Implementation Details
- **[Walkthrough 1](docs/reference/walkthrough1.md)**: Multi-mode architecture verification

---

## 🧪 Current Status

| Component | Status |
|-----------|--------|
| Core Architecture | ✅ Complete |
| Multi-Mode System | ✅ Complete |
| Jehovah Token Initialization | ✅ Complete |
| Training Infrastructure | ✅ Complete |
| Central Menu System | ✅ Complete |
| Hardware-Agnostic Configs | ✅ Complete |
| GitHub Organization | ✅ Complete |
| Logical Refinement | 🔄 In Progress (Phase 2) |
| Evaluation Framework | 🔴 Not Started |

**Next Milestone**: Implement weighted masking on logical connectives (see [roadmap](docs/roadmap/README.md))

---

## 🔬 Research Questions

### Core Hypotheses
1. **Depth > Width**: Does a 144-layer model outperform 12-layer on logical reasoning tasks?
2. **Coherence > Scale**: Can a 1M-token coherent corpus match billion-token models within its domain?
3. **Transfer Learning**: Do biblical reasoning patterns generalize to modern ethical dilemmas?

### Practical Applications
4. **Reasoning Engine Seed Dataset**: Can the Bible serve as an effective foundation for developing lightweight reasoning models that excel at:
   - Logical inference and deduction
   - Analogical reasoning across domains
   - Ethical and moral reasoning frameworks

5. **Dataset Curation Arbiter**: Can Bible-trained models act as quality evaluators for:
   - Curating existing datasets by identifying logical coherence
   - Filtering noisy or contradictory training data
   - Scoring reasoning quality in pre-training corpora

6. **Synthetic Data Validator**: Can these models automate the evaluation of synthetically generated datasets from larger models:
   - Assessing logical consistency in LLM-generated training data
   - Identifying high-quality reasoning examples for distillation
   - Creating feedback loops for iterative synthetic data refinement

---

## 📊 Corpus Statistics

- **Source**: New World Translation
- **Total Tokens**: 1,004,926
- **Books**: 66 (39 Hebrew Scriptures, 27 Christian Greek)
- **Logical Connectives**: 21,304 (~2.1% of corpus)
- **Genres**: Law, Poetry, Prophecy, Narrative, Wisdom, Epistles

---

## 🤝 Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for detailed guidelines on:
- Research methodology and experimental protocols
- Code standards and documentation requirements
- PR submission process and review criteria
- How to propose new research directions

**Key areas for extension:**
1. Alternative corpora (Qur'an, Analects, Principia Mathematica)
2. Evaluation benchmarks for analogical reasoning
3. Interpretability tooling (visualizing concept emergence across layers)
4. Multi-worldview deliberation systems
5. Logical refinement techniques (weighted masking, gradient focusing)

---

## ⚠️ Disclaimer

This is **AI research**, not an attempt to create "Christian AI." The Bible was chosen for its:
- Internal logical consistency
- Historical influence on Western reasoning
- Rich genre diversity within a unified framework
- Manageable corpus size (~1M tokens)

---

## 📜 License

*[To be determined]*

---

**Last Updated**: 2026-01-28  
**Current Phase**: Phase 2 (Logical Refinement) - In Progress

---

## Quick Links

- 🎮 **[Central Menu](run.py)** - Unified interface for all functionality
- 🚀 **[Training Engine](engine/)** - Core training and inference system
- 📖 **[Quick Reference](docs/reference/QUICK_REFERENCE.md)** - Project overview
- 🔬 **[Research Docs](docs/research/)** - Technical deep dives
- 🗺️ **[Roadmap](docs/roadmap/README.md)** - Implementation plan
- 🤝 **[Contributing](CONTRIBUTING.md)** - Contribution guidelines
