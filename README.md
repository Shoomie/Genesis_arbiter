# Genesis Arbiter: Deep Reasoning in Data-Constrained Regimes

Genesis Arbiter is an advanced research platform investigating **emergent reasoning in small language models** through corpus-specific optimization, extreme architectural depth, and extended training regimes that induce "grokking" phase transitions.

## 🎯 Project Mission

**Primary Objective**: Train 2B-parameter "Deep & Narrow" transformers exclusively on the Bible (~1M tokens) to demonstrate that:
- **Depth substitutes for volume**: 80-144 layer models can develop reasoning without trillion-token datasets
- **Extended training induces phase transitions**: Grokking enables generalization beyond memorization
- **Corpus-specific optimization maximizes signal**: Custom tokenizers and data augmentation extract latent structure

**Research Foundation**: Based on ["Architecting Emergent Reasoning in Data-Constrained Regimes"](docs/research/Architecting_Emergent_Reasoning_in_Data-Constrained_Regimes.md)

---

## 📁 Project Structure

```
Genesis_arbiter/
├── run.py                          # 🎮 Central menu system (START HERE!)
├── README.md                       # Project overview
├── CONTRIBUTING.md                 # Contribution guidelines
│
├── engine/                         # Core training, evaluation & automation
│   ├── models/                     # DeepNorm transformer architectures
│   ├── datasets/                   # Corpus loading and preprocessing
│   ├── components/                 # Checkpointing, optimization
│   │
│   ├── train.py                    # Interactive training system
│   ├── nwt_corpus.txt              # NWT corpus (1M tokens, 4.3 MB)
│   │
│   ├── train_configs/              # Legacy hardware-tier configs
│   │   ├── high_vram.toml
│   │   └── low_vram.toml
│   │
│   └── 🤖 ARBITER AUTOMATION (NEW - Phase 3)
│       ├── arbiter_logger.py           # SQLite + TensorBoard logging
│       ├── arbiter_quick_eval.py       # Fast (<1h) checkpoint evaluation
│       ├── arbiter_sweep_orchestrator.py  # Distributed parameter sweeps
│       └── arbiter_long_pipeline.py    # End-to-end training automation
│
├── scripts/                        # Utilities & data generation
│   ├── arbiter_tokenizer_factory.py   # Multi-vocab SentencePiece training
│   ├── arbiter_data_augmentor.py      # Synthetic reasoning traces
│   ├── arbiter_perplexity.py          # Legacy perplexity calculator
│   ├── friction_stress_test.py        # Adversarial evaluation
│   └── [see scripts/README.md for full listing]
│
├── configs/
│   └── sweep_templates/            # Parameter sweep configurations
│       ├── base_config.toml
│       ├── deep_narrow_template.toml
│       ├── grokking_regime.toml    # Extreme weight decay config
│       └── baseline_standard.toml
│
├── docs/
│   ├── research/                   # Technical papers
│   │   ├── Architecting_Emergent_Reasoning_in_Data-Constrained_Regimes.md
│   │   ├── theoretical_foundations.md
│   │   └── dynamic_masking_assessment.md
│   ├── reference/                  # Quick guides
│   └── roadmap/                    # Implementation plans
│
├── logs/                           # Auto-created by arbiter_logger
│   ├── experiments.db              # SQLite experiment database
│   └── tensorboard/                # TensorBoard logs
│
└── checkpoints/                    # Model snapshots (Git-ignored)
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
- **[5] Arbiter Automation** - Quick eval, long pipelines, parameter sweeps, data augmentation
- **[6] Project Information** - View statistics and project status

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

### Deep & Narrow Topologies (Phase 3)

| Configuration | Dim | Layers | Heads | Params | Purpose | Weight Decay |
|--------------|-----|--------|-------|--------|---------|-------------|
| **Theos-Small** | 1024 | 80 | 16 | ~1.8B | Grokking experiments | 0.1-0.2 |
| **Deep Narrow (Variable)** | 768-1280 | 60-100 | 12-20 | 1-3B | Parameter sweeps | Configurable |
| **Baseline Wide** | 2048 | 32 | 16 | ~2B | Comparison benchmark | 0.01 |

### Legacy Architectures (Phase 1-2)

| Protocol | Dim | Layers | Params | Purpose |
|----------|-----|--------|--------|----------|
| Microscope | 768 | 12 | 125M | Baseline |
| Tower of Truth | 144 | 144 | ~8M | Extreme depth experiment |
| High-Res Arbiter | 1024 | 24 | ~180M | Semantic resolution |

### Key Features
- **DeepNorm Stabilization**: Enables training of 80-144 layer networks
- **Custom Tokenizers**: SentencePiece BPE with corpus-specific MWE extraction
- **Grokking Detection**: Automatic monitoring for validation loss phase transitions
- **Distributed Training**: FSDP for 2B+ parameter models across multiple GPUs

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
| **Phase 1: Core Architecture** | ✅ Complete |
| Core Llama Implementation | ✅ Complete |
| Multi-Mode Training System | ✅ Complete |
| Hardware-Agnostic Configs | ✅ Complete |
| **Phase 2: Logical Refinement** | ✅ Complete |
| Jehovah Token Initialization | ✅ Complete |
| Dynamic Masking Research | ✅ Complete |
| **Phase 3: Arbiter Automation** | ✅ Complete (~2,850 LOC) |
| SQLite + TensorBoard Logger | ✅ Complete |
| Multi-Vocab Tokenizer Factory | ✅ Complete |
| Quick Evaluation Suite (<1h) | ✅ Complete |
| Data Augmentation System | ✅ Complete |
| Parameter Sweep Orchestrator | ✅ Complete |
| Long-Form Training Pipeline | ✅ Complete |
| Configuration Templates | ✅ Complete (4 TOML files) |
| **Phase 4: Visualization & Analysis** | 🔴 Not Started |
| Live Dashboard (Streamlit) | 🔴 Planned |
| Full Theological Benchmark | 🔴 Planned |
| Circuit Extraction Tools | 🔴 Planned |

**Current Milestone**: Phase 3 complete. Ready for large-scale grokking experiments.

**Next Phase**: Implement live monitoring dashboard and comprehensive evaluation benchmarks.

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

## 🙏 Acknowledgments

### Primary Corpus

This research utilizes the **New World Translation of the Holy Scriptures** published by the **Watch Tower Bible & Tract Society**. We extend sincere gratitude and appreciation to the Watch Tower organization and its devoted community of translators, scholars, and volunteers for their exceptional work.

**Why We Chose the NWT:**
- **Translation Excellence**: Decades of meticulous scholarship by biblical language experts
- **Linguistic Consistency**: Careful, uniform rendering of key terms throughout the text
- **Divine Name Restoration**: Principled use of "Jehovah" (7,000+ occurrences) provides a valuable lexical anchor for semantic analysis
- **Modern Clarity**: Contemporary English that maintains fidelity to source languages
- **Scholarly Rigor**: Multiple revisions (1950, 1961, 1984, 2013) demonstrating commitment to accuracy

The NWT's interpretational coherence and scholarly foundation make it an ideal corpus for investigating whether AI models can develop reasoning from internally consistent textual frameworks.

**Research Ethics**: We acknowledge that the NWT was created for religious education and spiritual instruction. Our computational research use represents a transformative application undertaken with deep respect for the spiritual significance of these texts and the community they serve. The NWT was selected for its exceptional methodological qualities: translation consistency, scholarly rigor, and interpretational coherence.

**See Also**: [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for our complete formal acknowledgment and commitment to responsible use.

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

## ⚠️ Research Scope

This is **computational linguistics and AI research** investigating how language models learn from coherent textual corpora. The Bible was chosen as the primary corpus for its:
- Internal logical consistency and unified interpretational framework
- Historical influence on Western reasoning and philosophical thought
- Rich genre diversity (law, poetry, narrative, prophecy, wisdom, epistles)
- Manageable corpus size (~1M tokens) enabling deep analysis

---

## 📜 Open Source Commitment

**This is an open source research project.** We believe that information should be free, and our work is made freely available for both research and production use.

### Usage Terms

**Research & Production Use**: This work may be freely used, studied, modified, and built upon for both academic research and production applications.

**Derivative & Commercial Works**: You may create derivative works and use this research in commercial applications (similar to how large language models are often trained on diverse corpora including copyrighted works). However, the research itself and its direct outputs **may not be directly commercialized or sold as-is**.

**Our Philosophy**: We believe AI should be:
- **Efficient**: Optimized architectures that maximize reasoning with minimal energy use
- **Truthful**: Grounded in coherent, well-structured knowledge frameworks  
- **Free**: Openly available to advance collective understanding

### Attribution

We kindly request that derivative works:
- Acknowledge the Genesis project and its contributors
- Credit the Watch Tower Bible & Tract Society for the New World Translation corpus
- Share improvements and findings with the research community when possible

**License**: MIT License (allows commercial derivative works while keeping the codebase open)

*For questions about usage or collaboration, please open an issue on our repository.*

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
- 🙏 **[Acknowledgments](ACKNOWLEDGMENTS.md)** - Corpus attribution and thanks
