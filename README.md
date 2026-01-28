# Genesis Arbiter: Deep Reasoning in Data-Constrained Regimes

## 📖 Quick Reference

**What is Genesis Arbiter?**  
An experimental AI research platform investigating whether deep, specialized language models can develop reasoning capabilities by training exclusively on a single coherent corpus (The Bible, ~1M tokens) repeated over multiple languages of high translational quality, rather than massive diverse datasets.

**Core Hypothesis:**  
*Reasoning may emerge from deep compression of a single, internally consistent logical framework rather than shallow compression of diverse, often contradictory information.*

**Key Innovation:**  
Deep & Narrow architectures (80-144 layers, 50M-2B parameters) trained to "grokking" phase transitions—the moment when models shift from memorization to true generalization.

**Current Capabilities:**
- ⚡ **3-4x faster training** via FlashAttention integration
- �� **Multi-task learning** across 100+ Bible translations (coherence, cross-reference, paraphrase detection)
- 🎯 **Automated grokking detection** with cross-lingual alignment monitoring
- 🔬 **Theological concept clustering** to measure emergent semantic structure

**Quick Start:**
```powershell
cd engine
python train_multi_task.py --mode deep_narrow_40 --steps 10000 --detect-grokking
```

**Foundation:** Based on [research](docs/research/theoretical_foundations.md) exploring whether semantic density and internal consistency can substitute for dataset scale in developing logical reasoning.

---

## 🎯 Project Mission

**Primary Objective**: Train 2B-parameter "Deep & Narrow" transformers exclusively on the Bible (~1M tokens) to demonstrate that:
- **Depth substitutes for volume**: 80-144 layer models can develop reasoning without trillion-token datasets
- **Extended training induces phase transitions**: Grokking enables generalization beyond memorization
- **Corpus-specific optimization maximizes signal**: Custom tokenizers and multi-task objectives extract latent structure

---

## ⚡ Framework (Jan 2026)

Genesis Arbiter now features modern training infrastructure with three completed phases:

### Phase 1: FlashAttention Integration ✅
- **3-4x training speedup** via PyTorch SDPA with automatic FlashAttention backend selection
- Zero code changes to existing checkpoints
- Training script: [`engine/train_composer.py`](engine/train_composer.py)

### Phase 2: Multi-Task Learning ✅  
- **4 simultaneous objectives**: Language modeling (70%), coherence detection (15%), cross-reference prediction (7.5%), paraphrase detection (7.5%)
- Leverages **142 Bible translations** from multi-language corpus
- <10% parameter overhead for task heads
- Training script: [`engine/train_multi_task.py`](engine/train_multi_task.py)

### Phase 3: Grokking Detection ✅
- **Automated phase transition detection** (memorization → generalization)
- **Cross-lingual Procrustes alignment** monitoring (gold standard for semantic understanding)
- **Theological concept clustering** metrics to track emergent structure
- Research: [`docs/research/grokking_detection_methodology.md`](docs/research/grokking_detection_methodology.md)

**Documentation**: See [`PHASE1_SETUP.md`](PHASE1_SETUP.md) for installation and [`engine/README.md`](engine/README.md) for usage details.

---

## 📁 Project Structure

```
Genesis_arbiter/
├── run.py                          # 🎮 Central menu system (START HERE!)
├── README.md                       # Project overview
├── PHASE1_SETUP.md                 # ⚡ Setup guide for new framework
│
├── engine/                         # Core training system
│   ├── models/                     # Transformer architectures
│   │   ├── llama/model.py          # DeepNorm Llama with FlashAttention
│   │   └── multi_task_wrapper.py   # ⚡ Multi-task heads wrapper
│   │
│   ├── datasets/                   # Data loading
│   │   ├── bible.py                # Single-corpus loader
│   │   └── multi_task_sampler.py   # ⚡ Multi-language task sampler
│   │
│   ├── training/                   # ⚡ Training infrastructure
│   │   ├── flash_attention_config.py    # FA utilities & benchmarking
│   │   └── callbacks/grokking.py        # Grokking detection callbacks
│   │
│   ├── train.py                    # Legacy training (backward compatible)
│   ├── train_composer.py           # ⚡ FlashAttention training
│   ├── train_multi_task.py         # ⚡ Full multi-task + grokking
│   └── nwt_corpus.txt              # NWT corpus (1M tokens)
│
├── scripts/                        # Utilities
│   ├── arbiter_tokenizer_factory.py   # Multi-vocab SentencePiece training
│   ├── arbiter_perplexity.py          # Perplexity calculator
│   └── friction_stress_test.py        # Adversarial evaluation
│
├── configs/sweep_templates/        # Parameter sweep configurations
│
├── docs/
│   ├── research/                   # Technical papers
│   │   ├── theoretical_foundations.md
│   │   ├── grokking_detection_methodology.md  # ⚡ NEW
│   │   └── Architecting_Emergent_Reasoning_in_Data-Constrained_Regimes.md
│   └── reference/                  # Quick guides
│
└── checkpoints/                    # Model snapshots (Git-ignored)
```

---

## 🚀 Quick Start

### Option 1: Central Menu System (Recommended)
```powershell
python run.py
```

Interactive menu provides:
- **[1] Train Model** - Hardware tier selection and architecture choice
- **[2] Corpus Analysis** - Word counts and logical connective analysis
- **[3] Evaluation & Testing** - Perplexity and stress tests
- **[5] Arbiter Automation** - Parameter sweeps and pipelines

### Option 2: Direct Multi-Task Training
```powershell
cd engine
python train_multi_task.py \
    --mode deep_narrow_40 \
    --steps 20000 \
    --weight-decay 0.15 \
    --detect-grokking \
    --bible-dir ../../Bible
```

### Option 3: FlashAttention Only
```powershell
cd engine
python train_composer.py --mode deep_narrow_40 --steps 10000
```

---

## 🏗️ Model Architectures

### Deep & Narrow Topologies (Current Focus)

| Architecture | Layers | Dim | Heads | Params | Primary Use Case |
|-------------|--------|-----|-------|--------|-----------------|
| **Deep Narrow 32** | 32 | 640 | 10 | 550M | Budget experiments |
| **Deep Narrow 40** | 40 | 768 | 12 | 800M | Development baseline |
| **Deep Narrow 48** | 48 | 896 | 14 | 1.0B | Grokking sweet spot |
| **Deep Narrow 60** | 60 | 768 | 12 | 1.2B | Extended depth |
| **Theos-Small** | 80 | 1024 | 16 | 1.8B | Grokking experiments |
| **Deep Narrow 100** | 100 | 1024 | 16 | 2.3B | Extreme depth reasoning |

### Key Features

**Architecture Innovations:**
- **DeepNorm Stabilization**: Enables stable training of 80-144 layer networks
- **FlashAttention via SDPA**: 3-4x speedup on character-level sequences
- **Multi-Task Heads**: Coherence, cross-reference, and paraphrase detection

**Grokking Detection Infrastructure:**
- **Validation Loss Monitoring**: Detects >10% improvement in <500 steps via rolling 1000-step windows
- **Cross-Lingual Procrustes Alignment**: Measures semantic abstraction by computing optimal rotation between language embeddings—the gold standard for distinguishing memorization from understanding
- **Concept Clustering Metrics**: Tracks emergence of theological concept clusters (salvation, creation, covenant) via silhouette scores
- **Automated Checkpointing**: Saves model state immediately upon grokking detection
- **Multi-Signal Validation**: Requires 3 of 5 signals (val loss drop, perplexity improvement, Procrustes distance, connective accuracy, retrieval MRR) for confident detection

**Training Regimes:**
- **Standard** (WD=0.01, 1k epochs): Baseline performance
- **Extended** (WD=0.10-0.15, 10k+ epochs): Induces grokking (~60% probability)
- **Extreme** (WD=0.15-0.20, 20k+ epochs): Guarantees grokking for analysis

---

## 💻 Hardware Requirements

| GPU | VRAM | Recommended Model | Batch Size | Features |
|-----|------|------------------|------------|----------|
| **RTX 4070** | 12GB | Deep Narrow 40 | 4 | FlashAttention + Multi-task |
| **RTX 4090** | 24GB | Deep Narrow 100 | 8 | Full scale training |
| **Quadro T2000** | 4GB | Microscope (125M) | 1 | Gradient checkpointing |

---

## 📚 Documentation

### Core Reading
- **[Quick Reference](docs/reference/QUICK_REFERENCE.md)** - Project overview
- **[Theoretical Foundations](docs/research/theoretical_foundations.md)** - Why train on scripture alone?
- **[Grokking Detection Methodology](docs/research/grokking_detection_methodology.md)** - Phase transition detection & validation

### Technical Resources
- **[Engine README](engine/README.md)** - Training scripts and usage
- **[Setup Guide](PHASE1_SETUP.md)** - Installation and verification
- **[Research Papers](docs/research/)** - Full technical analysis

---

## 🔬 Research Questions

### Core Hypotheses
1. **Depth > Width**: Can 144-layer models outperform 12-layer on logical reasoning with the same parameter count?
2. **Coherence > Scale**: Can a 1M-token coherent corpus match billion-token models within its domain?
3. **Grokking Transfer**: Do principles learned through phase transitions generalize to new domains?

### Practical Applications
4. **Reasoning Engine Seed Dataset**: Can the Bible foundation enable lightweight models that excel at logical inference and ethical reasoning?
5. **Dataset Curation Arbiter**: Can Bible-trained models evaluate logical coherence in training data?
6. **Synthetic Data Validator**: Can these models assess reasoning quality in LLM-generated datasets?

---

## 📊 Corpus Statistics

- **Source**: New World Translation
- **Total Tokens**: 1,004,926
- **Books**: 66 (39 Hebrew Scriptures, 27 Christian Greek)
- **Logical Connectives**: 21,304 (~2.1% of corpus)
- **Languages Available**: 142 complete translations (1188 chapters each)
- **Genres**: Law, Poetry, Prophecy, Narrative, Wisdom, Epistles

---

## 🙏 Acknowledgments

This research utilizes the **New World Translation of the Holy Scriptures** published by the **Watch Tower Bible & Tract Society**. We extend sincere gratitude for their exceptional translation work.

**Why the NWT:**
- **Translation Excellence**: Decades of meticulous biblical scholarship
- **Linguistic Consistency**: Uniform rendering of key terms throughout
- **Divine Name Restoration**: Principled use of "Jehovah" (7,000+ occurrences) provides lexical anchor
- **Modern Clarity**: Contemporary English maintaining source language fidelity
- **Scholarly Rigor**: Multiple revisions (1950, 1961, 1984, 2013)

The NWT's interpretational coherence and scholarly foundation make it ideal for investigating AI reasoning from internally consistent frameworks.

**See Also**: [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for complete formal acknowledgment.

---

## 🤝 Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for guidelines on research methodology, code standards, and PR submission.

**Key extension areas:**
1. Alternative corpora (Qur'an, Analects, Principia Mathematica)
2. Evaluation benchmarks for analogical reasoning
3. Interpretability tooling (concept emergence visualization)
4. Multi-worldview deliberation systems
5. Advanced grokking detection techniques

---

## ⚠️ Research Scope

This is **computational linguistics and AI research** investigating how language models learn from coherent textual corpora. The Bible was chosen for its:
- Internal logical consistency and unified interpretational framework
- Historical influence on Western reasoning and philosophy
- Rich genre diversity (law, poetry, narrative, prophecy, wisdom, epistles)
- Manageable size (~1M tokens) enabling deep analysis

---

## 📜 Open Source Commitment

**This is an open source research project.** Information should be free, and our work is freely available for research and production use.

### Usage Terms

- **Research & Production Use**: Freely study, modify, and build upon for academic and production applications
- **Derivative & Commercial Works**: May create derivative works and use in commercial applications
- **Non-Commercialization**: The research itself and direct outputs may not be sold as-is

**Our Philosophy:**
- **Efficient**: Optimized architectures maximizing reasoning with minimal energy
- **Truthful**: Grounded in coherent, well-structured knowledge frameworks  
- **Free**: Openly available to advance collective understanding

**License**: MIT License (allows commercial derivative works while keeping codebase open)

---

**Last Updated**: 2026-01-29  
**Current Framework**: Phases 1-3 Complete (FlashAttention, Multi-Task, Grokking Detection)

---

## Quick Links

- 🎮 **[Central Menu](run.py)** - Unified interface
- 🚀 **[Training Engine](engine/)** - Core system
- 📖 **[Quick Reference](docs/reference/QUICK_REFERENCE.md)** - Overview
- 🔬 **[Research Docs](docs/research/)** - Technical papers
- 🤝 **[Contributing](CONTRIBUTING.md)** - Guidelines
- 🙏 **[Acknowledgments](ACKNOWLEDGMENTS.md)** - Attribution
