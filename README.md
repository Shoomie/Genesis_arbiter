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
-  **Multi-task learning** across 100+ Bible translations (coherence, cross-reference, paraphrase detection)
- 🎯 **Automated grokking detection** with cross-lingual alignment monitoring
- 🔬 **Theological concept clustering** to measure emergent semantic structure

**Quick Start:**
```powershell
python run.py
```

**Foundation:** Based on [research](docs/research/theoretical_foundations.md) exploring whether semantic density and internal consistency can substitute for dataset scale in developing logical reasoning.

---

## 🎯 Project Mission

**Primary Objective**: Train transformers exclusively on the Bible to demonstrate that:
- **Depth substitutes for volume**: Models can develop reasoning without trillion-token datasets
- **Extended training induces phase transitions**: Grokking enables generalization beyond memorization
- **Training integrated learning of complete semantics maximizes signal**: Character-level tokenizers and multi-task objectives extract latent structure

---

**Documentation**: See [`docs/PHASE1_SETUP.md`](docs/PHASE1_SETUP.md) for installation and [`src/genesis/README.md`](src/genesis/README.md) for usage details.

---

## 📁 Project Structure

```text
Genesis_arbiter/
├── run.py                          # 🎮 Central menu system (START HERE!)
├── genesis_config.toml             # ⚙️ Central configuration
├── README.md                       # Project overview
├── data/                           # 📂 Data assets (Tokenizers, Caches)
├── src/                            # 🏗️ Source code (Genesis package)
├── tools/                          # 🛠️ Utility scripts & analysis tools
├── project_doc/                    # 📄 Core project documentation (Legal, Contribution)
├── docs/                           # 📖 Research papers & technical guides
└── checkpoints/                    # 💾 Model snapshots (Git-ignored)
```

---

## 🚀 Quick Start

### Central Menu System (Recommended)
```powershell
python run.py
```

All core parameters are managed in **`genesis_config.toml`**. To adjust training or interaction settings, edit that file and relaunch the script.

### Configuration
The project uses a unified configuration system:
- **`[training]`**: Control learning rates, batch sizes, and model modes.
- **`[interaction]`**: Adjust temperature and generation limits for model chatting.

---

## 📚 Documentation

### Core Reading
- **[Quick Reference](docs/reference/QUICK_REFERENCE.md)** - Project overview
- **[Theoretical Foundations](docs/research/theoretical_foundations.md)** - Why train on scripture alone?
- **[Grokking Detection Methodology](docs/research/grokking_detection_methodology.md)** - Phase transition detection & validation

### Project Resources
- **[Contributing](project_doc/CONTRIBUTING.md)** - Guidelines for research and code
- **[Setup Guide](docs/PHASE1_SETUP.md)** - Installation and verification
- **[Research Papers](docs/research/)** - Full technical analysis

---

## 🙏 Acknowledgments

This research utilizes the **New World Translation of the Holy Scriptures** published by the **Watch Tower Bible & Tract Society**. We extend sincere gratitude for their exceptional translation work.

**See Also**: [project_doc/ACKNOWLEDGMENTS.md](project_doc/ACKNOWLEDGMENTS.md) for complete formal acknowledgment.

---

## 📜 Open Source Commitment

**This is an open source research project.** Information should be free, and our work is freely available for research and production use.

**License**: MIT License (allows commercial derivative works while keeping codebase open)

---

**Last Updated**: 2026-01-31  
**Current Framework**: Phase 4 Complete (Codebase Professionalization & Centralized Configuration)

---

## Quick Links

- 🎮 **[Central Menu](run.py)** - Unified interface
- ⚙️ **[Config](genesis_config.toml)** - Central settings
- 📖 **[Quick Reference](docs/reference/QUICK_REFERENCE.md)** - Overview
- 🔬 **[Research Docs](docs/research/)** - Technical papers
- 🤝 **[Contributing](project_doc/CONTRIBUTING.md)** - Guidelines
- 🙏 **[Acknowledgments](project_doc/ACKNOWLEDGMENTS.md)** - Attribution
