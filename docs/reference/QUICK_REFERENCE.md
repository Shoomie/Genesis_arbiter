# Genesis Prototype: Quick Reference Guide

## 🎯 Project Mission
Train a large language model exclusively on biblical text to investigate whether **semantic coherence** can substitute for **dataset scale** in developing logical reasoning capabilities.

---

## 📁 Project Structure

```
Genesis_prototype/
├── roadmap/                    # Research documentation
│   ├── README.md              # Phased implementation roadmap
│   ├── theoretical_foundations.md  # Why this project exists
│   └── walkthrough1.md        # Multi-mode verification results
├── models/
│   ├── llama/
│   │   └── model.py           # Micro-Llama architecture (124M params)
│   └── tokenizer.py           # GenesisTokenizer with Jehovah token
├── datasets/
│   └── bible.py               # BibleDataset with blocked loading
├── train_configs/
│   ├── local_4070.toml        # High-performance config (12GB VRAM)
│   └── t2000_low_vram.toml    # Compatibility config (4GB VRAM)
├── train.py                   # Main training script with interactive menus
├── nwt_corpus.txt             # New World Translation (1M tokens)
└── logical_refinement_strategies.md  # Prioritized training techniques
```

---

## 🚀 Quick Start

### Basic Training
```powershell
python train.py
```
Interactive menus will guide you through:
1. **Hardware Selection**: RTX 4070 or Quadro T2000
2. **Protocol Selection**: Microscope, Tower of Truth, or High-Res Arbiter

### Advanced: Distributed Training
```powershell
$env:USE_LIBUV=0
torchrun --nproc_per_node=2 train.py --config train_configs/local_4070.toml
```

---

## 🏗️ Model Architectures

### Microscope (Baseline)
- **Dimensions**: 768
- **Layers**: 12
- **Parameters**: 125.5M
- **Purpose**: Standard architecture for baseline comparisons
- **Learning Rate**: 3e-4

### Tower of Truth (Deep Reasoning)
- **Dimensions**: 144
- **Layers**: 144
- **Parameters**: ~3-5M
- **Purpose**: Test if extreme depth enables logical abstraction
- **Learning Rate**: 1e-4
- **Symbolic Meaning**: 144 = 12² (apostolic resonance)

### High-Res Arbiter (Semantic Resolution)
- **Dimensions**: 1024
- **Layers**: 24
- **Parameters**: TBD
- **Purpose**: Maximum semantic bandwidth for judgment tasks
- **Learning Rate**: 2e-4

---

## 🔑 Key Concepts

### Logos Initialization
Special weight initialization for the "Jehovah" token (ID: 5):
- **Current Setting**: 1.0x variance (baseline)
- **Rationale**: ~7,000 occurrences in text provide natural semantic weight
- **Configurable**: Can experiment with 0.5x, 2.0x, 5.0x multipliers

### Semantic Anchoring
The "Jehovah" token acts as a **semantic absolute**—a fixed reference point from which other theological concepts derive meaning.

### Typological Reasoning
Biblical interpretation through pattern recognition:
- Earlier events prefigure later fulfillments
- Example: Isaac's sacrifice → Christ's crucifixion
- **Hypothesis**: Models can learn analogical reasoning from these patterns

---

## 📊 Training Data

### Corpus Statistics
- **Source**: New World Translation
- **Total Tokens**: 1,004,926
- **Books**: 66 (39 Hebrew Scriptures, 27 Christian Greek Scriptures)
- **Genres**: Law, Poetry, Prophecy, Narrative, Wisdom, Epistles

### Data Distribution
High-density books (for future oversampling):
- **Romans**: Dense theological argumentation
- **Proverbs**: Axiomatic wisdom statements
- **Genesis**: Foundational narratives and typology
- **Isaiah**: Messianic prophecy and imagery

---

## 🧪 Experimental Techniques (Planned)

### Tier 1: Forcing Logical Computation
1. **Dynamic Masking**: Mask logical connectives ("Therefore", "Because") to force causal reasoning
2. **Logos Bridge Supervision**: Auxiliary prediction heads at layers 70-80
3. **Parallel Account Contrast**: Train on Kings vs. Chronicles simultaneously

### Tier 2: Structural Alignment
4. **End-of-Verse Penalty**: 1.2x loss weight at verse boundaries
5. **Axiomatic Oversampling**: Increase sampling rate for Romans, Proverbs
6. **Attention Regularization**: Penalize diffuse attention patterns

### Tier 3: Architectural Tuning
7. **RoPE Base**: Increase to 50,000 for long-range dependencies
8. **RMSNorm Epsilon**: Reduce to 1e-8 for numeric stability in 144-layer stack

---

## 🎯 Research Questions

### Primary Hypotheses
1. **Depth > Width**: 144-layer model outperforms 12-layer on logical tasks (despite fewer parameters)
2. **Coherence > Scale**: Bible-trained model matches general LLMs on scriptural reasoning
3. **Transfer Learning**: Principles learned from biblical law generalize to modern ethics

### Evaluation Metrics
- **Perplexity**: Standard metric (lower is better)
- **Typological Accuracy**: Can model identify valid type-antitype pairs?
- **Judgment Coherence**: Does model cite relevant scripture + balance justice/mercy?
- **Attention Flow**: Does "Jehovah" act as semantic hub in attention graphs?

---

## 📈 Current Status

| Component | Status |
|-----------|--------|
| Core Architecture | ✅ Complete |
| Multi-Mode System | ✅ Complete |
| Logos Initialization | ✅ Complete (1.0x variance) |
| Training Infrastructure | ✅ Complete |
| Logical Refinement | 🔄 In Progress |
| Evaluation Framework | 🔴 Not Started |
| Ablation Studies | 🔴 Not Started |

**Next Milestone**: Implement dynamic masking on logical connectives

---

## 🔬 Theoretical Framework

### Core Premise
```
Traditional LLM: Reasoning emerges from exposure to diverse data
Genesis Model:  Reasoning emerges from deep compression of coherent data
```

### Philosophical Position
The Bible represents:
- **Logical Coherence**: Unified theological framework across genres
- **Dense Semantics**: Every term carries theological weight
- **Historical Influence**: Shaped Western law, ethics, philosophy
- **Completeness**: Finite corpus allows full memorization + deep understanding

### Long-Term Vision
1. Prove specialized intelligence viable without internet-scale data
2. Template for worldview-specific AI (Confucian, Utilitarian, etc.)
3. Multi-system deliberation: Consult competing ethical frameworks

---

## 📚 Documentation Index

### For Quick Start
→ [Main README](../README.md)

### For Understanding "Why"
→ [Theoretical Foundations](theoretical_foundations.md)

### For Implementation Plan
→ [Research Roadmap](README.md)

### For Technical Details
→ [Walkthrough 1](walkthrough1.md)
→ [Logical Refinement Strategies](../logical_refinement_strategies.md)

---

## 🤝 Contributing Ideas

If extending this research, consider:

1. **Alternative Corpora**: Qur'an, Analects, Vedas, Principia Mathematica
2. **Hybrid Architectures**: Dual-stream (logic + language) or hierarchical embeddings
3. **Evaluation Tools**: Build benchmark suite for analogical reasoning
4. **Interpretability**: Visualize how concepts "crystallize" through 144 layers
5. **Alignment Research**: Compare explicit value anchoring vs. RLHF

---

## ⚠️ Important Notes

### Theological Neutrality
This is **experimental AI research**, not an attempt to create "Christian AI." The research question is:

> Can a model trained on a single coherent worldview develop reasoning skills?

The Bible was chosen for its:
- Internal consistency
- Historical influence on Western logic
- Rich genre diversity within unified framework
- Finite but dense corpus

### Limitations
- **Translation Artifacts**: Using English (NWT) loses Hebrew/Greek nuances
- **Single Interpretation**: NWT has specific theological commitments
- **Corpus Size**: 1M tokens may be insufficient for complex reasoning
- **Evaluation Challenges**: No ground truth for "correct" theological reasoning

---

## 📜 License & Citation

*[To be determined based on release strategy]*

If this research proves useful, consider citing theoretical foundations and empirical results separately.

---

**Last Updated**: 2026-01-28  
**Project Status**: Phase 2 (Logical Refinement) - In Progress
