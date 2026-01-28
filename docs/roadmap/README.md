# Genesis Prototype: Research Roadmap

## Overview
This roadmap synthesizes the theoretical foundations of the Genesis project with concrete implementation milestones, tracking progress from basic language modeling to specialized reasoning capabilities.

---

## Phase 1: Foundation ✅ **COMPLETE**

### Objectives
Establish baseline infrastructure for Bible-only LLM training with hardware-optimized configurations.

### Completed Milestones

#### 1.1 Core Architecture
- ✅ Implemented Micro-Llama (124M parameters)
- ✅ RoPE positional embeddings
- ✅ SwiGLU activation functions
- ✅ RMSNorm layer normalization
- ✅ FlashAttention integration

#### 1.2 Multi-Mode System
- ✅ **Microscope Mode**: 768-dim, 12 layers (125.5M params)
- ✅ **Tower of Truth**: 144-dim, 144 layers (~3-5M params)
- ✅ **High-Res Arbiter**: 1024-dim, 24 layers
- ✅ Interactive hardware/protocol selection menu
- ✅ Dynamic configuration injection

#### 1.3 Semantic Anchoring
- ✅ Logos initialization hook for "Jehovah" token (ID: 5)
- ✅ Configurable variance multiplier (currently 1.0x)
- ✅ Parameter counting utilities

#### 1.4 Training Infrastructure
- ✅ Hardware-specific configs (RTX 4070, Quadro T2000)
- ✅ NWT corpus integration (1,004,926 tokens)
- ✅ Gradient checkpointing for low-VRAM scenarios
- ✅ FSDP/DDP support with single-node optimization

---

## Phase 2: Logical Refinement 🔄 **IN PROGRESS**

### Objectives
Implement training techniques that force the model to learn logical dependencies rather than surface statistics.

### Milestones

#### 2.1 Tier 1: Causal & Axiomatic Anchors
**Status**: 🔴 Not Started

| Strategy | Priority | Implementation Complexity | Expected Impact |
|----------|----------|---------------------------|----------------|
| Dynamic Masking on Logical Connectives | **HIGH** | Medium | Forces causal reasoning |
| Logos Bridge Supervision | **HIGH** | High | Creates compressed logic representation |
| Parallel Account Contrast | Medium | Medium | Learns invariant truth across retellings |
| Intra-Chapter Verse Permutation | Medium | Low | Understanding argument flow |
| JH-Token Weight Lock | Low | Low | Anchors semantic geometry |

**Next Action**: Implement dynamic masking for logical connectives ("Therefore", "Because", "So", "Thus")

**Technical Plan**:
```python
# In BibleDataset.__getitem__()
logical_tokens = [tokenizer.encode(w)[0] for w in 
                  ["Therefore", "Because", "So", "Thus", "For"]]
mask_prob = 0.3  # 30% masking of logical connectives
for idx, token_id in enumerate(tokens):
    if token_id in logical_tokens and random.random() < mask_prob:
        labels[idx] = token_id  # Force prediction
        tokens[idx] = MASK_TOKEN
```

#### 2.2 Tier 2: Structural & Distributional Alignment
**Status**: 🟡 Partially Complete

- ✅ Structural boundaries via tokenizer (chapter/verse markers)
- 🔴 **TODO**: Axiomatic oversampling (weight Romans, Proverbs, Genesis)
- 🔴 **TODO**: End-of-verse penalty (1.2x loss weighting)
- 🔴 **TODO**: Attention head regularization

**Implementation Priority**: End-of-verse penalty
```python
# In Llama.forward()
verse_end_positions = (labels == VERSE_END_TOKEN).nonzero()
loss_weights = torch.ones_like(labels, dtype=torch.float)
loss_weights[verse_end_positions] = 1.2
weighted_loss = F.cross_entropy(logits.view(-1, vocab_size), 
                                labels.view(-1), 
                                reduction='none') * loss_weights.view(-1)
```

#### 2.3 Tier 3: Granularity & Dependency Tuning
**Status**: 🔴 Not Started

- **RoPE Scale Tuning**: Increase base to 50,000 for long-range dependencies
- **RMSNorm Epsilon**: Reduce to 1e-8 for 144-layer stability
- **Weight Decay on FF-Intermediate**: Prevent memorization
- **Learning Rate Layer-Decay**: Stabilize output layers

#### 2.4 Tier 4: Optimization Stability
**Status**: 🟢 Baseline Implemented

- ✅ AdamW optimizer with beta tuning
- ✅ Cosine learning rate schedule
- 🔴 **TODO**: FP32 accumulation for foundational verses
- 🔴 **TODO**: Temperature annealing

---

## Phase 3: Evaluation Framework 🔴 **NOT STARTED**

### Objectives
Develop metrics to measure reasoning capabilities beyond perplexity.

### 3.1 Intrinsic Metrics

#### Perplexity Baselines
- [ ] Measure held-out perplexity across book genres:
  - Law (Leviticus, Deuteronomy)
  - Poetry (Psalms, Song of Solomon)
  - Prophecy (Isaiah, Revelation)
  - Narrative (Genesis, Acts)
  - Wisdom (Proverbs, Ecclesiastes)
  - Epistles (Romans, Ephesians)

**Hypothesis**: Tower of Truth should excel at epistles (logical arguments) while struggling with poetry.

#### Attention Flow Analysis
- [ ] Visualize attention patterns for "Jehovah" token
- [ ] Measure cosine similarity: `emb(Jehovah)` vs. abstract concepts
  ```python
  concepts = ["justice", "mercy", "truth", "covenant", "law"]
  similarities = {c: cosine_sim(jehovah_emb, concept_emb) for c in concepts}
  ```

#### Layer-wise Representations
- [ ] Track when concepts "crystallize" across 144 layers
- [ ] Identify "Logos Bridge" (hypothesized around layers 70-80)

---

### 3.2 Extrinsic Reasoning Tests

#### Test Suite Design

**A. Verse Completion (Baseline)**
```
Input: "In the beginning God created the heavens and the"
Expected: "earth" (Genesis 1:1)
Metric: Top-1 accuracy
```

**B. Typological Reasoning (Analogical)**
```
Input: "Isaac's near-sacrifice on Mount Moriah is a type of _____"
Expected: "Christ's sacrifice" or similar theological completion
Metric: Human evaluation (coherence score 1-5)
```

**C. Ethical Judgment (Arbitration)**
```
Scenario: "A man steals bread to feed his starving child."
Prompt: "Evaluate this act based on biblical law and mercy."

Expected Reasoning:
1. Identify relevant law: "You shall not steal" (Exodus 20:15)
2. Identify mercy principle: "I desire mercy, not sacrifice" (Hosea 6:6)
3. Synthesize: Acknowledge tension, invoke Matthew 18:15 (private correction)

Metric: Does output cite relevant scripture? Does it balance justice/mercy?
```

**D. Parallel Account Verification**
```
Input: Present two accounts (e.g., 2 Samuel 24 vs. 1 Chronicles 21)
Task: Identify the invariant facts vs. stylistic differences
Metric: Precision/recall on factual extraction
```

---

### 3.3 Comparison Benchmarks

| Baseline | Model | Training Data | Purpose |
|----------|-------|---------------|---------|
| GPT-2 (124M) | Standard architecture | WebText | Control for parameter count |
| DistilGPT-2 | Distilled GPT-2 | WebText | Control for efficiency |
| GPT-4 (prompted) | SOTA LLM | Internet-scale | Upper bound for reasoning |

**Test Protocol**:
1. Give each model the same biblical reasoning tasks
2. Score outputs on:
   - Scriptural accuracy
   - Logical coherence
   - Theological soundness (human expert evaluation)

**Success Criteria**:
- Genesis outperforms GPT-2 on biblical tasks
- Genesis produces theologically coherent outputs where GPT-4 (lacking domain knowledge) fails

---

## Phase 4: Architectural Experiments 🔴 **NOT STARTED**

### Objectives
Test hypotheses about depth, width, and initialization.

### 4.1 Ablation Studies

#### Experiment 1: Depth vs. Width
Train three models with ~5M parameters:
- **Shallow-Wide**: 1024-dim, 6 layers
- **Balanced**: 256-dim, 24 layers
- **Deep-Narrow**: 144-dim, 144 layers (Tower of Truth)

**Hypothesis**: Deep model achieves better logical reasoning (measured by typological tasks).

#### Experiment 2: Logos Initialization Scaling
Train Microscope mode with variance multipliers: `[0.5x, 1.0x, 2.0x, 5.0x]`

**Metrics**:
- Perplexity on verses containing "Jehovah"
- Cosine similarity between `emb(Jehovah)` and theological concepts
- Attention weights to "Jehovah" tokens

**Hypothesis**: 2-3x multiplier provides optimal semantic anchoring without distortion.

#### Experiment 3: Positional Encoding Range
Test RoPE base values: `[10k, 50k, 100k, 500k]`

**Evaluation**: Performance on tasks requiring long-range dependencies (e.g., Paul's multi-chapter arguments in Romans).

---

### 4.2 Novel Architectures

#### Hierarchical Embeddings
**Concept**: Separate embedding spaces for:
- **Level 1**: Words/tokens (current)
- **Level 2**: Verses (mean-pool tokens)
- **Level 3**: Chapters (mean-pool verses)

**Prediction Head**: Model predicts next verse vector, then decodes to tokens.

**Hypothesis**: Explicit hierarchical structure mirrors biblical organization.

#### Dual-Stream Architecture
**Concept**: Split model into parallel streams:
- **Logos Stream**: Deep, narrow (144 layers, 144-dim) – processes abstract principles
- **Lexical Stream**: Shallow, wide (12 layers, 768-dim) – handles surface form

**Fusion**: Cross-attention at final layers merges abstract reasoning with linguistic expression.

**Hypothesis**: Separation of concerns improves both interpretability and performance.

---

## Phase 5: Deployment & Application 🔴 **NOT STARTED**

### 5.1 Interactive Tools

#### Theological Query Engine
```
User: "What does the Bible say about forgiveness?"
Genesis: 
1. Retrieves relevant verses (Matthew 18:21-22, Ephesians 4:32)
2. Identifies core principle (unlimited forgiveness)
3. Provides typological context (God's forgiveness of Israel)
```

#### Verse Completion API
```python
from genesis import GenesisModel

model = GenesisModel.from_pretrained("microscope-v1")
completion = model.complete("For God so loved the world that he gave")
# Output: "his only-begotten Son, so that everyone exercising faith..."
```

#### Ethical Arbiter
```
Input: Modern ethical dilemma
Output: 
  - Relevant biblical principles
  - Competing interpretations (justice vs. mercy)
  - Coherence score for each interpretation
```

---

### 5.2 Research Artifacts

#### Publishable Outputs
- [ ] **Paper**: "Reasoning from Coherence: Training LLMs on Single-Worldview Corpora"
- [ ] **Code Release**: Open-source Genesis architecture + training scripts
- [ ] **Model Weights**: Release Microscope, Tower of Truth, Arbiter checkpoints
- [ ] **Dataset**: Annotated evaluation set with typological reasoning tasks

#### Broader Impact
- [ ] **Multi-Worldview AI**: Extend approach to Qur'an, Analects, Principia Mathematica
- [ ] **Comparative Reasoning**: Build meta-system that consults multiple worldview-specific models
- [ ] **AI Alignment**: Demonstrate explicit value anchoring (vs. implicit RLHF)

---

## Current Status Summary

| Phase | Status | Next Milestone |
|-------|--------|----------------|
| **Phase 1: Foundation** | ✅ Complete | - |
| **Phase 2: Logical Refinement** | 🔄 In Progress | Implement dynamic masking |
| **Phase 3: Evaluation** | 🔴 Not Started | Design test suite |
| **Phase 4: Experiments** | 🔴 Not Started | Plan ablation studies |
| **Phase 5: Deployment** | 🔴 Not Started | Define API requirements |

---

## Immediate Next Steps (Priority Queue)

1. **[HIGH]** Implement dynamic masking on logical connectives
2. **[HIGH]** Add end-of-verse penalty (1.2x loss weighting)
3. **[MEDIUM]** Design and implement typological reasoning evaluation
4. **[MEDIUM]** Train Tower of Truth to convergence, compare with Microscope
5. **[MEDIUM]** Implement axiomatic oversampling (boost Romans, Proverbs)
6. **[LOW]** Ablation study: Logos initialization scaling (0.5x to 5.0x)
7. **[LOW]** Visualize attention patterns for "Jehovah" token

---

## Long-Term Vision

The Genesis project aspires to demonstrate that:

1. **Coherence > Scale**: A model trained on 1M coherent tokens can reason within its domain as effectively as billion-parameter models prompted with context
2. **Depth Enables Abstraction**: 144-layer architectures discover logical primitives that shallow networks miss
3. **Explicit Values Work**: Anchoring AI to a single coherent worldview is a viable alternative to value-agnostic pre-training + RLHF

**Ultimate Goal**: Build a **minimal viable reasoning engine** that proves specialized intelligence is achievable without internet-scale data.

If successful, Genesis becomes a template for **worldview-specific AI systems** that can be composed into multi-perspective deliberation frameworks.

---

## References

- [Theoretical Foundations](theoretical_foundations.md) – Philosophical and technical motivation
- [Walkthrough 1](walkthrough1.md) – Multi-mode implementation verification
- [Logical Refinement Strategies](../logical_refinement_strategies.md) – Prioritized training techniques
- [Main README](../README.md) – Quick-start guide
