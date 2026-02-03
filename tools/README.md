# Genesis Arbiter: Utility Scripts

This directory contains utility scripts for corpus analysis, tokenizer training, model evaluation, and automated data augmentation for the Arbiter research infrastructure.

---

## 🤖 ARBITER AUTOMATION (Phase 3 - NEW)

### `arbiter_tokenizer_factory.py`
**Automated multi-vocabulary tokenizer training with corpus-specific optimization.**

**Features**:
- Batch generation at 4k, 8k, 16k, 32k vocab sizes
- Automatic MWE (multi-word entity) extraction via TF-IDF
- Compression ratio analysis for optimal vocabulary selection
- SentencePiece BPE integration with custom symbols

**Usage**:
```bash
python arbiter_tokenizer_factory.py ./engine/nwt_corpus.txt
```

**Output**:
- `./tokenizers/arbiter_nwt_4096.model`
- `./tokenizers/arbiter_nwt_8192.model`
- `./tokenizers/tokenizer_summary.json` (compression metrics)

**Key Insight**: Forces "Jehovah God", "Jesus Christ", "Kingdom of God" as atomic tokens.

---

### `arbiter_data_augmentor.py`
**Synthetic reasoning trace generation to maximize data utility in 1M-token regime.**

**Augmentation Strategies**:
1. **Q&A Pairs** (500 traces): "Jesus went to Jerusalem" → "Where did Jesus go? Jerusalem"
2. **Cross-Reference Chains** (300 traces): Link verses A → B → C for transitive reasoning
3. **Genealogy Reasoning** (200 traces): Extract "X fathered Y" → generate grandfather inference
4. **Adversarial Samples** (100 traces): Theologically inconsistent statements for contrastive learning

**Usage**:
```bash
python arbiter_data_augmentor.py --corpus ./engine/nwt_corpus.txt --output ./augmented_data.jsonl
```

**Output**:
- `augmented_data.jsonl` (JSONL format compatible with training loops)
- `augmented_sample.txt` (human-readable inspection file)

**Research Application**: Converts 1M tokens into 1,100+ reasoning traces.

---

## 📊 CORPUS ANALYSIS (Legacy)

### `count_unique_words.py`
Analyzes vocabulary diversity in the NWT corpus.

**Usage**:
```bash
python count_unique_words.py
```

**Output**: Total unique words, frequency distribution

---

### `count_logical_connectives.py`
Counts occurrences of logical connectives and transition words.

**Usage**:
```bash
python count_logical_connectives.py
```

**Output**: Frequency table of connectives (therefore, because, so, etc.)

**Key Results** (NWT Corpus):
- **Total connectives**: 21,304 (~2.1% of corpus)
- **Most frequent**: "for" (9,189), "but" (4,498), "so" (3,873)
- **Causal markers**: "because" (1,915), "therefore" (317), "thus" (94)

**Research Application**: Informs dynamic masking strategy for causal reasoning training.

---

## 🔧 TOKENIZER TRAINING (Legacy)

### `train_tokenizer.py`
Trains a basic BPE tokenizer on the biblical corpus (superseded by `arbiter_tokenizer_factory.py`).

**Usage**:
```bash
python train_tokenizer.py
```

**Configuration**:
- Vocabulary size: 8,000-12,000 tokens
- Special tokens: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
- Custom handling: "Jehovah" as atomic token

**Output**: 
- `genesis_tokenizer.json`
- `genesis_tokenizer-vocab.json`
- `genesis_tokenizer-merges.txt`

**Note**: For Phase 3 research, prefer `arbiter_tokenizer_factory.py` for multi-vocab sweeps.

---

## 📈 EVALUATION SCRIPTS

### `arbiter_perplexity.py`
Calculates perplexity on held-out biblical text (legacy evaluation tool).

**Usage**:
```bash
python arbiter_perplexity.py --checkpoint ../engine/checkpoints/step_1000.pt
```

**Metrics**:
- Overall perplexity
- Per-genre breakdown (Law, Epistles, Poetry, etc.)
- Token-level loss distributions

**Note**: For comprehensive evaluation, use `engine/arbiter_quick_eval.py` which includes:
- Perplexity (10 min)
- Memorization vs. Generalization (20 min)
- Reasoning Probe (30 min)

---

### `friction_stress_test.py`
Tests gradient flow and numerical stability during training.

**Usage**:
```bash
python friction_stress_test.py
```

**Purpose**: 
- Detects vanishing/exploding gradients in 80-144 layer models
- Measures "axiomatic friction" at logical transitions
- Validates RMSNorm/DeepNorm stability

**Output**: 
- `friction_data.npy` (gradient norms per layer)
- Visualization of gradient flow bottlenecks

---

### `test_weighted_masking.py`
Experimental script for testing dynamic masking on logical connectives.

**Usage**:
```bash
python test_weighted_masking.py
```

**Research Context**: See [Dynamic Masking Assessment](../docs/research/dynamic_masking_assessment.md)

---

## 🔍 Running Scripts

### From Scripts Directory
```bash
cd scripts
python arbiter_data_augmentor.py --corpus ../engine/nwt_corpus.txt
```

### From Project Root (via Menu)
```bash
python run.py
# Select [5d] Data Augmentation
```

### From Project Root (Direct)
```bash
python scripts/arbiter_tokenizer_factory.py ./engine/nwt_corpus.txt
```

---

## 📝 Script Summary

| Script | Purpose | Output | Phase |
|--------|---------|--------|-------|
| `arbiter_tokenizer_factory.py` | Multi-vocab tokenizer training | `.model` files + metrics | Phase 3 |
| `arbiter_data_augmentor.py` | Synthetic reasoning generation | JSONL traces | Phase 3 |
| `arbiter_perplexity.py` | Perplexity calculation | Metrics report | Phase 2 |
| `friction_stress_test.py` | Gradient stability testing | `.npy` data | Phase 2 |
| `count_unique_words.py` | Vocabulary analysis | Console output | Phase 1 |
| `count_logical_connectives.py` | Connective frequency | Console output | Phase 1 |
| `train_tokenizer.py` | Basic tokenizer (legacy) | `.json` files | Phase 1 |
| `test_weighted_masking.py` | Masking experiments | Experimental | Phase 2 |

---

## 🧪 Recommended Workflow

For new grokking experiments:

1. **Generate Custom Tokenizer**:
   ```bash
   python arbiter_tokenizer_factory.py ./engine/nwt_corpus.txt
   ```

2. **Create Augmented Data**:
   ```bash
   python arbiter_data_augmentor.py --corpus ./engine/nwt_corpus.txt
   ```

3. **Launch Training Pipeline**:
   ```bash
   cd ../engine
   python arbiter_long_pipeline.py --corpus ./nwt_corpus.txt --output-dir ./run_001
   ```

4. **Monitor with Quick Eval** (during training):
   ```bash
   python arbiter_quick_eval.py --checkpoint ./checkpoints/step_5000 --mode all
   ```

---

## 📖 Related Documentation

- **[Implementation Plan](../brain/.../implementation_plan.md)**: Full automation architecture
- **[Walkthrough](../brain/.../walkthrough.md)**: Detailed usage examples
- **[Research Foundation](../docs/research/Architecting_Emergent_Reasoning_in_Data-Constrained_Regimes.md)**: Theoretical basis
- **[Quick Reference](../docs/reference/QUICK_REFERENCE.md)**: Project overview

---

**Last Updated**: 2026-02-03  
**Current Phase**: Phase 4 Complete (Reliability & Accelerated Diagnostics)
