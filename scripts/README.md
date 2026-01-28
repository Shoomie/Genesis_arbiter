# Genesis Utility Scripts

This folder contains utility scripts for corpus analysis, tokenizer training, and model evaluation.

## 📊 Corpus Analysis Scripts

### `count_unique_words.py`
Analyzes vocabulary diversity in the NWT corpus.

**Usage**:
```powershell
python count_unique_words.py
```

**Output**: Total unique words, frequency distribution

---

### `count_logical_connectives.py`
Counts occurrences of logical connectives and transition words.

**Usage**:
```powershell
python count_logical_connectives.py
```

**Output**: Frequency table of connectives (therefore, because, so, etc.)

**Key Results** (NWT Corpus):
- **Total connectives**: 21,304 (~2.1% of corpus)
- **Most frequent**: "for" (9,189), "but" (4,498), "so" (3,873)
- **Causal markers**: "because" (1,915), "therefore" (317), "thus" (94)

**Research Application**: Informs dynamic masking strategy for causal reasoning training.

---

## 🔧 Tokenizer Scripts

### `train_tokenizer.py`
Trains a BPE tokenizer on the biblical corpus.

**Usage**:
```powershell
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

---

## 📈 Evaluation Scripts

### `arbiter_perplexity.py`
Calculates perplexity on held-out biblical text.

**Usage**:
```powershell
python arbiter_perplexity.py --checkpoint ../engine/checkpoints_t2000/step_1000.pt
```

**Metrics**:
- Overall perplexity
- Per-genre breakdown (Law, Epistles, Poetry, etc.)
- Token-level loss distributions

---

### `friction_stress_test.py`
Tests gradient flow and numerical stability during training.

**Usage**:
```powershell
python friction_stress_test.py
```

**Purpose**: 
- Detects vanishing/exploding gradients in 144-layer Tower of Truth
- Measures "axiomatic friction" at logical transitions
- Validates RMSNorm stability

**Output**: 
- `friction_data.npy` (gradient norms per layer)
- Visualization of gradient flow bottlenecks

---

## 🔍 Running Scripts from Root

All scripts assume execution from the `scripts/` directory:

```powershell
cd scripts
python count_logical_connectives.py
```

If running from project root, update paths:
```powershell
python scripts/count_logical_connectives.py
# May need to update internal file paths to: "../engine/nwt_corpus.txt"
```

---

## 📝 Adding New Scripts

When creating analysis scripts:

1. **Place in this folder** for organizational clarity
2. **Use relative paths** for data files (e.g., `../engine/nwt_corpus.txt`)
3. **Document in this README** with usage examples
4. **Add to `.gitignore`** any generated data files (`.npy`, `.csv`, etc.)

---

## 🧪 Suggested Future Scripts

- `visualize_attention.py`: Plot attention weights for key tokens (Jehovah, logical connectives)
- `typological_benchmark.py`: Evaluate analogical reasoning (Isaac → Christ parallels)
- `verse_completion_test.py`: Measure fill-in-the-blank accuracy
- `genre_perplexity.py`: Compare model performance across Law, Poetry, Prophecy, etc.
- `embedding_analysis.py`: Visualize token embeddings via t-SNE/UMAP

---

## 📖 Related Documentation

- **[Research Roadmap](../docs/roadmap/README.md)**: Phase 3 evaluation framework
- **[Dynamic Masking Assessment](../docs/research/dynamic_masking_assessment.md)**: Connective analysis context
- **[Quick Reference](../docs/reference/QUICK_REFERENCE.md)**: Project overview

---

**Last Updated**: 2026-01-28
