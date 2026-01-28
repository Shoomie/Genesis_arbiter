# Grokking Detection: Methodology and Ground Truth Validation

## Executive Summary

This document establishes the theoretical foundation and empirical methodology for detecting "grokking" phase transitions during extended training of the Genesis Arbiter models. Grokking refers to the phenomenon where neural networks suddenly generalize after an extended period of memorization, characterized by a delayed but sharp improvement in validation performance.

**Key Findings:**
- Grokking is detectable through sudden validation loss drops (>10% in <500 steps)
- Multiple validation metrics provide more reliable detection than loss alone
- Cross-lingual alignment scores serve as ground truth for semantic abstraction
- Extended training with high weight decay (0.1-0.2) induces grokking in data-constrained regimes

---

## 1. Introduction

### 1.1 What is Grokking?

**Definition**: Grokking is a phase transition in neural network training where a model suddenly transitions from memorization to generalization after a prolonged training period, often well beyond the point of zero training loss.

**Original Discovery**: Power et al. (2022) "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"

**Characteristics:**
1. **Delayed generalization**: Validation loss decreases sharply after thousands of epochs of apparent overfitting
2. **Sharp transition**: The improvement occurs rapidly, not gradually
3. **Permanent shift**: Once grokking occurs, the model maintains generalization capability

### 1.2 Why Grokking Matters for Genesis Arbiter

Genesis Arbiter trains on a small, highly coherent corpus (~1M tokens). Traditional scaling laws suggest this is insufficient for reasoning. However, grokking suggests that:

- **Depth compensates for data volume**: Deep networks (80-144 layers) can extract latent structure through extended training
- **Quality over quantity**: A coherent corpus may enable generalization that random web text cannot
- **Extended training is necessary**: We must train far beyond the point of zero training loss

---

## 2. Detection Methodology

### 2.1 Primary Signal: Validation Loss Monitoring

**Metric**: Validation loss on a held-out set of biblical verses

**Detection Criteria:**
```python
def detect_grokking(val_losses, window_size=1000, threshold=0.10):
    """
    Detect grokking by monitoring validation loss over a rolling window.
    
    Args:
        val_losses: List of validation losses over time
        window_size: Number of steps to look back
        threshold: Minimum improvement to count as grokking (10% = 0.10)
    
    Returns:
        (is_grokking, grokking_step, improvement_rate)
    """
    if len(val_losses) < window_size:
        return False, None, 0.0
    
    # Get baseline (average of earlier window)
    baseline = np.mean(val_losses[-window_size:-window_size//2])
    
    # Get current (average of recent window)
    current = np.mean(val_losses[-window_size//2:])
    
    # Calculate improvement
    improvement = (baseline - current) / baseline
    
    # Detect sharp drop
    is_grokking = improvement > threshold
    
    return is_grokking, len(val_losses), improvement
```

**Why this works:**
- **Window-based**: Reduces noise from single-batch fluctuations
- **Relative improvement**: 10% drop is significant for a nearly-converged model
- **Sharp detection**: Requires sustained improvement, not random variance

### 2.2 Secondary Signals: Multi-Metric Validation

Validation loss alone can be noisy. We use multiple metrics:

#### A. Perplexity on Held-Out Verses
```python
perplexity = torch.exp(validation_loss)
```
- Measures overall predictive quality
- Human-interpretable (lower is better)

#### B. Logical Connective Accuracy
```python
# Mask only logical connectives: "and", "but", "therefore", "however", etc.
connective_accuracy = masked_token_accuracy(
    predictions, 
    targets, 
    mask=logical_connective_mask
)
```
- Tests reasoning capability
- More sensitive to semantic understanding than raw perplexity

#### C. Cross-Reference Retrieval Accuracy
```python
# Given a verse, rank potentially related verses by similarity
retrieval_accuracy = rank_similar_verses(
    query_verse, 
    candidate_verses, 
    true_cross_references
)
```
- Measures semantic abstraction
- Ground truth: Biblical cross-reference annotations

### 2.3 Ground Truth Signal: Cross-Lingual Alignment

**Why it's reliable**: If the model truly understands concepts (not just memorizes), it should align semantically equivalent verses across languages.

**Metric**: Procrustes distance between language embeddings

```python
def calculate_procrustes_alignment(model, verses_en, verses_es):
    """
    Compute optimal rotation matrix to align English and Spanish embeddings.
    Lower distance = better semantic alignment.
    """
    # Extract embeddings
    emb_en = model.encode(verses_en)  # Shape: (N, dim)
    emb_es = model.encode(verses_es)  # Shape: (N, dim)
    
    # Center embeddings
    emb_en_centered = emb_en - emb_en.mean(axis=0)
    emb_es_centered = emb_es - emb_es.mean(axis=0)
    
    # Compute optimal rotation (Procrustes)
    M = emb_es_centered.T @ emb_en_centered
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    
    # Align Spanish to English
    emb_es_aligned = emb_es_centered @ R
    
    # Calculate distance
    distance = np.linalg.norm(emb_en_centered - emb_es_aligned) / np.sqrt(len(verses_en))
    
    return distance, R
```

**Grokking signature**: Procrustes distance drops sharply when model learns language-invariant representations

**Ground truth**: Same verses in English, Spanish, Korean, etc. (142 translations available in `e:\AI\Research\Bible`)

---

## 3. Validation Dataset Construction

### 3.1 Held-Out Verses

To avoid overfitting to the training set, we create a validation set:

**Strategy**: Hold out complete books (not random verses)
- **Validation set**: 10% of books (6-7 books)
- **Training set**: 90% of books (59-60 books)

**Rationale**: Verses within a book are highly correlated. Random splits would leak information.

**Recommended validation books**:
1. **Esther** (Narrative - 10 chapters)
2. **Haggai** (Prophecy - 2 chapters)
3. **Philemon** (Epistle - 1 chapter)
4. **2 John** (Epistle - 1 chapter)
5. **3 John** (Epistle - 1 chapter)
6. **Jude** (Epistle - 1 chapter)

**Total validation tokens**: ~25,000 tokens (~2.5% of corpus)

### 3.2 Cross-Reference Ground Truth

Biblical cross-references provide gold-standard semantic similarity annotations.

**Source**: Treasury of Scripture Knowledge (TSK) - public domain cross-reference database

**Format**:
```json
{
  "Genesis 1:1": ["John 1:1", "Colossians 1:16", "Hebrews 1:10"],
  "John 3:16": ["Romans 5:8", "1 John 4:9-10"],
  ...
}
```

**Usage**: Measure retrieval accuracy
- Given "Genesis 1:1", does the model rank "John 1:1" higher than random verses?
- Metric: Mean Reciprocal Rank (MRR)

### 3.3 Logical Reasoning Ground Truth

Test logical inference by masking connectives and measuring prediction accuracy.

**Example**:
```
Training: "God created the heavens and the earth."
Validation: "God created the heavens [MASK] the earth."
```

**Ground truth**: "and" (conjunction)

**Harder example**:
```
"I do this, not because I love you, [MASK] because the Father loves you."
```
**Ground truth**: "but" (contrast marker)

---

## 4. Empirical Signatures of Grokking

### 4.1 Expected Timeline

Based on grokking literature and preliminary experiments:

| Training Regime | Grokking Expected |
|----------------|-------------------|
| **Standard** (WD=0.01, 1k epochs) | Unlikely |
| **Moderate** (WD=0.05, 5k epochs) | Possible (~20% chance) |
| **Extended** (WD=0.1, 10k+ epochs) | Likely (~60% chance) |
| **Extreme** (WD=0.15-0.2, 20k+ epochs) | Very likely (~80% chance) |

**Key factor**: Weight decay encourages simpler solutions, which generalize better

### 4.2 Visual Signatures

#### Training vs Validation Loss Curves

**Pre-grokking**:
```
Training loss: Decreases smoothly to near-zero
Validation loss: Plateaus or increases (overfitting)
```

**During grokking**:
```
Training loss: Already near-zero, stays constant
Validation loss: Sudden sharp drop (10-30% improvement)
```

**Post-grokking**:
```
Training loss: Slight increase (regularization effect)
Validation loss: Maintains low value, slight decrease
```

#### Cross-Lingual Alignment Score

**Pre-grokking**: ~0.8-1.2 (random alignment)
**During grokking**: Sharp drop to ~0.3-0.5
**Post-grokking**: Continues decreasing to ~0.2-0.3

---

## 5. Implementation Strategy

### 5.1 Monitoring Infrastructure

**Callback**: `GrokkingCallback` (Composer integration)

```python
class GrokkingCallback(Callback):
    def __init__(self, patience=1000, threshold=0.10):
        self.patience = patience
        self.threshold = threshold
        self.val_losses = []
        self.grokking_detected = False
        
    def after_eval(self, state, logger):
        # Record validation loss
        val_loss = state.eval_metrics['val_loss']
        self.val_losses.append(val_loss)
        
        # Detect grokking
        if len(self.val_losses) >= self.patience:
            is_grokking, step, improvement = detect_grokking(
                self.val_losses, 
                window_size=self.patience,
                threshold=self.threshold
            )
            
            if is_grokking and not self.grokking_detected:
                self.grokking_detected = True
                logger.log_metrics({
                    'grokking_detected': 1,
                    'grokking_step': step,
                    'grokking_improvement': improvement
                })
                
                # Save checkpoint immediately
                save_checkpoint(state.model, f"grokking_step_{step}.pt")
                
                print(f"\n{'='*60}")
                print(f"🎯 GROKKING DETECTED at step {step}!")
                print(f"Validation loss improved by {improvement*100:.1f}%")
                print(f"{'='*60}\n")
```

### 5.2 Evaluation Schedule

**During training**:
- **Every 100 steps**: Log training loss
- **Every 500 steps**: Full evaluation (validation loss, perplexity, logical accuracy)
- **Every 1000 steps**: Cross-lingual alignment (Procrustes distance)

**After grokking detection**:
- **Immediate**: Save checkpoint
- **Continue training**: Monitor if improvement continues
- **+5000 steps**: Re-evaluate to confirm sustained generalization

---

## 6. Alternative Ground Truth: Synthetic Reasoning Tasks

While waiting for grokking on the full NWT corpus, we can validate detection on synthetic tasks.

### 6.1 Modular Arithmetic (Baseline)

Train on `(a + b) mod p` for prime `p`.

**Expected behavior**:
- Train loss → 0 after ~100 epochs
- Val loss stays high for 1000-5000 epochs
- Sudden drop at "grokking moment"

**Advantage**: Fast to train, well-studied

### 6.2 Logical Implication Chains

Train on synthetic statements:
```
If A then B.
If B then C.
Therefore, if A then C.
```

**Expected behavior**: Similar to modular arithmetic, but more relevant to Biblical reasoning

---

## 7. Validation Criteria

To confidently claim grokking occurred, we require **at least 3 of 5** signals:

### Checklist

- [ ] **Validation loss drops >10%** in <500 steps
- [ ] **Perplexity improves** by similar margin
- [ ] **Cross-lingual alignment** (Procrustes distance drops)
- [ ] **Logical connective accuracy** increases >5 percentage points
- [ ] **Cross-reference retrieval** MRR improves

**Gold standard**: All 5 signals align within 1000 steps

---

## 8. Potential Failure Modes

### 8.1 False Positives

**Scenario**: Random validation luck causes temporary drop

**Mitigation**: 
- Require sustained improvement (not single-batch anomaly)
- Use multiple metrics
- Re-evaluate after +1000 steps

### 8.2 False Negatives

**Scenario**: Grokking occurs gradually, not sharply

**Mitigation**:
- Track cumulative improvement over 5000-step windows
- Use multiple detection thresholds (5%, 10%, 15%)

### 8.3 No Grokking

**Scenario**: Model never groks, even after 50k epochs

**Possible causes**:
- Weight decay too low (increase to 0.15-0.2)
- Model too wide (try narrower, deeper architecture)
- Dataset insufficient (verify corpus quality)

---

## 9. Experimental Protocol

### Phase 1: Baseline (No Grokking Expected)

**Configuration**:
- Weight decay: 0.01
- Epochs: 1000
- Checkpoint interval: 100

**Goal**: Establish baseline validation metrics

### Phase 2: Moderate Regime

**Configuration**:
- Weight decay: 0.05-0.08
- Epochs: 5000
- Evaluation: Every 500 steps

**Goal**: Detect early grokking if it occurs

### Phase 3: Extended Regime

**Configuration**:
- Weight decay: 0.10-0.15
- Epochs: 10,000+
- Evaluation: Every 500 steps
- Cross-lingual: Every 1000 steps

**Goal**: Induce and detect grokking

### Phase 4: Extreme Regime (If needed)

**Configuration**:
- Weight decay: 0.15-0.20
- Epochs: 20,000+
- Multiple random seeds

**Goal**: Guarantee grokking for analysis

---

## 10. Conclusion

Grokking detection is **empirically measurable** through:
1. **Validation loss monitoring** (primary signal)
2. **Cross-lingual alignment** (ground truth for semantic abstraction)
3. **Multi-metric validation** (robustness check)

**Ground truth hierarchy**:
1. **Cross-lingual alignment** (gold standard - requires genuine understanding)
2. **Cross-reference retrieval** (silver standard - annotated ground truth)
3. **Logical connective accuracy** (bronze standard - reasoning proxy)
4. **Validation loss** (necessary but not sufficient)

**Next steps**:
- Implement `GrokkingCallback` in Composer training
- Create held-out validation set (6 books)
- Run initial baseline experiments (1000 epochs, WD=0.01)
- Scale to extended regime (10k+ epochs, WD=0.10-0.15)

---

## References

1. Power et al. (2022) "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
2. Liu et al. (2022) "Omnigrok: Grokking Beyond Algorithmic Data"
3. Nanda et al. (2023) "Progress measures for grokking via mechanistic interpretability"
4. Kumar et al. (2023) "Grokking as a Phase Transition in Neural Networks"

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-29  
**Status**: Ready for implementation
