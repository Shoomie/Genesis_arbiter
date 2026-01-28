# Dynamic Masking for Causal Reasoning: Technical Assessment

## Executive Summary

This document evaluates the implementation of **dynamic masking on logical connectives** as a training technique to induce causal reasoning capabilities in Bible-trained language models. Based on corpus analysis revealing **21,304 logical connectives** in the NWT (~2.1% of tokens), we propose multiple implementation strategies ranging from simple random masking to sophisticated causal structure learning.

**Key Finding**: The high frequency and consistent usage patterns of logical connectives in biblical text create an ideal training signal for causal reasoning, particularly in epistolary literature (Romans, Hebrews, etc.).

---

## I. Theoretical Foundation

### A. Why Logical Connectives Matter

Logical connectives serve as **explicit markers of reasoning structure**:

```
Premise: "All have sinned and fall short"
Connective: "THEREFORE" ← This token signals logical necessity
Conclusion: "justification is by grace"
```

**Hypothesis**: By forcing the model to predict masked connectives, we compel it to:
1. Understand the **semantic relationship** between surrounding clauses
2. Classify the relationship type (causal, contrastive, additive, temporal)
3. Select the appropriate connective from a constrained vocabulary

### B. Cognitive Science Parallel

Human readers use connectives to:
- **Parse discourse structure** (identify arguments vs. narratives)
- **Track logical dependencies** (premise → conclusion)
- **Resolve ambiguity** ("but" signals contrast; "so" signals consequence)

**Training Objective**: Replicate this cognitive process in the model by making connective prediction a high-stakes task with disproportionate loss weighting.

### C. Contrast with Standard Language Modeling

| Standard LM | Dynamic Masking on Connectives |
|-------------|-------------------------------|
| Predict next token from local context | Predict relationship from global context |
| "Therefore" learned as frequent word | "Therefore" learned as causal necessity marker |
| Loss uniform across all tokens | Loss weighted 5-10x on connectives |
| Shallow pattern matching sufficient | Deep semantic understanding required |

---

## II. Corpus Analysis: Opportunity Assessment

### Distribution by Connective Type

Based on the empirical count:

#### High-Frequency Causal Markers (Prime Targets)
- **for** (9,189 occurrences): 0.91% of corpus
  - Usage: Justification, explanation
  - Example: "God so loved the world, **for** he gave his Son"
  
- **because** (1,915 occurrences): 0.19% of corpus
  - Usage: Explicit causation
  - Example: "Faith was counted to Abraham as righteousness **because** he believed"

- **therefore** (317 occurrences): 0.03% of corpus
  - Usage: Logical conclusion
  - Example: "We are justified by faith; **therefore** we have peace with God" (Romans 5:1)

- **thus** (94 occurrences): 0.01% of corpus
  - Usage: Formal consequence
  - Example: "**Thus** it is written, and **thus** the Christ must suffer"

#### Contrastive Markers
- **but** (4,498): Signals reversal or exception
- **however** (384): Formal contrast
- **yet** (274): Concessive contrast
- **although** (139): Concessive subordination

#### Additive Markers
- **moreover** (101): Incremental support
- **furthermore** (55): Continuation of argument

### Genre-Specific Density

| Genre | Books | Expected Connective Density |
|-------|-------|----------------------------|
| **Epistles** | Romans, Galatians, Hebrews | **Very High** (logical argumentation) |
| **Wisdom** | Proverbs, Ecclesiastes | **High** (cause-effect teachings) |
| **Prophecy** | Isaiah, Revelation | **Medium** (conditional promises) |
| **Narrative** | Genesis, Acts | **Low** (chronological flow) |
| **Law** | Leviticus, Deuteronomy | **Medium** (conditional stipulations) |

**Strategic Implication**: Oversampling epistles during connective-masking training phases maximizes exposure to causal reasoning.

---

## III. Implementation Strategies

### Strategy 1: Uniform Random Masking (Baseline)

#### Algorithm
```python
def mask_logical_connectives(tokens, labels, mask_prob=0.3):
    """
    Randomly mask logical connectives with fixed probability.
    """
    logical_tokens = {
        "for": 5, "but": 12, "so": 23, "because": 45, 
        "therefore": 67, "thus": 89  # Token IDs
    }
    
    for i, token_id in enumerate(tokens):
        if token_id in logical_tokens.values() and random.random() < mask_prob:
            tokens[i] = MASK_TOKEN  # Replace with [MASK]
            # Loss will be computed on this position
    
    return tokens, labels
```

#### Characteristics
- **Simplicity**: Minimal code changes
- **Uniform treatment**: All connectives masked at same rate
- **No context awareness**: Doesn't account for difficulty

#### Expected Outcome
- Model learns basic connective distribution
- **Limitation**: May learn surface patterns ("'therefore' often follows Romans 5:1") without deeper understanding

---

### Strategy 2: Weighted Masking by Semantic Difficulty

#### Rationale
Some connectives are harder to infer than others:
- **Easy**: "but" (local contrast is usually obvious)
- **Hard**: "therefore" (requires understanding entire argument chain)

#### Algorithm
```python
def weighted_connective_masking(tokens, labels, base_prob=0.3):
    """
    Mask connectives with probability proportional to reasoning depth.
    """
    difficulty_weights = {
        "therefore": 1.0,    # Hardest: requires full argument comprehension
        "thus": 0.9,
        "consequently": 0.9,
        "because": 0.7,      # Medium: local causation
        "for": 0.6,
        "so": 0.5,
        "but": 0.3,          # Easier: local contrast
        "however": 0.3
    }
    
    for i, token_id in enumerate(tokens):
        word = tokenizer.decode([token_id]).lower()
        if word in difficulty_weights:
            mask_prob = base_prob * difficulty_weights[word]
            if random.random() < mask_prob:
                tokens[i] = MASK_TOKEN
    
    return tokens, labels
```

#### Expected Outcome
- **Curriculum learning effect**: Model forced to master hard reasoning first
- **Efficiency**: Training budget focused on high-value tokens

---

### Strategy 3: Context-Aware Structural Masking

#### Rationale
The **position** of a connective within discourse structure affects its role:
- **Verse-initial "Therefore"**: Likely concluding a multi-verse argument (hard)
- **Mid-clause "for"**: Likely local justification (easier)

#### Algorithm
```python
def structural_connective_masking(tokens, labels, verse_boundaries):
    """
    Mask connectives based on their structural position.
    """
    for i, token_id in enumerate(tokens):
        if token_id in logical_tokens:
            # Check if this is the first word of a new verse
            is_verse_initial = (i in verse_boundaries)
            
            # Check if previous clause is long (suggests complex reasoning)
            clause_length = i - find_previous_punctuation(tokens, i)
            
            # High-value position: verse-initial + long preceding context
            if is_verse_initial and clause_length > 10:
                mask_prob = 0.8  # Nearly always mask
            else:
                mask_prob = 0.3  # Standard rate
                
            if random.random() < mask_prob:
                tokens[i] = MASK_TOKEN
    
    return tokens, labels
```

#### Expected Outcome
- **Targeted learning**: Model learns to handle complex discourse structures
- **Mimics human reasoning**: Humans also find verse-initial "therefore" more significant

---

### Strategy 4: Contrastive Connective Prediction

#### Rationale
Rather than predicting the exact connective, force the model to **classify the relationship type**:
- **Causal**: {therefore, thus, consequently, hence}
- **Contrastive**: {but, however, yet, although}
- **Additive**: {moreover, furthermore, additionally}

#### Algorithm
```python
def contrastive_connective_task(tokens, labels):
    """
    Replace connective with [MASK] and add auxiliary classification head.
    """
    connective_classes = {
        "causal": ["therefore", "thus", "consequently", "so"],
        "contrastive": ["but", "however", "yet", "although"],
        "additive": ["moreover", "furthermore"],
        "explanatory": ["for", "because", "since"]
    }
    
    auxiliary_targets = []  # List to store class labels
    
    for i, token_id in enumerate(tokens):
        word = tokenizer.decode([token_id]).lower()
        
        for class_name, members in connective_classes.items():
            if word in members:
                tokens[i] = MASK_TOKEN
                # Store both the exact token AND the class
                auxiliary_targets.append({
                    "position": i,
                    "exact_token": token_id,
                    "class": class_name
                })
                break
    
    return tokens, labels, auxiliary_targets
```

#### Training Modification
Add a secondary classification head that operates on the [MASK] token's hidden state:
```python
# In model.forward()
if auxiliary_targets:
    masked_hidden_states = h[mask_positions]
    class_logits = self.connective_classifier(masked_hidden_states)
    class_loss = F.cross_entropy(class_logits, class_labels)
    total_loss = lm_loss + 0.5 * class_loss  # Multi-task learning
```

#### Expected Outcome
- **Abstraction**: Model learns relationship taxonomy, not just word prediction
- **Transfer**: May generalize to paraphrases ("so" ↔ "therefore")
- **Interpretability**: Classification probabilities reveal model's understanding

---

### Strategy 5: Causal Chain Masking (Advanced)

#### Rationale
Biblical arguments often form **multi-step chains**:
```
(1) All have sinned [premise]
(2) FOR the wages of sin is death [explanation]
(3) BUT the gift of God is eternal life [contrast]
(4) THEREFORE we are saved by grace [conclusion]
```

**Advanced Technique**: Mask **multiple connectives in a chain** simultaneously.

#### Algorithm
```python
def causal_chain_masking(tokens, labels):
    """
    Identify argument chains and mask all connectives within them.
    """
    # Step 1: Identify chain boundaries (use verse markers or punctuation)
    chains = detect_argument_chains(tokens)
    
    # Step 2: For each chain, mask all connectives with 50% probability
    for chain in chains:
        if random.random() < 0.5:  # Mask entire chain
            for i in chain["connective_positions"]:
                tokens[i] = MASK_TOKEN
    
    return tokens, labels
```

#### Expected Outcome
- **Holistic reasoning**: Model must understand entire argument to predict any connective
- **Disambiguation**: Context forces correct choice even among similar connectives
- **Risk**: May be too difficult early in training (use curriculum: easy → chain masking)

---

## IV. Loss Weighting Schemes

### A. Static Weighting
```python
# In training loop
connective_positions = (labels in logical_token_ids)
loss_weights = torch.ones_like(labels, dtype=torch.float)
loss_weights[connective_positions] = 5.0  # 5x weight on connectives

weighted_loss = (F.cross_entropy(..., reduction='none') * loss_weights).mean()
```

**Trade-off**: 
- ✅ Simple, stable
- ❌ May neglect non-connective tokens

### B. Adaptive Weighting (Difficulty-Based)
```python
# Weight based on how often the model gets it wrong
def update_weights(correct_predictions_history):
    """
    Increase weight for connectives the model struggles with.
    """
    for token_id in logical_tokens:
        accuracy = correct_predictions_history[token_id]
        if accuracy < 0.5:  # Below 50% accuracy
            weights[token_id] *= 1.2  # Increase attention
        elif accuracy > 0.9:  # Mastered
            weights[token_id] *= 0.9  # Reduce (don't waste capacity)
    
    return weights
```

**Benefit**: Self-adjusting curriculum that focuses on bottlenecks.

---

## V. Integration with Existing Genesis Architecture

### A. Dataloader Modification

**File**: `datasets/bible.py`

```python
class BibleDataset(Dataset):
    def __init__(self, tokenizer, corpus_path, masking_strategy="weighted"):
        self.masking_strategy = masking_strategy
        self.logical_tokens = self._identify_logical_tokens(tokenizer)
        # ... existing initialization
    
    def __getitem__(self, idx):
        tokens, labels = self._load_chunk(idx)
        
        # Apply connective masking based on strategy
        if self.masking_strategy == "uniform":
            tokens, labels = mask_logical_connectives(tokens, labels, mask_prob=0.3)
        elif self.masking_strategy == "weighted":
            tokens, labels = weighted_connective_masking(tokens, labels)
        elif self.masking_strategy == "structural":
            tokens, labels = structural_connective_masking(tokens, labels, self.verse_boundaries)
        
        return tokens, labels
```

### B. Model Modification (Optional: Auxiliary Head)

**File**: `models/llama/model.py`

```python
class Llama(nn.Module):
    def __init__(self, config):
        # ... existing layers
        
        # Add auxiliary connective classifier (optional)
        if config.get("connective_classification", False):
            self.connective_classifier = nn.Linear(config.dim, 4)  # 4 classes
    
    def forward(self, tokens, labels=None, auxiliary_targets=None):
        # ... existing forward pass
        
        lm_loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
        
        # Optional: Multi-task learning with connective classification
        if auxiliary_targets and hasattr(self, 'connective_classifier'):
            aux_loss = self._compute_connective_class_loss(h, auxiliary_targets)
            total_loss = lm_loss + 0.3 * aux_loss
            return logits, total_loss
        
        return logits, lm_loss
```

### C. Training Configuration

**File**: `train_configs/connective_masking.toml`

```toml
[model]
# Use Microscope for initial experimentation
dim = 768
n_layers = 12

[training]
learning_rate = 3e-4
connective_masking = true
masking_strategy = "weighted"  # Options: uniform, weighted, structural
mask_probability = 0.3
connective_loss_weight = 5.0

[connective_masking]
# Specific connective settings
high_priority = ["therefore", "thus", "because", "consequently"]
medium_priority = ["for", "so"]
low_priority = ["but", "however"]
```

---

## VI. Experimental Protocol

### Phase 1: Baseline Establishment
1. Train **Microscope** model without any masking (standard LM objective)
2. Evaluate on connective prediction accuracy (hold out 10% of connectives)
3. **Baseline Hypothesis**: Model achieves ~60-70% accuracy via local context

### Phase 2: Uniform Masking
1. Train with 30% uniform masking on all connectives
2. Measure improvement over baseline
3. **Expected**: 75-80% accuracy

### Phase 3: Weighted Masking
1. Train with difficulty-weighted masking
2. Compare per-connective accuracy (should excel at "therefore", "thus")
3. **Expected**: 80-85% accuracy, with narrowed gap between easy/hard connectives

### Phase 4: Structural Masking
1. Train with position-aware masking (emphasize verse-initial)
2. Test on out-of-domain reasoning tasks (modern ethical dilemmas)
3. **Expected**: Better transfer learning due to deeper structural understanding

### Phase 5: Ablation Studies
Test combinations:
- Masking strategy × loss weighting
- Masking rate (10%, 30%, 50%, 80%)
- Multi-task vs. single-task learning

---

## VII. Evaluation Metrics

### A. Intrinsic Metrics

#### 1. Connective Prediction Accuracy
```python
def evaluate_connective_accuracy(model, test_set):
    correct = 0
    total = 0
    
    for tokens, labels in test_set:
        # Mask all connectives in test data
        masked_tokens = mask_all_connectives(tokens)
        predictions = model(masked_tokens)
        
        for pos in connective_positions:
            if predictions[pos].argmax() == labels[pos]:
                correct += 1
            total += 1
    
    return correct / total
```

#### 2. Per-Class Accuracy
Break down by connective type:
```
Causal connectives:     82% accuracy
Contrastive:            91% accuracy
Additive:               76% accuracy
Explanatory:            88% accuracy
```

#### 3. Contextual Difficulty Score
Measure accuracy as a function of:
- Distance from preceding period (longer = harder)
- Position in verse (initial = harder)
- Genre (epistle reasoning = harder than narrative)

### B. Extrinsic Metrics: Causal Reasoning Tests

#### Test 1: Premise-Conclusion Matching
```
Prompt: "All have sinned and fall short of God's glory; _____ justification is by grace alone."
Options: [therefore, however, moreover, because]
Correct: "therefore"
```

**Evaluation**: Does model select correct connective given full context?

#### Test 2: Argument Completion
```
Prompt: "If righteousness came through the Law, then Christ died for nothing. But righteousness does NOT come through the Law. [INSERT CONCLUSION]"
Expected: Model generates conclusion using "Therefore" + logical consequence
```

#### Test 3: Typological Reasoning
```
Prompt: "Isaac's near-sacrifice prefigured Christ's actual sacrifice. _____ both involved a beloved son, a mountain, and willing obedience."
Correct: "For" or "Because" (explanatory)
Incorrect: "However" or "Therefore"
```

---

## VIII. Expected Outcomes & Risks

### A. Optimistic Scenario
- **Connective accuracy**: 85%+ on held-out test set
- **Transfer learning**: Model applies causal reasoning to modern scenarios
  - Example: "Lying harms trust; **therefore** honesty is virtuous" (learned from biblical examples)
- **Emergence**: Model develops internal representation of argument structure
  - Visualized via: Attention patterns showing premise → conclusion links

### B. Realistic Scenario
- **Connective accuracy**: 75-80%
- **Partial transfer**: Model performs well on biblical-style arguments but struggles with modern syntax
- **Genre specificity**: Excellent on epistles, mediocre on narratives

### C. Pessimistic Scenario
- **Overfitting**: Model memorizes common patterns ("Romans 5:1 always says 'therefore'") without understanding
- **No transfer**: Reasoning ability doesn't generalize beyond training distribution
- **Collapse**: High loss weighting destabilizes training (gradients explode)

### Risk Mitigation
1. **Regularization**: Use dropout on connective embeddings to prevent memorization
2. **Data augmentation**: Substitute synonymous connectives during training
3. **Gradient clipping**: Prevent instability from high loss weights
4. **Curriculum learning**: Start with easy connectives, progressively increase difficulty

---

## IX. Timeline & Resource Estimate

### Phase 1: Implementation (1 week)
- Modify `bible.py` dataloader: 1 day
- Implement masking strategies: 2 days
- Add evaluation metrics: 1 day
- Testing & debugging: 1 day

### Phase 2: Baseline Experiments (2 weeks)
- Train baseline (no masking): 3 days
- Train uniform masking: 3 days
- Train weighted masking: 3 days
- Analysis & reporting: 2 days

### Phase 3: Advanced Experiments (2 weeks)
- Structural masking: 4 days
- Multi-task learning: 4 days
- Ablation studies: 4 days
- Comprehensive evaluation: 2 days

### Compute Requirements
- **Per training run**: ~8-12 hours on RTX 4070 (Microscope mode)
- **Total runs**: ~15 experiments
- **Total compute**: ~180 GPU-hours (~$50-100 on cloud, or 7-8 days walltime on single GPU)

---

## X. Recommendations

### Immediate Next Steps
1. **Implement Strategy 2 (Weighted Masking)** first
   - Highest expected ROI
   - Minimal complexity
   - Aligns with curriculum learning principles

2. **Pilot on Small Dataset**
   - Test on Romans + Galatians only (~50k tokens)
   - Fast iteration (2 hours/run on T2000)
   - Verify implementation correctness

3. **Establish Evaluation Infrastructure**
   - Create held-out test set of connectives
   - Build premise-conclusion matching benchmark
   - Set up automated evaluation pipeline

### Long-Term Vision
If connective masking proves successful:
1. **Extend to implicit reasoning**: Mask not just connectives, but entire conclusions
   ```
   "All have sinned; therefore [MASK ENTIRE CLAUSE]"
   → Model must generate: "we need salvation"
   ```

2. **Chain masking**: As described in Strategy 5, mask entire argument chains

3. **Cross-genre transfer**: Train on epistles, test on Proverbs (different style, same logic)

4. **Modern application**: Fine-tune on ethical dilemmas using biblical reasoning patterns

---

## XI. Conclusion

**Dynamic masking of logical connectives represents a high-leverage intervention** for inducing causal reasoning in Bible-trained models. With 21,304 connectives in the NWT corpus, we have:

✅ **Sufficient training signal** (~2% of corpus)  
✅ **Diverse connective types** (causal, contrastive, additive)  
✅ **Genre variation** (epistles = high density, narratives = low)  
✅ **Clear evaluation metrics** (accuracy, transfer learning)

**Key Success Factor**: The biblical text's explicit logical structure (especially in Pauline epistles) provides a clearer training signal than general web text, where reasoning is often implicit.

**Next Milestone**: Implement weighted masking (Strategy 2) in `datasets/bible.py` and run pilot experiments on Romans to validate the approach before full-scale training.

---

## References

1. **Logical Refinement Strategies** (`../logical_refinement_strategies.md`) – Tier 1 prioritization
2. **Research Roadmap** (`README.md`) – Phase 2 context
3. **Theoretical Foundations** (`theoretical_foundations.md`) – Causal reasoning motivation
4. **Connective Count Script** (`../count_logical_connectives.py`) – Empirical data source
