# Genesis Arbiter Development Environment Restructure Prompt

## Objective
Restructure the Genesis Arbiter training infrastructure to leverage modern PyTorch optimization frameworks while maintaining full customizability for character-level, deep-narrow transformer experiments on the NWT corpus.

---

## Current State Analysis

### Existing Architecture
- **Model**: Custom Llama-style transformer (12-144 layers, DeepNorm stabilization)
- **Tokenization**: Character-level (UTF-8) or custom BPE tokenizer
- **Dataset**: NWT Bible corpus (~1M tokens, 30+ languages planned)
- **Training Scale**: 50M-2B parameters on RTX 4070 (12GB VRAM) + 64GB RAM
- **Current Stack**: Pure PyTorch with manual optimization

### Problems to Solve
1. **No FlashAttention integration** - Missing 2-4x speedup and linear memory scaling
2. **Manual optimization** - Reinventing gradient accumulation, checkpointing, distributed training
3. **Limited scalability** - Difficult to scale to multi-GPU or larger models
4. **Inefficient character-level sequences** - 4-5x longer sequences need specialized kernels
5. **No framework overhead** - Need to preserve deep customization for grokking experiments

---

## Target Framework Recommendations

### Option 1: PyTorch + Composer (Recommended for Your Use Case)

**Why Composer:**
- **Lightweight**: Minimal abstraction over PyTorch, easy to customize
- **FlashAttention-2/3 built-in**: Automatic kernel selection
- **Grokking-friendly**: Supports extended training with custom callbacks
- **Multi-task learning**: Native support for your coherence/cross-ref tasks
- **Excellent for research**: Used by MosaicML for ablation studies

**Integration effort**: ~1-2 days
**Customizability**: 9/10 (thin wrapper, full PyTorch access)

```python
# Example: Composer with your custom model
from composer import Trainer
from composer.models import HuggingFaceModel

# Your custom model stays exactly as-is
from models.llama.model import Llama

# Wrap with minimal boilerplate
model = HuggingFaceModel(Llama(...))

trainer = Trainer(
    model=model,
    train_dataloader=dataloader,
    max_duration="10000ep",  # Grokking regime
    optimizers=optimizer,
    algorithms=[
        # Add FlashAttention, gradient clipping, etc.
        FlashAttention2(),
        GradientClipping(clipping_threshold=1.0)
    ],
    callbacks=[
        # Your custom grokking detector
        GrokkingCallback(),
        ConceptClusteringMetrics()
    ]
)
```

### Option 2: NanoGPT + Extensions (Maximum Control)

**Why NanoGPT:**
- **Ultra-minimal**: ~300 lines of core training code
- **Educational**: Andrej Karpathy's clean reference implementation
- **Easy FlashAttention integration**: Just swap `F.scaled_dot_product_attention`
- **Full control**: No framework overhead

**Integration effort**: ~3-5 days (more manual work)
**Customizability**: 10/10 (you control everything)

```python
# Add FlashAttention to your existing model
import torch.nn.functional as F

# In your attention layer
attn_output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
    enable_gqa=True  # Grouped Query Attention for memory efficiency
)
```

### Option 3: HuggingFace Transformers + Accelerate (Industry Standard)

**Why HF + Accelerate:**
- **Mature ecosystem**: Pre-built DeepNorm, RoPE, everything
- **FlashAttention via SDPA**: Automatic backend selection
- **Accelerate = distributed magic**: Handles multi-GPU, mixed precision, checkpointing
- **Great for production**: Easy deployment path

**Integration effort**: ~2-3 days
**Customizability**: 7/10 (more opinionated, but well-documented extension points)

```python
from transformers import LlamaConfig, LlamaForCausalLM
from accelerate import Accelerator

# Configure your deep-narrow architecture
config = LlamaConfig(
    hidden_size=768,
    num_hidden_layers=80,  # Deep & Narrow
    num_attention_heads=12,
    vocab_size=256,  # Character-level
    max_position_embeddings=4096,
    use_flash_attention_2=True,  # Automatic FA2
    rope_scaling=None,
    tie_word_embeddings=False
)

model = LlamaForCausalLM(config)

# Accelerate handles device placement, mixed precision, gradient accumulation
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=4
)

model, optimizer, dataloader = accelerator.prepare(
    model, optimizer, dataloader
)
```

---

## Recommended Architecture: Composer + Custom Components

### Why This Combination Works Best

1. **Preserves your custom model** - Keep your DeepNorm Llama implementation as-is
2. **Adds FlashAttention** - 2-4x speedup on character-level sequences
3. **Enables multi-task learning** - Easy to add coherence/cross-ref tasks
4. **Grokking-friendly** - Extended training with custom callbacks
5. **Distributed training** - Scales to multi-GPU when you need it
6. **Minimal refactor** - ~200 lines of glue code

### Implementation Plan

#### Phase 1: Core Migration (Week 1)
```
1. Install Composer: pip install mosaicml
2. Wrap your Llama model in HuggingFaceModel
3. Replace manual training loop with Trainer
4. Enable FlashAttention2 algorithm
5. Verify training metrics match baseline
```

#### Phase 2: Multi-Task Integration (Week 2)
```
1. Implement MultiTaskModel wrapper
2. Add coherence detection head
3. Add cross-reference prediction head
4. Add cross-lingual paraphrase head
5. Configure task sampling (70% LM / 15% coherence / 15% paraphrase)
```

#### Phase 3: Grokking Infrastructure (Week 3)
```
1. Implement GrokkingCallback (tracks validation metrics)
2. Add ProcrustesDistanceCallback (measures cross-lingual alignment)
3. Add ConceptClusteringCallback (evaluates semantic abstraction)
4. Configure extended training (10,000+ epochs)
```

---

## Concrete Code Structure

### Proposed Directory Layout
```
Genesis_arbiter/
├── engine/
│   ├── models/
│   │   ├── llama/
│   │   │   ├── model.py              # Your existing Llama (unchanged)
│   │   │   └── config.py
│   │   └── multi_task_wrapper.py     # NEW: Multi-task heads
│   │
│   ├── datasets/
│   │   ├── bible.py                  # Your existing dataset
│   │   └── multi_task_sampler.py     # NEW: Task-aware sampling
│   │
│   ├── training/
│   │   ├── composer_trainer.py       # NEW: Composer integration
│   │   ├── callbacks/
│   │   │   ├── grokking.py           # NEW: Grokking detection
│   │   │   ├── procrustes.py         # NEW: Cross-lingual metrics
│   │   │   └── concept_clustering.py # NEW: Semantic evaluation
│   │   └── algorithms/
│   │       └── flash_attention.py    # NEW: FA2/FA3 config
│   │
│   └── train_composer.py             # NEW: Main training script
│
└── configs/
    └── composer/
        ├── base.yaml                 # Hardware configs
        ├── deep_narrow_80L.yaml      # Your 80-layer config
        └── grokking_regime.yaml      # Extended training config
```

### Key File: `training/composer_trainer.py`

```python
"""
Composer-based training for Genesis Arbiter
Integrates FlashAttention, multi-task learning, and grokking detection
"""

from composer import Trainer, algorithms
from composer.callbacks import LRMonitor, MemoryMonitor, SpeedMonitor
from composer.loggers import TensorBoardLogger
from composer.models import HuggingFaceModel

from models.llama.model import Llama
from models.multi_task_wrapper import MultiTaskLlama
from datasets.multi_task_sampler import MultiTaskDataLoader
from training.callbacks.grokking import GrokkingCallback
from training.callbacks.procrustes import ProcrustesCallback


class GenesisTrainer:
    def __init__(self, config):
        self.config = config
        
        # Your custom model (unchanged)
        base_model = Llama(
            dim=config.model.dim,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            vocab_size=config.model.vocab_size,
            max_seq_len=config.model.max_seq_len
        )
        
        # Add multi-task heads
        self.model = MultiTaskLlama(
            base_model=base_model,
            coherence_dim=1024,
            enable_cross_ref=True,
            enable_paraphrase=True
        )
        
        # Wrap for Composer
        self.composer_model = HuggingFaceModel(self.model)
        
        # Dataset with multi-task sampling
        self.dataloader = MultiTaskDataLoader(
            data_dir=config.data.path,
            languages=config.data.languages,
            batch_size=config.training.batch_size,
            task_distribution={
                'lm': 0.70,
                'coherence': 0.15,
                'cross_ref': 0.075,
                'paraphrase': 0.075
            }
        )
        
        # Optimizer (your existing AdamW config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
            betas=(0.9, 0.95)
        )
        
    def train(self):
        trainer = Trainer(
            model=self.composer_model,
            train_dataloader=self.dataloader,
            max_duration=f"{self.config.training.max_epochs}ep",
            optimizers=self.optimizer,
            
            # Enable FlashAttention
            algorithms=[
                algorithms.FlashAttention2(),
                algorithms.GradientClipping(clipping_threshold=1.0)
            ],
            
            # Grokking and evaluation callbacks
            callbacks=[
                GrokkingCallback(
                    patience=1000,  # Check every 1000 steps
                    threshold=0.1   # 10% improvement = grokking
                ),
                ProcrustesCallback(
                    eval_languages=['en', 'es', 'ko'],
                    eval_interval=500
                ),
                LRMonitor(),
                MemoryMonitor(),
                SpeedMonitor(window_size=100)
            ],
            
            # Logging
            loggers=[
                TensorBoardLogger(log_dir='logs/tensorboard')
            ],
            
            # Checkpoint every 1000 steps
            save_interval=f"1000ba",
            save_folder=f"checkpoints/{self.config.experiment_name}",
            
            # Mixed precision
            precision='amp_bf16',
            
            # Gradient accumulation
            device_train_microbatch_size=self.config.training.batch_size,
            
            # Compile model (PyTorch 2.0+)
            compile_config={
                'mode': 'reduce-overhead',
                'fullgraph': False
            }
        )
        
        trainer.fit()
```

---

## FlashAttention Integration Details

### Option A: Via Composer (Automatic)
```python
from composer.algorithms import FlashAttention2

# In your trainer
algorithms=[FlashAttention2()]  # Automatically replaces all attention
```

### Option B: Manual Integration (Maximum Control)
```python
# In your attention layer (models/llama/model.py)
import torch.nn.functional as F

class Attention(nn.Module):
    def forward(self, x):
        q, k, v = self.compute_qkv(x)
        
        # Use PyTorch's native SDPA (auto-selects FlashAttention if available)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True
        )
        
        return self.out_proj(attn_output)
```

### Verify FlashAttention is Active
```python
# Check if FA2 is being used
import torch.backends.cuda

print(f"FlashAttention available: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Math fallback: {torch.backends.cuda.math_sdp_enabled()}")

# Force FlashAttention (raises error if not available)
with torch.backends.cuda.sdp_kernel(
    enable_flash=True,
    enable_math=False,
    enable_mem_efficient=False
):
    output = model(input)
```

---

## Multi-Task Learning Implementation

### Multi-Task Model Wrapper
```python
class MultiTaskLlama(nn.Module):
    def __init__(self, base_model, coherence_dim=1024):
        super().__init__()
        self.base = base_model  # Your existing Llama
        
        # Additional heads
        self.lm_head = nn.Linear(base_model.dim, base_model.vocab_size)
        self.coherence_head = nn.Sequential(
            nn.Linear(base_model.dim * 2, coherence_dim),
            nn.ReLU(),
            nn.Linear(coherence_dim, 1)  # Binary classification
        )
        self.cross_ref_head = nn.Linear(base_model.dim, base_model.dim)
        
    def forward(self, batch):
        task = batch['task_type']
        
        if task == 'lm':
            h = self.base(batch['input_ids'])
            return self.lm_head(h), 'lm'
        
        elif task == 'coherence':
            h1 = self.base(batch['verse1']).mean(dim=1)  # Pool to sentence
            h2 = self.base(batch['verse2']).mean(dim=1)
            combined = torch.cat([h1, h2], dim=-1)
            return self.coherence_head(combined), 'coherence'
        
        elif task == 'cross_ref':
            h_anchor = self.base(batch['anchor']).mean(dim=1)
            h_pos = self.base(batch['positive']).mean(dim=1)
            h_neg = self.base(batch['negative']).mean(dim=1)
            
            # Triplet loss
            d_pos = F.cosine_similarity(h_anchor, h_pos)
            d_neg = F.cosine_similarity(h_anchor, h_neg)
            return (d_pos, d_neg), 'triplet'
```

---

## Hardware Optimization for RTX 4070

### Memory Budget Analysis
```
RTX 4070: 12GB VRAM
RAM: 64GB

Model sizes at BF16:
- 50M params: ~100MB
- 350M params: ~700MB
- 1B params: ~2GB
- 2B params: ~4GB

Overhead (activations, optimizer states, gradients):
- ~3-4x model size during training

Safe maximums:
- 50M: Batch size 16, seq len 4096 ✓
- 350M: Batch size 4, seq len 4096 ✓
- 1B: Batch size 2, seq len 2048 ✓ (with gradient checkpointing)
```

### Optimal Configuration
```yaml
# configs/composer/rtx4070_optimal.yaml
model:
  dim: 768
  n_layers: 48  # Deep-narrow sweet spot for 12GB
  n_heads: 12
  vocab_size: 256
  max_seq_len: 4096

training:
  batch_size: 4
  gradient_accumulation: 4  # Effective batch = 16
  precision: bf16
  compile: true
  
algorithms:
  - flash_attention_2
  - gradient_checkpointing  # Saves ~40% memory
  - fused_adam  # 10-20% speedup

hardware:
  device: cuda
  num_workers: 8  # Dataloader parallelism
  pin_memory: true
```

---

## Migration Checklist

### Phase 1: Foundation (Day 1-2)
- [ ] Install Composer: `pip install mosaicml`
- [ ] Install FlashAttention: `pip install flash-attn --no-build-isolation`
- [ ] Create `training/composer_trainer.py`
- [ ] Wrap existing model in `HuggingFaceModel`
- [ ] Test baseline training matches existing performance

### Phase 2: Optimization (Day 3-4)
- [ ] Enable FlashAttention2 algorithm
- [ ] Enable gradient checkpointing
- [ ] Enable `torch.compile()` (PyTorch 2.0+)
- [ ] Benchmark: measure tokens/sec improvement
- [ ] Verify memory usage stays within 12GB

### Phase 3: Multi-Task (Day 5-7)
- [ ] Implement `MultiTaskLlama` wrapper
- [ ] Implement `MultiTaskDataLoader`
- [ ] Add coherence detection head
- [ ] Add cross-reference head
- [ ] Add cross-lingual paraphrase head
- [ ] Test each task independently

### Phase 4: Grokking Infrastructure (Week 2)
- [ ] Implement `GrokkingCallback`
- [ ] Implement `ProcrustesCallback`
- [ ] Implement `ConceptClusteringCallback`
- [ ] Configure extended training (10k epochs)
- [ ] Test on small model (50M) to verify callbacks work

### Phase 5: Production (Week 3)
- [ ] Full training run on 350M model
- [ ] Monitor grokking metrics
- [ ] Analyze concept clustering
- [ ] Measure cross-lingual alignment
- [ ] Document results

---

## Expected Performance Improvements

### Baseline (Current)
- **Tokens/sec**: ~500-1000 (pure PyTorch)
- **Memory usage**: ~8-10GB (inefficient)
- **Training time (50M, 10k epochs)**: ~8-12 weeks

### After Migration (Composer + FlashAttention)
- **Tokens/sec**: ~2000-4000 (4x speedup from FA2 + compile)
- **Memory usage**: ~6-8GB (gradient checkpointing + efficient kernels)
- **Training time (50M, 10k epochs)**: ~2-3 weeks
- **Training time (350M, 10k epochs)**: ~8-10 weeks (previously impossible)

---

## Fallback: If Composer is Too Heavy

### Minimal NanoGPT-style Integration
```python
# Pure PyTorch + Manual FlashAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharacterLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, 256, bias=False)
    
    def forward(self, x):
        for block in self.transformer:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.dim)
        self.ln2 = nn.LayerNorm(config.dim)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.proj = nn.Linear(config.dim, config.dim)
        self.n_heads = config.n_heads
        self.dim = config.dim
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.dim, dim=2)
        
        # Reshape for multi-head
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        
        # FlashAttention via PyTorch SDPA
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

# Training loop
model = CharacterLM(config).to('cuda')
model = torch.compile(model)  # PyTorch 2.0 compiler

for epoch in range(max_epochs):
    for batch in dataloader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(batch['input'])
            loss = F.cross_entropy(
                logits.view(-1, 256),
                batch['target'].view(-1)
            )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## Questions for Antigravity Agent

When using this prompt with Antigravity or another AI coding assistant, include these context questions:

1. **Framework Selection**: "Based on the Genesis Arbiter requirements (character-level, deep-narrow, multi-task, grokking detection), should I use Composer, NanoGPT-style, or HuggingFace Accelerate? Explain trade-offs."

2. **Migration Strategy**: "What's the minimal refactor to add FlashAttention to my existing Llama implementation in `engine/models/llama/model.py` without breaking current checkpoints?"

3. **Multi-Task Integration**: "How do I implement the multi-task wrapper (`MultiTaskLlama`) to support 70% LM / 15% coherence / 15% cross-lingual training without slowing down the training loop?"

4. **Grokking Callbacks**: "Design a Composer callback that detects grokking by monitoring validation loss plateaus and sudden improvements. Include Procrustes distance calculation for cross-lingual alignment."

5. **Memory Optimization**: "Given RTX 4070 (12GB), what's the largest model I can train with FlashAttention + gradient checkpointing + BF16? Provide memory budget breakdown."

6. **Character-Level Performance**: "Character-level tokenization creates 4-5x longer sequences than BPE. What additional optimizations (beyond FlashAttention) are critical for this use case?"

7. **Configuration System**: "Convert my existing TOML configs in `train_configs/` to Composer YAML format. Preserve all hyperparameters and add FlashAttention settings."

8. **Distributed Training**: "How do I scale from single RTX 4070 to multi-GPU using Composer's FSDP when I eventually need to train 2B+ models?"

---

## Final Recommendation

**Use Composer + Manual FlashAttention Integration**

### Why:
1. ✅ **Best of both worlds**: Framework convenience + full customization
2. ✅ **Proven for research**: MosaicML uses it for ablation studies
3. ✅ **Grokking-friendly**: Easy to extend training indefinitely
4. ✅ **Multi-task native**: Built-in support for your coherence/cross-ref tasks
5. ✅ **Production path**: Easy to deploy trained models
6. ✅ **Active development**: Regular updates, good community support

### Timeline:
- **Week 1**: Core migration + FlashAttention (80% of performance gains)
- **Week 2**: Multi-task learning (enables concept abstraction)
- **Week 3**: Grokking infrastructure (research-specific features)
- **Week 4**: First production training run on 350M model

### Risk Mitigation:
- Keep existing `train.py` as fallback
- Implement new system in parallel (`train_composer.py`)
- Validate parity on small model (50M) before full migration
- Maintain backward compatibility with existing checkpoints

---

## Success Metrics

After migration, you should achieve:
- [ ] 3-4x faster training (tokens/sec)
- [ ] 2x larger models on same hardware (via memory optimization)
- [ ] Multi-task training with <10% overhead
- [ ] Automated grokking detection
- [ ] Cross-lingual alignment metrics every 500 steps
- [ ] Seamless scaling to multi-GPU when needed

**Ready to migrate?** Start with Phase 1 (FlashAttention only) to validate performance gains before committing to full Composer migration.
