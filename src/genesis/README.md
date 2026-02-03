# Genesis Core Engine (`src/genesis`)

The `genesis` package is the beating heart of the Arbiter project. It contains the complete source code for training, evaluating, and managing the Genesis language models.

This directory has been professionalized to follow modern Python package standards, ensuring modularity, type safety, and clear separation of concerns.

## 🏗️ Architecture Overview

The system is designed around a modular **Trainer-Callback** architecture:
-   **Trainer (`training/`)**: Handles the training loop, optimization, and state management.
-   **Models (`models/`)**: Pure PyTorch implementations of the Llama architecture with FlashAttention.
-   **Pipelines (`pipelines/`)**: High-level orchestration scripts for end-to-end workflows.
-   **Utils (`utils/`)**: Shared infrastructure for logging, configuration, and checkpoints.

## 📂 Directory Structure

```text
src/genesis/
├── models/                 # Neural network architectures
│   ├── llama/              # Core Transformer implementation (FlashAttention)
│   └── multi_task.py       # Multi-task heads wrapper
├── datasets/               # Data ingestion
│   └── multi_task.py       # Weighted sampling & data loading
├── training/               # Training Loop & Logic
│   ├── trainer.py          # Modular GenesisTrainer class (Research-Focused)
│   ├── analytics.py        # CPU-offloaded loss dynamics (EMA, Stagnation, Slopes)
│   └── scheduler.py        # Learning rate scheduling with Stun Support
├── pipelines/              # Orchestration Workflows
│   ├── long_pipeline.py    # Auto-resume long-term training
│   └── quick_eval.py       # < 1 hour checkpoint assessments
├── evaluation/             # Evaluation Suites
│   └── procedural.py       # Sub-morphemic alignment tests
├── utils/                  # Shared Utilities
│   ├── logger.py           # SQLite + TensorBoard logging
│   └── config_loader.py    # TOML configuration parser
├── train.py                # 🚀 Main Training Entry Point
└── verify.py               # 🔍 System Verification Script
```

## 🔑 Key Components

### 1. Training Engine (`training/trainer.py`)
The `GenesisTrainer` is the project's primary research instrument, featuring:
- **Curriculum Masking**: Ramping difficulty linearly via `wwm_ramp_steps`.
- **Plateau Recovery**: Automated "LR Stun" when stagnation limits are hit.
- **Phase Persistence**: Checkpoints save all research states, including EMA baselines and stagnation counts.
- **Diagnostics**: Multi-lingual perplexity and phase-anchored improvement tracking.

### 2. GPU-Resident Loader (`datasets/byte_loader.py`)
A custom, high-throughput `InfiniteGPULoader` that keeps the entire tokenized corpus in VRAM for zero-bottleneck training. It supports:
- **WWM & Span Masking**: Hardware-accelerated word and sequence masking.
- **Locale Targeting**: High-precision sampling for specific language evaluation.

### 2. Pipelines (`pipelines/`)
Automated workflows that combine training and evaluation.
-   **Long Pipeline**: Runs for days/weeks, handling crashes and auto-resuming.
-   **Quick Eval**: Rapid "Go/No-Go" assessment of checkpoints.

### 3. Models (`models/`)
A highly optimized, FlashAttention-enabled implementation of Llama.
-   **`models.llama`**: The base Transformer.
-   **`models.multi_task_wrapper`**: Adds heads for auxiliary tasks (Coherence, Reference, Paraphrase).

### 4. Utilities (`utils/`)
Infrastructure code used across the project.
-   **`ArbiterLogger`**: A unified logger that writes to both a structured SQLite database (for analysis) and TensorBoard (for real-time monitoring).

## 👩‍💻 Development Guidelines

-   **Imports**: Always use relative imports within the package (e.g., `from ..utils import logger`) and absolute imports for verifying scripts.
-   **Configuration**: Do not hardcode parameters. Retrieve them via `get_config_section("section_name")` from `utils/config_loader.py`.
-   **Type Hinting**: All new function signatures must be fully type-hinted.

## 🚀 Quick Start
To verify the integrity of the source installation, run:

```bash
python src/genesis/verify.py
```
