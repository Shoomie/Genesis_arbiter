# Contributing to Genesis

Thank you for your interest in contributing to the Genesis project! This document provides guidelines for contributing to this research prototype.

## 🎯 Project Vision

Genesis explores whether large language models trained exclusively on biblical text can develop logical reasoning capabilities through semantic coherence rather than dataset scale. We welcome contributions that advance this research question.

## 🤝 Ways to Contribute

### 1. Code Contributions
- **Training Techniques**: Implement strategies from [`docs/research/logical_refinement_strategies.md`](docs/research/logical_refinement_strategies.md)
- **Evaluation Metrics**: Build benchmarks for analogical reasoning, typological matching
- **Architecture Experiments**: Test novel configurations (dual-stream, hierarchical embeddings)
- **Optimization**: Improve training efficiency, reduce VRAM requirements

### 2. Documentation
- **Tutorials**: Create guides for specific use cases
- **Research Papers**: Contribute theoretical analyses or empirical results
- **Code Examples**: Add Jupyter notebooks demonstrating techniques

### 3. Testing & Analysis
- **Corpus Analysis**: Explore linguistic patterns in biblical text
- **Model Evaluation**: Test reasoning capabilities on benchmark tasks
- **Ablation Studies**: Systematically test architectural variations

### 4. Bug Reports & Feature Requests
Use GitHub Issues with clear descriptions and reproduction steps.

## 📋 Contribution Guidelines

### Code Style
- **Python**: Follow PEP 8 conventions
- **Documentation**: Include docstrings for all functions/classes
- **Comments**: Explain "why" not "what" for complex logic

### File Organization
```
genesis_prototype/
├── engine/          # Core training code - keep minimal and focused
├── scripts/         # Standalone utilities - one script, one purpose
├── docs/
│   ├── research/   # Technical analyses (10+ pages)
│   ├── reference/  # Quick guides (1-5 pages)
│   └── roadmap/    # Implementation planning
```

### Commit Messages
Use clear, descriptive commit messages:
```
✅ Good: "Add weighted masking for logical connectives"
❌ Bad: "Fix stuff"
```

### Pull Request Process
1. **Fork** the repository
2. **Create a branch** with a descriptive name (`feature/dynamic-masking`)
3. **Make focused changes** - one feature per PR
4. **Test thoroughly** - verify code runs without errors
5. **Update documentation** - reflect changes in relevant docs
6. **Submit PR** with clear description of changes

## 🧪 Research Contributions

### Experimental Protocol
If proposing new training techniques:
1. **Justify theoretically** - explain why it should improve reasoning
2. **Define metrics** - how will success be measured?
3. **Baseline comparison** - compare against standard training
4. **Document results** - create walkthrough in `docs/reference/`

### Sharing Results
- **Checkpoints**: Host large model files externally (Hugging Face, Google Drive)
- **Logs**: Include TensorBoard/Wandb links in PRs
- **Visualizations**: Add plots to `docs/reference/` for significant findings

## 🔬 Areas of Active Research

### High Priority
1. **Dynamic Masking Implementation** (Phase 2)
   - See [`docs/research/dynamic_masking_assessment.md`](docs/research/dynamic_masking_assessment.md)
   - Start with Strategy 2: Weighted Masking
   
2. **Evaluation Framework** (Phase 3)
   - Build premise-conclusion matching tests
   - Create typological reasoning benchmark
   
3. **Tower of Truth Training** (Phase 4)
   - Investigate 144-layer depth effects
   - Compare with Microscope baseline

### Open Questions
- Can typological reasoning transfer to modern analogies?
- What is the minimal viable corpus size for reasoning?
- Do models develop hierarchical concept representations?

## 🚫 What We're Not Looking For

- **General-purpose LLM features**: This is domain-specific research
- **Non-biblical corpora** (yet): Focus on single-worldview approach first
- **Production deployment code**: This is a research prototype
- **Massive architectural changes**: Incremental experiments preferred

## 📞 Communication

- **GitHub Issues**: Bug reports, feature requests
- **Discussions**: Theoretical questions, design debates
- **Pull Requests**: Code contributions with clear scope

##⚖️ Code of Conduct

- **Respectful dialogue**: Focus on research merit, not ideology
- **Academic integrity**: Cite sources, acknowledge prior work
- **Collaborative spirit**: Help others understand complex concepts

## 📚 Getting Started

1. **Read the docs**:
   - [`docs/reference/QUICK_REFERENCE.md`](docs/reference/QUICK_REFERENCE.md) - Project overview
   - [`docs/research/theoretical_foundations.md`](docs/research/theoretical_foundations.md) - Core hypotheses
   - [`docs/roadmap/README.md`](docs/roadmap/README.md) - Current priorities

2. **Set up environment**:
   ```bash
   git clone https://github.com/[your-repo]/Genesis.git
   cd Genesis
   pip install -r requirements.txt  # If we add one
   python run.py  # Launch central menu
   ```

3. **Pick a task**: Check Issues labeled `good-first-issue` or `help-wanted`

4. **Ask questions**: Don't hesitate to open a Discussion for clarification

---

## 🙏 Acknowledgments

Contributors will be acknowledged in:
- `README.md` contributors section
- Research paper acknowledgments (if published)
- Release changelogs for significant features

Thank you for helping advance the science of specialized language models!

---

**Note**: This is an experimental research project. Contributions should prioritize scientific rigor over production readiness.
