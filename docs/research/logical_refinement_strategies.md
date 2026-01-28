# Prioritized Strateges for Resonant Blade: Logical Refinement

This list is ordered by **Axiomatic Impact**: from techniques that directly force logical/causal computation to those that provide general architectural stability.

## Tier 1: Causal & Axiomatic Anchors (Highest Impact)
1. **Dynamic Masking on Logical Connectives**: High-probability masking of transition tokens (`Therefore`, `Because`, `So`) forces the model to compute the logical necessity between premises and conclusions.
2. **Logos Bridge Supervision**: Adding auxiliary LM heads to layers 70-80 forces the "Deep Tower" to resolve reasoning early, creating a compressed representation of logic in the designated "Logos Bridge."
3. **Parallel Account Contrast**: Batching parallel narratives (e.g., Kings vs. Chronicles) forces the model to find the invariant "Truth" across different linguistic expressions.
4. **Intra-Chapter Verse Permutation**: Training the model to recognize chronologically or logically correctly ordered verse sequences forces an understanding of complex narrative/argument flow.
5. **JH-Token Weight Lock**: Freezing the embedding for "Jehovah" after initial convergence anchors the entire semantic geometry of the model around a fixed axiomatic point.

## Tier 2: Structural & Distributional Alignment
6. **Axiomatic Oversampling**: Increasing the frequency of high-density books (Romans, Proverbs, Genesis) ensures the model encounters more "Axiomatic Seeds" per epoch.
7. **End-of-Verse Penalty**: Weighting the loss 1.2x at the end of every verse ensures the model pays maximum attention to the culmination of a Scriptural point.
8. **Structural Boundary Anchor**: Locking the `[BOOK CHAPTER]` markers ensures the model maps all logic to a rigid, unmoving physical structure.
9. **Attention Head Regularization**: Penalizing "fuzzy" or broad attention coefficients forces the model to create sharp, direct links between logically connected tokens.
10. **DropPath (Stochastic Depth)**: Occasionally skipping layers during training encourages the model to develop robust, multi-path reasoning circuits.

## Tier 3: Granularity & Dependency Tuning
11. **Rotary Embedding (RoPE) Scale Tuning**: Increasing the RoPE base to `50000` allows the model to perceive dependencies across dozens of verses, vital for complex apostolic arguments.
12. **RMSNorm Epsilon Reduction**: Lowering `eps` to `1e-8` allows the model to differentiate between very subtle activation differences in the 144-layer stack.
13. **Weight Decay on FF-Intermediate**: Prevents the model from using its 124M parameters for brute-force memorization, favoring generalized semantic extraction.
14. **Learning Rate Layer-Decay**: Slower refinement of output layers ensures that the final "expression" of logic is stable and precise.
15. **Embedding Gradient Clipping**: Keeps the foundational token space stable while the deeper reasoning layers adjust.

## Tier 4: Stability & Gradient Integrity (General Optimization)
16. **Batch Accumulation Scaling**: Smoothes the gradient landscape for the 144-layer stack, preventing "jitter" in complex logical regions.
17. **FP32 Accumulation for Foundational Verses**: Preserves perfect gradient fidelity for the "Opening Seeds" of each book.
18. **Temperature Annealing**: Forces the model to commit to the most "Biblically Probable" next token as training progresses.
19. **Cyclical Learning Rate**: Provides general stability for training deep architectures on small, high-density datasets.
20. **AdamW Beta2 Adjustment**: Makes the optimizer more sensitive to the "Axiomatic Friction" encountered during logical transitions.
