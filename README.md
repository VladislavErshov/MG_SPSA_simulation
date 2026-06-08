# Mixed-Gradient SPSA: Theory and Reinforcement-Learning Applications

This repository implements the mixed-variable SPSA optimizer from the article below and applies it to a 2D drone trajectory control problem.

## Article

**Mixed-Gradient SPSA: Theory and Reinforcement-Learning Applications**

The paper studies a mixed-parameter reinforcement-learning agent in which the actor block is updated by an exact policy-gradient estimator while the curiosity block is updated by a one-measurement SPSA probe. The analysis separates four layers: a unified gradient-operator framework, abstract Robbins–Siegmund convergence, defect-sensitive speed bounds, and mixed-variable verification. For off-center probing (effective defect order q=1) the balanced choice is gamma=1/4, yielding the mean-square rate O(n^{-1/2}). For centered stencils the defect improves to order q and the rate improves to O(n^{-q/(q+1)}). The results are verified on two RL policies: a linear curiosity model with exact unbiased mixed update and a probe-center biased q-bandit model with explicit defect, fluctuation order, and per-iteration probe cost.

## Quick Start

```bash
pip install -e .
python examples/run_and_plot.py --mode spsa2 --iterations 30
```

See [USAGE.md](USAGE.md) for full documentation.
