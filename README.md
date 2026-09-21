# Membership Inference Attacks on Bayesian Networks

This repository contains the experimental codebase developed for my Master's thesis **"Enhancing Membership Inference Attacks against Bayesian Networks in Data-Scarce Scenarios"**. It implements and evaluates several variants of the **Log-Likelihood Ratio (LLR)** attack, with a focus on the **Filtered LLR (FLLR)**, the novel contribution of the thesis, and compares it against the standard LLR baseline from Murakonda et al. (2021).

---

## Overview

Membership Inference Attacks attempt to determine whether a given data record was part of the training dataset of a machine learning model. In this work, the target models are **discrete Bayesian Networks**, and the attack strategy is based on the **log-likelihood ratio** between a reference model and the target model.

The main contribution of this work is the development of the **Filtered LLR**, a variant that improves attack performance by directly classifying instances as non-members when their LLR score is considered unreliable due to low-frequency parent configurations in the training set.

---

## Repository Structure

```
.
├── requirements.txt
│
├── src/
│   └── mia/
│       ├── InferenceEngine.py      # Wrapper for probabilistic inference
│       ├── llr/                    # LLR attack implementations
│       │   ├── AbstractLLR.py
│       │   ├── StandardLLR.py
│       │   ├── NodewiseLLR.py
│       │   ├── FilteredLLR.py      # Main contribution (FLLR)
│       │   ├── WeightedLLR.py
│       │   └── FilteredWeightedLLR.py
│       └── weighting/              # Weighting strategies for weighted variants
│           ├── WeightingStrategy.py
│           ├── EntropyWeightingStrategy.py
│           ├── KLWeightingStrategy.py
│           ├── ParentsCountsWeightingStrategy.py
│           └── ParentsProbabilityWeightingStrategy.py
│
└── experiment_suite/
    ├── run.py                      # Entry point to launch experiments
    ├── plots.ipynb                 # Notebook for boxplots
    ├── build_report.ipynb          # Notebook for HTML report generation (ROC curves)
    ├── core/                       # Experiment pipeline logic
    │   ├── benchmark_engine.py
    │   ├── synthetic_engine.py
    │   ├── experiment.py
    │   ├── generation.py
    │   ├── estimators.py
    │   ├── metrics.py
    │   ├── sample_size.py
    │   └── utils.py
    └── experiments/                # Experiment configs and results (JSON)
        ├── eval_balanced/
        │   ├── benchmark/          # Real-world BNs (Asia, Cancer, Alarm, ...)
        │   └── synthetic/          # Random and simple synthetic BNs
        └── eval_unbalanced/
            ├── benchmark/
            └── synthetic/
```

---

## Attack Methods

All attack variants implement the `AbstractLLR` interface and are located in `src/mia/llr/`.

### StandardLLR
Direct implementation of the LLR attack from Murakonda et al. (2021). Computes the log-likelihood ratio using joint inference over the full instance:

$$\text{LLR}(x) = \log P(x \mid \hat{\theta}_R) - \log P(x \mid \hat{\theta}_T)$$

### NodewiseLLR
A factored variant that decomposes the LLR node by node, exploiting the conditional independence structure of the Bayesian Network:

$$\text{LLR}(x) = \sum_{i} \log \frac{P(x_i \mid \text{pa}(x_i), \hat{\theta}_R)}{P(x_i \mid \text{pa}(x_i), \hat{\theta}_T)}$$

Node-level LLR values are cached to avoid redundant computations.

### FilteredLLR *(main contribution)*
Extends `NodewiseLLR` with a **filtering mechanism** that detects unreliable node contributions. A node's contribution is filtered if its parent configuration is too rare in the training set — formally, if:

$$P(\pi_i \mid \hat{\theta}_T) \cdot |\Pi_i| \cdot |T| \leq 2$$

where $|\Pi_i|$ is the number of possible parent configurations and $|T|$ is the estimated training set size. Filtered instances receive a monotonically increasing penalty instead of a log-ratio score, separating them from legitimate members.

### WeightedLLR
Extends `NodewiseLLR` with a pluggable weighting strategy, allowing each node's contribution to be scaled according to a relevance criterion (entropy, KL divergence, parent counts, or parent probability).

$$\text{LLR}(x) = \sum_{i} \log \frac{P(x_i \mid \text{pa}(x_i), \hat{\theta}_R)}{P(x_i \mid \text{pa}(x_i), \hat{\theta}_T)} \cdot w_i(x_i, \pi_i) \cdot$$

### FilteredWeightedLLR
Combines the filtering mechanism of `FilteredLLR` with the weighting mechanism of `WeightedLLR`. This and the weighted variants are included for completeness; results in the thesis focus on `StandardLLR` vs. `FilteredLLR`.

---

## Experimental Pipeline

The evaluation follows the procedure below for each experiment configuration:

1. **Network setup**: load a benchmark Bayesian Network (`.bif` format) or generate synthetic networks.
2. **For each coverage target** $c \in \{0.1, 0.25, 0.5, 0.75\}$:
   - Determine the population size $N_c$ needed to achieve coverage $c$.
   - Sample a population $P$ of size $N_c$ from the network.
3. **Repeat 20 times**:
   - Partition $P$ into reference set $R$, training set $T$, and evaluation set $E$.
   - Estimate $\hat{\theta}_R$ and $\hat{\theta}_T$ from $R$ and $T$.
   - For each attack method, compute LLR scores on $E$ and derive the ROC curve.
4. **Average** ROC curves over 20 repetitions per method, network, and coverage target.

Experiments are split into two evaluation settings:
- **`eval_balanced`**: balanced member/non-member ratio in $E$.
- **`eval_unbalanced`**: unbalanced ratio, closer to a realistic attack scenario.

Benchmark networks include: *Asia*, *Cancer*, *Earthquake*, *Survey*, *Sachs* (small), *Alarm*, *Child*, *Insurance* (medium).

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

## Usage

### Running an experiment

```bash
python -m experiment_suite.run --exp experiment_suite/experiments/eval_balanced/benchmark/small_networks/asia
```

The script auto-detects whether the experiment directory contains a `.bif` file (benchmark) or not (synthetic), and writes results to `experiment_results.json` in the same directory.

### Generating plots

Open `experiment_suite/plots.ipynb` in Jupyter and run all cells. The notebook reads `experiment_results.json` files and produces boxplots of the ΔpAUC@0.2 (FLLR − LLR) across experiments, coverage targets, and evaluation settings.

### Building the HTML report

Open `experiment_suite/build_report.ipynb` to generate the full interactive HTML evaluation report, comprising ROC curves for all methods, networks, coverage targets, and evaluation settings.

---

## Reference

The LLR attack implemented in this repository is based on the work of Murakonda et al. (2021), which this thesis extends, evaluates, and improves upon:

Sasi Kumar Murakonda, Reza Shokri, and George Theodorakopoulos. Quantifying the privacy risks of learning high-dimensional graphical models. In Arindam Banerjee and Kenji Fukumizu, editors, Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 2287–2295. PMLR, 13–15 Apr 2021.