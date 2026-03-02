# HyperPerturb

Predicting gene perturbation effects using hyperbolic embeddings and graph neural networks.

Gene regulatory networks are hierarchical — a handful of master regulators sit upstream, and targets fan out in tree-like structures below them. Euclidean embeddings need many dimensions to represent trees without distorting distances. Hyperbolic space (Poincaré ball model) grows exponentially with radius, so it can embed hierarchies more compactly. That's the core idea behind this project.

HyperPerturb takes single-cell CRISPR perturbation data (Perturb-seq screens), embeds genes in the Poincaré ball via graph convolutions over a PPI network, and learns to predict which perturbations most strongly affect which genes. It outputs two things per gene: a distribution over perturbations (policy head) and a scalar sensitivity score (value head).

> **Honest status note:** This is a work in progress. The hyperbolic pipeline is numerically fragile — training often diverges to NaN after epoch 1 (see [TRAINING_RUNS.md](TRAINING_RUNS.md)). The architecture needs more stabilization before producing reliable results. No benchmarks against GEARS, CPA, or scGEN have been run yet.

## What it actually does

- Poincaré ball manifold operations (exp map, log map, Möbius addition, geodesic distance)
- Riemannian Adam — converts Euclidean gradients to Riemannian ones before the Adam step
- Graph convolution layers that project through hyperbolic exp/log maps
- Dual-head graph model: **policy head** (per-gene perturbation impact distribution) + **value head** (per-gene sensitivity)
- Cosine-exponential LR decay schedule
- L2 weight regularization
- Preprocessing pipeline for `.h5ad` Perturb-seq data with optional STRING PPI network

## Installation

You need Python 3.8+ and ideally a GPU.

```bash
git clone https://github.com/menyawino/HyperPerturb
cd HyperPerturb
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

We use the Frangieh & Izar (2021) CRISPR perturbation dataset.

```bash
bash scripts/download_string_and_test_data.sh
```

This downloads RNA + protein `.h5ad` files and the STRING PPI network into `data/raw/`.

### Preprocessing

Preprocessing subsamples to 3000 cells by default (tuned for ~10 GB RAM machines), selects 2000 highly variable genes, runs PCA, and computes log fold-changes relative to control cells.

```python
from hyperpreturb.data import preprocess_data, prepare_perturbation_data

adata = preprocess_data("data/raw/FrangiehIzar2021_RNA.h5ad", max_cells=3000)
adata = prepare_perturbation_data(adata)
```

Control detection looks for `"non-targeting"` first, then falls back to `"control"` in the `perturbation` column of `adata.obs`.

## Training

### Quick start

```bash
python scripts/train_model.py
```

This runs the advanced trainer with defaults: 30 epochs, batch size 16, LR 1e-5, curvature 1.0.

### Python API

```python
from hyperpreturb.data import load_and_preprocess_perturbation_data
from hyperpreturb.models.train import train_model

adata, adj_matrix = load_and_preprocess_perturbation_data(
    "data/raw/FrangiehIzar2021_RNA.h5ad",
    network_path="data/raw/protein.links.full.v11.5.txt",
)

model, history = train_model(
    adata,
    adj_matrix=adj_matrix,
    epochs=30,
    batch_size=16,
    learning_rate=3e-4,
    curvature=1.0,
)
```

### CLI options

| Flag | Default | Notes |
|------|---------|-------|
| `--trainer` | `advanced` | `simple` uses plain Keras MSE regression |
| `--epochs` | `30` | |
| `--batch_size` | `16` | |
| `--learning_rate` | `1e-5` | Higher values cause NaN divergence |
| `--curvature` | `1.0` | Poincaré ball curvature |
| `--max_cells` | `3000` | Cap for memory-constrained machines |
| `--debug` | off | Constant LR + NaN checks |
| `--euclidean_baseline` | off | Ablation: replace hyperbolic convs with Euclidean |

## Inference

```python
from hyperpreturb import HyperPerturbInference

inference = HyperPerturbInference("models/saved/hyperperturb-20250413-123456")
top_k_indices, scores, values = inference.predict_perturbations(
    test_expression, k=5
)
```

## Project layout

```
hyperpreturb/
├── __init__.py
├── data.py             # Data loading, preprocessing, fold-change computation
├── models/
│   ├── __init__.py     # HyperPerturbModel, graph conv layers, regularizers
│   ├── hyperbolic.py   # Riemannian optimizer, LR schedule, hyperbolic dense/attention
│   ├── train.py        # Training loop (advanced trainer)
│   └── inference.py    # Model loading and prediction
└── utils/
    ├── data_loader.py  # STRING network parsing, adjacency matrix construction
    └── manifolds.py    # Poincaré ball: distance, expmap, logmap, egrad2rgrad
```

## Known issues and limitations

- **NaN divergence**: hyperbolic graph convolutions frequently produce NaN after 1-2 epochs. Currently mitigated with aggressive norm clipping, but not fully solved.
- **No validation during graph-level training**: the advanced trainer fits a single graph with `validation_split=0.0`, so early stopping / LR reduction callbacks don't actually trigger.
- **Reward definition ignores directionality**: policy targets are based on |log fold-change| magnitude, not whether a gene is up- or down-regulated.
- **No benchmarks**: haven't compared against GEARS (Roohani et al. 2023), CPA (Lotfollahi et al. 2023), or other perturbation prediction methods.
- **No test suite**: `pytest` is in requirements but there are no actual tests.
- **Single dataset**: only tested on Frangieh & Izar (2021); generalization to other screens is unknown.

## Why hyperbolic?

The short version: biological networks have hub-and-spoke / hierarchical structure. In a d-dimensional Poincaré ball, you can embed a tree with N nodes using distortion that scales as O(log N), whereas Euclidean embeddings need distortion scaling polynomially. For gene regulatory networks with clear hierarchical organization, this means the geometry of the embedding space matches the data better.

The longer discussion: Nickel & Kiela (2017) showed this works well for WordNet hierarchies. Chami et al. (2019) extended it to GNNs. Whether it actually helps for perturbation prediction specifically — vs. a well-tuned Euclidean GNN baseline — is an open empirical question that this project hasn't answered yet.

## Citation

Still in development. Feedback and collaboration welcome — open an issue or email me.
