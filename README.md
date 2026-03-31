# HyperPerturb

Strict, fail-fast perturbation response modeling with graph-aware hyperbolic deep learning.

Gene regulatory networks are hierarchical — a handful of master regulators sit upstream, and targets fan out in tree-like structures below them. Euclidean embeddings need many dimensions to represent trees without distorting distances. Hyperbolic space (Poincaré ball model) grows exponentially with radius, so it can embed hierarchies more compactly. That's the core idea behind this project.

HyperPerturb takes single-cell CRISPR perturbation data (Perturb-seq screens), embeds genes in the Poincaré ball via graph convolutions over a PPI network, and learns to predict which perturbations most strongly affect which genes. It outputs two things per gene: a distribution over perturbations (policy head) and a scalar sensitivity score (value head).

HyperPerturb is designed as a scientific-computing pipeline with explicit contracts:

- Required inputs are validated early; execution fails fast on missing or inconsistent data.
- Training and inference use a graph-level contract (`gene_features`, `adj_matrix`) with strict shape checks.
- Model serialization is single-path (`final_model.keras`) to keep runs reproducible and auditable.
- No implicit fallback logic is used in core training/inference paths.

## What it does

- Poincaré ball manifold operations (exp map, log map, Möbius addition, geodesic distance)
- Riemannian Adam — converts Euclidean gradients to Riemannian ones before the Adam step
- Graph convolution layers that project through hyperbolic exp/log maps
- Dual-head graph model: **policy head** (per-gene perturbation impact distribution) + **value head** (per-gene sensitivity)
- Cosine-exponential LR decay schedule
- L2 weight regularization
- Deterministic seeding controls for reproducible runs
- Preprocessing pipeline for `.h5ad` Perturb-seq data with optional STRING PPI network

## Scientific Contract

- `adata.obs` must include a perturbation column and explicit control labels.
- `adata.obsm['log_fold_change']` must exist before advanced graph training.
- Advanced trainer requires an explicit gene adjacency matrix with shape `(n_genes, n_genes)`.
- Inference requires both expression matrix `(n_cells, n_genes)` and matching adjacency matrix.
- Contract violations raise hard errors (`ValueError`/`FileNotFoundError`), not silent fallbacks.

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

Control handling is strict: by default `prepare_perturbation_data` expects `ctrl_key="perturbation"` and `ctrl_value="non-targeting"`. If your dataset uses different labels, pass them explicitly.

## Training

### Quick start

```bash
python scripts/train_model.py \
    --rna_path data/raw/FrangiehIzar2021_RNA.h5ad \
    --network_path data/raw/9606.protein.links.full.v11.5.txt
```

This runs the advanced trainer with defaults: 30 epochs, batch size 16, LR 1e-5, curvature 1.0, seed 42.
The CLI is fail-fast and requires explicit paths (no implicit data-path fallback).

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
| `--seed` | `42` | Global random seed for deterministic setup |
| `--max_cells` | `3000` | Cap for memory-constrained machines |
| `--debug` | off | Constant LR + NaN checks |
| `--euclidean_baseline` | off | Ablation: replace hyperbolic convs with Euclidean |

## Inference

```python
from hyperpreturb import HyperPerturbInference
from hyperpreturb.utils.data_loader import load_protein_network, create_adjacency_matrix

inference = HyperPerturbInference("models/saved/hyperperturb-20250413-123456")

gene_list = list(test_gene_names)
network_df = load_protein_network("data/raw/9606.protein.links.full.v11.5.txt")
adj_matrix = create_adjacency_matrix(network_df, gene_list)

top_k_indices, scores, values = inference.predict_perturbations(
    test_expression,
    adj_matrix=adj_matrix,
    k=5,
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

## Scope and limitations

- **Directionality not modeled in policy target**: policy supervision is based on magnitude `|log fold-change|`.
- **Benchmarking not included in this repository**: GEARS/CPA/scGEN comparisons are out of scope for the current codebase.
- **Dataset coverage**: primary workflow is built around Frangieh & Izar (2021)-style perturbation screens.

## Why hyperbolic?

The short version: biological networks have hub-and-spoke / hierarchical structure. In a d-dimensional Poincaré ball, you can embed a tree with N nodes using distortion that scales as O(log N), whereas Euclidean embeddings need distortion scaling polynomially. For gene regulatory networks with clear hierarchical organization, this means the geometry of the embedding space matches the data better.

The longer discussion: Nickel & Kiela (2017) showed this works well for WordNet hierarchies. Chami et al. (2019) extended it to GNNs. Whether it actually helps for perturbation prediction specifically — vs. a well-tuned Euclidean GNN baseline — is an open empirical question that this project hasn't answered yet.

## Citation

Still in development. Feedback and collaboration welcome — open an issue or email me.
