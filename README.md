# HyperPerturb: A Hyperbolic Geometry-Driven Framework for Gene Perturbation Inference

Hyperbolic representation emerges as a compelling choice for modeling gene perturbation networks due to its natural alignment with the inherent hierarchical and scale-free topology of biological regulatory systems. Traditional Euclidean embeddings often struggle to faithfully preserve the complex, tree-like relationships and modular organization prevalent in gene regulatory networks. Hyperbolic space, with its exponential volume growth and negative curvature, provides an efficient geometric substrate that can embed hierarchical structures with minimal distortion, enabling more accurate and interpretable representations of gene-gene interactions and regulatory cascades.

The motive behind adopting hyperbolic geometry lies in capturing these biological hierarchies continuously and at multiple scales, whereby genes at the network center (critical hubs) are embedded near the origin of the hyperbolic space, while peripheral genes with fewer interactions inhabit the outer regions. This facilitates nuanced modeling of gene influence and regulatory proximity, critical for robust perturbation prediction.

To integrate this geometric paradigm effectively, specialized optimization and regularization methods are indispensable. Conventional gradient descent techniques developed for Euclidean spaces become inadequate because they do not account for the curvature and topology of hyperbolic manifolds. Thus, HyperPerturb employs Riemannian optimization methods that respect hyperbolic geometry; these involve curvature-aware gradient calculations and update rules that ensure parameter trajectories remain on the manifold, preventing geometric inconsistencies and improving convergence stability and speed.

Furthermore, the novel neuromorphic-inspired regularization, modeled on spike-timing dependent plasticity (STDP), introduces biologically relevant constraints that promote sparsity and temporal coherence in model parameters. This regularization harmonizes with the hyperbolic framework by encouraging localized, hierarchical connectivity patterns consistent with biological gene regulatory logic. Combined with quantum-inspired annealing schedules, which provide a principled mechanism to avoid local minima and efficiently explore the curved parameter landscape, these methods coalesce to create a training regimen tailored for the unique demands of hyperbolic gene perturbation modeling.

Overall, this synergy of geometric embedding, manifold-appropriate optimization, and biologically motivated regularization forms the foundation of HyperPerturb’s innovative approach, enabling it to capture the complexity of gene regulatory networks with enhanced fidelity and predictive power.

## Key Features

- **Hyperbolic Representation Learning**: Models gene expression data in hyperbolic space to better capture hierarchical relationships
- **Quantum-Inspired Optimization**: Leverages quantum annealing schedules for efficient training
- **Riemannian Geometry**: Custom Riemannian optimizers for accurate gradient updates in hyperbolic space
- **Neuromorphic Regularization**: STDP-inspired regularization for biologically plausible models
- **Adaptive Curriculum Learning**: Progressive complexity increase for robust training
- **Distributed Training**: Efficient multi-GPU training via TensorFlow's distribution strategies
- **High-Performance Computing**: XLA compilation for accelerated model training and inference

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- TensorFlow 2.8+

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/menyawino/HyperPerturb
   cd hyperperturb
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Data Preparation

HyperPerturb supports various single-cell RNA-seq and perturbation datasets. The reference
setup in this repository assumes the Frangieh & Izar (2021) CRISPR perturbation dataset
in `AnnData` (`.h5ad`) format.

### 1. Download reference data and STRING network

This script downloads the RNA, protein, and STRING network files into `data/raw`:

```bash
bash scripts/download_string_and_test_data.sh
```

After running it, you should have at least:

- `data/raw/FrangiehIzar2021_RNA.h5ad`
- `data/raw/FrangiehIzar2021_protein.h5ad`
- a STRING network text file (e.g. `9606.protein.links.full.v11.5.txt`)

### 2. Expected metadata / control labels

The preprocessing pipeline automatically detects control samples based on columns in
`adata.obs`:

- Primary control key: `perturbation`
- Fallback keys (if needed): `perturbation_type`, `perturbation_2`

The control label is chosen in this order:

1. Use `"non-targeting"` if present in the chosen column.
2. Otherwise use `"control"` if present (this matches the Frangieh & Izar data).

You can inspect the available labels with the helper script:

```bash
chmod +x scripts/inspect_labels.sh
./scripts/inspect_labels.sh
```

This will print all columns in `obs` and sample values from `perturbation`,
`perturbation_type`, and `perturbation_2` so you can confirm that the control label
is detected correctly.

### 3. Preprocessing and memory expectations (~10 GB RAM)

For users on machines with around **10 GB of system RAM**, the default preprocessing
and training configuration is tuned to avoid out-of-memory errors:

- In `hyperpreturb/data.py`, `preprocess_data` subsamples to at most **3000 cells**
  (`max_cells=3000`).
- Highly variable genes are limited to 1000, and PCA is run on this subset.
- This keeps the dense matrices used by Scanpy within a safe memory budget.

If you have more RAM and want to use more cells, you can increase `max_cells` in
`preprocess_data` or override it when calling custom preprocessing functions, but for
10 GB we recommend keeping the default.

### 4. Using the built-in preprocessing pipeline

The main training script `scripts/train_model.py` uses the helper
`load_and_preprocess_perturbation_data` from `hyperpreturb.data`. If you want to
preprocess your own dataset, follow a similar pattern:

```python
from hyperpreturb.data import preprocess_data, prepare_perturbation_data

# Basic preprocessing with a 10 GB-friendly cell cap
adata = preprocess_data(
    "path/to/your/data.h5ad",
    output_path="data/processed/your_processed_data.h5ad",
    max_cells=3000,
)

# Add perturbation targets and log-fold changes relative to control samples
adata = prepare_perturbation_data(adata)
```

## Model Training

There are two main training entry points:

1. A **command-line script** (`scripts/train_model.py`) that handles data loading,
    preprocessing, and training end-to-end.
2. The lower-level **Python API** in `hyperpreturb.models.train` for custom workflows.

### 1. Command-line training script (recommended)

The script `scripts/train_model.py` is configured with defaults that work on a
machine with ~10 GB of RAM. It:

- Loads `data/raw/FrangiehIzar2021_RNA.h5ad` (and optional protein data) by default.
- Subsamples to at most 3000 cells for memory safety.
- Detects control samples from `adata.obs['perturbation']` (using `"control"` unless
   `"non-targeting"` is present).
- Trains the HyperPerturb model using the **advanced** hyperbolic trainer by default.

From the repository root:

```bash
python scripts/train_model.py
```

Key CLI options (all have 10 GB-friendly defaults):

- `--rna_path`: Path to RNA `.h5ad` file (default: `data/raw/FrangiehIzar2021_RNA.h5ad`).
- `--protein_path`: Optional protein `.h5ad` file.
- `--network_path`: Optional STRING network file.
- `--output_dir`: Directory to save models and logs.
- `--trainer`: `advanced` (default) or `simple`.
- `--epochs`: Default `30`.
- `--batch_size`: Default `16`.
- `--embedding_dim`: Default `16`.
- `--hidden_dim`: Default `32`.
- `--max_cells`: Default `3000` (preprocessing cell cap).

Example of a conservative run on a 10 GB machine:

```bash
python scripts/train_model.py \
   --trainer advanced \
   --epochs 30 \
   --batch_size 16 \
   --max_cells 3000
```

If you have more memory available, you can gradually increase `--max_cells` or
`--batch_size`. If you encounter an out-of-memory kill from the OS, reduce these
values again.

### 2. Python API example

For more control, you can call the training API directly:

```python
from hyperpreturb.data import load_and_preprocess_perturbation_data
from hyperpreturb.models.train import train_model

# Load and preprocess data with built-in defaults (including max_cells=3000)
adata, adj_matrix = load_and_preprocess_perturbation_data(
      "data/raw/FrangiehIzar2021_RNA.h5ad",
      network_path="data/raw/protein.links.full.v11.5.txt",  # adjust filename as needed
)

# Train HyperPerturb model (advanced trainer)
model, history = train_model(
      adata,
      adj_matrix=adj_matrix,
      epochs=30,
      batch_size=16,
      learning_rate=3e-4,
      curvature=1.0,
)
```

## Inference

Use a trained model to predict optimal perturbations:

```python
from hyperpreturb import HyperPerturbInference

# Initialize inference engine
inference = HyperPerturbInference("models/saved/hyperperturb-20250413-123456")

# Predict perturbations
test_expression = adata[0:10].X.toarray()
top_k_indices, scores, values = inference.predict_perturbations(
    test_expression,
    k=5  # Top-5 perturbations
)

# Interpret results
results_df = inference.interpret_perturbations(adata, top_k_indices)
print(results_df)
```

## Framework Architecture

```
hyperpreturb/
├── __init__.py         # Main package initialization
├── data.py             # Data download and preprocessing
├── models/
│   ├── __init__.py     # Core model architecture
│   ├── hyperbolic.py   # Hyperbolic layers and optimizers
│   ├── train.py        # Training pipeline
│   └── inference.py    # Inference engine
└── utils/
    ├── __init__.py     # Utility functions
    ├── data_loader.py  # Data loading utilities
    └── manifolds.py    # Riemannian manifold implementations
```

## Hyperbolic Space Advantages

HyperPerturb leverages hyperbolic geometry for several key advantages:

1. **Efficient Embedding of Hierarchies**: Hyperbolic space can embed hierarchical structures with lower distortion than Euclidean space
2. **Continuous Hierarchy Representation**: Enables smooth transitions between hierarchical levels
3. **Expressive Power**: The curvature parameter provides an additional degree of freedom for modeling
4. **Distance Preservation**: Better preserves distances between data points with hierarchical relationships

## Citation

This is still in the development phase, so you are free to use it as you wish
I appreciate feedback and collaboration oppurtunities!
