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

HyperPerturb supports various single-cell RNA-seq and perturbation datasets:

1. Download preprocessed datasets:
   ```bash
   bash scripts/download_string.sh
   ```

2. Preprocess your own data:
   ```python
   from hyperpreturb import preprocess_data
   
   # Load and preprocess data
   adata = preprocess_data("path/to/your/data.h5ad", 
                          output_path="data/processed/your_processed_data.h5ad")
   ```

## Model Training

Train the HyperPerturb model with customizable parameters:

```python
from hyperpreturb import load_and_preprocess_perturbation_data, train_model

# Load data
adata, adj_matrix = load_and_preprocess_perturbation_data(
    "data/raw/FrangiehIzar2021_RNA.h5ad",
    network_path="data/raw/9606.protein.links.full.v11.5.txt"
)

# Train model
model, history = train_model(
    adata,
    adj_matrix=adj_matrix,
    epochs=200,
    batch_size=128,
    learning_rate=3e-4,
    curvature=1.0
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
