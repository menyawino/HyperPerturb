# HyperPerturb: Advanced Hyperbolic Gene Perturbation Framework

HyperPerturb is a state-of-the-art framework for predicting optimal gene perturbations using hyperbolic geometry and deep learning. This framework integrates hyperbolic neural networks, reinforcement learning, and quantum-inspired optimization techniques to model complex gene regulatory relationships in high-dimensional biological data.

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