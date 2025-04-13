"""
HyperPerturb: A framework for analyzing and predicting gene perturbations in hyperbolic space.

The HyperPerturb package provides tools for analyzing single-cell RNA-sequencing data
and predicting optimal genetic perturbations using hyperbolic geometry and graph neural networks.

Main Components:
- Data: Functions for loading and preprocessing perturbation datasets
- Models: Hyperbolic neural network models and training procedures
- Utils: Utility functions for data processing and visualization
"""

__version__ = '0.1.0'

# Import key modules
from hyperpreturb.data import (
    download_data, download_string_network, preprocess_data, 
    prepare_perturbation_data, load_and_preprocess_perturbation_data
)
from hyperpreturb.utils.manifolds import PoincareBall
from hyperpreturb.models import HyperPerturbModel
from hyperpreturb.models.hyperbolic import (
    HyperbolicAdam, QuantumAnnealer, HyperbolicDense, HyperbolicAttention
)
from hyperpreturb.models.train import train_model
from hyperpreturb.models.inference import HyperPerturbInference