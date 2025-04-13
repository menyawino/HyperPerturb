import tensorflow as tf
import numpy as np
import scanpy as sc
import pandas as pd
import os
import json
from pathlib import Path

from hyperpreturb.models import HyperPerturbModel
from hyperpreturb.models.hyperbolic import HyperbolicAdam, QuantumAnnealer
from hyperpreturb.utils.manifolds import PoincareBall

class HyperPerturbInference:
    """
    Inference class for deploying trained HyperPerturb models.
    
    This class handles loading of trained models and provides methods
    for making perturbation predictions and analyzing results.
    """
    
    def __init__(self, model_path):
        """
        Initialize the inference engine with a trained model.
        
        Args:
            model_path: Path to the trained model directory
        """
        self.model_path = model_path
        self.model = None
        self.config = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and configuration."""
        # Load configuration
        config_path = os.path.join(self.model_path, "config.json")
        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found at {config_path}")
        
        # Load custom objects for model loading
        custom_objects = {
            'PoincareBall': PoincareBall,
            'HyperbolicAdam': HyperbolicAdam,
            'QuantumAnnealer': QuantumAnnealer
        }
        
        # Load model
        try:
            self.model = tf.keras.models.load_model(
                os.path.join(self.model_path, 'final_model'),
                custom_objects=custom_objects
            )
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
        
        print(f"Model successfully loaded from {self.model_path}")
    
    def predict_perturbations(self, expression_data, adj_matrix=None, k=5):
        """
        Predict optimal perturbations for given expression states.
        
        Args:
            expression_data: Gene expression data (n_cells, n_genes)
            adj_matrix: Adjacency matrix (optional)
            k: Number of top perturbations to return for each cell
            
        Returns:
            Tuple of (top_k_indices, top_k_scores, predicted_values)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Initialize with a valid model path.")
        
        # Validate input dimensions
        if expression_data.shape[1] != self.config.get("n_genes"):
            raise ValueError(f"Input data has {expression_data.shape[1]} genes, "
                             f"but model expects {self.config.get('n_genes')}.")
        
        # Ensure adj_matrix is provided or create default
        if adj_matrix is None:
            adj_matrix = tf.sparse.eye(expression_data.shape[1])
        
        # Make predictions
        logits, values = self.model.predict((expression_data, adj_matrix), verbose=0)
        
        # Get top k perturbations
        perturbation_scores = tf.nn.softmax(logits)
        top_k_values, top_k_indices = tf.math.top_k(perturbation_scores, k=k)
        
        return top_k_indices.numpy(), top_k_values.numpy(), values.numpy().flatten()
    
    def interpret_perturbations(self, adata, top_k_indices, gene_names=None):
        """
        Interpret perturbation predictions using gene names.
        
        Args:
            adata: AnnData object with gene metadata
            top_k_indices: Indices of top perturbations
            gene_names: Optional list of gene names (uses adata.var_names if None)
            
        Returns:
            DataFrame with gene perturbation interpretations
        """
        if gene_names is None:
            if hasattr(adata, 'var_names'):
                gene_names = adata.var_names
            else:
                raise ValueError("Gene names not provided and not found in AnnData object")
        
        # Convert indices to gene names
        results = []
        for i, cell_perturbations in enumerate(top_k_indices):
            cell_genes = [gene_names[idx] for idx in cell_perturbations]
            results.append({
                'cell_id': i,
                'perturbation_genes': cell_genes
            })
        
        return pd.DataFrame(results)
    
    def simulate_perturbation_effect(self, expression_data, perturbation_indices, 
                                   perturbation_strength=0.8, n_steps=5):
        """
        Simulate the effect of applying the predicted perturbations.
        
        Args:
            expression_data: Initial gene expression data
            perturbation_indices: Indices of genes to perturb
            perturbation_strength: Strength of perturbation (0-1). Default: 0.8
            n_steps: Number of simulation steps. Default: 5
            
        Returns:
            Trajectory of expression states after perturbation
        """
        # Create a copy of expression data to avoid modifying the original
        expression_trajectory = [expression_data.copy()]
        current_state = expression_data.copy()
        
        # Simulate perturbation effect over multiple steps
        for step in range(n_steps):
            # Apply perturbation effect
            for cell_idx, cell_perturbations in enumerate(perturbation_indices):
                # Get perturbation effect for this cell
                effect = np.zeros_like(current_state[cell_idx])
                effect[cell_perturbations] = perturbation_strength
                
                # Update cell state
                updated_cell = current_state[cell_idx] * (1 - effect)
                current_state[cell_idx] = updated_cell
            
            # Store the current state in the trajectory
            expression_trajectory.append(current_state.copy())
        
        return np.array(expression_trajectory)
    
    def visualize_perturbation_trajectory(self, adata, expression_trajectory, 
                                         perturbation_indices, gene_names=None,
                                         reduction='umap'):
        """
        Visualize the trajectory of expression after perturbation.
        
        Args:
            adata: AnnData object with dimensionality reduction
            expression_trajectory: Trajectory from simulate_perturbation_effect
            perturbation_indices: Indices of perturbed genes
            gene_names: Optional list of gene names
            reduction: Dimensionality reduction to use ('umap' or 'pca')
            
        Returns:
            New AnnData object with perturbation trajectory
        """
        try:
            import anndata as ad
            from scipy.sparse import issparse
        except ImportError:
            raise ImportError("Please install anndata and scipy for visualization")
        
        # Create a copy of the original AnnData
        adata_traj = ad.AnnData(
            X=np.vstack([
                adata.X.toarray() if issparse(adata.X) else adata.X,
                expression_trajectory.reshape(-1, expression_trajectory.shape[-1])
            ])
        )
        
        # Add observation metadata
        n_orig_cells = adata.n_obs
        n_traj_steps = expression_trajectory.shape[0]
        n_perturb_cells = expression_trajectory.shape[1]
        
        adata_traj.obs['dataset'] = ['original'] * n_orig_cells + ['perturbed'] * (n_traj_steps * n_perturb_cells)
        adata_traj.obs['time_step'] = [-1] * n_orig_cells + sum([[i] * n_perturb_cells for i in range(n_traj_steps)], [])
        adata_traj.obs['cell_id'] = list(range(n_orig_cells)) + sum([list(range(n_perturb_cells)) for _ in range(n_traj_steps)], [])
        
        # Add gene names if provided
        if gene_names is not None:
            adata_traj.var_names = gene_names
        elif hasattr(adata, 'var_names'):
            adata_traj.var_names = adata.var_names
        
        # Run dimensionality reduction
        if reduction == 'umap':
            sc.pp.neighbors(adata_traj)
            sc.tl.umap(adata_traj)
        elif reduction == 'pca':
            sc.pp.pca(adata_traj)
        
        return adata_traj