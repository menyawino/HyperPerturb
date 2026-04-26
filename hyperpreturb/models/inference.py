import tensorflow as tf
import numpy as np
import pandas as pd
import os
import json

from hyperpreturb.models import EuclideanPerturbModel, HyperPerturbModel, SignedHyperPerturbModel
from hyperpreturb.models.hyperbolic import HyperbolicAdam, QuantumAnnealer
from hyperpreturb.models.training_utils import build_graph_inputs
from hyperpreturb.utils.manifolds import PoincareBall

class HyperPerturbInference:
    """Load a trained model and run perturbation predictions."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.config = None
        self._load_model()

    def _load_model(self):
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
            'QuantumAnnealer': QuantumAnnealer,
            'HyperPerturbModel': HyperPerturbModel,
            'SignedHyperPerturbModel': SignedHyperPerturbModel,
            'EuclideanPerturbModel': EuclideanPerturbModel,
        }
        
        # Load model
        try:
            self.model = tf.keras.models.load_model(
                os.path.join(self.model_path, 'final_model.keras'),
                custom_objects=custom_objects,
                compile=False,
            )
        except (OSError, ValueError, TypeError) as e:
            raise ValueError(f"Failed to load model: {str(e)}")
        
        print(f"Model successfully loaded from {self.model_path}")
    
    def predict_perturbations(self, expression_data, adj_matrix, k=5):
        """Get top-k predicted perturbation genes for each response gene.

        Args:
            expression_data: Cell x gene matrix used to derive graph node features.
            adj_matrix: Gene x gene adjacency matrix used during training.
            k: Number of top perturbation genes per response gene.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Initialize with a valid model path.")

        if adj_matrix is None:
            raise ValueError("adj_matrix is required and cannot be None.")

        # Validate and build graph-level inputs from cell x gene matrix.
        expression_data = np.asarray(expression_data, dtype=np.float32)
        if expression_data.ndim != 2:
            raise ValueError(f"expression_data must be rank-2 (cells x genes), got shape {expression_data.shape}")

        if expression_data.shape[1] != self.config.get("n_genes"):
            raise ValueError(f"Input data has {expression_data.shape[1]} genes, "
                             f"but model expects {self.config.get('n_genes')}.")

        class _ExpressionWrapper:
            def __init__(self, matrix):
                self.X = matrix
                self.n_vars = matrix.shape[1]

        x_gene, x_adj = build_graph_inputs(_ExpressionWrapper(expression_data), adj_matrix=adj_matrix)

        # Make predictions
        signed_effects, values = self.model.predict((x_gene, x_adj), verbose=0)

        # Rank perturbation genes by highest positive predicted effect.
        perturbation_scores = tf.convert_to_tensor(signed_effects, dtype=tf.float32)
        n_perts = int(perturbation_scores.shape[-1])
        if k > n_perts:
            raise ValueError(f"k={k} exceeds number of perturbations ({n_perts})")
        top_k_values, top_k_indices = tf.math.top_k(perturbation_scores, k=k)

        return top_k_indices.numpy()[0], top_k_values.numpy()[0], np.asarray(values)[0, :, 0]
    
    def interpret_perturbations(self, adata, top_k_indices, gene_names=None):
        """Map perturbation indices back to gene names for each response gene.

        If ``gene_names`` is not provided, this first uses the saved model's
        gene_names from config.json and then falls back to ``adata.var_names``.
        """
        if gene_names is None:
            if self.config is not None and self.config.get("gene_names"):
                gene_names = self.config["gene_names"]
            elif hasattr(adata, 'var_names'):
                gene_names = adata.var_names
            else:
                raise ValueError("Gene names not provided and not found in AnnData object")
        gene_names = list(gene_names)
        
        # Each row corresponds to a response gene in gene space, not to an input cell.
        results = []
        for i, gene_perturbations in enumerate(top_k_indices):
            perturbation_genes = [gene_names[idx] for idx in gene_perturbations]
            results.append({
                'response_gene': gene_names[i],
                'perturbation_genes': perturbation_genes
            })
        
        return pd.DataFrame(results)
    
    def simulate_perturbation_effect(self, expression_data, perturbation_indices,
                                   perturbation_strength=0.8, n_steps=5):
        """Naive perturbation simulation: scales target genes down over n_steps.

        This is a very rough approximation -- just multiplies expression by
        (1 - strength) at perturbation targets. Not a real biological simulation.
        """
        if len(perturbation_indices) != expression_data.shape[0]:
            raise ValueError(
                "perturbation_indices must align with rows of expression_data. "
                "Do not pass the gene-space output of predict_perturbations directly into simulate_perturbation_effect."
            )

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
        """UMAP/PCA visualization of original + perturbed expression states."""
        try:
            import anndata as ad
            import scanpy as sc
            from scipy.sparse import issparse
        except ImportError:
            raise ImportError("Please install anndata, scanpy, and scipy for visualization")
        
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