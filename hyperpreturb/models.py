import tensorflow as tf
import numpy as np
from hyperpreturb.models.hyperbolic import HyperbolicLayer, HyperbolicPoincareBall
from hyperpreturb.utils.manifolds import PoincareBall

class HyperbolicPerturbationModel(tf.keras.Model):
    """
    A model for predicting gene expression changes in response to perturbations,
    using hyperbolic embeddings to capture the hierarchical structure of gene regulatory networks.
    """
    def __init__(self, n_genes, n_perturbations, embedding_dim=32, hidden_dim=64, adj_matrix=None):
        """
        Initialize the HyperbolicPerturbationModel.
        
        Args:
            n_genes: Number of genes in the dataset
            n_perturbations: Number of possible perturbations
            embedding_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden layers
            adj_matrix: Adjacency matrix from PPI network (optional)
        """
        super(HyperbolicPerturbationModel, self).__init__()
        
        self.n_genes = n_genes
        self.n_perturbations = n_perturbations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Initialize manifold
        self.manifold = PoincareBall(dim=embedding_dim)
        
        # Gene embeddings in hyperbolic space
        self.gene_embeddings = self.add_weight(
            name="gene_embeddings",
            shape=[n_genes, embedding_dim],
            initializer=tf.keras.initializers.RandomUniform(
                minval=-0.001, maxval=0.001
            ),
            trainable=True
        )
        
        # Perturbation embeddings in hyperbolic space
        self.perturbation_embeddings = self.add_weight(
            name="perturbation_embeddings",
            shape=[n_perturbations, embedding_dim],
            initializer=tf.keras.initializers.RandomUniform(
                minval=-0.001, maxval=0.001
            ),
            trainable=True
        )
        
        # Project perturbation input to hyperbolic space
        self.pert_projection = tf.keras.layers.Dense(embedding_dim, activation='relu')
        
        # Hyperbolic feed-forward layers
        self.hyperbolic_layer1 = HyperbolicLayer(
            manifold=self.manifold,
            units=hidden_dim,
            activation='relu'
        )
        
        self.hyperbolic_layer2 = HyperbolicLayer(
            manifold=self.manifold,
            units=hidden_dim // 2,
            activation='relu'
        )
        
        # Output projection layer (from hyperbolic to Euclidean space)
        self.output_projection = tf.keras.layers.Dense(n_genes, activation='linear')
        
    def call(self, inputs, training=False):
        """
        Forward pass of the model.
        
        Args:
            inputs: One-hot encoded perturbation targets
            training: Whether in training mode
            
        Returns:
            Predicted gene expression changes
        """
        # Project perturbation inputs to hyperbolic space
        pert_projected = self.pert_projection(inputs)
        
        # Normalize to ensure points are inside the Poincaré ball
        pert_norm = tf.norm(pert_projected, axis=-1, keepdims=True)
        max_norm = 0.999  # Keep away from the boundary
        pert_normalized = tf.where(
            pert_norm > max_norm,
            pert_projected * max_norm / pert_norm,
            pert_projected
        )
        
        # Get perturbation-specific embeddings
        # Here we compute a weighted sum of perturbation embeddings based on input
        pert_embeddings = tf.matmul(inputs, self.perturbation_embeddings)
        
        # Apply hyperbolic layers
        h = self.hyperbolic_layer1(pert_normalized)
        h = self.hyperbolic_layer2(h)
        
        # Project back to Euclidean space for gene expression predictions
        log_fc_predictions = self.output_projection(h)
        
        return log_fc_predictions