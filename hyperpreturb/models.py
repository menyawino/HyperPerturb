import tensorflow as tf
import numpy as np
from hyperpreturb.models.hyperbolic import HyperbolicLayer, HyperbolicPoincareBall
from hyperpreturb.utils.manifolds import PoincareBall

class HyperbolicPerturbationModel(tf.keras.Model):
    """Simple perturbation model: one-hot input -> hyperbolic dense -> predicted logFC.

    Standalone version (legacy). The main model is in models/__init__.py.
    """
    def __init__(self, n_genes, n_perturbations, embedding_dim=32, hidden_dim=64, curvature=1.0, adj_matrix=None):
        super().__init__()
        
        self.n_genes = n_genes
        self.n_perturbations = n_perturbations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.curvature = curvature
        
        # Initialize manifold
        self.manifold = PoincareBall(curvature=curvature)
        
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
        
    def call(self, inputs, training=None, mask=None):
        # project perturbation one-hot -> embedding space
        pert_projected = self.pert_projection(inputs)
        
        # clip to stay inside the ball
        pert_norm = tf.norm(pert_projected, axis=-1, keepdims=True)
        max_norm = 0.999
        pert_normalized = tf.where(
            pert_norm > max_norm,
            pert_projected * max_norm / pert_norm,
            pert_projected
        )

        # weighted sum of perturbation embeddings
        pert_embeddings = tf.matmul(inputs, self.perturbation_embeddings)

        # hyperbolic layers
        h = self.hyperbolic_layer1(pert_normalized)
        h = self.hyperbolic_layer2(h)

        # back to Euclidean for the final prediction
        log_fc_predictions = self.output_projection(h)
        
        return log_fc_predictions