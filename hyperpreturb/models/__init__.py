"""
Core model architecture for the HyperPerturb framework.
"""

from hyperpreturb.models.hyperbolic import (
    HyperbolicAdam, QuantumAnnealer, HyperbolicDense, HyperbolicAttention
)
from hyperpreturb.utils.manifolds import PoincareBall
import tensorflow as tf
import tensorflow_probability as tfp

# ----------------------------
# Hyperbolic Operations (XLA-optimized)
# ----------------------------
@tf.function(jit_compile=True)
def poincare_expmap(v, c=1.0):
    norm_v = tf.norm(v, axis=-1, keepdims=True)
    return tf.math.tanh(tf.sqrt(c) * norm_v) * v / (tf.sqrt(c) * norm_v + 1e-8)

@tf.function(jit_compile=True)
def poincare_logmap(y, c=1.0):
    norm_y = tf.norm(y, axis=-1, keepdims=True)
    return tf.math.atanh(tf.sqrt(c) * norm_y) * y / (tf.sqrt(c) * norm_y + 1e-8)

@tf.function(jit_compile=True)
def parallel_transport(x, y):
    return tf.linalg.expm(tf.linalg.logm(tf.linalg.inv(x) @ y))

# ----------------------------
# Quantum-Inspired Initialization
# ----------------------------
class HaarMeasureInitializer(tf.keras.initializers.Initializer):
    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon
        
    def __call__(self, shape, dtype=None):
        n = shape[0]
        q = tf.linalg.exp(tfp.math.random_rademacher(shape, dtype=tf.complex64))
        return tf.math.real(q) + self.epsilon * tf.random.normal(shape, dtype=dtype)

# ----------------------------
# Hierarchical Graph Convolution
# ----------------------------
class HyperbolicGraphConv(tf.keras.layers.Layer):
    def __init__(self, units, curvature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.curvature = curvature
        self.manifold = PoincareBall(curvature)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=HaarMeasureInitializer(),
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
    @tf.function(jit_compile=True)
    def call(self, inputs):
        x, adj = inputs
        x_proj = poincare_expmap(x, self.curvature)
        support = tf.sparse.sparse_dense_matmul(adj, x_proj)
        output = poincare_logmap(support, self.curvature)
        return output + self.bias

# ----------------------------
# Neuromorphic Regularization
# ----------------------------
class STDPRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, rho=0.05, beta=1e-3):
        self.rho = rho
        self.beta = beta
        self.spike_trace = None
        
    def __call__(self, weights):
        if self.spike_trace is None:
            self.spike_trace = tf.Variable(tf.zeros_like(weights), trainable=False)
            
        spike_penalty = self.beta * tf.reduce_sum(
            tf.math.square(weights * self.spike_trace)
        )
        self.spike_trace.assign_add(tf.math.abs(weights))
        return spike_penalty

# ----------------------------
# Core Model Architecture
# ----------------------------
class HyperPerturbModel(tf.keras.Model):
    def __init__(self, num_genes, curvature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.manifold = PoincareBall(curvature)
        # Encoder stack operating on (node_features, adjacency)
        self.encoder_gcn1 = HyperbolicGraphConv(512, curvature=curvature)
        self.encoder_norm = tf.keras.layers.LayerNormalization()
        self.encoder_gcn2 = HyperbolicGraphConv(256, curvature=curvature)
        self.encoder_dropout = tf.keras.layers.Dropout(0.3)

        # Policy and value heads; each takes (encoded_nodes, adjacency)
        self.policy_gcn = HyperbolicGraphConv(128, curvature=curvature)
        self.policy_dense = tf.keras.layers.Dense(
            num_genes,
            activity_regularizer=STDPRegularizer(),
            name="policy_output",
        )

        self.value_gcn = HyperbolicGraphConv(128, curvature=curvature)
        self.value_dense = tf.keras.layers.Dense(1, name="value_output")

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: Tuple ``(x, adj)`` where ``x`` has shape
                ``(batch, n_nodes, d)`` and ``adj`` is a (possibly
                sparse) adjacency matrix of shape ``(n_nodes, n_nodes)``.
        """
        x, adj = inputs

        # Encoder
        h = self.encoder_gcn1((x, adj))
        h = self.encoder_norm(h)
        h = self.encoder_gcn2((h, adj))
        h = self.encoder_dropout(h, training=training)

        # Policy head
        policy_h = self.policy_gcn((h, adj))
        policy_logits = self.policy_dense(policy_h)

        # Value head
        value_h = self.value_gcn((h, adj))
        value = self.value_dense(value_h)

        return policy_logits, value

# ----------------------------
# Simple HyperbolicPerturbationModel
# ----------------------------
class HyperbolicPerturbationModel(tf.keras.Model):
    """
    A simple model for predicting gene expression changes in response to perturbations.
    Uses hyperbolic embeddings to capture hierarchical structure.
    """
    def __init__(self, n_genes, n_perturbations, embedding_dim=32, hidden_dim=64, curvature=1.0, **kwargs):
        super().__init__(**kwargs)
        
        self.n_genes = n_genes
        self.n_perturbations = n_perturbations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.curvature = curvature
        
        # Initialize manifold
        self.manifold = PoincareBall(curvature=curvature)
        
        # Project perturbation input to hyperbolic space
        self.pert_projection = tf.keras.layers.Dense(embedding_dim, activation='relu')
        
        # Hyperbolic layers
        self.hyperbolic_layer1 = HyperbolicDense(units=hidden_dim, curvature=curvature)
        self.hyperbolic_layer2 = HyperbolicDense(units=hidden_dim // 2, curvature=curvature)
        
        # Output projection layer
        self.output_projection = tf.keras.layers.Dense(n_genes, activation='linear')
    
    def call(self, inputs, training=False):
        """Forward pass."""
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
        
        # Apply hyperbolic layers
        h = self.hyperbolic_layer1(pert_normalized)
        h = self.hyperbolic_layer2(h)
        
        # Project back to Euclidean space for gene expression predictions
        log_fc_predictions = self.output_projection(h)
        
        return log_fc_predictions

# Add to exports
__all__ = ['HyperPerturbModel', 'HyperbolicPerturbationModel', 'HyperbolicGraphConv', 'STDPRegularizer']