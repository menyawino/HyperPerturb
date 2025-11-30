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
# Hyperbolic Operations
# ----------------------------
def poincare_expmap(v, c=1.0):
    # Clip norms to avoid extremely large arguments to tanh
    norm_v = tf.norm(v, axis=-1, keepdims=True)
    norm_v = tf.clip_by_value(norm_v, 0.0, 10.0)
    return tf.math.tanh(tf.sqrt(c) * norm_v) * v / (tf.sqrt(c) * norm_v + 1e-8)

def poincare_logmap(y, c=1.0):
    # Clip norms to keep arguments to atanh in a safe range
    norm_y = tf.norm(y, axis=-1, keepdims=True)
    norm_y = tf.clip_by_value(norm_y, 0.0, 0.999)
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
        """Simple, robust weight initializer.

        Uses a normalized Gaussian initializer that is compatible with a wide
        range of TensorFlow versions and avoids advanced linear algebra ops
        that may be missing on some builds.
        """
        # Ensure ``dtype`` is a valid TensorFlow dtype
        if dtype is None:
            dtype = tf.float32
        dtype = tf.as_dtype(dtype)

        # Base Gaussian weights
        w = tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=dtype)

        # Optional small perturbation noise for numerical diversity
        if self.epsilon > 0:
            w += self.epsilon * tf.random.normal(shape, dtype=dtype)

        # L2-normalize over input dimension to keep scales reasonable
        if len(shape) > 0:
            w = tf.math.l2_normalize(w, axis=0)

        return w

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
        """Build layer weights.

        ``input_shape`` can be either a tuple ``(node_shape, adj_shape)`` or a
        single tensor shape when Keras traces the layer. We only need the
        feature dimension of the node representation.
        """
        # If we receive a tuple/list, take the node feature shape
        if isinstance(input_shape, (list, tuple)):
            node_shape = tf.TensorShape(input_shape[0])
        else:
            node_shape = tf.TensorShape(input_shape)

        feature_dim = int(node_shape[-1])

        self.kernel = self.add_weight(
            name="kernel",
            shape=(feature_dim, self.units),
            initializer=HaarMeasureInitializer(),
            trainable=True,
            dtype=self.dtype,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            dtype=self.dtype,
        )
        
    def call(self, inputs):
        # ``inputs`` is expected to be (x, adj)
        x, adj = inputs

        # Map node features to hyperbolic space
        x_proj = poincare_expmap(x, self.curvature)

        # Dense or sparse adjacency support
        if isinstance(adj, tf.SparseTensor):
            support = tf.sparse.sparse_dense_matmul(adj, x_proj)
        else:
            support = tf.linalg.matmul(adj, x_proj)

        # Map back to tangent space
        output = poincare_logmap(support, self.curvature)

        # Linear projection with bias
        output = tf.linalg.matmul(output, self.kernel) + self.bias
        return output

# ----------------------------
# Neuromorphic Regularization
# ----------------------------
class STDPRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, rho=0.05, beta=1e-3):
        self.rho = rho
        self.beta = beta
        
    def __call__(self, weights):
        """Stateless L2-style regularization on weights.

        Using a stateless formulation avoids graph scope and capture issues
        when this regularizer is used inside compiled `tf.function`s.
        """
        return self.beta * tf.reduce_sum(tf.square(weights))

# ----------------------------
# Core Model Architecture
# ----------------------------
class HyperPerturbModel(tf.keras.Model):
    def __init__(self, num_genes, num_perts, curvature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.manifold = PoincareBall(curvature)
        # Encoder stack operating on (node_features, adjacency)
        self.encoder_gcn1 = HyperbolicGraphConv(512, curvature=curvature)
        self.encoder_norm = tf.keras.layers.LayerNormalization()
        self.encoder_gcn2 = HyperbolicGraphConv(256, curvature=curvature)
        self.encoder_dropout = tf.keras.layers.Dropout(0.3)

        # Policy and value heads; each takes (encoded_nodes, adjacency)
        self.policy_gcn = HyperbolicGraphConv(128, curvature=curvature)
        # Policy head output: per-gene distribution over perturbations
        self.policy_dense = tf.keras.layers.Dense(
            num_perts,
            activation="softmax",  # force probabilities at the head
            name="policy_output",
        )

        self.value_gcn = HyperbolicGraphConv(128, curvature=curvature)
        self.value_dense = tf.keras.layers.Dense(1, name="value_output")

    def call(self, inputs, training=False, debug=False):
        """Forward pass.

        Args:
            inputs: Tuple ``(x, adj)`` where ``x`` has shape
                ``(batch, n_nodes, d)`` and ``adj`` is a (possibly
                sparse) adjacency matrix of shape ``(n_nodes, n_nodes)``.
        """
        x, adj = inputs

        # Encoder
        h = self.encoder_gcn1((x, adj))
        if debug:
            tf.debugging.check_numerics(h, "NaN/Inf after encoder_gcn1")
        h = self.encoder_norm(h)
        h = self.encoder_gcn2((h, adj))
        if debug:
            tf.debugging.check_numerics(h, "NaN/Inf after encoder_gcn2")
        h = self.encoder_dropout(h, training=training)

        # Policy head
        policy_h = self.policy_gcn((h, adj))
        if debug:
            tf.debugging.check_numerics(policy_h, "NaN/Inf after policy_gcn")
        policy_logits = self.policy_dense(policy_h)
        if debug:
            tf.debugging.check_numerics(policy_logits, "NaN/Inf after policy_dense (pre-clip)")

        # Extra numerical safety: clip and renormalize policy distribution
        eps = 1e-6
        policy_logits = tf.clip_by_value(policy_logits, eps, 1.0)
        policy_logits /= tf.reduce_sum(policy_logits, axis=-1, keepdims=True)
        if debug:
            tf.debugging.check_numerics(policy_logits, "NaN/Inf after policy renorm")

        # Value head
        value_h = self.value_gcn((h, adj))
        if debug:
            tf.debugging.check_numerics(value_h, "NaN/Inf after value_gcn")
        value = self.value_dense(value_h)
        if debug:
            tf.debugging.check_numerics(value, "NaN/Inf after value_dense")

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