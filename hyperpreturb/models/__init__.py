"""Core model architecture for the HyperPerturb framework."""

from hyperpreturb.models.hyperbolic import (
    HyperbolicAdam, QuantumAnnealer, HyperbolicDense, HyperbolicAttention
)
from hyperpreturb.utils.manifolds import PoincareBall
import tensorflow as tf
import tensorflow_probability as tfp

# Helpers for mapping to/from the Poincare ball
def poincare_expmap(v, c=1.0):
    """Exp map at the origin: project tangent vector v onto the ball."""
    norm_v = tf.norm(v, axis=-1, keepdims=True)
    norm_v = tf.clip_by_value(norm_v, 0.0, 10.0)
    return tf.math.tanh(tf.sqrt(c) * norm_v) * v / (tf.sqrt(c) * norm_v + 1e-8)

def poincare_logmap(y, c=1.0):
    """Log map at the origin: pull point y back to tangent space."""
    norm_y = tf.norm(y, axis=-1, keepdims=True)
    norm_y = tf.clip_by_value(norm_y, 0.0, 0.999)
    return tf.math.atanh(tf.sqrt(c) * norm_y) * y / (tf.sqrt(c) * norm_y + 1e-8)

# NOTE: parallel_transport via matrix expm/logm is not correct for the
# Poincare ball — it's a matrix Lie group operation. Left here for reference
# but don't use it for actual Riemannian transport.
@tf.function(jit_compile=True)
def parallel_transport(x, y):
    return tf.linalg.expm(tf.linalg.logm(tf.linalg.inv(x) @ y))


class ScaledNormalInitializer(tf.keras.initializers.Initializer):
    """L2-normalized Gaussian init. Nothing fancy, just keeps scales reasonable."""

    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon
        
    def __call__(self, shape, dtype=None):
        if dtype is None:
            dtype = tf.float32
        dtype = tf.as_dtype(dtype)
        w = tf.random.normal(shape, mean=0.0, stddev=1.0, dtype=dtype)
        if self.epsilon > 0:
            w += self.epsilon * tf.random.normal(shape, dtype=dtype)
        if len(shape) > 0:
            w = tf.math.l2_normalize(w, axis=0)
        return w

# Back-compat alias
HaarMeasureInitializer = ScaledNormalInitializer

class HyperbolicGraphConv(tf.keras.layers.Layer):
    """Graph convolution with optional Poincare ball projection.

    When euclidean_mode=False: expmap -> adj aggregation -> logmap -> linear.
    When euclidean_mode=True: just adj aggregation -> linear (for ablation).
    Norms are clipped to max_norm to avoid boundary explosions.
    """

    def __init__(self, units, curvature=1.0, euclidean_mode=False, max_norm=0.9, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.curvature = curvature
        self.euclidean_mode = euclidean_mode
        self.max_norm = max_norm
        self.manifold = PoincareBall(curvature)
        
    def build(self, input_shape):
        # input_shape may be (node_shape, adj_shape) or just node_shape
        if isinstance(input_shape, (list, tuple)):
            node_shape = tf.TensorShape(input_shape[0])
        else:
            node_shape = tf.TensorShape(input_shape)

        feature_dim = int(node_shape[-1])
        self.kernel = self.add_weight(
            'kernel', (feature_dim, self.units),
            initializer=ScaledNormalInitializer(), dtype=self.dtype)
        self.bias = self.add_weight(
            'bias', (self.units,), initializer='zeros', dtype=self.dtype)
        
    def _clip_to_ball(self, x):
        """Keep points inside the ball (norm < max_norm)."""
        norm = tf.norm(x, axis=-1, keepdims=True)
        scale = tf.where(norm > 0, tf.minimum(1.0, self.max_norm / (norm + 1e-8)), 1.0)
        return x * scale

    def call(self, inputs):
        # ``inputs`` is expected to be (x, adj)
        x, adj = inputs

        if self.euclidean_mode:
            # Pure Euclidean aggregation for debugging: mirror EuclideanGraphConv
            if isinstance(adj, tf.SparseTensor):
                support = tf.sparse.sparse_dense_matmul(adj, x)
            else:
                support = tf.linalg.matmul(adj, x)
            output = tf.linalg.matmul(support, self.kernel) + self.bias
            return output

        # Map node features to hyperbolic space with clipping
        x_clipped = self._clip_to_ball(x)
        x_proj = poincare_expmap(x_clipped, self.curvature)
        x_proj = self._clip_to_ball(x_proj)

        # Dense or sparse adjacency support in hyperbolic space
        if isinstance(adj, tf.SparseTensor):
            support = tf.sparse.sparse_dense_matmul(adj, x_proj)
        else:
            support = tf.linalg.matmul(adj, x_proj)
        support = self._clip_to_ball(support)

        # Map back to tangent space and project
        output = poincare_logmap(support, self.curvature)
        output = tf.linalg.matmul(output, self.kernel) + self.bias
        return output


class EuclideanGraphConv(tf.keras.layers.Layer):
    """Plain Euclidean GCN for ablation against the hyperbolic version."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            node_shape = tf.TensorShape(input_shape[0])
        else:
            node_shape = tf.TensorShape(input_shape)

        feature_dim = int(node_shape[-1])

        self.kernel = self.add_weight(
            name="kernel",
            shape=(feature_dim, self.units),
            initializer="glorot_uniform",
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
        x, adj = inputs

        # Dense or sparse adjacency support in Euclidean space
        if isinstance(adj, tf.SparseTensor):
            support = tf.sparse.sparse_dense_matmul(adj, x)
        else:
            support = tf.linalg.matmul(adj, x)

        output = tf.linalg.matmul(support, self.kernel) + self.bias
        return output

# Weight decay regularizer (plain L2)
# Previously called "STDPRegularizer" which was misleading — there’s no
# spike-timing dependent plasticity here, just L2 penalty on weights.
class WeightDecay(tf.keras.regularizers.Regularizer):
    def __init__(self, strength=1e-3):
        self.strength = strength

    def __call__(self, weights):
        return self.strength * tf.reduce_sum(tf.square(weights))

# Back-compat alias
STDPRegularizer = WeightDecay

# Core model
class HyperPerturbModel(tf.keras.Model):
    """Graph neural net with hyperbolic convolutions, dual policy+value heads.

    Encoder: 2 GCN layers (first hyperbolic, second euclidean) + norm + dropout.
    Policy head: GCN -> dense(softmax) -> per-gene distribution over perturbations.
    Value head: GCN -> dense -> per-gene scalar sensitivity score.
    """
    def __init__(self, num_genes, num_perts, curvature=1.0, **kwargs):
        super().__init__(**kwargs)
        self.manifold = PoincareBall(curvature)
        # Encoder stack operating on (node_features, adjacency)
        # Stage 2 ablation: make only the first encoder layer hyperbolic
        # while keeping the second encoder, policy, and value GCNs Euclidean.
        self.encoder_gcn1 = HyperbolicGraphConv(
            512, curvature=curvature, euclidean_mode=False
        )
        self.encoder_norm = tf.keras.layers.LayerNormalization()
        self.encoder_gcn2 = HyperbolicGraphConv(
            256, curvature=curvature, euclidean_mode=True
        )
        self.encoder_dropout = tf.keras.layers.Dropout(0.3)

        # Policy and value heads; each takes (encoded_nodes, adjacency)
        self.policy_gcn = HyperbolicGraphConv(
            128, curvature=curvature, euclidean_mode=True
        )
        # Policy head output: per-gene distribution over perturbations
        self.policy_dense = tf.keras.layers.Dense(
            num_perts,
            activation="softmax",  # force probabilities at the head
            name="policy_output",
        )

        # Stage 1 ablation: make only the value head hyperbolic while
        # keeping encoder and policy Euclidean.
        self.value_gcn = HyperbolicGraphConv(
            128, curvature=curvature, euclidean_mode=False
        )
        self.value_dense = tf.keras.layers.Dense(1, name="value_output")

    def call(self, inputs, training=False, debug=False):
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

class HyperbolicPerturbationModel(tf.keras.Model):
    """Simpler perturbation model: one-hot -> hyperbolic dense layers -> predicted logFC.

    Used by the 'simple' trainer. No graph structure involved.
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
__all__ = [
    'HyperPerturbModel',
    'HyperbolicPerturbationModel',
    'HyperbolicGraphConv',
    'EuclideanGraphConv',
    'STDPRegularizer',
]