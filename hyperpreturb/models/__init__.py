"""
Core model architecture for the HyperPerturb framework.
"""

from hyperpreturb.final_files.models.hyperbolic import (
    HyperbolicAdam, QuantumAnnealer, HyperbolicDense, HyperbolicAttention
)
from hyperpreturb.final_files.utils.manifolds import PoincareBall
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
        self.encoder = tf.keras.Sequential([
            HyperbolicGraphConv(512, curvature=curvature),
            tf.keras.layers.LayerNormalization(),
            HyperbolicGraphConv(256, curvature=curvature),
            tf.keras.layers.Dropout(0.3)
        ])
        
        self.policy_head = tf.keras.Sequential([
            HyperbolicGraphConv(128, curvature=curvature),
            tf.keras.layers.Dense(num_genes, activity_regularizer=STDPRegularizer())
        ], name='policy')
        
        self.value_head = tf.keras.Sequential([
            HyperbolicGraphConv(128, curvature=curvature),
            tf.keras.layers.Dense(1)
        ], name='value')
        
    @tf.function(jit_compile=True)
    def call(self, inputs):
        x, adj = inputs
        encoded = self.encoder((x, adj))
        return self.policy_head(encoded), self.value_head(encoded)