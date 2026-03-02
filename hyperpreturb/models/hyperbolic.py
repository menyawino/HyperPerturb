import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from hyperpreturb.utils.manifolds import PoincareBall

# ----------------------------
# Hyperbolic Optimization Tools
# ----------------------------
class HyperbolicAdam(tf.keras.optimizers.Adam):
    """Adam with Riemannian gradient correction for Poincaré ball params.
    Rescales Euclidean gradients by the inverse conformal factor before
    the standard Adam step. This keeps updates geometrically consistent
    with the manifold curvature, at least approximately.
    """

    def __init__(self, manifold, **kwargs):
        super().__init__(**kwargs)
        self.manifold = manifold

    def _resource_apply_dense(self, grad, var, apply_state=None):
        riemannian_grad = self.manifold.egrad2rgrad(var, grad)
        return super()._resource_apply_dense(riemannian_grad, var, apply_state)

# Cosine-exponential LR decay
# Combines exponential falloff with cosine oscillations — loosely
# resembles warm-restart schedules. Not actually quantum anything.
QuantumAnnealer = None  # kept as alias below for back-compat

class CosineExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """LR schedule: exponential decay modulated by cosine oscillations."""

    def __init__(self, initial_lr=1e-3, T_max=1000, alpha=0.1):
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.alpha = alpha
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        t_max = tf.cast(self.T_max, tf.float32)
        phase = tf.math.mod(step, t_max)
        decay_factor = tf.math.exp(-phase / t_max)
        cosine_factor = 0.5 * (1 + tf.math.cos(np.pi * phase / t_max))
        return self.initial_lr * ((1 - self.alpha) * decay_factor * cosine_factor + self.alpha)

    def get_config(self):
        return {
            'initial_lr': self.initial_lr,
            'T_max': self.T_max,
            'alpha': self.alpha
        }

# Back-compat alias
QuantumAnnealer = CosineExponentialDecay

# ----------------------------
# Advanced Hyperbolic Layers
# ----------------------------
class HyperbolicDense(tf.keras.layers.Layer):
    """Dense layer that operates via tangent-space linear transform.

    logmap to origin -> matmul + bias -> expmap back onto the ball.
    Activation (if any) is also applied in tangent space.
    """

    def __init__(self, units, curvature=1.0, activation=None, use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.curvature = curvature
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.manifold = PoincareBall(curvature)
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name='kernel', shape=(input_dim, self.units),
            initializer='glorot_uniform')
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias', shape=(self.units,),
                initializer='zeros')
        self.built = True

    @tf.function
    def call(self, inputs):
        # tangent space at origin
        tangent_inputs = self.manifold.logmap(tf.zeros_like(inputs), inputs)
        outputs = tf.matmul(tangent_inputs, self.kernel)
        if self.use_bias:
            outputs = outputs + self.bias

        # back onto the ball
        outputs = self.manifold.expmap(tf.zeros_like(outputs), outputs)

        if self.activation is not None:
            # activation in tangent space, then re-project
            tangent_outputs = self.manifold.logmap(tf.zeros_like(outputs), outputs)
            outputs = self.manifold.expmap(
                tf.zeros_like(tangent_outputs),
                self.activation(tangent_outputs)
            )
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'curvature': self.curvature,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        })
        return config

# ----------------------------
# Hyperbolic Attention
# ----------------------------
class HyperbolicAttention(tf.keras.layers.Layer):
    """Multi-head attention using hyperbolic distances instead of dot products.

    Projects Q/K/V through tangent space, computes attention weights from
    pairwise Poincaré distances, aggregates values in tangent space.
    NOTE: this layer is defined but not actually used in the current model.
    """

    def __init__(self, units, num_heads=8, curvature=1.0, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.curvature = curvature
        self.dropout_rate = dropout_rate
        self.manifold = PoincareBall(curvature)
        
        # Ensure units is divisible by num_heads
        assert units % num_heads == 0, "units must be divisible by num_heads"
        self.depth = units // num_heads
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        for name in ('query_weight', 'key_weight', 'value_weight'):
            setattr(self, name, self.add_weight(
                name=name, shape=(input_dim, self.units),
                initializer='glorot_uniform'))
        self.output_weight = self.add_weight(
            name='output_weight', shape=(self.units, self.units),
            initializer='glorot_uniform')
        self.built = True
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    @tf.function
    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]
        tangent_inputs = self.manifold.logmap(tf.zeros_like(inputs), inputs)
        
        # Compute query, key, and value projections
        q = tf.matmul(tangent_inputs, self.query_weight)
        k = tf.matmul(tangent_inputs, self.key_weight)
        v = tf.matmul(tangent_inputs, self.value_weight)
        
        # Split heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Map projections back to hyperbolic space
        q_hyp = self.manifold.expmap(tf.zeros_like(q), q)
        k_hyp = self.manifold.expmap(tf.zeros_like(k), k)
        
        # Compute hyperbolic distances
        # Reshape for broadcasting
        q_expanded = tf.expand_dims(q_hyp, 3)  # [batch, heads, seq_len_q, 1, depth]
        k_expanded = tf.expand_dims(k_hyp, 2)  # [batch, heads, 1, seq_len_k, depth]
        
        # Compute distances using einsum
        distances = -tf.reduce_sum(
            (q_expanded - k_expanded)**2,
            axis=-1
        )  # [batch, heads, seq_len_q, seq_len_k]
        
        # Scale distances
        distances = distances / tf.sqrt(tf.cast(self.depth, tf.float32))
        
        # Apply mask if provided
        if mask is not None:
            distances += (mask * -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(distances, axis=-1)
        
        # Apply dropout to attention weights
        if training:
            attention_weights = tf.nn.dropout(attention_weights, rate=self.dropout_rate)
        
        # Apply attention weights to values
        v_tangent = self.manifold.logmap(tf.zeros_like(v), v)
        context = tf.matmul(attention_weights, v_tangent)
        
        # Merge heads
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.units))
        
        # Final projection
        output = tf.matmul(context, self.output_weight)
        
        # Map back to hyperbolic space
        output = self.manifold.expmap(tf.zeros_like(output), output)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'num_heads': self.num_heads,
                       'curvature': self.curvature, 'dropout_rate': self.dropout_rate})
        return config


class HyperbolicPoincareBall:
    """Standalone Poincaré ball ops used by HyperbolicLayer.

    This duplicates some functionality from utils/manifolds.py but uses
    a different sign convention for curvature (stores -curvature internally).
    TODO: unify with the main PoincareBall class.
    """
    def __init__(self, dim, curvature=-1.0):
        self.dim = dim
        self.c = tf.convert_to_tensor(-curvature, dtype=tf.float32)
        self.eps = 1e-10

    def mobius_addition(self, x, y):
        """Möbius addition x ⊕ y."""
        x2 = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        y2 = tf.reduce_sum(tf.square(y), axis=-1, keepdims=True)
        xy = tf.reduce_sum(x * y, axis=-1, keepdims=True)
        
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c * self.c * x2 * y2
        
        return num / (denom + self.eps)

    def mobius_matrix_multiplication(self, M, x):
        """Möbius mat-vec: project x through matrix M while staying on the ball."""
        x_norm = tf.norm(x, axis=-1, keepdims=True)
        mx = tf.matmul(x, M, transpose_b=True)
        mx_norm = tf.norm(mx, axis=-1, keepdims=True)
        
        # Handle zero vectors
        res_c = tf.tanh(self.c * mx_norm / (1.0 + self.eps)) * mx / (mx_norm + self.eps)
        return tf.where(tf.abs(x_norm) < self.eps, mx, res_c)
        
    def exponential_map(self, x, v):
        """Exp map at x with tangent vector v."""
        v_norm = tf.norm(v, axis=-1, keepdims=True)
        second_term = tf.tanh(tf.sqrt(self.c) * v_norm / 2) * v / (tf.sqrt(self.c) * v_norm + self.eps)
        
        # Handle zero tangent vectors
        return tf.where(tf.abs(v_norm) < self.eps, x, self.mobius_addition(x, second_term))
        
    def logarithmic_map(self, x, y):
        """Log map at x towards y."""
        addition = self.mobius_addition(-x, y)
        addition_norm = tf.norm(addition, axis=-1, keepdims=True)
        
        log_term = tf.math.atanh(tf.sqrt(self.c) * addition_norm) * addition / (tf.sqrt(self.c) * addition_norm + self.eps)
        
        # Handle zero vectors
        return tf.where(tf.abs(addition_norm) < self.eps, y - x, 2 * log_term)

class HyperbolicLayer(tf.keras.layers.Layer):
    """Neural network layer operating in hyperbolic space via Möbius ops."""

    def __init__(self, manifold, units, activation=None, use_bias=True, **kwargs):
        super(HyperbolicLayer, self).__init__(**kwargs)
        self.manifold = manifold
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.weight_matrix = self.add_weight(
            name='hyperbolic_weight', shape=[input_dim, self.units],
            initializer='glorot_uniform')
        if self.use_bias:
            self.bias = self.add_weight(
                name='hyperbolic_bias', shape=[1, self.units],
                initializer='zeros')
        super().build(input_shape)

    def call(self, inputs):
        # Mobius mat-mul, optional bias, activation in tangent space
        outputs = self.manifold.mobius_matrix_multiplication(self.weight_matrix, inputs)
        
        # Apply bias in hyperbolic space if needed
        if self.use_bias:
            outputs = self.manifold.mobius_addition(outputs, self.bias)
        
        # Apply activation function if specified
        if self.activation is not None:
            # We need to map to tangent space, apply activation, and map back
            origin = tf.zeros_like(outputs)
            tangent = self.manifold.logarithmic_map(origin, outputs)
            activated = self.activation(tangent)
            outputs = self.manifold.exponential_map(origin, activated)
            
        return outputs