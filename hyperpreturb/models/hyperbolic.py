import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from hyperpreturb.final_files.utils.manifolds import PoincareBall

# ----------------------------
# Hyperbolic Optimization Tools
# ----------------------------
class HyperbolicAdam(tf.keras.optimizers.Adam):
    """Adam optimizer adapted for Riemannian manifolds.
    
    This optimizer correctly handles gradients on Riemannian manifolds by
    converting Euclidean gradients to Riemannian gradients before applying
    the standard Adam update rules.
    """
    
    def __init__(self, manifold, **kwargs):
        """
        Initialize the HyperbolicAdam optimizer.
        
        Args:
            manifold: Riemannian manifold instance (e.g. PoincareBall)
            **kwargs: Standard Adam optimizer parameters
        """
        super().__init__(**kwargs)
        self.manifold = manifold
    
    def _resource_apply_dense(self, grad, var, apply_state=None):
        """Apply gradients to variables, considering manifold structure."""
        # Convert Euclidean gradient to Riemannian gradient
        riemannian_grad = self.manifold.egrad2rgrad(var, grad)
        
        # Apply standard Adam update with Riemannian gradient
        return super()._resource_apply_dense(riemannian_grad, var, apply_state)

# ----------------------------
# Quantum Annealing Schedule
# ----------------------------
class QuantumAnnealer(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Learning rate scheduler inspired by quantum annealing.
    
    Provides a learning rate schedule that combines exponential decay
    with cosine oscillations, inspired by quantum annealing principles.
    """
    
    def __init__(self, initial_lr=1e-3, T_max=1000, alpha=0.1):
        """
        Initialize the quantum annealing schedule.
        
        Args:
            initial_lr: Initial learning rate. Default: 1e-3
            T_max: Maximum number of iterations. Default: 1000
            alpha: Minimum learning rate factor. Default: 0.1
        """
        self.initial_lr = initial_lr
        self.T_max = float(T_max)
        self.alpha = alpha
    
    def __call__(self, step):
        """Calculate the learning rate for the current step."""
        phase = tf.cast(step % self.T_max, tf.float32)
        decay_factor = tf.math.exp(-phase/self.T_max)
        cosine_factor = 0.5 * (1 + tf.math.cos(np.pi * phase/self.T_max))
        
        return self.initial_lr * ((1 - self.alpha) * decay_factor * cosine_factor + self.alpha)
    
    def get_config(self):
        """Return configuration for serialization."""
        return {
            'initial_lr': self.initial_lr,
            'T_max': self.T_max,
            'alpha': self.alpha
        }

# ----------------------------
# Advanced Hyperbolic Layers
# ----------------------------
class HyperbolicDense(tf.keras.layers.Layer):
    """
    Dense layer in hyperbolic space.
    
    This layer performs linear transformation in the tangent space,
    followed by exponential mapping back to the hyperbolic space.
    """
    
    def __init__(self, units, curvature=1.0, activation=None, use_bias=True, **kwargs):
        """
        Initialize the hyperbolic dense layer.
        
        Args:
            units: Number of output units
            curvature: Curvature of the hyperbolic space. Default: 1.0
            activation: Activation function. Default: None
            use_bias: Whether to use bias. Default: True
            **kwargs: Additional layer parameters
        """
        super().__init__(**kwargs)
        self.units = units
        self.curvature = curvature
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.manifold = PoincareBall(curvature)
    
    def build(self, input_shape):
        """Build the layer with the given input shape."""
        input_dim = input_shape[-1]
        
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True
            )
        
        self.built = True
    
    @tf.function
    def call(self, inputs):
        """Forward pass through the layer."""
        # Map points to tangent space at origin
        tangent_inputs = self.manifold.logmap(tf.zeros_like(inputs), inputs)
        
        # Apply linear transformation in tangent space
        outputs = tf.matmul(tangent_inputs, self.kernel)
        
        if self.use_bias:
            outputs = outputs + self.bias
        
        # Map back to hyperbolic space
        outputs = self.manifold.expmap(tf.zeros_like(outputs), outputs)
        
        if self.activation is not None:
            # Map to tangent space, apply activation, and map back
            tangent_outputs = self.manifold.logmap(tf.zeros_like(outputs), outputs)
            activated_outputs = self.activation(tangent_outputs)
            outputs = self.manifold.expmap(tf.zeros_like(activated_outputs), activated_outputs)
        
        return outputs
    
    def get_config(self):
        """Return layer configuration for serialization."""
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
    """
    Attention mechanism adapted for hyperbolic space.
    
    This layer computes attention weights based on hyperbolic distances
    and performs weighted aggregation in tangent space.
    """
    
    def __init__(self, units, num_heads=8, curvature=1.0, dropout_rate=0.1, **kwargs):
        """
        Initialize the hyperbolic attention layer.
        
        Args:
            units: Number of output units
            num_heads: Number of attention heads. Default: 8
            curvature: Curvature of hyperbolic space. Default: 1.0
            dropout_rate: Dropout rate for attention weights. Default: 0.1
            **kwargs: Additional layer parameters
        """
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
        """Build the layer with the given input shape."""
        input_dim = input_shape[-1]
        
        self.query_weight = self.add_weight(
            name='query_weight',
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        self.key_weight = self.add_weight(
            name='key_weight',
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        self.value_weight = self.add_weight(
            name='value_weight',
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        self.output_weight = self.add_weight(
            name='output_weight',
            shape=(self.units, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True
        )
        
        self.built = True
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    @tf.function
    def call(self, inputs, mask=None, training=None):
        """Forward pass through the layer."""
        batch_size = tf.shape(inputs)[0]
        
        # Transform inputs to tangent space
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
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads,
            'curvature': self.curvature,
            'dropout_rate': self.dropout_rate
        })
        return config