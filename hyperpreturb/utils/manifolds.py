import tensorflow as tf
import tensorflow_probability as tfp

class PoincareBall:
    """
    Poincare Ball manifold implementation with Riemannian geometry operations.
    
    This class provides operations for the Poincare ball model of hyperbolic geometry,
    including distance computations, parallel transport, and geodesic calculations.
    """
    
    def __init__(self, curvature=1.0, eps=1e-8):
        """
        Initialize the Poincare Ball manifold.
        
        Args:
            curvature: Curvature parameter (c). Default: 1.0
            eps: Small constant for numerical stability. Default: 1e-8
        """
        self.c = curvature
        self.eps = eps
    
    @tf.function(jit_compile=True)
    def distance(self, x, y):
        """
        Compute the geodesic distance between points x and y in the Poincare ball.
        
        Args:
            x: Point(s) in the Poincare ball
            y: Point(s) in the Poincare ball
            
        Returns:
            Geodesic distance between x and y
        """
        sqrt_c = tf.sqrt(self.c)
        
        # Compute the norm of x and y
        x_norm = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        y_norm = tf.reduce_sum(tf.square(y), axis=-1, keepdims=True)
        
        # Compute the Mobius addition of -x and y
        xy_dot = tf.reduce_sum(x * y, axis=-1, keepdims=True)
        num = (1 + 2 * self.c * xy_dot + self.c * y_norm) * x
        num = num - (1 - self.c * x_norm) * y
        denom = 1 + 2 * self.c * xy_dot + self.c * self.c * x_norm * y_norm
        mobius_add = num / (denom + self.eps)
        
        # Compute the norm of the result
        mobius_norm = tf.norm(mobius_add, axis=-1)
        
        # Return the distance
        return 2 / sqrt_c * tf.math.atanh(sqrt_c * mobius_norm)
    
    @tf.function(jit_compile=True)
    def expmap(self, x, v):
        """
        Exponential map at point x with tangent vector v.
        
        Args:
            x: Point in the Poincare ball
            v: Tangent vector at x
            
        Returns:
            Point in the Poincare ball
        """
        v_norm = tf.norm(v, axis=-1, keepdims=True)
        second_term = tf.tanh(tf.sqrt(self.c) / 2 * v_norm) * v / (tf.sqrt(self.c) * v_norm + self.eps)
        
        x_norm = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        denom = 1 + 2 * tf.reduce_sum(x * second_term, axis=-1, keepdims=True) + self.c * x_norm * tf.square(tf.norm(second_term, axis=-1, keepdims=True))
        
        return (x + second_term) / denom
    
    @tf.function(jit_compile=True)
    def logmap(self, x, y):
        """
        Logarithmic map at point x for target point y.
        
        Args:
            x: Base point in the Poincare ball
            y: Target point in the Poincare ball
            
        Returns:
            Tangent vector at x
        """
        sqrt_c = tf.sqrt(self.c)
        
        # Compute the Mobius addition of -x and y
        x_norm = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        y_norm = tf.reduce_sum(tf.square(y), axis=-1, keepdims=True)
        xy_dot = tf.reduce_sum(x * y, axis=-1, keepdims=True)
        
        num = (1 + 2 * self.c * xy_dot + self.c * y_norm) * x
        num = num - (1 - self.c * x_norm) * y
        denom = 1 + 2 * self.c * xy_dot + self.c * self.c * x_norm * y_norm
        mobius_add = num / (denom + self.eps)
        
        # Compute the norm of the result
        mobius_norm = tf.norm(mobius_add, axis=-1, keepdims=True)
        
        # Return the tangent vector
        coef = 2 / sqrt_c * tf.math.atanh(sqrt_c * mobius_norm) / (mobius_norm + self.eps)
        return coef * mobius_add
    
    @tf.function(jit_compile=True)
    def egrad2rgrad(self, x, grad):
        """
        Convert Euclidean gradient to Riemannian gradient.
        
        Args:
            x: Point in the Poincare ball
            grad: Euclidean gradient at x
            
        Returns:
            Riemannian gradient at x
        """
        x_norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        scaling = tf.square(1 - self.c * x_norm_sq) / 4
        return scaling * grad
    
    @tf.function(jit_compile=True)
    def mobius_addition(self, x, y):
        """
        Mobius addition of points x and y in the Poincare ball.
        
        Args:
            x: Point in the Poincare ball
            y: Point in the Poincare ball
            
        Returns:
            Mobius addition of x and y
        """
        # Compute dot products and norms
        x_norm_sq = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
        y_norm_sq = tf.reduce_sum(tf.square(y), axis=-1, keepdims=True)
        xy_dot = tf.reduce_sum(x * y, axis=-1, keepdims=True)
        
        # Compute numerator and denominator
        num = (1 + 2 * self.c * xy_dot + self.c * y_norm_sq) * x + (1 - self.c * x_norm_sq) * y
        denom = 1 + 2 * self.c * xy_dot + self.c * self.c * x_norm_sq * y_norm_sq
        
        return num / (denom + self.eps)