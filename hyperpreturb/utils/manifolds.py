class PoincareBall:
    def __init__(self, curvature=1.0):
        self.c = curvature
        
    def expmap(self, x, v):
        norm_v = tf.norm(v, axis=-1, keepdims=True)
        return tf.math.tanh(tf.sqrt(self.c)*norm_v) * v / (tf.sqrt(self.c)*norm_v + 1e-8)
    
    def logmap(self, x, y):
        norm_xy = tf.norm(y - x, axis=-1, keepdims=True)
        return tf.math.atanh(tf.sqrt(self.c)*norm_xy) * (y - x) / (tf.sqrt(self.c)*norm_xy + 1e-8)
    
    def egrad2rgrad(self, x, grad):
        return ((1 - self.c * tf.norm(x, axis=-1, keepdims=True)**2)**2 / 4) * grad
