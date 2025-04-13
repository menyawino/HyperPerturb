import tensorflow as tf
import numpy as np
import os
import logging
import json
from datetime import datetime
from pathlib import Path

from hyperpreturb.final_files.models import HyperPerturbModel
from hyperpreturb.final_files.models.hyperbolic import HyperbolicAdam, QuantumAnnealer
from hyperpreturb.final_files.utils.manifolds import PoincareBall

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------
# Adaptive Curriculum Learning
# ----------------------------
class ComplexityScheduler(tf.keras.callbacks.Callback):
    """
    Callback for curriculum learning by gradually increasing task complexity.
    """
    def __init__(self, env, factor=1.2, frequency=10):
        """
        Initialize the complexity scheduler.
        
        Args:
            env: Environment with increase_complexity method
            factor: Factor by which to increase complexity. Default: 1.2
            frequency: How often to increase complexity (epochs). Default: 10
        """
        super().__init__()
        self.env = env
        self.factor = factor
        self.frequency = frequency
        
    def on_epoch_end(self, epoch, logs=None):
        """Increase complexity at the end of specified epochs."""
        if epoch > 0 and epoch % self.frequency == 0:
            logger.info(f"Epoch {epoch}: Increasing complexity by factor {self.factor}")
            self.env.increase_complexity(self.factor)

# ----------------------------
# Perturbation Environment
# ----------------------------
class PerturbationEnv:
    """
    Environment for simulating gene perturbations and evaluating their effects.
    """
    
    def __init__(self, adata, complexity=1.0):
        """
        Initialize the perturbation environment.
        
        Args:
            adata: AnnData object with gene expression data
            complexity: Initial complexity level. Default: 1.0
        """
        self.adata = adata
        self.current_state = None
        self.complexity = complexity
        self.targets = None
        self._prepare_targets()
        self._reset()
        
    def _prepare_targets(self):
        """Prepare target variables for supervised learning."""
        # If perturbation annotations exist in the data
        if 'perturbation' in self.adata.obs:
            # Create one-hot encoding of perturbations
            perturbations = self.adata.obs['perturbation'].unique()
            pert_dict = {pert: i for i, pert in enumerate(perturbations)}
            
            self.targets = np.zeros((self.adata.n_obs, len(perturbations)))
            for i, pert in enumerate(self.adata.obs['perturbation']):
                self.targets[i, pert_dict[pert]] = 1
        else:
            # Use expression values directly as targets
            self.targets = self.adata.X.copy()
    
    def _reset(self):
        """Reset the environment to initial state."""
        self.current_state = tf.zeros(self.adata.n_vars)
        return self.current_state
    
    def step(self, action):
        """
        Take a step in the environment by applying a perturbation.
        
        Args:
            action: Perturbation action (gene indices to perturb)
            
        Returns:
            Tuple of (new_state, reward, done, info)
        """
        # Convert sparse/one-hot action to dense representation if needed
        if len(action.shape) > 1 and action.shape[1] > 1:
            action = tf.argmax(action, axis=1)
        
        # Apply perturbation effect based on action
        if hasattr(self.adata.X, 'toarray'):
            X = self.adata.X.toarray()
        else:
            X = self.adata.X
        
        # Simulate perturbation effect
        perturb_effect = np.zeros_like(self.current_state)
        for i, gene_idx in enumerate(action):
            # Apply stronger perturbation based on complexity
            perturb_effect += self.complexity * X[:, gene_idx].mean()
        
        self.current_state = perturb_effect
        
        # Calculate reward based on perturbation effect
        reward = tf.reduce_mean(perturb_effect)
        
        return self.current_state, reward, False, {}

    def increase_complexity(self, factor=1.2):
        """Increase the complexity of the environment."""
        self.complexity *= factor
        logger.info(f"Environment complexity increased to {self.complexity}")

# ----------------------------
# Distributed Training Strategy
# ----------------------------
def create_training_strategy(num_genes, curvature=1.0):
    """
    Create distributed training strategy and XLA-optimized training step.
    
    Args:
        num_genes: Number of genes in the model
        curvature: Curvature of hyperbolic space. Default: 1.0
        
    Returns:
        Tuple of (strategy, model, optimizer, train_step)
    """
    # Check available devices for distribution
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Using MirroredStrategy with {len(gpus)} GPUs")
    else:
        strategy = tf.distribute.get_strategy()  # Default strategy
        logger.info("Using default strategy")
    
    # Initialize model, optimizer, and environment within strategy scope
    with strategy.scope():
        model = HyperPerturbModel(num_genes, curvature)
        manifold = PoincareBall(curvature)
        optimizer = HyperbolicAdam(manifold=manifold, learning_rate=3e-4)
    
    # XLA-optimized training step
    @tf.function(jit_compile=True)
    def train_step(inputs):
        """Optimized training step with gradient tape."""
        with tf.GradientTape(persistent=True) as tape:
            states, actions, returns = inputs
            logits, values = model((states, tf.sparse.eye(num_genes)))  # Use identity matrix if no specific adjacency
            
            advantage = returns - values
            policy_loss = -tf.reduce_mean(actions * tf.nn.log_softmax(logits) * advantage)
            value_loss = tf.reduce_mean(tf.square(advantage))
            total_loss = policy_loss + 0.5 * value_loss
            
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total_loss
    
    return strategy, model, optimizer, train_step

# ----------------------------
# Full Training Pipeline
# ----------------------------
def train_model(adata, adj_matrix=None, model_dir="models/saved", 
                epochs=200, batch_size=128, learning_rate=3e-4,
                curvature=1.0, validation_split=0.1):
    """
    Full training pipeline for the HyperPerturb model.
    
    Args:
        adata: AnnData object with gene expression data
        adj_matrix: Adjacency matrix (optional)
        model_dir: Directory to save the model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        curvature: Curvature of hyperbolic space
        validation_split: Fraction of data to use for validation
        
    Returns:
        Trained model and training history
    """
    # Create timestamp for model directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(model_dir, f"hyperperturb-{timestamp}")
    os.makedirs(model_path, exist_ok=True)
    
    # Save training configuration
    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "curvature": curvature,
        "validation_split": validation_split,
        "n_genes": adata.n_vars,
        "n_cells": adata.n_obs,
    }
    with open(os.path.join(model_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Set up environment and curriculum learning
    env = PerturbationEnv(adata)
    curriculum = ComplexityScheduler(env)
    
    # Create training strategy and model
    strategy, model, _, _ = create_training_strategy(adata.n_vars, curvature)
    
    # Prepare data
    if adj_matrix is None:
        # Create identity adjacency matrix if none provided
        adj_matrix = tf.sparse.eye(adata.n_vars, adata.n_vars)
    
    # Convert sparse X to dense if needed
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # Set up quantum annealing schedule
    q_schedule = QuantumAnnealer(learning_rate, T_max=epochs)
    
    # Compile model with custom optimizer
    with strategy.scope():
        model.compile(
            optimizer=HyperbolicAdam(
                learning_rate=q_schedule,
                manifold=PoincareBall(curvature)
            ),
            loss=[
                tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                tf.keras.losses.MeanSquaredError()
            ],
            metrics={
                'policy': [
                    tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top1'),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top5')
                ],
                'value': [
                    tf.keras.metrics.MeanAbsoluteError(name='mae')
                ]
            }
        )
    
    # Set up callbacks
    callbacks = [
        curriculum,
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_path, 'logs'),
            histogram_freq=1,
            update_freq=100
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_path, 'checkpoints/model_{epoch:02d}'),
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )
    ]
    
    # Train model
    logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
    history = model.fit(
        x=(X, adj_matrix),
        y=env.targets,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(model_path, 'final_model'))
    logger.info(f"Model saved to {os.path.join(model_path, 'final_model')}")
    
    return model, history