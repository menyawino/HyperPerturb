import tensorflow as tf
import numpy as np
import os
import logging
import json
from datetime import datetime
from pathlib import Path

from hyperpreturb.models import HyperPerturbModel
from hyperpreturb.models.hyperbolic import QuantumAnnealer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def safe_kl_divergence(eps=1e-6, name="policy_kld"):
    """KL-divergence loss with safeguards for numerical stability.

    Clips predictions to [eps, 1] and renormalizes so they form
    valid probability distributions before computing KL.
    """
    base_kl = tf.keras.losses.KLDivergence(name=name)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, eps, 1.0)
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        return base_kl(y_true, y_pred)

    return loss_fn

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
# Full Training Pipeline (Graph-based HyperPerturbModel)
# ----------------------------
def train_model(adata, adj_matrix=None, model_dir="models/saved", 
                epochs=200, batch_size=128, learning_rate=1e-5,
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
    
    # Set up environment and curriculum learning (defines perturbation labels per cell)
    env = PerturbationEnv(adata)
    curriculum = ComplexityScheduler(env)

    # ------------------------
    # Build graph inputs
    # Nodes = genes, adjacency = gene-gene graph
    # ------------------------
    n_genes = adata.n_vars

    if adj_matrix is None:
        # Identity adjacency if none provided
        adj_matrix = tf.eye(n_genes, n_genes, dtype=tf.float32)
    else:
        # Ensure dense float32 (no extra batch dim)
        if isinstance(adj_matrix, tf.SparseTensor):
            adj_matrix = tf.sparse.to_dense(adj_matrix)
        adj_matrix = tf.cast(adj_matrix, tf.float32)

    # Expression matrix: cells × genes
    if hasattr(adata.X, "toarray"):
        X_dense = adata.X.toarray().astype("float32")
    else:
        X_dense = np.asarray(adata.X, dtype="float32")

    # Per-gene features via mean expression over cells
    # Result shape: (n_genes, 1); batch dimension will be handled by Keras
    gene_features = np.mean(X_dense, axis=0, keepdims=False).astype("float32")  # (n_genes,)
    gene_features = gene_features[:, np.newaxis]  # (n_genes, 1)

    # Sanity logging for inputs
    logger.info(
        "Gene features stats - min: %.4f, max: %.4f, mean: %.4f",
        float(gene_features.min()),
        float(gene_features.max()),
        float(gene_features.mean()),
    )
    logger.info(
        "Adjacency stats - min: %.4f, max: %.4f, mean: %.4f",
        float(tf.reduce_min(adj_matrix).numpy()),
        float(tf.reduce_max(adj_matrix).numpy()),
        float(tf.reduce_mean(adj_matrix).numpy()),
    )

    # ------------------------
    # Graph-level targets: per-gene × per-perturbation rewards (Option 2)
    # ------------------------
    # env.targets: (n_cells, n_perts) one-hot or similar
    targets = env.targets
    if hasattr(targets, "toarray"):
        targets = targets.toarray()
    targets = np.asarray(targets, dtype="float32")  # (n_cells, n_perts)

    # log fold-change: (n_cells, n_genes)
    if "log_fold_change" not in adata.obsm:
        raise ValueError("Expected 'log_fold_change' in adata.obsm for per-gene rewards.")

    lfc = adata.obsm["log_fold_change"]
    if hasattr(lfc, "toarray"):
        lfc = lfc.toarray()
    lfc = np.asarray(lfc, dtype="float32")  # (n_cells, n_genes)

    n_cells, n_perts = targets.shape
    _, n_genes_check = lfc.shape
    if n_genes_check != n_genes:
        raise ValueError(f"Mismatch between n_genes from adata.n_vars ({n_genes}) and log_fold_change second dim ({n_genes_check}).")

    # Compute per-perturbation × per-gene reward as mean |lfc| over cells with that perturbation
    per_pert_gene_reward = np.zeros((n_perts, n_genes), dtype="float32")
    for p in range(n_perts):
        mask = targets[:, p] > 0.5
        if np.any(mask):
            per_pert_gene_reward[p] = np.mean(np.abs(lfc[mask]), axis=0)
        else:
            per_pert_gene_reward[p] = 0.0

    # Transpose to (n_genes, n_perts)
    per_gene_pert_reward = per_pert_gene_reward.T  # (n_genes, n_perts)

    # Normalize per gene across perturbations to form probability distributions
    sum_per_gene = np.sum(per_gene_pert_reward, axis=-1, keepdims=True)
    sum_per_gene = np.where(sum_per_gene == 0.0, 1.0, sum_per_gene)  # avoid division by zero
    per_gene_pert_dist = per_gene_pert_reward / sum_per_gene  # (n_genes, n_perts)

    # Label smoothing / epsilon to avoid exact zeros and ones that can
    # destabilize KL divergence (log(0) -> -inf).
    eps = 1e-5
    per_gene_pert_dist = np.clip(per_gene_pert_dist, eps, 1.0)
    per_gene_pert_dist = per_gene_pert_dist / np.sum(per_gene_pert_dist, axis=-1, keepdims=True)

    # Sanity logging for policy targets
    logger.info(
        "Policy target stats - min: %.6f, max: %.6f, mean: %.6f",
        float(per_gene_pert_dist.min()),
        float(per_gene_pert_dist.max()),
        float(per_gene_pert_dist.mean()),
    )

    # Batch dimension for policy target
    graph_policy_target = per_gene_pert_dist[np.newaxis, ...]  # (1, n_genes, n_perts)

    # Per-gene scalar value target: mean (unnormalized) reward over perturbations
    per_gene_value = np.mean(per_gene_pert_reward, axis=-1, keepdims=True)  # (n_genes, 1)
    graph_value_target = per_gene_value[np.newaxis, ...]  # (1, n_genes, 1)

    # Sanity logging for value targets
    logger.info(
        "Value target stats - min: %.6f, max: %.6f, mean: %.6f",
        float(per_gene_value.min()),
        float(per_gene_value.max()),
        float(per_gene_value.mean()),
    )

    y_targets = [graph_policy_target, graph_value_target]

    # ------------------------
    # Build HyperPerturbModel on this graph
    # ------------------------
    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = HyperPerturbModel(
            num_genes=n_genes,
            num_perts=n_perts,
            curvature=curvature,
        )

        # Fallback: use standard Adam for stability diagnostics.
        # If this runs without NaNs, the issue likely lies in the
        # hyperbolic optimizer/manifold interaction rather than
        # the model architecture or losses.
        q_schedule = QuantumAnnealer(learning_rate, T_max=epochs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=q_schedule)
        model.compile(
            optimizer=optimizer,
            # Policy head: predict per-gene distribution over perturbations
            loss=[
                safe_kl_divergence(name="policy_kld"),
                tf.keras.losses.MeanSquaredError(name="value_mse"),
            ],
            loss_weights=[1.0, 0.5],
            metrics=[
                tf.keras.metrics.MeanAbsoluteError(name="policy_mae"),
                tf.keras.metrics.MeanAbsoluteError(name="value_mae"),
            ],
        )

        # Explicitly build model with expected input shapes: (batch, n_genes, feat_dim) and (batch, n_genes, n_genes)
        model.build(
            input_shape=[
                (None, n_genes, 1),        # gene_features
                (None, n_genes, n_genes),  # adj_matrix
            ]
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
            filepath=os.path.join(model_path, 'checkpoints', 'model_{epoch:02d}.keras'),
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
    
    # Prepare inputs with batch dimension = 1
    x_gene = gene_features[np.newaxis, ...]          # (1, n_genes, 1)
    x_adj = adj_matrix[np.newaxis, ...]             # (1, n_genes, n_genes)

    logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
    history = model.fit(
        x=[x_gene, x_adj],
        y=y_targets,
        epochs=epochs,
        batch_size=1,  # graph-level batch
        validation_split=0.0,  # single graph, use callbacks only
        callbacks=callbacks,
        verbose=1,
    )
    
    # Save final model in Keras format
    final_path = os.path.join(model_path, 'final_model.keras')
    model.save(final_path)
    logger.info(f"Model saved to {final_path}")
    
    return model, history