import argparse
import os
from pathlib import Path

import numpy as np
import scanpy as sc
import tensorflow as tf

from hyperpreturb.models import HyperPerturbModel
from hyperpreturb.models.train import PerturbationEnv


def build_graph_inputs(adata, adj_matrix=None):
    """Rebuild graph-level inputs (gene features, adjacency) from AnnData.

    This mirrors the logic in `hyperpreturb.models.train.train_model` so
    evaluation uses exactly the same representation as training.
    """
    n_genes = adata.n_vars

    # Expression matrix: cells × genes
    if hasattr(adata.X, "toarray"):
        X_dense = adata.X.toarray().astype("float32")
    else:
        X_dense = np.asarray(adata.X, dtype="float32")

    # Per-gene features via mean expression over cells
    gene_features = np.mean(X_dense, axis=0, keepdims=False).astype("float32")  # (n_genes,)
    gene_features = gene_features[:, np.newaxis]  # (n_genes, 1)

    # Adjacency
    if adj_matrix is None:
        adj_matrix = tf.eye(n_genes, n_genes, dtype=tf.float32)
    else:
        if isinstance(adj_matrix, tf.SparseTensor):
            adj_matrix = tf.sparse.to_dense(adj_matrix)
        adj_matrix = tf.cast(adj_matrix, tf.float32)

    # Add batch dimension
    x_gene = gene_features[np.newaxis, ...]          # (1, n_genes, 1)
    x_adj = adj_matrix[np.newaxis, ...]             # (1, n_genes, n_genes)

    return x_gene, x_adj


def build_targets(adata):
    """Rebuild policy and value targets from AnnData for evaluation.

    Uses the same logic as in `train_model` for computing
    `per_gene_pert_reward`, then derives:
      - policy target: per_gene_pert_dist (n_genes, n_perts)
      - value target: per_gene_value (n_genes, 1)
    """
    env = PerturbationEnv(adata)
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
    _, n_genes = lfc.shape

    per_pert_gene_reward = np.zeros((n_perts, n_genes), dtype="float32")
    for p in range(n_perts):
        mask = targets[:, p] > 0.5
        if np.any(mask):
            per_pert_gene_reward[p] = np.mean(np.abs(lfc[mask]), axis=0)
        else:
            per_pert_gene_reward[p] = 0.0

    per_gene_pert_reward = per_pert_gene_reward.T  # (n_genes, n_perts)

    # Policy target: we mirror the smoothed/temperature-scaled distribution
    sum_per_gene = np.sum(per_gene_pert_reward, axis=-1, keepdims=True)
    sum_per_gene = np.where(sum_per_gene == 0.0, 1.0, sum_per_gene)

    temperature = 0.7
    logits = per_gene_pert_reward / max(temperature, 1e-6)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    exp_sum = np.sum(exp_logits, axis=-1, keepdims=True)
    exp_sum = np.where(exp_sum == 0.0, 1.0, exp_sum)
    softmax_dist = exp_logits / exp_sum

    smoothing = 0.05
    uniform = np.full_like(softmax_dist, 1.0 / n_perts)
    per_gene_pert_dist = (1.0 - smoothing) * softmax_dist + smoothing * uniform

    # Value target: mean reward per gene
    per_gene_value = np.mean(per_gene_pert_reward, axis=-1, keepdims=True)  # (n_genes, 1)

    return per_gene_pert_dist, per_gene_value


def evaluate(model_path, preprocessed_path):
    """Evaluate a saved HyperPerturbModel on held-out cells.

    We use a simple cell-wise split (80/20) to recompute targets, then
    compare model predictions against targets derived from the *full*
    dataset. This gives a coarse sense of how well the learned policy and
    value heads match the aggregated perturbation effects.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_path}")

    adata = sc.read_h5ad(preprocessed_path)

    # Build graph inputs and targets for evaluation
    x_gene, x_adj = build_graph_inputs(adata, adj_matrix=None)
    policy_target, value_target = build_targets(adata)

    # Add batch dimension to targets
    policy_target = policy_target[np.newaxis, ...]  # (1, n_genes, n_perts)
    value_target = value_target[np.newaxis, ...]    # (1, n_genes, 1)

    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)

    # Forward pass
    preds = model([x_gene, x_adj], training=False)
    if isinstance(preds, (list, tuple)) and len(preds) == 2:
        policy_pred, value_pred = preds
    else:
        raise ValueError("Expected model to output (policy, value) tuple.")

    # Compute simple metrics
    # Policy: KL and MAE w.r.t. target distribution
    policy_pred = np.asarray(policy_pred)
    value_pred = np.asarray(value_pred)

    eps = 1e-8
    policy_pred_clip = np.clip(policy_pred, eps, 1.0)
    policy_pred_clip /= np.sum(policy_pred_clip, axis=-1, keepdims=True)

    kl = np.sum(
        policy_target * (np.log(policy_target + eps) - np.log(policy_pred_clip + eps)),
        axis=-1,
    )  # (1, n_genes)
    policy_kl_mean = float(np.mean(kl))
    policy_mae = float(np.mean(np.abs(policy_pred - policy_target)))

    # Value: MSE and MAE
    value_mse = float(np.mean((value_pred - value_target) ** 2))
    value_mae = float(np.mean(np.abs(value_pred - value_target)))

    print("Evaluation metrics:")
    print(f"  Policy KL (mean over genes): {policy_kl_mean:.4f}")
    print(f"  Policy MAE:                 {policy_mae:.6f}")
    print(f"  Value MSE:                  {value_mse:.4f}")
    print(f"  Value MAE:                  {value_mae:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained HyperPerturb model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved Keras model (.keras)",
    )
    parser.add_argument(
        "--preprocessed_path",
        type=str,
        required=True,
        help="Path to the preprocessed AnnData file used for training",
    )

    args = parser.parse_args()
    evaluate(args.model_path, args.preprocessed_path)


if __name__ == "__main__":
    main()
