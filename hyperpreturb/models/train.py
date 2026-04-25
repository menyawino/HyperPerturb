import json
import logging
import os
import random
from datetime import datetime

import numpy as np
import tensorflow as tf

from hyperpreturb.models import SignedHyperPerturbModel
from hyperpreturb.models.hyperbolic import HyperbolicAdam, QuantumAnnealer
from hyperpreturb.models.training_utils import (
    DEFAULT_CONTROL_VALUE,
    DEFAULT_PERTURBATION_KEY,
    build_graph_inputs,
    build_signed_effect_targets,
    masked_signed_huber_loss,
    masked_signed_mae,
    split_anndata_by_perturbation,
)
from hyperpreturb.utils.manifolds import PoincareBall


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def safe_kl_divergence(eps=1e-6, name="policy_kld"):
    """KL-divergence with clipping to avoid log(0) blowups."""
    base_kl = tf.keras.losses.KLDivergence(name=name)

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, eps, 1.0)
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        return base_kl(y_true, y_pred)

    return loss_fn


class ComplexityScheduler(tf.keras.callbacks.Callback):
    """Periodically bump the PerturbationEnv complexity (curriculum learning)."""

    def __init__(self, env, factor=1.2, frequency=10):
        super().__init__()
        self.env = env
        self.factor = factor
        self.frequency = frequency

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 0 and epoch % self.frequency == 0:
            logger.info("Epoch %s: bumping complexity by %.2fx", epoch, self.factor)
            self.env.increase_complexity(self.factor)


class PerturbationEnv:
    """Wraps AnnData for perturbation targets and a simple complexity dial."""

    def __init__(self, adata, complexity=1.0):
        self.adata = adata
        self.current_state = None
        self.complexity = complexity
        self.targets = None
        self._prepare_targets()
        self._reset()

    def _prepare_targets(self):
        """Build one-hot perturbation labels from adata.obs['perturbation']."""
        if "perturbation" in self.adata.obs:
            perturbations = self.adata.obs["perturbation"].unique()
            pert_dict = {pert: i for i, pert in enumerate(perturbations)}

            self.targets = np.zeros((self.adata.n_obs, len(perturbations)))
            for i, pert in enumerate(self.adata.obs["perturbation"]):
                self.targets[i, pert_dict[pert]] = 1
        else:
            self.targets = self.adata.X.copy()

    def _reset(self):
        """Reset the environment to initial state."""
        self.current_state = tf.zeros(self.adata.n_vars)
        return self.current_state

    def step(self, action):
        """RL-style step (not used in actual training, kept for API compat)."""
        if len(action.shape) > 1 and action.shape[1] > 1:
            action = tf.argmax(action, axis=1)
        action_np = np.asarray(action)

        if hasattr(self.adata.X, "toarray"):
            expression = self.adata.X.toarray()
        else:
            expression = self.adata.X

        perturb_effect = np.zeros_like(self.current_state)
        for gene_idx in action_np:
            perturb_effect += self.complexity * expression[:, gene_idx].mean()

        self.current_state = perturb_effect
        reward = tf.reduce_mean(perturb_effect)
        return self.current_state, reward, False, {}

    def increase_complexity(self, factor=1.2):
        """Increase the complexity of the environment."""
        self.complexity *= factor
        logger.info("Environment complexity increased to %s", self.complexity)


def train_model(
    adata,
    adj_matrix=None,
    model_dir="models/saved",
    epochs=200,
    batch_size=128,
    learning_rate=1e-5,
    curvature=1.0,
    validation_split=0.1,
    debug=False,
    policy_only=False,
    euclidean_baseline=False,
    seed=42,
    deterministic=True,
    perturbation_key=DEFAULT_PERTURBATION_KEY,
    control_value=DEFAULT_CONTROL_VALUE,
):
    """Train the advanced graph model with held-out perturbation validation.

    The policy head predicts signed log fold-change in gene space. Columns are
    masked so that train and validation can supervise different perturbation
    conditions without changing the output dimensionality.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(model_dir, f"hyperperturb-{timestamp}")
    os.makedirs(model_path, exist_ok=True)

    env = PerturbationEnv(adata)
    curriculum = ComplexityScheduler(env)

    n_genes = adata.n_vars
    if adj_matrix is None:
        raise ValueError("adj_matrix is required for graph training and cannot be None.")

    train_adata, validation_adata, split_metadata = split_anndata_by_perturbation(
        adata,
        validation_split=validation_split,
        perturbation_key=perturbation_key,
        control_value=control_value,
        seed=seed,
    )

    x_gene, x_adj = build_graph_inputs(train_adata, adj_matrix=adj_matrix)
    logger.info(
        "Gene features stats - min: %.4f, max: %.4f, mean: %.4f",
        float(np.min(x_gene)),
        float(np.max(x_gene)),
        float(np.mean(x_gene)),
    )
    logger.info(
        "Adjacency stats - min: %.4f, max: %.4f, mean: %.4f",
        float(tf.reduce_min(x_adj).numpy()),
        float(tf.reduce_max(x_adj).numpy()),
        float(tf.reduce_mean(x_adj).numpy()),
    )

    graph_policy_target, graph_value_target, target_metadata = build_signed_effect_targets(
        train_adata,
        supervised_perturbations=split_metadata["train_perturbations"],
        perturbation_key=perturbation_key,
        control_value=control_value,
    )
    train_effects = graph_policy_target[..., :n_genes]
    logger.info(
        "Policy target stats - min: %.6f, max: %.6f, mean: %.6f",
        float(np.min(train_effects)),
        float(np.max(train_effects)),
        float(np.mean(train_effects)),
    )
    logger.info(
        "Value target stats - min: %.6f, max: %.6f, mean: %.6f",
        float(np.min(graph_value_target)),
        float(np.max(graph_value_target)),
        float(np.mean(graph_value_target)),
    )

    y_targets = [graph_policy_target, graph_value_target]

    validation_data = None
    if validation_adata is not None:
        val_x_gene, val_x_adj = build_graph_inputs(validation_adata, adj_matrix=adj_matrix)
        val_policy_target, val_value_target, _ = build_signed_effect_targets(
            validation_adata,
            supervised_perturbations=split_metadata["validation_perturbations"],
            perturbation_key=perturbation_key,
            control_value=control_value,
        )
        val_targets = val_policy_target if policy_only else [val_policy_target, val_value_target]
        validation_data = ([val_x_gene, val_x_adj], val_targets)

    config = {
        "epochs": epochs,
        "batch_size": 1,
        "requested_batch_size": batch_size,
        "learning_rate": learning_rate,
        "curvature": curvature,
        "validation_split": validation_split,
        "seed": seed,
        "deterministic": deterministic,
        "n_genes": adata.n_vars,
        "n_cells": adata.n_obs,
        "policy_target": "signed_log_fold_change",
        "policy_target_space": "gene_space",
        "split_strategy": split_metadata["split_strategy"],
        "perturbation_key": perturbation_key,
        "control_value": control_value,
        "train_perturbations": split_metadata["train_perturbations"],
        "validation_perturbations": split_metadata["validation_perturbations"],
        "train_cells": int(train_adata.n_obs),
        "validation_cells": int(validation_adata.n_obs if validation_adata is not None else 0),
        "gene_names": [str(name) for name in adata.var_names.tolist()],
        "supervised_train_columns": target_metadata["perturbation_indices"],
    }
    with open(os.path.join(model_path, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with open(os.path.join(model_path, "split_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(split_metadata, handle, indent=2)

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        if euclidean_baseline:
            from hyperpreturb.models import EuclideanGraphConv

            class EuclideanPerturbModel(SignedHyperPerturbModel):
                """SignedHyperPerturbModel variant that uses EuclideanGraphConv."""

                def __init__(self, num_genes, curvature=1.0, **kwargs):
                    super().__init__(num_genes=num_genes, curvature=curvature, **kwargs)
                    self.encoder_gcn1 = EuclideanGraphConv(512)
                    self.encoder_gcn2 = EuclideanGraphConv(256)
                    self.policy_gcn = EuclideanGraphConv(128)
                    self.value_gcn = EuclideanGraphConv(128)

            base_model_cls = EuclideanPerturbModel
        else:
            base_model_cls = SignedHyperPerturbModel

        if debug:
            if policy_only:

                class DebugPolicyOnlyModel(base_model_cls):  # type: ignore[misc]
                    def call(self, inputs, training=False, debug=False):  # type: ignore[override]
                        policy_scores, _ = super().call(inputs, training=training, debug=True)
                        return policy_scores

                model = DebugPolicyOnlyModel(num_genes=n_genes, curvature=curvature)
            else:

                class DebugHyperPerturbModel(base_model_cls):  # type: ignore[misc]
                    def call(self, inputs, training=False, debug=False):  # type: ignore[override]
                        return super().call(inputs, training=training, debug=True)

                model = DebugHyperPerturbModel(num_genes=n_genes, curvature=curvature)
        else:
            model = base_model_cls(num_genes=n_genes, curvature=curvature)

        if debug:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            lr_schedule = QuantumAnnealer(learning_rate, T_max=epochs)
            optimizer = HyperbolicAdam(
                learning_rate=lr_schedule,
                manifold=PoincareBall(curvature),
            )

        if policy_only:
            loss = masked_signed_huber_loss(name="policy_huber")
            loss_weights = None
            metrics = [masked_signed_mae(name="policy_mae")]
            y_used = graph_policy_target
        else:
            loss = [
                masked_signed_huber_loss(name="policy_huber"),
                tf.keras.losses.MeanSquaredError(name="value_mse"),
            ]
            loss_weights = [1.0, 0.5]
            metrics = [
                masked_signed_mae(name="policy_mae"),
                tf.keras.metrics.MeanAbsoluteError(name="value_mae"),
            ]
            y_used = y_targets

        model.compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
        )

        _ = model(
            (
                tf.zeros((1, n_genes, 1), dtype=tf.float32),
                tf.zeros((1, n_genes, n_genes), dtype=tf.float32),
            )
        )

    class NaNMonitor(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            bad_metric = False
            for metric_name, metric_value in logs.items():
                if metric_value is None:
                    continue
                if isinstance(metric_value, (float, int)) and not np.isfinite(metric_value):
                    logger.warning("Epoch %s: metric %s is non-finite: %s", epoch, metric_name, metric_value)
                    bad_metric = True
            if bad_metric and debug:
                logger.error("Non-finite metrics detected; stopping training early due to debug mode.")
                self.model.stop_training = True

    callbacks: list[tf.keras.callbacks.Callback] = [curriculum]
    if debug:
        callbacks.append(NaNMonitor())
    else:
        monitor_metric = "val_loss" if validation_data is not None else "loss"
        os.makedirs(os.path.join(model_path, "checkpoints"), exist_ok=True)
        callbacks.extend(
            [
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(model_path, "logs"),
                    histogram_freq=1,
                    update_freq=100,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(model_path, "checkpoints", "model_{epoch:02d}.keras"),
                    save_best_only=True,
                    monitor=monitor_metric,
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor=monitor_metric,
                    patience=20,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=monitor_metric,
                    factor=0.5,
                    patience=10,
                    min_lr=1e-6,
                ),
            ]
        )

    logger.info(
        "Starting training for %s epochs with requested batch size %s (effective graph batch size is 1)",
        epochs,
        batch_size,
    )
    fit_kwargs = {
        "x": [x_gene, x_adj],
        "y": y_used,
        "epochs": epochs,
        "batch_size": 1,
        "validation_split": 0.0,
        "callbacks": callbacks,
        "verbose": 1,
    }
    if validation_data is not None:
        fit_kwargs["validation_data"] = validation_data

    history = model.fit(**fit_kwargs)

    final_path = os.path.join(model_path, "final_model.keras")
    model.save(final_path)
    logger.info("Model saved to %s", final_path)

    return model, history