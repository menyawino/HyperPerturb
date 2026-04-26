import json
import os

import numpy as np
import scipy.sparse as sp
import tensorflow as tf


DEFAULT_PERTURBATION_KEY = "perturbation"
DEFAULT_CONTROL_VALUE = "non-targeting"


def _normalize_mapping_targets(raw_targets):
    if isinstance(raw_targets, str):
        return [{"gene": raw_targets, "weight": 1.0}]

    if isinstance(raw_targets, (list, tuple)) and raw_targets and all(
        isinstance(target, dict) and "gene" in target and "weight" in target
        for target in raw_targets
    ):
        normalized = []
        total_weight = 0.0
        for target in raw_targets:
            weight = float(target["weight"])
            if weight <= 0.0:
                raise ValueError("Perturbation-to-gene mapping weights must be positive.")
            normalized.append({"gene": str(target["gene"]), "weight": weight})
            total_weight += weight
        for entry in normalized:
            entry["weight"] /= total_weight
        return normalized

    if isinstance(raw_targets, dict):
        if not raw_targets:
            raise ValueError("Perturbation-to-gene mapping entries cannot be empty.")
        normalized = []
        total_weight = 0.0
        for gene_name, raw_weight in raw_targets.items():
            weight = float(raw_weight)
            if weight <= 0.0:
                raise ValueError("Perturbation-to-gene mapping weights must be positive.")
            normalized.append({"gene": str(gene_name), "weight": weight})
            total_weight += weight
        for entry in normalized:
            entry["weight"] /= total_weight
        return normalized

    if isinstance(raw_targets, (list, tuple, set)):
        targets = [str(target) for target in raw_targets]
        if not targets:
            raise ValueError("Perturbation-to-gene mapping entries cannot be empty.")
        weight = 1.0 / len(targets)
        return [{"gene": target, "weight": weight} for target in targets]

    raise TypeError(
        "Perturbation-to-gene mapping entries must be a gene name, a list of gene names, or a gene-to-weight dictionary."
    )


def normalize_perturbation_gene_map(perturbation_gene_map):
    """Normalize heterogeneous mapping inputs to a JSON-serializable format."""
    if perturbation_gene_map is None:
        return None

    normalized = {}
    for perturbation_name, raw_targets in perturbation_gene_map.items():
        normalized[str(perturbation_name)] = _normalize_mapping_targets(raw_targets)
    return normalized


def load_perturbation_gene_map(mapping_path):
    """Load perturbation-to-gene mappings from JSON or delimited text."""
    if mapping_path is None:
        return None
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Perturbation mapping file not found: {mapping_path}")

    _, extension = os.path.splitext(mapping_path)
    extension = extension.lower()
    if extension == ".json":
        with open(mapping_path, "r", encoding="utf-8") as handle:
            return normalize_perturbation_gene_map(json.load(handle))

    import pandas as pd

    mapping_df = pd.read_csv(mapping_path, sep=None, engine="python")
    perturbation_column = next(
        (
            column
            for column in ["perturbation", "perturbation_label", "condition", "guide", "guide_id"]
            if column in mapping_df.columns
        ),
        None,
    )
    gene_column = next(
        (column for column in ["gene", "gene_symbol", "target_gene", "target"] if column in mapping_df.columns),
        None,
    )
    if perturbation_column is None or gene_column is None:
        raise ValueError(
            "Perturbation mapping tables must contain a perturbation column and a gene column. "
            f"Available columns: {list(mapping_df.columns)}"
        )

    weight_column = next((column for column in ["weight", "mapping_weight"] if column in mapping_df.columns), None)
    mapping = {}
    for perturbation_name, group in mapping_df.groupby(perturbation_column):
        entries = {}
        for _, row in group.iterrows():
            gene_name = str(row[gene_column])
            raw_weight = 1.0 if weight_column is None else float(row[weight_column])
            entries[gene_name] = entries.get(gene_name, 0.0) + raw_weight
        mapping[str(perturbation_name)] = entries

    return normalize_perturbation_gene_map(mapping)


def to_dense_array(matrix, dtype="float32"):
    """Convert sparse or array-like matrices to a dense NumPy array."""
    if sp.issparse(matrix):
        return matrix.toarray().astype(dtype)
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=dtype)
    return np.asarray(matrix, dtype=dtype)


def _validate_perturbation_annotations(adata, perturbation_key, control_value):
    if perturbation_key not in adata.obs.columns:
        raise ValueError(
            f"Perturbation key '{perturbation_key}' is missing from adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    labels = adata.obs[perturbation_key].astype(str).to_numpy()
    if control_value not in set(labels):
        raise ValueError(
            f"Control value '{control_value}' not found in adata.obs['{perturbation_key}']."
        )

    return labels


def build_graph_inputs(adata, adj_matrix=None):
    """Build graph-level gene features and adjacency tensors from AnnData."""
    n_genes = adata.n_vars
    x_dense = to_dense_array(adata.X)

    gene_features = np.mean(x_dense, axis=0, keepdims=False).astype("float32")
    gene_features = gene_features[:, np.newaxis]

    if adj_matrix is None:
        adj_tensor = tf.eye(n_genes, n_genes, dtype=tf.float32)
    else:
        if isinstance(adj_matrix, tf.SparseTensor):
            adj_matrix = tf.sparse.to_dense(adj_matrix)
        elif sp.issparse(adj_matrix):
            adj_matrix = adj_matrix.toarray()

        adj_tensor = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)
        if adj_tensor.shape.rank != 2:
            raise ValueError(f"adj_matrix must be rank-2, got rank {adj_tensor.shape.rank}")
        if adj_tensor.shape[0] != n_genes or adj_tensor.shape[1] != n_genes:
            raise ValueError(
                f"adj_matrix shape must be ({n_genes}, {n_genes}), got {adj_tensor.shape}"
            )

    return gene_features[np.newaxis, ...], adj_tensor[tf.newaxis, ...]


def split_anndata_by_perturbation(
    adata,
    validation_split=0.1,
    perturbation_key=DEFAULT_PERTURBATION_KEY,
    control_value=DEFAULT_CONTROL_VALUE,
    seed=42,
):
    """Split AnnData by perturbation condition while partitioning controls by cell.

    Every non-control perturbation is assigned wholly to train or validation.
    Control cells are partitioned by cell so each split can recompute logFC from
    its own controls without leaking held-out perturbation cells.
    """
    if validation_split < 0.0 or validation_split >= 1.0:
        raise ValueError("validation_split must be in the range [0.0, 1.0).")

    labels = _validate_perturbation_annotations(adata, perturbation_key, control_value)
    unique_perts = sorted({label for label in labels if label != control_value})

    if validation_split == 0.0:
        metadata = {
            "split_strategy": "held_out_perturbation",
            "perturbation_key": perturbation_key,
            "control_value": control_value,
            "train_perturbations": unique_perts,
            "validation_perturbations": [],
            "train_obs_names": adata.obs_names.astype(str).tolist(),
            "validation_obs_names": [],
        }
        return adata.copy(), None, metadata

    if len(unique_perts) < 2:
        raise ValueError(
            "Perturbation-held-out validation requires at least two non-control perturbations."
        )

    rng = np.random.default_rng(seed)
    shuffled_perts = list(unique_perts)
    rng.shuffle(shuffled_perts)

    n_val_perts = int(round(len(shuffled_perts) * validation_split))
    n_val_perts = min(max(n_val_perts, 1), len(shuffled_perts) - 1)

    validation_perturbations = sorted(shuffled_perts[:n_val_perts])
    train_perturbations = sorted(shuffled_perts[n_val_perts:])

    control_indices = np.flatnonzero(labels == control_value)
    if control_indices.size < 2:
        raise ValueError(
            "Perturbation-held-out validation requires at least two control cells."
        )

    rng.shuffle(control_indices)
    n_val_controls = int(round(control_indices.size * validation_split))
    n_val_controls = min(max(n_val_controls, 1), control_indices.size - 1)

    validation_control_indices = control_indices[:n_val_controls]
    train_control_indices = control_indices[n_val_controls:]

    all_indices = np.arange(adata.n_obs)
    train_mask = np.isin(labels, train_perturbations) | np.isin(all_indices, train_control_indices)
    validation_mask = np.isin(labels, validation_perturbations) | np.isin(all_indices, validation_control_indices)

    if np.any(train_mask & validation_mask):
        raise RuntimeError("Train and validation masks overlap unexpectedly.")
    if not np.any(train_mask):
        raise ValueError("Training split is empty after perturbation holdout.")
    if not np.any(validation_mask):
        raise ValueError("Validation split is empty after perturbation holdout.")

    metadata = {
        "split_strategy": "held_out_perturbation",
        "perturbation_key": perturbation_key,
        "control_value": control_value,
        "train_perturbations": train_perturbations,
        "validation_perturbations": validation_perturbations,
        "train_obs_names": adata.obs_names[train_mask].astype(str).tolist(),
        "validation_obs_names": adata.obs_names[validation_mask].astype(str).tolist(),
    }

    return adata[train_mask].copy(), adata[validation_mask].copy(), metadata


def compute_split_log_fold_change(
    adata,
    perturbation_key=DEFAULT_PERTURBATION_KEY,
    control_value=DEFAULT_CONTROL_VALUE,
):
    """Recompute split-local signed log fold change using only that split's controls."""
    labels = _validate_perturbation_annotations(adata, perturbation_key, control_value)
    control_mask = labels == control_value
    if not np.any(control_mask):
        raise ValueError(
            f"No control cells found with {perturbation_key}={control_value} in the provided split."
        )

    x_dense = to_dense_array(adata.X)
    control_mean = np.mean(x_dense[control_mask], axis=0).astype("float32")
    return x_dense - control_mean


def resolve_gene_aligned_perturbations(adata, perturbation_names, perturbation_gene_map=None):
    """Resolve perturbation labels into gene-space targets in adata.var_names."""
    gene_names = [str(name) for name in adata.var_names.tolist()]
    gene_to_index = {gene_name: idx for idx, gene_name in enumerate(gene_names)}
    normalized_map = normalize_perturbation_gene_map(perturbation_gene_map)

    resolved = {}
    missing = []
    for perturbation_name in perturbation_names:
        targets = None
        if normalized_map is not None:
            targets = normalized_map.get(str(perturbation_name))
        if targets is None:
            targets = [{"gene": str(perturbation_name), "weight": 1.0}]

        resolved_targets = []
        for target in targets:
            gene_name = str(target["gene"])
            if gene_name not in gene_to_index:
                missing.append(f"{perturbation_name}->{gene_name}")
                continue
            resolved_targets.append(
                {
                    "gene_name": gene_name,
                    "gene_index": gene_to_index[gene_name],
                    "weight": float(target["weight"]),
                }
            )
        resolved[str(perturbation_name)] = resolved_targets

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            "Perturbation labels must either match genes in adata.var_names or be provided via an explicit perturbation-to-gene map. "
            f"Missing mappings: {preview}"
        )

    return resolved


def pack_masked_policy_targets(policy_target, policy_mask):
    """Pack dense signed targets and a same-shape supervision mask for Keras."""
    return np.concatenate([policy_target, policy_mask], axis=-1).astype("float32")


def build_signed_effect_targets(
    adata,
    supervised_perturbations,
    perturbation_key=DEFAULT_PERTURBATION_KEY,
    control_value=DEFAULT_CONTROL_VALUE,
    perturbation_gene_map=None,
):
    """Build signed gene x perturbation targets in gene space.

    The policy target has shape (1, n_genes, 2 * n_genes):
    - first n_genes columns: signed mean logFC for each supervised perturbation gene
    - second n_genes columns: binary mask indicating which perturbation columns are supervised

    The value target remains a per-gene sensitivity score, defined here as the
    mean absolute signed effect across the supervised perturbations in the split.
    """
    labels = _validate_perturbation_annotations(adata, perturbation_key, control_value)
    if not supervised_perturbations:
        raise ValueError("supervised_perturbations must contain at least one held-in perturbation.")

    resolved_perturbations = resolve_gene_aligned_perturbations(
        adata,
        supervised_perturbations,
        perturbation_gene_map=perturbation_gene_map,
    )
    signed_log_fold_change = compute_split_log_fold_change(
        adata,
        perturbation_key=perturbation_key,
        control_value=control_value,
    )

    n_genes = adata.n_vars
    policy_target = np.zeros((n_genes, n_genes), dtype="float32")
    policy_mask = np.zeros((n_genes, n_genes), dtype="float32")
    policy_weight_sums = np.zeros((n_genes,), dtype="float32")

    for perturbation_name in supervised_perturbations:
        perturbation_mask = labels == perturbation_name
        if not np.any(perturbation_mask):
            raise ValueError(
                f"No cells found for supervised perturbation '{perturbation_name}' in the provided split."
            )

        mean_effect = np.mean(signed_log_fold_change[perturbation_mask], axis=0).astype("float32")
        for target in resolved_perturbations[str(perturbation_name)]:
            gene_index = target["gene_index"]
            weight = target["weight"]
            policy_target[:, gene_index] += mean_effect * weight
            policy_weight_sums[gene_index] += weight

    perturbation_indices = [int(index) for index in np.flatnonzero(policy_weight_sums > 0.0)]
    if not perturbation_indices:
        raise ValueError("No supervised gene columns were resolved for the provided perturbations.")

    for gene_index in perturbation_indices:
        policy_target[:, gene_index] /= policy_weight_sums[gene_index]
        policy_mask[:, gene_index] = 1.0

    supervised_effects = policy_target[:, perturbation_indices]
    value_target = np.mean(np.abs(supervised_effects), axis=-1, keepdims=True).astype("float32")

    return (
        pack_masked_policy_targets(policy_target[np.newaxis, ...], policy_mask[np.newaxis, ...]),
        value_target[np.newaxis, ...],
        {
            "perturbation_indices": perturbation_indices,
            "supervised_perturbations": list(supervised_perturbations),
            "resolved_perturbation_gene_map": {
                perturbation_name: resolved_perturbations[str(perturbation_name)]
                for perturbation_name in supervised_perturbations
            },
        },
    )


def _unpack_policy_targets(y_true, y_pred):
    width = tf.shape(y_pred)[-1]
    targets = y_true[..., :width]
    mask = y_true[..., width:width * 2]
    return targets, mask


def masked_signed_huber_loss(delta=1.0, name="policy_huber"):
    """Huber loss over only the supervised perturbation columns."""

    def loss_fn(y_true, y_pred):
        targets, mask = _unpack_policy_targets(y_true, y_pred)
        abs_error = tf.abs(y_pred - targets)
        quadratic = tf.minimum(abs_error, delta)
        linear = abs_error - quadratic
        losses = 0.5 * tf.square(quadratic) + delta * linear
        masked_losses = losses * mask
        normalizer = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(masked_losses) / normalizer

    loss_fn.__name__ = name
    return loss_fn


def masked_signed_mae(name="policy_mae"):
    """Masked MAE over the supervised signed-effect columns."""

    def metric_fn(y_true, y_pred):
        targets, mask = _unpack_policy_targets(y_true, y_pred)
        absolute_error = tf.abs(y_pred - targets) * mask
        normalizer = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(absolute_error) / normalizer

    metric_fn.__name__ = name
    return metric_fn


def compute_masked_policy_metrics(y_true, y_pred, delta=1.0):
    """NumPy metrics for signed policy evaluation."""
    width = y_pred.shape[-1]
    targets = y_true[..., :width]
    mask = y_true[..., width:width * 2]

    abs_error = np.abs(y_pred - targets)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    huber = 0.5 * np.square(quadratic) + delta * linear

    normalizer = max(float(np.sum(mask)), 1.0)
    masked_mae = float(np.sum(abs_error * mask) / normalizer)
    masked_huber = float(np.sum(huber * mask) / normalizer)

    sign_mask = mask * (np.abs(targets) > 1e-6)
    sign_normalizer = max(float(np.sum(sign_mask)), 1.0)
    sign_agreement = float(
        np.sum((np.sign(y_pred) == np.sign(targets)).astype("float32") * sign_mask) / sign_normalizer
    )

    return {
        "policy_huber": masked_huber,
        "policy_mae": masked_mae,
        "policy_sign_accuracy": sign_agreement,
    }