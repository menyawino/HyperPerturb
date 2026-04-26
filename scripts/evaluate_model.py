import argparse
import json
import os

import numpy as np
import scanpy as sc
import tensorflow as tf

from hyperpreturb.models import EuclideanPerturbModel, HyperPerturbModel, SignedHyperPerturbModel
from hyperpreturb.models.hyperbolic import HyperbolicAdam, QuantumAnnealer
from hyperpreturb.models.training_utils import (
    DEFAULT_CONTROL_VALUE,
    DEFAULT_PERTURBATION_KEY,
    build_graph_inputs,
    build_signed_effect_targets,
    compute_masked_policy_metrics,
    load_perturbation_gene_map,
    split_anndata_by_perturbation,
)
from hyperpreturb.utils.data_loader import create_adjacency_matrix, load_protein_network
from hyperpreturb.utils.manifolds import PoincareBall


def _resolve_model_dir(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if os.path.isdir(model_path):
        return model_path, os.path.join(model_path, "final_model.keras")
    if model_path.endswith(".keras"):
        return os.path.dirname(model_path), model_path
    raise ValueError("model_path must be a model directory or a final_model.keras file")


def _load_model_and_config(model_path):
    model_dir, keras_path = _resolve_model_dir(model_path)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    custom_objects = {
        "PoincareBall": PoincareBall,
        "HyperbolicAdam": HyperbolicAdam,
        "QuantumAnnealer": QuantumAnnealer,
        "HyperPerturbModel": HyperPerturbModel,
        "SignedHyperPerturbModel": SignedHyperPerturbModel,
        "EuclideanPerturbModel": EuclideanPerturbModel,
    }
    model = tf.keras.models.load_model(keras_path, custom_objects=custom_objects, compile=False)
    return model_dir, model, config


def _build_adjacency(adata, network_path=None, gene_mapping_path=None):
    if network_path is None:
        raise ValueError(
            "network_path is required for evaluation so held-out metrics use the same gene graph as training."
        )
    network_df = load_protein_network(network_path, gene_mapping_path=gene_mapping_path)
    return create_adjacency_matrix(network_df, adata.var_names.tolist())


def evaluate(
    model_path,
    preprocessed_path,
    network_path=None,
    gene_mapping_path=None,
    perturbation_gene_map=None,
    print_output=True,
):
    """Evaluate a trained model on held-out perturbation conditions.

    Returns a metrics dictionary so callers can aggregate benchmark runs.
    """
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"Preprocessed data not found: {preprocessed_path}")

    _, model, config = _load_model_and_config(model_path)
    adata = sc.read_h5ad(preprocessed_path)

    perturbation_key = config.get("perturbation_key", DEFAULT_PERTURBATION_KEY)
    control_value = config.get("control_value", DEFAULT_CONTROL_VALUE)
    validation_split = float(config.get("validation_split", 0.0))
    seed = int(config.get("seed", 42))
    if perturbation_gene_map is None:
        perturbation_gene_map = config.get("perturbation_gene_map")

    _, validation_adata, split_metadata = split_anndata_by_perturbation(
        adata,
        validation_split=validation_split,
        perturbation_key=perturbation_key,
        control_value=control_value,
        seed=seed,
    )
    if validation_adata is None:
        raise ValueError("No held-out perturbation split is available for this model configuration.")

    adj_matrix = _build_adjacency(
        validation_adata,
        network_path=network_path,
        gene_mapping_path=gene_mapping_path,
    )
    x_gene, x_adj = build_graph_inputs(validation_adata, adj_matrix=adj_matrix)
    policy_target, value_target, _ = build_signed_effect_targets(
        validation_adata,
        supervised_perturbations=split_metadata["validation_perturbations"],
        perturbation_key=perturbation_key,
        control_value=control_value,
        perturbation_gene_map=perturbation_gene_map,
    )

    preds = model([x_gene, x_adj], training=False)
    if not isinstance(preds, (list, tuple)) or len(preds) != 2:
        raise ValueError("Expected model to output a (policy, value) tuple.")

    policy_pred = np.asarray(preds[0], dtype="float32")
    value_pred = np.asarray(preds[1], dtype="float32")

    policy_metrics = compute_masked_policy_metrics(policy_target, policy_pred)
    value_mse = float(np.mean((value_pred - value_target) ** 2))
    value_mae = float(np.mean(np.abs(value_pred - value_target)))

    metrics = {
        "validation_perturbations": split_metadata["validation_perturbations"],
        "policy_huber": policy_metrics["policy_huber"],
        "policy_mae": policy_metrics["policy_mae"],
        "policy_sign_accuracy": policy_metrics["policy_sign_accuracy"],
        "value_mse": value_mse,
        "value_mae": value_mae,
    }

    if print_output:
        print("Evaluation metrics (held-out perturbations):")
        print(f"  Validation perturbations:   {', '.join(metrics['validation_perturbations'])}")
        print(f"  Policy Huber:              {metrics['policy_huber']:.6f}")
        print(f"  Policy MAE:                {metrics['policy_mae']:.6f}")
        print(f"  Policy Sign Accuracy:      {metrics['policy_sign_accuracy']:.4f}")
        print(f"  Value MSE:                 {metrics['value_mse']:.6f}")
        print(f"  Value MAE:                 {metrics['value_mae']:.6f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained HyperPerturb model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model directory or final_model.keras file.",
    )
    parser.add_argument(
        "--preprocessed_path",
        type=str,
        required=True,
        help="Path to the preprocessed AnnData file used for training.",
    )
    parser.add_argument(
        "--network_path",
        type=str,
        required=True,
        help="STRING network path used to rebuild the training adjacency.",
    )
    parser.add_argument(
        "--gene_mapping_path",
        type=str,
        default=None,
        help="Optional STRING protein-to-gene mapping file used to align network IDs with training genes.",
    )
    parser.add_argument(
        "--perturbation_gene_map_path",
        type=str,
        default=None,
        help="Optional JSON/CSV/TSV mapping from perturbation labels to target genes for gene-space evaluation.",
    )

    args = parser.parse_args()
    perturbation_gene_map = load_perturbation_gene_map(args.perturbation_gene_map_path)
    evaluate(
        args.model_path,
        args.preprocessed_path,
        network_path=args.network_path,
        gene_mapping_path=args.gene_mapping_path,
        perturbation_gene_map=perturbation_gene_map,
    )


if __name__ == "__main__":
    main()
