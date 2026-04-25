#!/usr/bin/env python3

"""Benchmark HyperPerturb hyperbolic vs Euclidean ablation on held-out perturbations.

This runner trains both variants with identical settings, evaluates each on
the held-out perturbation split, and writes a JSON summary.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is importable when running as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hyperpreturb.data import load_and_preprocess_perturbation_data
from hyperpreturb.models.train import train_model
from evaluate_model import evaluate


def _latest_run_dir(root: Path) -> Path:
    run_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("hyperperturb-")]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {root}")
    return sorted(run_dirs)[-1]


def _run_variant(name, adata, adj_matrix, output_root, args, euclidean_baseline):
    variant_root = output_root / name
    variant_root.mkdir(parents=True, exist_ok=True)

    train_model(
        adata=adata,
        adj_matrix=adj_matrix,
        model_dir=str(variant_root),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        curvature=args.curvature,
        validation_split=args.val_split,
        debug=args.debug,
        policy_only=False,
        euclidean_baseline=euclidean_baseline,
        seed=args.seed,
        deterministic=True,
        perturbation_key=args.perturbation_key,
        control_value=args.control_value,
    )

    run_dir = _latest_run_dir(variant_root)
    metrics = evaluate(
        model_path=str(run_dir),
        preprocessed_path=args.preprocessed_path,
        network_path=args.network_path,
        gene_mapping_path=args.gene_mapping_path,
        print_output=False,
    )

    return {
        "variant": name,
        "run_dir": str(run_dir),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark hyperbolic and Euclidean HyperPerturb variants on held-out perturbations."
    )
    parser.add_argument("--rna_path", type=str, required=True, help="Path to RNA .h5ad data")
    parser.add_argument("--network_path", type=str, required=True, help="Path to STRING network file")
    parser.add_argument("--gene_mapping_path", type=str, default=None, help="Optional STRING protein-to-gene mapping file")
    parser.add_argument(
        "--preprocessed_path",
        type=str,
        required=True,
        help="Path to preprocessed .h5ad (created if missing and reused for evaluation)",
    )
    parser.add_argument("--output_dir", type=str, default="models/benchmarks", help="Benchmark output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per variant")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--curvature", type=float, default=1.0, help="Hyperbolic curvature")
    parser.add_argument("--val_split", type=float, default=0.2, help="Held-out perturbation validation fraction")
    parser.add_argument("--max_cells", type=int, default=3000, help="Max cells in preprocessing")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--debug", action="store_true", help="Use debug training mode")
    parser.add_argument("--perturbation_key", type=str, default="perturbation", help="Perturbation column in adata.obs")
    parser.add_argument("--control_value", type=str, default="non-targeting", help="Control value in perturbation column")
    parser.add_argument(
        "--skip_hyperbolic",
        action="store_true",
        help="Skip the hyperbolic variant and run only Euclidean baseline",
    )
    parser.add_argument(
        "--skip_euclidean",
        action="store_true",
        help="Skip the Euclidean variant and run only hyperbolic variant",
    )
    args = parser.parse_args()

    if args.skip_hyperbolic and args.skip_euclidean:
        raise ValueError("At least one variant must be enabled.")

    if not os.path.exists(args.rna_path):
        raise FileNotFoundError(f"RNA data file not found: {args.rna_path}")
    if not os.path.exists(args.network_path):
        raise FileNotFoundError(f"Network file not found: {args.network_path}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    adata, adj_matrix = load_and_preprocess_perturbation_data(
        rna_path=args.rna_path,
        protein_path=None,
        network_path=args.network_path,
        gene_mapping_path=args.gene_mapping_path,
        preprocessed_path=args.preprocessed_path,
        max_cells=args.max_cells,
        ctrl_key=args.perturbation_key,
        ctrl_value=args.control_value,
    )

    if adj_matrix is None:
        raise ValueError("Benchmark requires adjacency; ensure --network_path is valid.")

    results = {
        "config": {
            "rna_path": args.rna_path,
            "network_path": args.network_path,
            "gene_mapping_path": args.gene_mapping_path,
            "preprocessed_path": args.preprocessed_path,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "curvature": args.curvature,
            "val_split": args.val_split,
            "max_cells": args.max_cells,
            "seed": args.seed,
            "debug": args.debug,
            "perturbation_key": args.perturbation_key,
            "control_value": args.control_value,
        },
        "variants": [],
    }

    if not args.skip_hyperbolic:
        results["variants"].append(
            _run_variant(
                name="hyperbolic",
                adata=adata,
                adj_matrix=adj_matrix,
                output_root=output_root,
                args=args,
                euclidean_baseline=False,
            )
        )

    if not args.skip_euclidean:
        results["variants"].append(
            _run_variant(
                name="euclidean",
                adata=adata,
                adj_matrix=adj_matrix,
                output_root=output_root,
                args=args,
                euclidean_baseline=True,
            )
        )

    metrics_path = output_root / "benchmark_results.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print("Benchmark completed.")
    print(f"Results written to: {metrics_path}")
    for variant in results["variants"]:
        name = variant["variant"]
        metrics = variant["metrics"]
        print(
            f"[{name}] policy_huber={metrics['policy_huber']:.6f}, "
            f"policy_sign_accuracy={metrics['policy_sign_accuracy']:.4f}, "
            f"value_mse={metrics['value_mse']:.6f}"
        )


if __name__ == "__main__":
    main()