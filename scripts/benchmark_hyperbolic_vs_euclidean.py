#!/usr/bin/env python3

"""Benchmark HyperPerturb hyperbolic vs Euclidean ablation on held-out perturbations.

This runner trains both variants across one or more seeds, evaluates each on
the held-out perturbation split, and writes JSON plus CSV/Markdown tables.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

# Ensure project root is importable when running as a script.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hyperpreturb.data import load_and_preprocess_perturbation_data
from hyperpreturb.models.train import train_model
from hyperpreturb.models.training_utils import load_perturbation_gene_map
from evaluate_model import evaluate


def _latest_run_dir(root: Path) -> Path:
    run_dirs = [path for path in root.iterdir() if path.is_dir() and path.name.startswith("hyperperturb-")]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under {root}")
    return sorted(run_dirs)[-1]


METRIC_COLUMNS = ["policy_huber", "policy_mae", "policy_sign_accuracy", "value_mse", "value_mae"]


def _run_variant(name, adata, adj_matrix, output_root, args, euclidean_baseline, seed, perturbation_gene_map):
    variant_root = output_root / name / f"seed-{seed}"
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
        seed=seed,
        deterministic=True,
        perturbation_key=args.perturbation_key,
        control_value=args.control_value,
        perturbation_gene_map=perturbation_gene_map,
    )

    run_dir = _latest_run_dir(variant_root)
    metrics = evaluate(
        model_path=str(run_dir),
        preprocessed_path=args.preprocessed_path,
        network_path=args.network_path,
        gene_mapping_path=args.gene_mapping_path,
        perturbation_gene_map=perturbation_gene_map,
        print_output=False,
    )

    return {
        "variant": name,
        "seed": seed,
        "run_dir": str(run_dir),
        "metrics": metrics,
    }


def _mean(values):
    return sum(values) / len(values)


def _std(values):
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / (len(values) - 1))


def _aggregate_variant_runs(name, runs):
    aggregate = {"variant": name, "n_runs": len(runs)}
    for metric in METRIC_COLUMNS:
        values = [float(run["metrics"][metric]) for run in runs]
        aggregate[f"{metric}_mean"] = _mean(values)
        aggregate[f"{metric}_std"] = _std(values)
    validation_sets = sorted({tuple(run["metrics"]["validation_perturbations"]) for run in runs})
    aggregate["validation_perturbations"] = [list(values) for values in validation_sets]
    return aggregate


def _write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value):
    return f"{value:.6f}"


def _write_markdown_table(path, rows, headers):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(row[header] for header in headers) + " |\n")


def _build_table_rows(results):
    run_rows = []
    summary_rows = []
    for variant_result in results["variants"]:
        for run in variant_result["runs"]:
            row = {
                "variant": variant_result["variant"],
                "seed": str(run["seed"]),
                "validation_perturbations": ", ".join(run["metrics"]["validation_perturbations"]),
            }
            for metric in METRIC_COLUMNS:
                row[metric] = _format_float(float(run["metrics"][metric]))
            run_rows.append(row)

        aggregate = variant_result["aggregate"]
        summary_row = {
            "variant": variant_result["variant"],
            "n_runs": str(aggregate["n_runs"]),
            "validation_perturbations": "; ".join(", ".join(values) for values in aggregate["validation_perturbations"]),
        }
        for metric in METRIC_COLUMNS:
            summary_row[f"{metric}_mean"] = _format_float(float(aggregate[f"{metric}_mean"]))
            summary_row[f"{metric}_std"] = _format_float(float(aggregate[f"{metric}_std"]))
        summary_rows.append(summary_row)

    return run_rows, summary_rows


def _write_result_tables(output_root, results):
    run_rows, summary_rows = _build_table_rows(results)

    run_fieldnames = ["variant", "seed", "validation_perturbations", *METRIC_COLUMNS]
    summary_fieldnames = ["variant", "n_runs", "validation_perturbations"]
    for metric in METRIC_COLUMNS:
        summary_fieldnames.extend([f"{metric}_mean", f"{metric}_std"])

    _write_csv(output_root / "benchmark_runs.csv", run_rows, run_fieldnames)
    _write_csv(output_root / "benchmark_summary.csv", summary_rows, summary_fieldnames)
    _write_markdown_table(output_root / "benchmark_runs.md", run_rows, run_fieldnames)
    _write_markdown_table(output_root / "benchmark_summary.md", summary_rows, summary_fieldnames)


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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="One or more random seeds for repeated held-out benchmark runs")
    parser.add_argument("--debug", action="store_true", help="Use debug training mode")
    parser.add_argument("--perturbation_key", type=str, default="perturbation", help="Perturbation column in adata.obs")
    parser.add_argument("--control_value", type=str, default="non-targeting", help="Control value in perturbation column")
    parser.add_argument(
        "--perturbation_gene_map_path",
        type=str,
        default=None,
        help="Optional JSON/CSV/TSV mapping from perturbation labels to target genes for gene-space supervision.",
    )
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
    if args.perturbation_gene_map_path is not None and not os.path.exists(args.perturbation_gene_map_path):
        raise FileNotFoundError(f"Perturbation mapping file not found: {args.perturbation_gene_map_path}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    perturbation_gene_map = load_perturbation_gene_map(args.perturbation_gene_map_path)

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
            "seeds": args.seeds,
            "debug": args.debug,
            "perturbation_key": args.perturbation_key,
            "control_value": args.control_value,
            "perturbation_gene_map_path": args.perturbation_gene_map_path,
        },
        "variants": [],
    }

    variant_specs = []
    if not args.skip_hyperbolic:
        variant_specs.append(("hyperbolic", False))
    if not args.skip_euclidean:
        variant_specs.append(("euclidean", True))

    for variant_name, euclidean_baseline in variant_specs:
        runs = []
        for seed in args.seeds:
            runs.append(
                _run_variant(
                    name=variant_name,
                    adata=adata,
                    adj_matrix=adj_matrix,
                    output_root=output_root,
                    args=args,
                    euclidean_baseline=euclidean_baseline,
                    seed=seed,
                    perturbation_gene_map=perturbation_gene_map,
                )
            )
        results["variants"].append(
            {
                "variant": variant_name,
                "runs": runs,
                "aggregate": _aggregate_variant_runs(variant_name, runs),
            }
        )

    metrics_path = output_root / "benchmark_results.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    _write_result_tables(output_root, results)

    print("Benchmark completed.")
    print(f"Results written to: {metrics_path}")
    print(f"Summary table: {output_root / 'benchmark_summary.md'}")
    print(f"Per-run table: {output_root / 'benchmark_runs.md'}")
    for variant in results["variants"]:
        name = variant["variant"]
        metrics = variant["aggregate"]
        print(
            f"[{name}] policy_huber={metrics['policy_huber_mean']:.6f} +/- {metrics['policy_huber_std']:.6f}, "
            f"policy_sign_accuracy={metrics['policy_sign_accuracy_mean']:.4f} +/- {metrics['policy_sign_accuracy_std']:.4f}, "
            f"value_mse={metrics['value_mse_mean']:.6f} +/- {metrics['value_mse_std']:.6f}"
        )


if __name__ == "__main__":
    main()