#!/usr/bin/env python3
"""
HyperPerturb model training script.
"""

import os
import argparse
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import shutil
import requests
import gzip

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project directory to the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import the hyperperturb modules
from hyperpreturb.data import (
    load_and_preprocess_perturbation_data,
    prepare_perturbation_data,
)
from hyperpreturb import models
from hyperpreturb.models.train import train_model as advanced_train_model
from hyperpreturb.models.training_utils import load_perturbation_gene_map

def download_file(url, output_path):
    """
    Download a file from a URL to a specified path.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download the file
    logger.info(f"Downloading from {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    logger.info(f"Downloaded to {output_path}")
    
    # Uncompress if it's a gzip file
    if output_path.endswith('.gz'):
        uncompressed_path = output_path[:-3]
        logger.info(f"Uncompressing to {uncompressed_path}")
        
        with gzip.open(output_path, 'rb') as f_in:
            with open(uncompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        return uncompressed_path
    
    return output_path

def main():
    """
    Main function to train the HyperPerturb model.
    """
    parser = argparse.ArgumentParser(description="Train the HyperPerturb model")
    parser.add_argument("--rna_path", type=str, required=True,
                        help="Path to the RNA expression data (h5ad)")
    parser.add_argument("--protein_path", type=str, default=None,
                        help="Path to the protein expression data (h5ad, optional)")
    parser.add_argument("--network_path", type=str, default=None,
                        help="Path to protein-protein interaction network (optional)")
    parser.add_argument("--gene_mapping_path", type=str, default=None,
                        help="Optional STRING protein-to-gene mapping file used to align network IDs with adata.var_names.")
    parser.add_argument("--output_dir", type=str, default="models/saved",
                        help="Directory to save the trained model")
    # Defaults tuned for ~10GB system RAM
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (10GB-safe default)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--embedding_dim", type=int, default=16, help="Dimension of the embedding")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Dimension of hidden layers")
    parser.add_argument("--curvature", type=float, default=1.0, 
                        help="Curvature parameter for hyperbolic space (1.0 for unit hyperboloid)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed for deterministic training setup")
    parser.add_argument("--use_protein_data", action="store_true", 
                        help="Include protein expression data in the model if available")
    parser.add_argument("--trainer", type=str, default="advanced",
                    choices=["simple", "advanced"],
                    help="Training pipeline: 'simple' Keras regression or 'advanced' HyperPerturb trainer.")
    parser.add_argument("--max_cells", type=int, default=3000,
                    help="Maximum number of cells to keep during preprocessing.")
    parser.add_argument("--val_split", type=float, default=0.2,
                    help="Validation split fraction for advanced trainer.")
    parser.add_argument("--preprocessed_path", type=str, default=None,
                    help="Path to preprocessed h5ad to skip Scanpy preprocessing.")
    parser.add_argument("--debug", action="store_true",
                    help="Enable debug mode for the advanced trainer (constant LR, NaN checks).")
    parser.add_argument("--policy_only", action="store_true",
                    help="Debug: train only the signed policy head.")
    parser.add_argument("--no_protein", action="store_true",
                    help="Debug: drop protein modality from adata.obsm if present.")
    parser.add_argument("--euclidean_baseline", action="store_true",
                    help="Debug: use Euclidean baseline model instead of hyperbolic graph convs.")
    parser.add_argument("--perturbation_key", type=str, default="perturbation",
                    help="adata.obs column containing perturbation labels.")
    parser.add_argument("--control_value", type=str, default="non-targeting",
                    help="Label in --perturbation_key that denotes control cells.")
    parser.add_argument("--perturbation_gene_map_path", type=str, default=None,
                    help="Optional JSON/CSV/TSV mapping from perturbation labels to target genes for gene-space supervision.")
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.rna_path):
        raise FileNotFoundError(f"RNA file not found: {args.rna_path}")

    if args.use_protein_data and args.protein_path is None:
        raise ValueError("--use_protein_data requires --protein_path to be set explicitly.")

    if args.protein_path is not None and not os.path.exists(args.protein_path):
        raise FileNotFoundError(f"Protein file not found: {args.protein_path}")

    if args.network_path is not None and not os.path.exists(args.network_path):
        raise FileNotFoundError(f"Network file not found: {args.network_path}")

    if args.perturbation_gene_map_path is not None and not os.path.exists(args.perturbation_gene_map_path):
        raise FileNotFoundError(f"Perturbation mapping file not found: {args.perturbation_gene_map_path}")

    if args.trainer == "advanced" and args.network_path is None:
        raise ValueError("Advanced trainer requires --network_path with a valid gene graph file.")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    perturbation_gene_map = load_perturbation_gene_map(args.perturbation_gene_map_path)
    adata, adj_matrix = load_and_preprocess_perturbation_data(
        args.rna_path,
        args.protein_path if args.use_protein_data else None,
        args.network_path,
        gene_mapping_path=args.gene_mapping_path,
        preprocessed_path=args.preprocessed_path,
        max_cells=args.max_cells,
        ctrl_key=args.perturbation_key,
        ctrl_value=args.control_value,
    )
    
    # Check if data was successfully loaded
    if adata is None:
        raise RuntimeError("Data loader returned None unexpectedly.")

    # Optional: debug ablation to drop protein modality entirely
    if args.no_protein and 'protein' in adata.obsm:
        logger.info("Debug: dropping protein modality from adata.obsm['protein']")
        del adata.obsm['protein']
    
    # Setup training data
    n_genes = adata.shape[1]
    
    # Check if perturbation targets were identified
    if 'perturbation_target' not in adata.obsm:
        raise ValueError("No perturbation targets found in adata.obsm['perturbation_target'].")
    
    n_perturbations = adata.obsm['perturbation_target'].shape[1]
    
    if n_perturbations == 0:
        raise ValueError("No perturbation targets found in the data.")
    
    # Check if log fold change data exists
    if 'log_fold_change' not in adata.obsm:
        raise ValueError("Log fold change data not found in adata.obsm['log_fold_change'].")
    
    logger.info(f"Data loaded: {n_genes} genes, {n_perturbations} perturbation targets")

    if args.trainer == "simple":
        logger.info("Using simple Keras regression trainer")
        # Create and train the model
        logger.info("Setting up hyperbolic perturbation model...")
        model = models.HyperbolicPerturbationModel(
            n_genes=n_genes,
            n_perturbations=n_perturbations,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            curvature=args.curvature,
        )

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        # Prepare training data
        X = adata.obsm['perturbation_target']
        y = adata.obsm['log_fold_change']

        # Add protein data if available (as additional features)
        if 'protein' in adata.obsm and args.use_protein_data:
            logger.info("Including protein expression data in model training")
            # Process protein data based on your model architecture
            # For example, you might want to include it as a separate input to the model

        # Split data into training and validation sets
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info(f"Starting training for {args.epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(args.output_dir, "model_checkpoint.keras"),
                    save_best_only=True
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(args.output_dir, "logs"),
                    histogram_freq=1
                )
            ]
        )

        # Evaluate model on validation data
        val_results = model.evaluate(X_val, y_val, verbose=1)
        logger.info(f"Validation loss: {val_results[0]}, Validation MAE: {val_results[1]}")

        # Save the trained model
        model_save_path = os.path.join(args.output_dir, "hyperbolic_model.keras")
        model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Mean Absolute Error')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')

        plot_save_path = os.path.join(args.output_dir, "training_history.png")
        plt.tight_layout()
        plt.savefig(plot_save_path)
        logger.info(f"Training plot saved to {plot_save_path}")
    else:
        logger.info("Using advanced HyperPerturb trainer from hyperpreturb.models.train")
        model, history = advanced_train_model(
            adata=adata,
            adj_matrix=adj_matrix,
            model_dir=str(args.output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            curvature=args.curvature,
            validation_split=args.val_split,
            debug=args.debug,
            policy_only=args.policy_only,
            euclidean_baseline=args.euclidean_baseline,
            seed=args.seed,
            deterministic=True,
            perturbation_key=args.perturbation_key,
            control_value=args.control_value,
            perturbation_gene_map=perturbation_gene_map,
        )
        logger.info("Advanced training completed")

if __name__ == "__main__":
    main()