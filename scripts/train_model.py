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
from pathlib import Path
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
sys.path.append("/mnt/omar/projects/hyperperturb")

# Import the hyperperturb modules
from hyperpreturb.data import (
    load_and_preprocess_perturbation_data, 
    prepare_perturbation_data
)
from hyperpreturb import models

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
    parser.add_argument("--rna_path", type=str, default=None,
                        help="Path to the RNA expression data (h5ad)")
    parser.add_argument("--protein_path", type=str, default=None,
                        help="Path to the protein expression data (h5ad, optional)")
    parser.add_argument("--network_path", type=str, default=None,
                        help="Path to protein-protein interaction network (optional)")
    parser.add_argument("--output_dir", type=str, default="/mnt/omar/projects/hyperperturb/models/saved",
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of the embedding")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimension of hidden layers")
    parser.add_argument("--curvature", type=float, default=1.0, 
                        help="Curvature parameter for hyperbolic space (1.0 for unit hyperboloid)")
    parser.add_argument("--use_protein_data", action="store_true", 
                        help="Include protein expression data in the model if available")
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set default paths for the data files if not provided
    default_data_dir = Path("/mnt/omar/projects/hyperperturb/data/raw")
    
    if args.rna_path is None:
        args.rna_path = default_data_dir / "FrangiehIzar2021_RNA.h5ad"
        logger.info(f"Using default RNA data path: {args.rna_path}")
    
    if args.protein_path is None and args.use_protein_data:
        default_protein_path = default_data_dir / "FrangiehIzar2021_protein.h5ad"
        if default_protein_path.exists():
            args.protein_path = default_protein_path
            logger.info(f"Using default protein data path: {args.protein_path}")
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    adata, adj_matrix = load_and_preprocess_perturbation_data(
        args.rna_path, 
        args.protein_path if args.use_protein_data else None,
        args.network_path
    )
    
    # Check if data was successfully loaded
    if adata is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Setup training data
    n_genes = adata.shape[1]
    
    # Check if perturbation targets were identified
    if 'perturbation_target' not in adata.obsm:
        logger.error("No perturbation targets found in the data. Check data preprocessing.")
        return
    
    n_perturbations = adata.obsm['perturbation_target'].shape[1]
    
    if n_perturbations == 0:
        logger.error("No perturbation targets found in the data. Check data preprocessing.")
        return
    
    # Check if log fold change data exists
    if 'log_fold_change' not in adata.obsm:
        logger.error("Log fold change data not found. Check data preprocessing.")
        return
    
    logger.info(f"Data loaded: {n_genes} genes, {n_perturbations} perturbation targets")
    
    # Create and train the model
    logger.info("Setting up hyperbolic perturbation model...")
    model = models.HyperbolicPerturbationModel(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        curvature=args.curvature,
        adj_matrix=adj_matrix
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
    if 'protein' in adata.layers and args.use_protein_data:
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
                filepath=os.path.join(args.output_dir, "model_checkpoint"),
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
    model_save_path = os.path.join(args.output_dir, "hyperbolic_model")
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

if __name__ == "__main__":
    main()