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
from hyperpreturb import (
    load_and_preprocess_perturbation_data, 
    models
)

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

def ensure_data_exists():
    """
    Check if the dataset files exist, and download them if they don't.
    """
    data_dir = "/mnt/omar/projects/hyperperturb/data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    # For this example, we'll use a smaller dataset for testing
    rna_path = os.path.join(data_dir, "subset_perturb_seq.h5ad")
    
    if not os.path.exists(rna_path):
        logger.info("Dataset not found. Downloading sample Perturb-seq dataset...")
        # URL to a sample Perturb-seq dataset
        # Note: This is a placeholder URL. Replace with the actual URL to your dataset
        url = "https://figshare.com/ndownloader/files/30306562"
        download_file(url, rna_path)
    
    return rna_path

def main():
    """
    Main function to train the HyperPerturb model.
    """
    parser = argparse.ArgumentParser(description="Train the HyperPerturb model")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to the input data. If not provided, will use the default dataset.")
    parser.add_argument("--output_dir", type=str, default="/mnt/omar/projects/hyperperturb/models/saved",
                        help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--embedding_dim", type=int, default=32, help="Dimension of the embedding")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Dimension of hidden layers")
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get data path
    data_path = args.data_path if args.data_path else ensure_data_exists()
    
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    adata, adj_matrix = load_and_preprocess_perturbation_data(data_path)
    
    # Setup training data
    n_genes = adata.shape[1]
    n_perturbations = adata.obsm['perturbation_target'].shape[1] if 'perturbation_target' in adata.obsm else 0
    
    if n_perturbations == 0:
        logger.error("No perturbation targets found in the data. Check data preprocessing.")
        return
    
    # Create and train the model
    logger.info("Setting up model...")
    model = models.HyperbolicPerturbationModel(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
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
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    history = model.fit(
        X, y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(args.output_dir, "model_checkpoint"),
                save_best_only=True
            )
        ]
    )
    
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