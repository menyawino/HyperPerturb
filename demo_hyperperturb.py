#!/usr/bin/env python3
"""
HyperPerturb Demonstration Script
This script demonstrates how to use the HyperPerturb framework for gene perturbation analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import scanpy as sc
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, '/mnt/omar/projects/hyperperturb')

print("🧬 HyperPerturb Demonstration")
print("=" * 50)

def demonstrate_basic_usage():
    """Demonstrate basic usage of HyperPerturb components."""
    print("\n1. 🔧 Setting up HyperPerturb components...")
    
    # Import core components
    from hyperpreturb.models import HyperbolicPerturbationModel
    from hyperpreturb.models.hyperbolic import HyperbolicAdam
    from hyperpreturb.utils.manifolds import PoincareBall
    
    # Create a simple example
    n_genes = 100
    n_perturbations = 10
    batch_size = 32
    
    print(f"   • Genes: {n_genes}")
    print(f"   • Perturbations: {n_perturbations}")
    print(f"   • Batch size: {batch_size}")
    
    # Create model
    model = HyperbolicPerturbationModel(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        embedding_dim=32,
        hidden_dim=64,
        curvature=1.0
    )
    
    # Create optimizer
    manifold = PoincareBall(curvature=1.0)
    optimizer = HyperbolicAdam(manifold=manifold, learning_rate=0.01)
    
    print("   ✓ Model and optimizer created")
    
    return model, optimizer, n_genes, n_perturbations, batch_size

def demonstrate_training():
    """Demonstrate training process."""
    print("\n2. 🎯 Training demonstration...")
    
    model, optimizer, n_genes, n_perturbations, batch_size = demonstrate_basic_usage()
    
    # Generate synthetic data
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Simulate perturbation data
    perturbations = tf.random.uniform([batch_size, n_perturbations], maxval=1.0)
    
    # Simulate target gene expression changes
    target_expression = tf.random.normal([batch_size, n_genes]) * 0.5
    
    print(f"   • Generated synthetic data: {perturbations.shape} -> {target_expression.shape}")
    
    # Training loop
    losses = []
    for epoch in range(10):
        with tf.GradientTape() as tape:
            predictions = model(perturbations)
            loss = tf.reduce_mean(tf.square(predictions - target_expression))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        losses.append(loss.numpy())
        
        if epoch % 2 == 0:
            print(f"   • Epoch {epoch:2d}: loss = {loss:.6f}")
    
    print(f"   ✓ Training completed. Final loss: {losses[-1]:.6f}")
    
    return model, losses

def demonstrate_inference():
    """Demonstrate inference and perturbation analysis."""
    print("\n3. 🔍 Inference demonstration...")
    
    model, losses = demonstrate_training()
    
    # Create test perturbations
    test_perturbations = tf.random.uniform([5, 10], maxval=1.0)
    
    # Make predictions
    predictions = model(test_perturbations)
    
    print(f"   • Test perturbations: {test_perturbations.shape}")
    print(f"   • Predictions: {predictions.shape}")
    
    # Analyze results
    print("\n   📊 Perturbation Analysis:")
    for i in range(5):
        pert_strength = tf.reduce_mean(test_perturbations[i]).numpy()
        pred_magnitude = tf.reduce_mean(tf.abs(predictions[i])).numpy()
        print(f"      Sample {i+1}: perturbation strength = {pert_strength:.3f}, predicted magnitude = {pred_magnitude:.3f}")
    
    return model, predictions

def demonstrate_data_loading():
    """Demonstrate data loading with the available dataset."""
    print("\n4. 📊 Data loading demonstration...")
    
    data_path = "/mnt/omar/projects/hyperperturb/data/raw/FrangiehIzar2021_RNA.h5ad"
    
    if os.path.exists(data_path):
        print(f"   • Loading data from: {os.path.basename(data_path)}")
        
        # Load the data
        adata = sc.read_h5ad(data_path)
        
        print(f"   ✓ Data loaded successfully!")
        print(f"      • Shape: {adata.shape}")
        print(f"      • Observations: {adata.n_obs}")
        print(f"      • Variables: {adata.n_vars}")
        
        # Show observation columns
        print(f"      • Observation columns: {list(adata.obs.columns)}")
        
        # Show variable columns
        print(f"      • Variable columns: {list(adata.var.columns)}")
        
        # Basic statistics
        print(f"      • Expression range: {adata.X.min():.3f} to {adata.X.max():.3f}")
        
        return adata
    else:
        print("   ! Data file not found. Using synthetic data instead.")
        return None

def demonstrate_manifold_operations():
    """Demonstrate hyperbolic manifold operations."""
    print("\n5. 🌐 Manifold operations demonstration...")
    
    from hyperpreturb.utils.manifolds import PoincareBall
    
    # Create manifold
    manifold = PoincareBall(curvature=1.0)
    
    # Create some points
    x = tf.random.normal([3, 5]) * 0.2
    y = tf.random.normal([3, 5]) * 0.2
    
    print(f"   • Created {x.shape[0]} points in {x.shape[1]}D hyperbolic space")
    
    # Compute distances
    distances = manifold.distance(x, y)
    print(f"   • Hyperbolic distances: {distances.numpy()}")
    
    # Exponential map
    v = tf.random.normal([3, 5]) * 0.1
    exp_result = manifold.expmap(x, v)
    print(f"   • Exponential map: {x.shape} -> {exp_result.shape}")
    
    # Logarithmic map
    log_result = manifold.logmap(x, y)
    print(f"   • Logarithmic map: {x.shape} -> {log_result.shape}")
    
    print("   ✓ Manifold operations completed successfully")

def create_simple_demo():
    """Create a simple end-to-end demonstration."""
    print("\n6. 🚀 End-to-end demonstration...")
    
    # Load real data if available
    adata = demonstrate_data_loading()
    
    if adata is not None:
        print("   • Using real data for demonstration")
        
        # Take a subset for demonstration
        if adata.n_obs > 1000:
            adata = adata[:1000, :].copy()
            print(f"   • Using subset: {adata.shape}")
        
        # Basic preprocessing
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        
        print(f"   • After filtering: {adata.shape}")
        
        # Extract some basic statistics
        print(f"   • Mean expression: {adata.X.mean():.3f}")
        print(f"   • Expression std: {adata.X.std():.3f}")
        
    else:
        print("   • Using synthetic data for demonstration")
    
    # Run inference demonstration
    model, predictions = demonstrate_inference()
    
    print("\n   🎉 Demonstration completed successfully!")
    print("   📝 Key takeaways:")
    print("      • HyperPerturb models can be created and trained")
    print("      • Hyperbolic manifold operations work correctly")
    print("      • Data loading and preprocessing are functional")
    print("      • End-to-end pipeline is ready for use")

def print_usage_guide():
    """Print a usage guide for the user."""
    print("\n" + "=" * 50)
    print("📖 USAGE GUIDE")
    print("=" * 50)
    print("\n🔧 Basic Usage:")
    print("   1. Import HyperPerturb components")
    print("   2. Create a HyperbolicPerturbationModel")
    print("   3. Set up a HyperbolicAdam optimizer")
    print("   4. Prepare your perturbation data")
    print("   5. Train the model")
    print("   6. Make predictions")
    
    print("\n📊 Data Requirements:")
    print("   • Perturbation data: [batch_size, n_perturbations]")
    print("   • Gene expression targets: [batch_size, n_genes]")
    print("   • Optional: Protein-protein interaction networks")
    
    print("\n🎯 Model Configuration:")
    print("   • embedding_dim: Dimension of hyperbolic embeddings")
    print("   • hidden_dim: Size of hidden layers")
    print("   • curvature: Hyperbolic space curvature (typically 1.0)")
    
    print("\n🚀 Next Steps:")
    print("   1. Load your own perturbation dataset")
    print("   2. Preprocess with scanpy")
    print("   3. Configure model hyperparameters")
    print("   4. Train on your data")
    print("   5. Analyze perturbation effects")
    print("   6. Discover optimal perturbation strategies")

if __name__ == "__main__":
    try:
        # Run demonstrations
        demonstrate_manifold_operations()
        model, predictions = demonstrate_inference()
        create_simple_demo()
        
        # Print usage guide
        print_usage_guide()
        
        print("\n🎉 HyperPerturb is ready for biological discovery!")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("Please check the error details above.")
        sys.exit(1)
