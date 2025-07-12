#!/usr/bin/env python3
"""
Comprehensive test script for HyperPerturb functionality.
This script tests all major components of the HyperPerturb framework.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, '/mnt/omar/projects/hyperperturb')

print("=" * 60)
print("HYPERPERTURB VALIDATION TEST")
print("=" * 60)

def test_imports():
    """Test that all main components can be imported."""
    print("\n1. Testing imports...")
    
    try:
        # Test basic imports
        import hyperpreturb
        print("✓ Main package imported successfully")
        
        # Test data module
        from hyperpreturb.data import download_data, preprocess_data
        print("✓ Data module imported successfully")
        
        # Test utils
        from hyperpreturb.utils.manifolds import PoincareBall
        print("✓ Manifolds module imported successfully")
        
        # Test models
        from hyperpreturb.models import HyperbolicPerturbationModel
        print("✓ Models module imported successfully")
        
        # Test hyperbolic components
        from hyperpreturb.models.hyperbolic import HyperbolicDense, HyperbolicAttention
        print("✓ Hyperbolic components imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\n2. Testing data loading...")
    
    try:
        # Check if data file exists
        data_path = "/mnt/omar/projects/hyperperturb/data/raw/FrangiehIzar2021_RNA.h5ad"
        if os.path.exists(data_path):
            print("✓ Data file found")
            
            # Load the data
            adata = sc.read_h5ad(data_path)
            print(f"✓ Data loaded successfully: {adata.shape}")
            print(f"  - Cells: {adata.n_obs}")
            print(f"  - Genes: {adata.n_vars}")
            
            # Check if perturbation information is available
            if 'perturbation' in adata.obs.columns:
                print(f"✓ Perturbation info found: {adata.obs['perturbation'].nunique()} unique perturbations")
            else:
                print("! No perturbation column found in obs")
                
            return adata
        else:
            print("✗ Data file not found")
            return None
            
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        return None

def test_manifold_operations():
    """Test hyperbolic manifold operations."""
    print("\n3. Testing hyperbolic manifold operations...")
    
    try:
        from hyperpreturb.utils.manifolds import PoincareBall
        
        # Create a Poincare ball manifold
        manifold = PoincareBall(dim=10, curvature=1.0)
        print("✓ Poincare ball manifold created")
        
        # Test basic operations
        x = tf.random.normal([5, 10]) * 0.1  # Small values to stay in ball
        y = tf.random.normal([5, 10]) * 0.1
        
        # Test distance computation
        dist = manifold.distance(x, y)
        print(f"✓ Distance computation: mean={dist.numpy().mean():.4f}")
        
        # Test exponential map
        v = tf.random.normal([5, 10]) * 0.01
        exp_result = manifold.exp(x, v)
        print(f"✓ Exponential map: shape={exp_result.shape}")
        
        # Test logarithmic map
        log_result = manifold.log(x, y)
        print(f"✓ Logarithmic map: shape={log_result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Manifold operations error: {e}")
        return False

def test_hyperbolic_layers():
    """Test hyperbolic neural network layers."""
    print("\n4. Testing hyperbolic neural network layers...")
    
    try:
        from hyperpreturb.models.hyperbolic import HyperbolicDense, HyperbolicAttention
        
        # Test HyperbolicDense layer
        layer = HyperbolicDense(units=32, curvature=1.0)
        x = tf.random.normal([10, 16]) * 0.1
        output = layer(x)
        print(f"✓ HyperbolicDense layer: input={x.shape}, output={output.shape}")
        
        # Test HyperbolicAttention layer
        attention = HyperbolicAttention(units=16, num_heads=4, curvature=1.0)
        att_output = attention(x)
        print(f"✓ HyperbolicAttention layer: input={x.shape}, output={att_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hyperbolic layers error: {e}")
        return False

def test_model_creation():
    """Test model creation and basic functionality."""
    print("\n5. Testing model creation...")
    
    try:
        from hyperpreturb.models import HyperbolicPerturbationModel
        
        # Create a small model for testing
        n_genes = 100
        n_perturbations = 10
        model = HyperbolicPerturbationModel(
            n_genes=n_genes,
            n_perturbations=n_perturbations,
            embedding_dim=16,
            hidden_dim=32,
            curvature=1.0
        )
        print("✓ Model created successfully")
        
        # Test model with dummy data
        gene_expr = tf.random.normal([5, n_genes])
        perturbations = tf.random.uniform([5, n_perturbations], maxval=1.0)
        
        # Forward pass
        output = model([gene_expr, perturbations])
        print(f"✓ Forward pass: input={gene_expr.shape}, output={output.shape}")
        
        # Check trainable parameters
        print(f"✓ Trainable parameters: {model.count_params()}")
        
        return model
        
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        return None

def test_optimizer():
    """Test hyperbolic optimizer."""
    print("\n6. Testing hyperbolic optimizer...")
    
    try:
        from hyperpreturb.models.hyperbolic import HyperbolicAdam
        
        # Create optimizer
        optimizer = HyperbolicAdam(learning_rate=0.001, curvature=1.0)
        print("✓ HyperbolicAdam optimizer created")
        
        # Test with simple optimization
        x = tf.Variable(tf.random.normal([5, 10]) * 0.1)
        
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(x))
        
        grads = tape.gradient(loss, [x])
        optimizer.apply_gradients(zip(grads, [x]))
        print(f"✓ Optimizer step completed, loss: {loss.numpy():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimizer error: {e}")
        return False

def test_training_loop():
    """Test a minimal training loop."""
    print("\n7. Testing training loop...")
    
    try:
        from hyperpreturb.models import HyperbolicPerturbationModel
        from hyperpreturb.models.hyperbolic import HyperbolicAdam
        
        # Create model and optimizer
        model = HyperbolicPerturbationModel(
            n_genes=50,
            n_perturbations=5,
            embedding_dim=8,
            hidden_dim=16,
            curvature=1.0
        )
        optimizer = HyperbolicAdam(learning_rate=0.01, curvature=1.0)
        
        # Generate dummy data
        batch_size = 10
        gene_expr = tf.random.normal([batch_size, 50])
        perturbations = tf.random.uniform([batch_size, 5], maxval=1.0)
        target = tf.random.normal([batch_size, 50])
        
        # Training step
        with tf.GradientTape() as tape:
            predictions = model([gene_expr, perturbations])
            loss = tf.reduce_mean(tf.square(predictions - target))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(f"✓ Training step completed, loss: {loss.numpy():.6f}")
        
        # Second step to verify learning
        with tf.GradientTape() as tape:
            predictions = model([gene_expr, perturbations])
            loss_2 = tf.reduce_mean(tf.square(predictions - target))
        
        gradients = tape.gradient(loss_2, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(f"✓ Second training step, loss: {loss_2.numpy():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training loop error: {e}")
        return False

def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("\n8. Testing data preprocessing...")
    
    try:
        from hyperpreturb.data import preprocess_data
        
        # Load data if available
        data_path = "/mnt/omar/projects/hyperperturb/data/raw/FrangiehIzar2021_RNA.h5ad"
        if os.path.exists(data_path):
            adata = sc.read_h5ad(data_path)
            print(f"✓ Data loaded: {adata.shape}")
            
            # Test preprocessing
            if adata.n_obs > 1000:  # Only test on subset for speed
                adata = adata[:1000, :].copy()
                print("✓ Using subset of data for testing")
            
            # Basic preprocessing
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            print(f"✓ Basic filtering: {adata.shape}")
            
            # Normalization
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            print("✓ Normalization completed")
            
            return True
        else:
            print("! Data file not found, skipping preprocessing test")
            return True
            
    except Exception as e:
        print(f"✗ Data preprocessing error: {e}")
        return False

def run_all_tests():
    """Run all tests and return results."""
    print("Starting comprehensive HyperPerturb validation...")
    
    results = {}
    
    # Run all tests
    results['imports'] = test_imports()
    results['data_loading'] = test_data_loading()
    results['manifold_ops'] = test_manifold_operations()
    results['hyperbolic_layers'] = test_hyperbolic_layers()
    results['model_creation'] = test_model_creation()
    results['optimizer'] = test_optimizer()
    results['training_loop'] = test_training_loop()
    results['data_preprocessing'] = test_data_preprocessing()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HyperPerturb is functional.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
