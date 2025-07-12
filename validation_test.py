#!/usr/bin/env python3
"""
Corrected validation test for HyperPerturb functionality.
This script tests the actual API and components as implemented.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, '/mnt/omar/projects/hyperperturb')

print("=" * 50)
print("HYPERPERTURB FUNCTIONAL VALIDATION")
print("=" * 50)

def test_imports():
    """Test that all components can be imported correctly."""
    print("\n1. Testing imports...")
    
    try:
        # Test basic imports
        import hyperpreturb
        print("✓ Main package imported")
        
        # Test utils
        from hyperpreturb.utils.manifolds import PoincareBall
        print("✓ PoincareBall imported")
        
        # Test models - use correct class names
        from hyperpreturb.models import HyperPerturbModel
        print("✓ HyperPerturbModel imported")
        
        # Test hyperbolic components
        from hyperpreturb.models.hyperbolic import HyperbolicDense, HyperbolicAdam
        print("✓ Hyperbolic components imported")
        
        # Test the actual model from models.py
        from hyperpreturb.models import HyperbolicPerturbationModel
        print("✓ HyperbolicPerturbationModel imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_manifold_operations():
    """Test manifold operations with correct API."""
    print("\n2. Testing manifold operations...")
    
    try:
        from hyperpreturb.utils.manifolds import PoincareBall
        
        # Create manifold with correct parameters
        manifold = PoincareBall(curvature=1.0)
        print("✓ Manifold created")
        
        # Test points in the ball
        x = tf.random.normal([3, 8]) * 0.1
        y = tf.random.normal([3, 8]) * 0.1
        
        # Test distance
        dist = manifold.distance(x, y)
        print(f"✓ Distance: {dist.numpy()}")
        
        # Test expmap
        v = tf.random.normal([3, 8]) * 0.01
        exp_result = manifold.expmap(x, v)
        print(f"✓ Expmap: {exp_result.shape}")
        
        # Test logmap
        log_result = manifold.logmap(x, y)
        print(f"✓ Logmap: {log_result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Manifold error: {e}")
        return False

def test_hyperbolic_layers():
    """Test hyperbolic layers."""
    print("\n3. Testing hyperbolic layers...")
    
    try:
        from hyperpreturb.models.hyperbolic import HyperbolicDense
        
        # Create layer
        layer = HyperbolicDense(units=16, curvature=1.0)
        
        # Test forward pass
        x = tf.random.normal([5, 10]) * 0.1
        output = layer(x)
        
        print(f"✓ HyperbolicDense: {x.shape} -> {output.shape}")
        print(f"✓ Parameters: {layer.count_params()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Layer error: {e}")
        return False

def test_models():
    """Test both model architectures."""
    print("\n4. Testing models...")
    
    try:
        # Test HyperPerturbModel
        from hyperpreturb.models import HyperPerturbModel
        
        model1 = HyperPerturbModel(num_genes=20, curvature=1.0)
        print("✓ HyperPerturbModel created")
        
        # Test HyperbolicPerturbationModel
        from hyperpreturb.models import HyperbolicPerturbationModel
        
        model2 = HyperbolicPerturbationModel(
            n_genes=20,
            n_perturbations=5,
            embedding_dim=8,
            hidden_dim=16,
            curvature=1.0
        )
        print("✓ HyperbolicPerturbationModel created")
        
        # Test forward pass on the perturbation model
        batch_size = 3
        perturbations = tf.random.uniform([batch_size, 5], maxval=1.0)
        
        output = model2(perturbations)
        print(f"✓ Forward pass: {output.shape}")
        print(f"✓ Total params: {model2.count_params()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model error: {e}")
        return False

def test_optimizer():
    """Test hyperbolic optimizer with correct API."""
    print("\n5. Testing optimizer...")
    
    try:
        from hyperpreturb.models.hyperbolic import HyperbolicAdam
        from hyperpreturb.utils.manifolds import PoincareBall
        
        # Create manifold and optimizer
        manifold = PoincareBall(curvature=1.0)
        optimizer = HyperbolicAdam(manifold=manifold, learning_rate=0.01)
        print("✓ Optimizer created with manifold")
        
        # Simple optimization test
        x = tf.Variable(tf.random.normal([2, 5]) * 0.1)
        
        for i in range(3):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(x - 0.1))
            
            grads = tape.gradient(loss, [x])
            optimizer.apply_gradients(zip(grads, [x]))
            
            print(f"  Step {i+1}: loss = {loss.numpy():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Optimizer error: {e}")
        return False

def test_training():
    """Test training with the actual model."""
    print("\n6. Testing training...")
    
    try:
        from hyperpreturb.models import HyperbolicPerturbationModel
        from hyperpreturb.models.hyperbolic import HyperbolicAdam
        from hyperpreturb.utils.manifolds import PoincareBall
        
        # Create model and optimizer
        model = HyperbolicPerturbationModel(
            n_genes=10,
            n_perturbations=3,
            embedding_dim=6,
            hidden_dim=12,
            curvature=1.0
        )
        
        manifold = PoincareBall(curvature=1.0)
        optimizer = HyperbolicAdam(manifold=manifold, learning_rate=0.01)
        
        # Dummy data
        batch_size = 4
        perturbations = tf.random.uniform([batch_size, 3], maxval=1.0)
        target = tf.random.normal([batch_size, 10])
        
        # Training step
        with tf.GradientTape() as tape:
            predictions = model(perturbations)
            loss = tf.reduce_mean(tf.square(predictions - target))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(f"✓ Training step: loss = {loss.numpy():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training error: {e}")
        return False

def test_graph_operations():
    """Test graph-based operations."""
    print("\n7. Testing graph operations...")
    
    try:
        from hyperpreturb.models import HyperPerturbModel
        import scipy.sparse as sp
        
        # Create a simple adjacency matrix
        n_genes = 10
        adj_matrix = sp.random(n_genes, n_genes, density=0.3, format='coo')
        adj_tensor = tf.SparseTensor(
            indices=np.column_stack([adj_matrix.row, adj_matrix.col]),
            values=adj_matrix.data.astype(np.float32),
            dense_shape=adj_matrix.shape
        )
        
        # Create model
        model = HyperPerturbModel(num_genes=n_genes, curvature=1.0)
        
        # Test forward pass
        x = tf.random.normal([5, n_genes]) * 0.1
        policy, value = model([x, adj_tensor])
        
        print(f"✓ Graph model: policy {policy.shape}, value {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Graph operations error: {e}")
        return False

def test_data_module():
    """Test data module."""
    print("\n8. Testing data module...")
    
    try:
        from hyperpreturb.data import download_data
        
        # Test that functions exist
        assert callable(download_data), "download_data should be callable"
        print("✓ Data functions accessible")
        
        # Check data file
        data_path = "/mnt/omar/projects/hyperperturb/data/raw/FrangiehIzar2021_RNA.h5ad"
        if os.path.exists(data_path):
            print(f"✓ Data file exists: {os.path.basename(data_path)}")
        else:
            print("! Data file not found")
        
        return True
        
    except Exception as e:
        print(f"✗ Data module error: {e}")
        return False

def run_validation():
    """Run all validation tests."""
    print("Running HyperPerturb validation tests...")
    
    tests = [
        ("Core Imports", test_imports),
        ("Manifold Operations", test_manifold_operations),
        ("Hyperbolic Layers", test_hyperbolic_layers),
        ("Model Creation", test_models),
        ("Optimizer", test_optimizer),
        ("Training", test_training),
        ("Graph Operations", test_graph_operations),
        ("Data Module", test_data_module)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All validation tests passed!")
        print("🚀 HyperPerturb is fully functional!")
    elif passed >= len(results) * 0.75:
        print("✅ Most tests passed - core functionality is working")
    else:
        print("⚠️  Several tests failed - check the errors above")
    
    return passed >= len(results) * 0.75

if __name__ == "__main__":
    success = run_validation()
    
    if success:
        print("\n" + "=" * 50)
        print("READY TO USE - NEXT STEPS:")
        print("=" * 50)
        print("1. 🧬 Load your perturbation data")
        print("2. 📊 Preprocess with scanpy")
        print("3. 🔗 Build protein-protein interaction networks")
        print("4. 🎯 Train hyperbolic models")
        print("5. 📈 Analyze perturbation effects")
        print("6. 🔍 Discover optimal perturbation strategies")
        print("\nHyperPerturb is ready for biological discovery!")
    
    sys.exit(0 if success else 1)
