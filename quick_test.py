#!/usr/bin/env python3
"""
Quick validation test for HyperPerturb core functionality.
This script tests the essential components without loading large datasets.
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
print("HYPERPERTURB QUICK VALIDATION")
print("=" * 50)

def test_core_imports():
    """Test that core components can be imported."""
    print("\n1. Testing core imports...")
    
    try:
        # Test basic imports
        import hyperpreturb
        print("✓ Main package imported")
        
        # Test utils
        from hyperpreturb.utils.manifolds import PoincareBall
        print("✓ PoincareBall imported")
        
        # Test models
        from hyperpreturb.models import HyperbolicPerturbationModel
        print("✓ HyperbolicPerturbationModel imported")
        
        # Test hyperbolic components
        from hyperpreturb.models.hyperbolic import HyperbolicDense, HyperbolicAdam
        print("✓ Hyperbolic components imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False

def test_manifold_basics():
    """Test basic manifold operations."""
    print("\n2. Testing manifold operations...")
    
    try:
        from hyperpreturb.utils.manifolds import PoincareBall
        
        # Create manifold
        manifold = PoincareBall(dim=8, curvature=1.0)
        print("✓ Manifold created")
        
        # Create some points in the ball
        x = tf.random.normal([3, 8]) * 0.1
        y = tf.random.normal([3, 8]) * 0.1
        
        # Test distance
        dist = manifold.distance(x, y)
        print(f"✓ Distance computed: {dist.numpy()}")
        
        # Test exp map
        v = tf.random.normal([3, 8]) * 0.01
        exp_result = manifold.exp(x, v)
        print(f"✓ Exp map: {exp_result.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Manifold error: {e}")
        return False

def test_hyperbolic_layer():
    """Test hyperbolic layer functionality."""
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

def test_model_creation():
    """Test model creation and basic forward pass."""
    print("\n4. Testing model creation...")
    
    try:
        from hyperpreturb.models import HyperbolicPerturbationModel
        
        # Small model for testing
        model = HyperbolicPerturbationModel(
            n_genes=20,
            n_perturbations=5,
            embedding_dim=8,
            hidden_dim=16,
            curvature=1.0
        )
        
        print("✓ Model created")
        
        # Test forward pass
        batch_size = 3
        gene_expr = tf.random.normal([batch_size, 20])
        perturbations = tf.random.uniform([batch_size, 5], maxval=1.0)
        
        output = model([gene_expr, perturbations])
        
        print(f"✓ Forward pass: {output.shape}")
        print(f"✓ Total params: {model.count_params()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model error: {e}")
        return False

def test_optimizer():
    """Test hyperbolic optimizer."""
    print("\n5. Testing optimizer...")
    
    try:
        from hyperpreturb.models.hyperbolic import HyperbolicAdam
        
        # Create optimizer
        optimizer = HyperbolicAdam(learning_rate=0.01, curvature=1.0)
        print("✓ Optimizer created")
        
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

def test_training_step():
    """Test a single training step."""
    print("\n6. Testing training step...")
    
    try:
        from hyperpreturb.models import HyperbolicPerturbationModel
        from hyperpreturb.models.hyperbolic import HyperbolicAdam
        
        # Create model and optimizer
        model = HyperbolicPerturbationModel(
            n_genes=10,
            n_perturbations=3,
            embedding_dim=6,
            hidden_dim=12,
            curvature=1.0
        )
        optimizer = HyperbolicAdam(learning_rate=0.01, curvature=1.0)
        
        # Dummy data
        batch_size = 4
        gene_expr = tf.random.normal([batch_size, 10])
        perturbations = tf.random.uniform([batch_size, 3], maxval=1.0)
        target = tf.random.normal([batch_size, 10])
        
        # Training step
        with tf.GradientTape() as tape:
            predictions = model([gene_expr, perturbations])
            loss = tf.reduce_mean(tf.square(predictions - target))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(f"✓ Training step: loss = {loss.numpy():.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training error: {e}")
        return False

def test_data_module():
    """Test data module functions."""
    print("\n7. Testing data module...")
    
    try:
        from hyperpreturb.data import download_data
        
        # Test that the function exists and is callable
        assert callable(download_data), "download_data should be callable"
        print("✓ Data functions are accessible")
        
        # Check if data file exists
        data_path = "/mnt/omar/projects/hyperperturb/data/raw/FrangiehIzar2021_RNA.h5ad"
        if os.path.exists(data_path):
            print(f"✓ Data file exists: {os.path.basename(data_path)}")
        else:
            print("! Data file not found (optional for core functionality)")
        
        return True
        
    except Exception as e:
        print(f"✗ Data module error: {e}")
        return False

def run_quick_tests():
    """Run all quick tests."""
    print("Running quick validation tests...")
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Manifold Basics", test_manifold_basics),
        ("Hyperbolic Layer", test_hyperbolic_layer),
        ("Model Creation", test_model_creation),
        ("Optimizer", test_optimizer),
        ("Training Step", test_training_step),
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
    print("QUICK TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All core functionality tests passed!")
        print("💡 HyperPerturb is ready to use!")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = run_quick_tests()
    
    if success:
        print("\n" + "=" * 50)
        print("NEXT STEPS:")
        print("=" * 50)
        print("1. ✓ Core functionality is working")
        print("2. 📊 Try loading your data with scanpy")
        print("3. 🧪 Run training experiments")
        print("4. 📈 Analyze results and perturbation effects")
        print("5. 🎯 Optimize hyperparameters for your use case")
    
    sys.exit(0 if success else 1)
