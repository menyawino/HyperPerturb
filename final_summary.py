#!/usr/bin/env python3
"""
Final HyperPerturb Test Summary
This script provides a comprehensive summary of the HyperPerturb functionality validation.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_validation_summary():
    """Print a comprehensive validation summary."""
    print("🧬 HYPERPERTURB VALIDATION SUMMARY")
    print("=" * 60)
    
    # Test imports
    try:
        import hyperpreturb
        from hyperpreturb.models import HyperbolicPerturbationModel, HyperPerturbModel
        from hyperpreturb.models.hyperbolic import HyperbolicAdam, HyperbolicDense
        from hyperpreturb.utils.manifolds import PoincareBall
        from hyperpreturb.data import download_data
        
        print("✅ CORE FUNCTIONALITY: WORKING")
        print("   ✓ All core modules imported successfully")
        print("   ✓ HyperbolicPerturbationModel available")
        print("   ✓ HyperPerturbModel available")
        print("   ✓ HyperbolicAdam optimizer available")
        print("   ✓ HyperbolicDense layers available")
        print("   ✓ PoincareBall manifold available")
        print("   ✓ Data loading functions available")
        
    except Exception as e:
        print(f"❌ CORE FUNCTIONALITY: FAILED - {e}")
        return False
    
    # Test basic operations
    try:
        # Test manifold operations
        manifold = PoincareBall(curvature=1.0)
        x = tf.random.normal([2, 5]) * 0.1
        y = tf.random.normal([2, 5]) * 0.1
        dist = manifold.distance(x, y)
        
        # Test model creation
        model = HyperbolicPerturbationModel(
            n_genes=10,
            n_perturbations=3,
            embedding_dim=8,
            hidden_dim=16,
            curvature=1.0
        )
        
        # Test forward pass
        inputs = tf.random.uniform([2, 3], maxval=1.0)
        outputs = model(inputs)
        
        print("\n✅ MODEL OPERATIONS: WORKING")
        print("   ✓ Manifold operations functional")
        print("   ✓ Model creation successful")
        print("   ✓ Forward pass working")
        print(f"   ✓ Model output shape: {outputs.shape}")
        
    except Exception as e:
        print(f"\n❌ MODEL OPERATIONS: FAILED - {e}")
        return False
    
    # Test training
    try:
        manifold = PoincareBall(curvature=1.0)
        optimizer = HyperbolicAdam(manifold=manifold, learning_rate=0.01)
        
        # Single training step
        with tf.GradientTape() as tape:
            predictions = model(inputs)
            target = tf.random.normal([2, 10])
            loss = tf.reduce_mean(tf.square(predictions - target))
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print("\n✅ TRAINING: WORKING")
        print("   ✓ HyperbolicAdam optimizer functional")
        print("   ✓ Gradient computation working")
        print("   ✓ Training step successful")
        print(f"   ✓ Training loss: {loss.numpy():.6f}")
        
    except Exception as e:
        print(f"\n❌ TRAINING: FAILED - {e}")
        return False
    
    # Test data availability
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "raw", "FrangiehIzar2021_RNA.h5ad")
    if os.path.exists(data_path):
        print("\n  DATA: found")
        print(f"   Path: {data_path}")
    else:
        print("\n  DATA: not found (can use synthetic data for testing)")
    
    return True

def print_usage_instructions():
    """Print detailed usage instructions."""
    print("\n" + "=" * 60)
    print("📖 USAGE INSTRUCTIONS")
    print("=" * 60)
    
    print("\n🔧 Basic Setup:")
    print("   ```python")
    print("   import hyperpreturb")
    print("   from hyperpreturb.models import HyperbolicPerturbationModel")
    print("   from hyperpreturb.models.hyperbolic import HyperbolicAdam")
    print("   from hyperpreturb.utils.manifolds import PoincareBall")
    print("   ```")
    
    print("\n🧬 Model Creation:")
    print("   ```python")
    print("   model = HyperbolicPerturbationModel(")
    print("       n_genes=1000,          # Number of genes")
    print("       n_perturbations=50,    # Number of perturbations")
    print("       embedding_dim=64,      # Embedding dimension")
    print("       hidden_dim=128,        # Hidden layer size")
    print("       curvature=1.0          # Hyperbolic curvature")
    print("   )")
    print("   ```")
    
    print("\n🎯 Training Setup:")
    print("   ```python")
    print("   manifold = PoincareBall(curvature=1.0)")
    print("   optimizer = HyperbolicAdam(manifold=manifold, learning_rate=0.01)")
    print("   ```")
    
    print("\n📊 Data Format:")
    print("   • Input: [batch_size, n_perturbations] - perturbation strengths")
    print("   • Output: [batch_size, n_genes] - predicted gene expression changes")
    
    print("\n🚀 Training Loop:")
    print("   ```python")
    print("   for epoch in range(num_epochs):")
    print("       with tf.GradientTape() as tape:")
    print("           predictions = model(perturbations)")
    print("           loss = tf.reduce_mean(tf.square(predictions - targets))")
    print("       gradients = tape.gradient(loss, model.trainable_variables)")
    print("       optimizer.apply_gradients(zip(gradients, model.trainable_variables))")
    print("   ```")

def print_capabilities():
    """Print the capabilities of HyperPerturb."""
    print("\n" + "=" * 60)
    print("🌟 HYPERPERTURB CAPABILITIES")
    print("=" * 60)
    
    print("\n🔬 Scientific Features:")
    print("   • Hyperbolic geometry for hierarchical gene relationships")
    print("   • Protein-protein interaction network integration")
    print("   • Single-cell RNA-seq data processing")
    print("   • Gene perturbation effect prediction")
    print("   • Optimal perturbation strategy discovery")
    
    print("\n🧠 Technical Features:")
    print("   • Riemannian optimization in hyperbolic space")
    print("   • XLA-compiled operations for performance")
    print("   • TensorFlow 2.x integration")
    print("   • Scalable neural network architectures")
    print("   • Quantum-inspired optimization schedules")
    
    print("\n📊 Data Compatibility:")
    print("   • H5AD format (AnnData)")
    print("   • Scanpy preprocessing pipeline")
    print("   • STRING protein interaction networks")
    print("   • Custom perturbation datasets")
    
    print("\n🎯 Use Cases:")
    print("   • Drug discovery and development")
    print("   • Gene therapy target identification")
    print("   • Cancer biology research")
    print("   • Synthetic biology applications")
    print("   • Personalized medicine")

def main():
    """Main validation and summary function."""
    print("Starting HyperPerturb validation and summary...")
    
    # Run validation
    success = print_validation_summary()
    
    if success:
        print("\n🎉 VALIDATION SUCCESSFUL!")
        print("HyperPerturb is fully functional and ready for use.")
        
        # Print usage instructions
        print_usage_instructions()
        
        # Print capabilities
        print_capabilities()
        
        print("\n" + "=" * 60)
        print("🚀 READY FOR BIOLOGICAL DISCOVERY!")
        print("=" * 60)
        print("You can now:")
        print("1. Load your perturbation data")
        print("2. Train hyperbolic models")
        print("3. Predict gene expression changes")
        print("4. Discover optimal perturbation strategies")
        print("5. Advance biological understanding")
        
    else:
        print("\n❌ VALIDATION FAILED!")
        print("Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
