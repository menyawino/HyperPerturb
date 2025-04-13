def main():
    # Initialize components
    strategy = tf.distribute.MirroredStrategy()
    data_loader = SingleCellDataset("data/raw/hematopoietic.h5ad")
    adata = data_loader.preprocess()
    
    with strategy.scope():
        model = HyperPerturbModel(
            num_genes=adata.n_vars, 
            curvature=0.8
        )
        optimizer = HyperbolicAdam(
            learning_rate=3e-4, 
            manifold=model.manifold
        )
    
    # XLA-optimized training step
    @tf.function(jit_compile=True)
    def train_step(batch):
        with tf.GradientTape() as tape:
            logits, values = model(batch)
            loss = compute_loss(batch, logits, values)
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    
    # Run training
    dataset = data_loader.to_tf_dataset()
    for epoch in range(200):
        for batch in dataset:
            loss = train_step(batch)
        print(f"Epoch {epoch+1}: Loss {loss.numpy():.4f}")
