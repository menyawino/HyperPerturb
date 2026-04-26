# HyperPerturb Training Theory and Pipeline

This document explains the theoretical foundations and practical wiring of the HyperPerturb training pipeline as currently implemented in the codebase.

## 1. Data Model and Preprocessing

### 1.1 Base object: AnnData

- HyperPerturb operates on single-cell perturbation data stored as an `AnnData` object.
- Core components used in training:
  - `adata.X`: cells × genes working matrix after preprocessing (log-transformed and later scaled for graph features).
  - `adata.layers['normalized_counts']`: normalized pre-log expression used to compute signed fold-change targets.
  - `adata.obs`: per-cell metadata, including perturbation labels and control indicators.
  - `adata.var`: per-gene metadata.
  - `adata.obsm['log_fold_change']`: per-cell, per-gene signed log fold-change vs control, computed from normalized counts.
  - `adata.obsm['perturbation_target']`: per-cell, one-hot non-control perturbation labels; control cells remain all-zero.
  - `adata.obsm['protein']`: optional per-cell protein expression matrix (e.g., 24 CITE-seq channels), stored as a separate modality when protein data are available.

### 1.2 Preprocessing pipeline (`hyperpreturb/data.py`)

1. **Loading**
   - RNA data is loaded from an `.h5ad` file via Scanpy.

2. **Subsampling for memory safety**
   - To keep memory usage within ~10 GB, cells are capped at `max_cells` (default 3000).

3. **Filtering and normalization**
   - Filter cells with too few genes; filter genes seen in too few cells.
  - Normalize total counts per cell and store the result in `adata.layers['normalized_counts']`.
  - Apply `log1p` transformation to the working matrix used for feature construction.

4. **Gene selection and scaling**
  - Select top 2000 highly variable genes.
   - Scale expression values (capped) for stable downstream PCA/neighbors.

5. **Optional graph building**
   - PCA and neighbor graph (`sc.pp.neighbors`) are computed for exploratory analysis.

6. **Perturbation-specific processing (`prepare_perturbation_data`)**
  - Requires explicit control metadata via `ctrl_key`/`ctrl_value` (defaults: `perturbation` / `non-targeting`).
  - Computes a **control mean expression per gene** using control cells from `adata.layers['normalized_counts']` when available.
  - Adds `adata.obsm['log_fold_change']` as `log1p(cell_expression) - log1p(control_mean)`.
  - Builds a one-hot matrix `adata.obsm['perturbation_target']` only for non-control perturbations.
  - Stores optional perturbation-to-gene mappings outside this step; those mappings are consumed later when gene-space supervision is built.

7. **Optional PPI / adjacency**
  - If a network file is provided, `load_protein_network` and `create_adjacency_matrix` build a gene–gene adjacency aligned to `adata.var_names`.
  - For STRING protein IDs, a matching `protein.info` mapping file is used to align proteins to gene symbols.

At the end of preprocessing, we have:

- Cell-level perturbation labels (`perturbation_target`).
- Per-cell, per-gene log fold-changes (`log_fold_change`).
- Optional cell-level protein expression (`protein`) stored in `adata.obsm['protein']` when a matching protein `.h5ad` is provided.
- Optional gene–gene adjacency matrix.

These are the raw ingredients for the hyperbolic training pipeline.

---

## 2. Hyperbolic Geometry & Manifolds

### 2.1 Poincaré Ball Manifold (`hyperpreturb/utils/manifolds.py`)

The `PoincareBall` class implements Riemannian geometry operations for the Poincaré ball model of hyperbolic space:

- `distance(x, y)`: geodesic distance between points.
- `expmap(x, v)`: exponential map from tangent vector `v` at point `x` onto the manifold.
- `logmap(x, y)`: logarithmic map from point `y` back to the tangent space at `x`.
- `egrad2rgrad(x, grad)`: converts Euclidean gradients to Riemannian gradients.
- `mobius_addition(x, y)`: Möbius addition, a fundamental operation in hyperbolic analogues of linear algebra.

**Theory:**

- Hyperbolic space (negative curvature) is well-suited for hierarchical and scale-free structures.
- The Poincaré ball model allows embedding of tree-like gene interaction networks with low distortion.

### 2.2 Hyperbolic Optimizer and Schedules (`hyperpreturb/models/hyperbolic.py`)

1. **HyperbolicAdam**
   - Inherits from `tf.keras.optimizers.Adam`.
   - Before applying an update, converts Euclidean gradients at a point `x` to Riemannian gradients via `PoincareBall.egrad2rgrad`.
   - This keeps parameter updates geometrically consistent with the manifold.

2. **QuantumAnnealer**
   - A custom learning-rate schedule combining exponential decay and cosine oscillations.
   - Mimics quantum annealing behavior: high exploration early, then gradual cooling.

3. **HyperbolicDense**
   - Dense layer operating in hyperbolic space.
   - Procedure:
     1. Map inputs from hyperbolic space to tangent space at origin via `logmap`.
     2. Apply linear transformation in tangent space.
     3. Map outputs back to hyperbolic space with `expmap`.
     4. If activation is used, it is applied in tangent space as well.

4. **HyperbolicAttention** (and supporting HyperbolicPoincareBall/HyperbolicLayer)
   - Multi-head attention adapted to hyperbolic geometry.
   - Attention weights are derived from hyperbolic distances rather than Euclidean dot products.
   - Aggregation and projections are done in tangent space, then mapped back.

**Scientific rationale:**

- Gradient-based optimization in curved space requires curvature-aware updates to remain on the manifold and avoid distortion.
- Quantum annealing-inspired schedules help explore the complex loss surface more robustly than simple monotonic decays.

---

## 3. Core Hyperbolic Models

### 3.1 SignedHyperPerturbModel (`hyperpreturb/models/__init__.py`)

`SignedHyperPerturbModel` is the main graph-based model used by the advanced trainer.

- **Inputs:**
  - `x`: node features of shape `(batch, n_nodes, d)`; here `batch=1`, `n_nodes = n_genes`, `d = 1`.
  - `adj`: adjacency matrix of shape `(n_nodes, n_nodes)`, typically sparse.

- **Backbone (encoder):**
  - Two `HyperbolicGraphConv` layers with LayerNorm and Dropout:
    - `encoder_gcn1((x, adj)) → LayerNormalization → encoder_gcn2((h, adj)) → Dropout`.
  - Each `HyperbolicGraphConv`:
    - Projects nodes into hyperbolic space via exp-map.
    - Aggregates neighbor information using sparse matrix multiplication with `adj`.
    - Maps back with log-map and adds a bias term.

- **Heads:**
  - **Policy head:**
    - `policy_gcn((h, adj))` produces latent gene embeddings.
    - Query/key projections score every response-gene/perturbation-gene pair.
    - In the current setup, outputs `(batch, n_genes, n_genes)`: signed effect scores in gene space.
  - **Value head:**
    - `value_gcn((h, adj))` → Dense to 1, output `(batch, n_genes, 1)`: scalar value per gene.

**Interpretation:**

- The encoder learns hyperbolic embeddings of genes that respect both expression-derived features and the gene–gene adjacency.
- The policy head learns, for each response gene, the signed effect associated with perturbing each supervised perturbation gene.
- The value head summarizes the overall perturbation sensitivity of each gene.

### 3.2 HyperbolicPerturbationModel

- A simpler hyperbolic model used in the "simple" trainer.
- Operates on perturbation embeddings rather than a graph.
- Uses `HyperbolicDense` layers to project perturbation inputs into hyperbolic space and back to Euclidean gene expression predictions.

---

## 4. Training Pipeline and Loss Design

### 4.1 Environment & Curriculum (`hyperpreturb/models/train.py`)

1. **PerturbationEnv**
   - Wraps an `AnnData` object.
   - Prepares `env.targets`:
     - If `perturbation` column exists in `adata.obs`, builds a one-hot matrix of shape `(n_cells, n_perts)`.
  - If the column is missing, advanced training fails earlier in preprocessing due to strict contract checks.
   - Provides simple RL-like `step` and `increase_complexity` methods (used primarily for curriculum learning and logging).

2. **ComplexityScheduler**
   - A `tf.keras.callbacks.Callback` that calls `env.increase_complexity` every `frequency` epochs.
   - Conceptually, this allows gradual introduction of task difficulty (e.g., more complex perturbation patterns).

### 4.2 Graph construction in `train_model`

`train_model` in `hyperpreturb/models/train.py` orchestrates the advanced training pipeline:

1. **Environment setup**
   - `env = PerturbationEnv(adata)` ensures `env.targets` and internal state are prepared.

2. **Gene graph inputs**
   - `n_genes = adata.n_vars`.
   - `adj_matrix`:
  - Must be provided explicitly from PPI/network preprocessing.
  - Shape is validated strictly as `(n_genes, n_genes)`; missing adjacency raises `ValueError`.
   - Node features (`gene_features`):
     - `X_dense = adata.X` converted to dense.
     - Per-gene mean across cells:
       - `gene_features = mean_cells(X_dense)  # (1, n_genes)`.
       - Add a feature dimension: `(1, n_genes, 1)`.

3. **Per-gene × per-perturbation signed targets (policy targets)**

Given:
- `lfc = adata.obsm['log_fold_change']` of shape `(n_cells, n_genes)`.
- `supervised_perturbations`: perturbation labels held in for the current split.
- `perturbation_gene_map`: optional explicit mapping from perturbation labels to one or more genes in `adata.var_names`.

We define a signed effect target for each gene–perturbation pair:

1. For each supervised perturbation gene `p`:
  - Select cells with that perturbation label.
  - Compute mean signed logFC for each response gene over those cells.
  - Project that label into one or more gene-space columns using the explicit perturbation-to-gene mapping layer.

2. Build a same-shape binary mask indicating which perturbation-gene columns are supervised in the current split.

3. Concatenate targets and mask along the last axis.
  - Final packed policy target has shape `(1, n_genes, 2 * n_genes)`.

**Interpretation:**

- For each response gene, the policy target preserves both effect magnitude and regulation direction for the supervised perturbation genes in the current split.

### 4.3 Per-gene scalar rewards (value targets)

- From the supervised signed effects, define a scalar per gene:
  - `per_gene_value[g] = mean_p |effect(g, p)|`.
- Final value target: `graph_value_target` of shape `(1, n_genes, 1)`.

**Interpretation:**

- Captures overall sensitivity of each gene to perturbations, independent of which perturbation.

### 4.4 Model compilation and training

1. **Model creation**

```python
model = HyperPerturbModel(num_genes=n_genes, curvature=curvature)
q_schedule = QuantumAnnealer(learning_rate, T_max=epochs)
optimizer = HyperbolicAdam(
    learning_rate=q_schedule,
    manifold=PoincareBall(curvature),
)
```

2. **Losses and metrics**

```python
model.compile(
  optimizer=optimizer,
  loss=[
    masked_signed_huber_loss(name="policy_huber"),
    tf.keras.losses.MeanSquaredError(name="value_mse"),  # value head
  ],
  loss_weights=[1.0, 0.5],
  metrics=[
    masked_signed_mae(name="policy_mae"),
    tf.keras.metrics.MeanAbsoluteError(name="value_mae"),
  ],
)
```

Here `masked_signed_huber_loss` applies the regression loss only on supervised perturbation-gene columns, which allows train and validation splits to hold out different perturbation conditions while preserving a fixed output shape.

- **Policy head:** learns to predict signed gene-space perturbation effects.
- **Value head:** approximates the scalar per-gene reward.

3. **Training call**

```python
history = model.fit(
    x=(gene_features, adj_matrix),
    y=[graph_policy_target, graph_value_target],
    epochs=epochs,
    batch_size=1,  # graph-level batch
    validation_split=0.0,
    callbacks=callbacks,
    verbose=1,
)
```

- Single-graph training: the full gene graph is treated as one example with rich internal structure.
- Curriculum learning and other callbacks operate at the epoch level.

---

## 5. Scientific Interpretation and Limitations

### 5.1 What the model is learning

- **Hyperbolic geometry:**
  - Embeds genes in a curved space tailored to hierarchical and scale-free structure.
  - Distances and aggregations reflect the geometry of gene–gene interactions more faithfully than Euclidean models in many biological settings.

- **Policy head:**
  - For each response gene, predicts signed effects for perturbing each supervised perturbation gene.
  - Useful for ranking candidate perturbations while preserving up- versus down-regulation.

- **Value head:**
  - For each gene, predicts an overall sensitivity or volatility under perturbations.
  - Can be used to identify genes broadly responsive to perturbation regimes.

### 5.2 Assumptions and caveats

- **Reward definition:**
  - The policy target now preserves directionality, but it is still an observational average over perturbed cells rather than a causal intervention estimate.
  - The target now supports explicit perturbation-to-gene mappings, including multiple labels mapping onto the same target gene column.

- **Averaging across cells:**
  - The per-gene, per-perturbation rewards aggregate across all cells with that perturbation.
  - This assumes that cell-to-cell heterogeneity is either limited or not the primary focus; refinements could condition on cell type or state.

- **Single-graph training:**
  - The current advanced trainer treats the entire gene network as one graph.
  - This is suitable for learning global structure but may be extended in the future to multiple graphs (e.g., different conditions or datasets).

### 5.3 Extensions

Potential future refinements include:

- Conditioning rewards on specific cell types or states (stratified `log_fold_change`).
- Extending supervision beyond single-gene target mappings to richer perturbation ontologies or combinatorial perturbation objects.
- Using genuine RL loops where actions are perturbation choices and rewards come from simulated or held-out expression responses.

---

## 6. Summary

- The training pipeline is built on a rigorous hyperbolic geometric foundation (Poincaré ball, Riemannian optimization) and tailored to perturbation data.
- Data preprocessing creates control-based signed log fold-changes and non-control perturbation labels that are transformed into masked gene-space targets.
- `SignedHyperPerturbModel` consumes a gene graph and learns, in hyperbolic space, both a signed policy (which perturbations affect which genes and in what direction) and a value (how sensitive each gene is overall).
- Losses and targets are shape-consistent and scientifically interpretable, making the training loop both mathematically sound and aligned with common biological questions about perturbation impact.
