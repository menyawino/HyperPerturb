import os
import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf
import scipy.sparse as sp
from pathlib import Path


def _infer_gene_mapping_path(string_path):
    path = Path(string_path)
    candidates = []
    if "protein.links" in path.name:
        candidates.append(path.with_name(path.name.replace("protein.links", "protein.info")))
    if ".links." in path.name:
        candidates.append(path.with_name(path.name.replace(".links.", ".info.")))
    candidates.extend(
        [
            path.with_name("protein.info.v12.0.txt"),
            path.with_name("protein.info.v12.0.txt.gz"),
            path.with_name("9606.protein.info.v12.0.txt"),
            path.with_name("9606.protein.info.v12.0.txt.gz"),
        ]
    )

    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.exists():
            return candidate_str
    return None


def _load_gene_mapping(gene_mapping_path):
    if not os.path.exists(gene_mapping_path):
        raise FileNotFoundError(f"Gene mapping file not found: {gene_mapping_path}")

    mapping_df = pd.read_csv(gene_mapping_path, sep="\t")
    protein_column = next(
        (column for column in ["protein_id", "protein_external_id", "#string_protein_id", "string_protein_id"] if column in mapping_df.columns),
        None,
    )
    gene_column = next(
        (column for column in ["gene_symbol", "preferred_name", "preferred_gene_name"] if column in mapping_df.columns),
        None,
    )

    if protein_column is None or gene_column is None:
        raise ValueError(
            "Gene mapping file must contain a STRING protein identifier column and a gene symbol column. "
            f"Available columns: {list(mapping_df.columns)}"
        )

    mapping_df = mapping_df[[protein_column, gene_column]].dropna()
    return dict(zip(mapping_df[protein_column].astype(str), mapping_df[gene_column].astype(str)))

def load_data(path, normalize=True, log_transform=True, highly_variable=3000):
    """Load h5ad and run basic preprocessing (normalize, log1p, HVG filter)."""
    adata = sc.read_h5ad(path)
    
    # Apply standard preprocessing steps
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
    
    if log_transform:
        sc.pp.log1p(adata)
    
    if highly_variable > 0:
        sc.pp.highly_variable_genes(adata, n_top_genes=highly_variable)
        adata = adata[:, adata.var.highly_variable]
    
    return adata

def load_protein_network(string_path="data/raw/protein.links.v12.0.txt",
                         gene_mapping_path=None,
                         confidence=700):
    """Load STRING PPI network, filter by confidence score."""
    network_df = pd.read_csv(string_path, sep=r'\s+', engine='python')
    
    # Filter by confidence score
    network_df = network_df[network_df['combined_score'] > confidence]
    
    # Map protein IDs to gene symbols if mapping provided
    resolved_mapping_path = gene_mapping_path or _infer_gene_mapping_path(string_path)
    if resolved_mapping_path:
        protein_to_gene = _load_gene_mapping(resolved_mapping_path)
        
        network_df['protein1_gene'] = network_df['protein1'].map(protein_to_gene)
        network_df['protein2_gene'] = network_df['protein2'].map(protein_to_gene)
        
        # Remove unmapped entries
        network_df = network_df.dropna(subset=['protein1_gene', 'protein2_gene'])
    
    return network_df

def create_adjacency_matrix(network_df, gene_list, weighted=True):
    """Build sparse adjacency matrix from network edges, aligned to gene_list."""
    gene_indices = {gene: i for i, gene in enumerate(gene_list)}
    
    # Get edges between genes that exist in the gene list
    edges = []
    weights = []
    
    for _, row in network_df.iterrows():
        gene1 = row.get('protein1_gene', row['protein1'])
        gene2 = row.get('protein2_gene', row['protein2'])
        
        if gene1 in gene_indices and gene2 in gene_indices:
            i, j = gene_indices[gene1], gene_indices[gene2]
            edges.append((i, j))
            edges.append((j, i))  # Undirected graph
            
            if weighted:
                weight = row['combined_score'] / 1000.0  # Normalize to [0, 1]
                weights.extend([weight, weight])
            else:
                weights.extend([1.0, 1.0])

    if not edges:
        raise ValueError(
            "No overlapping network edges were found for the provided gene list. "
            "If the network uses STRING protein IDs, map them to gene symbols before building adjacency."
        )
    
    # Create sparse adjacency matrix
    i, j = zip(*edges)
    adj = sp.coo_matrix((weights, (i, j)), shape=(len(gene_list), len(gene_list)))
    
    # Convert to CSR format for efficient operations
    return adj.tocsr()

def prepare_tf_dataset(adata, adj_matrix, batch_size=128, shuffle=True, targets=None):
    """Wrap AnnData + adjacency into a tf.data.Dataset for training."""
    # Dense expression matrix
    if sp.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    # Convert adjacency matrix to TensorFlow sparse tensor
    adj_indices = np.array(sp.find(adj_matrix)[0:2]).T
    adj_values = sp.find(adj_matrix)[2]
    adj_shape = adj_matrix.shape
    adj_tensor = tf.sparse.SparseTensor(
        indices=adj_indices,
        values=adj_values,
        dense_shape=adj_shape
    )
    
    # Create dataset
    if targets is not None:
        dataset = tf.data.Dataset.from_tensor_slices((
            {"expression": tf.convert_to_tensor(X, dtype=tf.float32), 
             "adjacency": adj_tensor},
            targets
        ))
    else:
        dataset = tf.data.Dataset.from_tensor_slices({
            "expression": tf.convert_to_tensor(X, dtype=tf.float32),
            "adjacency": adj_tensor
        })
    
    # Apply shuffling and batching
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)