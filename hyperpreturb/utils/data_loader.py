import os
import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf
import scipy.sparse as sp
from pathlib import Path

def load_data(path, normalize=True, log_transform=True, highly_variable=3000):
    """
    Load AnnData object from h5ad file with advanced preprocessing.
    
    Args:
        path: Path to the h5ad file
        normalize: Whether to normalize the data. Default: True
        log_transform: Whether to apply log transformation. Default: True
        highly_variable: Number of highly variable genes to keep. Default: 3000
        
    Returns:
        Processed AnnData object
    """
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
    """
    Load and process STRING protein-protein interaction network.
    
    Args:
        string_path: Path to STRING database file
        gene_mapping_path: Path to gene name mapping file (optional)
        confidence: Confidence threshold for interactions. Default: 700 (high confidence)
        
    Returns:
        Processed network as a pandas DataFrame
    """
    # Load the raw network data
    network_df = pd.read_csv(string_path, sep=' ')
    
    # Filter by confidence score
    network_df = network_df[network_df['combined_score'] > confidence]
    
    # Map protein IDs to gene symbols if mapping provided
    if gene_mapping_path:
        mapping_df = pd.read_csv(gene_mapping_path, sep='\t')
        protein_to_gene = dict(zip(mapping_df['protein_id'], mapping_df['gene_symbol']))
        
        network_df['protein1_gene'] = network_df['protein1'].map(protein_to_gene)
        network_df['protein2_gene'] = network_df['protein2'].map(protein_to_gene)
        
        # Remove unmapped entries
        network_df = network_df.dropna(subset=['protein1_gene', 'protein2_gene'])
    
    return network_df

def create_adjacency_matrix(network_df, gene_list, weighted=True):
    """
    Create adjacency matrix from network data.
    
    Args:
        network_df: DataFrame with network data
        gene_list: List of genes to include in the adjacency matrix
        weighted: Whether to use weighted edges. Default: True
        
    Returns:
        Sparse adjacency matrix
    """
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
    
    # Create sparse adjacency matrix
    i, j = zip(*edges)
    adj = sp.coo_matrix((weights, (i, j)), shape=(len(gene_list), len(gene_list)))
    
    # Convert to CSR format for efficient operations
    return adj.tocsr()

def prepare_tf_dataset(adata, adj_matrix, batch_size=128, shuffle=True, targets=None):
    """
    Prepare TensorFlow dataset for model training.
    
    Args:
        adata: AnnData object with gene expression data
        adj_matrix: Adjacency matrix as scipy sparse matrix
        batch_size: Batch size for training. Default: 128
        shuffle: Whether to shuffle the data. Default: True
        targets: Optional targets for supervised learning
        
    Returns:
        TensorFlow dataset
    """
    # Convert sparse to dense if needed
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