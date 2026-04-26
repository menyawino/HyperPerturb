#!/usr/bin/env python3
"""
Example script demonstrating how to load and process STRING network data
"""
import pandas as pd
from hyperpreturb.utils.data_loader import load_protein_network, create_adjacency_matrix

# Load network with high confidence (score > 700)
network_df = load_protein_network(
    string_path="data/raw/protein.links.v12.0.txt",
    gene_mapping_path="data/raw/protein.info.v12.0.txt",
    confidence=700
)

print(f"Loaded {len(network_df)} high-confidence protein interactions")

# Get an example set of genes aligned to gene symbols
example_genes = network_df['protein1_gene'].dropna().unique()[:1000]

# Create adjacency matrix
adj_matrix = create_adjacency_matrix(network_df, example_genes)

print(f"Created adjacency matrix with shape {adj_matrix.shape}")
print(f"Number of non-zero entries: {adj_matrix.nnz}")

# Save as example
import scipy.sparse as sp
sp.save_npz("data/raw/example_adj_matrix.npz", adj_matrix)
print("Example adjacency matrix saved to data/raw/example_adj_matrix.npz")
