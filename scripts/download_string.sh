#!/bin/bash
# Script to download and prepare STRING protein-protein interaction data

# Create directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed

# Define constants
SPECIES_ID=9606  # Human
STRING_VERSION="11.5"
BASE_URL="https://stringdb-static.org/download"

# Download STRING network data
echo "Downloading STRING protein-protein interaction network data..."
wget -O "data/raw/${SPECIES_ID}.protein.links.full.v${STRING_VERSION}.txt.gz" \
    "${BASE_URL}/protein.links.full.v${STRING_VERSION}/${SPECIES_ID}.protein.links.full.v${STRING_VERSION}.txt.gz"

# Uncompress the file
echo "Uncompressing network data..."
gunzip -f "data/raw/${SPECIES_ID}.protein.links.full.v${STRING_VERSION}.txt.gz"

# Download protein aliases for mapping to gene names
echo "Downloading protein name mapping data..."
wget -O "data/raw/${SPECIES_ID}.protein.info.v${STRING_VERSION}.txt.gz" \
    "${BASE_URL}/protein.info.v${STRING_VERSION}/${SPECIES_ID}.protein.info.v${STRING_VERSION}.txt.gz"

# Uncompress the aliases file
echo "Uncompressing protein name mapping data..."
gunzip -f "data/raw/${SPECIES_ID}.protein.info.v${STRING_VERSION}.txt.gz"

echo "Download complete. Files saved to data/raw/"
echo "You can now use these files with the HyperPerturb framework."

# Create an example Python script to demonstrate loading the network
cat > scripts/load_network_example.py << 'EOF'
#!/usr/bin/env python3
"""
Example script demonstrating how to load and process STRING network data
"""
import pandas as pd
from hyperpreturb.utils.data_loader import load_protein_network, create_adjacency_matrix

# Load network with high confidence (score > 700)
network_df = load_protein_network(
    string_path="data/raw/9606.protein.links.full.v11.5.txt",
    gene_mapping_path="data/raw/9606.protein.info.v11.5.txt",
    confidence=700
)

print(f"Loaded {len(network_df)} high-confidence protein interactions")

# Get an example set of genes
example_genes = network_df['protein1_gene'].dropna().unique()[:1000]

# Create adjacency matrix
adj_matrix = create_adjacency_matrix(network_df, example_genes)

print(f"Created adjacency matrix with shape {adj_matrix.shape}")
print(f"Number of non-zero entries: {adj_matrix.nnz}")

# Save as example
import scipy.sparse as sp
sp.save_npz("data/raw/example_adj_matrix.npz", adj_matrix)
print("Example adjacency matrix saved to data/raw/example_adj_matrix.npz")
EOF

chmod +x scripts/load_network_example.py
echo "Created example script at scripts/load_network_example.py"
