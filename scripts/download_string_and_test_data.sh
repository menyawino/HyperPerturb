#!/bin/bash
# Script to download STRING network and FrangiehIzar2021 test datasets for HyperPerturb
#
# This script will:
#   - Download and unpack STRING v12.0 protein-protein interaction network (all species)
#   - Download FrangiehIzar2021 protein and RNA perturbation datasets (h5ad) from Figshare
#
# All files are placed under data/raw/ and can be used directly
# with the examples in the README and the HyperPerturb codebase.

# Create directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed

# Download STRING network data (v12.0, all species, lighter file)
echo "Downloading STRING protein-protein interaction network data (v12.0)..."
wget -nc -O "data/raw/protein.links.v12.0.txt.gz" \
    "https://stringdb-downloads.org/download/protein.links.v12.0.txt.gz"

# Uncompress the file
echo "Uncompressing network data..."
gunzip -f "data/raw/protein.links.v12.0.txt.gz" || true

# NOTE: v12.0 generic file does not come with a per-species info table in this script.
# If you need gene symbol mappings, you can either:
#   - Download the corresponding protein.info file separately, or
#   - Work directly with protein IDs as nodes.

echo "Downloading FrangiehIzar2021 protein and RNA perturbation datasets (h5ad)..."

# Download protein expression perturbation data (h5ad)
PROTEIN_H5AD_URL="https://plus.figshare.com/ndownloader/files/42428325"
PROTEIN_H5AD_PATH="data/raw/FrangiehIzar2021_Protein.h5ad"
if [ -f "$PROTEIN_H5AD_PATH" ]; then
    echo "Protein h5ad already exists at $PROTEIN_H5AD_PATH, skipping download."
else
    wget -O "$PROTEIN_H5AD_PATH" "$PROTEIN_H5AD_URL"
fi

# Download RNA expression perturbation data (h5ad)
RNA_H5AD_URL="https://plus.figshare.com/ndownloader/files/42428808"
RNA_H5AD_PATH="data/raw/FrangiehIzar2021_RNA.h5ad"
if [ -f "$RNA_H5AD_PATH" ]; then
    echo "RNA h5ad already exists at $RNA_H5AD_PATH, skipping download."
else
    wget -O "$RNA_H5AD_PATH" "$RNA_H5AD_URL"
fi

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
    string_path="data/raw/protein.links.v12.0.txt",
    gene_mapping_path=None,
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
