#!/bin/bash
# Script to download STRING network and FrangiehIzar2021 test datasets for HyperPerturb
#
# This script will:
#   - Download and unpack the human STRING v12.0 protein-protein interaction network
#   - Download the matching human STRING protein info table for protein-to-gene mapping
#   - Download FrangiehIzar2021 protein and RNA perturbation datasets (h5ad) from Figshare
#
# All files are placed under data/raw/ and can be used directly
# with the examples in the README and the HyperPerturb codebase.

# Create directories if they don't exist
mkdir -p data/raw
mkdir -p data/processed

download_file() {
    local url="$1"
    local output_path="$2"

    if command -v wget >/dev/null 2>&1; then
        wget -nc -O "$output_path" "$url"
        return
    fi

    if command -v curl >/dev/null 2>&1; then
        if [ -f "$output_path" ]; then
            echo "File already exists at $output_path, skipping download."
            return
        fi
        curl -L --fail --output "$output_path" "$url"
        return
    fi

    echo "Neither wget nor curl is available for downloading $url" >&2
    return 1
}

# Download STRING network data (human / 9606)
echo "Downloading STRING protein-protein interaction network data (human, v12.0)..."
download_file \
    "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz" \
    "data/raw/protein.links.v12.0.txt.gz"

# Uncompress the file
echo "Uncompressing network data..."
gunzip -f "data/raw/protein.links.v12.0.txt.gz" || true

echo "Downloading STRING protein-to-gene mapping table (human, v12.0)..."
download_file \
    "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz" \
    "data/raw/protein.info.v12.0.txt.gz"

echo "Uncompressing protein info mapping..."
gunzip -f "data/raw/protein.info.v12.0.txt.gz" || true

# The generic STRING links file uses protein IDs. The matching protein.info
# table lets HyperPerturb align those IDs to gene symbols in adata.var_names.

echo "Downloading FrangiehIzar2021 protein and RNA perturbation datasets (h5ad)..."

# Download protein expression perturbation data (h5ad)
PROTEIN_H5AD_URL="https://plus.figshare.com/ndownloader/files/42428325"
PROTEIN_H5AD_PATH="data/raw/FrangiehIzar2021_Protein.h5ad"
if [ -f "$PROTEIN_H5AD_PATH" ]; then
    echo "Protein h5ad already exists at $PROTEIN_H5AD_PATH, skipping download."
else
    download_file "$PROTEIN_H5AD_URL" "$PROTEIN_H5AD_PATH"
fi

# Download RNA expression perturbation data (h5ad)
RNA_H5AD_URL="https://plus.figshare.com/ndownloader/files/42428808"
RNA_H5AD_PATH="data/raw/FrangiehIzar2021_RNA.h5ad"
if [ -f "$RNA_H5AD_PATH" ]; then
    echo "RNA h5ad already exists at $RNA_H5AD_PATH, skipping download."
else
    download_file "$RNA_H5AD_URL" "$RNA_H5AD_PATH"
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
EOF

chmod +x scripts/load_network_example.py
echo "Created example script at scripts/load_network_example.py"
