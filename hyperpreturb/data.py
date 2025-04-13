import os
import requests
import scanpy as sc
import pandas as pd
import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from pathlib import Path
from hyperpreturb.final_files.utils.data_loader import load_protein_network, create_adjacency_matrix

def download_data(url, output_path, force=False):
    """Download data from a given URL.
    
    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        force: Whether to force download even if file exists. Default: False
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if force or not os.path.exists(output_path):
        print(f"Downloading data from {url} to {output_path}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(output_path, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
        print(f"Download complete: {output_path}")
    else:
        print(f"File {output_path} already exists. Use force=True to re-download.")

def download_string_network(species_id=9606, output_dir="data/raw"):
    """Download STRING protein-protein interaction network.
    
    Args:
        species_id: NCBI Taxonomy identifier (default: 9606 for human)
        output_dir: Directory to save the downloaded file
        
    Returns:
        Path to the downloaded file
    """
    os.makedirs(output_dir, exist_ok=True)
    string_url = f"https://stringdb-static.org/download/protein.links.full.v11.5/{species_id}.protein.links.full.v11.5.txt.gz"
    output_path = os.path.join(output_dir, f"{species_id}.protein.links.full.v11.5.txt.gz")
    
    download_data(string_url, output_path)
    
    # Uncompress the file if it's gzipped
    if output_path.endswith('.gz'):
        import gzip
        import shutil
        
        uncompressed_path = output_path[:-3]  # Remove .gz extension
        with gzip.open(output_path, 'rb') as f_in:
            with open(uncompressed_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"Uncompressed file saved to: {uncompressed_path}")
        return uncompressed_path
    
    return output_path

def preprocess_data(input_path, output_path=None, n_neighbors=15, n_pcs=50):
    """Preprocess raw data and optionally save the processed version.
    
    Args:
        input_path: Path to the raw data file
        output_path: Path to save the processed data. If None, doesn't save.
        n_neighbors: Number of neighbors for the neighborhood graph. Default: 15
        n_pcs: Number of principal components. Default: 50
        
    Returns:
        Processed AnnData object
    """
    print(f"Preprocessing data from {input_path}")
    adata = sc.read(input_path)
    
    # Basic preprocessing
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]
    
    # Scale data
    sc.pp.scale(adata)
    
    # Run PCA
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
    
    # Compute neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    
    # Run UMAP for visualization
    sc.tl.umap(adata)
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        adata.write(output_path)
        print(f"Processed data saved to {output_path}")
    
    return adata

def prepare_perturbation_data(adata, ctrl_key='condition', ctrl_value='control'):
    """Prepare perturbation data by computing fold changes compared to control.
    
    Args:
        adata: AnnData object
        ctrl_key: Key in adata.obs containing control/perturbation information
        ctrl_value: Value in ctrl_key that identifies control samples
        
    Returns:
        AnnData object with added perturbation effects
    """
    # Split data into control and perturbation groups
    is_control = adata.obs[ctrl_key] == ctrl_value
    
    # Get control expression
    control_cells = adata[is_control]
    control_mean = control_cells.X.mean(axis=0)
    
    if sp.issparse(adata.X):
        control_mean = control_mean.toarray().flatten()
    
    # Compute log fold changes for each cell compared to control mean
    if sp.issparse(adata.X):
        adata.obsm['log_fold_change'] = np.array(adata.X.toarray() - control_mean)
    else:
        adata.obsm['log_fold_change'] = adata.X - control_mean
    
    # Add perturbation targets
    if 'perturbation' in adata.obs:
        # Create one-hot encoding of perturbation targets
        perturbations = adata.obs['perturbation'].unique()
        pert_dict = {pert: i for i, pert in enumerate(perturbations)}
        
        one_hot = np.zeros((adata.n_obs, len(perturbations)))
        for i, pert in enumerate(adata.obs['perturbation']):
            one_hot[i, pert_dict[pert]] = 1
        
        adata.obsm['perturbation_target'] = one_hot
    
    return adata

def load_and_preprocess_perturbation_data(rna_path, protein_path=None, network_path=None):
    """
    Load and preprocess perturbation data with optional protein data and PPI network.
    
    Args:
        rna_path: Path to RNA expression data (h5ad)
        protein_path: Path to protein expression data (h5ad, optional)
        network_path: Path to protein-protein interaction network (optional)
        
    Returns:
        Tuple of (processed RNA data, adjacency matrix)
    """
    # Load RNA data
    rna_adata = sc.read_h5ad(rna_path)
    
    # Preprocess RNA data
    rna_adata = preprocess_data(rna_adata)
    
    # Compute adjacency matrix from PPI network if provided
    adj_matrix = None
    if network_path:
        network_df = load_protein_network(network_path)
        gene_list = rna_adata.var_names.tolist()
        adj_matrix = create_adjacency_matrix(network_df, gene_list)
    
    # If protein data is provided, integrate it
    if protein_path:
        protein_adata = sc.read_h5ad(protein_path)
        # Process and integrate protein data
        # This is a placeholder - implementation depends on specific requirements
        
    # Prepare perturbation effects
    rna_adata = prepare_perturbation_data(rna_adata)
    
    return rna_adata, adj_matrix