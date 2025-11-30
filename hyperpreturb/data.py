import os
import requests
import scanpy as sc
import pandas as pd
import tensorflow as tf
import scipy.sparse as sp
import numpy as np
from pathlib import Path
from hyperpreturb.utils.data_loader import load_protein_network, create_adjacency_matrix

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

def preprocess_data(input_path, output_path=None, n_neighbors=15, n_pcs=20, max_cells=3000):
    """Preprocess raw data and optionally save the processed version.
    
    Args:
        input_path: Path to the raw data file or AnnData object
        output_path: Path to save the processed data. If None, doesn't save.
        n_neighbors: Number of neighbors for the neighborhood graph. Default: 15
        n_pcs: Number of principal components. Default: 50
        
    Returns:
        Processed AnnData object
    """
    print(f"Preprocessing data from {input_path}")
    
    # Handle both string paths and AnnData objects
    if isinstance(input_path, str) or isinstance(input_path, Path):
        adata = sc.read(input_path)
    else:
        # Assume input_path is already an AnnData object
        adata = input_path

    # Subsample cells to reduce memory footprint if necessary.
    # With 10GB RAM, we keep at most `max_cells` (default 3000).
    if max_cells is not None and adata.n_obs > max_cells:
        adata = adata[adata.obs_names[:max_cells]].copy()
        print(f"Subsampled to {adata.n_obs} cells for memory constraints (10GB-safe cap).")
    
    # Basic preprocessing
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Find highly variable genes (reduced number for memory efficiency)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000)
    adata = adata[:, adata.var.highly_variable]
    
    # Scale data (can be memory intensive; operates on subsampled data)
    sc.pp.scale(adata, max_value=10)
    
    # Run PCA with fewer components
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
    
    # Compute neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    
    # UMAP is primarily for visualization and can be expensive; skip by default
    
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        adata.write(output_path)
        print(f"Processed data saved to {output_path}")
    
    return adata

def prepare_perturbation_data(adata, ctrl_key='perturbation', ctrl_value='non-targeting'):
    """Prepare perturbation data by computing fold changes compared to control.
    
    Args:
        adata: AnnData object
        ctrl_key: Key in adata.obs containing control/perturbation information
        ctrl_value: Value in ctrl_key that identifies control samples
        
    Returns:
        AnnData object with added perturbation effects
    """
    # Select control column and label based on available metadata
    if ctrl_key not in adata.obs.columns:
        potential_keys = ['perturbation', 'perturbation_type', 'perturbation_2']
        for key in potential_keys:
            if key in adata.obs.columns:
                ctrl_key = key
                break

    if ctrl_key not in adata.obs.columns:
        raise ValueError(f"Could not find control key in data. Available columns: {list(adata.obs.columns)}")

    available = adata.obs[ctrl_key].unique()
    # Prefer 'non-targeting' if present; otherwise fall back to 'control'
    if 'non-targeting' in available:
        ctrl_value = 'non-targeting'
    elif 'control' in available:
        ctrl_value = 'control'

    print(f"Using '{ctrl_key}' column with control value '{ctrl_value}'")
    
    # Split data into control and perturbation groups
    is_control = adata.obs[ctrl_key] == ctrl_value
    
    if sum(is_control) == 0:
        raise ValueError(f"No control samples found with {ctrl_key}={ctrl_value}. Available values: {adata.obs[ctrl_key].unique()}")
    
    print(f"Found {sum(is_control)} control samples out of {adata.n_obs} total samples")
    
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

def load_and_preprocess_perturbation_data(rna_path, protein_path=None, network_path=None, preprocessed_path=None):
    """
    Load and preprocess perturbation data with optional protein data and PPI network.
    
    Args:
        rna_path: Path to RNA expression data (h5ad)
        protein_path: Path to protein expression data (h5ad, optional)
        network_path: Path to protein-protein interaction network (optional)
        
    Returns:
        Tuple of (processed RNA data, adjacency matrix)
    """
    import os
    from pathlib import Path
    
    # Fast path: load preprocessed AnnData if provided
    if preprocessed_path is not None and os.path.exists(preprocessed_path):
        print(f"Loading preprocessed perturbation data from {preprocessed_path}")
        rna_adata = sc.read_h5ad(preprocessed_path)
        adj_matrix = None
        return rna_adata, adj_matrix

    # Handle default paths for data files that exist in the project
    if rna_path is None or not os.path.exists(rna_path):
        default_rna_path = Path(__file__).parent.parent / "data" / "raw" / "FrangiehIzar2021_RNA.h5ad"
        if os.path.exists(default_rna_path):
            print(f"Using default RNA data file: {default_rna_path}")
            rna_path = default_rna_path
        else:
            raise FileNotFoundError(f"RNA data file not found at {rna_path}")
    
    if protein_path is None:
        default_protein_path = Path(__file__).parent.parent / "data" / "raw" / "FrangiehIzar2021_protein.h5ad"
        if os.path.exists(default_protein_path):
            print(f"Found protein data file: {default_protein_path}")
            protein_path = default_protein_path
    
    # Load RNA data
    print(f"Loading RNA data from {rna_path}")
    rna_adata = sc.read_h5ad(rna_path)
    
    # Preprocess RNA data
    rna_adata = preprocess_data(rna_adata)
    
    # Prepare perturbation data
    rna_adata = prepare_perturbation_data(rna_adata)
    
    # Compute adjacency matrix from PPI network if provided
    adj_matrix = None
    if network_path and os.path.exists(network_path):
        try:
            from hyperpreturb.utils.data_loader import load_protein_network, create_adjacency_matrix
            print(f"Loading protein network from {network_path}")
            network_df = load_protein_network(network_path)
            gene_list = rna_adata.var_names.tolist()
            adj_matrix = create_adjacency_matrix(network_df, gene_list)
        except Exception as e:
            print(f"Error loading protein network: {e}")
    
    # If protein data is provided, integrate it as a separate modality
    if protein_path and os.path.exists(protein_path):
        try:
            print(f"Loading protein data from {protein_path}")
            protein_adata = sc.read_h5ad(protein_path)
            
            # Match protein data with RNA data
            common_cells = list(set(protein_adata.obs_names).intersection(set(rna_adata.obs_names)))
            if len(common_cells) > 0:
                print(f"Found {len(common_cells)} common cells between RNA and protein data")
                
                # Subset both datasets to common cells
                rna_subset = rna_adata[common_cells]
                protein_subset = protein_adata[common_cells]
                
                # Store protein expression as separate modality; shape (n_cells, n_proteins)
                # Using obsm avoids the (n_obs, n_vars) constraint of layers
                rna_subset.obsm['protein'] = protein_subset.X
                
                # Use the subset with both modalities
                rna_adata = rna_subset
                print("Successfully integrated protein data with RNA data")
            else:
                print("No common cells found between RNA and protein data")
        except Exception as e:
            print(f"Error integrating protein data: {e}")
    
    return rna_adata, adj_matrix