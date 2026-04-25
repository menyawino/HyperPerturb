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
    """Download a file from url. Skips if already exists unless force=True."""
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
    """Grab STRING PPI network for a given species (default: human)."""
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
    """Standard scanpy preprocessing: filter, normalize, HVG, PCA, neighbors.

    Subsamples to max_cells first (default 3000) to stay under ~10GB RAM.
    """
    print(f"Preprocessing data from {input_path}")

    if isinstance(input_path, str) or isinstance(input_path, Path):
        adata = sc.read(input_path)
    else:
        adata = input_path

    # subsample for memory
    if max_cells is not None and adata.n_obs > max_cells:
        adata = adata[adata.obs_names[:max_cells]].copy()
        print(f"Subsampled to {adata.n_obs} cells for memory constraints (10GB-safe cap).")
    
    # standard scanpy pipeline
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    # Preserve normalized counts so downstream log fold-change targets are
    # computed from expression, not z-scored features.
    adata.layers['normalized_counts'] = adata.X.copy()
    sc.pp.log1p(adata)

    # 2000 HVGs — balance between signal and memory
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
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
    """Compute log fold-changes vs control and one-hot perturbation labels.

    This function is intentionally strict for reproducible scientific runs:
    the control column and control value must be explicitly present.
    """
    if ctrl_key not in adata.obs.columns:
        raise ValueError(
            f"Control key '{ctrl_key}' is missing from adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    available = adata.obs[ctrl_key].unique()
    if ctrl_value not in available:
        raise ValueError(
            f"Control value '{ctrl_value}' not found in adata.obs['{ctrl_key}']. "
            f"Available values: {available}"
        )

    print(f"Using '{ctrl_key}' column with control value '{ctrl_value}'")
    
    # Split data into control and perturbation groups
    is_control = (adata.obs[ctrl_key] == ctrl_value).to_numpy()
    
    if sum(is_control) == 0:
        raise ValueError(f"No control samples found with {ctrl_key}={ctrl_value}. Available values: {adata.obs[ctrl_key].unique()}")
    
    print(f"Found {sum(is_control)} control samples out of {adata.n_obs} total samples")
    
    effect_source = adata.layers['normalized_counts'] if 'normalized_counts' in adata.layers else adata.X
    if sp.issparse(effect_source):
        effect_matrix = effect_source.toarray().astype(np.float32)
    else:
        effect_matrix = np.asarray(effect_source, dtype=np.float32)

    control_mean = np.mean(effect_matrix[is_control], axis=0).astype(np.float32)

    if 'normalized_counts' in adata.layers:
        adata.obsm['log_fold_change'] = np.log1p(effect_matrix) - np.log1p(control_mean[np.newaxis, :])
    else:
        # Fallback for already-materialized AnnData objects that do not carry
        # normalized counts. This preserves backwards compatibility for tests
        # and older preprocessed files.
        adata.obsm['log_fold_change'] = effect_matrix - control_mean
    
    perturbation_column = 'perturbation' if 'perturbation' in adata.obs else ctrl_key
    if perturbation_column not in adata.obs:
        raise ValueError(
            "Expected a perturbation annotation column in adata.obs to build perturbation targets. "
            f"Tried 'perturbation' and '{ctrl_key}'."
        )

    # Create one-hot encoding only for non-control perturbations; control cells
    # remain all-zero rows instead of being treated as an intervention class.
    perturbations = [pert for pert in adata.obs[perturbation_column].unique() if pert != ctrl_value]
    if not perturbations:
        raise ValueError(
            f"No non-control perturbations found in adata.obs['{perturbation_column}'] after excluding '{ctrl_value}'."
        )
    pert_dict = {pert: i for i, pert in enumerate(perturbations)}

    one_hot = np.zeros((adata.n_obs, len(perturbations)), dtype=np.float32)
    for i, pert in enumerate(adata.obs[perturbation_column]):
        if pert != ctrl_value:
            one_hot[i, pert_dict[pert]] = 1.0

    adata.obsm['perturbation_target'] = one_hot
    adata.uns['perturbation_target_names'] = [str(pert) for pert in perturbations]
    
    return adata

def load_and_preprocess_perturbation_data(
    rna_path,
    protein_path=None,
    network_path=None,
    gene_mapping_path=None,
    preprocessed_path=None,
    max_cells=3000,
    ctrl_key='perturbation',
    ctrl_value='non-targeting',
):
    """
    Load and preprocess perturbation data with optional protein data and PPI network.
    
    Args:
        rna_path: Path to RNA expression data (h5ad)
        protein_path: Path to protein expression data (h5ad, optional)
        network_path: Path to protein-protein interaction network (optional)
        gene_mapping_path: Optional STRING protein-to-gene mapping file
        ctrl_key: Column in adata.obs used to identify control cells
        ctrl_value: Value in ctrl_key that denotes control cells
        
    Returns:
        Tuple of (processed RNA data, adjacency matrix)
    """
    import os
    from pathlib import Path
    
    if protein_path is not None and not os.path.exists(protein_path):
        raise FileNotFoundError(f"Protein data file not found at {protein_path}")

    if network_path is not None and not os.path.exists(network_path):
        raise FileNotFoundError(f"Protein network file not found at {network_path}")

    if preprocessed_path is not None and os.path.exists(preprocessed_path):
        print(f"Loading preprocessed perturbation data from {preprocessed_path}")
        rna_adata = sc.read_h5ad(preprocessed_path)
        rna_adata = prepare_perturbation_data(rna_adata, ctrl_key=ctrl_key, ctrl_value=ctrl_value)
    else:
        if rna_path is None:
            raise ValueError("rna_path must be provided explicitly.")
        if not os.path.exists(rna_path):
            raise FileNotFoundError(f"RNA data file not found at {rna_path}")

        # Load RNA data
        print(f"Loading RNA data from {rna_path}")
        rna_adata = sc.read_h5ad(rna_path)
        
        # Preprocess RNA data
        rna_adata = preprocess_data(rna_adata, max_cells=max_cells)
        
        # Prepare perturbation data
        rna_adata = prepare_perturbation_data(rna_adata, ctrl_key=ctrl_key, ctrl_value=ctrl_value)

        # Optionally save fully processed AnnData for reproducible reruns.
        if preprocessed_path is not None:
            os.makedirs(os.path.dirname(preprocessed_path), exist_ok=True)
            rna_adata.write(preprocessed_path)
            print(f"Saved preprocessed perturbation data to {preprocessed_path}")
    
    # Compute adjacency matrix from PPI network if provided
    adj_matrix = None
    if network_path is not None:
        print(f"Loading protein network from {network_path}")
        network_df = load_protein_network(network_path, gene_mapping_path=gene_mapping_path)
        gene_list = rna_adata.var_names.tolist()
        adj_matrix = create_adjacency_matrix(network_df, gene_list)
    
    # If protein data is provided, integrate it as a separate modality
    if protein_path is not None:
        print(f"Loading protein data from {protein_path}")
        protein_adata = sc.read_h5ad(protein_path)

        # Match protein data with RNA data
        common_cells = list(set(protein_adata.obs_names).intersection(set(rna_adata.obs_names)))
        if len(common_cells) == 0:
            raise ValueError("No common cells found between RNA and protein data")

        print(f"Found {len(common_cells)} common cells between RNA and protein data")

        # Subset both datasets to common cells
        rna_subset = rna_adata[common_cells]
        protein_subset = protein_adata[common_cells]

        # Store protein expression as separate modality; shape (n_cells, n_proteins)
        rna_subset.obsm['protein'] = protein_subset.X

        # Use the subset with both modalities
        rna_adata = rna_subset
        print("Successfully integrated protein data with RNA data")

    return rna_adata, adj_matrix