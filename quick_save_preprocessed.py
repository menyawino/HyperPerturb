from pathlib import Path

from hyperpreturb.data import load_and_preprocess_perturbation_data


if __name__ == "__main__":
    # Adjust these paths if your data lives elsewhere
    rna_path = Path("/mnt/omar/projects/hyperperturb/data/raw/FrangiehIzar2021_RNA.h5ad")
    protein_path = Path("/mnt/omar/projects/hyperperturb/data/raw/FrangiehIzar2021_protein.h5ad")
    out_path = Path("/mnt/omar/projects/hyperperturb/data/processed/preprocessed_perturbation.h5ad")

    print("Loading and preprocessing RNA (and optional protein) data...")
    adata, adj = load_and_preprocess_perturbation_data(
        rna_path=str(rna_path),
        protein_path=str(protein_path) if protein_path.exists() else None,
        network_path=None,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(out_path)
    print(f"Saved preprocessed AnnData to {out_path}")
