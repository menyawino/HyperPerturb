import numpy as np
import pandas as pd
import anndata as ad
import pytest

from hyperpreturb.data import prepare_perturbation_data
from hyperpreturb.models.train import train_model


def make_adata(n_cells=6, n_genes=4):
    x = np.random.rand(n_cells, n_genes).astype("float32")
    obs = pd.DataFrame(
        {
            "perturbation": ["non-targeting", "non-targeting", "TP53", "BRCA1", "TP53", "BRCA1"]
        }
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    return ad.AnnData(X=x, obs=obs, var=var)


def test_prepare_perturbation_data_requires_control_value():
    adata = make_adata()

    with pytest.raises(ValueError, match="Control value"):
        prepare_perturbation_data(adata, ctrl_key="perturbation", ctrl_value="missing-control")


def test_prepare_perturbation_data_requires_perturbation_column():
    adata = make_adata()
    adata.obs = pd.DataFrame({"condition": ["ctrl"] * adata.n_obs})

    with pytest.raises(ValueError, match="Control key"):
        prepare_perturbation_data(adata, ctrl_key="perturbation", ctrl_value="non-targeting")


def test_advanced_train_requires_adjacency_matrix():
    adata = make_adata()
    adata = prepare_perturbation_data(adata, ctrl_key="perturbation", ctrl_value="non-targeting")

    with pytest.raises(ValueError, match="adj_matrix is required"):
        train_model(adata=adata, adj_matrix=None, epochs=1, deterministic=True, seed=42)
