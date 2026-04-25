import numpy as np
import pandas as pd
import anndata as ad
import pytest

from hyperpreturb.data import prepare_perturbation_data
from hyperpreturb.models.training_utils import (
    build_signed_effect_targets,
    split_anndata_by_perturbation,
)
from hyperpreturb.models.train import train_model


def make_adata():
    x = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 0.5],
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, -2.0, 0.0],
        ],
        dtype="float32",
    )
    obs = pd.DataFrame(
        {
            "perturbation": ["non-targeting", "non-targeting", "TP53", "BRCA1", "TP53", "BRCA1"]
        }
    )
    var = pd.DataFrame(index=["TP53", "BRCA1", "EGFR"])
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


def test_split_anndata_by_perturbation_holds_out_conditions():
    adata = make_adata()

    train_adata, validation_adata, metadata = split_anndata_by_perturbation(
        adata,
        validation_split=0.5,
        seed=7,
    )

    assert validation_adata is not None
    assert set(metadata["train_perturbations"]).isdisjoint(set(metadata["validation_perturbations"]))
    assert set(train_adata.obs["perturbation"]) - {"non-targeting"} == set(metadata["train_perturbations"])
    assert set(validation_adata.obs["perturbation"]) - {"non-targeting"} == set(metadata["validation_perturbations"])


def test_build_signed_effect_targets_preserves_directionality():
    adata = make_adata()

    policy_target, value_target, metadata = build_signed_effect_targets(
        adata,
        supervised_perturbations=["TP53", "BRCA1"],
    )

    n_genes = adata.n_vars
    signed_effects = policy_target[0, :, :n_genes]
    supervision_mask = policy_target[0, :, n_genes:]

    np.testing.assert_allclose(signed_effects[:, metadata["perturbation_indices"][0]], [0.5, -0.5, 0.25])
    np.testing.assert_allclose(signed_effects[:, metadata["perturbation_indices"][1]], [1.5, -1.5, 0.0])
    np.testing.assert_allclose(value_target[0, :, 0], [1.0, 1.0, 0.125])
    assert np.all(supervision_mask[:, metadata["perturbation_indices"][0]] == 1.0)
    assert np.all(supervision_mask[:, metadata["perturbation_indices"][1]] == 1.0)
    assert np.all(supervision_mask[:, 2] == 0.0)


def test_advanced_train_supports_held_out_perturbations(tmp_path):
    adata = make_adata()
    adata = prepare_perturbation_data(adata, ctrl_key="perturbation", ctrl_value="non-targeting")
    adj_matrix = np.eye(adata.n_vars, dtype="float32")

    _, history = train_model(
        adata=adata,
        adj_matrix=adj_matrix,
        model_dir=str(tmp_path),
        epochs=1,
        validation_split=0.5,
        debug=True,
        euclidean_baseline=True,
        deterministic=True,
        seed=42,
    )

    assert "loss" in history.history
    assert "val_loss" in history.history
