import numpy as np
import pandas as pd
import anndata as ad
import pytest

from hyperpreturb.data import load_and_preprocess_perturbation_data, prepare_perturbation_data
from hyperpreturb.models.inference import HyperPerturbInference
from hyperpreturb.models.training_utils import (
    build_signed_effect_targets,
    split_anndata_by_perturbation,
)
from hyperpreturb.models.train import train_model
from hyperpreturb.utils.data_loader import create_adjacency_matrix, load_protein_network


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


def test_prepare_perturbation_data_excludes_control_from_targets():
    adata = make_adata()
    adata = prepare_perturbation_data(adata, ctrl_key="perturbation", ctrl_value="non-targeting")

    assert adata.obsm["perturbation_target"].shape == (adata.n_obs, 2)
    assert adata.uns["perturbation_target_names"] == ["TP53", "BRCA1"]
    np.testing.assert_allclose(adata.obsm["perturbation_target"][:2], 0.0)


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


def test_advanced_train_supports_custom_perturbation_key_and_control(tmp_path):
    adata = make_adata()
    adata.obs["condition"] = adata.obs["perturbation"]
    adata.obs = adata.obs.drop(columns=["perturbation"])
    adata = prepare_perturbation_data(adata, ctrl_key="condition", ctrl_value="non-targeting")
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
        perturbation_key="condition",
        control_value="non-targeting",
    )

    assert "loss" in history.history


def test_preprocessed_fast_path_rebuilds_adjacency(tmp_path):
    adata = make_adata()
    preprocessed_path = tmp_path / "preprocessed.h5ad"
    network_path = tmp_path / "network.txt"

    adata.write_h5ad(preprocessed_path)
    network_path.write_text(
        "protein1 protein2 combined_score\nTP53 BRCA1 900\n",
        encoding="utf-8",
    )

    loaded_adata, adj_matrix = load_and_preprocess_perturbation_data(
        rna_path=None,
        network_path=str(network_path),
        preprocessed_path=str(preprocessed_path),
        ctrl_key="perturbation",
        ctrl_value="non-targeting",
    )

    assert loaded_adata.n_vars == adata.n_vars
    assert adj_matrix is not None
    assert adj_matrix.shape == (adata.n_vars, adata.n_vars)
    assert adj_matrix.nnz == 2


def test_create_adjacency_matrix_requires_edge_overlap():
    network_df = pd.DataFrame(
        {
            "protein1": ["NOT_A_GENE"],
            "protein2": ["STILL_NOT_A_GENE"],
            "combined_score": [900],
        }
    )

    with pytest.raises(ValueError, match="No overlapping network edges"):
        create_adjacency_matrix(network_df, ["TP53", "BRCA1"])


def test_load_protein_network_auto_detects_mapping_file(tmp_path):
    network_path = tmp_path / "protein.links.v12.0.txt"
    mapping_path = tmp_path / "protein.info.v12.0.txt"

    network_path.write_text(
        "protein1 protein2 combined_score\n9606.ENSP1 9606.ENSP2 900\n",
        encoding="utf-8",
    )
    mapping_path.write_text(
        "#string_protein_id\tpreferred_name\n9606.ENSP1\tTP53\n9606.ENSP2\tBRCA1\n",
        encoding="utf-8",
    )

    network_df = load_protein_network(str(network_path), confidence=700)

    assert list(network_df["protein1_gene"]) == ["TP53"]
    assert list(network_df["protein2_gene"]) == ["BRCA1"]


def test_interpret_perturbations_reports_response_genes():
    inference = HyperPerturbInference.__new__(HyperPerturbInference)
    inference.config = {"gene_names": ["TP53", "BRCA1", "EGFR"]}

    interpreted = inference.interpret_perturbations(
        make_adata(),
        np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
    )

    assert list(interpreted["response_gene"]) == ["TP53", "BRCA1", "EGFR"]
    assert interpreted.loc[0, "perturbation_genes"] == ["BRCA1", "EGFR"]
