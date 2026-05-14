from __future__ import annotations

from waste2energy.models import MODEL_KEYS, get_model_ops


def test_catboost_is_registered_with_expected_artifact_metadata():
    assert "catboost" in MODEL_KEYS

    ops = get_model_ops("catboost")

    assert ops["model_key"] == "catboost"
    assert ops["model_family"] == "catboost_regressor"
    assert ops["model_file_name"] == "model.cbm"
    assert callable(ops["train_model"])
    assert callable(ops["build_feature_importance"])
    assert callable(ops["save_model"])
