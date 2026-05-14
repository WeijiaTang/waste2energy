from __future__ import annotations

from waste2energy.models import MODEL_KEYS, get_model_ops


def test_lightgbm_is_registered_with_expected_artifact_metadata():
    assert "lightgbm" in MODEL_KEYS

    ops = get_model_ops("lightgbm")

    assert ops["model_key"] == "lightgbm"
    assert ops["model_family"] == "lightgbm_regressor"
    assert ops["model_file_name"] == "model.txt"
    assert callable(ops["train_model"])
    assert callable(ops["build_feature_importance"])
    assert callable(ops["save_model"])
