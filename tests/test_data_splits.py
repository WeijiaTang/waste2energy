from __future__ import annotations

from waste2energy.data.loaders import load_dataset_bundle


def test_paper1_htc_scope_recommended_keeps_augmentation_rows_in_train():
    bundle = load_dataset_bundle("paper1_htc_scope", split_strategy="recommended")

    train_split_values = set(bundle.split_frames["train"]["recommended_split"].astype(str).tolist())

    assert "augmentation" in train_split_values


def test_paper1_htc_scope_strict_group_excludes_augmentation_rows_from_train():
    bundle = load_dataset_bundle("paper1_htc_scope", split_strategy="strict_group")

    train_split_values = set(bundle.split_frames["train"]["recommended_split"].astype(str).tolist())

    assert "augmentation" not in train_split_values


def test_paper1_htc_scope_leave_study_out_excludes_augmentation_rows_from_train():
    bundle = load_dataset_bundle("paper1_htc_scope", split_strategy="leave_study_out")

    train_split_values = set(bundle.split_frames["train"]["recommended_split"].astype(str).tolist())

    assert "augmentation" not in train_split_values
