"""Waste2Energy machine-learning utilities."""

from .runtime import apply_runtime_thread_defaults

apply_runtime_thread_defaults()

from .data import DATASET_KEYS, TARGET_COLUMNS

__all__ = ["DATASET_KEYS", "TARGET_COLUMNS"]
