"""Dataset specs, loading, and split logic for Waste2Energy."""

from .loaders import DatasetBundle, frame_to_xy, load_dataset_bundle
from .specs import (
    DATASET_KEYS,
    DATASET_SPECS,
    DEFAULT_EXCLUDED_COLUMNS,
    TARGET_COLUMNS,
    DatasetSpec,
    get_dataset_spec,
)
from .contracts import (
    DataContractSummary,
    annotate_data_contract,
    infer_row_provenance,
    read_and_summarize_planning_contract,
    summarize_data_contract,
    validate_planning_input_contract,
)

__all__ = [
    "DATASET_KEYS",
    "DATASET_SPECS",
    "DEFAULT_EXCLUDED_COLUMNS",
    "TARGET_COLUMNS",
    "DataContractSummary",
    "DatasetBundle",
    "DatasetSpec",
    "annotate_data_contract",
    "frame_to_xy",
    "get_dataset_spec",
    "infer_row_provenance",
    "load_dataset_bundle",
    "read_and_summarize_planning_contract",
    "summarize_data_contract",
    "validate_planning_input_contract",
]
