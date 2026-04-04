# Data Processing Plan

This folder contains all scripts that transform source-preserved raw data into reproducible processed datasets for Waste2Energy Paper 1.

## Paper 1 Processing Goal

Convert mixed-source raw data into a unified dataset for:

- surrogate modeling
- scenario generation
- optimization inputs
- figure and table reproduction

## Project-Specific Processing Logic

Waste2Energy Paper 1 focuses on a mixed waste system:

- dominant livestock manure
- secondary food or municipal wet waste

The processing pipeline therefore needs to do more than simple cleaning. It must also create mixed-feed descriptors that match the paper's scientific identity.

## Planned Raw-to-Processed Flow

### Stage 1: Ingest repository data

Inputs:

- `data/raw/Wet-Waste-Biomass-Opt/*.csv`
- `data/raw/ManurePyrolysisIAM/pyrolysis_energy_balance.xlsx`
- `data/raw/ManurePyrolysisIAM/baseline_supplementary_tables/*.csv`
- `data/raw/GCAM-CDR-policy/policy_reference_45Q-2040/**/*.csv`

Outputs:

- standardized source-level tables under `data/processed/unified_features`

### Stage 2: Schema harmonization

Tasks:

- standardize column names
- standardize units
- map pathway names to canonical labels
- separate pathway-performance fields from scenario-reference fields
- tag each row with a `source_repo`

Suggested canonical pathway labels:

- `baseline`
- `ad`
- `pyrolysis`
- `htc`

### Stage 3: Mixed-waste feature construction

This is the most project-specific processing step.

For each blended feed or pseudo-blended scenario, derive features such as:

- `manure_ratio`
- `wet_waste_ratio`
- `moisture_blend`
- `ash_blend`
- `volatile_matter_blend`
- `carbon_blend`
- `nitrogen_blend`
- `cn_ratio_blend`
- `hhv_blend`

If direct mixed-feed measurements are unavailable, use controlled blending rules with explicit assumptions.

### Stage 4: Target-variable alignment

Create model targets such as:

- product yield
- energy recovery
- carbon-emission indicator
- cost indicator

Each target should include:

- source
- unit
- transformation notes
- whether it is directly observed or derived

### Stage 5: Scenario-input extraction

Use policy and cost reference files to build:

- energy-price scenarios
- policy-support scenarios
- environmental-penalty scenarios
- technology-performance sensitivity scenarios

Outputs go to:

- `data/processed/scenario_inputs`

### Stage 6: Model-ready export

Create final datasets for:

- machine learning
- optimization
- figure generation

Outputs go to:

- `data/processed/model_ready`
- `data/processed/figures_tables`

## Recommended Processed Files

### Unified features

- `data/processed/unified_features/pathway_samples.csv`
- `data/processed/unified_features/source_variable_dictionary.csv`

### Scenario inputs

- `data/processed/scenario_inputs/policy_scenarios.csv`
- `data/processed/scenario_inputs/market_scenarios.csv`
- `data/processed/scenario_inputs/technology_sensitivity.csv`

### Model-ready

- `data/processed/model_ready/ml_training_dataset.csv`
- `data/processed/model_ready/ml_training_dataset_htc_direct.csv`
- `data/processed/model_ready/ml_training_matrix_htc_direct.csv`
- `data/processed/model_ready/ml_training_dataset_pyrolysis_direct.csv`
- `data/processed/model_ready/ml_training_matrix_pyrolysis_direct.csv`
- `data/processed/model_ready/paper1_ml_dataset_htc_scope.csv`
- `data/processed/model_ready/paper1_ml_matrix_htc_scope.csv`
- `data/processed/model_ready/optimization_input_dataset.csv`
- `data/processed/model_ready/blending_assumptions.csv`

## Current Direct-ML Exports

The legacy `ml_training_dataset.csv` keeps all observed rows in one unified table, but it is not the best direct training input because `htc` and `pyrolysis` do not share the same measured feature set.

Use the split exports below for direct machine-learning work:

- `python scripts/data-process/10_build_ml_ready_datasets.py`

This script writes:

- `ml_training_dataset_htc_direct.csv`: observed HTC rows only, with HTC-valid features and targets
- `ml_training_matrix_htc_direct.csv`: one-hot encoded HTC training matrix
- `ml_training_dataset_pyrolysis_direct.csv`: observed pyrolysis rows only, with pyrolysis-valid features and targets
- `ml_training_matrix_pyrolysis_direct.csv`: pyrolysis training matrix
- `paper1_ml_dataset_htc_scope.csv`: Paper 1 HTC scope combining observed manure and food-waste rows with synthetic mixed-feed reference rows
- `paper1_ml_matrix_htc_scope.csv`: matrix export for Paper 1 HTC scope
- `ml_ready_dataset_manifest.json`: export manifest with row counts, feature lists, and usage notes

Recommended usage:

- use `*_htc_direct*` for direct HTC supervised learning
- use `*_pyrolysis_direct*` for direct pyrolysis supervised learning
- use `paper1_ml_*` for Waste2Energy Paper 1 benchmarking, augmentation, and scenario screening
- do not treat the synthetic rows in `paper1_ml_dataset_htc_scope.csv` as blind test data

## Initial Script Ideas

- `scripts/data-process/01_ingest_wet_waste_biomass_opt.py`
- `scripts/data-process/02_ingest_manure_pyrolysis_iam.py`
- `scripts/data-process/03_ingest_gcam_policy_reference.py`
- `scripts/data-process/04_build_mixed_waste_features.py`
- `scripts/data-process/05_build_model_ready_tables.py`

## Guardrails

- no hand-edited processed files
- every derived field must be traceable
- keep source repository provenance attached to each transformed table
- keep GCAM policy data as scenario-reference input, not as the main ML training body

## Immediate Next Processing Target

The first processing milestone should be:

1. ingest Wet-Waste-Biomass-Opt CSV files
2. ingest manure pyrolysis cost and balance tables
3. define a canonical variable schema
4. create the first `ml_training_dataset.csv`
