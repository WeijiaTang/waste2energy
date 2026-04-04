# Raw Data Inventory

This folder stores source-preserved raw data for the Waste2Energy project.

## Principles

- Keep repository-derived raw data as close to original as possible.
- Do not manually edit files in `data/raw`.
- All cleaning, renaming, unit conversion, and feature generation should happen in `scripts/data-process`.
- Only processed outputs should flow into `data/processed`.

## Copied Data Sources

### 1. Wet-Waste-Biomass-Opt

**Source repository**

- https://github.com/PEESEgroup/Wet-Waste-Biomass-Opt

**Related paper**

- https://doi.org/10.1016/j.jclepro.2023.138606

**Copied files**

- `data/raw/Wet-Waste-Biomass-Opt/HTC_data.csv`
- `data/raw/Wet-Waste-Biomass-Opt/HTC_data.xlsx`
- `data/raw/Wet-Waste-Biomass-Opt/Pyrolysis_data.csv`
- `data/raw/Wet-Waste-Biomass-Opt/Pyrolysis_data.xlsx`
- `data/raw/Wet-Waste-Biomass-Opt/SI_Dataset(1).docx`
- `data/raw/Wet-Waste-Biomass-Opt/README.md`

**Role in Waste2Energy**

- core pathway-performance training data for HTC and pyrolysis
- source of process conditions, product metrics, and supplementary references
- template for JOCP-style data organization

### 2. ManurePyrolysisIAM

**Source repository**

- https://github.com/PEESEgroup/ManurePyrolysisIAM

**Copied files**

- `data/raw/ManurePyrolysisIAM/README.md`
- `data/raw/ManurePyrolysisIAM/pyrolysis_energy_balance.xlsx`
- `data/raw/ManurePyrolysisIAM/baseline_supplementary_tables/*.csv`

**Role in Waste2Energy**

- manure-focused pyrolysis cost, balance, and GHG reference data
- baseline scenario support for environmental and economic assumptions
- useful manure-side complements to the wet-waste pathway data

**Note**

The repository contains much larger GCAM result structures. For Paper 1, only the baseline supplementary tables and pyrolysis energy-balance workbook were copied because they are the most directly reusable raw references.

### 3. GCAM-CDR-policy

**Source repository**

- https://github.com/PEESEgroup/GCAM-CDR-policy

**Copied files**

- `data/raw/GCAM-CDR-policy/README.md`
- `data/raw/GCAM-CDR-policy/policy_reference_45Q-2040/high/*.csv`
- `data/raw/GCAM-CDR-policy/policy_reference_45Q-2040/low/*.csv`

**Role in Waste2Energy**

- policy-scenario inspiration only
- reference tables for cost, subsidy, market-share, and policy-spend framing
- not a direct training-data source for Paper 1

**Note**

This subset was copied to support scenario design, not to pull Waste2Energy into full GCAM-scale modeling.

### 4. External Region Data

**Folder**

- `data/raw/external-region-data`

**Status**

No external regional raw files have been copied yet.

**Expected future sources**

- livestock and agricultural statistics
- municipal food or wet-waste generation statistics
- regional energy prices
- regional emission factors
- policy and subsidy reference data

These should be added later with source URLs recorded in this file or in a companion manifest.

## Current Data Position

The raw layer now contains:

- technology-performance data from wet-waste conversion studies
- manure-pyrolysis economic and environmental reference tables
- policy-scenario reference tables for later scenario design

This is enough to start designing the first Waste2Energy processing scripts and unified feature schema.
