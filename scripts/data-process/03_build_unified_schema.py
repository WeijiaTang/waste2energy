from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
UNIFIED_DIR = ROOT / "data" / "processed" / "unified_features"
MODEL_READY_DIR = ROOT / "data" / "processed" / "model_ready"
SCENARIO_DIR = ROOT / "data" / "processed" / "scenario_inputs"


FEATURE_COLUMNS = [
    "pathway",
    "feedstock_group",
    "feedstock_carbon_pct",
    "feedstock_hydrogen_pct",
    "feedstock_nitrogen_pct",
    "feedstock_oxygen_pct",
    "feedstock_moisture_pct",
    "feedstock_volatile_matter_pct",
    "feedstock_fixed_carbon_pct",
    "feedstock_ash_pct",
    "feedstock_hhv_mj_per_kg",
    "process_temperature_c",
    "residence_time_min",
    "heating_rate_c_per_min",
    "blend_manure_ratio",
    "blend_wet_waste_ratio",
]


TARGET_COLUMNS = [
    "product_char_yield_pct",
    "product_char_hhv_mj_per_kg",
    "energy_recovery_pct",
    "carbon_retention_pct",
]


METADATA_COLUMNS = [
    "sample_id",
    "source_repo",
    "source_file",
    "source_dataset_kind",
    "feedstock_name",
    "blending_case",
    "reference_label",
]


VARIABLE_DICTIONARY = [
    ("sample_id", "metadata", "Unique sample identifier", ""),
    ("source_repo", "metadata", "Source repository name", ""),
    ("source_file", "metadata", "Source file name", ""),
    ("source_dataset_kind", "metadata", "Source dataset type", ""),
    ("feedstock_name", "metadata", "Original feedstock name when available", ""),
    ("feedstock_group", "feature", "Canonical feedstock grouping", ""),
    ("pathway", "feature", "Canonical conversion pathway label", ""),
    ("blending_case", "metadata", "Blend-design label for current sample", ""),
    ("blend_manure_ratio", "feature", "Manure fraction in blended feed", "ratio"),
    ("blend_wet_waste_ratio", "feature", "Wet-waste fraction in blended feed", "ratio"),
    ("feedstock_carbon_pct", "feature", "Feedstock carbon content", "pct"),
    ("feedstock_hydrogen_pct", "feature", "Feedstock hydrogen content", "pct"),
    ("feedstock_nitrogen_pct", "feature", "Feedstock nitrogen content", "pct"),
    ("feedstock_oxygen_pct", "feature", "Feedstock oxygen content", "pct"),
    ("feedstock_moisture_pct", "feature", "Feedstock moisture content", "pct"),
    ("feedstock_volatile_matter_pct", "feature", "Feedstock volatile matter", "pct"),
    ("feedstock_fixed_carbon_pct", "feature", "Feedstock fixed carbon", "pct"),
    ("feedstock_ash_pct", "feature", "Feedstock ash content", "pct"),
    ("feedstock_hhv_mj_per_kg", "feature", "Feedstock higher heating value", "MJ/kg"),
    ("process_temperature_c", "feature", "Processing temperature", "C"),
    ("residence_time_min", "feature", "Residence time", "min"),
    ("heating_rate_c_per_min", "feature", "Heating rate", "C/min"),
    ("product_char_yield_pct", "target", "Char yield", "pct"),
    ("product_char_hhv_mj_per_kg", "target", "Char higher heating value", "MJ/kg"),
    ("energy_recovery_pct", "target", "Energy recovery", "pct"),
    ("carbon_retention_pct", "target", "Carbon retention", "pct"),
    ("reference_label", "metadata", "Original reference label", ""),
]


def build_variable_dictionary() -> pd.DataFrame:
    rows = []
    for column_name, column_role, description, units in VARIABLE_DICTIONARY:
        rows.append(
            {
                "column_name": column_name,
                "column_role": column_role,
                "description": description,
                "units": units,
                "paper1_include": "yes",
            }
        )
    return pd.DataFrame(rows)


def build_scenario_template() -> pd.DataFrame:
    rows = [
        {
            "scenario_group": "feedstock_composition",
            "scenario_name": "manure_dominant",
            "blend_manure_ratio": 0.7,
            "blend_wet_waste_ratio": 0.3,
            "energy_price_multiplier": 1.0,
            "policy_multiplier": 1.0,
            "technology_efficiency_multiplier": 1.0,
        },
        {
            "scenario_group": "feedstock_composition",
            "scenario_name": "balanced_mixed_feed",
            "blend_manure_ratio": 0.5,
            "blend_wet_waste_ratio": 0.5,
            "energy_price_multiplier": 1.0,
            "policy_multiplier": 1.0,
            "technology_efficiency_multiplier": 1.0,
        },
        {
            "scenario_group": "feedstock_composition",
            "scenario_name": "wet_waste_enhanced",
            "blend_manure_ratio": 0.3,
            "blend_wet_waste_ratio": 0.7,
            "energy_price_multiplier": 1.0,
            "policy_multiplier": 1.0,
            "technology_efficiency_multiplier": 1.0,
        },
        {
            "scenario_group": "market_condition",
            "scenario_name": "high_energy_price",
            "blend_manure_ratio": None,
            "blend_wet_waste_ratio": None,
            "energy_price_multiplier": 1.2,
            "policy_multiplier": 1.0,
            "technology_efficiency_multiplier": 1.0,
        },
        {
            "scenario_group": "policy_environment",
            "scenario_name": "policy_support",
            "blend_manure_ratio": None,
            "blend_wet_waste_ratio": None,
            "energy_price_multiplier": 1.0,
            "policy_multiplier": 1.2,
            "technology_efficiency_multiplier": 1.0,
        },
        {
            "scenario_group": "technology_condition",
            "scenario_name": "optimistic_efficiency",
            "blend_manure_ratio": None,
            "blend_wet_waste_ratio": None,
            "energy_price_multiplier": 1.0,
            "policy_multiplier": 1.0,
            "technology_efficiency_multiplier": 1.1,
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

    combined = pd.read_csv(UNIFIED_DIR / "wet_waste_biomass_opt_combined_standardized.csv")

    model_ready = combined[METADATA_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS].copy()
    model_ready["blend_manure_ratio"] = model_ready["blend_manure_ratio"].astype("float64")
    model_ready["blend_wet_waste_ratio"] = model_ready["blend_wet_waste_ratio"].astype("float64")

    variable_dictionary = build_variable_dictionary()
    scenario_template = build_scenario_template()

    model_ready_out = MODEL_READY_DIR / "ml_training_dataset.csv"
    schema_out = MODEL_READY_DIR / "ml_training_dataset_schema.json"
    variable_dictionary_out = UNIFIED_DIR / "source_variable_dictionary.csv"
    scenario_template_out = SCENARIO_DIR / "paper1_scenario_template.csv"

    model_ready.to_csv(model_ready_out, index=False)
    variable_dictionary.to_csv(variable_dictionary_out, index=False)
    scenario_template.to_csv(scenario_template_out, index=False)

    schema_payload = {
        "dataset_name": "ml_training_dataset",
        "source_file": "wet_waste_biomass_opt_combined_standardized.csv",
        "row_count": int(len(model_ready)),
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "metadata_columns": METADATA_COLUMNS,
        "notes": [
            "This is the first model-ready text dataset for Waste2Energy Paper 1.",
            "Rows are currently dominated by repository-derived wet-waste and pyrolysis literature samples.",
            "Mixed-feed ratios remain empty until region-specific blending assumptions are introduced.",
        ],
    }
    schema_out.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")

    print(f"Wrote {model_ready_out}")
    print(f"Wrote {variable_dictionary_out}")
    print(f"Wrote {scenario_template_out}")
    print(f"Wrote {schema_out}")


if __name__ == "__main__":
    main()
