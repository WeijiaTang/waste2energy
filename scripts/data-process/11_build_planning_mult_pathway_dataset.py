from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
UNIFIED_DIR = ROOT / "data" / "processed" / "unified_features"
SCENARIO_DIR = ROOT / "data" / "processed" / "scenario_inputs"
MODEL_READY_DIR = ROOT / "data" / "processed" / "model_ready"
RAW_MANURE_DIR = ROOT / "data" / "raw" / "ManurePyrolysisIAM" / "baseline_supplementary_tables"

COMBINED_INPUT = UNIFIED_DIR / "wet_waste_biomass_opt_combined_standardized.csv"
MANURE_REFERENCE_INPUT = UNIFIED_DIR / "manure_pyrolysis_energy_balance_long.csv"
CALIFORNIA_MODEL_INPUT = MODEL_READY_DIR / "california_food_waste_model_input.csv"
REGION_SCENARIO_INPUT = SCENARIO_DIR / "paper1_region_scenario_placeholder.csv"
TREATMENT_MIX_INPUT = SCENARIO_DIR / "california_food_waste_treatment_mix_reference.csv"
WARM_REFERENCE_INPUT = (
    ROOT
    / "data"
    / "raw"
    / "external-region-data"
    / "california"
    / "emission_factors"
    / "california_waste_treatment_emission_factor_reference.csv"
)

PROTOTYPE_OUTPUT = UNIFIED_DIR / "paper1_planning_pathway_prototypes.csv"
OPTIMIZATION_OUTPUT = MODEL_READY_DIR / "optimization_input_dataset.csv"
ASSUMPTIONS_OUTPUT = MODEL_READY_DIR / "optimization_input_dataset_assumptions.json"
LEGACY_ASSUMPTIONS_OUTPUT = MODEL_READY_DIR / "mixed_waste_feature_assumptions.json"
READINESS_OUTPUT = MODEL_READY_DIR / "optimization_pathway_readiness_summary.csv"

TARGET_REGION_ID = "us_ca"

BLEND_CASES = [
    ("manure_dominant", 0.7, 0.3),
    ("balanced_mixed_feed", 0.5, 0.5),
    ("wet_waste_enhanced", 0.3, 0.7),
]

FEEDSTOCK_COLUMNS = [
    "feedstock_carbon_pct",
    "feedstock_hydrogen_pct",
    "feedstock_nitrogen_pct",
    "feedstock_oxygen_pct",
    "feedstock_moisture_pct",
    "feedstock_volatile_matter_pct",
    "feedstock_fixed_carbon_pct",
    "feedstock_ash_pct",
    "feedstock_hhv_mj_per_kg",
]

TARGET_COLUMNS = [
    "product_char_yield_pct",
    "product_char_hhv_mj_per_kg",
    "energy_recovery_pct",
    "carbon_retention_pct",
]

HTC_COST_PROXY_BASE = 0.95
PYROLYSIS_COST_PROXY_BASE = 1.05
AD_COST_PROXY_BASE = 0.75
BASELINE_COST_PROXY_BASE = 0.35

AD_WET_ELECTRICITY_KWH_PER_TON = 201.4
AD_WET_AVAILABLE_HEAT_MMBTU_PER_TON = 1.26
AD_WET_REACTOR_HEAT_MMBTU_PER_TON = 0.14
MMBTU_TO_MJ = 1055.056
KWH_TO_MJ = 3.6
LANDFILL_AVOIDED_UTILITY_EMISSIONS_KGCO2E_PER_SHORT_TON = 60.0

PATHWAY_READINESS_RULES = {
    "baseline": {
        "process_basis": "regional_weighted_management_mix_proxy",
        "performance_basis": "california_weighted_energy_proxy_from_landfill_lfg_and_ad_share",
        "environment_basis": "zero_improvement_anchor_vs_current_california_weighted_mix",
        "cost_basis": "relative_pathway_proxy_index_only",
        "claim_boundary": (
            "Baseline rows are planning anchors for comparison and optimization accounting, not "
            "generalizable process-performance evidence."
        ),
    },
    "ad": {
        "process_basis": "food_waste_ad_summary_proxy_with_canonical_operating_placeholders",
        "performance_basis": "explicit_energy_proxy_from_warm_wet_digestion_exhibit",
        "environment_basis": "explicit_food_waste_ad_emission_factor_from_epa_warm",
        "cost_basis": "relative_pathway_proxy_index_only",
        "claim_boundary": (
            "AD rows support regional planning comparison but should be written as pathway proxies rather "
            "than as calibrated facility-specific AD optimization."
        ),
    },
    "pyrolysis": {
        "process_basis": "representative_conditions_from_observed_pyrolysis_rows",
        "performance_basis": "synthetic_mixed_feed_rows_with_observed_pyrolysis_condition_anchor",
        "environment_basis": "carbon_retention_based_avoidance_proxy",
        "cost_basis": "relative_pathway_proxy_index_only_with_future_pyrolysis_cost_reference_columns",
        "claim_boundary": (
            "Pyrolysis rows are optimization-ready mixed-feed candidates, but blended mixed-feed behavior is "
            "still partly proxy-built rather than directly observed."
        ),
    },
    "htc": {
        "process_basis": "representative_conditions_from_observed_htc_rows",
        "performance_basis": "synthetic_mixed_feed_rows_weighted_from_observed_htc_food_and_manure_profiles",
        "environment_basis": "carbon_retention_based_avoidance_proxy",
        "cost_basis": "relative_pathway_proxy_index_only",
        "claim_boundary": (
            "HTC rows are the strongest current mixed-feed planning candidates in the repository, but they "
            "still do not justify strong cross-study generalization claims."
        ),
    },
}


def load_combined_observed() -> pd.DataFrame:
    return pd.read_csv(COMBINED_INPUT)


def load_california_food_waste_reference() -> dict[str, object]:
    frame = pd.read_csv(CALIFORNIA_MODEL_INPUT)
    if len(frame) != 1:
        raise RuntimeError(
            f"Expected exactly one California food-waste model-input row in {CALIFORNIA_MODEL_INPUT}"
        )
    row = frame.iloc[0].to_dict()
    if row["region_id"] != TARGET_REGION_ID:
        raise RuntimeError(
            f"Expected California model input region_id {TARGET_REGION_ID}, found {row['region_id']}"
        )
    return row


def load_region_scenarios() -> pd.DataFrame:
    frame = pd.read_csv(REGION_SCENARIO_INPUT)
    california = frame[frame["region_id"] == TARGET_REGION_ID].copy()
    if california.empty:
        raise RuntimeError(f"No {TARGET_REGION_ID} rows found in {REGION_SCENARIO_INPUT}")
    return california.reset_index(drop=True)


def load_treatment_mix_reference() -> pd.DataFrame:
    frame = pd.read_csv(TREATMENT_MIX_INPUT)
    california = frame[frame["region_id"] == TARGET_REGION_ID].copy()
    if california.empty:
        raise RuntimeError(f"No {TARGET_REGION_ID} rows found in {TREATMENT_MIX_INPUT}")
    return california.reset_index(drop=True)


def load_warm_reference() -> pd.DataFrame:
    frame = pd.read_csv(WARM_REFERENCE_INPUT)
    california = frame[frame["region_id"] == TARGET_REGION_ID].copy()
    if california.empty:
        raise RuntimeError(f"No {TARGET_REGION_ID} rows found in {WARM_REFERENCE_INPUT}")
    return california.reset_index(drop=True)


def load_manure_subtype_reference() -> pd.DataFrame:
    energy_balance = pd.read_csv(MANURE_REFERENCE_INPUT)
    keys = [
        "Initial Moisture Content",
        "dry manure calorific value",
        "biochar yield",
        "biochar calorific value",
    ]
    subset = energy_balance[energy_balance["constant_name"].isin(keys)].copy()
    pivot = subset.pivot_table(
        index="livestock_type",
        columns="constant_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(
        columns={
            "Initial Moisture Content": "subtype_moisture_ratio",
            "dry manure calorific value": "subtype_feedstock_hhv_mj_per_kg",
            "biochar yield": "subtype_char_yield_ratio",
            "biochar calorific value": "subtype_char_hhv_mj_per_kg",
        }
    )
    return pivot


def load_pyrolysis_cost_reference() -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {}
    unit_cost = pd.read_csv(RAW_MANURE_DIR / "unit_cost_pyrolysis.csv")
    total_cost = pd.read_csv(RAW_MANURE_DIR / "total_cost_pyrolysis.csv")
    feedstock_cost = pd.read_csv(RAW_MANURE_DIR / "feedstock_cost_pyrolysis.csv")

    subtype_aliases = {
        "beef": ("beef",),
        "dairy": ("dairy",),
        "goat": ("goat",),
        "poultry": ("poultry",),
        "swine": ("pork", "swine"),
    }

    for subtype, aliases in subtype_aliases.items():
        unit_mask = (
            unit_cost["Version"].astype(str).eq("Baseline")
            & unit_cost["GCAM"].astype(str).eq("USA")
            & unit_cost["output"].astype(str).str.lower().apply(
                lambda text: any(alias in text for alias in aliases)
            )
        )
        total_mask = (
            total_cost["Version"].astype(str).eq("Median")
            & total_cost["GCAM"].astype(str).eq("USA")
            & total_cost["output"].astype(str).str.lower().apply(
                lambda text: any(alias in text for alias in aliases)
            )
        )
        feedstock_mask = (
            feedstock_cost["Version"].astype(str).eq("Median")
            & feedstock_cost["GCAM"].astype(str).eq("USA")
            & feedstock_cost["product"].astype(str).str.lower().apply(
                lambda text: any(alias in text for alias in aliases)
            )
        )

        unit_row = unit_cost.loc[unit_mask].copy()
        total_row = total_cost.loc[total_mask].copy()
        feedstock_row = feedstock_cost.loc[feedstock_mask].copy()

        lookup[subtype] = {
            "unit_cost_usd_per_ton_biochar_reference_2040": _pick_numeric(unit_row, "2040"),
            "total_cost_usd_per_ton_biochar_reference_2040": _pick_numeric(total_row, "2040"),
            "feedstock_cost_usd_per_ton_manure_reference_2040": _pick_numeric(feedstock_row, "2040"),
        }

    return lookup


def _pick_numeric(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return float("nan")
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.iloc[0])


def representative_conditions(
    frame: pd.DataFrame,
    *,
    include_heating_rate: bool,
    max_conditions: int = 4,
    defaults: list[dict[str, float]] | None = None,
) -> pd.DataFrame:
    keys = ["process_temperature_c", "residence_time_min"]
    if include_heating_rate:
        keys.append("heating_rate_c_per_min")

    available = frame.dropna(subset=["process_temperature_c", "residence_time_min"]).copy()
    if include_heating_rate:
        available = available.dropna(subset=["heating_rate_c_per_min"]).copy()

    counts = (
        available.groupby(keys, dropna=True)
        .size()
        .reset_index(name="count")
        .sort_values(["count"] + keys, ascending=[False] + [True] * len(keys))
    )
    selected = counts.head(max_conditions).copy()
    if selected.empty:
        selected = pd.DataFrame(defaults or [])
    if "heating_rate_c_per_min" not in selected.columns:
        selected["heating_rate_c_per_min"] = pd.NA
    return selected.reset_index(drop=True)


def weighted_value(manure_value: float, wet_value: float, manure_ratio: float, wet_ratio: float) -> float:
    return manure_value * manure_ratio + wet_value * wet_ratio


def build_subtype_profiles(
    *,
    htc_manure_medians: pd.Series,
    manure_subtypes: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    profiles: dict[str, dict[str, object]] = {}
    for _, subtype in manure_subtypes.iterrows():
        subtype_name = str(subtype["livestock_type"])
        feedstock_profile = htc_manure_medians[FEEDSTOCK_COLUMNS].copy()
        feedstock_profile["feedstock_moisture_pct"] = float(subtype["subtype_moisture_ratio"]) * 100.0
        feedstock_profile["feedstock_hhv_mj_per_kg"] = float(subtype["subtype_feedstock_hhv_mj_per_kg"])

        char_yield_pct = float(subtype["subtype_char_yield_ratio"]) * 100.0
        char_hhv = float(subtype["subtype_char_hhv_mj_per_kg"])
        energy_recovery_pct = (
            float(subtype["subtype_char_yield_ratio"])
            * float(subtype["subtype_char_hhv_mj_per_kg"])
            / float(subtype["subtype_feedstock_hhv_mj_per_kg"])
            * 100.0
        )
        profiles[subtype_name] = {
            "feedstock_profile": {column: float(feedstock_profile[column]) for column in FEEDSTOCK_COLUMNS},
            "char_yield_pct": char_yield_pct,
            "char_hhv_mj_per_kg": char_hhv,
            "energy_recovery_pct": energy_recovery_pct,
        }
    return profiles


def build_common_feed_rows(
    *,
    food_feedstock_medians: pd.Series,
    subtype_profiles: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subtype_name, subtype_profile in subtype_profiles.items():
        manure_feedstock = subtype_profile["feedstock_profile"]
        for blend_name, manure_ratio, wet_ratio in BLEND_CASES:
            row: dict[str, object] = {
                "feedstock_name": f"{subtype_name}_manure_plus_food_waste",
                "blending_case": blend_name,
                "pathway_feedstock_scope": "mixed_manure_food_waste",
                "feedstock_group": "mixed_manure_wet_waste",
                "manure_subtype": subtype_name,
                "wet_waste_reference_group": "food_waste",
                "blend_manure_ratio": manure_ratio,
                "blend_wet_waste_ratio": wet_ratio,
            }
            for column in FEEDSTOCK_COLUMNS:
                row[column] = weighted_value(
                    float(manure_feedstock[column]),
                    float(food_feedstock_medians[column]),
                    manure_ratio,
                    wet_ratio,
                )
            rows.append(row)
    return rows


def build_htc_candidates(
    *,
    common_feed_rows: list[dict[str, object]],
    subtype_profiles: dict[str, dict[str, object]],
    htc_food_targets: pd.Series,
    htc_manure_targets: pd.Series,
    htc_conditions: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    row_id = 1
    for base in common_feed_rows:
        subtype_profile = subtype_profiles[str(base["manure_subtype"])]
        manure_targets = htc_manure_targets.copy()
        manure_targets["product_char_yield_pct"] = float(subtype_profile["char_yield_pct"])
        manure_targets["product_char_hhv_mj_per_kg"] = float(subtype_profile["char_hhv_mj_per_kg"])
        manure_targets["energy_recovery_pct"] = float(subtype_profile["energy_recovery_pct"])

        for _, condition in htc_conditions.iterrows():
            row = base.copy()
            row.update(
                {
                    "sample_id": f"Waste2Energy::planning::htc::{row_id:04d}",
                    "source_repo": "Waste2EnergyDerived",
                    "source_file": "11_build_planning_mult_pathway_dataset.py",
                    "source_dataset_kind": "synthetic_mixed_feed_htc_candidate",
                    "reference_label": "weighted_blend_from_htc_food_manure_references",
                    "pathway": "htc",
                    "pathway_variant": "mixed_feed_htc_condition_anchor",
                    "pathway_process_basis": PATHWAY_READINESS_RULES["htc"]["process_basis"],
                    "pathway_performance_basis": PATHWAY_READINESS_RULES["htc"]["performance_basis"],
                    "pathway_environment_basis": PATHWAY_READINESS_RULES["htc"]["environment_basis"],
                    "pathway_cost_basis": PATHWAY_READINESS_RULES["htc"]["cost_basis"],
                    "pathway_claim_boundary": PATHWAY_READINESS_RULES["htc"]["claim_boundary"],
                    "pathway_cost_proxy_base_factor": HTC_COST_PROXY_BASE,
                    "process_temperature_c": float(condition["process_temperature_c"]),
                    "residence_time_min": float(condition["residence_time_min"]),
                    "heating_rate_c_per_min": pd.NA,
                }
            )
            for column in TARGET_COLUMNS:
                row[column] = weighted_value(
                    float(manure_targets[column]),
                    float(htc_food_targets[column]),
                    float(base["blend_manure_ratio"]),
                    float(base["blend_wet_waste_ratio"]),
                )
            row["pathway_energy_intensity_mj_per_ton"] = (
                float(row["product_char_yield_pct"]) / 100.0 * float(row["product_char_hhv_mj_per_kg"]) * 1000.0
            )
            rows.append(row)
            row_id += 1
    return rows


def build_pyrolysis_candidates(
    *,
    common_feed_rows: list[dict[str, object]],
    subtype_profiles: dict[str, dict[str, object]],
    pyrolysis_targets: pd.Series,
    pyrolysis_conditions: pd.DataFrame,
    pyrolysis_cost_lookup: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    row_id = 1
    for base in common_feed_rows:
        subtype_name = str(base["manure_subtype"])
        subtype_profile = subtype_profiles[subtype_name]
        manure_targets = pyrolysis_targets.copy()
        manure_targets["product_char_yield_pct"] = float(subtype_profile["char_yield_pct"])
        manure_targets["product_char_hhv_mj_per_kg"] = float(subtype_profile["char_hhv_mj_per_kg"])
        manure_targets["energy_recovery_pct"] = float(subtype_profile["energy_recovery_pct"])
        cost_reference = pyrolysis_cost_lookup.get(subtype_name, {})

        for _, condition in pyrolysis_conditions.iterrows():
            row = base.copy()
            row.update(
                {
                    "sample_id": f"Waste2Energy::planning::pyrolysis::{row_id:04d}",
                    "source_repo": "Waste2EnergyDerived",
                    "source_file": "11_build_planning_mult_pathway_dataset.py",
                    "source_dataset_kind": "synthetic_mixed_feed_pyrolysis_candidate",
                    "reference_label": "weighted_blend_from_pyrolysis_anchor_and_manure_balance_references",
                    "pathway": "pyrolysis",
                    "pathway_variant": "mixed_feed_pyrolysis_condition_anchor",
                    "pathway_process_basis": PATHWAY_READINESS_RULES["pyrolysis"]["process_basis"],
                    "pathway_performance_basis": PATHWAY_READINESS_RULES["pyrolysis"]["performance_basis"],
                    "pathway_environment_basis": PATHWAY_READINESS_RULES["pyrolysis"]["environment_basis"],
                    "pathway_cost_basis": PATHWAY_READINESS_RULES["pyrolysis"]["cost_basis"],
                    "pathway_claim_boundary": PATHWAY_READINESS_RULES["pyrolysis"]["claim_boundary"],
                    "pathway_cost_proxy_base_factor": PYROLYSIS_COST_PROXY_BASE,
                    "process_temperature_c": float(condition["process_temperature_c"]),
                    "residence_time_min": float(condition["residence_time_min"]),
                    "heating_rate_c_per_min": float(condition["heating_rate_c_per_min"])
                    if pd.notna(condition["heating_rate_c_per_min"])
                    else pd.NA,
                    "unit_cost_usd_per_ton_biochar_reference_2040": cost_reference.get(
                        "unit_cost_usd_per_ton_biochar_reference_2040", float("nan")
                    ),
                    "total_cost_usd_per_ton_biochar_reference_2040": cost_reference.get(
                        "total_cost_usd_per_ton_biochar_reference_2040", float("nan")
                    ),
                    "feedstock_cost_usd_per_ton_manure_reference_2040": cost_reference.get(
                        "feedstock_cost_usd_per_ton_manure_reference_2040", float("nan")
                    ),
                }
            )
            for column in TARGET_COLUMNS:
                row[column] = weighted_value(
                    float(manure_targets[column]),
                    float(pyrolysis_targets[column]),
                    float(base["blend_manure_ratio"]),
                    float(base["blend_wet_waste_ratio"]),
                )
            row["pathway_energy_intensity_mj_per_ton"] = (
                float(row["product_char_yield_pct"]) / 100.0 * float(row["product_char_hhv_mj_per_kg"]) * 1000.0
            )
            rows.append(row)
            row_id += 1
    return rows


def build_ad_candidates(
    *,
    common_feed_rows: list[dict[str, object]],
    ad_emission_factor: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    row_id = 1
    ad_energy_mj = AD_WET_ELECTRICITY_KWH_PER_TON * KWH_TO_MJ + (
        max(AD_WET_AVAILABLE_HEAT_MMBTU_PER_TON - AD_WET_REACTOR_HEAT_MMBTU_PER_TON, 0.0) * MMBTU_TO_MJ
    )
    for base in common_feed_rows:
        row = base.copy()
        row.update(
            {
                "sample_id": f"Waste2Energy::planning::ad::{row_id:04d}",
                "source_repo": "CaliforniaRegionData; EPA_WARM",
                "source_file": (
                    "paper1_region_scenario_placeholder.csv; "
                    "california_waste_treatment_emission_factor_reference.csv"
                ),
                "source_dataset_kind": "regional_ad_summary_candidate",
                "reference_label": "food_waste_ad_summary_proxy_with_warm_energy_anchor",
                "pathway": "ad",
                "pathway_variant": "anaerobic_digestion_summary_proxy",
                "pathway_process_basis": PATHWAY_READINESS_RULES["ad"]["process_basis"],
                "pathway_performance_basis": PATHWAY_READINESS_RULES["ad"]["performance_basis"],
                "pathway_environment_basis": PATHWAY_READINESS_RULES["ad"]["environment_basis"],
                "pathway_cost_basis": PATHWAY_READINESS_RULES["ad"]["cost_basis"],
                "pathway_claim_boundary": PATHWAY_READINESS_RULES["ad"]["claim_boundary"],
                "pathway_cost_proxy_base_factor": AD_COST_PROXY_BASE,
                "pathway_emission_factor_kgco2e_per_short_ton_reference": ad_emission_factor,
                "pathway_energy_intensity_mj_per_ton": ad_energy_mj,
                "process_temperature_c": 37.0,
                "residence_time_min": 1440.0,
                "heating_rate_c_per_min": pd.NA,
                "product_char_yield_pct": 0.0,
                "product_char_hhv_mj_per_kg": 0.0,
                "energy_recovery_pct": 0.0,
                "carbon_retention_pct": 0.0,
            }
        )
        rows.append(row)
        row_id += 1
    return rows


def build_baseline_candidates(
    *,
    common_feed_rows: list[dict[str, object]],
    baseline_factor: float,
    baseline_energy_intensity_mj_per_ton: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    row_id = 1
    for base in common_feed_rows:
        row = base.copy()
        row.update(
            {
                "sample_id": f"Waste2Energy::planning::baseline::{row_id:04d}",
                "source_repo": "CaliforniaRegionData; EPA_WARM",
                "source_file": (
                    "california_food_waste_treatment_mix_reference.csv; "
                    "paper1_region_scenario_placeholder.csv"
                ),
                "source_dataset_kind": "regional_baseline_anchor_candidate",
                "reference_label": "california_weighted_food_waste_management_mix",
                "pathway": "baseline",
                "pathway_variant": "california_weighted_management_mix",
                "pathway_process_basis": PATHWAY_READINESS_RULES["baseline"]["process_basis"],
                "pathway_performance_basis": PATHWAY_READINESS_RULES["baseline"]["performance_basis"],
                "pathway_environment_basis": PATHWAY_READINESS_RULES["baseline"]["environment_basis"],
                "pathway_cost_basis": PATHWAY_READINESS_RULES["baseline"]["cost_basis"],
                "pathway_claim_boundary": PATHWAY_READINESS_RULES["baseline"]["claim_boundary"],
                "pathway_cost_proxy_base_factor": BASELINE_COST_PROXY_BASE,
                "pathway_emission_factor_kgco2e_per_short_ton_reference": baseline_factor,
                "pathway_energy_intensity_mj_per_ton": baseline_energy_intensity_mj_per_ton,
                "process_temperature_c": 0.0,
                "residence_time_min": 0.0,
                "heating_rate_c_per_min": pd.NA,
                "product_char_yield_pct": 0.0,
                "product_char_hhv_mj_per_kg": 0.0,
                "energy_recovery_pct": 0.0,
                "carbon_retention_pct": 0.0,
            }
        )
        rows.append(row)
        row_id += 1
    return rows


def build_prototypes() -> tuple[pd.DataFrame, dict[str, object]]:
    combined = load_combined_observed()
    htc = combined[combined["pathway"] == "htc"].copy()
    htc_manure = htc[htc["feedstock_group"] == "manure"].copy()
    htc_food = htc[htc["feedstock_group"] == "food_waste"].copy()
    pyrolysis = combined[combined["pathway"] == "pyrolysis"].copy()

    if htc_manure.empty or htc_food.empty or pyrolysis.empty:
        raise RuntimeError("Missing HTC manure, HTC food-waste, or pyrolysis observed rows.")

    htc_manure_medians = htc_manure[FEEDSTOCK_COLUMNS + TARGET_COLUMNS].median(numeric_only=True)
    htc_food_feedstock = htc_food[FEEDSTOCK_COLUMNS].median(numeric_only=True)
    htc_food_targets = htc_food[TARGET_COLUMNS].median(numeric_only=True)
    pyrolysis_targets = pyrolysis[TARGET_COLUMNS].median(numeric_only=True)

    manure_subtypes = load_manure_subtype_reference()
    subtype_profiles = build_subtype_profiles(
        htc_manure_medians=htc_manure_medians,
        manure_subtypes=manure_subtypes,
    )
    common_feed_rows = build_common_feed_rows(
        food_feedstock_medians=htc_food_feedstock,
        subtype_profiles=subtype_profiles,
    )

    htc_conditions = representative_conditions(
        htc_food,
        include_heating_rate=False,
        defaults=[
            {"process_temperature_c": 210.0, "residence_time_min": 30.0, "count": 0},
            {"process_temperature_c": 240.0, "residence_time_min": 30.0, "count": 0},
            {"process_temperature_c": 270.0, "residence_time_min": 60.0, "count": 0},
            {"process_temperature_c": 300.0, "residence_time_min": 60.0, "count": 0},
        ],
    )
    pyrolysis_conditions = representative_conditions(
        pyrolysis,
        include_heating_rate=True,
        defaults=[
            {
                "process_temperature_c": 400.0,
                "residence_time_min": 30.0,
                "heating_rate_c_per_min": 10.0,
                "count": 0,
            },
            {
                "process_temperature_c": 500.0,
                "residence_time_min": 60.0,
                "heating_rate_c_per_min": 10.0,
                "count": 0,
            },
            {
                "process_temperature_c": 600.0,
                "residence_time_min": 60.0,
                "heating_rate_c_per_min": 20.0,
                "count": 0,
            },
            {
                "process_temperature_c": 700.0,
                "residence_time_min": 60.0,
                "heating_rate_c_per_min": 20.0,
                "count": 0,
            },
        ],
    )

    warm_reference = load_warm_reference()
    treatment_mix = load_treatment_mix_reference()
    california_reference = load_california_food_waste_reference()

    ad_emission_factor = float(
        warm_reference.loc[
            warm_reference["management_pathway"] == "anaerobic_digestion_summary",
            "factor_value",
        ].iloc[0]
    )
    baseline_factor = float(
        treatment_mix.loc[
            treatment_mix["baseline_relevance"] == "baseline_default",
            "factor_value",
        ].iloc[0]
    )
    treatment_mix_components = treatment_mix[
        treatment_mix["baseline_relevance"] == "california_weighted_mix_component"
    ].copy()
    landfill_share = float(
        treatment_mix_components.loc[
            treatment_mix_components["management_pathway"]
            == "landfill_with_lfg_recovery_and_electricity_generation",
            "pathway_share_of_total_food_waste_management",
        ].iloc[0]
    )
    ad_share = float(
        treatment_mix_components.loc[
            treatment_mix_components["management_pathway"] == "anaerobic_digestion_summary",
            "pathway_share_of_total_food_waste_management",
        ].iloc[0]
    )
    reference_grid_factor = float(load_region_scenarios().iloc[0]["scenario_grid_electricity_emission_factor_kgco2e_per_kwh"])
    landfill_energy_kwh_per_ton = LANDFILL_AVOIDED_UTILITY_EMISSIONS_KGCO2E_PER_SHORT_TON / max(
        reference_grid_factor, 1e-9
    )
    ad_energy_mj = AD_WET_ELECTRICITY_KWH_PER_TON * KWH_TO_MJ + (
        max(AD_WET_AVAILABLE_HEAT_MMBTU_PER_TON - AD_WET_REACTOR_HEAT_MMBTU_PER_TON, 0.0) * MMBTU_TO_MJ
    )
    baseline_energy_intensity_mj_per_ton = landfill_share * landfill_energy_kwh_per_ton * KWH_TO_MJ + ad_share * ad_energy_mj

    pyrolysis_cost_lookup = load_pyrolysis_cost_reference()

    rows: list[dict[str, object]] = []
    rows.extend(
        build_baseline_candidates(
            common_feed_rows=common_feed_rows,
            baseline_factor=baseline_factor,
            baseline_energy_intensity_mj_per_ton=baseline_energy_intensity_mj_per_ton,
        )
    )
    rows.extend(
        build_ad_candidates(
            common_feed_rows=common_feed_rows,
            ad_emission_factor=ad_emission_factor,
        )
    )
    rows.extend(
        build_pyrolysis_candidates(
            common_feed_rows=common_feed_rows,
            subtype_profiles=subtype_profiles,
            pyrolysis_targets=pyrolysis_targets,
            pyrolysis_conditions=pyrolysis_conditions,
            pyrolysis_cost_lookup=pyrolysis_cost_lookup,
        )
    )
    rows.extend(
        build_htc_candidates(
            common_feed_rows=common_feed_rows,
            subtype_profiles=subtype_profiles,
            htc_food_targets=htc_food_targets,
            htc_manure_targets=htc_manure_medians[TARGET_COLUMNS],
            htc_conditions=htc_conditions,
        )
    )
    prototype_df = pd.DataFrame(rows)
    metadata = {
        "htc_condition_count": int(len(htc_conditions)),
        "pyrolysis_condition_count": int(len(pyrolysis_conditions)),
        "blend_case_count": len(BLEND_CASES),
        "manure_subtype_count": len(subtype_profiles),
    }
    return prototype_df, metadata


def build_optimization_rows(
    prototypes: pd.DataFrame,
    region_scenarios: pd.DataFrame,
    california_reference: dict[str, object],
) -> pd.DataFrame:
    reference_generation = float(california_reference["waste_generation_ton_per_year"])
    collection_rate = float(california_reference["collection_rate_pct_reference"])
    collectable_proxy = reference_generation * collection_rate / 100.0

    enriched = prototypes.copy()
    enriched["region_id"] = california_reference["region_id"]
    enriched["region_name"] = california_reference["region_name"]
    enriched["country"] = california_reference["country"]
    enriched["analysis_reference_year"] = int(california_reference["analysis_reference_year"])
    enriched["wet_waste_reference_stream_type"] = california_reference["waste_stream_type"]
    enriched["wet_waste_generation_ton_per_year_reference"] = reference_generation
    enriched["wet_waste_collection_rate_pct_reference"] = collection_rate
    enriched["wet_waste_collectable_ton_per_year_proxy_reference"] = collectable_proxy
    enriched["wet_waste_source_bundle"] = california_reference["source_bundle"]
    enriched["wet_waste_pathway_relevance"] = california_reference["pathway_relevance"]
    enriched["wet_waste_reference_notes"] = california_reference["notes"]

    optimization_df = (
        enriched.assign(_merge_key=1)
        .merge(region_scenarios.assign(_merge_key=1), on=["_merge_key", "region_id"], how="inner")
        .drop(columns="_merge_key")
    )
    optimization_df["optimization_case_id"] = (
        optimization_df["sample_id"] + "::" + optimization_df["scenario_name"]
    )
    optimization_df["scenario_wet_waste_feed_allocation_ton_per_year_proxy"] = (
        optimization_df["scenario_wet_waste_collectable_ton_per_year_proxy"]
        * optimization_df["blend_wet_waste_ratio"]
    )
    optimization_df["optimization_region_source"] = "paper1_region_scenario_placeholder.csv"
    optimization_df["optimization_wet_waste_source"] = "california_food_waste_model_input.csv"
    optimization_df["optimization_energy_source"] = "planning_pathway_candidate_layer"
    optimization_df["optimization_emission_source"] = "planning_pathway_candidate_layer"

    pathway_factor = pd.to_numeric(
        optimization_df.get("pathway_emission_factor_kgco2e_per_short_ton_reference", pd.NA),
        errors="coerce",
    )
    emission_multiplier = pd.to_numeric(
        optimization_df["emission_factor_multiplier"], errors="coerce"
    ).fillna(1.0)
    optimization_df["pathway_emission_factor_kgco2e_per_short_ton_scenario_proxy"] = (
        pathway_factor * emission_multiplier
    )
    carbon_retention = pd.to_numeric(
        optimization_df["carbon_retention_pct"], errors="coerce"
    ).fillna(0.0) / 100.0
    baseline_factor = pd.to_numeric(
        optimization_df["scenario_baseline_waste_treatment_emission_factor_kgco2e_per_short_ton"],
        errors="coerce",
    ).fillna(0.0)
    default_benefit = baseline_factor * carbon_retention
    explicit_benefit = (
        baseline_factor - optimization_df["pathway_emission_factor_kgco2e_per_short_ton_scenario_proxy"]
    )
    optimization_df["pathway_environment_benefit_kgco2e_per_ton"] = explicit_benefit.where(
        pathway_factor.notna(), default_benefit
    )
    optimization_df.loc[
        optimization_df["pathway"] == "baseline",
        "pathway_environment_benefit_kgco2e_per_ton",
    ] = 0.0
    return optimization_df


def build_pathway_readiness_summary(
    prototypes: pd.DataFrame,
    optimization_df: pd.DataFrame,
) -> pd.DataFrame:
    prototype_counts = prototypes.groupby("pathway").size().rename("prototype_row_count")
    optimization_counts = optimization_df.groupby("pathway").size().rename("optimization_row_count")
    summary = (
        pd.concat([prototype_counts, optimization_counts], axis=1)
        .fillna(0)
        .reset_index()
        .rename(columns={"pathway": "pathway"})
    )
    summary["prototype_row_count"] = summary["prototype_row_count"].astype(int)
    summary["optimization_row_count"] = summary["optimization_row_count"].astype(int)
    summary["distinct_manure_subtypes"] = summary["pathway"].map(
        prototypes.groupby("pathway")["manure_subtype"].nunique().to_dict()
    ).fillna(0).astype(int)
    summary["distinct_blend_cases"] = summary["pathway"].map(
        prototypes.groupby("pathway")["blending_case"].nunique().to_dict()
    ).fillna(0).astype(int)
    summary["distinct_variants"] = summary["pathway"].map(
        prototypes.groupby("pathway")["pathway_variant"].nunique().to_dict()
    ).fillna(0).astype(int)
    summary["process_basis"] = summary["pathway"].map(
        {key: value["process_basis"] for key, value in PATHWAY_READINESS_RULES.items()}
    )
    summary["performance_basis"] = summary["pathway"].map(
        {key: value["performance_basis"] for key, value in PATHWAY_READINESS_RULES.items()}
    )
    summary["environment_basis"] = summary["pathway"].map(
        {key: value["environment_basis"] for key, value in PATHWAY_READINESS_RULES.items()}
    )
    summary["cost_basis"] = summary["pathway"].map(
        {key: value["cost_basis"] for key, value in PATHWAY_READINESS_RULES.items()}
    )
    summary["claim_boundary"] = summary["pathway"].map(
        {key: value["claim_boundary"] for key, value in PATHWAY_READINESS_RULES.items()}
    )
    return summary.sort_values("pathway").reset_index(drop=True)


def build_assumptions_payload(
    *,
    prototypes: pd.DataFrame,
    optimization_df: pd.DataFrame,
    readiness_summary: pd.DataFrame,
    california_reference: dict[str, object],
    region_scenarios: pd.DataFrame,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "dataset_name": "optimization_input_dataset",
        "prototype_dataset_name": "paper1_planning_pathway_prototypes",
        "generation_method": (
            "Common mixed-feed prototypes crossed with pathway-specific planning rules for baseline, AD, "
            "pyrolysis, and HTC."
        ),
        "default_region_id": california_reference["region_id"],
        "default_region_name": california_reference["region_name"],
        "default_wet_waste_stream": california_reference["waste_stream_type"],
        "prototype_row_count": int(len(prototypes)),
        "optimization_row_count": int(len(optimization_df)),
        "scenario_names": region_scenarios["scenario_name"].dropna().astype(str).tolist(),
        "pathways": readiness_summary.to_dict("records"),
        "construction_metadata": metadata,
        "assumptions": [
            "Baseline and AD rows are optimization-ready regional planning anchors, not direct process-observation rows.",
            "Pyrolysis and HTC rows are synthetic mixed-feed candidates anchored to repository observations and manure subtype balance references.",
            "Energy objective values now prefer explicit pathway energy intensities when they are available, and thermochemical rows still remain consistent with char-yield-based recoverable energy.",
            "Environmental objective values now prefer pathway-level emission-factor differentials for baseline and AD, while pyrolysis and HTC keep carbon-retention-based avoidance proxies.",
            "Cost remains a relative proxy index across all pathways in the current planning layer; no pathway should be written as having a publication-ready real total-system-cost estimate yet.",
            "Pyrolysis economic reference columns from ManurePyrolysisIAM are carried into the planning candidates for future calibration, but they are not yet used as the cross-pathway cost objective.",
            "Scenario-specific environmental factors still come from the California region placeholder chain, so high-level emission multipliers affect both baseline and alternative pathway benefit calculations.",
        ],
    }


def main() -> None:
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

    california_reference = load_california_food_waste_reference()
    region_scenarios = load_region_scenarios()
    prototypes, metadata = build_prototypes()
    optimization_df = build_optimization_rows(
        prototypes=prototypes,
        region_scenarios=region_scenarios,
        california_reference=california_reference,
    )
    readiness_summary = build_pathway_readiness_summary(
        prototypes=prototypes,
        optimization_df=optimization_df,
    )

    prototypes.to_csv(PROTOTYPE_OUTPUT, index=False)
    optimization_df.to_csv(OPTIMIZATION_OUTPUT, index=False)
    readiness_summary.to_csv(READINESS_OUTPUT, index=False)

    assumptions_payload = build_assumptions_payload(
        prototypes=prototypes,
        optimization_df=optimization_df,
        readiness_summary=readiness_summary,
        california_reference=california_reference,
        region_scenarios=region_scenarios,
        metadata=metadata,
    )
    payload_text = json.dumps(assumptions_payload, indent=2)
    ASSUMPTIONS_OUTPUT.write_text(payload_text, encoding="utf-8")
    LEGACY_ASSUMPTIONS_OUTPUT.write_text(payload_text, encoding="utf-8")

    print(f"Wrote {PROTOTYPE_OUTPUT}")
    print(f"Wrote {OPTIMIZATION_OUTPUT}")
    print(f"Wrote {READINESS_OUTPUT}")
    print(f"Wrote {ASSUMPTIONS_OUTPUT}")
    print(f"Wrote {LEGACY_ASSUMPTIONS_OUTPUT}")


if __name__ == "__main__":
    main()
