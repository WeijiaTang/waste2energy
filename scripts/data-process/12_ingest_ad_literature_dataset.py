from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "literature-ad"
UNIFIED_DIR = ROOT / "data" / "processed" / "unified_features"
MODEL_READY_DIR = ROOT / "data" / "processed" / "model_ready"

RAW_AD_YIELDS = RAW_DIR / "ofmsw_ad_yields_sailer_2022_table14_raw.csv"
ADDITIONAL_RAW_AD_YIELDS = RAW_DIR / "ad_literature_yields_additional_raw.csv"
AD_STANDARDIZED_OUTPUT = UNIFIED_DIR / "ad_literature_standardized.csv"
AD_SUMMARY_OUTPUT = UNIFIED_DIR / "ad_literature_summary_by_feedstock_group.csv"
AD_MANIFEST_OUTPUT = MODEL_READY_DIR / "ad_literature_dataset_manifest.json"

# Lower heating value of methane at standard conditions. Used only to convert
# methane yield to an energy-yield proxy for pathway comparison.
METHANE_LHV_MJ_PER_M3 = 35.8

# The downloaded Sailer et al. OFMSW table reports yields on organic dry matter
# (oDM), while the current planning layer is wet-ton anchored. The basic
# characteristics source file is retained in data/raw; these conservative
# category defaults avoid pretending that this first AD expansion is a complete
# facility design dataset.
DEFAULT_DM_FRACTION_BY_GROUP = {
    "food_waste": 0.35,
    "green_waste": 0.65,
    "paper_waste": 0.95,
    "co_digestion_food_sanitation_waste": 0.18,
    "co_digestion_food_wastewater": 0.10,
    "industrial_food_waste": 0.20,
    "agroindustrial_waste": 0.35,
}
DEFAULT_ODM_FRACTION_OF_DM_BY_GROUP = {
    "food_waste": 0.94,
    "green_waste": 0.85,
    "paper_waste": 0.95,
    "co_digestion_food_sanitation_waste": 0.88,
    "co_digestion_food_wastewater": 0.80,
    "industrial_food_waste": 0.90,
    "agroindustrial_waste": 0.90,
}


def build_standardized_ad_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame["specific_biogas_yield_m3_per_kg_odm"] = pd.to_numeric(
        frame["specific_biogas_yield_m3_per_kg_odm"], errors="coerce"
    )
    frame["specific_methane_yield_m3_per_kg_odm"] = pd.to_numeric(
        frame["specific_methane_yield_m3_per_kg_odm"], errors="coerce"
    )
    frame["methane_concentration_pct"] = pd.to_numeric(frame["methane_concentration_pct"], errors="coerce")

    frame["sample_id"] = [
        f"Sailer2022_OFMSW_AD::{code}::rep{rep}"
        for code, rep in zip(frame["sample_code"].astype(str), frame["replicate"].astype(int), strict=False)
    ]
    frame["source_repo"] = "MendeleyData"
    frame["source_dataset_kind"] = "literature_ofmsw_ad_yield_observation"
    frame["reference_label"] = "Sailer et al. 2022 OFMSW AD Table 14"
    frame["blending_case"] = "single_source_or_ofmsw_subcategory_reference"
    frame["pathway"] = "ad"

    frame["feedstock_dry_matter_fraction_assumed"] = frame["feedstock_group"].map(
        DEFAULT_DM_FRACTION_BY_GROUP
    ).fillna(0.35)
    frame["feedstock_odm_fraction_of_dm_assumed"] = frame["feedstock_group"].map(
        DEFAULT_ODM_FRACTION_OF_DM_BY_GROUP
    ).fillna(0.90)
    frame["odm_kg_per_wet_ton_proxy"] = (
        1000.0
        * frame["feedstock_dry_matter_fraction_assumed"]
        * frame["feedstock_odm_fraction_of_dm_assumed"]
    )
    frame["methane_yield_m3_per_wet_ton_proxy"] = (
        frame["specific_methane_yield_m3_per_kg_odm"] * frame["odm_kg_per_wet_ton_proxy"]
    )
    frame["ad_energy_yield_mj_per_wet_ton_proxy"] = (
        frame["methane_yield_m3_per_wet_ton_proxy"] * METHANE_LHV_MJ_PER_M3
    )
    frame["evidence_scope"] = "OFMSW biochemical methane potential; not facility siting or full-scale AD design"
    frame["conversion_note"] = (
        "Methane-yield observations are reported per kg oDM. Wet-ton energy proxies use group-level "
        "DM and oDM assumptions and should be used as bounded comparison inputs."
    )

    columns = [
        "sample_id",
        "source_repo",
        "source_file",
        "source_dataset_kind",
        "reference_label",
        "study_id",
        "doi",
        "article_doi",
        "pathway",
        "sample_code",
        "replicate",
        "feedstock_name",
        "feedstock_group",
        "blending_case",
        "specific_biogas_yield_m3_per_kg_odm",
        "methane_concentration_pct",
        "specific_methane_yield_m3_per_kg_odm",
        "feedstock_dry_matter_fraction_assumed",
        "feedstock_odm_fraction_of_dm_assumed",
        "odm_kg_per_wet_ton_proxy",
        "methane_yield_m3_per_wet_ton_proxy",
        "ad_energy_yield_mj_per_wet_ton_proxy",
        "basis",
        "source_table",
        "evidence_scope",
        "conversion_note",
        "notes",
    ]
    return frame[columns].sort_values(["feedstock_group", "sample_code", "replicate"]).reset_index(drop=True)


def build_summary(standardized: pd.DataFrame) -> pd.DataFrame:
    grouped = standardized.groupby("feedstock_group", dropna=False)
    summary = grouped.agg(
        observations=("sample_id", "count"),
        sample_codes=("sample_code", "nunique"),
        methane_yield_m3_per_kg_odm_mean=("specific_methane_yield_m3_per_kg_odm", "mean"),
        methane_yield_m3_per_kg_odm_min=("specific_methane_yield_m3_per_kg_odm", "min"),
        methane_yield_m3_per_kg_odm_max=("specific_methane_yield_m3_per_kg_odm", "max"),
        methane_concentration_pct_mean=("methane_concentration_pct", "mean"),
        energy_yield_mj_per_wet_ton_proxy_mean=("ad_energy_yield_mj_per_wet_ton_proxy", "mean"),
        energy_yield_mj_per_wet_ton_proxy_min=("ad_energy_yield_mj_per_wet_ton_proxy", "min"),
        energy_yield_mj_per_wet_ton_proxy_max=("ad_energy_yield_mj_per_wet_ton_proxy", "max"),
    ).reset_index()
    numeric_cols = summary.select_dtypes(include="number").columns
    summary[numeric_cols] = summary[numeric_cols].round(3)
    return summary


def main() -> None:
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)
    raw_frames = [pd.read_csv(RAW_AD_YIELDS)]
    if ADDITIONAL_RAW_AD_YIELDS.exists():
        raw_frames.append(pd.read_csv(ADDITIONAL_RAW_AD_YIELDS))
    raw = pd.concat(raw_frames, ignore_index=True, sort=False)
    standardized = build_standardized_ad_dataset(raw)
    summary = build_summary(standardized)

    standardized.to_csv(AD_STANDARDIZED_OUTPUT, index=False)
    summary.to_csv(AD_SUMMARY_OUTPUT, index=False)

    manifest = {
        "dataset": "ad_literature_standardized",
        "raw_input": [
            str(RAW_AD_YIELDS.relative_to(ROOT)),
            str(ADDITIONAL_RAW_AD_YIELDS.relative_to(ROOT)) if ADDITIONAL_RAW_AD_YIELDS.exists() else None,
        ],
        "standardized_output": str(AD_STANDARDIZED_OUTPUT.relative_to(ROOT)),
        "summary_output": str(AD_SUMMARY_OUTPUT.relative_to(ROOT)),
        "source_manifest": str((RAW_DIR / "source_manifest.csv").relative_to(ROOT)),
        "rows": int(len(standardized)),
        "feedstock_group_counts": standardized["feedstock_group"].value_counts().to_dict(),
        "methane_lhv_mj_per_m3": METHANE_LHV_MJ_PER_M3,
        "assumption_note": (
            "First-pass AD expansion from traceable OFMSW methane-yield data. Wet-ton energy yields are "
            "proxy conversions from oDM-basis yields; use for bounded screening, not facility design."
        ),
    }
    AD_MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
