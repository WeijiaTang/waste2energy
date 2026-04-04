from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_CA_DIR = ROOT / "data" / "raw" / "external-region-data" / "california"
SCENARIO_DIR = ROOT / "data" / "processed" / "scenario_inputs"
MODEL_READY_DIR = ROOT / "data" / "processed" / "model_ready"

FOOD_WASTE_MODEL_INPUT = MODEL_READY_DIR / "california_food_waste_model_input.csv"
FACILITY_INVENTORY = SCENARIO_DIR / "california_organics_facility_inventory.csv"
WARM_REFERENCE = (
    RAW_CA_DIR / "emission_factors" / "california_waste_treatment_emission_factor_reference.csv"
)

REFERENCE_OUTPUT = SCENARIO_DIR / "california_food_waste_treatment_mix_reference.csv"
STATUS_OUTPUT = MODEL_READY_DIR / "california_food_waste_treatment_mix_status.json"


def utcnow_text() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def ensure_inputs() -> None:
    missing = [path for path in [FOOD_WASTE_MODEL_INPUT, FACILITY_INVENTORY, WARM_REFERENCE] if not path.exists()]
    if missing:
        joined = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(f"Missing treatment-mix input files: {joined}")


def load_single_row_csv(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise RuntimeError(f"Expected exactly one row in {path}, found {len(frame)}")
    return frame.iloc[0].to_dict()


def warm_factor(frame: pd.DataFrame, management_pathway: str) -> float:
    match = frame[frame["management_pathway"] == management_pathway].copy()
    if match.empty:
        raise RuntimeError(f"Could not find {management_pathway} in {WARM_REFERENCE}")
    return float(match.iloc[0]["factor_value"])


def final_treatment_share_by_technology(inventory: pd.DataFrame) -> tuple[dict[str, float], str]:
    subset = inventory[
        inventory["accepts_food_wastes"].astype(bool)
        & inventory["technology_group"].isin(["composting", "anaerobic_digestion"])
    ].copy()
    if subset.empty:
        raise RuntimeError("No active food-waste composting or anaerobic-digestion facilities were found")

    throughput_totals = (
        subset.groupby("technology_group")["annualized_throughput_ton_per_year"].sum(min_count=1).dropna()
    )
    if not throughput_totals.empty and throughput_totals.sum() > 0:
        shares = (throughput_totals / throughput_totals.sum()).to_dict()
        return {str(key): float(value) for key, value in shares.items()}, "annualized_throughput_ton_per_year"

    capacity_totals = (
        subset.groupby("technology_group")["annualized_capacity_ton_per_year"].sum(min_count=1).dropna()
    )
    if not capacity_totals.empty and capacity_totals.sum() > 0:
        shares = (capacity_totals / capacity_totals.sum()).to_dict()
        return {str(key): float(value) for key, value in shares.items()}, "annualized_capacity_ton_per_year"

    counts = subset.groupby("technology_group")["SWIS Number"].nunique()
    shares = (counts / counts.sum()).to_dict()
    return {str(key): float(value) for key, value in shares.items()}, "unique_food_waste_facility_count"


def build_reference_rows() -> list[dict[str, object]]:
    model_input = load_single_row_csv(FOOD_WASTE_MODEL_INPUT)
    inventory = pd.read_csv(FACILITY_INVENTORY)
    warm = pd.read_csv(WARM_REFERENCE)
    timestamp = utcnow_text()

    collection_rate = float(model_input["collection_rate_pct_reference"]) / 100.0
    landfill_share = max(0.0, 1.0 - collection_rate)
    treatment_shares, share_basis = final_treatment_share_by_technology(inventory)
    compost_share = collection_rate * float(treatment_shares.get("composting", 0.0))
    digestion_share = collection_rate * float(treatment_shares.get("anaerobic_digestion", 0.0))

    landfill_factor = warm_factor(warm, "landfill_with_lfg_recovery_and_electricity_generation")
    compost_factor = warm_factor(warm, "composting")
    digestion_factor = warm_factor(warm, "anaerobic_digestion_summary")

    weighted_factor = (
        landfill_share * landfill_factor
        + compost_share * compost_factor
        + digestion_share * digestion_factor
    )

    rows = [
        {
            "region_id": "us_ca",
            "region_name": "California",
            "country": "United States",
            "management_pathway": "landfill_with_lfg_recovery_and_electricity_generation",
            "mix_component_type": "observed_disposal_proxy",
            "baseline_relevance": "california_weighted_mix_component",
            "pathway_share_of_total_food_waste_management": landfill_share,
            "share_basis": "1_minus_collection_rate_proxy",
            "factor_value": landfill_factor,
            "factor_unit": "kgCO2e_per_short_ton",
            "weighted_contribution_kgco2e_per_short_ton": landfill_share * landfill_factor,
            "source_organization": "California Department of Resources Recycling and Recovery; U.S. Environmental Protection Agency",
            "source_url": "https://www2.calrecycle.ca.gov/WasteCharacterization/ | https://www2.calrecycle.ca.gov/SolidWaste/Site/DataExport | https://www.epa.gov/warm/versions-waste-reduction-model",
            "download_or_publication_date": timestamp,
            "raw_file_name": "california_food_waste_model_input.csv; california_organics_facility_inventory.csv; epa_warm_organic_materials_v16_dec_2023.pdf",
            "notes": (
                "Landfill share is proxied as one minus the California food-waste collection-rate reference, "
                "while the landfill emission factor remains the EPA WARM controlled-landfill proxy."
            ),
        },
        {
            "region_id": "us_ca",
            "region_name": "California",
            "country": "United States",
            "management_pathway": "composting",
            "mix_component_type": "observed_treatment_proxy",
            "baseline_relevance": "california_weighted_mix_component",
            "pathway_share_of_total_food_waste_management": compost_share,
            "share_basis": share_basis,
            "factor_value": compost_factor,
            "factor_unit": "kgCO2e_per_short_ton",
            "weighted_contribution_kgco2e_per_short_ton": compost_share * compost_factor,
            "source_organization": "California Department of Resources Recycling and Recovery; U.S. Environmental Protection Agency",
            "source_url": "https://www2.calrecycle.ca.gov/SolidWaste/Site/DataExport | https://www.epa.gov/warm/versions-waste-reduction-model",
            "download_or_publication_date": timestamp,
            "raw_file_name": "california_organics_facility_inventory.csv; epa_warm_organic_materials_v16_dec_2023.pdf",
            "notes": (
                "Composting share is allocated from the collected-food-waste portion using the active California "
                "SWIS food-waste facility technology mix."
            ),
        },
        {
            "region_id": "us_ca",
            "region_name": "California",
            "country": "United States",
            "management_pathway": "anaerobic_digestion_summary",
            "mix_component_type": "observed_treatment_proxy",
            "baseline_relevance": "california_weighted_mix_component",
            "pathway_share_of_total_food_waste_management": digestion_share,
            "share_basis": share_basis,
            "factor_value": digestion_factor,
            "factor_unit": "kgCO2e_per_short_ton",
            "weighted_contribution_kgco2e_per_short_ton": digestion_share * digestion_factor,
            "source_organization": "California Department of Resources Recycling and Recovery; U.S. Environmental Protection Agency",
            "source_url": "https://www2.calrecycle.ca.gov/SolidWaste/Site/DataExport | https://www.epa.gov/warm/versions-waste-reduction-model",
            "download_or_publication_date": timestamp,
            "raw_file_name": "california_organics_facility_inventory.csv; epa_warm_organic_materials_v16_dec_2023.pdf",
            "notes": (
                "Anaerobic-digestion share is allocated from the collected-food-waste portion using the active "
                "California SWIS food-waste facility technology mix. The WARM digestion summary factor is used "
                "because SWIS does not consistently distinguish wet vs dry digestion subtype."
            ),
        },
        {
            "region_id": "us_ca",
            "region_name": "California",
            "country": "United States",
            "management_pathway": "california_weighted_food_waste_management_mix",
            "mix_component_type": "weighted_baseline",
            "baseline_relevance": "baseline_default",
            "pathway_share_of_total_food_waste_management": 1.0,
            "share_basis": share_basis,
            "factor_value": weighted_factor,
            "factor_unit": "kgCO2e_per_short_ton",
            "weighted_contribution_kgco2e_per_short_ton": weighted_factor,
            "source_organization": "California Department of Resources Recycling and Recovery; U.S. Environmental Protection Agency",
            "source_url": "https://www2.calrecycle.ca.gov/WasteCharacterization/ | https://www2.calrecycle.ca.gov/SolidWaste/Site/DataExport | https://www.epa.gov/warm/versions-waste-reduction-model",
            "download_or_publication_date": timestamp,
            "raw_file_name": "california_food_waste_model_input.csv; california_organics_facility_inventory.csv; epa_warm_organic_materials_v16_dec_2023.pdf",
            "notes": (
                "California-weighted baseline for food-waste management. Disposal share comes from the current "
                "California collection-rate proxy; collected-food-waste treatment shares come from active SWIS "
                "food-waste facility technology mix; pathway factors come from EPA WARM."
            ),
        },
    ]
    return rows


def main() -> None:
    ensure_inputs()
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_reference_rows()
    frame = pd.DataFrame(rows)
    frame.to_csv(REFERENCE_OUTPUT, index=False)

    baseline_row = frame[frame["baseline_relevance"] == "baseline_default"].iloc[0]
    STATUS_OUTPUT.write_text(
        json.dumps(
            {
                "dataset_name": "california_food_waste_treatment_mix_reference",
                "status": "official_reference_connected",
                "checked_at_utc": utcnow_text(),
                "reference_output_file": str(REFERENCE_OUTPUT.relative_to(ROOT)),
                "baseline_factor_value": float(baseline_row["factor_value"]),
                "baseline_factor_unit": str(baseline_row["factor_unit"]),
                "baseline_management_pathway": str(baseline_row["management_pathway"]),
                "notes": [
                    "This weighted baseline supersedes the previous pure-WARM landfill proxy when regional inputs are rebuilt.",
                    "Treatment shares are California-weighted proxies inferred from active SWIS food-waste facilities.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {REFERENCE_OUTPUT}")
    print(f"Wrote {STATUS_OUTPUT}")


if __name__ == "__main__":
    main()
