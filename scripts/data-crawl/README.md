# California Official Data Crawl

This folder contains manifest-driven download scripts for official external-region data used by Waste2Energy Paper 1.

## Purpose

The crawl layer should:

- download source-preserved official data into `data/raw`
- record provenance for every downloaded file
- keep the download logic separate from `scripts/data-process`
- make it easy to extend the same workflow to other states later

## Current Scope

The first California crawl pass targets:

- electricity price data from EIA
- grid emission factor data from EPA eGRID
- waste-treatment emission factors from EPA WARM
- a livestock reference page snapshot from USDA NASS California
- SWIS facility/activity/waste exports from CalRecycle for facility-level organics infrastructure

The manifest also records California data sources that are relevant but not yet auto-downloaded, such as CalRecycle pages that currently require manual export or a future site-specific connector.

## Files

- `california_sources.json`
  Declarative source registry for California data collection.
- `download_official_sources.py`
  Generic downloader that reads the manifest, downloads supported entries, and writes raw provenance manifests.
- `crawl_california_energy_prices.py`
  California EIA energy-price collector that writes validated raw snapshots plus structured baseline price tables.
- `crawl_california_livestock.py`
  California USDA livestock collector that writes structured statewide livestock tables and release-text evidence from official PDFs.
- `crawl_california_policy_reference.py`
  California SB 1383 policy collector that writes official page snapshots plus a structured policy-reference table for Paper 1.
- `crawl_california_waste_emission_factors.py`
  California EPA WARM collector that structures food-waste treatment emission factors and marks the default baseline pathway used by Paper 1.
- `register_calrecycle_capacity_export.py`
  Manual-export intake helper that copies an official CalRecycle capacity-planning export into the canonical California raw path and records provenance.

## Usage

Run all currently auto-downloadable California sources:

```powershell
python scripts/data-crawl/download_official_sources.py
```

Run only selected source IDs:

```powershell
python scripts/data-crawl/download_official_sources.py --source-id california_eia_electricity_profile_html --source-id california_eia_natural_gas_industrial_price_xls --source-id california_epa_egrid_xlsx --source-id california_epa_warm_organic_materials_pdf --source-id california_swis_site_export_xlsx --source-id california_swis_site_activity_export_xlsx --source-id california_swis_site_waste_export_xlsx
```

Force re-download:

```powershell
python scripts/data-crawl/download_official_sources.py --force
```

Build California energy-price raw tables:

```powershell
python scripts/data-crawl/crawl_california_energy_prices.py
```

Build California livestock raw tables:

```powershell
python scripts/data-crawl/crawl_california_livestock.py
```

Build California SB 1383 policy raw tables:

```powershell
python scripts/data-crawl/crawl_california_policy_reference.py
```

Build California waste-treatment emission-factor reference tables from EPA WARM:

```powershell
python scripts/data-crawl/crawl_california_waste_emission_factors.py
```

Register an official manual export from the CalRecycle capacity-planning portal:

```powershell
python scripts/data-crawl/register_calrecycle_capacity_export.py --source-path "C:\path\to\capacity_export.xlsx"
```

## Output Layout

Downloaded files are written under:

- `data/raw/external-region-data/california/region_context`
- `data/raw/external-region-data/california/livestock_supply`
- `data/raw/external-region-data/california/wet_waste_supply`
- `data/raw/external-region-data/california/energy_prices`
- `data/raw/external-region-data/california/emission_factors`
- `data/raw/external-region-data/california/policy_reference`

Each run also updates:

- `data/raw/external-region-data/california/source_manifest.csv`
- `data/raw/external-region-data/california/source_manifest.json`

## Guardrails

- Raw files are stored as-downloaded.
- The crawl layer does not edit files in `data/processed`.
- Every manifest entry must include source organization, source URL, download time, target file path, and status.
- Sources blocked by robots, JavaScript-only export flows, or manual portal interactions should stay documented in the manifest with a non-downloaded status rather than being silently skipped.
- Manual-export-only sources should be copied into the canonical raw filename before any processing scripts consume them.
