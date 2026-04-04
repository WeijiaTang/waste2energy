# California External Region Raw Data

This folder stores California-specific official raw data for Waste2Energy Paper 1.

## Structure

- `region_context`
- `livestock_supply`
- `wet_waste_supply`
- `energy_prices`
- `emission_factors`
- `policy_reference`

## Provenance

The crawl layer writes:

- `source_manifest.json`
- `source_manifest.csv`

These manifests record source URLs, source organizations, retrieval status, download timestamps, file paths, and notes for each California data source.

## Rules

- Keep files here source-preserved.
- Do not hand-edit downloaded raw files.
- Add all new official California raw inputs through `scripts/data-crawl` when possible.
- If a source must be exported manually, record the export path and provenance in the manifest before using it in `scripts/data-process`.
