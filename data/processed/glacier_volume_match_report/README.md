# Glacier volume match report

This directory keeps only lightweight provenance and QA outputs for the
`glacier_fed_lakes_17regions_merged` volume match.

Final analysis table:

`../final/glacier_fed_lakes_17regions_valid_volume_timeseries_1990_2024.csv`

Key result:

- Source inventory: 102,929 unique `lake_id` values.
- Valid volume time-series lakes: 85,622.
- Final long table rows: 2,996,770.
- Years: 1990-2024.
- Join key: `lake_id`.

Large intermediate files and detailed grouped diagnostics were removed because
the final CSV, infographic, and source inventory/volume directories are
sufficient to document or reproduce them.

Infographic:

`../final/glacier_volume_match_summary_infographic.png`
