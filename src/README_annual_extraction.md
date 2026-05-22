# Annual Area Extraction Framework

This folder contains the first executable framework for extracting annual lake area time series from the cleaned glacier-fed lake database.

## Files

- `build_extraction_tasks.py`
  Builds the regional and region-year task tables from the prepared core-region batches.

- `gee_extract_annual_max_open_water.js`
  Earth Engine template script for extracting annual maximum open-water extent using Landsat Collection 2 Level-2.

- `gee_extract_annual_max_open_water_single_year.js`
  Safer single-year export version for slow downloads, resumable runs, and quick QC.

- `gee_extract_annual_max_open_water_chunked.js`
  Chunked single-year export version for very large regions such as Greenland Periphery.

- `build_region_chunk_plan.py`
  Builds region chunk plans for heavy test regions.

- `qc_annual_area_exports.py`
  Local post-export QC script for reviewing the test CSVs downloaded from Earth Engine.

## Config

Primary workflow config:

- `E:\Glacier\configs\annual_area_extraction_config.json`

Key defaults:

- Main analysis lakes: `IUL + ICL`
- Supplemental only: `SGL`
- Years: `2000-2024`
- Primary metric: `annual maximum open-water extent`
- Primary sensor: Landsat Collection 2 Level-2
- Test regions: `greenland_periphery`, `central_asia`

## Recommended execution order

1. Run `python .\src\build_extraction_tasks.py`
2. Upload one test region GeoJSON from `E:\Glacier\data\prepared\core_region_batches` to Earth Engine as a table asset
3. Update `regionKey`, `lakeAsset`, and `exportPrefix` in `gee_extract_annual_max_open_water.js`
4. If the full regional export is too large, switch to `gee_extract_annual_max_open_water_single_year.js`
5. If a single-year full region is still too slow, switch to `gee_extract_annual_max_open_water_chunked.js`
6. Use the chunk plan from `E:\Glacier\data\prepared\chunk_plans`
7. Run one chunk at a time in Earth Engine
8. Place the downloaded CSVs into `E:\Glacier\data\interim\annual_area_raw`
9. Run `python .\src\qc_annual_area_exports.py`
10. Review output CSVs and adjust QC rules before expanding to all 17 regions

## Current GEE outputs

The refined Earth Engine script now exports the following per-lake per-year fields:

- `annual_max_area_km2`
- `water_area_median_km2`
- `valid_area_any_km2`
- `baseline_valid_area_fraction`
- `annual_area_to_baseline_ratio`
- `image_count`
- `l5_count`, `l7_count`, `l8_count`, `l9_count`
- `qc_enough_images`
- `qc_enough_coverage`
- `qc_usable`

These fields are meant to support immediate post-export QC in the local project.

## Expected outputs

Task tables:

- `E:\Glacier\data\prepared\extraction_tasks\region_tasks.csv`
- `E:\Glacier\data\prepared\extraction_tasks\region_year_tasks_full.csv`
- `E:\Glacier\data\prepared\extraction_tasks\region_year_tasks_test.csv`

Annual extraction outputs should later be stored under:

- `E:\Glacier\data\interim\annual_area_raw`
- `E:\Glacier\data\interim\annual_area_qc`
