from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_SELECTED_FILES = ROOT / "data" / "interim" / "annual_area_qc" / "downloaded_exports_status_selected_files.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "annual_area_skeleton"
ANNUAL_AREA_ROOT = ROOT / "data" / "processed" / "GlacierAnnualArea"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a clean skeleton-year series for one region from selected annual-area export files."
    )
    p.add_argument("--region", required=True, help="Region key, e.g. central_asia")
    p.add_argument("--selected-files-csv", default=str(DEFAULT_SELECTED_FILES))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument(
        "--years",
        default="",
        help="Optional comma-separated year filter, e.g. 2000,2005,2010,2015,2020,2024",
    )
    p.add_argument("--tag", default="", help="Optional output tag override")
    return p.parse_args()


def normalize_boolish(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    return mapped.where(mapped.notna(), series).astype(bool)


def parse_years(raw: str) -> list[int] | None:
    if not raw.strip():
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

def resolve_export_path(row_path: str, file_name: str) -> Path:
    candidate = ROOT / Path(row_path)
    if candidate.exists():
        return candidate
    # Fall back to subdirectories under GlacierAnnualArea (e.g., 05/, 13/)
    matches = list(ANNUAL_AREA_ROOT.rglob(file_name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Export file not found: {file_name} (row path: {row_path})")


def main() -> None:
    args = parse_args()
    region = args.region
    year_filter = parse_years(args.years)
    selected_files = pd.read_csv(args.selected_files_csv)
    region_files = selected_files[selected_files["region_key"] == region].copy()
    if year_filter is not None:
        region_files = region_files[region_files["year"].isin(year_filter)].copy()

    if region_files.empty:
        raise SystemExit(f"No selected files found for region={region}")

    region_files = region_files.sort_values(["year", "chunk_start"]).reset_index(drop=True)

    frames: list[pd.DataFrame] = []
    for row in region_files.itertuples(index=False):
        resolved = resolve_export_path(row.path, row.file)
        df = pd.read_csv(resolved)
        df["source_file"] = row.file
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    if "glambie_region_key" in full.columns:
        full["glambie_region_key"] = full["glambie_region_key"].fillna(region)
    else:
        full["glambie_region_key"] = region

    full["qc_usable"] = normalize_boolish(full["qc_usable"])
    full["qc_enough_images"] = normalize_boolish(full["qc_enough_images"])
    full["qc_enough_coverage"] = normalize_boolish(full["qc_enough_coverage"])

    numeric_cols = [
        "annual_max_area_km2",
        "baseline_area_0_km2",
        "baseline_valid_area_fraction",
        "annual_area_to_baseline_ratio",
        "image_count",
        "water_area_median_km2",
        "valid_area_any_km2",
        "elevation_m",
        "latitude",
        "longitude",
    ]
    for col in numeric_cols:
        if col in full.columns:
            full[col] = pd.to_numeric(full[col], errors="coerce")

    full["year"] = pd.to_numeric(full["year"], errors="coerce").astype("Int64")
    full["flag_low_image_count"] = full["image_count"] < 3
    full["flag_low_valid_fraction"] = full["baseline_valid_area_fraction"] < 0.7
    full["flag_ratio_lt_0_1"] = full["annual_area_to_baseline_ratio"] < 0.1
    full["flag_ratio_gt_3"] = full["annual_area_to_baseline_ratio"] > 3

    usable = full[full["qc_usable"]].copy()

    year_summary = (
        full.groupby("year")
        .agg(
            rows=("lake_id", "count"),
            lake_count=("lake_id", "nunique"),
            usable_rows=("qc_usable", "sum"),
            usable_share=("qc_usable", "mean"),
            total_annual_area_km2=("annual_max_area_km2", "sum"),
            usable_total_annual_area_km2=("annual_max_area_km2", lambda s: float(full.loc[s.index, "annual_max_area_km2"][full.loc[s.index, "qc_usable"]].sum())),
            median_image_count=("image_count", "median"),
            mean_image_count=("image_count", "mean"),
            low_image_rows=("flag_low_image_count", "sum"),
            low_valid_rows=("flag_low_valid_fraction", "sum"),
            low_ratio_rows=("flag_ratio_lt_0_1", "sum"),
            high_ratio_rows=("flag_ratio_gt_3", "sum"),
        )
        .reset_index()
        .sort_values("year")
    )

    usable_year_metrics = (
        usable.groupby("year")
        .agg(
            usable_median_area_ratio=("annual_area_to_baseline_ratio", "median"),
            usable_mean_area_ratio=("annual_area_to_baseline_ratio", "mean"),
            usable_median_valid_fraction=("baseline_valid_area_fraction", "median"),
            usable_median_water_area_km2=("water_area_median_km2", "median"),
        )
        .reset_index()
    )
    year_summary = year_summary.merge(usable_year_metrics, on="year", how="left")
    year_summary["total_area_change_km2"] = year_summary["total_annual_area_km2"].diff()
    year_summary["usable_total_area_change_km2"] = year_summary["usable_total_annual_area_km2"].diff()
    year_summary["usable_share_change"] = year_summary["usable_share"].diff()

    lake_year = full.sort_values(["lake_id", "year"]).reset_index(drop=True)

    lake_summary = (
        full.groupby("lake_id")
        .agg(
            lake_type=("lake_type", "first"),
            harmonized_class=("harmonized_class", "first"),
            baseline_area_0_km2=("baseline_area_0_km2", "first"),
            elevation_m=("elevation_m", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            years_present=("year", "nunique"),
            usable_years=("qc_usable", "sum"),
            usable_share_across_years=("qc_usable", "mean"),
            median_image_count=("image_count", "median"),
            median_valid_fraction=("baseline_valid_area_fraction", "median"),
            mean_area_ratio=("annual_area_to_baseline_ratio", "mean"),
        )
        .reset_index()
    )

    low_quality_lakes = lake_summary[
        (lake_summary["usable_share_across_years"] < 0.5) | (lake_summary["median_valid_fraction"] < 0.7)
    ].sort_values(["usable_share_across_years", "median_valid_fraction", "lake_id"])

    years_sorted = sorted(int(y) for y in year_summary["year"].dropna().tolist())
    interyear_change = year_summary[
        [
            "year",
            "usable_share",
            "usable_share_change",
            "total_annual_area_km2",
            "total_area_change_km2",
            "usable_total_annual_area_km2",
            "usable_total_area_change_km2",
            "usable_median_area_ratio",
            "usable_mean_area_ratio",
        ]
    ].copy()

    latest_drop_candidates = pd.DataFrame()
    if len(years_sorted) >= 2:
        prev_year = years_sorted[-2]
        latest_year = years_sorted[-1]
        pair = full[full["year"].isin([prev_year, latest_year])].copy()
        pair_wide = (
            pair.pivot_table(
                index="lake_id",
                columns="year",
                values=[
                    "annual_area_to_baseline_ratio",
                    "annual_max_area_km2",
                    "qc_usable",
                    "baseline_valid_area_fraction",
                    "image_count",
                ],
                aggfunc="first",
            )
            .sort_index(axis=1)
        )
        pair_wide.columns = [f"{metric}_{year}" for metric, year in pair_wide.columns]
        pair_wide = pair_wide.reset_index()
        meta = full[full["year"] == latest_year][["lake_id", "lake_type", "harmonized_class"]].drop_duplicates("lake_id")
        latest_drop_candidates = meta.merge(pair_wide, on="lake_id", how="left")
        latest_drop_candidates[f"ratio_change_{latest_year}_minus_{prev_year}"] = (
            latest_drop_candidates.get(f"annual_area_to_baseline_ratio_{latest_year}")
            - latest_drop_candidates.get(f"annual_area_to_baseline_ratio_{prev_year}")
        )
        latest_drop_candidates[f"area_change_{latest_year}_minus_{prev_year}"] = (
            latest_drop_candidates.get(f"annual_max_area_km2_{latest_year}")
            - latest_drop_candidates.get(f"annual_max_area_km2_{prev_year}")
        )
        latest_drop_candidates = latest_drop_candidates.sort_values(
            [f"ratio_change_{latest_year}_minus_{prev_year}", f"area_change_{latest_year}_minus_{prev_year}", "lake_id"]
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or region

    lake_year_csv = output_dir / f"{tag}_skeleton_lake_year.csv"
    year_summary_csv = output_dir / f"{tag}_skeleton_year_summary.csv"
    interyear_change_csv = output_dir / f"{tag}_skeleton_interyear_change.csv"
    lake_summary_csv = output_dir / f"{tag}_skeleton_lake_summary.csv"
    low_quality_csv = output_dir / f"{tag}_skeleton_low_quality_lakes.csv"
    latest_drop_csv = output_dir / f"{tag}_skeleton_latest_year_drop_candidates.csv"
    summary_json = output_dir / f"{tag}_skeleton_summary.json"

    lake_year.to_csv(lake_year_csv, index=False)
    year_summary.to_csv(year_summary_csv, index=False)
    interyear_change.to_csv(interyear_change_csv, index=False)
    lake_summary.to_csv(lake_summary_csv, index=False)
    low_quality_lakes.to_csv(low_quality_csv, index=False)
    latest_drop_candidates.to_csv(latest_drop_csv, index=False)

    payload = {
        "region": region,
        "years": sorted(int(y) for y in year_summary["year"].dropna().tolist()),
        "input_file_count": int(len(region_files)),
        "lake_year_rows": int(len(lake_year)),
        "unique_lakes": int(lake_year["lake_id"].nunique()),
        "usable_share_overall": float(full["qc_usable"].mean()),
        "year_summary_csv": str(year_summary_csv),
        "interyear_change_csv": str(interyear_change_csv),
        "lake_year_csv": str(lake_year_csv),
        "lake_summary_csv": str(lake_summary_csv),
        "low_quality_lakes_csv": str(low_quality_csv),
        "latest_drop_candidates_csv": str(latest_drop_csv),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
