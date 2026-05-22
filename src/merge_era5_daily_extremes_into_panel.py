from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_DAILY_DIR = ROOT / "data" / "interim" / "era5_land_daily_extremes_region_year"
DEFAULT_PANEL = ROOT / "data" / "processed" / "formal_region_year_panel" / "formal_region_year_panel_with_era5_v1.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "formal_region_year_panel"
DEFAULT_TAG = "formal_region_year_panel_with_era5_and_daily_extremes_v1"

DROP_COLS = ["system:index", ".geo"]
EXPECTED_DAILY_COLS = [
    "region_code",
    "region_key",
    "year",
    "warm_months",
    "era5_dataset",
    "tx90_ref_years",
    "tx90_threshold_c",
    "warm_day_count",
    "hot_day_count",
    "era5l_tx90p",
    "era5l_wsdi",
    "warm_extreme_year_flag",
    "lake_point_count",
    "sampled_point_count",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge downloaded ERA5 daily warm-extreme region-year CSVs into the formal region-year panel.")
    p.add_argument("--daily-dir", default=str(DEFAULT_DAILY_DIR))
    p.add_argument("--panel-csv", default=str(DEFAULT_PANEL))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--tag", default=DEFAULT_TAG)
    return p.parse_args()


def read_daily_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=col)

    for col in EXPECTED_DAILY_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    if "region_code" in df.columns:
        df["region_code"] = pd.to_numeric(df["region_code"], errors="coerce").astype("Int64")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "warm_extreme_year_flag" in df.columns:
        df["warm_extreme_year_flag"] = pd.to_numeric(df["warm_extreme_year_flag"], errors="coerce").astype("Int64")

    df["daily_extreme_source_file"] = path.name
    return df[EXPECTED_DAILY_COLS + ["daily_extreme_source_file"]].copy()


def main() -> None:
    args = parse_args()
    daily_dir = Path(args.daily_dir)
    panel_path = Path(args.panel_csv)
    out_dir = Path(args.out_dir)
    tag = args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    daily_files = sorted(daily_dir.glob("*.csv"))
    if not daily_files:
        raise SystemExit(f"No daily extreme CSV files found in {daily_dir}")

    daily_frames = [read_daily_table(p) for p in daily_files]
    daily = pd.concat(daily_frames, ignore_index=True)
    daily = daily.sort_values(["region_key", "year", "daily_extreme_source_file"]).reset_index(drop=True)

    duplicate_keys = int(daily.duplicated(subset=["region_key", "year"]).sum())
    if duplicate_keys:
        raise ValueError(f"Daily extreme tables contain duplicate region_key+year rows: {duplicate_keys}")

    merged = panel.merge(
        daily,
        on=["region_code", "region_key", "year", "warm_months"],
        how="left",
        suffixes=("", "_daily"),
    )
    merged["daily_extreme_joined"] = merged["era5l_tx90p_daily"].notna()

    coverage = (
        merged.groupby("region_key")
        .agg(
            panel_rows=("year", "count"),
            daily_extreme_rows=("daily_extreme_joined", "sum"),
            year_min=("year", "min"),
            year_max=("year", "max"),
        )
        .reset_index()
    )
    coverage["daily_extreme_coverage_share"] = coverage["daily_extreme_rows"] / coverage["panel_rows"]

    manifest = {
        "tag": tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "panel_csv": str(panel_path),
            "daily_dir": str(daily_dir),
            "daily_files": [p.name for p in daily_files],
        },
        "counts": {
            "panel_rows": int(len(panel)),
            "panel_regions": int(panel["region_key"].nunique()),
            "daily_rows_total": int(len(daily)),
            "daily_regions": int(daily["region_key"].nunique()),
            "duplicate_daily_region_year_keys": duplicate_keys,
            "merged_rows": int(len(merged)),
            "merged_daily_rows": int(merged["daily_extreme_joined"].sum()),
        },
    }

    daily_combined_csv = out_dir / f"{tag}_daily_extremes_combined.csv"
    merged_csv = out_dir / f"{tag}.csv"
    coverage_csv = out_dir / f"{tag}_coverage.csv"
    manifest_json = out_dir / f"{tag}_manifest.json"

    daily.to_csv(daily_combined_csv, index=False)
    merged.to_csv(merged_csv, index=False)
    coverage.to_csv(coverage_csv, index=False)
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        **manifest,
        "outputs": {
            "daily_combined_csv": str(daily_combined_csv),
            "merged_panel_csv": str(merged_csv),
            "coverage_csv": str(coverage_csv),
            "manifest_json": str(manifest_json),
        }
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
