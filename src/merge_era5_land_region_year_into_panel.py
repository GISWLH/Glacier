from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_ERA5_DIR = ROOT / "data" / "interim" / "era5_land_region_year"
DEFAULT_PANEL = ROOT / "data" / "processed" / "formal_region_year_panel" / "formal_region_year_panel_v1.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "formal_region_year_panel"
DEFAULT_TAG = "formal_region_year_panel_with_era5_v1"


DROP_COLS = ["system:index", ".geo"]
EXPECTED_ERA5_COLS = [
    "region_code",
    "region_key",
    "year",
    "warm_months",
    "era5_month_count",
    "warm_season_t2m_mean_c",
    "warm_season_t2m_anomaly_c",
    "warm_season_precip_sum_mm",
    "warm_season_precip_anomaly_mm",
    "climatology_ref_years",
    "era5_dataset",
    "lake_point_count",
    "sampled_point_count",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge downloaded ERA5-Land region-year CSVs into the formal region-year panel.")
    p.add_argument("--era5-dir", default=str(DEFAULT_ERA5_DIR))
    p.add_argument("--panel-csv", default=str(DEFAULT_PANEL))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--tag", default=DEFAULT_TAG)
    return p.parse_args()


def read_era5_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=col)

    for col in EXPECTED_ERA5_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    if "region_code" in df.columns:
        df["region_code"] = pd.to_numeric(df["region_code"], errors="coerce").astype("Int64")
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    df["source_file"] = path.name
    return df[EXPECTED_ERA5_COLS + ["source_file"]].copy()


def main() -> None:
    args = parse_args()
    era5_dir = Path(args.era5_dir)
    panel_path = Path(args.panel_csv)
    out_dir = Path(args.out_dir)
    tag = args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    era5_files = sorted(era5_dir.glob("*.csv"))
    if not era5_files:
        raise SystemExit(f"No ERA5 CSV files found in {era5_dir}")

    era5_frames = [read_era5_table(p) for p in era5_files]
    era5 = pd.concat(era5_frames, ignore_index=True)
    era5 = era5.sort_values(["region_key", "year", "source_file"]).reset_index(drop=True)

    duplicate_keys = era5.duplicated(subset=["region_key", "year"]).sum()
    if duplicate_keys:
        raise ValueError(f"ERA5 tables contain duplicate region_key+year rows: {duplicate_keys}")

    merged = panel.merge(era5, on=["region_code", "region_key", "year"], how="left")
    merged["era5_joined"] = merged["warm_season_t2m_mean_c"].notna()

    era5_coverage = (
        merged.groupby("region_key")
        .agg(
            panel_rows=("year", "count"),
            era5_rows=("era5_joined", "sum"),
            year_min=("year", "min"),
            year_max=("year", "max"),
        )
        .reset_index()
    )
    era5_coverage["era5_coverage_share"] = era5_coverage["era5_rows"] / era5_coverage["panel_rows"]

    manifest = {
        "tag": tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "panel_csv": str(panel_path),
            "era5_dir": str(era5_dir),
            "era5_files": [p.name for p in era5_files],
        },
        "counts": {
            "panel_rows": int(len(panel)),
            "panel_regions": int(panel["region_key"].nunique()),
            "era5_rows_total": int(len(era5)),
            "era5_regions": int(era5["region_key"].nunique()),
            "duplicate_era5_region_year_keys": int(duplicate_keys),
            "merged_rows": int(len(merged)),
            "merged_era5_rows": int(merged["era5_joined"].sum()),
        },
    }

    era5_combined_csv = out_dir / f"{tag}_era5_combined.csv"
    merged_csv = out_dir / f"{tag}.csv"
    coverage_csv = out_dir / f"{tag}_coverage.csv"
    manifest_json = out_dir / f"{tag}_manifest.json"

    era5.to_csv(era5_combined_csv, index=False)
    merged.to_csv(merged_csv, index=False)
    era5_coverage.to_csv(coverage_csv, index=False)
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        **manifest,
        "outputs": {
            "era5_combined_csv": str(era5_combined_csv),
            "merged_panel_csv": str(merged_csv),
            "coverage_csv": str(coverage_csv),
            "manifest_json": str(manifest_json),
        }
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
