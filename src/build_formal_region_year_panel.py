from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_FREEZE_DIR = ROOT / "data" / "processed" / "formal_area_results_freeze" / "freeze_20260509"
DEFAULT_GLAMBIE_DIR = ROOT / "data" / "raw" / "GlaMBIE_Data" / "glambie_results_20240716" / "calendar_years"
DEFAULT_REGION_MANIFEST = ROOT / "data" / "prepared" / "region_batches" / "region_manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "formal_region_year_panel"
DEFAULT_PANEL_VERSION = "v1"


CLIMATE_RESERVED_COLS = [
    "era5l_temp_mean_c",
    "era5l_temp_anom_c",
    "era5l_precip_mm",
    "era5l_snowfall_mmwe",
    "era5l_solid_precip_frac",
    "era5l_pdd_c_days",
    "era5l_tx90p",
    "era5l_wsdi",
    "warm_extreme_year_flag",
    "era5l_data_start_year",
    "era5l_data_end_year",
    "era5l_source_version",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the formal region-year panel from frozen area results and GlaMBIE.")
    p.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE_DIR))
    p.add_argument("--glambie-dir", default=str(DEFAULT_GLAMBIE_DIR))
    p.add_argument("--region-manifest", default=str(DEFAULT_REGION_MANIFEST))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--panel-version", default=DEFAULT_PANEL_VERSION)
    return p.parse_args()


def build_glambie_normalized(glambie_dir: Path, included_regions: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(glambie_dir.glob("*.csv")):
        stem = csv_path.stem
        if stem == "0_global":
            continue
        parts = stem.split("_", 1)
        if len(parts) != 2:
            continue
        region_code_raw, region_key = parts
        if region_key not in included_regions:
            continue

        df = pd.read_csv(csv_path)
        df["year"] = pd.to_numeric(df["start_dates"], errors="coerce").astype("Int64")
        df["glambie_start_year"] = pd.to_numeric(df["start_dates"], errors="coerce")
        df["glambie_end_year"] = pd.to_numeric(df["end_dates"], errors="coerce")
        df["glambie_region_code"] = int(region_code_raw)
        df["region_key"] = region_key
        df["glambie_source_file"] = csv_path.name
        df["glambie_glacier_area_km2"] = pd.to_numeric(df["glacier_area"], errors="coerce")
        df["glambie_combined_gt"] = pd.to_numeric(df["combined_gt"], errors="coerce")
        df["glambie_combined_gt_errors"] = pd.to_numeric(df["combined_gt_errors"], errors="coerce")
        df["glambie_combined_mwe"] = pd.to_numeric(df["combined_mwe"], errors="coerce")
        df["glambie_combined_mwe_errors"] = pd.to_numeric(df["combined_mwe_errors"], errors="coerce")
        frames.append(
            df[
                [
                    "region_key",
                    "year",
                    "glambie_region_code",
                    "glambie_start_year",
                    "glambie_end_year",
                    "glambie_glacier_area_km2",
                    "glambie_combined_gt",
                    "glambie_combined_gt_errors",
                    "glambie_combined_mwe",
                    "glambie_combined_mwe_errors",
                    "glambie_source_file",
                ]
            ].copy()
        )

    glambie = pd.concat(frames, ignore_index=True).sort_values(["region_key", "year"]).reset_index(drop=True)

    duration = glambie["glambie_end_year"] - glambie["glambie_start_year"]
    if not duration.dropna().round(6).eq(1.0).all():
        raise ValueError("Found GlaMBIE calendar-year rows with duration not equal to 1 year.")

    for metric in ["glambie_combined_gt", "glambie_combined_mwe"]:
        glambie[f"{metric}_anomaly"] = glambie.groupby("region_key")[metric].transform(lambda s: s - s.mean())
        glambie[f"{metric}_rolling3"] = (
            glambie.groupby("region_key")[metric].transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        glambie[f"{metric}_rolling5"] = (
            glambie.groupby("region_key")[metric].transform(lambda s: s.rolling(5, min_periods=1).mean())
        )
        glambie[f"{metric}_cumulative"] = glambie.groupby("region_key")[metric].cumsum()

    return glambie


def main() -> None:
    args = parse_args()
    freeze_dir = Path(args.freeze_dir)
    glambie_dir = Path(args.glambie_dir)
    region_manifest_path = Path(args.region_manifest)
    output_dir = Path(args.output_dir)
    panel_version = args.panel_version
    output_dir.mkdir(parents=True, exist_ok=True)

    area_master_path = freeze_dir / "analysis_region_year_master.csv"
    quality_path = freeze_dir / "quality_region_diagnosis_index.csv"
    manifest_path = freeze_dir / "freeze_manifest.json"

    area_master = pd.read_csv(area_master_path)
    quality = pd.read_csv(quality_path)
    freeze_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    region_manifest = pd.read_csv(region_manifest_path)

    area_master["year"] = pd.to_numeric(area_master["year"], errors="coerce").astype("Int64")
    quality["year"] = pd.to_numeric(quality["year"], errors="coerce").astype("Int64")
    included_regions = sorted(region_manifest["glambie_region_key"].dropna().astype(str).unique().tolist())

    glambie = build_glambie_normalized(glambie_dir, included_regions)

    keep_area = [
        "freeze_version",
        "final_tag",
        "region_id",
        "region_name",
        "region_code",
        "region_key",
        "year",
        "rows",
        "lake_count",
        "valid_area_rows",
        "usable_rows",
        "mean_annual_area_km2",
        "median_annual_area_km2",
        "total_annual_area_km2",
        "mean_ratio_change",
        "median_ratio_change",
        "mean_ratio_change_region_diag",
        "median_ratio_change_region_diag",
        "total_area_change_km2_region_diag",
        "mean_ratio_change_including_zero",
        "zero_area_share",
        "zero_ratio_share",
        "usable_share_overall",
        "usable_share_nonzero",
    ]
    area = area_master[[c for c in keep_area if c in area_master.columns]].copy()

    quality_keep = [
        "region_key",
        "year",
        "suspicious_lake_count",
        "usable_rows",
        "lake_count",
        "usable_share_overall",
    ]
    quality_sub = quality[[c for c in quality_keep if c in quality.columns]].copy()
    quality_sub = quality_sub.rename(
        columns={
            "usable_rows": "quality_usable_rows",
            "lake_count": "quality_lake_count",
            "usable_share_overall": "quality_usable_share_overall",
        }
    )

    panel = area.merge(quality_sub, on=["region_key", "year"], how="left")
    panel = panel.merge(
        region_manifest[["glambie_region_id", "glambie_region_key"]].rename(columns={"glambie_region_key": "region_key"}),
        on="region_key",
        how="left",
    )
    panel = panel.merge(glambie, on=["region_key", "year"], how="left")

    climate_joined = False
    climate_source = None
    climate_path = freeze_dir / "analysis_region_year_climate_joined.csv"
    if climate_path.exists():
        climate = pd.read_csv(climate_path)
        climate["year"] = pd.to_numeric(climate["year"], errors="coerce").astype("Int64")
        climate_cols = [c for c in climate.columns if c not in {"region_key", "year"}]
        panel = panel.merge(climate[["region_key", "year", *climate_cols]], on=["region_key", "year"], how="left")
        climate_joined = True
        climate_source = str(climate_path)
    else:
        for col in CLIMATE_RESERVED_COLS:
            if col not in panel.columns:
                panel[col] = pd.NA

    panel.insert(1, "panel_version", panel_version)
    panel["era5_joined"] = climate_joined
    panel["era5_source_file"] = climate_source

    panel = panel.sort_values(["region_key", "year"]).reset_index(drop=True)

    output_csv = output_dir / f"formal_region_year_panel_{panel_version}.csv"
    manifest_json = output_dir / f"formal_region_year_panel_{panel_version}_manifest.json"
    panel.to_csv(output_csv, index=False)

    payload = {
        "panel_version": panel_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "freeze_dir": str(freeze_dir),
            "glambie_dir": str(glambie_dir),
            "region_manifest_csv": str(region_manifest_path),
            "freeze_manifest": freeze_manifest,
            "climate_joined": climate_joined,
            "climate_source": climate_source,
        },
        "outputs": {"panel_csv": str(output_csv)},
        "shape": {
            "rows": int(len(panel)),
            "regions": int(panel["region_key"].nunique()),
            "year_min": int(panel["year"].min()),
            "year_max": int(panel["year"].max()),
        },
        "missing_glambie_rows": int(panel["glambie_combined_mwe"].isna().sum()),
    }
    manifest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
