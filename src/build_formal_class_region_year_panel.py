from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_LAKE_YEAR_CSV = ROOT / "data" / "processed" / "formal_area_results_freeze" / "freeze_20260509" / "analysis_lake_year_master.csv"
DEFAULT_LAKE_MASTER_CSV = ROOT / "data" / "processed" / "analysis_lake_master.csv"
DEFAULT_REGION_PANEL_CSV = ROOT / "data" / "processed" / "formal_region_year_panel" / "formal_region_year_panel_with_era5_and_daily_extremes_v1.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "formal_class_region_year_panel"
DEFAULT_PANEL_VERSION = "v1"

PREFERRED_CLASS_ORDER = [
    "proglacial_detached",
    "proglacial_contacted",
    "supraglacial",
]

MASTER_CLASS_COLS = [
    "lake_id",
    "lake_type",
    "lake_type_label",
    "analysis_tier",
    "main_analysis_include",
    "harmonized_class",
    "glambie_region_key",
]

REGION_PANEL_KEEP = [
    "freeze_version",
    "panel_version",
    "region_code",
    "region_key",
    "region_name",
    "final_tag",
    "year",
    "rows",
    "lake_count",
    "usable_rows",
    "usable_share_overall",
    "suspicious_lake_count",
    "total_area_change_km2_region_diag",
    "mean_ratio_change_region_diag",
    "glambie_region_code",
    "glambie_start_year",
    "glambie_end_year",
    "glambie_glacier_area_km2",
    "glambie_combined_gt",
    "glambie_combined_gt_errors",
    "glambie_combined_mwe",
    "glambie_combined_mwe_errors",
    "glambie_source_file",
    "glambie_combined_gt_anomaly",
    "glambie_combined_gt_rolling3",
    "glambie_combined_gt_rolling5",
    "glambie_combined_gt_cumulative",
    "glambie_combined_mwe_anomaly",
    "glambie_combined_mwe_rolling3",
    "glambie_combined_mwe_rolling5",
    "glambie_combined_mwe_cumulative",
    "glambie_region_id",
    "era5_joined",
    "warm_season_t2m_mean_c",
    "warm_season_t2m_anomaly_c",
    "warm_season_precip_sum_mm",
    "warm_season_precip_anomaly_mm",
    "era5_dataset",
    "era5_dataset_daily",
    "era5l_tx90p_daily",
    "era5l_wsdi_daily",
    "warm_extreme_year_flag_daily",
    "daily_extreme_joined",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the formal class-region-year panel.")
    p.add_argument("--lake-year-csv", default=str(DEFAULT_LAKE_YEAR_CSV))
    p.add_argument("--lake-master-csv", default=str(DEFAULT_LAKE_MASTER_CSV))
    p.add_argument("--region-panel-csv", default=str(DEFAULT_REGION_PANEL_CSV))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--panel-version", default=DEFAULT_PANEL_VERSION)
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {label}: {missing}")


def normalize_boolish(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    return mapped.where(mapped.notna(), series)


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    return non_null.iloc[0] if not non_null.empty else np.nan


def joined_unique(series: pd.Series) -> str:
    vals = sorted({str(v) for v in series.dropna() if str(v)})
    return "|".join(vals)


def map_harmonized_from_type(series: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    out = out.mask(series.eq("IUL"), "proglacial_detached")
    out = out.mask(series.eq("ICL"), "proglacial_contacted")
    out = out.mask(series.eq("SGL"), "supraglacial")
    return out


def map_tier_from_type(series: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    out = out.mask(series.isin(["IUL", "ICL"]), "core")
    out = out.mask(series.eq("SGL"), "supplemental")
    return out


def compute_group_anomalies(df: pd.DataFrame, metric: str, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    trend_col = f"{metric}_trend"
    anomaly_col = f"{metric}_anomaly"
    z_col = f"{metric}_anomaly_z"
    out[trend_col] = np.nan
    out[anomaly_col] = np.nan
    out[z_col] = np.nan

    for _, idx in out.groupby(group_cols, observed=False).groups.items():
        sub = out.loc[idx].sort_values("year")
        x = pd.to_numeric(sub["year"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        trend = np.full(len(sub), np.nan, dtype=float)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            trend[mask] = slope * x[mask] + intercept
        elif mask.sum() == 1:
            trend[mask] = y[mask]

        anomaly = y - trend
        valid_anom = anomaly[np.isfinite(anomaly)]
        z = np.full(len(sub), np.nan, dtype=float)
        if valid_anom.size >= 2:
            std = float(valid_anom.std(ddof=0))
            if std > 0:
                mean = float(valid_anom.mean())
                z[np.isfinite(anomaly)] = (anomaly[np.isfinite(anomaly)] - mean) / std

        out.loc[sub.index, trend_col] = trend
        out.loc[sub.index, anomaly_col] = anomaly
        out.loc[sub.index, z_col] = z

    return out


def main() -> None:
    args = parse_args()
    lake_year_path = Path(args.lake_year_csv)
    lake_master_path = Path(args.lake_master_csv)
    region_panel_path = Path(args.region_panel_csv)
    output_dir = Path(args.output_dir)
    panel_version = args.panel_version
    output_dir.mkdir(parents=True, exist_ok=True)

    lake_year = pd.read_csv(lake_year_path, low_memory=False)
    lake_master = pd.read_csv(lake_master_path)
    region_panel = pd.read_csv(region_panel_path)

    require_columns(lake_year, ["lake_id", "region_key", "year", "annual_area_to_baseline_ratio", "annual_max_area_km2", "qc_usable"], "lake_year")
    require_columns(lake_master, MASTER_CLASS_COLS, "lake_master")
    require_columns(region_panel, ["region_key", "year", "region_name", "glambie_combined_mwe_anomaly", "warm_season_t2m_anomaly_c", "warm_season_precip_anomaly_mm"], "region_panel")

    for col in ["annual_area_to_baseline_ratio", "annual_max_area_km2", "valid_area_any_km2", "water_area_median_km2"]:
        if col in lake_year.columns:
            lake_year[col] = pd.to_numeric(lake_year[col], errors="coerce")
    for col in ["year", "lake_id"]:
        lake_year[col] = pd.to_numeric(lake_year[col], errors="coerce")
        lake_master[col] = pd.to_numeric(lake_master[col], errors="coerce") if col in lake_master.columns else lake_master.get(col)
    lake_year["qc_usable"] = normalize_boolish(lake_year["qc_usable"]).astype(bool)
    lake_master["main_analysis_include"] = normalize_boolish(lake_master["main_analysis_include"]).astype(bool)
    if "era5_joined" in region_panel.columns:
        region_panel["era5_joined"] = normalize_boolish(region_panel["era5_joined"]).astype(bool)
    if "daily_extreme_joined" in region_panel.columns:
        region_panel["daily_extreme_joined"] = normalize_boolish(region_panel["daily_extreme_joined"]).astype(bool)

    master = lake_master[MASTER_CLASS_COLS].drop_duplicates("lake_id").rename(
        columns={
            "lake_type": "lake_type_master",
            "lake_type_label": "lake_type_label_master",
            "analysis_tier": "analysis_tier_master",
            "main_analysis_include": "main_analysis_include_master",
            "harmonized_class": "harmonized_class_master",
            "glambie_region_key": "glambie_region_key_master",
        }
    )
    master_class_counts = lake_master["harmonized_class"].value_counts(dropna=False).to_dict()
    master_type_counts = lake_master["lake_type"].value_counts(dropna=False).to_dict()
    lake_year_type_counts = lake_year["lake_type"].value_counts(dropna=False).to_dict() if "lake_type" in lake_year.columns else {}
    lake_year_class_counts_original = lake_year["harmonized_class"].value_counts(dropna=False).to_dict() if "harmonized_class" in lake_year.columns else {}

    merged = lake_year.merge(master, on="lake_id", how="left")
    merged["lake_type_original"] = merged.get("lake_type")
    merged["harmonized_class_original"] = merged.get("harmonized_class")
    merged["glambie_region_key_original"] = merged.get("glambie_region_key")

    merged["lake_type"] = merged["lake_type_master"].combine_first(merged.get("lake_type"))
    merged["harmonized_class"] = merged["harmonized_class_master"].combine_first(merged.get("harmonized_class"))
    merged["analysis_tier"] = merged["analysis_tier_master"]
    merged["main_analysis_include"] = merged["main_analysis_include_master"]
    merged["glambie_region_key"] = merged["glambie_region_key_master"].combine_first(merged.get("glambie_region_key"))
    merged["lake_type_label"] = merged["lake_type_label_master"]

    merged["harmonized_class"] = merged["harmonized_class"].combine_first(map_harmonized_from_type(merged["lake_type"]))
    merged["analysis_tier"] = merged["analysis_tier"].combine_first(map_tier_from_type(merged["lake_type"]))
    merged["main_analysis_include"] = merged["main_analysis_include"].where(merged["main_analysis_include"].notna(), merged["analysis_tier"].eq("core"))

    merged["region_key"] = merged["glambie_region_key"].combine_first(merged["region_key"]).astype(str)
    merged["year"] = pd.to_numeric(merged["year"], errors="coerce").astype("Int64")

    merged["class_backfilled_from_master"] = merged["harmonized_class_original"].isna() & merged["harmonized_class"].notna()
    merged["class_changed_vs_original"] = (
        merged["harmonized_class_original"].notna()
        & merged["harmonized_class"].notna()
        & merged["harmonized_class_original"].astype(str).ne(merged["harmonized_class"].astype(str))
    )
    merged["region_changed_vs_original"] = (
        merged["glambie_region_key_original"].notna()
        & merged["glambie_region_key"].notna()
        & merged["glambie_region_key_original"].astype(str).ne(merged["glambie_region_key"].astype(str))
    )

    group_cols = ["region_key", "year", "harmonized_class"]
    class_year = (
        merged.groupby(group_cols, dropna=False)
        .agg(
            freeze_version=("freeze_version", first_non_null),
            final_tag=("final_tag", first_non_null),
            lake_type=("lake_type", joined_unique),
            lake_type_label=("lake_type_label", joined_unique),
            analysis_tier=("analysis_tier", first_non_null),
            main_analysis_include=("main_analysis_include", "max"),
            total_annual_area_km2=("annual_max_area_km2", "sum"),
            usable_total_annual_area_km2=("annual_max_area_km2", lambda s: float(s[merged.loc[s.index, "qc_usable"]].sum())),
            mean_ratio=("annual_area_to_baseline_ratio", "mean"),
            median_ratio=("annual_area_to_baseline_ratio", "median"),
            zero_ratio_share=("annual_area_to_baseline_ratio", lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0) == 0).mean())),
            usable_share=("qc_usable", "mean"),
            lake_count=("lake_id", "nunique"),
            usable_lake_count=("qc_usable", lambda s: int(pd.Series(s, dtype=bool).sum())),
            raw_row_count=("lake_id", "size"),
            source_relpath=("source_relpath", first_non_null),
            class_backfilled_rows=("class_backfilled_from_master", "sum"),
            class_changed_rows=("class_changed_vs_original", "sum"),
        )
        .reset_index()
    )

    class_year["harmonized_class"] = pd.Categorical(class_year["harmonized_class"], categories=PREFERRED_CLASS_ORDER, ordered=True)
    class_year = class_year.sort_values(["region_key", "harmonized_class", "year"]).reset_index(drop=True)

    for metric in ["total_annual_area_km2", "mean_ratio", "median_ratio", "zero_ratio_share"]:
        class_year = compute_group_anomalies(class_year, metric, ["region_key", "harmonized_class"])

    region_panel_join = region_panel[[c for c in REGION_PANEL_KEEP if c in region_panel.columns]].copy()
    region_panel_join = region_panel_join.rename(columns={"panel_version": "region_panel_version", "lake_count": "region_lake_count", "rows": "region_rows", "usable_rows": "region_usable_rows"})
    class_panel = class_year.merge(region_panel_join, on=["region_key", "year"], how="left")
    class_panel.insert(1, "panel_version", panel_version)

    class_panel_csv = output_dir / f"formal_class_region_year_panel_{panel_version}.csv"
    validation_csv = output_dir / f"formal_class_region_year_panel_{panel_version}_validation.csv"
    manifest_json = output_dir / f"formal_class_region_year_panel_{panel_version}_manifest.json"

    class_panel.to_csv(class_panel_csv, index=False)

    compare = (
        class_panel.groupby(["region_key", "year"], dropna=False)
        .agg(
            class_total_area_km2=("total_annual_area_km2", "sum"),
            class_lake_count=("lake_count", "sum"),
            class_count=("harmonized_class", "nunique"),
        )
        .reset_index()
        .merge(
            region_panel[["region_key", "year", "total_annual_area_km2", "lake_count", "region_name"]].rename(
                columns={"total_annual_area_km2": "region_total_area_km2", "lake_count": "region_lake_count"}
            ),
            on=["region_key", "year"],
            how="left",
        )
    )
    compare["total_area_diff_km2"] = compare["class_total_area_km2"] - compare["region_total_area_km2"]
    compare["lake_count_diff"] = compare["class_lake_count"] - compare["region_lake_count"]
    compare["abs_total_area_diff_km2"] = compare["total_area_diff_km2"].abs()
    compare = compare.sort_values(["region_key", "year"]).reset_index(drop=True)
    compare.to_csv(validation_csv, index=False)

    missing_class_rows = int(class_panel["harmonized_class"].isna().sum())
    duplicate_count = int(class_panel.duplicated(subset=["region_key", "year", "harmonized_class"]).sum())
    backfilled_rows = int(merged["class_backfilled_from_master"].sum())
    changed_rows = int(merged["class_changed_vs_original"].sum())

    observed_classes = sorted([str(v) for v in class_panel["harmonized_class"].dropna().unique().tolist()])
    expected_classes = sorted([str(v) for v in lake_master["harmonized_class"].dropna().unique().tolist()])
    missing_expected_classes = sorted(set(expected_classes) - set(observed_classes))

    manifest = {
        "panel_version": panel_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "lake_year_csv": str(lake_year_path),
            "lake_master_csv": str(lake_master_path),
            "region_panel_csv": str(region_panel_path),
        },
        "outputs": {
            "class_panel_csv": str(class_panel_csv),
            "validation_csv": str(validation_csv),
        },
        "shape": {
            "rows": int(len(class_panel)),
            "regions": int(class_panel["region_key"].nunique()),
            "classes": observed_classes,
            "year_min": int(class_panel["year"].min()),
            "year_max": int(class_panel["year"].max()),
            "duplicate_region_year_class_rows": duplicate_count,
        },
        "class_reconciliation": {
            "backfilled_rows": backfilled_rows,
            "changed_rows_vs_original": changed_rows,
            "missing_class_rows_after_backfill": missing_class_rows,
        },
        "class_coverage": {
            "expected_classes_from_master": expected_classes,
            "observed_classes_in_panel": observed_classes,
            "missing_expected_classes": missing_expected_classes,
            "master_harmonized_class_counts": master_class_counts,
            "master_lake_type_counts": master_type_counts,
            "lake_year_lake_type_counts": lake_year_type_counts,
            "lake_year_harmonized_class_counts_original": lake_year_class_counts_original,
        },
        "consistency_checks": {
            "max_abs_region_total_area_diff_km2": float(compare["abs_total_area_diff_km2"].max()) if not compare.empty else 0.0,
            "nonzero_region_total_area_diff_rows": int((compare["abs_total_area_diff_km2"] > 1e-9).sum()) if not compare.empty else 0,
            "nonzero_region_lake_count_diff_rows": int((compare["lake_count_diff"] != 0).sum()) if not compare.empty else 0,
        },
        "notes": {
            "missing_expected_classes_reason": "If a class exists in analysis_lake_master but is absent from analysis_lake_year_master, it will remain absent from the class-region-year panel even after metadata backfill.",
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
