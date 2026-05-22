from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
TAG = "central_asia_2000_2024_final"
ANALYSIS_DIR = ROOT / "data" / "processed" / "analysis" / TAG
SKELETON_DIR = ROOT / "data" / "processed" / "annual_area_skeleton"
MANUSCRIPT_TABLES = ROOT / "manuscript" / "tables"
MANUSCRIPT_FIG = ROOT / "manuscript" / "figures_for_paper"
MANIFEST_DIR = MANUSCRIPT_FIG / "package_manifest"


def main() -> None:
    MANUSCRIPT_TABLES.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    year_df = pd.read_csv(SKELETON_DIR / f"{TAG}_skeleton_year_summary.csv").sort_values("year")
    trend_df = pd.read_csv(ANALYSIS_DIR / f"{TAG}_trend_summary.csv")
    anomaly_df = pd.read_csv(ANALYSIS_DIR / f"{TAG}_year_anomalies.csv").sort_values("year")
    class_year_df = pd.read_csv(ANALYSIS_DIR / f"{TAG}_class_year_stats.csv").sort_values(["harmonized_class", "year"])
    class_trend_df = pd.read_csv(ANALYSIS_DIR / f"{TAG}_class_trends.csv").sort_values(["harmonized_class", "metric"])

    year_table = year_df[
        [
            "year",
            "usable_share",
            "total_annual_area_km2",
            "usable_total_annual_area_km2",
            "usable_median_area_ratio",
            "usable_mean_area_ratio",
            "median_image_count",
            "low_ratio_rows",
        ]
    ].copy()
    year_table["usable_share_pct"] = year_table["usable_share"] * 100
    year_table = year_table[
        [
            "year",
            "usable_share",
            "usable_share_pct",
            "total_annual_area_km2",
            "usable_total_annual_area_km2",
            "usable_median_area_ratio",
            "usable_mean_area_ratio",
            "median_image_count",
            "low_ratio_rows",
        ]
    ]

    trend_table = trend_df[
        ["segment", "metric", "year_start", "year_end", "slope", "intercept", "r2", "p_value", "n"]
    ].copy()
    class_trend_table = class_trend_df[
        ["harmonized_class", "metric", "slope", "intercept", "r2", "p_value", "n"]
    ].copy()

    figure_year = year_df.copy()
    figure_anomaly = anomaly_df[
        [
            "year",
            "total_annual_area_km2",
            "total_annual_area_km2_trend",
            "total_annual_area_km2_anomaly",
            "total_annual_area_km2_anomaly_z",
            "median_ratio",
            "median_ratio_trend",
            "median_ratio_anomaly",
            "median_ratio_anomaly_z",
            "zero_ratio_share",
            "zero_ratio_share_trend",
            "zero_ratio_share_anomaly",
            "zero_ratio_share_anomaly_z",
            "usable_share",
        ]
    ].copy()
    figure_class = class_year_df.copy()

    year_table_path = MANUSCRIPT_TABLES / f"{TAG}_year_summary_table.csv"
    trend_table_path = MANUSCRIPT_TABLES / f"{TAG}_trend_summary_table.csv"
    class_trend_table_path = MANUSCRIPT_TABLES / f"{TAG}_class_trend_table.csv"

    figure_year_path = MANUSCRIPT_FIG / f"{TAG}_figure_year_source.csv"
    figure_anomaly_path = MANUSCRIPT_FIG / f"{TAG}_figure_anomaly_source.csv"
    figure_class_path = MANUSCRIPT_FIG / f"{TAG}_figure_class_source.csv"

    year_table.to_csv(year_table_path, index=False, encoding="utf-8-sig")
    trend_table.to_csv(trend_table_path, index=False, encoding="utf-8-sig")
    class_trend_table.to_csv(class_trend_table_path, index=False, encoding="utf-8-sig")

    figure_year.to_csv(figure_year_path, index=False, encoding="utf-8-sig")
    figure_anomaly.to_csv(figure_anomaly_path, index=False, encoding="utf-8-sig")
    figure_class.to_csv(figure_class_path, index=False, encoding="utf-8-sig")

    strongest_area_peak = figure_anomaly.sort_values("total_annual_area_km2_anomaly", ascending=False).iloc[0]
    strongest_area_drop = figure_anomaly.sort_values("total_annual_area_km2_anomaly").iloc[0]
    strongest_ratio_peak = figure_anomaly.sort_values("median_ratio_anomaly", ascending=False).iloc[0]
    strongest_ratio_drop = figure_anomaly.sort_values("median_ratio_anomaly").iloc[0]

    payload = {
        "tag": TAG,
        "year_summary_table_csv": str(year_table_path),
        "trend_summary_table_csv": str(trend_table_path),
        "class_trend_table_csv": str(class_trend_table_path),
        "figure_year_source_csv": str(figure_year_path),
        "figure_anomaly_source_csv": str(figure_anomaly_path),
        "figure_class_source_csv": str(figure_class_path),
        "highlights": {
            "largest_positive_area_anomaly_year": int(strongest_area_peak["year"]),
            "largest_positive_area_anomaly_km2": float(strongest_area_peak["total_annual_area_km2_anomaly"]),
            "largest_negative_area_anomaly_year": int(strongest_area_drop["year"]),
            "largest_negative_area_anomaly_km2": float(strongest_area_drop["total_annual_area_km2_anomaly"]),
            "largest_positive_ratio_anomaly_year": int(strongest_ratio_peak["year"]),
            "largest_positive_ratio_anomaly": float(strongest_ratio_peak["median_ratio_anomaly"]),
            "largest_negative_ratio_anomaly_year": int(strongest_ratio_drop["year"]),
            "largest_negative_ratio_anomaly": float(strongest_ratio_drop["median_ratio_anomaly"]),
        },
    }
    manifest_path = MANIFEST_DIR / f"{TAG}_paper_package_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**payload, "manifest_json": str(manifest_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
