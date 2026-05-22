from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze one region skeleton without plotting.")
    p.add_argument("--region", required=True, help="Region key, e.g. central_asia")
    p.add_argument("--tag", default="", help="Output tag prefix (defaults to region)")
    p.add_argument(
        "--year-summary",
        default="",
        help="Path to skeleton year summary CSV (defaults to data/processed/annual_area_skeleton/{tag}_skeleton_year_summary.csv)",
    )
    p.add_argument(
        "--lake-year",
        default="",
        help="Path to skeleton lake-year CSV (defaults to data/processed/annual_area_skeleton/{tag}_skeleton_lake_year.csv)",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Output directory (defaults to data/processed/analysis/{tag})",
    )
    return p.parse_args()


def linregress_safe(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "p_value": np.nan, "n": int(len(x))}
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    p_value = np.nan
    try:
        from scipy.stats import linregress as _lr  # type: ignore

        p_value = float(_lr(x, y).pvalue)
    except Exception:
        p_value = np.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "p_value": p_value, "n": int(len(x))}


def segment_trends(df: pd.DataFrame, metric: str, segments: list[tuple[str, int, int]]) -> pd.DataFrame:
    rows = []
    for label, start, end in segments:
        sub = df[(df["year"] >= start) & (df["year"] <= end)].copy()
        stats = linregress_safe(sub["year"].values, sub[metric].values)
        rows.append(
            {
                "segment": label,
                "metric": metric,
                "year_start": start,
                "year_end": end,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def compute_anomalies(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    stats = linregress_safe(df["year"].values, df[metric].values)
    df = df.copy()
    df[f"{metric}_trend"] = stats["slope"] * df["year"] + stats["intercept"]
    df[f"{metric}_anomaly"] = df[metric] - df[f"{metric}_trend"]
    df[f"{metric}_anomaly_z"] = (df[f"{metric}_anomaly"] - df[f"{metric}_anomaly"].mean()) / df[
        f"{metric}_anomaly"
    ].std(ddof=0)
    return df


def normalize_boolish(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    return mapped.where(mapped.notna(), series).astype(bool)


def resolve_class_column(lake_year: pd.DataFrame) -> tuple[str | None, str]:
    candidates = [
        ("harmonized_class", "harmonized_class"),
        ("lake_type", "lake_type_fallback"),
    ]
    for col, source in candidates:
        if col in lake_year.columns and lake_year[col].notna().any():
            return col, source
    return None, "no_class_available"


def main() -> None:
    args = parse_args()
    tag = args.tag or args.region
    year_summary_path = Path(
        args.year_summary
        or ROOT / "data" / "processed" / "annual_area_skeleton" / f"{tag}_skeleton_year_summary.csv"
    )
    lake_year_path = Path(
        args.lake_year
        or ROOT / "data" / "processed" / "annual_area_skeleton" / f"{tag}_skeleton_lake_year.csv"
    )
    out_dir = Path(args.out_dir or ROOT / "data" / "processed" / "analysis" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    year_df = pd.read_csv(year_summary_path)
    lake_year = pd.read_csv(lake_year_path)

    year_metric_aliases = {
        "total_annual_area_km2": ["total_annual_area_km2"],
        "median_ratio": ["median_ratio", "usable_median_area_ratio"],
        "mean_ratio": ["mean_ratio", "usable_mean_area_ratio"],
        "zero_ratio_share": ["zero_ratio_share"],
        "usable_share": ["usable_share"],
    }
    resolved_metrics: dict[str, str] = {}
    for canonical, aliases in year_metric_aliases.items():
        for candidate in aliases:
            if candidate in year_df.columns:
                year_df[candidate] = pd.to_numeric(year_df[candidate], errors="coerce")
                if candidate != canonical:
                    year_df[canonical] = year_df[candidate]
                resolved_metrics[canonical] = candidate
                break
    for col in ["annual_max_area_km2", "annual_area_to_baseline_ratio"]:
        if col in lake_year.columns:
            lake_year[col] = pd.to_numeric(lake_year[col], errors="coerce")
    if "qc_usable" in lake_year.columns:
        lake_year["qc_usable"] = normalize_boolish(lake_year["qc_usable"])

    if "zero_ratio_share" not in year_df.columns:
        zero_ratio = (
            lake_year.groupby("year")["annual_area_to_baseline_ratio"]
            .apply(lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0) == 0).mean()))
            .reset_index(name="zero_ratio_share")
        )
        year_df = year_df.merge(zero_ratio, on="year", how="left")
        resolved_metrics["zero_ratio_share"] = "derived_from_lake_year"

    year_min = int(year_df["year"].min()) if year_df["year"].notna().any() else 2000
    year_max = int(year_df["year"].max()) if year_df["year"].notna().any() else 2024
    mid = (year_min + year_max) // 2
    segments = [
        ("full", year_min, year_max),
        ("early", year_min, mid),
        ("late", mid, year_max),
    ]

    trend_tables = []
    for metric in ["total_annual_area_km2", "median_ratio", "mean_ratio", "zero_ratio_share"]:
        trend_tables.append(segment_trends(year_df, metric, segments))
    trend_summary = pd.concat(trend_tables, ignore_index=True)
    trend_summary.to_csv(out_dir / f"{tag}_trend_summary.csv", index=False)

    anomaly_df = compute_anomalies(year_df, "total_annual_area_km2")
    anomaly_df = compute_anomalies(anomaly_df, "median_ratio")
    anomaly_df = compute_anomalies(anomaly_df, "mean_ratio")
    anomaly_df = compute_anomalies(anomaly_df, "zero_ratio_share")
    anomaly_df.to_csv(out_dir / f"{tag}_year_anomalies.csv", index=False)

    class_col, class_source = resolve_class_column(lake_year)
    if class_col is not None:
        class_year = (
            lake_year.groupby(["year", class_col])
            .agg(
                total_area_km2=("annual_max_area_km2", "sum"),
                median_ratio=("annual_area_to_baseline_ratio", "median"),
                mean_ratio=("annual_area_to_baseline_ratio", "mean"),
                zero_ratio_share=("annual_area_to_baseline_ratio", lambda s: float((s.fillna(0) == 0).mean())),
                usable_share=("qc_usable", "mean"),
                lake_count=("lake_id", "nunique"),
            )
            .reset_index()
            .rename(columns={class_col: "harmonized_class"})
            .sort_values(["harmonized_class", "year"])
        )
        class_year.insert(1, "class_source", class_source)
    else:
        class_year = pd.DataFrame(
            columns=[
                "year",
                "class_source",
                "harmonized_class",
                "total_area_km2",
                "median_ratio",
                "mean_ratio",
                "zero_ratio_share",
                "usable_share",
                "lake_count",
            ]
        )
    class_year.to_csv(out_dir / f"{tag}_class_year_stats.csv", index=False)

    class_trend_rows = []
    for cls in class_year["harmonized_class"].dropna().unique():
        sub = class_year[class_year["harmonized_class"] == cls]
        for metric in ["total_area_km2", "median_ratio"]:
            stats = linregress_safe(sub["year"].values, sub[metric].values)
            class_trend_rows.append(
                {
                    "class_source": class_source,
                    "harmonized_class": cls,
                    "metric": metric,
                    **stats,
                }
            )
    class_trends = pd.DataFrame(class_trend_rows)
    class_trends.to_csv(out_dir / f"{tag}_class_trends.csv", index=False)

    payload = {
        "resolved_year_metrics": resolved_metrics,
        "class_source": class_source,
        "trend_summary_csv": str(out_dir / f"{tag}_trend_summary.csv"),
        "year_anomalies_csv": str(out_dir / f"{tag}_year_anomalies.csv"),
        "class_year_stats_csv": str(out_dir / f"{tag}_class_year_stats.csv"),
        "class_trends_csv": str(out_dir / f"{tag}_class_trends.csv"),
    }
    (out_dir / f"{tag}_analysis_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
