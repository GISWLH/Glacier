from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
YEAR_SUMMARY = ROOT / "data" / "processed" / "annual_area_skeleton" / "central_asia_key_years_or2024_skeleton_year_summary.csv"
LAKE_YEAR = ROOT / "data" / "processed" / "annual_area_skeleton" / "central_asia_key_years_or2024_skeleton_lake_year.csv"
OUT_DIR = ROOT / "data" / "processed" / "analysis" / "central_asia_or2024"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze Central Asia OR-2024 skeleton without plotting.")
    p.add_argument("--year-summary", default=str(YEAR_SUMMARY))
    p.add_argument("--lake-year", default=str(LAKE_YEAR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    year_df = pd.read_csv(args.year_summary)
    lake_year = pd.read_csv(args.lake_year)

    # Normalize numeric columns
    for col in ["total_annual_area_km2", "median_ratio", "mean_ratio", "zero_ratio_share", "usable_share"]:
        if col in year_df.columns:
            year_df[col] = pd.to_numeric(year_df[col], errors="coerce")
    for col in ["annual_max_area_km2", "annual_area_to_baseline_ratio"]:
        if col in lake_year.columns:
            lake_year[col] = pd.to_numeric(lake_year[col], errors="coerce")
    if "qc_usable" in lake_year.columns:
        lake_year["qc_usable"] = lake_year["qc_usable"].astype(str).str.lower().map({"true": True, "false": False}).fillna(
            False
        )

    # Define segments: early (2000-2010), mid (2010-2015), late (2015-2024), full (2000-2024)
    segments = [
        ("full", 2000, 2024),
        ("early", 2000, 2010),
        ("mid", 2010, 2015),
        ("late", 2015, 2024),
    ]

    trend_tables = []
    for metric in ["total_annual_area_km2", "median_ratio", "mean_ratio", "zero_ratio_share"]:
        trend_tables.append(segment_trends(year_df, metric, segments))
    trend_summary = pd.concat(trend_tables, ignore_index=True)
    trend_summary.to_csv(out_dir / "central_asia_trend_summary.csv", index=False)

    # Anomalies
    anomaly_df = compute_anomalies(year_df, "total_annual_area_km2")
    anomaly_df = compute_anomalies(anomaly_df, "median_ratio")
    anomaly_df = compute_anomalies(anomaly_df, "mean_ratio")
    anomaly_df = compute_anomalies(anomaly_df, "zero_ratio_share")
    anomaly_df.to_csv(out_dir / "central_asia_year_anomalies.csv", index=False)

    # Class-level summaries
    class_year = (
        lake_year.groupby(["year", "harmonized_class"])
        .agg(
            total_area_km2=("annual_max_area_km2", "sum"),
            median_ratio=("annual_area_to_baseline_ratio", "median"),
            mean_ratio=("annual_area_to_baseline_ratio", "mean"),
            zero_ratio_share=("annual_area_to_baseline_ratio", lambda s: float((s.fillna(0) == 0).mean())),
            usable_share=("qc_usable", "mean"),
            lake_count=("lake_id", "nunique"),
        )
        .reset_index()
        .sort_values(["harmonized_class", "year"])
    )
    class_year.to_csv(out_dir / "central_asia_class_year_stats.csv", index=False)

    # Class-level trends on total_area and median_ratio
    class_trend_rows = []
    for cls in class_year["harmonized_class"].dropna().unique():
        sub = class_year[class_year["harmonized_class"] == cls]
        for metric in ["total_area_km2", "median_ratio"]:
            stats = linregress_safe(sub["year"].values, sub[metric].values)
            class_trend_rows.append({"harmonized_class": cls, "metric": metric, **stats})
    class_trends = pd.DataFrame(class_trend_rows)
    class_trends.to_csv(out_dir / "central_asia_class_trends.csv", index=False)

    payload = {
        "trend_summary_csv": str(out_dir / "central_asia_trend_summary.csv"),
        "year_anomalies_csv": str(out_dir / "central_asia_year_anomalies.csv"),
        "class_year_stats_csv": str(out_dir / "central_asia_class_year_stats.csv"),
        "class_trends_csv": str(out_dir / "central_asia_class_trends.csv"),
    }
    (out_dir / "central_asia_analysis_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
