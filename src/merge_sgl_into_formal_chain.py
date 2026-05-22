from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"E:\Glacier")
DEFAULT_REGISTRY = ROOT / "data" / "processed" / "region_final_results_summary.csv"
DEFAULT_SGL_QC_FULL = ROOT / "data" / "interim" / "annual_area_qc" / "sgl_incremental_all_qc_full.csv"
DEFAULT_SUMMARY_JSON = ROOT / "data" / "processed" / "sgl_formal_chain_merge_summary.json"

BOOL_COLS = ["qc_usable", "qc_enough_images", "qc_enough_coverage"]
NUMERIC_COLS = [
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
    "chunk_start",
    "chunk_size",
    "annual_max_pixel_count",
    "valid_pixel_any_count",
    "l5_count",
    "l7_count",
    "l8_count",
    "l9_count",
    "source_year_count",
]
HELPER_DROP_COLS = [
    "__source_file",
    "region_key_from_name",
    "year_from_name",
    "chunk_start_from_name",
    "chunk_size_from_name",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge SGL incremental results into the formal area chain.")
    p.add_argument("--registry-csv", default=str(DEFAULT_REGISTRY))
    p.add_argument("--sgl-qc-full", default=str(DEFAULT_SGL_QC_FULL))
    p.add_argument("--summary-json", default=str(DEFAULT_SUMMARY_JSON))
    p.add_argument("--regions", default="", help="Optional comma-separated region keys.")
    return p.parse_args()


def normalize_boolish(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    return mapped.where(mapped.notna(), series).fillna(False).astype(bool)


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
        rows.append({"segment": label, "metric": metric, "year_start": start, "year_end": end, **stats})
    return pd.DataFrame(rows)


def compute_anomalies(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    stats = linregress_safe(df["year"].values, df[metric].values)
    df = df.copy()
    df[f"{metric}_trend"] = stats["slope"] * df["year"] + stats["intercept"]
    df[f"{metric}_anomaly"] = df[metric] - df[f"{metric}_trend"]
    std = df[f"{metric}_anomaly"].std(ddof=0)
    if pd.isna(std) or std == 0:
        df[f"{metric}_anomaly_z"] = np.nan
    else:
        df[f"{metric}_anomaly_z"] = (df[f"{metric}_anomaly"] - df[f"{metric}_anomaly"].mean()) / std
    return df


def resolve_class_column(lake_year: pd.DataFrame) -> tuple[str | None, str]:
    candidates = [
        ("harmonized_class", "harmonized_class"),
        ("lake_type", "lake_type_fallback"),
    ]
    for col, source in candidates:
        if col in lake_year.columns and lake_year[col].notna().any():
            return col, source
    return None, "no_class_available"


def parse_region_list(raw: str) -> set[str] | None:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return set(values) if values else None


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_combined_lake_year(core_df: pd.DataFrame, sgl_df: pd.DataFrame, region_key: str) -> pd.DataFrame:
    core = core_df.copy()
    sgl = sgl_df.copy()

    if "source_file" not in sgl.columns and "__source_file" in sgl.columns:
        sgl = sgl.rename(columns={"__source_file": "source_file"})
    sgl = sgl.drop(columns=[c for c in HELPER_DROP_COLS if c in sgl.columns], errors="ignore")

    for df in [core, sgl]:
        if "glambie_region_key" in df.columns:
            df["glambie_region_key"] = df["glambie_region_key"].fillna(region_key)
        else:
            df["glambie_region_key"] = region_key
        for col in BOOL_COLS:
            if col in df.columns:
                df[col] = normalize_boolish(df[col])
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    if "lake_type" in core.columns and core["lake_type"].astype(str).eq("SGL").any():
        core = core[~core["lake_type"].astype(str).eq("SGL")].copy()

    combined = pd.concat([core, sgl], ignore_index=True, sort=False)
    dup = combined.duplicated(["lake_id", "year"], keep=False)
    if dup.any():
        sample = combined.loc[dup, ["lake_id", "year", "lake_type", "source_file"]].head(20)
        raise ValueError(f"Found duplicate lake_id/year rows for {region_key}:\n{sample.to_string(index=False)}")

    for col in BOOL_COLS:
        if col in combined.columns:
            combined[col] = normalize_boolish(combined[col])
    if "year" in combined.columns:
        combined["year"] = pd.to_numeric(combined["year"], errors="coerce").astype("Int64")
    for col in NUMERIC_COLS:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined["flag_low_image_count"] = combined["image_count"] < 3
    combined["flag_low_valid_fraction"] = combined["baseline_valid_area_fraction"] < 0.7
    combined["flag_ratio_lt_0_1"] = combined["annual_area_to_baseline_ratio"] < 0.1
    combined["flag_ratio_gt_3"] = combined["annual_area_to_baseline_ratio"] > 3

    return combined.sort_values(["lake_id", "year", "source_file"], na_position="last").reset_index(drop=True)


def build_skeleton_outputs(full: pd.DataFrame, region_key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    usable = full[full["qc_usable"]].copy()

    year_summary = (
        full.groupby("year")
        .agg(
            rows=("lake_id", "count"),
            lake_count=("lake_id", "nunique"),
            usable_rows=("qc_usable", "sum"),
            usable_share=("qc_usable", "mean"),
            total_annual_area_km2=("annual_max_area_km2", "sum"),
            usable_total_annual_area_km2=(
                "annual_max_area_km2",
                lambda s: float(full.loc[s.index, "annual_max_area_km2"][full.loc[s.index, "qc_usable"]].sum()),
            ),
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
        meta_cols = [c for c in ["lake_id", "lake_type", "harmonized_class"] if c in full.columns]
        meta = full[full["year"] == latest_year][meta_cols].drop_duplicates("lake_id")
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

    payload = {
        "region": region_key,
        "years": years_sorted,
        "input_file_count": int(full["source_file"].dropna().nunique()) if "source_file" in full.columns else 0,
        "lake_year_rows": int(len(lake_year)),
        "unique_lakes": int(lake_year["lake_id"].nunique()),
        "usable_share_overall": float(full["qc_usable"].mean()),
    }
    return lake_year, year_summary, interyear_change, lake_summary, low_quality_lakes, latest_drop_candidates, payload


def build_analysis_outputs(lake_year: pd.DataFrame, year_summary: pd.DataFrame, region_key: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    year_df = year_summary.copy()
    lake_year_df = lake_year.copy()

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
        if col in lake_year_df.columns:
            lake_year_df[col] = pd.to_numeric(lake_year_df[col], errors="coerce")
    if "qc_usable" in lake_year_df.columns:
        lake_year_df["qc_usable"] = normalize_boolish(lake_year_df["qc_usable"])

    if "zero_ratio_share" not in year_df.columns:
        zero_ratio = (
            lake_year_df.groupby("year")["annual_area_to_baseline_ratio"]
            .apply(lambda s: float((pd.to_numeric(s, errors="coerce").fillna(0) == 0).mean()))
            .reset_index(name="zero_ratio_share")
        )
        year_df = year_df.merge(zero_ratio, on="year", how="left")
        resolved_metrics["zero_ratio_share"] = "derived_from_lake_year"

    year_min = int(year_df["year"].min()) if year_df["year"].notna().any() else 2000
    year_max = int(year_df["year"].max()) if year_df["year"].notna().any() else 2024
    mid = (year_min + year_max) // 2
    segments = [("full", year_min, year_max), ("early", year_min, mid), ("late", mid, year_max)]

    trend_tables = []
    for metric in ["total_annual_area_km2", "median_ratio", "mean_ratio", "zero_ratio_share"]:
        trend_tables.append(segment_trends(year_df, metric, segments))
    trend_summary = pd.concat(trend_tables, ignore_index=True)

    anomaly_df = compute_anomalies(year_df, "total_annual_area_km2")
    anomaly_df = compute_anomalies(anomaly_df, "median_ratio")
    anomaly_df = compute_anomalies(anomaly_df, "mean_ratio")
    anomaly_df = compute_anomalies(anomaly_df, "zero_ratio_share")

    class_col, class_source = resolve_class_column(lake_year_df)
    if class_col is not None:
        class_year = (
            lake_year_df.groupby(["year", class_col])
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

    class_trend_rows = []
    for cls in class_year["harmonized_class"].dropna().unique():
        sub = class_year[class_year["harmonized_class"] == cls]
        for metric in ["total_area_km2", "median_ratio"]:
            stats = linregress_safe(sub["year"].values, sub[metric].values)
            class_trend_rows.append({"class_source": class_source, "harmonized_class": cls, "metric": metric, **stats})
    class_trends = pd.DataFrame(class_trend_rows)

    payload = {
        "resolved_year_metrics": resolved_metrics,
        "class_source": class_source,
        "region": region_key,
    }
    return trend_summary, anomaly_df, class_year, class_trends, payload


def build_diagnosis_outputs(
    lake_year: pd.DataFrame,
    region_key: str,
    ref_year: int,
    target_year: int,
    ratio_floor: float,
    min_images: int,
    min_valid_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df = lake_year.copy()
    if "glambie_region_key" in df.columns:
        df = df[df["glambie_region_key"] == region_key].copy()
    pair = df[df["year"].isin([ref_year, target_year])].copy()
    if pair.empty:
        raise ValueError(f"No rows found for region={region_key} years={ref_year},{target_year}")

    numeric_cols = [
        "annual_area_to_baseline_ratio",
        "annual_max_area_km2",
        "image_count",
        "baseline_valid_area_fraction",
        "chunk_start",
    ]
    for col in numeric_cols:
        if col in pair.columns:
            pair[col] = pd.to_numeric(pair[col], errors="coerce")
    pair["qc_usable"] = normalize_boolish(pair["qc_usable"])

    ysum = (
        pair.groupby("year")
        .agg(
            rows=("lake_id", "count"),
            usable_share=("qc_usable", "mean"),
            total_area_km2=("annual_max_area_km2", "sum"),
            median_ratio=("annual_area_to_baseline_ratio", "median"),
            zero_ratio_share=("annual_area_to_baseline_ratio", lambda s: float((s.fillna(0) == 0).mean())),
        )
        .reset_index()
        .sort_values("year")
    )

    pair_cols = [
        "year",
        "lake_id",
        "harmonized_class",
        "lake_type",
        "chunk_start",
        "annual_area_to_baseline_ratio",
        "annual_max_area_km2",
        "image_count",
        "baseline_valid_area_fraction",
        "qc_usable",
    ]
    pair = pair[[c for c in pair_cols if c in pair.columns]].copy()
    ref_df = pair[pair["year"] == ref_year].copy()
    tgt_df = pair[pair["year"] == target_year].copy()
    merged = ref_df.merge(tgt_df, on="lake_id", how="inner", suffixes=(f"_{ref_year}", f"_{target_year}"))

    merged["ratio_change"] = (
        merged[f"annual_area_to_baseline_ratio_{target_year}"] - merged[f"annual_area_to_baseline_ratio_{ref_year}"]
    )
    merged["area_change_km2"] = merged[f"annual_max_area_km2_{target_year}"] - merged[f"annual_max_area_km2_{ref_year}"]

    suspicious = merged[
        (merged[f"annual_area_to_baseline_ratio_{ref_year}"] >= ratio_floor)
        & (merged[f"annual_area_to_baseline_ratio_{target_year}"] == 0)
        & (merged[f"qc_usable_{target_year}"])
        & (merged[f"image_count_{target_year}"] >= min_images)
        & (merged[f"baseline_valid_area_fraction_{target_year}"] >= min_valid_fraction)
    ].copy()

    chunk_col = f"chunk_start_{target_year}"
    chunk_summary = (
        merged.groupby(chunk_col)
        .agg(
            lakes=("lake_id", "count"),
            mean_ratio_ref=(f"annual_area_to_baseline_ratio_{ref_year}", "mean"),
            mean_ratio_target=(f"annual_area_to_baseline_ratio_{target_year}", "mean"),
            mean_ratio_change=("ratio_change", "mean"),
            total_area_ref=(f"annual_max_area_km2_{ref_year}", "sum"),
            total_area_target=(f"annual_max_area_km2_{target_year}", "sum"),
            total_area_change=("area_change_km2", "sum"),
        )
        .reset_index()
        .rename(columns={chunk_col: "chunk_start"})
        .sort_values("total_area_change")
    )

    suspicious_chunk_summary = (
        suspicious.groupby(chunk_col)
        .agg(
            suspicious_lakes=("lake_id", "count"),
            mean_ratio_ref=(f"annual_area_to_baseline_ratio_{ref_year}", "mean"),
            mean_images_target=(f"image_count_{target_year}", "mean"),
            mean_valid_fraction_target=(f"baseline_valid_area_fraction_{target_year}", "mean"),
            mean_area_ref=(f"annual_max_area_km2_{ref_year}", "mean"),
        )
        .reset_index()
        .rename(columns={chunk_col: "chunk_start"})
        .sort_values("suspicious_lakes", ascending=False)
    )

    class_col = f"harmonized_class_{target_year}"
    class_summary = (
        merged.groupby(class_col)
        .agg(
            lakes=("lake_id", "count"),
            mean_ratio_ref=(f"annual_area_to_baseline_ratio_{ref_year}", "mean"),
            mean_ratio_target=(f"annual_area_to_baseline_ratio_{target_year}", "mean"),
            mean_ratio_change=("ratio_change", "mean"),
            zero_ratio_target=(f"annual_area_to_baseline_ratio_{target_year}", lambda s: int((s.fillna(0) == 0).sum())),
        )
        .reset_index()
        .rename(columns={class_col: "harmonized_class"})
        .sort_values("mean_ratio_change")
    )

    payload = {
        "region": region_key,
        "ref_year": ref_year,
        "target_year": target_year,
        "rows_ref_target": int(len(merged)),
        "total_area_change_km2": float(merged["area_change_km2"].sum()),
        "mean_ratio_change": float(merged["ratio_change"].mean()),
        "suspicious_lake_count": int(len(suspicious)),
        "ratio_floor": float(ratio_floor),
        "min_images": int(min_images),
        "min_valid_fraction": float(min_valid_fraction),
    }
    return ysum, chunk_summary, class_summary, suspicious, suspicious_chunk_summary, payload


def write_region_outputs(
    row: pd.Series,
    lake_year: pd.DataFrame,
    year_summary: pd.DataFrame,
    interyear_change: pd.DataFrame,
    lake_summary: pd.DataFrame,
    low_quality_lakes: pd.DataFrame,
    latest_drop_candidates: pd.DataFrame,
    skeleton_payload: dict,
    trend_summary: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    class_year: pd.DataFrame,
    class_trends: pd.DataFrame,
    analysis_payload: dict,
    diag_year_pair: pd.DataFrame,
    diag_chunk: pd.DataFrame,
    diag_class: pd.DataFrame,
    diag_suspicious: pd.DataFrame,
    diag_suspicious_chunk: pd.DataFrame,
    diagnosis_payload: dict,
) -> dict:
    final_tag = str(row["final_tag"])
    skeleton_summary_json = Path(str(row["skeleton_summary_json"]))
    analysis_dir = Path(str(row["analysis_dir"]))
    diagnosis_summary_json = Path(str(row["diagnosis_summary_json"]))
    skeleton_dir = skeleton_summary_json.parent
    diagnosis_tag = diagnosis_summary_json.name.removesuffix("_diagnosis_summary.json")

    skeleton_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    lake_year_csv = skeleton_dir / f"{final_tag}_skeleton_lake_year.csv"
    year_summary_csv = skeleton_dir / f"{final_tag}_skeleton_year_summary.csv"
    interyear_change_csv = skeleton_dir / f"{final_tag}_skeleton_interyear_change.csv"
    lake_summary_csv = skeleton_dir / f"{final_tag}_skeleton_lake_summary.csv"
    low_quality_csv = skeleton_dir / f"{final_tag}_skeleton_low_quality_lakes.csv"
    latest_drop_csv = skeleton_dir / f"{final_tag}_skeleton_latest_year_drop_candidates.csv"

    lake_year.to_csv(lake_year_csv, index=False)
    year_summary.to_csv(year_summary_csv, index=False)
    interyear_change.to_csv(interyear_change_csv, index=False)
    lake_summary.to_csv(lake_summary_csv, index=False)
    low_quality_lakes.to_csv(low_quality_csv, index=False)
    latest_drop_candidates.to_csv(latest_drop_csv, index=False)

    skeleton_payload = {
        **skeleton_payload,
        "year_summary_csv": str(year_summary_csv),
        "interyear_change_csv": str(interyear_change_csv),
        "lake_year_csv": str(lake_year_csv),
        "lake_summary_csv": str(lake_summary_csv),
        "low_quality_lakes_csv": str(low_quality_csv),
        "latest_drop_candidates_csv": str(latest_drop_csv),
    }
    skeleton_summary_json.write_text(json.dumps(skeleton_payload, indent=2), encoding="utf-8")

    trend_csv = analysis_dir / f"{final_tag}_trend_summary.csv"
    anomalies_csv = analysis_dir / f"{final_tag}_year_anomalies.csv"
    class_year_csv = analysis_dir / f"{final_tag}_class_year_stats.csv"
    class_trends_csv = analysis_dir / f"{final_tag}_class_trends.csv"
    analysis_summary_json = analysis_dir / f"{final_tag}_analysis_summary.json"

    trend_summary.to_csv(trend_csv, index=False)
    anomaly_df.to_csv(anomalies_csv, index=False)
    class_year.to_csv(class_year_csv, index=False)
    class_trends.to_csv(class_trends_csv, index=False)
    analysis_payload = {
        **analysis_payload,
        "trend_summary_csv": str(trend_csv),
        "year_anomalies_csv": str(anomalies_csv),
        "class_year_stats_csv": str(class_year_csv),
        "class_trends_csv": str(class_trends_csv),
    }
    analysis_summary_json.write_text(json.dumps(analysis_payload, indent=2), encoding="utf-8")

    diag_year_pair_csv = skeleton_dir / f"{diagnosis_tag}_year_pair_summary.csv"
    diag_chunk_csv = skeleton_dir / f"{diagnosis_tag}_chunk_drop_summary.csv"
    diag_class_csv = skeleton_dir / f"{diagnosis_tag}_class_drop_summary.csv"
    diag_suspicious_csv = skeleton_dir / f"{diagnosis_tag}_suspicious_lakes.csv"
    diag_suspicious_chunk_csv = skeleton_dir / f"{diagnosis_tag}_suspicious_chunk_priority.csv"

    diag_year_pair.to_csv(diag_year_pair_csv, index=False)
    diag_chunk.to_csv(diag_chunk_csv, index=False)
    diag_class.to_csv(diag_class_csv, index=False)
    diag_suspicious.to_csv(diag_suspicious_csv, index=False)
    diag_suspicious_chunk.to_csv(diag_suspicious_chunk_csv, index=False)
    diagnosis_payload = {
        **diagnosis_payload,
        "year_pair_summary_csv": str(diag_year_pair_csv),
        "chunk_drop_summary_csv": str(diag_chunk_csv),
        "class_drop_summary_csv": str(diag_class_csv),
        "suspicious_lakes_csv": str(diag_suspicious_csv),
        "suspicious_chunk_priority_csv": str(diag_suspicious_chunk_csv),
    }
    diagnosis_summary_json.write_text(json.dumps(diagnosis_payload, indent=2), encoding="utf-8")

    return {
        "skeleton_summary_json": str(skeleton_summary_json),
        "analysis_dir": str(analysis_dir),
        "diagnosis_summary_json": str(diagnosis_summary_json),
        "skeleton_payload": skeleton_payload,
        "analysis_payload": analysis_payload,
        "diagnosis_payload": diagnosis_payload,
    }


def extract_registry_metrics(row: pd.Series, skeleton_payload: dict, anomaly_df: pd.DataFrame, diagnosis_payload: dict) -> dict:
    metrics = row.to_dict()
    metrics["usable_share_overall"] = float(skeleton_payload["usable_share_overall"])
    metrics["input_file_count"] = int(skeleton_payload["input_file_count"])
    metrics["lake_year_rows"] = int(skeleton_payload["lake_year_rows"])
    metrics["unique_lakes"] = int(skeleton_payload["unique_lakes"])

    year_index = anomaly_df.set_index("year")
    for year in [2020, 2024]:
        if year not in year_index.index:
            continue
        metrics[f"usable_share_{year}"] = float(year_index.at[year, "usable_share"])
        metrics[f"total_annual_area_km2_{year}"] = float(year_index.at[year, "total_annual_area_km2"])
        metrics[f"usable_median_area_ratio_{year}"] = float(year_index.at[year, "usable_median_area_ratio"])
        metrics[f"usable_mean_area_ratio_{year}"] = float(year_index.at[year, "usable_mean_area_ratio"])

    metrics["total_area_change_km2_2020_2024"] = float(diagnosis_payload["total_area_change_km2"])
    metrics["mean_ratio_change_2020_2024"] = float(diagnosis_payload["mean_ratio_change"])
    metrics["suspicious_lake_count_2020_2024"] = int(diagnosis_payload["suspicious_lake_count"])
    metrics["skeleton_summary_json"] = skeleton_payload["year_summary_csv"].replace("_skeleton_year_summary.csv", "_skeleton_summary.json")
    return metrics


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry_csv)
    sgl_qc_full_path = Path(args.sgl_qc_full)
    summary_json = Path(args.summary_json)
    selected_regions = parse_region_list(args.regions)

    registry = pd.read_csv(registry_path)
    sgl_full = pd.read_csv(sgl_qc_full_path, low_memory=False)
    if "glambie_region_key" not in sgl_full.columns:
        raise ValueError("SGL QC full CSV is missing glambie_region_key")
    sgl_full["glambie_region_key"] = sgl_full["glambie_region_key"].astype(str)

    updated_rows = []
    summary_rows = []
    for row in registry.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        region_key = str(row_s["region_key"])
        if selected_regions is not None and region_key not in selected_regions:
            updated_rows.append(row_s.to_dict())
            continue

        final_tag = str(row_s["final_tag"])
        skeleton_summary_json = Path(str(row_s["skeleton_summary_json"]))
        diagnosis_summary_json = Path(str(row_s["diagnosis_summary_json"]))
        skeleton_dir = skeleton_summary_json.parent
        core_lake_year_path = skeleton_dir / f"{final_tag}_skeleton_lake_year.csv"

        core_df = pd.read_csv(core_lake_year_path, low_memory=False)
        sgl_df = sgl_full[sgl_full["glambie_region_key"] == region_key].copy()
        if sgl_df.empty:
            raise ValueError(f"No SGL rows found for {region_key}")

        combined = prepare_combined_lake_year(core_df, sgl_df, region_key)
        lake_year, year_summary, interyear_change, lake_summary, low_quality_lakes, latest_drop_candidates, skeleton_payload = build_skeleton_outputs(combined, region_key)
        trend_summary, anomaly_df, class_year, class_trends, analysis_payload = build_analysis_outputs(lake_year, year_summary, region_key)

        old_diag = load_json(diagnosis_summary_json)
        ref_year = int(old_diag.get("ref_year", 2020))
        target_year = int(old_diag.get("target_year", 2024))
        ratio_floor = float(old_diag.get("ratio_floor", 0.6))
        min_images = int(old_diag.get("min_images", 5))
        min_valid_fraction = float(old_diag.get("min_valid_fraction", 0.9))
        diag_year_pair, diag_chunk, diag_class, diag_suspicious, diag_suspicious_chunk, diagnosis_payload = build_diagnosis_outputs(
            lake_year,
            region_key,
            ref_year,
            target_year,
            ratio_floor,
            min_images,
            min_valid_fraction,
        )

        written = write_region_outputs(
            row_s,
            lake_year,
            year_summary,
            interyear_change,
            lake_summary,
            low_quality_lakes,
            latest_drop_candidates,
            skeleton_payload,
            trend_summary,
            anomaly_df,
            class_year,
            class_trends,
            analysis_payload,
            diag_year_pair,
            diag_chunk,
            diag_class,
            diag_suspicious,
            diag_suspicious_chunk,
            diagnosis_payload,
        )

        updated_row = extract_registry_metrics(row_s, written["skeleton_payload"], anomaly_df, written["diagnosis_payload"])
        updated_rows.append(updated_row)
        summary_rows.append(
            {
                "region_key": region_key,
                "final_tag": final_tag,
                "core_rows_before": int(len(core_df)),
                "sgl_rows_added": int(len(sgl_df)),
                "lake_year_rows_after": int(len(lake_year)),
                "unique_lakes_after": int(lake_year["lake_id"].nunique()),
                "usable_share_overall_after": float(skeleton_payload["usable_share_overall"]),
                "input_file_count_after": int(skeleton_payload["input_file_count"]),
                "diagnosis_ref_year": ref_year,
                "diagnosis_target_year": target_year,
                "suspicious_lake_count_after": int(diagnosis_payload["suspicious_lake_count"]),
            }
        )

    updated_registry = pd.DataFrame(updated_rows)
    updated_registry.to_csv(registry_path, index=False)

    payload = {
        "registry_csv": str(registry_path),
        "sgl_qc_full_csv": str(sgl_qc_full_path),
        "regions_updated": summary_rows,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
