from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_INPUT = ROOT / "data" / "processed" / "annual_area_skeleton" / "central_asia_key_years_skeleton_lake_year.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "annual_area_skeleton"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose anomaly between two years for one region skeleton lake-year table.")
    p.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    p.add_argument("--region", default="central_asia")
    p.add_argument("--ref-year", type=int, required=True, help="Reference year, e.g. 2020")
    p.add_argument("--target-year", type=int, required=True, help="Target year to diagnose, e.g. 2024")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--tag", default="")
    p.add_argument("--ratio-floor", type=float, default=0.6, help="Reference-year ratio threshold for suspicious lakes")
    p.add_argument("--min-images", type=int, default=5, help="Minimum target-year image_count for suspicious lakes")
    p.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.9,
        help="Minimum target-year baseline_valid_area_fraction for suspicious lakes",
    )
    return p.parse_args()


def normalize_boolish(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    return mapped.where(mapped.notna(), series).astype(bool)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if "glambie_region_key" in df.columns:
        df = df[df["glambie_region_key"] == args.region].copy()
    pair = df[df["year"].isin([args.ref_year, args.target_year])].copy()
    if pair.empty:
        raise SystemExit("No rows found for selected years/region.")

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
    pair = pair[pair_cols].copy()
    ref_df = pair[pair["year"] == args.ref_year].copy()
    tgt_df = pair[pair["year"] == args.target_year].copy()
    merged = ref_df.merge(
        tgt_df,
        on="lake_id",
        how="inner",
        suffixes=(f"_{args.ref_year}", f"_{args.target_year}"),
    )

    merged["ratio_change"] = (
        merged[f"annual_area_to_baseline_ratio_{args.target_year}"]
        - merged[f"annual_area_to_baseline_ratio_{args.ref_year}"]
    )
    merged["area_change_km2"] = (
        merged[f"annual_max_area_km2_{args.target_year}"] - merged[f"annual_max_area_km2_{args.ref_year}"]
    )

    suspicious = merged[
        (merged[f"annual_area_to_baseline_ratio_{args.ref_year}"] >= args.ratio_floor)
        & (merged[f"annual_area_to_baseline_ratio_{args.target_year}"] == 0)
        & (merged[f"qc_usable_{args.target_year}"])
        & (merged[f"image_count_{args.target_year}"] >= args.min_images)
        & (merged[f"baseline_valid_area_fraction_{args.target_year}"] >= args.min_valid_fraction)
    ].copy()

    chunk_col = f"chunk_start_{args.target_year}"
    chunk_summary = (
        merged.groupby(chunk_col)
        .agg(
            lakes=("lake_id", "count"),
            mean_ratio_ref=(f"annual_area_to_baseline_ratio_{args.ref_year}", "mean"),
            mean_ratio_target=(f"annual_area_to_baseline_ratio_{args.target_year}", "mean"),
            mean_ratio_change=("ratio_change", "mean"),
            total_area_ref=(f"annual_max_area_km2_{args.ref_year}", "sum"),
            total_area_target=(f"annual_max_area_km2_{args.target_year}", "sum"),
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
            mean_ratio_ref=(f"annual_area_to_baseline_ratio_{args.ref_year}", "mean"),
            mean_images_target=(f"image_count_{args.target_year}", "mean"),
            mean_valid_fraction_target=(f"baseline_valid_area_fraction_{args.target_year}", "mean"),
            mean_area_ref=(f"annual_max_area_km2_{args.ref_year}", "mean"),
        )
        .reset_index()
        .rename(columns={chunk_col: "chunk_start"})
        .sort_values("suspicious_lakes", ascending=False)
    )

    class_summary = (
        merged.groupby(f"harmonized_class_{args.target_year}")
        .agg(
            lakes=("lake_id", "count"),
            mean_ratio_ref=(f"annual_area_to_baseline_ratio_{args.ref_year}", "mean"),
            mean_ratio_target=(f"annual_area_to_baseline_ratio_{args.target_year}", "mean"),
            mean_ratio_change=("ratio_change", "mean"),
            zero_ratio_target=(f"annual_area_to_baseline_ratio_{args.target_year}", lambda s: int((s.fillna(0) == 0).sum())),
        )
        .reset_index()
        .rename(columns={f"harmonized_class_{args.target_year}": "harmonized_class"})
        .sort_values("mean_ratio_change")
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.region}_{args.ref_year}_{args.target_year}"
    out_ysum = out_dir / f"{tag}_year_pair_summary.csv"
    out_chunk = out_dir / f"{tag}_chunk_drop_summary.csv"
    out_class = out_dir / f"{tag}_class_drop_summary.csv"
    out_susp = out_dir / f"{tag}_suspicious_lakes.csv"
    out_susp_chunk = out_dir / f"{tag}_suspicious_chunk_priority.csv"
    out_json = out_dir / f"{tag}_diagnosis_summary.json"

    ysum.to_csv(out_ysum, index=False)
    chunk_summary.to_csv(out_chunk, index=False)
    class_summary.to_csv(out_class, index=False)
    suspicious.to_csv(out_susp, index=False)
    suspicious_chunk_summary.to_csv(out_susp_chunk, index=False)

    payload = {
        "region": args.region,
        "ref_year": args.ref_year,
        "target_year": args.target_year,
        "rows_ref_target": int(len(merged)),
        "total_area_change_km2": float(merged["area_change_km2"].sum()),
        "mean_ratio_change": float(merged["ratio_change"].mean()),
        "suspicious_lake_count": int(len(suspicious)),
        "ratio_floor": args.ratio_floor,
        "min_images": args.min_images,
        "min_valid_fraction": args.min_valid_fraction,
        "year_pair_summary_csv": str(out_ysum),
        "chunk_drop_summary_csv": str(out_chunk),
        "class_drop_summary_csv": str(out_class),
        "suspicious_lakes_csv": str(out_susp),
        "suspicious_chunk_priority_csv": str(out_susp_chunk),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
