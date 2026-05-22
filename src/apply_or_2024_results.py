from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_SKELETON = ROOT / "data" / "processed" / "annual_area_skeleton" / "central_asia_key_years_skeleton_lake_year.csv"
DEFAULT_OR_DIR = ROOT / "data" / "processed" / "GlacierAnnualArea" / "13"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "annual_area_skeleton"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replace 2024 rows with OR-rule outputs and write updated skeleton files.")
    p.add_argument("--skeleton-csv", default=str(DEFAULT_SKELETON))
    p.add_argument("--or-dir", default=str(DEFAULT_OR_DIR))
    p.add_argument("--region-key", default="central_asia")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--tag", default="central_asia_key_years_or2024")
    return p.parse_args()


def normalize_boolish(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    return mapped.where(mapped.notna(), series).astype(bool)


def load_or_files(or_dir: Path, year: int) -> pd.DataFrame:
    pattern = re.compile(
        rf"^annual_area_13_central_asia_{year}_chunk_(\d+)_(\d+)_4month_water_or_ndwip0p00_mndwip0p00\.csv$"
    )
    frames = []
    for p in sorted(or_dir.glob(f"annual_area_13_central_asia_{year}_chunk_*_4month_water_or_ndwip0p00_mndwip0p00.csv")):
        if not pattern.match(p.name):
            continue
        df = pd.read_csv(p)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No OR-rule CSVs found in {or_dir} for year {year}.")
    out = pd.concat(frames, ignore_index=True)
    return out


def summarize_year(df: pd.DataFrame) -> dict:
    df = df.copy()
    for c in ["annual_max_area_km2", "annual_area_to_baseline_ratio", "image_count", "baseline_valid_area_fraction"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["qc_usable"] = normalize_boolish(df["qc_usable"])
    return {
        "rows": int(len(df)),
        "usable_share": float(df["qc_usable"].mean()),
        "total_area_km2": float(df["annual_max_area_km2"].sum()),
        "median_ratio": float(df["annual_area_to_baseline_ratio"].median()),
        "mean_ratio": float(df["annual_area_to_baseline_ratio"].mean()),
        "zero_ratio_share": float((df["annual_area_to_baseline_ratio"].fillna(0) == 0).mean()),
        "median_images": float(pd.to_numeric(df["image_count"], errors="coerce").median()),
        "median_valid_fraction": float(pd.to_numeric(df["baseline_valid_area_fraction"], errors="coerce").median()),
    }


def main() -> None:
    args = parse_args()
    skeleton_path = Path(args.skeleton_csv)
    or_dir = Path(args.or_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skeleton = pd.read_csv(skeleton_path)
    base = skeleton[~((skeleton["glambie_region_key"] == args.region_key) & (skeleton["year"] == args.year))].copy()

    or_df = load_or_files(or_dir, args.year)
    if "glambie_region_key" not in or_df.columns:
        or_df["glambie_region_key"] = args.region_key
    or_df["year"] = args.year

    combined = pd.concat([base, or_df], ignore_index=True)

    out_lake_year = out_dir / f"{args.tag}_skeleton_lake_year.csv"
    combined.to_csv(out_lake_year, index=False)

    # Build updated year summary
    numeric_cols = [
        "annual_max_area_km2",
        "annual_area_to_baseline_ratio",
        "image_count",
        "baseline_valid_area_fraction",
    ]
    tmp = combined.copy()
    for c in numeric_cols:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp["qc_usable"] = normalize_boolish(tmp["qc_usable"])
    year_summary = (
        tmp.groupby("year")
        .agg(
            rows=("lake_id", "count"),
            lake_count=("lake_id", "nunique"),
            usable_rows=("qc_usable", "sum"),
            usable_share=("qc_usable", "mean"),
            total_annual_area_km2=("annual_max_area_km2", "sum"),
            median_ratio=("annual_area_to_baseline_ratio", "median"),
            mean_ratio=("annual_area_to_baseline_ratio", "mean"),
            zero_ratio_share=("annual_area_to_baseline_ratio", lambda s: float((s.fillna(0) == 0).mean())),
        )
        .reset_index()
        .sort_values("year")
    )
    out_year_summary = out_dir / f"{args.tag}_skeleton_year_summary.csv"
    year_summary.to_csv(out_year_summary, index=False)

    payload = {
        "region_key": args.region_key,
        "year_replaced": args.year,
        "or_dir": str(or_dir),
        "output_lake_year_csv": str(out_lake_year),
        "output_year_summary_csv": str(out_year_summary),
        "or_year_summary": summarize_year(or_df),
    }
    out_json = out_dir / f"{args.tag}_skeleton_summary.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
