from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
RAW_DIR = ROOT / "data" / "interim" / "annual_area_raw"
QC_DIR = ROOT / "data" / "interim" / "annual_area_qc"
FILENAME_RE = re.compile(
    r"^annual_area_(?P<region_code>\d{2})_(?P<region_key>.+)_(?P<year>\d{4})_chunk_"
    r"(?P<chunk_start>\d+)_(?P<chunk_end>\d+)(?:_(?P<variant>.+))?\.csv$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QC local annual-area CSV exports.")
    p.add_argument(
        "--input-dir",
        default=str(RAW_DIR),
        help=r"Directory containing downloaded CSV exports. Default: E:\Glacier\data\interim\annual_area_raw",
    )
    p.add_argument(
        "--pattern",
        default="*.csv",
        help="Filename glob pattern used inside the input directory. Default: *.csv",
    )
    p.add_argument(
        "--tag",
        default="annual_area_test",
        help="Short tag used in output filenames. Default: annual_area_test",
    )
    p.add_argument(
        "--exclude-substring",
        default="",
        help="Optional substring filter. Files containing this text will be excluded.",
    )
    return p.parse_args()


def normalize_boolish(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    return mapped.where(mapped.notna(), series).astype(bool)


def parse_filename_metadata(path: Path) -> dict:
    match = FILENAME_RE.match(path.name)
    if not match:
        return {}
    info = match.groupdict()
    chunk_start = int(info["chunk_start"])
    chunk_end = int(info["chunk_end"])
    return {
        "region_key_from_name": info["region_key"],
        "year_from_name": int(info["year"]),
        "chunk_start_from_name": chunk_start,
        "chunk_size_from_name": chunk_end - chunk_start + 1,
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    QC_DIR.mkdir(parents=True, exist_ok=True)
    csvs = sorted(input_dir.glob(args.pattern))
    if args.exclude_substring:
        csvs = [p for p in csvs if args.exclude_substring not in p.name]
    if not csvs:
        print(f"No CSV files found in {input_dir} matching pattern {args.pattern}")
        return

    frames = []
    for path in csvs:
        frame = pd.read_csv(path)
        meta = parse_filename_metadata(path)
        frame["__source_file"] = path.name
        for key, value in meta.items():
            frame[key] = value
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    required_cols = {
        "lake_id",
        "glambie_region_key",
        "year",
        "annual_max_area_km2",
        "baseline_area_0_km2",
        "baseline_valid_area_fraction",
        "annual_area_to_baseline_ratio",
        "image_count",
        "qc_enough_images",
        "qc_enough_coverage",
        "qc_usable",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["qc_usable"] = normalize_boolish(df["qc_usable"])
    df["qc_enough_images"] = normalize_boolish(df["qc_enough_images"])
    df["qc_enough_coverage"] = normalize_boolish(df["qc_enough_coverage"])

    if "glambie_region_key" in df.columns and "region_key_from_name" in df.columns:
        df["glambie_region_key"] = df["glambie_region_key"].fillna(df["region_key_from_name"])
    if "year" in df.columns and "year_from_name" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(df["year_from_name"]).astype(int)
    if "chunk_start" in df.columns and "chunk_start_from_name" in df.columns:
        df["chunk_start"] = (
            pd.to_numeric(df["chunk_start"], errors="coerce").fillna(df["chunk_start_from_name"]).astype(int)
        )
    if "chunk_size" in df.columns and "chunk_size_from_name" in df.columns:
        df["chunk_size"] = pd.to_numeric(df["chunk_size"], errors="coerce").fillna(df["chunk_size_from_name"]).astype(int)

    df["flag_area_ratio_gt_3"] = df["annual_area_to_baseline_ratio"] > 3
    df["flag_area_ratio_lt_0_1"] = df["annual_area_to_baseline_ratio"] < 0.1
    df["flag_low_image_count"] = df["image_count"] < 3
    df["flag_low_valid_fraction"] = df["baseline_valid_area_fraction"] < 0.7

    out_full = QC_DIR / f"{args.tag}_qc_full.csv"
    out_summary = QC_DIR / f"{args.tag}_qc_summary.csv"
    out_json = QC_DIR / f"{args.tag}_qc_summary.json"
    out_chunk_summary = QC_DIR / f"{args.tag}_qc_chunk_summary.csv"

    df.to_csv(out_full, index=False)

    summary = (
        df.groupby("glambie_region_key")
        .agg(
            rows=("lake_id", "count"),
            usable_rows=("qc_usable", lambda s: int(pd.Series(s).fillna(False).sum())),
            usable_share=("qc_usable", lambda s: float(pd.Series(s).fillna(False).mean())),
            low_image_rows=("flag_low_image_count", "sum"),
            low_valid_rows=("flag_low_valid_fraction", "sum"),
            high_ratio_rows=("flag_area_ratio_gt_3", "sum"),
            low_ratio_rows=("flag_area_ratio_lt_0_1", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(out_summary, index=False)

    chunk_summary_path = None
    if {"chunk_start", "chunk_size"}.issubset(df.columns):
        chunk_summary = (
            df.groupby(["glambie_region_key", "year", "chunk_start", "chunk_size"])
            .agg(
                rows=("lake_id", "count"),
                usable_rows=("qc_usable", "sum"),
                usable_share=("qc_usable", "mean"),
                median_image_count=("image_count", "median"),
                low_image_rows=("flag_low_image_count", "sum"),
                low_valid_rows=("flag_low_valid_fraction", "sum"),
            )
            .reset_index()
            .sort_values(["glambie_region_key", "year", "chunk_start"])
        )
        chunk_summary.to_csv(out_chunk_summary, index=False)
        chunk_summary_path = str(out_chunk_summary)

    payload = {
        "input_files": [str(p) for p in csvs],
        "input_dir": str(input_dir),
        "pattern": args.pattern,
        "exclude_substring": args.exclude_substring,
        "rows": int(len(df)),
        "regions": sorted(df["glambie_region_key"].dropna().unique().tolist()),
        "usable_share_overall": float(pd.Series(df["qc_usable"]).fillna(False).mean()),
        "summary_csv": str(out_summary),
        "chunk_summary_csv": chunk_summary_path,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
