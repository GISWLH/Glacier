from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_INPUT_DIR = ROOT / "data" / "processed" / "GlacierAnnualArea"
DEFAULT_PLAN_PATH = ROOT / "data" / "prepared" / "execution_plan" / "phase1_batch_runlist.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim" / "annual_area_qc"

FILENAME_RE = re.compile(
    r"^annual_area_(?P<region_code>\d{2})_(?P<region_key>.+)_(?P<year>\d{4})_chunk_"
    r"(?P<chunk_start>\d+)_(?P<chunk_end>\d+)(?:_(?P<variant>.+))?\.csv$"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inventory downloaded annual-area exports, resolve duplicate variants, and compare against the phase1 plan."
    )
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    p.add_argument("--plan-csv", default=str(DEFAULT_PLAN_PATH))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--tag", default="downloaded_inventory")
    return p.parse_args()


def normalize_boolish(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    return mapped.where(mapped.notna(), series).astype(bool)


def scan_files(input_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(input_dir.glob("*.csv")):
        match = FILENAME_RE.match(path.name)
        if not match:
            rows.append({"file": path.name, "path": str(path), "matched": False})
            continue
        info = match.groupdict()
        rows.append(
            {
                "file": path.name,
                "path": str(path),
                "matched": True,
                "region_code": int(info["region_code"]),
                "region_key": info["region_key"],
                "year": int(info["year"]),
                "chunk_start": int(info["chunk_start"]),
                "chunk_end": int(info["chunk_end"]),
                "variant": info["variant"] or "",
            }
        )
    return pd.DataFrame(rows)


def summarize_file(path: Path, fallback_region_key: str | None = None, fallback_year: int | None = None) -> dict:
    df = pd.read_csv(path)
    required_cols = {
        "lake_id",
        "glambie_region_key",
        "year",
        "image_count",
        "baseline_valid_area_fraction",
        "qc_usable",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    qc_usable = normalize_boolish(df["qc_usable"])
    region_key_mode = df["glambie_region_key"].dropna().astype(str).mode()
    year_mode = pd.to_numeric(df["year"], errors="coerce").dropna().mode()

    summary = {
        "rows": int(len(df)),
        "usable_rows": int(qc_usable.sum()),
        "usable_share": float(qc_usable.mean()),
        "median_image_count": float(pd.to_numeric(df["image_count"], errors="coerce").median()),
        "median_valid_fraction": float(
            pd.to_numeric(df["baseline_valid_area_fraction"], errors="coerce").median()
        ),
        "glambie_region_key_data": region_key_mode.iloc[0] if not region_key_mode.empty else (fallback_region_key or ""),
        "year_data": int(year_mode.iloc[0]) if not year_mode.empty else int(fallback_year or -1),
    }
    if {"chunk_start", "chunk_size"}.issubset(df.columns):
        summary["chunk_start_data"] = int(pd.to_numeric(df["chunk_start"], errors="coerce").dropna().mode().iloc[0])
        summary["chunk_size_data"] = int(pd.to_numeric(df["chunk_size"], errors="coerce").dropna().mode().iloc[0])
    return summary


def choose_best_variants(file_summary: pd.DataFrame) -> pd.DataFrame:
    ranked = file_summary.sort_values(
        [
            "region_key",
            "year",
            "chunk_start",
            "usable_share",
            "rows",
            "variant",
        ],
        ascending=[True, True, True, False, False, True],
    ).copy()
    ranked["selection_rank"] = ranked.groupby(["region_key", "year", "chunk_start", "chunk_end"]).cumcount() + 1
    ranked["selected"] = ranked["selection_rank"] == 1
    return ranked


def build_plan_status(selected_files: pd.DataFrame, plan_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    plan = plan_df.copy()
    if "phase1_wave" not in plan.columns:
        plan["phase1_wave"] = ""
    if "wave_order" not in plan.columns:
        plan["wave_order"] = range(len(plan))
    plan["chunk_start"] = plan["start_index"].astype(int)
    plan["chunk_end"] = (plan["end_index_exclusive"].astype(int) - 1).astype(int)

    selected = selected_files[
        ["region_key", "year", "chunk_start", "chunk_end", "file", "variant", "rows", "usable_rows", "usable_share"]
    ].copy()

    merged = plan.merge(
        selected,
        left_on=["glambie_region_key", "year", "chunk_start", "chunk_end"],
        right_on=["region_key", "year", "chunk_start", "chunk_end"],
        how="left",
    )
    merged["downloaded"] = merged["file"].notna()
    merged["row_match_expected"] = merged["rows"] == merged["expected_lake_count"]

    region_year = (
        merged.groupby(["glambie_region_key", "year", "phase1_wave"], dropna=False)
        .agg(
            expected_chunks=("task_id", "count"),
            downloaded_chunks=("downloaded", "sum"),
            expected_rows=("expected_lake_count", "sum"),
            downloaded_rows=("rows", lambda s: int(pd.Series(s).fillna(0).sum())),
            mean_usable_share=("usable_share", "mean"),
        )
        .reset_index()
    )
    region_year["download_completion"] = region_year["downloaded_chunks"] / region_year["expected_chunks"]
    return merged, region_year


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    plan_path = Path(args.plan_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    scanned = scan_files(input_dir)
    unmatched = scanned[~scanned["matched"]].copy()
    matched = scanned[scanned["matched"]].copy()

    if matched.empty:
        raise SystemExit(f"No matching annual-area CSVs found in {input_dir}")

    summaries = []
    for row in matched.itertuples(index=False):
        summary = summarize_file(Path(row.path), fallback_region_key=row.region_key, fallback_year=row.year)
        summaries.append({**row._asdict(), **summary})
    file_summary = pd.DataFrame(summaries)
    ranked = choose_best_variants(file_summary)
    selected = ranked[ranked["selected"]].copy()

    plan_df = pd.read_csv(plan_path)
    plan_status, region_year = build_plan_status(selected, plan_df)

    duplicates = ranked.groupby(["region_key", "year", "chunk_start", "chunk_end"]).filter(lambda x: len(x) > 1).copy()

    inventory_csv = output_dir / f"{args.tag}_file_inventory.csv"
    selected_csv = output_dir / f"{args.tag}_selected_files.csv"
    duplicate_csv = output_dir / f"{args.tag}_duplicate_candidates.csv"
    plan_status_csv = output_dir / f"{args.tag}_phase1_plan_status.csv"
    region_year_csv = output_dir / f"{args.tag}_phase1_region_year_status.csv"
    unmatched_csv = output_dir / f"{args.tag}_unmatched_files.csv"
    summary_json = output_dir / f"{args.tag}_summary.json"

    ranked.sort_values(["region_key", "year", "chunk_start", "selection_rank"]).to_csv(inventory_csv, index=False)
    selected.sort_values(["region_key", "year", "chunk_start"]).to_csv(selected_csv, index=False)
    duplicates.sort_values(["region_key", "year", "chunk_start", "selection_rank"]).to_csv(duplicate_csv, index=False)
    plan_status.sort_values(["wave_order", "glambie_region_key", "year", "chunk_id"]).to_csv(plan_status_csv, index=False)
    region_year.sort_values(["glambie_region_key", "year"]).to_csv(region_year_csv, index=False)
    unmatched.to_csv(unmatched_csv, index=False)

    payload = {
        "input_dir": str(input_dir),
        "plan_csv": str(plan_path),
        "matched_files": int(len(matched)),
        "selected_files": int(len(selected)),
        "duplicate_groups": int(
            ranked.groupby(["region_key", "year", "chunk_start", "chunk_end"]).size().gt(1).sum()
        ),
        "unmatched_files": int(len(unmatched)),
        "downloaded_phase1_chunks": int(plan_status["downloaded"].sum()),
        "expected_phase1_chunks": int(len(plan_status)),
        "fully_downloaded_region_years": region_year.loc[
            region_year["download_completion"] == 1,
            ["glambie_region_key", "year"],
        ].to_dict(orient="records"),
        "inventory_csv": str(inventory_csv),
        "selected_csv": str(selected_csv),
        "duplicate_csv": str(duplicate_csv),
        "plan_status_csv": str(plan_status_csv),
        "region_year_csv": str(region_year_csv),
        "unmatched_csv": str(unmatched_csv),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
