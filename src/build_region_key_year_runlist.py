from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
PARAMS_PATH = ROOT / "data" / "prepared" / "formal_parameters" / "formal_extraction_parameter_table.csv"
OUT_DIR = ROOT / "data" / "prepared" / "execution_plan"

DEFAULT_KEY_YEARS = [2000, 2005, 2010, 2015, 2020, 2024]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a key-year runlist for one region.")
    p.add_argument("--region-key", required=True, help="glambie_region_key, e.g. western_canada_us")
    p.add_argument("--years", default=",".join(str(y) for y in DEFAULT_KEY_YEARS), help="Comma-separated key years")
    p.add_argument(
        "--output-prefix",
        default="",
        help="Optional output filename prefix. Defaults to {region_key}_key_years.",
    )
    return p.parse_args()


def settings_for_year(region_key: str, year: int, row: pd.Series) -> tuple[str, int, str]:
    if region_key == "greenland_periphery":
        if 2000 <= year <= 2005:
            return row["early_year_months"], int(row["early_year_chunk_size"]), "early"
        if 2019 <= year <= 2024:
            return row["late_year_months"], int(row["late_year_chunk_size"]), "late"
        return row["late_year_months"], int(row["early_year_chunk_size"]), "middle"

    months = row["early_year_months"]
    chunk_size = int(row["early_year_chunk_size"])
    if 2019 <= year <= 2024:
        months = row["late_year_months"]
        chunk_size = int(row["late_year_chunk_size"])
        return months, chunk_size, "late"
    if 2000 <= year <= 2005:
        return months, chunk_size, "early"
    return months, chunk_size, "middle"


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    years = [int(part) for part in args.years.split(",") if part.strip()]
    output_prefix = args.output_prefix or f"{args.region_key}_key_years"
    params = pd.read_csv(PARAMS_PATH)
    region_row = params.loc[params["glambie_region_key"] == args.region_key]
    if region_row.empty:
        raise SystemExit(f"Region not found in parameter table: {args.region_key}")
    row = region_row.iloc[0]

    key_year_rows = []
    runlist_rows = []
    lake_count = int(row["lake_count"])
    for year in years:
        months, chunk_size, period_label = settings_for_year(args.region_key, year, row)
        chunk_count = math.ceil(lake_count / chunk_size)
        key_year_rows.append(
            {
                "year": year,
                "role": {
                    2000: "baseline_early",
                    2005: "early_transition",
                    2010: "mid_transition",
                    2015: "pre_recent_warm",
                    2020: "recent_reference",
                    2024: "latest_endpoint",
                }.get(year, "custom_key_year"),
                "rationale": {
                    2000: "early baseline year",
                    2005: "early period control point",
                    2010: "mid-period structural checkpoint",
                    2015: "late pre-recent benchmark",
                    2020: "recent reference year",
                    2024: "latest endpoint for skeleton series",
                }.get(year, "user-defined key year"),
                "current_status": "pending_submission",
            }
        )
        for chunk_id in range(chunk_count):
            start_index = chunk_id * chunk_size
            end_index_exclusive = min((chunk_id + 1) * chunk_size, lake_count)
            runlist_rows.append(
                {
                    "glambie_region_key": args.region_key,
                    "rgi_region_name": row["rgi_region_name"],
                    "year": year,
                    "period_label": period_label,
                    "months": months,
                    "chunk_size": chunk_size,
                    "chunk_id": chunk_id,
                    "start_index": start_index,
                    "end_index_exclusive": end_index_exclusive,
                    "expected_lake_count": end_index_exclusive - start_index,
                    "status": "pending",
                    "notes": "",
                    "task_id": f"{args.region_key}_{year}_chunk_{chunk_id:03d}",
                }
            )

    key_year_df = pd.DataFrame(key_year_rows).sort_values("year")
    runlist_df = pd.DataFrame(runlist_rows).sort_values(["year", "chunk_id"]).reset_index(drop=True)
    summary_df = (
        runlist_df.groupby(["year", "months", "chunk_size"], as_index=False)
        .agg(task_count=("task_id", "count"))
        .sort_values("year")
    )

    key_year_path = OUT_DIR / f"{output_prefix}.csv"
    runlist_path = OUT_DIR / f"{output_prefix}_runlist.csv"
    summary_path = OUT_DIR / f"{output_prefix}_runlist_summary.csv"
    manifest_path = OUT_DIR / f"{output_prefix}_runlist_manifest.json"

    key_year_df.to_csv(key_year_path, index=False)
    runlist_df.to_csv(runlist_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    payload = {
        "region_key": args.region_key,
        "rgi_region_name": row["rgi_region_name"],
        "lake_count": lake_count,
        "years": years,
        "key_year_csv": str(key_year_path),
        "runlist_csv": str(runlist_path),
        "summary_csv": str(summary_path),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**payload, "manifest_json": str(manifest_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
