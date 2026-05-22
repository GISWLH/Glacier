from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_STATUS = ROOT / "data" / "interim" / "annual_area_qc" / "downloaded_exports_status_phase1_region_year_status.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "analysis" / "region_download_coverage.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize region/year download coverage for phase1 exports.")
    p.add_argument("--status-csv", default=str(DEFAULT_STATUS))
    p.add_argument(
        "--years",
        default="2000,2005,2010,2015,2020,2024",
        help="Comma-separated key years to check.",
    )
    p.add_argument("--out-csv", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def parse_years(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    years = parse_years(args.years)
    df = pd.read_csv(args.status_csv)

    key = "glambie_region_key"
    subset = df[df["year"].isin(years)].copy()
    if subset.empty:
        raise SystemExit("No matching rows for requested years.")

    summary = (
        subset.groupby(key)
        .agg(
            years_present=("year", "nunique"),
            total_expected_chunks=("expected_chunks", "sum"),
            total_downloaded_chunks=("downloaded_chunks", "sum"),
            mean_completion=("download_completion", "mean"),
        )
        .reset_index()
    )

    incomplete_years = (
        subset[subset["download_completion"] < 1]
        .groupby(key)["year"]
        .apply(lambda s: sorted(set(s.tolist())))
        .reset_index()
        .rename(columns={"year": "incomplete_years"})
    )
    summary = summary.merge(incomplete_years, on=key, how="left")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)

    payload = {
        "years": years,
        "regions": summary[key].tolist(),
        "out_csv": str(out_path),
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
