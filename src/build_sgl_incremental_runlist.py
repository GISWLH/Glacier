from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
PARAMS_PATH = ROOT / "data" / "prepared" / "formal_parameters" / "formal_extraction_parameter_table.csv"
MANIFEST_PATH = ROOT / "data" / "prepared" / "region_batches" / "region_manifest.csv"
OUT_DIR = ROOT / "data" / "prepared" / "execution_plan"
YEARS = list(range(2000, 2025))


def settings_for_year(region_key: str, year: int, row: pd.Series) -> tuple[str, int, str]:
    if region_key == "greenland_periphery":
        if 2000 <= year <= 2005:
            return row["early_year_months"], int(row["early_year_chunk_size"]), "early"
        if 2019 <= year <= 2024:
            return row["late_year_months"], int(row["late_year_chunk_size"]), "late"
        return row["late_year_months"], int(row["early_year_chunk_size"]), "middle"

    if region_key == "central_asia":
        return row["early_year_months"], int(row["early_year_chunk_size"]), "all_years_default"

    return row["early_year_months"], int(row["early_year_chunk_size"]), "default"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    params = pd.read_csv(PARAMS_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)[["glambie_region_key", "rgi_region_name", "supraglacial_count"]]
    df = params.merge(manifest, on=["glambie_region_key", "rgi_region_name"], how="left")
    df = df[df["supraglacial_count"].fillna(0).astype(int) > 0].copy()

    rows = []
    for row in df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        region_key = row_s["glambie_region_key"]
        lake_count = int(row_s["supraglacial_count"])
        for year in YEARS:
            months, chunk_size, period_label = settings_for_year(region_key, year, row_s)
            chunk_count = math.ceil(lake_count / chunk_size)
            for chunk_id in range(chunk_count):
                start_index = chunk_id * chunk_size
                end_index_exclusive = min((chunk_id + 1) * chunk_size, lake_count)
                rows.append(
                    {
                        "glambie_region_key": region_key,
                        "rgi_region_name": row_s["rgi_region_name"],
                        "year": year,
                        "period_label": period_label,
                        "months": months,
                        "chunk_size": chunk_size,
                        "chunk_id": chunk_id,
                        "start_index": start_index,
                        "end_index_exclusive": end_index_exclusive,
                        "expected_lake_count": end_index_exclusive - start_index,
                        "lake_type_filter": "SGL",
                        "status": "pending",
                        "notes": "Incremental supraglacial backfill only.",
                        "task_id": f"sgl_{region_key}_{year}_chunk_{chunk_id:03d}",
                    }
                )

    out = pd.DataFrame(rows).sort_values(["glambie_region_key", "year", "chunk_id"]).reset_index(drop=True)
    csv_path = OUT_DIR / "sgl_incremental_runlist.csv"
    json_path = OUT_DIR / "sgl_incremental_runlist.json"
    summary_path = OUT_DIR / "sgl_incremental_runlist_summary.json"

    out.to_csv(csv_path, index=False)
    json_path.write_text(out.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")

    summary = {
        "task_count": int(len(out)),
        "region_count": int(out["glambie_region_key"].nunique()) if not out.empty else 0,
        "year_count": int(out["year"].nunique()) if not out.empty else 0,
        "regions": sorted(out["glambie_region_key"].dropna().astype(str).unique().tolist()) if not out.empty else [],
        "csv_path": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
