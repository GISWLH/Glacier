from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
PARAMS_PATH = ROOT / "data" / "prepared" / "formal_parameters" / "formal_extraction_parameter_table.csv"
OUT_DIR = ROOT / "data" / "prepared" / "execution_plan"


PHASE1_REGIONS = {"greenland_periphery", "central_asia"}
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


def phase1_wave(region_key: str, year: int) -> str:
    if region_key == "greenland_periphery":
        if year in {2000, 2020}:
            return "wave_1_anchor_years"
        if 2001 <= year <= 2005:
            return "wave_2_early_years"
        if 2019 <= year <= 2024:
            return "wave_3_late_years"
        return "wave_4_middle_years"

    if region_key == "central_asia":
        if year == 2000:
            return "wave_1_anchor_years"
        if 2001 <= year <= 2005:
            return "wave_2_early_years"
        if 2019 <= year <= 2024:
            return "wave_3_late_years"
        return "wave_4_middle_years"

    return "wave_unknown"


def tested_status(region_key: str, year: int, chunk_id: int, months: str, chunk_size: int) -> tuple[str, str]:
    if region_key == "greenland_periphery" and year == 2000 and chunk_id == 0 and months == "6,7,8,9" and chunk_size == 500:
        return "tested_passed", "Validated anchor chunk. Use this configuration for Greenland early years."
    if region_key == "greenland_periphery" and year == 2020 and chunk_id == 0 and months == "7,8,9" and chunk_size == 200:
        return "tested_passed", "Validated anchor chunk. Use this configuration for Greenland later years."
    if region_key == "central_asia" and year == 2000 and chunk_id == 0 and months == "7,8,9" and chunk_size == 500:
        return "tested_passed", "Validated anchor chunk. Keep 7-9 as the default Central Asia window."
    return "pending", ""


def build_runlist() -> pd.DataFrame:
    params = pd.read_csv(PARAMS_PATH)
    params = params[params["glambie_region_key"].isin(PHASE1_REGIONS)].copy()

    rows = []
    for row in params.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        region_key = row_s["glambie_region_key"]
        lake_count = int(row_s["lake_count"])

        for year in YEARS:
            months, chunk_size, period_label = settings_for_year(region_key, year, row_s)
            chunk_count = math.ceil(lake_count / chunk_size)
            for chunk_id in range(chunk_count):
                start_index = chunk_id * chunk_size
                end_index_exclusive = min((chunk_id + 1) * chunk_size, lake_count)
                status, notes = tested_status(region_key, year, chunk_id, months, chunk_size)
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
                        "phase1_wave": phase1_wave(region_key, year),
                        "status": status,
                        "notes": notes,
                        "task_id": f"{region_key}_{year}_chunk_{chunk_id:03d}",
                    }
                )

    df = pd.DataFrame(rows)
    wave_order = {
        "wave_1_anchor_years": 1,
        "wave_2_early_years": 2,
        "wave_3_late_years": 3,
        "wave_4_middle_years": 4,
    }
    df["wave_order"] = df["phase1_wave"].map(wave_order)
    df = df.sort_values(["wave_order", "glambie_region_key", "year", "chunk_id"]).reset_index(drop=True)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_runlist()

    csv_path = OUT_DIR / "phase1_batch_runlist.csv"
    json_path = OUT_DIR / "phase1_batch_runlist.json"
    summary_csv = OUT_DIR / "phase1_batch_runlist_summary.csv"
    summary_json = OUT_DIR / "phase1_batch_runlist_summary.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")

    summary = (
        df.groupby(["glambie_region_key", "phase1_wave"])
        .agg(task_count=("task_id", "count"), tested_passed=("status", lambda s: int((s == "tested_passed").sum())))
        .reset_index()
        .sort_values(["glambie_region_key", "phase1_wave"])
    )
    summary.to_csv(summary_csv, index=False)

    payload = {
        "task_count": int(len(df)),
        "tested_passed_tasks": int((df["status"] == "tested_passed").sum()),
        "anchor_years": df.loc[df["phase1_wave"] == "wave_1_anchor_years", ["glambie_region_key", "year"]]
        .drop_duplicates()
        .to_dict(orient="records"),
        "csv_path": str(csv_path),
    }
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
