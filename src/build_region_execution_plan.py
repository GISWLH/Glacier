from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
PARAMS_PATH = ROOT / "data" / "prepared" / "formal_parameters" / "formal_extraction_parameter_table.csv"
OUT_DIR = ROOT / "data" / "prepared" / "execution_plan"


PHASE_ORDER = {
    "validated_test_region": 1,
    "high_priority_large_region": 2,
    "medium_priority_region": 3,
    "standard_region": 4,
}

PHASE_LABEL = {
    1: "phase_1_validated_scaleout",
    2: "phase_2_large_regions",
    3: "phase_3_medium_regions",
    4: "phase_4_standard_regions",
}


def first_pilot_task(row: pd.Series) -> str:
    if row["glambie_region_key"] == "greenland_periphery":
        return "Run 2000 chunk 0 with months 6,7,8,9 and chunk_size 500, then run 2020 chunk 0 with months 7,8,9 and chunk_size 200."
    if row["glambie_region_key"] == "central_asia":
        return "Use 7,8,9 as the default window. Start with 2000 chunk 0 and then extend to additional years using chunk_size 500."
    return (
        f"Start with 2000 chunk 0 using months {row['early_year_months']} "
        f"and chunk_size {int(row['early_year_chunk_size'])}, then validate a later year using months {row['late_year_months']}."
    )


def production_strategy(row: pd.Series) -> str:
    if row["priority_class"] == "validated_test_region":
        return "Ready for controlled scale-out using the validated region-specific settings."
    if row["priority_class"] == "high_priority_large_region":
        return "Large region: validate one early-year chunk and one late-year chunk before batch expansion."
    if row["priority_class"] == "medium_priority_region":
        return "Medium-size region: start with the default configuration and expand if QC is stable."
    return "Standard region: default configuration is acceptable for the first production pass."


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PARAMS_PATH)
    df["phase_number"] = df["priority_class"].map(PHASE_ORDER)
    df["phase_label"] = df["phase_number"].map(PHASE_LABEL)
    df["run_rank_within_phase"] = (
        df.sort_values(["phase_number", "lake_count"], ascending=[True, False])
        .groupby("phase_number")
        .cumcount()
        .add(1)
    )
    df["recommended_first_task"] = df.apply(first_pilot_task, axis=1)
    df["recommended_production_strategy"] = df.apply(production_strategy, axis=1)
    df["recommended_status"] = "pending"
    df["notes"] = ""

    df = df.sort_values(["phase_number", "run_rank_within_phase", "glambie_region_id"]).reset_index(drop=True)

    csv_path = OUT_DIR / "region_execution_plan.csv"
    json_path = OUT_DIR / "region_execution_plan.json"
    summary_path = OUT_DIR / "region_execution_plan_summary.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")

    phase_summary = (
        df.groupby(["phase_number", "phase_label"])
        .agg(region_count=("glambie_region_key", "count"), total_lakes=("lake_count", "sum"))
        .reset_index()
        .sort_values("phase_number")
    )
    phase_summary_path = OUT_DIR / "region_execution_plan_phase_summary.csv"
    phase_summary.to_csv(phase_summary_path, index=False)

    summary = {
        "region_count": int(len(df)),
        "phase_summary_csv": str(phase_summary_path),
        "phase_1_regions": df.loc[df["phase_number"] == 1, "glambie_region_key"].tolist(),
        "phase_2_regions": df.loc[df["phase_number"] == 2, "glambie_region_key"].tolist(),
        "csv_path": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
