from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
MANIFEST_PATH = ROOT / "data" / "prepared" / "region_batches" / "region_manifest.csv"
OUT_DIR = ROOT / "data" / "prepared" / "formal_parameters"


def classify_period_strategy(region_key: str) -> dict:
    if region_key == "greenland_periphery":
        return {
            "early_year_months": "6,7,8,9",
            "late_year_months": "7,8,9",
            "early_year_chunk_size": 500,
            "late_year_chunk_size": 200,
            "early_year_notes": "Early years require a wider seasonal window due to sparse image counts.",
            "late_year_notes": "Later years retain the standard 3-month window but require smaller chunks because the image stack is much larger.",
        }
    if region_key == "central_asia":
        return {
            "early_year_months": "7,8,9",
            "late_year_months": "7,8,9",
            "early_year_chunk_size": 500,
            "late_year_chunk_size": 500,
            "early_year_notes": "A 4-month window gave only marginal improvement over 3 months, so 7-9 remains the default.",
            "late_year_notes": "No special later-year rule established yet; keep the default configuration pending broader validation.",
        }
    if region_key in {"southern_andes", "new_zealand"}:
        return {
            "early_year_months": "1,2,3",
            "late_year_months": "1,2,3",
            "early_year_chunk_size": 500,
            "late_year_chunk_size": 500,
            "early_year_notes": "Southern Hemisphere region: use austral summer months 1-3 instead of the Northern Hemisphere default 7-9.",
            "late_year_notes": "Southern Hemisphere region: keep 1-3 as the default austral summer window pending further validation.",
        }
    return {
        "early_year_months": "7,8,9",
        "late_year_months": "7,8,9",
        "early_year_chunk_size": 500,
        "late_year_chunk_size": 500,
        "early_year_notes": "Default configuration. No region-specific override validated yet.",
        "late_year_notes": "Default configuration. No region-specific override validated yet.",
    }


def classify_region_priority(region_key: str, lake_count: int) -> str:
    if region_key in {"greenland_periphery", "central_asia"}:
        return "validated_test_region"
    if lake_count >= 8000:
        return "high_priority_large_region"
    if lake_count >= 3000:
        return "medium_priority_region"
    return "standard_region"


def build_table() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH)
    rows = []
    for row in manifest.itertuples(index=False):
        strat = classify_period_strategy(row.glambie_region_key)
        rows.append(
            {
                "glambie_region_id": int(row.glambie_region_id),
                "glambie_region_key": row.glambie_region_key,
                "rgi_region_name": row.rgi_region_name,
                "lake_count": int(row.lake_count),
                "priority_class": classify_region_priority(row.glambie_region_key, int(row.lake_count)),
                "default_sensor": "landsat_collection2_level2",
                "default_metric": "annual_maximum_open_water_extent_km2",
                "default_qc_min_images": 3,
                "default_qc_min_valid_fraction": 0.7,
                "early_year_definition": "2000-2005",
                "late_year_definition": "2019-2024",
                "early_year_months": strat["early_year_months"],
                "late_year_months": strat["late_year_months"],
                "early_year_chunk_size": strat["early_year_chunk_size"],
                "late_year_chunk_size": strat["late_year_chunk_size"],
                "early_year_notes": strat["early_year_notes"],
                "late_year_notes": strat["late_year_notes"],
            }
        )
    return pd.DataFrame(rows).sort_values("glambie_region_id")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_table()
    csv_path = OUT_DIR / "formal_extraction_parameter_table.csv"
    json_path = OUT_DIR / "formal_extraction_parameter_table.json"
    summary_path = OUT_DIR / "formal_extraction_parameter_summary.json"

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")

    summary = {
        "region_count": int(len(df)),
        "validated_regions": df.loc[df["priority_class"] == "validated_test_region", "glambie_region_key"].tolist(),
        "default_rule": {
            "months": "7,8,9",
            "chunk_size": 500,
            "min_images": 3,
            "min_valid_fraction": 0.7,
        },
        "special_rules": {
            "greenland_periphery": {
                "early_year_months": "6,7,8,9",
                "late_year_chunk_size": 200,
            },
            "central_asia": {
                "early_year_months": "7,8,9",
                "late_year_chunk_size": 500,
            },
            "southern_hemisphere_regions": {
                "regions": ["southern_andes", "new_zealand"],
                "months": "1,2,3",
                "chunk_size": 500,
            },
        },
        "csv_path": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
