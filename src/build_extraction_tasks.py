from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
CONFIG_PATH = ROOT / "configs" / "annual_area_extraction_config.json"


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = Path(cfg["data_paths"]["region_manifest_csv"])
    task_dir = Path(cfg["data_paths"]["extraction_task_dir"])
    task_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    years = list(range(cfg["analysis_scope"]["year_start"], cfg["analysis_scope"]["year_end"] + 1))
    test_regions = set(cfg["test_regions"])
    default_months = cfg["annual_window"]["default_months"]
    region_overrides = cfg["annual_window"]["region_overrides"]

    region_rows = []
    year_rows = []
    for row in manifest.itertuples(index=False):
        region_key = row.glambie_region_key
        months = region_overrides.get(region_key, default_months)
        region_rows.append(
            {
                "glambie_region_id": row.glambie_region_id,
                "rgi_region_name": row.rgi_region_name,
                "glambie_region_key": region_key,
                "output_geojson": row.output_geojson,
                "lake_count": row.lake_count,
                "core_lake_count": row.core_lake_count,
                "supplemental_lake_count": row.supplemental_lake_count,
                "total_area_0_km2": row.total_area_0_km2,
                "contacted_count": row.contacted_count,
                "detached_count": row.detached_count,
                "supraglacial_count": row.supraglacial_count,
                "months": ",".join(str(m) for m in months),
                "is_test_region": region_key in test_regions,
            }
        )
        for year in years:
            year_rows.append(
                {
                    "glambie_region_id": row.glambie_region_id,
                    "glambie_region_key": region_key,
                    "rgi_region_name": row.rgi_region_name,
                    "output_geojson": row.output_geojson,
                    "year": year,
                    "months": ",".join(str(m) for m in months),
                    "is_test_region": region_key in test_regions,
                }
            )

    region_df = pd.DataFrame(region_rows).sort_values("glambie_region_id")
    year_df = pd.DataFrame(year_rows).sort_values(["glambie_region_id", "year"])
    test_df = year_df[year_df["is_test_region"]].copy()

    region_df.to_csv(task_dir / "region_tasks.csv", index=False)
    year_df.to_csv(task_dir / "region_year_tasks_full.csv", index=False)
    test_df.to_csv(task_dir / "region_year_tasks_test.csv", index=False)

    summary = {
        "config_path": str(CONFIG_PATH),
        "region_manifest_csv": str(manifest_path),
        "region_count": int(len(region_df)),
        "year_task_count_full": int(len(year_df)),
        "year_task_count_test": int(len(test_df)),
        "test_regions": sorted(test_regions),
    }
    (task_dir / "task_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
