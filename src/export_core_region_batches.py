from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROCESSED_DIR = Path(r"E:\Glacier\data\processed")
PREPARED_DIR = Path(r"E:\Glacier\data\prepared\region_batches")
MASTER_GPKG = PROCESSED_DIR / "analysis_lake_master.gpkg"
MANIFEST_CSV = PREPARED_DIR / "region_manifest.csv"
SUMMARY_JSON = PREPARED_DIR / "region_manifest_summary.json"


def _slug(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def main() -> None:
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    lakes = gpd.read_file(MASTER_GPKG, layer="analysis_master")

    manifest_rows = []
    for region_id, region_df in lakes.groupby("glambie_region_id"):
        region_name = region_df["rgi_region_name"].iloc[0]
        region_key = region_df["glambie_region_key"].iloc[0]
        slug = f"{int(region_id):02d}_{_slug(region_key or region_name)}"
        out_path = PREPARED_DIR / f"{slug}.geojson"
        region_df.to_file(out_path, driver="GeoJSON", engine="pyogrio")

        manifest_rows.append(
            {
                "glambie_region_id": int(region_id),
                "rgi_region_name": region_name,
                "glambie_region_key": region_key,
                "output_geojson": str(out_path),
                "lake_count": int(len(region_df)),
                "core_lake_count": int(region_df["main_analysis_include"].fillna(False).astype(bool).sum()),
                "supplemental_lake_count": int(region_df["analysis_tier"].eq("supplemental").sum()),
                "total_area_0_km2": float(region_df["area_0_km2"].sum()),
                "contacted_count": int((region_df["harmonized_class"] == "proglacial_contacted").sum()),
                "detached_count": int((region_df["harmonized_class"] == "proglacial_detached").sum()),
                "supraglacial_count": int((region_df["harmonized_class"] == "supraglacial").sum()),
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values("glambie_region_id")
    manifest.to_csv(MANIFEST_CSV, index=False)

    summary = {
        "region_file_count": int(len(manifest)),
        "total_lakes": int(manifest["lake_count"].sum()),
        "total_core_lakes": int(manifest["core_lake_count"].sum()),
        "total_supplemental_lakes": int(manifest["supplemental_lake_count"].sum()),
        "largest_region_by_lake_count": manifest.sort_values("lake_count", ascending=False).iloc[0]["rgi_region_name"] if len(manifest) else None,
        "manifest_csv": str(MANIFEST_CSV),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
