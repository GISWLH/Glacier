from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd


ROOT = Path(r"E:\Glacier")
MANIFEST_CSV = ROOT / "data" / "prepared" / "region_batches" / "region_manifest.csv"
OUT_ROOT = ROOT / "data" / "prepared" / "gee_uploads_slim"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a GEE upload shapefile zip for one region.")
    p.add_argument("--region-key", required=True, help="glambie_region_key, e.g. western_canada_us")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(MANIFEST_CSV)
    row = manifest.loc[manifest["glambie_region_key"] == args.region_key]
    if row.empty:
        raise SystemExit(f"Region not found in region manifest: {args.region_key}")
    row = row.iloc[0]

    region_id = int(row["glambie_region_id"])
    region_key = str(row["glambie_region_key"])
    base_name = f"{region_id:02d}_{region_key}"
    out_name = f"{base_name}_gee_upload"
    source_geojson = Path(row["output_geojson"])

    out_dir = OUT_ROOT / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(source_geojson)
    gdf = gdf[
        [
            "lake_id",
            "lake_type",
            "harmonized_class",
            "rgi_region_name",
            "glambie_region_key",
            "area_0_km2",
            "elevation_m",
            "latitude",
            "longitude",
            "geometry",
        ]
    ].rename(
        columns={
            "harmonized_class": "hclass",
            "rgi_region_name": "rgi_name",
            "glambie_region_key": "glmb_key",
            "area_0_km2": "area0_km2",
            "elevation_m": "elev_m",
            "latitude": "lat",
            "longitude": "lon",
        }
    )
    shp_path = out_dir / f"{base_name}.shp"
    gdf.to_file(shp_path, driver="ESRI Shapefile", engine="pyogrio")

    zip_path = OUT_ROOT / f"{out_name}.zip"
    sidecar_paths = [shp_path.with_suffix(ext) for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sidecar_paths:
            if path.exists():
                zf.write(path, arcname=path.name)

    payload = {
        "region_key": region_key,
        "region_id": region_id,
        "source_geojson": str(source_geojson),
        "output_dir": str(out_dir),
        "zip_path": str(zip_path),
        "files": [str(path) for path in sidecar_paths if path.exists()],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
