from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.validation import make_valid


RAW_DIR = Path(r"E:\Glacier\data\raw")
PROCESSED_DIR = Path(r"E:\Glacier\data\processed")
SRC_EXPORT_DIR = Path(r"E:\Glacier\src")

GIGLAK_PATH = RAW_DIR / "GIGLak_dataset" / "Data_Glacial lake layer" / "Global_Lake_Dataset.shp"
RGI_G_DIR = RAW_DIR / "RGI2000" / "RGI2000-v7.0-G-global"
RGI_O1_PATH = RAW_DIR / "RGI2000" / "RGI2000-v7.0-regions" / "RGI2000-v7.0-o1regions.shp"

AUDIT_GPKG = PROCESSED_DIR / "giglak_rgi7_audit.gpkg"
AUDIT_CSV = PROCESSED_DIR / "giglak_rgi7_audit.csv"
CLEAN_GPKG = PROCESSED_DIR / "clean_global_glacier_fed_lakes.gpkg"
CLEAN_CSV = PROCESSED_DIR / "clean_global_glacier_fed_lakes.csv"
SUMMARY_JSON = PROCESSED_DIR / "clean_global_glacier_fed_lakes_summary.json"


TYPE_LABELS = {
    "NGFL": "non_glacier_fed_lake",
    "IUL": "ice_uncontacted_proglacial_lake",
    "ICL": "ice_contacted_proglacial_lake",
    "SGL": "supraglacial_lake",
}


def _ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SRC_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _read_giglak() -> gpd.GeoDataFrame:
    lakes = gpd.read_file(GIGLAK_PATH)
    lakes = lakes.rename(
        columns={
            "ID": "source_id",
            "Lake_Type": "lake_type",
            "Area_km2": "area_0_km2",
            "Area_error": "area_error_km2",
            "Elevation": "elevation_m",
            "Region": "giglak_region",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Perimeter": "perimeter_km",
        }
    )
    lakes["lake_id"] = lakes["source_id"].astype(str)
    lakes["lake_type_label"] = lakes["lake_type"].map(TYPE_LABELS).fillna("unknown")
    lakes["is_valid_geom"] = lakes.geometry.notna() & ~lakes.geometry.is_empty
    invalid = ~lakes.geometry.is_valid
    if invalid.any():
        lakes.loc[invalid, "geometry"] = lakes.loc[invalid, "geometry"].map(make_valid)
    lakes["is_valid_geom"] = lakes.geometry.notna() & ~lakes.geometry.is_empty & lakes.geometry.is_valid
    lakes["positive_area"] = lakes["area_0_km2"].fillna(0) > 0
    lakes["keep_type"] = lakes["lake_type"].isin(["IUL", "ICL", "SGL"])
    lakes["exclude_reason"] = pd.NA
    lakes.loc[~lakes["keep_type"], "exclude_reason"] = "non_glacier_fed_type"
    lakes.loc[lakes["keep_type"] & ~lakes["is_valid_geom"], "exclude_reason"] = "invalid_geometry"
    lakes.loc[lakes["keep_type"] & lakes["is_valid_geom"] & ~lakes["positive_area"], "exclude_reason"] = "non_positive_area"
    lakes["keep_for_clean_db"] = lakes["keep_type"] & lakes["is_valid_geom"] & lakes["positive_area"]
    return lakes


def _assign_rgi_regions(lakes: gpd.GeoDataFrame) -> pd.DataFrame:
    regions = gpd.read_file(RGI_O1_PATH)[["o1region", "full_name", "geometry"]]
    points = gpd.GeoDataFrame(
        lakes[["lake_id"]].copy(),
        geometry=gpd.points_from_xy(lakes["longitude"], lakes["latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, regions, how="left", predicate="intersects")
    joined = joined[["lake_id", "o1region", "full_name"]].drop_duplicates(subset=["lake_id"])
    joined = joined.rename(columns={"o1region": "rgi_o1region", "full_name": "rgi_region_name"})
    return joined


def _read_rgi_glacier_polygons() -> gpd.GeoDataFrame:
    zip_paths = sorted(RGI_G_DIR.glob("RGI2000-v7.0-G-*.zip"))
    frames: list[gpd.GeoDataFrame] = []
    for zip_path in zip_paths:
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(zip_path) as zf:
                shp_names = [name for name in zf.namelist() if name.lower().endswith(".shp")]
                if len(shp_names) != 1:
                    raise RuntimeError(f"Expected one shapefile in {zip_path.name}, found {len(shp_names)}")
                zf.extractall(tmpdir)
                shp_path = Path(tmpdir) / shp_names[0]
                glaciers = gpd.read_file(shp_path, columns=[])
                glaciers = glaciers[["geometry"]].copy()
                glaciers["src"] = zip_path.stem
                frames.append(glaciers)
    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs="EPSG:4326")
    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()
    invalid = ~out.geometry.is_valid
    if invalid.any():
        out.loc[invalid, "geometry"] = out.loc[invalid, "geometry"].map(make_valid)
    return out[["geometry"]]


def _assign_contact_status(clean_lakes: gpd.GeoDataFrame, glaciers: gpd.GeoDataFrame) -> pd.DataFrame:
    lakes_wgs84 = clean_lakes.to_crs("EPSG:4326")[["lake_id", "lake_type", "geometry"]].copy()
    joined = gpd.sjoin(lakes_wgs84, glaciers, how="left", predicate="intersects")
    contact = (
        joined.groupby("lake_id", as_index=False)["index_right"]
        .apply(lambda s: int(s.notna().sum()))
        .rename(columns={"index_right": "contacting_glacier_count"})
    )
    contact["rgi2000_contact"] = contact["contacting_glacier_count"] > 0
    return contact


def _build_summary(audit: pd.DataFrame, clean: pd.DataFrame) -> dict:
    mismatch = clean[
        ((clean["lake_type"] == "ICL") | (clean["lake_type"] == "SGL")) & (~clean["rgi2000_contact"])
        | ((clean["lake_type"] == "IUL") & (clean["rgi2000_contact"]))
    ]
    summary = {
        "input_rows": int(len(audit)),
        "clean_rows": int(len(clean)),
        "excluded_rows": int((~audit["keep_for_clean_db"]).sum()),
        "clean_by_type": {k: int(v) for k, v in clean["lake_type"].value_counts().sort_index().items()},
        "clean_by_rgi_region": {str(k): int(v) for k, v in clean["rgi_region_name"].fillna("unassigned").value_counts().sort_index().items()},
        "rgi2000_contact_true": int(clean["rgi2000_contact"].sum()),
        "rgi2000_contact_false": int((~clean["rgi2000_contact"]).sum()),
        "type_contact_mismatch_rows": int(len(mismatch)),
        "type_contact_mismatch_share": float(len(mismatch) / len(clean)) if len(clean) else 0.0,
        "selection_rules": {
            "kept_lake_types": ["IUL", "ICL", "SGL"],
            "excluded_lake_types": ["NGFL"],
            "required_positive_area": True,
            "required_valid_geometry": True,
        },
        "rgi_choice": "RGI2000-v7.0-G-global glacier polygon product",
        "rgi_region_choice": "RGI2000-v7.0-o1regions.shp",
        "glambie_choice_for_next_step": "glambie_results_20240716/calendar_years regional CSV files",
    }
    return summary


def main() -> None:
    _ensure_dirs()

    lakes = _read_giglak()
    region_map = _assign_rgi_regions(lakes)
    lakes = lakes.merge(region_map, on="lake_id", how="left")

    clean = lakes[lakes["keep_for_clean_db"]].copy()
    glaciers = _read_rgi_glacier_polygons()
    contact = _assign_contact_status(clean, glaciers)
    clean = clean.merge(contact, on="lake_id", how="left")
    clean["contacting_glacier_count"] = clean["contacting_glacier_count"].fillna(0).astype(int)
    clean["rgi2000_contact"] = clean["rgi2000_contact"].fillna(False)
    clean["glacier_contact_status_rgi2000"] = clean["rgi2000_contact"].map({True: "contact", False: "non_contact"})
    clean["expected_contact_from_type"] = clean["lake_type"].isin(["ICL", "SGL"])
    clean["type_contact_consistent"] = clean["expected_contact_from_type"] == clean["rgi2000_contact"]

    audit = lakes.merge(
        clean[
            [
                "lake_id",
                "rgi_o1region",
                "rgi_region_name",
                "contacting_glacier_count",
                "rgi2000_contact",
                "glacier_contact_status_rgi2000",
                "expected_contact_from_type",
                "type_contact_consistent",
            ]
        ],
        on=["lake_id", "rgi_o1region", "rgi_region_name"],
        how="left",
    )

    clean_cols = [
        "lake_id",
        "source_id",
        "lake_type",
        "lake_type_label",
        "area_0_km2",
        "area_error_km2",
        "perimeter_km",
        "elevation_m",
        "latitude",
        "longitude",
        "giglak_region",
        "rgi_o1region",
        "rgi_region_name",
        "contacting_glacier_count",
        "rgi2000_contact",
        "glacier_contact_status_rgi2000",
        "expected_contact_from_type",
        "type_contact_consistent",
        "geometry",
    ]
    clean = clean[clean_cols].copy()

    audit.to_file(AUDIT_GPKG, layer="audit", driver="GPKG", engine="pyogrio")
    audit.drop(columns="geometry").to_csv(AUDIT_CSV, index=False)
    clean.to_file(CLEAN_GPKG, layer="clean_lakes", driver="GPKG", engine="pyogrio")
    clean.drop(columns="geometry").to_csv(CLEAN_CSV, index=False)

    summary = _build_summary(audit, clean)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
