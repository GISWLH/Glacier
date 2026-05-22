from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROCESSED_DIR = Path(r"E:\Glacier\data\processed")
RAW_GLAMBIE_DIR = Path(r"E:\Glacier\data\raw\GlaMBIE_Data\glambie_results_20240716\calendar_years")

CLEAN_GPKG = PROCESSED_DIR / "clean_global_glacier_fed_lakes.gpkg"
MASTER_GPKG = PROCESSED_DIR / "analysis_lake_master.gpkg"
MASTER_CSV = PROCESSED_DIR / "analysis_lake_master.csv"
CORE_GPKG = PROCESSED_DIR / "analysis_lake_core.gpkg"
CORE_CSV = PROCESSED_DIR / "analysis_lake_core.csv"
REGION_SUMMARY_CSV = PROCESSED_DIR / "analysis_lake_region_summary.csv"
SUMMARY_JSON = PROCESSED_DIR / "analysis_lake_summary.json"


GLAMBIE_REGION_MAP = {
    1: ("alaska", "1_alaska.csv"),
    2: ("western_canada_us", "2_western_canada_us.csv"),
    3: ("arctic_canada_north", "3_arctic_canada_north.csv"),
    4: ("arctic_canada_south", "4_arctic_canada_south.csv"),
    5: ("greenland_periphery", "5_greenland_periphery.csv"),
    6: ("iceland", "6_iceland.csv"),
    7: ("svalbard", "7_svalbard.csv"),
    8: ("scandinavia", "8_scandinavia.csv"),
    9: ("russian_arctic", "9_russian_arctic.csv"),
    10: ("north_asia", "10_north_asia.csv"),
    11: ("central_europe", "11_central_europe.csv"),
    12: ("caucasus_middle_east", "12_caucasus_middle_east.csv"),
    13: ("central_asia", "13_central_asia.csv"),
    14: ("south_asia_west", "14_south_asia_west.csv"),
    15: ("south_asia_east", "15_south_asia_east.csv"),
    16: ("low_latitudes", "16_low_latitudes.csv"),
    17: ("southern_andes", "17_southern_andes.csv"),
    18: ("new_zealand", "18_new_zealand.csv"),
    19: ("antarctic_and_subantarctic", "19_antarctic_and_subantarctic.csv"),
}


def _load_clean() -> gpd.GeoDataFrame:
    lakes = gpd.read_file(CLEAN_GPKG, layer="clean_lakes")
    lakes["analysis_tier"] = "supplemental"
    lakes.loc[lakes["lake_type"].isin(["IUL", "ICL"]), "analysis_tier"] = "core"
    lakes["main_analysis_include"] = lakes["analysis_tier"].eq("core")

    lakes["harmonized_class"] = "supraglacial"
    lakes.loc[lakes["lake_type"] == "IUL", "harmonized_class"] = "proglacial_detached"
    lakes.loc[lakes["lake_type"] == "ICL", "harmonized_class"] = "proglacial_contacted"

    lakes["contact_qc_flag"] = "consistent"
    lakes.loc[(lakes["lake_type"] == "ICL") & (~lakes["rgi2000_contact"]), "contact_qc_flag"] = "type_contact_mismatch"
    lakes.loc[(lakes["lake_type"] == "SGL") & (~lakes["rgi2000_contact"]), "contact_qc_flag"] = "type_contact_mismatch"
    lakes.loc[(lakes["lake_type"] == "IUL") & (lakes["rgi2000_contact"]), "contact_qc_flag"] = "type_contact_mismatch"
    return lakes


def _attach_glambie_fields(lakes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    lakes["glambie_region_id"] = lakes["rgi_o1region"].astype("Int64")
    lakes["glambie_region_key"] = pd.NA
    lakes["glambie_calendar_file"] = pd.NA
    lakes["glambie_calendar_path"] = pd.NA

    for rid, (key, filename) in GLAMBIE_REGION_MAP.items():
        mask = lakes["glambie_region_id"] == rid
        lakes.loc[mask, "glambie_region_key"] = key
        lakes.loc[mask, "glambie_calendar_file"] = filename
        lakes.loc[mask, "glambie_calendar_path"] = str(RAW_GLAMBIE_DIR / filename)

    lakes["has_glambie_region"] = lakes["glambie_region_key"].notna()
    return lakes


def _region_summary(lakes: pd.DataFrame) -> pd.DataFrame:
    summary = (
        lakes.groupby(["glambie_region_id", "rgi_region_name", "analysis_tier", "harmonized_class"], dropna=False)
        .agg(
            lake_count=("lake_id", "count"),
            total_area_0_km2=("area_0_km2", "sum"),
            median_area_0_km2=("area_0_km2", "median"),
            contact_count=("rgi2000_contact", "sum"),
            mismatch_count=("type_contact_consistent", lambda s: int((~s).sum())),
        )
        .reset_index()
        .sort_values(["glambie_region_id", "analysis_tier", "harmonized_class"])
    )
    return summary


def _summary_json(master: pd.DataFrame, core: pd.DataFrame) -> dict:
    return {
        "master_rows": int(len(master)),
        "core_rows": int(len(core)),
        "supplemental_rows": int((master["analysis_tier"] == "supplemental").sum()),
        "core_classes": {k: int(v) for k, v in core["harmonized_class"].value_counts().sort_index().items()},
        "master_contact_qc": {k: int(v) for k, v in master["contact_qc_flag"].value_counts().sort_index().items()},
        "regions_with_core_lakes": int(core["glambie_region_id"].nunique()),
        "regions_without_glambie_mapping_rows": int((~master["has_glambie_region"]).sum()),
        "core_selection_rule": "Main analysis includes IUL and ICL only; SGL retained as supplemental due to lower temporal stability for annual open-water extent analysis.",
    }


def main() -> None:
    lakes = _load_clean()
    lakes = _attach_glambie_fields(lakes)

    master_cols = [
        "lake_id",
        "source_id",
        "lake_type",
        "lake_type_label",
        "analysis_tier",
        "main_analysis_include",
        "harmonized_class",
        "contact_qc_flag",
        "area_0_km2",
        "area_error_km2",
        "perimeter_km",
        "elevation_m",
        "latitude",
        "longitude",
        "giglak_region",
        "rgi_o1region",
        "rgi_region_name",
        "glambie_region_id",
        "glambie_region_key",
        "glambie_calendar_file",
        "glambie_calendar_path",
        "has_glambie_region",
        "contacting_glacier_count",
        "rgi2000_contact",
        "glacier_contact_status_rgi2000",
        "expected_contact_from_type",
        "type_contact_consistent",
        "geometry",
    ]
    master = lakes[master_cols].copy()
    core = master[master["main_analysis_include"]].copy()
    region_summary = _region_summary(master.drop(columns="geometry"))

    master.to_file(MASTER_GPKG, layer="analysis_master", driver="GPKG", engine="pyogrio")
    master.drop(columns="geometry").to_csv(MASTER_CSV, index=False)
    core.to_file(CORE_GPKG, layer="analysis_core", driver="GPKG", engine="pyogrio")
    core.drop(columns="geometry").to_csv(CORE_CSV, index=False)
    region_summary.to_csv(REGION_SUMMARY_CSV, index=False)

    summary = _summary_json(master.drop(columns="geometry"), core.drop(columns="geometry"))
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
