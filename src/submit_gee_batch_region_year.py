from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd

try:
    import ee  # type: ignore
except ImportError:
    ee = None


ROOT = Path(r"E:\Glacier")
RUNLIST_PATH = ROOT / "data" / "prepared" / "execution_plan" / "phase1_batch_runlist.csv"
ASSET_CFG_PATH = ROOT / "configs" / "gee_batch_assets.json"
REGION_MANIFEST_PATH = ROOT / "data" / "prepared" / "region_batches" / "region_manifest.csv"
NDWI_THRESHOLD = 0.0
MNDWI_THRESHOLD = 0.0
WATER_RULE = "and"
LAKE_TYPE_FILTER = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Submit Earth Engine export tasks for one region-year using the Phase 1 batch run list."
    )
    p.add_argument("--region", required=True, help="glambie_region_key, e.g. greenland_periphery or central_asia")
    p.add_argument("--year", required=True, type=int, help="Target year, e.g. 2000")
    p.add_argument(
        "--status-filter",
        default="pending",
        choices=["pending", "tested_passed", "all"],
        help="Which tasks from the run list to submit",
    )
    p.add_argument("--max-tasks", type=int, default=5, help="Maximum number of chunk tasks to submit in this run")
    p.add_argument("--start-chunk-id", type=int, default=None, help="Optional minimum chunk_id to submit")
    p.add_argument("--end-chunk-id", type=int, default=None, help="Optional maximum chunk_id to submit")
    p.add_argument(
        "--runlist-csv",
        default=str(RUNLIST_PATH),
        help=f"Path to run list CSV. Default: {RUNLIST_PATH}",
    )
    p.add_argument(
        "--ee-project-override",
        default="",
        help="Optional Earth Engine Cloud project override used for ee.Initialize().",
    )
    p.add_argument("--ndwi-threshold", type=float, default=0.0, help="NDWI threshold for water mask. Default: 0.0")
    p.add_argument("--mndwi-threshold", type=float, default=0.0, help="MNDWI threshold for water mask. Default: 0.0")
    p.add_argument(
        "--water-rule",
        default="and",
        choices=["and", "or"],
        help="Combine NDWI and MNDWI with AND (default) or OR.",
    )
    p.add_argument(
        "--name-suffix",
        default="",
        help="Optional suffix appended to EE task description and output filename, e.g. _m123.",
    )
    p.add_argument(
        "--source-years",
        default="",
        help="Optional comma-separated source years used to build the image stack while keeping --year as the target label, e.g. 2023,2024.",
    )
    p.add_argument("--lake-type-filter", default="", help="Optional lake_type filter, e.g. SGL")
    p.add_argument("--dry-run", action="store_true", help="Preview the tasks without submitting")
    return p.parse_args()


def parse_source_years(raw: str, target_year: int) -> list[int]:
    if not raw.strip():
        return [target_year]
    years = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not years:
        return [target_year]
    return years


def prop_or(feature: ee.Feature, primary: str, fallback: str):
    return ee.Algorithms.If(feature.propertyNames().contains(primary), feature.get(primary), feature.get(fallback))


def safe_number(dictionary, key: str):
    dictionary = ee.Dictionary(dictionary)
    return ee.Number(ee.Algorithms.If(dictionary.contains(key), dictionary.get(key), 0))


def safe_first_number(dictionary, keys):
    dictionary = ee.Dictionary(dictionary)
    keys = ee.List(keys)
    found = (
        keys.map(lambda k: ee.Algorithms.If(dictionary.contains(ee.String(k)), dictionary.get(ee.String(k)), None))
        .removeAll([None])
    )
    return ee.Number(ee.Algorithms.If(ee.List(found).size().gt(0), ee.List(found).get(0), 0))


def median_from_list(values):
    values = ee.List(values).sort()
    n = values.size()
    return ee.Number(
        ee.Algorithms.If(
            n.eq(0),
            0,
            ee.Algorithms.If(
                n.mod(2).eq(1),
                ee.Number(values.get(n.divide(2).floor())),
                ee.Number(values.get(n.divide(2).subtract(1))).add(ee.Number(values.get(n.divide(2)))).divide(2),
            ),
        )
    )


def infer_project_id(asset_id: str) -> str | None:
    match = re.match(r"^projects/([^/]+)/assets/", asset_id)
    if match:
        return match.group(1)
    return None


def apply_scale_factors(img: ee.Image) -> ee.Image:
    optical = img.select("SR_B.*").multiply(0.0000275).add(-0.2)
    return img.addBands(optical, overwrite=True)


def mask_landsat_l2(img: ee.Image) -> ee.Image:
    qa = img.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 0).eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    sat = img.select("QA_RADSAT").eq(0)
    return img.updateMask(mask).updateMask(sat)


def rename_bands_and_tag_sensor(img: ee.Image) -> ee.Image:
    spacecraft = ee.String(img.get("SPACECRAFT_ID"))
    bands = img.bandNames()
    is_l57 = bands.contains("SR_B1")
    renamed = ee.Image(
        ee.Algorithms.If(
            is_l57,
            img.select(["SR_B2", "SR_B4", "SR_B5", "SR_B7"], ["green", "nir", "swir1", "swir2"]),
            img.select(["SR_B3", "SR_B5", "SR_B6", "SR_B7"], ["green", "nir", "swir1", "swir2"]),
        )
    ).copyProperties(img, img.propertyNames())

    sensor_code = ee.Number(
        ee.Algorithms.If(
            spacecraft.compareTo("LANDSAT_5").eq(0),
            5,
            ee.Algorithms.If(
                spacecraft.compareTo("LANDSAT_7").eq(0),
                7,
                ee.Algorithms.If(
                    spacecraft.compareTo("LANDSAT_8").eq(0),
                    8,
                    ee.Algorithms.If(spacecraft.compareTo("LANDSAT_9").eq(0), 9, -1),
                ),
            ),
        )
    )
    return renamed.set("sensor_code", sensor_code)


def add_water_and_validity(img: ee.Image) -> ee.Image:
    ndwi = img.normalizedDifference(["green", "nir"]).rename("NDWI")
    mndwi = img.normalizedDifference(["green", "swir1"]).rename("MNDWI")
    valid = img.select("green").mask().rename("valid")
    if WATER_RULE == "or":
        water = ndwi.gte(NDWI_THRESHOLD).Or(mndwi.gte(MNDWI_THRESHOLD)).rename("water")
    else:
        water = ndwi.gte(NDWI_THRESHOLD).And(mndwi.gte(MNDWI_THRESHOLD)).rename("water")
    return img.addBands([ndwi, mndwi, valid, water])


def landsat_collection(source_years: list[int], months_list: list[int], geom: ee.Geometry) -> ee.ImageCollection:
    def prep(collection_id: str) -> ee.ImageCollection:
        merged = ee.ImageCollection([])
        for year in source_years:
            start = ee.Date.fromYMD(year, months_list[0], 1)
            end = ee.Date.fromYMD(year, months_list[-1], 1).advance(1, "month")
            merged = merged.merge(
                ee.ImageCollection(collection_id)
                .filterDate(start, end)
                .filterBounds(geom)
                .map(apply_scale_factors)
                .map(mask_landsat_l2)
                .map(rename_bands_and_tag_sensor)
                .map(add_water_and_validity)
                .map(lambda img: img.set("source_year", year))
            )
        return merged

    return prep("LANDSAT/LT05/C02/T1_L2").merge(prep("LANDSAT/LE07/C02/T1_L2")).merge(
        prep("LANDSAT/LC08/C02/T1_L2")
    ).merge(prep("LANDSAT/LC09/C02/T1_L2"))


def summarize_lake_year(
    lake: ee.Feature,
    target_year: int,
    months_list: list[int],
    source_years: list[int],
    chunk_start: int,
    chunk_size: int,
    scale_meters: int = 30,
    roi_buffer_meters: int = 120,
    minimum_images_per_year: int = 3,
    minimum_valid_area_fraction: float = 0.70,
) -> ee.Feature:
    lake_geom = lake.geometry()
    roi = lake_geom.buffer(roi_buffer_meters)
    baseline_area = ee.Number(prop_or(lake, "area_0_km2", "area0_km2"))
    col = landsat_collection(source_years, months_list, roi)
    image_count = col.size()
    pixel_area_km2 = ee.Image.pixelArea().divide(1e6).rename("area_km2")

    empty_stats = ee.Dictionary(
        {
            "annual_max_area_km2": 0,
            "annual_max_pixel_count": 0,
            "valid_area_any_km2": 0,
            "valid_pixel_any_count": 0,
            "water_area_median_km2": 0,
            "l5_count": 0,
            "l7_count": 0,
            "l8_count": 0,
            "l9_count": 0,
            "image_count": 0,
        }
    )

    def stats_when_images():
        valid_any = col.select("valid").max().rename("valid_any")
        water_max = col.select("water").max().rename("water_max")

        def area_feature(img):
            area_stats = pixel_area_km2.updateMask(img.select("water")).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=lake_geom,
                scale=scale_meters,
                maxPixels=1_000_000_000,
                tileScale=4,
            )
            return ee.Feature(None, {"area_km2": safe_first_number(area_stats, ["area_km2", "area", "sum"])})

        water_area_col = col.map(area_feature)
        water_area_median = ee.Number(
            ee.Algorithms.If(
                water_area_col.size().gt(0),
                median_from_list(water_area_col.aggregate_array("area_km2")),
                0,
            )
        )

        max_stats = pixel_area_km2.updateMask(water_max).reduceRegion(
            reducer=ee.Reducer.sum().combine(reducer2=ee.Reducer.count(), sharedInputs=True),
            geometry=lake_geom,
            scale=scale_meters,
            maxPixels=1_000_000_000,
            tileScale=4,
        )
        valid_stats = pixel_area_km2.updateMask(valid_any).reduceRegion(
            reducer=ee.Reducer.sum().combine(reducer2=ee.Reducer.count(), sharedInputs=True),
            geometry=lake_geom,
            scale=scale_meters,
            maxPixels=1_000_000_000,
            tileScale=4,
        )
        sensor_hist = ee.Dictionary(ee.List(col.aggregate_array("sensor_code")).reduce(ee.Reducer.frequencyHistogram()))

        return ee.Dictionary(
            {
                "annual_max_area_km2": safe_first_number(max_stats, ["area_km2_sum", "area_km2", "area", "sum"]),
                "annual_max_pixel_count": safe_first_number(max_stats, ["area_km2_count", "count"]),
                "valid_area_any_km2": safe_first_number(valid_stats, ["area_km2_sum", "area_km2", "area", "sum"]),
                "valid_pixel_any_count": safe_first_number(valid_stats, ["area_km2_count", "count"]),
                "water_area_median_km2": water_area_median,
                "l5_count": safe_number(sensor_hist, "5"),
                "l7_count": safe_number(sensor_hist, "7"),
                "l8_count": safe_number(sensor_hist, "8"),
                "l9_count": safe_number(sensor_hist, "9"),
                "image_count": image_count,
            }
        )

    stats = ee.Dictionary(ee.Algorithms.If(image_count.gt(0), stats_when_images(), empty_stats))

    annual_max_area = safe_number(stats, "annual_max_area_km2")
    valid_area_any = safe_number(stats, "valid_area_any_km2")
    image_count_num = safe_number(stats, "image_count")
    valid_area_fraction = ee.Number(ee.Algorithms.If(baseline_area.gt(0), valid_area_any.divide(baseline_area), 0))
    annual_area_fraction = ee.Number(ee.Algorithms.If(baseline_area.gt(0), annual_max_area.divide(baseline_area), 0))
    qc_enough_images = image_count_num.gte(minimum_images_per_year)
    qc_enough_coverage = valid_area_fraction.gte(minimum_valid_area_fraction)
    qc_usable = qc_enough_images.And(qc_enough_coverage)

    return ee.Feature(
        None,
        {
            "lake_id": prop_or(lake, "lake_id", "lake_id"),
            "lake_type": prop_or(lake, "lake_type", "lake_type"),
            "harmonized_class": prop_or(lake, "harmonized_class", "hclass"),
            "rgi_region_name": prop_or(lake, "rgi_region_name", "rgi_name"),
            "glambie_region_key": prop_or(lake, "glambie_region_key", "glmb_key"),
            "baseline_area_0_km2": baseline_area,
            "elevation_m": prop_or(lake, "elevation_m", "elev_m"),
            "latitude": prop_or(lake, "latitude", "lat"),
            "longitude": prop_or(lake, "longitude", "lon"),
            "year": target_year,
            "months": ",".join(str(m) for m in months_list),
            "source_years": ",".join(str(y) for y in source_years),
            "source_year_count": len(source_years),
            "chunk_start": chunk_start,
            "chunk_size": chunk_size,
            "image_count": image_count_num,
            "annual_max_area_km2": annual_max_area,
            "annual_max_pixel_count": safe_number(stats, "annual_max_pixel_count"),
            "valid_area_any_km2": valid_area_any,
            "valid_pixel_any_count": safe_number(stats, "valid_pixel_any_count"),
            "water_area_median_km2": safe_number(stats, "water_area_median_km2"),
            "baseline_valid_area_fraction": valid_area_fraction,
            "annual_area_to_baseline_ratio": annual_area_fraction,
            "l5_count": safe_number(stats, "l5_count"),
            "l7_count": safe_number(stats, "l7_count"),
            "l8_count": safe_number(stats, "l8_count"),
            "l9_count": safe_number(stats, "l9_count"),
            "qc_enough_images": qc_enough_images,
            "qc_enough_coverage": qc_enough_coverage,
            "qc_usable": qc_usable,
        },
    )


def build_export_collection(asset_id: str, task_row: pd.Series) -> ee.FeatureCollection:
    all_lakes = ee.FeatureCollection(asset_id).sort("lake_id")
    if LAKE_TYPE_FILTER:
        all_lakes = all_lakes.filter(ee.Filter.eq("lake_type", LAKE_TYPE_FILTER))
    chunk_size = int(task_row["chunk_size"])
    start_index = int(task_row["start_index"])
    chunk_lakes = ee.FeatureCollection(all_lakes.toList(chunk_size, start_index))
    months_list = [int(x) for x in str(task_row["months"]).split(",")]
    target_year = int(task_row["year"])
    source_years = parse_source_years(str(task_row.get("source_years", "")), target_year)

    def mapper(lake):
        return summarize_lake_year(
            lake,
            target_year=target_year,
            months_list=months_list,
            source_years=source_years,
            chunk_start=start_index,
            chunk_size=chunk_size,
        )

    return chunk_lakes.map(mapper)


def format_threshold(value: float) -> str:
    sign = "m" if value < 0 else "p"
    return f"{sign}{abs(value):.2f}".replace(".", "p")


def water_suffix() -> str:
    if WATER_RULE == "and" and NDWI_THRESHOLD == 0.0 and MNDWI_THRESHOLD == 0.0:
        return ""
    ndwi_tag = format_threshold(NDWI_THRESHOLD)
    mndwi_tag = format_threshold(MNDWI_THRESHOLD)
    return f"_water_{WATER_RULE}_ndwi{ndwi_tag}_mndwi{mndwi_tag}"


def load_region_code_map() -> dict[str, str]:
    manifest = pd.read_csv(REGION_MANIFEST_PATH)
    return {
        str(row["glambie_region_key"]): f"{int(row['glambie_region_id']):02d}"
        for _, row in manifest.iterrows()
    }


def task_description(task_row: pd.Series, name_suffix: str = "") -> str:
    region_key = task_row["glambie_region_key"]
    year = int(task_row["year"])
    start = int(task_row["start_index"])
    end = int(task_row["end_index_exclusive"]) - 1
    month_count = len(str(task_row["months"]).split(","))
    suffix = f"_{month_count}month" if month_count != 3 else ""
    region_code = REGION_CODE_MAP.get(region_key, "xx")
    lake_suffix = f"_{LAKE_TYPE_FILTER.lower()}" if LAKE_TYPE_FILTER else ""
    return f"annual_area_{region_code}_{region_key}_{year}_chunk_{start}_{end}{suffix}{lake_suffix}{name_suffix}{water_suffix()}"


def filter_tasks(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df[(df["glambie_region_key"] == args.region) & (df["year"] == args.year)].copy()
    if args.status_filter != "all":
        out = out[out["status"] == args.status_filter].copy()
    if args.start_chunk_id is not None:
        out = out[out["chunk_id"] >= args.start_chunk_id].copy()
    if args.end_chunk_id is not None:
        out = out[out["chunk_id"] <= args.end_chunk_id].copy()
    out = out.sort_values("chunk_id").head(args.max_tasks).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    global NDWI_THRESHOLD, MNDWI_THRESHOLD, WATER_RULE, LAKE_TYPE_FILTER
    NDWI_THRESHOLD = float(args.ndwi_threshold)
    MNDWI_THRESHOLD = float(args.mndwi_threshold)
    WATER_RULE = args.water_rule
    LAKE_TYPE_FILTER = str(args.lake_type_filter or "").strip().upper()
    runlist = pd.read_csv(args.runlist_csv)
    asset_cfg = json.loads(ASSET_CFG_PATH.read_text(encoding="utf-8"))
    asset_map = asset_cfg["region_assets"]
    drive_folder = asset_cfg.get("drive_folder", "GlacierAnnualArea")
    ee_project = args.ee_project_override or asset_cfg.get("ee_project") or infer_project_id(asset_map.get(args.region, ""))

    if args.region not in asset_map or asset_map[args.region].startswith("users/your_username/"):
        raise SystemExit(
            f"Asset path for region '{args.region}' is not configured. Please edit {ASSET_CFG_PATH} first."
        )

    tasks_df = filter_tasks(runlist, args)
    if tasks_df.empty:
        raise SystemExit("No tasks matched the requested filters.")
    tasks_df["source_years"] = args.source_years

    if not args.dry_run:
        if ee is None:
            raise SystemExit("Earth Engine Python API (ee) is not installed in the current environment.")
        try:
            if ee_project:
                ee.Initialize(project=ee_project)
            else:
                ee.Initialize()
        except Exception as e:
            raise SystemExit(
                "Earth Engine initialization failed. Check that the authenticated Google account has Earth Engine "
                f"access to project '{ee_project or 'DEFAULT'}', then retry.\nOriginal error: {e}"
            )

    print(f"Submitting tasks for region={args.region}, year={args.year}, count={len(tasks_df)}")
    print(f"Using asset: {asset_map[args.region]}")
    if args.source_years.strip():
        print(f"Using source years: {args.source_years}")
    if ee_project:
        print(f"Using EE project: {ee_project}")
    print(f"Drive folder: {drive_folder}")

    submitted = []
    safe_name_suffix = str(args.name_suffix or "")
    for row in tasks_df.itertuples(index=False):
        task_row = pd.Series(row._asdict())
        desc = task_description(task_row, name_suffix=safe_name_suffix)
        record = {
            "task_id": task_row["task_id"],
            "description": desc,
            "chunk_id": int(task_row["chunk_id"]),
            "start_index": int(task_row["start_index"]),
            "end_index_exclusive": int(task_row["end_index_exclusive"]),
            "months": task_row["months"],
            "source_years": task_row["source_years"],
            "chunk_size": int(task_row["chunk_size"]),
            "ndwi_threshold": NDWI_THRESHOLD,
            "mndwi_threshold": MNDWI_THRESHOLD,
            "water_rule": WATER_RULE,
            "status": "prepared" if args.dry_run else "submitted",
        }
        if not args.dry_run:
            collection = build_export_collection(asset_map[args.region], task_row)
            task = ee.batch.Export.table.toDrive(
                collection=collection,
                description=desc,
                folder=drive_folder,
                fileNamePrefix=desc,
                fileFormat="CSV",
            )
            task.start()
            record["ee_task_id"] = task.id
        submitted.append(record)
        print(json.dumps(record, ensure_ascii=False))

    out_dir = ROOT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "dryrun" if args.dry_run else "submitted"
    chunk_min = int(tasks_df["chunk_id"].min())
    chunk_max = int(tasks_df["chunk_id"].max())
    log_name_suffix = ""
    if safe_name_suffix:
        cleaned = safe_name_suffix.strip("_")
        if cleaned:
            log_name_suffix = f"_{cleaned}"
    out_path = out_dir / (
        f"gee_batch_submit_{args.region}_{args.year}_chunk_{chunk_min:03d}_{chunk_max:03d}{log_name_suffix}_{suffix}.json"
    )
    out_path.write_text(json.dumps(submitted, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved submission log to {out_path}")


if __name__ == "__main__":
    REGION_CODE_MAP = load_region_code_map()
    main()
