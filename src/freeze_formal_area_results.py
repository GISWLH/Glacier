from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_REGISTRY = ROOT / "data" / "processed" / "region_final_results_summary.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed" / "formal_area_results_freeze"
DEFAULT_FREEZE_VERSION = "freeze_20260509"


SKELETON_SUFFIXES = [
    "_skeleton_lake_year.csv",
    "_skeleton_year_summary.csv",
    "_skeleton_interyear_change.csv",
    "_skeleton_lake_summary.csv",
    "_skeleton_summary.json",
]

ANALYSIS_SUFFIXES = [
    "_trend_summary.csv",
    "_year_anomalies.csv",
    "_class_year_stats.csv",
    "_class_trends.csv",
    "_analysis_summary.json",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze formal area results into a versioned, registry-driven release.")
    p.add_argument("--registry-csv", default=str(DEFAULT_REGISTRY))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--freeze-version", default=DEFAULT_FREEZE_VERSION)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def must_exist(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def copy_with_hash(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": str(src),
        "dest": str(dst),
        "size_bytes": dst.stat().st_size,
        "sha256": sha256_file(dst),
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_region_paths(row: pd.Series) -> dict[str, Path]:
    final_tag = str(row["final_tag"])
    skeleton_summary_json = Path(str(row["skeleton_summary_json"]))
    analysis_dir = Path(str(row["analysis_dir"]))
    diagnosis_summary_json = Path(str(row["diagnosis_summary_json"]))
    skeleton_dir = skeleton_summary_json.parent

    paths: dict[str, Path] = {
        "skeleton_summary_json": must_exist(skeleton_summary_json, "skeleton_summary_json"),
        "analysis_dir": must_exist(analysis_dir, "analysis_dir"),
        "diagnosis_summary_json": must_exist(diagnosis_summary_json, "diagnosis_summary_json"),
    }

    for suffix in SKELETON_SUFFIXES:
        key = suffix.removeprefix("_")
        paths[key] = must_exist(skeleton_dir / f"{final_tag}{suffix}", f"skeleton asset {suffix}")

    for suffix in ANALYSIS_SUFFIXES:
        key = f"analysis{suffix}"
        paths[key] = must_exist(analysis_dir / f"{final_tag}{suffix}", f"analysis asset {suffix}")

    return paths


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    registry_path = Path(args.registry_csv)
    output_root = Path(args.output_root)
    freeze_version = args.freeze_version
    freeze_dir = output_root / freeze_version
    regions_root = freeze_dir / "regions"

    registry = pd.read_csv(registry_path)
    required_cols = {
        "region_code",
        "region_key",
        "final_tag",
        "skeleton_summary_json",
        "analysis_dir",
        "diagnosis_summary_json",
    }
    missing = sorted(required_cols - set(registry.columns))
    if missing:
        raise ValueError(f"Registry missing required columns: {missing}")

    freeze_dir.mkdir(parents=True, exist_ok=True)
    registry_snapshot = freeze_dir / "freeze_registry_snapshot.csv"
    registry.to_csv(registry_snapshot, index=False)

    copied_files: list[dict] = []
    lake_year_frames: list[pd.DataFrame] = []
    region_year_frames: list[pd.DataFrame] = []
    quality_rows: list[dict] = []
    verification_rows: list[dict] = []

    for row in registry.itertuples(index=False):
        region_code = int(row.region_code)
        region_key = str(row.region_key)
        final_tag = str(row.final_tag)
        row_series = pd.Series(row._asdict())
        paths = build_region_paths(row_series)

        region_dir = regions_root / f"{region_code:02d}_{region_key}" / final_tag

        for key, src in paths.items():
            if key == "analysis_dir":
                continue
            dst = region_dir / src.name
            copied_files.append({"region_key": region_key, "final_tag": final_tag, "asset_key": key, **copy_with_hash(src, dst)})

        lake_year_src = paths["skeleton_lake_year.csv"]
        year_summary_src = paths["skeleton_year_summary.csv"]
        anomalies_src = paths["analysis_year_anomalies.csv"]

        lake_year_df = pd.read_csv(lake_year_src, low_memory=False)
        lake_year_df.insert(0, "freeze_version", freeze_version)
        lake_year_df.insert(1, "region_code", region_code)
        lake_year_df.insert(2, "region_key", region_key)
        lake_year_df.insert(3, "final_tag", final_tag)
        lake_year_df["source_relpath"] = relative_to_root(lake_year_src)
        lake_year_frames.append(lake_year_df)

        year_summary_df = pd.read_csv(year_summary_src)
        anomalies_df = pd.read_csv(anomalies_src)
        anomaly_merge_cols = [c for c in anomalies_df.columns if c != "year" and c not in year_summary_df.columns]
        merged_year = year_summary_df.merge(anomalies_df[["year", *anomaly_merge_cols]], on="year", how="left")
        merged_year.insert(0, "freeze_version", freeze_version)
        merged_year.insert(1, "region_code", region_code)
        merged_year.insert(2, "region_key", region_key)
        merged_year.insert(3, "final_tag", final_tag)
        merged_year["source_relpath"] = relative_to_root(year_summary_src)
        region_year_frames.append(merged_year)

        skeleton_summary = read_json(paths["skeleton_summary_json"])
        diagnosis_summary = read_json(paths["diagnosis_summary_json"])

        quality_rows.append(
            {
                "freeze_version": freeze_version,
                "region_code": region_code,
                "region_key": region_key,
                "final_tag": final_tag,
                "skeleton_summary_json": relative_to_root(paths["skeleton_summary_json"]),
                "diagnosis_summary_json": relative_to_root(paths["diagnosis_summary_json"]),
                "input_file_count": skeleton_summary.get("input_file_count"),
                "lake_year_rows": skeleton_summary.get("lake_year_rows"),
                "unique_lakes": skeleton_summary.get("unique_lakes"),
                "usable_share_overall": skeleton_summary.get("usable_share_overall"),
                "ref_year": diagnosis_summary.get("ref_year"),
                "target_year": diagnosis_summary.get("target_year"),
                "suspicious_lake_count": diagnosis_summary.get("suspicious_lake_count"),
                "total_area_change_km2": diagnosis_summary.get("total_area_change_km2"),
                "mean_ratio_change": diagnosis_summary.get("mean_ratio_change"),
            }
        )

        year_count = int(year_summary_df["year"].nunique())
        verification_rows.append(
            {
                "region_code": region_code,
                "region_key": region_key,
                "final_tag": final_tag,
                "lake_year_rows_registry": int(getattr(row, "lake_year_rows")),
                "lake_year_rows_actual": int(len(lake_year_df)),
                "unique_lakes_registry": int(getattr(row, "unique_lakes")),
                "unique_lakes_actual": int(lake_year_df["lake_id"].nunique()),
                "usable_share_registry": float(getattr(row, "usable_share_overall")),
                "usable_share_actual": float(skeleton_summary.get("usable_share_overall")),
                "year_count_actual": year_count,
            }
        )

    lake_year_master = pd.concat(lake_year_frames, ignore_index=True)
    region_year_master = pd.concat(region_year_frames, ignore_index=True)
    quality_index = pd.DataFrame(quality_rows).sort_values(["region_code", "region_key"])
    verification_df = pd.DataFrame(verification_rows).sort_values(["region_code", "region_key"])
    copied_df = pd.DataFrame(copied_files)

    lake_year_master_path = freeze_dir / "analysis_lake_year_master.csv"
    region_year_master_path = freeze_dir / "analysis_region_year_master.csv"
    quality_index_path = freeze_dir / "quality_region_diagnosis_index.csv"
    verification_path = freeze_dir / "freeze_verification_checks.csv"
    copied_manifest_path = freeze_dir / "freeze_copied_files.csv"

    lake_year_master.to_csv(lake_year_master_path, index=False)
    region_year_master.to_csv(region_year_master_path, index=False)
    quality_index.to_csv(quality_index_path, index=False)
    verification_df.to_csv(verification_path, index=False)
    copied_df.to_csv(copied_manifest_path, index=False)

    manifest = {
        "freeze_version": freeze_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_csv": str(registry_path),
        "registry_snapshot": str(registry_snapshot),
        "region_count": int(len(registry)),
        "lake_year_master_csv": str(lake_year_master_path),
        "region_year_master_csv": str(region_year_master_path),
        "quality_region_diagnosis_index_csv": str(quality_index_path),
        "verification_checks_csv": str(verification_path),
        "copied_files_csv": str(copied_manifest_path),
        "lake_year_rows_total": int(len(lake_year_master)),
        "region_year_rows_total": int(len(region_year_master)),
        "unique_regions": sorted(registry["region_key"].astype(str).tolist()),
        "source_files": {
            "registry": str(registry_path),
            "script": str(Path(__file__)),
        },
        "column_notes": {
            "analysis_lake_year_master": [
                "freeze_version",
                "region_code",
                "region_key",
                "final_tag",
                "source_relpath",
            ],
            "analysis_region_year_master": [
                "freeze_version",
                "region_code",
                "region_key",
                "final_tag",
                "source_relpath",
            ],
        },
    }
    manifest_path = freeze_dir / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
