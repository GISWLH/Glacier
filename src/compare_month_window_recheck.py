from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "GlacierAnnualArea"
DEFAULT_OUT_DIR = ROOT / "data" / "interim" / "annual_area_qc"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare baseline 3-month exports vs recheck 4-month exports for same region/year/chunks."
    )
    p.add_argument("--region-code", required=True, help="Region code in file name, e.g. 13")
    p.add_argument("--region-key", required=True, help="Region key in file name, e.g. central_asia")
    p.add_argument("--year", type=int, required=True)
    p.add_argument(
        "--chunk-starts",
        required=True,
        help="Comma-separated chunk starts, e.g. 2500,3000,3500,6000,6500",
    )
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument(
        "--baseline-dir",
        default="",
        help="Optional directory containing baseline files. Defaults to --data-dir.",
    )
    p.add_argument(
        "--recheck-dir",
        default="",
        help="Optional directory containing recheck files. Defaults to --data-dir.",
    )
    p.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--tag", default="")
    return p.parse_args()


def normalize_boolish(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    return mapped.where(mapped.notna(), series).astype(bool)


def find_chunk_file(
    data_dir: Path,
    region_code: str,
    region_key: str,
    year: int,
    chunk_start: int,
    expected_end: int,
) -> tuple[Path | None, Path | None]:
    """Resolve baseline and 4-month files for one chunk start in a single directory.

    Works for both full chunks (e.g., 2500_2999) and the final partial chunk
    (e.g., 7000_7262).
    """
    exact_base = data_dir / f"annual_area_{region_code}_{region_key}_{year}_chunk_{chunk_start}_{expected_end}.csv"
    exact_recheck = (
        data_dir / f"annual_area_{region_code}_{region_key}_{year}_chunk_{chunk_start}_{expected_end}_4month.csv"
    )
    if exact_base.exists() and exact_recheck.exists():
        return exact_base, exact_recheck

    pattern = re.compile(
        rf"^annual_area_{re.escape(region_code)}_{re.escape(region_key)}_{year}_chunk_{chunk_start}_(\d+)(?:_(.+))?\.csv$"
    )
    base_candidates: list[tuple[int, Path]] = []
    recheck_candidates: list[tuple[int, Path]] = []
    for p in data_dir.glob(f"annual_area_{region_code}_{region_key}_{year}_chunk_{chunk_start}_*.csv"):
        m = pattern.match(p.name)
        if not m:
            continue
        chunk_end = int(m.group(1))
        suffix = m.group(2) or ""
        if suffix == "":
            base_candidates.append((chunk_end, p))
        elif suffix.startswith("4month"):
            recheck_candidates.append((chunk_end, p))

    def pick(candidates: list[tuple[int, Path]]) -> Path | None:
        if not candidates:
            return None
        # Prefer exact end if present, otherwise use the smallest end (final partial chunk)
        exact = [p for end, p in candidates if end == expected_end]
        if exact:
            return exact[0]
        return sorted(candidates, key=lambda t: t[0])[0][1]

    return pick(base_candidates), pick(recheck_candidates)


def resolve_chunk_files(
    baseline_dir: Path,
    recheck_dir: Path,
    region_code: str,
    region_key: str,
    year: int,
    chunk_start: int,
    expected_end: int,
) -> tuple[Path | None, Path | None]:
    """Resolve baseline and recheck files, supporting separate directories."""
    if baseline_dir == recheck_dir:
        return find_chunk_file(
            data_dir=baseline_dir,
            region_code=region_code,
            region_key=region_key,
            year=year,
            chunk_start=chunk_start,
            expected_end=expected_end,
        )

    baseline_path = baseline_dir / f"annual_area_{region_code}_{region_key}_{year}_chunk_{chunk_start}_{expected_end}.csv"
    if not baseline_path.exists():
        baseline_path, _ = find_chunk_file(
            data_dir=baseline_dir,
            region_code=region_code,
            region_key=region_key,
            year=year,
            chunk_start=chunk_start,
            expected_end=expected_end,
        )

    recheck_path = (
        recheck_dir / f"annual_area_{region_code}_{region_key}_{year}_chunk_{chunk_start}_{expected_end}_4month.csv"
    )
    if not recheck_path.exists():
        _, recheck_path = find_chunk_file(
            data_dir=recheck_dir,
            region_code=region_code,
            region_key=region_key,
            year=year,
            chunk_start=chunk_start,
            expected_end=expected_end,
        )

    return baseline_path, recheck_path


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["qc_usable"] = normalize_boolish(df["qc_usable"])
    for c in ["annual_max_area_km2", "annual_area_to_baseline_ratio", "image_count", "baseline_valid_area_fraction"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "usable_rows": int(df["qc_usable"].sum()),
        "usable_share": float(df["qc_usable"].mean()),
        "total_area_km2": float(df["annual_max_area_km2"].sum()),
        "median_ratio": float(df["annual_area_to_baseline_ratio"].median()),
        "zero_ratio_share": float((df["annual_area_to_baseline_ratio"].fillna(0) == 0).mean()),
        "median_images": float(df["image_count"].median()),
        "median_valid_fraction": float(df["baseline_valid_area_fraction"].median()),
    }


def main() -> None:
    args = parse_args()
    starts = [int(x.strip()) for x in args.chunk_starts.split(",") if x.strip()]
    data_dir = Path(args.data_dir)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else data_dir
    recheck_dir = Path(args.recheck_dir) if args.recheck_dir else data_dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for s in starts:
        e = s + args.chunk_size - 1
        base_path, recheck_path = resolve_chunk_files(
            baseline_dir=baseline_dir,
            recheck_dir=recheck_dir,
            region_code=args.region_code,
            region_key=args.region_key,
            year=args.year,
            chunk_start=s,
            expected_end=e,
        )
        base_name = base_path.name if base_path else f"annual_area_{args.region_code}_{args.region_key}_{args.year}_chunk_{s}_{e}.csv"
        recheck_name = (
            recheck_path.name
            if recheck_path
            else f"annual_area_{args.region_code}_{args.region_key}_{args.year}_chunk_{s}_{e}_4month.csv"
        )
        if (base_path is None or not base_path.exists()) or (recheck_path is None or not recheck_path.exists()):
            missing.append(
                {
                    "chunk_start": s,
                    "baseline_exists": bool(base_path and base_path.exists()),
                    "recheck_exists": bool(recheck_path and recheck_path.exists()),
                    "baseline_file": str(base_path) if base_path else "",
                    "recheck_file": str(recheck_path) if recheck_path else "",
                }
            )
            continue

        base_df = load_csv(base_path)
        chk_df = load_csv(recheck_path)
        base_s = summarize(base_df)
        chk_s = summarize(chk_df)
        rows.append(
            {
                "chunk_start": s,
                "chunk_end": e,
                "baseline_file": base_name,
                "recheck_file": recheck_name,
                "baseline_rows": base_s["rows"],
                "recheck_rows": chk_s["rows"],
                "baseline_usable_share": base_s["usable_share"],
                "recheck_usable_share": chk_s["usable_share"],
                "usable_share_change": chk_s["usable_share"] - base_s["usable_share"],
                "baseline_total_area_km2": base_s["total_area_km2"],
                "recheck_total_area_km2": chk_s["total_area_km2"],
                "total_area_change_km2": chk_s["total_area_km2"] - base_s["total_area_km2"],
                "baseline_median_ratio": base_s["median_ratio"],
                "recheck_median_ratio": chk_s["median_ratio"],
                "median_ratio_change": chk_s["median_ratio"] - base_s["median_ratio"],
                "baseline_zero_ratio_share": base_s["zero_ratio_share"],
                "recheck_zero_ratio_share": chk_s["zero_ratio_share"],
                "zero_ratio_share_change": chk_s["zero_ratio_share"] - base_s["zero_ratio_share"],
            }
        )

    comp = pd.DataFrame(rows).sort_values("chunk_start") if rows else pd.DataFrame()
    miss_df = pd.DataFrame(missing).sort_values("chunk_start") if missing else pd.DataFrame()

    tag = args.tag or f"{args.region_key}_{args.year}_month_window_recheck"
    out_comp = out_dir / f"{tag}_chunk_comparison.csv"
    out_miss = out_dir / f"{tag}_missing_files.csv"
    out_json = out_dir / f"{tag}_summary.json"

    comp.to_csv(out_comp, index=False)
    miss_df.to_csv(out_miss, index=False)

    payload = {
        "region_key": args.region_key,
        "year": args.year,
        "requested_chunks": starts,
        "baseline_dir": str(baseline_dir),
        "recheck_dir": str(recheck_dir),
        "compared_chunks": int(len(comp)),
        "missing_chunks": int(len(miss_df)),
        "comparison_csv": str(out_comp),
        "missing_csv": str(out_miss),
    }
    if not comp.empty:
        payload.update(
            {
                "mean_usable_share_change": float(comp["usable_share_change"].mean()),
                "sum_total_area_change_km2": float(comp["total_area_change_km2"].sum()),
                "mean_median_ratio_change": float(comp["median_ratio_change"].mean()),
                "mean_zero_ratio_share_change": float(comp["zero_ratio_share_change"].mean()),
            }
        )
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
