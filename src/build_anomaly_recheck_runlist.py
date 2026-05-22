from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_BASE_RUNLIST = ROOT / "data" / "prepared" / "execution_plan" / "phase1_batch_runlist.csv"
DEFAULT_PRIORITY_CSV = ROOT / "data" / "processed" / "annual_area_skeleton" / "central_asia_2020_2024_anomaly_suspicious_chunk_priority.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "prepared" / "execution_plan"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a focused anomaly recheck runlist from suspicious chunk priority.")
    p.add_argument("--region", default="central_asia")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--base-runlist-csv", default=str(DEFAULT_BASE_RUNLIST))
    p.add_argument("--priority-csv", default=str(DEFAULT_PRIORITY_CSV))
    p.add_argument("--top-chunks", type=int, default=5, help="Number of top-priority chunks to include")
    p.add_argument("--months", default="6,7,8,9", help="Months override used for recheck runs")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--tag", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = pd.read_csv(args.base_runlist_csv)
    priority = pd.read_csv(args.priority_csv)

    wanted_chunks = (
        priority.sort_values("suspicious_lakes", ascending=False)["chunk_start"].head(args.top_chunks).astype(int).tolist()
    )
    chunk_set = set(wanted_chunks)

    sel = base[
        (base["glambie_region_key"] == args.region)
        & (base["year"] == args.year)
        & (base["start_index"].astype(int).isin(chunk_set))
    ].copy()

    if sel.empty:
        raise SystemExit("No matching tasks found in base runlist for requested region/year/chunks.")

    sel["months"] = args.months
    sel["status"] = "pending"
    sel["phase1_wave"] = "wave_recheck_anomaly"
    sel["notes"] = (
        "Anomaly recheck run (target-year low-ratio diagnostics). "
        f"Months overridden to {args.months}. Prioritized by suspicious_lakes."
    )
    sel["task_id"] = sel.apply(
        lambda r: f"{r['glambie_region_key']}_{int(r['year'])}_chunk_{int(r['chunk_id']):03d}_recheck", axis=1
    )

    sel = sel.sort_values("chunk_id").reset_index(drop=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.region}_{args.year}_anomaly_recheck_top{args.top_chunks}"
    out_csv = out_dir / f"{tag}_runlist.csv"
    out_json = out_dir / f"{tag}_runlist_summary.json"
    sel.to_csv(out_csv, index=False)

    payload = {
        "region": args.region,
        "year": args.year,
        "top_chunks": args.top_chunks,
        "months_override": args.months,
        "selected_chunk_starts": wanted_chunks,
        "task_count": int(len(sel)),
        "runlist_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
