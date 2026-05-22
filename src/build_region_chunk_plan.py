from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
MANIFEST_PATH = ROOT / "data" / "prepared" / "region_batches" / "region_manifest.csv"
OUT_DIR = ROOT / "data" / "prepared" / "chunk_plans"


CHUNK_SIZE_BY_REGION = {
    "greenland_periphery": 500,
    "central_asia": 500,
}


def build_chunk_rows(region_key: str, lake_count: int, chunk_size: int) -> list[dict]:
    rows = []
    chunk_id = 0
    start_idx = 0
    while start_idx < lake_count:
        end_idx = min(start_idx + chunk_size, lake_count)
        rows.append(
            {
                "glambie_region_key": region_key,
                "chunk_id": chunk_id,
                "chunk_size_target": chunk_size,
                "start_index": start_idx,
                "end_index_exclusive": end_idx,
                "expected_lake_count": end_idx - start_idx,
            }
        )
        chunk_id += 1
        start_idx = end_idx
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)

    all_rows = []
    summary_rows = []
    for row in manifest.itertuples(index=False):
        region_key = row.glambie_region_key
        if region_key not in CHUNK_SIZE_BY_REGION:
            continue
        chunk_size = CHUNK_SIZE_BY_REGION[region_key]
        rows = build_chunk_rows(region_key, int(row.lake_count), chunk_size)
        all_rows.extend(rows)
        summary_rows.append(
            {
                "glambie_region_key": region_key,
                "lake_count": int(row.lake_count),
                "chunk_size": chunk_size,
                "chunk_count": len(rows),
            }
        )

    chunk_df = pd.DataFrame(all_rows).sort_values(["glambie_region_key", "chunk_id"])
    summary_df = pd.DataFrame(summary_rows).sort_values("glambie_region_key")

    chunk_df.to_csv(OUT_DIR / "test_region_chunk_plan.csv", index=False)
    summary_df.to_csv(OUT_DIR / "test_region_chunk_summary.csv", index=False)

    payload = {
        "regions": summary_df.to_dict(orient="records"),
        "chunk_plan_csv": str(OUT_DIR / "test_region_chunk_plan.csv"),
    }
    (OUT_DIR / "test_region_chunk_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
