from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_PANEL = ROOT / "data" / "processed" / "analysis" / "glambie_coupling_v1" / "glambie_coupling_v1_analysis_ready_panel.csv"
DEFAULT_REGIONWISE = ROOT / "data" / "processed" / "analysis" / "glambie_coupling_v1" / "glambie_coupling_v1_regionwise_correlations.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "analysis" / "glambie_anomaly_robustness_v1"
DEFAULT_TAG = "glambie_anomaly_robustness_v1"


PRIMARY_TESTS = [
    ("total_annual_area_km2_anomaly_z", "glambie_combined_mwe_anomaly", "area_anomaly_primary"),
    ("mean_ratio_anomaly_z", "glambie_combined_mwe_anomaly", "ratio_anomaly_primary"),
    ("zero_ratio_share_anomaly_z", "glambie_combined_mwe_anomaly", "zero_ratio_supportive"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run robustness and stratified analyses for GlaMBIE anomaly main results.")
    p.add_argument("--panel-csv", default=str(DEFAULT_PANEL))
    p.add_argument("--regionwise-csv", default=str(DEFAULT_REGIONWISE))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--tag", default=DEFAULT_TAG)
    return p.parse_args()


def corr_stats(x: pd.Series, y: pd.Series, method: str) -> dict:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    n = int(len(x))
    if n < 2:
        return {"n": n, "estimate": np.nan, "p_value": np.nan}
    try:
        from scipy.stats import pearsonr, spearmanr  # type: ignore

        if method == "pearson":
            est, p = pearsonr(x, y)
        elif method == "spearman":
            est, p = spearmanr(x, y)
        else:
            raise ValueError(method)
        return {"n": n, "estimate": float(est), "p_value": float(p)}
    except Exception:
        est = x.corr(y, method=method)
        return {"n": n, "estimate": float(est) if pd.notna(est) else np.nan, "p_value": np.nan}


def ols_stats(x: pd.Series, y: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask].astype(float)
    y = y[mask].astype(float)
    n = int(len(x))
    if n < 2:
        return {"n": n, "slope": np.nan, "intercept": np.nan, "r2": np.nan, "p_value": np.nan}
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    p_value = np.nan
    try:
        from scipy.stats import linregress as _lr  # type: ignore

        p_value = float(_lr(x, y).pvalue)
    except Exception:
        p_value = np.nan
    return {"n": n, "slope": float(slope), "intercept": float(intercept), "r2": float(r2), "p_value": p_value}


def evaluate_subset(df: pd.DataFrame, subset_name: str) -> list[dict]:
    rows = []
    for response, predictor, family in PRIMARY_TESTS:
        pearson = corr_stats(df[predictor], df[response], method="pearson")
        spearman = corr_stats(df[predictor], df[response], method="spearman")
        ols = ols_stats(df[predictor], df[response])
        rows.extend(
            [
                {
                    "subset_name": subset_name,
                    "response": response,
                    "predictor": predictor,
                    "predictor_family": family,
                    "test_name": "pearson",
                    "n_rows": pearson["n"],
                    "n_regions": int(df["region_key"].nunique()),
                    "estimate": pearson["estimate"],
                    "p_value": pearson["p_value"],
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                },
                {
                    "subset_name": subset_name,
                    "response": response,
                    "predictor": predictor,
                    "predictor_family": family,
                    "test_name": "spearman",
                    "n_rows": spearman["n"],
                    "n_regions": int(df["region_key"].nunique()),
                    "estimate": spearman["estimate"],
                    "p_value": spearman["p_value"],
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                },
                {
                    "subset_name": subset_name,
                    "response": response,
                    "predictor": predictor,
                    "predictor_family": family,
                    "test_name": "ols",
                    "n_rows": ols["n"],
                    "n_regions": int(df["region_key"].nunique()),
                    "estimate": ols["slope"],
                    "p_value": ols["p_value"],
                    "slope": ols["slope"],
                    "intercept": ols["intercept"],
                    "r2": ols["r2"],
                },
            ]
        )
    return rows


def categorize_regions(regionwise: pd.DataFrame) -> pd.DataFrame:
    target = regionwise[
        (regionwise["response"] == "total_annual_area_km2_anomaly_z")
        & (regionwise["predictor"] == "glambie_combined_mwe_anomaly")
        & (regionwise["test_name"] == "pearson")
    ].copy()
    target["effect_direction"] = np.where(target["estimate"] < 0, "negative", "positive_or_zero")
    target["significant_negative"] = (target["estimate"] < 0) & (target["p_value"] < 0.05)
    target["supportive_negative"] = (target["estimate"] < 0) & (target["p_value"] >= 0.05)
    target["weak_or_opposite"] = target["estimate"] >= 0
    target["stratum"] = np.select(
        [target["significant_negative"], target["supportive_negative"], target["weak_or_opposite"]],
        ["core_negative", "supporting_negative", "weak_or_opposite"],
        default="weak_or_opposite",
    )
    return target[["region_key", "region_name", "estimate", "p_value", "stratum"]].sort_values(["stratum", "estimate"])


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel_csv)
    regionwise_path = Path(args.regionwise_csv)
    out_dir = Path(args.out_dir)
    tag = args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    regionwise = pd.read_csv(regionwise_path)

    shared = panel[panel["shared_sample_flag"]].copy()

    robustness_specs: list[tuple[str, pd.DataFrame]] = []
    robustness_specs.append(("baseline_shared_2000_2023", shared.copy()))
    robustness_specs.append(("usable_share_ge_0.50", shared[shared["usable_share"] >= 0.50].copy()))
    robustness_specs.append(("usable_share_ge_0.70", shared[shared["usable_share"] >= 0.70].copy()))
    robustness_specs.append(("drop_high_suspicious_ge_100", shared[shared["suspicious_lake_count"] < 100].copy()))
    robustness_specs.append(("drop_low_overall_quality_lt_0.60", shared[shared["usable_share_overall"] >= 0.60].copy()))
    robustness_specs.append(("drop_both_low_quality_and_high_suspicious", shared[(shared["usable_share_overall"] >= 0.60) & (shared["suspicious_lake_count"] < 100)].copy()))

    robustness_rows = []
    subset_inventory_rows = []
    for subset_name, df in robustness_specs:
        robustness_rows.extend(evaluate_subset(df, subset_name))
        subset_inventory_rows.append(
            {
                "subset_name": subset_name,
                "n_rows": int(len(df)),
                "n_regions": int(df["region_key"].nunique()),
                "year_min": int(df["year"].min()) if len(df) else np.nan,
                "year_max": int(df["year"].max()) if len(df) else np.nan,
                "usable_share_mean": float(df["usable_share"].mean()) if len(df) else np.nan,
                "usable_share_overall_region_mean": float(df.groupby("region_key")["usable_share_overall"].first().mean()) if len(df) else np.nan,
                "regions": ",".join(sorted(df["region_key"].unique().tolist())) if len(df) else "",
            }
        )

    robustness_df = pd.DataFrame(robustness_rows)
    subset_inventory_df = pd.DataFrame(subset_inventory_rows)

    stratified_regions = categorize_regions(regionwise)
    stratum_map = stratified_regions[["region_key", "stratum"]].drop_duplicates()
    stratified_panel = shared.merge(stratum_map, on="region_key", how="left")

    stratum_rows = []
    stratum_inventory_rows = []
    for stratum, df in stratified_panel.groupby("stratum"):
        stratum_rows.extend(evaluate_subset(df, f"stratum::{stratum}"))
        stratum_inventory_rows.append(
            {
                "stratum": stratum,
                "n_rows": int(len(df)),
                "n_regions": int(df["region_key"].nunique()),
                "regions": ",".join(sorted(df["region_key"].unique().tolist())),
                "usable_share_mean": float(df["usable_share"].mean()),
            }
        )
    stratum_tests_df = pd.DataFrame(stratum_rows)
    stratum_inventory_df = pd.DataFrame(stratum_inventory_rows)

    robustness_csv = out_dir / f"{tag}_robustness_tests.csv"
    subset_inventory_csv = out_dir / f"{tag}_subset_inventory.csv"
    stratified_regions_csv = out_dir / f"{tag}_stratified_regions.csv"
    stratum_tests_csv = out_dir / f"{tag}_stratum_tests.csv"
    stratum_inventory_csv = out_dir / f"{tag}_stratum_inventory.csv"
    summary_json = out_dir / f"{tag}_analysis_summary.json"

    robustness_df.to_csv(robustness_csv, index=False)
    subset_inventory_df.to_csv(subset_inventory_csv, index=False)
    stratified_regions.to_csv(stratified_regions_csv, index=False)
    stratum_tests_df.to_csv(stratum_tests_csv, index=False)
    stratum_inventory_df.to_csv(stratum_inventory_csv, index=False)

    summary = {
        "tag": tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "analysis_ready_panel_csv": str(panel_path),
            "regionwise_correlations_csv": str(regionwise_path),
        },
        "outputs": {
            "robustness_tests_csv": str(robustness_csv),
            "subset_inventory_csv": str(subset_inventory_csv),
            "stratified_regions_csv": str(stratified_regions_csv),
            "stratum_tests_csv": str(stratum_tests_csv),
            "stratum_inventory_csv": str(stratum_inventory_csv),
        },
        "sample": {
            "shared_rows": int(len(shared)),
            "shared_regions": int(shared['region_key'].nunique()),
            "shared_year_min": int(shared['year'].min()),
            "shared_year_max": int(shared['year'].max()),
        },
        "stratum_counts": stratum_inventory_df.to_dict(orient="records"),
        "headline_preview": {
            "baseline_tests": robustness_df[robustness_df['subset_name'] == 'baseline_shared_2000_2023'].head(12).to_dict(orient='records'),
            "core_regions": stratified_regions[stratified_regions['stratum'] == 'core_negative']['region_key'].tolist(),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
