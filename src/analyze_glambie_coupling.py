from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_PANEL = ROOT / "data" / "processed" / "formal_region_year_panel" / "formal_region_year_panel_v1.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "analysis" / "glambie_coupling_v1"
DEFAULT_TAG = "glambie_coupling_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run first-round GlaMBIE coupling analyses on the formal region-year panel.")
    p.add_argument("--panel-csv", default=str(DEFAULT_PANEL))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--tag", default=DEFAULT_TAG)
    return p.parse_args()


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
    return {
        "n": n,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "p_value": p_value,
    }


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


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel_csv)
    out_dir = Path(args.out_dir)
    tag = args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(panel_path)
    require_columns(
        panel,
        [
            "region_key",
            "region_name",
            "year",
            "usable_total_area_change_km2",
            "total_annual_area_km2",
            "total_annual_area_km2_anomaly_z",
            "mean_ratio_anomaly_z",
            "zero_ratio_share_anomaly_z",
            "glambie_combined_mwe",
            "glambie_combined_mwe_anomaly",
            "glambie_combined_mwe_rolling3",
            "glambie_combined_mwe_rolling5",
            "glambie_combined_mwe_cumulative",
            "glambie_combined_gt_cumulative",
            "glambie_glacier_area_km2",
            "usable_share",
        ],
    )

    panel["glambie_available"] = panel["glambie_combined_mwe"].notna()
    panel["shared_sample_flag"] = panel["glambie_available"] & (panel["year"] <= 2023)

    analysis_cols = [
        "freeze_version",
        "panel_version",
        "region_code",
        "region_key",
        "region_name",
        "final_tag",
        "year",
        "usable_share",
        "glambie_available",
        "shared_sample_flag",
        "usable_total_area_change_km2",
        "total_annual_area_km2",
        "total_annual_area_km2_anomaly_z",
        "mean_ratio_anomaly_z",
        "zero_ratio_share_anomaly_z",
        "glambie_glacier_area_km2",
        "glambie_combined_mwe",
        "glambie_combined_mwe_anomaly",
        "glambie_combined_mwe_rolling3",
        "glambie_combined_mwe_rolling5",
        "glambie_combined_mwe_cumulative",
        "glambie_combined_gt_cumulative",
        "suspicious_lake_count",
        "usable_share_overall",
    ]
    analysis_ready = panel[analysis_cols].copy().sort_values(["region_key", "year"])

    shared = panel[panel["shared_sample_flag"]].copy().sort_values(["region_key", "year"])

    region_summary_rows = []
    for region_key, sub in shared.groupby("region_key"):
        sub = sub.sort_values("year")
        first = sub.iloc[0]
        last = sub.iloc[-1]
        region_summary_rows.append(
            {
                "region_key": region_key,
                "region_name": first["region_name"],
                "year_start": int(sub["year"].min()),
                "year_end": int(sub["year"].max()),
                "n_shared_years": int(len(sub)),
                "lake_area_2000_km2": float(first["total_annual_area_km2"]),
                "lake_area_2023_km2": float(last["total_annual_area_km2"]),
                "lake_area_change_2000_2023_km2": float(last["total_annual_area_km2"] - first["total_annual_area_km2"]),
                "glambie_combined_mwe_cumulative_2023": float(last["glambie_combined_mwe_cumulative"]),
                "glambie_combined_gt_cumulative_2023": float(last["glambie_combined_gt_cumulative"]),
                "glambie_glacier_area_km2_mean": float(sub["glambie_glacier_area_km2"].mean()),
                "usable_share_mean": float(sub["usable_share"].mean()),
            }
        )
    region_summary = pd.DataFrame(region_summary_rows).sort_values("region_key")

    longterm_tests = []
    longterm_pairs = [
        ("lake_area_change_2000_2023_km2", "glambie_combined_mwe_cumulative_2023", "mwe_cumulative_primary"),
        ("lake_area_change_2000_2023_km2", "glambie_combined_gt_cumulative_2023", "gt_cumulative_sensitivity"),
    ]
    for response, predictor, family in longterm_pairs:
        pearson = corr_stats(region_summary[predictor], region_summary[response], method="pearson")
        spearman = corr_stats(region_summary[predictor], region_summary[response], method="spearman")
        ols = ols_stats(region_summary[predictor], region_summary[response])
        longterm_tests.extend(
            [
                {
                    "response": response,
                    "predictor": predictor,
                    "predictor_family": family,
                    "test_name": "pearson",
                    "n_regions": pearson["n"],
                    "estimate": pearson["estimate"],
                    "p_value": pearson["p_value"],
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                },
                {
                    "response": response,
                    "predictor": predictor,
                    "predictor_family": family,
                    "test_name": "spearman",
                    "n_regions": spearman["n"],
                    "estimate": spearman["estimate"],
                    "p_value": spearman["p_value"],
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                },
                {
                    "response": response,
                    "predictor": predictor,
                    "predictor_family": family,
                    "test_name": "ols",
                    "n_regions": ols["n"],
                    "estimate": ols["slope"],
                    "p_value": ols["p_value"],
                    "slope": ols["slope"],
                    "intercept": ols["intercept"],
                    "r2": ols["r2"],
                },
            ]
        )
    longterm_tests_df = pd.DataFrame(longterm_tests)

    panel_tests = []
    panel_pairs = [
        ("total_annual_area_km2_anomaly_z", "glambie_combined_mwe_anomaly", "mwe_anomaly_primary"),
        ("mean_ratio_anomaly_z", "glambie_combined_mwe_anomaly", "mwe_anomaly_primary"),
        ("usable_total_area_change_km2", "glambie_combined_mwe_rolling3", "mwe_rolling3_supportive"),
        ("usable_total_area_change_km2", "glambie_combined_mwe_rolling5", "mwe_rolling5_supportive"),
        ("mean_ratio_anomaly_z", "glambie_combined_mwe_rolling3", "mwe_rolling3_supportive"),
        ("usable_total_area_change_km2", "glambie_combined_mwe_cumulative", "mwe_cumulative_supportive"),
        ("total_annual_area_km2", "glambie_combined_mwe_cumulative", "mwe_cumulative_supportive"),
        ("zero_ratio_share_anomaly_z", "glambie_combined_mwe_anomaly", "mwe_anomaly_supportive"),
    ]
    for response, predictor, family in panel_pairs:
        ols = ols_stats(shared[predictor], shared[response])
        pearson = corr_stats(shared[predictor], shared[response], method="pearson")
        panel_tests.extend(
            [
                {
                    "response": response,
                    "predictor": predictor,
                    "predictor_family": family,
                    "test_name": "pearson",
                    "n_rows": pearson["n"],
                    "n_regions": int(shared["region_key"].nunique()),
                    "year_min": int(shared["year"].min()),
                    "year_max": int(shared["year"].max()),
                    "estimate": pearson["estimate"],
                    "p_value": pearson["p_value"],
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                },
                {
                    "response": response,
                    "predictor": predictor,
                    "predictor_family": family,
                    "test_name": "ols",
                    "n_rows": ols["n"],
                    "n_regions": int(shared["region_key"].nunique()),
                    "year_min": int(shared["year"].min()),
                    "year_max": int(shared["year"].max()),
                    "estimate": ols["slope"],
                    "p_value": ols["p_value"],
                    "slope": ols["slope"],
                    "intercept": ols["intercept"],
                    "r2": ols["r2"],
                },
            ]
        )
    panel_tests_df = pd.DataFrame(panel_tests)

    regionwise_rows = []
    region_pairs = [
        ("total_annual_area_km2_anomaly_z", "glambie_combined_mwe_anomaly", "mwe_anomaly_primary"),
        ("mean_ratio_anomaly_z", "glambie_combined_mwe_anomaly", "mwe_anomaly_primary"),
        ("usable_total_area_change_km2", "glambie_combined_mwe_rolling3", "mwe_rolling3_supportive"),
    ]
    for region_key, sub in shared.groupby("region_key"):
        for response, predictor, family in region_pairs:
            pearson = corr_stats(sub[predictor], sub[response], method="pearson")
            spearman = corr_stats(sub[predictor], sub[response], method="spearman")
            regionwise_rows.extend(
                [
                    {
                        "region_key": region_key,
                        "region_name": sub["region_name"].iloc[0],
                        "response": response,
                        "predictor": predictor,
                        "predictor_family": family,
                        "test_name": "pearson",
                        "n": pearson["n"],
                        "estimate": pearson["estimate"],
                        "p_value": pearson["p_value"],
                    },
                    {
                        "region_key": region_key,
                        "region_name": sub["region_name"].iloc[0],
                        "response": response,
                        "predictor": predictor,
                        "predictor_family": family,
                        "test_name": "spearman",
                        "n": spearman["n"],
                        "estimate": spearman["estimate"],
                        "p_value": spearman["p_value"],
                    },
                ]
            )
    regionwise_df = pd.DataFrame(regionwise_rows).sort_values(["response", "predictor", "region_key", "test_name"])

    ready_path = out_dir / f"{tag}_analysis_ready_panel.csv"
    region_summary_path = out_dir / f"{tag}_longterm_region_summary.csv"
    longterm_tests_path = out_dir / f"{tag}_longterm_tests.csv"
    panel_tests_path = out_dir / f"{tag}_panel_tests.csv"
    regionwise_path = out_dir / f"{tag}_regionwise_correlations.csv"
    summary_json_path = out_dir / f"{tag}_analysis_summary.json"

    analysis_ready.to_csv(ready_path, index=False)
    region_summary.to_csv(region_summary_path, index=False)
    longterm_tests_df.to_csv(longterm_tests_path, index=False)
    panel_tests_df.to_csv(panel_tests_path, index=False)
    regionwise_df.to_csv(regionwise_path, index=False)

    summary = {
        "tag": tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "panel_csv": str(panel_path),
        },
        "outputs": {
            "analysis_ready_panel_csv": str(ready_path),
            "longterm_region_summary_csv": str(region_summary_path),
            "longterm_tests_csv": str(longterm_tests_path),
            "panel_tests_csv": str(panel_tests_path),
            "regionwise_correlations_csv": str(regionwise_path),
        },
        "sample_checks": {
            "panel_rows_total": int(len(panel)),
            "panel_regions_total": int(panel["region_key"].nunique()),
            "glambie_missing_rows": int(panel["glambie_combined_mwe"].isna().sum()),
            "glambie_missing_years": sorted(panel.loc[panel["glambie_combined_mwe"].isna(), "year"].dropna().unique().tolist()),
            "shared_sample_rows": int(len(shared)),
            "shared_sample_regions": int(shared["region_key"].nunique()),
            "shared_year_min": int(shared["year"].min()),
            "shared_year_max": int(shared["year"].max()),
        },
        "analysis_design": {
            "primary_predictor_family": "mwe",
            "exclude_2024_from_glambie_tests": True,
            "longterm_period": "2000-2023",
            "panel_period": "2000-2023",
        },
        "headline_results": {
            "longterm_tests_preview": longterm_tests_df.head(6).to_dict(orient="records"),
            "panel_tests_preview": panel_tests_df.head(8).to_dict(orient="records"),
        },
    }
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
