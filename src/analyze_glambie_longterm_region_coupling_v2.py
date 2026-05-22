from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_PANEL = ROOT / "data" / "processed" / "formal_region_year_panel" / "formal_region_year_panel_v1.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "analysis" / "glambie_longterm_region_coupling_v2"
DEFAULT_TAG = "glambie_longterm_region_coupling_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Redesign long-term region-level GlaMBIE coupling metrics and tests.")
    p.add_argument("--panel-csv", default=str(DEFAULT_PANEL))
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


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_region_metrics(shared: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region_key, sub in shared.groupby("region_key"):
        sub = sub.sort_values("year")
        first = sub.iloc[0]
        last = sub.iloc[-1]
        lake_area_start = float(first["total_annual_area_km2"])
        lake_area_end = float(last["total_annual_area_km2"])
        lake_area_change = lake_area_end - lake_area_start
        glacier_area_mean = float(sub["glambie_glacier_area_km2"].mean())
        lake_area_change_frac = lake_area_change / lake_area_start if lake_area_start != 0 else np.nan
        lake_area_change_per_glacier_area = lake_area_change / glacier_area_mean if glacier_area_mean != 0 else np.nan

        lake_area_slope = ols_stats(sub["year"], sub["total_annual_area_km2"])
        mean_ratio_slope = ols_stats(sub["year"], sub["mean_ratio"])
        mwe_slope = ols_stats(sub["year"], sub["glambie_combined_mwe"])
        gt_slope = ols_stats(sub["year"], sub["glambie_combined_gt"])

        rows.append(
            {
                "region_code": int(first["region_code"]),
                "region_key": region_key,
                "region_name": first["region_name"],
                "year_start": int(sub["year"].min()),
                "year_end": int(sub["year"].max()),
                "n_shared_years": int(len(sub)),
                "lake_area_2000_km2": lake_area_start,
                "lake_area_2023_km2": lake_area_end,
                "lake_area_change_2000_2023_km2": lake_area_change,
                "lake_area_change_frac_2000_2023": lake_area_change_frac,
                "lake_area_change_per_glacier_area": lake_area_change_per_glacier_area,
                "total_annual_area_km2_slope_per_year": lake_area_slope["slope"],
                "total_annual_area_km2_slope_r2": lake_area_slope["r2"],
                "mean_ratio_slope_per_year": mean_ratio_slope["slope"],
                "mean_ratio_slope_r2": mean_ratio_slope["r2"],
                "usable_total_area_change_km2_mean_2000_2023": float(pd.to_numeric(sub["usable_total_area_change_km2"], errors="coerce").mean()),
                "glambie_combined_mwe_cumulative_2023": float(last["glambie_combined_mwe_cumulative"]),
                "glambie_combined_gt_cumulative_2023": float(last["glambie_combined_gt_cumulative"]),
                "glambie_combined_mwe_slope_per_year": mwe_slope["slope"],
                "glambie_combined_mwe_slope_r2": mwe_slope["r2"],
                "glambie_combined_gt_slope_per_year": gt_slope["slope"],
                "glambie_combined_gt_slope_r2": gt_slope["r2"],
                "glambie_glacier_area_km2_mean": glacier_area_mean,
                "usable_share_mean": float(sub["usable_share"].mean()),
                "usable_share_overall": float(first["usable_share_overall"]),
                "suspicious_lake_count": int(first["suspicious_lake_count"]),
            }
        )
    return pd.DataFrame(rows).sort_values("region_key")


def run_tests(df: pd.DataFrame, subset_name: str) -> list[dict]:
    tests = [
        ("lake_area_change_frac_2000_2023", "glambie_combined_mwe_cumulative_2023", "primary", "endpoint_normalized"),
        ("lake_area_change_per_glacier_area", "glambie_combined_mwe_cumulative_2023", "primary", "endpoint_glacier_normalized"),
        ("total_annual_area_km2_slope_per_year", "glambie_combined_mwe_slope_per_year", "supportive", "slope_vs_slope_area"),
        ("mean_ratio_slope_per_year", "glambie_combined_mwe_slope_per_year", "supportive", "slope_vs_slope_ratio"),
        ("usable_total_area_change_km2_mean_2000_2023", "glambie_combined_mwe_cumulative_2023", "supportive", "mean_change_vs_cumulative"),
        ("lake_area_change_2000_2023_km2", "glambie_combined_mwe_cumulative_2023", "sensitivity", "raw_endpoint_old_metric"),
        ("lake_area_change_frac_2000_2023", "glambie_combined_gt_cumulative_2023", "sensitivity", "endpoint_normalized_gt"),
        ("total_annual_area_km2_slope_per_year", "glambie_combined_gt_slope_per_year", "sensitivity", "slope_vs_slope_area_gt"),
    ]

    rows = []
    for response, predictor, family, metric_system in tests:
        pearson = corr_stats(df[predictor], df[response], method="pearson")
        spearman = corr_stats(df[predictor], df[response], method="spearman")
        ols = ols_stats(df[predictor], df[response])
        rows.extend(
            [
                {
                    "subset_name": subset_name,
                    "metric_system": metric_system,
                    "test_family": family,
                    "response": response,
                    "predictor": predictor,
                    "test_name": "pearson",
                    "n_regions": pearson["n"],
                    "estimate": pearson["estimate"],
                    "p_value": pearson["p_value"],
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                },
                {
                    "subset_name": subset_name,
                    "metric_system": metric_system,
                    "test_family": family,
                    "response": response,
                    "predictor": predictor,
                    "test_name": "spearman",
                    "n_regions": spearman["n"],
                    "estimate": spearman["estimate"],
                    "p_value": spearman["p_value"],
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r2": np.nan,
                },
                {
                    "subset_name": subset_name,
                    "metric_system": metric_system,
                    "test_family": family,
                    "response": response,
                    "predictor": predictor,
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
    return rows


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
            "region_code",
            "region_key",
            "region_name",
            "year",
            "total_annual_area_km2",
            "mean_ratio",
            "usable_total_area_change_km2",
            "glambie_combined_mwe",
            "glambie_combined_gt",
            "glambie_combined_mwe_cumulative",
            "glambie_combined_gt_cumulative",
            "glambie_glacier_area_km2",
            "usable_share",
            "usable_share_overall",
            "suspicious_lake_count",
        ],
    )

    panel["shared_sample_flag"] = panel["glambie_combined_mwe"].notna() & (panel["year"] <= 2023)
    shared = panel[panel["shared_sample_flag"]].copy().sort_values(["region_key", "year"])

    region_metrics = build_region_metrics(shared)

    quality_inventory = region_metrics[
        [
            "region_code",
            "region_key",
            "region_name",
            "n_shared_years",
            "usable_share_mean",
            "usable_share_overall",
            "suspicious_lake_count",
            "glambie_glacier_area_km2_mean",
        ]
    ].copy()

    subset_specs: list[tuple[str, pd.DataFrame]] = []
    subset_specs.append(("all_valid_regions", region_metrics.copy()))
    subset_specs.append(("usable_share_mean_ge_0.60", region_metrics[region_metrics["usable_share_mean"] >= 0.60].copy()))
    subset_specs.append(("usable_share_overall_ge_0.60", region_metrics[region_metrics["usable_share_overall"] >= 0.60].copy()))
    subset_specs.append(("drop_high_suspicious_ge_100", region_metrics[region_metrics["suspicious_lake_count"] < 100].copy()))

    test_rows = []
    for subset_name, df in subset_specs:
        test_rows.extend(run_tests(df, subset_name))
    tests_df = pd.DataFrame(test_rows)

    region_metrics_csv = out_dir / f"{tag}_region_metrics.csv"
    tests_csv = out_dir / f"{tag}_longterm_tests.csv"
    quality_csv = out_dir / f"{tag}_quality_inventory.csv"
    summary_json = out_dir / f"{tag}_analysis_summary.json"

    region_metrics.to_csv(region_metrics_csv, index=False)
    tests_df.to_csv(tests_csv, index=False)
    quality_inventory.to_csv(quality_csv, index=False)

    summary = {
        "tag": tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "panel_csv": str(panel_path),
        },
        "outputs": {
            "region_metrics_csv": str(region_metrics_csv),
            "longterm_tests_csv": str(tests_csv),
            "quality_inventory_csv": str(quality_csv),
        },
        "sample": {
            "shared_regions": int(region_metrics["region_key"].nunique()),
            "shared_year_start": int(region_metrics["year_start"].min()),
            "shared_year_end": int(region_metrics["year_end"].max()),
            "min_shared_years": int(region_metrics["n_shared_years"].min()),
            "max_shared_years": int(region_metrics["n_shared_years"].max()),
        },
        "headline_preview": tests_df.head(18).to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
