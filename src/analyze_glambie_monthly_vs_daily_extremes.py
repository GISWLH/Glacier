from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_PANEL = ROOT / "data" / "processed" / "formal_region_year_panel" / "formal_region_year_panel_with_era5_and_daily_extremes_v1.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "analysis" / "glambie_monthly_vs_daily_extremes_v1"
DEFAULT_TAG = "glambie_monthly_vs_daily_extremes_v1"

RESPONSES = [
    "total_annual_area_km2_anomaly_z",
    "mean_ratio_anomaly_z",
    "zero_ratio_share_anomaly_z",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare monthly warm-season anomalies against daily extreme warm indicators in joint models.")
    p.add_argument("--panel-csv", default=str(DEFAULT_PANEL))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--tag", default=DEFAULT_TAG)
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


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


def ols_stats_from_xy(X: pd.DataFrame, y: pd.Series) -> dict:
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].astype(float)
    y = y.loc[mask].astype(float)
    n = int(len(y))
    if n < 3:
        return {"n": n, "coefficients": {}, "intercept": np.nan, "r2": np.nan, "adj_r2": np.nan}

    Xmat = np.column_stack([np.ones(n), X.values])
    beta, *_ = np.linalg.lstsq(Xmat, y.values, rcond=None)
    yhat = Xmat @ beta
    ss_res = float(np.sum((y.values - yhat) ** 2))
    ss_tot = float(np.sum((y.values - np.mean(y.values)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    p = X.shape[1]
    adj_r2 = np.nan if n <= p + 1 or np.isnan(r2) else 1 - (1 - r2) * (n - 1) / (n - p - 1)
    coeffs = {col: float(beta[i + 1]) for i, col in enumerate(X.columns)}
    return {
        "n": n,
        "coefficients": coeffs,
        "intercept": float(beta[0]),
        "r2": float(r2),
        "adj_r2": float(adj_r2) if pd.notna(adj_r2) else np.nan,
    }


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / std


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
            "freeze_version",
            "panel_version",
            "region_code",
            "region_key",
            "region_name",
            "final_tag",
            "year",
            "usable_share",
            "glambie_combined_mwe_anomaly",
            "warm_season_t2m_anomaly_c",
            "warm_season_precip_anomaly_mm",
            "era5l_tx90p_daily",
            "era5l_wsdi_daily",
            "warm_extreme_year_flag_daily",
            "total_annual_area_km2_anomaly_z",
            "mean_ratio_anomaly_z",
            "zero_ratio_share_anomaly_z",
        ],
    )

    panel["comparison_sample_flag"] = (
        (panel["year"] <= 2023)
        & panel["glambie_combined_mwe_anomaly"].notna()
        & panel["warm_season_t2m_anomaly_c"].notna()
        & panel["warm_season_precip_anomaly_mm"].notna()
        & panel["era5l_tx90p_daily"].notna()
        & panel["era5l_wsdi_daily"].notna()
        & panel["warm_extreme_year_flag_daily"].notna()
    )

    analysis_cols = [
        "freeze_version",
        "panel_version",
        "region_code",
        "region_key",
        "region_name",
        "final_tag",
        "year",
        "usable_share",
        "comparison_sample_flag",
        "glambie_combined_mwe_anomaly",
        "warm_season_t2m_anomaly_c",
        "warm_season_precip_anomaly_mm",
        "era5l_tx90p_daily",
        "era5l_wsdi_daily",
        "warm_extreme_year_flag_daily",
        "total_annual_area_km2_anomaly_z",
        "mean_ratio_anomaly_z",
        "zero_ratio_share_anomaly_z",
    ]
    analysis_ready = panel[analysis_cols].copy().sort_values(["region_key", "year"])
    compare = analysis_ready[analysis_ready["comparison_sample_flag"]].copy().sort_values(["region_key", "year"])

    compare["z_era5l_tx90p_daily"] = zscore(compare["era5l_tx90p_daily"])
    compare["z_era5l_wsdi_daily"] = zscore(compare["era5l_wsdi_daily"])

    sample_inventory = (
        compare.groupby("region_key")
        .agg(
            n_rows=("year", "count"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            usable_share_mean=("usable_share", "mean"),
        )
        .reset_index()
    )
    sample_inventory = sample_inventory.merge(compare[["region_key", "region_name"]].drop_duplicates(), on="region_key", how="left")
    sample_inventory = sample_inventory.sort_values("region_key")

    model_specs = [
        ("M0_glambie_only", ["glambie_combined_mwe_anomaly"]),
        ("M1_monthly_temp", ["glambie_combined_mwe_anomaly", "warm_season_t2m_anomaly_c"]),
        ("M2_monthly_temp_precip", ["glambie_combined_mwe_anomaly", "warm_season_t2m_anomaly_c", "warm_season_precip_anomaly_mm"]),
        ("M3_daily_tx90p", ["glambie_combined_mwe_anomaly", "z_era5l_tx90p_daily"]),
        ("M4_daily_tx90p_wsdi", ["glambie_combined_mwe_anomaly", "z_era5l_tx90p_daily", "z_era5l_wsdi_daily"]),
        ("M5_daily_full", ["glambie_combined_mwe_anomaly", "z_era5l_tx90p_daily", "z_era5l_wsdi_daily", "warm_extreme_year_flag_daily"]),
        ("M6_hybrid_monthly_plus_daily", ["glambie_combined_mwe_anomaly", "warm_season_t2m_anomaly_c", "warm_season_precip_anomaly_mm", "z_era5l_tx90p_daily", "z_era5l_wsdi_daily", "warm_extreme_year_flag_daily"]),
    ]

    model_rows = []
    regionwise_rows = []

    for response in RESPONSES:
        base_r2 = None
        monthly_r2 = None
        for model_name, predictors in model_specs:
            stats = ols_stats_from_xy(compare[predictors], compare[response])
            row = {
                "response": response,
                "model_name": model_name,
                "predictors": ",".join(predictors),
                "n_rows": stats["n"],
                "n_regions": int(compare["region_key"].nunique()),
                "year_min": int(compare["year"].min()),
                "year_max": int(compare["year"].max()),
                "intercept": stats["intercept"],
                "r2": stats["r2"],
                "adj_r2": stats["adj_r2"],
                "delta_r2_vs_M0": np.nan,
                "delta_r2_vs_M1": np.nan,
            }
            for predictor in [
                "glambie_combined_mwe_anomaly",
                "warm_season_t2m_anomaly_c",
                "warm_season_precip_anomaly_mm",
                "z_era5l_tx90p_daily",
                "z_era5l_wsdi_daily",
                "warm_extreme_year_flag_daily",
            ]:
                row[f"coef__{predictor}"] = stats["coefficients"].get(predictor, np.nan)

            if model_name == "M0_glambie_only":
                base_r2 = stats["r2"]
            if model_name == "M1_monthly_temp":
                monthly_r2 = stats["r2"]
            if base_r2 is not None and pd.notna(stats["r2"]):
                row["delta_r2_vs_M0"] = float(stats["r2"] - base_r2)
            if monthly_r2 is not None and pd.notna(stats["r2"]):
                row["delta_r2_vs_M1"] = float(stats["r2"] - monthly_r2)

            model_rows.append(row)

    for region_key, sub in compare.groupby("region_key"):
        for response in RESPONSES:
            for predictor in [
                "warm_season_t2m_anomaly_c",
                "era5l_tx90p_daily",
                "era5l_wsdi_daily",
                "warm_extreme_year_flag_daily",
            ]:
                pearson = corr_stats(sub[predictor], sub[response], method="pearson")
                spearman = corr_stats(sub[predictor], sub[response], method="spearman")
                regionwise_rows.extend([
                    {
                        "region_key": region_key,
                        "region_name": sub["region_name"].iloc[0],
                        "response": response,
                        "predictor": predictor,
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
                        "test_name": "spearman",
                        "n": spearman["n"],
                        "estimate": spearman["estimate"],
                        "p_value": spearman["p_value"],
                    },
                ])

    model_tests_df = pd.DataFrame(model_rows)
    regionwise_df = pd.DataFrame(regionwise_rows).sort_values(["response", "predictor", "region_key", "test_name"])

    analysis_ready_csv = out_dir / f"{tag}_analysis_ready_panel.csv"
    sample_inventory_csv = out_dir / f"{tag}_sample_inventory.csv"
    model_tests_csv = out_dir / f"{tag}_model_tests.csv"
    regionwise_csv = out_dir / f"{tag}_regionwise_correlations.csv"
    summary_json = out_dir / f"{tag}_analysis_summary.json"

    analysis_ready.to_csv(analysis_ready_csv, index=False)
    sample_inventory.to_csv(sample_inventory_csv, index=False)
    model_tests_df.to_csv(model_tests_csv, index=False)
    regionwise_df.to_csv(regionwise_csv, index=False)

    summary = {
        "tag": tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "panel_csv": str(panel_path),
        },
        "outputs": {
            "analysis_ready_panel_csv": str(analysis_ready_csv),
            "sample_inventory_csv": str(sample_inventory_csv),
            "model_tests_csv": str(model_tests_csv),
            "regionwise_correlations_csv": str(regionwise_csv),
        },
        "sample": {
            "rows": int(len(compare)),
            "regions": int(compare["region_key"].nunique()),
            "year_min": int(compare["year"].min()),
            "year_max": int(compare["year"].max()),
        },
        "headline_preview": model_tests_df.head(21).to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
