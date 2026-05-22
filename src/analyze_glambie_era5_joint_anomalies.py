from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_PANEL = ROOT / "data" / "processed" / "formal_region_year_panel" / "formal_region_year_panel_with_era5_v1.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "analysis" / "glambie_era5_joint_anomalies_v1"
DEFAULT_TAG = "glambie_era5_joint_anomalies_v1"

RESPONSES = [
    "total_annual_area_km2_anomaly_z",
    "mean_ratio_anomaly_z",
    "zero_ratio_share_anomaly_z",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run first-round joint GlaMBIE + ERA5 anomaly analyses.")
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
            "region_code",
            "region_key",
            "region_name",
            "year",
            "era5_joined",
            "glambie_combined_mwe_anomaly",
            "glambie_combined_gt_anomaly",
            "warm_season_t2m_anomaly_c",
            "warm_season_precip_anomaly_mm",
            "total_annual_area_km2_anomaly_z",
            "mean_ratio_anomaly_z",
            "zero_ratio_share_anomaly_z",
            "usable_share",
            "glambie_glacier_area_km2",
        ],
    )

    panel["joint_sample_flag"] = (
        panel["era5_joined"].eq(True)
        & panel["glambie_combined_mwe_anomaly"].notna()
        & panel["warm_season_t2m_anomaly_c"].notna()
        & (panel["year"] <= 2023)
    )

    analysis_cols = [
        "freeze_version",
        "panel_version",
        "region_code",
        "region_key",
        "region_name",
        "final_tag",
        "year",
        "joint_sample_flag",
        "usable_share",
        "glambie_glacier_area_km2",
        "glambie_combined_mwe_anomaly",
        "glambie_combined_gt_anomaly",
        "warm_season_t2m_anomaly_c",
        "warm_season_precip_anomaly_mm",
        "total_annual_area_km2_anomaly_z",
        "mean_ratio_anomaly_z",
        "zero_ratio_share_anomaly_z",
    ]
    analysis_ready = panel[analysis_cols].copy().sort_values(["region_key", "year"])

    joint = analysis_ready[analysis_ready["joint_sample_flag"]].copy().sort_values(["region_key", "year"])

    sample_inventory = (
        joint.groupby("region_key")
        .agg(
            n_rows=("year", "count"),
            year_min=("year", "min"),
            year_max=("year", "max"),
            usable_share_mean=("usable_share", "mean"),
        )
        .reset_index()
    )
    sample_inventory = sample_inventory.merge(
        joint[["region_key", "region_name"]].drop_duplicates(),
        on="region_key",
        how="left",
    )
    sample_inventory = sample_inventory.sort_values("region_key")

    model_rows = []
    regionwise_rows = []

    model_specs = [
        ("M1_glambie_only", ["glambie_combined_mwe_anomaly"]),
        ("M2_temp_only", ["warm_season_t2m_anomaly_c"]),
        ("M3_joint_core", ["glambie_combined_mwe_anomaly", "warm_season_t2m_anomaly_c"]),
        ("M4_joint_plus_precip", ["glambie_combined_mwe_anomaly", "warm_season_t2m_anomaly_c", "warm_season_precip_anomaly_mm"]),
        ("M5_gt_swap", ["glambie_combined_gt_anomaly", "warm_season_t2m_anomaly_c"]),
    ]

    for response in RESPONSES:
        for model_name, predictors in model_specs:
            stats = ols_stats_from_xy(joint[predictors], joint[response])
            row = {
                "response": response,
                "model_name": model_name,
                "predictors": ",".join(predictors),
                "n_rows": stats["n"],
                "n_regions": int(joint["region_key"].nunique()),
                "year_min": int(joint["year"].min()),
                "year_max": int(joint["year"].max()),
                "intercept": stats["intercept"],
                "r2": stats["r2"],
                "adj_r2": stats["adj_r2"],
            }
            for predictor in [
                "glambie_combined_mwe_anomaly",
                "glambie_combined_gt_anomaly",
                "warm_season_t2m_anomaly_c",
                "warm_season_precip_anomaly_mm",
            ]:
                row[f"coef__{predictor}"] = stats["coefficients"].get(predictor, np.nan)
            model_rows.append(row)

    for region_key, sub in joint.groupby("region_key"):
        for response in RESPONSES:
            for predictor in [
                "glambie_combined_mwe_anomaly",
                "warm_season_t2m_anomaly_c",
                "warm_season_precip_anomaly_mm",
            ]:
                pearson = corr_stats(sub[predictor], sub[response], method="pearson")
                spearman = corr_stats(sub[predictor], sub[response], method="spearman")
                regionwise_rows.extend(
                    [
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
                    ]
                )

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
            "joint_rows": int(len(joint)),
            "joint_regions": int(joint["region_key"].nunique()),
            "joint_region_list": sorted(joint["region_key"].unique().tolist()),
            "joint_year_min": int(joint["year"].min()),
            "joint_year_max": int(joint["year"].max()),
        },
        "models": model_tests_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
