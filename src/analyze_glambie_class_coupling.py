from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
DEFAULT_PANEL = ROOT / "data" / "processed" / "formal_class_region_year_panel" / "formal_class_region_year_panel_v1.csv"
DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "analysis" / "glambie_class_coupling_v1"
DEFAULT_TAG = "glambie_class_coupling_v1"

RESPONSES = [
    "total_annual_area_km2_anomaly_z",
    "mean_ratio_anomaly_z",
    "zero_ratio_share_anomaly_z",
]

PREDICTORS = [
    "glambie_combined_mwe_anomaly",
    "warm_season_t2m_anomaly_c",
    "warm_season_precip_anomaly_mm",
]

PREFERRED_CLASS_ORDER = [
    "proglacial_detached",
    "proglacial_contacted",
    "supraglacial",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run class-region-year GlaMBIE + climate coupling analyses.")
    p.add_argument("--panel-csv", default=str(DEFAULT_PANEL))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--tag", default=DEFAULT_TAG)
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_boolish(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False})
    return mapped.where(mapped.notna(), series)


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


def ols_inference(X: pd.DataFrame, y: pd.Series) -> dict:
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask].astype(float)
    y = y.loc[mask].astype(float)
    n = int(len(y))
    names = ["intercept", *X.columns.tolist()]
    if n < 3:
        return {"n": n, "df_resid": np.nan, "r2": np.nan, "adj_r2": np.nan, "rows": []}

    Xmat = np.column_stack([np.ones(n), X.values])
    beta, *_ = np.linalg.lstsq(Xmat, y.values, rcond=None)
    yhat = Xmat @ beta
    resid = y.values - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y.values - np.mean(y.values)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    p_no_intercept = X.shape[1]
    df_resid = n - p_no_intercept - 1
    adj_r2 = np.nan if df_resid <= 0 or np.isnan(r2) else 1 - (1 - r2) * (n - 1) / df_resid

    cov = np.full((len(names), len(names)), np.nan)
    se = np.full(len(names), np.nan)
    t_stat = np.full(len(names), np.nan)
    p_values = np.full(len(names), np.nan)
    if df_resid > 0:
        xtx_inv = np.linalg.pinv(Xmat.T @ Xmat)
        sigma2 = ss_res / df_resid
        cov = sigma2 * xtx_inv
        se = np.sqrt(np.diag(cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = beta / se
        try:
            from scipy.stats import t as student_t  # type: ignore

            p_values = student_t.sf(np.abs(t_stat), df=df_resid) * 2
        except Exception:
            p_values = np.full(len(names), np.nan)

    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "term": name,
                "estimate": float(beta[i]) if pd.notna(beta[i]) else np.nan,
                "std_error": float(se[i]) if pd.notna(se[i]) else np.nan,
                "t_value": float(t_stat[i]) if pd.notna(t_stat[i]) else np.nan,
                "p_value": float(p_values[i]) if pd.notna(p_values[i]) else np.nan,
            }
        )
    return {
        "n": n,
        "df_resid": int(df_resid) if pd.notna(df_resid) else np.nan,
        "r2": float(r2) if pd.notna(r2) else np.nan,
        "adj_r2": float(adj_r2) if pd.notna(adj_r2) else np.nan,
        "rows": rows,
    }


def ordered_classes(values: pd.Series) -> list[str]:
    present = [c for c in PREFERRED_CLASS_ORDER if c in set(values.dropna().astype(str))]
    extras = sorted(set(values.dropna().astype(str)) - set(present))
    return [*present, *extras]


def build_interaction_matrix(df: pd.DataFrame, predictors: list[str], class_col: str, reference_class: str) -> pd.DataFrame:
    X = df[predictors].apply(pd.to_numeric, errors="coerce").copy()
    classes = ordered_classes(df[class_col])
    for cls in classes:
        if cls == reference_class:
            continue
        dummy = df[class_col].astype(str).eq(cls).astype(float)
        X[f"class__{cls}"] = dummy
        for predictor in predictors:
            X[f"{predictor}__x__{cls}"] = pd.to_numeric(df[predictor], errors="coerce") * dummy
    return X


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
            "harmonized_class",
            "lake_type",
            "analysis_tier",
            "main_analysis_include",
            *RESPONSES,
            *PREDICTORS,
            "usable_share",
            "lake_count",
        ],
    )

    panel["main_analysis_include"] = normalize_boolish(panel["main_analysis_include"]).astype(bool)
    panel["predictor_sample_flag"] = (
        panel["glambie_combined_mwe_anomaly"].notna()
        & panel["warm_season_t2m_anomaly_c"].notna()
        & panel["warm_season_precip_anomaly_mm"].notna()
        & (pd.to_numeric(panel["year"], errors="coerce") <= 2023)
    )
    panel["core_primary_sample_flag"] = panel["predictor_sample_flag"] & panel["main_analysis_include"]
    panel["supplemental_sample_flag"] = panel["predictor_sample_flag"] & panel["analysis_tier"].eq("supplemental")
    panel["all_class_sample_flag"] = panel["predictor_sample_flag"]

    analysis_cols = [
        "freeze_version",
        "panel_version",
        "region_panel_version",
        "region_code",
        "region_key",
        "region_name",
        "final_tag",
        "year",
        "harmonized_class",
        "lake_type",
        "lake_type_label",
        "analysis_tier",
        "main_analysis_include",
        "predictor_sample_flag",
        "core_primary_sample_flag",
        "supplemental_sample_flag",
        "all_class_sample_flag",
        "lake_count",
        "usable_lake_count",
        "usable_share",
        *RESPONSES,
        *PREDICTORS,
        "glambie_combined_mwe_rolling3",
        "glambie_combined_mwe_rolling5",
        "glambie_combined_mwe_cumulative",
        "glambie_glacier_area_km2",
    ]
    analysis_ready = panel[[c for c in analysis_cols if c in panel.columns]].copy().sort_values(["region_key", "harmonized_class", "year"])

    sample_rows = []
    sample_defs = {
        "core_primary": analysis_ready[analysis_ready["core_primary_sample_flag"]].copy(),
        "supplemental_only": analysis_ready[analysis_ready["supplemental_sample_flag"]].copy(),
        "all_classes": analysis_ready[analysis_ready["all_class_sample_flag"]].copy(),
    }
    for sample_name, sample_df in sample_defs.items():
        if sample_df.empty:
            continue
        global_row = {
            "sample_name": sample_name,
            "harmonized_class": "__all__",
            "n_rows": int(len(sample_df)),
            "n_regions": int(sample_df["region_key"].nunique()),
            "n_region_class": int(sample_df[["region_key", "harmonized_class"]].drop_duplicates().shape[0]),
            "year_min": int(sample_df["year"].min()),
            "year_max": int(sample_df["year"].max()),
            "lake_count_mean": float(sample_df["lake_count"].mean()),
            "usable_share_mean": float(sample_df["usable_share"].mean()),
        }
        sample_rows.append(global_row)
        for cls, sub in sample_df.groupby("harmonized_class"):
            sample_rows.append(
                {
                    "sample_name": sample_name,
                    "harmonized_class": cls,
                    "n_rows": int(len(sub)),
                    "n_regions": int(sub["region_key"].nunique()),
                    "n_region_class": int(sub[["region_key", "harmonized_class"]].drop_duplicates().shape[0]),
                    "year_min": int(sub["year"].min()),
                    "year_max": int(sub["year"].max()),
                    "lake_count_mean": float(sub["lake_count"].mean()),
                    "usable_share_mean": float(sub["usable_share"].mean()),
                }
            )
    sample_inventory = pd.DataFrame(sample_rows)

    pooled_rows = []
    pooled_predictors = [
        "glambie_combined_mwe_anomaly",
        "warm_season_t2m_anomaly_c",
        "warm_season_precip_anomaly_mm",
    ]
    for cls, sub in sample_defs["all_classes"].groupby("harmonized_class"):
        for response in RESPONSES:
            for predictor in pooled_predictors:
                pearson = corr_stats(sub[predictor], sub[response], method="pearson")
                spearman = corr_stats(sub[predictor], sub[response], method="spearman")
                ols = ols_stats_from_xy(sub[[predictor]], sub[response])
                pooled_rows.extend(
                    [
                        {
                            "harmonized_class": cls,
                            "response": response,
                            "predictor": predictor,
                            "test_name": "pearson",
                            "n_rows": pearson["n"],
                            "n_regions": int(sub["region_key"].nunique()),
                            "year_min": int(sub["year"].min()),
                            "year_max": int(sub["year"].max()),
                            "estimate": pearson["estimate"],
                            "p_value": pearson["p_value"],
                            "slope": np.nan,
                            "intercept": np.nan,
                            "r2": np.nan,
                            "adj_r2": np.nan,
                        },
                        {
                            "harmonized_class": cls,
                            "response": response,
                            "predictor": predictor,
                            "test_name": "spearman",
                            "n_rows": spearman["n"],
                            "n_regions": int(sub["region_key"].nunique()),
                            "year_min": int(sub["year"].min()),
                            "year_max": int(sub["year"].max()),
                            "estimate": spearman["estimate"],
                            "p_value": spearman["p_value"],
                            "slope": np.nan,
                            "intercept": np.nan,
                            "r2": np.nan,
                            "adj_r2": np.nan,
                        },
                        {
                            "harmonized_class": cls,
                            "response": response,
                            "predictor": predictor,
                            "test_name": "ols",
                            "n_rows": ols["n"],
                            "n_regions": int(sub["region_key"].nunique()),
                            "year_min": int(sub["year"].min()),
                            "year_max": int(sub["year"].max()),
                            "estimate": ols["coefficients"].get(predictor, np.nan),
                            "p_value": np.nan,
                            "slope": ols["coefficients"].get(predictor, np.nan),
                            "intercept": ols["intercept"],
                            "r2": ols["r2"],
                            "adj_r2": ols["adj_r2"],
                        },
                    ]
                )
    pooled_tests = pd.DataFrame(pooled_rows)

    joint_rows = []
    for cls, sub in sample_defs["all_classes"].groupby("harmonized_class"):
        for response in RESPONSES:
            stats = ols_stats_from_xy(sub[PREDICTORS], sub[response])
            row = {
                "harmonized_class": cls,
                "response": response,
                "predictors": ",".join(PREDICTORS),
                "n_rows": stats["n"],
                "n_regions": int(sub["region_key"].nunique()),
                "year_min": int(sub["year"].min()),
                "year_max": int(sub["year"].max()),
                "intercept": stats["intercept"],
                "r2": stats["r2"],
                "adj_r2": stats["adj_r2"],
            }
            for predictor in PREDICTORS:
                row[f"coef__{predictor}"] = stats["coefficients"].get(predictor, np.nan)
            joint_rows.append(row)
    class_joint_models = pd.DataFrame(joint_rows)

    contrast_rows = []
    contrast_samples = {
        "core_primary": sample_defs["core_primary"],
        "all_classes": sample_defs["all_classes"],
    }
    for sample_name, sample_df in contrast_samples.items():
        if sample_df.empty:
            continue
        classes = ordered_classes(sample_df["harmonized_class"])
        if len(classes) < 2:
            continue
        reference_class = classes[0]
        for response in RESPONSES:
            X = build_interaction_matrix(sample_df, PREDICTORS, "harmonized_class", reference_class)
            stats = ols_inference(X, sample_df[response])
            for row in stats["rows"]:
                contrast_rows.append(
                    {
                        "sample_name": sample_name,
                        "response": response,
                        "reference_class": reference_class,
                        "n_rows": stats["n"],
                        "n_regions": int(sample_df["region_key"].nunique()),
                        "year_min": int(sample_df["year"].min()),
                        "year_max": int(sample_df["year"].max()),
                        "df_resid": stats["df_resid"],
                        "r2": stats["r2"],
                        "adj_r2": stats["adj_r2"],
                        **row,
                    }
                )
    class_contrast_tests = pd.DataFrame(contrast_rows)

    regionwise_rows = []
    for (region_key, cls), sub in sample_defs["all_classes"].groupby(["region_key", "harmonized_class"]):
        if sub.empty:
            continue
        for response in RESPONSES:
            for predictor in pooled_predictors:
                pearson = corr_stats(sub[predictor], sub[response], method="pearson")
                spearman = corr_stats(sub[predictor], sub[response], method="spearman")
                regionwise_rows.extend(
                    [
                        {
                            "region_key": region_key,
                            "region_name": sub["region_name"].iloc[0],
                            "harmonized_class": cls,
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
                            "harmonized_class": cls,
                            "response": response,
                            "predictor": predictor,
                            "test_name": "spearman",
                            "n": spearman["n"],
                            "estimate": spearman["estimate"],
                            "p_value": spearman["p_value"],
                        },
                    ]
                )
    regionwise_class_tests = pd.DataFrame(regionwise_rows)
    if not regionwise_class_tests.empty:
        regionwise_class_tests = regionwise_class_tests.sort_values(["response", "predictor", "region_key", "harmonized_class", "test_name"])

    analysis_ready_csv = out_dir / f"{tag}_analysis_ready_panel.csv"
    sample_inventory_csv = out_dir / f"{tag}_sample_inventory.csv"
    class_pooled_tests_csv = out_dir / f"{tag}_class_pooled_tests.csv"
    class_joint_models_csv = out_dir / f"{tag}_class_joint_models.csv"
    class_contrast_tests_csv = out_dir / f"{tag}_class_contrast_tests.csv"
    regionwise_class_tests_csv = out_dir / f"{tag}_regionwise_class_tests.csv"
    summary_json = out_dir / f"{tag}_analysis_summary.json"

    analysis_ready.to_csv(analysis_ready_csv, index=False)
    sample_inventory.to_csv(sample_inventory_csv, index=False)
    pooled_tests.to_csv(class_pooled_tests_csv, index=False)
    class_joint_models.to_csv(class_joint_models_csv, index=False)
    class_contrast_tests.to_csv(class_contrast_tests_csv, index=False)
    regionwise_class_tests.to_csv(regionwise_class_tests_csv, index=False)

    observed_classes = sorted(panel["harmonized_class"].dropna().astype(str).unique().tolist())
    expected_classes = sorted(panel.loc[panel["analysis_tier"].notna(), "harmonized_class"].dropna().astype(str).unique().tolist())
    supplemental_present = sorted(panel.loc[panel["analysis_tier"].eq("supplemental"), "harmonized_class"].dropna().astype(str).unique().tolist())

    summary = {
        "tag": tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "panel_csv": str(panel_path),
        },
        "outputs": {
            "analysis_ready_panel_csv": str(analysis_ready_csv),
            "sample_inventory_csv": str(sample_inventory_csv),
            "class_pooled_tests_csv": str(class_pooled_tests_csv),
            "class_joint_models_csv": str(class_joint_models_csv),
            "class_contrast_tests_csv": str(class_contrast_tests_csv),
            "regionwise_class_tests_csv": str(regionwise_class_tests_csv),
        },
        "sample_checks": {
            "panel_rows_total": int(len(panel)),
            "panel_regions_total": int(panel["region_key"].nunique()),
            "classes_total": observed_classes,
            "core_primary_rows": int(sample_defs["core_primary"].shape[0]),
            "supplemental_rows": int(sample_defs["supplemental_only"].shape[0]),
            "all_class_rows": int(sample_defs["all_classes"].shape[0]),
            "supplemental_classes_present": supplemental_present,
        },
        "coverage_notes": {
            "observed_classes_in_panel": observed_classes,
            "supplemental_classes_present": supplemental_present,
            "supplemental_absent_from_panel": len(supplemental_present) == 0,
            "interpretation": "If supplemental classes are absent from the built class panel, the current frozen annual lake-year data do not support a formal SGL response test yet.",
        },
        "headline_results": {
            "class_joint_models_preview": class_joint_models.head(9).to_dict(orient="records"),
            "class_contrast_preview": class_contrast_tests.head(12).to_dict(orient="records"),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
