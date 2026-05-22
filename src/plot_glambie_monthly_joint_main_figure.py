from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
ANALYSIS_DIR = ROOT / "data" / "processed" / "analysis" / "glambie_era5_joint_anomalies_v1"
FIG_SOURCE_DIR = ROOT / "manuscript" / "figures_for_paper"
OUT_DIR = FIG_SOURCE_DIR / "rendered" / "glambie_monthly_joint_main"
MANIFEST_PATH = OUT_DIR / "glambie_monthly_joint_figure02_manifest.json"
PANEL_A_SOURCE = FIG_SOURCE_DIR / "glambie_monthly_joint_figure02_panel_a_source.csv"
PANEL_B_SOURCE = FIG_SOURCE_DIR / "glambie_monthly_joint_figure02_panel_b_source.csv"
PANEL_C_SOURCE = FIG_SOURCE_DIR / "glambie_monthly_joint_figure02_panel_c_source.csv"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "black": "#000000",
    "gray": "#7A7A7A",
    "light_gray": "#D9D9D9",
}

MODEL_COLORS = {
    "M1_glambie_only": OKABE_ITO["gray"],
    "M2_temp_only": OKABE_ITO["orange"],
    "M3_joint_core": OKABE_ITO["blue"],
    "M4_joint_plus_precip": OKABE_ITO["green"],
    "M5_gt_swap": OKABE_ITO["vermillion"],
}

MODEL_LABELS = {
    "M1_glambie_only": "GlaMBIE only",
    "M2_temp_only": "Temperature only",
    "M3_joint_core": "Joint core",
    "M4_joint_plus_precip": "Joint + precip",
    "M5_gt_swap": "Gt swap",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.grid": False,
        }
    )


def add_panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.14,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=OKABE_ITO["black"],
    )


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


def load_model_summary() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_DIR / "glambie_era5_joint_anomalies_v1_model_tests.csv")
    df = df.loc[df["response"] == "total_annual_area_km2_anomaly_z"].copy()
    order = ["M1_glambie_only", "M2_temp_only", "M3_joint_core", "M4_joint_plus_precip", "M5_gt_swap"]
    df["model_name"] = pd.Categorical(df["model_name"], categories=order, ordered=True)
    df = df.sort_values("model_name").reset_index(drop=True)
    df["label"] = df["model_name"].map(MODEL_LABELS)
    best_single = df.loc[df["model_name"].isin(["M1_glambie_only", "M2_temp_only"]), "r2"].max()
    df["delta_vs_best_single"] = df["r2"] - best_single
    return df


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    Xmat = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xmat, y, rcond=None)
    return beta


def residualize(target: np.ndarray, others: np.ndarray) -> np.ndarray:
    beta = fit_ols(others, target)
    Xmat = np.column_stack([np.ones(len(target)), others])
    fitted = Xmat @ beta
    return target - fitted


def regression_band(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    x_mean = x.mean()
    y_mean = y.mean()
    sxx = np.sum((x - x_mean) ** 2)
    slope = np.sum((x - x_mean) * (y - y_mean)) / sxx
    intercept = y_mean - slope * x_mean
    y_hat = intercept + slope * x_grid
    residuals = y - (intercept + slope * x)
    s_err = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
    se_mean = s_err * np.sqrt(1 / len(x) + (x_grid - x_mean) ** 2 / sxx)
    delta = 1.96 * se_mean
    r = float(np.corrcoef(x, y)[0, 1])
    r2 = float(r**2)
    return y_hat, y_hat - delta, y_hat + delta, float(slope), float(intercept), r2


def load_joint_sample() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_DIR / "glambie_era5_joint_anomalies_v1_analysis_ready_panel.csv")
    df = df.loc[df["joint_sample_flag"] == True].copy()
    return df


def build_partial_panel(df: pd.DataFrame, focal: str, controls: list[str], focal_label: str) -> pd.DataFrame:
    y = pd.to_numeric(df["total_annual_area_km2_anomaly_z"], errors="coerce").to_numpy(dtype=float)
    x = pd.to_numeric(df[focal], errors="coerce").to_numpy(dtype=float)
    controls_mat = df[controls].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y_resid = residualize(y, controls_mat)
    x_resid = residualize(x, controls_mat)
    out = df[["region_key", "region_name", "year"]].copy()
    out["focal_variable"] = focal
    out["focal_label"] = focal_label
    out["x_residual"] = x_resid
    out["y_residual"] = y_resid
    out["x_raw"] = x
    out["y_raw"] = y
    return out


def build_figure(model_df: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame) -> dict[str, str]:
    fig = plt.figure(figsize=(7.2, 5.3))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.85, 1.0], hspace=0.42, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    x = np.arange(len(model_df))
    colors = [MODEL_COLORS[m] for m in model_df["model_name"]]
    ax1.bar(x, model_df["r2"], color=colors, width=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_df["label"])
    ax1.set_ylabel("Model R$^2$")
    ax1.set_ylim(0, max(model_df["r2"]) * 1.28)
    ax1.set_title("Joint monthly climate models improve explanation", pad=4)
    ax1.axhline(0, color=OKABE_ITO["black"], linewidth=0.8)
    for xi, row in model_df.iterrows():
        ax1.text(xi, row["r2"] + 0.004, f"{row['r2']:.3f}", ha="center", va="bottom", fontsize=5.8)
        if row["model_name"] in {"M3_joint_core", "M4_joint_plus_precip", "M5_gt_swap"}:
            ax1.text(
                xi,
                row["r2"] + 0.018,
                f"Δ {row['delta_vs_best_single']:+.3f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color=OKABE_ITO["black"],
            )
    ax1.text(
        0.01,
        0.98,
        "Response: lake area anomaly (z score)\nSample: 408 region-years, 17 regions, 2000–2023",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
    )
    add_panel_label(ax1, "a")

    for ax, panel_df, line_color, title, xlabel in [
        (
            ax2,
            panel_b,
            OKABE_ITO["blue"],
            "Independent glacier contribution",
            "Residual glacier anomaly (m w.e.)",
        ),
        (
            ax3,
            panel_c,
            OKABE_ITO["orange"],
            "Independent temperature contribution",
            "Residual warm-season temperature anomaly (°C)",
        ),
    ]:
        xv = panel_df["x_residual"].to_numpy(dtype=float)
        yv = panel_df["y_residual"].to_numpy(dtype=float)
        x_grid = np.linspace(xv.min() - 0.05 * (xv.max() - xv.min()), xv.max() + 0.05 * (xv.max() - xv.min()), 250)
        y_hat, y_lo, y_hi, slope, _, r2 = regression_band(xv, yv, x_grid)
        ax.scatter(xv, yv, s=13, color=OKABE_ITO["gray"], alpha=0.45, linewidth=0, rasterized=True, zorder=2)
        ax.fill_between(x_grid, y_lo, y_hi, color=line_color, alpha=0.18, zorder=1)
        ax.plot(x_grid, y_hat, color=line_color, linewidth=1.2, zorder=3)
        ax.axhline(0, color=OKABE_ITO["light_gray"], linewidth=0.8, zorder=0)
        ax.axvline(0, color=OKABE_ITO["light_gray"], linewidth=0.8, zorder=0)
        ax.set_title(title, pad=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Residual lake area anomaly")
        ax.text(
            0.03,
            0.97,
            f"Slope = {slope:.3f}\nR$^2$ = {r2:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
        )

    add_panel_label(ax2, "b")
    add_panel_label(ax3, "c")

    outputs = save_figure(fig, "glambie_monthly_joint_figure02")
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "panel_a_source": str(PANEL_A_SOURCE),
                "panel_b_source": str(PANEL_B_SOURCE),
                "panel_c_source": str(PANEL_C_SOURCE),
                "analysis_inputs": [
                    str(ANALYSIS_DIR / "glambie_era5_joint_anomalies_v1_model_tests.csv"),
                    str(ANALYSIS_DIR / "glambie_era5_joint_anomalies_v1_analysis_ready_panel.csv"),
                ],
                "figure_style": "Nature-style main-text figure with model-comparison panel plus two added-variable panels",
                "nature_specs_applied": {
                    "width_mm": 183,
                    "font_family": "Arial",
                    "body_text_pt": "5-7",
                    "panel_label_pt": 8,
                    "vector_text": True,
                    "pdf_fonttype": 42,
                    "line_width_pt_range": "0.25-1",
                    "color_palette": "Okabe-Ito"
                },
                "source_notes": {
                    "panel_b": "Residual lake area anomaly and residual glacier anomaly after controlling for warm_season_t2m_anomaly_c and warm_season_precip_anomaly_mm",
                    "panel_c": "Residual lake area anomaly and residual warm_season_t2m_anomaly_c after controlling for glambie_combined_mwe_anomaly and warm_season_precip_anomaly_mm"
                },
                **outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    setup_style()
    model_df = load_model_summary()
    joint_df = load_joint_sample()
    panel_b = build_partial_panel(
        joint_df,
        focal="glambie_combined_mwe_anomaly",
        controls=["warm_season_t2m_anomaly_c", "warm_season_precip_anomaly_mm"],
        focal_label="Glacier mass-balance anomaly",
    )
    panel_c = build_partial_panel(
        joint_df,
        focal="warm_season_t2m_anomaly_c",
        controls=["glambie_combined_mwe_anomaly", "warm_season_precip_anomaly_mm"],
        focal_label="Warm-season temperature anomaly",
    )
    PANEL_A_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(PANEL_A_SOURCE, index=False)
    panel_b.to_csv(PANEL_B_SOURCE, index=False)
    panel_c.to_csv(PANEL_C_SOURCE, index=False)
    outputs = build_figure(model_df, panel_b, panel_c)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
