from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
ANALYSIS_DIR = ROOT / "data" / "processed" / "analysis" / "glambie_monthly_vs_daily_extremes_v1"
FIG_SOURCE_DIR = ROOT / "manuscript" / "figures_for_paper"
OUT_DIR = FIG_SOURCE_DIR / "rendered" / "monthly_vs_daily_extremes_main"
MANIFEST_PATH = OUT_DIR / "monthly_vs_daily_extremes_figure03_manifest.json"
PANEL_A_SOURCE = FIG_SOURCE_DIR / "monthly_vs_daily_extremes_figure03_panel_a_source.csv"
PANEL_B_SOURCE = FIG_SOURCE_DIR / "monthly_vs_daily_extremes_figure03_panel_b_source.csv"
PANEL_C_SOURCE = FIG_SOURCE_DIR / "monthly_vs_daily_extremes_figure03_panel_c_source.csv"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#7A7A7A",
    "light_gray": "#D9D9D9",
}

MODEL_LABELS = {
    "M0_glambie_only": "GlaMBIE only",
    "M1_monthly_temp": "Monthly temp",
    "M2_monthly_temp_precip": "Monthly + precip",
    "M3_daily_tx90p": "Daily TX90p",
    "M4_daily_tx90p_wsdi": "Daily TX90p + WSDI",
    "M5_daily_full": "Daily full",
    "M6_hybrid_monthly_plus_daily": "Hybrid",
}

MODEL_COLORS = {
    "M0_glambie_only": OKABE_ITO["gray"],
    "M1_monthly_temp": OKABE_ITO["orange"],
    "M2_monthly_temp_precip": OKABE_ITO["orange"],
    "M3_daily_tx90p": OKABE_ITO["sky"],
    "M4_daily_tx90p_wsdi": OKABE_ITO["blue"],
    "M5_daily_full": OKABE_ITO["purple"],
    "M6_hybrid_monthly_plus_daily": OKABE_ITO["green"],
}

PREDICTOR_LABELS = {
    "warm_season_t2m_anomaly_c": "Monthly temp",
    "era5l_tx90p_daily": "TX90p",
    "era5l_wsdi_daily": "WSDI",
    "warm_extreme_year_flag_daily": "Extreme-year flag",
}

PREDICTOR_COLORS = {
    "warm_season_t2m_anomaly_c": OKABE_ITO["orange"],
    "era5l_tx90p_daily": OKABE_ITO["sky"],
    "era5l_wsdi_daily": OKABE_ITO["blue"],
    "warm_extreme_year_flag_daily": OKABE_ITO["purple"],
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
    df = pd.read_csv(ANALYSIS_DIR / "glambie_monthly_vs_daily_extremes_v1_model_tests.csv")
    df = df.loc[df["response"] == "total_annual_area_km2_anomaly_z"].copy()
    order = [
        "M0_glambie_only",
        "M1_monthly_temp",
        "M2_monthly_temp_precip",
        "M3_daily_tx90p",
        "M4_daily_tx90p_wsdi",
        "M5_daily_full",
        "M6_hybrid_monthly_plus_daily",
    ]
    df["model_name"] = pd.Categorical(df["model_name"], categories=order, ordered=True)
    df = df.sort_values("model_name").reset_index(drop=True)
    df["label"] = df["model_name"].map(MODEL_LABELS)
    df["family"] = np.where(
        df["model_name"].eq("M6_hybrid_monthly_plus_daily"),
        "hybrid",
        np.where(df["model_name"].isin(["M1_monthly_temp", "M2_monthly_temp_precip"]), "monthly", np.where(df["model_name"].isin(["M3_daily_tx90p", "M4_daily_tx90p_wsdi", "M5_daily_full"]), "daily", "glacier")),
    )
    return df


def load_regionwise_distribution() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_DIR / "glambie_monthly_vs_daily_extremes_v1_regionwise_correlations.csv")
    df = df.loc[
        (df["response"] == "total_annual_area_km2_anomaly_z")
        & (df["test_name"] == "pearson")
        & (df["predictor"].isin(list(PREDICTOR_LABELS.keys())))
    ].copy()
    df["predictor_label"] = df["predictor"].map(PREDICTOR_LABELS)
    return df


def build_figure(model_df: pd.DataFrame, region_df: pd.DataFrame) -> dict[str, str]:
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.95, 1.05], hspace=0.42, wspace=0.34)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    x = np.arange(len(model_df))
    colors = [MODEL_COLORS[m] for m in model_df["model_name"]]
    ax1.bar(x, model_df["r2"], color=colors, width=0.72)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_df["label"])
    ax1.set_ylabel("Model R$^2$")
    ax1.set_ylim(0, max(model_df["r2"]) * 1.3)
    ax1.set_title("Monthly climate outperforms daily-only extreme blocks", pad=4)
    ax1.axhline(0, color=OKABE_ITO["black"], linewidth=0.8)
    for xi, row in model_df.iterrows():
        ax1.text(xi, row["r2"] + 0.0038, f"{row['r2']:.3f}", ha="center", va="bottom", fontsize=5.8)
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

    compare_df = model_df.loc[model_df["model_name"].isin([
        "M2_monthly_temp_precip",
        "M3_daily_tx90p",
        "M4_daily_tx90p_wsdi",
        "M5_daily_full",
        "M6_hybrid_monthly_plus_daily",
    ])].copy()
    compare_df = compare_df.sort_values("delta_r2_vs_M1", ascending=True).reset_index(drop=True)
    y = np.arange(len(compare_df))
    ax2.axvline(0, color=OKABE_ITO["light_gray"], linewidth=0.8)
    ax2.barh(y, compare_df["delta_r2_vs_M1"], color=[MODEL_COLORS[m] for m in compare_df["model_name"]], height=0.62)
    ax2.set_yticks(y)
    ax2.set_yticklabels(compare_df["label"])
    ax2.set_xlabel("ΔR$^2$ versus monthly-temp model")
    ax2.set_title("Only hybrid and monthly + precip improve on monthly temp", pad=4)
    for yi, row in compare_df.iterrows():
        ha = "left" if row["delta_r2_vs_M1"] >= 0 else "right"
        offset = 0.0012 if row["delta_r2_vs_M1"] >= 0 else -0.0012
        ax2.text(row["delta_r2_vs_M1"] + offset, yi, f"{row['delta_r2_vs_M1']:+.3f}", va="center", ha=ha, fontsize=5.8)
    add_panel_label(ax2, "b")

    pred_order = [
        "warm_season_t2m_anomaly_c",
        "era5l_tx90p_daily",
        "era5l_wsdi_daily",
        "warm_extreme_year_flag_daily",
    ]
    positions = np.arange(len(pred_order))
    for pos, pred in zip(positions, pred_order):
        sub = region_df.loc[region_df["predictor"] == pred, "estimate"].to_numpy(dtype=float)
        ax3.boxplot(
            [sub],
            positions=[pos],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": OKABE_ITO["black"], "linewidth": 1.0},
            boxprops={"facecolor": PREDICTOR_COLORS[pred], "alpha": 0.35, "edgecolor": PREDICTOR_COLORS[pred], "linewidth": 1.0},
            whiskerprops={"color": PREDICTOR_COLORS[pred], "linewidth": 1.0},
            capprops={"color": PREDICTOR_COLORS[pred], "linewidth": 1.0},
        )
        offsets = np.linspace(-0.14, 0.14, len(sub))
        ax3.scatter(np.full(len(sub), pos) + offsets, sub, s=15, color=PREDICTOR_COLORS[pred], alpha=0.75, linewidth=0, zorder=3)
    ax3.axhline(0, color=OKABE_ITO["light_gray"], linewidth=0.8)
    ax3.set_xticks(positions)
    ax3.set_xticklabels([PREDICTOR_LABELS[p] for p in pred_order], rotation=15, ha="right")
    ax3.set_ylabel("Region-level Pearson r")
    ax3.set_title("Monthly temperature is the most consistent climate signal", pad=4)
    summary_lines = []
    for pred in pred_order:
        sig = int((region_df.loc[region_df["predictor"] == pred, "p_value"] < 0.05).sum())
        summary_lines.append(f"{PREDICTOR_LABELS[pred]}: {sig}/17 sig")
    ax3.text(0.02, 0.98, "\n".join(summary_lines), transform=ax3.transAxes, ha="left", va="top", fontsize=5.6)
    add_panel_label(ax3, "c")

    outputs = save_figure(fig, "monthly_vs_daily_extremes_figure03")
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "panel_a_source": str(PANEL_A_SOURCE),
                "panel_b_source": str(PANEL_B_SOURCE),
                "panel_c_source": str(PANEL_C_SOURCE),
                "analysis_inputs": [
                    str(ANALYSIS_DIR / "glambie_monthly_vs_daily_extremes_v1_model_tests.csv"),
                    str(ANALYSIS_DIR / "glambie_monthly_vs_daily_extremes_v1_regionwise_correlations.csv"),
                    str(ANALYSIS_DIR / "glambie_monthly_vs_daily_extremes_v1_analysis_ready_panel.csv"),
                ],
                "figure_style": "Nature-style three-panel figure comparing monthly and daily-extreme climate model performance",
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
                    "panel_a": "Nested model ladder on common 408-row sample",
                    "panel_b": "Delta R2 relative to monthly temperature baseline M1",
                    "panel_c": "Regional Pearson correlation distributions for total_annual_area_km2_anomaly_z against monthly and daily climate indicators"
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
    region_df = load_regionwise_distribution()
    PANEL_A_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_csv(PANEL_A_SOURCE, index=False)
    model_df.loc[model_df["model_name"].isin([
        "M2_monthly_temp_precip",
        "M3_daily_tx90p",
        "M4_daily_tx90p_wsdi",
        "M5_daily_full",
        "M6_hybrid_monthly_plus_daily",
    ])].to_csv(PANEL_B_SOURCE, index=False)
    region_df.to_csv(PANEL_C_SOURCE, index=False)
    outputs = build_figure(model_df, region_df)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
