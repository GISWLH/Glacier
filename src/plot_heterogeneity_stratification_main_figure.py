from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
ANALYSIS_DIR = ROOT / "data" / "processed" / "analysis" / "glambie_anomaly_robustness_v1"
FIG_SOURCE_DIR = ROOT / "manuscript" / "figures_for_paper"
OUT_DIR = FIG_SOURCE_DIR / "rendered" / "heterogeneity_stratification_main"
MANIFEST_PATH = OUT_DIR / "heterogeneity_stratification_figure04_manifest.json"
PANEL_A_SOURCE = FIG_SOURCE_DIR / "heterogeneity_stratification_figure04_panel_a_source.csv"
PANEL_B_SOURCE = FIG_SOURCE_DIR / "heterogeneity_stratification_figure04_panel_b_source.csv"
PANEL_C_SOURCE = FIG_SOURCE_DIR / "heterogeneity_stratification_figure04_panel_c_source.csv"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "black": "#000000",
    "gray": "#7A7A7A",
    "light_gray": "#D9D9D9",
}

STRATUM_COLORS = {
    "core_negative": OKABE_ITO["blue"],
    "supporting_negative": OKABE_ITO["green"],
    "weak_or_opposite": OKABE_ITO["orange"],
}

STRATUM_LABELS = {
    "core_negative": "Core negative",
    "supporting_negative": "Supporting negative",
    "weak_or_opposite": "Weak or opposite",
}

SUBSET_LABELS = {
    "baseline_shared_2000_2023": "Baseline",
    "usable_share_ge_0.50": "Usable share ≥ 0.50",
    "usable_share_ge_0.70": "Usable share ≥ 0.70",
    "drop_high_suspicious_ge_100": "Drop suspicious ≥ 100",
    "drop_low_overall_quality_lt_0.60": "Drop overall quality < 0.60",
    "drop_both_low_quality_and_high_suspicious": "Drop both filters",
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


def short_label(region_key: str) -> str:
    mapping = {
        "greenland_periphery": "Greenland",
        "arctic_canada_south": "Arctic Can. S",
        "southern_andes": "S. Andes",
        "alaska": "Alaska",
        "central_asia": "Central Asia",
        "scandinavia": "Scandinavia",
        "western_canada_us": "W. Can./US",
        "russian_arctic": "Russian Arctic",
        "south_asia_east": "South Asia E",
        "low_latitudes": "Low Latitudes",
        "north_asia": "North Asia",
        "arctic_canada_north": "Arctic Can. N",
        "south_asia_west": "South Asia W",
        "central_europe": "Central Europe",
        "caucasus_middle_east": "Caucasus/ME",
        "iceland": "Iceland",
        "new_zealand": "New Zealand",
    }
    return mapping.get(region_key, region_key)


def load_panel_a() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_DIR / "glambie_anomaly_robustness_v1_stratified_regions.csv")
    df["label"] = df["region_key"].map(short_label)
    df = df.sort_values("estimate", ascending=True).reset_index(drop=True)
    df["significant"] = df["p_value"] < 0.05
    return df


def load_panel_b() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_DIR / "glambie_anomaly_robustness_v1_stratum_tests.csv")
    inv = pd.read_csv(ANALYSIS_DIR / "glambie_anomaly_robustness_v1_stratum_inventory.csv")
    pearson = df.loc[
        (df["response"] == "total_annual_area_km2_anomaly_z")
        & (df["predictor"] == "glambie_combined_mwe_anomaly")
        & (df["test_name"] == "pearson")
    ].copy()
    ols = df.loc[
        (df["response"] == "total_annual_area_km2_anomaly_z")
        & (df["predictor"] == "glambie_combined_mwe_anomaly")
        & (df["test_name"] == "ols")
    ][["subset_name", "r2", "slope"]].copy().rename(columns={"r2": "ols_r2", "slope": "ols_slope"})
    pearson = pearson.merge(ols, on="subset_name", how="left")
    pearson["stratum"] = pearson["subset_name"].str.replace("stratum::", "", regex=False)
    pearson = pearson.merge(inv[["stratum", "n_rows", "n_regions"]], on="stratum", how="left", suffixes=("", "_inventory"))
    pearson["label"] = pearson["stratum"].map(STRATUM_LABELS)
    order = ["core_negative", "supporting_negative", "weak_or_opposite"]
    pearson["stratum"] = pd.Categorical(pearson["stratum"], categories=order, ordered=True)
    pearson = pearson.sort_values("stratum").reset_index(drop=True)
    return pearson


def load_panel_c() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_DIR / "glambie_anomaly_robustness_v1_robustness_tests.csv")
    pearson = df.loc[
        (df["response"] == "total_annual_area_km2_anomaly_z")
        & (df["predictor"] == "glambie_combined_mwe_anomaly")
        & (df["test_name"] == "pearson")
    ].copy()
    ols = df.loc[
        (df["response"] == "total_annual_area_km2_anomaly_z")
        & (df["predictor"] == "glambie_combined_mwe_anomaly")
        & (df["test_name"] == "ols")
    ][["subset_name", "r2", "slope"]].copy().rename(columns={"r2": "ols_r2", "slope": "ols_slope"})
    out = pearson.merge(ols, on="subset_name", how="left")
    order = [
        "baseline_shared_2000_2023",
        "usable_share_ge_0.50",
        "usable_share_ge_0.70",
        "drop_high_suspicious_ge_100",
        "drop_low_overall_quality_lt_0.60",
        "drop_both_low_quality_and_high_suspicious",
    ]
    out["subset_name"] = pd.Categorical(out["subset_name"], categories=order, ordered=True)
    out = out.sort_values("subset_name").reset_index(drop=True)
    out["label"] = out["subset_name"].map(SUBSET_LABELS)
    out["significant"] = out["p_value"] < 0.05
    return out


def build_figure(panel_a: pd.DataFrame, panel_b: pd.DataFrame, panel_c: pd.DataFrame) -> dict[str, str]:
    fig = plt.figure(figsize=(7.2, 5.35))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.42)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    y = np.arange(len(panel_a))
    ax1.axvline(0, color=OKABE_ITO["light_gray"], linewidth=0.8)
    for idx, row in panel_a.iterrows():
        color = STRATUM_COLORS[row["stratum"]]
        ax1.hlines(idx, 0, row["estimate"], color=color, linewidth=1.2, zorder=1)
        face = color if bool(row["significant"]) else "white"
        ax1.scatter(row["estimate"], idx, s=24, facecolor=face, edgecolor=color, linewidth=1.0, zorder=2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(panel_a["label"])
    ax1.set_xlabel("Region-level Pearson r")
    ax1.set_title("Global heterogeneity is structured, not random", pad=4)
    ax1.set_xlim(-0.82, 0.32)
    ax1.set_ylim(-0.7, len(panel_a) - 0.3)
    core_count = int((panel_a["stratum"] == "core_negative").sum())
    support_count = int((panel_a["stratum"] == "supporting_negative").sum())
    weak_count = int((panel_a["stratum"] == "weak_or_opposite").sum())
    ax1.text(0.02, 0.98, f"Core: {core_count}\nSupporting: {support_count}\nWeak/opposite: {weak_count}", transform=ax1.transAxes, ha="left", va="top", fontsize=5.8)
    add_panel_label(ax1, "a")

    x2 = np.arange(len(panel_b))
    colors_b = [STRATUM_COLORS[s] for s in panel_b["stratum"]]
    ax2.bar(x2, panel_b["estimate"], color=colors_b, width=0.68)
    ax2.axhline(0, color=OKABE_ITO["light_gray"], linewidth=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(panel_b["label"])
    ax2.set_ylabel("Pooled Pearson r")
    ax2.set_title("The main signal is concentrated in core regions", pad=4)
    for xi, row in panel_b.iterrows():
        ax2.text(xi, row["estimate"] - 0.03 if row["estimate"] < 0 else row["estimate"] + 0.02, f"{row['estimate']:.3f}\nR$^2$={row['ols_r2']:.3f}", ha="center", va="top" if row["estimate"] < 0 else "bottom", fontsize=5.6)
    add_panel_label(ax2, "b")

    y3 = np.arange(len(panel_c))
    ax3.axvline(0, color=OKABE_ITO["light_gray"], linewidth=0.8)
    facecolors = [OKABE_ITO["blue"] if sig else "white" for sig in panel_c["significant"]]
    ax3.hlines(y3, 0, panel_c["estimate"], color=OKABE_ITO["gray"], linewidth=1.0, zorder=1)
    ax3.scatter(panel_c["estimate"], y3, s=24, facecolor=facecolors, edgecolor=OKABE_ITO["blue"], linewidth=1.0, zorder=2)
    ax3.set_yticks(y3)
    ax3.set_yticklabels(panel_c["label"])
    ax3.set_xlabel("Pearson r")
    ax3.set_title("The primary effect is robust to filtering", pad=4)
    for yi, row in panel_c.iterrows():
        ax3.text(row["estimate"] + 0.01, yi, f"n={int(row['n_rows'])}", va="center", ha="left", fontsize=5.5)
    add_panel_label(ax3, "c")

    outputs = save_figure(fig, "heterogeneity_stratification_figure04")
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "panel_a_source": str(PANEL_A_SOURCE),
                "panel_b_source": str(PANEL_B_SOURCE),
                "panel_c_source": str(PANEL_C_SOURCE),
                "analysis_inputs": [
                    str(ANALYSIS_DIR / "glambie_anomaly_robustness_v1_stratified_regions.csv"),
                    str(ANALYSIS_DIR / "glambie_anomaly_robustness_v1_stratum_tests.csv"),
                    str(ANALYSIS_DIR / "glambie_anomaly_robustness_v1_stratum_inventory.csv"),
                    str(ANALYSIS_DIR / "glambie_anomaly_robustness_v1_robustness_tests.csv"),
                ],
                "figure_style": "Nature-style heterogeneity figure with region ranking, stratum summary, and robustness stability panels",
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
                    "panel_a": "Regionwise primary anomaly correlations and strata",
                    "panel_b": "Pooled primary anomaly correlations by stratum",
                    "panel_c": "Primary anomaly correlations across robustness subsets"
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
    panel_a = load_panel_a()
    panel_b = load_panel_b()
    panel_c = load_panel_c()
    PANEL_A_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    panel_a.to_csv(PANEL_A_SOURCE, index=False)
    panel_b.to_csv(PANEL_B_SOURCE, index=False)
    panel_c.to_csv(PANEL_C_SOURCE, index=False)
    outputs = build_figure(panel_a, panel_b, panel_c)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
