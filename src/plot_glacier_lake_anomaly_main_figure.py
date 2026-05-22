from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
ANALYSIS_DIR = ROOT / "data" / "processed" / "analysis" / "glambie_coupling_v1"
FIG_SOURCE_DIR = ROOT / "manuscript" / "figures_for_paper"
OUT_DIR = FIG_SOURCE_DIR / "rendered" / "glacier_lake_anomaly_main"
MANIFEST_PATH = OUT_DIR / "glacier_lake_anomaly_figure01_manifest.json"
PANEL_A_SOURCE = FIG_SOURCE_DIR / "glacier_lake_anomaly_figure01_panel_a_source.csv"
PANEL_B_SOURCE = FIG_SOURCE_DIR / "glacier_lake_anomaly_figure01_panel_b_source.csv"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "sky": "#56B4E9",
    "black": "#000000",
    "gray": "#7A7A7A",
    "light_gray": "#D9D9D9",
}

STRATUM_COLORS = {
    "core_negative": OKABE_ITO["blue"],
    "supporting_negative": OKABE_ITO["green"],
    "weak_or_opposite": OKABE_ITO["orange"],
}


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


def load_panel_a() -> pd.DataFrame:
    panel = pd.read_csv(ANALYSIS_DIR / "glambie_coupling_v1_analysis_ready_panel.csv")
    data = panel.loc[
        panel["shared_sample_flag"] == True,
        [
            "region_key",
            "region_name",
            "year",
            "glambie_combined_mwe_anomaly",
            "total_annual_area_km2_anomaly_z",
        ],
    ].copy()
    data = data.rename(
        columns={
            "glambie_combined_mwe_anomaly": "x_glacier_mwe_anomaly",
            "total_annual_area_km2_anomaly_z": "y_lake_area_anomaly_z",
        }
    )
    return data


def fisher_ci(r: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 3:
        return np.nan, np.nan
    r = min(max(r, -0.999999), 0.999999)
    zr = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo = np.tanh(zr - z * se)
    hi = np.tanh(zr + z * se)
    return float(lo), float(hi)


def load_panel_b() -> pd.DataFrame:
    corr = pd.read_csv(ANALYSIS_DIR / "glambie_coupling_v1_regionwise_correlations.csv")
    corr = corr.loc[
        (corr["response"] == "total_annual_area_km2_anomaly_z")
        & (corr["predictor"] == "glambie_combined_mwe_anomaly")
        & (corr["test_name"] == "pearson")
    ].copy()
    strata = pd.read_csv(
        ROOT / "data" / "processed" / "analysis" / "glambie_anomaly_robustness_v1" / "glambie_anomaly_robustness_v1_stratified_regions.csv"
    )[["region_key", "stratum"]].copy()
    panel_b = corr.merge(strata, on="region_key", how="left")
    panel_b["label"] = panel_b["region_key"].map(short_label)
    ci = panel_b.apply(lambda row: fisher_ci(float(row["estimate"]), int(row["n"])), axis=1, result_type="expand")
    panel_b[["ci_low", "ci_high"]] = ci
    panel_b["significant"] = panel_b["p_value"] < 0.05
    panel_b = panel_b.sort_values(["estimate", "region_key"]).reset_index(drop=True)
    return panel_b


def load_overall_stats() -> dict[str, float]:
    tests = pd.read_csv(ANALYSIS_DIR / "glambie_coupling_v1_panel_tests.csv")
    row = tests.loc[
        (tests["response"] == "total_annual_area_km2_anomaly_z")
        & (tests["predictor"] == "glambie_combined_mwe_anomaly")
        & (tests["test_name"] == "pearson")
    ].iloc[0]
    ols_row = tests.loc[
        (tests["response"] == "total_annual_area_km2_anomaly_z")
        & (tests["predictor"] == "glambie_combined_mwe_anomaly")
        & (tests["test_name"] == "ols")
    ].iloc[0]
    return {
        "n_rows": int(row["n_rows"]),
        "n_regions": int(row["n_regions"]),
        "year_min": int(row["year_min"]),
        "year_max": int(row["year_max"]),
        "pearson_r": float(row["estimate"]),
        "p_value": float(row["p_value"]),
        "slope": float(ols_row["slope"]),
        "intercept": float(ols_row["intercept"]),
        "r2": float(ols_row["r2"]),
    }


def p_label(p: float) -> str:
    if p < 0.001:
        return f"P = {p:.1e}"
    return f"P = {p:.3f}"


def regression_band(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return y_hat, y_hat - delta, y_hat + delta


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


def build_figure(panel_a: pd.DataFrame, panel_b: pd.DataFrame, stats: dict[str, float]) -> dict[str, str]:
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.6),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.28},
    )

    x = panel_a["x_glacier_mwe_anomaly"].to_numpy(dtype=float)
    y = panel_a["y_lake_area_anomaly_z"].to_numpy(dtype=float)
    x_grid = np.linspace(x.min() - 0.08, x.max() + 0.08, 250)
    y_hat, y_lo, y_hi = regression_band(x, y, x_grid)

    ax1.scatter(
        x,
        y,
        s=13,
        color=OKABE_ITO["gray"],
        alpha=0.45,
        linewidth=0,
        rasterized=True,
        zorder=2,
    )
    ax1.fill_between(x_grid, y_lo, y_hi, color=OKABE_ITO["sky"], alpha=0.25, zorder=1)
    ax1.plot(x_grid, y_hat, color=OKABE_ITO["black"], linewidth=1.2, zorder=3)
    ax1.axhline(0, color=OKABE_ITO["light_gray"], linewidth=0.8, zorder=0)
    ax1.axvline(0, color=OKABE_ITO["light_gray"], linewidth=0.8, zorder=0)
    ax1.set_xlabel("Glacier mass-balance anomaly (m w.e.)")
    ax1.set_ylabel("Lake area anomaly (z score)")
    ax1.set_title("Pooled 2000–2023 anomalies", pad=4)
    ax1.text(
        0.03,
        0.97,
        f"n = {stats['n_rows']}, {stats['n_regions']} regions\nr = {stats['pearson_r']:.3f}, {p_label(stats['p_value'])}\nR$^2$ = {stats['r2']:.3f}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=6,
        color=OKABE_ITO["black"],
    )
    add_panel_label(ax1, "a")

    y_pos = np.arange(len(panel_b))
    ax2.axvline(0, color=OKABE_ITO["light_gray"], linewidth=0.8, zorder=0)
    for idx, row in panel_b.iterrows():
        color = STRATUM_COLORS.get(row["stratum"], OKABE_ITO["gray"])
        ax2.hlines(idx, row["ci_low"], row["ci_high"], color=color, linewidth=1.1, zorder=1)
        face = color if bool(row["significant"]) else "white"
        ax2.scatter(
            row["estimate"],
            idx,
            s=24,
            facecolor=face,
            edgecolor=color,
            linewidth=1.0,
            zorder=2,
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(panel_b["label"])
    ax2.set_xlabel("Region-level Pearson r")
    ax2.set_title("Regional heterogeneity", pad=4)
    ax2.set_xlim(-0.85, 0.35)
    ax2.set_ylim(-0.7, len(panel_b) - 0.3)
    add_panel_label(ax2, "b")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=STRATUM_COLORS["core_negative"], markeredgecolor=STRATUM_COLORS["core_negative"], markersize=4.5, label="Core negative"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=STRATUM_COLORS["supporting_negative"], markeredgecolor=STRATUM_COLORS["supporting_negative"], markersize=4.5, label="Supporting negative"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=STRATUM_COLORS["weak_or_opposite"], markeredgecolor=STRATUM_COLORS["weak_or_opposite"], markersize=4.5, label="Weak or opposite"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=OKABE_ITO["black"], markeredgecolor=OKABE_ITO["black"], markersize=4.5, label="P < 0.05"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=OKABE_ITO["black"], markersize=4.5, label="P ≥ 0.05"),
    ]
    ax2.legend(handles=legend_handles, frameon=False, loc="lower right", handletextpad=0.6, borderpad=0.2)

    outputs = save_figure(fig, "glacier_lake_anomaly_figure01")
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "panel_a_source": str(PANEL_A_SOURCE),
                "panel_b_source": str(PANEL_B_SOURCE),
                "analysis_inputs": [
                    str(ANALYSIS_DIR / "glambie_coupling_v1_analysis_ready_panel.csv"),
                    str(ANALYSIS_DIR / "glambie_coupling_v1_panel_tests.csv"),
                    str(ANALYSIS_DIR / "glambie_coupling_v1_regionwise_correlations.csv"),
                    str(ROOT / "data" / "processed" / "analysis" / "glambie_anomaly_robustness_v1" / "glambie_anomaly_robustness_v1_stratified_regions.csv"),
                ],
                "figure_style": "Nature-style double-column figure with pooled anomaly scatter and region-level forest plot",
                "nature_specs_applied": {
                    "width_mm": 183,
                    "font_family": "Arial",
                    "body_text_pt": "5-7",
                    "panel_label_pt": 8,
                    "vector_text": True,
                    "pdf_fonttype": 42,
                    "color_palette": "Okabe-Ito"
                },
                "overall_stats": stats,
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
    stats = load_overall_stats()
    PANEL_A_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    panel_a.to_csv(PANEL_A_SOURCE, index=False)
    panel_b.to_csv(PANEL_B_SOURCE, index=False)
    outputs = build_figure(panel_a, panel_b, stats)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
