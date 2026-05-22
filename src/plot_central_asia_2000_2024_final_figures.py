from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


ROOT = Path(r"E:\Glacier")
TAG = "central_asia_2000_2024_final"
FIG_SOURCE_DIR = ROOT / "manuscript" / "figures_for_paper"
OUT_DIR = FIG_SOURCE_DIR / "rendered" / TAG
YEAR_SOURCE = FIG_SOURCE_DIR / f"{TAG}_figure_year_source.csv"
ANOMALY_SOURCE = FIG_SOURCE_DIR / f"{TAG}_figure_anomaly_source.csv"
CLASS_SOURCE = FIG_SOURCE_DIR / f"{TAG}_figure_class_source.csv"
MANIFEST_PATH = OUT_DIR / f"{TAG}_figure_manifest.json"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
    "gray": "#6E6E6E",
}

CLASS_ORDER = ["proglacial_detached", "proglacial_contacted", "supraglacial"]

CLASS_COLORS = {
    "proglacial_detached": OKABE_ITO["blue"],
    "proglacial_contacted": OKABE_ITO["orange"],
    "supraglacial": OKABE_ITO["sky"],
}

CLASS_MARKERS = {
    "proglacial_detached": "o",
    "proglacial_contacted": "o",
    "supraglacial": "^",
}


def pct_formatter(decimals: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda x, _: f"{x * 100:.{decimals}f}%")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "grid.linewidth": 0.6,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


def plot_year_overview(year_df: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.8), sharex=True, height_ratios=[1.15, 1.0])
    ax1, ax2 = axes
    years = year_df["year"]

    ax1.plot(years, year_df["total_annual_area_km2"], color=OKABE_ITO["blue"], marker="o", linewidth=1.8, label="Total annual area")
    ax1.plot(
        years,
        year_df["usable_total_annual_area_km2"],
        color=OKABE_ITO["green"],
        marker="s",
        linewidth=1.6,
        label="Usable annual area",
    )
    ax1.set_ylabel("Area (km$^2$)")
    ax1.set_title("A. Central Asia annual area trajectory")
    ax1.legend(frameon=False, ncol=2, loc="upper left")

    max_row = year_df.loc[year_df["total_annual_area_km2"].idxmax()]
    min_row = year_df.loc[year_df["total_annual_area_km2"].idxmin()]
    for row, yoff in [(max_row, 10), (min_row, -24)]:
        ax1.annotate(
            f"{int(row['year'])}\n{row['total_annual_area_km2']:.1f}",
            xy=(row["year"], row["total_annual_area_km2"]),
            xytext=(0, yoff),
            textcoords="offset points",
            ha="center",
            color=OKABE_ITO["blue"],
        )

    ax2.plot(years, year_df["usable_share"], color=OKABE_ITO["orange"], marker="o", linewidth=1.6, label="Usable share")
    ax2.plot(
        years,
        year_df["usable_median_area_ratio"],
        color=OKABE_ITO["purple"],
        marker="D",
        linewidth=1.6,
        label="Median area ratio",
    )
    ax2.set_ylabel("Share / ratio")
    ax2.set_xlabel("Year")
    ax2.yaxis.set_major_formatter(pct_formatter(0))
    ax2.set_title("B. Coverage quality and normalized area signal")
    ax2.legend(frameon=False, ncol=2, loc="lower left")

    return save_figure(fig, f"{TAG}_figure01_year_overview")


def plot_trend_anomalies(anomaly_df: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.8), sharex=True, height_ratios=[1.1, 1.0])
    ax1, ax2 = axes
    years = anomaly_df["year"]

    ax1.plot(years, anomaly_df["total_annual_area_km2"], color=OKABE_ITO["blue"], marker="o", linewidth=1.8, label="Observed")
    ax1.plot(
        years,
        anomaly_df["total_annual_area_km2_trend"],
        color=OKABE_ITO["gray"],
        linestyle="--",
        linewidth=1.4,
        label="Full-period trend",
    )
    ax1.set_ylabel("Area (km$^2$)")
    ax1.set_title("A. Total area and full-period trend")
    ax1.legend(frameon=False, loc="upper left")

    strongest_area_peak = anomaly_df.loc[anomaly_df["total_annual_area_km2_anomaly"].idxmax()]
    strongest_area_drop = anomaly_df.loc[anomaly_df["total_annual_area_km2_anomaly"].idxmin()]
    for row, color, va in [
        (strongest_area_peak, OKABE_ITO["green"], "bottom"),
        (strongest_area_drop, OKABE_ITO["vermillion"], "top"),
    ]:
        ax1.scatter(row["year"], row["total_annual_area_km2"], color=color, s=42, zorder=4)
        ax1.annotate(
            f"{int(row['year'])}\n{row['total_annual_area_km2_anomaly']:+.1f}",
            xy=(row["year"], row["total_annual_area_km2"]),
            xytext=(0, 10 if va == "bottom" else -14),
            textcoords="offset points",
            ha="center",
            va=va,
            color=color,
        )

    ratio_colors = [
        OKABE_ITO["green"] if value >= 0 else OKABE_ITO["vermillion"] for value in anomaly_df["median_ratio_anomaly"]
    ]
    ax2.bar(years, anomaly_df["median_ratio_anomaly"], color=ratio_colors, width=0.75, alpha=0.65, label="Median ratio anomaly")
    ax2.plot(years, anomaly_df["median_ratio"], color=OKABE_ITO["purple"], marker="o", linewidth=1.5, label="Observed median ratio")
    ax2.plot(
        years,
        anomaly_df["median_ratio_trend"],
        color=OKABE_ITO["gray"],
        linestyle="--",
        linewidth=1.2,
        label="Trend baseline",
    )
    ax2.axhline(0, color=OKABE_ITO["black"], linewidth=0.8)
    ax2.yaxis.set_major_formatter(pct_formatter(0))
    ax2.set_ylabel("Ratio / anomaly")
    ax2.set_xlabel("Year")
    ax2.set_title("B. Median ratio anomaly structure")
    ax2.legend(frameon=False, ncol=3, loc="upper left")

    strongest_ratio_peak = anomaly_df.loc[anomaly_df["median_ratio_anomaly"].idxmax()]
    strongest_ratio_drop = anomaly_df.loc[anomaly_df["median_ratio_anomaly"].idxmin()]
    for row, color in [
        (strongest_ratio_peak, OKABE_ITO["green"]),
        (strongest_ratio_drop, OKABE_ITO["vermillion"]),
    ]:
        ax2.annotate(
            f"{int(row['year'])}\n{row['median_ratio_anomaly']:+.3f}",
            xy=(row["year"], row["median_ratio_anomaly"]),
            xytext=(0, 10 if row["median_ratio_anomaly"] >= 0 else -14),
            textcoords="offset points",
            ha="center",
            va="bottom" if row["median_ratio_anomaly"] >= 0 else "top",
            color=color,
        )

    return save_figure(fig, f"{TAG}_figure02_trend_anomalies")


def plot_class_trajectories(class_df: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.8), sharex=True, height_ratios=[1.1, 1.0])
    ax1, ax2 = axes

    present_classes = [
        class_name
        for class_name in CLASS_ORDER
        if class_name in class_df["harmonized_class"].dropna().astype(str).unique().tolist()
    ]

    for class_name in present_classes:
        sub = class_df[class_df["harmonized_class"] == class_name].sort_values("year")
        label = class_name.replace("_", " ")
        color = CLASS_COLORS[class_name]
        marker = CLASS_MARKERS[class_name]
        ax1.plot(sub["year"], sub["total_area_km2"], color=color, marker=marker, linewidth=1.8, label=label)
        ax2.plot(sub["year"], sub["median_ratio"], color=color, marker=marker, linewidth=1.8, label=label)

    ax1.set_ylabel("Area (km$^2$)")
    ax1.set_title("A. Area trajectory by lake class")
    ax1.legend(frameon=False, ncol=min(len(present_classes), 3), loc="upper left")

    ax2.yaxis.set_major_formatter(pct_formatter(0))
    ax2.set_ylabel("Median ratio")
    ax2.set_xlabel("Year")
    ax2.set_title("B. Normalized area trajectory by lake class")
    ax2.legend(frameon=False, ncol=min(len(present_classes), 3), loc="lower left")

    contacted = class_df[class_df["harmonized_class"] == "proglacial_contacted"].sort_values("year")
    if not contacted.empty:
        peak_contacted = contacted.loc[contacted["median_ratio"].idxmax()]
        ax2.annotate(
            f"peak {int(peak_contacted['year'])}\n{peak_contacted['median_ratio']:.3f}",
            xy=(peak_contacted["year"], peak_contacted["median_ratio"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=CLASS_COLORS["proglacial_contacted"],
        )

    return save_figure(fig, f"{TAG}_figure03_class_trajectories")


def build_manifest(outputs: dict[str, dict[str, str]]) -> None:
    payload = {
        "tag": TAG,
        "source_year_csv": str(YEAR_SOURCE),
        "source_anomaly_csv": str(ANOMALY_SOURCE),
        "source_class_csv": str(CLASS_SOURCE),
        "figures": outputs,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    setup_style()
    year_df = pd.read_csv(YEAR_SOURCE).sort_values("year")
    anomaly_df = pd.read_csv(ANOMALY_SOURCE).sort_values("year")
    class_df = pd.read_csv(CLASS_SOURCE).sort_values(["harmonized_class", "year"])

    outputs = {
        "figure01_year_overview": {
            "title": "Central Asia annual area and quality overview",
            **plot_year_overview(year_df),
        },
        "figure02_trend_anomalies": {
            "title": "Central Asia trend and anomaly diagnostics",
            **plot_trend_anomalies(anomaly_df),
        },
        "figure03_class_trajectories": {
            "title": "Central Asia class-specific trajectories",
            **plot_class_trajectories(class_df),
        },
    }
    build_manifest(outputs)
    print(json.dumps({"output_dir": str(OUT_DIR), "manifest": str(MANIFEST_PATH), "figures": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
