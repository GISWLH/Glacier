from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


ROOT = Path(r"E:\Glacier")
FIG_SOURCE_DIR = ROOT / "manuscript" / "figures_for_paper"
OUT_DIR = FIG_SOURCE_DIR / "rendered" / "greenland_key_years_final"
YEAR_SOURCE = FIG_SOURCE_DIR / "greenland_key_years_final_figure_year_source.csv"
STAGE_SOURCE = FIG_SOURCE_DIR / "greenland_key_years_final_figure_stage_source.csv"
CLASS_SOURCE = FIG_SOURCE_DIR / "greenland_key_years_final_figure_stage_class_source.csv"
MANIFEST_PATH = OUT_DIR / "greenland_key_years_final_figure_manifest.json"

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
    "proglacial_contacted": "s",
    "supraglacial": "^",
}


def pct_formatter(decimals: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda x, _: f"{x * 100:.{decimals}f}%")


def signed_number(value: float, decimals: int = 0, suffix: str = "") -> str:
    return f"{value:+.{decimals}f}{suffix}"


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
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True, height_ratios=[1.15, 1.0])
    ax1, ax2 = axes

    years = year_df["year"]

    ax1.plot(
        years,
        year_df["total_annual_area_km2"],
        color=OKABE_ITO["blue"],
        marker="o",
        linewidth=2.0,
        label="Total annual area",
    )
    ax1.plot(
        years,
        year_df["usable_total_annual_area_km2"],
        color=OKABE_ITO["green"],
        marker="s",
        linewidth=1.8,
        label="Usable annual area",
    )
    ax1.set_ylabel("Area (km$^2$)")
    ax1.set_title("A. Greenland Periphery key-year area trajectory")
    ax1.legend(frameon=False, ncol=2, loc="upper right")

    for year in [2010, 2015, 2024]:
        row = year_df.loc[year_df["year"] == year].iloc[0]
        ax1.annotate(
            f"{year}\n{row['total_annual_area_km2']:.0f}",
            xy=(year, row["total_annual_area_km2"]),
            xytext=(0, 10 if year != 2015 else -28),
            textcoords="offset points",
            ha="center",
            color=OKABE_ITO["blue"],
        )

    ax2.plot(
        years,
        year_df["usable_share"],
        color=OKABE_ITO["orange"],
        marker="o",
        linewidth=1.8,
        label="Usable share",
    )
    ax2.plot(
        years,
        year_df["usable_median_area_ratio"],
        color=OKABE_ITO["purple"],
        marker="D",
        linewidth=1.8,
        label="Median area ratio",
    )
    ax2.axvline(2015, color=OKABE_ITO["gray"], linestyle="--", linewidth=1.0, alpha=0.8)
    ax2.text(2015 + 0.2, 0.99, "2015 4-month replacement", fontsize=8, color=OKABE_ITO["gray"])
    ax2.set_ylim(0.55, 1.02)
    ax2.yaxis.set_major_formatter(pct_formatter(0))
    ax2.set_ylabel("Share / ratio")
    ax2.set_xlabel("Year")
    ax2.set_title("B. Coverage quality and normalized area signal")
    ax2.legend(frameon=False, ncol=2, loc="lower left")

    return save_figure(fig, "greenland_key_years_final_figure01_year_overview")


def plot_stage_changes(stage_df: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True, height_ratios=[1.2, 0.9])
    ax1, ax2 = axes

    interval_labels = stage_df["interval"].str.replace("_", "-", regex=False)
    colors = [
        OKABE_ITO["green"] if value > 0 else OKABE_ITO["vermillion"]
        for value in stage_df["total_area_change_km2"]
    ]

    bars = ax1.bar(interval_labels, stage_df["total_area_change_km2"], color=colors, width=0.65)
    ax1.axhline(0, color=OKABE_ITO["black"], linewidth=0.9)
    ax1.set_ylabel("Area change (km$^2$)")
    ax1.set_title("A. Interval-scale area change")

    for bar, value in zip(bars, stage_df["total_area_change_km2"]):
        offset = 40 if value >= 0 else -70
        ax1.annotate(
            signed_number(value, 0),
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            color=OKABE_ITO["black"],
        )

    ax2.axhline(0, color=OKABE_ITO["black"], linewidth=0.9)
    ax2.plot(
        interval_labels,
        stage_df["median_ratio_change"],
        color=OKABE_ITO["purple"],
        marker="o",
        linewidth=1.8,
    )
    ax2.scatter(interval_labels, stage_df["median_ratio_change"], color=colors, s=38, zorder=3)
    ax2.yaxis.set_major_formatter(pct_formatter(0))
    ax2.set_ylabel("Median ratio change")
    ax2.set_xlabel("Interval")
    ax2.set_title("B. Interval-scale normalized area change")

    for x_value, ratio, interpretation in zip(
        interval_labels,
        stage_df["median_ratio_change"],
        stage_df["interpretation"],
    ):
        ax2.annotate(
            interpretation,
            xy=(x_value, ratio),
            xytext=(0, 10 if ratio >= 0 else -14),
            textcoords="offset points",
            ha="center",
            va="bottom" if ratio >= 0 else "top",
            fontsize=7,
            color=OKABE_ITO["gray"],
        )

    return save_figure(fig, "greenland_key_years_final_figure02_stage_changes")


def plot_class_contributions(class_df: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True, height_ratios=[1.2, 1.0])
    ax1, ax2 = axes

    intervals = ["2000_2005", "2005_2010", "2010_2015", "2015_2020", "2020_2024"]
    interval_labels = [interval.replace("_", "-") for interval in intervals]
    pivot_area = (
        class_df.pivot(index="interval", columns="harmonized_class", values="area_change_km2")
        .reindex(intervals)
        .reindex(columns=CLASS_ORDER, fill_value=0.0)
        .fillna(0.0)
    )
    pivot_ratio = (
        class_df.pivot(index="interval", columns="harmonized_class", values="median_ratio_change")
        .reindex(intervals)
        .reindex(columns=CLASS_ORDER, fill_value=0.0)
        .fillna(0.0)
    )
    present_classes = [
        class_name
        for class_name in CLASS_ORDER
        if class_name in class_df["harmonized_class"].dropna().astype(str).unique().tolist()
    ]

    bottoms = [0.0] * len(intervals)
    for class_name in present_classes:
        values = pivot_area[class_name].tolist()
        ax1.bar(
            interval_labels,
            values,
            bottom=bottoms,
            color=CLASS_COLORS[class_name],
            width=0.65,
            label=class_name.replace("_", " "),
        )
        bottoms = [base + value for base, value in zip(bottoms, values)]

    ax1.axhline(0, color=OKABE_ITO["black"], linewidth=0.9)
    ax1.set_ylabel("Area change (km$^2$)")
    ax1.set_title("A. Class contribution to interval area change")
    ax1.legend(frameon=False, ncol=min(len(present_classes), 3), loc="upper right")

    totals = class_df.groupby("interval", as_index=True)["interval_total_area_change_km2"].first().reindex(intervals)
    for x_value, total in zip(interval_labels, totals):
        ax1.annotate(
            signed_number(total, 0),
            xy=(x_value, total),
            xytext=(0, 10 if total >= 0 else -16),
            textcoords="offset points",
            ha="center",
            va="bottom" if total >= 0 else "top",
            fontsize=8,
            color=OKABE_ITO["black"],
        )

    x_positions = list(range(len(intervals)))
    offsets = {
        "proglacial_detached": -0.18,
        "proglacial_contacted": 0.0,
        "supraglacial": 0.18,
    }
    for class_name in present_classes:
        y_values = pivot_ratio[class_name].tolist()
        x_values = [x + offsets[class_name] for x in x_positions]
        ax2.plot(
            x_values,
            y_values,
            color=CLASS_COLORS[class_name],
            marker=CLASS_MARKERS[class_name],
            linewidth=1.6,
            label=class_name.replace("_", " "),
        )

    ax2.axhline(0, color=OKABE_ITO["black"], linewidth=0.9)
    ax2.set_xticks(x_positions, interval_labels)
    ax2.yaxis.set_major_formatter(pct_formatter(0))
    ax2.set_ylabel("Median ratio change")
    ax2.set_xlabel("Interval")
    ax2.set_title("B. Class-specific normalized area change")
    ax2.legend(frameon=False, ncol=min(len(present_classes), 3), loc="lower left")

    return save_figure(fig, "greenland_key_years_final_figure03_class_contributions")


def build_manifest(outputs: dict[str, dict[str, str]]) -> None:
    payload = {
        "source_year_csv": str(YEAR_SOURCE),
        "source_stage_csv": str(STAGE_SOURCE),
        "source_class_csv": str(CLASS_SOURCE),
        "figures": outputs,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    setup_style()
    year_df = pd.read_csv(YEAR_SOURCE).sort_values("year")
    stage_df = pd.read_csv(STAGE_SOURCE).sort_values("year_start")
    class_df = pd.read_csv(CLASS_SOURCE).sort_values(["interval", "harmonized_class"])

    outputs = {
        "figure01_year_overview": {
            "title": "Greenland Periphery key-year overview",
            **plot_year_overview(year_df),
        },
        "figure02_stage_changes": {
            "title": "Greenland Periphery interval-scale changes",
            **plot_stage_changes(stage_df),
        },
        "figure03_class_contributions": {
            "title": "Greenland Periphery class-specific contributions",
            **plot_class_contributions(class_df),
        },
    }
    build_manifest(outputs)
    print(json.dumps({"output_dir": str(OUT_DIR), "manifest": str(MANIFEST_PATH), "figures": outputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
