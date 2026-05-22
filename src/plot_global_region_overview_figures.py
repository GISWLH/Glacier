from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import pandas as pd


ROOT = Path(r"E:\Glacier")
FIG_SOURCE_DIR = ROOT / "manuscript" / "figures_for_paper"
OUT_DIR = FIG_SOURCE_DIR / "rendered" / "global_region_overview"
SOURCE = FIG_SOURCE_DIR / "global_region_overview_figure_source.csv"
MANIFEST_PATH = OUT_DIR / "global_region_overview_figure_manifest.json"

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
    "gray": "#7A7A7A",
    "light_gray": "#D9D9D9",
}

PRIORITY_COLORS = {
    "high": OKABE_ITO["vermillion"],
    "medium": OKABE_ITO["orange"],
    "low": OKABE_ITO["gray"],
}


def pct_formatter(decimals: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda x, _: f"{x * 100:.{decimals}f}%")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.grid": False,
        }
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


def build_figure(df: pd.DataFrame) -> dict[str, str]:
    order = df.sort_values("total_area_change_km2_2020_2024", ascending=True).reset_index(drop=True)
    order["label"] = order["region_key"].map(short_label)
    y = range(len(order))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.2, 4.6),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.18},
    )

    ax1.axvline(0, color=OKABE_ITO["black"], linewidth=0.8, zorder=1)
    for idx, row in order.iterrows():
        color = PRIORITY_COLORS[row["priority_level"]]
        ax1.hlines(
            y=idx,
            xmin=0,
            xmax=row["total_area_change_km2_2020_2024"],
            color=color,
            linewidth=1.2,
            zorder=2,
        )
        ax1.scatter(
            row["total_area_change_km2_2020_2024"],
            idx,
            s=28,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax1.set_yticks(list(y))
    ax1.set_yticklabels(order["label"])
    ax1.set_xlabel("Area change, 2020-2024 (km$^2$)")
    ax1.set_title("a, Regional area change")

    core_labels = {"Greenland", "Arctic Can. S", "S. Andes", "Central Asia", "Alaska"}
    for idx, row in order.iterrows():
        if row["label"] in core_labels:
            ax1.text(
                row["total_area_change_km2_2020_2024"],
                idx + 0.18,
                f"{row['total_area_change_km2_2020_2024']:.0f}",
                fontsize=6.2,
                color=OKABE_ITO["black"],
                ha="center",
                va="bottom",
            )

    ax2.axvline(0, color=OKABE_ITO["black"], linewidth=0.8, zorder=1)
    for idx, row in order.iterrows():
        color = PRIORITY_COLORS[row["priority_level"]]
        size = 18 + min(row["suspicious_lake_count_2020_2024"], 250) * 0.22
        ax2.scatter(
            row["mean_ratio_change_2020_2024"],
            idx,
            s=size,
            color=color,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

    ax2.set_yticks(list(y))
    ax2.set_yticklabels([])
    ax2.set_xlabel("Mean ratio change, 2020-2024")
    ax2.xaxis.set_major_formatter(pct_formatter(0))
    ax2.set_title("b, Normalized change and anomaly weight")

    legend_priority = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PRIORITY_COLORS[key], markeredgecolor="none", markersize=5, label=key.capitalize())
        for key in ["high", "medium", "low"]
    ]
    legend_size = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=OKABE_ITO["gray"], markeredgecolor="none", markersize=size, label=label)
        for size, label in [(4, "Low suspicious count"), (7, "Moderate"), (10, "High")]
    ]
    leg1 = ax2.legend(handles=legend_priority, title="Priority", frameon=False, loc="lower right")
    ax2.add_artist(leg1)
    ax2.legend(handles=legend_size, title="Bubble size", frameon=False, loc="upper left")

    manifest = {
        "source": str(SOURCE),
        "figure_style": "Nature-inspired compact two-panel overview; no background grid; color-blind-safe palette; position-first encoding",
        "sorting": "ascending total_area_change_km2_2020_2024",
        "priority_color_map": PRIORITY_COLORS,
    }
    outputs = save_figure(fig, "global_region_overview_figure01")
    MANIFEST_PATH.write_text(json.dumps({**manifest, **outputs}, indent=2), encoding="utf-8")
    return outputs


def main() -> None:
    setup_style()
    df = pd.read_csv(SOURCE)
    outputs = build_figure(df)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
