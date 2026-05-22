from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"E:\Glacier")
YEAR_SUMMARY = ROOT / "data" / "processed" / "annual_area_skeleton" / "central_asia_key_years_or2024_skeleton_year_summary.csv"
LAKE_YEAR = ROOT / "data" / "processed" / "annual_area_skeleton" / "central_asia_key_years_or2024_skeleton_lake_year.csv"
OUT_DIR = ROOT / "figures" / "results" / "central_asia_or2024"


def setup_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_total_area_and_usable(df: pd.DataFrame, out_dir: Path):
    fig, ax1 = plt.subplots(figsize=(7.2, 4.0))
    ax1.plot(df["year"], df["total_annual_area_km2"], marker="o", color="#2E6E9E", label="Total area (km²)")
    ax1.set_ylabel("Total area (km²)")
    ax1.set_xlabel("Year")

    ax2 = ax1.twinx()
    ax2.plot(df["year"], df["usable_share"], marker="s", color="#C27B0E", label="Usable share")
    ax2.set_ylabel("Usable share")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    ax1.set_title("Central Asia: Total Area and Usable Share (OR 2024)")

    save_fig(out_dir / "central_asia_total_area_usable_share.png")


def plot_ratio_metrics(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(df["year"], df["median_ratio"], marker="o", color="#3A7D44", label="Median ratio")
    ax.plot(df["year"], df["mean_ratio"], marker="s", color="#8A2BE2", label="Mean ratio")
    ax.set_xlabel("Year")
    ax.set_ylabel("Area to baseline ratio")
    ax.set_title("Central Asia: Ratio Metrics (OR 2024)")
    ax.legend()
    save_fig(out_dir / "central_asia_ratio_metrics.png")


def plot_zero_ratio(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(df["year"], df["zero_ratio_share"], marker="o", color="#A3382C")
    ax.set_xlabel("Year")
    ax.set_ylabel("Zero-ratio share")
    ax.set_title("Central Asia: Zero-Ratio Share (OR 2024)")
    save_fig(out_dir / "central_asia_zero_ratio_share.png")


def plot_class_total_area(lake_year: pd.DataFrame, out_dir: Path):
    grouped = (
        lake_year.groupby(["year", "harmonized_class"])
        .agg(total_area_km2=("annual_max_area_km2", "sum"))
        .reset_index()
    )
    classes = grouped["harmonized_class"].dropna().unique().tolist()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for cls in classes:
        sub = grouped[grouped["harmonized_class"] == cls]
        ax.plot(sub["year"], sub["total_area_km2"], marker="o", label=cls)
    ax.set_xlabel("Year")
    ax.set_ylabel("Total area (km²)")
    ax.set_title("Central Asia: Total Area by Lake Class (OR 2024)")
    ax.legend()
    save_fig(out_dir / "central_asia_total_area_by_class.png")


def plot_class_median_ratio(lake_year: pd.DataFrame, out_dir: Path):
    grouped = (
        lake_year.groupby(["year", "harmonized_class"])
        .agg(median_ratio=("annual_area_to_baseline_ratio", "median"))
        .reset_index()
    )
    classes = grouped["harmonized_class"].dropna().unique().tolist()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for cls in classes:
        sub = grouped[grouped["harmonized_class"] == cls]
        ax.plot(sub["year"], sub["median_ratio"], marker="o", label=cls)
    ax.set_xlabel("Year")
    ax.set_ylabel("Median ratio")
    ax.set_title("Central Asia: Median Ratio by Lake Class (OR 2024)")
    ax.legend()
    save_fig(out_dir / "central_asia_median_ratio_by_class.png")


def main():
    setup_style()
    out_dir = OUT_DIR

    year_df = pd.read_csv(YEAR_SUMMARY)
    lake_year = pd.read_csv(LAKE_YEAR)
    for col in ["annual_max_area_km2", "annual_area_to_baseline_ratio"]:
        if col in lake_year.columns:
            lake_year[col] = pd.to_numeric(lake_year[col], errors="coerce")

    plot_total_area_and_usable(year_df, out_dir)
    plot_ratio_metrics(year_df, out_dir)
    plot_zero_ratio(year_df, out_dir)
    plot_class_total_area(lake_year, out_dir)
    plot_class_median_ratio(lake_year, out_dir)

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
