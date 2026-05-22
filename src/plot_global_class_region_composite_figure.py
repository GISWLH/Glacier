from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd


ROOT = Path(r"E:\Glacier")
FIG_SOURCE_DIR = ROOT / "manuscript" / "figures_for_paper"
OUT_DIR = FIG_SOURCE_DIR / "rendered" / "global_class_region_composite"
MANIFEST_PATH = OUT_DIR / "global_class_region_composite_figure01_manifest.json"
PANEL_SOURCE = FIG_SOURCE_DIR / "global_class_region_composite_figure01_panel_source.csv"
MAP_SOURCE = FIG_SOURCE_DIR / "global_class_region_composite_figure01_map_source.csv"

CLASS_PANEL_CSV = ROOT / "data" / "processed" / "formal_class_region_year_panel" / "formal_class_region_year_panel_v1.csv"
COUPLING_PANEL_CSV = (
    ROOT / "data" / "processed" / "analysis" / "glambie_class_coupling_v1" / "glambie_class_coupling_v1_analysis_ready_panel.csv"
)
REGION_MANIFEST_CSV = ROOT / "data" / "prepared" / "core_region_batches" / "core_region_manifest.csv"
RGI_REGION_SHP = ROOT / "data" / "raw" / "RGI2000" / "RGI2000-v7.0-regions" / "RGI2000-v7.0-o1regions.shp"
LAND_SHP = ROOT / "data" / "raw" / "global_map" / "40e47-main" / "global_map" / "110m_physical" / "ne_110m_land.shp"
COAST_SHP = ROOT / "data" / "raw" / "global_map" / "40e47-main" / "global_map" / "110m_physical" / "ne_110m_coastline.shp"
ANALYSIS_MASTER_GPKG = ROOT / "data" / "processed" / "analysis_lake_master.gpkg"

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
    "link_gray": "#9A9A9A",
    "region_gray": "#B8B8B8",
}

CLASS_ORDER = ["proglacial_detached", "proglacial_contacted", "supraglacial"]
CLASS_COLORS = {
    "proglacial_detached": "#1F77B4",
    "proglacial_contacted": "#F1C40F",
    "supraglacial": "#D62728",
}
CLASS_MARKERS = {
    "proglacial_detached": "o",
    "proglacial_contacted": "s",
    "supraglacial": "^",
}

SHORT_LABELS = {
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

MAP_AX_POS = [0.16, 0.20, 0.68, 0.56]

CALLOUT_POSITIONS = {
    "alaska": (0.01, 0.69, 0.12, 0.12),
    "western_canada_us": (0.01, 0.55, 0.12, 0.12),
    "arctic_canada_north": (0.12, 0.86, 0.12, 0.12),
    "arctic_canada_south": (0.14, 0.72, 0.12, 0.12),
    "greenland_periphery": (0.29, 0.86, 0.12, 0.12),
    "iceland": (0.42, 0.86, 0.12, 0.12),
    "scandinavia": (0.55, 0.86, 0.12, 0.12),
    "russian_arctic": (0.68, 0.86, 0.12, 0.12),
    "north_asia": (0.87, 0.72, 0.12, 0.12),
    "central_europe": (0.76, 0.86, 0.12, 0.12),
    "caucasus_middle_east": (0.87, 0.56, 0.12, 0.12),
    "central_asia": (0.87, 0.42, 0.12, 0.12),
    "south_asia_west": (0.87, 0.28, 0.12, 0.12),
    "south_asia_east": (0.87, 0.14, 0.12, 0.12),
    "low_latitudes": (0.15, 0.01, 0.12, 0.12),
    "southern_andes": (0.32, 0.01, 0.12, 0.12),
    "new_zealand": (0.76, 0.01, 0.12, 0.12),
}

MAP_EXTENT = (-180, 180, -62, 88)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "Arial",
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 7.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
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
    return SHORT_LABELS.get(region_key, region_key)


def compute_simple_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int] | None:
    if len(x) < 2 or np.allclose(x, x[0]):
        return None
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else np.nan
    r2 = float(r**2) if np.isfinite(r) else np.nan
    return float(slope), float(intercept), r2, int(len(x))


def load_region_geometries() -> gpd.GeoDataFrame:
    manifest = pd.read_csv(REGION_MANIFEST_CSV)
    gdf = gpd.read_file(RGI_REGION_SHP)
    gdf["o1region"] = pd.to_numeric(gdf["o1region"], errors="coerce")
    gdf = gdf[gdf["o1region"].isin(manifest["glambie_region_id"])].copy()
    gdf = gdf[["o1region", "full_name", "geometry"]].dissolve(by="o1region", as_index=False)
    gdf = gdf.merge(
        manifest[["glambie_region_id", "glambie_region_key", "rgi_region_name"]],
        left_on="o1region",
        right_on="glambie_region_id",
        how="left",
    )
    return gdf.to_crs(4326)


def load_lake_map_data(region_polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    lake_gdf = gpd.read_file(ANALYSIS_MASTER_GPKG, layer="analysis_master")
    region_keys = set(region_polygons["glambie_region_key"].dropna().astype(str))
    lake_gdf = lake_gdf[lake_gdf["glambie_region_key"].astype(str).isin(region_keys)].copy()
    lake_gdf = lake_gdf[lake_gdf["harmonized_class"].isin(CLASS_ORDER)].copy()
    lake_gdf = lake_gdf.dropna(subset=["longitude", "latitude"]).copy()
    return lake_gdf.to_crs(4326)


def compute_region_anchors(region_polygons: gpd.GeoDataFrame, lake_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    fallback = region_polygons[["o1region", "glambie_region_key", "geometry"]].copy()
    fallback["anchor"] = fallback.representative_point()
    fallback["anchor_lon"] = fallback["anchor"].x
    fallback["anchor_lat"] = fallback["anchor"].y

    anchor_rows: list[dict[str, float | str | int]] = []
    for _, region_row in fallback.iterrows():
        region_key = region_row["glambie_region_key"]
        region_lakes = lake_gdf[lake_gdf["glambie_region_key"] == region_key].copy()
        core_lakes = region_lakes[region_lakes["main_analysis_include"] == True].copy()
        candidate_lakes = core_lakes if not core_lakes.empty else region_lakes
        if not candidate_lakes.empty:
            med_lon = candidate_lakes["longitude"].median()
            med_lat = candidate_lakes["latitude"].median()
            distances = (candidate_lakes["longitude"] - med_lon) ** 2 + (candidate_lakes["latitude"] - med_lat) ** 2
            best_idx = distances.idxmin()
            best = candidate_lakes.loc[best_idx]
            anchor_rows.append(
                {
                    "o1region": int(region_row["o1region"]),
                    "glambie_region_key": str(region_key),
                    "anchor_lon": float(best["longitude"]),
                    "anchor_lat": float(best["latitude"]),
                    "anchor_source": "lake_median_nearest_point",
                }
            )
        else:
            anchor_rows.append(
                {
                    "o1region": int(region_row["o1region"]),
                    "glambie_region_key": str(region_key),
                    "anchor_lon": float(region_row["anchor_lon"]),
                    "anchor_lat": float(region_row["anchor_lat"]),
                    "anchor_source": "region_representative_point",
                }
            )
    return pd.DataFrame(anchor_rows)


def load_world_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    land = gpd.read_file(LAND_SHP).to_crs(4326)
    coast = gpd.read_file(COAST_SHP).to_crs(4326)
    return land, coast


def load_panel_data() -> pd.DataFrame:
    line_df = pd.read_csv(CLASS_PANEL_CSV)
    scatter_df = pd.read_csv(COUPLING_PANEL_CSV)

    line_df = line_df.loc[
        (line_df["year"] >= 2000) & (line_df["year"] <= 2023),
        ["region_key", "region_name", "year", "harmonized_class", "total_annual_area_km2_anomaly_z"],
    ].copy()
    scatter_df = scatter_df.loc[
        (scatter_df["year"] >= 2000)
        & (scatter_df["year"] <= 2023)
        & (scatter_df["predictor_sample_flag"] == True),
        [
            "region_key",
            "region_name",
            "year",
            "harmonized_class",
            "glambie_combined_mwe_anomaly",
            "total_annual_area_km2_anomaly_z",
        ],
    ].copy()

    panel = line_df.merge(
        scatter_df,
        on=["region_key", "region_name", "year", "harmonized_class", "total_annual_area_km2_anomaly_z"],
        how="outer",
    )
    return panel


def add_panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.02,
        1.02,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=OKABE_ITO["black"],
    )


def draw_region_panel(ax: plt.Axes, region_df: pd.DataFrame, region_key: str) -> None:
    ax.set_facecolor("white")
    ax.patch.set_alpha(0.96)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_edgecolor(OKABE_ITO["gray"])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    line_ax = ax.inset_axes([0.14, 0.56, 0.80, 0.34])
    scatter_ax = ax.inset_axes([0.14, 0.16, 0.80, 0.24])

    line_data = region_df[["year", "harmonized_class", "total_annual_area_km2_anomaly_z"]].dropna().copy()
    scatter_data = region_df[
        ["harmonized_class", "glambie_combined_mwe_anomaly", "total_annual_area_km2_anomaly_z"]
    ].dropna().copy()

    metric_lines: list[tuple[str, str]] = []
    metric_labels = {
        "proglacial_detached": "IUL",
        "proglacial_contacted": "ICL",
        "supraglacial": "SGL",
    }

    for class_name in CLASS_ORDER:
        sub_line = line_data[line_data["harmonized_class"] == class_name].sort_values("year")
        if not sub_line.empty:
            line_ax.plot(
                sub_line["year"],
                sub_line["total_annual_area_km2_anomaly_z"],
                color=CLASS_COLORS[class_name],
                marker=CLASS_MARKERS[class_name],
                markersize=1.8,
                linewidth=0.85,
            )

        sub_scatter = scatter_data[scatter_data["harmonized_class"] == class_name].copy()
        if not sub_scatter.empty:
            x = sub_scatter["glambie_combined_mwe_anomaly"].to_numpy(dtype=float)
            y = sub_scatter["total_annual_area_km2_anomaly_z"].to_numpy(dtype=float)
            scatter_ax.scatter(
                x,
                y,
                s=8,
                color=CLASS_COLORS[class_name],
                alpha=0.65,
                linewidths=0,
                rasterized=True,
            )
            fit = compute_simple_fit(x, y)
            if fit is not None:
                slope, intercept, r2, n_obs = fit
                xgrid = np.linspace(np.nanmin(x), np.nanmax(x), 40)
                scatter_ax.plot(xgrid, intercept + slope * xgrid, color=CLASS_COLORS[class_name], linewidth=0.75)
                metric_lines.append((class_name, f"{metric_labels[class_name]} R²={r2:.2f}, n={n_obs}"))

    line_ax.axhline(0, color=OKABE_ITO["light_gray"], linewidth=0.5, zorder=0)
    scatter_ax.axhline(0, color=OKABE_ITO["light_gray"], linewidth=0.5, zorder=0)
    scatter_ax.axvline(0, color=OKABE_ITO["light_gray"], linewidth=0.5, zorder=0)

    line_ax.set_xlim(2000, 2023)
    line_ax.set_xticks([2000, 2010, 2020])
    line_ax.set_yticks([])
    line_ax.tick_params(axis="x", labelsize=4.8, pad=0.8)

    scatter_ax.set_yticks([])
    scatter_ax.set_xticks([])

    if metric_lines:
        metric_text = "\n".join(text for _, text in metric_lines)
        metric_colors = [CLASS_COLORS[class_name] for class_name, _ in metric_lines]
        y0 = 0.96
        for color, (_, text) in zip(metric_colors, metric_lines):
            scatter_ax.text(0.03, y0, text, transform=scatter_ax.transAxes, ha="left", va="top", fontsize=3.8, color=color)
            y0 -= 0.16

    for sub_ax in [line_ax, scatter_ax]:
        sub_ax.spines["top"].set_visible(False)
        sub_ax.spines["right"].set_visible(False)
        sub_ax.spines["left"].set_linewidth(0.8)
        sub_ax.spines["bottom"].set_linewidth(0.8)

    ax.text(0.03, 0.96, short_label(region_key), transform=ax.transAxes, ha="left", va="top", fontsize=5.6)


def build_figure(
    region_polygons: gpd.GeoDataFrame,
    anchor_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    lake_map_gdf: gpd.GeoDataFrame,
) -> dict[str, str]:
    fig = plt.figure(figsize=(18.0, 12.0))
    gs = GridSpec(1, 1, figure=fig)
    map_ax = fig.add_subplot(gs[0, 0])
    map_ax.set_position(MAP_AX_POS)

    land, coast = load_world_layers()
    land.plot(ax=map_ax, color="#ECECEC", edgecolor="none", zorder=0)
    coast.plot(ax=map_ax, color="#A8A8A8", linewidth=0.35, zorder=1)

    for class_name in CLASS_ORDER:
        class_points = lake_map_gdf[lake_map_gdf["harmonized_class"] == class_name].copy()
        if class_points.empty:
            continue
        map_ax.scatter(
            class_points["longitude"],
            class_points["latitude"],
            s=3.5 if class_name != "supraglacial" else 2.8,
            color=CLASS_COLORS[class_name],
            alpha=0.22 if class_name != "supraglacial" else 0.18,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    map_ax.scatter(
        anchor_df["anchor_lon"],
        anchor_df["anchor_lat"],
        s=20,
        color=OKABE_ITO["black"],
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )

    map_ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    map_ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    map_ax.set_xticks([-120, -60, 0, 60, 120])
    map_ax.set_yticks([-40, 0, 40, 80])
    map_ax.set_xticklabels(["120°W", "60°W", "0°", "60°E", "120°E"])
    map_ax.set_yticklabels(["40°S", "0°", "40°N", "80°N"])
    map_ax.tick_params(axis="both", length=2.0, width=0.5, labelsize=5.2, colors=OKABE_ITO["gray"], pad=1.2)
    map_ax.grid(True, color=OKABE_ITO["light_gray"], linewidth=0.35, alpha=0.7, zorder=0)
    map_ax.spines["top"].set_visible(False)
    map_ax.spines["right"].set_visible(False)
    map_ax.spines["left"].set_linewidth(0.9)
    map_ax.spines["bottom"].set_linewidth(0.9)
    map_ax.spines["left"].set_color(OKABE_ITO["gray"])
    map_ax.spines["bottom"].set_color(OKABE_ITO["gray"])
    add_panel_label(map_ax, "a")

    panel_axes: dict[str, plt.Axes] = {}
    for region_key, box in CALLOUT_POSITIONS.items():
        pax = fig.add_axes(box)
        panel_axes[region_key] = pax
        draw_region_panel(pax, panel_df[panel_df["region_key"] == region_key].copy(), region_key)

    merged = anchor_df.merge(region_polygons[["glambie_region_key"]], on="glambie_region_key", how="left")
    for _, row in merged.iterrows():
        region_key = row["glambie_region_key"]
        if region_key not in panel_axes:
            continue
        pax = panel_axes[region_key]
        pax_box = pax.get_position()
        cx = pax_box.x0 + pax_box.width / 2
        cy = pax_box.y0 + pax_box.height / 2
        side_x = pax_box.x1 if cx < 0.5 else pax_box.x0
        side_y = cy
        con = ConnectionPatch(
            xyA=(row["anchor_lon"], row["anchor_lat"]),
            coordsA=map_ax.transData,
            xyB=(side_x, side_y),
            coordsB=fig.transFigure,
            arrowstyle="-",
            linewidth=0.65,
            color=OKABE_ITO["link_gray"],
            alpha=0.95,
            zorder=1,
        )
        fig.add_artist(con)

    legend_label_map = {
        "proglacial_detached": "IUL (Ice-Uncontacted Proglacial Lake)",
        "proglacial_contacted": "ICL (Ice-Contacted Proglacial Lake)",
        "supraglacial": "SGL (Supraglacial Lake)",
    }
    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markerfacecolor=CLASS_COLORS[c],
            markeredgecolor="none",
            markersize=6,
            alpha=0.8,
            label=legend_label_map[c],
        )
        for c in CLASS_ORDER
    ]
    legend_handles.extend(
        [
            Line2D([0], [0], color=OKABE_ITO["black"], linewidth=1.2, label="Top mini-panel: annual class anomaly trajectory"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=OKABE_ITO["gray"], markersize=6, label="Bottom mini-panel: glacier vs lake anomaly scatter + fit"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=OKABE_ITO["black"], markeredgecolor="white", markersize=6, label="Regional callout anchor"),
        ]
    )
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        handletextpad=0.9,
        columnspacing=1.6,
    )

    fig.suptitle(
        "Global glacier-fed lake classes across 17 regions: annual anomaly trajectories and glacier-lake coupling",
        fontsize=10,
        y=0.985,
    )

    outputs = save_figure(fig, "global_class_region_composite_figure01")
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "map_sources": [str(RGI_REGION_SHP), str(REGION_MANIFEST_CSV), str(LAND_SHP), str(COAST_SHP), str(ANALYSIS_MASTER_GPKG)],
                "panel_sources": [str(CLASS_PANEL_CSV), str(COUPLING_PANEL_CSV)],
                "exported_panel_source_csv": str(PANEL_SOURCE),
                "exported_map_source_csv": str(MAP_SOURCE),
                "class_order": CLASS_ORDER,
                "anchor_logic": "For each region, anchor uses the real lake point nearest the regional median longitude/latitude of core analysis lakes; falls back to region representative point if needed.",
                "figure_style": "Global map with 17 regional callout mini-panels arranged around the map; map layer combines land/coast basemap, latitude-longitude reference grid, weak region outlines, class-colored lake spatial distributions, and regional callout anchors. Top mini-panels show class anomaly trajectories and bottom mini-panels show glacier-lake coupling scatters with fit summaries.",
                **outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    setup_style()
    region_polygons = load_region_geometries()
    lake_map_gdf = load_lake_map_data(region_polygons)
    anchor_df = compute_region_anchors(region_polygons, lake_map_gdf)
    panel_df = load_panel_data()
    PANEL_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(PANEL_SOURCE, index=False)
    anchor_df.to_csv(MAP_SOURCE, index=False)
    outputs = build_figure(region_polygons, anchor_df, panel_df, lake_map_gdf)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
