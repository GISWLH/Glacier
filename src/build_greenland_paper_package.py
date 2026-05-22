from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(r"E:\Glacier")
TAG = "greenland_key_years_final"
ANALYSIS_DIR = ROOT / "data" / "processed" / "analysis" / TAG
SKELETON_DIR = ROOT / "data" / "processed" / "annual_area_skeleton"
QC_DIR = ROOT / "data" / "interim" / "annual_area_qc"
MANUSCRIPT_TABLES = ROOT / "manuscript" / "tables"
MANUSCRIPT_DRAFT = ROOT / "manuscript" / "draft"
MANUSCRIPT_FIG = ROOT / "manuscript" / "figures_for_paper"


def fmt_float(v: float, digits: int = 2) -> str:
    return f"{float(v):.{digits}f}"


def fmt_share(v: float, digits: int = 1) -> str:
    return f"{float(v) * 100:.{digits}f}%"


def main() -> None:
    MANUSCRIPT_TABLES.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_DRAFT.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIG.mkdir(parents=True, exist_ok=True)

    year_df = pd.read_csv(SKELETON_DIR / f"{TAG}_skeleton_year_summary.csv").sort_values("year")
    stage_df = pd.read_csv(ANALYSIS_DIR / f"{TAG}_stage_change_summary.csv").sort_values("year_start")
    class_stage_df = pd.read_csv(ANALYSIS_DIR / f"{TAG}_stage_class_change_summary.csv").sort_values(
        ["interval", "harmonized_class"]
    )
    old_year_df = pd.read_csv(SKELETON_DIR / "greenland_key_years_skeleton_year_summary.csv").sort_values("year")
    qc_2015 = pd.read_json(QC_DIR / "greenland_2015_4month_formal_qc_summary.json", typ="series")
    all84 = pd.read_json(QC_DIR / "greenland_2015_recheck_all84_summary.json", typ="series")
    anom_2010_2015_old = pd.read_json(
        SKELETON_DIR / "greenland_2010_2015_anomaly_diagnosis_summary.json", typ="series"
    )
    anom_2010_2015_new = pd.read_json(
        SKELETON_DIR / "greenland_2010_2015_final_anomaly_diagnosis_summary.json", typ="series"
    )

    old_2015 = old_year_df.loc[old_year_df["year"] == 2015].iloc[0]
    new_2015 = year_df.loc[year_df["year"] == 2015].iloc[0]

    year_table = year_df[
        [
            "year",
            "usable_share",
            "total_annual_area_km2",
            "usable_total_annual_area_km2",
            "usable_median_area_ratio",
            "usable_mean_area_ratio",
            "median_image_count",
            "low_ratio_rows",
        ]
    ].copy()
    year_table["usable_share_pct"] = year_table["usable_share"] * 100
    year_table = year_table[
        [
            "year",
            "usable_share",
            "usable_share_pct",
            "total_annual_area_km2",
            "usable_total_annual_area_km2",
            "usable_median_area_ratio",
            "usable_mean_area_ratio",
            "median_image_count",
            "low_ratio_rows",
        ]
    ]

    stage_table = stage_df[
        [
            "interval",
            "year_start",
            "year_end",
            "usable_share_change",
            "total_area_change_km2",
            "usable_area_change_km2",
            "median_ratio_change",
            "mean_ratio_change",
            "low_ratio_share_change",
            "interpretation",
        ]
    ].copy()
    stage_table["usable_share_change_pct"] = stage_table["usable_share_change"] * 100
    stage_table["low_ratio_share_change_pct"] = stage_table["low_ratio_share_change"] * 100
    stage_table = stage_table[
        [
            "interval",
            "year_start",
            "year_end",
            "usable_share_change",
            "usable_share_change_pct",
            "total_area_change_km2",
            "usable_area_change_km2",
            "median_ratio_change",
            "mean_ratio_change",
            "low_ratio_share_change",
            "low_ratio_share_change_pct",
            "interpretation",
        ]
    ]

    class_table = class_stage_df[
        [
            "interval",
            "harmonized_class",
            "area_change_km2",
            "area_change_share_of_interval",
            "median_ratio_change",
            "mean_ratio_change",
            "zero_ratio_share_change",
            "usable_share_change",
            "lake_count",
        ]
    ].copy()
    class_table["area_change_share_of_interval_pct"] = class_table["area_change_share_of_interval"] * 100
    class_table["zero_ratio_share_change_pct"] = class_table["zero_ratio_share_change"] * 100
    class_table["usable_share_change_pct"] = class_table["usable_share_change"] * 100

    year_table_path = MANUSCRIPT_TABLES / "greenland_key_years_final_year_summary_table.csv"
    stage_table_path = MANUSCRIPT_TABLES / "greenland_key_years_final_stage_summary_table.csv"
    class_table_path = MANUSCRIPT_TABLES / "greenland_key_years_final_stage_class_table.csv"
    figure_year_path = MANUSCRIPT_FIG / "greenland_key_years_final_figure_year_source.csv"
    figure_stage_path = MANUSCRIPT_FIG / "greenland_key_years_final_figure_stage_source.csv"
    figure_class_path = MANUSCRIPT_FIG / "greenland_key_years_final_figure_stage_class_source.csv"

    year_table.to_csv(year_table_path, index=False, encoding="utf-8-sig")
    stage_table.to_csv(stage_table_path, index=False, encoding="utf-8-sig")
    class_table.to_csv(class_table_path, index=False, encoding="utf-8-sig")
    year_df.to_csv(figure_year_path, index=False, encoding="utf-8-sig")
    stage_df.to_csv(figure_stage_path, index=False, encoding="utf-8-sig")
    class_stage_df.to_csv(figure_class_path, index=False, encoding="utf-8-sig")

    largest_decline = stage_df.sort_values("total_area_change_km2").iloc[0]
    largest_increase = stage_df.sort_values("total_area_change_km2", ascending=False).iloc[0]

    class_2010_2015 = class_stage_df[class_stage_df["interval"] == "2010_2015"].sort_values("area_change_km2")
    class_2015_2020 = class_stage_df[class_stage_df["interval"] == "2015_2020"].sort_values("area_change_km2")
    class_2020_2024 = class_stage_df[class_stage_df["interval"] == "2020_2024"].sort_values(
        "area_change_km2", ascending=False
    )

    draft_lines = [
        "# Greenland Periphery 结果小节草稿",
        "",
        "## 骨架年结果概述",
        (
            f"Greenland Periphery 区域最终采用 `2000/2005/2010/2015/2020/2024` 六个骨架年作为阶段性变化分析基础。"
            f" 在正式替换 `2015` 的 `4month` 版本后，最终骨架共覆盖 `41,548` 个湖泊和 `249,288` 条 lake-year 记录，"
            f"整体 `usable_share_overall` 为 `{fmt_share(year_df['usable_share'].mean(), 2)}`。"
        ),
        (
            f"从年际状态看，`2015` 的正式 `4month` 版本相较原始 `7,8,9` 结果明显改善："
            f"`usable_share` 从 `{fmt_share(old_2015['usable_share'])}` 提升到 `{fmt_share(new_2015['usable_share'])}`，"
            f"`total_annual_area_km2` 从 `{fmt_float(old_2015['total_annual_area_km2'])}` 增加到 `{fmt_float(new_2015['total_annual_area_km2'])}`，"
            f"`usable_median_area_ratio` 从 `{fmt_float(old_2015['usable_median_area_ratio'], 3)}` 提升到 `{fmt_float(new_2015['usable_median_area_ratio'], 3)}`，"
            f"同时 `low_ratio_rows` 从 `{int(old_2015['low_ratio_rows'])}` 降至 `{int(new_2015['low_ratio_rows'])}`。"
        ),
        "",
        "## 阶段变化主结论",
        (
            f"按相邻骨架年划分，Greenland Periphery 最强的下降阶段出现在 `{largest_decline['interval']}`，"
            f"`total_area_change_km2 = {fmt_float(largest_decline['total_area_change_km2'])}`，"
            f"`median_ratio_change = {fmt_float(largest_decline['median_ratio_change'], 3)}`，"
            f"对应解释标签为 `{largest_decline['interpretation']}`。"
        ),
        (
            f"最强的恢复阶段出现在 `{largest_increase['interval']}`，"
            f"`total_area_change_km2 = +{fmt_float(largest_increase['total_area_change_km2'])}`，"
            f"`median_ratio_change = +{fmt_float(largest_increase['median_ratio_change'], 3)}`。"
        ),
        (
            f"具体来看，`2005 -> 2010` 呈现明显面积增长"
            f"（`+{fmt_float(stage_df.loc[stage_df['interval']=='2005_2010', 'total_area_change_km2'].iloc[0])} km²`），"
            f"但 `median_ratio` 几乎不变，说明该阶段更接近广泛面积扩张而非整体比例跃升。"
            f"`2010 -> 2015` 则表现为系统性下降，`2015 -> 2020` 继续下降但幅度明显减弱，"
            f"`2020 -> 2024` 则出现显著恢复。"
        ),
        "",
        "## 湖类贡献解释",
        (
            f"在最关键的 `2010 -> 2015` 阶段，两类湖泊均贡献了显著面积下降。"
            f"`proglacial_detached` 的面积变化为 `{fmt_float(class_2010_2015.iloc[0]['area_change_km2'])} km²`，"
            f"约占该阶段总降幅的 `{fmt_share(class_2010_2015.iloc[0]['area_change_share_of_interval'], 1)}`；"
            f"`proglacial_contacted` 的面积变化为 `{fmt_float(class_2010_2015.iloc[1]['area_change_km2'])} km²`，"
            f"约占 `{fmt_share(class_2010_2015.iloc[1]['area_change_share_of_interval'], 1)}`。"
            f"尽管两类湖的面积贡献接近，但 `proglacial_contacted` 的 `median_ratio_change` 更剧烈"
            f"（`{fmt_float(class_2010_2015.iloc[1]['median_ratio_change'], 3)}`），"
            f"说明该类湖在下降阶段的相对收缩更强。"
        ),
        (
            f"在 `2015 -> 2020` 阶段，两类湖继续同步下降，其中 `proglacial_detached` 贡献了"
            f"`{fmt_share(class_2015_2020.iloc[0]['area_change_share_of_interval'], 1)}` 的面积减量。"
            f"而在 `2020 -> 2024` 恢复阶段，`proglacial_detached` 贡献约"
            f"`{fmt_share(class_2020_2024.iloc[0]['area_change_share_of_interval'], 1)}` 的总增量，"
            f"`proglacial_contacted` 贡献约 `{fmt_share(class_2020_2024.iloc[1]['area_change_share_of_interval'], 1)}`。"
        ),
        "",
        "## 方法修正后的解释边界",
        (
            f"`2015 4month` 全区 `84` 个 chunk 的对照结果显示，`4month` 相比原始 `3month` 版本"
            f"具有一致正向改善：`sum_total_area_change_km2 = +{fmt_float(all84['sum_total_area_change_km2'])}`，"
            f"`mean_median_ratio_change = +{fmt_float(all84['mean_median_ratio_change'], 3)}`，"
            f"`mean_zero_ratio_share_change = {fmt_float(all84['mean_zero_ratio_share_change'], 3)}`。"
            f"因此，`2015` 的低值问题并非真实变化与方法误差无法区分，而是已经通过更宽时间窗得到部分修复。"
        ),
        (
            f"不过，方法修正后 `2010 -> 2015` 的异常并未完全消失。相较原始版本，"
            f"`suspicious_lake_count` 从 `{int(anom_2010_2015_old['suspicious_lake_count'])}` 降到 "
            f"`{int(anom_2010_2015_new['suspicious_lake_count'])}`，"
            f"`mean_ratio_change` 从 `{fmt_float(anom_2010_2015_old['mean_ratio_change'], 3)}` 收敛到 "
            f"`{fmt_float(anom_2010_2015_new['mean_ratio_change'], 3)}`。"
            f"这说明 `2015` 的方法问题已经被显著削弱，但 Greenland 在 `2010s` 中期的阶段性下降信号仍然存在，"
            f"后续更适合进入机制解释而非继续局部方法修补。"
        ),
        "",
        "## 可直接引用的结果性表述",
        (
            "Greenland Periphery 的最终骨架显示，该区域在 `2005 -> 2010` 经历了一次明显增长，"
            "随后在 `2010 -> 2015` 出现最强的阶段性下降，在 `2015 -> 2020` 延续较弱下降后，"
            "于 `2020 -> 2024` 再次表现出显著恢复。该阶段格局在两类 proglacial lakes 中均可观察到，"
            "但 `proglacial_contacted` 湖在下降阶段表现出更剧烈的相对收缩，而 `proglacial_detached` 湖则贡献了更大的绝对面积变化。"
        ),
        "",
        "## 对应表格与图件源",
        f"- 年汇总表：`{year_table_path}`",
        f"- 阶段汇总表：`{stage_table_path}`",
        f"- 类别贡献表：`{class_table_path}`",
        f"- 年序列图源：`{figure_year_path}`",
        f"- 阶段图源：`{figure_stage_path}`",
        f"- 类别阶段贡献图源：`{figure_class_path}`",
    ]

    draft_path = MANUSCRIPT_DRAFT / "greenland_key_years_final_results_draft.md"
    draft_path.write_text("\n".join(draft_lines), encoding="utf-8-sig")

    payload = {
        "year_table_csv": str(year_table_path),
        "stage_table_csv": str(stage_table_path),
        "class_table_csv": str(class_table_path),
        "figure_year_source_csv": str(figure_year_path),
        "figure_stage_source_csv": str(figure_stage_path),
        "figure_class_source_csv": str(figure_class_path),
        "draft_md": str(draft_path),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
