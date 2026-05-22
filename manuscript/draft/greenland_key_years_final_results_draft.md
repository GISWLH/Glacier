# Greenland Periphery 结果小节草稿

## 骨架年结果概述
Greenland Periphery 区域最终采用 `2000/2005/2010/2015/2020/2024` 六个骨架年作为阶段性变化分析基础。 在正式替换 `2015` 的 `4month` 版本后，最终骨架共覆盖 `41,548` 个湖泊和 `249,288` 条 lake-year 记录，整体 `usable_share_overall` 为 `93.80%`。
从年际状态看，`2015` 的正式 `4month` 版本相较原始 `7,8,9` 结果明显改善：`usable_share` 从 `96.5%` 提升到 `96.7%`，`total_annual_area_km2` 从 `5421.54` 增加到 `5800.23`，`usable_median_area_ratio` 从 `0.644` 提升到 `0.712`，同时 `low_ratio_rows` 从 `6154` 降至 `4899`。

## 阶段变化主结论
按相邻骨架年划分，Greenland Periphery 最强的下降阶段出现在 `2010_2015`，`total_area_change_km2 = -2354.68`，`median_ratio_change = -0.124`，对应解释标签为 `broad_decline`。
最强的恢复阶段出现在 `2020_2024`，`total_area_change_km2 = +1150.31`，`median_ratio_change = +0.146`。
具体来看，`2005 -> 2010` 呈现明显面积增长（`+589.18 km²`），但 `median_ratio` 几乎不变，说明该阶段更接近广泛面积扩张而非整体比例跃升。`2010 -> 2015` 则表现为系统性下降，`2015 -> 2020` 继续下降但幅度明显减弱，`2020 -> 2024` 则出现显著恢复。

## 湖类贡献解释
在最关键的 `2010 -> 2015` 阶段，两类湖泊均贡献了显著面积下降。`proglacial_detached` 的面积变化为 `-1279.71 km²`，约占该阶段总降幅的 `54.3%`；`proglacial_contacted` 的面积变化为 `-1074.97 km²`，约占 `45.7%`。尽管两类湖的面积贡献接近，但 `proglacial_contacted` 的 `median_ratio_change` 更剧烈（`-0.663`），说明该类湖在下降阶段的相对收缩更强。
在 `2015 -> 2020` 阶段，两类湖继续同步下降，其中 `proglacial_detached` 贡献了`60.6%` 的面积减量。而在 `2020 -> 2024` 恢复阶段，`proglacial_detached` 贡献约`66.0%` 的总增量，`proglacial_contacted` 贡献约 `34.0%`。

## 方法修正后的解释边界
`2015 4month` 全区 `84` 个 chunk 的对照结果显示，`4month` 相比原始 `3month` 版本具有一致正向改善：`sum_total_area_change_km2 = +378.69`，`mean_median_ratio_change = +0.051`，`mean_zero_ratio_share_change = -0.025`。因此，`2015` 的低值问题并非真实变化与方法误差无法区分，而是已经通过更宽时间窗得到部分修复。
不过，方法修正后 `2010 -> 2015` 的异常并未完全消失。相较原始版本，`suspicious_lake_count` 从 `1351` 降到 `775`，`mean_ratio_change` 从 `-0.168` 收敛到 `-0.122`。这说明 `2015` 的方法问题已经被显著削弱，但 Greenland 在 `2010s` 中期的阶段性下降信号仍然存在，后续更适合进入机制解释而非继续局部方法修补。

## 可直接引用的结果性表述
Greenland Periphery 的最终骨架显示，该区域在 `2005 -> 2010` 经历了一次明显增长，随后在 `2010 -> 2015` 出现最强的阶段性下降，在 `2015 -> 2020` 延续较弱下降后，于 `2020 -> 2024` 再次表现出显著恢复。该阶段格局在两类 proglacial lakes 中均可观察到，但 `proglacial_contacted` 湖在下降阶段表现出更剧烈的相对收缩，而 `proglacial_detached` 湖则贡献了更大的绝对面积变化。

## 对应表格与图件源
- 年汇总表：`E:\Glacier\manuscript\tables\greenland_key_years_final_year_summary_table.csv`
- 阶段汇总表：`E:\Glacier\manuscript\tables\greenland_key_years_final_stage_summary_table.csv`
- 类别贡献表：`E:\Glacier\manuscript\tables\greenland_key_years_final_stage_class_table.csv`
- 年序列图源：`E:\Glacier\manuscript\figures_for_paper\greenland_key_years_final_figure_year_source.csv`
- 阶段图源：`E:\Glacier\manuscript\figures_for_paper\greenland_key_years_final_figure_stage_source.csv`
- 类别阶段贡献图源：`E:\Glacier\manuscript\figures_for_paper\greenland_key_years_final_figure_stage_class_source.csv`