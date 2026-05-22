# Greenland Periphery 结果小节正式稿

## 结果段落主稿

Greenland Periphery 最终采用 `2000/2005/2010/2015/2020/2024` 六个骨架年表征区域阶段性变化。完成 `2015` 年 `4month` 版本替换后，最终骨架共覆盖 `41,548` 个湖泊和 `249,288` 条 lake-year 记录，整体 `usable_share_overall` 为 `93.80%`。从骨架年总体格局看，区域湖泊总面积在 `2000` 至 `2010` 年间总体维持高位并于 `2010` 年达到阶段性峰值（`8154.91 km²`），随后在 `2015` 年显著下降至 `5800.23 km²`，`2020` 年进一步降至 `5467.13 km²`，而到 `2024` 年则恢复至 `6617.43 km²`。与面积变化相对应，`usable_median_area_ratio` 从 `2010` 年的 `0.836` 下降到 `2015` 年的 `0.712`，在 `2020` 年进一步降至 `0.632`，并在 `2024` 年回升至 `0.778`，表明 Greenland Periphery 在 `2010s` 中期经历了明显收缩，并在最近阶段出现恢复信号。上述年际变化与数据质量指标方向一致：`usable_share` 从 `2000` 年的 `88.1%` 逐步提升到 `2024` 年的 `98.6%`，说明最终骨架不仅保留了阶段变化信息，也具备较稳定的可用观测覆盖（图 1）。

在 `2015` 年处理口径上，采用 `6,7,8,9` 四个月窗口的正式版本后，`2015` 年结果相较原始 `7,8,9` 三个月版本得到系统改善。具体而言，`usable_share` 由 `96.47%` 提升至 `96.68%`，`total_annual_area_km2` 由 `5421.54 km²` 增至 `5800.23 km²`，`usable_median_area_ratio` 由 `0.644` 提升至 `0.712`，同时 `low_ratio_rows` 从 `6154` 降至 `4899`。全区 `84` 个 chunk 的复核结果也表明，`4month` 版本相较 baseline 具有一致正向修正效果，表现为 `sum_total_area_change_km2 = +378.69 km²`、`mean_median_ratio_change = +0.051` 和 `mean_zero_ratio_share_change = -0.025`。因此，`2015` 年原始结果中的部分低值确实带有方法学偏差，但修正之后 Greenland Periphery 在 `2010 -> 2015` 阶段的下降并未消失，而是由极端异常收敛为更可信的阶段性负变化信号（图 1）。

按相邻骨架年划分，Greenland Periphery 的阶段变化具有明显的非单调特征（图 2）。其中，`2000 -> 2005` 仅表现为弱幅下降，`total_area_change_km2 = -53.81`，`median_ratio_change = -0.035`，可视为轻微波动阶段。`2005 -> 2010` 则转为正增长，`total_area_change_km2 = +589.18`，但 `median_ratio_change` 仅为 `-0.001`，说明这一阶段更接近区域总面积扩张，而非湖泊面积相对基准状态的整体抬升。最显著的转折出现在 `2010 -> 2015`，该阶段 `total_area_change_km2 = -2354.68`，`usable_area_change_km2 = -2190.71`，`median_ratio_change = -0.124`，是当前 Greenland Periphery 骨架中最强的下降阶段。`2015 -> 2020` 延续下降趋势，但幅度明显减弱，`total_area_change_km2 = -333.11`，`median_ratio_change = -0.080`。相比之下，`2020 -> 2024` 表现出最强恢复，`total_area_change_km2 = +1150.31`，`median_ratio_change = +0.146`，说明区域在最近阶段出现了较强的恢复性增长。整体上，Greenland Periphery 的阶段演化可概括为“`2005 -> 2010` 增长、`2010 -> 2020` 连续收缩、`2020 -> 2024` 恢复”的三段式格局（图 2）。

类别拆分结果显示，上述阶段变化并非由单一类型湖泊驱动，而是两类 proglacial lakes 共同参与，但贡献方式有所差异（图 3）。在最关键的 `2010 -> 2015` 阶段，`proglacial_detached` 的面积变化为 `-1279.71 km²`，约占该阶段总降幅的 `54.3%`；`proglacial_contacted` 的面积变化为 `-1074.97 km²`，约占 `45.7%`。这说明最强下降阶段不是个别类别主导的局地异常，而是两类湖共同收缩的区域性现象。不过，两类湖在相对变化强度上的表现并不一致：`proglacial_contacted` 的 `median_ratio_change = -0.663`，显著强于 `proglacial_detached` 的 `-0.080`，表明接触型冰川湖在该阶段经历了更剧烈的相对收缩。此后在 `2015 -> 2020` 阶段，两类湖继续同步下降，其中 `proglacial_detached` 贡献了 `60.6%` 的面积减量；而在 `2020 -> 2024` 恢复阶段，`proglacial_detached` 贡献了约 `66.0%` 的总增量，`proglacial_contacted` 贡献约 `34.0%`。因此，从绝对面积变化看，`proglacial_detached` 对区域总量波动的贡献更大；从相对收缩强度看，`proglacial_contacted` 在下降阶段更敏感（图 3）。

需要指出的是，方法修正虽显著削弱了 `2015` 年的异常程度，但并未完全抹去 `2010 -> 2015` 阶段的负向信号。相较于原始版本，异常诊断中的 `suspicious_lake_count` 已从 `1351` 降至 `775`，`mean_ratio_change` 也从 `-0.168` 收敛到 `-0.122`。这意味着 Greenland Periphery 在 `2010s` 中期的下降不能再简单归因于局部提取失败，而更可能反映真实区域变化与残余方法误差共同叠加后的综合结果。基于当前证据，后续工作的重点不宜继续停留在 `2015` 年局部口径修补，而应进一步转向对 `2010 -> 2015` 强下降和 `2020 -> 2024` 恢复阶段的机制解释。

## 精简版结果表述

Greenland Periphery 的最终骨架表明，该区域在 `2005 -> 2010` 经历了一次明显增长，随后在 `2010 -> 2015` 出现最强的阶段性下降，并在 `2015 -> 2020` 维持较弱负增长后，于 `2020 -> 2024` 再次表现出显著恢复。该阶段格局在 `proglacial_detached` 与 `proglacial_contacted` 两类湖泊中均可观察到，其中前者贡献了更大的绝对面积变化，而后者在下降阶段表现出更剧烈的相对收缩。

## 图文对应关系

- 图 1 对应内容：
  - 骨架年总体格局
  - `2015 4month` 替换后的年际位置
  - `usable_share` 与 `usable_median_area_ratio` 的协同变化
- 图 2 对应内容：
  - 相邻阶段的面积增减幅度
  - `2010 -> 2015` 最强下降与 `2020 -> 2024` 最强恢复
  - `2005 -> 2010` 面积增长但 ratio 基本持平的过渡性质
- 图 3 对应内容：
  - 两类 proglacial lakes 对阶段面积变化的贡献拆分
  - `proglacial_contacted` 的更强相对收缩
  - `proglacial_detached` 的更大绝对面积贡献

## 对应文件

- 年汇总表：`E:\Glacier\manuscript\tables\greenland_key_years_final_year_summary_table.csv`
- 阶段汇总表：`E:\Glacier\manuscript\tables\greenland_key_years_final_stage_summary_table.csv`
- 类别贡献表：`E:\Glacier\manuscript\tables\greenland_key_years_final_stage_class_table.csv`
- 图件清单：`E:\Glacier\manuscript\figures_for_paper\rendered\greenland_key_years_final\greenland_key_years_final_figure_manifest.json`
- Figure 01：`E:\Glacier\manuscript\figures_for_paper\rendered\greenland_key_years_final\greenland_key_years_final_figure01_year_overview.pdf`
- Figure 02：`E:\Glacier\manuscript\figures_for_paper\rendered\greenland_key_years_final\greenland_key_years_final_figure02_stage_changes.pdf`
- Figure 03：`E:\Glacier\manuscript\figures_for_paper\rendered\greenland_key_years_final\greenland_key_years_final_figure03_class_contributions.pdf`
