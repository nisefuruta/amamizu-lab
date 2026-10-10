"""時間降水量の基本統計の Markdown（旧 analyzer の rainfall_statistics.md と同じ構成）。"""
from ...usecases.rainfall_tools import HourlyStatistics

MONTHS = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]


def render(s: HourlyStatistics, location: str | None = None) -> str:
    out = ["# 降水量データ分析レポート", ""]
    if location:
        out += [f"## {location}", ""]
    out += ["## 基本統計", ""]
    out += [f"- **{k}**: {v}" for k, v in s.basic.items()]
    out += ["", "## 月別統計", "",
            "| 月 | 平均降水量 (mm/時) | 最大降水量 (mm/時) |",
            "|-----|-------------------|-------------------|"]
    for m in range(1, 13):
        if m in s.monthly.index:
            out.append(f"| {MONTHS[m - 1]} | {s.monthly.loc[m, '平均']} | {s.monthly.loc[m, '最大']} |")
    out += ["", "## 年別統計", "",
            "| 年 | 平均降水量 (mm/時) | 年間総降水量 (mm) | 最大降水量 (mm/時) |",
            "|-----|-------------------|-------------------|-------------------|"]
    for y in sorted(s.yearly.index):
        r = s.yearly.loc[y]
        out.append(f"| {y} | {r['平均']} | {r['年間総降水量']} | {r['最大']} |")
    return "\r\n".join(out) + "\r\n"
