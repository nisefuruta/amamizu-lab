"""降雨強度の検証（kind＝return_period）のレポートの Markdown を組み立てる。

文に記載する値（題名、地域、地点数、確率年、除いた年、事象の期間、取得日、出典）は、study.json と手引の定数から作る。
条件を変えたときに、前の検討の文が黙って残らないようにするため。
"""
from ...usecases.return_period_study import StationReturnPeriod


class Text:
    """study.json と手引の定数から、レポートの文に使う語を集める（study.json の項目は study_config.check_study で確かめたもの）。"""

    def __init__(self, conf: dict, guideline, n_stations: int, retrieved: str):
        self.title = conf["title"]
        self.T = conf["standard_period"]
        self.duration = conf["duration_min"]
        self.excluded = conf["last_year"] + 1
        self.event, self.month = conf["event"]["label"], conf["event"]["month"]
        y, m, d = map(int, retrieved.split("-"))
        self.retrieved = f"{y}年{m}月{d}日"                 # 取得日（studies/_jma/ の記録）
        self.region, self.issuer = guideline.report_text("region"), guideline.report_text("issuer")
        self.citation = guideline.report_text("citation")
        self.cite_formulas, self.cite_districts = guideline.cite("formulas"), guideline.cite("districts")
        self.D = guideline.district_term()        # 区域の呼び方（千葉県は「地区」）
        self.n = n_stations

    def header(self) -> list[str]:
        return f"""# {self.title}

この検討の計算を、{self.region}の観測所{self.n}地点ごとに示す。

標本は気象庁が公表する年最大10分間降水量（mm）をそのまま使う。
{self.issuer}の{self.T}年確率式の値は mm/hr なので、6で割って mm/10分 に直し、その値の再現期間を読む。
継続時間は{self.duration}分とする。

年最大10分間降水量にグンベル分布をモーメント法で算定した。
標本は{self.excluded}年と部分年を除く。

地点の並びは、{self.D}の{self.T}年確率式の値の再現期間が短い順とする。
上にあるほど、その地点の準拠基準が名乗る{self.T}年確率に対して実際の頻度が高い。""".split("\n")

    def footer(self) -> list[str]:
        return f"""---

出典は気象庁「過去の気象データ検索」（年ごとの値、観測史上1〜10位の値、日別値表）、取得日 {self.retrieved}。
{self.D}別の確率降雨強度式と適用範囲は、{self.citation}の{self.cite_formulas} と{self.cite_districts} による。""".split("\n")


def _std_mm10(r: StationReturnPeriod) -> float:
    """表示用の基準の値 mm/10分。表の値（小数1桁の mm/hr）を6で割る。再現期間は丸めない値で求めている。"""
    return round(r.standard_mmhr, 1) / 6


def _years(t: float) -> str:
    return f"{t:.0f}年" if t >= 10 else f"{t:.1f}年"


def _section(r: StationReturnPeriod, tx: Text) -> list[str]:
    y0, y1, n = r.years[0], r.years[-1], len(r.years)
    vmax, ymax = r.max_value
    lines = [f"## {r.station}（{r.district}{tx.D}、{r.kind}、{y0}〜{y1}年、{n}年）", "",
             "| 項目 | 値 |", "| --- | ---: |",
             f"| 標本年数 | {n}年（{y0}〜{y1}年） |",
             f"| 年最大10分間降水量の平均 | {r.mean:.2f} mm/10分 |",
             f"| 同 標準偏差 | {r.std:.2f} mm/10分 |",
             f"| 同 最大 | {vmax:.1f} mm/10分（{ymax}年） |",
             f"| 規模パラメータ beta | {r.gumbel.beta:.2f} mm/10分 |",
             f"| 位置パラメータ mu | {r.gumbel.mu:.2f} mm/10分 |",
             f"| {r.district}{tx.D}の{tx.T}年確率式の値 | **{_std_mm10(r):.2f} mm/10分**（{r.standard_mmhr:.1f} mm/hr ÷ 6） |",
             f"| 同 再現期間 | **{r.return_period:.1f}年** |",
             f"| {tx.event}の実測 | {r.event_mm10:.1f} mm/10分 |",
             f"| 同 再現期間 | {_years(r.event_return_period)} |", "",
             "標本（年最大10分間降水量、mm/10分）", "", "```"]
    cells = [f"{y}{v:7.1f}" for y, v in zip(r.years, r.sample)]
    for i in range(0, len(cells), 6):
        lines.append("  ".join(cells[i:i + 6]))
    lines += ["```", ""]
    return lines


def render(results: list[StationReturnPeriod], conf: dict, guideline, retrieved: str) -> str:
    """conf は study.json の内容、guideline は手引の定数（JsonGuideline）、retrieved は標本を取得した日（YYYY-MM-DD）。"""
    tx = Text(conf, guideline, len(results), retrieved)
    head = tx.header()
    quasi = sum(q == "quasi" for r in results for q in r.qualities)
    insufficient = sum(q == "insufficient" for r in results for q in r.qualities)
    parts = [p for p in (f"準正常値（`)`）が {quasi} 個" if quasi else "", f"資料不足値（`]`）が {insufficient} 個" if insufficient else "") if p]
    if parts:                                   # 準正常値・資料不足値も公表値として使い、含まれることだけを添える
        at = next(i for i, s in enumerate(head) if s.startswith("標本は気象庁")) + 1
        head[at:at] = [f"標本には、{'、'.join(parts)}含まれる。"]
    lines = head + ["", "## 一覧", "",
                           f"| 地点 | {tx.D} | 種別 | 標本年数 | 平均 | {tx.issuer}式{tx.T}年 | 再現期間 | {tx.month}の実測 |",
                           "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r.station} | {r.district} | {r.kind} | {len(r.years)}年 | {r.mean:.2f} | "
                     f"{_std_mm10(r):.2f} | **{r.return_period:.1f}年** | {r.event_mm10:.1f} |")
    lines += ["", "単位 mm/10分。標本年数と平均は年最大10分間降水量の系列のもの。", ""]
    for r in results:
        lines += _section(r, tx)
    lines += tx.footer()
    return "\r\n".join(lines) + "\r\n"
