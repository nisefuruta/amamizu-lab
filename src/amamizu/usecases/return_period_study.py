"""基準の降雨強度（県の10年確率式の t＝10分の値）が、観測記録では何年に1回になるかを求める。

標本は年最大10分間降水量。観測史上1〜10位の値の統計期間の内側で、完全な暦年に限る
（統計期間の開始月が1月でない開始年は除く）。検証したい年（2026年）は当てはめに入れない。
グンベル分布をモーメント法で当てはめ、基準の値の再現期間を読む。
"""
from dataclasses import dataclass, field

from ..domain.frequency import Gumbel, sample_stats
from ..domain.rainfall import NO_VALUE
from .ports import AnnualMaxRecords, Guideline


@dataclass(frozen=True)
class StationReturnPeriod:
    station: str
    district: str
    kind: str                       # アメダス / 官署
    years: list[int]
    sample: list[float]             # mm/10分
    mean: float
    std: float
    gumbel: Gumbel
    standard_mmhr: float            # 基準の強度 mm/hr（式から求めたままの値。再現期間は、これを mm/10分にした値で求める）
    return_period: float            # 基準の値の再現期間 年
    event_mm10: float               # 検証したい事象の実測 mm/10分
    event_return_period: float
    qualities: list[str] = field(default_factory=list)   # 標本の各年の品質（normal・quasi・insufficient など）

    @property
    def max_value(self) -> tuple[float, int]:
        v = max(self.sample)
        return v, self.years[self.sample.index(v)]


def evaluate(guideline: Guideline, records: AnnualMaxRecords, standard_period: int = 10,
             duration_min: int = 10, last_year: int = 2025) -> list[StationReturnPeriod]:
    """全地点を評価し、基準の値の再現期間が短い順に返す。"""
    if duration_min != 10:
        # 標本（AnnualMaxRecords.annual_max_10min）は年最大10分間降水量に固定されている。
        # 他の継続時間の基準を10分の標本と比べると、誤った再現期間がエラーなしで出てしまう
        raise ValueError(f"study.json の duration_min（{duration_min}）は 10 にしてください。標本が年最大10分間降水量なので、継続時間は10分だけ扱えます。")
    out = []
    for st in records.stations():
        name = st["地点"]
        district = guideline.district_of(name)
        y0, m0 = records.stat_period_start(name)
        first = y0 + 1 if m0 != 1 else y0
        rows = records.annual_max_10min(name, first, last_year)
        lack = [f"{y}年（年最大値系列にない）" for y in range(first, last_year + 1) if y not in {r[0] for r in rows}]
        lack += [f"{y}年（{'値なし' if v is None else '品質 ' + q}）" for y, v, q in rows if v is None or q in NO_VALUE]
        if lack:                                # 値のない年を標本から黙って除かない（年数と再現期間が変わる）
            raise ValueError(f"{name}: 年最大10分間降水量の標本（{first}〜{last_year}年）に、値のない年があります: {'、'.join(lack)}。"
                             "studies/_jma/ のこの地点の年最大値系列（annual_max.csv）を確かめるか、study.json の last_year を見直してください。")
        pairs = [(y, float(v)) for y, v, _ in rows if v is not None]     # 値のない年は上で止めてある
        qualities = [q for _, v, q in rows if v is not None]
        years = [y for y, _ in pairs]
        sample = [v for _, v in pairs]
        g = Gumbel.by_moments(sample)
        m, s = sample_stats(sample)
        std_mmhr = guideline.formula(district, standard_period).intensity(duration_min)   # 丸めない（表示だけ小数1桁）
        std_mm10 = std_mmhr * duration_min / 60
        ev = st["事象の最大10分_mm"]
        out.append(StationReturnPeriod(name, district, st["種別"], years, sample, m, s, g, std_mmhr,
                                       g.return_period(std_mm10), ev, g.return_period(ev), qualities))
    return sorted(out, key=lambda r: r.return_period)
