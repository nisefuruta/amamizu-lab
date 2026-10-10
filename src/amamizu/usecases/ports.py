"""usecases が外側に求める読み出し口（抽象）。実装は adapters にある。

データの表は pandas.DataFrame で受け渡す。列の約束は各メソッドの説明のとおり。
"""
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ..domain.intensity import IntensityFormula
from ..domain.outlet import Hydraulics


@dataclass(frozen=True)
class HayamiCell:
    """早見表の1マス。"""
    district: str
    impervious_pct: int
    f: float
    qc: float
    volume: float       # m3/ha


@dataclass(frozen=True)
class InfiltrationCell:
    """浸透施設を設けたときの必要洪水調節容量の表（千葉県の手引 表10－2）の1マス。"""
    district: str
    intensity_mmh: float    # 設計浸透強度
    treated_pct: float      # 浸透処理面積率
    qc: float               # 放流比流量 m3/s/ha
    volume: float           # m3/ha


class Guideline(Protocol):
    """計画の基準（手引）。"""
    name: str

    def formula(self, district: str, return_period: int) -> IntensityFormula: ...
    def formula_table(self) -> list[tuple[str, int, int, float]]: ...   # (地区, 確率年, t分, 表の強度)
    def formula_table_conditions(self) -> dict: ...      # t_min（表の継続時間 分）, digits（表の値の小数の桁数）
    def qc_min(self, district: str) -> float: ...
    def district_of(self, station: str) -> str: ...
    def districts(self) -> list[str]: ...
    def default_district(self) -> str: ...               # WebGUI で計画降雨を選ぶときの地区の初期値
    def district_term(self) -> str: ...                  # 区域の呼び方（千葉県は「地区」）
    def runoff(self) -> dict: ...                        # impervious / pervious / cap
    def hydraulics(self) -> Hydraulics: ...              # 重力加速度と放流係数
    def hayami(self) -> list[HayamiCell]: ...          # 早見表のない基準では空
    def hayami_conditions(self) -> dict: ...           # tc_min, pond_area_ratio, return_period（早見表があるときだけ呼ぶ）
    def infiltration_table(self) -> list[InfiltrationCell]: ...      # 浸透ありの容量の表。ない基準では空
    def infiltration_table_conditions(self) -> dict: ...             # f, area_ha, pond_area_ratio, tc_min, return_period


class StationRecords(Protocol):
    """気象庁の観測値（studies/_jma/。地点ごとに1か所）。"""

    def daily(self, station: str, year_from: str = "") -> pd.DataFrame:
        """日別値。列: 日付（YYYY-MM-DD の文字列）, 日合計_日別値_mm, 最大1時間_日別値_mm, 日別値_品質"""

    def hourly(self, station: str) -> pd.DataFrame:
        """時間値。列: 日付, 時(1〜24), 降水量_mm, 品質"""

    def tenmin(self, station: str, start: str, end: str) -> pd.DataFrame | None:
        """10分値（start〜end の日がそろったもの）。列: 日付, 時分, 降水量_mm, 品質。なければ None。
        値のある品質で雨量が空欄の行や、同じ日時の行が値の違うまま重なっていれば、読み込むときにエラーにする（時間値・日別値も同じ）"""

    def annual_nhour(self, stations: list[str]) -> pd.DataFrame:
        """年ごとの値の N時間降水量（stations の地点）。列: 地点, 年, 最大{h}時間/(mm), 最大{h}時間/月/日 時:分"""


class AnnualMaxRecords(Protocol):
    """年最大値系列と観測史上順位（前回の検討）。"""

    def stations(self) -> list[dict]: ...                       # 地点, 種別, 2026年8月の実測 など
    def annual_max_10min(self, station: str, first: int, last: int) -> list[tuple[int, float | None, str]]: ...   # first〜last 年の (年, 値 mm（なければ None）, 品質)
    def stat_period_start(self, station: str) -> tuple[int, int]: ...  # 最大10分の統計期間の開始（年, 月）
    def fetched_on(self) -> str: ...                                   # 年最大値系列と観測史上順位を取得した日（YYYY-MM-DD）
