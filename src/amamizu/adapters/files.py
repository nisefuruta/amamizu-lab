"""降雨強度の検証の入力を読む（AnnualMaxRecords の実装）。

年最大値系列と観測史上順位は、気象庁の観測値として studies/_jma/ に地点ごとに持つ（adapters/jma_store.py の JmaStore）。
検証したい事象の集計（report_data.json）は、その検討の集計なので検討フォルダの raw/ にある。
"""
import json
import math
from pathlib import Path

from .jma_store import JmaStore

RANK_10MIN = "日最大10分間降水量 (mm)"      # 観測史上順位の要素の名前（気象庁のページの表のとおり）


class AnnualMaxFiles:
    """年最大値系列と観測史上順位（studies/_jma/）、事象の集計（検討の raw/report_data.json）。"""

    def __init__(self, raw_dir: Path | str, store: JmaStore):
        raw = Path(raw_dir)
        self.store = store
        self._event = {s["地点"]: s for s in json.loads((raw / "report_data.json").read_text(encoding="utf-8"))["stations"]}

    def stations(self) -> list[dict]:
        return [{"地点": n, "種別": s["種別"], "事象の最大10分_mm": s["最大10分_mm"]} for n, s in self._event.items()]

    def annual_max_10min(self, station: str, first: int, last: int) -> list[tuple[int, float | None, str]]:
        t = self.store.annual_max(station, first, last)
        return [(int(y), None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v), str(q))
                for y, v, q in zip(t["年"], t["10分"], t["10分_品質"], strict=True)]

    def fetched_on(self) -> str:
        """年最大値系列と観測史上順位を取得した日（地点のうち最も新しい日。YYYY-MM-DD）。レポートの出典に書く。"""
        return max(self.store.fetched_on(kind, n) for n in self._event for kind in ("annual_max", "rank"))

    def stat_period_start(self, station: str) -> tuple[int, int]:
        y, m = self.store.rank(station)[RANK_10MIN]["統計期間"].split()[0].split("/")
        return int(y), int(m)
