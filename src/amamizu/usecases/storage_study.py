"""手引どおりに計画した貯留施設に、実測の降雨を観測どおりの順序で入れて、溢れるかを確かめる。"""
import datetime
import math
import re
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..domain import rainfall as rf
from ..domain.design import DesignedFacility, design_for_depth
from ..domain.outlet import SquareOrifice
from ..domain.pond import DEFAULT_TC_MIN, Facility, PrismaticPond, route
from .design_study import (OVERDRAW_TIMES, Catchment, DesignResult, Inflow, ObservedRain, observed_inflow, overdraw_note, planned_inflow,
                           run_design)
from .ports import Guideline, StationRecords

DURATIONS = (1, 3, 6, 12, 24)
TOTAL_TOL_MM = 1.0      # 時間値・10分値の合計と日別値の合計が一致したとみなす差 mm


def rounded(df: pd.DataFrame) -> pd.DataFrame:
    """結果の表（TSV）に書く桁に丸める。最大貯留率は小数3桁、溢水量は0.1m3（0.0 と読める 0.05 未満は有効数字2桁にし、
    溢水の回数と食い違わせない）。継続時間ごとの雨量（実測・計画）は0.1mm、計画との比は小数3桁。
    判定とレポートの表示は丸めない値を使う（丸めた値をもう一度丸めると、0.93494 が 0.935 を経て 94% になる）。"""
    df = df.copy()
    for c in df.columns:
        if c.endswith("最大貯留率"):
            df[c] = df[c].map(lambda v: round(v, 3))
        elif c.endswith("溢水量_m3"):
            df[c] = df[c].map(lambda v: float(f"{v:.2g}") if 0 < v < 0.05 else round(v, 1))
        elif re.search(r"\d+h_(.+_)?mm$", c):
            df[c] = df[c].round(1)
        elif re.search(r"\d+h_.+比$", c):
            df[c] = df[c].round(3)
    return df


@dataclass(frozen=True)
class ModelConditions:
    """モデル施設と判定の条件。"""
    f: float = 0.8                  # 流出係数（不浸透面積率50%、手引 図5－2 の例）
    depth: float = 2.0              # 柱状の池の深さ（満水位）m
    area_ha: float = 1.0            # 開発面積 ha
    return_period: int = 50         # 計画降雨の確率年
    tail_hours: int = 48            # 降雨の後に足す無降水の時間（池が下がりきるまで）
    fill_ratio: float = 0.5         # 時間値で計算する候補: 総量が容量のこの割合に相当する雨量以上
    dry_gap: int = 2                # 降雨を区切る無降水日の日数
    tenmin_from: str = "2008-01-01"  # 10分値で計算し直す降雨の開始日の下限
    tenmin_fill: float = 0.7        # 10分値で計算し直す降雨の最大貯留率の下限
    tc_min: int = DEFAULT_TC_MIN    # 洪水到達時間（計画降雨の区切り、追跡の刻み）分

    def __post_init__(self):
        # 観測は10分値なので tc は10の倍数。レポートの時刻の表示は1時間ごとなので60の約数に限る
        if self.tc_min not in (10, 20, 30, 60):
            raise ValueError(f"study.json の conditions.tc_min（{self.tc_min} 分）は扱えません。貯留施設の検証では 10・20・30・60 のどれかにしてください。")


# 降雨の日に、日別値表の最大1時間降水量（公表値）がない日の数の列。1以上なら、その降雨の1時間は
# ほかの日の公表値と毎正時で区切った1時間雨量の大きいほうで、公表値のない日の最大1時間は含まない
NO_PUBLISHED = "1h_公表値のない日"


def _rain(frame: pd.DataFrame) -> tuple[npt.NDArray[np.float64], int]:
    """取得データ（時間値・10分値）の雨量 mm と、値のないコマの数。値のない品質（missing・doubtful・blank）のコマは0mm にする。
    値のある品質で雨量が空欄の行は、読み込むときにエラーにしてある（StationRecords の実装）。"""
    return frame["降水量_mm"].fillna(0.0).to_numpy(dtype=float), int(frame["品質"].isin(rf.NO_VALUE).sum())


def _day(frame: pd.DataFrame, i: int) -> str:
    """日別値の i 行目の日付（YYYY-MM-DD の文字列。StationRecords.daily の取り決め）。"""
    return str(frame["日付"].iat[int(i)])


@dataclass(frozen=True)
class ModelFacility:
    """地区のモデル施設。orifice は designed.facility.outlet と同じもの（正方形のオリフィスとして寸法を読むために型を持つ）。"""
    designed: DesignedFacility[PrismaticPond]
    orifice: SquareOrifice


@dataclass
class StorageStudy:
    guideline: Guideline
    records: StationRecords
    cond: ModelConditions = field(default_factory=ModelConditions)

    def __post_init__(self):
        self._models: dict[str, ModelFacility] = {}     # 地区ごとのモデル施設（model で決める）
        self.notices: list[str] = []     # 計算は続けるが、結果を読む人に知らせること
        self._hourly: dict[str, pd.DataFrame | None] = {}

    def _tenmin(self, name: str, start: str, end: str) -> pd.DataFrame:
        """start〜end の10分値。日がそろっていなければ、取得のコマンドを示してエラーにする。"""
        t = self.records.tenmin(name, start, end)
        if t is None:
            raise ValueError(f"{name} {start}〜{end}: 10分値がそろっていません。"
                             f"amamizu fetch <検討フォルダ> tenmin --stations {name} --start {start} --end {end} で取得してください。")
        return t

    @property
    def plan_label(self) -> str:
        """計画の確率年の表記（1/50 など）。結果の表の列名とレポートに使う。"""
        return f"1/{self.cond.return_period}"

    # 施設と1降雨の計算 ------------------------------------------------------
    def model(self, district: str) -> ModelFacility:
        """地区の計画降雨でモデル施設を決める。計画降雨で満水位にちょうど達する寸法を、判定が変わらない向き（安全側）に丸める:
        オリフィスは1mm単位で切り捨て（許容放流量を超えない）、そのオリフィスで計画降雨の最大水位がちょうど満水位になる
        池の面積を求め、1m^2単位で切り上げる（溢れない）。ちょうど達する寸法は、四捨五入すると判定が揺れる。
        地点ごとの検討（amamizu event-studies）もこの寸法で作る。"""
        if district not in self._models:
            tc, depth, hy = self.cond.tc_min, self.cond.depth, self.guideline.hydraulics()
            inflow = self.planned(district).nodes
            q_allow = self.allowable_q(district)
            sized = SquareOrifice.sized_for(depth, q_allow, hy)
            orifice = SquareOrifice(math.floor(sized.size * 1000) / 1000, hy)
            fitted = design_for_depth(inflow, None, depth, dt=tc * 60, orifice=orifice)
            fac = Facility(PrismaticPond(float(math.ceil(fitted.facility.pond.area)), depth), orifice)
            self._models[district] = ModelFacility(DesignedFacility(fac, route(fac, inflow, capped=False, dt=tc * 60)), orifice)
        return self._models[district]

    def facility(self, district: str) -> DesignedFacility[PrismaticPond]:
        """地区のモデル施設と、計画降雨での追跡結果（model）。"""
        return self.model(district).designed

    def planned(self, district: str) -> Inflow:
        """地区の計画降雨（後方集中型24時間）の流入（設計の検討の planned_inflow）。"""
        return planned_inflow(self.guideline.formula(district, self.cond.return_period), self.catchments(),
                              self.cond.tc_min, "rear")

    def catchments(self) -> list[Catchment]:
        return [Catchment("開発区域", self.cond.area_ha, self.cond.f)]

    def allowable_q(self, district: str) -> float:
        return self.guideline.qc_min(district) * self.cond.area_ha

    def outlet_spec(self, district: str) -> dict:
        """モデル施設のオリフィスを、設計の検討の書き方（study.json の orifice）にしたもの。"""
        s = self.model(district).orifice.size
        return {"sill_m": 0.0, "width_m": s, "height_m": s}

    def run(self, district: str, inflow: Inflow, dry_blocks: int) -> DesignResult:
        """流入を地区のモデル施設に入れる（設計の検討と同じ run_design）。流入の後に dry_blocks 区切りの無降水を足す。
        前の刻みの放流量が池の水と流入の和を超えた刻みがあれば、notices に残す。"""
        tc = self.cond.tc_min
        if inflow.rain is None:
            raise ValueError("降雨のない流入（ハイドログラフ）は、モデル施設の検討に入れられません。計画降雨か実測降雨を使ってください。")
        res = run_design(inflow, self.facility(district).facility.pond, [self.outlet_spec(district)], None,
                         (len(inflow.rain) + dry_blocks + 1) * tc, hy=self.guideline.hydraulics(),
                         allowable_q=self.allowable_q(district))
        if res.routing.overdraw:
            when = f"{inflow.start:%Y-%m-%d} からの降雨" if inflow.start else "計画降雨"
            self.notices.append(f"{district}{self.guideline.district_term()}の{when}: {overdraw_note(res.routing, tc, OVERDRAW_TIMES)}")
        return res

    def route_rain(self, district: str, mm, step_min: int, start: str) -> tuple[ObservedRain, DesignResult]:
        """観測の区間ごとの雨量 mm（step_min が 10 なら10分値、60 なら時間値）を、設計の検討と同じ observed_inflow で
        tc ごとの流入にし、モデル施設に入れる。降雨の後に tail_hours の無降水を足す。"""
        obs = ObservedRain(f"{step_min}分", datetime.datetime.fromisoformat(start), step_min, np.asarray(mm, dtype=float))
        return obs, self.run(district, observed_inflow(obs, self.catchments(), self.cond.tc_min),
                             self.cond.tail_hours * 60 // self.cond.tc_min)

    def storage_cols(self, district: str, mm, step_min: int, start: str) -> dict:
        s = self.route_rain(district, mm, step_min, start)[1].summary
        return {"最大貯留率": s["最大貯留率"], "溢水量_m3": s["溢水量_m3"]}

    def duration_cols(self, district: str, hourly_mm, published: pd.Series) -> dict:
        """継続時間ごとの雨量の列と、公表値のない日の数（NO_PUBLISHED）。published は降雨の日ごとの
        日別値表の最大1時間降水量（公表値。ない日は NaN）。"""
        formula = self.guideline.formula(district, self.cond.return_period)
        out: dict[str, float] = {}
        max1h = published.max() if published.notna().any() else None
        for d in rf.duration_depths(hourly_mm, lambda h: formula.depth(60 * h), DURATIONS, max1h):
            out |= dict(zip(self.duration_names(d.hours), (d.observed, d.plan, d.ratio)))
        out[NO_PUBLISHED] = int(published.isna().sum())
        return out

    def _check_total(self, where: str, what: str, total: float, daily_total: float, missing: list[str]) -> None:
        """記録（時間値・10分値）の合計 total と日別値の合計 daily_total を比べる。TOTAL_TOL_MM を超えて違うとき、
        欠測がなければ（missing が空）エラー、欠測があれば notices に知らせる（欠測は0mm として計算するので合わないことがある）。"""
        if abs(total - daily_total) <= TOTAL_TOL_MM:
            return
        if not missing:
            raise ValueError(f"{where}: {what}の合計 {total:.1f}mm が日別値の合計 {daily_total:.1f}mm と {TOTAL_TOL_MM:g}mm を超えて違い、"
                             "欠測もないため、計算を中止しました。"
                             f"気象庁の「過去の気象データ検索」で、この期間の日別値と{what}を照合してください。")
        self.notices.append(f"{where}: {'、'.join(missing)}があり、{what}の合計 {total:.1f}mm が日別値の合計 {daily_total:.1f}mm と"
                            "合いません（欠測は0mm として計算）")

    @staticmethod
    def _missing(what: str, cells: int, unit: str, daily_days: int) -> list[str]:
        """欠測の説明（「10分値の欠測 3 コマ」「日別値の欠測 1 日」）。欠測がなければ空。"""
        return ([f"{what}の欠測 {cells} {unit}"] if cells else []) + ([f"日別値の欠測 {daily_days} 日"] if daily_days else [])

    def duration_names(self, hours: int) -> tuple[str, str, str]:
        """継続時間 hours 時間の列の名前（実測の雨量、計画の雨量、計画との比）。"""
        return f"{hours}h_mm", f"{hours}h_{self.plan_label}_mm", f"{hours}h_{self.plan_label}比"

    def threshold_mm(self, district: str) -> float:
        """候補にする降雨の総量の下限 mm。池に入る水量は f·R·A·10 m3 を超えないので、これ未満は容量の fill_ratio に達しない。"""
        return self.cond.fill_ratio * self.facility(district).capacity / (self.cond.f * 10 * self.cond.area_ha)

    def left_after_dry(self, district: str) -> float:
        """計画降雨で満水にした池が、雨が止んで dry_gap 日後に残す貯留量の、容量に対する割合。
        降雨を dry_gap 日の無降水で区切ってよい（前の降雨の水がほとんど残らない）ことの裏付け。"""
        res = self.run(district, self.planned(district), self.cond.dry_gap * 24 * 60 // self.cond.tc_min)
        pond = res.facility.pond
        return pond.volume(res.routing.h[-1]) / pond.capacity

    # 1つの降雨を多地点で ----------------------------------------------------
    def event_at_stations(self, stations: list[str], start: str, end: str) -> pd.DataFrame:
        """同じ期間の10分値を各地点のモデル施設に入れる（千葉豪雨）。stations は study.json の events の stations。
        10分値の合計は日別値の合計と比べる（_check_total）。"""
        if not stations:
            raise ValueError(f"{start}〜{end} の降雨（study.json の events）の stations に、計算する地点がありません。地点名を書いてください。")
        twice = sorted({n for n in stations if stations.count(n) > 1})
        if twice:                               # 同じ地点の行が2つ並ぶ（表とレポートの地点の数が変わる）
            raise ValueError(f"{start}〜{end} の降雨（study.json の events）の stations に、同じ地点が2回以上あります: {'、'.join(twice)}。1回にしてください。")
        days = len(pd.date_range(start, end))
        rows = []
        for name in stations:
            d = self.guideline.district_of(name)
            t = self._tenmin(name, start, end)
            mm10, miss = _rain(t)
            daily = self.records.daily(name)
            self._check_daily_complete(name, daily, start, end)
            dd = daily[(daily["日付"] >= start) & (daily["日付"] <= end)]
            self._check_total(f"{name} {start}〜{end}", "10分値", mm10.sum(), dd["日合計_日別値_mm"].fillna(0.0).sum(),
                              self._missing("10分値", miss, "コマ", int(dd["日別値_品質"].isin(rf.NO_VALUE).sum())))
            row = {"地点": name, "地区": d, f"{days}日合計_mm": round(mm10.sum(), 1)}
            row.update(self.duration_cols(d, rf.hourly_from_10min(mm10), dd["最大1時間_日別値_mm"]))
            row["計画容量_m3"] = round(self.facility(d).capacity)
            row.update(self.storage_cols(d, mm10, 10, start))
            row["欠測コマ"] = miss
            rows.append(row)
        return pd.DataFrame(rows).sort_values("最大貯留率", ascending=False)

    # 長期記録 --------------------------------------------------------------
    def candidates(self, station: str, date_from: str, date_to: str) -> list[dict]:
        """日別値を降雨に区切り、開始日が date_from〜date_to にあり総量が基準以上の降雨を返す。
        区切りは記録全体で行い、期間の端（年の境目など）で降雨を途中から切らない。
        date_from〜date_to に日別値のない日があればエラー（その期間の降雨を黙って見落とさないように）。"""
        for day in (date_from, date_to):         # 日付は文字列で比べるので、書式が違うと候補が黙って0件になる
            if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
                raise ValueError(f"期間の日付「{day}」を読めません（study.json の partial_year の from・to など）。"
                                 "2026-01-01 の形で書いてください。")
        dd = self.records.daily(station)
        self._check_daily_complete(station, dd, date_from, date_to)
        thr = self.threshold_mm(self.guideline.district_of(station))
        out = []
        last = len(dd) - 1
        for e in rf.split_events(dd["日合計_日別値_mm"], self.cond.dry_gap):
            if not date_from <= _day(dd, e.start) <= date_to:
                continue
            if e.end == last and dd["日合計_日別値_mm"].to_numpy(dtype=float)[last] > 0:     # 記録の後も降り続いていれば、総量が小さく出る
                self.notices.append(f"{station} {_day(dd, e.start)}〜{_day(dd, last)}: 日別値の最後の日（{_day(dd, last)}）に雨があり、"
                                    "その後も降り続いた可能性があります（総量は記録の最後の日までで計算）")
            if e.total >= thr:
                seg = dd.iloc[e.start:e.end + 1]
                out.append({"開始": seg["日付"].iloc[0], "終了": seg["日付"].iloc[-1], "総量_mm": e.total,
                            "日数": e.days, "欠測日": int(seg["日別値_品質"].isin(rf.NO_VALUE).sum())})
        return out

    def screening(self, starts: dict[str, int], end_year: int, final: pd.DataFrame) -> pd.DataFrame:
        """長期記録で計算する降雨をどう絞り込んだか。地点ごとに、日別値の日数から溢水までの件数を絞り込みの順に数える。
        final は recompute_with_10min で置き換えた長期記録（最大貯留率は丸めない値）。
        列名はそのまま out/screening.tsv に出力される。"""
        rows = []
        for name, y0 in starts.items():
            date_from, date_to = f"{y0}-01-01", f"{end_year}-12-31"
            dd = self.records.daily(name)
            self._check_daily_complete(name, dd, date_from, date_to)
            d = self.guideline.district_of(name)
            inside = (dd["日付"] >= date_from) & (dd["日付"] <= date_to)
            events = [e for e in rf.split_events(dd["日合計_日別値_mm"], self.cond.dry_gap)
                      if date_from <= _day(dd, e.start) <= date_to]
            e = final[final["地点"] == name]
            cands = self.candidates(name, date_from, date_to)
            n_cand = len(cands)
            if n_cand != len(e):
                raise ValueError(f"{name}: 基準雨量以上の降雨 {n_cand} 件に対して、長期記録の計算結果が {len(e)} 件しかなく、表を作れません。"
                                 "計算の途中で降雨の数が合わなくなったもので、プログラムの誤りの可能性があります。"
                                 "study.json と studies/_jma/ をそのままにして、この表示を開発者に知らせてください。")
            thr = self.threshold_mm(d)
            rows.append({"地点": name, "地区": d, "開始年": y0, "終了年": end_year,
                         "対象期間の日数": int(inside.sum()),
                         "降水のあった日": int((inside & (dd["日合計_日別値_mm"] > 0)).sum()),
                         "降雨": len(events), "計画容量_m3": self.facility(d).capacity,
                         "容量に相当する雨量_mm": thr / self.cond.fill_ratio, "基準雨量_mm": thr,
                         "計算した降雨": n_cand, "計算した降雨の日数": sum(c["日数"] for c in cands),
                         f"最大貯留率{self.cond.fill_ratio:.0%}以上": int((e["最大貯留率"] >= self.cond.fill_ratio).sum()),
                         "10分値で計算し直した降雨": int((e["出所"] == "10分値").sum()),
                         "溢水": int((e["溢水量_m3"] > 0).sum())})
        return pd.DataFrame(rows)

    @staticmethod
    def _check_daily_complete(station: str, dd: pd.DataFrame, date_from: str, date_to: str):
        want = pd.date_range(date_from, date_to).strftime("%Y-%m-%d")
        lack = want.difference(pd.Index(dd["日付"]))
        if len(lack):
            raise ValueError(f"{station}: 日別値が {date_from}〜{date_to} のうち {len(lack)} 日ありません（{lack[0]} など）。"
                             f"amamizu fetch <検討フォルダ> daily --stations {station} でその年の日別値を取得するか、"
                             "study.json の期間（long_record の starts・end_year、partial_year）を取得済みの期間に合わせてください。")

    def missing_days(self, station: str, date_from: str, date_to: str) -> list[dict]:
        """日別値が欠測（値なし）の日と、その日を含むか前後 dry_gap 日以内にある降雨の組（1行が1組）。
        欠測の日は雨量0mm で、降雨を区切らない。欠測の日に降った雨は総量に入っていないので、
        計算しなかった降雨は基準雨量までの差を示す。近くに降雨がない日は、その日だけの降雨があってもわからない。"""
        dd = self.records.daily(station)
        thr = self.threshold_mm(self.guideline.district_of(station))
        g = self.cond.dry_gap
        events = rf.split_events(dd["日合計_日別値_mm"], g)
        out = []
        for i in np.where(dd["日合計_日別値_mm"].isna())[0].tolist():
            day = _day(dd, i)
            if not date_from <= day <= date_to:
                continue
            near = [e for e in events if e.start - g <= i <= e.end + g]
            for e in near:
                out.append({"地点": station, "日別値が欠測の日": day,
                            "降雨": f"{_day(dd, e.start)}〜{_day(dd, e.end)}", "観測できた総量_mm": e.total,
                            "判定": "計算した" if e.total >= thr
                            else f"計算していない（基準雨量 {thr:.1f}mm まで {thr - e.total:.1f}mm）"})
            if not near:
                out.append({"地点": station, "日別値が欠測の日": day, "降雨": "", "観測できた総量_mm": None,
                            "判定": f"前後{g}日に降雨なし"})
        return out

    # 欠測の時間 ------------------------------------------------------------
    def _hourly_of(self, station: str) -> pd.DataFrame | None:
        if station not in self._hourly:
            try:
                self._hourly[station] = self.records.hourly(station)
            except FileNotFoundError:
                self._hourly[station] = None
        return self._hourly[station]

    def _missing_spans(self, station: str, start: str, end: str, use_tenmin: bool) -> list[tuple] | None:
        """start〜end の日で欠測の時間（(始まり, 終わり) の Timestamp）。その日の記録がなければ None。
        use_tenmin なら10分値、そうでなければ時間値を見る。時間値の「時」、10分値の「時分」は区間の終わりの時刻。"""
        if use_tenmin:
            t = self.records.tenmin(station, start, end)
            if t is None:
                return None
            m = t[t["品質"].isin(rf.NO_VALUE)]
            ends = pd.to_datetime(m["日付"]) + pd.to_timedelta(
                m["時分"].str[:2].astype(int) * 60 + m["時分"].str[3:].astype(int), unit="min")
            return [(e - pd.Timedelta(minutes=10), e) for e in ends]
        h = self._hourly_of(station)
        if h is None:
            return None
        seg = h[(h["日付"] >= start) & (h["日付"] <= end)]
        if len(seg) != len(pd.date_range(start, end)) * 24:
            return None
        m = seg[seg["品質"].isin(rf.NO_VALUE)]
        ends = pd.to_datetime(m["日付"]) + pd.to_timedelta(m["時"].astype(int), unit="h")
        return [(e - pd.Timedelta(hours=1), e) for e in ends]

    def unobserved(self, station: str, start: str, end: str, source: str, margin: int | None = None) -> list[tuple[str, str]]:
        """降雨の期間と前後 margin 日（既定 dry_gap）のうち、欠測の時間。("YYYY-MM-DD HH:MM", 同) の組を時刻順に返す。

        期間内は計算に使った記録（source が「10分値」なら10分値、それ以外は時間値）の欠測。
        前後の日は、日別値が欠測（値なし）の日だけを見る。その日の時間値か10分値があればその欠測の時間、なければ終日。
        欠測の雨量は推測しない。どの時間が欠測かを示すだけ。"""
        margin = self.cond.dry_gap if margin is None else margin
        s0, e0 = pd.Timestamp(start), pd.Timestamp(end)
        inside = self._missing_spans(station, start, end, source == "10分値")
        if inside is None:                      # 計算に使った記録がそろっていない。欠測の時間がないものとして示さない
            raise ValueError(f"{station} {start}〜{end}: {source if source == '10分値' else '時間値'}がそろっていないので、欠測の時間を示せません。"
                             "amamizu fetch でこの期間の値を取得してください。")
        spans = list(inside)
        daily = self.records.daily(station).set_index("日付")["日合計_日別値_mm"]
        for day in pd.date_range(s0 - pd.Timedelta(days=margin), e0 + pd.Timedelta(days=margin)):
            ds = day.strftime("%Y-%m-%d")
            if s0 <= day <= e0 or ds not in daily.index or not pd.isna(daily[ds]):
                continue
            got = self._missing_spans(station, ds, ds, False)
            if got is None:
                got = self._missing_spans(station, ds, ds, True)
            spans += got if got is not None else [(day, day + pd.Timedelta(days=1))]
        merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        for a, b in sorted(spans):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        return [(a.strftime("%Y-%m-%d %H:%M"), b.strftime("%Y-%m-%d %H:%M")) for a, b in merged]

    def uncomputed_near_missing(self, station: str, date_from: str, date_to: str) -> list[dict]:
        """候補にならず計算しなかった降雨のうち、期間内か前後 dry_gap 日以内に日別値が欠測（値なし）の日があるもの。
        観測できた総量が基準未満でも、欠測の日の雨量はわからない。"""
        dd = self.records.daily(station)
        thr = self.threshold_mm(self.guideline.district_of(station))
        nan = set(np.where(dd["日合計_日別値_mm"].isna())[0].tolist())
        g = self.cond.dry_gap
        out = []
        for e in rf.split_events(dd["日合計_日別値_mm"], g):
            day = _day(dd, e.start)
            if e.total >= thr or not date_from <= day <= date_to:
                continue
            near = sorted(i for i in nan if e.start - g <= i <= e.end + g)
            if near:
                out.append({"地点": station, "開始": day, "終了": _day(dd, e.end), "観測できた総量_mm": e.total,
                            "基準雨量_mm": round(thr, 1), "日別値が欠測の日": [_day(dd, i) for i in near]})
        return out

    def long_record(self, starts: dict[str, int], end_year: int) -> pd.DataFrame:
        """時間値で長期記録を計算する。starts は地点ごとの標本の開始年。
        時間値の合計が日別値の合計と TOTAL_TOL_MM 以内で合わない降雨があれば止める（欠測のある降雨は除く）。"""
        rows = []
        for name, y0 in starts.items():
            d = self.guideline.district_of(name)
            hr = self.records.hourly(name)
            dd = self.records.daily(name).set_index("日付")
            for ev in self.candidates(name, f"{y0}-01-01", f"{end_year}-12-31"):
                seg = self._hourly_days(hr, name, ev["開始"], ev["終了"])
                rain, miss = _rain(seg)
                self._check_total(f"{name} {ev['開始']}〜{ev['終了']}", "時間値", rain.sum(), ev["総量_mm"],
                                  self._missing("時間値", miss, "時間", ev["欠測日"]))
                row = {"地点": name, "地区": d, "開始": ev["開始"], "終了": ev["終了"], "総量_mm": round(rain.sum(), 1),
                       "日別値の総量_mm": ev["総量_mm"], "出所": "時間値"}
                row.update(self.duration_cols(d, rain, dd.loc[ev["開始"]:ev["終了"], "最大1時間_日別値_mm"]))
                row.update(self.storage_cols(d, rain, 60, ev["開始"]))
                row["欠測時間"] = miss
                rows.append(row)
        return self._event_frame(rows, ["地点", "地区", "開始", "終了", "総量_mm", "日別値の総量_mm", "出所"], "欠測時間")

    @staticmethod
    def _hourly_days(hr: pd.DataFrame, name: str, start: str, end: str) -> pd.DataFrame:
        """時間値 hr の start〜end の行。日がそろっていなければ（1日24行）、取得のコマンドを示してエラーにする
        （そろっていない時間値を、短い降雨として計算しない）。"""
        seg = hr[(hr["日付"] >= start) & (hr["日付"] <= end)]
        if len(seg) != len(pd.date_range(start, end)) * 24:
            raise ValueError(f"{name} {start}〜{end}: 時間値がそろっていません。"
                             f"amamizu candidates <検討フォルダ> を実行してから、amamizu fetch <検討フォルダ> hourly --stations {name} "
                             "で取得してください（取得済みの日は取得しません）。")
        return seg

    def _event_frame(self, rows: list[dict], head: list[str], tail: str) -> pd.DataFrame:
        """降雨ごとの結果の表。基準雨量以上の降雨が1件もないときも、列のある0行の表にする
        （列のない表では、並べ替えや絞り込みが列名の KeyError で止まる）。列の順は行の項目の順と同じ。"""
        durations = [c for h in DURATIONS for c in self.duration_names(h)] + [NO_PUBLISHED]
        return pd.DataFrame(rows, columns=[*head, *durations, "最大貯留率", "溢水量_m3", tail])

    def recompute_with_10min(self, long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """開始日が tenmin_from 以降で最大貯留率が tenmin_fill 以上の降雨を、10分値で計算し直す。戻り値は（比較表, 置き換えた長期記録）。
        継続時間ごとの雨量は置き換えない（毎正時で区切った時間雨量にそろえるため）。"""
        long = long.copy()
        target = long[(long["開始"] >= self.cond.tenmin_from) & (long["最大貯留率"] >= self.cond.tenmin_fill)]
        rows = []
        for idx, r in target.iterrows():
            name, d = r["地点"], r["地区"]
            t = self.records.tenmin(name, r["開始"], r["終了"])
            if t is None:
                raise ValueError(f"{name} {r['開始']}〜{r['終了']}: 10分値で計算し直す降雨ですが、10分値がそろっていません。"
                                 f"amamizu fetch <検討フォルダ> tenmin --stations {name} --start {r['開始']} --end {r['終了']} で取得してください。")
            mm10, miss = _rain(t)
            x = self.storage_cols(d, mm10, 10, r["開始"])
            y = self.storage_cols(d, rf.hourly_from_10min(mm10), 60, r["開始"])
            rows.append({"地点": name, "開始": r["開始"], "終了": r["終了"], "10分値の合計_mm": round(mm10.sum(), 1),
                         "時間値の合計_mm": r["総量_mm"],
                         "10分値_最大貯留率": x["最大貯留率"], "10分値_溢水量_m3": x["溢水量_m3"],
                         "時間値_最大貯留率": r["最大貯留率"], "時間値_溢水量_m3": r["溢水量_m3"],
                         "10分値を1時間ごとに合計した場合_最大貯留率": y["最大貯留率"],
                         "10分値の欠測コマ": miss})
            long.loc[idx, "最大貯留率"] = x["最大貯留率"]
            long.loc[idx, "溢水量_m3"] = x["溢水量_m3"]
            long.loc[idx, "出所"] = "10分値"
        return pd.DataFrame(rows), long

    def trace(self, station: str, start: str, end: str, source: str):
        """1つの降雨の時系列（レポートの推移表とハイドログラフ用）。
        source が「10分値」なら10分値、それ以外は時間値を6等分する。
        戻り値は (10分ごとの雨量 mm, 出所の説明, 流入量 m3/s, 追跡結果, 施設)。"""
        d = self.guideline.district_of(station)
        if source == "10分値":
            mm, step = _rain(self._tenmin(station, start, end))[0], 10
            label = "10分値"
        else:
            mm, step = _rain(self._hourly_days(self.records.hourly(station), station, start, end))[0], 60
            label = "時間値（1時間を6等分）"
        obs, res = self.route_rain(d, mm, step, start)
        return obs.mmhr_10min() / 6, label, res.pond_inflow, res.routing, self.facility(d)

    def partial_year(self, stations: list[str], date_from: str, date_to: str) -> pd.DataFrame:
        """途中の年（2026年など）の候補降雨を10分値で計算する。"""
        rows = []
        for name in stations:
            d = self.guideline.district_of(name)
            dd = self.records.daily(name).set_index("日付")
            for ev in self.candidates(name, date_from, date_to):
                t = self._tenmin(name, ev["開始"], ev["終了"])
                mm10, miss = _rain(t)
                self._check_total(f"{name} {ev['開始']}〜{ev['終了']}", "10分値", mm10.sum(), ev["総量_mm"],
                                  self._missing("10分値", miss, "コマ", ev["欠測日"]))
                row = {"地点": name, "地区": d, "開始": ev["開始"], "終了": ev["終了"], "総量_mm": round(mm10.sum(), 1),
                       "出所": "10分値"}
                row.update(self.duration_cols(d, rf.hourly_from_10min(mm10), dd.loc[ev["開始"]:ev["終了"], "最大1時間_日別値_mm"]))
                row.update(self.storage_cols(d, mm10, 10, ev["開始"]))
                row["欠測コマ"] = miss
                rows.append(row)
        return self._event_frame(rows, ["地点", "地区", "開始", "終了", "総量_mm", "出所"], "欠測コマ").sort_values(["地点", "開始"])
