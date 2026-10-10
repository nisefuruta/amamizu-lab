"""調整池の設計の厳密計算（kind: design）。

流入（計画降雨または実測降雨と区域ごとの面積・流出係数、またはハイドログラフ）、池（柱状または水位・面積・容量の表）、
放流施設（長方形のオリフィスと洪水吐）を与えて貯留追跡計算を行う。オリフィスの寸法を与えて追跡するほか、
最大放流量が目標値になるオリフィスの寸法を求めることもできる。刻みは洪水到達時間 tc。
"""
import datetime
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..domain import infiltration as inf
from ..domain import rainfall as rf
from ..domain.design import required_pond_scale, size_orifice_for_peak
from ..domain.design_storm import planned_storm
from ..domain.intensity import IntensityFormula
from ..domain.outlet import CircleOrifice, Hydraulics, PumpTable, RectOrifice, Spillway, combine
from ..domain.pond import Facility, RoutingResult, route
from ..domain.runoff import rational_inflow


@dataclass(frozen=True)
class Catchment:
    """流域の区域。infiltration はこの区域の浸透施設（書いた区域が浸透処理区域。None は無処理区域）。"""
    name: str
    area_ha: float
    f: float
    infiltration: "Infiltration | None" = None

    def __post_init__(self):
        if self.area_ha <= 0:
            raise ValueError(f"区域「{self.name}」の面積 {self.area_ha} ha は正の値にしてください。")
        if not 0 < self.f <= 1:
            raise ValueError(f"区域「{self.name}」の流出係数 {self.f} は 0 より大きく 1 以下にしてください。")


@dataclass(frozen=True)
class Inflow:
    """追跡に入れる流入。nodes は tc ごとの節点の流入量 m3/s（節点0は0）、rain は区切りごとの降雨強度 mm/hr（なければ None）。
    start は節点0の日時（実測降雨のとき。計画降雨・ハイドログラフは None）。
    zones は区域ごとの流入（計画降雨・実測降雨。ハイドログラフは区域がないので空）。nodes はその合計。"""
    nodes: np.ndarray
    rain: np.ndarray | None
    tc_min: int
    start: datetime.datetime | None = None
    zones: tuple = ()

    def fitted(self, n: int) -> "Inflow":
        """節点を n 個にそろえた写し（足りなければ後ろを0で埋め、多ければ切る。区域ごとの流入も同じ）。"""
        def fit(x):
            return np.concatenate([x, np.zeros(max(0, n - len(x)))])[:n]
        return replace(self, nodes=fit(self.nodes), zones=tuple(ZoneInflow(z.catchment, fit(z.nodes)) for z in self.zones))


@dataclass(frozen=True)
class ZoneInflow:
    """区域1つの流入（節点の流入量 m3/s）。"""
    catchment: Catchment
    nodes: np.ndarray


@dataclass(frozen=True)
class ObservedRain:
    """実測降雨（気象庁の10分値か時間値）。mm は観測の区間ごとの雨量 mm（欠測と取得していない区間は0）、
    step_min は区間の長さ（10 か 60）、start は最初の区間の始まり、
    missing は欠測の時間、absent は取得していない時間（studies/_jma/ にない日など）の（始まり, 終わり）の並び。"""
    label: str
    start: datetime.datetime
    step_min: int
    mm: np.ndarray
    missing: tuple = ()
    absent: tuple = ()

    def mmhr_10min(self) -> np.ndarray:
        """10分ごとの強度 mm/hr。時間値は1時間の雨量を10分ずつ均等に配分する（貯留施設の検証と同じ）。"""
        if self.step_min == 10:
            return np.asarray(self.mm, dtype=float) * 6
        if self.step_min == 60:
            return rf.spread_hourly(self.mm)
        raise ValueError(f"区間の長さ {self.step_min} 分の実測降雨は扱えません（10分値か時間値）。")


def _add_up(arrays: list[np.ndarray]) -> np.ndarray:
    """流入量の配列を並びの順に足す（1つ以上）。0 から足し始めるのと同じ計算で、結果は新しい配列（区域ごとの配列と共有しない）。"""
    total = arrays[0] + 0.0
    for q in arrays[1:]:
        total = total + q
    return total


def _sum_inflow(storm, catchments: list[Catchment]) -> tuple[np.ndarray, tuple]:
    """区域ごとの合理式連続モデルの流入量の合計と、区域ごとの流入。"""
    if not catchments:
        raise ValueError("区域（catchments）がありません。流入の区域を1つ以上書いてください（WebGUI では「流入」のタブ）。")
    names = [c.name for c in catchments]
    if len(set(names)) < len(names):
        raise ValueError(f"区域の名前が重なっています（{'、'.join(n for n in dict.fromkeys(names) if names.count(n) > 1)}）。"
                         "区域ごとに別の名前にしてください。")
    each = [rational_inflow(storm, c.f, c.area_ha) for c in catchments]
    return _add_up(each), tuple(ZoneInflow(c, q) for c, q in zip(catchments, each))


def planned_inflow(formula: IntensityFormula, catchments: list[Catchment], tc_min: int, shape: str,
                   hours: int = 24, round_digits: int | None = None) -> Inflow:
    """計画降雨（shape: rear 後方集中型、center 中央集中型）を作り、区域ごとの合理式連続モデルの流入量を合計する。
    round_digits を与えると、降雨強度を小数点以下その桁に四捨五入してから流入量を求める。"""
    storm = planned_storm(formula, shape, hours, tc_min, round_digits)
    nodes, zones = _sum_inflow(storm, catchments)
    return Inflow(nodes, storm, tc_min, zones=zones)


def observed_storm(obs: ObservedRain, tc_min: int) -> np.ndarray:
    """実測降雨を観測どおりの順序で tc ごとの強度 mm/hr にまとめる（雨量を tc ごとに合計）。tc は10の倍数。"""
    return rf.blocks_from_10min(obs.mmhr_10min(), tc_min)


def observed_inflow(obs: ObservedRain, catchments: list[Catchment], tc_min: int) -> Inflow:
    """実測降雨を tc ごとの強度にまとめ（observed_storm）、区域ごとの合理式連続モデルの流入量を合計する。"""
    storm = observed_storm(obs, tc_min)
    nodes, zones = _sum_inflow(storm, catchments)
    return Inflow(nodes, storm, tc_min, obs.start, zones)


def hydrograph_inflow(times_min, q_m3s, tc_min: int, rain=None) -> Inflow:
    """ハイドログラフ（区切りの終わりの時刻と流入量）を節点の流入量にする。時刻は tc, 2tc, 3tc … であること。"""
    t = np.asarray(times_min, dtype=float)
    want = tc_min * np.arange(1, len(t) + 1)
    if len(t) == 0 or np.any(np.abs(t - want) > 1e-9):
        bad = int(np.argmax(np.abs(t - want) > 1e-9)) if len(t) else 0
        raise ValueError(f"ハイドログラフの時刻が tc＝{tc_min} 分ごと（{tc_min}, {2 * tc_min} …）になっていません"
                         f"（値の {bad + 1} 行目）。時刻の列を tc_min ごとの区切りの終わりの時刻にするか、"
                         "study.json の tc_min をハイドログラフの時刻の刻みに合わせてください。")
    q = np.asarray(q_m3s, dtype=float)
    if np.any(np.isnan(q)) or np.any(q < 0):
        raise ValueError("ハイドログラフの流入量に空欄か負の値があります。流入量の列には、すべての行に0以上の値を書いてください。")
    r = None if rain is None else np.asarray(rain, dtype=float)
    if r is not None and (np.any(np.isnan(r)) or np.any(r < 0)):
        raise ValueError("ハイドログラフの降雨強度に空欄か負の値があります。降雨強度の列には、すべての行に0以上の値を書いてください。")
    return Inflow(np.concatenate([[0.0], q, [0.0]]), r, tc_min)


def hm(minutes: float) -> str:
    """経過時間（分）の表記 H:MM（節点の時刻は tc の整数倍なので、分は整数に丸める）。"""
    h, m = divmod(int(round(minutes)), 60)
    return f"{h}:{m:02d}"


def clock(start: datetime.datetime | None, minutes: float) -> str:
    """節点の時刻の表記。実測降雨（start が節点0の日時）なら日時、計画降雨・ハイドログラフなら経過時間 H:MM。
    グラフの凡例・要約・WebGUI で同じ表記にする。"""
    if start is None:
        return hm(minutes)
    return f"{start + datetime.timedelta(minutes=float(minutes)):%Y-%m-%d %H:%M}"


def rain_table(storm, tc_min: int, begin: datetime.datetime | None = None) -> pd.DataFrame:
    """計画降雨の表（区間ごとの時刻、降雨強度 mm/h、雨量 mm）。
    begin（実測降雨の最初の区間の始まり）を与えると、先頭に区間の日時の列を足す（区間・時・分は begin からの経過時間）。"""
    storm = np.asarray(storm, dtype=float)
    start = np.arange(len(storm)) * tc_min
    dated = {}
    if begin is not None:
        step = datetime.timedelta(minutes=tc_min)
        dated["区間の日時"] = [f"{begin + k * step:%Y-%m-%d %H:%M}〜{begin + (k + 1) * step:%H:%M}"
                          for k in range(len(storm))]
    return pd.DataFrame(dated | {
        "区間": [f"{hm(int(s))}〜{hm(int(s) + tc_min)}" for s in start],
        "時": (start // 60).astype(int),
        "分": [f"{int(s % 60)}〜{int(s % 60) + tc_min}" for s in start],
        "降雨強度(mm/h)": storm,
        "雨量(mm)": storm * tc_min / 60,
    })


def formula_text(formula: IntensityFormula, kind: str | None) -> str:
    """強度式の書き方（レポートや要約に使う）。kind は FORMULA_TYPES の種類（手引の式なら None）。"""
    from ..domain.intensity import FORMULA_TYPES
    name = FORMULA_TYPES[kind or "cleveland"][0]
    a, b, n = formula.a, formula.b, formula.n
    body = {"talbot": f"{a:g}/(t{b:+g})", "sherman": f"{a:g}/t^{n:g}", "kuno_ishiguro": f"{a:g}/(t^0.5{b:+g})"}.get(
        kind or "cleveland", f"{a:g}/(t^{n:g}{b:+g})")
    return f"{name}式 r＝{body}"


@dataclass(frozen=True)
class Infiltration:
    """浸透（一定量差し引き）1単位。rate_m3s は設計浸透量、void_m3 は空隙貯留量、facilities は内訳（施設で書いたとき）、
    intensity_mmh は設計浸透強度（強度で書いたとき）。"""
    rate_m3s: float
    void_m3: float
    facilities: tuple = ()
    intensity_mmh: float | None = None


@dataclass(frozen=True)
class TreatedZone:
    """設計浸透量を差し引く単位（浸透処理区域）。区域ごとの浸透なら区域1つ、最上位の浸透なら全体（name は WHOLE）。
    area_ha はハイドログラフの全体なら None。nodes は差し引く前、after は差し引いた後の流入量。"""
    name: str
    area_ha: float | None
    infiltration: Infiltration
    nodes: np.ndarray
    after: np.ndarray


WHOLE = "全体"


def infiltration_units(inflow: Inflow, infiltration: Infiltration | None) -> list[tuple]:
    """設計浸透量を差し引く単位の並び（名前, 面積 ha, 差し引く前の流入量, 浸透）。どの流入からどの浸透を差し引くかはここだけで決める。
    最上位の浸透（infiltration）は全体を1単位にする。区域の浸透は区域ごと（浸透のない区域は浸透 None の単位）。"""
    zoned = any(z.catchment.infiltration for z in inflow.zones)
    if infiltration is not None and zoned:
        raise ValueError("infiltration は最上位（区域全体で1つ）か区域ごと（catchments の各区域）のどちらか一方に書いてください。")
    if infiltration is not None:
        area = sum(z.catchment.area_ha for z in inflow.zones) if inflow.zones else None
        return [(WHOLE, area, inflow.nodes, infiltration)]
    if zoned:
        return [(z.catchment.name, z.catchment.area_ha, z.nodes, z.catchment.infiltration) for z in inflow.zones]
    return []


def infiltrated(inflow: Inflow, infiltration: Infiltration | None, dt: float) -> tuple[np.ndarray, tuple]:
    """池に入れる流入量（浸透を差し引いた後）と、浸透処理区域の内訳（TreatedZone の並び）。
    単位ごとに一定量差し引き（subtract_constant）を行い、浸透のない区域の流入はそのまま足す（手引 第10条 図10－3）。"""
    units = infiltration_units(inflow, infiltration)
    if not units:
        return inflow.nodes, ()
    treated, parts = [], []
    for name, area, q, it in units:
        if it is None:
            parts.append(q)
            continue
        after = inf.subtract_constant(q, it.rate_m3s, it.void_m3, dt)
        treated.append(TreatedZone(name, area, it, q, after))
        parts.append(after)
    return _add_up(parts), tuple(treated)


FACTOR_NAMES = {"clogging": "目づまり", "groundwater": "地下水位", "safety": "安全率 α"}    # 浸透施設の影響係数（factors）の項目


def build_infiltration(spec: dict | None, area_ha: float | None = None) -> Infiltration | None:
    """study.json の infiltration（最上位か区域の）から浸透を作る。書かなければ浸透なし（None）。

    rate_m3s（設計浸透量。void_storage_m3 も書ける）、intensity_mmh（設計浸透強度 mm/h。設計浸透量は
    強度 × area_ha × 10 m3/h。void_storage_m3 も書ける）、facilities（浸透施設）のどれか一つを書く。
    area_ha は浸透処理区域の面積（区域の浸透なら区域の面積、最上位なら区域の合計。ハイドログラフは None）。
    施設は name、比浸透量（specific_m2 か、shape の種類と寸法）、permeability（value と unit）、count、factors（目づまり clogging、地下水位 groundwater、安全率 safety）、
    void_parts（1単位あたりの部分ごとの volume_m3 と porosity_pct。任意）。
    """
    if spec is None:
        return None
    kinds = [k for k in ("rate_m3s", "intensity_mmh", "facilities") if k in spec]
    if len(kinds) != 1:
        raise ValueError("infiltration には rate_m3s、intensity_mmh、facilities のどれか一つを書いてください。")
    if kinds == ["rate_m3s"]:
        return Infiltration(spec["rate_m3s"], spec.get("void_storage_m3", 0.0))
    if kinds == ["intensity_mmh"]:
        if area_ha is None:
            raise ValueError("浸透強度（intensity_mmh）から設計浸透量を求めるには区域の面積が要ります"
                             "（ハイドログラフの流入では rate_m3s か facilities で書いてください）。")
        return Infiltration(inf.design_rate_m3s(spec["intensity_mmh"], area_ha), spec.get("void_storage_m3", 0.0),
                            intensity_mmh=spec["intensity_mmh"])
    if "void_storage_m3" in spec:
        raise ValueError("facilities を書くときは、空隙貯留量は各施設の void_parts に書いてください（void_storage_m3 は使わない）。")
    facs = []
    for f in spec["facilities"]:
        name = f.get("name", "浸透施設")
        if ("specific_m2" in f) == ("shape" in f):
            raise ValueError(f"浸透施設「{name}」には specific_m2 か shape のどちらか一つを書いてください。")
        if "shape" in f:
            s = f["shape"]
            k = inf.specific_infiltration(s["kind"], s["H_m"], s.get("W_m"), s.get("D_m"), s.get("L_m"))
        else:
            k = f["specific_m2"]
        unit = f["permeability"]["unit"]
        if unit not in inf.PERMEABILITY_UNITS:
            raise ValueError(f"飽和透水係数の単位「{unit}」は扱えません（{', '.join(inf.PERMEABILITY_UNITS)}）。")
        void = sum(p["volume_m3"] * p["porosity_pct"] / 100 for p in f.get("void_parts", []))
        c = f["factors"]
        lack = [label for key, label in FACTOR_NAMES.items() if key not in c]
        if lack:
            raise ValueError(f"浸透施設「{name}」の影響係数（{'、'.join(lack)}）がありません。"
                             f"factors に {'、'.join(f'{k}（{v}）' for k, v in FACTOR_NAMES.items())} を書いてください。")
        facs.append(inf.Facility(name, k, f["permeability"]["value"] * inf.PERMEABILITY_UNITS[unit], f["count"],
                                 c["clogging"], c["groundwater"], c["safety"], void))
    return Infiltration(sum(x.rate_m3_h for x in facs) / 3600, sum(x.void_m3 for x in facs), tuple(facs))


@dataclass
class DesignResult:
    table: pd.DataFrame | None       # 時刻ごとの結果（run_design の with_table のときだけ）
    summary: dict            # 最大値など
    facility: Facility
    routing: RoutingResult
    inflow: Inflow
    tc_min: int
    orifices: list                   # オリフィス（RectOrifice、CircleOrifice）。敷高の低い順
    spillway: Spillway | None
    pump: PumpTable | None
    allowable_q: float | None
    infiltration: Infiltration | None = None     # 最上位の浸透（区域全体で1つ）
    treated: tuple = ()                           # 浸透処理区域ごとの差し引きの内訳（TreatedZone）
    pond_inflow: np.ndarray | None = None         # 池に入れた流入量（浸透を差し引いた後。丸める前。節点は table と同じ）


# オリフィスの形ごとの寸法の項目
SHAPE_SIZES = {"rect": ("width_m", "height_m"), "circle": ("diameter_m",)}


def _shape_of(spec: dict) -> str:
    shape = spec.get("shape", "rect")
    if shape not in SHAPE_SIZES:
        raise ValueError(f"オリフィスの形「{shape}」は扱えません。shape は rect（長方形）か circle（円形）にしてください。")
    wrong = [k for s, keys in SHAPE_SIZES.items() if s != shape for k in keys if k in spec]
    if wrong:
        raise ValueError(f"オリフィスの形 {shape} には {', '.join(wrong)} を書かないでください。")
    return shape


def _hole(spec: dict, hy: Hydraulics):
    shape = _shape_of(spec)
    lack = [k for k in SHAPE_SIZES[shape] if k not in spec]
    if lack:
        raise ValueError(f"オリフィスの寸法 {', '.join(lack)} がありません（mode: route）。")
    if shape == "rect":
        return RectOrifice(spec["sill_m"], spec["width_m"], spec["height_m"], hy)
    return CircleOrifice(spec["sill_m"], spec["diameter_m"], hy)


TARGET_Q_NEEDED = "size_orifice には目標の最大放流量 target_q_m3s（正の値）が必要です。"


def check_options(pond, orifices: list[dict], spillway: dict | None, end_min: int, tc: int, inflow_nodes: int,
                  mode: str = "route", target_q: float | None = None, allowable_q: float | None = None,
                  storage_margin: float | None = None, pump: dict | None = None) -> None:
    """計算の前の確かめ（計算はしない。WebGUI では入力のたびに呼ぶ）。誤りは ValueError。

    inflow_nodes は流入の節点の数（節点0と、最後の三角形が0に戻る節点を含む）。
    """
    if end_min % tc:
        raise ValueError(f"追跡の終了時刻 {end_min} 分が tc＝{tc} 分で割り切れません。end_min を {tc} 分の倍数にしてください。")
    if end_min // tc + 1 < inflow_nodes - 1:
        raise ValueError(f"追跡の終了時刻 {end_min} 分が、流入の期間 {(inflow_nodes - 2) * tc} 分より短いです。"
                         f"end_min を {(inflow_nodes - 2) * tc} 分以上にしてください。")
    if not orifices:
        raise ValueError("オリフィスがありません。study.json の orifice（WebGUI では「放流施設」のタブ）に1つ以上書いてください。")
    for spec in orifices:
        if spec["sill_m"] < pond.bottom:
            raise ValueError(f"オリフィスの敷高 {spec['sill_m']} m が池底 {pond.bottom} m より低いです。敷高を池底以上にしてください。")
    if pump:
        start, q0 = pump["levels_m"][0], pump["q_m3s"][0]
        if start < pond.bottom or (start == pond.bottom and q0 > 0):
            raise ValueError(f"ポンプの表の最初の水位 {start} m が池底 {pond.bottom} m より低いか、池底で排水しています"
                             "（池が空のときは排水できません）。")
    lowest = min(spec["sill_m"] for spec in orifices)
    if spillway and spillway["crest_m"] < lowest:
        raise ValueError(f"洪水吐の敷高 {spillway['crest_m']} m がオリフィスの敷高 {lowest} m より低いです。"
                         "洪水吐の敷高を、最も低いオリフィスの敷高以上にしてください。")
    if mode == "route":
        if target_q is not None:
            raise ValueError("target_q_m3s は mode: size_orifice のときだけ書いてください。")
        for spec in orifices:
            shape = _shape_of(spec)
            lack = [k for k in SHAPE_SIZES[shape] if k not in spec]
            if lack:
                raise ValueError(f"オリフィスの寸法 {', '.join(lack)} がありません（mode: route）。")
    elif mode == "size_orifice":
        if len(orifices) != 1:
            raise ValueError("size_orifice で寸法を求めるオリフィスは1つです（orifice に書いてください）。")
        spec = orifices[0]
        shape = _shape_of(spec)
        given = [k for k in SHAPE_SIZES[shape] if k in spec]
        if given:
            raise ValueError(f"size_orifice ではオリフィスの寸法を求めるので、orifice に {', '.join(given)} を書かないでください。")
        if target_q is None or target_q <= 0:
            raise ValueError(TARGET_Q_NEEDED)
        if allowable_q is not None:
            raise ValueError("size_orifice では target_q_m3s が許容放流量になるので、allowable_q_m3s を書かないでください。")
    else:
        raise ValueError(f"mode「{mode}」は扱えません。route（寸法を決めて追跡する）か size_orifice（オリフィスの寸法を求める）にしてください。")
    if storage_margin is not None and storage_margin < 0:
        raise ValueError(f"容量の余裕 {storage_margin} m3 は0以上にしてください。")


def build_outlets(orifices: list[dict], spillway: dict | None, pump: dict | None, hy: Hydraulics,
                  with_holes: bool = True):
    """放流施設の部品を作る。戻り値は（オリフィスの並び（敷高の低い順）, 洪水吐, ポンプ）。
    with_holes が False ならオリフィスは作らない（size_orifice で寸法を求めるとき）。"""
    holes = sorted((_hole(spec, hy) for spec in orifices), key=lambda o: o.sill) if with_holes else []
    spill_obj = Spillway(spillway["crest_m"], spillway["width_m"], hy) if spillway else None
    pump_obj = PumpTable(tuple(pump["levels_m"]), tuple(pump["q_m3s"])) if pump else None
    return holes, spill_obj, pump_obj


def run_design(inflow: Inflow, pond, orifices: list[dict], spillway: dict | None, end_min: int, mode: str = "route",
               target_q: float | None = None, *, hy: Hydraulics,
               allowable_q: float | None = None, pump: dict | None = None,
               storage_margin: float | None = None, infiltration: Infiltration | None = None,
               with_table: bool = False) -> DesignResult:
    """貯留追跡計算を行う。

    orifices はオリフィスの並び。各オリフィスは {"sill_m", "shape"（rect／circle、省略時 rect）, 寸法}。
    寸法は rect なら width_m と height_m、circle なら diameter_m。2つ書けば2段のオリフィス（放流量は和）。
    mode が size_orifice ならオリフィスは1つで、寸法は書かない（rect は正方形の一辺、circle は直径を求める）。
    spillway は {"crest_m", "width_m"}、pump は {"levels_m": [...], "q_m3s": [...]}（水位と排水量の表）。
    allowable_q は許容放流量（route のとき任意）。size_orifice では target_q が許容放流量になる。
    storage_margin は判定に使う容量の余裕 m3（任意。容量 ≧ 最大貯留量＋余裕 なら O.K）。
    放流量・判定の放流量は、オリフィス・洪水吐・ポンプの和。
    追跡の前に、浸透（最上位の infiltration は全体から、区域の浸透はその区域の流入から）の設計浸透量を差し引く
    （一定量差し引きモデル。infiltrated）。
    with_table なら時刻ごとの結果の表（DesignResult.table）も作る（route.csv・グラフ・WebGUI の表に使う）。
    """
    tc = inflow.tc_min
    check_options(pond, orifices, spillway, end_min, tc, len(inflow.nodes), mode, target_q, allowable_q, storage_margin, pump)
    shortage = None
    fit = inflow.fitted(end_min // tc + 1)
    raw = fit.nodes
    nodes, treated = infiltrated(fit, infiltration, tc * 60)
    holes, spill_obj, pump_obj = build_outlets(orifices, spillway, pump, hy, with_holes=(mode == "route"))
    others = tuple(x for x in (spill_obj, pump_obj) if x is not None)
    dt = tc * 60
    if mode == "route":
        fac = Facility(pond, combine(tuple(holes) + others))
        r = route(fac, nodes, capped=True, dt=dt)
    else:
        if target_q is None:                 # check_options で確かめてある（型の上でも None でないことを示す）
            raise ValueError(TARGET_Q_NEEDED)
        spec = orifices[0]
        shape = "square" if _shape_of(spec) == "rect" else "circle"
        hole, r = size_orifice_for_peak(nodes, pond, spec["sill_m"], target_q, dt, others, hy=hy, shape=shape, allow_spill=True)
        if r.spill > 0:               # 池の容量が足りない: 水深はそのままで、溢れなくなる面積の倍率を求め、その池で追跡する
            k, hole, big, r = required_pond_scale(nodes, pond, spec["sill_m"], target_q, dt, others, hy=hy, shape=shape)
            shortage = {"面積の倍率": k, "入力の容量_m3": pond.capacity, "必要な容量_m3": big.capacity,
                        "不足量_m3": big.capacity - pond.capacity}
            pond = big
        holes = [hole]
        fac = Facility(pond, combine((hole,) + others))
        allowable_q = target_q
    table = _table(inflow, raw, r, pond, tc) if with_table else None
    summary = _summary(nodes, r, pond, holes, tc, allowable_q, storage_margin)
    if shortage:
        summary["池の不足"] = shortage
        summary["判定"]["溢水"] = f"N.G（入力の池では溢れる。面積を {shortage['面積の倍率']:.3f} 倍にすると溢れない）"
        summary["判定"]["総合"] = "N.G"
    if treated:
        zones = [] if infiltration is not None else list(treated)    # 区域の名前が「全体」でも区域ごとの浸透として扱う
        if table is not None:
            at = list(table.columns).index("流入量(m3/s)") + 1
            for t in zones:
                table.insert(at, f"{t.name}の流入量(m3/s)", np.round(t.nodes, 4))
                table.insert(at + 1, f"{t.name}の浸透後流入量(m3/s)", np.round(t.after, 4))
                at += 2
            table.insert(at, "浸透後流入量(m3/s)", np.round(nodes, 4))
        if zones:
            summary["浸透処理区域"] = [{"名前": t.name, "面積_ha": t.area_ha, "設計浸透量_m3s": t.infiltration.rate_m3s,
                                   "最大流入量_m3s": float(t.nodes.max()), "浸透後の最大流入量_m3s": float(t.after.max())}
                                  for t in zones]
            summary["浸透処理面積率_%"] = sum(t.area_ha for t in zones) / sum(z.catchment.area_ha for z in fit.zones) * 100
        summary["浸透前の最大流入量_m3s"] = float(raw.max())
    return DesignResult(table, summary, fac, r, inflow, tc, holes, spill_obj, pump_obj, allowable_q, infiltration, treated,
                        nodes)


OVERDRAW_TIMES = 5      # 池が空になった時刻を知らせる文に並べる数（ほかは回数を示す）


def overdraw_note(r: RoutingResult, tc: int, limit: int) -> str:
    """刻みの中で池が空になった時刻（route の overdraw の節点）を知らせる文。多ければ先頭 limit 個と、ほかの回数。
    その刻みの放流量は池にあった水と流入の和（route の説明）。"""
    times = [f"{i * tc} 分" for i, _ in r.overdraw[:limit]]
    if len(r.overdraw) > limit:
        times.append(f"ほか {len(r.overdraw) - limit} 回")
    return f"{'、'.join(times)}に池が空になりました（貯留量 0 m^3）。"


def _table(inflow: Inflow, nodes, r: RoutingResult, pond, tc: int) -> pd.DataFrame:
    n = len(nodes)
    cols = {"計算時間(min)": np.arange(n) * tc}
    if inflow.rain is not None:       # 節点 k の降雨強度は、その節点で終わる区切りのもの
        rain = np.zeros(n)
        m = min(len(inflow.rain), n - 1)
        rain[1:1 + m] = inflow.rain[:m]
        cols["降雨強度(mm/h)"] = np.round(rain, 3)
    cols |= {"流入量(m3/s)": np.round(nodes, 4), "放流量(m3/s)": np.round(r.out, 4), "水位(m)": np.round(r.h, 4)}
    if pond.has_area:                 # 池の表に面積がなければ、水面積の列は出力しない
        cols["水面積(m2)"] = np.round([pond.area_at(x) for x in r.h], 2)
    return pd.DataFrame(cols | {"貯留量(m3)": np.round(pond.volume(r.h), 2)})


def _summary(nodes, r: RoutingResult, pond, holes, tc: int, allowable_q: float | None,
             storage_margin: float | None) -> dict:
    iq, io, ih = int(np.argmax(nodes)), int(np.argmax(r.out)), int(np.argmax(r.h))
    qmax, spare = float(r.out[io]), pond.capacity - r.vmax
    judge = {"溢水": "O.K" if r.spill <= 0 else "N.G（溢れる）"}
    if allowable_q is not None:
        judge["放流量"] = "O.K（許容放流量以下）" if qmax <= allowable_q * (1 + 1e-9) else "N.G（許容放流量を超える）"
    if storage_margin is not None:
        judge["容量の余裕"] = (f"O.K（余裕 {storage_margin:g} m^3 以上）" if spare >= storage_margin
                          else f"N.G（余裕が {storage_margin:g} m^3 に足りない）")
    judge["総合"] = "O.K" if all(v.startswith("O.K") for v in judge.values()) else "N.G"
    return {
        "最大流入量_m3s": float(nodes[iq]), "最大流入量の時刻_min": iq * tc,
        "最大放流量_m3s": float(r.out[io]), "最大放流量の時刻_min": io * tc,
        "最大水位_m": float(r.h[ih]), "最大水位の時刻_min": ih * tc,
        "最大貯留量_m3": r.vmax, "容量_m3": pond.capacity, "満水位_m": pond.hwl,
        "最大貯留率": r.vmax / pond.capacity, "溢水量_m3": r.spill,
        "余裕量_m3": spare, "余裕率_%": spare / pond.capacity * 100,
        "オリフィス": [(h.sill, h.describe()) for h in holes],
        "許容放流量_m3s": allowable_q, "判定": judge,
    }
