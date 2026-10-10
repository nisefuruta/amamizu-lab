"""貯留施設と貯留追跡計算。"""
import math
from dataclasses import dataclass
from typing import ClassVar, Generic, Protocol, TypeVar, overload

import numpy as np
import numpy.typing as npt

from .outlet import Outlet

DEFAULT_TC_MIN = 10           # 洪水到達時間 tc（計算の刻み）の既定 分。study.json・amamizu rain・WebGUI の既定はすべてこの値
DT = DEFAULT_TC_MIN * 60.0    # 計算の刻み 秒。tc を変えるときは route の dt に tc×60 を渡す

FloatArray = npt.NDArray[np.float64]


class Pond(Protocol):
    """貯留施設が持つもの（貯留追跡 route と寸法の探索が使う）。

    計算の基準は水位と容量の関係 volume(h)。容量は水位の増加関数で、池底で0、満水位より上も rise_rate で増え続ける。
    """

    @property
    def bottom(self) -> float: ...

    @property
    def hwl(self) -> float: ...

    @property
    def capacity(self) -> float: ...

    @property
    def has_area(self) -> bool: ...

    @overload
    def volume(self, h: float) -> float: ...

    @overload
    def volume(self, h: FloatArray) -> FloatArray: ...

    def level(self, v: float) -> float: ...

    def area_at(self, h: float) -> float: ...

    def rise_rate(self) -> float: ...

    def scaled(self, k: float) -> "Pond": ...


@dataclass
class PrismaticPond:
    """柱状の貯留施設（水深×一定の面積）。水位は池底からの水深なので、池底は 0 m（容量は 面積×水位）。
    標高で書く池は TablePond を使う。"""
    area: float     # m2
    hwl: float      # 満水位（深さ）m

    bottom: ClassVar[float] = 0.0    # 池底（水深で表すので 0。容量 面積×水位 と矛盾しないよう、変えられる項目にしない）

    def __post_init__(self) -> None:
        if not self.area > 0 or not self.hwl > 0:
            raise ValueError(f"柱状の池の面積 {self.area} m^2 と満水位 {self.hwl} m は正の値にしてください。")

    @property
    def capacity(self) -> float:
        return self.area * self.hwl

    @overload
    def volume(self, h: float) -> float: ...

    @overload
    def volume(self, h: FloatArray) -> FloatArray: ...

    def volume(self, h: float | FloatArray) -> float | FloatArray:
        return self.area * h

    def level(self, v: float) -> float:
        return v / self.area

    def area_at(self, h: float) -> float:
        return self.area

    def rise_rate(self) -> float:
        """水位 1 m あたりの容量の増え方 m3/m（柱状なので面積そのもの）。"""
        return self.area

    has_area: ClassVar[bool] = True

    def scaled(self, k: float) -> "PrismaticPond":
        """水深（満水位）はそのままで、面積を k 倍した池。"""
        return PrismaticPond(self.area * k, self.hwl)


@dataclass(init=False)
class TablePond:
    """水位・面積・容量の表で表す貯留施設（段のある池を含む）。表の間は線形補間する。

    levels は水位 m（昇順）、volumes は容量 m3（昇順。最初の水位で0）、areas は水面積 m2（なければ None）。
    容量がなく面積だけあるときは from_areas で容量を求める。
    満水位 hwl を省略すると表の最上段。表より上は、最後の区間の容量の増え方（rise_rate）のまま立ち上がるものとして扱う。
    計算の基準は容量なので、面積の列の有無で延長を変えない（面積の列は水面積の表示にだけ使う）。容量は水位とともに増えるので
    （__init__ で確かめる）、この増え方は正で、表より上でも容量は水位の増加関数になる。
    """
    levels: FloatArray
    areas: FloatArray | None
    volumes: FloatArray
    hwl: float

    def __init__(self, levels: npt.ArrayLike, areas: npt.ArrayLike | None, volumes: npt.ArrayLike, hwl: float | None = None):
        # hwl は省略時に表の最上段にするので、項目は float、引数は float | None（dataclass の自動の __init__ を使わない）
        self.levels = np.asarray(levels, dtype=float)
        self.areas = None if areas is None else np.asarray(areas, dtype=float)
        self.volumes = np.asarray(volumes, dtype=float)
        rows = {len(self.levels), len(self.volumes)} | ({len(self.areas)} if self.areas is not None else set())
        if len(rows) > 1 or len(self.levels) < 2:
            raise ValueError("池の表は、水位・面積・容量が同じ行数で2行以上必要です。")
        if np.any(np.diff(self.levels) <= 0):
            raise ValueError("池の表の水位は、下の行ほど高くしてください（同じ水位の行は置けません）。")
        if np.any(np.diff(self.volumes) <= 0):
            raise ValueError("池の表の容量は、水位が高いほど大きくしてください。")
        if self.volumes[0] != 0:
            raise ValueError(f"池の表の最初の行（水位 {self.levels[0]} m）の容量は0にしてください（{self.volumes[0]}）。")
        if self.areas is not None and np.any(self.areas < 0):
            raise ValueError("池の表の面積に負の値があります。面積は0以上にしてください。")
        self.hwl = float(self.levels[-1]) if hwl is None else hwl
        if not self.levels[0] < self.hwl <= self.levels[-1]:
            raise ValueError(f"満水位 {self.hwl} m が池の表の範囲（{self.levels[0]}〜{self.levels[-1]} m）にありません。")

    @classmethod
    def from_areas(cls, levels: npt.ArrayLike, areas: npt.ArrayLike, hwl: float | None = None) -> "TablePond":
        """水位と面積の表から、容量を平均断面法（V[i] = V[i-1] + (A[i-1] + A[i]) / 2 × (h[i] - h[i-1])）で求めた池。"""
        h, a = np.asarray(levels, dtype=float), np.asarray(areas, dtype=float)
        if len(h) != len(a) or len(h) < 2:
            raise ValueError("池の表は、水位・面積・容量が同じ行数で2行以上必要です。")
        if np.any((a[:-1] <= 0) & (a[1:] <= 0)):
            i = int(np.argmax((a[:-1] <= 0) & (a[1:] <= 0)))
            raise ValueError(f"面積から容量を求めるには、隣り合う行の面積の少なくとも一方を正にしてください"
                             f"（水位 {h[i]:g} m と {h[i + 1]:g} m の行）。")
        return cls(h, a, np.concatenate([[0.0], np.cumsum((a[:-1] + a[1:]) / 2 * np.diff(h))]), hwl)

    @property
    def has_area(self) -> bool:
        return self.areas is not None

    def scaled(self, k: float) -> "TablePond":
        """水位（満水位）はそのままで、面積と容量を k 倍した池。"""
        return TablePond(self.levels, None if self.areas is None else self.areas * k, self.volumes * k, self.hwl)

    def rise_rate(self) -> float:
        """表より上の、水位 1 m あたりの容量の増え方 m3/m（最後の区間の容量の増え方。表の容量と途切れずにつながる）。"""
        return float((self.volumes[-1] - self.volumes[-2]) / (self.levels[-1] - self.levels[-2]))

    @property
    def bottom(self) -> float:
        return float(self.levels[0])

    @property
    def capacity(self) -> float:
        return float(self.volume(self.hwl))

    @overload
    def volume(self, h: float) -> float: ...

    @overload
    def volume(self, h: FloatArray) -> FloatArray: ...

    def volume(self, h: float | FloatArray) -> float | FloatArray:
        # 追跡の求解では水位を1個ずつ評価する。表内は補間、表より上は最後の勾配で延長する。
        if np.ndim(h) == 0:
            level = float(h)
            if level > self.levels[-1]:
                return float(self.volumes[-1] + self.rise_rate() * (level - self.levels[-1]))
            return float(np.interp(level, self.levels, self.volumes))
        top, vtop, rate = self.levels[-1], self.volumes[-1], self.rise_rate()
        x = np.asarray(h, dtype=float)
        v = np.where(x > top, vtop + rate * (x - top), np.interp(x, self.levels, self.volumes))
        return v if x.ndim else float(v)

    def level(self, v: float) -> float:
        top, vtop, rate = self.levels[-1], self.volumes[-1], self.rise_rate()
        if v > vtop:
            return float(top + (v - vtop) / rate)
        return float(np.interp(v, self.volumes, self.levels))

    def area_at(self, h: float) -> float:
        if self.areas is None:
            raise ValueError("池の表に面積の列がないため、水位ごとの面積を求められません。池の表に面積の列を加えてください。")
        if h > self.levels[-1]:                  # 表より上は延長した池の水面積（容量の増え方）
            return self.rise_rate()
        return float(np.interp(h, self.levels, self.areas))


P = TypeVar("P", bound=Pond)


@dataclass
class Facility(Generic[P]):
    """貯留施設と放流施設の組。P は池の種類（寸法を求めた施設は PrismaticPond で、面積を読める）。"""
    pond: P
    outlet: Outlet


@dataclass
class RoutingResult:
    h: FloatArray        # 各節点の水位 m
    out: FloatArray      # 各節点の放流量 m3/s
    spill: float         # 溢水量 m3
    vmax: float          # 最大貯留量 m3
    overdraw: tuple[tuple[int, float], ...] = ()     # 刻みの中で池が空になった節点と、放流しなかった池にない量 m3 の組の並び（route の説明）


def _S(pond: Pond, outlet: Outlet, h: float, dt: float) -> float:
    """連続式の左辺 S(h) ＝ V(h) ＋ dt/2·O(h)。V は増加、O は減らないので、S は水位の増加関数。"""
    return pond.volume(h) + dt / 2 * outlet.discharge(h)


def _upper(pond: Pond, outlet: Outlet, rhs: float, dt: float) -> float:
    """S(b) ＞ rhs となる水位 b（_solve の区間の上端）。表より上も容量は増え方 rise_rate（正）で増え続けるので、S は際限なく増え、b がある。

    O ≧ 0 なので S(h) ≧ V(h)。容量から逆算した水位 level(rhs) では、V は丸めの誤差（数 ulp）を除いて rhs に等しい。
    S(b) ≦ rhs の間は、b の ulp から始めて刻みを倍にしながら b を上げる（丸めの誤差の分だけ上げれば済み、回数は誤差の桁数で決まる）。
    rhs ÷ 容量の増え方 が float の範囲を超えて逆算できないときは、満水位から、池の深さを刻みの始めとして倍にしながら上げる。
    真の水位が float の範囲を超えるときは inf を返す（表せる解がない。route がエラーにする）。
    """
    b = pond.level(rhs)
    if math.isfinite(b):
        b = max(b, pond.bottom)
        step = math.ulp(max(abs(b), 1.0))
    else:
        b = pond.hwl
        step = max(pond.hwl - pond.bottom, 1.0)
    while _S(pond, outlet, b, dt) <= rhs:
        b += step
        step *= 2
    return b


def _solve(pond: Pond, outlet: Outlet, rhs: float, dt: float, b: float) -> float:
    """S(h) ＝ rhs の水位 h。S(h) ≦ rhs ＜ S(h の次の float) となる h（隣り合う float の下側）を返す。

    b は S(b) ＞ rhs となる区間の上端（呼び出し側が保証する。_upper の値。溢水を数える計算では満水位との小さい方）。
    S は水位の単調増加関数なので、この h は一つに決まる。Illinois 法（はさみうち法の改良）で区間を狭め、
    最後に二分法で隣り合う float まで詰める（二分法だけで詰めるのと同じ h になり、放流量の計算の回数は約4割）。
    """
    a = pond.bottom
    fa = _S(pond, outlet, a, dt) - rhs
    if fa >= 0:                                 # 池底で S ≧ rhs（右辺が0以下など）。池底が解
        return a
    fb, side = _S(pond, outlet, b, dt) - rhs, 0
    if not (fb > 0 and math.isfinite(b)):
        raise RuntimeError(f"貯留追跡の内部の誤り: 区間の上端 {b} m で、有限の水位で S ＞ rhs が成り立ちません（S－rhs＝{fb}）。")
    for _ in range(60):
        c = (a * fb - b * fa) / (fb - fa)
        if not a < c < b:
            break
        fc = _S(pond, outlet, c, dt) - rhs
        if fc > 0:
            b, fb = c, fc
            if side == 1:
                fa /= 2
            side = 1
        else:
            a, fa = c, fc
            if side == -1:
                fb /= 2
            side = -1
        if b - a <= 4e-16 * max(abs(a), abs(b)):     # 区間が数 ulp まで詰まった（池底が負の標高でも同じ判定）
            break
    while True:
        mid = (a + b) / 2
        if mid == a or mid == b:
            return a
        if _S(pond, outlet, mid, dt) > rhs:
            b = mid
        else:
            a = mid


def route(facility: Facility[P], inflow: npt.ArrayLike, capped: bool = True, dt: float = DT) -> RoutingResult:
    """貯留追跡計算。連続式 V(n) + dt/2·O(n) = V(n-1) + dt/2·(I(n-1)+I(n)-O(n-1)) を台形公式で解く。

    左辺は水位の単調増加関数になるので、各節点で水位を求める（_solve）。dt は秒（tc×60）。
    capped=True なら満水位を超えた分は溢水として数え、水位は満水位で止める。
    放流量が段で変わる水位（ポンプの始動など。outlet.steps）で式に解がないときは、水位をその段に置き、
    放流量を式が成り立つ値（段の下と上の放流量の間。ポンプなら間欠運転）にする。
    右辺が負になる節点（前の節点の放流量の半刻み分が池の水と流入の和を超える。池が1刻みの中で空になる）は、
    水位を池底・放流量を0とし、その刻みの放流量を池にあった水と流入の和にする。前後の節点の放流量の平均から求める量との差
    （池にない量）を、節点と組にして overdraw に残す（節点の放流量の台形公式の和からこれを引くと、水収支が合う）。
    capped=True では、rhs が満水位での左辺 S(満水位) を超える節点は満水位で溢れたとし、そうでなければ池底と満水位の間で解く。
    capped=False では満水位より上も、表の最後の区間の容量の増え方で池を延ばして解く。
    流入量に NaN・無限大があるか、右辺が float の範囲を超えれば ValueError（S と比べられない）。
    池底で放流量が正の放流施設は ValueError（空の池からは放流できない。S(池底)＝0 がこの解き方の前提で、
    これが崩れると 0 ≦ rhs ＜ S(池底) の節点で水が増える）。
    """
    pond, outlet = facility.pond, facility.outlet
    steps = outlet.steps
    q = np.asarray(inflow, dtype=float)
    if not np.all(np.isfinite(q)):
        raise ValueError("流入量に数でない値（NaN）か無限大があります。流入量の入力を確かめてください。")
    if outlet.discharge(pond.bottom) > 0:
        raise ValueError(f"池底（{pond.bottom} m）で放流量が {outlet.discharge(pond.bottom):g} m^3/s あります。空の池からは放流できないので、"
                         "オリフィスの敷高とポンプの最初の水位を池底以上にし、池底での排水量を0にしてください。")
    n = len(q)
    h = np.full(n, pond.bottom)
    o = np.zeros(n)
    cap = pond.capacity
    s_hwl = _S(pond, outlet, pond.hwl, dt)
    spill = 0.0
    v_prev = 0.0
    overdraw = []
    qf = q.tolist()                               # 右辺は Python の float で計算する（範囲を超えれば警告なしに inf になり、下で確かめる）
    for i in range(1, n):
        rhs = v_prev + dt / 2 * (qf[i - 1] + qf[i] - float(o[i - 1]))
        if not math.isfinite(rhs):
            raise ValueError(f"{i}番目の節点で、連続式の右辺（前の貯留量＋流入量－放流量の刻みの水量）が float の範囲を超えました。"
                             "流入量の値と単位（m^3/s）を確かめてください。")
        if rhs < 0:
            overdraw.append((i, -rhs))
        if capped and rhs >= s_hwl:               # 満水位で S ≦ rhs。満水位で止め、超えた分を溢水にする
            h[i], o[i] = pond.hwl, outlet.discharge(pond.hwl)
            spill += rhs - s_hwl
            v_prev = cap
            continue
        b = _upper(pond, outlet, rhs, dt)
        if capped:                                # rhs ＜ S(満水位) なので満水位も上端になる。挟む区間の狭い方を使う
            b = min(b, pond.hwl)
        if not math.isfinite(b):
            raise ValueError(f"{i}番目の節点で、水位が float の範囲を超えます（池の容量の増え方に比べて流入量が大きすぎます）。"
                             "池の大きさと流入量の値・単位を確かめてください。")
        hh = _solve(pond, outlet, rhs, dt, b)
        oo = None
        if steps:
            up = math.nextafter(hh, math.inf)
            for s in steps:
                if hh < s <= up:                  # 解が段の上にある（段の下では S ≦ rhs、段では S ＞ rhs）
                    hh = s
                    oo = min(max((rhs - pond.volume(s)) * 2 / dt, 0.0), outlet.discharge(s))
                    break
        h[i] = hh
        o[i] = outlet.discharge(hh) if oo is None else oo
        v_prev = pond.volume(hh)
    return RoutingResult(h=h, out=o, spill=spill, vmax=float(np.max(pond.volume(h))), overdraw=tuple(overdraw))
