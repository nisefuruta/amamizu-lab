"""放流施設（自然放流のオリフィスと洪水吐）。

水位 h は、池の水位と同じ基準の高さ（柱状の池では池底からの水深）。
係数（越流係数、オリフィスの流量係数）と重力加速度は基準で決まるので、Hydraulics で渡す（既定値は持たない）。
"""
import math
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np

MIN_ORIFICE_M = 1e-4     # オリフィスの寸法を二分法で探す範囲の下限 m
SOLVE_RTOL = 1e-6        # 二分法で求めた値が目標の値と一致したとみなす相対差

class Outlet(Protocol):
    """放流施設が持つもの（貯留追跡 route が使う）。

    discharge は水位 h（池と同じ基準）の放流量 m3/s で、水位とともに減らないこと（追跡の水位が一つに決まる前提）。
    steps は放流量が段で変わる水位（ポンプの始動など。連続なら空）。
    """

    @property
    def steps(self) -> tuple[float, ...]: ...

    def discharge(self, h: float) -> float: ...


@dataclass(frozen=True)
class Hydraulics:
    """放流の式の係数（重力加速度 g m/s2、越流係数 c_weir、オリフィスの流量係数 c_orifice）。"""
    g: float
    c_weir: float
    c_orifice: float

    def __post_init__(self) -> None:
        # 越流（1.2D まで）とオリフィス（1.8D から）の間を直線で補間するので、1.8D の放流量が 1.2D より大きくないと
        # 放流量が水位とともに減る区間ができ、追跡の水位が一つに決まらない。長方形と円形の両方で確かめる（D によらない）
        if not (self.g > 0 and self.c_weir > 0 and self.c_orifice > 0):
            raise ValueError(f"放流の係数（g {self.g}、越流 {self.c_weir}、オリフィス {self.c_orifice}）は正の値にしてください。")
        rect = self.c_orifice * math.sqrt(2 * self.g * 1.3) / (self.c_weir * 1.2 ** 1.5)
        circle = self.c_orifice * math.pi / 4 * math.sqrt(2 * self.g * 1.3) / (1.5 * self.c_weir * circle_weir_integral(1.2, 1.0))
        if not (rect > 1 and circle > 1):
            raise ValueError(f"放流の係数（g {self.g}、越流 {self.c_weir}、オリフィス {self.c_orifice}）では、"
                             "オリフィスの放流量が水位の上がる途中で減ります（1.8D の放流量が 1.2D より小さい）。係数を見直してください。")


def orifice_discharge(y: float, width: float, height: float, hy: Hydraulics) -> float:
    """敷高からの水深 y m の、幅 width・高さ height のオリフィスの放流量 m3/s。

    低い水位は越流（Q = c_weir·B·y^1.5）、高い水位はオリフィス（Q = c_orifice·B·D·sqrt(2g(y-D/2))）で、
    水深がオリフィスの高さの1.2倍〜1.8倍の間は直線で補間する（上端もオリフィスの式の値。1.8D で値がつながる）。
    """
    b, d = width, height
    if y <= 0:
        return 0.0
    if y >= 1.8 * d:
        return hy.c_orifice * b * d * math.sqrt(2 * hy.g * (y - d / 2))
    if y <= 1.2 * d:
        return hy.c_weir * b * float(y ** 1.5)      # float(): 底が正なので実数（型の上では float ** float は Any）
    q1 = hy.c_weir * b * float((1.2 * d) ** 1.5)
    q2 = hy.c_orifice * b * d * math.sqrt(2 * hy.g * (1.8 * d - d / 2))
    return q1 + (q2 - q1) * (y - 1.2 * d) / (0.6 * d)


@dataclass
class SquareOrifice:
    """正方形のオリフィス（一辺 size m、下端は池底）。モデル施設と早見表の計算に使う。"""
    size: float
    hy: Hydraulics

    steps: ClassVar[tuple[float, ...]] = ()      # 放流量が段で変わる水位（連続なのでなし。route の説明）

    def discharge(self, h: float) -> float:
        return orifice_discharge(h, self.size, self.size, self.hy)

    @staticmethod
    def sized_for(hwl: float, q_allow: float, hy: Hydraulics, strict: bool = True) -> "SquareOrifice":
        """HWL での放流量が q_allow になるオリフィス（二分法）。

        オリフィスの高さは HWL/1.8 までに限る。その範囲で q_allow にできなければ、strict なら ValueError、
        そうでなければ最も近いオリフィスを返す（寸法の探索の途中で試す水位に使う。最後の施設は design で確かめる）。
        """
        lo, hi = MIN_ORIFICE_M, hwl / 1.8
        o = SquareOrifice(lo, hy)
        for _ in range(100):
            mid = (lo + hi) / 2
            o.size = mid                         # 詰め切った後の mid も、打ち切らずに回したときと同じ値
            if mid == lo or mid == hi:           # 隣り合う float まで詰めた
                break
            if o.discharge(hwl) > q_allow:
                hi = o.size
            else:
                lo = o.size
        if strict and abs(o.discharge(hwl) - q_allow) > SOLVE_RTOL * q_allow:
            # オリフィスの高さの上限（HWL/1.8）でも許容放流量に達しない、または下限でも超える
            raise ValueError(f"満水位 {hwl} m で放流量 {q_allow} m3/s になるオリフィスが見つかりません"
                             f"（{o.discharge(hwl):.6g} m3/s で止まりました）。満水位か許容放流量を見直してください。")
        return o


@dataclass
class RectOrifice:
    """長方形のオリフィス（敷高 sill m、幅 width m、高さ height m）。"""
    sill: float
    width: float
    height: float
    hy: Hydraulics

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"オリフィスの幅と高さは正の値にしてください（幅 {self.width}、高さ {self.height}）。")

    steps: ClassVar[tuple[float, ...]] = ()      # 放流量が段で変わる水位（連続なのでなし。route の説明）

    def discharge(self, h: float) -> float:
        return orifice_discharge(h - self.sill, self.width, self.height, self.hy)

    def describe(self) -> str:
        return f"長方形 幅 {self.width:.4f} m、高さ {self.height:.4f} m"


_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)
_W = (_GL_X + 1) / 2          # [0, 1] のガウス・ルジャンドル点
_WW = _GL_W / 2


def circle_weir_integral(y: float, d: float) -> float:
    """直径 d の円の、敷高（円の下端）からの水深 y までの ∫ b(z)·sqrt(y-z) dz（m^2.5）。

    b(z) は高さ z での円の幅 2sqrt(z(d-z))。水面より上と円より上は積分しない。
    z＝d/2·(1-cosθ) で b·dz＝(d^2/2)·sin^2θ·dθ とし、さらに θ＝θtop·(1-w^2) として
    水面の位置の平方根の変化を消してから、w について64点のガウス・ルジャンドル積分で求める。
    """
    zt = min(y, d)
    if zt <= 0:
        return 0.0
    th_top = math.acos(max(-1.0, 1 - 2 * zt / d))
    th = th_top * (1 - _W ** 2)
    z = d / 2 * (1 - np.cos(th))
    f = (d * d / 2) * np.sin(th) ** 2 * np.sqrt(np.maximum(y - z, 0.0)) * 2 * th_top * _W
    return float(np.dot(_WW, f))


def circle_orifice_discharge(y: float, d: float, hy: Hydraulics) -> float:
    """敷高からの水深 y m の、直径 d の円形のオリフィスの放流量 m3/s。

    越流（y ≦ 1.2D）は水面下の円の幅を積分する: Q = (3/2)·c_weir·∫ b(z)sqrt(y-z) dz
    （幅一定の長方形なら c_weir·B·y^1.5 に一致する係数）。円より上の幅は0として積分しない。
    オリフィス（y ≧ 1.8D）は Q = c_orifice·(πD^2/4)·sqrt(2g(y-D/2))。その間は直線で補間する。
    """
    if y <= 0:
        return 0.0
    area = math.pi * d * d / 4
    if y >= 1.8 * d:
        return hy.c_orifice * area * math.sqrt(2 * hy.g * (y - d / 2))
    k = 1.5 * hy.c_weir
    if y <= 1.2 * d:
        return k * circle_weir_integral(y, d)
    q1 = k * circle_weir_integral(1.2 * d, d)
    q2 = hy.c_orifice * area * math.sqrt(2 * hy.g * (1.8 * d - d / 2))
    return q1 + (q2 - q1) * (y - 1.2 * d) / (0.6 * d)


@dataclass
class CircleOrifice:
    """円形のオリフィス（敷高 sill m＝円の下端、直径 diameter m）。"""
    sill: float
    diameter: float
    hy: Hydraulics

    def __post_init__(self) -> None:
        if self.diameter <= 0:
            raise ValueError(f"オリフィスの直径は正の値にしてください（{self.diameter}）。")

    steps: ClassVar[tuple[float, ...]] = ()      # 放流量が段で変わる水位（連続なのでなし。route の説明）

    def discharge(self, h: float) -> float:
        return circle_orifice_discharge(h - self.sill, self.diameter, self.hy)

    def describe(self) -> str:
        return f"円形 直径 {self.diameter:.4f} m"


@dataclass
class Spillway:
    """洪水吐（敷高 crest m、幅 width m）。越流の式 Q = c_weir·B·(h-crest)^1.5。"""
    crest: float
    width: float
    hy: Hydraulics

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(f"洪水吐の幅は正の値にしてください（{self.width}）。")

    steps: ClassVar[tuple[float, ...]] = ()      # 放流量が段で変わる水位（連続なのでなし。route の説明）

    def discharge(self, h: float) -> float:
        y = h - self.crest
        if not y > 0:
            return 0.0
        try:
            return self.hy.c_weir * self.width * float(y ** 1.5)     # y ＞ 0 なので実数
        except OverflowError:       # y^1.5 が float の範囲を超える。Python の float のべき乗は inf を返さず例外になるので、
            return math.inf         # IEEE の浮動小数点数と同じく +inf にする（放流量は水位の増加関数のまま）


@dataclass
class PumpTable:
    """ポンプ（水位と排水量の表。表の間は線形補間）。

    水位は池と同じ基準。表の最初の水位より低いときは0、最後の水位より高いときは最後の排水量。
    排水量は水位とともに減らないこと（減ると追跡の水位が一つに決まらないため、エラーにする）。
    """
    levels: tuple[float, ...]
    q: tuple[float, ...]

    def __post_init__(self) -> None:
        lv, q = np.asarray(self.levels, dtype=float), np.asarray(self.q, dtype=float)
        if len(lv) != len(q) or len(lv) < 2:
            raise ValueError("ポンプの表は、水位と排水量が同じ個数で2点以上必要です。")
        if not (np.all(np.isfinite(lv)) and np.all(np.isfinite(q))):      # 空欄（NaN）は大小の確かめをすり抜ける
            raise ValueError("ポンプの表に空欄か数でない値があります。水位と排水量をすべての行に入力してください。")
        if np.any(np.diff(lv) <= 0):
            raise ValueError("ポンプの表の水位は、後ろほど高くしてください。")
        if np.any(q < 0) or np.any(np.diff(q) < 0):
            raise ValueError("ポンプの排水量は0以上で、水位が高いほど減らないようにしてください。")
        self._lv, self._q = lv, q

    @property
    def steps(self) -> tuple[float, ...]:
        """放流量が段で変わる水位。最初の排水量が正なら、最初の水位で0から段になる（route は間欠運転として扱う）。"""
        return (float(self._lv[0]),) if self._q[0] > 0 else ()

    def discharge(self, h: float) -> float:
        if h < self._lv[0]:
            return 0.0
        return float(np.interp(h, self._lv, self._q))

    def describe(self) -> str:
        return f"ポンプ（水位 {self._lv[0]:g}〜{self._lv[-1]:g} m、排水量 {self._q[0]:g}〜{self._q[-1]:g} m^3/s）"


@dataclass
class Outlets:
    """複数の放流施設（オリフィスと洪水吐など）。放流量は各施設の和。"""
    parts: tuple[Outlet, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("放流施設がありません。オリフィスを1つ以上書いてください。")

    @property
    def steps(self) -> tuple[float, ...]:
        return tuple(sorted({s for p in self.parts for s in p.steps}))

    def discharge(self, h: float) -> float:
        # 追跡の求解で節点ごとに何十回も呼ぶので、sum とジェネレータを使わずに足す（同じ順に足すので値は同じ）
        q = 0.0
        for p in self.parts:
            q += p.discharge(h)
        return q


def combine(parts: tuple[Outlet, ...]) -> Outlet:
    """放流施設の組。1つならその施設をそのまま返す（追跡で和を取る呼び出しを省く。放流量は同じ）。"""
    return parts[0] if len(parts) == 1 else Outlets(parts)
