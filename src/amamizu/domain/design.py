"""計画降雨で貯留施設を決める（必要洪水調節容量、放流オリフィスの寸法）。"""
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

import numpy.typing as npt

from .outlet import MIN_ORIFICE_M, SOLVE_RTOL, CircleOrifice, Hydraulics, Outlet, Outlets, RectOrifice, SquareOrifice
from .pond import DT, Facility, P, Pond, PrismaticPond, RoutingResult, route


@dataclass
class DesignedFacility(Generic[P]):
    facility: Facility[P]
    design_result: RoutingResult     # 計画降雨での追跡結果

    @property
    def capacity(self) -> float:
        """計画容量 m3（満水位までの池の容量）。"""
        return self.facility.pond.capacity


def _fit(design_inflow: npt.ArrayLike, q_allow: float | None, make: Callable[[float], Facility[P]], lo: float, hi: float,
         dt: float = DT) -> DesignedFacility[P]:
    """計画降雨の最大水位がちょうど満水位になる param を、等比の二分法で探す。
    make(param) で作る施設は、param を増やすほど「最大水位÷満水位」が下がること。
    見つけた施設で、最大水位が満水位に、満水位での放流量が q_allow に一致しなければ ValueError
    （q_allow が None なら放流量は確かめない。オリフィスを与えて池を決めるとき）。"""
    def trial(param: float) -> tuple[float, Facility[P], RoutingResult]:
        fac = make(param)
        r = route(fac, design_inflow, capped=False, dt=dt)
        return float(r.h.max()) / fac.pond.hwl, fac, r

    lo0, hi0 = lo, hi
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if mid == lo or mid == hi:             # 隣り合う float まで詰めた（これ以上は区間が変わらない）
            break
        ratio, _, _ = trial(mid)
        if ratio > 1:
            lo = mid
        else:
            hi = mid
    ratio, fac, r = trial(hi)
    q_hwl = fac.outlet.discharge(fac.pond.hwl)
    if q_allow is not None and abs(q_hwl - q_allow) > SOLVE_RTOL * q_allow:
        raise ValueError(f"満水位 {fac.pond.hwl:.3f} m で許容放流量 {q_allow} m3/s になるオリフィスが作れません"
                         f"（{q_hwl:.6g} m3/s）。満水位か許容放流量を見直してください。")
    if abs(ratio - 1) > SOLVE_RTOL:
        # 探索の範囲に解がなく、範囲の端で止まった
        raise ValueError(f"計画降雨の最大水位を満水位に合わせる寸法が範囲 {lo0:g}〜{hi0:g} にありません"
                         f"（最大水位÷満水位 {ratio:.4f}）。開発面積や満水位を見直してください。")
    return DesignedFacility(fac, r)


def design_for_depth(design_inflow: npt.ArrayLike, q_allow: float | None, depth: float, dt: float = DT,
                     hy: Hydraulics | None = None, orifice: Outlet | None = None) -> DesignedFacility[PrismaticPond]:
    """満水位（深さ）を決め、池の面積を求める。放流オリフィスは満水位で許容放流量になる寸法。
    orifice を与えると、そのオリフィスで池の面積を求める（q_allow は None にし、放流量は確かめない）。"""
    if orifice is None:
        if q_allow is None or hy is None:
            raise ValueError("池の面積を求めるには、許容放流量（q_allow）と放流の係数（hy）か、放流オリフィス（orifice）のどちらかが必要です。")
        orifice = SquareOrifice.sized_for(depth, q_allow, hy, strict=False)
    hole = orifice
    return _fit(design_inflow, q_allow, lambda area: Facility(PrismaticPond(area, depth), hole), 10.0, 1e5, dt)


def design_for_area(design_inflow: npt.ArrayLike, q_allow: float, area: float, dt: float = DT, *,
                    hy: Hydraulics) -> DesignedFacility[PrismaticPond]:
    """池の面積を決め、満水位（深さ）を求める（手引の早見表の条件）。"""
    return _fit(design_inflow, q_allow,
                lambda hwl: Facility(PrismaticPond(area, hwl), SquareOrifice.sized_for(hwl, q_allow, hy, strict=False)),
                0.05, 50.0, dt)


def _orifice_maker(shape: str, sill: float, hy: Hydraulics) -> Callable[[float], RectOrifice | CircleOrifice]:
    """寸法を求めるオリフィスの作り方（square は正方形の一辺、circle は円の直径）。"""
    makers: dict[str, Callable[[float], RectOrifice | CircleOrifice]] = {
        "square": lambda s: RectOrifice(sill, s, s, hy), "circle": lambda s: CircleOrifice(sill, s, hy)}
    if shape not in makers:
        raise ValueError(f"オリフィスの形「{shape}」は寸法を求められません（square、circle）。")
    return makers[shape]


def size_orifice_for_peak(inflow: npt.ArrayLike, pond: Pond, sill: float, target_q: float, dt: float = DT,
                          others: tuple[Outlet, ...] = (), *, hy: Hydraulics,
                          shape: str = "square", allow_spill: bool = False) -> tuple[RectOrifice | CircleOrifice, RoutingResult]:
    """池を決め、最大放流量が target_q になるオリフィス（敷高 sill）の寸法を二分法で求める。

    shape が square なら正方形の一辺、circle なら円の直径を求める。
    others は一緒に使う放流施設（洪水吐など）。オリフィスが大きいほど最大放流量は大きくなる。
    見つけたオリフィスで、最大放流量が target_q と SOLVE_RTOL の割合で一致しなければ ValueError。
    そのオリフィスで池が溢れるときも ValueError（池の容量が足りない）。allow_spill なら溢れてもオリフィスと追跡の結果を返す
    （足りない容量は required_pond_scale で求める）。
    """
    make = _orifice_maker(shape, sill, hy)

    def trial(size: float) -> tuple[float, RectOrifice | CircleOrifice, RoutingResult]:
        hole = make(size)
        r = route(Facility(pond, Outlets((hole,) + tuple(others))), inflow, capped=True, dt=dt)
        return float(r.out.max()), hole, r

    lo, hi = MIN_ORIFICE_M, pond.hwl - sill
    if hi <= lo:
        raise ValueError(f"オリフィスの敷高 {sill} m が満水位 {pond.hwl} m 以上です。敷高を満水位より低くしてください。")
    for _ in range(100):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        qmax, _, _ = trial(mid)
        if qmax > target_q:
            hi = mid
        else:
            lo = mid
    qmax, hole, r = trial(lo)
    if r.spill > 0 and not allow_spill:
        raise ValueError(f"最大放流量を {target_q} m3/s 以下にするオリフィスでは池が溢れます（溢水量 {r.spill:.1f} m3）。"
                         "池の容量が足りません。池の面積か満水位を大きくするか、目標の最大放流量（target_q_m3s）を見直してください。")
    if abs(qmax - target_q) > SOLVE_RTOL * target_q:
        raise ValueError(f"最大放流量が {target_q} m3/s になるオリフィスが、寸法 {MIN_ORIFICE_M:g}〜{pond.hwl - sill:.3f} m の範囲にありません"
                         f"（{qmax:.6g} m3/s で止まりました）。目標の最大放流量（target_q_m3s）かオリフィスの敷高を見直してください。")
    return hole, r


def required_pond_scale(inflow: npt.ArrayLike, pond: Pond, sill: float, target_q: float, dt: float = DT,
                        others: tuple[Outlet, ...] = (), *, hy: Hydraulics, shape: str = "square",
                        k_max: float = 1e4) -> tuple[float, RectOrifice | CircleOrifice, Pond, RoutingResult]:
    """最大放流量を target_q にするオリフィスでは池が溢れるとき、水深（満水位）はそのままで面積を何倍にすれば溢れないかを求める。

    溢れるかどうかの境目の池では、最大水位がちょうど満水位になり、そのときの放流量が target_q になる。
    そこでオリフィスは「満水位での放流量（others を含む）が target_q になる寸法」に決まり（池の大きさによらない）、
    そのオリフィスで溢れない最小の面積の倍率 k を等比の二分法で求める。戻り値は（k, オリフィス, 面積を k 倍した池, 追跡の結果）。"""
    make = _orifice_maker(shape, sill, hy)
    lo, hi = MIN_ORIFICE_M, pond.hwl - sill
    if hi <= lo:
        raise ValueError(f"オリフィスの敷高 {sill} m が満水位 {pond.hwl} m 以上です。敷高を満水位より低くしてください。")
    if others and Outlets(tuple(others)).discharge(pond.hwl) >= target_q:
        raise ValueError(f"オリフィスのほかの放流施設だけで、満水位 {pond.hwl} m の放流量が {target_q} m3/s 以上になります。")
    for _ in range(100):
        mid = (lo + hi) / 2
        if mid == lo or mid == hi:
            break
        if Outlets((make(mid),) + tuple(others)).discharge(pond.hwl) > target_q:
            hi = mid
        else:
            lo = mid
    hole = make(lo)
    q_hwl = Outlets((hole,) + tuple(others)).discharge(pond.hwl)
    if abs(q_hwl - target_q) > SOLVE_RTOL * target_q:           # 二分法が範囲の端で止まった（満水位で目標に達するオリフィスがない）
        raise ValueError(f"満水位 {pond.hwl} m で放流量が {target_q} m3/s になるオリフィスが、寸法 {MIN_ORIFICE_M:g}〜{pond.hwl - sill:.3f} m の"
                         f"範囲にありません（{q_hwl:.6g} m3/s）。目標の最大放流量（target_q_m3s）かオリフィスの敷高を見直してください。")

    def run(k: float) -> tuple[Pond, RoutingResult]:
        big = pond.scaled(k)
        return big, route(Facility(big, Outlets((hole,) + tuple(others))), inflow, capped=True, dt=dt)

    k_lo, k_hi = 1.0, 2.0
    while run(k_hi)[1].spill > 0:
        k_lo, k_hi = k_hi, k_hi * 2
        if k_hi > k_max:
            raise ValueError(f"池の面積を {k_max:g} 倍にしても溢れます。流入（区域の面積・計画降雨）とオリフィスの敷高を見直してください。")
    for _ in range(60):
        mid = math.sqrt(k_lo * k_hi)
        if mid == k_lo or mid == k_hi:
            break
        if run(mid)[1].spill > 0:
            k_lo = mid
        else:
            k_hi = mid
    big, r = run(k_hi)
    return k_hi, hole, big, r
