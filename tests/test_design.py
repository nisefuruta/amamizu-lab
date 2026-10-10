"""設計の厳密計算（kind: design）と、tc・表の池・放流施設の一般化の試験。"""
import datetime
import json
import math
from pathlib import Path

import numpy as np
import pytest

from amamizu.adapters.report.design_plots import GraphError
from amamizu.cli import load_study, main, run
from amamizu.domain.design import design_for_depth, required_pond_scale, size_orifice_for_peak
from amamizu.domain.design_storm import rear_loaded
from amamizu.domain.intensity import IntensityFormula
from amamizu.domain.outlet import Hydraulics, Outlets, PumpTable, RectOrifice, Spillway, SquareOrifice
from amamizu.domain.pond import Facility, PrismaticPond, TablePond, route
from amamizu.domain.rainfall import blocks_from_10min
from amamizu.domain.runoff import rational_inflow
from amamizu.usecases.design_study import Catchment, hydrograph_inflow, overdraw_note, planned_inflow, run_design

CHIBA = Path(__file__).parents[1] / "studies" / "_guideline" / "chiba_2006.json"      # 千葉県の手引の定数

HY = Hydraulics(9.8, 1.8, 0.6)      # 試験に使う放流の係数（重力加速度、越流係数、オリフィスの流量係数）

CHIBA_50 = IntensityFormula(2439, 0.75, 11.9)


def storm_inflow(tc, area_ha=1.0, f=0.8):
    return rational_inflow(rear_loaded(CHIBA_50, 24, tc), f, area_ha)


# 表の池と放流施設 ------------------------------------------------------------
def test_table_pond_with_constant_area_equals_prismatic():
    """面積一定の表の池（池底の標高 10m）は、柱状の池と同じ水深・放流量になる。"""
    q = np.concatenate([storm_inflow(10), np.zeros(288)])
    prism = route(Facility(PrismaticPond(600.0, 2.0), SquareOrifice(0.08, HY)), q)
    table = TablePond([10.0, 11.0, 12.0], [600.0, 600.0, 600.0], [0.0, 600.0, 1200.0])
    tab = route(Facility(table, RectOrifice(10.0, 0.08, 0.08, HY)), q)
    assert np.allclose(tab.h - 10.0, prism.h, atol=1e-9)
    assert np.allclose(tab.out, prism.out, atol=1e-12)
    assert tab.vmax == pytest.approx(prism.vmax) and tab.spill == pytest.approx(prism.spill)


@pytest.mark.parametrize("levels, areas, volumes, msg", [
    ([1, 1, 2], [1, 1, 1], [0, 1, 2], "水位"),
    ([0, 1, 2], [1, 1, 1], [0, 2, 1], "容量"),
    ([0, 1, 2], [1, 1, 1], [5, 6, 7], "0"),
])
def test_bad_pond_table_is_an_error(levels, areas, volumes, msg):
    with pytest.raises(ValueError, match=msg):
        TablePond(levels, areas, volumes)


def test_pond_table_from_areas():
    """面積だけの表は、容量を平均断面法で求める（解説 資料－4 計算例－1 の図－10 の水位容量表と一致する）。"""
    pond = TablePond.from_areas([0, 1, 2, 3, 4, 5, 6, 7], [1000, 1386, 1772, 2157, 2543, 2929, 3314, 3700])
    assert np.allclose(pond.volumes, [0, 1193, 2772, 4736.5, 7086.5, 9822.5, 12944, 16451])
    with pytest.raises(ValueError, match="少なくとも一方を正"):
        TablePond.from_areas([0, 1, 2], [0, 0, 100])


def test_pond_table_without_areas():
    """面積のない表: 水面積は求められない。表より上は最後の区間の容量の増え方で立ち上がる。"""
    pond = TablePond([0, 1, 2], None, [0, 100, 300])
    assert not pond.has_area and pond.volume(3.0) == pytest.approx(500) and pond.level(500) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="面積の列がないため"):
        pond.area_at(1.0)


def test_prismatic_pond_bottom_is_zero():
    """柱状の池は水深で表すので、池底は 0 m（容量 面積×水位 と矛盾する池底を作れない）。"""
    pond = PrismaticPond(100.0, 2.0)
    assert pond.bottom == 0.0 and pond.volume(pond.bottom) == 0.0 and pond.scaled(2.0).bottom == 0.0
    with pytest.raises(TypeError, match="bottom"):
        PrismaticPond(100.0, 2.0, bottom=-5.0)


def test_scaled_pond_keeps_depth():
    """面積を k 倍した池は、満水位（水深）はそのままで容量が k 倍になる。"""
    t = TablePond([0, 1, 2], [100, 100, 100], [0, 100, 200], hwl=1.5)
    for pond in (PrismaticPond(100.0, 2.0), t, TablePond([0, 1, 2], None, [0, 100, 200])):
        big = pond.scaled(1.5)
        assert big.hwl == pond.hwl and big.capacity == pytest.approx(pond.capacity * 1.5)


def test_spillway_adds_weir_flow_above_crest():
    hy = HY
    o = RectOrifice(0.0, 0.2, 0.2, hy)
    s = Spillway(1.0, 1.5, hy)
    both = Outlets((o, s))
    assert both.discharge(0.9) == pytest.approx(o.discharge(0.9))
    assert both.discharge(1.3) == pytest.approx(o.discharge(1.3) + 1.8 * 1.5 * 0.3 ** 1.5)


def test_orifice_formula_is_continuous():
    """越流とオリフィスの間の補間は、両端で式の値につながる（1.8D で値が飛ばない）。"""
    o = RectOrifice(0.0, 0.9, 0.9, Hydraulics(9.81, 1.8, 0.6))
    for y in (1.2 * 0.9, 1.8 * 0.9):
        assert o.discharge(y - 1e-9) == pytest.approx(o.discharge(y + 1e-9), rel=1e-6)


# tc -----------------------------------------------------------------------
@pytest.mark.parametrize("tc", [5, 10, 15, 20, 30, 60])
def test_tc_keeps_storm_depth_and_runoff_volume(tc):
    """tc を変えても、計画降雨の24時間雨量と流入の総量（f·R·A·10）は変わらない。"""
    storm = rear_loaded(CHIBA_50, 24, tc)
    assert storm.sum() * tc / 60 == pytest.approx(CHIBA_50.depth(1440))
    q = rational_inflow(storm, 0.8, 1.0)
    assert np.trapezoid(q, dx=tc * 60) == pytest.approx(0.8 * CHIBA_50.depth(1440) * 10)


@pytest.mark.parametrize("tc", [5, 10, 20])
def test_routing_conserves_water(tc):
    """流入の総量＝放流の総量＋最後の貯留量＋溢水量（台形公式の連続式）。"""
    q = np.concatenate([storm_inflow(tc), np.zeros(18 * 60 // tc)])
    pond = TablePond([0.0, 1.0, 3.0], [300.0, 400.0, 600.0], [0.0, 350.0, 1350.0])
    r = route(Facility(pond, RectOrifice(0.0, 0.1, 0.1, HY)), q, dt=tc * 60)
    vin, vout = np.trapezoid(q, dx=tc * 60), np.trapezoid(r.out, dx=tc * 60)
    assert vin == pytest.approx(vout + pond.volume(r.h[-1]) + r.spill, rel=1e-9)


def test_rear_loaded_rejects_tc_not_dividing_duration():
    with pytest.raises(ValueError, match="割り切りません"):
        rear_loaded(CHIBA_50, 24, 7)


def test_blocks_from_10min():
    assert list(blocks_from_10min([6.0, 12.0, 30.0], 20)) == [9.0, 15.0]
    with pytest.raises(ValueError, match="10の倍数"):
        blocks_from_10min([1.0], 15)


# オリフィスの寸法の探索 ------------------------------------------------------------
def test_size_orifice_for_peak_hits_target():
    q = np.concatenate([storm_inflow(10), np.zeros(108)])
    pond = PrismaticPond(700.0, 3.0)
    rect, r = size_orifice_for_peak(q, pond, 0.0, 0.03, hy=HY)
    assert r.out.max() == pytest.approx(0.03, rel=1e-6) and r.spill == 0
    assert route(Facility(pond, rect), q).out.max() == pytest.approx(0.03, rel=1e-6)


def test_size_orifice_reports_small_pond():
    q = np.concatenate([storm_inflow(10), np.zeros(108)])
    with pytest.raises(ValueError, match="容量が足りません"):
        size_orifice_for_peak(q, PrismaticPond(50.0, 1.0), 0.0, 0.01, hy=HY)


def test_required_pond_scale_needs_a_reachable_hole():
    """満水位で目標の放流量にできるオリフィスがない（オリフィスの敷高が満水位に近すぎる）ときは、そのオリフィスで倍率を探さずエラーにする。"""
    q = np.concatenate([storm_inflow(10), np.zeros(108)])
    with pytest.raises(ValueError, match="範囲にありません"):
        required_pond_scale(q, PrismaticPond(300.0, 2.0), 1.99, 5.0, hy=HY)


def test_size_orifice_needs_sill_below_hwl():
    q = np.concatenate([storm_inflow(10), np.zeros(108)])
    for f in (size_orifice_for_peak, required_pond_scale):
        with pytest.raises(ValueError, match="敷高 3.0 m が満水位"):
            f(q, PrismaticPond(700.0, 3.0), 3.0, 0.03, hy=HY)


def test_size_orifice_stopped_at_range_end_is_an_error():
    """最も大きいオリフィスでも目標の最大放流量に達しないときは、範囲の端の寸法を黙って返さない。"""
    q = np.concatenate([storm_inflow(10), np.zeros(108)])
    with pytest.raises(ValueError, match="範囲にありません.*止まりました"):
        size_orifice_for_peak(q, PrismaticPond(5000.0, 3.0), 0.0, 100.0, hy=HY)


def test_required_pond_scale_other_outlets_already_exceed_target():
    q = np.concatenate([storm_inflow(10), np.zeros(108)])
    with pytest.raises(ValueError, match="ほかの放流施設だけで"):
        required_pond_scale(q, PrismaticPond(50.0, 1.0), 0.0, 0.01, others=(Spillway(0.5, 1.0, HY),), hy=HY)


def test_required_pond_scale_beyond_limit_is_an_error():
    """面積の倍率が上限を超えるときは、上限の池を黙って返さない。"""
    q = np.concatenate([storm_inflow(10), np.zeros(108)])
    with pytest.raises(ValueError, match="2 倍にしても溢れます"):
        required_pond_scale(q, PrismaticPond(50.0, 1.0), 0.0, 0.01, k_max=2.0, hy=HY)


def test_depth_design_with_unreachable_allowable_discharge_is_an_error():
    """満水位で許容放流量にできるオリフィスがないときは、最も近いオリフィスで池を決めずにエラーにする。"""
    with pytest.raises(ValueError, match="オリフィスが作れません"):
        design_for_depth(storm_inflow(10), 100.0, 1.0, hy=HY)


def _check_nodes(q, r, fac, capped, dt):
    """各節点の水位が route の決めどおりか: 溢水の節点は満水位、右辺が0以下は池底、ポンプの段は式を満たす放流量、
    それ以外は S(h) ≦ rhs ＜ S(h の次の float)。"""
    from amamizu.domain.pond import _S
    pond, outlet = fac.pond, fac.outlet
    s_hwl = _S(pond, outlet, pond.hwl, dt)
    v_prev = 0.0
    for i in range(1, len(q)):
        rhs = v_prev + dt / 2 * (q[i - 1] + q[i] - r.out[i - 1])
        hh = r.h[i]
        if capped and rhs >= s_hwl:
            assert hh == pond.hwl, i
        elif rhs <= 0:
            assert hh == pond.bottom, i
        elif hh in outlet.steps:
            assert 0 <= r.out[i] <= outlet.discharge(hh) and pond.volume(hh) + dt / 2 * r.out[i] == pytest.approx(rhs), i
        else:
            assert _S(pond, outlet, hh, dt) <= rhs < _S(pond, outlet, math.nextafter(hh, math.inf), dt), i
        v_prev = pond.volume(hh)


def _balance(q, r, pond, dt):
    """流入の総量と、放流・溢水・最後の貯留量の和の差 m3。放流量は、節点の放流量の台形公式の和から、池が空になった刻みで
    放流しなかった池にない量（overdraw）を引いたもの（0 になるのが正しい）。"""
    return np.trapezoid(q, dx=dt) - (np.trapezoid(r.out, dx=dt) + r.spill + pond.volume(r.h[-1])) + sum(x for _, x in r.overdraw)


def test_extension_above_table_follows_storage():
    """表より上は、面積の列の有無によらず、最後の区間の容量の増え方で延びる（計算の基準は容量）。最上段の面積が0でも同じ。
    表より上の水面積は、その増え方（延長した池の水面積）。"""
    with_areas = TablePond.from_areas([0.0, 1.0, 2.0], [100.0, 50.0, 0.0])          # 容量 0、75、100
    volumes_only = TablePond([0.0, 1.0, 2.0], None, [0.0, 75.0, 100.0])
    for pond in (with_areas, volumes_only):
        assert pond.rise_rate() == pytest.approx(25.0)
        assert pond.volume(3.0) == pytest.approx(125.0) and pond.level(125.0) == pytest.approx(3.0)
    assert with_areas.area_at(2.0) == 0.0 and with_areas.area_at(3.0) == pytest.approx(25.0)


@pytest.mark.parametrize("outlet", [SquareOrifice(0.05, HY), PumpTable((0.5, 1.0), (0.01, 0.02))])
@pytest.mark.parametrize("capped", [True, False])
def test_route_does_not_depend_on_area_column(outlet, capped):
    """同じ水位と容量の表なら、面積の列があってもなくても、追跡の結果（水位・放流量・溢水量）は同じ。水収支も合う。"""
    q = np.array([0.0] + [0.5] * 12 + [0.0] * 30)
    runs = [route(Facility(pond, outlet), q, capped=capped, dt=600)
            for pond in (TablePond.from_areas([0.0, 1.0, 2.0], [100.0, 50.0, 0.0]), TablePond([0.0, 1.0, 2.0], None, [0.0, 75.0, 100.0]))]
    assert np.array_equal(runs[0].h, runs[1].h) and np.array_equal(runs[0].out, runs[1].out) and runs[0].spill == runs[1].spill
    pond = TablePond([0.0, 1.0, 2.0], None, [0.0, 75.0, 100.0])
    assert (runs[0].h.max() > pond.levels[-1]) == (not capped)
    assert _balance(q, runs[0], pond, 600) == pytest.approx(0, abs=1e-6)


def test_route_solution_is_the_float_below_the_root():
    """各節点の水位は S(h) ≦ rhs ＜ S(h の次の float) を満たす。面積がごく小さく容量が大きい池では、容量から逆算した水位の丸めの誤差が
    固定の余裕（1e-9 m など）より大きくなり、余裕を固定すると区間が解を挟まずに誤った水位を返す。放流のない水位で確かめる。"""
    from amamizu.domain.pond import _S
    pond = PrismaticPond(1e-4, 1e12)
    fac = Facility(pond, RectOrifice(1e11, 0.01, 0.01, HY))
    q = np.full(9, 3000.0)
    for capped in (True, False):
        r = route(fac, q, capped=capped, dt=600)
        v_prev = 0.0
        for i in range(1, len(q)):
            rhs = v_prev + 300 * (q[i - 1] + q[i] - r.out[i - 1])
            hh = r.h[i]
            if not (capped and hh == pond.hwl and rhs >= _S(pond, fac.outlet, hh, 600)):
                assert _S(pond, fac.outlet, hh, 600) <= rhs < _S(pond, fac.outlet, math.nextafter(hh, math.inf), 600), i
            v_prev = pond.volume(hh)
        assert _balance(q, r, pond, 600) == pytest.approx(0, abs=1e-9 * np.trapezoid(q, dx=600))


@pytest.mark.parametrize("outlet", [PumpTable((0.0, 1.0), (0.02, 0.05)), RectOrifice(-0.5, 0.1, 0.1, HY)])
def test_route_rejects_discharge_from_empty_pond(outlet):
    """池底で放流量が正の施設（池底で排水するポンプ、池底より低いオリフィス）は、空の池から放流して水が増えるのでエラーにする。"""
    q = np.array([0.0, 0.005, 0.005, 0.005, 0.0, 0.0])
    with pytest.raises(ValueError, match="空の池からは放流できない"):
        route(Facility(PrismaticPond(100.0, 2.0), outlet), q, dt=600)


def test_route_is_the_same_for_negative_bottom():
    """池底を負の標高にしても（標高で書いた池）、同じ水深・放流量になる。"""
    q = np.concatenate([storm_inflow(10), np.zeros(288)])
    runs = {}
    for bottom in (0.0, -5.0):
        pond = TablePond.from_areas([bottom, bottom + 1.0, bottom + 2.0], [300.0, 300.0, 300.0])
        runs[bottom] = route(Facility(pond, RectOrifice(bottom, 0.08, 0.08, HY)), q)
    assert np.allclose(runs[-5.0].h + 5.0, runs[0.0].h, atol=1e-9)
    assert np.allclose(runs[-5.0].out, runs[0.0].out, rtol=1e-9, atol=1e-15)


def test_route_balance_with_overdraw():
    """右辺が負になる節点（池が1刻みの中で空になる）では、その刻みの放流量を池にあった水と流入の和にする。
    節点の放流量の台形公式の和から、放流しなかった池にない量（overdraw）を引くと、水収支が合う。"""
    fac = Facility(PrismaticPond(1.0, 2.0), RectOrifice(0.0, 1.0, 1.0, HY))
    q = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
    r = route(fac, q, capped=False, dt=600)
    assert r.overdraw
    _check_nodes(q, r, fac, False, 600)
    assert _balance(q, r, fac.pond, 600) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("capped", [True, False])
def test_route_rejects_overflowing_right_hand_side(capped):
    """流入量は有限でも、連続式の右辺が float の範囲を超えるときはエラーにする（終わらない、溢水量が無限大になる、を防ぐ）。"""
    with pytest.raises(ValueError, match="float の範囲を超えました"):
        route(Facility(PrismaticPond(300.0, 2.0), SquareOrifice(0.05, HY)), np.array([0.0, 1e306, 0.0]), capped=capped, dt=600)


def test_spillway_discharge_beyond_float_range_is_inf():
    """洪水吐の放流量が float の範囲を超える水位では +inf（Python の例外にしない）。容量の増え方がごく小さい池でも追跡が終わり、水収支が合う。"""
    assert Spillway(0.5, 1.0, HY).discharge(1e300) == math.inf
    pond = TablePond([0.0, 1.0], None, [0.0, 1e-300])
    fac = Facility(pond, Spillway(0.5, 1.0, HY))
    q = np.array([0.0, 1.0, 1.0, 0.0])
    r = route(fac, q, capped=False, dt=600)
    assert np.all(np.isfinite(r.h))
    _check_nodes(q, r, fac, False, 600)
    assert _balance(q, r, pond, 600) == pytest.approx(0, abs=1e-9 * np.trapezoid(q, dx=600))


def test_route_when_level_cannot_be_inverted():
    """右辺 ÷ 容量の増え方 が float の範囲を超えて水位を逆算できなくても、真の水位が有限なら求める（満水位から上端を上げる）。
    真の水位も float の範囲を超えるとき（放流量に上限があるポンプだけ）は、表せる解がないのでエラーにする。"""
    pond = PrismaticPond(1e-300, 1.0)
    q = np.array([0.0, 1e200, 0.0])
    fac = Facility(pond, Spillway(0.5, 1.0, HY))
    r = route(fac, q, capped=False, dt=600)
    assert np.all(np.isfinite(r.h)) and r.h.max() > 1e100
    _check_nodes(q, r, fac, False, 600)
    with pytest.raises(ValueError, match="水位が float の範囲を超えます"):
        route(Facility(pond, PumpTable((0.5, 1.0), (0.01, 0.02))), np.array([0.0, 1e10, 0.0]), capped=False, dt=600)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_route_rejects_non_finite_inflow(bad):
    """流入量に NaN・無限大があれば、水位の探索に入らずエラーにする（NaN では二分法が終わらない）。"""
    q = np.array([0.0, 0.1, bad, 0.0])
    with pytest.raises(ValueError, match="NaN"):
        route(Facility(PrismaticPond(100.0, 2.0), SquareOrifice(0.05, HY)), q, dt=600)


PONDS = {
    "柱状": PrismaticPond(300.0, 2.0),
    "表・面積が上で広がる": TablePond.from_areas([10.0, 11.0, 12.0], [100.0, 200.0, 400.0]),
    "表・面積が上で狭まる": TablePond.from_areas([0.0, 1.0, 2.0], [400.0, 50.0, 1e-9]),
    "表・最上段の面積が0": TablePond.from_areas([0.0, 1.0, 2.0], [300.0, 100.0, 0.0]),
    "表・池底の面積が0": TablePond.from_areas([0.0, 0.5, 2.0], [0.0, 200.0, 300.0]),
    "表・容量だけ": TablePond([0.0, 1.0, 2.0], None, [0.0, 250.0, 600.0]),
    "表・満水位が途中": TablePond.from_areas([0.0, 1.0, 2.0, 3.0], [200.0, 250.0, 300.0, 350.0], hwl=1.5),
}
OUTLETS = {
    "オリフィス": lambda b: RectOrifice(b, 0.1, 0.1, HY),
    "オリフィスと洪水吐": lambda b: Outlets((RectOrifice(b, 0.1, 0.1, HY), Spillway(b + 1.5, 1.0, HY))),
    "ポンプ": lambda b: PumpTable((b + 0.3, b + 1.0), (0.02, 0.05)),
    "オリフィスとポンプ": lambda b: Outlets((RectOrifice(b, 0.05, 0.05, HY), PumpTable((b + 0.3, b + 1.0), (0.02, 0.05)))),
}


@pytest.mark.parametrize("pond_name", PONDS)
@pytest.mark.parametrize("outlet_name", OUTLETS)
@pytest.mark.parametrize("capped", [True, False])
def test_route_conserves_water(pond_name, outlet_name, capped):
    """池の形・放流施設・溢水の扱いのどの組み合わせでも、水量が消えたり増えたりしない
    （右辺が負になる節点がなければ、流入の総量 ＝ 放流 ＋ 溢水 ＋ 最後の貯留量）。流入は満水位を超える大きさ。"""
    pond = PONDS[pond_name]
    fac = Facility(pond, OUTLETS[outlet_name](pond.bottom))
    q = np.concatenate([storm_inflow(10, area_ha=20.0), np.zeros(288)])
    r = route(fac, q, capped=capped)
    assert not r.overdraw
    assert r.h.max() >= pond.hwl                    # 満水位に達する流入で確かめる
    _check_nodes(q, r, fac, capped, 600)
    assert _balance(q, r, pond, 600) == pytest.approx(0, abs=1e-6 * np.trapezoid(q, dx=600))


def test_design_reports_pond_shortage():
    """オリフィスの寸法を求める設計で池が溢れるときは、エラーにせず、面積の倍率・必要な容量・不足量を要約に入れ、判定を N.G にする。
    追跡の結果は、面積を広げた池（溢れない）のもの。"""
    inflow = planned_inflow(CHIBA_50, [Catchment("区域", 1.0, 0.8)], 10, "rear")
    res = run_design(inflow, PrismaticPond(300.0, 2.0), [{"sill_m": 0.0}], None, 2520, "size_orifice", 0.02, hy=HY)
    sh = res.summary["池の不足"]
    assert sh["面積の倍率"] > 1 and sh["必要な容量_m3"] == pytest.approx(600.0 * sh["面積の倍率"])
    assert sh["不足量_m3"] == pytest.approx(sh["必要な容量_m3"] - 600.0)
    assert res.summary["溢水量_m3"] == 0 and res.summary["判定"]["総合"] == "N.G"
    assert res.summary["判定"]["溢水"].startswith("N.G（入力の池では溢れる")


@pytest.mark.parametrize("pond", [PrismaticPond(300.0, 2.0), TablePond([0, 1, 2], [200, 300, 400], [0, 250, 600])])
def test_required_pond_scale(pond):
    """容量が足りない池で、水深はそのままで溢れなくなる最小の面積の倍率。少し小さいと溢れ、少し大きければオリフィスを求め直しても
    同じ寸法のオリフィスで溢れない。"""
    q = np.concatenate([storm_inflow(10), np.zeros(108)])
    hole, r = size_orifice_for_peak(q, pond, 0.0, 0.02, allow_spill=True, hy=HY)
    assert r.spill > 0
    k, hole, big, r = required_pond_scale(q, pond, 0.0, 0.02, hy=HY)
    assert k > 1 and r.spill == 0 and big.capacity == pytest.approx(pond.capacity * k)
    assert r.out.max() == pytest.approx(0.02, rel=1e-3) and r.h.max() == pytest.approx(pond.hwl, abs=1e-3)
    assert route(Facility(pond.scaled(k * 0.99), hole), q, capped=True).spill > 0
    resized, r2 = size_orifice_for_peak(q, pond.scaled(k * 1.001), 0.0, 0.02, hy=HY)
    assert r2.spill == 0 and resized.width == pytest.approx(hole.width, rel=1e-3)


# 使用ケース ---------------------------------------------------------------
def test_hydrograph_times_must_follow_tc():
    with pytest.raises(ValueError, match="tc＝10"):
        hydrograph_inflow([10, 20, 40], [1.0, 2.0, 3.0], 10)
    assert list(hydrograph_inflow([10, 20], [1.0, 2.0], 10).nodes) == [0.0, 1.0, 2.0, 0.0]


def test_hydrograph_rain_must_not_be_negative():
    """ハイドログラフの降雨強度の列の負の値を、そのまま rain.csv とグラフに出力しない。"""
    with pytest.raises(ValueError, match="降雨強度に空欄か負の値"):
        hydrograph_inflow([10, 20], [1.0, 2.0], 10, rain=[5.0, -1.0])
    assert list(hydrograph_inflow([10, 20], [1.0, 2.0], 10, rain=[5.0, 0.0]).rain) == [5.0, 0.0]


def test_hydrograph_reverse():
    """inflow_csv.reverse が true なら値の並びを逆にする（true か false であることは study.json を読むときに確かめる）。"""
    import pandas as pd

    from amamizu.adapters.design_inputs import hydrograph_from_frame
    df = pd.DataFrame({"時刻_分": [10, 20, 30], "流入量_m3s": [1.0, 2.0, 3.0]})
    assert list(hydrograph_from_frame(df, {"reverse": True}, 10).nodes) == [0.0, 3.0, 2.0, 1.0, 0.0]
    assert list(hydrograph_from_frame(df, {}, 10).nodes) == [0.0, 1.0, 2.0, 3.0, 0.0]


def test_catchments_are_summed():
    inflow = planned_inflow(CHIBA_50, [Catchment("a", 1.0, 0.9), Catchment("b", 2.0, 0.6)], 10, "rear")
    assert np.allclose(inflow.nodes, storm_inflow(10, 1.0, 0.9) + storm_inflow(10, 2.0, 0.6))


def test_run_design_rejects_bad_levels():
    inflow = planned_inflow(CHIBA_50, [Catchment("a", 1.0, 0.8)], 10, "rear")
    pond = TablePond([10.0, 12.0], [500.0, 500.0], [0.0, 1000.0])
    with pytest.raises(ValueError, match="池底"):
        run_design(inflow, pond, [{"sill_m": 9.0, "width_m": 0.1, "height_m": 0.1}], None, 2520, hy=HY)
    with pytest.raises(ValueError, match="洪水吐"):
        run_design(inflow, pond, [{"sill_m": 10.0, "width_m": 0.1, "height_m": 0.1}], {"crest_m": 9.5, "width_m": 1}, 2520, hy=HY)
    with pytest.raises(ValueError, match="短い"):
        run_design(inflow, pond, [{"sill_m": 10.0, "width_m": 0.1, "height_m": 0.1}], None, 600, hy=HY)


# study.json とコマンド -------------------------------------------------------
def write_design(tmp_path, **over):
    (tmp_path / "input").mkdir(exist_ok=True)
    (tmp_path / "input" / "pond.csv").write_text("水位_m,面積_m2,容量_m3\n10.0,500,0\n11.0,600,550\n13.0,800,1950\n",
                                               encoding="utf-8")
    conf = {"kind": "design", "guideline": "chiba_2006", "title": "試験", "tc_min": 20,
            "rain": {"formula": {"type": "cleveland", "a": 2439, "b": 11.9, "n": "3/4"}, "shape": "rear"},
            "catchments": [{"name": "浸透なし", "area_ha": 1.0, "f": 0.9}],
            "pond": {"table": {"file": "input/pond.csv"}},
            "orifice": {"sill_m": 10.0, "width_m": 0.1, "height_m": 0.1}} | over
    (tmp_path / "study.json").write_text(json.dumps(conf, ensure_ascii=False), encoding="utf-8")
    return tmp_path


@pytest.mark.needs_japanese_font
def test_design_study_runs_from_command(tmp_path, capsys):
    study = write_design(tmp_path)
    main(["study", str(study)])
    table = (study / "out" / "route.csv").read_text(encoding="utf-8-sig")
    rows = table.splitlines()
    assert rows[0].startswith("計算時間(min),降雨強度(mm/h),流入量(m3/s)")
    assert rows[1].startswith("0,") and rows[2].startswith("20,")                # tc＝20 分ごと
    assert len(rows) - 1 == (1440 + 18 * 60) // 20 + 1                           # 既定の終了は流入の後18時間
    assert "最大放流量" in capsys.readouterr().out


def test_design_study_rejects_unknown_nested_keys(tmp_path):
    with pytest.raises(ValueError, match=r"pond\.table\.levl_col"):
        load_study(write_design(tmp_path, pond={"table": {"file": "input/pond.csv", "levl_col": "水位"}}))
    with pytest.raises(ValueError, match=r"catchments\[0\]\.are"):
        load_study(write_design(tmp_path, catchments=[{"name": "a", "are": 1, "f": 0.9}]))


def test_design_study_needs_one_inflow(tmp_path):
    study = write_design(tmp_path, inflow_csv={"file": "input/x.csv"})
    with pytest.raises(ValueError, match="一つ"):
        main(["study", str(study)])


def test_design_study_has_no_report(tmp_path):
    with pytest.raises(ValueError, match="レポート"):
        main(["study", str(write_design(tmp_path)), "--report", str(tmp_path / "b.md")])


def test_command_shows_the_reason_of_an_error(tmp_path, capsys):
    """amamizu コマンドは、理由を書いた誤りを経過（Traceback）なしで表示し、終了コード1にする。"""
    study = write_design(tmp_path, orifice={"sill_m": 9.0, "width_m": 0.1, "height_m": 0.1})
    with pytest.raises(SystemExit) as e:
        run(["study", str(study)])
    assert e.value.code == 1
    assert capsys.readouterr().err == "エラー（放流施設）: オリフィスの敷高 9.0 m が池底 10.0 m より低いです。敷高を池底以上にしてください。\n"
    (study / "study.json").write_text('{"kind": "design", "tc_mn": 10}', encoding="utf-8")
    with pytest.raises(SystemExit):
        run(["study", str(study)])
    assert capsys.readouterr().err == ("エラー: study.json に不明な項目（amamizu が使わない項目）があります: tc_mn（「a.b」は a の中の b、「[0]」はリストの1番目）。"
                                          "項目名の書き間違いなら直し、不要なら削除してください（書き間違いのまま既定値で計算しないよう、中止しました）。\n")


def test_command_keeps_the_traceback_of_an_unexpected_error(tmp_path):
    """pandas などが発生させた誤りは、理由が文に書かれていないことがあるので、原因をたどれるよう例外のまま送出する。"""
    study = write_design(tmp_path)
    (study / "input" / "pond.csv").write_text("水位_m,面積_m2\n10,500\n11,600,5,6\n", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        run(["study", str(study)])
    assert type(e.value.__cause__.__cause__).__module__.startswith("pandas")


def test_missing_pond_column_is_an_error(tmp_path):
    from amamizu.app.design_runner import ConfigError
    study = write_design(tmp_path, pond={"table": {"file": "input/pond.csv", "level_col": "標高"}})
    with pytest.raises(ConfigError, match="標高") as e:
        main(["study", str(study)])
    assert e.value.section == "池"                      # WebGUI ではこの誤りを「池」のタブに表示する


# 貯留施設の検証の tc ------------------------------------------------------------
def test_storage_tc_must_be_a_divisor_of_60_and_multiple_of_10():
    from amamizu.usecases.storage_study import ModelConditions
    ModelConditions(tc_min=20)
    for bad in (5, 15, 40):
        with pytest.raises(ValueError, match="tc_min"):
            ModelConditions(tc_min=bad)


def test_storage_facility_follows_tc():
    """tc＝20 のモデル施設は、20分の計画降雨と20分刻みの追跡で決まる（10分とは容量が変わる）。"""
    from amamizu.adapters.guideline.json_guideline import JsonGuideline
    from amamizu.usecases.storage_study import ModelConditions, StorageStudy
    g = JsonGuideline(CHIBA)
    s10 = StorageStudy(g, None, ModelConditions())
    s20 = StorageStudy(g, None, ModelConditions(tc_min=20))
    f20 = s20.facility("千葉")
    assert len(f20.design_result.h) == 24 * 3 + 2
    assert f20.capacity != pytest.approx(s10.facility("千葉").capacity, rel=1e-6)
    _, res = s20.route_rain("千葉", np.full(6, 5.0), 10, "2026-01-01")   # 1時間 30mm/hr の10分値
    assert len(res.pond_inflow) == 3 + 48 * 3 + 2


def test_guideline_hydraulics():
    from amamizu.adapters.guideline.json_guideline import JsonGuideline
    assert JsonGuideline(CHIBA).hydraulics() == Hydraulics(9.8, 1.8, 0.6)
    minimal = JsonGuideline(Path(__file__).parent / "fixtures" / "guideline_minimal.json")
    with pytest.raises(KeyError, match="hydraulics"):
        minimal.hydraulics()


def test_design_mode_rejects_ignored_items():
    """計算に使わない項目を書いたら、黙って無視せずエラーにする。"""
    inflow = planned_inflow(CHIBA_50, [Catchment("a", 1.0, 0.8)], 10, "rear")
    pond = PrismaticPond(700.0, 3.0)
    with pytest.raises(ValueError, match="width_m"):
        run_design(inflow, pond, [{"sill_m": 0.0, "width_m": 0.1}], None, 2520, "size_orifice", 0.03, hy=HY)
    with pytest.raises(ValueError, match="target_q_m3s"):
        run_design(inflow, pond, [{"sill_m": 0.0, "width_m": 0.1, "height_m": 0.1}], None, 2520, "route", 0.03, hy=HY)


# 円形のオリフィス --------------------------------------------------------------------
def test_circle_weir_integral_matches_fine_sum():
    """越流の積分は、細かく分けた中点の和と一致する（水面が円の途中・上端・上端より上）。"""
    from amamizu.domain.outlet import circle_weir_integral
    d = 0.2
    for y in (0.002, 0.06, 0.1, 0.199, 0.2, 0.24):
        zt = min(y, d)
        z = (np.arange(400_000) + 0.5) * zt / 400_000
        fine = float(np.sum(2 * np.sqrt(z * (d - z)) * np.sqrt(y - z)) * zt / 400_000)
        assert circle_weir_integral(y, d) == pytest.approx(fine, rel=1e-6)


def test_circle_orifice_formula():
    from amamizu.domain.outlet import CircleOrifice
    import math
    hy = HY
    c = CircleOrifice(5.0, 0.2, hy)
    y = 1e-4                                                # 水位が低いときは (3/2)·1.8·(π/4)·sqrt(D)·y^2
    assert c.discharge(5.0 + y) == pytest.approx(1.5 * 1.8 * math.pi / 4 * math.sqrt(0.2) * y * y, rel=1e-3)
    y = 0.5                                                 # 1.8D 以上はオリフィスの式
    assert c.discharge(5.0 + y) == pytest.approx(0.6 * math.pi * 0.04 / 4 * math.sqrt(2 * 9.8 * (y - 0.1)))
    for k in (1.2, 1.8):                                    # 補間の両端で値がつながる
        assert c.discharge(5.0 + k * 0.2 - 1e-9) == pytest.approx(c.discharge(5.0 + k * 0.2 + 1e-9), rel=1e-6)
    q = [c.discharge(5.0 + v) for v in np.linspace(1e-6, 0.8, 2000)]
    assert np.all(np.diff(q) > 0)
    assert c.discharge(4.9) == 0.0


def test_size_circle_orifice_for_peak():
    from amamizu.domain.outlet import CircleOrifice
    q = np.concatenate([storm_inflow(10), np.zeros(108)])
    hole, r = size_orifice_for_peak(q, PrismaticPond(700.0, 3.0), 0.0, 0.03, shape="circle", hy=HY)
    assert isinstance(hole, CircleOrifice) and r.out.max() == pytest.approx(0.03, rel=1e-6)


def test_orifice_shape_items_are_checked():
    inflow = planned_inflow(CHIBA_50, [Catchment("a", 1.0, 0.8)], 10, "rear")
    pond = PrismaticPond(700.0, 3.0)
    with pytest.raises(ValueError, match="diameter_m"):
        run_design(inflow, pond, [{"sill_m": 0.0, "width_m": 0.1, "height_m": 0.1, "diameter_m": 0.1}], None, 2520, hy=HY)
    with pytest.raises(ValueError, match="diameter_m"):
        run_design(inflow, pond, [{"sill_m": 0.0, "shape": "circle"}], None, 2520, hy=HY)
    with pytest.raises(ValueError, match="扱えません"):
        run_design(inflow, pond, [{"sill_m": 0.0, "shape": "oval", "diameter_m": 0.1}], None, 2520, hy=HY)
    with pytest.raises(ValueError, match="allowable_q_m3s"):
        run_design(inflow, pond, [{"sill_m": 0.0, "shape": "circle"}], None, 2520, "size_orifice", 0.03, allowable_q=0.03, hy=HY)
    res = run_design(inflow, pond, [{"sill_m": 0.0, "shape": "circle", "diameter_m": 0.1}], None, 2520, hy=HY)
    assert res.summary["オリフィス"] == [(0.0, "円形 直径 0.1000 m")]


# グラフ ---------------------------------------------------------------------
@pytest.mark.needs_japanese_font
def test_design_study_writes_graphs(tmp_path, capsys):
    study = write_design(tmp_path, orifice={"sill_m": 10.0, "shape": "circle", "diameter_m": 0.12},
                         spillway={"crest_m": 12.5, "width_m": 1.0}, allowable_q_m3s=0.001)
    main(["study", str(study)])
    for name in ("hydrograph.png", "water_level.png", "pond_curves.png"):
        data = (study / "out" / name).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 10_000
    summary = (study / "out" / "design_summary.md").read_text(encoding="utf-8")
    assert "円形 直径 0.1200 m" in summary and "洪水吐" in summary and "![水位](water_level.png)" in summary
    assert "許容放流量 0.001 m^3/s を超えました" in capsys.readouterr().out


@pytest.mark.needs_japanese_font
def test_design_study_writes_pdf(tmp_path):
    """要約とグラフを清書した PDF を書く。日本語のフォントを埋め込み、作るたびに同じバイト
    （WebGUI の「保存して出力」と amamizu study の out/ を一致させるため）。"""
    study = write_design(tmp_path)
    main(["study", str(study)])
    first = (study / "out" / "design_summary.pdf").read_bytes()
    assert first[:5] == b"%PDF-" and b"/FontFile2" in first            # TrueType のフォントを埋め込んでいる
    assert b"/Subtype /Image" in first                                 # グラフの画像
    main(["study", str(study)])
    assert (study / "out" / "design_summary.pdf").read_bytes() == first


def test_readme_lists_every_japanese_font():
    """README の「日本語のフォント」は、コードが探すフォント（JP_FONTS）をすべて載せている（一覧はコードが唯一の定義）。"""
    from amamizu.adapters.report.design_plots import JP_FONTS
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    assert JP_FONTS and all(name in readme for name in JP_FONTS)


def test_time_text_is_shared():
    """節点の時刻の表記は1つ（hm・clock）。計画降雨は経過時間 H:MM、実測降雨は日時。グラフ・要約・WebGUI が同じ関数を使う。"""
    from amamizu.usecases.design_study import clock, hm
    assert hm(0) == "0:00" and hm(70) == "1:10" and hm(1230) == "20:30" and hm(np.int64(600)) == "10:00"
    assert clock(None, 1230) == "20:30"
    assert clock(datetime.datetime(2013, 10, 15, 12, 0), 1230) == "2013-10-16 08:30"


def test_graphs_need_japanese_font(monkeypatch):
    from amamizu.adapters.report import design_plots
    monkeypatch.setattr(design_plots, "JP_FONTS", ["存在しないフォント"])
    with pytest.raises(GraphError, match="日本語のフォント"):
        design_plots.japanese_font()


@pytest.mark.needs_japanese_font
def test_graphs_refuse_characters_missing_from_font(tmp_path):
    """フォントにない文字（豆腐になる文字）が題名にあれば、グラフを出力しない。"""
    study = write_design(tmp_path, title="調整池 \U0001F600")
    with pytest.raises(GraphError, match="フォントにない文字"):
        main(["study", str(study)])
    assert not any((study / "out").glob("*"))      # 途中まで書いた route.csv・inflow.csv も残さない


@pytest.mark.needs_japanese_font
def test_long_title_is_wrapped_and_legends_do_not_overlap(tmp_path):
    """長い題名でも、凡例が文字や描画の領域に重ならずに出力できる（重なれば _check_legends がエラーにする）。"""
    study = write_design(tmp_path, title="とても長い題の調整池の検討（" + "区域" * 30 + "）",
                         spillway={"crest_m": 12.5, "width_m": 1.0}, allowable_q_m3s=0.02)
    main(["study", str(study)])
    assert (study / "out" / "pond_curves.png").exists()
    from amamizu.adapters.report.design_plots import TITLE_WIDTH, _wrap
    lines = _wrap("とても長い題の調整池の検討（" + "区域" * 30 + "）").splitlines()
    assert len(lines) > 1 and all(len(x) <= TITLE_WIDTH for x in lines)


def test_overlapping_legend_is_an_error():
    """凡例が線や文字に重なる図は出力しない（重なりの確認が実際に働くこと）。"""
    import matplotlib.pyplot as plt
    from amamizu.adapters.report.design_plots import _check_legends
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="line")      # 文字の種類は重なりの確認に関係しない（フォントを設定しない）
    ax.legend(loc="center")
    fig.canvas.draw()
    with pytest.raises(GraphError, match="重なり、文字が読めなくなる"):
        _check_legends(fig)
    plt.close(fig)


# 2段のオリフィス、ポンプ、判定 ------------------------------------------------------------
def test_two_tier_orifices_are_summed():
    """2段のオリフィスの放流量は、各オリフィスの放流量の和（上のオリフィスは敷高を超えてから流れる）。"""
    from amamizu.domain.outlet import CircleOrifice
    inflow = planned_inflow(CHIBA_50, [Catchment("a", 1.0, 0.8)], 10, "rear")
    pond = TablePond([10.0, 13.0], [500.0, 500.0], [0.0, 1500.0])
    res = run_design(inflow, pond, [{"sill_m": 11.5, "shape": "circle", "diameter_m": 0.2},
                                    {"sill_m": 10.0, "width_m": 0.08, "height_m": 0.08}], None, 2520, hy=HY)
    low, high = RectOrifice(10.0, 0.08, 0.08, HY), CircleOrifice(11.5, 0.2, HY)
    for h in (10.5, 11.4, 11.9, 12.8):
        assert res.facility.outlet.discharge(h) == pytest.approx(low.discharge(h) + high.discharge(h))
    assert [s for s, _ in res.summary["オリフィス"]] == [10.0, 11.5]              # 敷高の低い順


def test_pump_table():
    from amamizu.domain.outlet import PumpTable
    p = PumpTable((10.5, 11.0, 12.0), (0.02, 0.05, 0.05))
    assert p.discharge(10.4) == 0.0 and p.discharge(10.75) == pytest.approx(0.035) and p.discharge(13.0) == 0.05
    with pytest.raises(ValueError, match="減らない"):
        PumpTable((10.0, 11.0), (0.05, 0.02))
    for levels, q in [((10.0, None, 12.0), (0.01, 0.02, 0.03)), ((10.0, 11.0), (0.01, float("nan")))]:
        with pytest.raises(ValueError, match="空欄か数でない値"):      # 空欄（WebGUI で消した欄）を NaN のまま通さない
            PumpTable(levels, q)
    inflow = planned_inflow(CHIBA_50, [Catchment("a", 1.0, 0.8)], 10, "rear")
    pond = TablePond([10.0, 13.0], [500.0, 500.0], [0.0, 1500.0])
    orifice = [{"sill_m": 10.0, "width_m": 0.05, "height_m": 0.05}]
    without = run_design(inflow, pond, orifice, None, 2520, hy=HY)
    with_pump = run_design(inflow, pond, orifice, None, 2520, pump={"levels_m": [10.5, 11.0], "q_m3s": [0.02, 0.05]}, hy=HY)
    assert with_pump.summary["最大水位_m"] < without.summary["最大水位_m"]
    assert "ポンプ" in with_pump.pump.describe()


def test_route_solves_each_node_to_adjacent_floats():
    """各節点の水位 h は S(h) ≦ rhs ＜ S(h の次の float) を満たす（S は単調なので一つに決まる。
    求解の方法を変えても、隣り合う float まで詰めれば同じ h になる）。溢れない池・表の池・円形・洪水吐で確かめる。"""
    from amamizu.domain.outlet import CircleOrifice
    q = np.concatenate([storm_inflow(10), np.zeros(288)])
    tp = TablePond([10.0, 10.5, 11.0, 12.0], [0.0, 300.0, 300.0, 600.0], [0.0, 75.0, 225.0, 675.0])
    cases = [Facility(PrismaticPond(1500.0, 2.0), SquareOrifice(0.08, HY)),
             Facility(tp, CircleOrifice(10.0, 0.1, HY)),
             Facility(PrismaticPond(900.0, 2.0), Outlets((RectOrifice(0.0, 0.06, 0.06, HY), Spillway(1.2, 0.5, HY))))]
    for fac in cases:
        r = route(fac, q, capped=False)
        _check_nodes(q, r, fac, False, 600)


def test_pump_step_keeps_mass():
    """ポンプの最初の排水量が正（最初の水位で0から段になる）とき、段の上に解がある節点は水位を段に置き、
    放流量を式が成り立つ値（0〜最初の排水量。間欠運転）にする。水収支（流入 ＝ 貯留 ＋ 放流）が合う。"""
    from amamizu.domain.outlet import PumpTable
    pond = TablePond([0.0, 3.0], [500.0, 500.0], [0.0, 1500.0])
    fac = Facility(pond, PumpTable((0.5, 1.0), (0.02, 0.05)))
    q = np.concatenate([[0.0], np.full(60, 0.008), np.zeros(80)])
    r = route(fac, q)
    at = r.h == 0.5
    assert at.sum() > 0 and np.all((0 < r.out[at]) & (r.out[at] < 0.02))
    inflow = 600.0 * (q[:-1] + q[1:]).sum() / 2
    outflow = 600.0 * (r.out[:-1] + r.out[1:]).sum() / 2
    assert inflow == pytest.approx(pond.volume(r.h[-1]) + outflow, abs=1e-6)


def test_inputs_that_break_routing_are_errors():
    """追跡の水位が一つに決まらない入力はエラー: オリフィスの係数（補間区間で放流量が減る）、池底で排水するポンプ。"""
    with pytest.raises(ValueError, match="途中で減ります"):
        Hydraulics(9.8, 1.8, 0.2)
    inflow = planned_inflow(CHIBA_50, [Catchment("a", 1.0, 0.8)], 10, "rear")
    pond = TablePond([10.0, 13.0], [500.0, 500.0], [0.0, 1500.0])
    orifice = [{"sill_m": 10.0, "width_m": 0.05, "height_m": 0.05}]
    with pytest.raises(ValueError, match="池底"):
        run_design(inflow, pond, orifice, None, 2520, pump={"levels_m": [9.5, 11.0], "q_m3s": [0.0, 0.05]}, hy=HY)
    with pytest.raises(ValueError, match="池底で排水"):
        run_design(inflow, pond, orifice, None, 2520, pump={"levels_m": [10.0, 11.0], "q_m3s": [0.01, 0.05]}, hy=HY)


def test_overdraw_is_noticed():
    """刻みの中で池が空になる刻み（右辺が負）は、水位を池底・放流量を0にし、その刻みの放流量を池にあった水と流入の和にして、
    注意で知らせる。例 hydrograph（池底の面積 79 m^2、tc 20分）は、540〜560 分に空になり、放流量は 13.6 m^3
    （前後の時刻の放流量の平均からは 23.1 m^3。池にない 9.5 m^3 は放流しない）。"""
    from amamizu.app.design_runner import trial
    d = Path(__file__).parent / "fixtures" / "design_cases" / "hydrograph"
    run = trial(json.loads((d / "study.json").read_text(encoding="utf-8")), d)
    [(node, v)] = run.result.routing.overdraw
    assert node * 20 == 560 and v == pytest.approx(9.53, abs=0.01)
    r, fac = run.result.routing, run.result.facility
    nodes = run.result.pond_inflow
    released = fac.pond.volume(r.h[node - 1]) + 600 * (nodes[node - 1] + nodes[node])   # 池にあった水と流入の和
    assert 600 * r.out[node - 1] - v == pytest.approx(released) and released == pytest.approx(13.6, abs=0.05)
    assert r.h[node] == fac.pond.bottom and r.out[node] == 0
    assert "560 分に池が空になりました（貯留量 0 m^3）。" in run.notices
    inflow = planned_inflow(CHIBA_50, [Catchment("a", 1.0, 0.8)], 10, "rear")
    assert not run_design(inflow, PrismaticPond(1150.0, 2.0), [{"sill_m": 0.0, "width_m": 0.08, "height_m": 0.08}],
                          None, 2520, hy=HY).routing.overdraw


def test_overdraw_note_lists_the_first_times():
    """池が空になった時刻は先頭 limit 個を並べ、ほかは回数を示す。"""
    from types import SimpleNamespace
    r = SimpleNamespace(overdraw=[(k, 1.0) for k in (3, 5, 9)])
    assert overdraw_note(r, 10, 2) == "30 分、50 分、ほか 1 回に池が空になりました（貯留量 0 m^3）。"
    assert overdraw_note(r, 10, 3) == "30 分、50 分、90 分に池が空になりました（貯留量 0 m^3）。"


def test_judgements():
    """判定: 許容放流量以下か、容量の余裕（指定したとき）、溢水。余裕量と余裕率はいつも書く。"""
    inflow = planned_inflow(CHIBA_50, [Catchment("a", 1.0, 0.8)], 10, "rear")
    pond = TablePond([10.0, 13.0], [500.0, 500.0], [0.0, 1500.0])
    orifice = [{"sill_m": 10.0, "width_m": 0.08, "height_m": 0.08}]
    res = run_design(inflow, pond, orifice, None, 2520, allowable_q=1.0, storage_margin=30, hy=HY)
    s = res.summary
    assert s["余裕量_m3"] == pytest.approx(1500.0 - s["最大貯留量_m3"])
    assert s["判定"]["放流量"].startswith("O.K") and s["判定"]["総合"] == "O.K"
    tight = run_design(inflow, pond, orifice, None, 2520, allowable_q=0.001, storage_margin=s["余裕量_m3"] + 1, hy=HY)
    assert tight.summary["判定"]["放流量"].startswith("N.G") and tight.summary["判定"]["容量の余裕"].startswith("N.G")
    assert tight.summary["判定"]["総合"] == "N.G"
    assert "容量の余裕" not in run_design(inflow, pond, orifice, None, 2520, hy=HY).summary["判定"]


def test_orifice_and_orifices_are_exclusive(tmp_path):
    study = write_design(tmp_path, orifices=[{"sill_m": 10.0, "width_m": 0.1, "height_m": 0.1}])
    with pytest.raises(ValueError, match="オリフィスには"):
        main(["study", str(study)])
    inflow = planned_inflow(CHIBA_50, [Catchment("a", 1.0, 0.8)], 10, "rear")
    with pytest.raises(ValueError, match="1つ"):
        run_design(inflow, PrismaticPond(700.0, 3.0), [{"sill_m": 0.0}, {"sill_m": 1.0}], None, 2520, "size_orifice", 0.03, hy=HY)


@pytest.mark.needs_japanese_font
def test_design_summary_has_judgements(tmp_path):
    study = write_design(tmp_path, allowable_q_m3s=0.5, judge={"storage_margin_m3": 30},
                         pump={"levels_m": [10.5, 11.0], "q_m3s": [0.01, 0.02]})
    main(["study", str(study)])
    text = (study / "out" / "design_summary.md").read_text(encoding="utf-8")
    assert "| 判定（放流量） |" in text and "| 判定（容量の余裕） |" in text and "| 判定（総合） |" in text
    assert "| ポンプ |" in text and "余裕率" in text


def test_spill_text_never_shows_zero_when_spilling():
    """溢れている（判定は N.G）のに「0.0」と表示しない。"""
    from amamizu.app.design_runner import spill_text
    assert spill_text(0.0) == "0.0"
    assert spill_text(0.013123) == "0.013"
    assert spill_text(0.00004) == "0.00004"
    assert spill_text(0.05) == "0.1" and spill_text(964.23) == "964.2"


def test_command_shows_file_error_without_traceback(tmp_path, capsys, monkeypatch):
    """ファイルを読み書きできないとき（出力先を Excel で開いているなど）は、理由の文を表示して終了コード1にする。"""
    from amamizu import cli

    def locked(argv=None):
        raise PermissionError(13, "Permission denied", "out/route.csv")
    monkeypatch.setattr(cli, "main", locked)
    with pytest.raises(SystemExit) as e:
        cli.run(["study", str(tmp_path)])
    err = capsys.readouterr().err
    assert e.value.code == 1 and "ファイルを読み書きできません" in err and "route.csv" in err and "Traceback" not in err

@pytest.mark.parametrize("bottom", [-10.0, 0.0, 10.0])
def test_table_volume_scalar_matches_array_at_boundaries(bottom):
    """水位1個でも配列でも、表の端・段の前後・表外の延長で同じ容量になる。"""
    pond = TablePond(np.array([0.0, 1.0, 2.0]) + bottom, None, [0.0, 100.0, 350.0])
    levels = np.array([bottom - 1, bottom, np.nextafter(bottom + 1, -np.inf), bottom + 1,
                       np.nextafter(bottom + 1, np.inf), bottom + 2, bottom + 3])
    expected = pond.volume(levels)
    for level, volume in zip(levels, expected, strict=True):
        for value in (float(level), level, np.asarray(level)):
            got = pond.volume(value)
            assert isinstance(got, float)
            assert got == volume
    assert expected[0] == 0 and expected[-1] == 600
