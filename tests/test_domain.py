"""domain 層の単体試験。"""
import math

import numpy as np
import pandas as pd
import pytest

from amamizu.domain.capture import capture_rate
from amamizu.domain.design import design_for_depth
from amamizu.domain.design_storm import rear_loaded
from amamizu.domain.frequency import Gumbel
from amamizu.domain.intensity import IntensityFormula
from amamizu.domain.outlet import SquareOrifice
from amamizu.domain.rainfall import duration_depths, hourly_from_10min, max_sum, split_events
from amamizu.domain.runoff import rational_inflow
from amamizu.usecases.rainfall_tools import capture_table, hourly_statistics
from amamizu.domain.outlet import Hydraulics

HY = Hydraulics(9.8, 1.8, 0.6)      # 試験に使う放流の係数（重力加速度、越流係数、オリフィスの流量係数）

CHIBA_50 = IntensityFormula(2439, 0.75, 11.9)


def test_rear_loaded_keeps_depth_and_peak():
    h = rear_loaded(CHIBA_50)
    assert len(h) == 144
    assert h.sum() / 6 == pytest.approx(CHIBA_50.depth(1440))          # 24時間雨量 238.3mm
    assert h[-1] == pytest.approx(CHIBA_50.intensity(10))                # 最後の10分がピーク
    assert np.all(np.diff(h) >= 0)


def test_rational_inflow_volume_equals_runoff():
    rain = np.array([10.0, 20.0, 0.0])
    q = rational_inflow(rain, 0.8, 1.0)
    # 三角形の重ね合わせ（節点を直線で結ぶ）の体積 = f·R·A·10
    vol = np.trapezoid(q, dx=600)
    assert vol == pytest.approx(0.8 * rain.sum() / 6 * 1.0 * 10)


def test_orifice_sized_for_allowable_discharge():
    o = SquareOrifice.sized_for(2.0, 0.025, hy=HY)
    assert o.discharge(2.0) == pytest.approx(0.025, rel=1e-6)
    assert o.discharge(0) == 0


def test_orifice_that_cannot_reach_allowable_discharge_is_an_error():
    """オリフィスの高さの上限（HWL/1.8）でも許容放流量に達しないときは、小さいオリフィスのまま黙って返さない。"""
    with pytest.raises(ValueError, match="オリフィスが見つかりません"):
        SquareOrifice.sized_for(0.5, 1.0, hy=HY)


def test_depth_design_needs_allowable_discharge_or_orifice():
    """許容放流量もオリフィスも与えずに池の面積を求めるときは、Python の例外ではなくエラーの文で止める。"""
    inflow = rational_inflow(rear_loaded(CHIBA_50), 0.8, 1.0)
    with pytest.raises(ValueError, match="許容放流量（q_allow）と放流の係数（hy）か、放流オリフィス（orifice）"):
        design_for_depth(inflow, None, 2.0, hy=HY)


def test_pond_area_out_of_search_range_is_an_error():
    """必要な池の面積が探索の範囲（〜100,000 m2）を超えるときは、範囲の端の寸法を黙って返さない。"""
    inflow = rational_inflow(rear_loaded(CHIBA_50), 0.8, 500.0)
    with pytest.raises(ValueError, match="範囲"):
        design_for_depth(inflow, 0.025 * 500, 5.0, hy=HY)


def test_rainfall_helpers():
    assert max_sum([1, 5, 2, 7], 2) == 9
    with pytest.raises(ValueError, match="3 時間以上の記録"):     # 記録の外を0とみなした値を返さない
        max_sum([1, 5], 3)
    assert list(hourly_from_10min([1] * 12)) == [6, 6]
    ev = split_events([0, 3, 0, 4, 0, 0, 5, 1])
    assert [(e.start, e.end, e.total) for e in ev] == [(1, 3, 7.0), (6, 7, 6.0)]
    d = duration_depths([1, 2, 3], lambda h: 10.0 * h, hours=(1, 2), max1h_published=4.0)
    assert [(x.hours, x.observed, x.plan) for x in d] == [(1, 4.0, 10.0), (2, 5.0, 20.0)]


def test_split_events_missing_day_keeps_continuity():
    """欠測（NaN）の日は無降水に数えない。雨量は0で、欠測の日だけでは降雨を始めない。"""
    nan = float("nan")
    ev = split_events([1, 0, 2, nan, 0, 3, 0, 0, nan, 0, 0, 4])
    assert [(e.start, e.end, e.total) for e in ev] == [(0, 5, 6.0), (11, 11, 4.0)]


def test_gumbel_moments():
    g = Gumbel.by_moments([10, 12, 15, 11, 20])
    m, s = 13.6, math.sqrt(sum((v - 13.6) ** 2 for v in [10, 12, 15, 11, 20]) / 4)
    assert g.beta == pytest.approx(s * math.sqrt(6) / math.pi)
    assert g.mu == pytest.approx(m - 0.5772157 * g.beta)
    assert g.return_period(g.mu) == pytest.approx(1 / (1 - math.exp(-1)))


def test_capture_rate_matches_old_definition():
    """旧 capturer: 捕捉量 = 強度以下の雨量の和 + 強度を超えた時間数 × 強度。"""
    r = np.array([0, 2, 8, 15, 30, np.nan])
    old = (r[r <= 10].sum() + (r > 10).sum() * 10) / np.nansum(r) * 100
    assert capture_rate(r, 10)[0] == pytest.approx(old)
    t = capture_table(r, 5, 15, 5)
    assert list(t["設計浸透強度 (mm/hr)"]) == [5, 10, 15]


def test_hourly_statistics():
    df = pd.DataFrame({"date": ["2025-01-01"] * 3 + ["2025-02-01"], "hour": [1, 2, 24, 1], "rainfall_mm": [0, 12, 35, None]})
    s = hourly_statistics(df)
    assert s.basic["30mm/時以上の豪雨発生回数"] == 1 and s.basic["欠損値数"] == 1
    assert s.basic["最大時間降水量の発生日時"] == "2025-01-01 23:00"
    assert s.monthly.loc[1, "最大"] == 35


def test_prismatic_pond_rejects_non_positive():
    from amamizu.domain.pond import PrismaticPond
    for area, hwl in ((-1, 2.0), (0, 2.0), (600, 0), (600, -1)):
        with pytest.raises(ValueError, match="正の値"):
            PrismaticPond(area, hwl)
