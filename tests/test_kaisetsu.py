"""手引の解説 資料－4 計算例－1（印刷104〜122ページ）を再現できること。

A開発地区 9.45 ha、千葉地区 1/50 後方集中型24時間、tc 10分、許容放流量 0.236 m3/s、G調整池（図－10 の水位容量表）、
正方形のオリフィス（敷高 0）。浸透施設併用は、浸透処理区域 4.329 ha（流出率 1.0、設計浸透強度 13.05 mm/h）と
無処理区域 5.121 ha（流出率 0.773）。調整池単独は 9.45 ha、流出率 0.877（本文の 0.88 は丸めた値。表－9 は 0.877 で計算してある）。
解説の値は tests/fixtures/kaisetsu_rei1/ の表－9（流出量）と表－11（貯留追跡）。

解説の表－11 の連続式は、流入を1行先の値と平均する（V(t) = V(t-10) + [(Qin(t)+Qin(t+10))/2 - (Qout(t-10)+Qout(t))/2]×600）。
流入が10分早く並ぶので、ここの計算の時刻 t+10 が解説の時刻 t にあたる（最大値の大きさは変わらない）。
解説の水深は、3 m を超えると図－10 の表と合わない水位容量曲線で求めてあるので、比べるのは容量と放流量。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amamizu.adapters.guideline.json_guideline import JsonGuideline
from amamizu.domain.pond import TablePond
from amamizu.usecases.design_study import Catchment, build_infiltration, planned_inflow, run_design
from amamizu.domain.outlet import Hydraulics

CHIBA = Path(__file__).parents[1] / "studies" / "_guideline" / "chiba_2006.json"      # 千葉県の手引の定数

HY = Hydraulics(9.8, 1.8, 0.6)      # 試験に使う放流の係数（重力加速度、越流係数、オリフィスの流量係数）

DATA = Path(__file__).parent / "fixtures" / "kaisetsu_rei1"
LEVELS = [0, 1, 2, 3, 4, 5, 6, 7]
AREAS = [1000, 1386, 1772, 2157, 2543, 2929, 3314, 3700]
VOLUMES = [0, 1193, 2772, 4736.5, 7086.5, 9822.5, 12944, 16451]
# 浸透処理区域は、区域の浸透を設計浸透強度 13.05 mm/h（study.json の intensity_mmh と同じ書き方）で与える
TREATED = Catchment("浸透処理区域", 4.329, 1.0, build_infiltration({"intensity_mmh": 13.05}, 4.329))
UNTREATED = Catchment("無処理区域", 5.121, 0.773)


@pytest.fixture(scope="module")
def formula():
    return JsonGuideline(CHIBA).formula("千葉", 50)


def with_infiltration(formula, orifice):
    inflow = planned_inflow(formula, [TREATED, UNTREATED], 10, "rear")
    return run_design(inflow, TablePond(LEVELS, AREAS, VOLUMES), [orifice], None, 2500, with_table=True, **_size(orifice), hy=HY)


def alone(formula, orifice):
    inflow = planned_inflow(formula, [Catchment("開発区域", 9.45, 0.877)], 10, "rear")
    return run_design(inflow, TablePond(LEVELS, AREAS, VOLUMES), [orifice], None, 2500, with_table=True, **_size(orifice), hy=HY)


def _size(orifice):
    """寸法のないオリフィスは、最大放流量が許容放流量 0.236 m3/s になる寸法を求める。"""
    return {"mode": "size_orifice", "target_q": 0.236} if "width_m" not in orifice else {"allowable_q": 0.236}


def test_runoff_matches_table9(formula):
    """表－9: 計画雨量、浸透なし、浸透処理区域（浸透後）、無処理区域、計の全144行（表は小数3桁、雨量は2桁）。"""
    t9 = pd.read_csv(DATA / "table9_runoff.csv")
    res = with_infiltration(formula, {"sill_m": 0.0, "width_m": 0.211, "height_m": 0.211})
    tab = res.table.set_index("計算時間(min)").loc[t9["時刻_分"]]
    none = planned_inflow(formula, [Catchment("開発区域", 9.45, 0.877)], 10, "rear").nodes[1:145]
    untreated = planned_inflow(formula, [UNTREATED], 10, "rear").nodes[1:145]
    treated_after = tab["浸透処理区域の浸透後流入量(m3/s)"].to_numpy()
    assert np.abs(tab["降雨強度(mm/h)"].to_numpy() - t9["計画雨量_mmh"]).max() <= 0.0051
    for calc, col in ((none, "浸透なし_m3s"), (treated_after, "浸透処理区域_m3s"), (untreated, "無処理区域_m3s"),
                      (tab["浸透後流入量(m3/s)"].to_numpy(), "計_m3s")):
        assert np.abs(calc - t9[col]).max() <= 0.0006, col


def test_routing_matches_table11(formula):
    """表－11: 浸透施設併用、オリフィス 0.211 m 角。ここの時刻 t+10 を解説の時刻 t と比べる。"""
    t11 = pd.read_csv(DATA / "table11_routing.csv")
    res = with_infiltration(formula, {"sill_m": 0.0, "width_m": 0.211, "height_m": 0.211})
    ours = res.table.set_index("計算時間(min)").loc[t11["時刻_分"] + 10]
    # 解説の表－11 は流入と放流で平均する区間が合っていないので、値は一致しない（docs/厳密計算.md の「10 手引との照合」）。
    # 全248行の最大の差は、放流量 0.0033 m^3/s、貯留量 39.7 m^3
    assert np.abs(ours["放流量(m3/s)"].to_numpy() - t11["放流量_m3s"]).max() <= 0.004
    assert np.abs(ours["貯留量(m3)"].to_numpy() - t11["池容量_m3"]).max() <= 40
    assert res.summary["最大貯留量_m3"] == pytest.approx(7612.5, rel=0.003)
    assert res.summary["最大流入量_m3s"] == pytest.approx(3.047, abs=0.0005)


@pytest.mark.parametrize("case, side, volume", [(with_infiltration, 0.211, 7613), (alone, 0.195, 11344)])
def test_orifice_size_matches_table10(formula, case, side, volume):
    """表－10: 許容放流量 0.236 m3/s を満たすオリフィスの口径（解説は数回の仮定で決めた 0.211・0.195 m 角）と洪水調節容量。"""
    res = case(formula, {"sill_m": 0.0, "shape": "rect"})
    assert res.orifices[0].width == pytest.approx(side, abs=0.002)
    assert res.summary["最大放流量_m3s"] == pytest.approx(0.236, rel=1e-4)
    assert res.summary["最大貯留量_m3"] == pytest.approx(volume, rel=0.003)
