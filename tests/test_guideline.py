"""計算が手引（資料－1 の強度式、資料－2 の早見表、第10条 表10－2 の浸透ありの容量）と一致すること。"""
from pathlib import Path
import pytest

from amamizu.adapters.guideline.json_guideline import JsonGuideline
import numpy as np

from amamizu.domain.design import design_for_area
from amamizu.domain.design_storm import rear_loaded
from amamizu.domain.infiltration import subtract_constant
from amamizu.domain.runoff import rational_inflow
from amamizu.usecases.guideline_check import MATCH_REL, formula_mismatches, reproduce_hayami, reproduce_infiltration_table

CHIBA = Path(__file__).parents[1] / "studies" / "_guideline" / "chiba_2006.json"      # 千葉県の手引の定数


@pytest.fixture(scope="module")
def guideline():
    return JsonGuideline(CHIBA)


def test_formulas_reproduce_table(guideline):
    assert formula_mismatches(guideline) == []


@pytest.fixture(scope="module")
def hayami(guideline):
    return reproduce_hayami(guideline)


def test_hayami_all_cells(hayami):
    assert len(hayami) == 140
    for r in hayami:
        assert abs(r.rel_err) <= MATCH_REL, (r.cell, r.calc)


def test_hayami_main_rows_within_03pct(hayami):
    """流出率0.8・0.9、許容放流比流量が下限値の14件は ±0.3% 以内（本冊2章。最大は館山 f0.9 の -0.25%）。"""
    main = [r for r in hayami if r.cell.f in (0.8, 0.9) and r.cell.qc == (0.035 if r.cell.district in ("勝浦", "館山") else 0.025)]
    assert len(main) == 14
    assert max(abs(r.rel_err) for r in main) <= 0.003


def test_infiltration_table_all_cells(guideline):
    """表10－2（7地区 × 浸透なしと浸透強度 5・10・15 mm/h × 浸透処理面積率 20〜100%）の112マスが ±0.7% 以内（最大は -0.55%）。"""
    res = reproduce_infiltration_table(guideline)
    assert len(res) == 112
    worst = max(res, key=lambda r: abs(r.rel_err))
    assert abs(worst.rel_err) <= MATCH_REL, (worst.cell, worst.calc)


def test_subtracting_from_whole_inflow_underestimates(guideline):
    """設計浸透量を浸透処理区域ではなく全体の流入から差し引くと、浸透処理面積率が100%未満で容量を小さく見積もる
    （千葉 15 mm/h・20% で表 952 に対して約 870 m3/ha）。浸透処理区域の流入だけから差し引く理由。"""
    c = next(c for c in guideline.infiltration_table() if (c.district, c.intensity_mmh, c.treated_pct) == ("千葉", 15, 20))
    storm = rear_loaded(guideline.formula("千葉", 50), block_min=10)
    whole = subtract_constant(rational_inflow(storm, 0.8, 1.0), 15 * 0.2 * 10 / 3600, 0.0, 600)
    calc = design_for_area(whole, 0.025, 500.0, dt=600, hy=guideline.hydraulics()).capacity
    assert calc / c.volume - 1 < -0.05
    assert np.isclose(reproduce_infiltration_table(guideline, [c])[0].calc, c.volume, rtol=MATCH_REL)


def test_unknown_district_is_an_error(guideline):
    """地区名の書き間違いが、許容放流比流量の既定値で黙って通らない。"""
    with pytest.raises(KeyError, match="地区「千葉市」がありません"):
        guideline.qc_min("千葉市")


def test_return_period_study_only_handles_10min(guideline):
    """標本は年最大10分間降水量なので、他の継続時間の基準と比べない。"""
    from amamizu.usecases.return_period_study import evaluate
    with pytest.raises(ValueError, match="10分"):
        evaluate(guideline, None, duration_min=60)
