"""計算が手引と一致するかを確かめる。"""
from dataclasses import dataclass

from ..domain.design import design_for_area
from .design_study import Catchment, build_infiltration, infiltrated, planned_inflow
from .ports import Guideline, HayamiCell, InfiltrationCell


MATCH_REL = 0.007     # 早見表・浸透ありの容量の表と計算が一致したとみなす相対差（±0.7%）


def formula_mismatches(guideline: Guideline) -> list[tuple]:
    """転記した強度式が手引の表の値を再現しない組（表の値の小数の桁数に丸めて比べる）。空なら一致。"""
    table = guideline.formula_table()
    digits = guideline.formula_table_conditions()["digits"]
    bad = []
    for d, T, t, v in table:
        calc = round(guideline.formula(d, T).intensity(t), digits)
        if calc != v:
            bad.append((d, T, t, calc, v))
    return bad


@dataclass(frozen=True)
class HayamiResult:
    cell: HayamiCell
    calc: float         # m3/ha

    @property
    def rel_err(self) -> float:
        return self.calc / self.cell.volume - 1


def reproduce_hayami(guideline: Guideline, cells: list[HayamiCell] | None = None) -> list[HayamiResult]:
    """早見表の条件（確率年、池の面積を開発面積の割合で固定、1ha）で必要洪水調節容量を計算する。早見表がなければ空。"""
    cells = cells if cells is not None else guideline.hayami()
    if not cells:
        return []
    c0, hy = guideline.hayami_conditions(), guideline.hydraulics()
    ratio, tc = c0["pond_area_ratio"], c0["tc_min"]
    out = []
    for c in cells:
        inflow = planned_inflow(guideline.formula(c.district, c0["return_period"]), [Catchment("開発区域", 1.0, c.f)], tc, "rear").nodes
        fac = design_for_area(inflow, c.qc, ratio * 1e4, dt=tc * 60, hy=hy)
        out.append(HayamiResult(c, fac.capacity))
    return out


@dataclass(frozen=True)
class InfiltrationResult:
    cell: InfiltrationCell
    calc: float         # m3/ha

    @property
    def rel_err(self) -> float:
        return self.calc / self.cell.volume - 1


def reproduce_infiltration_table(guideline: Guideline,
                                 cells: list[InfiltrationCell] | None = None) -> list[InfiltrationResult]:
    """浸透施設を設けたときの容量の表（手引 表10－2）を、表の条件で計算する。表がなければ空。
    開発区域を浸透処理区域（開発面積 × 浸透処理面積率）と無処理区域に分け、設計の厳密計算と同じ流入の作り方と
    浸透の書き方の解釈（build_infiltration の intensity_mmh）、浸透の差し引き（planned_inflow、infiltrated）で、
    浸透処理区域の流入だけから設計浸透量（浸透強度 × 浸透処理面積 × 10 m3/h）を差し引く。"""
    cells = cells if cells is not None else guideline.infiltration_table()
    if not cells:
        return []
    c0, hy = guideline.infiltration_table_conditions(), guideline.hydraulics()
    f, area, tc = c0["f"], c0["area_ha"], c0["tc_min"]
    out = []
    for c in cells:
        treated_ha = area * c.treated_pct / 100
        zones = [Catchment("浸透処理区域", treated_ha, f, build_infiltration({"intensity_mmh": c.intensity_mmh}, treated_ha))
                 ] if treated_ha > 0 else []
        zones += [Catchment("無処理区域", area - treated_ha, f)] if treated_ha < area else []
        inflow = planned_inflow(guideline.formula(c.district, c0["return_period"]), zones, tc, "rear")
        nodes, _ = infiltrated(inflow, None, tc * 60)
        fac = design_for_area(nodes, c.qc * area, c0["pond_area_ratio"] * area * 1e4, dt=tc * 60, hy=hy)
        out.append(InfiltrationResult(c, fac.capacity / area))
    return out
