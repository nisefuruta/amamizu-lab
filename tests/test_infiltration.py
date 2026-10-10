"""浸透（一定量差し引きモデル）と比浸透量の式。

期待値は、式を試験の中で書き下した値と、参照値。
"""
import json
import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from amamizu.app.design_runner import routing_table_lines, trial
from amamizu.cli import main
from amamizu.domain.infiltration import Facility, design_rate_m3s, specific_infiltration, subtract_constant
from amamizu.usecases.design_study import build_infiltration


@pytest.mark.parametrize("kind, dims, expected", [
    ("pavement", {"H": 0.5}, 0.014 * 0.5 + 1.287),
    ("trench", {"H": 1.0, "W": 0.6}, 3.093 * 1.0 + 1.34 * 0.6 + 0.677),
    ("circle_side_bottom", {"H": 1.2, "D": 0.5}, (0.475 * 0.5 + 0.945) * 1.44 + (6.07 * 0.5 + 1.01) * 1.2 + 2.57 * 0.5 - 0.188),
    ("circle_side_bottom", {"H": 1.0, "D": 2.0}, (6.244 * 2 + 2.853) * 1.0 + 0.93 * 4 + 1.606 * 2 - 0.773),
    ("circle_bottom", {"H": 2.0, "D": 0.6}, (1.497 * 0.6 - 0.1) * 2.0 + 1.13 * 0.36 + 0.638 * 0.6 - 0.011),
    ("square_side_bottom", {"H": 1.0, "W": 5.0}, (-0.453 * 25 + 8.289 * 5 + 0.753) + 1.458 * 25 + 1.27 * 5 + 0.362),
    ("square_bottom", {"H": 1.0, "W": 20.0}, (1.265 * 20 - 15.67) + 1.259 * 400 + 2.336 * 20 - 8.13),
    ("rect_side_bottom", {"H": 1.0, "W": 4.0, "L": 10.0}, (3.297 * 10 + 1.971 * 4 + 4.663) + (1.401 * 4 + 0.684) * 10 + 1.214 * 4 - 0.834),
    ("rect_side_bottom", {"H": 1.0, "W": 2.0, "L": 10.0}, (3.297 * 10 + 1.971 * 2 + 4.663) + (1.401 * 2 + 0.684) * 10 + 1.214 * 2 - 0.834),
    ("large_side_bottom", {"H": 2.0, "W": 10.0, "L": 20.0}, (7.88 * 2.0 ** -0.446 * 2.0 + 14.0) * 20.0),
])
def test_specific_infiltration_formulas(kind, dims, expected):
    assert specific_infiltration(kind, **dims) == pytest.approx(expected, rel=1e-12)


def test_specific_infiltration_above_1_5m_uses_pressure_index():
    """設計水頭が1.5 m を超える円筒ます（側面および底面）は、H＝1.0 と 1.5 の K を静水圧指標で比例配分する。"""
    D, H = 0.5, 3.0
    k = lambda h: (0.475 * D + 0.945) * h ** 2 + (6.07 * D + 1.01) * h + 2.57 * D - 0.188
    p = lambda h: 0.5 * h ** 2 * math.pi * D + h * (D / 2) ** 2 * math.pi
    expected = k(1.0) + (k(1.5) - k(1.0)) * (p(H) - p(1.0)) / (p(1.5) - p(1.0))
    assert specific_infiltration("circle_side_bottom", H=H, D=D) == pytest.approx(expected, rel=1e-12)
    k = lambda h: 1.105 * h ** 2 + 8.657 * h + 2.575                           # W＝1 ちょうども、W≦1 の式で計算する
    p = lambda h: 2 * h ** 2 + h
    expected = k(1.0) + (k(1.5) - k(1.0)) * (p(2.0) - p(1.0)) / (p(1.5) - p(1.0))
    assert specific_infiltration("square_side_bottom", H=2.0, W=1.0) == pytest.approx(expected, rel=1e-12)


def test_large_tank_interpolates_between_widths():
    W, L, H = 15.0, 30.0, 2.0
    x = L / W
    k10 = (7.88 * x ** -0.446 * H + 14.0) * L
    k20 = (7.06 * x ** -0.452 * H + 27.06) * L
    assert specific_infiltration("large_side_bottom", H=H, W=W, L=L) == pytest.approx((k10 + k20) / 2, rel=1e-12)


@pytest.mark.parametrize("kind, dims, msg", [
    ("rect_side_bottom", {"H": 1.209, "W": 7.26, "L": 17.34}, "適用範囲外"),
    ("rect_side_bottom", {"H": 1.0, "W": 4.5, "L": 10.0}, "適用範囲外"),
    ("pavement", {"H": 2.0}, "適用範囲外"),
    ("trench", {"H": 1.0}, "W が必要"),
    ("pavement", {"H": 1.0, "W": 1.0}, "W を使わない"),
    ("hoge", {"H": 1.0}, "扱えません"),
])
def test_specific_infiltration_errors(kind, dims, msg):
    with pytest.raises(ValueError, match=msg):
        specific_infiltration(kind, **dims)


def test_facility_rate_matches_reference():
    """透水性舗装の参照値: 比浸透量 1.294 × 0.022 m/h × 2035 × 0.9 × 0.9 × 0.1 ＝ 4.69252278 m^3/h。"""
    f = Facility("透水性舗装", specific_infiltration("pavement", H=0.5), 0.022, 2035, 0.9, 0.9, 0.1)
    assert f.specific_m2 == pytest.approx(1.294)
    assert f.rate_m3_h == pytest.approx(4.69252278, rel=1e-9)


def test_subtract_constant():
    q = [0.0, 0.01, 0.03, 0.05, 0.02]
    out = subtract_constant(q, 0.015, 10.0, 600)
    # 0.03: 超えた 0.015×600＝9 m^3 は空隙へ。0.05: 超えた 21 m^3 のうち 1 m^3 で満ち、残り 20 m^3 を流す
    assert out == pytest.approx([0.0, 0.0, 0.0, 0.035 - 1 / 600, 0.005])
    exact = subtract_constant([0.0, 0.03, 0.05], 0.015, 9.0, 600)     # ちょうど満ちるとき、水量を消さない
    assert exact == pytest.approx([0.0, 0.0, 0.035])
    assert list(subtract_constant(q, 0.0, 0.0, 600)) == q


def test_build_infiltration():
    it = build_infiltration({"facilities": [
        {"name": "浸透ます", "shape": {"kind": "circle_side_bottom", "H_m": 1.0, "D_m": 0.5},
         "permeability": {"value": 1e-5, "unit": "m/s"}, "count": 10,
         "factors": {"clogging": 0.9, "groundwater": 0.9, "safety": 1.0},
         "void_parts": [{"volume_m3": 0.2, "porosity_pct": 100}, {"volume_m3": 1.0, "porosity_pct": 40}]}]})
    k = specific_infiltration("circle_side_bottom", H=1.0, D=0.5)
    assert it.rate_m3s == pytest.approx(k * 0.036 * 10 * 0.81 / 3600)
    assert it.void_m3 == pytest.approx((0.2 + 0.4) * 10)
    assert build_infiltration({"rate_m3s": 0.01}).void_m3 == 0.0
    assert build_infiltration(None) is None
    with pytest.raises(ValueError, match="どれか一つ"):
        build_infiltration({"rate_m3s": 0.01, "facilities": []})
    with pytest.raises(ValueError, match="どれか一つ"):
        build_infiltration({"rate_m3s": 0.01, "intensity_mmh": 10})
    with pytest.raises(ValueError, match="単位"):
        build_infiltration({"facilities": [{"specific_m2": 1.0, "permeability": {"value": 1, "unit": "mm/h"},
                                            "count": 1, "factors": {"clogging": 1, "groundwater": 1, "safety": 1}}]})
    with pytest.raises(ValueError, match="影響係数（安全率 α）がありません"):
        build_infiltration({"facilities": [{"specific_m2": 1.0, "permeability": {"value": 1, "unit": "m/h"},
                                            "count": 1, "factors": {"clogging": 0.9, "groundwater": 0.9}}]})
    with pytest.raises(ValueError, match="影響係数（地下水位）は 0 より大きく 1 以下"):
        build_infiltration({"facilities": [{"specific_m2": 1.0, "permeability": {"value": 1, "unit": "m/h"},
                                            "count": 1, "factors": {"clogging": 0.9, "groundwater": 1.2, "safety": 1}}]})


def test_intensity_to_rate():
    """設計浸透強度 mm/h × 浸透処理面積 ha × 10 が設計浸透量 m3/h（手引 第10条(5) の逆）。面積はどの書き方でも区域の面積。"""
    assert design_rate_m3s(13.05, 4.329) * 3600 == pytest.approx(13.05 * 4.329 * 10)
    it = build_infiltration({"intensity_mmh": 15, "void_storage_m3": 5}, 2.0)
    assert it.rate_m3s == design_rate_m3s(15, 2.0) and it.intensity_mmh == 15 and it.void_m3 == 5
    with pytest.raises(ValueError, match="区域の面積"):
        build_infiltration({"intensity_mmh": 15})           # ハイドログラフ（区域なし）では面積がない
    with pytest.raises(ValueError, match="正の値"):
        design_rate_m3s(10, 0.0)


@pytest.mark.needs_japanese_font
def test_design_with_and_without_infiltration(tmp_path):
    """浸透を書かなければ今までどおり。書けば流入量から差し引いてから追跡する。"""
    base = {"kind": "design", "guideline": "chiba_2006", "title": "浸透の確認",
            "rain": {"formula": {"type": "talbot", "a": 1914, "b": 12.3}, "shape": "rear"},
            "catchments": [{"name": "区域", "area_ha": 1.0, "f": 0.9}],
            "pond": {"prismatic": {"area_m2": 800.0, "hwl_m": 2.0}},
            "orifice": {"sill_m": 0.0, "width_m": 0.1, "height_m": 0.1}}
    maxes = {}
    for name, extra in (("none", {}), ("with", {"infiltration": {"rate_m3s": 0.05, "void_storage_m3": 20}})):
        d = tmp_path / name
        d.mkdir()
        (d / "study.json").write_text(json.dumps(base | extra, ensure_ascii=False), encoding="utf-8")
        main(["study", str(d)])
        head = (d / "out" / "route.csv").read_text(encoding="utf-8-sig").splitlines()[0]
        maxes[name] = (d / "out" / "design_summary.md").read_text(encoding="utf-8")
        assert ("浸透後流入量" in head) == (name == "with")
    assert "浸透施設（区域全体）" in maxes["with"] and "浸透施設（区域全体）" not in maxes["none"]


@pytest.mark.needs_japanese_font
def test_design_with_zone_infiltration(tmp_path):
    """区域ごとの浸透は、その区域の流入だけから差し引く。同じ設計浸透量を最上位（全体）で差し引くより最大貯留量が大きい
    （ある区域の施設の余りで、ほかの区域の水を浸透させない）。区域が1つなら、区域ごとに書いても最上位に書いても同じ値。"""
    base = {"kind": "design", "guideline": "chiba_2006", "title": "区域ごとの浸透の確認",
            "rain": {"formula": {"type": "talbot", "a": 1914, "b": 12.3}, "shape": "rear"},
            "pond": {"prismatic": {"area_m2": 800.0, "hwl_m": 3.0}},
            "orifice": {"sill_m": 0.0, "width_m": 0.1, "height_m": 0.1}}
    roof, green = {"name": "屋根", "area_ha": 0.3, "f": 1.0}, {"name": "緑地", "area_ha": 0.7, "f": 0.6}
    cases = {"whole": {"catchments": [roof, green], "infiltration": {"rate_m3s": 0.05}},
             "zones": {"catchments": [roof | {"infiltration": {"rate_m3s": 0.05}}, green]},
             "one_whole": {"catchments": [roof], "infiltration": {"intensity_mmh": 20}},
             "one_zone": {"catchments": [roof | {"infiltration": {"intensity_mmh": 20}}]}}
    out = {}
    for name, extra in cases.items():
        d = tmp_path / name
        d.mkdir()
        (d / "study.json").write_text(json.dumps(base | extra, ensure_ascii=False), encoding="utf-8")
        main(["study", str(d)])
        route = (d / "out" / "route.csv").read_text(encoding="utf-8-sig")
        out[name] = (route, (d / "out" / "design_summary.md").read_text(encoding="utf-8"))
    head = out["zones"][0].splitlines()[0]
    assert "屋根の流入量(m3/s)" in head and "屋根の浸透後流入量(m3/s)" in head and "緑地の" not in head
    assert "| 屋根の浸透施設（0.3 ha） | 設計浸透量 0.050000 m^3/s" in out["zones"][1]
    assert "| 屋根の浸透施設（0.3 ha） | 設計浸透強度 20 mm/h |" in out["one_zone"][1] and "| 浸透処理面積率 | 30.0% |" in out["zones"][1]
    vmax = {k: float(next(x for x in v[1].splitlines() if x.startswith("| 最大貯留量")).split()[3]) for k, v in out.items()}
    assert vmax["zones"] > vmax["whole"]
    one = {k: out[k][0].splitlines() for k in ("one_whole", "one_zone")}
    zone_col, whole_col = (one[k][0].split(",").index("浸透後流入量(m3/s)") for k in ("one_zone", "one_whole"))
    assert [r.split(",")[zone_col] for r in one["one_zone"][1:]] == [r.split(",")[whole_col] for r in one["one_whole"][1:]]
    assert vmax["one_whole"] == vmax["one_zone"]


@pytest.mark.needs_japanese_font
def test_zone_named_whole_is_treated_as_zone(tmp_path):
    """区域の名前が「全体」でも、区域ごとに書いた浸透は区域ごとの浸透として扱う（最上位の浸透と取り違えない）。"""
    d = tmp_path / "zones"
    d.mkdir()
    conf = {"kind": "design", "guideline": "chiba_2006", "title": "区域の名前が全体",
            "rain": {"formula": {"type": "talbot", "a": 1914, "b": 12.3}, "shape": "rear"},
            "catchments": [{"name": "全体", "area_ha": 0.3, "f": 1.0, "infiltration": {"rate_m3s": 0.01}},
                           {"name": "B", "area_ha": 0.7, "f": 0.6, "infiltration": {"rate_m3s": 0.02}}],
            "pond": {"prismatic": {"area_m2": 800.0, "hwl_m": 3.0}},
            "orifice": {"sill_m": 0.0, "width_m": 0.1, "height_m": 0.1}}
    (d / "study.json").write_text(json.dumps(conf, ensure_ascii=False), encoding="utf-8")
    main(["study", str(d)])
    head = (d / "out" / "route.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    summary = (d / "out" / "design_summary.md").read_text(encoding="utf-8")
    assert "全体の流入量(m3/s)" in head and "全体の浸透後流入量(m3/s)" in head and "Bの浸透後流入量(m3/s)" in head
    assert "| 全体の浸透施設（0.3 ha） |" in summary and "| Bの浸透施設（0.7 ha） |" in summary
    assert "| 浸透処理面積率 | 100.0% |" in summary and "浸透施設（区域全体）" not in summary


@pytest.mark.needs_japanese_font
def test_inflow_csv_round_trip(tmp_path):
    """out/inflow.csv（流出量の計算結果）は、ハイドログラフの入力（inflow_csv）にそのまま使える。区域ごとの浸透を差し引いた
    流入量を読み込み、同じ池とオリフィスで追跡すると、元の検討と同じ放流量と水位になる（表示の桁の範囲）。"""
    base = {"kind": "design", "guideline": "chiba_2006", "pond": {"prismatic": {"area_m2": 800.0, "hwl_m": 3.0}},
            "orifice": {"sill_m": 0.0, "width_m": 0.1, "height_m": 0.1}}
    first = tmp_path / "zones"
    first.mkdir()
    (first / "study.json").write_text(json.dumps(base | {
        "title": "区域ごとの浸透", "rain": {"formula": {"type": "talbot", "a": 1914, "b": 12.3}, "shape": "rear"},
        "catchments": [{"name": "屋根", "area_ha": 0.3, "f": 1.0, "infiltration": {"intensity_mmh": 15}},
                       {"name": "緑地", "area_ha": 0.7, "f": 0.6}]}, ensure_ascii=False), encoding="utf-8")
    main(["study", str(first)])
    head = (first / "out" / "inflow.csv").read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert head == ["時刻_分", "降雨強度_mmh", "屋根の流入量_m3s", "屋根の浸透後流入量_m3s", "緑地の流入量_m3s",
                    "浸透前流入量_m3s", "流入量_m3s"]
    second = tmp_path / "hydrograph"
    (second / "input").mkdir(parents=True)
    (second / "input" / "inflow.csv").write_bytes((first / "out" / "inflow.csv").read_bytes())
    (second / "study.json").write_text(json.dumps(base | {
        "title": "読み込んだ流入", "inflow_csv": {"file": "input/inflow.csv", "rain_col": "降雨強度_mmh"}},
        ensure_ascii=False), encoding="utf-8")
    main(["study", str(second)])
    a = pd.read_csv(first / "out" / "route.csv", encoding="utf-8-sig")
    b = pd.read_csv(second / "out" / "route.csv", encoding="utf-8-sig")
    assert len(a) == len(b)
    assert (a["浸透後流入量(m3/s)"] - b["流入量(m3/s)"]).abs().max() <= 1e-4
    for col in ("放流量(m3/s)", "水位(m)", "降雨強度(mm/h)"):
        assert (a[col] - b[col]).abs().max() <= 1e-4, col


@pytest.mark.needs_japanese_font
def test_outputs_align_time_charts_and_list_routing(tmp_path):
    """PDF 用の時間の図（降雨と流入・放流、水位）は幅と描画の領域の左右がそろう。清書した PDF には時刻ごとの追跡の結果の表
    （route.csv の全行。列ごとに桁をそろえる）を載せる（要約の design_summary.md には入れない）。"""
    conf = {"kind": "design", "guideline": "chiba_2006", "title": "図の確認", "rain": {"formula": {"type": "talbot", "a": 1914, "b": 12.3}, "shape": "rear"},
            "catchments": [{"name": "区域", "area_ha": 1.0, "f": 0.9}],
            "pond": {"prismatic": {"area_m2": 800.0, "hwl_m": 3.0}}, "orifice": {"sill_m": 0.0, "width_m": 0.1, "height_m": 0.1}}
    (tmp_path / "study.json").write_text(json.dumps(conf, ensure_ascii=False), encoding="utf-8")
    main(["study", str(tmp_path)])
    frames = []
    for name in ("hydrograph.png", "water_level.png"):
        im = np.asarray(Image.open(tmp_path / "out" / name).convert("L"))
        cols = np.where((im < 80).sum(axis=0) > im.shape[0] * 0.25)[0]     # 描画の領域の枠の縦線
        frames.append((im.shape[1], cols.min(), cols.max()))
    assert frames[0] == frames[1]
    res = trial(conf, tmp_path).result
    lines = routing_table_lines(res)
    assert len(lines) == 4 + len(res.table) and lines[4].startswith("| 0 | 0.000 |")
    assert "追跡計算の結果" not in (tmp_path / "out" / "design_summary.md").read_text(encoding="utf-8")


def test_routing_table_decimals_without_exponent():
    """PDF の追跡の結果の表は、小さい値（1e-05）でも指数表記にせず、列ごとに桁をそろえる。NaN があっても止まらない。"""
    from types import SimpleNamespace
    tab = pd.DataFrame({"計算時間(min)": [0, 10, 20], "放流量(m3/s)": [0.0, 1e-05, float("nan")], "水位(m)": [0.0, 0.25, 1.5]})
    lines = routing_table_lines(SimpleNamespace(table=tab))
    assert lines[4:] == ["| 0 | 0.00000 | 0.00 |", "| 10 | 0.00001 | 0.25 |", "| 20 | nan | 1.50 |"]
