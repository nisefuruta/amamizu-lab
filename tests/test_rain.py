"""確率降雨強度式の5方式と計画降雨の波形（後方集中型、中央集中型）。

期待値は、a＝1914、b＝12.3、n＝3/4 で作った参照値。
"""
import json

import numpy as np
import pytest

from amamizu.cli import main
from amamizu.domain.design_storm import center_loaded, planned_storm, ranked_blocks, rear_loaded
from amamizu.domain.intensity import formula_of

A, B, N = 1914, 12.3, 0.75


def test_center_loaded_reproduces_reference():
    """クリーブランド式・中央集中型の24時間（10分ごと）が、参照値と一致する。"""
    s = center_loaded(formula_of("cleveland", A, B, N))
    expected = {0: 2.255214748044409, 36: 4.002984222026498, 71: 106.7876956860731,
                72: 69.15231410919904, 108: 3.8678367183150613}
    for i, v in expected.items():
        assert s[i] == pytest.approx(v, rel=1e-12), i
    assert int(np.argmax(s)) == 71                          # 11:50〜12:00


@pytest.mark.parametrize("kind, b, n, first, second", [
    ("talbot", B, None, 85.82959641255606, 32.684335475988846),
    ("sherman", None, N, 340.3626790814498, 64.39904056359813),
    ("kuno_ishiguro", B, None, 123.78512674950612, 104.45056191118097),
    ("cleveland", B, N, 106.7876956860731, 69.15231410919904),
    ("kuno_ishiguro_n", B, N, 106.7876956860731, 69.15231410919904),
])
def test_five_formula_types_match_reference(kind, b, n, first, second):
    """5方式の1番目・2番目の区間の強度が、参照値と一致する。"""
    blocks = ranked_blocks(formula_of(kind, A, b, n))
    assert blocks[0] == pytest.approx(first, rel=1e-12)
    assert blocks[1] == pytest.approx(second, rel=1e-12)


@pytest.mark.parametrize("kind, kw, msg", [
    ("talbot", {"a": 1, "b": 1, "n": 0.5}, "n を書かないで"),
    ("sherman", {"a": 1, "b": 1, "n": 0.5}, "b を書かないで"),
    ("kuno_ishiguro", {"a": 1, "b": 1, "n": 0.5}, "n を書かないで"),
    ("cleveland", {"a": 1, "b": 1}, "n が必要"),
    ("sherman", {"a": 1}, "n が必要"),
    ("hoge", {"a": 1}, "扱えません"),
])
def test_formula_items_are_checked(kind, kw, msg):
    """その方式で使わない係数を書いたら、黙って無視せずエラーにする。"""
    with pytest.raises(ValueError, match=msg):
        formula_of(kind, **kw)


def test_formula_with_nonpositive_denominator_is_an_error():
    """b が負で分母 t^n＋b が0以下になる継続時間では、負や無限大の強度を返さずエラーにする。分母が正なら計算する。"""
    f = formula_of("talbot", A, -15.0)
    assert f.intensity(20) == pytest.approx(A / 5)
    for t in (10, 15):
        with pytest.raises(ValueError, match="分母"):
            f.intensity(t)
    with pytest.raises(ValueError, match="分母"):
        planned_storm(f, "rear")


@pytest.mark.parametrize("hours, tc", [(24, 10), (3, 20), (24, 60)])
def test_center_and_rear_use_the_same_blocks(hours, tc):
    """中央集中型は後方集中型と同じ区間を並べ替えたもの。1番目を中央、2番目をその後ろ、3番目をその前に置く。"""
    f = formula_of("cleveland", A, B, N)
    c, r = center_loaded(f, hours, tc), rear_loaded(f, hours, tc)
    assert sorted(c) == pytest.approx(sorted(r))
    ranked = ranked_blocks(f, hours, tc)
    mid = (len(c) - 1) // 2
    assert c[mid] == ranked[0] and c[mid + 1] == ranked[1] and c[mid - 1] == ranked[2]


def test_unknown_shape_is_an_error():
    with pytest.raises(ValueError, match="降雨波形"):
        planned_storm(formula_of("talbot", A, B), "front")


@pytest.mark.parametrize("hours", [0, -2])
def test_nonpositive_hours_is_an_error(hours):
    """継続時間が0以下の計画降雨は、区間のない（流入0の）降雨にせずエラーにする（試算が流入0で O.K になるため）。"""
    with pytest.raises(ValueError, match="継続時間"):
        planned_storm(formula_of("talbot", A, B), "rear", hours)


@pytest.mark.needs_japanese_font
def test_rain_command(tmp_path, capsys):
    out = tmp_path / "rain.csv"
    main(["rain", "--type", "cleveland", "--a", "1914", "--b", "12.3", "--n", "3/4", "--shape", "center",
          "--output", str(out)])
    rows = out.read_text(encoding="utf-8-sig").splitlines()
    assert rows[0] == "区間,時,分,降雨強度(mm/h),雨量(mm)"
    assert rows[72].startswith("11:50〜12:00,11,50〜60,106.787695686")
    assert out.with_suffix(".png").read_bytes()[:4] == b"\x89PNG"
    assert "中央集中型24時間" in capsys.readouterr().out


@pytest.mark.needs_japanese_font
def test_rain_command_with_guideline(tmp_path, capsys):
    out = tmp_path / "rain.csv"
    main(["rain", "--district", "千葉", "--guideline", "chiba_2006", "--shape", "rear", "--tc", "20", "--output", str(out)])
    assert "手引 千葉地区 1/50" in capsys.readouterr().out
    assert len(out.read_text(encoding="utf-8-sig").splitlines()) == 1 + 72
    with pytest.raises(ValueError, match=r"式の種類と係数（type"):
        main(["rain", "--district", "千葉", "--type", "talbot", "--shape", "rear", "--output", str(out)])
    with pytest.raises(ValueError, match=r"式の種類と係数（b、n"):       # 指定した係数を黙って使わずに手引の式で作らない
        main(["rain", "--district", "千葉", "--guideline", "chiba_2006", "--b", "99", "--n", "0.9", "--shape", "rear",
              "--output", str(out)])
    with pytest.raises(ValueError, match=r"手引の式の条件（return_period、guideline"):
        main(["rain", "--type", "talbot", "--a", "5000", "--b", "30", "--return-period", "10", "--guideline", "chiba_2006",
              "--shape", "rear", "--output", str(out)])


@pytest.mark.needs_japanese_font
def test_design_with_center_loaded_rain(tmp_path):
    (tmp_path / "study.json").write_text(json.dumps({
        "kind": "design", "guideline": "chiba_2006", "title": "中央集中型の確認",
        "rain": {"formula": {"type": "talbot", "a": A, "b": B}, "shape": "center"},
        "catchments": [{"name": "区域", "area_ha": 1.0, "f": 0.9}],
        "pond": {"prismatic": {"area_m2": 800.0, "hwl_m": 2.0}},
        "orifice": {"sill_m": 0.0, "width_m": 0.1, "height_m": 0.1}}, ensure_ascii=False), encoding="utf-8")
    main(["study", str(tmp_path)])
    summary = (tmp_path / "out" / "design_summary.md").read_text(encoding="utf-8")
    assert "| 計画降雨 | 中央集中型24時間 |" in summary and "| 降雨強度式 | タルボット式 r＝1914/(t+12.3) |" in summary
    assert (tmp_path / "out" / "rain.csv").exists()


def test_design_needs_shape(tmp_path):
    (tmp_path / "study.json").write_text(json.dumps({
        "kind": "design", "guideline": "chiba_2006", "rain": {"formula": {"type": "talbot", "a": A, "b": B}},
        "catchments": [{"name": "区域", "area_ha": 1.0, "f": 0.9}],
        "pond": {"prismatic": {"area_m2": 800.0, "hwl_m": 2.0}},
        "orifice": {"sill_m": 0.0, "width_m": 0.1, "height_m": 0.1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="rain.shape"):
        main(["study", str(tmp_path)])


# 降雨強度の丸め --------------------------------------------------------------
def test_round_half_up_like_excel():
    """四捨五入は Excel の ROUND と同じ（numpy の偶数への丸めではない）。"""
    from amamizu.domain.design_storm import round_half_up
    assert list(round_half_up([2.5, 3.5, 2.675, 0.125, 106.7876956860731], 0)) == [3.0, 4.0, 3.0, 0.0, 107.0]
    assert list(round_half_up([2.675, 0.125, 1.005], 2)) == [2.68, 0.13, 1.01]
    for bad in (-1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="桁数"):
            round_half_up([1.0], bad)


def test_rounding_is_optional():
    """丸めを省略したら今までどおり（丸めない）。指定したら四捨五入した値になる。"""
    f = formula_of("cleveland", A, B, N)
    raw = planned_storm(f, "center")
    assert np.array_equal(planned_storm(f, "center", round_digits=None), raw)
    r1 = planned_storm(f, "center", round_digits=1)
    assert r1[71] == 106.8 and r1[72] == 69.2 and np.all(np.abs(r1 - raw) <= 0.05 + 1e-12)


@pytest.mark.needs_japanese_font
def test_rain_command_rounding(tmp_path, capsys):
    out = tmp_path / "rain.csv"
    main(["rain", "--type", "cleveland", "--a", "1914", "--b", "12.3", "--n", "3/4", "--shape", "center",
          "--round", "2", "--output", str(out)])
    rows = out.read_text(encoding="utf-8-sig").splitlines()
    assert rows[72].startswith("11:50〜12:00,11,50〜60,106.79,")
    assert "小数2桁に四捨五入" in capsys.readouterr().out


@pytest.mark.needs_japanese_font
def test_design_uses_rounded_rain(tmp_path):
    """設計の検討で丸めを指定したら、丸めた降雨強度で流入量を計算する。"""
    from amamizu.usecases.design_study import Catchment, planned_inflow
    f = formula_of("talbot", A, B)
    c = [Catchment("区域", 1.0, 0.9)]
    raw, rounded = planned_inflow(f, c, 10, "rear"), planned_inflow(f, c, 10, "rear", round_digits=1)
    assert np.allclose(rounded.nodes[1:-1], 0.9 * rounded.rain * 1.0 / 360)
    assert not np.array_equal(raw.nodes, rounded.nodes)
    (tmp_path / "study.json").write_text(json.dumps({
        "kind": "design", "guideline": "chiba_2006", "rain": {"formula": {"type": "talbot", "a": A, "b": B}, "shape": "rear", "round_digits": 1},
        "catchments": [{"name": "区域", "area_ha": 1.0, "f": 0.9}],
        "pond": {"prismatic": {"area_m2": 800.0, "hwl_m": 2.0}},
        "orifice": {"sill_m": 0.0, "width_m": 0.1, "height_m": 0.1}}, ensure_ascii=False), encoding="utf-8")
    main(["study", str(tmp_path)])
    assert "降雨強度を小数1桁に四捨五入" in (tmp_path / "out" / "design_summary.md").read_text(encoding="utf-8")


def test_saved_rain_does_not_take_names_of_design_outputs(tmp_path):
    """降雨のページの「out/ に保存」は、「保存して出力」の out/ のファイル（rain.csv など）と同じ名前で書かない
    （同じ名前だと、どちらかが黙って上書きされ、前の出力の記録から削除されることもある）。初期値の名前は重ならない。"""
    from amamizu.app.design_runner import output_names
    from amamizu.app.rain_runner import saved_rain_name, saved_rain_path
    for source in ("planned", "observed"):
        name = saved_rain_name(source)
        assert saved_rain_path(tmp_path, name) == tmp_path / "out" / f"{name}.csv"
        assert not {f"{name}{ext}" for ext in (".csv", ".png", ".json")} & set(output_names())
    for name in ("rain", "Rain", "route", "hydrograph", "conditions", " rain "):
        with pytest.raises(ValueError, match="保存して出力"):
            saved_rain_path(tmp_path, name)
    for name in ("", "  ", "a/b"):
        with pytest.raises(ValueError, match="使えない文字"):
            saved_rain_path(tmp_path, name)
