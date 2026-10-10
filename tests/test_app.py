"""組み立ての層（app）と study.json の読み込み・確認（adapters/study_config）の試験。"""
import json
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

from amamizu.adapters.design_inputs import read_table
from amamizu.adapters.study_config import check_study, dump_study, missing_items, remove_keys, unknown_keys, value_errors
from amamizu.app.design_runner import ConfigError, check_sections, prepare, record_outputs, run_study, stale_outputs, trial

ROOT = Path(__file__).parents[1]
CASES = Path(__file__).parent / "fixtures" / "design_cases"


def conf_of(case: str) -> dict:
    return json.loads((CASES / case / "study.json").read_text(encoding="utf-8"))


# study.json の確認 ---------------------------------------------------------------
def test_storage_conditions_and_required_items():
    """貯留施設の検証・降雨強度の検証は、conditions の中の不明な項目と、必要な項目の不足を、項目名を示すエラーにする。"""
    storage = json.loads((ROOT / "studies" / "2026-09_貯留施設の検証" / "study.json").read_text(encoding="utf-8"))
    assert check_study(storage) is storage
    storage["conditions"]["depht"] = 2.0
    with pytest.raises(ValueError, match=r"不明な項目.*conditions\.depht"):
        check_study(storage)
    del storage["conditions"]["depht"], storage["partial_year"], storage["events"][1]["end"]
    assert missing_items(storage) == ["partial_year", "events[1].end"]
    with pytest.raises(ValueError, match=r"必要な項目がありません: partial_year, events\[1\]\.end"):
        check_study(storage)
    del storage["title"]
    assert missing_items(storage)[0] == "title"                  # 題名はレポートの冒頭に書く
    assert missing_items({"kind": "return_period", "title": "t"}) == [
        "standard_period", "duration_min", "last_year", "event"]


CHIBA = json.loads((ROOT / "studies" / "_guideline" / "chiba_2006.json").read_text(encoding="utf-8"))


def test_guideline_file_without_item_names_it(tmp_path):
    """手引の定数のファイルに項目がなければ、項目名と使い道を示すエラーにする。"""
    from amamizu.adapters.guideline.json_guideline import JsonGuideline
    from amamizu.usecases.guideline_check import formula_mismatches
    d = {k: v for k, v in CHIBA.items() if k != "formula_table"}
    p = tmp_path / "part.json"
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(KeyError, match="「formula_table」がありません。amamizu verify"):
        formula_mismatches(JsonGuideline(p))
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError, match="「name」"):
        JsonGuideline(p)


@pytest.mark.parametrize("change, message", [
    (lambda d: d["formulas"]["千葉"].__setitem__("50", [2439, "3/x", 11.9]), r"「formulas\.千葉\.50\[1\]」は数か分数の文字列"),
    (lambda d: d["formulas"]["千葉"].__setitem__("五十", d["formulas"]["千葉"].pop("50")), r"「formulas\.千葉\.五十」は項目名を確率年の整数"),
    (lambda d: d["formula_table"]["千葉"].__setitem__("50", [154.7, 63.0]),
     r"「formula_table\.千葉\.50」の値の数（2）を、formula_table_conditions\.t_min の数（3）"),
    (lambda d: d.pop("formula_table_conditions"), r"「formula_table_conditions」がありません"),
    (lambda d: d["hayami_conditions"].__setitem__("return_periods", 30),
     r"「hayami_conditions」はtc_min、pond_area_ratio、return_period だけで書いて"),
    (lambda d: d["hydraulics"].__setitem__("c_orfice", 0.6), r"hydraulics の c_orfice は使えない項目"),
    (lambda d: d["qc_min"].pop("default"), r"「qc_min\.default」がありません"),
    (lambda d: d["qc_min"].__setitem__("勝浦 ", d["qc_min"].pop("勝浦")), r"qc_min の項目名に、formulas にない地区「勝浦 」があります"),
    (lambda d: d["district_of_station"].__setitem__("千葉", "千葉市"), r"district_of_station の値に、formulas にない地区「千葉市」があります"),
    (lambda d: d.__setitem__("runoff", [1.0, 0.6, 0.9]), r"「runoff」は項目を持つ形"),
    (lambda d: d["infiltration_table_conditions"].__setitem__("f", "0.8"), r"「infiltration_table_conditions\.f」は数で書いて"),
    (lambda d: d["hayami"]["千葉"]["v"][0].pop(), r"「hayami\.千葉\.v\[0\]」の値の数（3）を、「hayami\.千葉\.qc」の値の数（4）"),
    (lambda d: d["hayami"]["千葉"]["v"].pop(), r"「hayami\.千葉\.v」の行の数（4）を、hayami_imp の値の数（5）"),
    (lambda d: d["infiltration_table"]["千葉"]["v"].pop(), r"「infiltration_table\.千葉\.v」の値の数（15）を、infiltration_table_rows の行の数（16）"),
])
def test_guideline_file_with_wrong_shape_names_the_place(tmp_path, change, message):
    """手引の定数の表の中身の形が違えば、読み込むときに場所を示すエラーにする（値の足りない行を黙って照合から外さない）。"""
    import copy

    from amamizu.adapters.guideline.json_guideline import JsonGuideline
    d = copy.deepcopy(CHIBA)
    change(d)
    p = tmp_path / "wrong.json"
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    with pytest.raises((KeyError, ValueError), match=message):
        JsonGuideline(p)


def test_value_errors():
    """WebGUI の数値の欄は 10.0 を返す。整数の項目は 10.0 を通さない（要約が「10.0 分」になるため）。
    どの項目も SCHEMA の型で確かめる（数の項目の true や "1.0"、true・false の項目の "false" を黙って計算に使わない）。"""
    c = conf_of("guideline")
    assert value_errors(c) == []
    c["tc_min"] = 10.0
    c["rain"]["hours"] = 24.5
    assert [p for p, _ in value_errors(c)] == ["tc_min", "rain.hours"]
    with pytest.raises(ValueError, match="10.0 ではなく 10"):
        check_study(c)
    s = {"kind": "storage", "conditions": {"tc_min": 10, "f": True}, "long_record": {"starts": {"千葉": 1967.0}, "end_year": 2025},
         "events": [{"in_report": True, "overflow_detail": False}, {"in_report": "false"}]}
    assert [p for p, _ in value_errors(s)] == ["conditions.f", "long_record.starts.千葉", "events[1].in_report"]
    d = conf_of("infiltration")
    assert value_errors(d) == []
    fac = d["infiltration"]["facilities"][0]
    fac |= {"count": True, "factors": {"clogging": 0.9, "groundwater": "0.9", "safety": 1.0}, "void_parts": "1×50×2"}
    d["catchments"][0]["area_ha"] = "1.0"
    d["rain"] = {"formula": {"type": "cleveland", "a": 2439, "b": 11.9, "n": "3/x"}}
    d["inflow_csv"] = {"reverse": "false"}
    assert [t for _, t in value_errors(d)] == [
        'rain.formula.n（今は "3/x"。数か分数の文字列（"3/4" など）で書く）',
        'catchments[0].area_ha（今は "1.0"。数で書く）',
        "infiltration.facilities[0].count（今は true。数で書く）",
        'infiltration.facilities[0].factors.groundwater（今は "0.9"。数で書く）',
        'infiltration.facilities[0].void_parts（今は "1×50×2"。リストで書く）',
        'inflow_csv.reverse（今は "false"。true か false で書く）']
    with pytest.raises(ValueError, match="数で書く"):
        check_study(d)
    d["inflow_csv"]["reverse"] = True
    d["catchments"][0]["area_ha"] = 1.0
    d["rain"]["formula"]["n"] = "3/4"
    fac |= {"count": 1, "factors": {"clogging": 0.9, "groundwater": 0.9, "safety": 1.0}, "void_parts": []}
    assert value_errors(d) == []


def test_output_folder_that_is_a_file_is_an_error(tmp_path):
    """out がフォルダでなくファイルなら、理由を示して止まる（ファイルを消したり上書きしたりしない）。"""
    study = tmp_path / "hydrograph"
    shutil.copytree(CASES / "hydrograph", study)
    (study / "out").write_text("メモ", encoding="utf-8")
    with pytest.raises(ValueError, match="フォルダではなくファイル"):
        run_study(conf_of("hydrograph"), study)
    assert (study / "out").read_text(encoding="utf-8") == "メモ"


def test_unknown_keys_and_removal():
    c = conf_of("two_tier_pump") | {"titel": "書き間違い"}
    c["orifices"][1]["diamter_m"] = 0.1
    assert sorted(unknown_keys(c)) == ["orifices[1].diamter_m", "titel"]
    fixed = remove_keys(c, unknown_keys(c))
    assert unknown_keys(fixed) == [] and "titel" in c                   # 元の dict は変えない


def test_dump_study_format():
    """書き戻し: SCHEMA の順、分数の文字列はそのまま、UTF-8、CRLF、2字下げ。書く前に確かめる。"""
    c = conf_of("two_tier_pump")
    shuffled = dict(reversed(list(c.items())))
    data = dump_study(shuffled)
    text = data.decode("utf-8")
    assert "\r\n" in text and "\n" not in text.replace("\r\n", "")
    from amamizu.adapters.study_config import META, SCHEMA
    order = [k for k in list(META) + list(SCHEMA["design"]) if k in c]
    assert list(json.loads(text)) == order                            # 並べ替えた dict を渡しても SCHEMA の順
    assert json.loads(text)["rain"]["formula"]["n"] == "3/4"
    assert json.loads(text) == c
    with pytest.raises(ValueError, match="不明な項目"):
        dump_study(c | {"titel": "x"})


# 組み立て ---------------------------------------------------------------------
def test_errors_carry_their_section(tmp_path):
    """誤りには、WebGUI のどのタブの誤りかが付く。"""
    cases = [
        ({"rain": {"formula": {"type": "talbot", "a": 1, "b": 1, "n": 0.5}, "shape": "rear"}}, "流入", "n を書かないでください"),
        ({"pond": {"prismatic": {"area_m2": 100.0}}}, "池", "必要な項目 hwl_m がありません"),
        ({"orifice": {"sill_m": 0.0, "shape": "oval"}}, "放流施設", "オリフィスの形「oval」は扱えません"),
        ({"infiltration": {"rate_m3s": 0.01, "facilities": []}}, "浸透", "のどれか一つを書いてください"),
        ({"catchments": [{"name": "屋根", "area_ha": 1.0, "f": 0.9, "infiltration": {"rate_m3s": 0.01}}],
          "infiltration": {"rate_m3s": 0.01}}, "浸透", "のどちらか一方に書いてください"),      # 最上位と区域ごとの両方
        ({"catchments": [{"name": "屋根", "area_ha": 1.0, "f": 0.9,
                          "infiltration": {"rate_m3s": 0.01, "intensity_mmh": 10}}]}, "浸透", "のどれか一つを書いてください"),
        ({"catchments": [{"name": "屋根", "area_ha": 1.0, "f": 0.9}, {"name": "屋根", "area_ha": 1.0, "f": 0.9}]}, "流入",
         "区域の名前が重なっています"),
        ({"end_min": 100}, "判定と実行", "流入の期間 1440 分より短いです"),
        ({"guideline": "tokyo_2020"}, "全体", "手引の定数のファイルがありません"),
    ]
    for over, section, message in cases:
        with pytest.raises(ConfigError, match=message) as e:
            trial(conf_of("guideline") | over, tmp_path)
        assert e.value.section == section, (over, str(e.value))


def test_zone_error_goes_only_to_inflow(tmp_path):
    """区域の面積・流出係数の誤りは「流入」のタブだけに付く。区域に浸透を書いていても「浸透」のタブに重ねない。"""
    c = conf_of("guideline")
    c["catchments"] = [{"name": "屋根", "area_ha": 0.0, "f": 0.9, "infiltration": {"intensity_mmh": 10}}]
    chk = check_sections(c, CASES / "guideline")
    assert "面積" in chk.errors["流入"] and "浸透" not in chk.errors
    c["catchments"][0] |= {"area_ha": 1.0, "f": 1.5}
    chk = check_sections(c, CASES / "guideline")
    assert "流出係数" in chk.errors["流入"] and "浸透" not in chk.errors


def test_tables_in_memory_are_used(tmp_path):
    """WebGUI で編集中の表（メモリ上の表）を渡すと、input/ のファイルの代わりに使う。"""
    c = conf_of("five_methods")
    table = read_table(CASES / "five_methods", "input/pond.csv")
    from_file = trial(c, CASES / "five_methods")
    in_memory = trial(c, tmp_path, {"pond": table})                   # tmp_path には input/ がない
    assert in_memory.lines == from_file.lines
    bigger = table.assign(容量_m3=table["容量_m3"] * 2)
    assert trial(c, tmp_path, {"pond": bigger}).result.summary["最大水位_m"] < from_file.result.summary["最大水位_m"]
    for bad, msg in ((table.drop(columns=["面積_m2", "容量_m3"]), "どちらか"),
                     (table.assign(容量_m3=[None] + list(table["容量_m3"][1:])), "一部が空欄")):
        with pytest.raises(ConfigError, match=msg) as e:
            trial(c, tmp_path, {"pond": bad})
        assert e.value.section == "池"


def test_pond_table_without_area_or_volume(tmp_path):
    """池の表は、面積か容量の一方を省ける（列がないか、列がすべて空欄）。容量を省くと面積から平均断面法で求め、
    面積を省くと結果の表に水面積の列を出力しない。"""
    c = conf_of("five_methods")
    table = read_table(CASES / "five_methods", "input/pond.csv")
    h, a = table["水位_m"].to_numpy(), table["面積_m2"].to_numpy()
    trapezoid = np.concatenate([[0.0], np.cumsum((a[:-1] + a[1:]) / 2 * np.diff(h))])
    by_area = trial(c, tmp_path, {"pond": table.assign(容量_m3=None)})
    assert np.allclose(by_area.prepared.pond.volumes, trapezoid)
    same = trial(c, tmp_path, {"pond": table.assign(容量_m3=trapezoid)})
    assert by_area.lines == same.lines
    no_area = trial(c, tmp_path, {"pond": table.drop(columns=["面積_m2"])})
    assert "水面積(m2)" not in no_area.result.table.columns and "水面積(m2)" in same.result.table.columns
    assert no_area.result.summary["最大水位_m"] == trial(c, tmp_path, {"pond": table}).result.summary["最大水位_m"]


@pytest.mark.needs_japanese_font
def test_trial_writes_nothing_and_run_study_writes_outputs(tmp_path):
    """試算は out/ に書かない。run_study（cli と WebGUI の「保存して出力」）は out/ に書き、表・要約の順に返す。
    出力の中身は tests/test_design_golden.py が確認済みの出力と比べる。"""
    study = tmp_path / "hydrograph"
    shutil.copytree(CASES / "hydrograph", study)
    trial(conf_of("hydrograph"), study)
    assert not (study / "out").exists()
    r = run_study(conf_of("hydrograph"), study)
    assert [p.name for p in r.files][:2] == ["route.csv", "design_summary.md"]
    assert all(p.is_file() and p.parent == study / "out" for p in r.files)


@pytest.mark.needs_japanese_font
def test_run_study_removes_rain_csv_of_previous_conditions(tmp_path):
    """流入をハイドログラフにして出力し直したら、前の条件（計画降雨）の rain.csv を out/ に残さない。
    検討の出力ではないファイル（降雨のページで保存した降雨、利用者が保存したファイル）は削除しない。"""
    study = tmp_path / "study"
    shutil.copytree(CASES / "guideline", study)
    shutil.copytree(CASES / "hydrograph" / "input", study / "input", dirs_exist_ok=True)
    run_study(conf_of("guideline"), study)
    assert (study / "out" / "rain.csv").exists()
    for name in ("observed_rain.csv", "memo.txt"):
        (study / "out" / name).write_text("前の内容", encoding="utf-8")
    r = run_study(conf_of("hydrograph"), study)
    assert {p.name for p in (study / "out").iterdir()} == {p.name for p in r.files} | {"observed_rain.csv", "memo.txt"}


@pytest.mark.needs_japanese_font
def test_failed_output_also_removes_files_of_previous_output(tmp_path, monkeypatch):
    """出力の途中で失敗したときも、前の出力にだけあるファイル（前の条件の rain.csv）を残さない。
    残すと、conditions.json が消えて前の出力のファイルと分からなくなり、次に出力し直しても削除されない。"""
    from amamizu.adapters.report import design_plots
    study = tmp_path / "study"
    shutil.copytree(CASES / "guideline", study)
    shutil.copytree(CASES / "hydrograph" / "input", study / "input", dirs_exist_ok=True)
    run_study(conf_of("guideline"), study)
    (study / "out" / "memo.txt").write_text("前の内容", encoding="utf-8")

    def fail(*args, **kwargs):
        raise ValueError("グラフを出力できません")
    with monkeypatch.context() as m:
        m.setattr(design_plots, "write_all", fail)
        with pytest.raises(ValueError, match="グラフを出力できません"):
            run_study(conf_of("hydrograph"), study)
    assert {p.name for p in (study / "out").iterdir()} == {"memo.txt"}
    r = run_study(conf_of("hydrograph"), study)
    assert {p.name for p in (study / "out").iterdir()} == {p.name for p in r.files} | {"memo.txt"}


def test_record_outputs_removes_only_files_of_previous_output(tmp_path):
    """貯留施設の検証の表: 前の出力にあって今回の出力にない表（study.json から削除した降雨の表、欠測日の表）だけを削除する。"""
    out = tmp_path / "out"
    out.mkdir()
    names = ("event_20191025.tsv", "event_20230908.tsv", "daily_missing.tsv", "long_events.tsv")
    for name in (*names, "rain.csv"):
        (out / name).write_text("前の内容", encoding="utf-8")
    record_outputs(out, {"conf": {}, "pond": None}, [out / n for n in names])
    files = record_outputs(out, {"conf": {}, "pond": None}, [out / "event_20191025.tsv", out / "long_events.tsv"])
    assert {p.name for p in out.iterdir()} == {p.name for p in files} | {"rain.csv"}
    assert [p.name for p in files] == ["event_20191025.tsv", "long_events.tsv", "conditions.json"]
    (out / "conditions.json").write_text('{"files": ["../study.json", "rain.csv"]}', encoding="utf-8")
    assert [p.name for p in stale_outputs(out, [])] == ["rain.csv"]      # 記録が書き換えられていても、out/ の外は削除しない


# 計算の基準（手引の定数と放流の係数） ------------------------------------------------


def no_guideline(conf: dict) -> dict:
    """手引を使用しない検討にした条件（係数は千葉県の手引と同じ値）。"""
    return conf | {"guideline": None, "hydraulics": {"g": 9.8, "c_weir": 1.8, "c_orifice": 0.6}}


def test_guideline_must_be_written():
    """guideline を書いていない study.json は、どれかの基準の値で計算せずにエラーにする。"""
    conf = {k: v for k, v in conf_of("hydrograph").items() if k != "guideline"}
    with pytest.raises(ConfigError, match="guideline がありません.*chiba_2006.*null") as e:
        prepare(conf, CASES / "hydrograph")
    assert e.value.section == "全体"


def test_no_guideline_needs_hydraulics():
    conf = conf_of("hydrograph") | {"guideline": None}
    with pytest.raises(ConfigError, match="hydraulics（g、c_weir、c_orifice）を study.json に"):
        prepare(conf, CASES / "hydrograph")
    with pytest.raises(ConfigError, match="hydraulics に c_orifice がありません"):
        prepare(conf | {"hydraulics": {"g": 9.8, "c_weir": 1.8}}, CASES / "hydrograph")
    with pytest.raises(ConfigError, match="c_weir は数で"):
        prepare(conf | {"hydraulics": {"g": 9.8, "c_weir": "1.8", "c_orifice": 0.6}}, CASES / "hydrograph")


def test_guideline_and_hydraulics_are_exclusive():
    """手引の定数を選んだ検討に hydraulics を書いたら、どちらの係数かを黙って決めずにエラーにする。"""
    conf = conf_of("hydrograph") | {"hydraulics": {"g": 9.8, "c_weir": 1.8, "c_orifice": 0.6}}
    with pytest.raises(ConfigError, match="hydraulics を書かないでください"):
        prepare(conf, CASES / "hydrograph")


def test_no_guideline_cannot_use_guideline_rain():
    with pytest.raises(ConfigError, match="rain.guideline（手引の地区と確率年）は使えません") as e:
        prepare(no_guideline(conf_of("guideline")), CASES / "guideline")
    assert e.value.section == "流入"


def test_no_guideline_with_same_values_gives_same_result():
    """手引を使用しない検討で、手引と同じ係数と同じ強度式を書けば、結果は手引を選んだ検討と同じ。"""
    conf = conf_of("guideline")
    gl = conf["rain"]["guideline"]
    a, n, b = json.loads((CASES.parents[2] / "studies" / "_guideline" / "chiba_2006.json").read_text(encoding="utf-8"))[
        "formulas"][gl["district"]][str(gl["return_period"])]
    rain = {k: v for k, v in conf["rain"].items() if k != "guideline"} | {"formula": {"type": "cleveland", "a": a, "b": b, "n": n}}
    with_g, without = trial(conf, CASES / "guideline"), trial(no_guideline(conf) | {"rain": rain}, CASES / "guideline")
    assert without.prepared.guideline is None
    assert without.result.summary == with_g.result.summary
    assert np.array_equal(without.result.routing.out, with_g.result.routing.out)


def test_hydraulics_of_the_guideline_file_are_used(tmp_path):
    """放流の係数は選んだ手引の定数のファイルの値（係数の違うファイルでは結果が変わる）。手引を使用しない検討では study.json の値。"""
    study = tmp_path / "hydrograph"
    shutil.copytree(CASES / "hydrograph", study)
    src = json.loads((CASES.parents[2] / "studies" / "_guideline" / "chiba_2006.json").read_text(encoding="utf-8"))
    (tmp_path / "_guideline").mkdir()
    for name, hy in (("chiba_2006", src["hydraulics"]), ("other", {"g": 9.81, "c_weir": 1.7, "c_orifice": 0.62})):
        (tmp_path / "_guideline" / f"{name}.json").write_text(json.dumps(src | {"hydraulics": hy}, ensure_ascii=False), encoding="utf-8")
    conf = conf_of("hydrograph")
    base, other = trial(conf, study), trial(conf | {"guideline": "other"}, study)
    assert base.result.summary["最大放流量_m3s"] != other.result.summary["最大放流量_m3s"]
    custom = trial(conf | {"guideline": None, "hydraulics": {"g": 9.81, "c_weir": 1.7, "c_orifice": 0.62}}, study)
    assert custom.result.summary["最大放流量_m3s"] == other.result.summary["最大放流量_m3s"]
    with pytest.raises(ConfigError, match="手引の定数のファイルがありません.*chiba_2006、other"):
        prepare(conf | {"guideline": "tokyo"}, study)


def test_storage_study_needs_a_guideline(tmp_path):
    """手引の計画降雨と比べる検討では、guideline に null を書けない。"""
    from amamizu.adapters.guideline.json_guideline import JsonGuideline
    with pytest.raises(ValueError, match="kind: storage.*null は書けません"):
        JsonGuideline.required({"kind": "storage", "guideline": None}, tmp_path)


def test_notice_of_allowable_discharge_follows_the_judgement():
    """許容放流量を超えた知らせは、判定（放流量）と同じ基準で表示する（判定が N.G なのに知らせがない、にしない）。"""
    from types import SimpleNamespace

    from amamizu.app.design_runner import notices
    from amamizu.usecases.design_study import _summary
    r = SimpleNamespace(out=np.array([0.0, 0.1 * (1 + 5e-7)]), h=np.array([0.0, 1.0]), spill=0.0, vmax=100.0, overdraw=())
    pond = SimpleNamespace(capacity=1000.0, hwl=2.0)
    for allow, over in ((0.1, True), (0.1 * (1 + 1e-6), False)):
        s = _summary(np.array([0.0, 1.0]), r, pond, [], 10, allow, None)
        assert s["判定"]["放流量"].startswith("N.G") is over
        res = SimpleNamespace(summary=s, routing=r)
        assert any("許容放流量" in n for n in notices(res)) is over
