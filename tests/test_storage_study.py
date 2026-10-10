"""貯留施設の検証（2026年9月）の数値が、確認済みの出力（tests/golden/）と一致すること。

レポートの結論になる数値で、ほかに正解がないので、意図しない変化をここで見つける。意図して変えたときだけ golden を更新する。"""
from pathlib import Path

import pandas as pd
import pytest

from amamizu.adapters.jma_store import JmaStore
from amamizu.adapters.guideline.json_guideline import JsonGuideline
from amamizu.usecases.published_check import exceedance_sets
from amamizu.usecases.storage_study import DURATIONS, StorageStudy, rounded

CHIBA = Path(__file__).parents[1] / "studies" / "_guideline" / "chiba_2006.json"      # 千葉県の手引の定数

ROOT = Path(__file__).parents[1]
GOLDEN = ROOT / "tests" / "golden"
STUDY = ROOT / "studies" / "2026-09_貯留施設の検証"
STARTS = {"千葉": 1967, "館山": 1969, "勝浦": 1969, "銚子": 1964}


def test_storage_tables_described():
    """amamizu study が out/ に書く表は、WebGUI で目的を示す一覧（STORAGE_TABLES と降雨ごとの表）にすべてある。"""
    import re
    from amamizu.adapters.study_config import read_study
    from amamizu.app.viewer import STORAGE_TABLES, event_tables, storage_table_notes
    written = set(re.findall(r'"(\w+\.tsv)"', (ROOT / "src" / "amamizu" / "cli.py").read_text(encoding="utf-8")))
    assert written == set(STORAGE_TABLES)
    conf = read_study(STUDY)
    assert [n for n, _ in event_tables(conf)] == ["event_20260813.tsv", "event_20260920.tsv"]
    assert list(storage_table_notes(conf, STUDY))[:2] == ["event_20260813.tsv", "event_20260920.tsv"]


@pytest.fixture(scope="module")
def event_tables_computed():
    """貯留施設の検証の降雨ごとの表（地点ごとの最大貯留率・溢水量）。"""
    from amamizu.adapters.study_config import read_study
    conf = read_study(STUDY)
    study = StorageStudy(JsonGuideline(CHIBA), JmaStore(ROOT / "studies"))
    return {ev["name"]: study.event_at_stations(ev["stations"], ev["start"], ev["end"])
            .set_index("地点") for ev in conf["events"]}


def test_overflow_levels_section(event_tables_computed):
    """溢れた地点の水位の節は、表で溢れた地点を溢水量の多い順に並べる。図の横軸はどの地点も同じで、
    参考の計画降雨の水位は満水位のすぐ下まで達する（毎正時の値だけを拾うとピークが欠けて 0.93 倍になる）。"""
    from amamizu.adapters.report.storage_report import sec_overflow_levels
    from amamizu.adapters.study_config import read_study
    ev = next(e for e in read_study(STUDY)["events"] if e["name"] == "台風25号")
    df = event_tables_computed["台風25号"].reset_index()
    o = sec_overflow_levels(StorageStudy(JsonGuideline(CHIBA), JmaStore(ROOT / "studies")), df, ev, 5, 4)
    assert o[0] == "## 5 台風25号で溢れた6地点の貯留施設の水位"
    # 最初の24時間の水を放流しきらないうちに、2日目の雨で満水位に達した（値は計算から作る）
    assert ("6地点とも、最初の24時間の雨で水位は0.56〜0.90 mまで上がり、その後も0.46〜0.81 mを下回らないまま、"
            "9/21 17:10〜19:50（41〜44時間目）の雨で満水位（2.0 m）に達した。降り始めから降り終わりまでは39〜43時間だった。") in o
    rows = [s.split(" | ") for s in o if s.startswith("| ") and not s.startswith(("| 地点", "| ---"))]
    over = df[df["溢水量_m3"] > 0].sort_values("溢水量_m3", ascending=False)
    assert [r[0][2:] for r in rows] == list(over["地点"]) == ["牛久", "千葉", "茂原", "木更津", "船橋", "佐倉"]
    assert all(float(r[2]) > 0 for r in rows)                                  # 満水の時間
    assert len({s for s in o if "x-axis" in s}) == 1                           # 横軸はどの地点も同じ
    # 雨量の棒は縦軸を0からにし、値0の棒を背景色で重ねて0より下を隠す（降っていない時間を着色しない）
    bars = [s for s in o if s.startswith("    bar [")]
    assert len(bars) == 12 and all(set(s[9:-1].split(", ")) == {"0"} for s in bars[1::2])
    assert all(s.startswith('    y-axis "mm" 0 --> ') for s in o if 'y-axis "mm"' in s)
    lines = [s for s in o if s.startswith("    line [")]                  # 地点ごとに 実測の水位、計画降雨の水位、満水位
    plan_top = [max(float(v) for v in s[10:-1].split(", ")) for s in lines[1::3]]
    assert len(plan_top) == 6 and all(0.99 * 2.0 <= v <= 2.0 for v in plan_top)


def test_event_studies_match_storage_study(event_tables_computed):
    """降雨ごと・地点ごとの設計の検討（amamizu event-studies で作ったもの）は、すべての地点がそろい、
    池とオリフィスが安全側に丸めたモデル施設なので、流入を手引の1/50の計画降雨にすると余裕を持って O.K になる。
    実測降雨では、貯留施設の検証の表と同じ施設・同じ計算なので、計画容量・最大貯留率・溢水量が表と一致する。"""
    from amamizu.adapters.study_config import read_study
    from amamizu.app.design_runner import trial
    from amamizu.app.event_studies import event_study_dir
    assert sum(len(t) for t in event_tables_computed.values()) == 36
    for name, table in event_tables_computed.items():
        for station, row in table.iterrows():
            d = event_study_dir(ROOT / "studies", STUDY, name, station)
            conf = read_study(d)
            s = trial(conf, d).result.summary
            assert s["容量_m3"] == row["計画容量_m3"], (name, station)
            assert s["溢水量_m3"] == pytest.approx(row["溢水量_m3"], abs=1e-9), (name, station)
            assert s["最大貯留率"] == pytest.approx(row["最大貯留率"], abs=1e-12), (name, station)
            conf.pop("observed_rain")
            conf["rain"] = {"guideline": {"district": row["地区"], "return_period": 50}, "shape": "rear"}
            p = trial(conf, d).result.summary
            assert p["判定"]["総合"] == "O.K" and p["余裕量_m3"] > 0.1, (name, station)


def test_event_studies_are_reproducible(tmp_path):
    """amamizu event-studies で作り直すと、コミットした検討と同じ study.json（作成日時を除く）になる。実測降雨は studies/_jma/ の
    10分値を地点・期間で指定する。同じ名前のフォルダがあれば作らずに止まる。フォルダ名に空白を入れない。"""
    from amamizu.adapters.study_config import read_study
    from amamizu.app.event_studies import create_event_studies
    made = create_event_studies(STUDY, tmp_path)
    assert len(made) == 36 and not any(" " in d.name or "　" in d.name for d in made)
    for d in made:
        mine, ours = read_study(d), read_study(ROOT / "studies" / d.name)
        mine.pop("created"), ours.pop("created")
        assert mine == ours, d.name
        assert mine["observed_rain"]["series"] == "tenmin" and not (d / "input").exists()
    with pytest.raises(ValueError, match="同じ名前"):
        create_event_studies(STUDY, tmp_path)


def test_event_studies_rewrite_relative_guideline(tmp_path):
    """貯留施設の検証の guideline が .json の相対パスなら、作る検討の study.json には、その検討から見たパスを書く。"""
    import shutil
    from amamizu.adapters.study_config import dump_study, read_study
    from amamizu.app.event_studies import create_event_studies
    src = tmp_path / "src" / STUDY.name
    src.mkdir(parents=True)
    shutil.copy2(CHIBA, src / "custom.json")
    conf = read_study(STUDY)
    ev = conf["events"][0] | {"stations": ["佐倉"]}
    (src / "study.json").write_bytes(dump_study(conf | {"guideline": "custom.json", "events": [ev]}))
    (tmp_path / "src" / "_jma").mkdir()
    shutil.copy2(ROOT / "studies" / "_jma" / "config.json", tmp_path / "src" / "_jma")
    folder = JmaStore(ROOT / "studies").station_dir("佐倉")
    shutil.copytree(folder, tmp_path / "src" / "_jma" / folder.name)
    [made] = create_event_studies(src, tmp_path / "dst")
    assert read_study(made)["guideline"] == f"../../src/{STUDY.name}/custom.json"
    assert JsonGuideline.of_study(read_study(made), made) is not None


def test_report_events_and_unique_start():
    """レポートに表を載せる降雨は1つ以上。overflow_detail は表を載せる降雨にだけ付けられる。
    開始日が同じ降雨は表の名前が重なるのでエラー。"""
    from amamizu.adapters.report.storage_report import section_numbers
    from amamizu.app.viewer import report_events, event_tables
    ev = {"name": "a", "stations": ["千葉"], "start": "2026-08-13", "end": "2026-08-15"}
    with pytest.raises(ValueError, match="1つ以上"):
        report_events({"events": [ev]})
    with pytest.raises(ValueError, match="overflow_detail"):
        report_events({"events": [dict(ev, in_report=True), dict(ev, name="b", overflow_detail=True)]})
    two = {"events": [dict(ev, in_report=True), dict(ev, name="b", start="2026-09-20", in_report=True, overflow_detail=True)]}
    assert [e["name"] for e in report_events(two)] == ["a", "b"]
    assert section_numbers(2, 1) == {"events": [3, 4], "overflow": [5], "long": 6, "details": 7, "unobserved": 8}
    assert section_numbers(1, 0) == {"events": [3], "overflow": [], "long": 4, "details": 5, "unobserved": 6}
    with pytest.raises(ValueError, match="開始日が同じ"):
        event_tables({"events": [ev, dict(ev)]})


def golden(name):
    return pd.read_csv(GOLDEN / name, sep="\t", encoding="utf-8-sig")


def same(a: pd.DataFrame, b: pd.DataFrame, keys):
    """a は usecase の結果（丸めない値）。TSV に書く桁に丸めて golden と比べる。"""
    assert list(a.columns) == list(b.columns)           # 列がなくなる・増える・順が変わるのも見つける
    a = rounded(a).sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=0, atol=1e-9)


@pytest.fixture(scope="module")
def study():
    return StorageStudy(JsonGuideline(CHIBA), JmaStore(ROOT / "studies"))


@pytest.fixture(scope="module")
def long(study):
    return study.long_record(STARTS, 2025)


def test_chiba_event(study):
    same(study.event_at_stations(["佐倉", "勝浦", "千葉", "坂畑", "大多喜", "成田", "我孫子", "木更津", "東庄", "横芝光", "牛久", "船橋", "茂原", "銚子", "鋸南", "館山", "香取", "鴨川"], "2026-08-13", "2026-08-15"), golden("chiba_event.tsv"), ["地点"])


def test_fill_not_rounded_twice(study):
    """最大貯留率は丸めずに返し、レポートの % は1回だけ丸める（0.93494 を 0.935 経由で 94% にしない）。"""
    from amamizu.adapters.report.storage_report import fill
    df = study.event_at_stations(["我孫子"], "2026-08-13", "2026-08-15")
    v = df["最大貯留率"].iat[0]
    assert v != round(v, 3)
    assert fill(0.93494, 0) == "93%"


def test_long_record(long):
    same(long, golden("long_events.tsv"), ["地点", "開始"])


def test_recompute_with_10min(study, long):
    check, final = study.recompute_with_10min(long)
    same(check, golden("tenmin_check.tsv"), ["地点", "開始"])
    same(final, golden("long_events_final.tsv"), ["地点", "開始"])
    assert int((final["溢水量_m3"] > 0).sum()) == 7


def test_year_2026(study):
    same(study.partial_year(list(STARTS), "2026-01-01", "2026-09-22"), golden("year2026.tsv"), ["地点", "開始"])


def test_published_nhour_agrees(study, long):
    _, final = study.recompute_with_10min(long)
    res = exceedance_sets(study.guideline, study.records.annual_nhour(list(STARTS)), final, study.cond.return_period, DURATIONS)
    assert all(v["一致"] for v in res.values()), res
    assert {h: len(v["計算で超過"]) for h, v in res.items()} == {1: 1, 3: 2, 6: 5, 12: 6, 24: 2}


def test_spill_never_shows_zero_when_spilling():
    """溢水の回数に入る降雨（溢水量が0より大きい）を、レポートと表（TSV）で溢水量 0 と表示しない。"""
    from amamizu.adapters.report.storage_report import spill_text
    assert spill_text(0.0) == "0"
    assert spill_text(0.013123) == "0.013" and spill_text(0.37) == "0.37"
    assert spill_text(0.00004) == "0.00004"
    assert spill_text(0.5) == "0.5" and spill_text(0.51) == "1"     # 0.5 は偶数への丸めで「0」になる
    assert spill_text(964.23) == "964"
    df = rounded(pd.DataFrame({"溢水量_m3": [0.0, 0.013123, 0.05, 964.23]}))
    assert list(df["溢水量_m3"]) == [0.0, 0.013, 0.1, 964.2]


def test_station_without_records_is_an_error():
    """観測値のない地点は、空の表で続けずに取得のコマンドを示してエラーにする。"""
    with pytest.raises(FileNotFoundError, match="地点「存在しない地点」の観測値がありません。amamizu fetch で取得してください"):
        JmaStore(ROOT / "studies").daily("存在しない地点")


@pytest.mark.parametrize("day", ["2026/01/01", "2026-1-1"])
def test_period_date_in_other_format_is_an_error(study, day):
    """期間の日付の書式が違うと、日付の文字列の比較で候補が黙って0件になる（途中の年の表が空のまま進む）ので、エラーにする。"""
    assert len(study.candidates("千葉", "2026-01-01", "2026-09-22")) == 4
    with pytest.raises(ValueError, match="2026-01-01 の形"):
        study.candidates("千葉", day, "2026-09-22")

@pytest.mark.parametrize("district, area, size, vmax", [
    ("我孫子", 527.0, 0.082, 1053.1293514933582),
    ("横利根", 701.0, 0.082, 1400.9759483741084),
    ("銚子", 763.0, 0.082, 1525.456350627754),
    ("松戸", 751.0, 0.082, 1500.8220852507725),
    ("千葉", 576.0, 0.082, 1150.5384832136099),
    ("勝浦", 876.0, 0.097, 1750.941782837781),
    ("館山", 750.0, 0.097, 1499.6479160864494),
])
def test_model_dimensions_and_storage_unchanged(district, area, size, vmax):
    """面積探索を省く前の全地区の寸法と最大貯留量を、丸め前の値で維持する。"""
    study = StorageStudy(JsonGuideline(CHIBA), JmaStore(ROOT / "studies"))
    model = study.model(district)
    assert model.orifice.size == size
    assert model.designed.facility.pond.area == area
    assert model.designed.design_result.vmax == vmax


def test_model_fits_area_once(monkeypatch):
    """丸めたオリフィスに合わせて池の面積を1回だけ探索する。"""
    from unittest.mock import Mock
    from amamizu.usecases import storage_study
    fit = Mock(wraps=storage_study.design_for_depth)
    monkeypatch.setattr(storage_study, "design_for_depth", fit)
    study = StorageStudy(JsonGuideline(CHIBA), JmaStore(ROOT / "studies"))
    model = study.model("千葉")
    assert study.model("千葉") is model
    assert fit.call_count == 1


@pytest.mark.parametrize("qc, message", [(100.0, "オリフィスが見つかりません"), (1e-12, "オリフィスが見つかりません")])
def test_model_rejects_unattainable_discharge(monkeypatch, qc, message):
    """直接算定でも寸法の上下限で放流量が合わない条件を拒否する。"""
    guideline = JsonGuideline(CHIBA)
    monkeypatch.setattr(guideline, "qc_min", lambda district: qc)
    with pytest.raises(ValueError, match=message):
        StorageStudy(guideline, JmaStore(ROOT / "studies")).model("千葉")
