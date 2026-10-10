"""条件を変えたときに、誤った結果を黙って出力しないこと（基準の選択、study.json の項目、確率年）。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amamizu.adapters.guideline.json_guideline import JsonGuideline
from amamizu.cli import load_study, main
from amamizu.usecases.guideline_check import formula_mismatches, reproduce_hayami
from amamizu.usecases.storage_study import ModelConditions, StorageStudy

CHIBA = Path(__file__).parents[1] / "studies" / "_guideline" / "chiba_2006.json"      # 千葉県の手引の定数

MINIMAL = Path(__file__).parent / "fixtures" / "guideline_minimal.json"


def test_guideline_without_hayami_verifies_formulas_only(capsys):
    g = JsonGuideline(MINIMAL)
    assert g.hayami() == []
    assert reproduce_hayami(g) == []
    main(["verify", "--guideline", str(MINIMAL)])
    out = capsys.readouterr().out
    assert "表の値とのずれ 0 件" in out and "早見表: この基準にはありません" in out


def test_unknown_guideline_is_an_error():
    with pytest.raises(FileNotFoundError, match="手引の定数のファイルがありません.*tokyo_2020.json"):
        JsonGuideline.load("tokyo_2020", CHIBA.parents[1], CHIBA.parents[1])


def test_relative_guideline_path_is_read_from_base(tmp_path, monkeypatch):
    """.json の相対パスは、今のフォルダではなく指定した base から読む。"""
    monkeypatch.chdir(tmp_path)
    assert JsonGuideline.load("_guideline/chiba_2006.json", tmp_path, CHIBA.parents[1]).name == JsonGuideline(CHIBA).name
    with pytest.raises(FileNotFoundError, match="手引の定数のファイルがありません"):
        JsonGuideline.load("_guideline/chiba_2006.json", tmp_path, tmp_path)


def test_relocated_guideline(tmp_path):
    """別のフォルダの study.json へ写すとき、.json の相対パスだけを写す先から見たパスに直す。名前・絶対パス・null はそのまま。
    写す先へコピーするフォルダ（copied）の中のファイルは、コピーしたほうを指す。"""
    from amamizu.adapters.guideline.json_guideline import relocated
    src, dst = tmp_path / "a", tmp_path / "b" / "c"
    assert relocated("custom.json", src, dst, ()) == "../../a/custom.json"
    assert relocated("../_guideline/x.json", src, tmp_path / "b", ()) == "../_guideline/x.json"
    assert relocated("chiba_2006", src, dst, ()) == "chiba_2006"
    assert relocated(None, src, dst, ()) is None
    assert relocated(str(CHIBA), src, dst, ()) == str(CHIBA)
    assert relocated("input/custom.json", src, dst, ("input",)) == "input/custom.json"
    assert relocated("../a/input/custom.json", src, dst, ("input",)) == "input/custom.json"
    assert relocated("custom.json", src, dst, ("input",)) == "../../a/custom.json"
    assert relocated("input/custom.json", src, dst, ()) == "../../a/input/custom.json"


def test_guideline_names_what_is_missing():
    """手引にない地区・確率年・地点は、何がないかと選べる値を示すエラーにする。"""
    g = JsonGuideline(CHIBA)
    with pytest.raises(KeyError, match="地区「東京」"):
        g.formula("東京", 50)
    with pytest.raises(KeyError, match="確率年 7 年"):
        g.formula("千葉", 7)
    with pytest.raises(KeyError, match="地点「東京」"):
        g.district_of("東京")


def guideline_with(tmp_path, base: Path, **changes) -> JsonGuideline:
    """base の手引の定数の項目を changes で置き換えたファイルを読む。"""
    d = json.loads(base.read_text(encoding="utf-8")) | changes
    p = tmp_path / "guideline.json"
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return JsonGuideline(p)


def test_hayami_return_period_is_read_from_the_guideline(tmp_path):
    """早見表の照合の確率年は、手引の定数の hayami_conditions.return_period の値で計算する。"""
    base = JsonGuideline(CHIBA)
    cells = [c for c in base.hayami() if c.district == "千葉"][:2]
    c0 = base.hayami_conditions()
    g30 = guideline_with(tmp_path, CHIBA, hayami_conditions=c0 | {"return_period": 30})
    v50 = [r.calc for r in reproduce_hayami(base, cells)]
    v30 = [r.calc for r in reproduce_hayami(g30, cells)]
    assert len(v30) == 2 and all(a < b for a, b in zip(v30, v50, strict=True))


def test_formula_table_is_compared_at_its_digits(tmp_path):
    """強度式の照合は、表の値を formula_table_conditions.digits の桁に丸めて比べる。"""
    assert formula_mismatches(JsonGuideline(MINIMAL)) == []
    g = guideline_with(tmp_path, MINIMAL, formula_table_conditions={"t_min": [10, 60, 1440], "digits": 0})
    assert len(formula_mismatches(g)) == 3


def write_study(tmp_path, conf):
    (tmp_path / "study.json").write_text(json.dumps(conf, ensure_ascii=False), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("conf, bad", [
    ({"kind": "return_period", "standard_period": 10, "duration_min": 10, "last_year": 2025, "guidline": "x"}, "guidline"),
    ({"kind": "storage", "conditions": {}, "long_record": {"starts": {}, "end_year": 2025, "end": 1},
      "partial_year": {}, "events": []}, "long_record.end"),
])
def test_unknown_study_keys_are_errors(tmp_path, conf, bad):
    """書き間違えた項目が、既定値のまま黙って通らない。"""
    with pytest.raises(ValueError, match=bad.replace(".", r"\.")):
        load_study(write_study(tmp_path, conf))


def test_unknown_kind_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        load_study(write_study(tmp_path, {"kind": "storge"}))
    with pytest.raises(ValueError, match="kind がありません"):
        load_study(write_study(tmp_path, {"title": "x"}))


@pytest.mark.parametrize("raw, why", [
    (b"{ kind: design,", "JSON として読めません"),
    (b"", "JSON として読めません"),
    (b"[1, 2]", "項目を持つ形"),
    ('{"kind": "design", "title": "題名"}'.encode("cp932"), "UTF-8 ではありません"),
])
def test_study_json_that_cannot_be_read_says_why(tmp_path, raw, why):
    """読めない study.json は、理由と直し方を示すエラーにする（コマンドと WebGUI で同じ文）。"""
    (tmp_path / "study.json").write_bytes(raw)
    with pytest.raises(ValueError, match=why):
        load_study(tmp_path)


def test_shape_errors_are_told_apart_from_unknown_keys():
    """形の違う項目（並びのはずが辞書など）は、不明な項目とは別に取り出せる（WebGUI は開かずに理由を示す）。
    study.json の確認では、形の違いを不明な項目とせず、値を直す項目として示す。"""
    from amamizu.adapters.study_config import check_study, shape_errors, unknown_keys
    conf = {"kind": "design", "catchments": {"name": "x"}, "pond": []}
    assert shape_errors(conf) == ["catchments（リストで書く）", "pond（項目を持つ形で書く）"]
    assert unknown_keys(conf | {"memo": 1}) == ["memo"]
    with pytest.raises(ValueError, match=r'値を直してください.*catchments（今は \{"name": "x"\}。リストで書く）'):
        check_study(conf)


def test_columns_follow_return_period():
    """確率年を30にしたら、列名も計画の雨量も1/30になる（1/50 と表示しない）。"""
    g = JsonGuideline(CHIBA)
    study = StorageStudy(g, None, ModelConditions(return_period=30))
    cols = study.duration_cols("千葉", np.array([10.0, 20.0, 5.0] + [0.0] * 21), pd.Series([np.nan]))
    assert "1h_1/30_mm" in cols and not any("1/50" in k for k in cols)
    assert cols["24h_1/30_mm"] == g.formula("千葉", 30).depth(1440)      # 丸めるのは表に書くとき（rounded）


# 日別値と10分値 ------------------------------------------------------------
ROOT = Path(__file__).parents[1]
STORAGE = ROOT / "studies" / "2026-09_貯留施設の検証"
DANSEN = ROOT / "studies" / "2026-09_降雨強度の検証"


@pytest.fixture(scope="module")
def records():
    from amamizu.adapters.jma_store import JmaStore
    return JmaStore(ROOT / "studies")


def test_daily_gap_in_period_is_an_error(records):
    """日別値のない期間の降雨を、黙って見落とさない（千葉の日別値は1967年から）。"""
    study = StorageStudy(JsonGuideline(CHIBA), records)
    with pytest.raises(ValueError, match="日別値が"):
        study.candidates("千葉", "1966-01-01", "1970-12-31")


def test_missing_daily_values_are_listed(records):
    """日別値が欠測の日は、降雨を区切らず、その降雨と判定と一緒に一覧に載る。
    館山 2016-08-19・20 が欠測で、前後の降雨は1つにつながる。2024-05-10 も欠測で、つながった降雨は基準雨量以上になる。"""
    study = StorageStudy(JsonGuideline(CHIBA), records)
    rows = study.missing_days("館山", "2016-01-01", "2025-12-31")
    by_day = {}
    for r in rows:
        by_day.setdefault(r["日別値が欠測の日"], []).append(r)
    assert [(r["降雨"], r["観測できた総量_mm"], r["判定"]) for r in by_day["2016-08-19"]] == [
        ("2016-08-16〜2016-08-22", 168.0, "計算した")]
    assert [r["降雨"] for r in by_day["2024-05-10"]] == ["2024-05-07〜2024-05-16"]
    assert by_day["2023-11-01"][0]["判定"] == "前後2日に降雨なし"
    assert by_day["2023-12-09"][0]["判定"] == "計算していない（基準雨量 93.8mm まで 60.2mm）"


def test_period_without_heavy_rain_gives_empty_table(records):
    """基準雨量以上の降雨が1件もない期間でも、列のある0行の表を返す（列名の KeyError で止まらない）。"""
    study = StorageStudy(JsonGuideline(CHIBA), records)
    assert study.candidates("千葉", "2026-01-01", "2026-01-31") == []
    y = study.partial_year(["千葉"], "2026-01-01", "2026-01-31")
    assert len(y) == 0 and {"地点", "開始", "最大貯留率", "溢水量_m3", "24h_1/50比", "欠測コマ"} <= set(y.columns)
    full = study.partial_year(["千葉"], "2026-01-01", "2026-09-22")
    assert list(full.columns) == list(y.columns)


def test_exceedance_is_judged_with_unrounded_values():
    """計画の雨量を超えたかは、丸めた比（0.9996 → 1.000）ではなく雨量で決める。"""
    from amamizu.usecases.published_check import exceedance_sets
    g = JsonGuideline(CHIBA)
    plan = g.formula(g.district_of("銚子"), 50).depth(60 * 12)       # 銚子の12時間は 234.52 mm
    below = 234.5
    assert below < plan and round(below / plan, 3) == 1.0
    events = pd.DataFrame([{"地点": "銚子", "開始": "2000-07-01", "12h_mm": below, "12h_1/50比": round(below / plan, 3)},
                           {"地点": "銚子", "開始": "2001-07-01", "12h_mm": 235.0, "12h_1/50比": round(235.0 / plan, 3)}])
    published = pd.DataFrame([{"地点": "銚子", "年": 2001, "最大12時間/(mm)": "235.0"}])
    res = exceedance_sets(g, published, events, 50, durations=(12,))
    assert res[12]["計算で超過"] == {("銚子", 2001)} and res[12]["一致"]


def test_event_without_tenmin_is_an_error(records):
    study = StorageStudy(JsonGuideline(CHIBA), records)
    with pytest.raises(ValueError, match="10分値がそろっていません"):
        study.event_at_stations(["千葉"], "1990-01-01", "1990-01-02")
    with pytest.raises(ValueError, match="stations に、計算する地点がありません"):
        study.event_at_stations([], "2026-08-13", "2026-08-15")
    with pytest.raises(ValueError, match="同じ地点が2回以上あります: 千葉"):
        study.event_at_stations(["千葉", "勝浦", "千葉"], "2026-08-13", "2026-08-15")


# レポートの文 ------------------------------------------------------------------
def test_storage_report_text_follows_conditions(records):
    """開発面積・流出係数・確率年を変えたら、レポートの文もそれに従う（前の検討の文が残らない）。"""
    from amamizu.adapters.report.storage_report import sec_design_rain, sec_ponds
    study = StorageStudy(JsonGuideline(CHIBA), records, ModelConditions(f=0.9, area_ha=2.0, return_period=30))
    text = "\n".join(sec_ponds(study, []) + sec_design_rain(study, {d: np.zeros(144) for d in study.guideline.districts()}))
    assert "2ha" in text and "不浸透面積率75%" in text and "1/30計画降雨" in text and "計画降雨（1/30" in text
    assert "1/50" not in text and "1ha" not in text and "早見表" not in text


def test_storage_report_refuses_runoff_it_cannot_describe(records):
    from amamizu.adapters.report.storage_report import impervious_ratio
    with pytest.raises(ValueError, match="範囲外"):
        impervious_ratio(StorageStudy(JsonGuideline(CHIBA), records, ModelConditions(f=0.5)))


def test_return_period_report_text_follows_study():
    from amamizu.adapters.files import AnnualMaxFiles
    from amamizu.adapters.report.return_period_report import render
    from amamizu.usecases.return_period_study import evaluate
    g = JsonGuideline(CHIBA)
    conf = load_study(DANSEN)
    conf = {**conf, "standard_period": 5}
    from amamizu.adapters.jma_store import JmaStore
    records = AnnualMaxFiles(DANSEN / "raw", JmaStore(ROOT / "studies"))
    text = render(evaluate(g, records, 5, 10, 2025), conf, g, "2027-01-01")
    assert "県の5年確率式" in text and "県式5年" in text and "取得日 2027年1月1日" in text
    assert "10年確率" not in text


def test_hayami_sentence_needs_the_row_of_f(records):
    """f＝0.73 は早見表の行にないので、上限（0.9）の行だけを照合して「f＝0.9 の行も含めた」と書かない。"""
    from amamizu.adapters.report.storage_report import sec_ponds
    g = JsonGuideline(CHIBA)
    study = StorageStudy(g, records, ModelConditions(f=0.73))
    cap_cells = [c for c in g.hayami() if c.f in {0.73, 0.9} and c.qc == g.qc_min(c.district)]
    assert len(cap_cells) == 7
    text = "\n".join(sec_ponds(study, reproduce_hayami(g, cap_cells)))
    assert "早見表" not in text and "不浸透面積率32%" in text


class FakeRecords:
    """時間値に欠測があり、合計が日別値と合わない降雨を1つだけ持つ記録。"""

    def daily(self, station, year_from=""):
        import pandas as pd
        days = pd.date_range("2000-01-01", "2000-12-31").strftime("%Y-%m-%d")
        rain = [0.0] * len(days)
        rain[3] = 200.0      # 2000-01-04
        return pd.DataFrame({"日付": days, "日合計_日別値_mm": rain, "最大1時間_日別値_mm": [x / 4 for x in rain],
                             "日別値_品質": "normal"})

    def hourly(self, station):
        import pandas as pd
        rain = [0.0] * 24
        rain[10:20] = [10.0] * 10    # 100mm。残りの時間は欠測
        q = ["normal"] * 24
        q[20:24] = ["missing"] * 4
        return pd.DataFrame({"日付": "2000-01-04", "時": range(1, 25), "降水量_mm": rain, "品質": q})


def test_hourly_mismatch_with_missing_is_noticed():
    """欠測を含む降雨は止めずに計算するが、日別値と合わないことを notices で知らせる。"""
    study = StorageStudy(JsonGuideline(CHIBA), FakeRecords())
    long = study.long_record({"千葉": 2000}, 2000)
    assert len(long) == 1 and long["欠測時間"].iat[0] == 4
    assert len(study.notices) == 1 and "欠測 4 時間" in study.notices[0]


# 欠測の時間 ------------------------------------------------------------
def test_unobserved_time_is_shown_not_filled(records):
    """欠測は推測で埋めず、どの時間が欠測かを示す（館山は 2016-08-19 0時〜8/21 17時が欠測）。"""
    from amamizu.adapters.report.storage_report import spans_text
    study = StorageStudy(JsonGuideline(CHIBA), records)
    spans = study.unobserved("館山", "2016-08-21", "2016-08-22", "時間値")
    assert spans == [("2016-08-19 00:00", "2016-08-21 17:00")]
    assert spans_text(spans, 2016) == "8/19 0:00〜8/21 17:00"
    assert study.unobserved("館山", "2024-07-11", "2024-07-17", "時間値") == [("2024-07-09 09:00", "2024-07-09 17:00")]
    assert spans_text([("2010-10-30 14:00", "2010-10-31 00:00")], 2010) == "10/30 14:00〜24:00"


def test_uncomputed_events_near_missing_days_are_listed(records):
    """観測できた総量が基準未満で計算しなかった降雨も、近くに日別値の欠測があれば一覧に載せる。"""
    study = StorageStudy(JsonGuideline(CHIBA), records)
    rows = {r["開始"]: r for r in study.uncomputed_near_missing("館山", "2023-01-01", "2024-12-31")}
    assert rows["2023-12-11"]["日別値が欠測の日"] == ["2023-12-09", "2023-12-10"]
    assert rows["2023-12-11"]["観測できた総量_mm"] == 33.5
    assert "2024-05-07" not in rows      # 欠測の日でつながり、基準雨量以上になって計算した

# 気象庁からの取得（studies/_jma/ に書く） ---------------------------------------------
def jma_dir(tmp_path):
    """試験の studies/（tmp_path）に _jma/config.json を作る。"""
    (tmp_path / "_jma").mkdir(exist_ok=True)
    (tmp_path / "_jma" / "config.json").write_text('{"refresh_months": 1}', encoding="utf-8")
    return tmp_path


def test_fetch_hourly_writes_to_store_and_skips_final_days(tmp_path, monkeypatch):
    """時間値は studies/_jma/ に書き、確定の日（観測日の1か月より後に取得した日）は取り直さない。"""
    import datetime
    from amamizu.adapters import jma_fetch
    from amamizu.adapters.jma_store import JmaStore
    asked = []

    class FakeCollector:
        def __init__(self, station, **kw):
            self.failures = []

        def collect(self, dates, on_day=None):
            for d in dates:
                asked.append(d)
                on_day(d, [{"地点": "千葉", "block_no": "47682", "日付": d.isoformat(), "時": h, "降水量_mm": 0.5, "品質": "normal"}
                           for h in range(1, 25)])

    monkeypatch.setattr(jma_fetch, "JMAHourlyCollector", FakeCollector)
    monkeypatch.setattr(jma_fetch.JmaFetcher, "station", lambda self, name: {"name": name, "block_no": "47682", "pref": "千葉県"})
    studies = jma_dir(tmp_path)
    jma_fetch.JmaFetcher(studies).hourly("千葉", [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)])
    jma_fetch.JmaFetcher(studies).hourly("千葉", [datetime.date(2020, 1, 2), datetime.date(2020, 1, 3)])
    assert asked == [datetime.date(2020, 1, 1), datetime.date(2020, 1, 2), datetime.date(2020, 1, 3)]
    df = JmaStore(studies).hourly("千葉")
    assert len(df) == 72 and set(df["地点"]) == {"千葉"}


def test_fetch_same_name_other_station_is_error(tmp_path, monkeypatch):
    """studies/_jma/ に取得済みの地点名で、別の観測所（福島県と岐阜県の金山）を取得しようとしたら、取得を始める前にエラーにする。"""
    import datetime
    from amamizu.adapters import jma_fetch

    fetched = []

    class FakeCollector:
        def __init__(self, station, **kw):
            self.station, self.failures = station, []

        def collect(self, dates, on_day=None):
            for d in dates:
                fetched.append((self.station["block_no"], d.isoformat()))
                on_day(d, [{"地点": "金山", "block_no": self.station["block_no"], "日付": d.isoformat(), "時": h,
                            "降水量_mm": 0.0, "品質": "normal"} for h in range(1, 25)])

    monkeypatch.setattr(jma_fetch, "JMAHourlyCollector", FakeCollector)
    studies = jma_dir(tmp_path)
    jma_fetch.JmaFetcher(studies, pref="福島県").hourly("金山", [datetime.date(2020, 1, 1)])
    assert (studies / "_jma" / "金山_1044").is_dir()
    jma_fetch.JmaFetcher(studies, pref="福島県").hourly("金山", [datetime.date(2020, 1, 2)])   # 同じ観測所は続きを取得する
    with pytest.raises(jma_fetch.StationMismatch, match="地点「金山」（地点番号（block_no）1044）の値を取得済み"):
        jma_fetch.JmaFetcher(studies, pref="岐阜県").hourly("金山", [datetime.date(2020, 1, 1)])
    assert fetched == [("1044", "2020-01-01"), ("1044", "2020-01-02")]


def test_fetch_daily_asks_only_missing_or_provisional_days(tmp_path, monkeypatch):
    """日別値は、値がない日と暫定の日（観測日から1か月の間に取得した日）のある月だけ、月ごとにその最初から最後までを取得する。
    確定の日は取り直さない。取得は昨日まで。"""
    import datetime
    from amamizu.adapters import jma_fetch
    from amamizu.adapters.jma_store import JmaStore

    calls = []

    class FakeDaily:
        def __init__(self, station, **kw):
            self.failures = []

        def collect(self, start, end):
            calls.append((start, end))
            days = pd.date_range(start, end).strftime("%Y-%m-%d")
            return pd.DataFrame({"日付": days, "日合計_日別値_mm": 1.0, "最大1時間_日別値_mm": 1.0,
                                 "最大10分_日別値_mm": 1.0, "日別値_品質": "normal"})

    monkeypatch.setattr(jma_fetch, "JMADailyCollector", FakeDaily)
    monkeypatch.setattr(jma_fetch.JmaFetcher, "station", lambda self, name: {"name": name, "block_no": "47682", "pref": "千葉県"})
    studies = jma_dir(tmp_path)

    class Today(datetime.date):                  # 今日の日付を固定する（1月1日に実行すると、今年の取得する日がなくなる）
        @classmethod
        def today(cls):
            return cls(2026, 10, 9)
    monkeypatch.setattr(jma_fetch, "datetime", type(datetime)("datetime"))
    jma_fetch.datetime.date, jma_fetch.datetime.timedelta = Today, datetime.timedelta
    today = datetime.date(2026, 10, 9)
    y = today.year
    store = JmaStore(studies)
    st = {"地点": "千葉", "block_no": "47682", "pref": "千葉県"}
    last_year = pd.DataFrame({"日付": pd.date_range(f"{y - 1}-01-01", f"{y - 1}-12-31").strftime("%Y-%m-%d"), "日合計_日別値_mm": 0.0,
                              "最大1時間_日別値_mm": 0.0, "最大10分_日別値_mm": 0.0, "日別値_品質": "normal"})
    mid = last_year["日付"] == f"{y - 1}-06-15"
    store.put("daily", st, last_year[~mid], datetime.date(y, 2, 15), "試験")                 # 確定（観測日の1か月より後に取得）
    store.put("daily", st, last_year[mid], datetime.date(y - 1, 6, 16), "試験")              # 暫定（翌日に取得）
    first = pd.DataFrame({"日付": [f"{y}-01-01"], "日合計_日別値_mm": 0.0, "最大1時間_日別値_mm": 0.0, "最大10分_日別値_mm": 0.0,
                          "日別値_品質": "normal"})
    store.put("daily", st, first, datetime.date(y, 1, 2), "試験")                            # 暫定（翌日に取得）
    jma_fetch.JmaFetcher(studies).daily("千葉", y - 1, y)
    yesterday = today - datetime.timedelta(days=1)
    days = pd.date_range(f"{y}-01-01", yesterday)
    assert calls == [(datetime.date(y - 1, 6, 15), datetime.date(y - 1, 6, 15))] + [      # 前年は暫定の日の月だけ
        (days[days.month == m][0].date(), days[days.month == m][-1].date()) for m in sorted(set(days.month))]
    df = store.daily("千葉", f"{y}-01-01")
    assert df["日付"].iloc[-1] == yesterday.isoformat() and df["日付"].is_unique and (df["日合計_日別値_mm"] == 1.0).all()


def test_fetch_annual_records_failure_and_writes_the_others(tmp_path, monkeypatch):
    """年ごとの値は地点ごとに書く。取れなかった地点は failures に残し、その地点は書かない。"""
    from amamizu.adapters import jma_fetch
    from amamizu.adapters.jma_store import JmaStore
    from jma_rainfall.fetch import FetchError, Pacer

    def fake_annual(station, view, session, verify):
        if station["name"] == "館山":
            raise FetchError("HTTP 404")
        return pd.DataFrame({"年": ["2013"], "最大1時間/(mm)": ["33"]})

    monkeypatch.setattr(jma_fetch, "fetch_annual", fake_annual)
    monkeypatch.setattr(jma_fetch.JmaFetcher, "station", lambda self, name: {"name": name, "block_no": "0", "pref": "千葉県"})
    studies = jma_dir(tmp_path)
    f = jma_fetch.JmaFetcher(studies)
    f.pacer = Pacer(sleep=lambda s: None)
    f.annual(["千葉", "館山"])
    assert [x["地点"] for x in f.failures] == ["館山"]
    assert JmaStore(studies).annual_nhour(["千葉"])["最大1時間/(mm)"].tolist() == ["33"]
    assert not (studies / "_jma" / "館山_0").exists()


def test_fetch_annual_max_reads_both_header_forms(tmp_path, monkeypatch):
    """年最大値系列は、官署とアメダスで見出しの形が違う表から列を選び、記号を品質にする。観測史上順位は要素ごとに置き換える。
    列が1列に決まらない表は、取れなかったものとして残す。"""
    from amamizu.adapters import jma_fetch
    from amamizu.adapters.jma_store import JmaStore
    from jma_rainfall.fetch import Pacer
    tables = {"千葉": {"年": ["2013", "2014"], "降水量(mm)/最大/1時間": ["61.5", "30.0 )"], "降水量(mm)/最大/10分間": ["19.5", "12.0 ]"]},
              "我孫子": {"年": ["2013"], "降水量/最大/1時間(mm)": ["48.5"], "降水量/最大/10分間(mm)": ["×"]},
              "館山": {"年": ["2013"], "降水量/最大/1時間(mm)": ["1"]}}
    monkeypatch.setattr(jma_fetch, "fetch_annual", lambda station, view, session, verify: pd.DataFrame(tables[station["name"]]))
    monkeypatch.setattr(jma_fetch, "fetch_rank", lambda station, session, verify: pd.DataFrame(
        {"地点": [station["name"]], "要素": ["日最大10分間降水量 (mm)"], "1位": ["19.5 (2013/10/16)"], "統計期間": ["2008/12 2026/9"]}))
    monkeypatch.setattr(jma_fetch.JmaFetcher, "station", lambda self, name: {"name": name, "block_no": name, "pref": "千葉県"})
    studies = jma_dir(tmp_path)
    f = jma_fetch.JmaFetcher(studies)
    f.pacer = Pacer(sleep=lambda s: None)
    f.annual_max(["千葉", "我孫子", "館山"])
    store = JmaStore(studies)
    assert store.annual_max("千葉", 2013, 2014).values.tolist() == [[2013, 19.5, "normal", 61.5, "normal"], [2014, 12.0, "insufficient", 30.0, "quasi"]]
    t = store.annual_max("我孫子", 2013, 2013)
    assert pd.isna(t["10分"].iat[0]) and t["10分_品質"].iat[0] == "missing"
    assert store.rank("我孫子") == {"日最大10分間降水量 (mm)": {"1位": "19.5 (2013/10/16)", "統計期間": "2008/12 2026/9"}}
    assert [(x["地点"], x["対象"]) for x in f.failures] == [("館山", "年最大値系列")]


def test_fetch_hourly_without_days_file_shows_the_reason(tmp_path, monkeypatch, capsys):
    """時間値を取得する日のファイルがない・読めないときは、経過（Traceback）なしで理由を表示し、どの地点も取得しない。"""
    from amamizu.adapters import jma_fetch
    from amamizu.cli import run

    class NoFetch:
        def __init__(self, *a, **kw):
            raise AssertionError("取得を始めてはいけない")

    monkeypatch.setattr(jma_fetch, "JMAHourlyCollector", NoFetch)
    study = jma_dir(tmp_path) / "study"
    (study / "out").mkdir(parents=True)
    (study / "out" / "candidate_days_千葉.txt").write_text("2020-01-01\n", encoding="utf-8")
    with pytest.raises(SystemExit) as e:        # 2地点目の候補日のファイルがない。1地点目も取得しない
        run(["fetch", str(study), "hourly", "--stations", "千葉", "銚子"])
    err = capsys.readouterr().err
    assert e.value.code == 1 and "Traceback" not in err
    assert "銚子 の時間値を取得する日の一覧のファイルがありません" in err and f"amamizu candidates {study}" in err

    days = tmp_path / "days.txt"
    days.write_text("2020-01-01\n2020-13-01\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        run(["fetch", str(study), "hourly", "--stations", "千葉", "--dates-file", str(days)])
    assert capsys.readouterr().err == f"エラー: {days} の 2 行目の「2020-13-01」を日付として読めません。日付は 2026-08-13 の形で書いてください。\n"
    with pytest.raises(SystemExit):
        run(["fetch", str(study), "hourly", "--stations", "千葉", "--dates-file", str(tmp_path / "none_{station}.txt")])
    assert "--dates-file のファイル名（" in capsys.readouterr().err
    days.write_bytes("2020-01-01\r\n".encode("utf-16"))      # メモ帳などで「Unicode テキスト」として保存したファイル
    with pytest.raises(SystemExit):
        run(["fetch", str(study), "hourly", "--stations", "千葉", "--dates-file", str(days)])
    assert capsys.readouterr().err == (f"エラー: {days} の文字コードが UTF-8 ではないため、読めません。"
                                       "メモ帳で開き、「名前を付けて保存」で文字コードを UTF-8 にして保存し直してください。\n")


@pytest.mark.parametrize("what, options, message", [
    ("tenmin", [], "fetch tenmin には --start と --end を指定してください"),
    ("tenmin", ["--start", "2026-08-13"], "fetch tenmin には --end を指定してください"),
    ("tenmin", ["--start", "2026/08/13", "--end", "2026-08-15"], "--start の「2026/08/13」を日付として読めません"),
    ("tenmin", ["--start", "2026-08-15", "--end", "2026-08-13"], "--end（2026-08-13）は --start（2026-08-15）以降"),
    ("daily", [], "fetch daily には --year-from と --year-to を指定してください"),
    ("daily", ["--year-from", "2025"], "fetch daily には --year-to を指定してください"),
    ("daily", ["--year-from", "2025", "--year-to", "2020"], "--year-to（2020）は --year-from（2025）以降"),
])
def test_fetch_without_period_shows_the_reason(tmp_path, capsys, what, options, message):
    """10分値・日別値の期間の指定がない・読めないときは、経過（Traceback）なしで理由を表示し、取得を始めない（_jma/ も変えない）。"""
    from amamizu.cli import run
    study = jma_dir(tmp_path) / "study"
    study.mkdir()
    with pytest.raises(SystemExit) as e:
        run(["fetch", str(study), what, "--stations", "千葉", *options])
    err = capsys.readouterr().err
    assert e.value.code == 1 and err.startswith("エラー: ") and message in err
    assert [p.name for p in (tmp_path / "_jma").iterdir()] == ["config.json"]


def test_periods_command_lists_the_same_periods_as_webgui(tmp_path, capsys):
    """amamizu periods は、WebGUI の実測降雨の期間の選択肢と同じ一覧（app.observations）を地点ごとに表示し、--output に TSV を書く。
    取得していない地点・値の種類、重複した地点、取得済みの地点がないときは、経過（Traceback）なしでエラーにし、TSV を書かない。"""
    from amamizu.adapters.jma_store import JmaStore
    from amamizu.app.observations import PERIOD_COLUMNS, fetched_periods, period_label, periods_summary
    from amamizu.cli import run
    store = JmaStore(tmp_path)                      # 試験の studies/ に _jma/ がないので、リポジトリの studies/_jma/ を読む（conftest）
    out = tmp_path / "periods.tsv"
    run(["periods", "tenmin", "--stations", "館山", "千葉", "--studies", str(tmp_path), "--output", str(out)])
    found = fetched_periods(store, "tenmin", ["館山", "千葉"])
    expected = [line for n, ps in found.items() for line in [periods_summary(n, "tenmin", ps), *(f"  {period_label(p)}" for p in ps)]]
    assert capsys.readouterr().out.splitlines() == expected + [f"  {out}"]
    data = out.read_bytes()
    assert data.startswith("﻿".encode("utf-8")) and data.count(b"\n") == data.count(b"\r\n")
    t = pd.read_csv(out, sep="\t", encoding="utf-8-sig", dtype={"開始日": str})
    assert list(t.columns) == PERIOD_COLUMNS and len(t) == sum(len(ps) for ps in found.values())
    assert t.iloc[0].tolist()[:4] == ["館山", found["館山"][0].start, found["館山"][0].end, found["館山"][0].days]

    run(["periods", "hourly", "--studies", str(tmp_path)])       # 地点を省略すると、値の種類を取得済みの全地点
    heads = [x.split("の時間値は")[0] for x in capsys.readouterr().out.splitlines() if not x.startswith(" ")]
    assert heads == store.stations_with("hourly")

    (tmp_path / "empty").mkdir()
    cases = [(["hourly", "--stations", "佐倉"], tmp_path, "佐倉_0916 に時間値がありません"),
             (["tenmin", "--stations", "館山", "館山x"], tmp_path, "地点「館山x」の観測値がありません"),
             (["tenmin", "--stations", "館山", "館山"], tmp_path, "地点 館山 を2回以上指定しています"),
             (["tenmin"], jma_dir(tmp_path / "empty"), "10分値を取得した地点がありません。先に amamizu fetch <検討フォルダ> tenmin")]
    for args, studies, message in cases:
        bad = tmp_path / "bad.tsv"
        with pytest.raises(SystemExit) as e:
            run(["periods", *args, "--studies", str(studies), "--output", str(bad)])
        cap = capsys.readouterr()
        assert e.value.code == 1 and cap.err.startswith("エラー: ") and message in cap.err and "Traceback" not in cap.err
        assert cap.out == "" and not bad.exists()
