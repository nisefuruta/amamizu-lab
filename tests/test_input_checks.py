"""入力の件数・形・値が想定と違うとき、値・行・照合を黙って飛ばしたり埋めたりせず、エラーにするか知らせる。"""
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amamizu.adapters.jma_store import JmaStore
from amamizu.adapters.guideline.json_guideline import JsonGuideline
from amamizu.usecases.storage_study import NO_PUBLISHED, StorageStudy

ROOT = Path(__file__).parents[1]
CHIBA = ROOT / "studies" / "_guideline" / "chiba_2006.json"
ST, BLK = "千葉", "47682"
DAYS = ["2026-08-13", "2026-08-14", "2026-08-15"]


def tenmin(mm=lambda d, k: 0.0, quality="normal") -> pd.DataFrame:
    """DAYS の10分値（8/14 12:10〜13:00 に 10mm×6 コマ）。"""
    rows = []
    for d in DAYS:
        for k in range(1, 145):
            hh, mi = divmod(k * 10, 60)
            rows.append({"日付": d, "時分": f"{hh:02d}:{mi:02d}", "区間終了時刻": f"{d} {hh:02d}:{mi:02d}", "降水量_mm": mm(d, k),
                         "品質": quality})
    return pd.DataFrame(rows)


def rain(d, k):
    return 10.0 if d == DAYS[1] and 72 < k <= 78 else 0.0


def daily(total=60.0, max1h=60.0, days=DAYS) -> pd.DataFrame:
    return pd.DataFrame([{"日付": d, "日合計_日別値_mm": total if d == DAYS[1] else 0.0, "最大1時間_日別値_mm": max1h if d == DAYS[1] else 0.0,
                          "最大10分_日別値_mm": 0.0, "日別値_品質": "normal"} for d in days])


STATION = {"地点": ST, "block_no": BLK, "pref": "千葉県"}


def store_with(tmp_path, **tables: pd.DataFrame) -> JmaStore:
    """試験の studies/_jma/ に、tables（種類 → 行）を確定の値（観測日の1か月より後の取得）として書く。"""
    (tmp_path / "_jma").mkdir(parents=True)
    (tmp_path / "_jma" / "config.json").write_text('{"refresh_months": 1}', encoding="utf-8")
    store = JmaStore(tmp_path)
    for kind, rows in tables.items():
        store.put(kind, STATION, rows, datetime.date(2026, 12, 1), "試験")
    return store


# 観測値の書き込み ------------------------------------------------------------------
def test_blank_rain_with_valued_quality_is_an_error(tmp_path):
    """品質が値のある品質なのに雨量が空欄の行は、0mm や欠測として黙って扱わない。"""
    t = tenmin(rain)
    t.loc[75, "降水量_mm"] = None
    with pytest.raises(ValueError, match=r"千葉 2026-08-13 12:40 の「降水量_mm」（空欄、品質 normal）を雨量として読めません。"):
        store_with(tmp_path, tenmin=t)
    t.loc[75, "品質"] = "missing"                                                    # 欠測の品質なら書く（0mm として計算し、欠測に数える）
    assert store_with(tmp_path / "ok", tenmin=t).tenmin(ST, DAYS[0], DAYS[-1])["降水量_mm"].isna().sum() == 1


# 貯留施設の検証の降雨ごとの表 ------------------------------------------------------
def event(tmp_path, daily_frame) -> pd.DataFrame:
    study = StorageStudy(JsonGuideline(CHIBA), store_with(tmp_path, tenmin=tenmin(rain), daily=daily_frame))
    return study.event_at_stations([ST], DAYS[0], DAYS[-1])


def test_event_needs_daily_values_of_its_days(tmp_path):
    with pytest.raises(ValueError, match="日別値が 2026-08-13〜2026-08-15 のうち 3 日ありません"):
        event(tmp_path, daily(days=["2026-07-01"]))


def test_event_counts_days_without_published_1h(tmp_path):
    """1時間の公表値がない日は数えて表に出力する（毎正時の値を公表値のように黙って出力しない）。"""
    assert event(tmp_path / "a", daily(max1h=75.5)).iloc[0][["1h_mm", NO_PUBLISHED]].tolist() == [75.5, 0]
    r = event(tmp_path / "b", daily(max1h=np.nan)).iloc[0]
    assert r["1h_mm"] == 60.0 and r[NO_PUBLISHED] == 1


def test_event_total_is_checked_against_daily(tmp_path):
    """10分値の合計が日別値と合わず、欠測もなければ止める（長期記録・途中の年と同じ照合）。"""
    with pytest.raises(ValueError, match="10分値の合計 60.0mm が日別値の合計 30.0mm と 1mm を超えて違い、欠測もない"):
        event(tmp_path, daily(total=30.0))


def test_event_reaching_the_last_day_of_records_is_noticed():
    """日別値の最後の日にも雨がある降雨は、その後も続いた可能性を知らせる（総量が小さく出ることを黙らない）。"""
    class Records:
        def daily(self, station, year_from=""):
            return pd.DataFrame({"日付": DAYS, "日合計_日別値_mm": [0.0, 80.0, 120.0], "最大1時間_日別値_mm": [0.0, 20.0, 30.0],
                                 "日別値_品質": "normal"})
    study = StorageStudy(JsonGuideline(CHIBA), Records())
    study.candidates(ST, DAYS[0], DAYS[-1])
    assert any("2026-08-15）に雨があり、その後も降り続いた可能性があります" in n for n in study.notices)


# 降雨強度の検証・公表値との照合 ---------------------------------------------------
def test_return_period_sample_with_missing_year_is_an_error():
    from amamizu.usecases.return_period_study import evaluate

    class Records:
        def stations(self):
            return [{"地点": ST, "種別": "官署", "事象の最大10分_mm": 20.0}]

        def annual_max_10min(self, station, first, last):
            return [(y, None if y == 2019 else 15.0, "missing" if y == 2019 else "normal") for y in range(first, last + 1)]

        def stat_period_start(self, station):
            return 2010, 1
    with pytest.raises(ValueError, match=r"2010〜2025年）に、値のない年があります: 2019年（値なし）"):
        evaluate(JsonGuideline(CHIBA), Records())


def test_published_value_that_cannot_be_read_is_not_compared():
    """公表値が読めない地点・年は「照合できない」に入れ、「超過していない」として黙って比べない。"""
    from amamizu.usecases.published_check import exceedance_sets
    published = pd.DataFrame([{"地点": "銚子", "年": 2001, "最大12時間/(mm)": "///"}])
    events = pd.DataFrame([{"地点": "銚子", "開始": "2001-10-10", "12h_mm": 400.0}])
    res = exceedance_sets(JsonGuideline(CHIBA), published, events, 50, durations=(12,))[12]
    assert res["照合できない"] == {("銚子", 2001)} and res["一致"]
    res = exceedance_sets(JsonGuideline(CHIBA), published.iloc[:0], events, 50, durations=(12,))[12]
    assert res["照合できない"] == {("銚子", 2001)}                                  # 公表値の行がない年も同じ


# 設計の検討の実測降雨・雨水捕捉率 --------------------------------------------------
def test_observed_rain_text_value_is_an_error():
    """数に読めない雨量（O.5 など）を欠測として黙って 0mm にしない。空欄は欠測として記録する。"""
    from amamizu.adapters.design_inputs import observed_from_frame
    t = tenmin(rain).astype({"降水量_mm": object})
    t.loc[75, "降水量_mm"] = "O.5"
    with pytest.raises(ValueError, match=r"2026-08-13 12:40 の降水量_mm「O.5」を数として読めません。"):
        observed_from_frame(t, {})
    t.loc[75, "降水量_mm"] = None
    assert observed_from_frame(t, {}).missing
    with pytest.raises(ValueError, match="observed_rain.start）の「」を日付として読めません"):
        observed_from_frame(t, {"start": ""})


def test_capture_shows_missing_hours():
    from amamizu.usecases.rainfall_tools import capture_table
    table = capture_table(pd.Series([10.0, None, 30.0, None]), 10, 10, 5)
    assert table.iloc[0][["総降雨量 (mm)", "計算した時間数", "欠測の時間数"]].tolist() == [40.0, 2, 2]
    with pytest.raises(ValueError, match="2 行目の値「x」を数として読めません。"):
        capture_table(pd.Series([10.0, "x"]), 10, 10, 5)


# 取得 ------------------------------------------------------------------------
def test_fetched_year_without_rows_is_a_failure(tmp_path, monkeypatch):
    """失敗なしで行が0の年も、取得できなかった年として数える（書かずに黙って次の年へ進まない）。"""
    from amamizu.adapters import jma_fetch

    class Empty:
        def __init__(self, *a, **k):
            self.failures = []

        def collect(self, start, end):
            return pd.DataFrame()
    monkeypatch.setattr(jma_fetch, "JMADailyCollector", Empty)
    monkeypatch.setattr(jma_fetch.JmaFetcher, "station", lambda self, name: {"name": name, "block_no": BLK, "pref": "千葉県"})
    store = store_with(tmp_path)
    f = jma_fetch.JmaFetcher(tmp_path)
    f.daily(ST, 2020, 2020)
    assert [x["対象"] for x in f.failures] == ["2020年の日別値"] and not store.root.joinpath(f"{ST}_{BLK}").exists()


# 原因の違う誤りを、値がない・出力していないものとして扱わない -----------------------------
def test_station_name_with_two_folders_is_not_treated_as_no_values(tmp_path):
    """同じ地点名のフォルダが2つあるときは、値がないものとして扱わない（取得の案内だけを表示しない。取得済みの日を取得し直さない）。"""
    store = store_with(tmp_path, tenmin=tenmin(rain))
    assert store.needed_days("tenmin", STATION, [datetime.date.fromisoformat(DAYS[0])]) == []
    (store.root / f"{ST}_99999").mkdir()
    for call in (lambda: store.tenmin(ST, DAYS[0], DAYS[-1]),
                 lambda: store.needed_days("tenmin", STATION, [datetime.date.fromisoformat(DAYS[0])])):
        with pytest.raises(ValueError, match="地点「千葉」の観測所が 2 つあります"):
            call()
    with pytest.raises(FileNotFoundError, match="地点「銚子」の観測値がありません"):
        store.station_dir("銚子")


def hourly(days=DAYS) -> pd.DataFrame:
    return pd.DataFrame([{"日付": d, "時": h, "降水量_mm": 0.0, "品質": "normal"} for d in days for h in range(1, 25)])


def test_unobserved_time_needs_the_records_used(tmp_path):
    """計算に使った記録がそろっていない期間を、欠測の時間がないものとして示さない。"""
    study = StorageStudy(JsonGuideline(CHIBA), store_with(tmp_path, tenmin=tenmin(rain), daily=daily(), hourly=hourly(DAYS[:1])))
    assert study.unobserved(ST, DAYS[0], DAYS[-1], "10分値") == []
    with pytest.raises(ValueError, match="時間値がそろっていないので、欠測の時間を示せません"):
        study.unobserved(ST, DAYS[0], DAYS[-1], "時間値")


def test_trace_needs_complete_hourly_values(tmp_path):
    """レポートの推移表と図に、日のそろっていない時間値を短い降雨として使わない（長期記録の計算と同じ確かめ）。"""
    study = StorageStudy(JsonGuideline(CHIBA), store_with(tmp_path, daily=daily(), hourly=hourly(DAYS[:1])))
    with pytest.raises(ValueError, match="千葉 2026-08-13〜2026-08-15: 時間値がそろっていません"):
        study.trace(ST, DAYS[0], DAYS[-1], "時間値")
    assert len(study.trace(ST, DAYS[0], DAYS[0], "時間値")[0]) == 144


def test_broken_output_record_is_not_treated_as_no_output(tmp_path):
    """out/conditions.json を読めないときは、出力していないものとして扱わない（前の出力にだけあるファイルが残り、条件と観測値の変化も示されない）。"""
    from amamizu.app.design_runner import read_output_conditions, stale_outputs
    assert read_output_conditions(tmp_path) is None                 # out/ がない
    out = tmp_path / "out"
    out.mkdir()
    assert read_output_conditions(tmp_path) is None                 # conditions.json がない
    for data in (b"{", b"[]", "{}".encode("utf-16")):
        (out / "conditions.json").write_bytes(data)
        with pytest.raises(ValueError, match="conditions.json（前の出力の条件と、出力したファイルの名前の記録）を読めません"):
            read_output_conditions(tmp_path)
        with pytest.raises(ValueError, match="このファイルを削除してから、もう一度出力してください"):
            stale_outputs(out, [])


def test_candidates_command_shows_notices(tmp_path, monkeypatch, capsys):
    """amamizu candidates は、候補を選ぶときの注意（日別値の最後の日に雨がある降雨など）を表示する。"""
    import shutil

    from amamizu import cli
    d = tmp_path / "2026-09_貯留施設の検証"
    d.mkdir()
    shutil.copy(ROOT / "studies" / d.name / "study.json", d / "study.json")

    def candidates(self, station, date_from, date_to):
        self.notices.append(f"{station}: 試験の注意")
        return []
    monkeypatch.setattr(StorageStudy, "candidates", candidates)
    cli.main(["candidates", str(d)])
    assert "注意: 千葉: 試験の注意" in capsys.readouterr().out


def test_published_check_lines_name_what_differs():
    """公表値との照合が一致しないときは、どの地点・年が違うかを表示する（「不一致」だけで終えない）。"""
    from amamizu.cli import published_check_lines
    res = {1: {"公表値で超過": {("千葉", 2019)}, "計算で超過": {("千葉", 2019)}, "照合できない": set(), "一致": True},
           3: {"公表値で超過": {("千葉", 2019), ("銚子", 2001)}, "計算で超過": {("千葉", 2019), ("館山", 1996), ("勝浦", 1980)},
               "照合できない": {("勝浦", 1980)}, "一致": False}}
    assert published_check_lines(res, "1/50") == [
        "公表値（N時間降水量）との照合: {1: '一致', 3: '不一致'}",
        "注意: 3時間の1/50の超過が公表値と一致しません（公表値だけが超過）: 銚子 2001年",
        "注意: 3時間の1/50の超過が公表値と一致しません（計算だけが超過）: 館山 1996年",
        "注意: 3時間の公表値がない（数として読めない）地点・年は照合していません: 勝浦 1980年"]


def test_abort_then_unsaved_rows_are_both_reported(tmp_path, monkeypatch):
    """取得の打ち切りの後に取得済みの行を書けなかったときは、打ち切りと書けなかった理由の両方を、打ち切りとして知らせる。"""
    from jma_rainfall.fetch import FetchAborted

    from amamizu.adapters import jma_fetch
    day = datetime.date(2020, 1, 1)

    class Aborting:
        def __init__(self, station, **kw):
            self.failures = [{"対象": "2020-01-02"}]

        def collect(self, dates, on_day=None):
            on_day(day, [{"日付": day.isoformat(), "時": h, "降水量_mm": 0.5, "品質": "normal"} for h in range(1, 25)])
            raise FetchAborted("失敗が続いたため中止しました。")
    monkeypatch.setattr(jma_fetch, "JMAHourlyCollector", Aborting)
    monkeypatch.setattr(jma_fetch.JmaFetcher, "station", lambda self, name: {"name": name, "block_no": BLK, "pref": "千葉県"})
    store = store_with(tmp_path)
    f = jma_fetch.JmaFetcher(tmp_path)
    with pytest.raises(FetchAborted, match=r"^失敗が続いたため中止しました。$"):      # 書けたときは、打ち切りの文のまま
        f.hourly(ST, [day, datetime.date(2020, 1, 2)])
    assert len(store.hourly(ST)) == 24 and len(f.failures) == 1

    def cannot_put(self, *a, **k):
        raise ValueError("書けない理由")
    monkeypatch.setattr(JmaStore, "put", cannot_put)
    with pytest.raises(FetchAborted, match="失敗が続いたため中止しました。\n  取得済みの分は保存できませんでした: 書けない理由"):
        jma_fetch.JmaFetcher(tmp_path).hourly(ST, [day, datetime.date(2020, 1, 2)])


def test_unreadable_json_names_the_file(tmp_path):
    """手引の定数のファイルと観測値のフォルダの設定を JSON として読めないときは、ファイルを示す。"""
    bad = tmp_path / "基準.json"
    bad.write_text("{ name: 基準,", encoding="utf-8")
    with pytest.raises(ValueError, match="手引の定数のファイル .*基準.json を JSON として読めません"):
        JsonGuideline(bad)
    bad.write_bytes("{}".encode("utf-16"))
    with pytest.raises(ValueError, match="手引の定数のファイル .*基準.json の文字コードが UTF-8 ではありません"):
        JsonGuideline(bad)
    (tmp_path / "_jma").mkdir()
    for text in ("{ refresh_months: 1 }", "[1]"):
        (tmp_path / "_jma" / "config.json").write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="config.json (を JSON として読めません|の refresh_months)"):
            JmaStore(tmp_path)


def test_output_record_does_not_hide_an_unreadable_pond_table(tmp_path):
    """出力の条件の記録を作るときに池の表を読めなければ、池の表なしとして記録せずにエラーにする。"""
    from amamizu.app.design_runner import output_conditions
    assert output_conditions({"pond": {"prismatic": {"area_m2": 600.0, "hwl_m": 2.0}}}, tmp_path) == {
        "conf": {"pond": {"prismatic": {"area_m2": 600.0, "hwl_m": 2.0}}}, "pond": None}
    with pytest.raises(FileNotFoundError, match="入力ファイルがありません"):
        output_conditions({"pond": {"table": {"file": "input/pond.csv"}}}, tmp_path)
