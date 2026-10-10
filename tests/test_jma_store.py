"""気象庁の観測値を地点ごとに1か所で持つ studies/_jma/（JmaStore）。取得日で暫定と確定を分けて書き込むことを確かめる。"""
import datetime
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from amamizu.adapters.jma_store import JmaStore

ROOT = Path(__file__).parents[1]
# 取り込んだ raw/ で、同じ日時の値が取得の時期で品質だけ違った行（準正常値 → 正常値。_jma/ は後の取得を持ち、revisions.tsv に記録）
REVISED = {("勝浦", "daily", "2026-09-21"), ("勝浦", "tenmin", "2026-09-21 21:30"), ("勝浦", "tenmin", "2026-09-21 22:20")}


def test_revisions_are_the_known_quality_changes():
    rev = pd.read_csv(ROOT / "studies" / "_jma" / "revisions.tsv", sep="\t", encoding="utf-8-sig", dtype=str)
    rev = rev[rev["出所"].str.startswith("取り込み ")]                  # 取り込んだ後の取得による訂正は含めない
    assert {(r["地点"], r["種類"], r["日時"]) for _, r in rev.iterrows()} == REVISED


# 書き込みの規則 ------------------------------------------------------------------
def new_store(tmp_path, months=1) -> JmaStore:
    (tmp_path / "_jma").mkdir()
    (tmp_path / "_jma" / "config.json").write_text(json.dumps({"refresh_months": months}), encoding="utf-8")
    return JmaStore(tmp_path)


def test_final_from_the_same_day_or_the_month_end(tmp_path):
    """確定になるのは、観測日の1か月後の同じ日（その月にない日は月末。1月31日なら2月28日）以後に取得した行。年ごとの種類は12月31日から数える。"""
    store = new_store(tmp_path)
    def final(kind, key, observed, fetched):
        return list(store.final(kind, pd.DataFrame({key: observed, "取得日": fetched})))
    assert final("tenmin", "日付", ["2026-01-31"] * 3, ["2026-02-27", "2026-02-28", "2026-03-01"]) == [False, True, True]
    assert final("daily", "日付", ["2026-12-15"] * 2, ["2027-01-14", "2027-01-15"]) == [False, True]
    assert final("annual", "年", [2025, 2025], ["2026-01-30", "2026-01-31"]) == [False, True]


def tenmin_rows(mm: float, quality: str) -> pd.DataFrame:
    return pd.DataFrame([{"日付": "2026-09-21", "時分": "21:30", "区間終了時刻": "2026-09-21 21:30", "降水量_mm": mm, "品質": quality}])


ST = {"地点": "勝浦", "block_no": "47674", "pref": "千葉県"}


def test_provisional_value_is_replaced_and_recorded(tmp_path):
    """観測日から1か月の間に取得した値は暫定。取り直して違えば上書きし、訂正を記録する。"""
    store = new_store(tmp_path)
    store.put("tenmin", ST, tenmin_rows(0.5, "quasi"), datetime.date(2026, 9, 22), "試験")
    r = store.put("tenmin", ST, tenmin_rows(0.5, "normal"), datetime.date(2026, 9, 24), "試験")
    assert (r.replaced, r.conflicts) == (1, [])
    t = store.tenmin("勝浦", "2026-09-21", "2026-09-21")
    assert t is None                                                # 1日の144コマがそろっていないので None
    row = pd.read_csv(store.root / "勝浦_47674" / "tenmin" / "2026.tsv", sep="\t", encoding="utf-8-sig", dtype=str).iloc[0]
    assert (row["品質"], row["取得日"]) == ("normal", "2026-09-24")          # 上書きした値と取得日を書いている
    rev = pd.read_csv(store.root / "revisions.tsv", sep="\t", encoding="utf-8-sig", dtype=str)
    assert rev.iloc[0][["日時", "前の値", "新しい値", "前の取得日"]].tolist() == [
        "2026-09-21 21:30", "品質 quasi", "品質 normal", "2026-09-22"]


def test_final_value_is_not_overwritten(tmp_path):
    """観測日から1か月より後に取得した値は確定。取り直して違えば、書かずに一覧を返す。"""
    store = new_store(tmp_path)
    store.put("tenmin", ST, tenmin_rows(0.5, "normal"), datetime.date(2026, 10, 22), "試験")
    path = store.root / "勝浦_47674" / "tenmin" / "2026.tsv"
    before = path.read_bytes()
    r = store.put("tenmin", ST, tenmin_rows(1.0, "normal"), datetime.date(2026, 11, 1), "試験")
    assert r.replaced == 0 and r.conflicts == ["勝浦 10分値 2026-09-21 21:30: 降水量_mm 0.5（取得日 2026-10-22、確定）→ 降水量_mm 1.0"]
    assert path.read_bytes() == before


def test_rows_other_than_the_conflicts_are_written(tmp_path):
    """確定の値と違う行があっても、ほかの行（新しい行・暫定の行）は書く。"""
    store = new_store(tmp_path)
    store.put("tenmin", ST, tenmin_rows(0.5, "normal"), datetime.date(2026, 10, 22), "試験")
    new = tenmin_rows(1.0, "normal").assign(時分="21:40", 区間終了時刻="2026-09-21 21:40")
    r = store.put("tenmin", ST, pd.concat([tenmin_rows(1.0, "normal"), new]), datetime.date(2026, 11, 1), "試験")
    assert (r.added, r.replaced, len(r.conflicts)) == (1, 0, 1)
    t = pd.read_csv(store.root / "勝浦_47674" / "tenmin" / "2026.tsv", sep="\t", encoding="utf-8-sig", dtype=str)
    assert t[["時分", "降水量_mm"]].values.tolist() == [["21:30", "0.5"], ["21:40", "1.0"]]


def test_put_without_rows_is_an_error(tmp_path):
    store = new_store(tmp_path)
    with pytest.raises(ValueError, match="書く行がありません"):
        store.put("tenmin", ST, tenmin_rows(0.5, "normal").iloc[:0], datetime.date(2026, 11, 1), "試験")


def test_same_value_fetched_later_becomes_final(tmp_path):
    """同じ値を1か月より後に取り直したら、取得日を新しくして確定にする。"""
    store = new_store(tmp_path)
    store.put("tenmin", ST, tenmin_rows(0.5, "normal"), datetime.date(2026, 9, 22), "試験")
    row = pd.read_csv(store.root / "勝浦_47674" / "tenmin" / "2026.tsv", sep="\t", encoding="utf-8-sig", dtype=str).iloc[0]
    assert row["取得日"] == "2026-09-22" and not store.final("tenmin", pd.DataFrame([row])).any()
    store.put("tenmin", ST, tenmin_rows(0.5, "normal"), datetime.date(2026, 11, 1), "試験")
    row = pd.read_csv(store.root / "勝浦_47674" / "tenmin" / "2026.tsv", sep="\t", encoding="utf-8-sig", dtype=str).iloc[0]
    assert row["取得日"] == "2026-11-01" and store.final("tenmin", pd.DataFrame([row])).all()


def test_store_needs_its_config(tmp_path):
    (tmp_path / "_jma").mkdir()
    with pytest.raises(FileNotFoundError, match="refresh_months"):
        JmaStore(tmp_path)
    (tmp_path / "_jma" / "config.json").write_text('{"refresh_months": 1.5}', encoding="utf-8")
    with pytest.raises(ValueError, match="0以上の整数"):
        JmaStore(tmp_path)


def test_store_keeps_one_station_per_name(tmp_path):
    """地点名が同じ別の観測所（福島県と岐阜県の金山）の値は書かない。地点名で読む観測所は1つに決まる。"""
    from amamizu.adapters.jma_store import StationMismatch
    store = new_store(tmp_path)
    kaneyama = lambda block: {"地点": "金山", "block_no": block, "pref": "福島県"}
    store.put("tenmin", kaneyama("1044"), tenmin_rows(0.5, "normal"), datetime.date(2026, 11, 1), "試験")
    with pytest.raises(StationMismatch, match="地点番号 0487 は地点名が同じ別の観測所"):
        store.put("tenmin", kaneyama("0487"), tenmin_rows(0.5, "normal"), datetime.date(2026, 11, 1), "試験")
    assert [p.name for p in store.root.iterdir() if p.is_dir()] == ["金山_1044"]
    assert store.station_dir("金山").name == "金山_1044"


@pytest.mark.parametrize("name", ["/x", "C:/x", "**", "..", "*", "金[山]", "", "金"])
def test_station_name_is_compared_and_not_used_as_a_pattern(tmp_path, name):
    """取得していない地点名（study.json に書いた名前）は、どんな文字でも「観測値がありません」の扱いにする
    （地点名をフォルダを探すパターンにしない。パターンにすると「/x」「C:/x」「**」は探す処理のエラーになる）。"""
    store = new_store(tmp_path)
    store.put("tenmin", {"地点": "金山", "block_no": "1044", "pref": "福島県"}, tenmin_rows(0.5, "normal"), datetime.date(2026, 11, 1), "試験")
    assert store.pref(name) is None
    with pytest.raises(FileNotFoundError, match="の観測値がありません"):
        store.station_dir(name)
    store.check_station(name, "0487")                               # 同じ地点名の観測所はないので、エラーにしない


def test_station_name_with_pattern_characters_is_found(tmp_path):
    """パターンの文字（[ ]）を含む地点名のフォルダも、名前のとおりに見つける。"""
    store = new_store(tmp_path)
    store.put("tenmin", {"地点": "金[山]", "block_no": "1044", "pref": "福島県"}, tenmin_rows(0.5, "normal"), datetime.date(2026, 11, 1), "試験")
    assert store.station_dir("金[山]").name == "金[山]_1044" and store.pref("金[山]") == "福島県"


def test_folder_without_block_no_is_not_a_station(tmp_path):
    """名前に「_」のないフォルダ（手で作ったフォルダなど）は地点のフォルダとして扱わない。地点名と同じ名前でも、地点のフォルダは1つのまま。"""
    store = new_store(tmp_path)
    store.put("tenmin", ST, tenmin_rows(0.5, "normal"), datetime.date(2026, 11, 1), "試験")
    shutil.copytree(store.root / "勝浦_47674", store.root / "勝浦")
    shutil.copytree(store.root / "勝浦_47674", store.root / "backup")
    assert store.station_dir("勝浦").name == "勝浦_47674" and store.pref("勝浦") == "千葉県"
    store.check_station("勝浦", "47674")
    store.check_station("backup", "1")
    assert store.stations_with("tenmin") == ["勝浦"]
    with pytest.raises(FileNotFoundError, match="の観測値がありません"):
        store.station_dir("backup")


def test_pref_of_a_station(tmp_path):
    """都道府県は station.json の pref。station.json か pref がない、同じ地点名のフォルダが2つあるときは None。
    station.json が項目を持つ形でなければ、読めないファイルとしてエラーにする（黙って都道府県なしにしない）。"""
    store = new_store(tmp_path)
    store.put("tenmin", ST, tenmin_rows(0.5, "normal"), datetime.date(2026, 11, 1), "試験")
    info = store.root / "勝浦_47674" / "station.json"
    assert store.pref("勝浦") == "千葉県"
    for text in ("[]", "null", '"千葉県"'):
        info.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="station.json は項目を持つ形"):
            store.pref("勝浦")
    info.write_text('{"地点": "勝浦", "block_no": "47674", "pref": null}', encoding="utf-8")
    assert store.pref("勝浦") is None
    info.unlink()
    assert store.pref("勝浦") is None
    (store.root / "勝浦_9999").mkdir()
    assert store.pref("勝浦") is None


# 検討が使った観測値の記録 ------------------------------------------------------------
def test_fingerprint_and_changes(tmp_path):
    """指紋は値と品質から作り、同じ値を取り直しても変わらない。前の出力の記録と比べて、行の増減・値の変化と訂正を示す。"""
    from amamizu.app.observations import observation_changes, observation_record
    store = new_store(tmp_path)
    day = lambda hm, mm, q: pd.DataFrame([{"日付": "2026-09-21", "時分": hm, "区間終了時刻": f"2026-09-21 {hm}", "降水量_mm": mm, "品質": q}])
    store.put("tenmin", ST, pd.concat([day("21:30", 0.5, "quasi"), day("21:40", 0.0, "insufficient")]), datetime.date(2026, 9, 22), "試験")
    store.reads.clear()
    store.note_read("tenmin", "勝浦", "2026-09-21", "2026-09-21")
    record = observation_record(store) | {"出力日": "2026-09-22"}         # 9/22 に出力した記録
    r = record["範囲"][0]
    assert (r["行数"], r["準正常値"], r["資料不足値"], r["暫定の行"]) == (2, 1, 1, 2)
    store.put("tenmin", ST, day("21:40", 0.0, "insufficient"), datetime.date(2026, 11, 1), "試験")   # 同じ値を取り直した
    assert store.fingerprint("tenmin", "勝浦", "2026-09-21", "2026-09-21")["指紋"] == r["指紋"]
    assert observation_changes({"観測値": record}, tmp_path) == []
    store.put("tenmin", ST, day("21:30", 0.5, "normal"), datetime.date(2026, 9, 24), "試験")          # 暫定の値が直った
    changes = observation_changes({"観測値": record}, tmp_path)
    assert changes[0] == "勝浦 10分値 2026-09-21〜2026-09-21: 値か品質が変わりました"
    assert "訂正 2026-09-21 21:30: 品質 quasi → 品質 normal（取得日 2026-09-24）" in changes[1]
    store.put("tenmin", ST, day("21:50", 1.0, "normal"), datetime.date(2026, 9, 24), "試験")
    assert observation_changes({"観測値": record}, tmp_path)[0] == "勝浦 10分値 2026-09-21〜2026-09-21: 行数が 2 → 3"


def test_reads_are_recorded_per_range(tmp_path):
    """離れた期間を読んだら別々に記録する（使っていない間の行の変化を、使った値の変化としない）。重なる期間は1つにする。"""
    store = new_store(tmp_path)
    for a, b in (("2026-09-01", "2026-09-02"), ("2026-09-20", "2026-09-21"), ("2026-09-02", "2026-09-03")):
        store.note_read("tenmin", "勝浦", a, b)
    assert store.reads[("tenmin", "勝浦")] == [("2026-09-01", "2026-09-03"), ("2026-09-20", "2026-09-21")]


def test_periods_are_runs_of_fetched_days(tmp_path):
    """取得済みの期間は、取得した日が続いている期間（新しい順）。合計雨量と行のそろっていない日の数を持ち、読んだ期間には記録しない。"""
    store = new_store(tmp_path)
    rows = [{"日付": d, "時": h, "降水量_mm": 1.0, "品質": "normal"} for d in ("2026-09-01", "2026-09-02") for h in range(1, 25)]
    rows += [{"日付": "2026-09-05", "時": 1, "降水量_mm": 2.5, "品質": "normal"}, {"日付": "2026-09-05", "時": 2, "降水量_mm": None, "品質": "missing"}]
    store.put("hourly", ST, pd.DataFrame(rows), datetime.date(2026, 10, 22), "試験")
    assert [(p.start, p.end, p.days, p.total_mm, p.short_days) for p in store.periods("hourly", "勝浦")] == [
        ("2026-09-05", "2026-09-05", 1, 2.5, 1), ("2026-09-01", "2026-09-02", 2, 48.0, 0)]
    assert store.reads == {}
    with pytest.raises(FileNotFoundError, match="10分値がありません"):      # 値の種類のファイルがない地点を、空の一覧にしない
        store.periods("tenmin", "勝浦")


def test_fingerprint_without_the_file_is_an_error(tmp_path):
    store = new_store(tmp_path)
    store.put("tenmin", ST, tenmin_rows(0.5, "normal"), datetime.date(2026, 9, 22), "試験")
    with pytest.raises(FileNotFoundError, match="年最大値系列がありません"):
        store.fingerprint("annual_max", "勝浦", "2025", "2026")


def test_annual_max_and_rank(tmp_path):
    """年最大値系列は年ごとの行（値のない年は空欄）、観測史上順位はその時点の表。品質の数は 10分_品質・1時間_品質 から数える。"""
    from amamizu.adapters.jma_store import StationMismatch
    store = new_store(tmp_path)
    rows = pd.DataFrame([{"年": 2025, "10分": 20.5, "10分_品質": "quasi", "1時間": 60.0, "1時間_品質": "normal"},
                         {"年": 2026, "10分": None, "10分_品質": "missing", "1時間": 55.0, "1時間_品質": "insufficient"}])
    store.put("annual_max", ST, rows, datetime.date(2026, 9, 22), "試験")
    t = store.annual_max("勝浦", 2025, 2026)
    assert list(t["年"]) == [2025, 2026] and pd.isna(t["10分"].iat[1]) and list(t["10分_品質"]) == ["quasi", "missing"]
    r = store.fingerprint("annual_max", "勝浦", "2025", "2026")
    assert (r["準正常値"], r["資料不足値"], r["暫定の行"]) == (1, 1, 1)          # 2026年は年が終わる前に取得した（暫定）
    store.put_rank(ST, {"日最大10分間降水量 (mm)": {"統計期間": "2008/12 2026/8"}}, datetime.date(2026, 9, 25), "試験")
    assert store.rank("勝浦") == {"日最大10分間降水量 (mm)": {"統計期間": "2008/12 2026/8"}}
    assert (store.fetched_on("annual_max", "勝浦"), store.fetched_on("rank", "勝浦")) == ("2026-09-22", "2026-09-25")
    assert [(r["種類"], r["期間"]) for r in store.used()] == [("annual_max", ["2025", "2026"]), ("rank", ["", ""])]
    with pytest.raises(StationMismatch, match="地点番号 0487 は地点名が同じ別の観測所"):
        store.put_rank(ST | {"block_no": "0487"}, {}, datetime.date(2026, 9, 22), "試験")
