"""WebGUI（gui）の試験。状態（gui/state.py）とグラフの設定（gui/charts.py）はブラウザなしで、
WebGUI の操作は NiceGUI の user_simulation で確かめる。「保存して出力」の out/ は amamizu study とバイト単位で一致させる。"""
import asyncio
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nicegui")

from nicegui import ui  # noqa: E402
from nicegui.testing.user_simulation import user_simulation  # noqa: E402

from amamizu.adapters.jma_store import JmaStore  # noqa: E402
from amamizu.app.observations import fetched_periods, period_label  # noqa: E402
from amamizu.adapters.study_config import dump_study, read_study  # noqa: E402
from amamizu.app.design_runner import SECTIONS, check_sections, run_study, trial  # noqa: E402
from amamizu.cli import main  # noqa: E402
from amamizu.gui import charts  # noqa: E402
from amamizu.gui.app import root_factory  # noqa: E402
from amamizu.gui.state import (DesignForm, changes, create_design_study, delete_plan, delete_study,  # noqa: E402
                                duplicate_study, new_study_dir, normalized)

CHIBA = Path(__file__).parents[1] / "studies" / "_guideline" / "chiba_2006.json"      # 千葉県の手引の定数

CASES = Path(__file__).parent / "fixtures" / "design_cases"
STUDIES = Path(__file__).parents[1] / "studies"


@pytest.fixture(autouse=True)
def opened(monkeypatch):
    """試験では、フォルダや PDF を実際には開かず、開こうとしたパスを記録する。"""
    from amamizu.gui import app as gui_app
    calls = []
    monkeypatch.setattr(gui_app, "open_path", calls.append)
    return calls


def copy_case(case: str, studies: Path, name: str | None = None) -> Path:
    d = studies / (name or f"2026-09_{case}")
    shutil.copytree(CASES / case, d)
    return d


def copy_for_cli(study: Path, studies: Path) -> Path:
    """検討（out/ を除く）を studies/ にコピーする。amamizu コマンドは別のプロセスで試験の設定が効かないので、
    手引の定数と、貯留施設の検証なら観測値（studies/_jma/ の設定とその地点）も置く。"""
    d = studies / study.name
    shutil.copytree(study, d, ignore=shutil.ignore_patterns("out"))
    shutil.copytree(CHIBA.parent, studies / "_guideline")
    spec = read_study(d).get("observed_rain")
    if spec:
        (studies / "_jma").mkdir()
        shutil.copy2(STUDIES / "_jma" / "config.json", studies / "_jma")
        folder = JmaStore(STUDIES).station_dir(spec["station"])
        shutil.copytree(folder, studies / "_jma" / folder.name)
    return d


# 状態 ------------------------------------------------------------------------
def ordered_studies(tmp_path) -> list[str]:
    """一覧は検証の例（降雨強度の検証、貯留施設の検証の順）を先頭に固定し、利用者が作った検討は作成日の新しい順（題名の順ではない）。"""
    from amamizu.gui.state import list_order
    for name in ("2026-09_貯留施設の検証", "2026-09_降雨強度の検証"):
        (tmp_path / name).mkdir()
        shutil.copy(STUDIES / name / "study.json", tmp_path / name / "study.json")
    old = create_design_study(tmp_path, "あ 先に作った")
    new = create_design_study(tmp_path, "ん 後で作った")
    conf = read_study(new)
    conf["created"] = (datetime.datetime.fromisoformat(read_study(old)["created"]) + datetime.timedelta(minutes=1)).isoformat()
    (new / "study.json").write_bytes(dump_study(conf))
    (tmp_path / "2026-09_壊れた").mkdir()
    (tmp_path / "2026-09_壊れた" / "study.json").write_text("{", encoding="utf-8")
    items = []
    for d in tmp_path.iterdir():
        try:
            items.append((d, read_study(d)))
        except ValueError:
            items.append((d, None))
    order = [d.name for d, c in sorted(items, key=lambda x: list_order(*x))]
    expected = ["2026-09_降雨強度の検証", "2026-09_貯留施設の検証", new.name, old.name, "2026-09_壊れた"]
    assert order == expected
    return expected


def test_list_order_guideline_first_then_newest(tmp_path):
    ordered_studies(tmp_path)


@pytest.mark.asyncio
async def test_home_shows_studies_in_list_order(tmp_path):
    """WebGUI の一覧の行は、list_order の順に並ぶ（「開く」のボタンは行の順に作る）。"""
    expected = ordered_studies(tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open("/")
        await user.should_see("検討の一覧")
        ids = {n: user.find(marker=f"open:{n}").elements.pop().id for n in expected}
    assert sorted(ids, key=ids.get) == expected


def test_unreadable_created_is_shown_in_the_list(tmp_path):
    """study.json の created を日時として読めない検討は、フォルダの作成日時で並べ、読めないことを一覧の行に示す。"""
    from amamizu.gui.app import study_row
    d = create_design_study(tmp_path, "作成日時")
    assert study_row(d)["err"] == ""
    conf = read_study(d)
    conf["created"] = "9月1日"
    (d / "study.json").write_bytes(dump_study(conf))
    row = study_row(d)
    assert "study.json の created「9月1日」を日時として読めない" in row["err"] and row["created"].startswith("作成 ")


def test_new_study_dir_rules(tmp_path):
    today = datetime.date(2026, 9, 23)
    assert new_study_dir(tmp_path, " 調整池A ", today) == tmp_path / "2026-09_調整池A"
    assert new_study_dir(tmp_path, "調整池 A　2期", today) == tmp_path / "2026-09_調整池_A_2期"   # 空白（全角も）は _
    for bad, why in (("", "題名を入力してください"), ("  ", "題名を入力してください"), ("a/b", "フォルダ名に使えない文字"),
                     ("a:b", "フォルダ名に使えない文字"), ("末尾.", "フォルダ名に使えない文字")):
        with pytest.raises(ValueError, match=why):
            new_study_dir(tmp_path, bad, today)
    (tmp_path / "2026-09_調整池A").mkdir()
    with pytest.raises(ValueError, match="同じ名前"):
        new_study_dir(tmp_path, "調整池A", today)


def test_new_design_study_is_valid_and_runs(tmp_path):
    d = create_design_study(tmp_path, "新しい池")
    raw = (d / "study.json").read_bytes()
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")
    f = DesignForm.load(d)
    assert f.conf["title"] == "新しい池" and f.check().ok and not f.dirty


def test_normalized_integers():
    c = {"kind": "design", "tc_min": 10.0, "end_min": 2520.5, "rain": {"hours": 24.0, "guideline": {"return_period": 50.0}}}
    n = normalized(c)
    assert n["tc_min"] == 10 and isinstance(n["tc_min"], int) and n["end_min"] == 2520.5
    assert isinstance(n["rain"]["hours"], int) and isinstance(n["rain"]["guideline"]["return_period"], int)
    assert c["tc_min"] == 10.0                                   # 元は変えない


def test_integer_error_goes_to_its_section(tmp_path):
    f = DesignForm.load(copy_case("guideline", tmp_path))
    f.conf["tc_min"] = 10.5
    f.conf["rain"]["hours"] = 24.5
    chk = f.check()
    assert "整数" in chk.errors["全体"] and "rain.hours" in chk.errors["流入"]
    with pytest.raises(ValueError, match="保存できません"):
        f.save()


def test_sections_are_checked_independently(tmp_path):
    d = copy_case("guideline", tmp_path)
    c = read_study(d)
    c["catchments"][0]["f"] = 1.5                                 # 流入の誤り
    c["pond"]["prismatic"]["area_m2"] = -1                        # 池の誤り
    chk = check_sections(c, d)
    assert {"流入", "池"} <= set(chk.errors) and "放流施設" not in chk.errors
    c["guideline"] = "ない手引"
    chk = check_sections(c, d)
    assert set(chk.errors) == set(SECTIONS) and "全体" in chk.errors["流入"]


def test_dirty_snapshot_and_changes(tmp_path):
    f = DesignForm.load(copy_case("two_tier_pump", tmp_path))
    before = f.snapshot()
    assert not f.dirty
    f.conf["orifices"][0]["width_m"] = 0.08
    f.pond_rows[1]["面積_m2"] = 530.0
    assert f.dirty
    ch = changes(before, f.snapshot())
    assert "放流施設 オリフィス1 幅（orifices[0].width_m）: 0.07 → 0.08" in ch
    assert any(x.startswith("池の表 2 行目 面積_m2") for x in ch)
    assert changes(None, before) == ["（最初の試算）"] and changes(before, before) == ["（変更なし）"]


def test_every_condition_has_a_name():
    """変えた条件の表示は、study.json のどの項目も WebGUI の名前で示す（SCHEMA に項目を足したら名前も足す）。"""
    from amamizu.adapters.study_config import META, SCHEMA
    from amamizu.gui.state import condition_name

    def paths(schema, prefix):
        for k, v in schema.items():
            p = f"{prefix}{k}"
            yield p
            if isinstance(v, dict):
                yield from paths(v, p + ".")
            elif isinstance(v, list) and isinstance(v[0], dict):       # 値のリスト（["num"]）は項目そのものの名前
                yield p + "[0]"
                yield from paths(v[0], p + "[0].")
    missing = [p for p in paths(META | SCHEMA["design"], "") if condition_name(p) == p]
    assert missing == []
    assert condition_name("catchments[1].infiltration.facilities[0].shape.W_m") == (
        "流入 区域2 浸透 施設1 施設の形 幅 W（catchments[1].infiltration.facilities[0].shape.W_m）")
    assert condition_name("unknown.key") == "unknown.key"


def test_table_changes_show_what_is_lost(tmp_path):
    d = copy_case("two_tier_pump", tmp_path)
    f = DesignForm.load(d)
    assert f.table_changes() == []                               # 変えていなければ書かない
    f.pond_rows.append({"水位_m": 14.0, "面積_m2": 900.0, "容量_m3": 2792.5})
    [ch] = f.table_changes()
    assert ch.lost and "行" in ch.lost[0]
    df = pd.read_csv(d / "input" / "pond.csv", encoding="utf-8-sig")
    df["備考"] = "x"
    df.to_csv(d / "input" / "pond.csv", index=False, encoding="utf-8-sig")
    [ch] = f.table_changes()
    assert any("備考" in x for x in ch.lost)


def test_save_writes_csv_and_study_json(tmp_path):
    d = copy_case("two_tier_pump", tmp_path)
    f = DesignForm.load(d)
    f.pond_rows[-1]["面積_m2"] = 830.0
    written = f.save()
    raw = (d / "input" / "pond.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and b"\r\n" in raw and b"830" in raw
    assert (d / "study.json").read_bytes() == dump_study(read_study(d))
    assert d / "study.json" in written and not f.dirty


def test_edits_after_save_reach_the_conditions(tmp_path):
    """保存しても条件の入れ子の辞書は置き換えない。WebGUI の入力欄が持つ辞書（区域の行など）への保存の後の変更が、条件に反映される。"""
    d = copy_case("infiltration", tmp_path)
    f = DesignForm.load(d)
    row, rain = f.conf["catchments"][0], f.conf["rain"]
    f.save()
    row["area_ha"] = 2.5
    rain["hours"] = 12
    assert f.conf["catchments"][0]["area_ha"] == 2.5 and f.conf["rain"]["hours"] == 12 and f.dirty
    f.save()
    assert read_study(d)["catchments"][0]["area_ha"] == 2.5 and read_study(d)["rain"]["hours"] == 12


def test_pond_rows_need_a_level_column():
    """池の表の CSV を WebGUI の表の行にするとき、面積・容量はどちらか一方を省けるが、水位の列がなければエラー（空の行を作らない）。"""
    from amamizu.adapters.design_inputs import pond_rows_of
    spec = {"file": "input/pond.csv"}
    assert pond_rows_of(pd.DataFrame({"水位_m": [0.0, 1.0], "面積_m2": [10.0, None]}), spec) == [
        {"水位_m": 0.0, "面積_m2": 10.0}, {"水位_m": 1.0}]
    with pytest.raises(ValueError, match="水位の列"):
        pond_rows_of(pd.DataFrame({"H": [0.0], "F": [10.0], "V": [0.0]}), spec)


def test_save_blocked_by_unknown_keys_until_removed(tmp_path):
    d = copy_case("guideline", tmp_path)
    c = read_study(d) | {"titel": "書き間違い"}
    (d / "study.json").write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    f = DesignForm.load(d)
    assert f.unknown == ["titel"]
    with pytest.raises(ValueError, match="不明な項目"):
        f.save()
    f.save(remove_unknown=True)
    assert "titel" not in read_study(d) and f.unknown == []


def test_save_keeps_input_with_calculation_errors(tmp_path):
    """「保存」は計算の誤りがあっても保存する（入力の途中を残せる）。study.json として読み直せない書き方の違う値は保存しない。"""
    d = copy_case("guideline", tmp_path)
    f = DesignForm.load(d)
    f.conf.pop("orifice")
    assert not f.check().ok and f.can_save
    f.save()
    assert "orifice" not in read_study(d) and not f.dirty
    assert "orifice" not in DesignForm.load(d).conf                 # 開き直せる
    f.conf["tc_min"] = 10.5
    assert not f.can_save
    with pytest.raises(ValueError, match="値を直す項目"):
        f.save()
    assert read_study(d)["tc_min"] == 10


def test_type_error_is_shown_on_its_tab(tmp_path):
    """型の違う値（数の項目の "1.0" や true）は、その項目のタブに表示する（区域の浸透は「浸透」のタブ）。保存もしない。"""
    f = DesignForm.load(copy_case("guideline", tmp_path))
    f.conf["catchments"][0]["area_ha"] = "1.0"
    f.conf["catchments"][1]["infiltration"] = {"rate_m3s": True}
    errors = f.check().errors
    assert '値を直してください: 流入 区域1 面積（catchments[0].area_ha）、今は "1.0"。数で書く。' == errors["流入"]
    assert "設計浸透量（catchments[1].infiltration.rate_m3s）、今は true。数で書く。" in errors["浸透"]
    assert not f.can_save


@pytest.mark.needs_japanese_font
def test_output_changes_list_what_changed_after_output(tmp_path):
    """out/ の出力の条件（conditions.json）と今の条件（保存していない変更を含む）の違い。池の表の違いも示す。"""
    d = copy_case("two_tier_pump", tmp_path)
    f = DesignForm.load(d)
    assert f.output_changes() is None                              # まだ出力していない
    run_study(f.normalized(), d)
    f.reload_output()
    assert f.output_changes() == []
    f.conf["orifices"][0]["width_m"] = 0.08
    f.pond_rows[1]["面積_m2"] = 530.0
    ch = f.output_changes()
    assert "放流施設 オリフィス1 幅（orifices[0].width_m）: 0.07 → 0.08" in ch
    assert any(x.startswith("池の表 2 行目 面積_m2") for x in ch) and len(ch) == 2
    f.save()                                                       # 保存しても、出力し直すまでは違いを示す
    assert len(DesignForm.load(d).output_changes()) == 2


def test_output_changes_show_a_cell_that_is_not_a_number(tmp_path):
    """池の表に数でない値があるときも、表を「なし」とせずに、その行と列の違いとして示す。"""
    d = copy_case("two_tier_pump", tmp_path)
    run_study(read_study(d), d)
    f = DesignForm.load(d)
    before = f.output["pond"][1]["面積_m2"]                         # 出力の条件の記録の値（小数）
    f.pond_rows[1]["面積_m2"] = "約500"
    assert f.output_changes() == [f"池の表 2 行目 面積_m2: {before} → 約500"]
    f.pond_rows = None                                             # 池の表を読めていないとき
    assert f.output_changes() == [f"池の表: {len(f.output['pond'])} 行 → なし"]


def test_duplicate_study_copies_conditions_but_not_output(tmp_path):
    """複製は study.json（題名と作成日時は新しいもの）と input/ をコピーし、out/ はコピーしない。"""
    d = copy_case("two_tier_pump", tmp_path)
    (d / "out").mkdir()
    (d / "out" / "route.csv").write_text("x", encoding="utf-8")
    new = duplicate_study(tmp_path, d.name, "ケース2 オリフィス")
    assert new.name.endswith("_ケース2_オリフィス") and not (new / "out").exists()
    assert (new / "input" / "pond.csv").read_bytes() == (d / "input" / "pond.csv").read_bytes()
    a, b = read_study(d), read_study(new)
    assert b["title"] == "ケース2 オリフィス" and b.get("created") != a.get("created")
    assert {k: v for k, v in b.items() if k not in ("title", "created")} == {
        k: v for k, v in a.items() if k not in ("title", "created")}
    assert (new / "study.json").read_bytes() == dump_study(b)
    with pytest.raises(ValueError, match="同じ名前のフォルダ"):
        duplicate_study(tmp_path, d.name, "ケース2 オリフィス")
    with pytest.raises(ValueError, match="使えない文字"):
        duplicate_study(tmp_path, d.name, "a/b")
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "study.json").write_text('{"kind": "storage"}', encoding="utf-8")
    with pytest.raises(ValueError, match="複製できるのは設計の検討だけ"):
        duplicate_study(tmp_path, "storage", "x")


# グラフ ----------------------------------------------------------------------
def test_time_charts_share_margins():
    """流量（降雨つき）と水位のグラフは、描画の領域の左右の余白が同じ（上下に並べて同じ時刻が同じ横の位置になる）。"""
    t = np.arange(0, 300, 10)
    flows = charts.flows(t, {"流入量": np.ones(30)}, 0.05, np.ones(28), 10)
    level = charts.level(t, {"水位": np.ones(30)}, [])
    for g in [*flows["grid"], level["grid"]]:
        assert (g["left"], g["right"]) == (charts.GRID_LEFT, charts.GRID_RIGHT)
    assert flows["xAxis"][-1]["max"] == level["xAxis"]["max"]


def test_charts_keep_legend_outside_and_ascii_units():
    t = np.arange(0, 300, 10)
    opts = [charts.rain_bars(np.ones(6), 10, "計画降雨"),
            charts.flows(t, {"流入量": np.ones(30), "放流量": np.ones(30)}, 0.05, np.ones(28), 10),
            charts.level(t, {"水位": np.ones(30)}, [("満水位", 1.0, charts.RED)]),
            charts.curve([0, 1], [0, 1], "容量 m^3", [], "水位と容量")]
    for o in opts:
        assert o["legend"]["bottom"] == 0
        grids = o["grid"] if isinstance(o["grid"], list) else [o["grid"]]
        assert grids[-1]["bottom"] >= 72                          # 凡例と軸の名前が重ならない余白
        text = json.dumps(o, ensure_ascii=False)
        assert not set(text) & set("²³㎡㎥")
    assert len(opts[1]["grid"]) == 2 and opts[1]["yAxis"][0]["inverse"]


def test_rain_and_inflow_share_time_axis():
    """降雨（24時間で終わる）と流入量（洪水到達時間だけ後ろへ続く）は、同じ横軸の範囲で描く。"""
    rain = np.ones(144)
    o = charts.flows(np.arange(146) * 10, {"流入量": np.ones(146)}, rain=rain, tc=10)
    (a0, a1) = o["xAxis"]
    assert (a0["min"], a0["max"], a0["interval"]) == (a1["min"], a1["max"], a1["interval"])
    assert o["axisPointer"]["link"] == [{"xAxisIndex": "all"}]


# WebGUI の操作 --------------------------------------------------------------------
async def wait_for(user, text, seconds=20.0):
    for _ in range(int(seconds / 0.2)):
        try:
            await user.should_see(text, retries=1)
            return
        except AssertionError:
            await asyncio.sleep(0.2)
    await user.should_see(text)


async def wait_until(check, seconds=20.0):
    """check() が真になるまで待つ（文字の表示では確かめられない変化を待つ）。"""
    for _ in range(int(seconds / 0.2)):
        if check():
            return
        await asyncio.sleep(0.2)
    assert check()


def inside(user, box, *, text=None, marker=None):
    """印 box の要素の中に、text を含む文字か印 marker の要素があるか。シミュレーションは閉じたメニューや折りたたみの中も
    表示しているとみなし、クリックでも開かないので、中にあることは表示ではなく要素の親子で確かめる。"""
    [outer] = user.find(marker=box).elements
    inner = list(outer.descendants())
    if marker is not None:
        return any(e in inner for e in user.find(marker=marker).elements)
    return any(text in str(getattr(e, "text", "")) for e in inner)


def station_options(user):
    """実測降雨の地点の選択肢。"""
    return user.find(marker="observed_station").elements.pop().options


@pytest.mark.asyncio
async def test_home_lists_and_creates(tmp_path):
    copy_case("guideline", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open("/")
        await user.should_see("手引の式・柱状の池")
        user.find("貯留追跡計算を行う").click()
        user.find(marker="new_title").type("画面の池")
        await user.should_see("作るフォルダ")
        user.find(marker="create").click()
        await wait_for(user, "out/: まだ出力していません")             # 新しい検討の画面（題名は入力欄にもあるので待つ目印にしない）
    assert list(tmp_path.glob("*_画面の池/study.json"))


@pytest.mark.asyncio
async def test_storage_tables_show_purpose(tmp_path):
    """貯留施設の検証の表は STORAGE_TABLES の順（名前の順ではない）に並び、選んだ表の目的を表示する。"""
    from amamizu.app.viewer import storage_table_notes
    d = tmp_path / "2026-09_貯留施設の検証"
    (d / "out").mkdir(parents=True)
    shutil.copy(STUDIES / d.name / "study.json", d / "study.json")
    for name in ("daily_missing.tsv", "long_events.tsv"):
        (d / "out" / name).write_text("地点\n千葉\n", encoding="utf-8-sig")
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("long_events.tsv（1 行）")
        notes = storage_table_notes(read_study(d), d)
        await user.should_see(notes["long_events.tsv"][1])
        assert notes["event_20260813.tsv"][1].startswith("千葉豪雨（2026年8月13〜15日）の降雨を")   # どの降雨かを study.json から書く
        assert notes["event_20260920.tsv"][0] == "台風25号の各地点"


@pytest.mark.asyncio
async def test_viewer_shows_conditions_changed_after_output(tmp_path):
    """見るだけの検討も、out/ の結果を作った条件（conditions.json）と今の study.json が違えば、変えた条件を示す
    （前の条件の表とレポートを、今の条件の結果として黙って見せない）。同じなら何も示さない。"""
    d = tmp_path / "2026-09_貯留施設の検証"
    (d / "out").mkdir(parents=True)
    shutil.copy(STUDIES / d.name / "study.json", d / "study.json")
    conf = read_study(d)
    (d / "out" / "long_events.tsv").write_text("地点\n千葉\n", encoding="utf-8-sig")
    (d / "out" / "conditions.json").write_text(json.dumps({"conf": conf, "pond": None, "files": ["long_events.tsv"]}),
                                               encoding="utf-8")
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("long_events.tsv（1 行）")
        await user.should_not_see(marker="output_changes")
    old = json.loads(json.dumps(conf))
    old["conditions"]["return_period"] = 10
    (d / "out" / "conditions.json").write_text(json.dumps({"conf": old, "pond": None, "files": ["long_events.tsv"]}),
                                               encoding="utf-8")
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("条件が変更されています（1 件）")
        assert inside(user, "output_changes", text="conditions.return_period: 10 → 50")


@pytest.mark.asyncio
async def test_report_draws_mermaid_charts(tmp_path):
    """out/report.md の図（Mermaid）は、図として描く（コードのまま表示しない）。"""
    d = tmp_path / "2026-09_貯留施設の検証"
    (d / "out").mkdir(parents=True)
    rep = d / "out" / "report.md"
    rep.write_text("# レポート\n\n```mermaid\nxychart-beta\n    bar [1, 2]\n```\n", encoding="utf-8")
    shutil.copy(STUDIES / d.name / "study.json", d / "study.json")
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_not_see(str(rep))                                  # 保存先のパスは表示しない
        md =[e for e in user.find(ui.markdown).elements if "xychart-beta" in e.content]
        assert md and "mermaid" in md[0].extras


def test_ctrl_c_ends_without_traceback(tmp_path):
    """Ctrl+C（SIGINT）で止めると、サーバの終了処理のあとトレースバックを表示せずに終わる。"""
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    script = tmp_path / "run.py"
    script.write_text(
        "import signal, threading\n"
        "from nicegui import app\n"
        "app.on_startup(lambda: threading.Timer(1, signal.raise_signal, args=(signal.SIGINT,)).start())\n"
        "app.on_shutdown(lambda: print('終了処理', flush=True))\n"
        "from amamizu.cli import main\n"
        f"main(['gui', '--studies', r'{STUDIES}', '--port', '{port}', '--no-browser'])\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(script)], capture_output=True, timeout=60,
                       env={**{k: v for k, v in os.environ.items() if not k.startswith(("NICEGUI", "PYTEST"))},
                            "PYTHONIOENCODING": "utf-8"})   # pytest の中の試験用の設定を引き継がない
    out, err = r.stdout.decode("utf-8"), r.stderr.decode("utf-8")
    assert r.returncode == 0, err
    assert "Traceback" not in err
    assert out.index("終了処理") < out.index("終了しました。")


@pytest.mark.asyncio
async def test_input_error_disables_run_and_shows_on_tab(tmp_path):
    copy_case("guideline", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open("/study/2026-09_guideline")
        await user.should_see("保存済み")
        user.find(marker="tc").clear().type("10.5")
        await wait_for(user, "洪水到達時間 tc（tc_min）、今は 10.5。整数で書く")
        await user.should_see("⚠ 全体")
        await user.should_see("未保存の変更あり")
        trial = user.find(marker="trial").elements.pop()
        assert not trial.enabled
        user.find(marker="tc").clear().type("10")
        await wait_until(lambda: trial.enabled)
        await user.should_not_see("⚠ 全体")                          # タブの印が消える


@pytest.mark.asyncio
async def test_trial_compare_and_restore(tmp_path):
    d = copy_case("two_tier_pump", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        user.find(marker="trial").click()
        await wait_for(user, "試算の結果（ファイルには書いていません）")
        await user.should_see("許容放流量 0.06 m^3/s を超えました")
        user.find(marker="allowable").clear().type("0.07")
        await asyncio.sleep(0.6)
        user.find(marker="trial").click()
        await wait_for(user, "No2")
        [table] = user.find(ui.table).elements
        assert [r["changes"] for r in table.rows] == ["（最初の試算）", "判定と実行 許容放流量（allowable_q_m3s）: 0.06 → 0.07"]
        assert [r["judge"] for r in table.rows] == ["N.G", "O.K"]
        pick = user.find(marker="restore_pick").elements.pop()
        with user:
            pick.value = 1
        restore = user.find(marker="restore").elements.pop()
        user.find(marker="restore").click()
        await wait_for(user, "No1 の条件を画面に読み戻しました")
        # 読み戻しは入力の欄だけを作り直し、押したボタン（試算の比較）は消さない。消すと、その処理の続きの ui.notify が
        # 画面を見つけられずに失敗する（実際の WebGUI で起きた。試験の仕組みはボタンを持ち続けるので、消えたかどうかで確かめる）
        assert not restore.is_deleted
        assert user.find(marker="allowable").elements.pop().value == 0.06
    assert not (d / "out").exists()                               # 試算はファイルに書かない


@pytest.mark.asyncio
@pytest.mark.needs_japanese_font
async def test_switch_observed_to_rain_presets_station_district(tmp_path):
    """実測降雨から計画降雨に切り替えると、地区の初期値は実測降雨の地点が属する地区（成田 → 我孫子）。千葉に決め打ちしない。"""
    d = tmp_path / "2026-09_千葉豪雨_成田"
    shutil.copytree(STUDIES / d.name, d, ignore=shutil.ignore_patterns("out"))    # 10分値はリポジトリの studies/_jma/ を読む
    assert DesignForm.load(d).observed_district() == "我孫子"
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("保存済み")
        # 実測降雨のときは、試算の左に使う降雨記録を示す
        assert user.find(marker="inflow_source_button").elements.pop().text == "降雨記録 成田 10分値 2026-08-13〜2026-08-15"
        toggle = user.find(marker="inflow_source").elements.pop()
        with user:
            toggle.value = "rain"
        await wait_for(user, "手引 我孫子地区 1/50")
        btn = user.find(marker="inflow_source_button").elements.pop()
        assert btn.text == "計画降雨"                     # 計画降雨のときは計画降雨の画面を開くボタン
        user.find(marker="save").click()
        await wait_for(user, "保存して out/ に出力しました", 60)
    assert read_study(d)["rain"]["guideline"]["district"] == "我孫子"


@pytest.mark.asyncio
async def test_observed_choices_come_from_store_and_nothing_is_preselected(tmp_path):
    """流入を実測降雨に切り替えても、値の種類・地点を自動では選ばない。地点の選択肢は studies/_jma/ に取得済みの地点だけで、
    取得のコマンドを示す（input/ の池の表などのファイルを選択肢にしない）。ハイドログラフに切り替えたときも、input/ の最初の CSV を選ばない。"""
    d = copy_case("hydrograph", tmp_path)                       # input/ に池の表 pond.csv がある
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("保存済み")
        toggle = user.find(marker="inflow_source").elements.pop()
        with user:
            toggle.value = "observed_rain"
        await wait_for(user, "実測降雨は、studies/_jma/ に取得済みの地点から選びます")
        await user.should_see("値の種類（10分値か時間値）を選んでください")
        [station] = user.find(marker="observed_station").elements
        assert station.value is None and station.options == {}
        with user:
            user.find(marker="observed_series").elements.pop().value = "tenmin"
        await wait_until(lambda: station_options(user))               # 値の種類を選ぶと地点の選択肢が入る
        [station] = user.find(marker="observed_station").elements
        assert station.value is None and list(station.options) == JmaStore(STUDIES).stations_with("tenmin")
        assert station.options["館山"] == "館山（千葉県）"              # 選ぶ値は地点名、表示は都道府県付き
        assert not any(".csv" in str(o) for o in station.options)
        toggle = user.find(marker="inflow_source").elements.pop()
        with user:
            toggle.value = "inflow_csv"
        await wait_for(user, "ハイドログラフの CSV（input/）")          # 切り替えの選択肢と同じ名前なので、欄の名前を待つ
        [csv] = [e for e in user.find(ui.select).elements if "ハイドログラフの CSV" in (e.props.get("label") or "")]
        assert csv.value is None


@pytest.mark.asyncio
async def test_observed_without_fetched_stations_says_to_fetch(tmp_path):
    """studies/_jma/ に値の種類の地点がなければ、地点を選べないことと取得のコマンドを示す。"""
    (tmp_path / "_jma").mkdir()
    (tmp_path / "_jma" / "config.json").write_text('{"refresh_months": 1}', encoding="utf-8")
    d = copy_case("guideline", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("保存済み")
        with user:
            user.find(marker="inflow_source").elements.pop().value = "observed_rain"
        await wait_for(user, "amamizu fetch <検討フォルダ> tenmin")
        for series, kind in (("tenmin", "10分値"), ("hourly", "時間値")):
            with user:
                user.find(marker="observed_series").elements.pop().value = series
            await wait_for(user, f"studies/_jma/ に{kind}を取得した地点がありません")
            assert station_options(user) == {}


def test_district_term_from_guideline(tmp_path):
    """区域の呼び方が違う手引（district_term が「地域」）では、計画降雨の説明にその語を使う（「地区」と決め打ちしない）。"""
    from amamizu.adapters.guideline.json_guideline import JsonGuideline
    from amamizu.app.rain_runner import make_rain
    src = json.loads((Path(__file__).parent / "fixtures" / "guideline_minimal.json").read_text(encoding="utf-8"))
    path = tmp_path / "other.json"
    path.write_text(json.dumps(src | {"district_term": "地域"}, ensure_ascii=False), encoding="utf-8")
    g = JsonGuideline(path)
    d0 = g.districts()[0]
    rp = int(next(iter(src["formulas"][d0])))
    r = make_rain({"district": d0, "return_period": rp, "shape": "rear", "guideline": str(path)}, tmp_path, tmp_path)
    assert f"手引 {d0}地域 1/{rp}" in r.label and "地区" not in r.label


def test_rain_reads_relative_guideline_from_study_dir(tmp_path, monkeypatch):
    """study.json の guideline が .json の相対パスでも、降雨のページは検討のフォルダから読んで計画降雨を作る（今のフォルダによらない）。"""
    from amamizu.app.rain_runner import make_rain
    d = copy_case("guideline", tmp_path)
    shutil.copy2(CHIBA, d / "custom.json")
    monkeypatch.chdir(tmp_path.parent)
    spec = {"district": "千葉", "return_period": 50, "shape": "rear", "guideline": "custom.json"}
    assert make_rain(spec, tmp_path, d).label.startswith("手引 千葉")
    with pytest.raises(FileNotFoundError):
        make_rain(spec, tmp_path, tmp_path)


def test_duplicate_study_rewrites_relative_guideline(tmp_path):
    """複製は、guideline の .json の相対パスを複製から見たパスに直して写す（複製から手引の定数を読める）。"""
    from amamizu.adapters.guideline.json_guideline import JsonGuideline
    d = copy_case("guideline", tmp_path)
    shutil.copy2(CHIBA, d / "custom.json")
    conf = read_study(d) | {"guideline": "custom.json"}
    (d / "study.json").write_bytes(dump_study(conf))
    new = duplicate_study(tmp_path, d.name, "複製")
    assert read_study(new)["guideline"] == f"../{d.name}/custom.json"
    assert JsonGuideline.of_study(read_study(new), new) is not None


def test_duplicate_study_keeps_guideline_copied_in_input(tmp_path):
    """guideline が input/ の中の .json なら、複製は input/ ごとコピーしたほうを指す（元の検討を削除しても計算できる）。"""
    from amamizu.adapters.guideline.json_guideline import JsonGuideline
    d = copy_case("guideline", tmp_path)
    (d / "input").mkdir()
    shutil.copy2(CHIBA, d / "input" / "custom.json")
    (d / "study.json").write_bytes(dump_study(read_study(d) | {"guideline": "input/custom.json"}))
    new = duplicate_study(tmp_path, d.name, "複製")
    assert read_study(new)["guideline"] == "input/custom.json"
    shutil.rmtree(d)
    assert JsonGuideline.of_study(read_study(new), new) is not None


def test_default_district_comes_from_guideline(tmp_path):
    """地区の初期値と呼び方は手引の JSON による（他県の基準で千葉を表示しない）。JSON になければ最初の地区と「地区」。"""
    from amamizu.adapters.guideline.json_guideline import JsonGuideline
    g = JsonGuideline(Path(__file__).parent / "fixtures" / "guideline_minimal.json")
    assert g.default_district() == g.districts()[0] and g.district_term() == "地区"
    assert JsonGuideline(CHIBA).default_district() == "千葉"
    d = create_design_study(tmp_path, "他県の池")
    conf = read_study(d)
    conf["guideline"] = str(Path(__file__).parent / "fixtures" / "guideline_minimal.json")
    (d / "study.json").write_bytes(dump_study(conf))
    assert DesignForm.load(d).default_district() == g.districts()[0]


@pytest.mark.asyncio
@pytest.mark.needs_japanese_font
async def test_switch_inflow_to_observed_rain(tmp_path):
    """流入を実測降雨に切り替え、値の種類・地点・期間を選ぶ（studies/_jma/ の値）。区域はそのまま使う。試算に欠測の時間が表示される。"""
    d = copy_case("guideline", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("保存済み")
        with user:
            user.find(marker="inflow_source").elements.pop().value = "observed_rain"
        await wait_for(user, "値の種類（10分値か時間値）を選んでください")
        with user:
            user.find(marker="observed_series").elements.pop().value = "tenmin"
        await wait_until(lambda: station_options(user))               # 値の種類を選ぶと地点の選択肢が入る
        with user:
            user.find(marker="observed_station").elements.pop().value = "館山"
        await wait_for(user, "館山の10分値は")
        user.find(marker="observed_start").type("2026-02-23")
        user.find(marker="observed_end").type("2026-02-25")
        await wait_for(user, "欠測の時間（0 mm として計算）: 2026-02-24 05:50〜06:40")
        user.find(marker="observed_start").clear().type("2026-02-24")
        await wait_for(user, "実測降雨: 館山 10分値 2026-02-24〜2026-02-25")
        user.find(marker="trial").click()
        await wait_for(user, "0 mm として計算しました", 60)
        user.find(marker="save").click()
        await wait_for(user, "保存して out/ に出力しました", 60)
    conf = read_study(d)
    assert conf["observed_rain"] == {"station": "館山", "series": "tenmin", "start": "2026-02-24", "end": "2026-02-25"}
    assert "rain" not in conf and conf["catchments"]
    assert (d / "out" / "rain.csv").read_text(encoding="utf-8-sig").splitlines()[1].startswith("2026-02-24 00:00〜00:10")


@pytest.mark.asyncio
@pytest.mark.needs_japanese_font
@pytest.mark.parametrize("series, station", [("tenmin", "館山"), ("hourly", "千葉")])
async def test_observed_period_is_chosen_from_fetched_periods(tmp_path, series, station):
    """地点を選ぶと、取得済みの期間（取得した日が続いている期間）の一覧が表示される。選ぶまでは開始日・終了日を選ぶよう示し、
    選ぶと開始日・終了日が入る。study.json には値の種類・地点・開始日・終了日の4つだけを書く。"""
    period = JmaStore(STUDIES).periods(series, station)[1]
    d = copy_case("guideline", tmp_path)                        # 観測値はリポジトリの studies/_jma/ を読む（conftest）
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("保存済み")
        with user:
            user.find(marker="inflow_source").elements.pop().value = "observed_rain"
        await wait_for(user, "値の種類（10分値か時間値）を選んでください")
        with user:
            user.find(marker="observed_series").elements.pop().value = series
        await wait_until(lambda: station_options(user))               # 値の種類を選ぶと地点の選択肢が入る
        with user:
            user.find(marker="observed_station").elements.pop().value = station
        await wait_for(user, "取得済みの期間から選ぶか、開始日・終了日を入力してください")
        await user.should_see("実測降雨の開始日・終了日を選んでください")
        [start_input] = user.find(marker="observed_start").elements
        [cal] = [e for e in start_input.descendants() if isinstance(e, ui.date)]
        assert f"{period.start.replace('-', '/')}" in cal._props["options"]
        with user:
            cal.value = period.start
        await wait_for(user, "実測降雨の終了日を選んでください")
        assert user.find(marker="observed_start").elements.pop().value == period.start
        [choice] = user.find(marker="observed_period").elements
        found = fetched_periods(JmaStore(STUDIES), series, [station])[station]      # amamizu periods と同じ一覧
        assert choice.value is None and choice.options == {f"{p.start}〜{p.end}": period_label(p) for p in found}
        with user:
            choice.value = f"{period.start}〜{period.end}"
        await wait_for(user, f"実測降雨: {station}")
        assert user.find(marker="observed_start").elements.pop().value == period.start
        assert user.find(marker="observed_end").elements.pop().value == period.end
        user.find(marker="save").click()
        await wait_for(user, "保存して out/ に出力しました", 60)
    assert read_study(d)["observed_rain"] == {"series": series, "station": station, "start": period.start, "end": period.end}


@pytest.mark.asyncio
@pytest.mark.needs_japanese_font
async def test_infiltration_by_zone(tmp_path):
    """浸透のタブで「区域ごと」にすると、区域名の見出しの下に区域ごとの欄が出る。区域2を浸透強度にして保存すると、
    その区域の infiltration に書き、最上位の infiltration は消す。区域を足すと欄も増える。「区域全体で1つ」に戻すと区域の浸透は消す。"""
    d = copy_case("infiltration", tmp_path)                    # 最上位の infiltration（施設）がある例
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        user.find("区域を追加").click()
        scope = next(iter(user.find(marker="inf_scope").elements))
        scope.set_value("zones")
        await user.should_see("区域2（1 ha）")
        kinds = sorted((e for e in user.find(kind=ui.toggle).elements if "intensity" in e.options), key=lambda e: e.id)
        assert len(kinds) == 2 and all(k.value == "none" for k in kinds)
        kinds[1].set_value("intensity")
        user.find(marker="save").click()
        await wait_for(user, "保存して out/ に出力しました", 60)
        c = read_study(d)
        assert "infiltration" not in c and "infiltration" not in c["catchments"][0]
        assert c["catchments"][1]["infiltration"] == {"intensity_mmh": 10}
        head = (d / "out" / "route.csv").read_text(encoding="utf-8-sig").splitlines()[0]
        assert "区域2の浸透後流入量(m3/s)" in head
        scope.set_value("whole")
        user.find(marker="save").click()
        for _ in range(300):                    # 前の保存の通知が残っているので、study.json が変わるのを待つ
            c = read_study(d)
            if "infiltration" in c:
                break
            await asyncio.sleep(0.2)
    assert c["infiltration"] == {"rate_m3s": 0.001} and all("infiltration" not in r for r in c["catchments"])


@pytest.mark.asyncio
async def test_infiltration_factors_by_name(tmp_path):
    """浸透施設の影響係数は、目づまり・地下水位・安全率 α の欄に分けて入力する。施設を追加すると 0.9・0.9・1.0 が入る。
    欄の近くに技術指針の値と、選んだ手引の定数の文（千葉県の手引の解説 p.32・p.33）を表示する。空隙の欄の読めない入力は、
    項目の名前を付けて「浸透」のタブに示し、「保存」を押せなくする。"""
    d = copy_case("infiltration", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("雨水浸透施設技術指針［案］調査・計画編 p.47・p.55")
        await user.should_see("解説 p.32: 影響係数 C＝K1（目詰まり）×K2（地下水位）×α（安全率）")
        await user.should_see("解説 p.33: 砕石空隙貯留浸透施設")
        def fields(kind, label):
            return sorted((e for e in user.find(kind=kind).elements if str(e.props.get("label", "")).startswith(label)),
                          key=lambda e: e.id)

        def safety():
            return [e.value for e in fields(ui.number, "安全率 α")]
        assert safety() == [1.0, 0.1]
        user.find("施設を追加").click()
        await wait_for(user, "浸透施設（区域全体）: 3 種類")               # 変更は 0.4 秒ごとに画面へ反映する
        assert safety() == [1.0, 0.1, 1.0]
        ground = fields(ui.number, "地下水位")
        ground[0].set_value(1.0)
        voids = fields(ui.input, "空隙")
        voids[0].set_value("1×50×2")
        await wait_for(user, "空隙（infiltration.facilities[0].void_parts）、今は \"1×50×2\"。リストで書く")
        save = next(iter(user.find(marker="save_only").elements))
        assert not save.enabled
        voids[0].set_value("1×50")
        for _ in range(20):
            if save.enabled:
                break
            await asyncio.sleep(0.1)
        user.find(marker="save_only").click()
        for _ in range(100):
            c = read_study(d)
            if len(c["infiltration"]["facilities"]) == 3:
                break
            await asyncio.sleep(0.2)
    facs = c["infiltration"]["facilities"]
    assert facs[0]["factors"] == {"clogging": 0.9, "groundwater": 1.0, "safety": 1.0}
    assert facs[0]["void_parts"] == [{"volume_m3": 1.0, "porosity_pct": 50.0}]
    assert facs[2]["factors"] == {"clogging": 0.9, "groundwater": 0.9, "safety": 1.0}


@pytest.mark.asyncio
@pytest.mark.needs_japanese_font
async def test_save_after_confirm_dialog(tmp_path, opened):
    """確認のダイアログ（不明な項目を削除して保存する）から保存しても、画面の文脈で最後まで進み、結果の表示と PDF を開くまで行う。"""
    d = copy_case("guideline", tmp_path)
    c = read_study(d) | {"titel": "書き間違い"}
    (d / "study.json").write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        user.find(marker="save").click()
        await wait_for(user, "不明な項目 titel を削除して保存しますか")
        user.find(marker="confirm_save").click()
        await wait_for(user, "保存して out/ に出力しました", 60)
    assert "titel" not in read_study(d)
    assert opened == [d / "out" / "design_summary.pdf"]


@pytest.mark.asyncio
async def test_switch_to_pond_table_with_other_columns(tmp_path):
    """池を「表」に切り替えたとき、input/pond.csv に水位の列がなければ（列名が違う）、例の2行を表示する（空の行にしない）。"""
    d = copy_case("guideline", tmp_path)
    (d / "input").mkdir(exist_ok=True)
    pd.DataFrame({"H": [0, 1, 2], "F": [100, 100, 100], "V": [0, 100, 200]}).to_csv(d / "input" / "pond.csv", index=False)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        toggle = next(e for e in user.find(kind=ui.toggle).elements if "table" in e.options)
        toggle.set_value("table")
        await user.should_see("池の表（上の行から順に")
        levels = sorted(user.find(kind=ui.number, content="水位 m").elements, key=lambda e: e.id)
        assert [e.value for e in levels] == [0.0, 2.0]


@pytest.mark.asyncio
async def test_import_pond_with_other_columns_is_an_error(tmp_path):
    """池の表の「読み込む」で、選んだ CSV に水位の列がなければエラーを表示し、表を置き換えない（列の並びから推測しない）。"""
    d = copy_case("hydrograph", tmp_path)
    pd.DataFrame({"A": [0, 1, 2], "B": [100, 100, 100], "C": [0, 100, 200]}).to_csv(d / "input" / "other.csv", index=False)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("保存済み")
        before = [e.value for e in sorted(user.find(kind=ui.number, content="水位 m").elements, key=lambda e: e.id)]
        sel = [e for e in user.find(ui.select).elements if (e.props.get("label") or "") == "CSV から読み込む"][0]
        with user:
            sel.value = "input/other.csv"
        user.find(kind=ui.button, content="読み込む").click()
        await asyncio.sleep(0.2)
        assert any("池の表に水位の列" in m for m in user.notify.messages)
        await asyncio.sleep(0.6)                                     # 状態の表示は 0.4 秒ごとに更新する
        await user.should_not_see("未保存の変更あり")
        assert [e.value for e in sorted(user.find(kind=ui.number, content="水位 m").elements, key=lambda e: e.id)] == before


@pytest.mark.asyncio
async def test_result_time_charts_are_aligned(tmp_path, monkeypatch):
    """試算の結果の流量と水位のグラフは、同じ幅の枠・同じ左右の余白・同じ横軸の範囲で描き、描いた後に2つとも
    大きさを置いた場所に合わせ直す（初期化の時と幅が変わっても狭いまま残さない）。"""
    resized = []
    monkeypatch.setattr(ui.echart, "run_chart_method", lambda self, name, *a, **k: resized.append((self._markers, name)))
    d = copy_case("infiltration", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        user.find(marker="trial").click()
        await wait_for(user, "試算の結果", 60)
        flows = next(iter(user.find(marker="result_flows").elements))
        level = next(iter(user.find(marker="result_level").elements))
        for _ in range(20):
            if {m[0] for m, _ in resized if m} >= {"result_flows", "result_level"}:
                break
            await asyncio.sleep(0.1)
    assert "w-full" in flows.classes and "w-full" in level.classes and flows.parent_slot.parent is level.parent_slot.parent
    grids = [*flows.options["grid"], level.options["grid"]]
    assert {(g["left"], g["right"]) for g in grids} == {(charts.GRID_LEFT, charts.GRID_RIGHT)}
    assert flows.options["xAxis"][-1]["max"] == level.options["xAxis"]["max"]
    assert {m[0] for m, name in resized if name == "resize" and m} >= {"result_flows", "result_level"}


@pytest.mark.asyncio
async def test_charts_follow_the_browser_width(tmp_path):
    """グラフを置いた場所の幅は、ブラウザの表示幅から決まる（中身の幅では決まらない）。グラフから上の列・行・ページの枠は
    どれも幅を持つ（w-full か、左右に並べる行の右の列）。幅のない枠が1つあると、描いたグラフの幅がページの幅を固定し、
    表示幅を変えてもグラフも表も元の幅のまま残る。タブは開いているものだけをブラウザに置く（開いていないタブの中で
    作り直した部品は、画面の外で作られて説明が出ない）。"""
    from amamizu.gui.app import CHART_COLUMN
    d = copy_case("infiltration", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        user.find(marker="trial").click()
        await wait_for(user, "試算の結果", 60)
        found = list(user.find(kind=ui.echart).elements)
        assert len(found) >= 4                           # 条件のタブのグラフ、結果の2つ、試算の比較
        for chart in found:
            assert "w-full" in chart.classes
            e = chart.parent_slot.parent
            while e.parent_slot is not None:
                if isinstance(e, (ui.column, ui.row, ui.sub_pages)):
                    assert "w-full" in e.classes or set(CHART_COLUMN.split()) <= set(e.classes), (type(e).__name__, e.classes)
                e = e.parent_slot.parent
        panels = list(user.find(kind=ui.tab_panels).elements)
        assert panels and not any(p.props.get("keep-alive") for p in panels)


@pytest.mark.asyncio
async def test_save_pond_csv_only(tmp_path):
    """池のタブの「CSV に保存」は、池の表だけを保存先の CSV に書く（study.json と out/ は書かない）。
    池の表の変更は保存済みになり、ほかの条件の変更は未保存のまま。"""
    d = copy_case("five_methods", tmp_path)
    before = (d / "study.json").read_bytes()
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        volume = sorted((e for e in user.find(kind=ui.number, content="容量 m^3").elements), key=lambda e: e.id)
        volume[-1].set_value(9999.0)
        user.find(marker="save_pond").click()
        await wait_for(user, "既存の CSV を上書きします")          # 「保存して出力」と同じ確認
        await wait_for(user, "未保存の変更あり")
        user.find(marker="confirm_save").click()
        await wait_for(user, "池の表を pond.csv に保存しました")
        await wait_for(user, "保存済み")                           # 池の表だけの変更は、保存済みになる
    assert (d / "input" / "pond.csv").read_text(encoding="utf-8-sig").splitlines()[-1].endswith(",9999.0")
    assert (d / "study.json").read_bytes() == before and not (d / "out").exists()
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("保存済み")
        user.find(marker="allowable").elements.pop().set_value(0.123)
        volume = sorted((e for e in user.find(kind=ui.number, content="容量 m^3").elements), key=lambda e: e.id)
        volume[-1].set_value(8888.0)
        user.find(marker="save_pond").click()
        await wait_for(user, "既存の CSV を上書きします")
        user.find(marker="confirm_save").click()
        await wait_for(user, "池の表を pond.csv に保存しました")
        await asyncio.sleep(0.6)                                    # 状態の表示は 0.4 秒ごとに更新する
        await user.should_see("未保存の変更あり")                    # ほかの条件の変更は未保存のまま
    assert (d / "input" / "pond.csv").read_text(encoding="utf-8-sig").splitlines()[-1].endswith(",8888.0")
    assert (d / "study.json").read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("drop", ["容量_m3", "面積_m2"])
@pytest.mark.needs_japanese_font
async def test_pond_without_a_column_matches_cli(tmp_path, drop, opened):
    """池の表の面積か容量の列を空欄にした検討でも、WebGUI の「保存して出力」の out/ が amamizu study とバイト単位で一致する。"""
    gui_dir = copy_case("five_methods", tmp_path / "gui")
    pond = pd.read_csv(gui_dir / "input" / "pond.csv", encoding="utf-8-sig").assign(**{drop: None})
    pond.to_csv(gui_dir / "input" / "pond.csv", index=False, encoding="utf-8-sig", lineterminator="\r\n")
    async with user_simulation(root=root_factory(tmp_path / "gui")) as user:
        await user.open(f"/study/{gui_dir.name}")
        user.find(marker="save").click()
        await wait_for(user, "保存して out/ に出力しました", 60)
    cli_dir = copy_for_cli(gui_dir, tmp_path / "cli")
    subprocess.run([sys.executable, "-m", "amamizu.cli", "study", str(cli_dir)], check=True, capture_output=True)
    names = sorted(p.name for p in (gui_dir / "out").iterdir())
    assert names == sorted(p.name for p in (cli_dir / "out").iterdir())
    for n in names:
        assert (gui_dir / "out" / n).read_bytes() == (cli_dir / "out" / n).read_bytes(), n
    assert ("水面積(m2)" in (gui_dir / "out" / "route.csv").read_text(encoding="utf-8-sig")) == (drop != "面積_m2")


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["two_tier_pump", "hydrograph", "infiltration", "observed_10min"])
@pytest.mark.needs_japanese_font
async def test_save_and_output_matches_cli(tmp_path, case, opened):
    """WebGUI の「保存して出力」の out/ と、同じ study.json での amamizu study の out/ がバイト単位で一致する。
    出力した後に、清書した PDF を開く。"""
    gui_dir = copy_case(case, tmp_path / "gui")
    async with user_simulation(root=root_factory(tmp_path / "gui")) as user:
        await user.open(f"/study/{gui_dir.name}")
        user.find(marker="save").click()
        await wait_for(user, "保存して out/ に出力しました", 60)
    assert opened == [gui_dir / "out" / "design_summary.pdf"]
    cli_dir = copy_for_cli(gui_dir, tmp_path / "cli")
    subprocess.run([sys.executable, "-m", "amamizu.cli", "study", str(cli_dir)], check=True, capture_output=True)
    names = sorted(p.name for p in (gui_dir / "out").iterdir())
    assert names == sorted(p.name for p in (cli_dir / "out").iterdir()) and "route.csv" in names
    for n in names:
        assert (gui_dir / "out" / n).read_bytes() == (cli_dir / "out" / n).read_bytes(), n


@pytest.mark.asyncio
@pytest.mark.needs_japanese_font
async def test_save_shows_graph_error(tmp_path, opened):
    """グラフを出力できないとき（題名にフォントにない文字がある）は、「保存して出力」で WebGUI にその理由を表示する。
    図を残さず、PDF を開かない。"""
    d = copy_case("guideline", tmp_path)
    conf = read_study(d)
    conf["title"] = "調整池 \U0001F600"
    (d / "study.json").write_bytes(dump_study(conf))
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        user.find(marker="save").click()
        await wait_for(user, "フォントにない文字", 60)
    assert opened == [] and not any((d / "out").glob("*"))      # 途中まで書いた route.csv・inflow.csv も残さない


@pytest.mark.asyncio
async def test_rain_page_shows_missing_font(tmp_path, monkeypatch):
    """日本語のフォントがないときは、降雨の画面の保存で WebGUI にその理由を表示する。"""
    from amamizu.adapters.report import design_plots
    monkeypatch.setattr(design_plots, "JP_FONTS", ["存在しないフォント"])
    d = copy_case("guideline", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/rain/{d.name}")
        await user.should_see("総雨量")
        user.find("out/ に保存").click()
        await wait_for(user, "日本語のフォント")
    assert not any((d / "out").glob("planned_rain.*"))      # グラフを出力できないときは CSV・JSON も残さない


@pytest.mark.asyncio
@pytest.mark.needs_japanese_font
async def test_rain_page_saves_with_conditions(tmp_path):
    d = copy_case("guideline", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/rain/{d.name}")
        await user.should_see("総雨量")
        user.find("out/ に保存").click()
        await wait_for(user, "保存しました")
    assert {p.name for p in (d / "out").iterdir()} == {"planned_rain.csv", "planned_rain.png", "planned_rain.json"}
    assert json.loads((d / "out" / "planned_rain.json").read_text(encoding="utf-8"))["district"] == "千葉"


@pytest.mark.asyncio
async def test_rain_page_starts_from_passed_planned_rain(tmp_path):
    """検討の画面の「計画降雨」は、選んでいる（保存前を含む）計画降雨の条件を降雨の画面に渡し、それを初期値にする。"""
    from urllib.parse import urlencode
    d = copy_case("guideline", tmp_path)
    q = urlencode({"source": "planned", "district": "我孫子", "return_period": 10, "shape": "center", "hours": 12, "tc": 10})
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/rain/{d.name}?{q}")
        await wait_for(user, "手引 我孫子地区 1/10")
        await user.should_see("中央集中型12時間")


def test_rain_query_passes_integers(tmp_path):
    """数値の欄で入力した整数は小数（20.0）になる。降雨のページへ渡す条件は整数に直す（降雨のページは int で読む）。"""
    f = DesignForm.load(copy_case("guideline", tmp_path))
    f.conf["tc_min"] = 20.0
    f.conf["rain"]["hours"] = 12.0
    q = f.rain_query()
    assert q["source"] == "planned" and (q["tc"], q["hours"]) == (20, 12)
    assert all(not isinstance(v, float) or not v.is_integer() for v in q.values())
    assert DesignForm.load(copy_case("hydrograph", tmp_path)).rain_query() is None


@pytest.mark.asyncio
async def test_rain_page_starts_from_study_tc(tmp_path):
    """URL で tc を渡さないときは、検討の tc_min を計画降雨の区間にする。"""
    d = copy_case("guideline", tmp_path)
    conf = read_study(d)
    conf["tc_min"] = 20
    (d / "study.json").write_bytes(dump_study(conf))
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/rain/{d.name}")
        await wait_for(user, "後方集中型24時間、20分ごと")


@pytest.mark.asyncio
@pytest.mark.needs_japanese_font
async def test_rain_page_shows_observed_record_like_planned(tmp_path):
    """降雨の画面は、計画降雨と同じ形（説明、ハイエトグラフ、区間ごとの表、out/ への保存）で降雨記録（実測降雨）を見る。
    検討の画面から渡したファイル・期間を初期値にし、計画降雨に切り替えられる。総雨量は貯留施設の検証の表と同じ。"""
    from urllib.parse import urlencode
    d = tmp_path / "2026-09_台風25号_坂畑"
    shutil.copytree(STUDIES / d.name, d, ignore=shutil.ignore_patterns("out"))    # 10分値はリポジトリの studies/_jma/ を読む
    spec = read_study(d)["observed_rain"]
    q = urlencode({"source": "observed"} | spec | {"tc": 10})
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/rain/{d.name}?{q}")
        await wait_for(user, "降雨記録 坂畑 10分値 2026-09-20〜2026-09-22")
        await user.should_see("432 区間、総雨量 476.5 mm")                  # 3日×144区間、event_20260920.tsv の3日合計
        await user.should_see("欠測の時間（0 mm として計算）")
        [table] = user.find(ui.table).elements
        assert table.rows[0]["区間の日時"] == "2026-09-20 00:00〜00:10"
        user.find("out/ に保存").click()
        await wait_for(user, "保存しました")
        toggle = user.find(marker="rain_source").elements.pop()
        with user:
            toggle.value = "planned"
        await wait_for(user, "手引 館山地区 1/50")      # 計画降雨の地区は、降雨記録の地点（坂畑）が属する館山地区
    assert {p.name for p in (d / "out").iterdir()} == {"observed_rain.csv", "observed_rain.png", "observed_rain.json"}
    assert {k: v for k, v in json.loads((d / "out" / "observed_rain.json").read_text(encoding="utf-8")).items() if k in spec} == spec


@pytest.mark.asyncio
async def test_viewer_computes_return_period(tmp_path):
    src = STUDIES / "2026-09_降雨強度の検証"
    shutil.copytree(src, tmp_path / src.name, ignore=shutil.ignore_patterns("out"))
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{src.name}")
        await wait_for(user, "地点ごとの再現期間", 30)
        [table] = user.find(ui.table).elements
    from amamizu.app.viewer import return_period_rows
    expected = return_period_rows(tmp_path / src.name, read_study(tmp_path / src.name))
    assert [r["再現期間 年"] for r in table.rows] == expected["再現期間 年"].astype(str).tolist()
    assert len(table.rows) > 0


def test_level_axis_includes_lines():
    """満水位などの線が縦軸の範囲の外にあると描かれないので、範囲に入れる。"""
    o = charts.level(np.arange(0, 60, 10), {"水位": [0.1, 0.5, 0.8, 0.6, 0.3, 0.1]}, [("満水位", 2.0, charts.RED)])
    assert o["yAxis"]["max"] >= 2.0 and o["yAxis"]["min"] <= 0.1
    assert o["xAxis"]["max"] % o["xAxis"]["interval"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(p.name for p in CASES.iterdir()))
@pytest.mark.needs_japanese_font
async def test_save_and_output_matches_command(tmp_path, case):
    """設計の検討の例すべてで、WebGUI の「保存して出力」の out/ が、同じ検討の amamizu study の out/ とバイト単位で一致する。"""
    d = copy_case(case, tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        user.find(marker="save").click()
        await wait_for(user, "保存して out/ に出力しました", 60)
    cmd = copy_case(case, tmp_path / "command")
    main(["study", str(cmd)])
    names = sorted(p.name for p in (cmd / "out").iterdir())
    assert sorted(p.name for p in (d / "out").iterdir()) == names
    for name in names:
        assert (d / "out" / name).read_bytes() == (cmd / "out" / name).read_bytes(), name


def test_gui_command_without_nicegui(monkeypatch, capsys):
    """NiceGUI がインストールされていなければ、インストールの方法を示して終わる（終了コード 1）。"""
    from amamizu import cli
    monkeypatch.setitem(sys.modules, "nicegui", None)
    monkeypatch.setattr(sys.stdout, "reconfigure", lambda **kw: None, raising=False)
    assert cli.main(["gui"]) == 1
    assert "uv sync --extra gui" in capsys.readouterr().err


def test_delete_only_design_studies(tmp_path):
    d = copy_case("two_tier_pump", tmp_path)
    plan = delete_plan(tmp_path, d.name)
    assert plan.path == d.resolve() and plan.files == 2 and not plan.has_out
    assert plan.entries == ["input/", "study.json"]
    removed = []
    delete_study(plan, trash=removed.append)
    assert removed == [str(d.resolve())] and d.exists()
    delete_study(plan, trash=removed.append, permanent=True)          # 完全な削除はごみ箱を使わない
    assert removed == [str(d.resolve())] and not d.exists()
    src = STUDIES / "2026-09_降雨強度の検証"
    shutil.copytree(src, tmp_path / src.name, ignore=shutil.ignore_patterns("out", "raw"))
    with pytest.raises(ValueError, match="設計の検討だけ"):
        delete_plan(tmp_path, src.name)
    for bad in ("..", "../x", "ない検討"):
        with pytest.raises(ValueError, match="検討のフォルダではありません"):
            delete_plan(tmp_path, bad)


@pytest.mark.asyncio
async def test_delete_from_home(tmp_path, monkeypatch):
    import send2trash
    monkeypatch.setattr(send2trash, "send2trash", shutil.rmtree)      # 試験ではごみ箱を使わない
    d = copy_case("guideline", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open("/")
        await user.should_see("手引の式・柱状の池")
        assert inside(user, f"more:{d.name}", marker=f"delete:{d.name}")   # 削除は「⋮」のメニューの中
        user.find(marker=f"delete:{d.name}").click()
        await user.should_see("を削除します")
        user.find(marker="delete_ok").click()
        await wait_for(user, "を削除しました")
        await user.should_not_see(marker=f"delete:{d.name}")                # 一覧から消える
    assert not d.exists()


@pytest.mark.asyncio
async def test_delete_from_study_page(tmp_path, monkeypatch):
    import send2trash
    monkeypatch.setattr(send2trash, "send2trash", shutil.rmtree)      # 試験ではごみ箱を使わない
    d = copy_case("guideline", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        assert inside(user, f"more:{d.name}", marker=f"delete:{d.name}")   # 削除は上部の「⋮」のメニューの中
        assert inside(user, f"more:{d.name}", text="この検討を削除する")
        user.find(marker=f"delete:{d.name}").click()
        await user.should_see("中身: input/、study.json" if (d / "input").exists() else "中身: study.json")
        user.find(marker="delete_ok").click()
        await wait_for(user, "貯留追跡計算を行う")                       # 一覧の画面（「検討の一覧」は検討の画面の戻るボタンにもある）
        await user.should_not_see(marker="save")                       # 検討の画面から一覧へ移った
    assert not d.exists()


@pytest.mark.asyncio
async def test_save_only_does_not_output(tmp_path):
    """「保存」は study.json を書き、計算も出力もしない。"""
    d = copy_case("guideline", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("out/: まだ出力していません")
        user.find(marker="allowable").clear().type("0.07")
        await wait_for(user, "未保存の変更あり")
        user.find(marker="save_only").click()
        await wait_for(user, "保存しました（出力はしていません）")
    assert read_study(d)["allowable_q_m3s"] == 0.07 and not (d / "out").exists()


@pytest.mark.asyncio
@pytest.mark.needs_japanese_font
async def test_study_page_shows_conditions_changed_after_output(tmp_path, opened):
    """out/ の出力の後に条件を変えたら、そのことと変えた内容を示す。「保存して出力」で今の条件の結果になる。"""
    d = copy_case("guideline", tmp_path)
    run_study(read_study(d), d)
    conf = read_study(d)
    conf["orifice"]["width_m"] = 0.09
    (d / "study.json").write_bytes(dump_study(conf))
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("条件が変更されています（1 件）")
        assert inside(user, "output_changes", text="放流施設 オリフィス 幅（orifice.width_m）: 0.08 → 0.09")
        user.find(marker="save").click()
        await wait_for(user, "out/: 今の条件で出力した結果です", 60)


@pytest.mark.asyncio
async def test_duplicate_asks_for_a_title_and_opens_the_copy(tmp_path):
    """一覧の「⋮」のメニューで複製する。新しい検討の題名を入力し、複製したら新しい検討を開く。"""
    d = copy_case("two_tier_pump", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open("/")
        assert inside(user, f"more:{d.name}", marker=f"duplicate:{d.name}")
        user.find(marker=f"duplicate:{d.name}").click()               # 複製は削除の上
        await user.should_see("検討を複製する")
        assert user.find(marker="duplicate_title").elements.pop().value in ("", None)
        user.find(marker="duplicate_ok").click()
        await user.should_see("題名を入力してください")                # 空欄のままでは作らない
        assert [p for p in tmp_path.iterdir() if p != d] == []
        user.find(marker="duplicate_title").type("ケース2")
        folder = new_study_dir(tmp_path, "ケース2")
        user.find(marker="duplicate_ok").click()
        await wait_for(user, f"フォルダ {folder}")                     # 新しい検討の画面（題名と作るフォルダは複製のダイアログにもある）
    [new] = [p for p in tmp_path.iterdir() if p != d]
    assert read_study(new)["title"] == "ケース2" and (new / "input" / "pond.csv").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["edits", "save_first"])
async def test_duplicate_with_unsaved_changes_keeps_them(tmp_path, how):
    """保存していない変更があるときの複製は、変更を失わない。変更を含めて複製（元は保存した状態のまま）か、
    保存してから複製（元にも保存）を選び、新しい検討を開く。"""
    d = copy_case("guideline", tmp_path)
    saved = read_study(d)["allowable_q_m3s"]
    edited = round(saved + 0.01, 4)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        user.find(marker="allowable").clear().type(str(edited))
        await wait_for(user, "未保存の変更あり")
        assert inside(user, f"more:{d.name}", marker=f"duplicate:{d.name}")
        user.find(marker=f"duplicate:{d.name}").click()
        await user.should_see("保存していない変更があります")
        user.find(marker="duplicate_title").type("ケース3")
        folder = new_study_dir(tmp_path, "ケース3")
        user.find(marker=f"duplicate_{how}").click()
        await wait_for(user, f"フォルダ {folder}")                     # 新しい検討の画面
        await user.should_see("保存済み")                              # 開いた新しい検討は、保存した状態
    [new] = [p for p in tmp_path.iterdir() if p != d]
    assert read_study(new)["title"] == "ケース3" and read_study(new)["allowable_q_m3s"] == edited
    assert read_study(d)["allowable_q_m3s"] == (edited if how == "save_first" else saved)


def test_duplicate_with_edits_writes_the_edited_pond_table(tmp_path):
    """変更を含めて複製すると、編集中の池の表も新しい検討に書く。元の検討の表は変えない。"""
    d = copy_case("two_tier_pump", tmp_path)
    f = DesignForm.load(d)
    before = (d / "input" / "pond.csv").read_bytes()
    f.pond_rows[1]["面積_m2"] = 531.0
    f.conf["orifices"][0]["width_m"] = 0.08
    new = f.duplicate("ケース4")
    g = DesignForm.load(new)
    assert g.pond_rows[1]["面積_m2"] == 531.0 and g.conf["orifices"][0]["width_m"] == 0.08
    assert (d / "input" / "pond.csv").read_bytes() == before and read_study(d)["orifices"][0]["width_m"] == 0.07
    assert f.dirty                                                 # 元の検討の編集中の状態はそのまま
    f.conf["tc_min"] = 10.5
    with pytest.raises(ValueError, match="値を直す項目"):
        f.duplicate("ケース5")


# 利用者が studies/ に置いた想定外のフォルダ ------------------------------------------
GOOD = json.loads((CASES / "guideline" / "study.json").read_text(encoding="utf-8"))
# フォルダ名 → (study.json の中身（bytes か dict）, 手を加える処理, 検討の画面に示す理由。None は開ける)
ODD_STUDIES = {
    "正常": (GOOD, None, None),
    "正常 - コピー": (GOOD, None, None),
    "JSONの誤り": (b"{ kind: design,", None, "JSON として読めません"),
    "空のファイル": (b"", None, "JSON として読めません"),
    "並び": (b"[1, 2]", None, "項目を持つ形"),
    "ShiftJIS": (json.dumps(GOOD, ensure_ascii=False).encode("cp932"), None, "UTF-8 ではありません"),
    "kindなし": ({k: v for k, v in GOOD.items() if k != "kind"}, None, "kind がありません"),
    "kind不明": (GOOD | {"kind": "foo"}, None, "kind「foo」は扱えません"),
    "不明な項目": (GOOD | {"memo": "自分用"}, None, None),
    "中身が空": ({"kind": "design"}, None, None),
    "入力ファイルなし": (GOOD | {"pond": {"table": {"file": "input/ない.csv"}}}, None, None),
    "outがファイル": (GOOD, lambda d: (d / "out").write_text("x"), None),
    "outに下位フォルダ": (GOOD, lambda d: (d / "out" / "old").mkdir(parents=True), None),
    "created不正": (GOOD | {"created": "きのう"}, None, None),
    "titleが数": (GOOD | {"title": 123}, None, None),
    "整数でない": (GOOD | {"tc_min": 10.5}, None, None),
    "storageでrawなし": ({"kind": "storage", "title": "raw なし"}, None, "必要な項目がありません: conditions"),
    "return_periodでrawなし": ({"kind": "return_period", "title": "raw なし"}, None, "必要な項目がありません: standard_period"),
    "区域が辞書": (GOOD | {"catchments": {"name": "x"}}, None, "catchments（リストで書く）"),
}


LIST_REASONS = ("JSONの誤り", "空のファイル", "並び", "ShiftJIS", "kindなし", "kind不明")     # 一覧の行に理由を示す検討


def odd_studies(studies: Path) -> None:
    for name, (data, touch, _) in ODD_STUDIES.items():
        d = studies / name
        d.mkdir(parents=True)
        (d / "study.json").write_bytes(data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode("utf-8"))
        if touch:
            touch(d)
    (studies / "study.json のないフォルダ").mkdir()


@pytest.mark.asyncio
async def test_home_lists_every_study_even_with_odd_folders(tmp_path):
    """想定外のフォルダがあっても一覧は出る。どのフォルダも1行ずつ表示し、読めない study.json はその行に理由を示す。"""
    odd_studies(tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open("/")
        await user.should_see("検討の一覧")
        for name in ODD_STUDIES:
            await user.should_see(marker=f"open:{name}")
        await user.should_not_see(marker="open:study.json のないフォルダ")
        for name, (_, _, why) in ODD_STUDIES.items():
            card = user.find(marker=f"open:{name}").elements.pop()
            while not isinstance(card, ui.card):
                card = card.parent_slot.parent
            texts = [str(getattr(e, "text", "")) for e in card.descendants()]
            if name in LIST_REASONS:                    # 一覧は study.json を読めるかと kind だけを確かめる
                assert any(why in t for t in texts), (name, texts)
            assert any("created「きのう」を日時として読めない" in t for t in texts) == (name == "created不正"), (name, texts)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(ODD_STUDIES))
async def test_study_page_opens_or_tells_why(tmp_path, name):
    """想定外のフォルダでも検討の画面はエラー（500）にしない。開けるものは開き（開いただけで条件を書き換えない）、
    開けないものは理由を示す。"""
    odd_studies(tmp_path)
    why = ODD_STUDIES[name][2]
    before = (tmp_path / name / "study.json").read_bytes()
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{name}")
        await user.should_not_see("produced an error")
        assert (tmp_path / name / "study.json").read_bytes() == before
        if why is None:
            await user.should_not_see(marker="cannot_open")
            data = ODD_STUDIES[name][0]
            if isinstance(data, dict) and data.get("kind") == "design":
                await user.should_see("保存済み")
        else:
            await user.should_see(marker="cannot_open")
            await user.should_see(why)


@pytest.mark.asyncio
async def test_delete_permanently_when_trash_fails(tmp_path, monkeypatch):
    """ごみ箱へ移せないときは理由を表示し、「完全に削除する」を押したときだけ完全に削除する。"""
    import send2trash

    def no_trash(path):
        raise OSError("ごみ箱がありません")
    monkeypatch.setattr(send2trash, "send2trash", no_trash)
    d = copy_case("guideline", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        assert inside(user, f"more:{d.name}", marker=f"delete:{d.name}")
        user.find(marker=f"delete:{d.name}").click()
        user.find(marker="delete_ok").click()
        await wait_for(user, "ごみ箱へ移せませんでした: ごみ箱がありません")
        assert d.exists()                                                  # 押すまでは消さない
        user.find(marker="delete_permanent").click()
        await wait_for(user, "貯留追跡計算を行う")                       # 一覧の画面（「検討の一覧」は検討の画面の戻るボタンにもある）
        await user.should_not_see(marker="save")
    assert not d.exists()


@pytest.mark.parametrize("platform, command", [("darwin", "open"), ("linux", "xdg-open")])
def test_open_path_on_other_os(tmp_path, monkeypatch, platform, command):
    """os.startfile のない OS では、macOS は open、Linux などは xdg-open で開く。"""
    from amamizu.gui import app as gui_app
    monkeypatch.undo()                         # 開く処理を記録に置き換えた fixture を戻し、本来の open_path を試す
    monkeypatch.delattr(gui_app.os, "startfile", raising=False)
    monkeypatch.setattr(gui_app.sys, "platform", platform)
    calls = []
    monkeypatch.setattr(gui_app.subprocess, "Popen", lambda args: calls.append(args))
    gui_app.open_path(tmp_path)
    assert calls == [[command, str(tmp_path)]]


# 手引の定数を使用しない検討 ---------------------------------------------------------


def test_new_study_without_guideline_files(tmp_path):
    """手引の定数のファイルがない検討のフォルダでは、新しい検討を「使用しない」で作る（そのまま計算できる）。"""
    (tmp_path / "_guideline").mkdir()
    f = DesignForm.load(create_design_study(tmp_path, "手引なし"))
    assert f.no_guideline and f.conf["hydraulics"] == {"g": 9.8, "c_weir": 1.8, "c_orifice": 0.6}
    assert "formula" in f.conf["rain"] and f.check().ok


def test_new_study_selects_chiba(tmp_path):
    conf = read_study(create_design_study(tmp_path, "新しい池"))
    assert list(conf)[:3] == ["kind", "title", "guideline"] and conf["guideline"] == "chiba_2006" and "hydraulics" not in conf


def test_switch_to_no_guideline_keeps_the_result(tmp_path):
    """「使用しない」に切り替えると、放流の係数に一般的な値が入り、手引の地区と確率年の計画降雨は同じ式の係数の入力になる
    （千葉県の手引は係数が一般的な値と同じなので、結果は変わらない）。手引の定数を選び直すと、係数の項目を削除する。"""
    d = copy_case("guideline", tmp_path)
    f = DesignForm.load(d)
    before = trial(f.normalized(), d).result.summary
    f.set_guideline(None)
    assert f.no_guideline and f.conf["hydraulics"] == {"g": 9.8, "c_weir": 1.8, "c_orifice": 0.6}
    assert "guideline" not in f.conf["rain"] and f.conf["rain"]["formula"]["type"] == "cleveland"
    assert f.check().ok and trial(f.normalized(), d).result.summary == before
    assert b'"guideline": null' in dump_study(f.normalized())           # 「使用しない」は null と書く（項目を省略しない）
    f.set_guideline("chiba_2006")
    assert "hydraulics" not in f.conf and f.check().ok and trial(f.normalized(), d).result.summary == before


def test_template_is_not_a_choice():
    """雛形（_template.json）は読めるが、選択肢には表示しない。"""
    from amamizu.adapters.guideline.json_guideline import JsonGuideline, guideline_names
    assert guideline_names(STUDIES) == ["chiba_2006"]
    g = JsonGuideline(STUDIES / "_guideline" / "_template.json")
    assert g.hydraulics().g == 9.8 and g.return_periods("地区A") == [10, 50]


@pytest.mark.asyncio
async def test_no_guideline_shows_hydraulics_fields(tmp_path):
    """「使用しない」の検討では、放流の係数の入力欄と説明を表示し、計画降雨は係数の入力だけにする。"""
    d = copy_case("guideline", tmp_path)
    f = DesignForm.load(d)
    f.set_guideline(None)
    (d / "study.json").write_bytes(dump_study(f.normalized()))
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("越流係数")
        assert user.find(marker="guideline").elements.pop().value == "使用しない"
        await user.should_see("初期値は一般的な値です。使用する基準の値を確認して入力してください。")
        await user.should_not_see("手引の式（地区・確率年）")


# エラーを表示せずに処理を終えない ----------------------------------------------------


def test_sample_formula_is_chiba_50():
    """強度式の係数の入力欄の初期値は、千葉県の手引の千葉地区・50年確率の式。"""
    from amamizu.adapters.guideline.json_guideline import JsonGuideline
    from amamizu.gui.state import SAMPLE_FORMULA
    assert SAMPLE_FORMULA == JsonGuideline(CHIBA).formula_spec("千葉", 50)


def test_switch_to_no_guideline_tells_when_formula_is_lost(tmp_path):
    """「使用しない」に切り替えるときに、選んでいた計画降雨の式を読めなければ、初期値にしたことを知らせる。"""
    d = copy_case("guideline", tmp_path)
    f = DesignForm.load(d)
    assert f.set_guideline(None) is None                         # 読めたときは知らせない
    f.set_guideline("chiba_2006")
    f.conf["rain"] = {"guideline": {"district": "東京", "return_period": 50}, "shape": "rear"}
    notice = f.set_guideline(None)
    assert "東京" in notice and "初期値にしました" in notice and f.conf["rain"]["formula"]["a"] == 2439


@pytest.mark.asyncio
async def test_trial_shows_unexpected_error(tmp_path, monkeypatch, capsys):
    """試算の途中の想定していない例外は、WebGUI に表示し、経過をターミナルに出力する（何も表示せずに終えない）。"""
    from amamizu.gui import app as gui_app
    d = copy_case("guideline", tmp_path)

    def broken(*args, **kw):
        raise RuntimeError("内部の誤り")
    monkeypatch.setattr(gui_app, "trial", broken)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        user.find(marker="trial").click()
        await wait_for(user, "試算の途中で、想定していないエラーが起きました（RuntimeError: 内部の誤り）")
    assert "RuntimeError: 内部の誤り" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_rain_page_shows_why_it_cannot_open(tmp_path):
    """降雨のページは、URL の値を数として読めないときと、手引の定数を読めないときに、理由を表示する。"""
    d = copy_case("guideline", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/rain/{d.name}?tc=abc")
        await user.should_see("降雨のページを開けません")
        await user.open(f"/rain/{d.name}?basis=tokyo")
        await user.should_see("手引の定数（tokyo）を読めません")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["/x", "C:/x", "**"])
async def test_observed_station_not_fetched_is_shown_as_written(tmp_path, name):
    """study.json の地点が取得していない名前（フォルダを探すパターンとして読めない名前も）なら、
    その名前のまま選択肢に示す（WebGUI を止めない。取得済みの地点の都道府県も表示したままにする）。"""
    d = copy_case("observed_10min", tmp_path)
    conf = read_study(d)
    conf["observed_rain"]["station"] = name
    (d / "study.json").write_text(json.dumps(conf, ensure_ascii=False, indent=2), encoding="utf-8")
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await wait_until(lambda: user.find(marker="observed_station").elements)
        [station] = user.find(marker="observed_station").elements
        assert station.value == name and station.options[name] == name
        assert station.options["館山"] == "館山（千葉県）"


@pytest.mark.asyncio
async def test_observed_station_info_error_is_shown_near_the_fields(tmp_path):
    """station.json が項目を持つ形でないときは、欄の近くに理由を表示し、地点名だけの選択肢にする（WebGUI を止めない）。"""
    (tmp_path / "_jma").mkdir()
    shutil.copy2(STUDIES / "_jma" / "config.json", tmp_path / "_jma")
    shutil.copytree(STUDIES / "_jma" / "館山_47672", tmp_path / "_jma" / "館山_47672")
    (tmp_path / "_jma" / "館山_47672" / "station.json").write_text("[]", encoding="utf-8")
    d = copy_case("observed_10min", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await wait_for(user, "地点の都道府県を読めません（地点名だけを表示します）")
        await user.should_see("station.json は項目を持つ形")
        assert station_options(user) == {"館山": "館山"}


@pytest.mark.asyncio
async def test_observed_period_error_is_shown_near_the_fields(tmp_path):
    """地点の選択肢に表示される地点でも、期間を読めないとき（同じ地点名の観測所のフォルダが2つある、値のファイルが読めない）は、
    欄の近くに理由を表示する（WebGUI を止めない。空の一覧にして黙って飛ばさない）。"""
    (tmp_path / "_jma").mkdir()
    shutil.copy2(STUDIES / "_jma" / "config.json", tmp_path / "_jma")
    for folder in ("千葉_47682", "千葉_9999", "館山_47672"):
        shutil.copytree(STUDIES / "_jma" / folder.replace("_9999", "_47682"), tmp_path / "_jma" / folder)
    bad = tmp_path / "_jma" / "館山_47672" / "tenmin" / "2026.tsv"
    lines = bad.read_bytes().split(b"\r\n")
    lines[1] = lines[1].replace(b"\t0.0\t", b"\tO.5\t", 1)          # 雨量に数として読めない値（0 の代わりに O）
    bad.write_bytes(b"\r\n".join(lines))
    d = copy_case("guideline", tmp_path)
    async with user_simulation(root=root_factory(tmp_path)) as user:
        await user.open(f"/study/{d.name}")
        await user.should_see("保存済み")
        with user:
            user.find(marker="inflow_source").elements.pop().value = "observed_rain"
        await wait_for(user, "値の種類（10分値か時間値）を選んでください")
        with user:
            user.find(marker="observed_series").elements.pop().value = "tenmin"
        await wait_until(lambda: station_options(user))               # 値の種類を選ぶと地点の選択肢が入る
        assert station_options(user) == {"千葉": "千葉", "館山": "館山（千葉県）"}   # 観測所が2つある地点は都道府県を決められない
        for station, message in (("千葉", "地点「千葉」の観測所が 2 つあります"), ("館山", "を雨量として読めません")):
            with user:
                user.find(marker="observed_station").elements.pop().value = station
            await wait_for(user, message)
            await user.should_not_see(marker="observed_period")            # 読めない地点の期間の一覧は表示しない
