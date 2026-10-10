"""WebGUI（NiceGUI）。起動は `amamizu gui`。

WebGUI は入力と表示だけを担う。組み立て・確かめ・計算・出力は app 層（design_runner、rain_runner）と
gui/state.py をそのまま使う（WebGUI で同じ確かめや計算を書き直さない）。

ページ:
  /                 検討の一覧、設計の検討を新しく作る
  /study/{name}     設計の検討（条件のタブ、試算、結果、試算の比較、保存して出力）。貯留施設の検証・降雨強度の検証は見るだけ
  /rain/{name}      降雨（計画降雨か降雨記録を選び、ハイエトグラフと区間ごとの表を見て、検討フォルダの out/ に出力する）。
                    ?source=observed&file=&station=&start=&end=&tc= で降雨記録（実測降雨）を開く
"""
import datetime
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from urllib.parse import quote, unquote, urlencode

import numpy as np
from nicegui import PageArguments, run, ui
from uvicorn.config import STARTUP_FAILURE

from ..adapters.design_inputs import OBSERVED_SERIES, POND_COLUMNS, fetch_command, pond_columns, pond_rows_of, read_table
from ..adapters.report.design_plots import GraphError
from ..adapters.guideline.json_guideline import JsonGuideline, guideline_dir, guideline_names
from ..adapters.jma_store import JmaStore
from ..adapters.study_config import check_kind, check_study, read_study, shape_errors
from ..app.observations import changed_count, fetched_days, fetched_periods, observation_changes, period_label, periods_summary
from ..app.design_runner import (INFLOW_SOURCES, SECTIONS, ConfigError, gap_notes, infiltration_label, infiltration_rows,
                                 read_output_conditions, run_study, spill_text, trial)
from ..app.rain_runner import make_observed_rain, make_rain, saved_rain_name, saved_rain_path, write_rain
from ..app.webgui_port import HOST, IS_WINDOWS, PortState, checked_state, unusable_message
from ..domain.infiltration import KINDS, PERMEABILITY_UNITS
from ..domain.intensity import FORMULA_TYPES
from ..domain.pond import DEFAULT_TC_MIN
from ..usecases.design_study import FACTOR_NAMES, clock
from . import charts
from .state import (SAMPLE_FORMULA, SAMPLE_FORMULA_NOTE, DesignForm, changes, create_design_study, created_at, created_error, delete_plan, delete_study, duplicate_study,
                    differences, list_order, new_study_dir, station_district)

# 浸透施設の影響係数: 施設を追加したときの値と、欄の近くに表示する出典の値
NEW_FACTORS = {"clogging": 0.9, "groundwater": 0.9, "safety": 1.0}
FACTOR_NOTE = ("雨水浸透施設技術指針［案］調査・計画編 p.47・p.55: 各種影響係数は一般的には 0.81。目づまりの標準 0.9、"
               "地下水位はボアホール法・土研法などの簡易試験によった場合 0.9 を標準とする")
KIND_LABEL = {"design": "設計（貯留追跡計算）", "storage": "貯留施設の検証", "return_period": "降雨強度の検証"}
OK_COLOR, NG_COLOR = "positive", "negative"
# 左に入力や表、右にグラフを並べる行と、その右の列。表示幅が足りないときは、右の列を下の行へ折り返す
TWO_COLUMNS = "w-full gap-6"
CHART_COLUMN = "grow basis-0 min-w-[24rem] max-w-full"
# タブの中身は、開いているタブだけをブラウザに置く。開いていないタブを保持すると、その中で作り直した部品が画面の外で作られ、
# タブを開いても説明（ツールチップ）が出ない
TAB_PANELS = {"animated": False, "keep_alive": False}


def fit_charts(*charts):
    """グラフの大きさを、置いた場所の大きさに合わせ直す（描いた少し後に1回）。
    NiceGUI のグラフは、初期化した時とその後で置いた場所の幅が変わると、狭いまま残ることがある。
    上下に並べる時間のグラフ（流量と水位）の横幅と横の位置をそろえるため、結果を描いたら呼ぶ。"""
    def fit():
        for c in charts:
            if not c.is_deleted:
                c.run_chart_method("resize")
    ui.timer(0.3, fit, once=True)
    return charts


def open_path(path: Path):
    """フォルダやファイルを OS の既定のアプリで開く（Windows は os.startfile、macOS は open、Linux などは xdg-open）。"""
    try:
        if hasattr(os, "startfile"):
            os.startfile(str(path))
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
    except OSError as e:
        ui.notify(f"{path} を既定のアプリで開けませんでした（{e}）。エクスプローラーからこのファイルを開いてください。",
                  type="negative", multi_line=True)


def unexpected(err: Exception, what: str):
    """想定していない例外を WebGUI に表示する（表示せずに処理を終えない）。経過は amamizu gui を実行したターミナルに出力する。"""
    traceback.print_exception(err)
    ui.notify(f"{what}の途中で、想定していないエラーが起きました（{type(err).__name__}: {err}）。"
              "経過は amamizu gui を実行したターミナルに表示しています。", type="negative", multi_line=True, timeout=0,
              close_button="閉じる")


NO_GUIDELINE = "使用しない"
# 手引を使用しない検討の、放流の係数の入力欄（study.json の項目, 名前, 単位, 説明）
HYDRAULICS_FIELDS = (
    ("g", "重力加速度 g", "m/s^2", "オリフィスの式に使います"),
    ("c_weir", "越流係数", "", "水深がオリフィスの高さの1.2倍以下のときの式 Q＝C·B·h^1.5 の C。洪水吐の式にも使います"),
    ("c_orifice", "オリフィスの流量係数", "", "水深がオリフィスの高さの1.8倍以上のときの式 Q＝C·A·√(2gh) の C"),
)


# 保存していない変更を黙って捨てない。検討の画面は保存状態を window.amamizuDirty に伝え、
# 再読み込み・タブを閉じるときはブラウザの確認、WebGUI の中の移動（go）は確かめてから移る。
LEAVE_GUARD = """<script>
window.amamizuDirty = false;
window.addEventListener('beforeunload', e => { if (window.amamizuDirty) { e.preventDefault(); e.returnValue = ''; } });
window.amamizuGo = (url, force) => {
  if (!force && window.amamizuDirty && !confirm('保存していない変更があります。変更を捨てて移動しますか？')) return;
  window.amamizuDirty = false;
  window.location.href = url;
};
</script>"""


def guideline_choices(g: JsonGuideline | None, chosen: dict) -> tuple[list, list]:
    """計画降雨の地区と確率年の選択肢（手引の定数のファイルにあるもの）。選んでいる値が選択肢になければ足す
    （入力欄に値を表示し、誤りは入力の確かめで示す）。手引を読めなければ、選んでいる値だけ。"""
    district, period = chosen.get("district"), chosen.get("return_period")
    districts = g.districts() if g else []
    periods = g.return_periods(district) if g and district in districts else []
    return (districts + [v for v in [district] if v is not None and v not in districts],
            periods + [v for v in [period] if v is not None and v not in periods])


def go(url: str, force: bool = False):
    """画面を移る。検討の画面に保存していない変更があれば確かめる。force なら確かめない（検討を削除したとき、
    保存していない変更を複製に含めたか保存したときなど）。確かめない移動は、ブラウザの確認（beforeunload）も表示しないよう
    保存していない変更の印（window.amamizuDirty）をオフにしてから、NiceGUI の画面移動で移る。"""
    if force:
        ui.run_javascript("window.amamizuDirty = false;")
        ui.navigate.to(url)
    else:
        ui.run_javascript(f"amamizuGo({json.dumps(url)}, false)")


def num(label: str, value, on_change, *, suffix: str = "", width: str = "w-36", **kw):
    """数値の欄。空欄は None。"""
    return ui.number(label=label, value=value, on_change=lambda e: on_change(e.value), suffix=suffix or None,
                     **kw).classes(width).props("dense outlined")


FETCH_HINT = (f"実測降雨は、studies/_jma/ に取得済みの地点から選びます。ほかの地点・期間は、先に {fetch_command('tenmin')} "
              f"（時間値は {fetch_command('hourly', start='開始日', end='終了日')}）で取得してください。")


def date_input(label: str, value, days: list[str], on_change, marker: str) -> None:
    """日付の欄。入力もでき、右のアイコンから days（取得済みの日。YYYY-MM-DD）だけを選べるカレンダーを開く。
    on_change(値) は欄を変えたとき（空欄は None）。"""
    inp = ui.input(label, value=value or "", placeholder="2013-10-15",
                   on_change=lambda e: on_change(e.value.strip() or None)).classes("w-40").props("dense outlined").mark(marker)
    if not days:
        return
    with inp:
        with ui.menu().props("no-parent-event") as menu:
            cal = ui.date(value=value, on_change=lambda e: (inp.set_value(e.value or ""), menu.close()))
            cal._props["options"] = [d.replace("-", "/") for d in days]          # Quasar の QDate は YYYY/MM/DD で選べる日を受け取る
            cal._props["default-year-month"] = (value if value in days else days[-1])[:7].replace("-", "/")
    with inp.add_slot("append"):
        ui.icon("event").on("click", menu.open).classes("cursor-pointer")


def observed_fields(spec: dict, studies: Path, on_change) -> None:
    """実測降雨の条件の欄（値の種類、地点、取得済みの期間、開始日・終了日）。検討の画面の流入と、降雨のページの降雨記録が使う。
    何も自動では選ばない。on_change(項目と値の辞書, 描き直すか) は欄を変えたとき（値が None の項目は削除する）。
    期間を選ぶと開始日と終了日を一度に変える（片方だけ変えた途中の状態で確かめない）。"""
    ui.label(FETCH_HINT).classes("text-sm text-gray-600")
    try:
        store = JmaStore(studies)
    except (FileNotFoundError, ValueError) as e:
        ui.label(f"⚠ {e}").classes(f"text-{NG_COLOR} text-sm")
        return
    series = spec.get("series")
    ui.toggle({"tenmin": "10分値", "hourly": "時間値"}, value=series,
              on_change=lambda e: on_change({"series": e.value}, True)).mark("observed_series")
    stations = store.stations_with(series) if series in OBSERVED_SERIES else []
    station = spec.get("station")
    if series is None:
        ui.label("値の種類（10分値か時間値）を選んでください。").classes("text-sm")
    elif not stations:
        ui.label(f"studies/_jma/ に{OBSERVED_SERIES[series]}を取得した地点がありません。上の amamizu fetch で取得してください。"
                 ).classes(f"text-{NG_COLOR} text-sm")
    periods, error = [], None
    if series in OBSERVED_SERIES and station in stations:
        try:
            periods = fetched_periods(store, series, [station])[station]
        except (FileNotFoundError, ValueError) as e:    # 同じ地点名の観測所が2つある、値のファイルが読めない
            error = str(e)
    names = stations + ([station] if station and station not in stations else [])   # 取得していない地点も隠さずに示す
    try:
        prefs = {s: store.pref(s) for s in names}
    except (OSError, ValueError) as e:                  # station.json を読めない
        prefs = {}
        ui.label(f"⚠ 地点の都道府県を読めません（地点名だけを表示します）: {e}").classes(f"text-{NG_COLOR} text-sm")
    with ui.row().classes("gap-2 items-end"):
        options = {s: f"{s}（{prefs[s]}）" if prefs.get(s) else s for s in names}       # 選ぶ値は地点名、表示は「地点（都道府県）」
        ui.select(options, value=station, label="地点", on_change=lambda e: on_change({"station": e.value}, True)).classes(
            "w-56").props("dense outlined").mark("observed_station")
        if periods:
            choices = {f"{p.start}〜{p.end}": period_label(p) for p in periods}
            chosen = f"{spec.get('start')}〜{spec.get('end')}"
            ui.select(choices, value=chosen if chosen in choices else None, label="取得済みの期間（新しい順。入力して絞り込めます）",
                      with_input=True, on_change=lambda e: e.value and on_change(dict(zip(("start", "end"), e.value.split("〜"))), True)
                      ).classes("w-[26rem]").props("dense outlined").mark("observed_period")
    if error:
        ui.label(f"⚠ {station}の{OBSERVED_SERIES[series]}の取得済みの期間を読めません: {error}").classes(f"text-{NG_COLOR} text-sm")
    days = fetched_days(periods)
    with ui.row().classes("gap-2 items-end"):
        for key, label in (("start", "開始日"), ("end", "終了日")):
            date_input(label, spec.get(key), days, lambda v, k=key: on_change({k: v}, False), f"observed_{key}")
    if periods:
        ui.label(periods_summary(station, series, periods) + "取得済みの期間から選ぶか、開始日・終了日を入力してください"
                 "（取得していない日は0mm として計算し、要約に示します）。").classes("text-sm text-gray-700")


# ============================================================================ 一覧
def home_page(studies: Path):
    ui.label("検討の一覧").classes("text-xl font-bold")
    ui.label(f"フォルダ: {studies}").classes("text-sm text-gray-600")
    with ui.row().classes("items-center gap-2 mt-2"):
        ui.button("貯留追跡計算を行う", icon="add", on_click=lambda: new_dialog(studies).open()).props(
            "color=primary").tooltip("新しい検討のフォルダを作り、池・放流施設・降雨の条件を入力して貯留追跡計算（厳密計算）を行います")
    box = ui.column().classes("w-full gap-2 mt-2")
    study_list(studies, box)


def study_row(d: Path) -> dict:
    """一覧の1行（題名、種類、作成日時、最後の出力の日時、study.json の誤り）。"""
    try:
        conf = read_study(d)
        kind, title, err = conf.get("kind"), conf.get("title", ""), ""
    except ValueError as e:
        conf, kind, title, err = None, "", "", str(e)
    except OSError as e:
        conf, kind, title, err = None, "", "", f"study.json を読めません（{e}）。"
    if not err:
        try:
            check_kind(conf)
        except ValueError as e:
            err = str(e)
    if not err and kind == "design":
        err = created_error(conf) or ""
    out = d / "out"
    last = max((p.stat().st_mtime for p in out.iterdir()), default=None) if out.is_dir() else None
    created = (f"作成 {datetime.datetime.fromtimestamp(created_at(d, conf)):%Y-%m-%d %H:%M}　"
               if kind == "design" else "")
    return {"name": d.name, "title": str(title or d.name), "kind": KIND_LABEL.get(kind, kind or "―"),
            "design": kind == "design", "created": created, "order": list_order(d, conf),
            "last": datetime.datetime.fromtimestamp(last).strftime("%Y-%m-%d %H:%M") if last else "未計算", "err": err}


def study_list(studies: Path, box):
    """検討の一覧（削除したら作り直す）。並びは state.list_order（検証の例（降雨強度の検証、貯留施設の検証）が先頭、その後は作成日の新しい順）。"""
    rows = []
    for d in (p for p in Path(studies).iterdir() if (p / "study.json").exists()) if Path(studies).exists() else []:
        try:
            rows.append(study_row(d))
        except Exception as e:      # 想定外のフォルダ1つで、一覧全体を表示できなくしない（理由はその行に示す）
            rows.append({"name": d.name, "title": d.name, "kind": "―", "design": False, "created": "",
                         "order": list_order(d, None), "last": "", "err": f"この検討を一覧に表示できません（{type(e).__name__}: {e}）。"})
    rows.sort(key=lambda r: r["order"])
    box.clear()
    with box:
        if not rows:
            ui.label("検討がありません。「貯留追跡計算を行う」から作ってください。").classes("mt-2")
        for r in rows:
            with ui.card().classes("w-full").props("flat bordered"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.column().classes("gap-0"):
                        ui.label(r["title"]).classes("text-lg font-medium")
                        ui.label(f"{r['kind']}　フォルダ {r['name']}　{r['created']}最後の出力 {r['last']}").classes(
                            "text-sm text-gray-600")
                        if r["err"]:
                            ui.label(r["err"]).classes(f"text-{NG_COLOR}")
                    with ui.row().classes("items-center no-wrap gap-2"):     # 幅を固定し、「開く」の位置をどの行でもそろえる
                        ui.button("開く", icon="folder_open",
                                  on_click=lambda n=r["name"]: ui.navigate.to(f"/study/{quote(n)}")).props("outline").mark(
                            f"open:{r['name']}").tooltip(
                            "設計の検討は条件を編集できます。貯留施設の検証・降雨強度の検証は結果を見るだけです")
                        if r["design"]:
                            study_menu(studies, r["name"], "", lambda: study_list(studies, box))
                        else:
                            ui.element("div").classes("w-8")                  # 「⋮」の分の空き（「開く」の位置をそろえる）


def study_menu(studies: Path, name: str, target: str, on_deleted, form: DesignForm | None = None):
    """検討の「⋮」のメニュー（一覧の行と、検討の画面の上部）。複製と削除。削除はふだん隠しておき（赤いボタンが常に見えると
    目立つため）、メニューを開いたときだけ示す。target は項目名の「この検討を」の部分（一覧の行では空）。
    form は検討の画面で編集中の条件（保存していない変更の扱いを複製のダイアログで選ぶ）。"""
    with ui.button(icon="more_vert").props("flat round dense color=grey-7").mark(f"more:{name}"):
        with ui.menu().props("anchor='bottom right' self='top right'"):
            with ui.menu_item(on_click=lambda: duplicate_dialog(studies, name, form)).mark(f"duplicate:{name}"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.icon("content_copy")
                    with ui.column().classes("gap-0"):
                        ui.label(f"{target}複製する")
                        ui.label("study.json と input/ をコピーした新しい検討を作って開く").classes("text-xs text-gray-600")
            with ui.menu_item(on_click=lambda: delete_dialog(studies, name, on_deleted)).mark(f"delete:{name}"):
                with ui.row().classes("items-center gap-2 no-wrap text-negative"):
                    ui.icon("delete")
                    with ui.column().classes("gap-0"):
                        ui.label(f"{target}削除する")
                        ui.label("OS のごみ箱へ移す削除（元に戻せます）").classes("text-xs text-gray-600")


def duplicate_dialog(studies: Path, name: str, form: DesignForm | None = None):
    """複製の確認。新しい検討の題名を入力してもらい、複製したら新しい検討を開く。
    検討の画面に保存していない変更があれば、その変更を失わない2つの方法から選ぶ:
    変更を複製に含める（この検討は保存した状態のまま）、この検討に保存してから複製する。"""
    try:
        title0 = read_study(Path(studies) / name).get("title") or name
    except (ValueError, OSError) as e:
        ui.notify(str(e), type="negative")
        return
    dirty = form is not None and form.dirty
    with ui.dialog() as dialog, ui.card().classes("w-[34rem]"):
        ui.label("検討を複製する").classes("text-lg font-bold")
        ui.label(f"「{title0}」の条件（study.json と input/ の表）をコピーして新しい検討を作り、開きます。"
                 "out/ はコピーしません。").classes("text-sm")
        if dirty:
            ui.label("この検討には保存していない変更があります。変更をどちらに残すかを選んでください。").classes(
                "text-sm font-medium text-orange-800")
            ui.label("・変更を含めて複製: 新しい検討に今の条件を保存します。この検討は保存した状態のままです。").classes("text-sm")
            ui.label("・保存してから複製: この検討に今の条件を保存してから、同じ条件で新しい検討を作ります。").classes("text-sm")
        title = ui.input("新しい検討の題名（フォルダ名にも使います）").classes("w-full").props(
            "autofocus").mark("duplicate_title")
        preview = ui.label("").classes("text-sm text-gray-600")
        err = ui.label("").classes(f"text-{NG_COLOR} text-sm")

        def show():
            try:
                preview.text = f"作るフォルダ: {new_study_dir(studies, title.value or '')}"
                err.text = ""
            except ValueError as e:
                preview.text, err.text = "", str(e)
        title.on_value_change(lambda _: show())

        def run(how: str):
            """how: saved（保存した条件）、edits（変更を含める）、save_first（保存してから）。"""
            try:
                new_study_dir(studies, title.value or "")             # 題名の誤りは、保存する前に示す
                if how == "edits":
                    d = form.duplicate(title.value or "")
                else:
                    if how == "save_first":
                        if [ch for ch in form.table_changes() if ch.lost]:
                            raise ValueError("池の表の保存で、元の CSV から失われるものがあります。"
                                             "先に「保存」で内容を確かめてから複製してください。")
                        form.save()
                    d = duplicate_study(studies, name, title.value or "")
            except (ValueError, OSError) as e:
                err.text = str(e)
                return
            dialog.close()
            ui.notify(f"「{title.value.strip()}」を作りました。開きます。", type="positive")
            go(f"/study/{quote(d.name)}", force=True)     # 保存していない変更は、複製に含めたか保存したので失われない
        with ui.row().classes("justify-end w-full"):
            ui.button("やめる", on_click=dialog.close).props("flat")
            if dirty:
                ui.button("保存してから複製", icon="save", on_click=lambda: run("save_first")).props("outline").mark(
                    "duplicate_save_first")
                ui.button("変更を含めて複製", icon="content_copy", on_click=lambda: run("edits")).props(
                    "color=primary").mark("duplicate_edits")
            else:
                ui.button("複製して開く", icon="content_copy", on_click=lambda: run("saved")).props("color=primary").mark(
                    "duplicate_ok")
    dialog.open()


def delete_dialog(studies: Path, name: str, on_done):
    """削除の確認。削除（OS のごみ箱へ移す。移せなければ確かめてから完全に削除）したら on_done を呼ぶ
    （一覧は作り直し、検討の画面は一覧へ戻る）。"""
    try:
        plan = delete_plan(studies, name)
    except (ValueError, OSError) as e:
        ui.notify(str(e), type="negative")
        return
    with ui.dialog() as dialog, ui.card().classes("w-[34rem]"):
        ui.label("検討を削除する").classes("text-lg font-bold")
        ui.label(f"「{plan.title}」を削除します。").classes("font-medium")
        ui.label(f"フォルダ: {plan.path}").classes("text-sm text-gray-600 break-all")
        ui.label(f"中身: {'、'.join(plan.entries)}（ファイル {plan.files} 個）").classes("text-sm")
        if plan.has_out:
            ui.label("計算結果の out/ も一緒に移ります。").classes("text-sm")
        ui.label("フォルダは OS のごみ箱へ移ります（ごみ箱から元に戻せます）。").classes("text-sm text-gray-600")
        # ごみ箱へ移せなかったときだけ表示する（完全な削除は元に戻せないので、黙って切り替えない）
        fallback = ui.column().classes("w-full gap-1")
        fallback.set_visibility(False)

        def run(permanent=False):
            try:
                delete_study(plan, permanent=permanent)
            except OSError as e:
                if permanent:
                    ui.notify(f"削除できませんでした（{e}）。フォルダ内のファイルを開いているアプリがあれば閉じてから、もう一度削除してください。",
                              type="negative", multi_line=True)
                    return
                fallback.clear()
                with fallback:
                    ui.label(f"ごみ箱へ移せませんでした: {e}").classes(f"text-{NG_COLOR} text-sm break-all")
                    ui.label("完全に削除することはできます。ごみ箱には残らず、元に戻せません。").classes("text-sm")
                    ui.button("完全に削除する", icon="delete_forever", on_click=lambda: run(permanent=True)).props(
                        "color=negative").mark("delete_permanent")
                fallback.set_visibility(True)
                return
            dialog.close()
            done = "完全に削除しました" if permanent else "OS のごみ箱へ移しました"
            ui.notify(f"「{plan.title}」を削除しました（{done}）。", type="positive")
            on_done()
        with ui.row().classes("justify-end w-full"):
            ui.button("やめる", on_click=dialog.close).props("flat")
            ui.button("削除する", icon="delete", on_click=lambda: run()).props("color=negative").mark("delete_ok")
    dialog.open()


def new_dialog(studies: Path):
    with ui.dialog() as dialog, ui.card().classes("w-[32rem]"):
        ui.label("設計の検討を新しく作る").classes("text-lg font-bold")
        title = ui.input("題名（フォルダ名にも使います）").classes("w-full").props("autofocus").mark("new_title")
        preview = ui.label("").classes("text-sm text-gray-600")
        err = ui.label("").classes(f"text-{NG_COLOR} text-sm")

        def show():
            try:
                preview.text = f"作るフォルダ: {new_study_dir(studies, title.value or '')}"
                err.text = ""
            except ValueError as e:
                preview.text, err.text = "", str(e)
        title.on_value_change(lambda _: show())

        def create():
            try:
                d = create_design_study(studies, title.value or "")
            except ValueError as e:
                err.text = str(e)
                return
            dialog.close()
            ui.navigate.to(f"/study/{quote(d.name)}")
        with ui.row().classes("justify-end w-full"):
            ui.button("やめる", on_click=dialog.close).props("flat")
            ui.button("作る", on_click=create).props("color=primary").mark("create")
    return dialog


# ============================================================================ 検討
def study_page(studies: Path, name: str):
    d = Path(studies) / unquote(name)
    if not (d / "study.json").exists():
        ui.label(f"検討が見つかりません（study.json がありません）: {d}。一覧へ戻って選び直してください。").classes(f"text-{NG_COLOR}")
        ui.link("一覧へ", "/")
        return
    try:
        conf = read_study(d)
    except ValueError as e:
        cannot_open(d, str(e))
        return
    except OSError as e:
        cannot_open(d, f"study.json を読めません（{e}）。")
        return
    try:
        kind = check_kind(conf)
    except ValueError as e:
        cannot_open(d, str(e), conf)
        return
    shape = shape_errors(conf)
    if shape:
        cannot_open(d, f"study.json の次の項目の形が違います: {', '.join(shape)}（「a.b」は a の中の b）。"
                       "テキストエディタで直してから開き直してください。", conf)
        return
    body = ui.column().classes("w-full gap-2")
    try:
        with body:
            if kind == "design":
                DesignPage(d).build()
            else:
                viewer_page(d, check_study(conf))       # 見るだけの検討は、amamizu study と同じ確かめを通す
    except Exception as e:       # 想定外の study.json で 500 にしない（理由を示し、コマンドで確かめられるようにする）
        body.clear()
        cannot_open(d, f"この検討を WebGUI で表示できません（{type(e).__name__}: {e}）。"
                       f"amamizu study {d} を実行すると、誤りを確かめられます。", conf)


def cannot_open(d: Path, message: str, conf: dict | None = None):
    """検討を開けないときの画面（理由と直し方。読めた study.json はそのまま見せる）。"""
    with ui.row().classes("items-center gap-2"):
        ui.button(icon="arrow_back", on_click=lambda: go("/")).props("flat round dense")
        ui.label(d.name).classes("text-xl font-bold")
    ui.label(f"フォルダ {d}").classes("text-sm text-gray-600")
    ui.label(message).classes(f"text-{NG_COLOR}").mark("cannot_open")
    if conf is not None:
        ui.code(json.dumps(conf, ensure_ascii=False, indent=2), language="json").classes("w-full")


def observation_changes_box(changes: list[str], redo: str):
    """out/ の出力の後に変わった、使った観測値（app/observations.observation_changes）。redo は作り直し方。変化がなければ何も表示しない
    （取り直して変わった観測値の結果を、今の値の結果として黙って見せない）。"""
    if not changes:
        return
    with ui.expansion(f"out/ の出力の後に、使った観測値（studies/_jma/）が変わっています（{changed_count(changes)} 件）",
                      icon="warning").classes("text-sm text-orange-800").props("dense").mark("observation_changes"):
        for line in changes:
            ui.label(line).classes("text-sm")
        ui.label(redo).classes("text-sm text-gray-600")


class DesignPage:
    """設計の検討の画面。"""

    def __init__(self, study_dir: Path):
        self.form = DesignForm.load(study_dir)
        self.trials: list[dict] = []          # 試算の結果（画面を開いている間保持する）
        self.pending = False                  # 入力が変わり、確かめ直しが要る
        self.busy = False
        self.tabs: dict[str, object] = {}
        self.errors: dict[str, object] = {}
        self.previews: dict[str, object] = {}
        self.inputs: dict[str, object] = {}         # 条件のタブの入力の欄（読み戻しで中身だけ作り直す）
        self.dirty_sent = None                       # ブラウザに伝えた保存状態

    # --- 共通 ---------------------------------------------------------------
    @property
    def conf(self) -> dict:
        return self.form.conf

    def changed(self, *_):
        self.pending = True

    def setter(self, container: dict, key: str, drop_none: bool = True):
        def f(v):
            if v is None or v == "":
                if drop_none:
                    container.pop(key, None)
            else:
                container[key] = v
            self.changed()
        return f

    # --- 画面の組み立て ------------------------------------------------------
    def build(self):
        self.root = ui.column().classes("w-full gap-2")
        self.body()
        ui.timer(0.4, self.tick)

    def body(self):
        f = self.form
        self.root.clear()
        with self.root:
            self._body(f)
        self.render_unknown()
        self.refresh()

    def _body(self, f):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                with ui.row().classes("items-center gap-2"):
                    ui.button(icon="arrow_back", on_click=lambda: go("/")).props("flat round dense").tooltip(
                        "検討の一覧に戻ります（保存していない変更があれば確かめます）")
                    self.title_label = ui.label(self.conf.get("title") or f.study_dir.name).classes("text-xl font-bold")
                    self.status = ui.badge("").props("outline")
                ui.label(f"フォルダ {f.study_dir}").classes("text-sm text-gray-600 ml-12")
                self.output_box = ui.column().classes("gap-0 ml-12")     # out/ の出力と今の条件の違い（render_output_state）
                self.output_shown = ()
            with ui.row().classes("gap-2"):
                # 流入の種類に合わせて表示を変える（計画降雨: 計画降雨の画面、実測降雨・ハイドログラフ: 使うデータを示し流入のタブへ）
                self.source_btn = ui.button("計画降雨", icon="water_drop", on_click=self.open_source).props(
                    "outline no-caps").mark("inflow_source_button")
                with self.source_btn:
                    self.source_tip = ui.tooltip("")
                self.trial_btn = ui.button("試算", icon="play_arrow", on_click=self.run_trial).props("color=primary").mark("trial").tooltip(
                    "今の条件で計算し、結果を画面に表示します。ファイルには保存しません。結果は「試算の比較」に残ります")
                self.save_only_btn = ui.button("保存", icon="save", on_click=lambda: self.save(output=False)).props(
                    "outline").mark("save_only").tooltip(
                    "study.json と input/ の池の表を保存します。計算と出力はしません")
                self.save_btn = ui.button("保存して出力", icon="save_as", on_click=lambda: self.save(output=True)).props(
                    "color=secondary").mark("save").tooltip(
                    "study.json と input/ の池の表を保存してから計算し、表・要約・グラフを out/ に出力します（amamizu study と同じ出力）")
                with ui.row().classes("items-center gap-1") as self.busy_note:
                    ui.spinner(size="sm")
                    ui.label("計算中です（オリフィスの寸法を求めるときは十数秒かかります）").classes("text-sm")
                self.busy_note.set_visibility(False)
                study_menu(f.study_dir.parent, f.study_dir.name, "この検討を", lambda: go("/", force=True), form=f)
        self.unknown_box = ui.column().classes("w-full")
        with ui.tabs().classes("w-full").props("align=left dense") as tabs:
            for s in SECTIONS:
                self.tabs[s] = ui.tab(s)
            self.tabs["結果"] = ui.tab("結果", icon="insights").tooltip("最後に行った試算、または保存して出力の結果")
            self.tabs["試算の比較"] = ui.tab("試算の比較", icon="compare").tooltip(
                "画面を開いてからの試算を並べ、主な値・変えた条件・水位のグラフを比較します。選んだ試算の条件を読み戻せます")
        with ui.tab_panels(tabs, value=self.tabs["全体"], **TAB_PANELS).classes("w-full") as self.panels:
            self.builders = dict(zip(SECTIONS, (self.general, self.inflow, self.pond, self.outlets, self.infiltration,
                                                self.run_options)))
            for s, builder in self.builders.items():
                with ui.tab_panel(self.tabs[s]):
                    self.errors[s] = ui.label("").classes(f"text-{NG_COLOR} font-medium")
                    with ui.row().classes(TWO_COLUMNS):
                        self.inputs[s] = ui.column().classes("w-[46%] min-w-[26rem] gap-2")
                        with self.inputs[s]:
                            builder()
                        self.previews[s] = ui.column().classes(f"{CHART_COLUMN} gap-2")
            with ui.tab_panel(self.tabs["結果"]):
                self.result_box = ui.column().classes("w-full")
                with self.result_box:
                    ui.label("「試算」を押すと、ここに結果が表示されます。").classes("text-gray-600")
            with ui.tab_panel(self.tabs["試算の比較"]):
                self.compare_box = ui.column().classes("w-full")
                self.render_compare()

    def tick(self):
        if self.pending:
            self.pending = False
            self.refresh()

    # --- 各タブの入力 --------------------------------------------------------
    def general(self):
        c = self.conf
        inp = ui.input("題名", value=c.get("title", "")).classes("w-full").props("dense outlined")

        def set_title(e):
            c["title"] = e.value
            self.title_label.text = e.value or self.form.study_dir.name
            self.changed()
        inp.on_value_change(set_title)
        names = guideline_names(self.form.studies)
        chosen = NO_GUIDELINE if self.form.no_guideline else c.get("guideline")
        if isinstance(chosen, str) and chosen not in (*names, NO_GUIDELINE):
            names.append(chosen)                 # .json のパスで書いた手引の定数
        with ui.row().classes("items-center gap-3 no-wrap"):
            ui.select([*names, NO_GUIDELINE], value=chosen if isinstance(chosen, str) else None, label="手引の定数",
                      on_change=lambda e: self.switch_guideline(e.value)).classes("w-64").props("dense outlined").mark(
                "guideline")
            ui.label(f"{guideline_dir(self.form.studies).name}/ のファイルから選びます。"
                     "「使用しない」では、放流の係数と強度式の係数を入力します").classes("text-sm text-gray-600")
        self.hydraulics_box = ui.column().classes("w-full gap-1")
        self.render_hydraulics()
        num("洪水到達時間 tc（計算の刻み）", c.get("tc_min", DEFAULT_TC_MIN), self.setter(c, "tc_min"), suffix="分", step=1, min=1,
            width="w-64").mark("tc")

    def switch_guideline(self, value):
        notice = self.form.set_guideline(None if value == NO_GUIDELINE else value)
        if notice:
            ui.notify(notice, type="warning", multi_line=True, timeout=0, close_button="閉じる")
        self.render_hydraulics()
        if "rain" in self.conf:
            self.render_inflow_form()
        self.render_infiltration()
        self.changed()

    def render_hydraulics(self):
        """手引を使用しない検討の、放流の係数の入力欄。"""
        self.hydraulics_box.clear()
        if not self.form.no_guideline:
            return
        hy = self.conf.setdefault("hydraulics", {})
        with self.hydraulics_box:
            ui.label("放流の係数").classes("font-medium")
            for key, label, unit, note in HYDRAULICS_FIELDS:
                with ui.row().classes("items-center gap-3 no-wrap"):
                    num(label, hy.get(key), self.setter(hy, key), suffix=unit, format="%.4g", width="w-56").mark(key)
                    ui.label(note).classes("text-sm text-gray-600")
            ui.label("初期値は一般的な値です。使用する基準の値を確認して入力してください。").classes("text-sm text-gray-600")

    def inflow(self):
        c = self.conf
        src = next((k for k in INFLOW_SOURCES if k in c), None)
        ui.toggle({"rain": "計画降雨", "observed_rain": "実測降雨", "inflow_csv": "ハイドログラフの CSV"}, value=src,
                  on_change=lambda e: self.switch_inflow(e.value)).mark("inflow_source").tooltip(
            "計画降雨: 強度式から作る。実測降雨: studies/_jma/ に取得済みの気象庁の10分値・時間値。"
            "ハイドログラフ: 流入量の表")
        self.inflow_form = ui.column().classes("w-full gap-2")
        self.render_inflow_form()

    def switch_inflow(self, src):
        c = self.conf
        if src not in c:
            # 実測降雨から切り替えるときは、その地点が属する地区を初期値にする（地点のモデル施設と同じ地区の計画降雨）
            district = (self.form.observed_district() if "observed_rain" in c else None) or self.form.default_district()
            for k in INFLOW_SOURCES:
                c.pop(k, None)
            if src == "rain":
                how = ({"formula": dict(SAMPLE_FORMULA)} if self.form.no_guideline
                       else {"guideline": {"district": district, "return_period": 50}})
                c["rain"] = how | {"shape": "rear"}
                c.setdefault("catchments", [{"name": "区域", "area_ha": 1.0, "f": 0.9}])
            elif src == "observed_rain":
                c["observed_rain"] = {}                 # 地点・値の種類・期間は利用者が選ぶ（自動では選ばない）
                c.setdefault("catchments", [{"name": "区域", "area_ha": 1.0, "f": 0.9}])
            else:
                c.pop("catchments", None)
                c["inflow_csv"] = {}                    # ファイルは利用者が選ぶ（input/ の最初の CSV を選ばない）
                self.form.load_hydrograph()
        self.render_inflow_form()
        self.render_infiltration_tab()
        self.changed()

    def input_files(self) -> list[str]:
        """input/ の CSV（ハイドログラフ・池の表のファイルの選択肢）。"""
        d = self.form.study_dir / "input"
        return sorted(f"input/{p.name}" for p in d.glob("*.csv")) if d.exists() else []

    def render_inflow_form(self):
        c = self.conf
        self.inflow_form.clear()
        with self.inflow_form:
            if "rain" in c:
                rain = c["rain"]
                how = "formula" if "formula" in rain or self.form.no_guideline else "guideline"
                term = self.form.district_term()
                if not self.form.no_guideline:       # 手引を使用しない検討は、係数の直接指定だけ
                    ui.toggle({"guideline": f"手引の式（{term}・確率年）", "formula": "5方式から選ぶ"}, value=how,
                              on_change=lambda e: self.switch_formula(e.value))
                if how == "guideline":
                    gl = rain.setdefault("guideline", {"district": self.form.default_district(), "return_period": 50})
                    districts, periods = guideline_choices(self.form.guideline(), gl)

                    def set_district(e):
                        gl["district"] = e.value
                        self.render_inflow_form()        # 確率年の選択肢は地区で決まる
                        self.changed()
                    with ui.row().classes("gap-2"):
                        ui.select(districts, value=gl.get("district"), label=term, on_change=set_district).classes(
                            "w-40").props("dense outlined")
                        ui.select(periods, value=gl.get("return_period"), label="確率年",
                                  on_change=lambda e: (gl.__setitem__("return_period", e.value), self.changed())
                                  ).classes("w-32").props("dense outlined")
                else:
                    rain.pop("guideline", None)
                    fo = rain.setdefault("formula", dict(SAMPLE_FORMULA))
                    if fo == SAMPLE_FORMULA:
                        ui.label(SAMPLE_FORMULA_NOTE).classes("text-sm text-gray-600")
                    ui.select({k: f"{v[0]}　r＝{v[1]}" for k, v in FORMULA_TYPES.items()}, value=fo.get("type"),
                              label="強度式の種類", on_change=lambda e: self.switch_type(e.value)).classes(
                        "w-80").props("dense outlined")
                    needed = FORMULA_TYPES.get(fo.get("type"), ("", "", ("a", "b", "n"), {}))[2]
                    with ui.row().classes("gap-2"):
                        for k in needed:
                            if k == "n":
                                ui.input("n（3/4 のような分数も可）", value=str(fo.get("n", "")),
                                         on_change=lambda e: (fo.__setitem__("n", self._n_value(e.value)),
                                                              self.changed())).classes("w-44").props("dense outlined")
                            else:
                                num(k, fo.get(k), self.setter(fo, k), width="w-32")
                with ui.row().classes("gap-2 items-end"):
                    ui.select({"rear": "後方集中型", "center": "中央集中型"}, value=rain.get("shape"), label="降雨波形",
                              on_change=lambda e: (rain.__setitem__("shape", e.value), self.changed())).classes(
                        "w-40").props("dense outlined")
                    num("継続時間", rain.get("hours", 24), self.setter(rain, "hours"), suffix="時間", step=1, width="w-32")
                    num("丸め（小数の桁。空欄は丸めない）", rain.get("round_digits"), self.setter(rain, "round_digits"),
                        step=1, min=0, width="w-56")
                ui.label("区域（流出係数の違う区域ごと）").classes("font-medium mt-2")
                self.catchment_box = ui.column().classes("w-full gap-1")
                self.render_catchments()
            elif "observed_rain" in c:
                self.render_observed_form(c["observed_rain"])
                ui.label("区域（流出係数の違う区域ごと）").classes("font-medium mt-2")
                self.catchment_box = ui.column().classes("w-full gap-1")
                self.render_catchments()
            elif "inflow_csv" in c:
                spec = c["inflow_csv"]
                files = self.input_files()
                if spec.get("file") and spec["file"] not in files:
                    files.append(spec["file"])
                ui.select(files, value=spec.get("file"), label="ハイドログラフの CSV（input/）",
                          on_change=lambda e: self.set_csv(e.value)).classes("w-80").props("dense outlined")
                cols = list(self.form.hydrograph.columns) if self.form.hydrograph is not None else []
                with ui.row().classes("gap-2"):
                    for key, label, default in (("time_col", "時刻の列（分）", "時刻_分"), ("q_col", "流入量の列（m^3/s）", "流入量_m3s")):
                        opts = cols if cols else [spec.get(key, default)]
                        ui.select(opts, value=spec.get(key, default) if spec.get(key, default) in opts else None,
                                  label=label, on_change=lambda e, k=key: (spec.__setitem__(k, e.value), self.changed())
                                  ).classes("w-48").props("dense outlined")
                ui.checkbox("値の並びを逆にする（強度の大きい順の表を後方集中型にする）", value=spec.get("reverse", False),
                            on_change=lambda e: (spec.__setitem__("reverse", True) if e.value else spec.pop("reverse", None),
                                                 self.changed()))
            else:
                ui.label("流入の種類（計画降雨、実測降雨、ハイドログラフの CSV）を上で選んでください。").classes("text-sm")

    def render_observed_form(self, spec):
        """実測降雨: studies/_jma/ に取得済みの地点、値の種類（10分値・時間値）、期間。"""
        observed_fields(spec, self.form.studies, lambda values, redraw: self.set_observed(spec, values, redraw))

    def set_observed(self, spec, values, redraw):
        for key, value in values.items():
            if value is None:
                spec.pop(key, None)
            else:
                spec[key] = value
        if redraw:
            self.render_inflow_form()
        self.changed()

    def _n_value(self, text):
        text = (text or "").strip()
        if not text:
            return None
        try:
            return int(text) if text.isdigit() else float(text)
        except ValueError:
            return text                      # "3/4" はそのまま（study.json にも分数の文字列で書く）

    def switch_formula(self, how):
        rain = self.conf["rain"]
        if how == "formula" and "formula" not in rain:
            rain.pop("guideline", None)
            rain["formula"] = dict(SAMPLE_FORMULA)
        elif how == "guideline" and "guideline" not in rain:
            rain.pop("formula", None)
            rain["guideline"] = {"district": self.form.default_district(), "return_period": 50}
        self.render_inflow_form()
        self.changed()

    def switch_type(self, kind):
        fo = self.conf["rain"]["formula"]
        fo["type"] = kind
        for k in FORMULA_TYPES[kind][3]:
            fo.pop(k, None)                  # その方式で使わない係数は消す（残すと誤りになる）
        for k in FORMULA_TYPES[kind][2]:
            fo.setdefault(k, SAMPLE_FORMULA[k])
        self.render_inflow_form()
        self.changed()

    def set_csv(self, file):
        self.conf["inflow_csv"]["file"] = file
        self.form.load_hydrograph()
        self.render_inflow_form()
        self.changed()

    def render_catchments(self):
        rows = self.conf.setdefault("catchments", [])
        self.catchment_box.clear()
        with self.catchment_box:
            for i, r in enumerate(rows):
                with ui.row(wrap=False).classes("gap-2 items-center"):
                    ui.input("名前", value=r.get("name", ""), on_change=lambda e, r=r: (r.__setitem__("name", e.value),
                                                                                       self.zones_changed())).classes(
                        "w-28").props("dense outlined")
                    num("面積 ha", r.get("area_ha"), lambda v, r=r: (self.setter(r, "area_ha")(v), self.zones_changed()),
                        width="w-24")
                    num("流出係数", r.get("f"), self.setter(r, "f"), step=0.01, width="w-20")
                    ui.button(icon="delete", on_click=lambda i=i: self.del_row(rows, i, self.catchments_changed)).props(
                        "flat round dense")
            ui.button("区域を追加", icon="add", on_click=lambda: self.add_row(
                rows, {"name": f"区域{len(rows) + 1}", "area_ha": 1.0, "f": 0.9}, self.catchments_changed)).props("flat dense")

    def catchments_changed(self):
        """区域を足した・消したとき。流入のタブの区域の行と、浸透のタブの区域ごとの欄を作り直す。"""
        self.render_catchments()
        self.zones_changed()

    def zones_changed(self):
        """区域の名前・面積が変わったとき。浸透が区域ごとなら、浸透のタブの見出しと欄を作り直す。"""
        if self.inf_scope == "zones":
            self.render_infiltration()
        self.changed()

    def add_row(self, rows, row, render):
        rows.append(row)
        render()
        self.changed()

    def del_row(self, rows, i, render):
        rows.pop(i)
        render()
        self.changed()

    def pond(self):
        c = self.conf
        pond = c.get("pond") or {}
        shape = "table" if "table" in pond else "prismatic" if "prismatic" in pond else None
        ui.toggle({"prismatic": "柱状（面積一定）", "table": "水位・面積・容量の表"}, value=shape,
                  on_change=lambda e: self.switch_pond(e.value))
        self.pond_form = ui.column().classes("w-full gap-2")
        self.render_pond_form()

    def switch_pond(self, shape):
        p = self.conf.setdefault("pond", {})
        if shape == "table" and "table" not in p:
            p.pop("prismatic", None)
            p["table"] = {"file": "input/pond.csv"}
            try:
                self.form.pond_rows = pond_rows_of(read_table(self.form.study_dir, "input/pond.csv"), p["table"])
            except (FileNotFoundError, ValueError) as e:
                if not isinstance(e, FileNotFoundError):      # ファイルはあるが読めない。初期値の表にしたことを知らせる
                    ui.notify(f"input/pond.csv を読めなかったので、池の表を初期値にしました（{e}）。",
                              type="warning", multi_line=True, timeout=0, close_button="閉じる")
                self.form.pond_rows = [{POND_COLUMNS[0]: 0.0, POND_COLUMNS[1]: 500.0, POND_COLUMNS[2]: 0.0},
                                       {POND_COLUMNS[0]: 2.0, POND_COLUMNS[1]: 600.0, POND_COLUMNS[2]: 1100.0}]
        elif shape == "prismatic" and "prismatic" not in p:
            p.pop("table", None)
            p["prismatic"] = {"area_m2": 600.0, "hwl_m": 2.0}
            self.form.pond_rows = None
        self.render_pond_form()
        self.changed()

    def render_pond_form(self):
        p = self.conf.get("pond") or {}
        self.pond_form.clear()
        with self.pond_form:
            if "prismatic" in p:
                pr = p["prismatic"]
                with ui.row().classes("gap-2"):
                    num("面積", pr.get("area_m2"), self.setter(pr, "area_m2"), suffix="m^2")
                    num("満水位（深さ）", pr.get("hwl_m"), self.setter(pr, "hwl_m"), suffix="m", step=0.1)
                return
            if "table" not in p:
                ui.label("池の形（柱状、または水位・面積・容量の表）を上で選んでください。").classes("text-sm")
                return
            spec = p["table"]
            with ui.row().classes("items-center gap-2"):
                ui.input("保存先の CSV（input/）", value=spec.get("file", "input/pond.csv"),
                         on_change=lambda e: (spec.__setitem__("file", e.value), self.changed())).classes("w-80").props(
                    "dense outlined")
                ui.button("CSV に保存", icon="save", on_click=lambda: self.save_pond_csv()).mark("save_pond").props(
                    "flat dense").tooltip("下の池の表を保存先の CSV にすぐ保存します（study.json は保存しません）")
            num("満水位（空欄は表の最上段）", spec.get("hwl_m"), self.setter(spec, "hwl_m"), suffix="m", width="w-56")
            files = self.input_files()
            if files:
                with ui.row().classes("items-center gap-2"):
                    pick = ui.select(files, label="CSV から読み込む", value=None).classes("w-64").props("dense outlined")
                    ui.button("読み込む", on_click=lambda: self.import_pond(pick.value)).props("flat dense").tooltip(
                        "選んだ CSV の水位・面積・容量で下の表を置き換えます（CSV は書き換えません）")
            cols = pond_columns(spec)
            rows = self.form.pond_rows or []       # 表を読めないときは空で示す（開いただけで条件を書き換えない）
            ui.label("池の表（上の行から順に。水位と容量は下の行ほど大きく）").classes("font-medium")
            ui.label("面積と容量は、どちらか一方を列ごと空欄にできます（容量が空欄なら、面積から平均断面法で求めます）。"
                     ).classes("text-sm text-gray-600")
            for i, r in enumerate(rows):
                with ui.row(wrap=False).classes("gap-2 items-center"):
                    for col, label in zip(cols, ("水位 m", "面積 m^2", "容量 m^3")):
                        num(label, r.get(col), self.setter(r, col), width="w-28")
                    ui.button(icon="delete", on_click=lambda i=i: self.del_row(rows, i, self.render_pond_form)).props(
                        "flat round dense")
            ui.button("行を追加", icon="add", on_click=lambda: self.add_pond_row(cols)).props("flat dense")

    def add_pond_row(self, cols):
        """池の表に行を足す。前の行の値を写す（空欄の列は空欄のまま）。表がなければ作る。"""
        if self.form.pond_rows is None:
            self.form.pond_rows = []
        rows = self.form.pond_rows
        self.add_row(rows, {c_: rows[-1][c_] for c_ in cols if c_ in rows[-1]} if rows else {c_: 0.0 for c_ in cols},
                     self.render_pond_form)

    def import_pond(self, file):
        if not file:
            return
        try:                                                  # 水位の列がなければ ValueError（列の並びから推測しない）
            rows = pond_rows_of(read_table(self.form.study_dir, file), self.conf["pond"]["table"])
        except (FileNotFoundError, ValueError) as e:
            ui.notify(str(e), type="negative", multi_line=True)
            return
        self.form.pond_rows = rows                            # 面積・容量の列がなければ空欄（どちらか一方は省ける）
        self.render_pond_form()
        self.changed()

    def outlets(self):
        c = self.conf
        ui.label("オリフィス（2つ入力すれば2段のオリフィス。放流量は和）").classes("font-medium")
        self.orifice_box = ui.column().classes("w-full gap-1")
        self.render_orifices()
        ui.separator()
        sp_on = ui.checkbox("洪水吐", value="spillway" in c, on_change=lambda e: self.toggle_part("spillway", e.value,
                                                                                            {"crest_m": 2.0, "width_m": 1.0}))
        self.spill_box = ui.row().classes("gap-2")
        self.render_spillway()
        ui.separator()
        ui.checkbox("ポンプ（水位と排水量の表）", value="pump" in c,
                    on_change=lambda e: self.toggle_part("pump", e.value, {"levels_m": [0.5, 1.0], "q_m3s": [0.01, 0.02]}))
        self.pump_box = ui.column().classes("gap-1")
        self.render_pump()
        return sp_on

    def holes(self) -> list:
        """オリフィスの並び（表示用。条件は書き換えない。足すのは add_hole）。"""
        c = self.conf
        if "orifice" in c:
            return [c["orifice"]]
        return c.get("orifices") or []

    def render_orifices(self):
        c = self.conf
        size_mode = c.get("mode") == "size_orifice"
        holes = self.holes()
        self.orifice_box.clear()
        with self.orifice_box:
            for i, h in enumerate(holes):
                with ui.row().classes("gap-2 items-center"):
                    ui.select({"rect": "長方形", "circle": "円形"}, value=h.get("shape", "rect"), label="形",
                              on_change=lambda e, h=h: self.set_shape(h, e.value)).classes("w-28").props("dense outlined")
                    num("敷高", h.get("sill_m"), self.setter(h, "sill_m", drop_none=False), suffix="m", width="w-28")
                    if size_mode:
                        ui.label("寸法は試算で求めます").classes("text-gray-600")
                    elif h.get("shape", "rect") == "circle":
                        num("直径", h.get("diameter_m"), self.setter(h, "diameter_m"), suffix="m", step=0.01, width="w-28")
                    else:
                        num("幅", h.get("width_m"), self.setter(h, "width_m"), suffix="m", step=0.01, width="w-28")
                        num("高さ", h.get("height_m"), self.setter(h, "height_m"), suffix="m", step=0.01, width="w-28")
                    if len(holes) > 1:
                        ui.button(icon="delete", on_click=lambda i=i: self.del_hole(i)).props("flat round dense")
            if not size_mode:
                ui.button("オリフィスを追加（2段）", icon="add", on_click=self.add_hole).props("flat dense").tooltip(
                    "敷高の違うオリフィスを足します。放流量は各オリフィスの和です")

    def set_shape(self, h, shape):
        for k in ("width_m", "height_m", "diameter_m"):
            h.pop(k, None)
        if shape == "circle":
            h["shape"] = "circle"
            if self.conf.get("mode") != "size_orifice":
                h["diameter_m"] = 0.1
        else:
            h.pop("shape", None)
            if self.conf.get("mode") != "size_orifice":
                h["width_m"] = h["height_m"] = 0.1
        self.render_orifices()
        self.changed()

    def add_hole(self):
        """オリフィスを1つ足す。1つ目は orifice、2つ目からは orifices の並びにする（study.json の書き方）。"""
        c = self.conf
        holes = self.holes()
        top = max((h.get("sill_m") or 0.0) for h in holes) if holes else None
        new = {"sill_m": 0.0 if top is None else round(top + 0.5, 3), "width_m": 0.1, "height_m": 0.1}
        if not holes:
            c.pop("orifices", None)
            c["orifice"] = new
        else:
            if "orifice" in c:
                c["orifices"] = [c.pop("orifice")]
            c["orifices"].append(new)
        self.render_orifices()
        self.changed()

    def del_hole(self, i):
        c = self.conf
        c["orifices"].pop(i)
        if len(c["orifices"]) == 1:
            c["orifice"] = c.pop("orifices")[0]
        self.render_orifices()
        self.changed()

    def toggle_part(self, key, on, default):
        if on:
            self.conf.setdefault(key, default)
        else:
            self.conf.pop(key, None)
        self.render_spillway()
        self.render_pump()
        self.changed()

    def render_spillway(self):
        self.spill_box.clear()
        sp = self.conf.get("spillway")
        if sp is None:
            return
        with self.spill_box:
            num("敷高", sp.get("crest_m"), self.setter(sp, "crest_m", drop_none=False), suffix="m", width="w-28")
            num("幅", sp.get("width_m"), self.setter(sp, "width_m", drop_none=False), suffix="m", width="w-28")

    def render_pump(self):
        self.pump_box.clear()
        pu = self.conf.get("pump")
        if pu is None:
            return
        with self.pump_box:
            for i in range(len(pu["levels_m"])):
                with ui.row().classes("gap-2 items-center"):
                    num("水位", pu["levels_m"][i], self._list_setter(pu["levels_m"], i), suffix="m", width="w-28")
                    num("排水量", pu["q_m3s"][i], self._list_setter(pu["q_m3s"], i), suffix="m^3/s", width="w-36")
                    ui.button(icon="delete", on_click=lambda i=i: self._del_pump(i)).props("flat round dense")
            ui.button("行を追加", icon="add", on_click=self._add_pump).props("flat dense")

    def _list_setter(self, lst, i):
        def f(v):
            lst[i] = v
            self.changed()
        return f

    def _add_pump(self):
        pu = self.conf["pump"]
        pu["levels_m"].append((pu["levels_m"][-1] or 0) + 0.5 if pu["levels_m"] else 0.5)
        pu["q_m3s"].append(pu["q_m3s"][-1] if pu["q_m3s"] else 0.01)
        self.render_pump()
        self.changed()

    def _del_pump(self, i):
        pu = self.conf["pump"]
        pu["levels_m"].pop(i)
        pu["q_m3s"].pop(i)
        self.render_pump()
        self.changed()

    def infiltration(self):
        self.inf_tab = ui.column().classes("w-full gap-2")
        self.render_infiltration_tab()

    def render_infiltration_tab(self):
        """浸透のタブ。上に「浸透なし／区域全体で1つ／区域ごと」（区域がない流入は「浸透なし／浸透あり」）、下に欄。"""
        c = self.conf
        zoned = [r for r in c.get("catchments", []) if "infiltration" in r]
        self.inf_scope = "whole" if "infiltration" in c else "zones" if zoned else "none"
        opts = ({"none": "浸透なし", "whole": "区域全体で1つ", "zones": "区域ごと"} if "catchments" in c
                else {"none": "浸透なし", "whole": "浸透あり"})
        self.inf_tab.clear()
        with self.inf_tab:
            ui.toggle(opts, value=self.inf_scope, on_change=lambda e: self.switch_scope(e.value)).mark("inf_scope")
            self.inf_box = ui.column().classes("w-full gap-2")
        self.render_infiltration()

    def switch_scope(self, scope):
        """浸透を書く所を切り替える。区域全体で1つは最上位の infiltration、区域ごとは各区域の infiltration（どちらか一方）。"""
        c = self.conf
        self.inf_scope = scope
        if scope != "whole":
            c.pop("infiltration", None)
        if scope != "zones":
            for r in c.get("catchments", []):
                r.pop("infiltration", None)
        if scope == "whole":
            c.setdefault("infiltration", {"rate_m3s": 0.001})
        self.render_infiltration()
        self.changed()

    def render_infiltration(self):
        """浸透の欄。区域全体で1つなら1つ、区域ごとなら区域名の見出しと、その区域の欄を区域の数だけ並べる。"""
        c = self.conf
        self.inf_box.clear()
        with self.inf_box:
            if self.inf_scope == "whole":
                whole = sum(r.get("area_ha") or 0 for r in c.get("catchments", [])) if "catchments" in c else None
                self.infiltration_form(c, whole, allow_none=False)
            elif self.inf_scope == "zones":
                for r in c.get("catchments", []):
                    ui.label(f"{r.get('name', '')}（{r.get('area_ha', 0):g} ha）").classes("font-medium mt-2")
                    self.infiltration_form(r, r.get("area_ha"), allow_none=True)

    def infiltration_form(self, owner: dict, area_ha: float | None, allow_none: bool):
        """浸透1つ分の欄（owner["infiltration"] を書き換える。owner は最上位の条件か区域の行）。
        書き方は設計浸透量・浸透強度（区域の面積があるときだけ）・浸透施設。区域ごとなら「浸透なし」（無処理区域）も選べる。"""
        kinds = {"none": "浸透なし"} if allow_none else {}
        kinds |= {"rate": "設計浸透量"} | ({"intensity": "浸透強度"} if area_ha else {}) | {"facilities": "浸透施設"}
        it = owner.get("infiltration")
        kind = ("none" if it is None else "rate" if "rate_m3s" in it else "intensity" if "intensity_mmh" in it
                else "facilities")
        box = {}
        ui.toggle(kinds, value=kind, on_change=lambda e: self.switch_kind(owner, e.value, box["detail"])).props("dense")
        box["detail"] = ui.column().classes("w-full gap-2")
        self.render_kind(owner, box["detail"])

    def switch_kind(self, owner: dict, kind: str, detail):
        new = {"none": None, "rate": {"rate_m3s": 0.001}, "intensity": {"intensity_mmh": 10.0},
               "facilities": {"facilities": [self._new_facility()]}}[kind]
        if new is None:
            owner.pop("infiltration", None)
        else:
            owner["infiltration"] = new
        self.render_kind(owner, detail)
        self.changed()

    def render_kind(self, owner: dict, detail):
        """浸透1つ分の値の欄（書き方を切り替えたときは、この欄だけを作り直す）。"""
        it = owner.get("infiltration")
        detail.clear()
        if it is None:
            return
        rerender = lambda: self.render_kind(owner, detail)      # noqa: E731
        with detail:
            if "rate_m3s" in it or "intensity_mmh" in it:
                with ui.row().classes("gap-2"):
                    if "rate_m3s" in it:
                        num("設計浸透量", it.get("rate_m3s"), self.setter(it, "rate_m3s", drop_none=False), suffix="m^3/s",
                            format="%.6f", width="w-44")
                    else:
                        num("設計浸透強度", it.get("intensity_mmh"), self.setter(it, "intensity_mmh", drop_none=False),
                            suffix="mm/h", width="w-36")
                    num("空隙貯留量", it.get("void_storage_m3"), self.setter(it, "void_storage_m3"), suffix="m^3", width="w-36")
                return
            facs = it.setdefault("facilities", [])
            for i, f in enumerate(facs):
                with ui.card().classes("w-full").props("flat bordered"):
                    with ui.row().classes("gap-2 items-center"):
                        ui.input("名前", value=f.get("name", ""), on_change=lambda e, f=f: (f.__setitem__("name", e.value),
                                                                                         self.changed())).classes(
                            "w-40").props("dense outlined")
                        how = "shape" if "shape" in f else "specific"
                        ui.select({"shape": "比浸透量を式で求める", "specific": "比浸透量を直接"}, value=how, label="比浸透量",
                                  on_change=lambda e, f=f: self._facility_k(f, e.value, rerender)).classes("w-48").props(
                            "dense outlined")
                        ui.button(icon="delete", on_click=lambda i=i: self.del_row(facs, i, rerender)).props(
                            "flat round dense")
                    if "shape" in f:
                        s = f["shape"]
                        with ui.row().classes("gap-2 items-center"):
                            ui.select({k: v[0] for k, v in KINDS.items()}, value=s.get("kind"), label="施設の種類",
                                      on_change=lambda e, s=s: self._kind(s, e.value, rerender)).classes("w-80").props(
                                "dense outlined")
                            for dim in KINDS.get(s.get("kind"), ("", ("H",)))[1]:
                                num({"H": "設計水頭 H", "W": "幅 W", "D": "直径 D", "L": "延長 L"}[dim], s.get(f"{dim}_m"),
                                    self.setter(s, f"{dim}_m", drop_none=False), suffix="m", width="w-28")
                    else:
                        num("比浸透量 K", f.get("specific_m2"), self.setter(f, "specific_m2", drop_none=False),
                            suffix="m^2", width="w-36")
                    with ui.row().classes("gap-2 items-center"):
                        pm = f.setdefault("permeability", {"value": 1e-5, "unit": "m/s"})
                        num("飽和透水係数", pm.get("value"), self.setter(pm, "value", drop_none=False), format="%.2e",
                            width="w-36")
                        ui.select(list(PERMEABILITY_UNITS), value=pm.get("unit"), label="単位",
                                  on_change=lambda e, pm=pm: (pm.__setitem__("unit", e.value), self.changed())).classes(
                            "w-24").props("dense outlined")
                        num("数量", f.get("count"), self.setter(f, "count", drop_none=False), width="w-24")
                    with ui.row().classes("gap-2 items-center"):
                        ui.label("影響係数").classes("text-sm")
                        fc = f.setdefault("factors", {})
                        for key, label in FACTOR_NAMES.items():
                            num(label, fc.get(key), self.setter(fc, key, drop_none=False), width="w-28")
                    vp = f.get("void_parts", [])
                    ui.input("空隙（体積 m^3 × 空隙率 % を「0.2×100, 0.8×40」の形で。任意）",
                             value=", ".join(f"{p['volume_m3']:g}×{p['porosity_pct']:g}" for p in vp),
                             on_change=lambda e, f=f: self._voids(f, e.value)).classes("w-full").props("dense outlined")
            ui.button("施設を追加", icon="add", on_click=lambda: self.add_row(facs, self._new_facility(), rerender)).props(
                "flat dense")
            ui.label(FACTOR_NOTE).classes("text-sm text-gray-600")
            g = self.form.guideline()
            if g is not None:
                for text in g.infiltration_notes().values():
                    ui.label(f"{g.name}の{text}").classes("text-sm text-gray-600")

    def _new_facility(self):
        return {"name": "浸透ます", "shape": {"kind": "circle_side_bottom", "H_m": 1.0, "D_m": 0.5},
                "permeability": {"value": 1e-5, "unit": "m/s"}, "count": 10,
                "factors": dict(NEW_FACTORS)}

    def _facility_k(self, f, how, rerender):
        if how == "shape":
            f.pop("specific_m2", None)
            f["shape"] = {"kind": "circle_side_bottom", "H_m": 1.0, "D_m": 0.5}
        else:
            f.pop("shape", None)
            f["specific_m2"] = 1.0
        rerender()
        self.changed()

    def _kind(self, s, kind, rerender):
        s["kind"] = kind
        for dim in ("H", "W", "D", "L"):
            if dim not in KINDS[kind][1]:
                s.pop(f"{dim}_m", None)              # その種類で使わない寸法は消す（残すと誤りになる）
            else:
                s.setdefault(f"{dim}_m", 1.0)
        rerender()
        self.changed()

    def _voids(self, f, text):
        parts = []
        try:
            for item in [x for x in text.replace("、", ",").split(",") if x.strip()]:
                v, p = item.replace("x", "×").split("×")
                parts.append({"volume_m3": float(v), "porosity_pct": float(p)})
        except ValueError:
            parts = text
        if parts:
            f["void_parts"] = parts
        else:
            f.pop("void_parts", None)
        self.changed()

    def run_options(self):
        c = self.conf
        ui.toggle({"route": "与えたオリフィスで追跡", "size_orifice": "最大放流量が目標値になるオリフィスの寸法を求める"},
                  value=c.get("mode", "route"), on_change=lambda e: self.switch_mode(e.value)).tooltip(
            "寸法を求めるときはオリフィスを1つにし、最大放流量が目標値（許容放流量）に一致する幅・高さ（円形は直径）を探します")
        self.run_box = ui.column().classes("w-full gap-2")
        self.render_run()

    def switch_mode(self, mode):
        c = self.conf
        if mode == "size_orifice":
            c["mode"] = "size_orifice"
            holes = self.holes()
            first = dict(holes[0]) if holes else {"sill_m": 0.0}
            for k in ("width_m", "height_m", "diameter_m"):
                first.pop(k, None)                  # 寸法は求める側なので書かない
            c.pop("orifices", None)
            c["orifice"] = first
            allow = c.pop("allowable_q_m3s", None)
            c.setdefault("target_q_m3s", allow if allow is not None else 0.05)
        else:
            c.pop("mode", None)
            c.pop("target_q_m3s", None)
            h = c.get("orifice", {"sill_m": 0.0})
            if h.get("shape") == "circle":
                h.setdefault("diameter_m", 0.1)
            else:
                h.setdefault("width_m", 0.1)
                h.setdefault("height_m", 0.1)
        self.render_orifices()
        self.render_run()
        self.changed()

    def render_run(self):
        c = self.conf
        self.run_box.clear()
        with self.run_box:
            if c.get("mode") == "size_orifice":
                num("目標の最大放流量（許容放流量）", c.get("target_q_m3s"), self.setter(c, "target_q_m3s"),
                    suffix="m^3/s", format="%.4f", width="w-72")
                ui.label("オリフィスは1つにし、その形の寸法を求めます。オリフィスの形は「放流施設」のタブで選びます。").classes(
                    "text-sm text-gray-600")
            else:
                num("許容放流量（任意）", c.get("allowable_q_m3s"), self.setter(c, "allowable_q_m3s"), suffix="m^3/s",
                    format="%.4f", width="w-56").mark("allowable")
            judge = c.setdefault("judge", {})
            num("容量の余裕（任意）", judge.get("storage_margin_m3"),
                self._judge_setter(judge), suffix="m^3", width="w-72")
            num("追跡の終了時刻（空欄は流入の期間＋18時間）", c.get("end_min"), self.setter(c, "end_min"), suffix="分",
                step=10, width="w-80")

    def _judge_setter(self, judge):
        def f(v):
            if v is None:
                judge.pop("storage_margin_m3", None)
            else:
                judge["storage_margin_m3"] = v
            if not judge:
                self.conf.pop("judge", None)
            else:
                self.conf["judge"] = judge
            self.changed()
        return f

    # --- 確かめとプレビュー ------------------------------------------------
    def refresh(self):
        f = self.form
        if not self.conf.get("judge"):
            self.conf.pop("judge", None)
        chk = f.check()
        for s in SECTIONS:
            err = chk.errors.get(s, "")
            self.errors[s].text = f"⚠ {err}" if err else ""
            self.tabs[s]._props["label"] = f"⚠ {s}" if err else s      # 誤りのあるタブに印を付ける
            self.tabs[s].classes(remove="text-negative")
            if err:
                self.tabs[s].classes("text-negative")
            self.tabs[s].update()
        self.status.text = "未保存の変更あり" if f.dirty else "保存済み"
        self.status.props(f'color={"warning" if f.dirty else "positive"}')
        if f.dirty != self.dirty_sent:                 # 変わったときだけブラウザに伝える（移動の確かめに使う）
            ui.run_javascript(f"window.amamizuDirty = {'true' if f.dirty else 'false'};")
            self.dirty_sent = f.dirty
        can_run = chk.ok and not self.busy
        self.trial_btn.set_enabled(can_run)
        self.save_btn.set_enabled(can_run)
        self.save_only_btn.set_enabled(f.can_save and not self.busy)
        self.render_output_state()
        self.render_source_button(chk)
        self.render_previews(chk)

    def open_source(self):
        """降雨の画面を新しいタブで開く（この画面の保存していない変更を失わないように）。流入が実測降雨なら降雨記録で開き、
        この画面で選んでいる（保存前を含む）ファイル・地点・期間・tc を渡す。ハイドログラフは流入のタブへ。"""
        q = self.form.rain_query()
        if q is None:
            self.panels.set_value(self.tabs["流入"])
        else:
            ui.navigate.to(f"/rain/{quote(self.form.study_dir.name)}?{urlencode(q)}", new_tab=True)

    def render_output_state(self):
        """out/ の出力と今の条件（保存していない変更を含む）の違いを、検討の名前の下に示す（変わったときだけ作り直す）。"""
        f = self.form
        diff = f.output_changes()
        if diff is None:
            has_out = (f.study_dir / "out").is_dir() and any((f.study_dir / "out").iterdir())
            state = ("old",) if has_out else ("none",)
        else:
            state = ("diff", *diff)
        state = (*state, "観測値", *f.observation_changes)   # 観測値の変化が変わったときも描き直す
        if state == self.output_shown:
            return
        self.output_shown = state
        self.output_box.clear()
        with self.output_box:
            if state[0] == "none":
                ui.label("out/: まだ出力していません").classes("text-sm text-gray-600")
            elif state[0] == "old":
                ui.label("out/: 出力の条件の記録がありません。「保存して出力」で出力し直すと、今の条件との違いを表示します").classes(
                    "text-sm text-gray-600")
            elif not diff:
                ui.label("out/: 今の条件で出力した結果です").classes(f"text-sm text-{OK_COLOR}")
            else:
                with ui.expansion(f"out/ の出力の後に、条件が変更されています（{len(diff)} 件）",
                                  icon="warning").classes("text-sm text-orange-800").props("dense").mark("output_changes"):
                    for line in diff:
                        ui.label(line).classes("text-sm")
                    ui.label("「保存して出力」で、今の条件の結果に作り直します").classes("text-sm text-gray-600")
            observation_changes_box(f.observation_changes, "「保存して出力」で、今の観測値の結果に作り直します")

    def render_source_button(self, chk):
        """試算の左のボタン。流入に使う降雨（または流入量）を示す。"""
        c, b = self.conf, self.source_btn
        if "observed_rain" in c:
            spec = c["observed_rain"]
            b.text = f"降雨記録 {chk.parts.get('observed_label') or spec.get('station') or '（地点を選んでいません）'}"
            b.props("icon=history")
            self.source_tip.text = ("流入に使う実測降雨（studies/_jma/ の値）。押すと新しいタブで、計画降雨と同じ形"
                                    "（ハイエトグラフと区間ごとの表）の降雨記録を開きます")
        elif "inflow_csv" in c:
            b.text = f"ハイドログラフ {c['inflow_csv'].get('file', '')}"
            b.props("icon=show_chart")
            self.source_tip.text = "流入に使うハイドログラフ。押すと流入のタブを開きます"
        else:
            b.text = "計画降雨"
            b.props("icon=water_drop")
            self.source_tip.text = "池を使わずに計画降雨だけを作り、この検討の out/ に保存します（amamizu rain と同じ）。新しいタブで開きます"

    def render_previews(self, chk):
        P = chk.parts
        with self._preview("全体"):
            if "guideline" in P:
                hy = P["hy"]
                ui.label(P["guideline"].name if P["guideline"] else "手引の定数: 使用しない").classes("text-sm")
                ui.label(f"重力加速度 g＝{hy.g:g} m/s^2、越流係数 {hy.c_weir:g}、オリフィスの流量係数 {hy.c_orifice:g}").classes(
                    "text-sm text-gray-700")
                ui.label(f"計算の刻み {P['tc']} 分（計画降雨の区切り、合理式連続モデルの三角形の底辺 2×tc）").classes(
                    "text-sm text-gray-700")
        with self._preview("流入"):
            inflow = P.get("inflow")
            if inflow is not None:
                tc = P["tc"]
                if inflow.rain is not None:
                    rain = inflow.rain
                    peak = int(np.argmax(rain))
                    when = f"{clock(inflow.start, peak * tc)}〜"
                    for k, v in P.get("inflow_rows", []):           # 要約と同じ行（見出し: 値）
                        ui.label(f"{k}: {v}").classes("text-sm")
                    ui.label(f"総雨量 {rain.sum() * tc / 60:.1f} mm、最大 {rain[peak]:.1f} mm/h"
                             f"（{when}）、流入量の最大 {inflow.nodes.max():.4f} m^3/s、"
                             f"区域の合計 {sum(c.get('area_ha') or 0 for c in self.conf.get('catchments', [])):g} ha").classes(
                        "text-sm text-gray-700")
                    for label, text in gap_notes(P.get("missing", ()), P.get("absent", ())):
                        ui.label(f"{label}（0 mm として計算）: {text}").classes("text-sm text-orange-800")
                    # 降雨と流入量は横軸を共有する1つの図にする（結果のタブと同じ形。別の図にすると横軸の範囲がずれる）
                    ui.echart(charts.flows(np.arange(len(inflow.nodes)) * tc, {"流入量": inflow.nodes}, rain=rain, tc=tc)).classes(
                        "w-full h-96")
                else:
                    ui.label(f"節点 {len(inflow.nodes) - 2} 個、流入量の最大 {inflow.nodes.max():.4f} m^3/s").classes("text-sm")
                    ui.echart(charts.flows(np.arange(len(inflow.nodes)) * tc, {"流入量": inflow.nodes})).classes("w-full h-64")
            if self.form.hydrograph is not None and "inflow_csv" in self.conf:
                ui.label("CSV の先頭 10 行").classes("font-medium text-sm")
                df = self.form.hydrograph.head(10)
                ui.table(columns=[{"name": str(c_), "label": str(c_), "field": str(c_)} for c_ in df.columns],
                         rows=df.astype(str).to_dict("records")).props("dense flat bordered")
        with self._preview("池"):
            pond = P.get("pond")
            if pond is not None:
                h = np.linspace(pond.bottom, pond.hwl, 200)
                ui.label(f"満水位 {pond.hwl:.3f} m、容量 {pond.capacity:.1f} m^3").classes("text-sm")
                lines = [("満水位", pond.hwl, charts.RED)]
                ui.echart(charts.curve(pond.volume(h), h, "容量 m^3", lines, "水位と容量")).classes("w-full h-64")
                if pond.has_area:
                    ui.echart(charts.curve([pond.area_at(x) for x in h], h, "面積 m^2", lines, "水位と面積")).classes(
                        "w-full h-56")
        with self._preview("放流施設"):
            if "outlets" in P:
                holes, spill, pump = P["outlets"]
                pond = P.get("pond")
                lo = pond.bottom if pond is not None else min(((o.get("sill_m") or 0) for o in P["orifices"]), default=0.0)
                hi = pond.hwl if pond is not None else lo + 3
                h = np.linspace(lo, hi, 300)
                parts = list(holes) + [x for x in (spill, pump) if x is not None]
                lines = [(f"オリフィス{i + 1 if len(holes) > 1 else ''}の敷高", o.sill, charts.GRAY) for i, o in enumerate(holes)]
                if spill is not None:
                    lines.append(("洪水吐の敷高", spill.crest, charts.ORANGE))
                if pond is not None:
                    lines.append(("満水位", pond.hwl, charts.RED))
                if parts:
                    q = [sum(p.discharge(x) for p in parts) for x in h]
                    ui.label(f"満水位での放流量 {q[-1]:.4f} m^3/s" if pond is not None else "").classes("text-sm")
                    ui.echart(charts.curve(q, h, "放流量 m^3/s", lines, "水位と放流量")).classes("w-full h-72")
                if self.conf.get("mode") == "size_orifice":
                    ui.label("オリフィスの寸法は試算で求めるので、ここにはオリフィスを含めていません。").classes("text-sm text-gray-600")
        with self._preview("浸透"):
            units = ([(infiltration_label(), P["infiltration"])] if P.get("infiltration") is not None else
                     [(infiltration_label(c.name, c.area_ha), c.infiltration) for c in P.get("catchments", [])
                      if c.infiltration is not None])
            if not units and "浸透" not in chk.errors:
                ui.label("浸透は考えません（今までどおり）。").classes("text-sm")
            for name, it in units:                   # 要約と同じ行（infiltration_rows）
                for i, (a, b) in enumerate(infiltration_rows(name, it)):
                    ui.label(f"{a.strip()}: {b}").classes("text-sm" if i == 0 else "text-sm text-gray-600 ml-4")
        with self._preview("判定と実行"):
            if "end_min" in P:
                ui.label(f"追跡の終了時刻 {P['end_min']} 分（{P['end_min'] / 60:.1f} 時間）").classes("text-sm")
            ui.label("「試算」は画面に結果を表示するだけで、ファイルには保存しません。"
                     "「保存して出力」で study.json と input/ を保存し、結果を out/ に出力します。").classes("text-sm text-gray-600")

    def _preview(self, s):
        box = self.previews[s]
        box.clear()
        return box

    # --- 不明な項目 --------------------------------------------------------
    def render_unknown(self):
        self.unknown_box.clear()
        if not self.form.unknown:
            return
        with self.unknown_box:
            with ui.card().classes("w-full bg-orange-50").props("flat bordered"):
                ui.label(f"study.json に不明な項目（amamizu が使わない項目）があります: {', '.join(self.form.unknown)}").classes(
                    "font-medium")
                ui.label("この項目は計算に使いません。項目名の書き間違いなら、study.json をテキストエディタで直してから開き直してください。"
                         "不要な項目なら、「保存」か「保存して出力」で削除して保存できます。").classes("text-sm")

    # --- 試算・保存 ----------------------------------------------------------
    async def run_trial(self):
        chk = self.form.check()
        if not chk.ok:
            ui.notify("入力に誤りがあります。⚠ の付いたタブを開くと、誤りの内容を表示しています。", type="negative")
            return
        self.set_busy(True)
        try:
            conf, tables = self.form.normalized(), self.form.tables()
            r = await run.io_bound(trial, conf, self.form.study_dir, tables)
        except ConfigError as e:
            ui.notify(f"「{e.section}」のタブの誤り: {e}", type="negative")
            return
        except Exception as e:
            unexpected(e, "試算")
            return
        finally:
            self.set_busy(False)
        snap = self.form.snapshot()
        prev = self.trials[-1]["snapshot"] if self.trials else None
        self.trials.append({"no": len(self.trials) + 1, "time": datetime.datetime.now().strftime("%H:%M:%S"),
                            "snapshot": snap, "changes": changes(prev, snap), "run": r, "show": True})
        self.render_result(r)
        self.render_compare()
        self.panels.set_value(self.tabs["結果"])

    def set_busy(self, busy):
        self.busy = busy
        self.trial_btn.set_enabled(not busy)
        self.save_btn.set_enabled(not busy)
        self.save_only_btn.set_enabled(not busy)
        self.busy_note.set_visibility(busy)
        if not busy:
            self.pending = True             # 計算中に変えた入力も確かめ直し、ボタンの状態を入力に合わせる

    async def save(self, output: bool, remove_unknown: bool = False, overwrite_ok: bool = False):
        """「保存」（output が偽）と「保存して出力」。保存して出力は、保存した条件で計算して out/ に書く
        （途中で失敗すれば out/ に何も残さない。out/ はいつも出力の条件の記録 conditions.json と一致する）。"""
        f = self.form
        if f.unknown and not remove_unknown:
            with ui.dialog() as dlg, ui.card():
                ui.label(f"study.json の不明な項目 {', '.join(f.unknown)} を削除して保存しますか（この項目は計算に使いません）。")
                with ui.row():
                    ui.button("やめる", on_click=dlg.close).props("flat")
                    ui.button("削除して保存する", on_click=lambda: self._confirmed(dlg, output, True, overwrite_ok)).mark(
                        "confirm_save").props("color=negative")
            dlg.open()
            return
        if not overwrite_ok and self._ask_overwrite(lambda: self.save(output, remove_unknown, True)):
            return
        if not output:
            try:
                f.save(remove_unknown=remove_unknown)
            except (ValueError, OSError) as e:
                ui.notify(str(e), type="negative", multi_line=True)
                return
            self.render_unknown()
            self.refresh()
            ui.notify("保存しました（出力はしていません）。", type="positive")
            return
        self.set_busy(True)
        try:
            f.save(remove_unknown=remove_unknown)
            r = await run.io_bound(run_study, f.normalized(), f.study_dir, f.tables())
        except (ValueError, OSError, GraphError) as e:
            ui.notify(str(e), type="negative", multi_line=True)
            return
        except Exception as e:
            unexpected(e, "保存して出力")
            return
        finally:
            f.reload_output()                # 成功しても失敗しても（失敗なら out/ は空）、out/ の状態を読み直す
            self.set_busy(False)
        self.render_unknown()
        self.refresh()
        self.render_result(r, saved=True)
        self.panels.set_value(self.tabs["結果"])
        ui.notify("保存して out/ に出力しました。", type="positive")
        pdf = next((p for p in r.files if p.suffix == ".pdf"), None)
        if pdf:
            open_path(pdf)

    async def _confirmed(self, dlg, output: bool, remove_unknown: bool, overwrite_ok: bool):
        """確認のダイアログで保存を選んだとき。ボタンの処理として待つので、画面の文脈（slot）の中で保存と表示を行う
        （別のタスクで起動すると ui.notify や ui.run_javascript が画面を見つけられない）。"""
        dlg.close()
        await self.save(output, remove_unknown, overwrite_ok)

    def _ask_overwrite(self, then) -> bool:
        """池の表の保存で既存の CSV から失われるものがあれば、確認のダイアログを出して True を返す（then は上書きを選んだときの処理。
        coroutine を返す関数なら、ボタンの処理として待つ）。失われるものがなければ False。"""
        lost = [(ch.path, ch.lost) for ch in self.form.table_changes() if ch.lost]
        if not lost:
            return False

        async def confirmed():
            dlg.close()
            result = then()
            if hasattr(result, "__await__"):
                await result
        with ui.dialog() as dlg, ui.card():
            ui.label("既存の CSV を上書きします。次のものが失われます。").classes("font-medium")
            for p, items in lost:
                ui.label(f"{p.name}: {'、'.join(items)}")
            with ui.row():
                ui.button("やめる", on_click=dlg.close).props("flat")
                ui.button("上書きして保存する", on_click=confirmed).mark("confirm_save").props("color=negative")
        dlg.open()
        return True

    def save_pond_csv(self, overwrite_ok: bool = False):
        """池の表だけを保存先の CSV に書く（study.json は書かない）。"""
        if not overwrite_ok and self._ask_overwrite(lambda: self.save_pond_csv(True)):
            return
        try:
            written = self.form.save_tables()
        except (ValueError, OSError) as e:
            ui.notify(str(e), type="negative", multi_line=True)
            return
        ui.notify(f"池の表を {written[0].name} に保存しました。" if written else "池の表は CSV と同じなので、保存しませんでした。",
                  type="positive")
        self.refresh()

    # --- 結果 ----------------------------------------------------------------
    def render_result(self, r, saved: bool = False):
        s = r.result.summary
        self.result_box.clear()
        with self.result_box:
            with ui.row().classes("items-center gap-2"):
                ui.label("保存した条件の結果（out/ に出力済み）" if saved else "試算の結果（ファイルには書いていません）").classes(
                    "text-lg font-bold")
                for k, v in s["判定"].items():
                    ok = v.startswith("O.K")
                    ui.badge(f"{k}: {v}").props(f"color={OK_COLOR if ok else NG_COLOR}")
            for n in r.notices:
                ui.label(f"⚠ {n}").classes(f"text-{NG_COLOR} font-medium")
            with ui.row().classes(TWO_COLUMNS):
                with ui.column().classes("w-[44%] min-w-[26rem]"):
                    ui.markdown("\n".join(x for x in r.lines[:-7] if not x.startswith("# "))).classes("text-sm result-table")
                with ui.column().classes(CHART_COLUMN):
                    tbl, tc = r.result.table, r.prepared.tc
                    series = {}
                    if "浸透後流入量(m3/s)" in tbl:
                        series["浸透前の流入量"] = tbl["流入量(m3/s)"]
                        series["浸透後の流入量"] = tbl["浸透後流入量(m3/s)"]
                    else:
                        series["流入量"] = tbl["流入量(m3/s)"]
                    series["放流量"] = tbl["放流量(m3/s)"]
                    rain = r.result.inflow.rain
                    fit_charts(
                        ui.echart(charts.flows(tbl["計算時間(min)"], series, s["許容放流量_m3s"], rain, tc)).classes(
                            "w-full h-96").mark("result_flows"),
                        ui.echart(charts.level(tbl["計算時間(min)"], {"水位": tbl["水位(m)"]}, self._level_lines(r))).classes(
                            "w-full h-72").mark("result_level"))
            if saved and r.files:
                ui.label("出力したファイル").classes("font-medium mt-2")
                with ui.row().classes("gap-2 items-center"):
                    for p in r.files:
                        ui.label(p.name).classes("text-sm px-2 py-1 bg-gray-100 rounded")
                    ui.button("フォルダを開く", icon="folder",
                              on_click=lambda: open_path(r.files[0].parent)).props("flat dense").tooltip(
                        "out/ を OS のファイル管理の画面（Windows はエクスプローラー、macOS は Finder）で開きます")
                with ui.row().classes("gap-2"):
                    for p in r.files:
                        if p.suffix == ".png":
                            ui.image(str(p)).classes("w-80").props("fit=contain")

    def _level_lines(self, r):
        lines = [(f"オリフィス{i + 1 if len(r.result.orifices) > 1 else ''}の敷高", o.sill, charts.GRAY)
                 for i, o in enumerate(r.result.orifices)]
        if r.result.spillway:
            lines.append(("洪水吐の敷高", r.result.spillway.crest, charts.ORANGE))
        lines.append(("満水位", r.result.facility.pond.hwl, charts.RED))
        return lines

    # --- 試算の比較 --------------------------------------------------------------
    def render_compare(self):
        self.compare_box.clear()
        with self.compare_box:
            if not self.trials:
                ui.label("試算すると、ここに試算ごとの主な値と、前の試算から変えた条件が並びます。").classes("text-gray-600")
                return
            cols = [("no", "No"), ("time", "時刻"), ("changes", "前の試算から変えた条件"), ("q", "最大放流量 m^3/s"),
                    ("h", "最大水位 m"), ("v", "最大貯留量 m^3"), ("spare", "余裕量 m^3"), ("spill", "溢水量 m^3"),
                    ("judge", "判定")]
            rows = []
            for t in self.trials:
                s = t["run"].result.summary
                rows.append({"no": t["no"], "time": t["time"], "changes": "\n".join(t["changes"]),
                             "q": f"{s['最大放流量_m3s']:.4f}", "h": f"{s['最大水位_m']:.4f}", "v": f"{s['最大貯留量_m3']:.1f}",
                             "spare": f"{s['余裕量_m3']:.1f}", "spill": spill_text(s['溢水量_m3']), "judge": s["判定"]["総合"]})
            table = ui.table(columns=[{"name": k, "label": v, "field": k, "align": "left"} for k, v in cols],
                             rows=rows, row_key="no").props("dense flat bordered wrap-cells").classes("w-full")
            table.add_slot("body-cell-changes", '''
                <q-td :props="props"><div style="white-space: pre-line; max-width: 36rem">{{ props.value }}</div></q-td>''')
            table.add_slot("body-cell-judge", '''
                <q-td :props="props"><q-badge :color="props.value === 'O.K' ? 'positive' : 'negative'">{{ props.value }}</q-badge></q-td>''')
            with ui.row().classes("gap-2 items-center mt-2"):
                ui.label("水位を重ねる試算:").classes("text-sm").tooltip("チェックした試算の水位を下のグラフに重ねて表示します")
                for t in self.trials:
                    ui.checkbox(f"No{t['no']}", value=t["show"],
                                on_change=lambda e, t=t: (t.__setitem__("show", e.value), self.render_compare()))
                ui.separator().props("vertical")
                pick = ui.select({t["no"]: f"No{t['no']}" for t in self.trials}, label="条件を画面に読み戻す",
                                 value=None).classes("w-48").props("dense outlined").mark("restore_pick")
                ui.button("読み戻す", on_click=lambda: self.restore(pick.value)).props("flat dense").mark("restore").tooltip(
                    "選んだ試算の条件を入力欄に戻します。今の入力は置き換わります（保存はしません）")
            shown = [t for t in self.trials if t["show"]]
            if shown:
                last = shown[-1]["run"]
                fit_charts(ui.echart(charts.level({f"No{t['no']}": t["run"].result.table["計算時間(min)"] for t in shown},
                                                  {f"No{t['no']}": t["run"].result.table["水位(m)"] for t in shown},
                                                  self._level_lines(last))).classes("w-full h-80"))

    def restore(self, no):
        t = next((t for t in self.trials if t["no"] == no), None)
        if t is None:
            return
        import copy
        self.form.conf = copy.deepcopy(t["snapshot"]["conf"])
        self.form.pond_rows = copy.deepcopy(t["snapshot"]["pond"])
        self.form.load_hydrograph()         # 読み戻した条件のファイルを読み直す（試算ごとにファイルが違うことがある）
        self.rebuild_inputs()
        ui.notify(f"No{no} の条件を画面に読み戻しました。")

    def rebuild_inputs(self):
        """条件のタブの入力の欄だけを、今の条件で作り直す。画面の上部・結果・試算の比較（押された「読み戻す」ボタンを含む）は
        作り直さない（押されたボタンを消すと、その処理の続きの ui.* が画面を見つけられなくなる）。"""
        for s, builder in self.builders.items():
            self.inputs[s].clear()
            with self.inputs[s]:
                builder()
        self.title_label.text = self.conf.get("title") or self.form.study_dir.name
        self.render_unknown()
        self.refresh()


# ============================================================================ 実測の検討（見るだけ）
def viewer_page(d: Path, conf: dict):
    with ui.row().classes("items-center gap-2"):
        ui.button(icon="arrow_back", on_click=lambda: go("/")).props("flat round dense")
        ui.label(conf.get("title") or d.name).classes("text-xl font-bold")
        ui.badge(KIND_LABEL.get(conf.get("kind"), conf.get("kind"))).props("outline")
    ui.label(f"フォルダ {d}　この検討は WebGUI では見るだけです（取得と計算は amamizu コマンドで行います）。").classes(
        "text-sm text-gray-600")
    rec = read_output_conditions(d)                 # out/ の結果を作った条件（amamizu study が書く conditions.json）
    diff = differences(rec, {"conf": conf, "pond": None}) if rec is not None else []
    if diff:                                        # 前の条件の表とレポートを、今の条件の結果として黙って見せない
        with ui.expansion(f"out/ の出力の後に、study.json の条件が変更されています（{len(diff)} 件）",
                          icon="warning").classes("text-sm text-orange-800").props("dense").mark("output_changes"):
            for line in diff:
                ui.label(line).classes("text-sm")
            ui.label(f"コマンドで amamizu study {d} を実行すると、今の条件の結果に作り直します").classes("text-sm text-gray-600")
    observation_changes_box(observation_changes(rec, d.resolve().parent),
                            f"コマンドで amamizu study {d} を実行すると、今の観測値の結果に作り直します")
    with ui.tabs().props("align=left dense") as tabs:
        t_conf, t_out, t_rep = ui.tab("条件"), ui.tab("計算結果の表"), ui.tab("レポート")
    with ui.tab_panels(tabs, value=t_out, **TAB_PANELS).classes("w-full"):
        with ui.tab_panel(t_conf):
            ui.code(json.dumps(conf, ensure_ascii=False, indent=2), language="json").classes("w-full")
        with ui.tab_panel(t_out):
            if conf.get("kind") == "return_period":
                box = ui.column().classes("w-full")
                with box:
                    ui.spinner()

                async def load():
                    from ..app.viewer import return_period_rows
                    try:
                        rows = await run.io_bound(return_period_rows, d, conf)
                    except Exception as e:       # 想定していない例外でも、計算中の表示のまま止めない
                        if not isinstance(e, (ValueError, OSError, KeyError)):
                            traceback.print_exception(e)
                        box.clear()
                        with box:
                            ui.label(f"地点ごとの再現期間を計算できません（{type(e).__name__}: {e}）。").classes(f"text-{NG_COLOR}")
                        return
                    box.clear()
                    with box:
                        ui.label("地点ごとの再現期間").classes("font-medium")
                        show_frame(rows)
                ui.timer(0.1, load, once=True)
            else:
                from ..app.viewer import storage_table_notes
                try:
                    notes = storage_table_notes(conf, d)
                except (KeyError, TypeError, ValueError, OSError) as e:
                    notes = {}
                    ui.label(f"study.json の項目が足りないか誤っているため、表の説明を作れません（{type(e).__name__}: {e}）。"
                             f"amamizu study {d} を実行すると、誤りを確かめられます。").classes(f"text-sm text-{NG_COLOR}")
                out = d / "out"
                order = list(notes)
                files = sorted(out.glob("*.tsv"), key=lambda p: (order.index(p.name) if p.name in order else len(order),
                                                                  p.name)) if out.is_dir() else []
                if not files:
                    ui.label(f"out/ に計算結果の表がありません。コマンドで amamizu study {d} を実行すると作ります。")
                else:
                    pick = ui.select({p.name: f"{notes[p.name][0]}（{p.name}）" if p.name in notes
                                      else p.name for p in files},
                                     value=files[0].name, label="表").classes("w-96").props("dense outlined")
                    holder = ui.column().classes("w-full")

                    def show(name):
                        import pandas as pd
                        holder.clear()
                        with holder:
                            df = pd.read_csv(out / name, sep="\t", encoding="utf-8-sig")
                            ui.label(f"{name}（{len(df)} 行）").classes("font-medium")
                            if name in notes:
                                ui.label(notes[name][1]).classes("text-sm text-gray-600")
                            show_frame(df)
                    pick.on_value_change(lambda e: show(e.value))
                    show(files[0].name)
        with ui.tab_panel(t_rep):
            path = d / "out" / "report.md"
            if path.exists():
                ui.markdown(path.read_text(encoding="utf-8"),
                            extras=["tables", "fenced-code-blocks", "mermaid"]).classes("w-full")   # レポートの図は Mermaid
            else:
                ui.label(f"レポート（out/report.md）がありません。コマンドで amamizu study {d} を実行すると作ります。")


# 数値の列の並べ替え（表示は TSV と同じ文字のまま。文字として比べると 311.5 が 252.0 より前になる）。数でない値は先頭
NUMERIC_SORT = "(a, b) => { const x = parseFloat(a), y = parseFloat(b); "                "if (isNaN(x)) return isNaN(y) ? 0 : -1; if (isNaN(y)) return 1; return x - y; }"


def show_frame(df):
    """表を画面に表示する。値は TSV と同じ書き方の文字で示し、数値の列は数として並べ替える。"""
    import pandas as pd
    ui.table(columns=[{"name": str(c), "label": str(c), "field": str(c), "align": "left", "sortable": True}
                      | ({":sort": NUMERIC_SORT} if pd.api.types.is_numeric_dtype(df[c]) else {})
                      for c in df.columns],
             rows=df.astype(str).to_dict("records"), pagination=25).props("dense flat bordered").classes("w-full")


# ============================================================================ 計画降雨
def rain_page(studies: Path, name: str, query: dict | None = None):
    """降雨の画面。計画降雨（強度式と波形）か降雨記録（実測降雨のファイル）を選び、同じ形（説明、ハイエトグラフ、
    区間ごとの表）で見て、検討フォルダの out/ に CSV・PNG・条件を出力する（計画降雨は amamizu rain と同じ）。
    query の source=observed で降雨記録を開き、file・station・start・end・tc を初期値にする（検討の画面で選んでいるもの）。
    ここで変えた条件は検討の条件（study.json）には書かない。"""
    query = query or {}
    d = Path(studies) / unquote(name)
    try:
        conf = read_study(d) if (d / "study.json").exists() else {}
        for k in ("tc", "hours", "round", "return_period"):
            int(query.get(k) or 0)
        for k in ("a", "b"):
            float(query.get(k) or 0)
        tc0 = int(query.get("tc") or conf.get("tc_min", DEFAULT_TC_MIN))      # 検討の tc（URL で渡されていればその値）
    except (ValueError, TypeError, OSError) as e:
        ui.label(f"降雨のページを開けません（{e}）。検討の study.json と、URL の値（tc、hours、round、return_period、a、b は数）を"
                 "確かめてください。").classes(f"text-{NG_COLOR}")
        ui.link("一覧へ", "/")
        return
    # 検討で選んだ手引の定数（検討の画面から開いたときは、保存前に選んでいるもの）。使用しない検討と読めないときは None
    gname = query.get("basis") or conf.get("guideline")
    g0, g_error = None, ""
    try:
        g0 = JsonGuideline.load(gname, studies, d) if isinstance(gname, str) else None
    except (ValueError, OSError, KeyError) as e:
        g_error = f"手引の定数（{gname}）を読めません（{e.args[0] if e.args else e}）。計画降雨は、強度式の係数を入力して作ります。"
    spec = {"shape": "rear", "hours": 24, "tc": tc0} | ({"district": g0.default_district(), "return_period": 50} if g0
                                                         else dict(SAMPLE_FORMULA))
    base = conf.get("observed_rain") or {}
    obs = {k: query.get(k) or base.get(k) for k in ("station", "series", "start", "end") if query.get(k) or base.get(k)}
    source = "observed" if query.get("source") == "observed" else "planned"
    # 検討の画面から渡された計画降雨の条件（保存前を含む）を初期値にする
    if query.get("type"):
        for k in ("district", "return_period"):
            spec.pop(k, None)
        spec |= {"type": query["type"]} | {k: float(query[k]) for k in ("a", "b") if query.get(k)} \
            | ({"n": query["n"]} if query.get("n") else {})
    elif query.get("district") and g0:              # 手引の定数を読めないときは、強度式の係数で作る（地区を残すと作れない）
        spec |= {"district": query["district"]} | (
            {"return_period": int(query["return_period"])} if query.get("return_period") else {})
    spec |= {k: int(query[k]) for k in ("hours", "round") if query.get(k)} \
        | ({"shape": query["shape"]} if query.get("shape") else {})
    near = station_district(obs, g0) if g0 else None      # 実測降雨の地点が属する地区を、計画降雨の地区の初期値にする
    if g0 and not (query.get("district") or query.get("type")):   # 渡された計画降雨の条件があれば、それを優先する
        spec["district"] = near or spec["district"]
    state = {"source": source, "how": "formula" if "type" in spec else "guideline",
             "save_name": saved_rain_name(source),
             "tc": tc0}
    with ui.row().classes("items-center gap-2"):
        back = json.dumps(f"/study/{quote(d.name)}")
        ui.button(icon="arrow_back", on_click=lambda: ui.run_javascript(
            f"if (window.opener) window.close(); else amamizuGo({back}, false);")).props("flat round dense").tooltip(
            "検討の画面から開いたときは、このタブを閉じて元のタブ（編集中の条件）に戻ります")
        title = ui.label("").classes("text-xl font-bold")
        ui.label(f"（{d.name}）").classes("text-gray-600")
    if g_error:
        ui.label(f"⚠ {g_error}").classes(f"text-{NG_COLOR} text-sm")
    with ui.row().classes(TWO_COLUMNS):
        form = ui.column().classes("w-[40%] min-w-[24rem] gap-2")
        view = ui.column().classes(CHART_COLUMN)

    def build_form():
        title.text = "降雨記録" if state["source"] == "observed" else "計画降雨"
        form.clear()
        with form:
            ui.toggle({"planned": "計画降雨", "observed": "降雨記録（実測降雨）"}, value=state["source"],
                      on_change=lambda e: switch_source(e.value)).mark("rain_source")
            if state["source"] == "observed":
                observed_form()
            else:
                planned_form()
            with ui.row().classes("gap-2 items-center"):
                ui.input("保存する名前", value=state["save_name"],
                         on_change=lambda e: state.__setitem__("save_name", e.value)).classes("w-48").props("dense outlined")
                ui.button("out/ に保存", icon="save", on_click=save).props("color=primary").tooltip(
                    "同じ名前のファイルがあれば上書きします")
            ui.label("CSV と PNG と、作り直すための条件（.json）を検討フォルダの out/ に出します。"
                     "ここで変えた条件は、検討の条件（study.json）には反映しません。").classes("text-sm text-gray-600")

    def planned_form():
        if g0:                                   # 手引を使用しない検討は、係数の直接指定だけ
            ui.toggle({"guideline": f"手引の式（{g0.district_term()}・確率年）", "formula": "5方式から選ぶ"}, value=state["how"],
                      on_change=lambda e: switch(e.value))
        if g0 and state["how"] == "guideline":
            districts, periods = guideline_choices(g0, spec)

            def set_district(e):
                spec["district"] = e.value
                build_form()                     # 確率年の選択肢は地区で決まる
                draw()
            with ui.row().classes("gap-2"):
                ui.select(districts, value=spec.get("district"), label=g0.district_term(),
                          on_change=set_district).classes("w-40").props("dense outlined")
                ui.select(periods, value=spec.get("return_period"), label="確率年",
                          on_change=lambda e: set_("return_period", e.value)).classes("w-32").props("dense outlined")
        else:
            ui.select({k: f"{v[0]}　r＝{v[1]}" for k, v in FORMULA_TYPES.items()}, value=spec.get("type"),
                      label="強度式の種類", on_change=lambda e: set_type(e.value)).classes("w-80").props("dense outlined")
            if all(spec.get(k) == v for k, v in SAMPLE_FORMULA.items()):
                ui.label(SAMPLE_FORMULA_NOTE).classes("text-sm text-gray-600")
            with ui.row().classes("gap-2"):
                for k in FORMULA_TYPES[spec["type"]][2]:
                    if k == "n":
                        ui.input("n（3/4 も可）", value=str(spec.get("n", "")),
                                 on_change=lambda e: set_("n", e.value)).classes("w-40").props("dense outlined")
                    else:
                        num(k, spec.get(k), lambda v, k=k: set_(k, v), width="w-32")
        with ui.row().classes("gap-2"):
            ui.select({"rear": "後方集中型", "center": "中央集中型"}, value=spec["shape"], label="降雨波形",
                      on_change=lambda e: set_("shape", e.value)).classes("w-40").props("dense outlined")
            num("継続時間", spec["hours"], lambda v: set_("hours", v), suffix="時間", width="w-28")
            num("区間 tc", spec["tc"], lambda v: set_("tc", v), suffix="分", width="w-28")
            num("丸め（小数の桁）", spec.get("round"), lambda v: set_("round", v), width="w-36")

    def observed_form():
        observed_fields(obs, studies, set_obs)
        num("区間 tc", state["tc"], set_tc, suffix="分", width="w-28")
        ui.label("tc は10の倍数で、10分値を tc ごとに合計する（貯留追跡計算の流入と同じ）。").classes("text-sm text-gray-600")

    def switch_source(source):
        state["source"] = source
        state["save_name"] = saved_rain_name(source)
        build_form()
        draw()

    def switch(how):
        state["how"] = how
        for k in ("district", "return_period", "type", "a", "b", "n"):
            spec.pop(k, None)
        if how == "guideline":
            spec.update({"district": near or g0.default_district(), "return_period": 50})
        else:
            spec.update(SAMPLE_FORMULA)
        build_form()
        draw()

    def set_type(kind):
        spec["type"] = kind
        for k in ("a", "b", "n"):
            if k not in FORMULA_TYPES[kind][2]:
                spec.pop(k, None)
            else:
                spec.setdefault(k, SAMPLE_FORMULA[k])
        build_form()
        draw()

    def set_(k, v):
        if v is None or v == "":
            spec.pop(k, None)
        else:
            spec[k] = int(v) if isinstance(v, float) and v.is_integer() and k in ("hours", "tc", "round",
                                                                                  "return_period") else v
        draw()

    def set_obs(values, redraw=False):
        for k, v in values.items():
            if v in (None, ""):
                obs.pop(k, None)
            else:
                obs[k] = v
        if redraw:
            build_form()
        draw()

    def set_tc(v):
        state["tc"] = int(v) if isinstance(v, float) and v.is_integer() else v
        draw()

    def current():
        s = dict(spec) | ({"guideline": gname} if g0 and spec.get("district") else {})
        if s.get("n") not in (None, ""):
            s["n"] = str(s["n"])
        return s

    def make():
        """選んでいる出どころの降雨（計画降雨も降雨記録も同じ Rain の形）。"""
        if state["source"] == "observed":
            return make_observed_rain(d, dict(obs), state["tc"])
        return make_rain(current(), studies, d)     # .json の相対パスは、g0 と同じく検討のフォルダから

    def draw():
        view.clear()
        with view:
            try:
                r = make()
            except (FileNotFoundError, ValueError, KeyError, TypeError, ZeroDivisionError) as e:
                ui.label(f"⚠ {e}").classes(f"text-{NG_COLOR} font-medium")
                return
            except Exception as e:       # 想定していない例外でも、空のまま表示しない
                traceback.print_exception(e)
                ui.label(f"⚠ 降雨を作れません（{type(e).__name__}: {e}）。").classes(f"text-{NG_COLOR} font-medium")
                return
            when = r.table["区間の日時" if r.start is not None else "区間"][r.peak]
            axis = f"。横軸は {r.start:%Y-%m-%d %H:%M} からの時間" if r.start is not None else ""
            ui.label(r.label).classes("text-sm")
            ui.label(f"{len(r.storm)} 区間、総雨量 {r.total_mm:.1f} mm、最大 {r.storm[r.peak]:.1f} mm/h（{when}）{axis}").classes(
                "text-sm text-gray-700")
            for label, text in r.notes:
                ui.label(f"{label}（0 mm として計算）: {text}").classes("text-sm text-orange-800")
            ui.echart(charts.rain_bars(r.storm, r.tc, "降雨記録" if r.start is not None else "計画降雨")).classes("w-full h-80")
            show_frame(r.table.round({"降雨強度(mm/h)": 3, "雨量(mm)": 3}))

    def save():
        try:
            r = make()
            files = write_rain(r, saved_rain_path(d, state["save_name"]), conditions_json=True)
        except (FileNotFoundError, ValueError, OSError, GraphError) as e:
            ui.notify(str(e), type="negative", multi_line=True)
            return
        except Exception as e:
            unexpected(e, "降雨の保存")
            return
        ui.notify("保存しました: " + "、".join(p.name for p in files), type="positive")

    build_form()
    draw()


# ============================================================================ 起動
def root_factory(studies: Path):
    studies = Path(studies)

    def rain(args: PageArguments):
        rain_page(studies, args.path_parameters["name"], dict(args.query_parameters))

    def root():
        ui.page_title("雨水ラボ amamizu")
        ui.button.default_props("no-caps")          # 英字を大文字にしない（out/ など）
        ui.add_head_html("<style>body{font-family:'Yu Gothic UI','Meiryo',sans-serif}.q-tooltip{font-size:13px;max-width:28rem;line-height:1.5}"
                         # 結果の表は語の途中で折り返さない（空白や「、」で折り返す。長い語だけは枠の中で折り返す）
                         ".result-table td{word-break:keep-all;overflow-wrap:anywhere}</style>")
        ui.add_head_html(LEAVE_GUARD)
        with ui.header().classes("items-center bg-slate-800 py-2"):
            ui.label("雨水ラボ amamizu").classes("text-lg font-bold cursor-pointer").on("click", lambda: go("/"))
        with ui.column().classes("w-full max-w-[1600px] mx-auto p-4"):
            # 幅を持たせる（幅がないと中身の幅で決まり、描いたグラフの幅がページの幅を固定して、ブラウザの表示幅に追従しなくなる）
            ui.sub_pages({
                "/": lambda: home_page(studies),
                "/study/{name}": lambda name: study_page(studies, name),
                "/rain/{name}": rain,
            }).classes("w-full")
    return root


# サーバの asyncio のイベントループ（uvicorn の loop に渡す名前か「モジュール:クラス」）。Windows の既定（ProactorEventLoop）は、
# 相手が切った接続を閉じるときにソケットの shutdown を呼び、それが ConnectionResetError（WinError 10054）になると、ソケットを閉じず、
# 接続を閉じ終えた扱いにもしない。そのあと Ctrl+C を押しても、サーバは接続が閉じ終わるのを待ち続けて終わらない。
# SelectorEventLoop は shutdown を呼ばずにソケットを閉じる。Windows 以外は uvicorn の既定のまま
EVENT_LOOP = "asyncio:SelectorEventLoop" if IS_WINDOWS else "auto"


def main(studies: Path, port: int, show: bool):
    try:
        ui.run(root_factory(studies), host=HOST, port=port, reload=False, show=show, title="雨水ラボ amamizu",
               show_welcome_message=True, favicon="💧", language="ja", loop=EVENT_LOOP)
    except KeyboardInterrupt:
        # Ctrl+C はサーバが止めたあと asyncio が KeyboardInterrupt を送り直す。正常な終了なので、トレースバックを表示しない
        print("終了しました。")
    except SystemExit as e:
        # uvicorn は起動に失敗すると STARTUP_FAILURE で終わる。番号を確かめたあと起動するまでの間に他のアプリが使い始めた場合は、その理由を表示する
        if e.code != STARTUP_FAILURE:
            raise
        state = checked_state(port)
        if state is PortState.IN_USE:
            raise ValueError(f"WebGUI のポート番号 {port} は、起動の途中で他のアプリが使用を始めました。数秒待ってから、もう一度実行してください。") from None
        if state is PortState.DENIED:
            raise ValueError(unusable_message(port, state)) from None
        raise ValueError("WebGUI の起動に失敗しました。上に表示された内容を確かめて、もう一度実行してください。") from None
