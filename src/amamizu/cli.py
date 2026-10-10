"""amamizu コマンド（外側の層）。検討フォルダの study.json を読み、usecases と adapters を組み立てて実行する。

  amamizu verify                          計算が手引（強度式・早見表）と一致するか
  amamizu study <検討フォルダ> [--report 保存先]   検討の計算を行い、結果の表とレポートを出力する（kind: design は設計の厳密計算）
  amamizu capture --input 時間値.csv ...   浸透施設の雨水捕捉率
  amamizu stats --input 時間値.csv ...     時間降水量の基本統計
  amamizu candidates <検討フォルダ>        時間値を取得すべき日（候補降雨の日）を出力する
  amamizu event-studies <貯留施設の検証>   降雨ごと・地点ごとに設計の検討を作る
  amamizu fetch <検討フォルダ> daily|hourly|tenmin|annual|annual_max --stations ...   気象庁から取得して studies/_jma/ に置く
  amamizu periods tenmin|hourly [--stations ...]   studies/_jma/ に取得済みの日が続いている期間の一覧
"""
import argparse
import functools
import linecache
import sys
from pathlib import Path

from .adapters.files import AnnualMaxFiles
from .adapters.design_inputs import OBSERVED_SERIES
from .adapters.jma_store import JmaStore, table_bytes
from .adapters.guideline.json_guideline import JsonGuideline
from .adapters.study_config import load_study
from .app import webgui_port
from .app.observations import fetched_periods, observation_changes, observation_record, period_label, periods_summary, periods_table
from .domain.pond import DEFAULT_TC_MIN
from .usecases.guideline_check import MATCH_REL, formula_mismatches, reproduce_hayami, reproduce_infiltration_table
from .usecases.published_check import exceedance_sets
from .usecases.storage_study import DURATIONS, ModelConditions, StorageStudy, rounded


# study.json の読み込みと確認は adapters/study_config.py（cli と gui が共通に使う）


def cmd_verify(args):
    g = JsonGuideline.load(args.guideline, args.studies, Path.cwd())    # --guideline の相対パスは今のフォルダから
    print(f"手引の定数: {g.name}")
    bad = formula_mismatches(g)
    print(f"強度式: 表の値とのずれ {len(bad)} 件", *bad, sep="\n")
    res = reproduce_hayami(g)
    if not res:
        print("早見表: この基準にはありません")
        return
    worst = max(res, key=lambda r: abs(r.rel_err))
    print(f"早見表: {len(res)} 件、最大の差 {worst.rel_err:+.2%}（{worst.cell.district} f={worst.cell.f} qc={worst.cell.qc}: "
          f"表 {worst.cell.volume}、計算 {worst.calc:.0f}）")
    for r in res:
        if abs(r.rel_err) > MATCH_REL:
            print(f"  ±{MATCH_REL:.1%} を超える: {r.cell.district} f={r.cell.f} qc={r.cell.qc} 表 {r.cell.volume} 計算 {r.calc:.0f}")
    inf = reproduce_infiltration_table(g)
    if not inf:
        print("浸透ありの容量の表: この基準にはありません")
        return
    worst = max(inf, key=lambda r: abs(r.rel_err))
    print(f"浸透ありの容量の表（{g.cite('infiltration_table')}）: {len(inf)} 件、最大の差 {worst.rel_err:+.2%}"
          f"（{worst.cell.district} 浸透強度 {worst.cell.intensity_mmh:g} mm/h 浸透処理面積率 {worst.cell.treated_pct:g}%: "
          f"表 {worst.cell.volume}、計算 {worst.calc:.0f}）")
    for r in inf:
        if abs(r.rel_err) > MATCH_REL:
            print(f"  ±{MATCH_REL:.1%} を超える: {r.cell.district} {r.cell.intensity_mmh:g} mm/h {r.cell.treated_pct:g}% "
                  f"表 {r.cell.volume} 計算 {r.calc:.0f}")


def _write_tsv(df, path: Path) -> Path:
    path.write_bytes(table_bytes(df, path))          # UTF-8 BOM 付き、改行 CRLF
    print(f"  {path}")
    return path


def _report_path(study_dir: Path, report: str | None) -> Path:
    """レポートの保存先。省略時は out/report.md。"""
    return Path(report) if report else study_dir / "out" / "report.md"


def run_storage(study_dir: Path, conf: dict, report: str | None):
    from .adapters.report import storage_report
    from .app.design_runner import CONDITIONS_FILE, all_or_nothing, record_outputs, stale_outputs
    from .app.viewer import event_tables, report_events
    g = JsonGuideline.required(conf, study_dir)
    study = StorageStudy(g, JmaStore(study_dir.resolve().parent), ModelConditions(**conf["conditions"]))
    lr, py = conf["long_record"], conf["partial_year"]
    # 計算をすべて終えてから out/ に書く（途中で止まったときに、新しい表と前の出力の表・レポートを out/ に混ぜない）
    events, tables = {}, []
    for name, ev in event_tables(conf):
        events[name] = study.event_at_stations(ev["stations"], ev["start"], ev["end"])
        tables.append((name, rounded(events[name])))
    long = study.long_record(lr["starts"], lr["end_year"])
    tables.append(("long_events.tsv", rounded(long)))
    check, final = study.recompute_with_10min(long)
    tables += [("tenmin_check.tsv", rounded(check)), ("long_events_final.tsv", rounded(final))]
    sc = study.screening(lr["starts"], lr["end_year"], final)
    tables.append(("screening.tsv", sc.round({"計画容量_m3": 0, "容量に相当する雨量_mm": 1, "基準雨量_mm": 1}).astype({"計画容量_m3": int})))
    y26 = study.partial_year(list(lr["starts"]), py["from"], py["to"])
    tables.append(("partial_year.tsv", rounded(y26)))
    res = exceedance_sets(g, study.records.annual_nhour(list(lr["starts"])), final, study.cond.return_period, DURATIONS)
    missing, missing_note = _missing_days(study, lr["starts"], py["to"])
    if missing is not None:
        tables.append(("daily_missing.tsv", missing))
    # 早見表との照合は、流出率が f と上限の行、許容放流比流量が下限値の列（早見表がなければ空）
    fs = {study.cond.f, g.runoff()["cap"]}
    cells = [c for c in g.hayami() if c.f in fs and c.qc == g.qc_min(c.district)]
    storms = {d: study.planned(d).rain for d in g.districts()}
    table_of = {id(e): n for n, e in event_tables(conf)}
    marked = [(ev, events[table_of[id(ev)]]) for ev in report_events(conf)]
    changed = _observation_changes(study_dir)
    # 使った観測値の記録は、レポートの節が読む観測値も含めるので、レポートを作った後に作る（1回だけ）
    record = functools.cache(lambda: observation_record(study.records))
    text = storage_report.render(study, conf, reproduce_hayami(g, cells), storms, marked, final, y26, record)
    out = study_dir / "out"
    out.mkdir(exist_ok=True)
    path = _report_path(study_dir, report)
    files = [out / name for name, _ in tables] + ([path] if path.parent.resolve() == out.resolve() else [])
    # 前の出力にだけある表（study.json から削除した降雨の表、欠測の日がなくなったときの欠測日の表）を残さない。
    # 出力の途中で失敗したときは、今回の表と前の出力にだけある表をすべて削除する（all_or_nothing）
    stale = stale_outputs(out, [*files, out / CONDITIONS_FILE])
    print("計算結果:")
    with all_or_nothing([*files, out / CONDITIONS_FILE, *stale]):
        for name, df in tables:
            _write_tsv(df, out / name)
        path.write_bytes(text.encode("utf-8"))
        record_outputs(out, {"conf": conf, "pond": None, "観測値": record()}, files)
    for line in published_check_lines(res, study.plan_label):
        print(line)
    print("溢水:", {n: int((final[final["地点"] == n]["溢水量_m3"] > 0).sum()) for n in lr["starts"]})
    for s in ([missing_note] if missing_note else []) + study.notices:
        print(f"注意: {s}")
    _print_observation_changes(changed)
    print(f"レポート: {path}")


def published_check_lines(res: dict, plan_label: str) -> list[str]:
    """公表値（N時間降水量）との照合（exceedance_sets の結果 res）を表示する行。一致しない継続時間は、公表値だけが超えた地点・年と
    計算だけが超えた地点・年を示す。照合していない地点・年も示す。plan_label は計画の確率年の表記（1/50 など）。"""
    def names(pairs):
        return "、".join(f"{s} {y}年" for s, y in sorted(pairs))
    lines = ["公表値（N時間降水量）との照合: " + str({h: ("一致" if v["一致"] else "不一致") for h, v in res.items()})]
    for h, v in res.items():
        known = v["計算で超過"] - v["照合できない"]
        for label, only in (("公表値だけが超過", v["公表値で超過"] - known), ("計算だけが超過", known - v["公表値で超過"])):
            if only:
                lines.append(f"注意: {h}時間の{plan_label}の超過が公表値と一致しません（{label}）: {names(only)}")
        if v["照合できない"]:
            lines.append(f"注意: {h}時間の公表値がない（数として読めない）地点・年は照合していません: {names(v['照合できない'])}")
    return lines


def _observation_changes(study_dir: Path) -> list[str]:
    """前の出力（out/）の後に変わった、使った観測値（出力し直す前に調べる）。"""
    from .app.design_runner import read_output_conditions
    return observation_changes(read_output_conditions(study_dir), study_dir.resolve().parent)


def _print_observation_changes(changed: list[str]):
    if changed:
        print("注意: 前の出力の後に、使った観測値（studies/_jma/）が変わりました（今回の出力は今の値で計算しました）:\n  " + "\n  ".join(changed))


def _missing_days(study: StorageStudy, starts: dict[str, int], date_to: str):
    """日別値が欠測（値なし）の日の表（daily_missing.tsv）と、知らせる文。0mm として降雨を区切らずに計算するので、
    その日に降った雨は総量に入っていない。欠測の日がなければ（None, None）。"""
    import pandas as pd
    rows = [r for n, y0 in starts.items() for r in study.missing_days(n, f"{y0}-01-01", date_to)]
    if not rows:
        return None, None
    df = pd.DataFrame(rows)
    days = df.drop_duplicates(["地点", "日別値が欠測の日"])
    near = int((days["降雨"] != "").sum())
    return df, (f"日別値が欠測（値なし）の日が {len(days)} 日あり、うち {near} 日は降雨の近くにあります。"
                "欠測の日は0mm として、降雨を区切らずに計算しました。その日に降った雨は総量に入っていません。")


def run_return_period(study_dir: Path, conf: dict, report: str | None):
    from .adapters.report import return_period_report
    from .usecases.return_period_study import evaluate
    g = JsonGuideline.required(conf, study_dir)
    changed = _observation_changes(study_dir)
    store = JmaStore(study_dir.resolve().parent)
    records = AnnualMaxFiles(study_dir / "raw", store)
    res = evaluate(g, records, conf["standard_period"], conf["duration_min"], conf["last_year"])
    for r in res:
        print(f"  {r.station}（{r.district}） {len(r.years)}年  再現期間 {r.return_period:.1f}年")
    path = _report_path(study_dir, report)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(return_period_report.render(res, conf, g, records.fetched_on()).encode("utf-8"))
    out = study_dir / "out"
    if path.parent.resolve() == out.resolve():      # out/ のレポートを作った条件を記録する（WebGUI が今の study.json との違いを示す）
        from .app.design_runner import record_outputs
        record_outputs(out, {"conf": conf, "pond": None, "観測値": observation_record(store)}, [path])
    _print_observation_changes(changed)
    print(f"レポート: {path}")


def run_design(study_dir: Path, conf: dict, report: str | None):
    """設計の厳密計算。結果は out/route.csv（時刻ごと）、out/design_summary.md（要約）、グラフ3枚（PNG）。
    組み立て・計算・出力は app/design_runner.py（WebGUI の「保存して出力」と同じ）。"""
    from .app.design_runner import run_study
    if report:
        raise ValueError("設計の検討（kind: design）はレポートを作らないので、--report は指定できません。"
                         "--report を外して実行してください（結果は検討フォルダの out/ に保存します）。")
    changed = _observation_changes(study_dir)
    r = run_study(conf, study_dir)
    print("\n".join(r.lines[:-7]))
    for p in r.files:
        print(f"  {p}")
    for n in r.notices:
        print(f"注意: {n}")
    _print_observation_changes(changed)


def cmd_rain(args):
    """計画降雨だけを作る。組み立ては app/rain_runner.py。"""
    from .app.rain_runner import make_rain, write_rain
    spec = {k: getattr(args, k) for k in ("district", "return_period", "guideline", "type", "a", "b", "n")}   # 組み合わせは make_rain が確かめる
    spec |= {"shape": args.shape, "hours": args.hours, "tc": args.tc, "round": args.round}
    r = make_rain(spec, args.studies, Path.cwd())                       # --guideline の相対パスは今のフォルダから
    csv, png = write_rain(r, Path(args.output))
    print(r.label)
    print(f"  {len(r.storm)} 区間、総雨量 {r.total_mm:.1f} mm、最大 {r.storm[r.peak]:.1f} mm/h（{r.table['区間'][r.peak]}）")
    print(f"  {csv}\n  {png}")


def cmd_gui(args):
    """WebGUI を開く（任意の依存 gui が要る）。"""
    try:
        import nicegui  # noqa: F401
    except ImportError:
        print("WebGUI を使うには NiceGUI のインストールが必要です。amamizu-lab のフォルダで次を実行してください: uv sync --extra gui",
              file=sys.stderr)
        return 1
    from .gui.app import main as gui_main
    studies = Path(args.studies).resolve()
    if not studies.is_dir():
        raise ValueError(f"検討のフォルダがありません: {studies}。amamizu-lab のフォルダで実行するか、--studies で検討のフォルダを指定してください。")
    port, skipped = webgui_port.resolve_port(args.port)
    if skipped is webgui_port.PortState.IN_USE:
        print(f"ポート {webgui_port.DEFAULT_PORT} は使用中のため、{port} で起動します。")
    elif skipped is webgui_port.PortState.DENIED:
        print(f"ポート {webgui_port.DEFAULT_PORT} は{webgui_port.DENIED_CAUSE}ため、{port} で起動します。")
    print(f"ブラウザで http://{webgui_port.HOST}:{port}/ を開きます（終了は Ctrl+C）。")
    gui_main(studies, port, show=not args.no_browser)


def cmd_study(args):
    study_dir = Path(args.study)
    conf = load_study(study_dir)
    runner = {"storage": run_storage, "return_period": run_return_period, "design": run_design}[conf["kind"]]
    runner(study_dir, conf, args.report)


def cmd_capture(args):
    import pandas as pd
    from .usecases.rainfall_tools import capture_table, normalize_hourly
    df = normalize_hourly(pd.read_csv(args.input, encoding="utf-8-sig"))
    table = capture_table(df["rainfall_mm"], args.min_intensity, args.max_intensity, args.step)
    print(table.to_string(index=False))
    if table["欠測の時間数"].iat[0]:
        print(f"注意: 欠測（空欄）の {table['欠測の時間数'].iat[0]} 時間を除いて計算しました（{len(df)} 時間のうち）。")
    if args.output:
        table.to_csv(args.output, index=False, encoding="utf-8-sig")


def cmd_stats(args):
    from .usecases.rainfall_tools import hourly_statistics
    from .adapters.report.stats_markdown import render
    import pandas as pd
    df = pd.read_csv(args.input, encoding="utf-8-sig")
    text = render(hourly_statistics(df), args.location)
    if args.output:
        Path(args.output).write_bytes(text.encode("utf-8"))
    else:
        print(text)


def cmd_event_studies(args):
    """貯留施設の検証の降雨ごと・地点ごとに、設計の検討を作る。"""
    from .app.event_studies import create_event_studies
    made = create_event_studies(Path(args.study), Path(args.studies))
    for d in made:
        print(f"  {d}")
    print(f"{len(made)} 件の検討を作りました。")


def cmd_periods(args):
    """studies/_jma/ に取得済みの日が続いている期間を、地点ごとに表示する（WebGUI の実測降雨の期間の選択肢と同じ一覧）。
    全地点を読んでから表示・保存する（途中の地点のエラーで一部だけの表を残さない）。"""
    found = fetched_periods(JmaStore(args.studies), args.series, args.stations)
    for station, periods in found.items():
        print(periods_summary(station, args.series, periods))
        for x in periods:
            print(f"  {period_label(x)}")
    if args.output:
        _write_tsv(periods_table(found), Path(args.output))


def cmd_candidates(args):
    """時間値を取得すべき日（候補降雨の日）を出力する。"""
    import pandas as pd
    study_dir = Path(args.study)
    conf = load_study(study_dir)
    study = StorageStudy(JsonGuideline.required(conf, study_dir), JmaStore(study_dir.resolve().parent),
                         ModelConditions(**conf["conditions"]))
    lr = conf["long_record"]
    for name, y0 in lr["starts"].items():
        evs = study.candidates(name, f"{y0}-01-01", f"{lr['end_year']}-12-31")
        days = sorted({str(d.date()) for e in evs for d in pd.date_range(e["開始"], e["終了"])})
        path = candidate_days_path(study_dir, name)
        path.parent.mkdir(exist_ok=True)
        path.write_text("\n".join(days) + "\n", encoding="utf-8")
        print(f"{name}: 基準 {study.threshold_mm(study.guideline.district_of(name)):.1f}mm、候補 {len(evs)} 降雨、{len(days)} 日 → {path}")
    for s in study.notices:          # 日別値の最後の日に雨がある降雨など（候補の総量が小さく出ていることがある）
        print(f"注意: {s}")


def candidate_days_path(study_dir: Path, station: str) -> Path:
    """amamizu candidates が書き、amamizu fetch hourly が読む候補日のファイル（検討の out/。作り直せるのでコミットしない）。"""
    return Path(study_dir) / "out" / f"candidate_days_{station}.txt"


def fetch_days(args, station: str) -> list:
    """時間値を取得する日を読む。--dates-file がなければ amamizu candidates が書いた候補日のファイル。"""
    import datetime
    src = Path(args.dates_file.format(station=station)) if args.dates_file else candidate_days_path(Path(args.study), station)
    if not src.exists():
        how = (f"--dates-file のファイル名（{args.dates_file}。{{station}} は地点名に置き換わります）を直してください" if args.dates_file
               else f"先に amamizu candidates {args.study} を実行して候補日のファイルを作るか、--dates-file で取得する日の一覧のファイルを指定してください")
        raise FileNotFoundError(f"{station} の時間値を取得する日の一覧のファイルがありません: {src}。{how}。")
    try:
        lines = src.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:   # メモ帳や Excel で「Unicode テキスト」（UTF-16）として保存したファイルなど
        raise ValueError(f"{src} の文字コードが UTF-8 ではないため、読めません。"
                         "メモ帳で開き、「名前を付けて保存」で文字コードを UTF-8 にして保存し直してください。") from None
    days = []
    for n, line in enumerate(lines, 1):
        for text in line.split():
            try:
                days.append(datetime.date.fromisoformat(text))
            except ValueError:
                raise ValueError(f"{src} の {n} 行目の「{text}」を日付として読めません。日付は 2026-08-13 の形で書いてください。") from None
    if not days:
        raise ValueError(f"{src} に日付が1つもないので、取得する日がありません。取得する日を 2026-08-13 の形で1行に1つ書いてください。")
    return days


def require_options(args, *options: tuple[str, str], example: str):
    """fetch の取得の種類に必要なオプション（(属性名, オプション名) の並び）がそろっているか確かめる。"""
    missing = [opt for attr, opt in options if getattr(args, attr) is None]
    if missing:
        raise ValueError(f"fetch {args.what} には {' と '.join(missing)} を指定してください（例: {example}）。")


def fetch_period(args) -> tuple:
    """fetch tenmin の --start・--end を日付にする。"""
    import datetime
    require_options(args, ("start", "--start"), ("end", "--end"), example="--start 2026-08-13 --end 2026-08-15")
    period = []
    for opt, text in (("--start", args.start), ("--end", args.end)):
        try:
            period.append(datetime.date.fromisoformat(text))
        except ValueError:
            raise ValueError(f"{opt} の「{text}」を日付として読めません。2026-08-13 の形で指定してください。") from None
    if period[1] < period[0]:
        raise ValueError(f"--end（{args.end}）は --start（{args.start}）以降の日付にしてください。")
    return tuple(period)


def fetch_years(args) -> tuple[int, int]:
    """fetch daily の --year-from・--year-to。"""
    require_options(args, ("year_from", "--year-from"), ("year_to", "--year-to"), example="--year-from 1976 --year-to 2025")
    if args.year_to < args.year_from:
        raise ValueError(f"--year-to（{args.year_to}）は --year-from（{args.year_from}）以降の年にしてください。")
    return args.year_from, args.year_to


def cmd_fetch(args):
    import logging

    from jma_rainfall.fetch import FetchAborted

    from .adapters.jma_fetch import JmaFetcher
    # jma_rainfall は画面に出力せず logging で知らせる。取得の失敗の理由と休止を標準エラーに出力する
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
    # 取得を始める前に、取得の種類に必要な指定と全地点の日付を確かめる
    if args.what == "daily":
        years = fetch_years(args)
    elif args.what == "hourly":
        days = {n: fetch_days(args, n) for n in args.stations}
    elif args.what == "tenmin":
        start, end = fetch_period(args)
    # 観測値は検討のフォルダ（studies/）の下の _jma/ に地点ごとに1か所で持つ
    f = JmaFetcher(Path(args.study).resolve().parent, pref=args.pref, interval=args.interval,
                   verify_ssl=not args.disable_ssl_verify, accept=args.accept_revisions)
    aborted = False
    try:
        if args.what == "daily":
            for n in args.stations:
                f.daily(n, *years)
        elif args.what == "hourly":
            for n in args.stations:
                f.hourly(n, days[n])
        elif args.what == "tenmin":
            for n in args.stations:
                f.tenmin(n, start, end)
        elif args.what == "annual":
            f.annual(args.stations, args.view)
        elif args.what == "annual_max":
            f.annual_max(args.stations)
    except FetchAborted as e:   # 失敗が続いた。取得済みの分は保存してある（保存できなかったときは、その理由が文に入っている）
        aborted = True
        print(f"エラー: {e}", file=sys.stderr)
    if f.conflicts:
        print(f"取得した値が、確定の値（観測日から{f.store.refresh_months}か月より後に取得した値）と違う行が {len(f.conflicts)} 行あり、"
              "その行は保存しませんでした（ほかの行は保存しました）:\n  " + "\n  ".join(f.conflicts)
              + "\n取得した値で上書きするなら、--accept-revisions を付けてもう一度実行してください"
              "（上書きした値は studies/_jma/revisions.tsv に記録します）。", file=sys.stderr)
    if f.failures:
        print(f"取得できなかったものが {len(f.failures)} 件あります（理由は上に表示しています）。"
              "時間を置いて同じコマンドをもう一度実行すると、取得できなかった分を取得し直します。", file=sys.stderr)
    if aborted or f.conflicts or f.failures:
        sys.exit(1)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="amamizu", description="雨水ラボ: 調整池の貯留追跡計算（厳密計算）をするツール")
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="計算が手引と一致するか確かめる")
    v.add_argument("--guideline", default="chiba_2006", help="照合する手引の定数（名前または .json のパス。既定 chiba_2006）")
    v.add_argument("--studies", default="studies", help="検討のフォルダ（既定 ./studies。手引の定数はこの下の _guideline/ から探す）")
    v.set_defaults(func=cmd_verify)
    s = sub.add_parser("study", help="検討フォルダの計算を行う")
    s.add_argument("study", help="studies/ の下の検討フォルダ")
    s.add_argument("--report", help="レポート（Markdown）の保存先（省略時は検討フォルダの out/report.md）")
    s.set_defaults(func=cmd_study)
    rn = sub.add_parser("rain", help="計画降雨を作る（強度式の5方式、後方集中型・中央集中型）")
    rn.add_argument("--type", choices=["talbot", "sherman", "kuno_ishiguro", "cleveland", "kuno_ishiguro_n"],
                    help="強度式の種類")
    rn.add_argument("--a", type=float, help="強度式の係数 a（--type と一緒に指定する）")
    rn.add_argument("--b", type=float, help="強度式の係数 b（式の種類によっては使わない）")
    rn.add_argument("--n", help="指数（3/4 のような分数でもよい）")
    rn.add_argument("--district", help="手引の地区（--type の代わりに手引の式を使う）")
    rn.add_argument("--return-period", dest="return_period", type=int, help="手引の式の確率年（既定 50）")
    rn.add_argument("--guideline", help="手引の定数（名前または .json のパス）。--district を使うときに指定する")
    rn.add_argument("--studies", default="studies", help="検討のフォルダ（既定 ./studies。手引の定数はこの下の _guideline/ から探す）")
    rn.add_argument("--shape", required=True, choices=["rear", "center"], help="rear 後方集中型、center 中央集中型")
    rn.add_argument("--hours", type=int, default=24, help="計画降雨の継続時間（時間）。既定 24")
    rn.add_argument("--tc", type=int, default=DEFAULT_TC_MIN, help=f"区間の長さ（洪水到達時間）分。既定 {DEFAULT_TC_MIN}")
    rn.add_argument("--round", type=int, default=None, help="降雨強度（mm/h）を小数点以下この桁に四捨五入する。省略すると丸めない")
    rn.add_argument("--output", required=True, help="計画降雨の CSV（同じ名前の .png も出力する）")
    rn.set_defaults(func=cmd_rain)
    gu = sub.add_parser("gui", help="WebGUI（ブラウザで操作する画面）を開く（uv sync --extra gui が要る）")
    gu.add_argument("--studies", default="studies", help="検討のフォルダ（既定 ./studies）")
    gu.add_argument("--port", type=int, default=None,
                    help=f"WebGUI のポート番号（省略すると {webgui_port.DEFAULT_PORT}。使用中または使用できないときは次の空いている番号）")
    gu.add_argument("--no-browser", dest="no_browser", action="store_true", help="ブラウザを自動で開かない")
    gu.set_defaults(func=cmd_gui)
    c = sub.add_parser("capture", help="浸透施設の雨水捕捉率")
    c.add_argument("--input", required=True, help="時間降水量の CSV（降水量_mm または rainfall_mm 列）")
    c.add_argument("--min_intensity", type=float, default=5, help="表の最初の設計浸透強度（mm/h）。既定 5")
    c.add_argument("--max_intensity", type=float, default=50, help="表の最後の設計浸透強度（mm/h）。既定 50")
    c.add_argument("--step", type=float, default=5, help="設計浸透強度の刻み（mm/h）。既定 5")
    c.add_argument("--output", help="結果の CSV")
    c.set_defaults(func=cmd_capture)
    t = sub.add_parser("stats", help="時間降水量の基本統計")
    t.add_argument("--input", required=True, help="時間降水量の CSV（日付・時・降水量_mm、または date・hour・rainfall_mm）")
    t.add_argument("--location", default=None, help="結果の見出しに書く地点名")
    t.add_argument("--output", help="Markdown の出力先")
    t.set_defaults(func=cmd_stats)
    ev = sub.add_parser("event-studies",
                        help="貯留施設の検証の降雨（events）を、地点ごとの設計の検討にする（モデル施設を安全側に丸めた池とオリフィス）")
    ev.add_argument("study", help="貯留施設の検証のフォルダ（kind: storage）")
    ev.add_argument("--studies", default="studies", help="検討を作るフォルダ（既定 ./studies）")
    ev.set_defaults(func=cmd_event_studies)
    k = sub.add_parser("candidates", help="時間値を取得すべき日（候補降雨の日）を出力する")
    k.add_argument("study", help="貯留施設の検証のフォルダ（kind: storage）")
    k.set_defaults(func=cmd_candidates)
    pe = sub.add_parser("periods", help="studies/_jma/ に取得済みの日が続いている期間（実測降雨の期間の選択肢）を、地点ごとに新しい順に表示する")
    pe.add_argument("series", choices=list(OBSERVED_SERIES),
                    help="値の種類（" + "、".join(f"{k} {v}" for k, v in OBSERVED_SERIES.items()) + "）")
    pe.add_argument("--stations", nargs="+", help="地点名（複数を空白で区切って指定できる。省略すると、値の種類を取得済みの全地点）")
    pe.add_argument("--studies", default="studies", help="検討のフォルダ（既定 ./studies。観測値はこの下の _jma/ から読む）")
    pe.add_argument("--output", help="一覧の保存先（.tsv はタブ区切り、それ以外はカンマ区切り。UTF-8 BOM 付き、改行 CRLF）")
    pe.set_defaults(func=cmd_periods)
    fe = sub.add_parser("fetch", help="気象庁から取得して studies/_jma/ に保存する（値がそろっていない日と暫定の日だけを取得する）")
    fe.add_argument("study", help="検討のフォルダ。観測値はその親のフォルダ（studies/）の _jma/ に保存し、時間値は out/ の候補日を取得する")
    fe.add_argument("what", choices=["daily", "hourly", "tenmin", "annual", "annual_max"],
                    help="取得するデータ（daily 日別値、hourly 時間値、tenmin 10分値、annual 年ごとの値（N時間降水量）、"
                         "annual_max 年最大値系列（最大10分間・最大1時間降水量）と観測史上順位）")
    fe.add_argument("--stations", nargs="+", required=True, help="地点名（複数を空白で区切って指定できる）")
    fe.add_argument("--pref", default=None, help="都道府県（地点名が同じ観測所がほかの都道府県にあるときに指定する）")
    fe.add_argument("--year-from", dest="year_from", type=int, help="日別値（daily）の最初の年")
    fe.add_argument("--year-to", dest="year_to", type=int, help="日別値（daily）の最後の年")
    fe.add_argument("--dates-file", dest="dates_file", help="時間値を取得する日の一覧（{station} で地点名を差し込める）")
    fe.add_argument("--start", help="10分値（tenmin）の開始日（YYYY-MM-DD）")
    fe.add_argument("--end", help="10分値（tenmin）の終了日（YYYY-MM-DD）")
    fe.add_argument("--view", default="a5", help="年ごとの値（annual）の表示の種類（気象庁のページの view。保存できるのは a5）")
    fe.add_argument("--accept-revisions", dest="accept_revisions", action="store_true",
                    help="確定の値（観測日から studies/_jma/config.json の refresh_months か月より後に取得した値）と違う取得値でも上書きする（studies/_jma/revisions.tsv に記録する）")
    fe.add_argument("--interval", type=float, help="アクセス間隔の秒数（省略時は jma-rainfall の既定）")
    fe.add_argument("--disable_ssl_verify", action="store_true",
                    help="SSL証明書の検証を無効にする（一時的な対処。通常は Windows の証明書ストアで検証する。"
                         "jma-rainfall の README の「SSL証明書の問題」を参照）")
    fe.set_defaults(func=cmd_fetch)
    args = p.parse_args(argv)
    return args.func(args)


PACKAGE = Path(__file__).resolve().parent


def _chain(err: BaseException):
    """例外と、その原因の例外（raise ... from、処理中に起きた例外）を順に返す。"""
    while err is not None:
        yield err
        err = err.__cause__ or (None if err.__suppress_context__ else err.__context__)


def _raised_by_amamizu(err: BaseException) -> bool:
    """amamizu の raise 文が、理由を書いた文と一緒に送出した例外か。
    pandas・Python などが送出した例外や、amamizu の中でも raise 文でない行（辞書を引く、数値に直すなど）で起きた例外は、
    文に理由が書かれていないことがあるので False にする。"""
    tb = err.__traceback__
    if tb is None:
        return False
    while tb.tb_next:
        tb = tb.tb_next
    path = Path(tb.tb_frame.f_code.co_filename).resolve()
    return path.is_relative_to(PACKAGE) and linecache.getline(str(path), tb.tb_lineno).lstrip().startswith("raise ")


def run(argv=None):
    """amamizu コマンドの入口。study.json や入力の誤りは、Python の経過（Traceback）を表示せずに、理由の文を表示して終了コード1にする。
    amamizu が理由を書いていない例外（原因の例外を含む）は、原因がわかるように今までどおり経過ごと表示する。"""
    try:
        return main(argv)
    except (ValueError, KeyError, TypeError, FileNotFoundError, RuntimeError) as e:
        if not all(_raised_by_amamizu(x) for x in _chain(e)):
            raise
        where = f"（{e.section}）" if getattr(e, "section", None) else ""        # ConfigError は項目のまとまり
        print(f"エラー{where}: {e.args[0] if e.args else e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:        # ファイルを読み書きできない（出力先のファイルを Excel などで開いている、権限がない、など）
        print(f"エラー: ファイルを読み書きできません（{e}）。このファイルを開いているアプリがあれば閉じてから、もう一度実行してください。",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(run())
