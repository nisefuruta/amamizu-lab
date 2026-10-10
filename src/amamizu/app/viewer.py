"""検討を見るための組み立て（WebGUI で見るだけの検討が使う）。気象庁へはアクセスしない（取得済みの値だけを読む）。"""
from pathlib import Path

import pandas as pd

from ..adapters.files import AnnualMaxFiles
from ..adapters.jma_store import JmaStore
from ..adapters.guideline.json_guideline import JsonGuideline
from ..adapters.report.storage_report import date_from_text, period_text
from ..usecases.return_period_study import evaluate
from ..usecases.storage_study import ModelConditions


# 貯留施設の検証（kind＝storage）が out/ に書く表。この一覧の順に並べ、長期記録は絞り込みの段階に番号を付ける。
# WebGUI で表を選ぶときに、名前と目的を示す。{…} には storage_table_notes が study.json の値を入れる。
# 降雨ごとの表（events の数だけ、event_<開始日>.tsv）は event_tables が名前を決め、この一覧の前に並べる。
EVENT_TABLE = ("{event_name}の各地点",
               "{event}の降雨を、10分値のある地点ごとに、その{term}のモデル施設に与えて計算した結果。1行が1地点。")
STORAGE_TABLES = {
    "screening.tsv": ("長期記録 1 絞り込みの件数",
                      "{long}の地点ごとに、対象期間の日数 → 降雨 → 基準雨量以上で計算した降雨 → 最大貯留率{fill_ratio}以上 → "
                      "10分値で計算し直した降雨 → 溢水 と、絞り込みの順に件数を並べた表。手順と理由はレポートの長期記録の節。"),
    "long_events.tsv": ("長期記録 2 時間値で計算",
                        "{long}で、総雨量が基準雨量以上の降雨を時間値で計算した結果。1行が1降雨。"
                        "10分値で計算し直す降雨を選び、その10分値を取得するのに使う途中の表。"),
    "tenmin_check.tsv": ("長期記録 3 10分値で計算し直し",
                         "2 のうち開始日が{tenmin_from}以降で最大貯留率{tenmin_fill}以上の降雨を、"
                         "10分値で計算し直し、時間値の結果と並べた表。"),
    "long_events_final.tsv": ("長期記録 4 最終の結果",
                              "2 のうち 3 で計算し直した降雨を、10分値の結果に置き換えた表（出所の列が「10分値」）。"
                              "溢れた回数とレポートの長期記録の節の表はこの表による。"),
    "partial_year.tsv": ("途中の年（{partial}）",
                         "{partial}の降雨を10分値で計算した結果。長期記録の回数には含めない。"),
    "daily_missing.tsv": ("日別値の欠測日",
                          "日別値が欠測（値なし）の日と、その日を含むか前後{dry_gap}日以内の降雨。欠測の日は0mm として、"
                          "降雨を区切らずに計算した。その日の雨は総量に入っていないので、計算しなかった降雨は"
                          "基準雨量までの差を示す。"),
}


def event_tables(conf: dict) -> list[tuple[str, dict]]:
    """study.json の events の降雨ごとの表の名前（event_<開始日>.tsv）と降雨。開始日が同じ降雨があればエラー。"""
    out = [(f"event_{ev['start'].replace('-', '')}.tsv", ev) for ev in conf["events"]]
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"study.json の events に開始日が同じ降雨があります。結果の表の名前（event_<開始日>.tsv）が重なるので、"
                         f"start を降雨ごとに違う日にしてください（表の名前: {', '.join(names)}）。")
    return out


def report_events(conf: dict) -> list[dict]:
    """レポートに表を載せる降雨（events のうち "in_report": true のもの。study.json の順）。1つもなければエラー。
    "overflow_detail": true（表で溢れた地点の水位を載せる）は、表を載せる降雨にだけ付けられる。"""
    marked = [ev for ev in conf["events"] if ev.get("in_report")]
    if not marked:
        raise ValueError('レポートに表を載せる降雨がありません。study.json の events のうち1つ以上を "in_report": true にしてください。')
    stray = [ev["name"] for ev in conf["events"] if ev.get("overflow_detail") and not ev.get("in_report")]
    if stray:
        raise ValueError(f'study.json の events で "overflow_detail": true の降雨（{", ".join(stray)}）は、"in_report": true にもしてください'
                         '（溢れた地点の水位は、表を載せる降雨にだけ載せます）。')
    return marked


def storage_table_notes(conf: dict, study_dir: Path) -> dict[str, tuple[str, str]]:
    """表の名前と説明に、study.json の値（降雨の名前と期間、長期記録の範囲、条件）を入れる。降雨ごとの表を先に並べる。"""
    c = ModelConditions(**conf["conditions"])
    py, lr = conf["partial_year"], conf["long_record"]
    notes = {}
    for name, ev in event_tables(conf):
        v = {"event_name": ev["name"], "event": f"{ev['name']}（{period_text(ev['start'], ev['end'])}）",
             "term": JsonGuideline.required(conf, study_dir).district_term()}
        notes[name] = (EVENT_TABLE[0].format(**v), EVENT_TABLE[1].format(**v))
    values = {"partial": period_text(py["from"], py["to"]),
              "long": f"{lr.get('station_kind', '')}{len(lr['starts'])}地点の長期記録（〜{lr['end_year']}年）",
              "fill_ratio": f"{c.fill_ratio:.0%}", "tenmin_from": date_from_text(c.tenmin_from),
              "tenmin_fill": f"{c.tenmin_fill:.0%}", "dry_gap": c.dry_gap}
    return notes | {name: (label.format(**values), text.format(**values)) for name, (label, text) in STORAGE_TABLES.items()}


def return_period_rows(study_dir: Path, conf: dict) -> pd.DataFrame:
    """降雨強度の検証を計算し、地点ごとの表にする（amamizu study と同じ値。WebGUI の表示に使う）。"""
    g = JsonGuideline.required(conf, study_dir)
    res = evaluate(g, AnnualMaxFiles(Path(study_dir) / "raw", JmaStore(Path(study_dir).resolve().parent)), conf["standard_period"],
                   conf["duration_min"],
                   conf["last_year"])
    return pd.DataFrame([{"地点": r.station, g.district_term(): r.district, "種別": r.kind, "年数": len(r.years),
                          "平均 mm/10分": round(r.mean, 2), "標準偏差": round(r.std, 2),
                          "基準の強度 mm/h": round(r.standard_mmhr, 1), "再現期間 年": round(r.return_period, 1)}
                         for r in res])
