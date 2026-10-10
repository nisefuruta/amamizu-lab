"""検討が使った気象庁の観測値（studies/_jma/）の記録と、前の出力からの変化（amamizu study と WebGUI が使う）。
取得済みの期間の一覧（amamizu periods と、WebGUI の実測降雨の期間の選択肢が使う）。

記録は out/conditions.json の「観測値」に書く: 出力日、暫定とする月数、読んだ種類・地点・期間ごとの要約（JmaStore.fingerprint）。
"""
import datetime
from pathlib import Path

import pandas as pd

from ..adapters.design_inputs import OBSERVED_SERIES, fetch_command
from ..adapters.jma_store import KIND_NAMES, KINDS, JmaStore, Period


def observation_record(store: JmaStore) -> dict:
    """store で読んだ観測値の記録（conditions.json の「観測値」）。"""
    return {"出力日": datetime.date.today().isoformat(), "暫定の月数": store.refresh_months, "範囲": store.used()}


def observation_changes(output: dict | None, studies: Path) -> list[str]:
    """out/ を作ったときに使った観測値（出力の条件の記録 output（conditions.json）の「観測値」）と、今の studies/_jma/ の違い
    （amamizu study・WebGUI が、前の値の結果を今の値の結果として黙って見せないために示す）。記録がなければ空。
    studies/_jma/ の設定が読めなければ、その理由を1行で返す。"""
    recorded = (output or {}).get("観測値")
    if not recorded:
        return []
    try:
        store = JmaStore(studies)
    except (FileNotFoundError, ValueError) as e:
        return [str(e)]
    return _changes(recorded, store)


def changed_count(changes: list[str]) -> int:
    """observation_changes の行のうち、種類・地点・期間の行の数（訂正の行は字下げしてある）。"""
    return len([x for x in changes if not x.startswith(" ")])


def _changes(recorded: dict, store: JmaStore) -> list[str]:
    """行数が違えば行の増減、行数が同じで指紋が違えば値か品質の変化とし、その期間の訂正（revisions.tsv の、出力日以後の取得）を添える。"""
    path = store.root / "revisions.tsv"
    rev = pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype=str) if path.exists() else None
    out = []
    for r in recorded["範囲"]:
        kind, station, (first, last) = r["種類"], r["地点"], r["期間"]
        name = f"{station} {KIND_NAMES[kind]}" + (f" {first}〜{last}" if first else "")
        try:
            now = store.fingerprint(kind, station, first, last)
        except (FileNotFoundError, ValueError) as e:     # 値のファイルがない・読めない
            out.append(f"{name}: {e}")
            continue
        if now["行数"] != r["行数"]:
            out.append(f"{name}: 行数が {r['行数']} → {now['行数']}")
        elif now["指紋"] != r["指紋"]:
            out.append(f"{name}: 値か品質が変わりました")
        else:
            continue
        if rev is not None and kind in KINDS:          # 観測史上順位（rank）は取り直すと置き換える（訂正の記録はない）
            day = rev["日時"].str[:4] if KINDS[kind].yearly else rev["日時"].str[:10]
            hit = rev[(rev["地点"] == station) & (rev["種類"] == kind) & (rev["取得日"] >= recorded["出力日"])
                      & (day >= first) & (day <= last)]
            out += [f"  訂正 {x['日時']}: {x['前の値']} → {x['新しい値']}（取得日 {x['取得日']}）" for _, x in hit.iterrows()]
    return out


# 取得済みの期間 ---------------------------------------------------------------------
PERIOD_COLUMNS = ["地点", "開始日", "終了日", "日数", "合計雨量_mm", "そろっていない日"]


def fetched_periods(store: JmaStore, series: str, stations: list[str] | None = None) -> dict[str, list[Period]]:
    """地点ごとの、series（tenmin 10分値、hourly 時間値）を取得済みの日が続いている期間（JmaStore.periods。新しい順）。
    stations を省略すると、series を取得済みの全地点。どの地点も読んでから返す（途中の地点のエラーで一部だけを返さない）。"""
    if series not in OBSERVED_SERIES:
        raise ValueError(f"値の種類「{series}」は扱えません。tenmin（10分値）か hourly（時間値）にしてください。")
    names = list(stations) if stations else store.stations_with(series)
    if not names:
        raise FileNotFoundError(f"studies/_jma/ に{OBSERVED_SERIES[series]}を取得した地点がありません。先に {fetch_command(series)} で取得してください。")
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        raise ValueError(f"地点 {'、'.join(twice)} を2回以上指定しています。1回にしてください。")
    return {n: store.periods(series, n) for n in names}


def period_label(p: Period) -> str:
    """期間の1行の表示（期間、日数、合計雨量、行のそろっていない日の数）。"""
    short = f"、そろっていない日 {p.short_days}" if p.short_days else ""
    return f"{p.start}〜{p.end}（{p.days}日、合計 {p.total_mm:.1f} mm{short}）"


def periods_summary(station: str, series: str, periods: list[Period]) -> str:
    """地点の取得済みの期間の要約（件数と、最初と最後の日）。"""
    kind = OBSERVED_SERIES[series]
    if not periods:
        return f"{station}の{kind}のファイルに値の行がありません。"
    return f"{station}の{kind}は {len(periods)} 件の期間（{periods[-1].start}〜{periods[0].end} の間）を取得済みです。"


def fetched_days(periods: list[Period]) -> list[str]:
    """期間に含まれる日（YYYY-MM-DD。古い順）。"""
    return [str(datetime.date.fromisoformat(p.start) + datetime.timedelta(days=i)) for p in reversed(periods) for i in range(p.days)]


def periods_table(found: dict[str, list[Period]]) -> pd.DataFrame:
    """fetched_periods の結果の表（列は PERIOD_COLUMNS。地点ごとに新しい順。合計雨量は小数1桁に丸める）。"""
    rows = [[n, p.start, p.end, p.days, round(p.total_mm, 1), p.short_days] for n, ps in found.items() for p in ps]
    return pd.DataFrame(rows, columns=PERIOD_COLUMNS)
