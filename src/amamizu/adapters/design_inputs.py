"""設計の検討（kind: design）の入力ファイルを読み書きする。検討フォルダの input/ に置いた CSV（UTF-8、BOM 付きでもよい）。
拡張子が .tsv のファイルはタブ区切りとして読む（jma_rainfall の10分値の既定の出力）。

列名は study.json で指定する。指定した列がなければエラーにする（別の列を黙って使わない）。
「CSV → 表」（read_table）と「表 → 部品」（pond_from_frame、hydrograph_from_frame、observed_from_frame）を分けてあるので、
WebGUI で編集した表（メモリ上の表）も同じ確認を通して部品にできる。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain.pond import TablePond
from ..domain.rainfall import NO_VALUE
from ..usecases.design_study import DesignResult, Inflow, ObservedRain, hydrograph_inflow

POND_COLUMNS = ("水位_m", "面積_m2", "容量_m3")
HYDROGRAPH_COLUMNS = ("時刻_分", "流入量_m3s")


def read_table(study_dir: Path, file: str) -> pd.DataFrame:
    """input/ などの CSV（.tsv はタブ区切り）を読む（列名の前後の空白は取る）。"""
    path = Path(study_dir) / file
    if not path.is_file():
        raise FileNotFoundError(f"入力ファイルがありません: {path}。study.json（WebGUI では該当するタブ）のファイル名を直すか、"
                                "そのパスにファイルを保存してください。")
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True, sep=sep,
                         dtype={"block_no": str})   # 地点番号の先頭の 0 を残す（0916 など）
    except UnicodeDecodeError:
        raise ValueError(f"{file} の文字コードが UTF-8 ではありません。UTF-8 で保存し直してください"
                         "（Excel では「CSV UTF-8 (コンマ区切り)」で保存する）。") from None
    df.columns = [str(c).strip() for c in df.columns]
    return df


def write_table(df: pd.DataFrame, path: Path):
    """表を CSV に書く（utf-8-sig、改行は CRLF）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\r\n")


def _columns(df: pd.DataFrame, cols: list[str], name: str):
    lack = [c for c in cols if c not in df.columns]
    if lack:
        raise KeyError(f"{name} に列 {', '.join(lack)} がありません（ファイルの列: {', '.join(map(str, df.columns))}）。"
                       "列は1行目の列名で探します。列名を直すか、study.json で使う列名を指定してください。")


def _numbers(df: pd.DataFrame, col: str, name: str) -> pd.Series:
    s = pd.to_numeric(df[col], errors="coerce")
    if s.isna().any():
        raise ValueError(f"{name} の列「{col}」に数値でない値か空欄があります（ファイルの {int(s.isna().idxmax()) + 2} 行目）。"
                         "その行に数値を入力してください。")
    return s


def pond_columns(spec: dict) -> list[str]:
    return [spec.get("level_col", POND_COLUMNS[0]), spec.get("area_col", POND_COLUMNS[1]),
            spec.get("volume_col", POND_COLUMNS[2])]


def pond_rows_of(df: pd.DataFrame, spec: dict) -> list[dict]:
    """池の表の CSV を WebGUI の表の行（列名 → 値）にする。使う列（pond_columns）だけを取り、空欄はキーなしにする
    （WebGUI で欄を空にしたときと同じ形。NaN を入れると変更の有無を誤る）。水位の列がなければ ValueError
    （面積・容量はどちらか一方を省けるが、水位は省けない。空の行を作らない）。"""
    cols = [c for c in pond_columns(spec) if c in df.columns]
    if pond_columns(spec)[0] not in cols:
        raise ValueError(f"池の表に水位の列 {pond_columns(spec)[0]} がありません（ファイルの列: {', '.join(map(str, df.columns))}）。"
                         f"CSV の1行目の水位の列名を {pond_columns(spec)[0]} にするか、study.json の pond.table.level_col に水位の列名を書いてください。")
    return [{k: v for k, v in r.items() if pd.notna(v)} for r in df[cols].to_dict("records")]


def hydrograph_columns(spec: dict) -> list[str]:
    cols = [spec.get("time_col", HYDROGRAPH_COLUMNS[0]), spec.get("q_col", HYDROGRAPH_COLUMNS[1])]
    if spec.get("rain_col"):
        cols.append(spec["rain_col"])
    return cols


def _optional_numbers(df: pd.DataFrame, col: str, name: str, named: bool) -> np.ndarray | None:
    """省ける列の数値。列がない（study.json で列名を指定していないとき）かすべて空欄なら None、一部だけ空欄ならエラー。"""
    if col not in df.columns:
        if named:
            _columns(df, [col], name)                   # 指定した列名がなければエラー（別の列を黙って使わない）
        return None
    blank = df[col].isna() | df[col].astype(str).str.strip().eq("")
    if blank.all():
        return None
    if blank.any():
        raise ValueError(f"{name} の列「{col}」の一部が空欄です（{int(blank.idxmax()) + 2} 行目）。"
                         "すべての行に書くか、列をすべて空欄にしてください。")
    return _numbers(df, col, name).to_numpy()


def pond_from_frame(df: pd.DataFrame, spec: dict, name: str = "池の表") -> TablePond:
    """池の表（水位と、面積・容量）から池を作る。spec: level_col, area_col, volume_col, hwl_m（省略時は表の最上段）。
    面積と容量はどちらか一方を省ける（列がないか、列がすべて空欄）。容量を省くと、面積から平均断面法で求める。"""
    lcol, acol, vcol = pond_columns(spec)
    _columns(df, [lcol], name)
    lv = _numbers(df, lcol, name).to_numpy()
    ar = _optional_numbers(df, acol, name, "area_col" in spec)
    vo = _optional_numbers(df, vcol, name, "volume_col" in spec)
    if vo is not None:
        return TablePond(lv, ar, vo, spec.get("hwl_m"))
    if ar is None:
        raise ValueError(f"{name} に面積（{acol}）か容量（{vcol}）のどちらかを書いてください。")
    return TablePond.from_areas(lv, ar, spec.get("hwl_m"))


def hydrograph_from_frame(df: pd.DataFrame, spec: dict, tc_min: int, name: str = "ハイドログラフ") -> Inflow:
    """ハイドログラフの表から流入を作る。spec: time_col（区切りの終わりの時刻 分）, q_col（流入量 m3/s）, rain_col（任意）,
    reverse（true なら値の並びを逆にする。強度の大きい順に並んだ表を後方集中型にするとき）。"""
    cols = hydrograph_columns(spec)
    _columns(df, cols, name)
    t = _numbers(df, cols[0], name).to_numpy()
    q = _numbers(df, cols[1], name).to_numpy()
    rain = _numbers(df, cols[2], name).to_numpy() if len(cols) > 2 else None
    if spec.get("reverse", False):             # true か false であることは study.json を読むときに確かめる（study_config.SCHEMA）
        q = q[::-1]
        rain = None if rain is None else rain[::-1]
    return hydrograph_inflow(t, q, tc_min, rain)


def inflow_frame(res: DesignResult) -> pd.DataFrame:
    """流出量の計算結果の表（解説 資料－4 の表－9 にあたる）。ハイドログラフの入力（inflow_csv）と同じ形なので、
    そのまま流入に使える（既定の列名 時刻_分・流入量_m3s。降雨強度は rain_col に 降雨強度_mmh を書けば読む）。

    行は流入の区切りの終わりの時刻 tc, 2tc, …（節点0と、最後の三角形が0に戻る節点は書かない。読み込みで足される）。
    流入量_m3s は池に入れた流入量（浸透を差し引いた後）。区域が2つ以上か区域ごとの浸透があれば区域ごとの流入量
    （浸透のある区域は浸透後も）、浸透があれば浸透前流入量_m3s も書く。値は追跡に使った丸める前の値。"""
    inflow, tc = res.inflow, res.tc_min
    k = np.arange(1, len(inflow.nodes) - 1)
    cols = {HYDROGRAPH_COLUMNS[0]: k * tc}
    if inflow.rain is not None:
        cols["降雨強度_mmh"] = np.asarray(inflow.rain, dtype=float)[k - 1]
    zoned = res.infiltration is None and bool(res.treated)
    after = {t.name: t.after for t in res.treated} if zoned else {}
    if len(inflow.zones) > 1 or zoned:
        for z in inflow.zones:
            cols[f"{z.catchment.name}の流入量_m3s"] = z.nodes[k]
            if z.catchment.name in after:
                cols[f"{z.catchment.name}の浸透後流入量_m3s"] = after[z.catchment.name][k]
    if res.treated:
        cols["浸透前流入量_m3s"] = inflow.nodes[k]
    cols[HYDROGRAPH_COLUMNS[1]] = res.pond_inflow[k]
    return pd.DataFrame(cols)


def load_pond_table(study_dir: Path, spec: dict) -> TablePond:
    """input/ の池の表の CSV から池を作る。spec: file と pond_from_frame の項目。"""
    return pond_from_frame(read_table(study_dir, spec["file"]), spec, spec["file"])


def load_hydrograph(study_dir: Path, spec: dict, tc_min: int) -> Inflow:
    """input/ のハイドログラフの CSV から流入を作る。spec: file と hydrograph_from_frame の項目。"""
    return hydrograph_from_frame(read_table(study_dir, spec["file"]), spec, tc_min, spec["file"])


# 実測降雨（studies/_jma/ の10分値・時間値）------------------------------------------
# 10分値: 地点, block_no, 日付, 時分（00:10〜24:00）, 区間終了時刻, 降水量_mm, 品質
# 時間値: 地点, block_no, 日付, 時（1〜24）, 降水量_mm, 品質
# 列はヘッダの名前で決める（位置で決めない）。値がない品質（domain.rainfall.NO_VALUE）の区間と、取得していない区間は0mm で計算する。
OBSERVED_STEP = {"時分": 10, "時": 60}
OBSERVED_KIND = {10: "10分値", 60: "時間値"}
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def check_period(start, end) -> None:
    """実測降雨の開始日・終了日が 2013-10-15 の形で、開始日が終了日以前か。違えばエラー
    （日付は文字列で比べるので、形が違うと期間の値が黙って0行になる）。"""
    for key, v in (("start", start), ("end", end)):
        if not isinstance(v, str) or not DATE.match(v):
            raise ValueError(f"実測降雨の{'開始日' if key == 'start' else '終了日'}（observed_rain.{key}）の「{v}」を日付として読めません。"
                             "2013-10-15 の形で書いてください。")
    if start > end:
        raise ValueError(f"実測降雨の開始日（observed_rain.start）{start} が終了日（end）{end} より後です。開始日を終了日以前の日にしてください。")


def _time_col(df: pd.DataFrame, name: str) -> str:
    for col in OBSERVED_STEP:
        if col in df.columns:
            return col
    raise KeyError(f"{name} に列 時分（10分値）か 時（時間値）がありません（ある列: {', '.join(map(str, df.columns))}）。"
                   "気象庁の10分値・時間値のファイル（jma_rainfall の出力）を選んでください。")


def observed_from_frame(df: pd.DataFrame, spec: dict, name: str = "実測降雨") -> ObservedRain:
    """1地点の10分値・時間値の表（studies/_jma/ の行と同じ列）から実測降雨を作る。spec: station（表示に使う地点名）、
    start・end（日付 YYYY-MM-DD。その日の 0:00〜24:00 を含む。省略時は表の最初・最後の日）。
    欠測の区間と、表にない区間（取得していない日など）は 0 mm とし、それぞれの時間を記録する。
    同じ区間が2回ある、時刻が区切りに合わない場合はエラーにする（どの値を使うか決められないため）。"""
    col = _time_col(df, name)
    step = OBSERVED_STEP[col]
    _columns(df, ["日付", col, "降水量_mm", "品質"], name)
    station = spec.get("station")
    dates = df["日付"].astype(str)
    if dates.empty:
        raise ValueError(f"{name} に値の行がありません（1行目の列名だけです）。気象庁の10分値・時間値を取得したファイルを選んでください。")
    start, end = spec.get("start", dates.min()), spec.get("end", dates.max())      # 空の文字列は下で日付として読めずエラー
    check_period(start, end)
    seg = df[(dates >= start) & (dates <= end)]
    if seg.empty:
        raise ValueError(f"{name} に {start}〜{end} の行がありません（ファイルの期間は {dates.min()}〜{dates.max()}）。"
                         "実測降雨の開始日・終了日を、ファイルの期間の中にしてください。")
    if col == "時分":
        hm = seg[col].astype(str).str.strip().str.split(":", expand=True)
        minutes = pd.to_numeric(hm[0], errors="coerce") * 60 + pd.to_numeric(hm[1], errors="coerce")
    else:
        minutes = pd.to_numeric(seg[col], errors="coerce") * 60
    if minutes.isna().any():
        raise ValueError(f"{name} の列「{col}」に時刻として読めない値があります。気象庁の10分値・時間値を取得したファイル（jma-rainfall の出力）を選んでください。")
    ends = pd.to_datetime(seg["日付"].astype(str)) + pd.to_timedelta(minutes, unit="min")
    begin = pd.Timestamp(start)
    want = pd.date_range(begin + pd.Timedelta(minutes=step), pd.Timestamp(end) + pd.Timedelta(days=1), freq=f"{step}min")
    if ends.duplicated().any():
        t = ends[ends.duplicated()].iloc[0]
        raise ValueError(f"{name} に {t:%Y-%m-%d %H:%M} までの区間の値が2回あり、どちらを使うか決められないので中止しました。"
                         "同じ期間の値を重ねて保存したファイルか、複数の地点の値を含むファイルです。気象庁から取得し直したファイルを選んでください。")
    extra = pd.DatetimeIndex(ends).difference(want)
    if len(extra):
        raise ValueError(f"{name} の時刻 {extra[0]:%Y-%m-%d %H:%M} が{OBSERVED_KIND[step]}の区切りになっていません"
                         "（10分値は10分ごと、時間値は毎正時）。気象庁の10分値・時間値を取得したファイル（jma-rainfall の出力）を選んでください。")
    # 期間のすべての区間に並べ直す。表にない区間（取得していない日など）と欠測の区間は 0 mm とし、それぞれ記録する
    num = pd.to_numeric(seg["降水量_mm"], errors="coerce")
    text = seg["降水量_mm"].notna() & num.isna()        # 空欄は欠測。数に読めない文字（O.5 など）を欠測として黙って 0 mm にしない
    if text.any():
        r = seg[text].iloc[0]
        more = f"（ほかに {int(text.sum()) - 1} 行）" if text.sum() > 1 else ""
        raise ValueError(f"{name} の {r['日付']} {r[col]} の降水量_mm「{r['降水量_mm']}」を数として読めません{more}。"
                         "値を直すか、欠測なら空欄にしてください。")
    val = num.to_numpy(dtype=float)
    bad = np.isnan(val) | seg["品質"].astype(str).isin(NO_VALUE).to_numpy()
    if (val[~bad] < 0).any():
        raise ValueError(f"{name} の降水量_mm に負の値があります。気象庁の10分値・時間値を取得したファイル（jma-rainfall の出力）を選んでください。")
    at = want.get_indexer(pd.DatetimeIndex(ends))
    mm = np.zeros(len(want))
    mm[at] = np.where(bad, 0.0, val)
    missing, absent = np.zeros(len(want), dtype=bool), np.ones(len(want), dtype=bool)
    missing[at] = bad
    absent[at] = False
    label = f"{station + ' ' if station else ''}{OBSERVED_KIND[step]} {start}〜{end}"
    return ObservedRain(label, begin.to_pydatetime(), step, mm, _spans(missing, want, step), _spans(absent, want, step))


def _spans(flags: np.ndarray, ends: pd.DatetimeIndex, step: int) -> tuple:
    """flags が続く区間ごとの（始まり, 終わり）。ends は各区間の終わりの時刻。"""
    f = np.concatenate([[False], flags, [False]]).astype(int)
    first, last = np.flatnonzero(np.diff(f) == 1), np.flatnonzero(np.diff(f) == -1) - 1
    width = pd.Timedelta(minutes=step)
    return tuple(((ends[a] - width).to_pydatetime(), ends[b].to_pydatetime()) for a, b in zip(first, last))


OBSERVED_SERIES = {"tenmin": "10分値", "hourly": "時間値"}      # study.json の observed_rain.series
OBSERVED_ITEMS = {"series": "値の種類", "station": "地点", "start": "開始日", "end": "終了日"}      # study.json の observed_rain の項目


def fetch_command(series: str, station: str = "<地点>", start: str = "<開始日>", end: str = "<終了日>") -> str:
    """実測降雨の値を取得するコマンド（10分値は期間、時間値は日の一覧のファイルで取得する日を指定する）。"""
    if series == "tenmin":
        return f"amamizu fetch <検討フォルダ> tenmin --stations {station} --start {start} --end {end}"
    return f"amamizu fetch <検討フォルダ> hourly --stations {station} --dates-file <{start}〜{end} の日を1行に1日書いたファイル>"


def load_observed(store, spec: dict) -> ObservedRain:
    """study.json の observed_rain（station 地点、series 値の種類（tenmin 10分値、hourly 時間値）、start・end 日付）の実測降雨を、
    studies/_jma/ の値（store は JmaStore）から作る。その期間の値がなければ、取得のコマンドを示してエラーにする。"""
    lack = [k for k in OBSERVED_ITEMS if k not in spec]
    if lack:
        how = "（取得済みの期間の一覧から選ぶか、日付を入力します）" if {"start", "end"} & set(lack) else ""
        raise KeyError(f"実測降雨の{'・'.join(OBSERVED_ITEMS[k] for k in lack)}を選んでください{how}。"
                       f"study.json では observed_rain の {'・'.join(lack)} に書きます。")
    station, series, start, end = spec["station"], spec["series"], spec["start"], spec["end"]
    if series not in OBSERVED_SERIES:
        raise ValueError(f"実測降雨の値の種類（observed_rain.series）の「{series}」は扱えません。tenmin（10分値）か hourly（時間値）にしてください。")
    check_period(start, end)
    rows = store.observed(station, series, start, end)
    kind = OBSERVED_SERIES[series]
    if rows.empty:
        raise ValueError(f"studies/_jma/ に{station}の{kind}の {start}〜{end} の値がありません。先に {fetch_command(series, station, start, end)} "
                         "で取得してください。")
    return observed_from_frame(rows, spec, f"studies/_jma/ の{station}の{kind}")
