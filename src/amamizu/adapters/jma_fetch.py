"""気象庁からの取得（jma-rainfall の jma_rainfall パッケージを使う）。

取得した値は、観測値のフォルダ studies/_jma/ に JmaStore.put で書く（取得日で暫定と確定を分け、暫定の値は取り直して上書きする）。
取得するのは、値がそろっていない日と暫定の値のある日だけ。歩調は Pacer（2秒間隔・200回ごとに60秒休止・3回連続失敗で打ち切り）。
"""
import datetime
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from jma_rainfall import stations as st
from jma_rainfall.annual import fetch_annual, fetch_rank
from jma_rainfall.daily import JMADailyCollector
from jma_rainfall.fetch import FetchAborted, FetchError, Pacer, new_session, record_failure
from jma_rainfall.hourly import JMAHourlyCollector
from jma_rainfall.tables import TableError, parse_value
from jma_rainfall.tenmin import JMATenMinCollector

from .jma_store import JmaStore, StationMismatch

log = logging.getLogger(__name__)


class JmaFetcher:
    def __init__(self, studies: Path | str, pref: str | None = None, interval: float | None = None, verify_ssl: bool = True,
                 accept: bool = False):
        """studies は検討のフォルダ（その下の _jma/ に書く）。accept なら、確定の値と違う取得値でも上書きして revisions.tsv に記録する。"""
        self.store = JmaStore(studies)
        self.pref = pref
        self.master = st.load_stations()
        self.pacer = Pacer() if interval is None else Pacer(interval=interval)   # 既定の間隔は Pacer の既定
        self.session = new_session()   # サーバ証明書を Windows の証明書ストアで検証する（社内網の UTM の SSL 検査にも対応）
        self.verify_ssl = verify_ssl
        self.accept = accept
        self.failures: list[dict] = []
        self.conflicts: list[str] = []      # 確定の値と違った行（書かなかった）

    def station(self, name: str) -> dict:
        """地点名から観測所を決める。studies/_jma/ に同じ地点名の別の観測所（「金山」など）があれば、取得を始める前にエラーにする。"""
        s = st.find_station(name, self.master, pref=self.pref)
        self.store.check_station(name, s["block_no"])        # 取得を始める前に確かめる（書くときも put が確かめる）
        for note in st.limitations(s):
            log.warning("注意: %s の「%s」（地点番号（block_no）%s）は、気象庁の観測所の一覧で「%s」となっている地点です。"
                        "値を取得できない期間があります。", s["pref"], s["name"], s["block_no"], note)
        return s

    @staticmethod
    def _info(s: dict) -> dict:
        return {"地点": s["name"], "block_no": s["block_no"], "pref": s["pref"]}

    def _put(self, kind: str, s: dict, rows: pd.DataFrame, what: str):
        r = self.store.put(kind, self._info(s), rows, datetime.date.today(), f"amamizu fetch {what}", accept=self.accept)
        self.conflicts += r.conflicts

    def _collect(self, col, collect: Callable[[], object], put: Callable[[], None]) -> None:
        """collect() で取得し、取得済みの行を put() で書く。取得が途中で終わっても（打ち切り FetchAborted、中断など）、取得済みの行は書く。
        打ち切りの後に取得済みの行を書けなかったときは、打ち切りの文に書けなかった理由を足した FetchAborted にする
        （書けなかった理由だけが表示され、打ち切りが表示されなくなるのを防ぐ）。"""
        aborted: FetchAborted | None = None
        try:
            collect()
        except FetchAborted as e:
            aborted = e
            raise
        finally:
            self.failures += col.failures
            try:
                put()
            except (ValueError, OSError) as e:
                if aborted is None:
                    raise
                raise FetchAborted(f"{aborted}\n  取得済みの分は保存できませんでした: {e}") from e

    def daily(self, name: str, year_from: int, year_to: int):
        """日別値を取得する（昨日まで。当日はまだ公表されていない）。要る日（値がない日と暫定の日）のある月だけ、
        月ごとにその月の要る日の最初から最後までを取得する（日別値のページは1か月ごと）。
        年ごとに書き、1日でも取れなかった年は書かない（再実行で取り直す）。"""
        s = self.station(name)
        last = datetime.date.today() - datetime.timedelta(days=1)
        days = [d.date() for d in pd.date_range(f"{year_from}-01-01", min(datetime.date(year_to, 12, 31), last))]
        need = self.store.needed_days("daily", self._info(s), days)
        for y in sorted({d.year for d in need}):
            col = JMADailyCollector(s, pacer=self.pacer, session=self.session, verify_ssl=self.verify_ssl)
            parts = []
            try:
                for m in sorted({d.month for d in need if d.year == y}):
                    span = [d for d in need if (d.year, d.month) == (y, m)]
                    parts.append(col.collect(span[0], span[-1]))
            finally:
                self.failures += col.failures
            df = pd.concat(parts, ignore_index=True)
            if df.empty and not col.failures:       # 失敗なしで行が0（その年に観測していないなど）も、取れなかった年として知らせる
                record_failure(self.failures, s, f"{y}年の日別値", "取得できた行がありません。その年に観測しているか、気象庁のページで確かめてください")
            if col.failures or df.empty:
                continue
            self._put("daily", s, df, f"daily {y}年")   # 1年ごとに書く（中断しても取得済みの年は残る）

    def hourly(self, name: str, dates: list[datetime.date]):
        """時間値を指定の日のうち要る日（値がそろっていない日と暫定の日）だけ取得する。打ち切られても取得済みの日は書く。"""
        s = self.station(name)
        need = self.store.needed_days("hourly", self._info(s), dates)
        got: list[dict] = []
        col = JMAHourlyCollector(s, verify_ssl=self.verify_ssl, pacer=self.pacer, session=self.session)

        def put():
            if got:
                self._put("hourly", s, pd.DataFrame(got), f"hourly {len({r['日付'] for r in got})} 日")
        self._collect(col, lambda: col.collect(need, on_day=lambda d, recs: got.extend(recs)), put)

    def tenmin(self, name: str, start: datetime.date, end: datetime.date):
        """10分値を期間のうち要る日（値がそろっていない日と暫定の日）だけ取得する。打ち切られても取得済みの日は書く。"""
        s = self.station(name)
        need = self.store.needed_days("tenmin", self._info(s), [d.date() for d in pd.date_range(start, end)])
        col = JMATenMinCollector(s, start, end, output_dir=self.store.root, verify_ssl=self.verify_ssl,   # ファイルは書かない（値は frame から）
                                 pacer=self.pacer, session=self.session)

        def put():
            if col.records:
                self._put("tenmin", s, col.frame(), f"tenmin {start}〜{end}")
        self._collect(col, lambda: col.collect(need), put)

    def annual_max(self, names: list[str]):
        """年最大値系列（年ごとの値の view="" の最大10分間・最大1時間降水量）と観測史上順位を、地点ごとに取得する（取得のたびに全部の年）。
        取れなかったものは failures に残し、残りを取得する。"""
        for n in names:
            s = self.station(n)
            for what, fetch in (("年最大値系列", lambda: annual_max_rows(fetch_annual(s, "", self.session, self.verify_ssl))),
                                ("観測史上順位", lambda: rank_record(fetch_rank(s, self.session, self.verify_ssl)))):
                failed = False
                try:
                    got = fetch()                   # 表の形が違えば TableError（取れなかったものとして残す）
                except (FetchError, TableError) as e:
                    failed = True
                    record_failure(self.failures, s, what, str(e))
                else:
                    if what == "年最大値系列":
                        self._put("annual_max", s, got, "annual_max")
                    else:
                        self.store.put_rank(self._info(s), got, datetime.date.today(), "amamizu fetch annual_max")
                self.pacer.done(failed)

    def annual(self, names: list[str], view: str = "a5"):
        """年ごとの値（N時間降水量。気象庁のページの view=a5）を地点ごとに取得する。取れなかった地点は failures に残し、残りの地点を取得する。"""
        if view != "a5":
            raise ValueError(f"年ごとの値は view a5（N時間降水量）だけを保存できます（今は {view!r}）。")
        for n in names:
            s = self.station(n)
            failed = False
            try:
                df = fetch_annual(s, view, self.session, self.verify_ssl)
            except (FetchError, TableError) as e:
                failed = True
                record_failure(self.failures, s, f"年ごとの値（view={view!r}）", str(e))
            else:
                self._put("annual", s, df.drop(columns=[c for c in ("地点", "block_no") if c in df.columns]), f"annual {view}")
            self.pacer.done(failed)


# 年ごとの値（view=""）の年最大値の列。見出しの形は種別で違う（官署 降水量(mm)/最大/10分間、アメダス 降水量/最大/10分間(mm)）ので、
# 見出しに「降水量」「最大」と次の語を含む列が1列に決まることを確かめて使う
ANNUAL_MAX_COLUMNS = {"10分": "10分間", "1時間": "1時間"}


def annual_max_rows(df: pd.DataFrame) -> pd.DataFrame:
    """年ごとの値（view=""）の表から、年最大値系列の行（年、10分、10分_品質、1時間、1時間_品質）を作る。
    値の記号（`)`・`]`・`×` など）は jma_rainfall の parse_value で品質にする。"""
    cols = {}
    for key, word in ANNUAL_MAX_COLUMNS.items():
        hits = [c for c in df.columns if "降水量" in c and "最大" in c and word in c]
        if len(hits) != 1:
            raise TableError(f"年ごとの値の表の見出しから、降水量の最大「{word}」の列を1列に決められません（該当 {len(hits)} 列）。"
                             f"表の見出し: {'、'.join(df.columns)}")
        cols[key] = hits[0]
    rows = []
    for _, r in df.iterrows():
        row = {"年": int(r["年"])}
        for key, col in cols.items():
            row[key], row[f"{key}_品質"] = parse_value(r[col])
        rows.append(row)
    return pd.DataFrame(rows)


def rank_record(df: pd.DataFrame) -> dict:
    """観測史上順位の表（要素, 1位〜10位, 統計期間）を、要素ごとの辞書にする（値は公表の文字列のまま）。"""
    return {r["要素"]: {c: r[c] for c in df.columns if c not in ("地点", "block_no", "要素")} for _, r in df.iterrows()}
