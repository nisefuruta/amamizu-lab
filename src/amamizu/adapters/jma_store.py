"""気象庁の観測値を地点ごとに1か所で持つ（studies/_jma/。StationRecords の実装）。

地点ごとのフォルダ <地点名>_<地点番号>/ に、日別値 daily.csv、時間値 hourly.csv、10分値 tenmin/<年>.tsv、
年ごとの値 annual_a5.csv（N時間降水量）、年最大値系列 annual_max.csv（年最大10分間・1時間降水量と品質）、
観測史上順位 rank.json、地点の情報 station.json を置く。どの行（rank.json はファイル）にも取得日がある。
取得の記録は fetch_log.tsv、値の訂正の記録は revisions.tsv（どちらも _jma/ の直下）。

暫定と確定: 観測日から config.json の refresh_months か月の間に取得した値は暫定とし、取得のたびに取り直して上書きする
（準正常値・資料不足値が後日正常値に直ることがある）。変わった値は revisions.tsv に記録する。
それより後に取得した値は確定とし、取り直して値が違えば、その行は書かずに一覧で知らせる（accept を指定したときだけ書く）。
ファイルは jma-rainfall の出力と同じ UTF-8（CSV・TSV は BOM 付き）、改行は CRLF。書き込みはすべて _write_bytes を通る。
"""
import datetime
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain.rainfall import NO_VALUE

FOLDER = "_jma"                 # 検討のフォルダ（studies/）の下の、観測値のフォルダ
CONFIG = "config.json"
FETCHED = "取得日"
RANK = "rank.json"


@dataclass(frozen=True)
class Kind:
    name: str                   # 表示の名前
    path: str                   # 地点のフォルダから見たファイル（{year} は観測の年）
    key: list[str]              # 行を決める列
    value: str | None = None    # 雨量の列（数として確かめる）
    quality: str | None = None  # 品質の列
    columns: list[str] | None = None      # 列の並び（None なら取得した表のまま）
    per_day: int = 0            # 1日の行数（そろっているかの判定。0 は日ごとでない種類）

    @property
    def yearly(self) -> bool:
        return self.key == ["年"]


KINDS = {
    "daily": Kind("日別値", "daily.csv", ["日付"], "日合計_日別値_mm", "日別値_品質",
                  ["地点", "block_no", "日付", "日合計_日別値_mm", "最大1時間_日別値_mm", "最大10分_日別値_mm", "日別値_品質"], 1),
    "hourly": Kind("時間値", "hourly.csv", ["日付", "時"], "降水量_mm", "品質", ["地点", "block_no", "日付", "時", "降水量_mm", "品質"], 24),
    "tenmin": Kind("10分値", "tenmin/{year}.tsv", ["日付", "時分"], "降水量_mm", "品質",
                   ["地点", "block_no", "日付", "時分", "区間終了時刻", "降水量_mm", "品質"], 144),
    "annual": Kind("年ごとの値", "annual_a5.csv", ["年"]),
    "annual_max": Kind("年最大値系列", "annual_max.csv", ["年"]),
}
KIND_NAMES = {k: v.name for k, v in KINDS.items()} | {"rank": "観測史上順位"}
LOG_COLUMNS = ["取得日", "地点", "block_no", "種類", "期間", "追加", "同じ", "上書き", "出所"]
REVISION_COLUMNS = ["取得日", "地点", "block_no", "種類", "日時", "前の値", "新しい値", "前の取得日", "出所"]


def rain_values(df: pd.DataFrame, value: str, quality: str, name: str) -> pd.DataFrame:
    """雨量の列を数にする。数に読めない値や、品質が値のある品質（正常値・準正常値など）なのに空欄の値があればエラーにする
    （値のない品質は missing・doubtful・blank だけ。それ以外の空欄を 0 mm や欠測として黙って扱わない）。"""
    num = pd.to_numeric(df[value], errors="coerce")
    bad = (df[value].notna() & num.isna()) | (num.isna() & ~df[quality].isin(NO_VALUE))
    if bad.any():
        r = df[bad].iloc[0]
        when = " ".join(str(r[c]) for c in ("日付", "時分", "時") if c in df.columns)
        shown = "空欄" if pd.isna(r[value]) else f"「{r[value]}」"
        more = f"（ほかに {int(bad.sum()) - 1} 行）" if bad.sum() > 1 else ""
        raise ValueError(f"{name} の {r['地点']} {when} の「{value}」（{shown}、品質 {r[quality]}）を雨量として読めません{more}。"
                         "値のない行の品質は missing・doubtful・blank のどれかです。")
    return df.assign(**{value: num})


@dataclass(frozen=True)
class Period:
    """取得済みの日が続いている期間（JmaStore.periods）。"""
    start: str
    end: str
    days: int
    total_mm: float             # 期間の雨量の合計（欠測の区間は含めない）
    short_days: int             # 行のそろっていない日の数（取得していない区間は0mm として計算する）


class StationMismatch(ValueError):
    """studies/_jma/ にある地点名で、別の観測所（地点番号の違うもの）の値を書こうとした。"""


@dataclass
class PutResult:
    added: int = 0
    same: int = 0
    replaced: int = 0
    conflicts: list[str] = field(default_factory=list)      # 確定の値と違う行（その行は書かなかった）


def store_dir(studies: Path | str) -> Path:
    return Path(studies) / FOLDER


def _write_bytes(path: Path, data: bytes) -> None:
    """studies/_jma/ のファイルを書く（書き込みはすべてここを通る）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _station_name(folder: Path) -> str | None:
    """地点のフォルダ（<地点名>_<地点番号>）の地点名。地点のフォルダでなければ（フォルダでない、名前に「_」がない）None。"""
    return folder.name.rsplit("_", 1)[0] if folder.is_dir() and "_" in folder.name else None


class JmaStore:
    def __init__(self, studies: Path | str):
        self.root = store_dir(studies)
        cfg = self.root / CONFIG
        if not cfg.exists():
            raise FileNotFoundError(f"観測値のフォルダの設定 {cfg} がありません。refresh_months（観測日から何か月の間に取得した値を暫定とし、"
                                    "取得のたびに取り直すか）を書いてください。")
        try:
            conf = json.loads(cfg.read_text(encoding="utf-8"))
        except ValueError as e:              # JSON の書き方の誤り、UTF-8 でない文字コード
            raise ValueError(f"観測値のフォルダの設定 {cfg} を JSON として読めません（{e}）。"
                             '{"refresh_months": 整数} の形で書いてください（観測日から何か月の間に取得した値を暫定とするか）。') from None
        months = conf.get("refresh_months") if isinstance(conf, dict) else None
        if isinstance(months, bool) or not isinstance(months, int) or months < 0:
            raise ValueError(f"{cfg} の refresh_months（{months!r}）は0以上の整数で書いてください（観測日から何か月の間に取得した値を暫定とするか）。")
        self.refresh_months: int = months
        # (種類, 地点) → 読んだ期間（最初, 最後）のリスト。重なる期間は1つにする。検討が使った観測値の記録（used）に使う
        self.reads: dict[tuple[str, str], list[tuple[str, str]]] = {}

    def note_read(self, kind: str, station: str, first: str, last: str) -> None:
        """読んだ期間を記録する。重なる期間は1つにまとめ、離れた期間は別に持つ（使っていない間の行の変化を、使った値の変化としない）。"""
        spans = sorted([*self.reads.get((kind, station), []), (first, last)])
        merged: list[tuple[str, str]] = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self.reads[(kind, station)] = merged

    # 地点 ----------------------------------------------------------------
    def _station_folders(self, name: str) -> list[Path]:
        """地点名 name のフォルダ（<地点名>_<地点番号>）。地点名のフォルダを探す処理はすべてここを通る。
        name は study.json に書いた取得していない地点名のこともあるので、フォルダの名前と比べるだけにして、glob のパターンにしない
        （「/」で始まる名前と「**」は glob がエラーにし、「[」を含む名前はその名前のフォルダがあっても見つからない）。"""
        return sorted(p for p in self.root.iterdir() if _station_name(p) == name)

    def check_station(self, name: str, block: str) -> None:
        """地点名 name で地点番号 block の値を書いてよいか。studies/_jma/ では地点名が1つの観測所だけを指す（書くのは put と put_rank だけで、
        どちらもこれを確かめるので、同じ地点名のフォルダは2つできない）。別の観測所の値があればエラー。"""
        have = [p.name.rsplit("_", 1)[1] for p in self._station_folders(name)]
        if have and str(block) not in have:
            raise StationMismatch(
                f"{self.root} には、地点「{name}」（地点番号（block_no）{'、'.join(have)}）の値を取得済みです。"
                f"地点番号 {block} は地点名が同じ別の観測所なので、値を混ぜないよう中止しました。"
                "取得済みの観測所の続きを取得するなら、--pref でその都道府県を指定してください。")

    def station_dir(self, name: str) -> Path:
        """地点名のフォルダ（<地点名>_<地点番号>）。なければ FileNotFoundError（同じ地点名のフォルダは put・put_rank が作らせない）。
        手でフォルダを足して2つあるときは、どちらを使うか決められないので ValueError（値がないときの FileNotFoundError と分ける。
        値がないものとして扱うと、取得済みの日をすべて取得し直したり、取得の案内だけを表示したりする）。"""
        found = self._station_folders(name)
        if not found:
            raise FileNotFoundError(f"{self.root} に地点「{name}」の観測値がありません。amamizu fetch で取得してください。")
        if len(found) > 1:
            raise ValueError(f"{self.root} に地点「{name}」の観測所が {len(found)} つあります（{'、'.join(p.name for p in found)}）。"
                             "地点名が同じ別の観測所なので、どちらを使うか決められません。使わないほうのフォルダを studies/_jma/ の外へ移してください。")
        return found[0]

    def _ensure_station(self, station: dict) -> Path:
        """書く地点のフォルダ（なければ作り、station.json を書く）。"""
        name, block = station["地点"], str(station["block_no"])
        self.check_station(name, block)
        folder = self.root / f"{name}_{block}"
        info = folder / "station.json"
        if not info.exists():
            text = json.dumps({"地点": name, "block_no": block, "pref": station.get("pref")}, ensure_ascii=False, indent=1) + "\n"
            _write_bytes(info, text.replace("\n", "\r\n").encode("utf-8"))
        return folder

    def _read(self, path: Path, kind: str) -> pd.DataFrame:
        sep = "\t" if path.suffix == ".tsv" else ","
        # 年ごとの値は「70 ]」のような記号付きの文字列を含むので、書いてあるとおりの文字列で読む
        types = str if kind == "annual" else {"block_no": str, FETCHED: str, "10分_品質": str, "1時間_品質": str}
        df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", dtype=types)
        k = KINDS[kind]
        return rain_values(df, k.value, k.quality, str(path.relative_to(self.root))) if k.value and k.quality else df

    def _frame(self, kind: str, station: str, years: range | None = None) -> pd.DataFrame:
        """地点・種類のファイルを読んでつないだ表（years は10分値の年ファイルを絞るとき）。ファイルがなければ取得の案内のエラー。"""
        d = self.station_dir(station)
        k = KINDS[kind]
        files = ([p for y in years if (p := d / k.path.format(year=y)).exists()] if years is not None
                 else sorted(d.glob(k.path.format(year="*"))))
        if not files:
            raise FileNotFoundError(f"{d} に{k.name}がありません。amamizu fetch で取得してください。")
        return pd.concat([self._read(p, kind) for p in files], ignore_index=True)

    def final(self, kind: str, t: pd.DataFrame) -> pd.Series:
        """行ごとに確定か（観測日の refresh_months か月後の同じ日（その月にない日は月末）以後に取得した行）。
        年ごとの種類の観測日はその年の12月31日。"""
        observed = pd.to_datetime(t[KINDS[kind].key[0]].astype(str) + ("-12-31" if KINDS[kind].yearly else ""))
        return pd.to_datetime(t[FETCHED]) >= observed + pd.DateOffset(months=self.refresh_months)

    # 読む（StationRecords） -----------------------------------------------------
    def daily(self, station: str, year_from: str = "") -> pd.DataFrame:
        df = self._frame("daily", station).sort_values("日付").reset_index(drop=True)
        df = df[["日付", "日合計_日別値_mm", "最大1時間_日別値_mm", "日別値_品質"]]
        df = df[df["日付"] >= year_from].reset_index(drop=True) if year_from else df
        if len(df):
            self.note_read("daily", station, df["日付"].iat[0], df["日付"].iat[-1])
        return df

    def hourly(self, station: str) -> pd.DataFrame:
        df = self._frame("hourly", station)[KINDS["hourly"].columns].sort_values(["日付", "時"]).reset_index(drop=True)
        if len(df):
            self.note_read("hourly", station, df["日付"].iat[0], df["日付"].iat[-1])
        return df

    def tenmin(self, station: str, start: str, end: str) -> pd.DataFrame | None:
        try:
            t = self._frame("tenmin", station, range(int(start[:4]), int(end[:4]) + 1))
        except FileNotFoundError:
            return None
        seg = t[(t["日付"] >= start) & (t["日付"] <= end)][KINDS["tenmin"].columns].sort_values(["日付", "時分"])
        if len(seg) != len(pd.date_range(start, end)) * 144:
            return None
        self.note_read("tenmin", station, start, end)
        return seg.reset_index(drop=True)

    def observed(self, station: str, series: str, start: str, end: str) -> pd.DataFrame:
        """設計の検討の実測降雨: station の series（tenmin 10分値、hourly 時間値）の start〜end にある行（そろっていない日もそのまま返す。
        ない区間は、実測降雨を作るときに「取得していない時間」として記録する）。"""
        if series not in ("tenmin", "hourly"):
            raise ValueError(f"実測降雨の値の種類 series（{series!r}）は tenmin（10分値）か hourly（時間値）にしてください。")
        t = self._frame(series, station)
        k = KINDS[series]
        t = t[(t["日付"] >= start) & (t["日付"] <= end)][k.columns].sort_values(k.key).reset_index(drop=True)
        self.note_read(series, station, start, end)
        return t

    def periods(self, series: str, station: str) -> list[Period]:
        """station の series（tenmin・hourly）を取得済みの日が続いている期間（新しい順。実測降雨の期間の選択肢）。
        値は候補降雨などの事象ごとに取得するので、続いている期間がほぼ1つの降雨になる。行のそろっていない日の数も数える。
        一覧を示すだけで計算には使わないので、読んだ期間（note_read）には記録しない。地点や値の種類のファイルがなければ取得の案内のエラー。"""
        t = self._frame(series, station)
        k = KINDS[series]
        g = t.groupby("日付")[k.value].agg(["size", "sum"])
        days = pd.to_datetime(g.index.to_series())
        run = (days.diff() != pd.Timedelta(days=1)).cumsum().to_numpy()
        out = [Period(str(r.index[0]), str(r.index[-1]), len(r), float(r["sum"].sum()), int((r["size"] < k.per_day).sum()))
               for _, r in g.groupby(run)]
        return out[::-1]

    def stations_with(self, series: str) -> list[str]:
        """series（tenmin・hourly）の値を取得済みの地点名（実測降雨の地点の選択肢）。"""
        pattern = KINDS[series].path.format(year="*")
        return sorted(n for p in self.root.iterdir() if (n := _station_name(p)) is not None and any(p.glob(pattern)))

    def pref(self, name: str) -> str | None:
        """地点名 name の観測所の都道府県（station.json の pref。地点の選択肢の表示に使う）。
        同じ地点名のフォルダが1つでないとき、station.json か pref がないときは None。station.json が項目を持つ形でなければエラー。"""
        found = self._station_folders(name)
        if len(found) != 1 or not (found[0] / "station.json").exists():
            return None
        info = json.loads((found[0] / "station.json").read_text(encoding="utf-8"))
        if not isinstance(info, dict):
            raise ValueError(f"{found[0] / 'station.json'} は項目を持つ形（{{\"地点\": …, \"block_no\": …, \"pref\": …}}）で書いてください。")
        pref = info.get("pref")
        return pref if isinstance(pref, str) and pref else None

    def annual_max(self, station: str, first: int, last: int) -> pd.DataFrame:
        """年最大値系列の first〜last 年（年、10分、10分_品質、1時間、1時間_品質。値のない年は空欄）。"""
        t = self._frame("annual_max", station).drop(columns=["地点", "block_no", FETCHED]).sort_values("年")
        t = t[(t["年"].astype(int) >= first) & (t["年"].astype(int) <= last)].reset_index(drop=True)
        self.note_read("annual_max", station, str(first), str(last))
        return t

    def fetched_on(self, kind: str, station: str) -> str:
        """地点・種類の最も新しい取得日（YYYY-MM-DD）。観測史上順位（rank）はそのファイルの取得日。"""
        if kind == "rank":
            return str(self._rank_file(station)[FETCHED])
        return str(self._frame(kind, station)[FETCHED].max())

    def _rank_file(self, station: str) -> dict:
        path = self.station_dir(station) / RANK
        if not path.exists():
            raise FileNotFoundError(f"{path.parent} に観測史上順位（{RANK}）がありません。amamizu fetch <検討フォルダ> annual_max で取得してください。")
        return json.loads(path.read_text(encoding="utf-8"))

    def rank(self, station: str) -> dict:
        """観測史上順位（気象庁の観測史上1〜10位の表の項目と、統計期間）。"""
        self.note_read("rank", station, "", "")
        return {k: v for k, v in self._rank_file(station).items() if k != FETCHED}

    def put_rank(self, station: dict, record: dict, fetched: datetime.date, source: str) -> None:
        """観測史上順位を rank.json に書く（その時点の順位の表なので、取り直すと置き換える）。"""
        folder = self._ensure_station(station)
        text = json.dumps({FETCHED: fetched.isoformat(), **record}, ensure_ascii=False, indent=1) + "\n"
        _write_bytes(folder / RANK, text.replace("\n", "\r\n").encode("utf-8"))
        self._append("fetch_log.tsv", [[fetched.isoformat(), station["地点"], str(station["block_no"]), "rank", "", 1, 0, 0, source]],
                     LOG_COLUMNS)

    def annual_nhour(self, stations: list[str]) -> pd.DataFrame:
        """年ごとの値の N時間降水量（stations の地点だけ。照合する地点を明示する）。"""
        parts = []
        for n in stations:
            t = self._frame("annual", n).drop(columns=[FETCHED])
            self.note_read("annual", n, str(t["年"].min()), str(t["年"].max()))
            parts.append(t)
        return pd.concat(parts, ignore_index=True)

    # 使った観測値の記録 ------------------------------------------------------------
    def fingerprint(self, kind: str, station: str, first: str, last: str) -> dict:
        """地点・種類の first〜last の行の要約。指紋は値と品質（取得日を除く）から作るので、同じ値を取り直しても変わらない。
        準正常値（`)`）・資料不足値（`]`）の数と、暫定の行（観測日から refresh_months か月の間に取得した行）の数も数える。
        観測史上順位（rank）は表全体（期間は空）。値がなければ FileNotFoundError。"""
        summary = {"種類": kind, "地点": station, "期間": [first, last]}
        if kind == "rank":
            record = self._rank_file(station)
            body = json.dumps({k: v for k, v in record.items() if k != FETCHED}, ensure_ascii=False, sort_keys=True)
            return summary | {"行数": 1, "指紋": hashlib.sha256(body.encode("utf-8")).hexdigest()[:16],
                              "最後の取得日": str(record[FETCHED]), "暫定の行": 0, "準正常値": 0, "資料不足値": 0}
        k = KINDS[kind]
        t = self._frame(kind, station)
        t = t[(t[k.key[0]].astype(str) >= first) & (t[k.key[0]].astype(str) <= last)].sort_values(k.key)
        cols = [c for c in t.columns if c not in ("地点", "block_no", FETCHED)]
        body = t[cols].to_csv(index=False, lineterminator="\n")
        if kind == "annual":                    # 年ごとの値は、値の文字列の末尾に記号がある（「70 ]」）
            text = t[[c for c in cols if c.endswith("/(mm)")]].astype(str)
            quasi = int(text.apply(lambda s: s.str.contains(")", regex=False)).to_numpy().sum())
            insufficient = int(text.apply(lambda s: s.str.contains("]", regex=False)).to_numpy().sum())
        else:                                   # 品質の列（年最大値系列は 10分_品質・1時間_品質）
            q = t[[k.quality] if k.quality else [c for c in cols if c.endswith("_品質")]]
            quasi, insufficient = int((q == "quasi").to_numpy().sum()), int((q == "insufficient").to_numpy().sum())
        return summary | {"行数": len(t), "指紋": hashlib.sha256(body.encode("utf-8")).hexdigest()[:16],
                          "最後の取得日": str(t[FETCHED].max()) if len(t) else None, "暫定の行": int((~self.final(kind, t)).sum()),
                          "準正常値": quasi, "資料不足値": insufficient}

    def used(self) -> list[dict]:
        """この JmaStore で読んだ観測値（種類・地点・期間ごと）の要約（fingerprint）。種類・地点・期間の順。"""
        return [self.fingerprint(k, s, a, b) for (k, s), spans in sorted(self.reads.items()) for a, b in spans]

    def needed_days(self, kind: str, station: dict, days: list[datetime.date]) -> list[datetime.date]:
        """days のうち取得が要る日: 値がそろっていない日（日別値は1行、時間値は24行、10分値は144行）と、暫定の行がある日。"""
        self.check_station(station["地点"], str(station["block_no"]))     # 別の観測所の値で判定しない
        try:
            t = self._frame(kind, station["地点"])
        except FileNotFoundError:
            return list(days)
        g = t.assign(_final=self.final(kind, t)).groupby("日付").agg(n=("日付", "size"), final=("_final", "all"))
        ready = set(g[(g["n"] == KINDS[kind].per_day) & g["final"]].index)
        return [d for d in days if d.isoformat() not in ready]

    # 書く ----------------------------------------------------------------
    def put(self, kind: str, station: dict, rows: pd.DataFrame, fetched: datetime.date, source: str,
            accept: bool = False) -> PutResult:
        """取得した行を地点のファイルに書く。新しい行は足す。同じ値の行は取得日を新しくする（取り直して確かめた）。
        違う値の行は、前の行が暫定なら上書きして revisions.tsv に記録し、確定ならその行は書かずに conflicts に入れる
        （accept なら上書きして記録）。ほかの行は書く。station は {"地点", "block_no", "pref"}。rows が空ならエラー。"""
        if rows.empty:
            raise ValueError(f"{station['地点']} の{KINDS[kind].name}に、書く行がありません（{source}）。")
        k = KINDS[kind]
        name, block = station["地点"], str(station["block_no"])
        folder = self._ensure_station(station)
        rows = rows.assign(地点=name, block_no=block)
        if k.value and k.quality:
            rows = rain_values(rows, k.value, k.quality, source)
        cols = k.columns or list(rows.columns)
        compare = [c for c in cols if c not in ("地点", "block_no", *k.key)]
        result = PutResult()
        groups = rows.groupby(rows["日付"].str[:4]) if "{year}" in k.path else [(None, rows)]
        for year, part in groups:
            path = folder / k.path.format(year=year)
            old = self._read(path, kind) if path.exists() else pd.DataFrame(columns=[*cols, FETCHED])
            merged, revisions = self._merge(kind, old, part[cols], compare, fetched, accept, result, station, source)
            _write_bytes(path, table_bytes(merged.sort_values(k.key).reset_index(drop=True), path))
            if revisions:
                self._append("revisions.tsv", revisions, REVISION_COLUMNS)
        period = f"{rows[k.key[0]].min()}〜{rows[k.key[0]].max()}"
        self._append("fetch_log.tsv", [[fetched.isoformat(), name, block, kind, period, result.added, result.same, result.replaced, source]],
                     LOG_COLUMNS)
        return result

    def _merge(self, kind, old, new, compare, fetched, accept, result, station, source):
        """old（保存済み）に new（取得した行）を合わせた表と、訂正の記録の行。確定の値と違う行は old のまま残す。"""
        key = KINDS[kind].key
        day = fetched.isoformat()
        o, n = old.set_index(key), new.assign(**{FETCHED: day}).set_index(key)
        if n.index.duplicated().any():
            raise ValueError(f"{station['地点']} の取得した{KINDS[kind].name}に、同じ日時の行が2つあります（{n.index[n.index.duplicated()][0]}）。")
        common = n.index.intersection(o.index)
        same = np.ones(len(common), dtype=bool)
        for c in compare:
            same &= _equal(o.loc[common, c], n.loc[common, c]).to_numpy()
        result.added += len(n.index.difference(o.index))
        result.same += int(same.sum())
        later = common[same]
        later = later[o.loc[later, FETCHED].astype(str).to_numpy() < day]
        o.loc[later, FETCHED] = day
        changed = common[~same]
        final = self.final(kind, o.loc[changed].reset_index()).to_numpy() if len(changed) else np.array([], dtype=bool)
        revisions, replace = [], []
        for k, is_final in zip(changed, final):
            prev, r = o.loc[k], n.loc[k]
            when = " ".join(map(str, k if isinstance(k, tuple) else (k,)))
            diff = [c for c in compare if not _equal(pd.Series([prev[c]]), pd.Series([r[c]])).iat[0]]
            before = "、".join(f"{c} {prev[c]}" for c in diff)
            after = "、".join(f"{c} {r[c]}" for c in diff)
            if is_final and not accept:
                result.conflicts.append(f"{station['地点']} {KINDS[kind].name} {when}: {before}（取得日 {prev[FETCHED]}、確定）→ {after}")
                continue
            revisions.append([day, station["地点"], str(station["block_no"]), kind, when, before, after, prev[FETCHED], source])
            replace.append(k)
        result.replaced += len(replace)
        o.loc[replace, n.columns] = n.loc[replace]
        merged = pd.concat([o, n.loc[n.index.difference(o.index)]])
        return merged.reset_index(), revisions

    def _append(self, name: str, rows: list[list], columns: list[str]) -> None:
        """_jma/ の直下の記録（fetch_log.tsv・revisions.tsv）に行を足す。"""
        path = self.root / name
        df = pd.DataFrame(rows, columns=columns).astype(str)
        old = pd.read_csv(path, sep="\t", encoding="utf-8-sig", dtype=str) if path.exists() else None
        _write_bytes(path, table_bytes(df if old is None else pd.concat([old, df], ignore_index=True), path))


def _equal(a: pd.Series, b: pd.Series) -> pd.Series:
    """値が同じか（空欄どうしは同じ。数に読める値は小数6桁で比べ、ほかは文字列で比べる）。"""
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    na, nb = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    both_num = na.notna() & nb.notna()
    eq = (a.isna() & b.isna()) | (both_num & (na.round(6) == nb.round(6))) | (~both_num & a.notna() & b.notna() & (a.astype(str) == b.astype(str)))
    return pd.Series(eq.to_numpy(), index=range(len(eq)))


def table_bytes(df: pd.DataFrame, path: Path) -> bytes:
    """jma-rainfall の出力と同じ形（UTF-8 BOM 付き、改行 CRLF。.tsv はタブ区切り）のバイト。"""
    return df.to_csv(sep="\t" if Path(path).suffix == ".tsv" else ",", index=False, lineterminator="\r\n").encode("utf-8-sig")
