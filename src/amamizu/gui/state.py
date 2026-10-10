"""設計の検討の画面の状態（NiceGUI の部品には依存しない。単体で試験できる）。

- 編集中の study.json の内容（conf）と、池の表（編集できる）・ハイドログラフと実測降雨の表（見るだけ）を持つ
- WebGUI の数値の欄は小数（10.0）を返すので、整数の項目は小数部が0なら整数に直す（normalized）
- 確かめは app/design_runner.check_sections と adapters/study_config をそのまま使う（WebGUI で同じ確かめを書き直さない）
- 保存は、study.json を dump_study（SCHEMA の順、CRLF）で書き、池の表を input/ に書く。
  既存の CSV を上書きして失われるもの（ほかの列、行）があれば、保存の前に示す
"""
import copy
import datetime
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..adapters.design_inputs import pond_columns, pond_rows_of, read_table, write_table
from ..app.observations import observation_changes
from ..adapters.guideline.json_guideline import NEW_STUDY, JsonGuideline, guideline_names, relocated
from ..domain.pond import DEFAULT_TC_MIN
from ..adapters.study_config import (INT_PATHS, check_kind, dump_study, read_study, remove_keys, unknown_keys,
                                     value_errors)
from ..app.design_runner import SectionCheck, check_sections, output_conditions, read_output_conditions

# 整数の項目 → WebGUI のタブ

# 手引を使用しない検討の、放流の係数の入力欄の初期値（一般的な値）。計算には study.json に書いた値を使う
GENERAL_HYDRAULICS = {"g": 9.8, "c_weir": 1.8, "c_orifice": 0.6}
# 強度式の係数を直接指定するときの入力欄の初期値（千葉県の手引の千葉地区、50年確率の式）
SAMPLE_FORMULA = {"type": "cleveland", "a": 2439, "b": 11.9, "n": "3/4"}
SAMPLE_FORMULA_NOTE = "初期値は千葉県の手引の千葉地区・50年確率の式です。使用する基準の値を確認して入力してください。"
# 新しい設計の検討の初めの条件（そのまま計算できるもの）。手引の定数と計画降雨は、作るときに決める（create_design_study）
NEW_DESIGN = {
    "kind": "design", "title": "", "tc_min": DEFAULT_TC_MIN,
    "catchments": [{"name": "区域", "area_ha": 1.0, "f": 0.9}],
    "pond": {"prismatic": {"area_m2": 600.0, "hwl_m": 2.0}},
    "orifice": {"sill_m": 0.0, "width_m": 0.08, "height_m": 0.08},
}

INVALID_NAME = re.compile(r'[\\/:*?"<>|]')


# 変えた条件の表示に使う、study.json の項目の名前（WebGUI の入力欄の名前にそろえる）。値は名前か（名前, 中の項目）。
# 並びの項目は、名前に番号を付ける（orifices[1] → オリフィス2）。SCHEMA の項目をすべて持つことを試験で確かめる
_INFILTRATION_NAMES = {
    "rate_m3s": "設計浸透量", "intensity_mmh": "設計浸透強度", "void_storage_m3": "空隙貯留量",
    "facilities": ("施設", {
        "name": "名前", "specific_m2": "比浸透量 K", "count": "数量",
        "factors": ("影響係数", {"clogging": "目づまり", "groundwater": "地下水位", "safety": "安全率 α"}),
        "shape": ("施設の形", {"kind": "施設の種類", "H_m": "設計水頭 H", "W_m": "幅 W", "D_m": "直径 D", "L_m": "延長 L"}),
        "permeability": ("飽和透水係数", {"value": "値", "unit": "単位"}),
        "void_parts": ("空隙", {"volume_m3": "体積", "porosity_pct": "空隙率"}),
    }),
}
_OUTLET_NAMES = {"sill_m": "敷高", "shape": "形", "width_m": "幅", "height_m": "高さ", "diameter_m": "直径"}
CONDITION_NAMES = {
    "kind": "検討の種類", "title": "題名", "guideline": "手引の定数", "created": "作成日時",
    "hydraulics": ("放流の係数", {"g": "重力加速度 g", "c_weir": "越流係数", "c_orifice": "オリフィスの流量係数"}),
    "tc_min": "洪水到達時間 tc", "end_min": "追跡の終了時刻", "mode": "計算の方法",
    "target_q_m3s": "目標の最大放流量", "allowable_q_m3s": "許容放流量",
    "rain": ("計画降雨", {
        "guideline": ("手引の式", {"district": "地区", "return_period": "確率年"}),
        "formula": ("強度式", {"type": "強度式の種類", "a": "a", "b": "b", "n": "n"}),
        "shape": "降雨波形", "hours": "継続時間", "round_digits": "丸め",
    }),
    "catchments": ("区域", {"name": "名前", "area_ha": "面積", "f": "流出係数", "infiltration": ("浸透", _INFILTRATION_NAMES)}),
    "inflow_csv": ("ハイドログラフ", {"file": "CSV", "time_col": "時刻の列", "q_col": "流入量の列", "rain_col": "降雨強度の列",
                                  "reverse": "値の並びを逆にする"}),
    "observed_rain": ("実測降雨", {"station": "地点", "series": "値の種類", "start": "開始日", "end": "終了日"}),
    "pond": ("池", {
        "prismatic": ("柱状の池", {"area_m2": "面積", "hwl_m": "満水位"}),
        "table": ("池の表", {"file": "保存先の CSV", "level_col": "水位の列", "area_col": "面積の列", "volume_col": "容量の列",
                           "hwl_m": "満水位"}),
    }),
    "orifice": ("オリフィス", _OUTLET_NAMES),
    "orifices": ("オリフィス", _OUTLET_NAMES),
    "spillway": ("洪水吐", {"crest_m": "敷高", "width_m": "幅"}),
    "pump": ("ポンプ", {"levels_m": "水位", "q_m3s": "排水量"}),
    "judge": ("判定", {"storage_margin_m3": "容量の余裕"}),
    "infiltration": ("浸透（区域全体）", _INFILTRATION_NAMES),
}
# 最上位の項目 → WebGUI のタブ
CONDITION_TABS = {
    "title": "全体", "guideline": "全体", "hydraulics": "全体", "tc_min": "全体", "rain": "流入", "catchments": "流入", "inflow_csv": "流入",
    "observed_rain": "流入", "pond": "池", "orifice": "放流施設", "orifices": "放流施設", "spillway": "放流施設",
    "pump": "放流施設", "infiltration": "浸透", "mode": "判定と実行", "end_min": "判定と実行", "target_q_m3s": "判定と実行",
    "allowable_q_m3s": "判定と実行", "judge": "判定と実行",
}


def condition_name(path: str) -> str:
    """study.json の項目の場所（orifices[0].width_m）を、利用者が読める名前にする（「放流施設 オリフィス1 幅（orifices[0].width_m）」）。
    識別子は、study.json を直接見る人のために括弧で添える。名前の表にない項目は、識別子だけにする。"""
    tokens = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    node: dict | None = CONDITION_NAMES
    parts: list[str] = []
    for tok in tokens:
        if tok.startswith("["):
            if not parts:
                return path
            parts[-1] += str(int(tok[1:-1]) + 1)
            continue
        entry = (node or {}).get(tok)
        if entry is None:
            return path
        label, node = (entry, None) if isinstance(entry, str) else entry
        parts.append(label)
    if not parts:
        return path
    tab = CONDITION_TABS.get(tokens[0])
    return f"{tab + ' ' if tab else ''}{' '.join(parts)}（{path}）"


def new_study_dir(studies: Path, title: str, today: datetime.date | None = None) -> Path:
    """新しい検討のフォルダ studies/<年-月>_<題名>。題名の空白は _ にする（コマンドでパスを扱いやすく）。
    題名が空・使えない文字・同じ名前のフォルダはエラー。"""
    title = title.strip()
    if not title:
        raise ValueError("題名を入力してください。")
    if INVALID_NAME.search(title) or title.endswith("."):
        raise ValueError('題名にフォルダ名に使えない文字（\\ / : * ? " < > |、末尾の .）があります。')
    today = today or datetime.date.today()
    d = Path(studies) / f"{today:%Y-%m}_{'_'.join(title.split())}"
    if d.exists():
        raise ValueError(f"同じ名前のフォルダがあります: {d}。別の題名にしてください。")
    return d


def create_design_study(studies: Path, title: str) -> Path:
    """新しい設計の検討を作り、study.json を書く。"""
    d = new_study_dir(studies, title)
    d.mkdir(parents=True)
    conf = copy.deepcopy(NEW_DESIGN) | {"title": title.strip(),
                                        "created": datetime.datetime.now().isoformat(timespec="seconds")}
    if NEW_STUDY in guideline_names(studies):
        district = JsonGuideline.load(NEW_STUDY, studies, studies).default_district()
        conf |= {"guideline": NEW_STUDY, "rain": {"guideline": {"district": district, "return_period": 50}, "shape": "rear"}}
    else:                                        # 手引の定数のファイルがない検討のフォルダ。手引を使用しない検討にする
        conf |= {"guideline": None, "hydraulics": dict(GENERAL_HYDRAULICS),
                 "rain": {"formula": dict(SAMPLE_FORMULA), "shape": "rear"}}
    (d / "study.json").write_bytes(dump_study(conf))
    return d


def station_district(spec: dict, g: JsonGuideline) -> str | None:
    """実測降雨の地点（spec の station）が属する手引の地区（計画降雨の地区の初期値。検討の画面と降雨の画面が使う）。
    地点を選んでいないか、手引の地点表にない地点なら None。"""
    station = spec.get("station")
    return g.district_of(station) if station in g.stations() else None


GUIDELINE_KINDS = ("return_period", "storage")   # 検証の例（降雨強度の検証、貯留施設の検証。報告の順）。一覧の先頭に固定する


def _created(conf: dict | None) -> tuple[float | None, str | None]:
    """study.json の created の（日時（秒）, 読めないときの文）。created がなければ（None, None）。"""
    if not conf or not conf.get("created"):
        return None, None
    try:
        return datetime.datetime.fromisoformat(conf["created"]).timestamp(), None
    except (TypeError, ValueError):
        return None, (f"study.json の created「{conf['created']}」を日時として読めないので、作成日時にはフォルダの作成日時を使っています。"
                      "created を 2026-09-01T10:00:00 の形に直してください。")


def created_error(conf: dict | None) -> str | None:
    """study.json の created を日時として読めないときに一覧に示す文（読めるか、created がなければ None）。"""
    return _created(conf)[1]


def created_at(d: Path, conf: dict | None) -> float:
    """検討を作った日時（秒）。study.json の created、なければ（日時として読めなければ）フォルダの作成日時。
    読めない created は、一覧を表示できなくせずに created_error の文で知らせる。"""
    at = _created(conf)[0]
    if at is not None:
        return at
    st = Path(d).stat()
    return getattr(st, "st_birthtime", st.st_ctime)


def list_order(d: Path, conf: dict | None) -> tuple:
    """一覧の並び。検証の例（降雨強度の検証、貯留施設の検証）を報告の順で先頭に置き、利用者が作った検討は作成日の新しい順に並べる。
    study.json を読めない検討は最後に置く。"""
    kind = (conf or {}).get("kind")
    if kind in GUIDELINE_KINDS:
        return (0, GUIDELINE_KINDS.index(kind), 0.0, d.name)
    if conf is not None:
        return (1, 0, -created_at(d, conf), d.name)
    return (2, 0, 0.0, d.name)


@dataclass
class DeletePlan:
    """削除する検討と、確認のために示す中身。"""
    path: Path
    title: str
    files: int
    has_out: bool
    entries: list          # フォルダの直下にあるもの（フォルダは末尾に /）


def design_study_dir(studies: Path, name: str, action: str) -> tuple[Path, dict]:
    """WebGUI から削除・複製してよい検討のフォルダと study.json の内容。扱えるのは studies/ の直下の設計の検討
    （kind: design）だけ（貯留施設の検証・降雨強度の検証は、WebGUI では結果を見るだけ）。"""
    studies = Path(studies).resolve()
    d = (studies / name).resolve()
    if d.parent != studies or not (d / "study.json").is_file():
        raise ValueError(f"検討のフォルダではありません（study.json がありません）: {d}。一覧から検討を選び直してください。")
    conf = read_study(d)
    if conf.get("kind") != "design":
        raise ValueError(f"WebGUI から{action}できるのは設計の検討だけです。")
    return d, conf


def delete_plan(studies: Path, name: str) -> DeletePlan:
    """WebGUI から削除してよい検討か確かめ（design_study_dir）、確認のために示す中身を集める。"""
    d, conf = design_study_dir(studies, name, "削除")
    files = sum(1 for p in d.rglob("*") if p.is_file())
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in d.iterdir())
    return DeletePlan(d, conf.get("title") or d.name, files, (d / "out").is_dir(), entries)


def duplicate_study(studies: Path, name: str, title: str, conf: dict | None = None,
                    tables: "list[TableChange] | None" = None) -> Path:
    """検討を複製して新しい検討を作る（条件を変えた別のケースの出力を、別のフォルダに残すため）。
    study.json（題名と作成日時は新しいもの）と input/ をコピーし、out/ はコピーしない（新しい検討はまだ出力していない）。
    conf と tables を与えると、保存した条件の代わりにそれを書く（WebGUI で編集中の条件を複製に含めるとき。DesignForm.duplicate）。
    実測降雨は studies/_jma/ の値を地点名で指すので、複製した検討でもそのまま使える。"""
    src, saved = design_study_dir(studies, name, "複製")
    d = new_study_dir(studies, title)
    targets = []
    for ch in tables or []:
        rel = ch.path.resolve().relative_to(src) if ch.path.resolve().is_relative_to(src) else None
        if rel is None:
            raise ValueError(f"池の表の保存先 {ch.path} が検討のフォルダの外にあります。池の表の保存先を input/ にしてから複製してください。")
        targets.append((ch.frame, d / rel))
    base = saved if conf is None else conf
    # .json の相対パスは複製から見たパスに直す。input/ の中のファイルはコピーするので、コピーしたほうを指す
    moved = {"guideline": relocated(base["guideline"], src, d, ("input",))} if "guideline" in base else {}
    data = dump_study(base | moved | {
        "title": title.strip(), "created": datetime.datetime.now().isoformat(timespec="seconds")})
    d.mkdir(parents=True)
    try:
        if (src / "input").is_dir():
            shutil.copytree(src / "input", d / "input")
        for frame, path in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_table(frame, path)
        (d / "study.json").write_bytes(data)
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)       # 途中まで作ったフォルダを残さない
        raise
    return d


def delete_study(plan: DeletePlan, trash=None, permanent: bool = False) -> None:
    """検討のフォルダを削除する。ふだんは OS のごみ箱へ移す（ごみ箱から元に戻せる）。

    send2trash が OS ごとの仕組みを使う: Windows はごみ箱、macOS はゴミ箱、
    Linux などは freedesktop.org の仕様のゴミ箱（~/.local/share/Trash など）。
    ごみ箱が使えない環境（ごみ箱のフォルダを作れない、別のドライブで移せないなど）では OSError になる。
    黙って完全な削除に切り替えず、WebGUI で確かめてから permanent=True で完全に削除する。
    trash は試験で差し替えるための引数。
    """
    if permanent:
        shutil.rmtree(plan.path)
        return
    if trash is None:
        from send2trash import send2trash as trash
    trash(str(plan.path))


def normalized(conf: dict) -> dict:
    """整数の項目で小数部が0の小数（10.0）を整数にした写し。小数部があるもの（10.5）はそのまま（確かめで誤りになる）。"""
    out = copy.deepcopy(conf)
    if out.get("kind") not in INT_PATHS:
        return out
    for path in INT_PATHS[out["kind"]]:
        keys = path.split(".")
        parent = out
        for k in keys[:-1]:
            parent = parent.get(k) if isinstance(parent, dict) else None
        if isinstance(parent, dict) and keys[-1] in parent:
            v = parent[keys[-1]]
            if isinstance(v, float) and v.is_integer():
                parent[keys[-1]] = int(v)
    return out


@dataclass
class TableChange:
    path: Path
    frame: pd.DataFrame
    lost: list = field(default_factory=list)     # 上書きで失われるもの（空なら確かめずに書いてよい）


@dataclass
class DesignForm:
    study_dir: Path
    conf: dict
    pond_rows: list | None = None          # 池の表（行の dict の並び。列名は study.json の指定どおり）
    hydrograph: pd.DataFrame | None = None  # ハイドログラフの表（見るだけ）
    unknown: list = field(default_factory=list)     # 読み込んだ study.json の不明な項目
    saved: dict | None = None              # 最後に保存（または読み込み）した状態の写し
    output: dict | None = None             # out/ の結果を作った条件（conditions.json。まだ出力していなければ None）
    observation_changes: list = field(default_factory=list)   # 出力の後に変わった、使った観測値（studies/_jma/）

    # 読み込み --------------------------------------------------------------
    @classmethod
    def load(cls, study_dir: Path) -> "DesignForm":
        study_dir = Path(study_dir)
        conf = read_study(study_dir)
        check_kind(conf)
        f = cls(study_dir, conf, unknown=unknown_keys(conf))
        f.reload_tables()
        f.saved = f.snapshot()
        f.reload_output()
        return f

    def reload_output(self):
        """out/ の結果を作った条件を読み直す（読み込みと出力の後だけ。確かめのたびにファイルを読まない）。"""
        self.output = read_output_conditions(self.study_dir)
        self.observation_changes = observation_changes(self.output, self.studies)

    def reload_tables(self):
        """study.json が参照する input/ の表を読み直す（読めなければ空のまま。誤りは確かめで出る）。"""
        spec = (self.conf.get("pond") or {}).get("table")
        self.pond_rows = None
        if spec and spec.get("file"):
            try:
                df = read_table(self.study_dir, spec["file"])
                self.pond_rows = pond_rows_of(df, spec)
            except (FileNotFoundError, ValueError, KeyError):
                self.pond_rows = None
        self.load_hydrograph()

    @property
    def studies(self) -> Path:
        return self.study_dir.resolve().parent

    @property
    def no_guideline(self) -> bool:
        """手引を使用しない検討（study.json の guideline が null）。"""
        return "guideline" in self.conf and self.conf["guideline"] is None

    def guideline(self) -> JsonGuideline | None:
        """検討で選んだ手引の定数。手引を使用しないか、読めなければ None（読めない誤りは WebGUI で示す）。"""
        try:
            return JsonGuideline.of_study(self.conf, self.study_dir)
        except (ValueError, OSError, KeyError):
            return None

    def set_guideline(self, name: str | None) -> str | None:
        """手引の定数を選び直す。None は「使用しない」で、放流の係数の入力欄に一般的な値を入れ、手引の地区と確率年の
        計画降雨は、同じ式の係数の直接指定に直す（選んでいた手引を使用しない検討では地区を選べない）。
        手引の定数を選んだら、study.json の放流の係数を削除する（係数は手引の定数のファイルの値）。
        戻り値は利用者に知らせる文（計画降雨の式を引き継げず、強度式の係数を初期値にしたとき。なければ None）。"""
        rain, notice = self.conf.get("rain"), None
        if name is None:
            if isinstance(rain, dict) and "guideline" in rain:
                gl = rain.pop("guideline")
                try:
                    g = JsonGuideline.of_study(self.conf, self.study_dir)
                    if g is None:
                        raise ValueError("手引の定数を選んでいません。")
                    rain["formula"] = g.formula_spec(gl["district"], gl["return_period"])
                except (KeyError, TypeError, ValueError, OSError) as e:
                    rain["formula"] = dict(SAMPLE_FORMULA)
                    notice = (f"選んでいた計画降雨（{gl.get('district')}、確率年 {gl.get('return_period')}）の式を読めなかったので、"
                              f"強度式の係数を初期値にしました（{e.args[0] if e.args else e}）。「流入」のタブで係数を入力してください。")
            self.conf.setdefault("hydraulics", dict(GENERAL_HYDRAULICS))
        else:
            self.conf.pop("hydraulics", None)
        self.conf["guideline"] = name
        return notice

    def default_district(self) -> str | None:
        """計画降雨の地区の初期値（検討で選んだ手引の default_district）。手引がなければ None。"""
        g = self.guideline()
        return g.default_district() if g else None

    def district_term(self) -> str:
        """区域の呼び方（検討で選んだ手引の district_term）。手引がなければ「地区」。"""
        g = self.guideline()
        return g.district_term() if g else "地区"

    def observed_district(self) -> str | None:
        """実測降雨の地点が属する手引の地区（計画降雨へ切り替えるときの初期値）。地点を選んでいないか、手引の地点表にない地点なら None。"""
        g = self.guideline()
        return station_district(self.conf.get("observed_rain") or {}, g) if g else None

    def load_hydrograph(self):
        spec = self.conf.get("inflow_csv")
        self.hydrograph = None
        if spec and spec.get("file"):
            try:
                self.hydrograph = read_table(self.study_dir, spec["file"])
            except (FileNotFoundError, ValueError):
                self.hydrograph = None

    # 状態 ------------------------------------------------------------------
    def normalized(self) -> dict:
        return normalized(self.conf)

    def rain_query(self) -> dict | None:
        """降雨のページに渡す条件（URL のクエリ）。流入がハイドログラフなら None。
        数値の欄は小数（20.0）を返すので、整数の項目を整数に直した条件（normalized）から作る。"""
        c = self.normalized()
        tc = {"tc": c.get("tc_min", DEFAULT_TC_MIN)} | ({"basis": c["guideline"]} if isinstance(c.get("guideline"), str) else {})
        if "rain" in c:
            rain = c["rain"]
            return ({"source": "planned"} | (rain.get("guideline") or {}) | (rain.get("formula") or {})
                    | {k: rain[k] for k in ("shape", "hours") if rain.get(k) is not None}
                    | ({"round": rain["round_digits"]} if rain.get("round_digits") is not None else {}) | tc)
        if "observed_rain" in c:
            spec = c["observed_rain"]
            return {"source": "observed"} | {k: spec[k] for k in ("station", "series", "start", "end") if spec.get(k)} | tc
        return None

    def pond_frame(self) -> pd.DataFrame | None:
        if self.pond_rows is None:
            return None
        spec = self.conf["pond"]["table"]
        return pd.DataFrame(self.pond_rows, columns=pond_columns(spec))

    def tables(self) -> dict:
        """計算に渡す表（WebGUI で編集中の池の表）。"""
        out = {}
        if (self.conf.get("pond") or {}).get("table") is not None and self.pond_rows is not None:
            out["pond"] = self.pond_frame()
        return out

    def snapshot(self) -> dict:
        """比べるための写し（条件と、池の表の中身）。"""
        return {"conf": self.normalized(), "pond": copy.deepcopy(self.pond_rows)}

    @property
    def dirty(self) -> bool:
        return self.saved is None or self.snapshot() != self.saved

    @property
    def can_save(self) -> bool:
        """「保存」できるか（study.json として読み直せる条件か。書き方の違う値がない）。計算の誤りは問わない。"""
        return not value_errors(self.normalized())

    def output_changes(self) -> list[str] | None:
        """out/ の結果を作った条件から、今の条件（保存していない変更を含む）で変わったもの。変わっていなければ空。
        まだ出力していなければ None。"""
        if self.output is None:
            return None
        if (self.conf.get("pond") or {}).get("table") is not None and self.pond_rows is None:
            # 池の表を読めていない（誤りは「池」のタブに表示される）。今の条件には池の表がないものとして比べる
            return differences(self.output, {"conf": self.normalized(), "pond": None})
        return differences(self.output, output_conditions(self.normalized(), self.study_dir, self.tables()))

    def check(self) -> SectionCheck:
        """まとまりごとの確認（計算はしない）。型・形の違う値（整数の項目の 10.5、数の項目の "1.0" や true、
        空隙の欄の読めない入力など）は、項目の名前を付けてその項目のタブに表示する（区域の浸透は「浸透」のタブ）。型の違う値で組み立てに失敗した文（Python の文）より先に示す。"""
        conf = self.normalized()
        chk = check_sections(conf, self.study_dir, self.tables())
        typed: dict[str, str] = {}
        for path, text in value_errors(conf):
            s = "浸透" if ".infiltration" in path else CONDITION_TABS.get(re.split(r"[.\[]", path)[0], "全体")
            typed.setdefault(s, f"値を直してください: {condition_name(path)}、{text.removeprefix(path)[1:-1]}。")
        chk.errors.update(typed)
        return chk

    # 保存 ------------------------------------------------------------------
    def table_changes(self) -> list[TableChange]:
        """保存で書く池の表。既存の CSV を上書きして失われるもの（ほかの列、書式の違い）を lost に入れる。"""
        spec = (self.conf.get("pond") or {}).get("table")
        if spec is None or self.pond_rows is None:
            return []
        path = self.study_dir / spec["file"]
        frame = self.pond_frame()
        lost = []
        if path.exists():
            old = read_table(self.study_dir, spec["file"])
            try:
                same = old.reindex(columns=frame.columns).astype(float).equals(frame.astype(float))   # 空欄の列も比べる
            except (KeyError, ValueError):
                same = False
            if same:
                return []                                   # 使う列が変わっていなければ書かない（ほかの列も残る）
            lost.append(f"元の表の中身（{len(old)} 行。書き換えると {len(frame)} 行）")
            extra = [c for c in old.columns if c not in frame.columns]
            if extra:
                lost.append(f"計算に使わない列 {', '.join(extra)}")
        return [TableChange(path, frame, lost)]

    def duplicate(self, title: str) -> Path:
        """編集中の条件（保存していない変更を含む）で、この検討を複製する。この検討の study.json と input/ は変えない。
        保存と同じく、study.json として読み直せない条件（不明な項目、書き方の違う値）では複製しない。"""
        if self.unknown:
            raise ValueError(f"study.json に不明な項目があります: {', '.join(self.unknown)}。"
                             "「保存」で削除して保存してから複製してください。")
        conf = self.normalized()
        bad = value_errors(conf)
        if bad:
            raise ValueError(f"値を直す項目があるので複製できません: {', '.join(t for _, t in bad)}。")
        return duplicate_study(self.study_dir.parent, self.study_dir.name, title, conf, self.table_changes())

    def save_tables(self) -> list[Path]:
        """池の表だけを保存先の CSV に書く（study.json は書かない。入力の誤りがあっても書く）。
        上書きで失われるものの確認は、呼ぶ側（WebGUI）が table_changes で行う。"""
        written = []
        for ch in self.table_changes():
            write_table(ch.frame, ch.path)
            written.append(ch.path)
        if self.saved is not None:
            self.saved["pond"] = copy.deepcopy(self.pond_rows)     # 池の表は保存済み（study.json の変更は残る）
        return written

    def save(self, remove_unknown: bool = False) -> list[Path]:
        """study.json と池の表を保存する（出力はしない）。計算の誤り（オリフィスがない、など）があっても保存する
        （入力の途中を残せる。amamizu study はその誤りで止まる）。study.json として読み直せない条件
        （不明な項目、書き方の違う値）は保存しない。不明な項目は、remove_unknown を選べば削除して保存する。
        上書きで失われるものの確認は、呼ぶ側（WebGUI）が table_changes で行う。"""
        conf = self.normalized()
        if self.unknown:
            if not remove_unknown:
                raise ValueError(f"study.json に不明な項目（amamizu が使わない項目）があります: {', '.join(self.unknown)}。"
                                 "保存するには、これらの項目を削除して保存することを選んでください。")
            conf = remove_keys(conf, self.unknown)
        bad = value_errors(conf)
        if bad:
            raise ValueError(f"値を直す項目があるので保存できません: {', '.join(t for _, t in bad)}。")
        written = self.save_tables()
        (self.study_dir / "study.json").write_bytes(dump_study(conf))
        written.append(self.study_dir / "study.json")
        replace_in_place(self.conf, conf)
        self.unknown = []
        self.saved = self.snapshot()
        return written


def replace_in_place(target, new):
    """target（辞書か並び）の中身を new と同じにする。入れ子の辞書・並びは置き換えずに中身を書き換える
    （WebGUI の入力欄は入れ子の辞書を持っているので、置き換えると保存の後の入力が条件に反映されなくなる）。"""
    if isinstance(target, dict):
        for k in [k for k in target if k not in new]:
            del target[k]
        for k, v in new.items():
            if k in target and type(target[k]) is type(v) and isinstance(v, (dict, list)):
                replace_in_place(target[k], v)
            else:
                target[k] = v
    else:
        del target[len(new):]
        for i, v in enumerate(new):
            if i < len(target) and type(target[i]) is type(v) and isinstance(v, (dict, list)):
                replace_in_place(target[i], v)
            elif i < len(target):
                target[i] = v
            else:
                target.append(v)


def changes(before: dict | None, after: dict) -> list[str]:
    """試算の比較に示す、前の試算からの違い（differences）。最初の試算と、違いがないときは、そう書いた1行。"""
    if before is None:
        return ["（最初の試算）"]
    return differences(before, after) or ["（変更なし）"]


def _rows_text(rows: list | None) -> str:
    return "なし" if rows is None else f"{len(rows)} 行"


def differences(before: dict, after: dict) -> list[str]:
    """2つの写し（条件 conf と池の表 pond）の違い（変えた条件）。違いがなければ空。"""
    out: list[str] = []

    def walk(a, b, prefix):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in list(a) + [k for k in b if k not in a]:
                walk(a.get(k), b.get(k), f"{prefix}.{k}".lstrip("."))
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{prefix}[{i}]")
        elif a != b:
            fmt = lambda v: "なし" if v is None else json.dumps(v, ensure_ascii=False)
            out.append(f"{condition_name(prefix)}: {fmt(a)} → {fmt(b)}")
    walk(before["conf"], after["conf"], "")
    if before.get("pond") != after.get("pond"):
        a, b = before.get("pond"), after.get("pond")
        if a is None or b is None or len(a) != len(b):
            out.append(f"池の表: {_rows_text(a)} → {_rows_text(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                for k in list(x) + [k for k in y if k not in x]:       # 空欄にした列（キーがなくなる）も示す
                    if x.get(k) != y.get(k):
                        out.append(f"池の表 {i + 1} 行目 {k}: {x.get(k)} → {y.get(k)}")
    return out
