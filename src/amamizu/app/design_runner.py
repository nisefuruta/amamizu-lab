"""設計の検討（kind: design）の組み立て・計算・出力（cli と gui が使う。組み立ての層）。

- prepare: study.json の内容（と、WebGUI で編集中の表）から計算の入力を作る。誤りは ConfigError（どの項目のまとまりの誤りか）
- compute: 使用ケースで計算する（WebGUI の試算にも使う。ファイルには書かない）
- write_outputs: out/ に結果を書く（route.csv、rain.csv、design_summary.md、グラフ3枚、出力の条件 conditions.json）
- run_study: prepare → compute → write_outputs（cli の amamizu study と、WebGUI の「保存して出力」）

計算とグラフの出力は、同時に2つ行わない（グラフの pyplot がグローバルな状態を持つため）。
"""
import contextlib
import json
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from fractions import Fraction
from numbers import Real
from pathlib import Path

import numpy as np

from ..adapters.design_inputs import (hydrograph_from_frame, inflow_frame, load_hydrograph, load_observed, load_pond_table,
                                      observed_from_frame, pond_from_frame, pond_rows_of, read_table, write_table)
from ..adapters.guideline.json_guideline import JsonGuideline, hydraulics_from
from ..adapters.jma_store import JmaStore
from ..adapters.report import numbers
from ..domain.design_storm import SHAPES
from ..domain.intensity import formula_of
from ..domain.outlet import Hydraulics
from ..domain.pond import DEFAULT_TC_MIN, PrismaticPond
from .observations import observation_record
from ..usecases.design_study import (OVERDRAW_TIMES, Catchment, DesignResult, Inflow, build_infiltration, build_outlets,
                                     check_options, clock, formula_text, infiltration_units, observed_inflow,
                                     overdraw_note, planned_inflow, rain_table, run_design)

INFLOW_SOURCES = ("rain", "inflow_csv", "observed_rain")
SPILL_DIGITS = 1        # 設計の検討の溢水量の桁（要約と WebGUI で同じ）
CONDITIONS_FILE = "conditions.json"     # out/ の結果を作った条件（output_conditions）

LOCK = threading.Lock()

# 項目のまとまり（WebGUI のタブと同じ）
SECTIONS = ("全体", "流入", "池", "放流施設", "浸透", "判定と実行")


class ConfigError(ValueError):
    """study.json（または WebGUI の入力）の誤り。section はどの項目のまとまりの誤りか。"""

    def __init__(self, section: str, message: str):
        super().__init__(message)
        self.section = section


class _In:
    """with _In("流入"): の中で出た誤りを ConfigError にする。KeyError は項目がないことを示す文にする。"""

    def __init__(self, section: str):
        self.section = section

    def __enter__(self):
        return self

    def __exit__(self, typ, err, tb):
        if err is None or isinstance(err, ConfigError):
            return False
        if isinstance(err, KeyError) and not str(err.args[0]).endswith("。"):
            raise ConfigError(self.section, f"必要な項目 {err.args[0]} がありません。study.json（WebGUI では「{self.section}」のタブ）に書いてください。") from err
        if isinstance(err, (ValueError, KeyError, TypeError, FileNotFoundError)):
            raise ConfigError(self.section, str(err.args[0]) if err.args else str(err)) from err
        return False


def one_of(conf: dict, keys: tuple, where: str) -> str:
    given = [k for k in keys if k in conf]
    if len(given) != 1:
        raise ValueError(f"{where}には {' か '.join(keys)} のどれか一つを書いてください（今は {len(given)} 個）。")
    return given[0]


def basis(conf: dict, study_dir: Path) -> tuple[JsonGuideline | None, Hydraulics]:
    """計算の基準（手引の定数, 放流の係数）。study.json の guideline が手引の定数なら、係数はそのファイルの値
    （study.json の hydraulics は書けない）。null（手引を使用しない）なら手引の定数は None で、係数は study.json の hydraulics。"""
    g = JsonGuideline.of_study(conf, study_dir)
    if g is None:
        if "hydraulics" not in conf:
            raise ValueError("guideline が null（手引を使用しない）のときは、放流の係数 hydraulics（g、c_weir、c_orifice）を study.json に書いてください。")
        return None, hydraulics_from(conf["hydraulics"], "study.json")
    if "hydraulics" in conf:
        raise ValueError("手引の定数（guideline）を選んだときは、study.json に hydraulics を書かないでください。放流の係数は手引の定数のファイルの値を使います。"
                         "係数を自分で決めるなら、guideline を null にしてください。")
    return g, g.hydraulics()


def formula_from(rain: dict, g: JsonGuideline | None):
    """rain.guideline（手引の地区と確率年）か rain.formula（種類と係数）から強度式を作る。戻り値は（式, 種類）。
    g は手引の定数（手引を使用しない検討では None で、rain.guideline は使えない）。"""
    how = one_of(rain, ("guideline", "formula"), "rain ")
    if how == "guideline":
        if g is None:
            raise ValueError("guideline が null（手引を使用しない）のときは、rain.guideline（手引の地区と確率年）は使えません。"
                             "rain.formula（強度式の種類と係数）を書いてください。")
        return g.formula(rain["guideline"]["district"], rain["guideline"]["return_period"]), None
    f = rain["formula"]
    if "type" not in f:
        raise ValueError("rain.formula.type（talbot、sherman、kuno_ishiguro、cleveland、kuno_ishiguro_n）を書いてください。")
    n = f.get("n")
    return formula_of(f["type"], f.get("a"), f.get("b"), None if n is None else float(Fraction(str(n)))), f["type"]


def round_note(digits) -> str:
    return "" if digits is None else f"、降雨強度を小数{digits}桁に四捨五入"


def planned_rows(formula, kind, rain: dict, district_term: str = "地区") -> list[tuple[str, str]]:
    """計画降雨の説明の行（見出し, 値）。計画降雨（手引の地区と確率年、降雨波形）と降雨強度式を分ける
    （1つの欄に詰めると途中で折り返されて読みにくいため）。区域の呼び方は手引の district_term。"""
    shape = f"{SHAPES[rain['shape']]}{rain.get('hours', 24)}時間" + round_note(rain.get("round_digits"))
    if "guideline" in rain:
        gl = rain["guideline"]
        shape = f"手引 {gl['district']}{district_term} 1/{gl['return_period']}、{shape}"
    return [("計画降雨", shape), ("降雨強度式", formula_text(formula, kind))]


@dataclass
class Prepared:
    """計算の入力（study.json の内容から作ったもの）。"""
    conf: dict
    study_dir: Path
    title: str
    guideline: JsonGuideline | None             # 手引の定数（手引を使用しない検討では None）
    tc: int
    inflow: Inflow
    inflow_rows: list                          # 流入の説明の行（見出し, 値）。要約と WebGUI のプレビューで使う
    pond: object
    orifices: list
    end_min: int
    kw: dict = field(default_factory=dict)     # run_design のそのほかの引数
    missing: tuple = ()                        # 実測降雨の欠測の時間（始まり, 終わり）の並び。0mm として計算した
    absent: tuple = ()                         # 実測降雨の studies/_jma/ にない時間（取得していない日など）。0mm として計算した
    observations: JmaStore | None = None       # 実測降雨を読んだ studies/_jma/（使った観測値を conditions.json に記録する）


# 計算の段階の誤りのうち、「判定と実行」のまとまりの項目に関するもの（文に含まれる語で見分ける）
_RUN_WORDS = ("終了時刻", "size_orifice", "target_q", "allowable_q", "余裕", "mode", "容量が足りません", "目標")


def _section_of(msg: str) -> str:
    return "判定と実行" if any(w in msg for w in _RUN_WORDS) else "放流施設"


def _message(err: Exception) -> str:
    msg = str(err.args[0]) if err.args else str(err)
    if isinstance(err, KeyError) and not msg.endswith("。"):
        msg = f"必要な項目 {msg} がありません。"
    return msg


@dataclass
class SectionCheck:
    """項目のまとまりごとの確かめの結果。errors はまとまり → 誤りの文（誤りがなければ入らない）、parts は作れた部品。
    causes はまとまり → 誤りの例外（prepare が原因としてつなぎ、amamizu コマンドが理由の書かれていない誤りを経過ごと表示できるようにする）。"""
    errors: dict = field(default_factory=dict)
    parts: dict = field(default_factory=dict)
    causes: dict = field(default_factory=dict)

    def add(self, section: str, message: str, err: Exception):
        """まとまりの最初の誤りを残す。"""
        if section not in self.errors:
            self.errors[section], self.causes[section] = message, err

    @property
    def ok(self) -> bool:
        return not self.errors


def check_sections(conf: dict, study_dir: Path, tables: dict | None = None) -> SectionCheck:
    """項目のまとまりごとに組み立てて確かめる（計算はしない）。あるまとまりの誤りで、ほかのまとまりの確かめを止めない。
    tables に "pond"・"hydrograph" の表（DataFrame）を渡すと、input/ のファイルの代わりに使う
    （WebGUI で編集中・読み込み済みの表）。"""
    tables = tables or {}
    study_dir = Path(study_dir)
    chk = SectionCheck()
    P = chk.parts

    def attempt(section, fn):
        try:
            with _In(section):
                fn()
        except ConfigError as e:
            chk.errors[e.section], chk.causes[e.section] = str(e), e

    def general():
        P["guideline"], P["hy"] = basis(conf, study_dir)
        P["tc"] = conf.get("tc_min", DEFAULT_TC_MIN)
        P["title"] = conf.get("title", study_dir.name)
    attempt("全体", general)
    if "全体" in chk.errors:
        for s in SECTIONS[1:]:
            chk.errors[s] = "「全体」の誤りを先に直してください。"
        return chk
    g, tc = P["guideline"], P["tc"]

    def zone_infiltration(spec, area):
        """区域の浸透。誤りは「浸透」のタブに付け、流入の組み立ては続ける（試算は浸透の誤りで止まる）。"""
        try:
            with _In("浸透"):
                return build_infiltration(spec, area)
        except ConfigError as e:
            chk.add(e.section, str(e), e)
            return None

    def inflow():
        src = one_of(conf, INFLOW_SOURCES, "流入")
        # 区域の面積・流出係数を先に確かめる（誤りは「流入」のタブだけに付け、同じ誤りを「浸透」のタブに重ねない）
        catchments = [Catchment(c["name"], c["area_ha"], c["f"]) for c in conf.get("catchments", [])]
        catchments = [replace(z, infiltration=zone_infiltration(c.get("infiltration"), z.area_ha))
                      for z, c in zip(catchments, conf.get("catchments", []))]
        P["catchments"] = catchments
        if src == "rain":
            rain = conf["rain"]
            formula, kind = formula_from(rain, g)
            if "shape" not in rain:
                raise ValueError("rain.shape（rear 後方集中型、center 中央集中型）を書いてください。")
            P["inflow"] = planned_inflow(formula, catchments, tc, rain["shape"], rain.get("hours", 24),
                                         rain.get("round_digits"))
            P["inflow_rows"] = planned_rows(formula, kind, rain, g.district_term() if g else "地区")
        elif src == "observed_rain":
            spec = conf["observed_rain"]
            store = JmaStore(Path(study_dir).resolve().parent)
            obs = load_observed(store, spec)
            P["inflow"] = observed_inflow(obs, catchments, tc)
            P["missing"], P["absent"] = obs.missing, obs.absent
            P["observations"] = store                # 使った観測値の記録（conditions.json の「観測値」）に使う
            P["inflow_rows"] = [("実測降雨", obs.label), ("観測値", f"studies/_jma/{store.station_dir(spec['station']).name}"),
                                ("時刻0分", f"{obs.start:%Y-%m-%d %H:%M}")]
            P["observed_label"] = obs.label          # 地点・10分値か時間値か・期間（検討の画面の上部に使う降雨を示す）
        else:
            if "catchments" in conf:
                raise ValueError("inflow_csv を使うときは catchments を書かないでください（流入量はハイドログラフのもの）。")
            spec = conf["inflow_csv"]
            P["inflow"] = (hydrograph_from_frame(tables["hydrograph"], spec, tc) if "hydrograph" in tables
                           else load_hydrograph(study_dir, spec, tc))
            P["inflow_rows"] = [("流入ハイドログラフ", spec["file"])]
    attempt("流入", inflow)

    def pond():
        shape = one_of(conf["pond"], ("prismatic", "table"), "pond ")
        if shape == "prismatic":
            p = conf["pond"]["prismatic"]
            P["pond"] = PrismaticPond(p["area_m2"], p["hwl_m"])
        else:
            spec = conf["pond"]["table"]
            P["pond"] = pond_from_frame(tables["pond"], spec) if "pond" in tables else load_pond_table(study_dir, spec)
    attempt("池", pond)

    def outlets():
        which = one_of(conf, ("orifice", "orifices"), "オリフィス")
        P["orifices"] = [conf["orifice"]] if which == "orifice" else conf["orifices"]
        P["outlets"] = build_outlets(P["orifices"], conf.get("spillway"), conf.get("pump"), P["hy"],
                                     with_holes=(conf.get("mode", "route") == "route"))
    attempt("放流施設", outlets)

    def infiltration():
        if "catchments" in conf and "catchments" not in P and "intensity_mmh" in (conf.get("infiltration") or {}):
            raise ValueError("浸透強度から設計浸透量を求めるには区域の面積が要ります。「流入」の区域の誤りを先に直してください。")
        whole = sum(c.area_ha for c in P["catchments"]) if P.get("catchments") else None
        P["infiltration"] = build_infiltration(conf.get("infiltration"), whole)
        if "inflow" in P:                            # 最上位と区域ごとの両方に書いていないか
            infiltration_units(P["inflow"], P["infiltration"])
    if "浸透" not in chk.errors:                    # 区域の浸透の誤りを先に付けてあれば、それを残す
        attempt("浸透", infiltration)

    def run_options():
        # 追跡の終了時刻の既定は、流入の期間の後に18時間（24時間の計画降雨なら 2520分）
        if "inflow" in P:
            P["end_min"] = conf.get("end_min", (len(P["inflow"].nodes) - 2) * tc + 18 * 60)
        if not {"inflow", "pond", "orifices"} <= set(P):
            return                                   # 流入・池・オリフィスの誤りを直してから確かめる
        try:
            check_options(P["pond"], P["orifices"], conf.get("spillway"), P["end_min"], tc, len(P["inflow"].nodes),
                          conf.get("mode", "route"), conf.get("target_q_m3s"), conf.get("allowable_q_m3s"),
                          conf.get("judge", {}).get("storage_margin_m3"), conf.get("pump"))
        except (ValueError, KeyError, TypeError) as e:
            msg = _message(e)
            chk.add(_section_of(msg), msg, e)
    attempt("判定と実行", run_options)
    return chk


def prepare(conf: dict, study_dir: Path, tables: dict | None = None) -> Prepared:
    """study.json の内容から計算の入力を作る。tables は check_sections と同じ。
    誤りがあれば、最初のまとまり（SECTIONS の順）の誤りを ConfigError にする。"""
    chk = check_sections(conf, study_dir, tables)
    for s in SECTIONS:
        if s in chk.errors:
            raise ConfigError(s, chk.errors[s]) from chk.causes.get(s)
    P = chk.parts
    kw = {"spillway": conf.get("spillway"), "mode": conf.get("mode", "route"), "target_q": conf.get("target_q_m3s"),
          "hy": P["hy"], "allowable_q": conf.get("allowable_q_m3s"), "pump": conf.get("pump"),
          "storage_margin": conf.get("judge", {}).get("storage_margin_m3"), "infiltration": P["infiltration"]}
    return Prepared(conf, Path(study_dir), P["title"], P["guideline"], P["tc"], P["inflow"], P["inflow_rows"], P["pond"],
                    P["orifices"], P["end_min"], kw, P.get("missing", ()), P.get("absent", ()), P.get("observations"))


def compute(p: Prepared) -> DesignResult:
    """計算する（ファイルには書かない）。使用ケースの誤りは、関係する項目のまとまりの ConfigError にする。"""
    with LOCK:
        try:
            return run_design(p.inflow, p.pond, p.orifices, p.kw["spillway"], p.end_min, p.kw["mode"],
                              p.kw["target_q"], hy=p.kw["hy"], allowable_q=p.kw["allowable_q"], pump=p.kw["pump"],
                              storage_margin=p.kw["storage_margin"], infiltration=p.kw["infiltration"], with_table=True)
        except (ValueError, KeyError, TypeError) as e:
            msg = _message(e)
            raise ConfigError(_section_of(msg), msg) from e


def infiltration_label(zone: str | None = None, area_ha: float | None = None) -> str:
    """浸透の説明の見出し。区域ごとなら「宅地の浸透施設（2.385 ha）」、区域全体で1つなら「浸透施設（区域全体）」。"""
    return "浸透施設（区域全体）" if zone is None else f"{zone}の浸透施設（{area_ha:g} ha）"


def infiltration_rows(label: str, it) -> list[tuple[str, str]]:
    """浸透1単位の説明の行（見出し, 値）。要約と WebGUI のプレビューで使う。1つの欄に詰めると途中で折り返されて読みにくいので、
    書き方（浸透強度・設計浸透量・浸透施設）、設計浸透量、空隙貯留量（0 でなければ）、浸透施設ごとの内訳を行に分ける。"""
    rate = f"{it.rate_m3s:.6f} m^3/s（{it.rate_m3s * 3600:.3f} m^3/h）"
    if it.intensity_mmh is not None:
        rows = [(label, f"設計浸透強度 {it.intensity_mmh:g} mm/h"), ("　設計浸透量", rate)]
    elif it.facilities:
        rows = [(label, f"{len(it.facilities)} 種類（内訳は下の行）"), ("　設計浸透量", rate)]
    else:
        rows = [(label, f"設計浸透量 {rate}")]
    if it.void_m3 > 0:
        rows.append(("　空隙貯留量", f"{it.void_m3:.2f} m^3"))
    return rows + facility_rows(it)


def facility_rows(it) -> list[tuple[str, str]]:
    """浸透施設ごとの（名前, 比浸透量・飽和透水係数・数量・影響係数と設計浸透量の文）。施設で書いたときだけ。"""
    return [(f"　{f.name}", f"比浸透量 {f.specific_m2:.4f} m^2、飽和透水係数 {f.permeability_m_h:g} m/h、数量 {f.count:g}、"
                            f"影響係数 {' × '.join(f'{label} {x:g}' for label, x in f.factors())} → {f.rate_m3_h:.4f} m^3/h")
            for f in it.facilities]


def summary_lines(p: Prepared, res: DesignResult) -> list[str]:
    """要約（Markdown の行。最後の7行はグラフの画像）。"""
    s = res.summary
    allow = s["許容放流量_m3s"]
    lines = [f"# {p.title}", "",
             "| 項目 | 値 |", "| --- | ---: |",
             f"| 洪水到達時間 tc（計算の刻み） | {p.tc} 分 |",
            ] + [f"| {k} | {v} |" for k, v in p.inflow_rows]
    lines += [f"| {label}（0 mm として計算） | {text} |" for label, text in gap_notes(p.missing, p.absent)]
    for k, (sill, text) in enumerate(s["オリフィス"], 1):
        lines.append(f"| オリフィス{k if len(s['オリフィス']) > 1 else ''} | 敷高 {sill:.3f} m、{text} |")
    if res.spillway:
        lines.append(f"| 洪水吐 | 敷高 {res.spillway.crest:.3f} m、幅 {res.spillway.width:g} m |")
    if res.pump:
        lines.append(f"| ポンプ | {res.pump.describe()} |")
    whole = res.infiltration is not None             # 最上位の浸透（区域の名前では見分けない）
    for t in res.treated:
        label = infiltration_label() if whole else infiltration_label(t.name, t.area_ha)
        lines += [f"| {a} | {b} |" for a, b in infiltration_rows(label, t.infiltration)]
        if not whole:
            lines.append(f"| 　{t.name}の最大流入量 | 浸透前 {t.nodes.max():.4f} m^3/s、浸透後 {t.after.max():.4f} m^3/s |")
    if "浸透処理面積率_%" in s:
        lines.append(f"| 浸透処理面積率 | {s['浸透処理面積率_%']:.1f}% |")
    if res.treated:
        lines.append(f"| 浸透前の最大流入量 | {s['浸透前の最大流入量_m3s']:.4f} m^3/s |")
    if allow is not None:
        lines.append(f"| 許容放流量 | {allow:g} m^3/s |")
    if "池の不足" in s:
        sh = s["池の不足"]
        lines.append(f"| 池の容量の不足 | 入力の池（容量 {sh['入力の容量_m3']:.1f} m^3）では、最大放流量を許容放流量にするオリフィスで溢れる。"
                     f"水深はそのままで面積を {sh['面積の倍率']:.3f} 倍（容量 {sh['必要な容量_m3']:.1f} m^3、不足 {sh['不足量_m3']:.1f} m^3）"
                     "にすると溢れない。オリフィスと以下の値は、面積を広げた池のもの |")
    lines += [f"| 最大流入量 | {s['最大流入量_m3s']:.4f} m^3/s（{at_time(p, s['最大流入量の時刻_min'])}） |",
              f"| 最大放流量 | {s['最大放流量_m3s']:.4f} m^3/s（{at_time(p, s['最大放流量の時刻_min'])}） |",
              f"| 最大水位 | {s['最大水位_m']:.4f} m（{at_time(p, s['最大水位の時刻_min'])}） |",
              f"| 最大貯留量 | {s['最大貯留量_m3']:.1f} m^3 |",
              f"| 容量（満水位 {s['満水位_m']:.3f} m まで） | {s['容量_m3']:.1f} m^3 |",
              f"| 最大貯留率 | {s['最大貯留率']:.3f} |",
              f"| 溢水量 | {spill_text(s['溢水量_m3'])} m^3 |",
              f"| 余裕量（容量 - 最大貯留量） | {s['余裕量_m3']:.1f} m^3（余裕率 {s['余裕率_%']:.1f}%） |"]
    lines += [f"| 判定（{k}） | {v} |" for k, v in s["判定"].items()]
    lines += ["", "![降雨と流入・放流](hydrograph.png)", "", "![水位](water_level.png)", "",
              "![池と放流施設の特性](pond_curves.png)", ""]
    return lines


def routing_table_lines(res: DesignResult) -> list[str]:
    """時刻ごとの追跡の結果の表（解説 資料－4 の表－11 にあたる。清書した PDF に載せる）。列と値は route.csv と同じ
    （列名に単位。値は route.csv と同じ丸め）。"""
    tab = res.table

    def decimals(col) -> int:                     # route.csv の丸めの桁（列の値の小数の桁の最大）で、列の桁をそろえる
        return max((len(np.format_float_positional(float(v), trim="-").partition(".")[2])  # 指数表記にしない
                    for v in tab[col] if np.isfinite(v)), default=0)
    fmt = {c: (lambda v, d=decimals(c): f"{v:.{d}f}") if tab[c].dtype.kind == "f" else str for c in tab.columns}
    lines = ["## 追跡計算の結果（route.csv）", "",
             "| " + " | ".join(tab.columns) + " |", "|" + " ---: |" * len(tab.columns)]
    for row in tab.itertuples(index=False):
        lines.append("| " + " | ".join(fmt[c](v) for c, v in zip(tab.columns, row)) + " |")
    return lines


def at_time(p: Prepared, minutes: int) -> str:
    """時刻の表記。実測降雨のときは日時も書く（1230 分、2013-10-16 08:30）。"""
    if p.inflow.start is None:
        return f"{minutes} 分"
    return f"{minutes} 分、{clock(p.inflow.start, minutes)}"


MISSING_SPANS = 10      # 実測降雨の欠測の時間・取得していない時間を知らせる文に並べる数（ほかは数と合計の時間を示す）


def missing_text(spans, limit: int) -> str:
    """時間の並びの表記（2013-10-15 14:00〜15:00、…）。多ければ先頭 limit 個と、ほかの数と合計の時間。"""
    def one(a, b):
        return f"{a:%Y-%m-%d %H:%M}〜{b:%H:%M}" if b.date() == a.date() else f"{a:%Y-%m-%d %H:%M}〜{b:%Y-%m-%d %H:%M}"
    text = "、".join(one(a, b) for a, b in spans[:limit])
    if len(spans) > limit:
        hours = sum((b - a).total_seconds() for a, b in spans) / 3600
        text += f" ほか {len(spans) - limit} か所（合計 {hours:g} 時間）"
    return text


def gap_notes(missing, absent) -> list[tuple[str, str]]:
    """実測降雨の欠測の時間と、取得していない時間（studies/_jma/ にない日など）の（見出し, 表記）。0 mm として計算したもの。"""
    return [(label, missing_text(spans, MISSING_SPANS)) for label, spans in (("欠測の時間", missing), ("取得していない時間", absent))
            if spans]


def spill_text(v: float) -> str:
    """設計の検討の溢水量 m^3 の表示（小数 SPILL_DIGITS 桁。adapters/report/numbers.spill_text）。"""
    return numbers.spill_text(v, SPILL_DIGITS)


def notices(res: DesignResult, p: Prepared | None = None) -> list[str]:
    """結果を読む人に知らせること（実測降雨の欠測の時間と取得していない時間、溢水、許容放流量の超過、
    前の刻みの放流量が池の水と流入の和を超えた刻み）。"""
    s, out = res.summary, []
    if p is not None:
        out += [f"実測降雨の{label}を 0 mm として計算しました: {text}" for label, text in gap_notes(p.missing, p.absent)]
    if s["溢水量_m3"] > 0:
        out.append(f"満水位を超え、{spill_text(s['溢水量_m3'])} m^3 溢れました。")
    if s["判定"].get("放流量", "").startswith("N.G"):     # 判定と同じ基準で知らせる（判定だけ N.G で知らせない、にしない）
        out.append(f"最大放流量 {s['最大放流量_m3s']:.6f} m^3/s が許容放流量 {s['許容放流量_m3s']:g} m^3/s を超えました。")
    if res.routing.overdraw:
        out.append(overdraw_note(res.routing, res.tc_min, OVERDRAW_TIMES))
    return out


def ensure_dir(path: Path) -> Path:
    """出力先のフォルダを作る。同じ名前のファイルがあれば、その理由を示して ValueError。"""
    path = Path(path)
    if path.exists() and not path.is_dir():
        raise ValueError(f"出力先の {path} がフォルダではなくファイルです。このファイルの名前を変えるか削除してから、もう一度出力してください。")
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def all_or_nothing(files: list[Path]) -> Iterator[None]:
    """out/ への出力を、一式そろって成功するか、何も残さないかのどちらかにする。途中でエラーになれば
    （グラフを出力できないときなど）、この一式のファイルを前の実行の残りも含めてすべて削除してから例外を通す
    （out/ はいつも保存した条件と一致し、途中の結果や古い結果が混ざらない）。"""
    try:
        yield
    except BaseException:
        for f in files:
            f.unlink(missing_ok=True)
        raise


def output_conditions(conf: dict, study_dir: Path, tables: dict | None = None) -> dict:
    """結果を作った条件（study.json の内容と池の表の行）。out/ の conditions.json に書き、WebGUI は今の条件と比べて
    出力の後に変えた条件を示す（どちらもこの関数で作るので、同じ条件なら違いは出ない）。
    池の表は、計算に使う表（tables の "pond"、なければ input/ のファイル）の行。表でない池は None。
    表を読めないときは、その例外を通す（読めない表を「池の表なし」として記録しない）。数でない値はそのまま記録する。"""
    spec = (conf.get("pond") or {}).get("table")
    rows = None
    if spec:
        frame = (tables or {}).get("pond")
        frame = read_table(study_dir, spec["file"]) if frame is None else frame
        rows = [{k: float(v) if isinstance(v, Real) else v for k, v in r.items()} for r in pond_rows_of(frame, spec)]
    return {"conf": conf, "pond": rows}


def read_output_conditions(study_dir: Path) -> dict | None:
    """out/ の結果を作った条件（conditions.json）。まだ出力していない（またはこの記録より前の出力）なら None。
    ファイルがあるのに読めない（JSON として読めない、項目を持つ形でない）ときは ValueError
    （出力していないものとして扱うと、前の出力にだけあるファイルが削除されず、条件と観測値の変化も示されない）。"""
    path = Path(study_dir) / "out" / CONDITIONS_FILE
    try:
        data = path.read_bytes()
    except (FileNotFoundError, NotADirectoryError):      # out/ か conditions.json がない
        return None
    try:
        record = json.loads(data.decode("utf-8"))        # UTF-8 でない文字コードも、JSON の誤りと同じ ValueError
        if not isinstance(record, dict):
            raise ValueError("項目を持つ形（{ で始まる形）ではありません")
    except ValueError as e:
        raise ValueError(f"{path}（前の出力の条件と、出力したファイルの名前の記録）を読めません（{e}）。"
                         "このファイルを削除してから、もう一度出力してください"
                         "（削除すると、前の出力にだけあるファイルは out/ に残ります）。") from None
    return record


def stale_outputs(out: Path, files: Iterable[Path]) -> list[Path]:
    """前の出力が書いたファイル（conditions.json の files）のうち、今回の出力 files にないもの。
    出力を書き終えたら削除する（条件によって出ないファイルに、前の条件の結果を残さない）。
    記録にないファイル（降雨のページで保存した降雨、利用者が保存したファイル）は含めない。"""
    keep = {Path(f).name for f in files}
    names = (read_output_conditions(out.parent) or {}).get("files") or []
    return [out / n for n in names if isinstance(n, str) and Path(n).name == n and n not in keep]


def write_conditions(out: Path, conditions: dict, files: Iterable[Path]) -> None:
    """結果を作った条件と、出力したファイルの名前（files）を out/ の conditions.json に書く。"""
    record = conditions | {"files": [Path(f).name for f in files]}
    text = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (out / CONDITIONS_FILE).write_bytes(text.replace("\n", "\r\n").encode("utf-8"))


def record_outputs(out: Path, conditions: dict, files: list[Path]) -> list[Path]:
    """出力を書き終えた後に、条件とファイルの名前を conditions.json に書き、前の出力にだけあるファイルを削除する。
    戻り値は、conditions.json を足した出力のファイル。"""
    files = [*files, out / CONDITIONS_FILE]
    stale = stale_outputs(out, files)
    write_conditions(out, conditions, files)
    for f in stale:
        f.unlink(missing_ok=True)
    return files


def output_names(with_rain: bool = True) -> list[str]:
    """write_outputs が out/ に書くファイルの名前（その順）。rain.csv は降雨強度のある流入（計画降雨・実測降雨と、降雨強度の列 rain_col のあるハイドログラフ）のときだけ。
    降雨のページの「out/ に保存」は、これと同じ名前のファイルを書かない（rain_runner.saved_rain_path）。"""
    from ..adapters.report import design_plots
    return ["route.csv", "design_summary.md", "inflow.csv", *(["rain.csv"] if with_rain else []),
            *design_plots.GRAPH_NAMES, "design_summary.pdf", CONDITIONS_FILE]


def write_outputs(p: Prepared, res: DesignResult, lines: list[str], conditions: dict) -> list[Path]:
    """out/ に結果を書き、書いたファイルを返す（route.csv、design_summary.md、inflow.csv、rain.csv、グラフ3枚、
    design_summary.pdf、conditions.json の順）。inflow.csv は流出量の計算結果で、ハイドログラフの入力（inflow_csv）にそのまま使える。
    conditions.json は結果を作った条件（output_conditions）と出力したファイルの名前。どれかを書けなければ、一式をすべて削除する
    （all_or_nothing）。前の出力にだけあるファイル（降雨強度の列のないハイドログラフにしたときの rain.csv）は、書き終えたら削除する（stale_outputs）。"""
    from ..adapters.report import design_pdf, design_plots
    out = ensure_dir(p.study_dir / "out")
    rain = [out / "rain.csv"] if res.inflow.rain is not None else []
    files = [out / name for name in output_names(with_rain=bool(rain))]
    with LOCK:
        stale = stale_outputs(out, files)
        # 失敗したときは、前の出力にだけあるファイルも削除する（conditions.json が消えると、次の出力で削除できなくなる）
        with all_or_nothing(files + stale):
            res.table.to_csv(out / "route.csv", index=False, encoding="utf-8-sig", lineterminator="\r\n")
            write_table(inflow_frame(res).round(6), out / "inflow.csv")
            design_plots.write_all(res, out, p.title)
            if rain:
                rain_table(res.inflow.rain, p.tc, res.inflow.start).to_csv(rain[0], index=False, encoding="utf-8-sig",
                                                                           lineterminator="\r\n")
            (out / "design_summary.md").write_bytes("\r\n".join(lines).encode("utf-8"))
            # 要約とグラフ、時刻ごとの追跡の結果の表を清書した PDF（表は要約の design_summary.md には入れない）
            design_pdf.write_pdf(lines + [""] + routing_table_lines(res), out, out / "design_summary.pdf")
            write_conditions(out, conditions, files)
            for f in stale:
                f.unlink(missing_ok=True)
    return files


@dataclass
class RunOutput:
    prepared: Prepared
    result: DesignResult
    lines: list[str]
    notices: list[str]
    files: list[Path] = field(default_factory=list)


def trial(conf: dict, study_dir: Path, tables: dict | None = None) -> RunOutput:
    """試算: 計算して要約と注意を返す。ファイルには書かない。"""
    p = prepare(conf, study_dir, tables)
    res = compute(p)
    return RunOutput(p, res, summary_lines(p, res), notices(res, p))


def run_study(conf: dict, study_dir: Path, tables: dict | None = None) -> RunOutput:
    """計算して out/ に書く（cli の amamizu study と、WebGUI の「保存して出力」）。"""
    r = trial(conf, study_dir, tables)
    conditions = output_conditions(conf, Path(study_dir), tables)
    if r.prepared.observations is not None:
        conditions["観測値"] = observation_record(r.prepared.observations)
    r.files = write_outputs(r.prepared, r.result, r.lines, conditions)
    return r
