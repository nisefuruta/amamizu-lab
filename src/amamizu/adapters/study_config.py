"""study.json の読み込み・確認・書き戻し（cli と gui が使う）。

書き間違いが既定値で黙って通らないよう、SCHEMA にない項目があればエラーにする。
SCHEMA の値は項目の形: dict は入れ子、[形] はその形のリスト、MAP(形) は項目名を自由に書ける dict（値はどれも同じ形）、
文字列は値の型（TYPES。"num" は数、"int" は整数、"str" は文字列、"bool" は true か false、"frac" は数か分数の文字列）、
None は確かめない（kind と guideline。check_kind と JsonGuideline.of_study が確かめる）。
型の違う値はエラーにする（整数の項目の 10.0 を 10 として、数の項目の true を 1 として、"false" を真として計算しない）。
計算に必要な項目（REQUIRED）がなければエラーにする（書き忘れを、項目名を示さない誤りで止めない）。
"""
import json
from dataclasses import fields
from fractions import Fraction
from pathlib import Path

from ..usecases.storage_study import ModelConditions


def MAP(shape) -> tuple:
    """項目名を自由に書ける dict（long_record.starts の地点名など）。値はどれも shape の形。"""
    return ("map", shape)


META = {"kind": None, "title": "str", "guideline": None}
# 浸透の書き方（最上位の infiltration と、区域ごとの catchments[].infiltration で同じ）
INFILTRATION = {
    "rate_m3s": "num", "intensity_mmh": "num", "void_storage_m3": "num",
    "facilities": [{"name": "str", "specific_m2": "num",
                    "shape": {"kind": "str", "H_m": "num", "W_m": "num", "D_m": "num", "L_m": "num"},
                    "permeability": {"value": "num", "unit": "str"}, "count": "num",
                    "factors": {"clogging": "num", "groundwater": "num", "safety": "num"},
                    "void_parts": [{"volume_m3": "num", "porosity_pct": "num"}]}],
}
ORIFICE = {"sill_m": "num", "shape": "str", "width_m": "num", "height_m": "num", "diameter_m": "num"}
SCHEMA = {
    "storage": {
        "conditions": {f.name: {float: "num", int: "int", str: "str"}[f.type] for f in fields(ModelConditions)},
        "long_record": {"note": "str", "station_kind": "str", "starts": MAP("int"), "end_year": "int"},
        "partial_year": {"from": "str", "to": "str"},
        "events": [{"name": "str", "stations": ["str"], "start": "str", "end": "str", "in_report": "bool", "overflow_detail": "bool"}],
    },
    "return_period": {
        "standard_period": "int", "duration_min": "int", "last_year": "int",
        "event": {"label": "str", "month": "str"},
    },
    "design": {
        "hydraulics": {"g": "num", "c_weir": "num", "c_orifice": "num"},      # guideline が null（手引を使用しない）のとき
        "created": "str", "tc_min": "int", "end_min": "int", "mode": "str", "target_q_m3s": "num", "allowable_q_m3s": "num",
        "rain": {"guideline": {"district": "str", "return_period": "int"},
                 "formula": {"type": "str", "a": "num", "b": "num", "n": "frac"},
                 "shape": "str", "hours": "int", "round_digits": "int"},
        "catchments": [{"name": "str", "area_ha": "num", "f": "num", "infiltration": INFILTRATION}],
        "inflow_csv": {"file": "str", "time_col": "str", "q_col": "str", "rain_col": "str", "reverse": "bool"},
        "observed_rain": {"station": "str", "series": "str", "start": "str", "end": "str"},
        "pond": {"prismatic": {"area_m2": "num", "hwl_m": "num"},
                 "table": {"file": "str", "level_col": "str", "area_col": "str", "volume_col": "str", "hwl_m": "num"}},
        "orifice": ORIFICE,
        "orifices": [ORIFICE],
        "spillway": {"crest_m": "num", "width_m": "num"},
        "pump": {"levels_m": ["num"], "q_m3s": ["num"]},
        "judge": {"storage_margin_m3": "num"},
        "infiltration": INFILTRATION,
    },
}
# 値の型 → 違うときの直し方
TYPES = {"num": "数で書く", "int": "整数で書く", "str": "文字列（\"…\"）で書く", "bool": "true か false で書く",
         "frac": "数か分数の文字列（\"3/4\" など）で書く"}


def _int_paths(schema, prefix: str = "") -> list[str]:
    """整数の項目の場所（リストの中を除く。"*" は MAP のどの項目も）。WebGUI が 10.0 を 10 に直すのに使う。"""
    if schema == "int":
        return [prefix]
    if isinstance(schema, tuple):
        return _int_paths(schema[1], f"{prefix}.*")
    if isinstance(schema, dict):
        return [p for k, s in schema.items() for p in _int_paths(s, f"{prefix}.{k}".lstrip("."))]
    return []


INT_PATHS = {kind: _int_paths(s) for kind, s in SCHEMA.items()}
# 必ず書く項目（リストの中は番号を除いた形。events.name は events の各降雨の name）。design は項目の組み合わせが
# 計算の種類で変わるので、check_sections がまとまりごとに確かめる
REQUIRED = {
    "storage": ["title", "conditions", "long_record", "long_record.starts", "long_record.end_year", "partial_year",
                "partial_year.from", "partial_year.to", "events", "events.name", "events.stations", "events.start", "events.end"],
    "return_period": ["title", "standard_period", "duration_min", "last_year", "event", "event.label", "event.month"],
    "design": [],
}


def _walk(conf, schema, prefix, unknown: list[str], shape: list[str]) -> None:
    """SCHEMA にない項目を unknown に、形の違う項目（リストのはずがリストでない、など）を shape に集める。"""
    if schema is None or isinstance(schema, str):
        return
    if isinstance(schema, list):
        if not isinstance(conf, list):
            shape.append(f"{prefix.rstrip('.')}（リストで書く）")
            return
        for i, c in enumerate(conf):
            _walk(c, schema[0], f"{prefix.rstrip('.')}[{i}].", unknown, shape)
        return
    if not isinstance(conf, dict):
        shape.append(f"{prefix.rstrip('.')}（項目を持つ形で書く）")
        return
    if isinstance(schema, tuple):
        return
    for k, v in conf.items():
        if k not in schema:
            unknown.append(prefix + k)
        else:
            _walk(v, schema[k], f"{prefix}{k}.", unknown, shape)


def _unknown(conf: dict) -> tuple[list[str], list[str]]:
    unknown: list[str] = []
    shape: list[str] = []
    _walk(conf, META | SCHEMA[check_kind(conf)], "", unknown, shape)
    return unknown, shape


def check_kind(conf: dict) -> str:
    kind = conf.get("kind")
    if kind not in SCHEMA:
        head = "study.json に kind がありません。" if kind is None else f"study.json の kind「{kind}」は扱えません。"
        raise ValueError(f"{head}kind は design（設計の検討）、storage（貯留施設の検証）、"
                         "return_period（降雨強度の検証）のどれかにしてください。")
    return kind


def unknown_keys(conf: dict) -> list[str]:
    """study.json の不明な項目（書き間違い）の一覧。形の違う項目は shape_errors と value_errors が示す。"""
    return _unknown(conf)[0]


def shape_errors(conf: dict) -> list[str]:
    """形の違う項目（リストで書く項目がリストでない、項目を持つ形で書く項目がそうでない）の一覧。"""
    return _unknown(conf)[1]


def _type_ok(v: object, t: str) -> bool:
    num = isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "frac" and isinstance(v, str):
        try:
            Fraction(v)
        except (ValueError, ZeroDivisionError):
            return False
        return True
    return {"num": num, "frac": num, "int": num and isinstance(v, int), "str": isinstance(v, str),
            "bool": isinstance(v, bool)}[t]


def _typed(conf, schema, path: str, bad: list[tuple[str, str]]) -> None:
    """SCHEMA の型・形と違う値を bad に集める（不明な項目は _walk が集める）。形の違い（リストで書く項目が文字列、など）も
    集める（WebGUI で読めない入力をそのまま条件に入れたとき、保存を止めて項目を示す）。"""
    def wrong(how: str) -> None:
        bad.append((path, f"{path}（今は {json.dumps(conf, ensure_ascii=False)}。{how}）"))
    if isinstance(schema, str):
        if not _type_ok(conf, schema):
            wrong(TYPES[schema])
    elif isinstance(schema, list):
        if not isinstance(conf, list):
            wrong("リストで書く")
            return
        for i, c in enumerate(conf):
            _typed(c, schema[0], f"{path}[{i}]", bad)
    elif isinstance(schema, (tuple, dict)) and not isinstance(conf, dict):
        wrong("項目を持つ形で書く")
    elif isinstance(schema, tuple):
        for k, c in conf.items():
            _typed(c, schema[1], f"{path}.{k}", bad)
    elif isinstance(schema, dict):
        for k, c in conf.items():
            if k in schema:
                _typed(c, schema[k], f"{path}.{k}".lstrip("."), bad)


def value_errors(conf: dict) -> list[tuple[str, str]]:
    """型・形の違う値の（場所, 説明）の一覧（整数の項目の 10.0、数の項目の true や "1.0"、true・false の項目の "false"、
    リストで書く項目の文字列など）。"""
    bad: list[tuple[str, str]] = []
    _typed(conf, META | SCHEMA[check_kind(conf)], "", bad)
    return bad


def missing_items(conf: dict) -> list[str]:
    """必ず書く項目（REQUIRED）のうち、ない項目の場所の一覧（events[1].name など）。上の項目がなければ、その項目だけを示す。"""
    out = []
    for path in REQUIRED[check_kind(conf)]:
        *parents, key = path.split(".")
        nodes = [("", conf)]
        for k in parents:
            nxt = []
            for name, node in nodes:
                v = node.get(k) if isinstance(node, dict) else None
                sub = f"{name}.{k}".lstrip(".")
                nxt += [(f"{sub}[{i}]", x) for i, x in enumerate(v)] if isinstance(v, list) else [(sub, v)]
            nodes = nxt
        out += [f"{name}.{key}".lstrip(".") for name, node in nodes if isinstance(node, dict) and key not in node]
    return out


def check_study(conf: dict) -> dict:
    """不明な項目・必要な項目の不足・書き方の違う値があればエラーにする。問題がなければ conf をそのまま返す。"""
    unknown = unknown_keys(conf)
    if unknown:
        raise ValueError(f"study.json に不明な項目（amamizu が使わない項目）があります: {', '.join(unknown)}"
                         "（「a.b」は a の中の b、「[0]」はリストの1番目）。項目名の書き間違いなら直し、不要なら削除してください"
                         "（書き間違いのまま既定値で計算しないよう、中止しました）。")
    bad = value_errors(conf)      # 形の違い（リストで書く項目が辞書など）を、その中の項目の不足より先に示す
    if bad:
        raise ValueError(f"study.json の次の項目の値を直してください（整数の項目は 10.0 ではなく 10）: {', '.join(t for _, t in bad)}"
                         "（「a.b」は a の中の b、「[0]」はリストの1番目）")
    missing = missing_items(conf)
    if missing:
        raise ValueError(f"study.json に必要な項目がありません: {', '.join(missing)}"
                         "（「a.b」は a の中の b、「[0]」はリストの1番目）。study.json に書いてください（書き方は docs/条件設定.md）。")
    return conf


def load_study(study_dir: Path) -> dict:
    """study.json を読み、確かめて返す。"""
    return check_study(read_study(study_dir))


def read_study(study_dir: Path) -> dict:
    """study.json を読むだけ（項目は確かめない。GUI で誤りのある study.json を開くときに使う）。
    JSON として読めない、または項目を持つ形（{ で始まる形）でなければ ValueError。"""
    path = Path(study_dir) / "study.json"
    try:
        conf = json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        raise ValueError(f"{path} の文字コードが UTF-8 ではありません。UTF-8 で保存し直してください。") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} を JSON として読めません（{e}）。JSON の書き方の誤り（カンマ・括弧・引用符の過不足など）を、"
                         "テキストエディタで直してください。") from None
    if not isinstance(conf, dict):
        raise ValueError(f"{path} の中身が、項目を持つ形（{{ \"kind\": ... }} のように {{ で始まる形）ではありません。"
                         "テキストエディタで直してください。")
    return conf


def remove_keys(conf: dict, paths: list[str]) -> dict:
    """unknown_keys が返した項目を取り除いた写しを返す（GUI で、利用者が削除を選んだときだけ使う）。"""
    import copy
    import re
    out = copy.deepcopy(conf)
    for path in paths:
        parts = re.findall(r"[^.\[\]]+|\[\d+\]", path.split("（")[0])
        node = out
        for p in parts[:-1]:
            node = node[int(p[1:-1])] if p.startswith("[") else node[p]
        last = parts[-1]
        if isinstance(node, dict) and not last.startswith("["):
            node.pop(last, None)
    return out


def _ordered(conf, schema):
    """SCHEMA の順に項目を並べる（SCHEMA が None の中身はそのままの順）。"""
    if schema is None or isinstance(schema, (str, tuple)):
        return conf
    if isinstance(schema, list):
        return [_ordered(c, schema[0]) for c in conf]
    return {k: _ordered(conf[k], schema[k]) for k in schema if k in conf}


def dump_study(conf: dict) -> bytes:
    """study.json に書くバイト。確かめてから、SCHEMA の順、2字下げ、UTF-8、改行は CRLF。
    値は受け取ったまま書く（n の "3/4" のような文字列はそのまま）。"""
    check_study(conf)
    ordered = _ordered(conf, META | SCHEMA[conf["kind"]])
    text = json.dumps(ordered, ensure_ascii=False, indent=2) + "\n"
    return text.replace("\n", "\r\n").encode("utf-8")
