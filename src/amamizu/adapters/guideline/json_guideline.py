"""手引の定数を JSON から読む（Guideline の実装）。

手引の定数のファイルは、検討のフォルダ（studies/）の下の _guideline/ に置く（<名前>.json）。
study.json の guideline は、その名前か .json のパス（検討フォルダから見た相対パスか絶対パス）。
手引を使用しない検討は null（放流の係数は study.json の hydraulics に書く。貯留追跡計算だけ）。
"""
import json
import os
from fractions import Fraction
from pathlib import Path

from ...domain.intensity import IntensityFormula
from ...domain.outlet import Hydraulics
from ...usecases.ports import HayamiCell, InfiltrationCell

FOLDER = "_guideline"          # 検討のフォルダ（studies/）の下の、手引の定数のファイルのフォルダ
NEW_STUDY = "chiba_2006"       # WebGUI で新しい検討を作るときに選んでおく手引の定数（_guideline/ にあるとき）
HYDRAULICS_KEYS = ("g", "c_weir", "c_orifice")
HOW_TO_WRITE = ("手引の定数のファイルを使うときはその名前（{names}）か .json のパス、"
                "使用しないときは null と、放流の係数 hydraulics（g、c_weir、c_orifice）を書いてください。")


def guideline_dir(studies: Path | str) -> Path:
    return Path(studies) / FOLDER


def guideline_names(studies: Path | str) -> list[str]:
    """選べる手引の定数の名前（_guideline/ の <名前>.json。「_」で始まるファイルは雛形なので除く）。"""
    return sorted(p.stem for p in guideline_dir(studies).glob("*.json") if not p.name.startswith("_"))


def relocated(name: object, src_dir: Path | str, dst_dir: Path | str, copied: tuple[str, ...]) -> object:
    """study.json の guideline を、src_dir の検討から dst_dir の検討の study.json へ写すときの値（複製・event-studies）。
    .json の相対パスは src_dir から見たパスなので、同じファイルを dst_dir から見たパスに直す（ドライブが違えば絶対パス）。
    copied は dst_dir へコピーする src_dir の中のフォルダ（複製の input/ など）。その中のファイルは、コピーしたほうを指す
    （元の検討を削除・編集しても、写した検討は変わらない）。名前・絶対パス・null はそのまま。"""
    if not (isinstance(name, str) and name.endswith(".json")) or Path(name).is_absolute():
        return name
    src = Path(src_dir).resolve()
    target = (src / name).resolve()
    if any(target.is_relative_to(src / c) for c in copied):
        return target.relative_to(src).as_posix()
    try:
        return Path(os.path.relpath(target, Path(dst_dir).resolve())).as_posix()
    except ValueError:
        return str(target)


def _names_text(studies: Path | str) -> str:
    names = guideline_names(studies)
    return "、".join(names) if names else f"{guideline_dir(studies)} にファイルがありません"


# 手引の定数の項目の形（読み込むときに、ファイルに書いてある項目をすべて確かめる）。"num" は数（true・false を除く）、"int" は整数、"str" は文字列、
# "frac" は数か分数の文字列（"3/4"）。("map", 形, [必ずある項目]) は項目を持つ形で値がどれも同じ形、("map_int", 形) はその項目名が
# 確率年の整数、("list", 形) はリスト、("row", [形, …]) は決まった個数のリスト、("obj", {項目: 形}) は決まった項目だけを持つ形
SHAPES: dict = {
    "name": "str", "region": "str", "issuer": "str", "citation": "str", "default_district": "str", "district_term": "str",
    "cite": ("map", "str"),
    "infiltration_notes": ("obj", {"factors": "str", "void": "str"}),
    "formulas": ("map", ("map_int", ("row", ["num", "frac", "num"]))),
    "formula_table": ("map", ("map_int", ("list", "num"))),
    "formula_table_conditions": ("obj", {"t_min": ("list", "int"), "digits": "int"}),
    "qc_min": ("map", "num", ["default"]),
    "runoff": ("obj", {"impervious": "num", "pervious": "num", "cap": "num"}),
    "district_of_station": ("map", "str"),
    "hayami": ("map", ("obj", {"qc": ("list", "num"), "v": ("list", ("list", "num"))})),
    "hayami_imp": ("list", "num"),
    "hayami_f": ("list", "num"),
    "hayami_conditions": ("obj", {"tc_min": "int", "pond_area_ratio": "num", "return_period": "int"}),
    "infiltration_table": ("map", ("obj", {"qc": "num", "v": ("list", "num")})),
    "infiltration_table_rows": ("list", ("row", ["num", "num"])),
    "infiltration_table_conditions": ("obj", {"f": "num", "area_ha": "num", "tc_min": "int", "return_period": "int",
                                              "pond_area_ratio": "num"}),
}
_HOW = {"num": "数で書いて", "int": "整数で書いて", "str": "文字列で書いて", "frac": "数か分数の文字列（\"3/4\" など）で書いて"}


def _is_fraction(v: object) -> bool:
    try:
        Fraction(str(v))
    except (ValueError, ZeroDivisionError):
        return False
    return True


def _shape_error(v: object, shape, path: str) -> tuple[str, str | None, object] | None:
    """v が shape の形でなければ、最初の誤りの（場所, 直し方（項目がないときは None）, 値）。形が合っていれば None。"""
    if isinstance(shape, str):
        num = isinstance(v, (int, float)) and not isinstance(v, bool)
        ok = {"num": num, "int": num and isinstance(v, int), "str": isinstance(v, str),
              "frac": num or (isinstance(v, str) and _is_fraction(v))}[shape]
        return None if ok else (path, _HOW[shape], v)
    kind, inner, *rest = shape
    if kind in ("map", "map_int", "obj"):
        if not isinstance(v, dict):
            return path, "項目を持つ形（{ }）で書いて", v
        for k in list(inner) if kind == "obj" else (rest[0] if rest else []):
            if k not in v:
                return f"{path}.{k}", None, None
        if kind == "obj":
            unknown = [k for k in v if k not in inner]
            if unknown:
                return path, f"{'、'.join(inner)} だけで書いて", v
        items = inner.items() if kind == "obj" else [(k, inner) for k in v]
        for k, s in items:
            if kind == "map_int" and not k.isdigit():
                return f"{path}.{k}", "項目名を確率年の整数（\"50\" など）にして", k
            err = _shape_error(v[k], s, f"{path}.{k}")
            if err:
                return err
        return None
    if not isinstance(v, list):
        return path, "リスト（[ ]）で書いて", v
    if kind == "row" and len(v) != len(inner):
        return path, f"{len(inner)}個の値のリストで書いて", v
    for i, x in enumerate(v):
        err = _shape_error(x, inner[i] if kind == "row" else inner, f"{path}[{i}]")
        if err:
            return err
    return None


def hydraulics_from(spec: object, where: str) -> Hydraulics:
    """hydraulics（g、c_weir、c_orifice）から放流の係数を作る。where は誤りの文に示すファイル。"""
    if not isinstance(spec, dict):
        raise ValueError(f"{where} の hydraulics は {{\"g\": 9.8, \"c_weir\": 1.8, \"c_orifice\": 0.6}} の形で書いてください。")
    lack = [k for k in HYDRAULICS_KEYS if spec.get(k) is None]
    if lack:
        raise ValueError(f"{where} の hydraulics に {'、'.join(lack)} がありません"
                         "（g は重力加速度 m/s2、c_weir は越流係数、c_orifice はオリフィスの流量係数）。")
    unknown = [k for k in spec if k not in HYDRAULICS_KEYS]
    if unknown:
        raise ValueError(f"{where} の hydraulics の {'、'.join(unknown)} は使えない項目です。g、c_weir、c_orifice だけで書いてください。")
    bad = [k for k in HYDRAULICS_KEYS if isinstance(spec[k], bool) or not isinstance(spec[k], (int, float))]
    if bad:
        raise ValueError(f"{where} の hydraulics の {'、'.join(bad)} は数で書いてください。")
    return Hydraulics(g=float(spec["g"]), c_weir=float(spec["c_weir"]), c_orifice=float(spec["c_orifice"]))


class JsonGuideline:
    @classmethod
    def load(cls, name_or_path: str, studies: Path | str, base: Path | str) -> "JsonGuideline":
        """名前（studies/_guideline/<名前>.json）またはパス（.json。相対パスは base から）で読む。
        base は省略できない（study.json の値なら検討のフォルダ、コマンドの引数なら今のフォルダを呼び出し側が選ぶ）。"""
        if name_or_path.endswith(".json"):
            p = Path(name_or_path)
            p = p if p.is_absolute() else Path(base) / p
        else:
            p = guideline_dir(studies) / f"{name_or_path}.json"
        if not p.exists():
            raise FileNotFoundError(f"手引の定数のファイルがありません: {p}。study.json の guideline（コマンドでは --guideline）に、"
                                    f"名前（{_names_text(studies)}）か、手引の定数の .json のパスを書いてください。"
                                    f"手引の定数のファイルは {guideline_dir(studies)} に置きます。")
        return cls(p)

    @classmethod
    def of_study(cls, conf: dict, study_dir: Path | str) -> "JsonGuideline | None":
        """study.json の guideline の手引の定数。null（手引を使用しない）なら None。guideline がなければエラー
        （書き忘れたまま、どれかの基準の値で計算しない）。"""
        studies = Path(study_dir).resolve().parent
        if "guideline" not in conf:
            raise ValueError("study.json に guideline がありません。" + HOW_TO_WRITE.format(names=_names_text(studies)))
        name = conf["guideline"]
        if name is None:
            return None
        if not isinstance(name, str) or not name:
            raise ValueError(f"study.json の guideline（{name!r}）が読めません。" + HOW_TO_WRITE.format(names=_names_text(studies)))
        return cls.load(name, studies, Path(study_dir).resolve())

    @classmethod
    def required(cls, conf: dict, study_dir: Path | str) -> "JsonGuideline":
        """手引の計画降雨と比べる検討（kind が storage、return_period）の手引の定数。null は書けない。"""
        g = cls.of_study(conf, study_dir)
        if g is None:
            raise ValueError(f"この検討（kind: {conf.get('kind')}）は手引の計画降雨と比べるので、study.json の guideline に null は書けません。"
                             f"手引の定数の名前（{_names_text(Path(study_dir).resolve().parent)}）か .json のパスを書いてください。")
        return g

    def __init__(self, path: Path | str):
        try:
            self._d = json.loads(Path(path).read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            raise ValueError(f"手引の定数のファイル {path} の文字コードが UTF-8 ではありません。UTF-8 で保存し直してください。") from None
        except json.JSONDecodeError as e:
            raise ValueError(f"手引の定数のファイル {path} を JSON として読めません（{e}）。JSON の書き方の誤り"
                             "（カンマ・括弧・引用符の過不足など）を、テキストエディタで直してください。") from None
        if not isinstance(self._d, dict) or not isinstance(self._d.get("name"), str):
            raise KeyError(f"手引の定数のファイル {path} に「name」（基準の名前。文字列）がありません。手引の定数の .json に書いてください。")
        self.name = self._d["name"]
        self._check_shapes()

    def _check_shapes(self) -> None:
        """書いてある項目の中身の形（SHAPES）と、表どうしの大きさがそろっているかを確かめる。違えば場所を示すエラーにする。"""
        for key, shape in SHAPES.items():
            err = _shape_error(self._d[key], shape, key) if key in self._d else None
            if err is None:
                continue
            where, how, value = err
            if how is None:
                raise KeyError(f"手引の定数（{self.name}）に「{where}」がありません。手引の定数の .json に書いてください。")
            raise ValueError(f"手引の定数（{self.name}）の「{where}」は{how}ください（今は {json.dumps(value, ensure_ascii=False)}）。")
        d = self._d
        if "hydraulics" in d:
            hydraulics_from(d["hydraulics"], f"手引の定数（{self.name}）")
        if "formula_table" in d:
            ts = self._need("formula_table_conditions", "formula_table の値の継続時間と小数の桁数を示す")["t_min"]
            for name, rows in d["formula_table"].items():
                for T, vals in rows.items():
                    if len(vals) != len(ts):
                        raise ValueError(f"手引の定数（{self.name}）の「formula_table.{name}.{T}」の値の数（{len(vals)}）を、"
                                         f"formula_table_conditions.t_min の数（{len(ts)}）とそろえてください。")
        if "formulas" in d:                    # 地区名の書き間違い（「勝浦 」など）を、default の値や別の地区で黙って通さない
            where = [("qc_min の項目名", [k for k in d.get("qc_min", {}) if k != "default"]),
                     ("formula_table の項目名", list(d.get("formula_table", {}))), ("hayami の項目名", list(d.get("hayami", {}))),
                     ("infiltration_table の項目名", list(d.get("infiltration_table", {}))),
                     ("district_of_station の値", list(d.get("district_of_station", {}).values())),
                     ("default_district", [d["default_district"]] if "default_district" in d else [])]
            for label, names in where:
                unknown = [n for n in names if n not in d["formulas"]]
                if unknown:
                    raise ValueError(f"手引の定数（{self.name}）の {label}に、formulas にない地区{'、'.join(f'「{n}」' for n in unknown)}があります。"
                                     f"地区は {'、'.join(d['formulas'])} のどれかにしてください。")
        if "hayami" in d and "hayami_imp" in d and "hayami_f" in d:
            imps, fs = d["hayami_imp"], d["hayami_f"]
            if len(imps) != len(fs):
                raise ValueError(f"手引の定数（{self.name}）の hayami_imp と hayami_f の値の数（{len(imps)}、{len(fs)}）をそろえてください。")
            for name, t in d["hayami"].items():
                if len(t["v"]) != len(imps):
                    raise ValueError(f"手引の定数（{self.name}）の「hayami.{name}.v」の行の数（{len(t['v'])}）を、"
                                     f"hayami_imp の値の数（{len(imps)}）とそろえてください。")
                for i, row in enumerate(t["v"]):
                    if len(row) != len(t["qc"]):
                        raise ValueError(f"手引の定数（{self.name}）の「hayami.{name}.v[{i}]」の値の数（{len(row)}）を、"
                                         f"「hayami.{name}.qc」の値の数（{len(t['qc'])}）とそろえてください。")
        if "infiltration_table" in d and "infiltration_table_rows" in d:
            for name, t in d["infiltration_table"].items():
                if len(t["v"]) != len(d["infiltration_table_rows"]):
                    raise ValueError(f"手引の定数（{self.name}）の「infiltration_table.{name}.v」の値の数（{len(t['v'])}）を、"
                                     f"infiltration_table_rows の行の数（{len(d['infiltration_table_rows'])}）とそろえてください。")

    def _need(self, key: str, use: str):
        """手引の定数の項目（形は読み込むときに確かめてある）。なければ、項目名と使い道（use）を示すエラーにする。"""
        if key not in self._d:
            raise KeyError(f"手引の定数（{self.name}）に「{key}」がありません。{use}ので、手引の定数の .json に書いてください。")
        return self._d[key]

    def formula(self, district: str, return_period: int) -> IntensityFormula:
        rows = self._formulas_of(district)
        if str(return_period) not in rows:
            raise KeyError(f"手引（{self.name}）の地区「{district}」に確率年 {return_period} 年の強度式がありません。"
                           f"確率年は {'、'.join(rows)} のどれかにしてください。")
        a, n, b = rows[str(return_period)]
        return IntensityFormula(a, float(Fraction(n)), b)

    def formula_spec(self, district: str, return_period: int) -> dict:
        """地区と確率年の強度式を、study.json の rain.formula の書き方にしたもの（n は分数の文字列のまま）。"""
        self.formula(district, return_period)
        a, n, b = self._formulas_of(district)[str(return_period)]
        return {"type": "cleveland", "a": a, "b": b, "n": n}

    def formula_table(self):
        out = []
        table = self._need("formula_table", "amamizu verify で強度式を表の値と照合する")
        ts = self.formula_table_conditions()["t_min"]
        for d, rows in table.items():
            for T, vals in rows.items():
                for t, v in zip(ts, vals, strict=True):
                    out.append((d, int(T), t, v))
        return out

    def formula_table_conditions(self) -> dict:
        return dict(self._need("formula_table_conditions", "amamizu verify で強度式を表の値と照合する"))

    def _formulas_of(self, district: str) -> dict:
        """地区の強度式（確率年ごと）。手引にない地区は、選べる地区を示すエラー。"""
        rows = self._need("formulas", "強度式（計画降雨）に使う")
        if district not in rows:
            raise KeyError(f"手引（{self.name}）に地区「{district}」がありません。地区は {'、'.join(rows)} のどれかにしてください。")
        return rows[district]

    def qc_min(self, district: str) -> float:
        self._formulas_of(district)
        q = self._need("qc_min", "モデル施設の許容放流量に使う")
        return q.get(district, q["default"])

    def district_of(self, station: str) -> str:
        stations = self._need("district_of_station", "観測所の地区を決めるのに使う")
        if station not in stations:
            raise KeyError(f"手引（{self.name}）に地点「{station}」の地区がありません。地点は {'、'.join(stations)} のどれかにしてください。")
        return stations[station]

    def stations(self) -> list[str]:
        return list(self._need("district_of_station", "観測所の地区を決めるのに使う"))

    def districts(self) -> list[str]:
        return list(self._need("formulas", "強度式（計画降雨）に使う"))

    def return_periods(self, district: str) -> list[int]:
        """地区の強度式がある確率年（WebGUI の選択肢）。"""
        return [int(k) for k in self._formulas_of(district)]

    def default_district(self) -> str:
        """WebGUI で計画降雨を選ぶときの地区の初期値（default_district）。JSON になければ式の表の最初の地区。"""
        if "default_district" in self._d:
            return self._need("default_district", "WebGUI の地区の初期値に使う")
        return self.districts()[0]

    def district_term(self) -> str:
        """強度式を分ける区域の呼び方（district_term。千葉県の手引は「地区」）。JSON になければ「地区」。"""
        return self._need("district_term", "区域の呼び方に使う") if "district_term" in self._d else "地区"

    def runoff(self) -> dict:
        return dict(self._need("runoff", "レポートと早見表との照合に使う"))

    def hayami(self) -> list[HayamiCell]:
        """早見表の全マス。早見表のない基準では空。"""
        if "hayami" not in self._d:
            return []
        use = "早見表との照合に使う"
        table, imps, fs = self._need("hayami", use), self._need("hayami_imp", use), self._need("hayami_f", use)
        return [HayamiCell(d, imps[i], fs[i], qc, v)
                for d, t in table.items() for i, row in enumerate(t["v"]) for qc, v in zip(t["qc"], row, strict=True)]

    def infiltration_table(self) -> list[InfiltrationCell]:
        """浸透施設を設けたときの必要洪水調節容量の表の全マス。表のない基準では空。"""
        if "infiltration_table" not in self._d:
            return []
        use = "浸透ありの容量の表との照合に使う"
        table, rows = self._need("infiltration_table", use), self._need("infiltration_table_rows", use)
        return [InfiltrationCell(d, i, a, t["qc"], v)
                for d, t in table.items() for (i, a), v in zip(rows, t["v"], strict=True)]

    def infiltration_table_conditions(self) -> dict:
        return dict(self._need("infiltration_table_conditions", "浸透ありの容量の表との照合に使う"))

    def infiltration_notes(self) -> dict[str, str]:
        """WebGUI の浸透施設の欄の近くに表示する、手引の影響係数（factors）と空隙貯留量（void）の文。JSON になければ空。"""
        return dict(self._d.get("infiltration_notes", {}))

    def report_text(self, key: str) -> str:
        """レポートの文に使う語（region / issuer / citation）。JSON にないときは、レポートに千葉県の文を記載しないようエラーにする。"""
        return self._need(key, "レポートの文に使う")

    def cite(self, item: str) -> str:
        """項目（formulas / districts / design_storm / qc_min / hayami）の手引の箇所。"""
        c = self._need("cite", "レポートの文に使う")
        if item not in c:
            raise KeyError(f"手引の定数（{self.name}）の cite に「{item}」がありません。レポートの文に使うので、手引の定数の .json に書いてください。")
        return c[item]

    def hydraulics(self) -> Hydraulics:
        """重力加速度と放流係数（越流、オリフィス）。"""
        spec = self._need("hydraulics", "放流の係数（重力加速度 g、越流係数 c_weir、オリフィスの流量係数 c_orifice）に使う")
        return hydraulics_from(spec, f"手引の定数（{self.name}）")

    def hayami_conditions(self) -> dict:
        return dict(self._need("hayami_conditions", "早見表との照合に使う"))
