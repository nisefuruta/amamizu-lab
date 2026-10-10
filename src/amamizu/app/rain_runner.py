"""計画降雨だけを作る（cli の amamizu rain と、WebGUI の「計画降雨」が使う。組み立ての層）。
実測降雨の記録も同じ形（ハイエトグラフと区間ごとの表）にする（WebGUI の「降雨記録」が使う。make_observed_rain）。

条件は dict で受け取る:
  手引の式: {"district", "return_period", "guideline"（任意）}
  5方式:    {"type", "a", "b", "n"}（n は "3/4" のような分数の文字列でもよい）
  共通:     {"shape"（rear／center）, "hours"（既定24）, "tc"（既定は domain.pond の DEFAULT_TC_MIN）, "round"（任意。小数点以下の桁）}
"""
import json
import datetime
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from ..adapters.design_inputs import OBSERVED_SERIES, load_observed
from ..adapters.jma_store import JmaStore
from ..adapters.guideline.json_guideline import JsonGuideline
from ..domain.design_storm import SHAPES, planned_storm
from ..domain.intensity import formula_of
from ..domain.pond import DEFAULT_TC_MIN
from ..usecases.design_study import formula_text, observed_storm, rain_table
from .design_runner import LOCK, all_or_nothing, ensure_dir, gap_notes, output_names, round_note


@dataclass
class Rain:
    spec: dict
    label: str
    storm: np.ndarray
    table: pd.DataFrame
    tc: int
    start: datetime.datetime | None = None           # 実測降雨の最初の区間の始まり（計画降雨は None）
    notes: list = field(default_factory=list)        # 実測降雨の欠測の時間など（見出し, 表記）

    @property
    def total_mm(self) -> float:
        return float(self.storm.sum() * self.tc / 60)

    @property
    def peak(self) -> int:
        return int(self.storm.argmax())


def make_rain(spec: dict, studies: Path | str, base: Path | str) -> Rain:
    """条件から計画降雨を作る。誤りは ValueError。studies は検討のフォルダ（手引の定数を名前で読むとき。_guideline/ から探す）。
    base は手引の定数の .json の相対パスの基準（WebGUI では検討のフォルダ、コマンドでは今のフォルダ）。"""
    tc, hours, digits = spec.get("tc", DEFAULT_TC_MIN), spec.get("hours", 24), spec.get("round")
    for name, v in (("区間の長さ tc", tc), ("継続時間 hours", hours), ("丸めの桁数 round", digits)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
            raise ValueError(f"{name} は整数で指定してください（今は {v}）。")
    if spec.get("district"):
        extra = [k for k in ("type", "a", "b", "n") if spec.get(k) is not None]
        if extra:                                  # 指定した係数を黙って使わずに、手引の式で作らない
            raise ValueError(f"手引の地区を使うときは、式の種類と係数（{'、'.join(extra)}。コマンドでは --type・--a・--b・--n）を指定しないでください。")
        if not spec.get("guideline"):
            raise ValueError("手引の地区を使うときは、手引の定数（guideline。コマンドでは --guideline）を指定してください。")
        g = JsonGuideline.load(spec["guideline"], studies, base)
        rp = 50 if spec.get("return_period") is None else spec["return_period"]
        formula, kind = g.formula(spec["district"], rp), None
        label = f"手引 {spec['district']}{g.district_term()} 1/{rp} "
    else:
        if not spec.get("type"):
            raise ValueError("式の種類か手引の地区のどちらかを指定してください（コマンドでは --type か --district）。")
        extra = [k for k in ("return_period", "guideline") if spec.get(k) is not None]
        if extra:                                  # 指定した確率年・手引の定数を黙って使わない
            raise ValueError(f"式の種類と係数を使うときは、手引の式の条件（{'、'.join(extra)}。コマンドでは --return-period・--guideline）を"
                             "指定しないでください。")
        n = spec.get("n")
        formula = formula_of(spec["type"], spec.get("a"), spec.get("b"), None if n is None else float(Fraction(str(n))))
        kind, label = spec["type"], ""
    shape = spec.get("shape")
    if shape not in SHAPES:
        raise ValueError("降雨波形（rear 後方集中型、center 中央集中型）を指定してください。")
    label += f"{formula_text(formula, kind)}、{SHAPES[shape]}{hours}時間、{tc}分ごと{round_note(digits)}"
    storm = planned_storm(formula, shape, hours, tc, digits)
    return Rain(spec, label, storm, rain_table(storm, tc), tc)


def make_observed_rain(study_dir: Path, spec: dict, tc: int = DEFAULT_TC_MIN) -> Rain:
    """実測降雨（studies/_jma/ の10分値・時間値）を、tc ごとの降雨強度にまとめる（貯留追跡計算の流入と同じまとめ方）。
    spec は study.json の observed_rain と同じ（station、series、start、end）。"""
    if isinstance(tc, bool) or not isinstance(tc, int) or tc <= 0 or tc % 10:
        raise ValueError(f"区間 tc は10の倍数の分で指定してください（{tc}）。")
    obs = load_observed(JmaStore(Path(study_dir).resolve().parent), spec)
    storm = observed_storm(obs, tc)
    label = f"降雨記録 {obs.label}（studies/_jma/ の{OBSERVED_SERIES[spec['series']]}）、{tc}分ごと"
    return Rain(dict(spec) | {"tc": tc}, label, storm, rain_table(storm, tc, obs.start), tc, obs.start,
                gap_notes(obs.missing, obs.absent))


def saved_rain_name(source: str) -> str:
    """降雨のページの「保存する名前」の初期値（source は observed か planned）。「保存して出力」の rain.csv と重ならない名前。"""
    return "observed_rain" if source == "observed" else "planned_rain"


def saved_rain_path(study_dir: Path, name: str) -> Path:
    """降雨のページの「out/ に保存」の保存先 out/<名前>.csv（同じ名前の .png・.json も書く）。名前が空か使えない文字があるとき、
    貯留追跡計算の出力（「保存して出力」の out/ のファイル）と同じ名前になるときは ValueError（どちらかが黙って上書きされるため）。"""
    name = (name or "").strip()
    if not name or any(ch in name for ch in '\\/:*?"<>|'):
        raise ValueError('保存する名前が空か、使えない文字（\\ / : * ? " < > |）があります。別の名前にしてください。')
    reserved = {n.lower() for n in output_names()}          # Windows ではファイル名の大文字と小文字を区別しない
    clash = [f"{name}{ext}" for ext in (".csv", ".png", ".json") if f"{name}{ext}".lower() in reserved]
    if clash:
        raise ValueError(f"保存する名前「{name}」では、「保存して出力」の結果のファイル（{'、'.join(clash)}）と同じ名前になり、"
                         "どちらかが上書きされます。別の名前にしてください。")
    return Path(study_dir) / "out" / f"{name}.csv"


def write_rain(r: Rain, csv_path: Path, conditions_json: bool = False) -> list[Path]:
    """計画降雨の CSV（utf-8-sig、CRLF）と、同じ名前の PNG を書く。conditions_json なら条件を同じ名前の .json にも書く。
    どれかを出力できなければ（グラフを出力できないときなど）、一式をすべて削除する（all_or_nothing）。"""
    from ..adapters.report import design_plots
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    files = [csv_path, csv_path.with_suffix(".png")] + ([csv_path.with_suffix(".json")] if conditions_json else [])
    with LOCK, all_or_nothing(files):
        r.table.to_csv(csv_path, index=False, encoding="utf-8-sig", lineterminator="\r\n")
        design_plots.hyetograph(r.storm, r.tc, files[1], r.label)
        if conditions_json:
            files[2].write_bytes((json.dumps(r.spec, ensure_ascii=False, indent=2) + "\n").replace("\n", "\r\n").encode("utf-8"))
    return files
