"""継続時間ごとの雨量を、気象庁の年ごとの値「N時間降水量」（公表値）と照合する。

公表値は1976年以降の年最大（1・3・6・12・24・48・72時間）。2002年以前は毎正時で区切った値（1mm単位）、
2003年以降は10分間隔で求めた値。本検討の計算で計画の確率年（1/50 など）を超えた降雨の地点・年の集合が、
公表値で同じ確率年を超えた地点・年の集合と一致するかを継続時間ごとに確かめる。
"""
import re

import pandas as pd

from .ports import Guideline

PUBLISHED_FROM = 1976      # N時間降水量の公表値がある最初の年


def _num(x):
    m = re.search(r"[0-9.]+", str(x))
    return float(m.group()) if m else None


def exceedance_sets(guideline: Guideline, published: pd.DataFrame, events: pd.DataFrame,
                    return_period: int, durations: tuple[int, ...]) -> dict:
    """継続時間ごとに {公表値で超過: set, 計算で超過: set, 照合できない: set, 一致: bool}。
    超過は、どちらも雨量 mm を丸めない計画の雨量と比べて決める（表の比の列は小数3桁に丸めてあり、
    0.9996 が 1.000 になるので判定に使わない）。公表値が数として読めない地点・年（/// や空欄）と、計算で超過したのに
    公表値の行がない地点・年は「照合できない」に入れ、一致はそれ以外で比べる（公表値がないのを「超過していない」として黙って比べない）。"""
    out = {}
    for h in durations:
        def plan(station: str) -> float:
            return guideline.formula(guideline.district_of(station), return_period).depth(60 * h)
        a, unknown = set(), set()
        for _, p in published.iterrows():
            v = _num(p[f"最大{h}時間/(mm)"])
            if v is None:
                unknown.add((p["地点"], int(p["年"])))
            elif v >= plan(p["地点"]):
                a.add((p["地点"], int(p["年"])))
        b = {(r["地点"], int(r["開始"][:4])) for _, r in events.iterrows()
             if int(r["開始"][:4]) >= PUBLISHED_FROM and r[f"{h}h_mm"] >= plan(r["地点"])}
        rows = {(p["地点"], int(p["年"])) for _, p in published.iterrows()}
        unknown |= {k for k in b if k not in rows}
        out[h] = {"公表値で超過": a, "計算で超過": b, "照合できない": unknown, "一致": a == b - unknown}
    return out
