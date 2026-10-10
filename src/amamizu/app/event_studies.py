"""貯留施設の検証（kind: storage）の降雨（events）を、地点ごとの設計の検討（kind: design）にする。

1つの降雨の1地点を1つの検討にし、studies/<年-月>_<降雨>_<地点>/ に study.json を置く。
実測降雨は、studies/_jma/ の10分値を地点・期間で指定する（observed_rain の station・series・start・end）。
池とオリフィスは、地点が属する地区のモデル施設（StorageStudy.facility。安全側に丸めた寸法）にする。
貯留施設の検証の表と同じ施設なので、表と同じ結果になる。
"""
import datetime
from pathlib import Path

import numpy as np

from ..adapters.jma_store import JmaStore
from ..adapters.guideline.json_guideline import JsonGuideline, relocated
from ..adapters.study_config import check_study, dump_study, read_study
from ..usecases.storage_study import ModelConditions, StorageStudy

def event_study_dir(studies: Path, storage_dir: Path, event_name: str, station: str) -> Path:
    """studies/<年-月>_<降雨>_<地点>。年-月は貯留施設の検証のフォルダ名の先頭。空白は _ にする（コマンドで扱いやすく）。"""
    name = f"{storage_dir.name[:7]}_{event_name}_{station}"
    return Path(studies) / "_".join(name.split())


def create_event_studies(storage_dir: Path, studies: Path) -> list[Path]:
    """貯留施設の検証の events の降雨ごと・地点ごとに、設計の検討を作る。同じ名前のフォルダがあればエラー。"""
    storage_dir = Path(storage_dir)
    conf = read_study(storage_dir)
    if conf.get("kind") != "storage":
        raise ValueError(f"{storage_dir} は貯留施設の検証（study.json の kind が storage）ではありません。"
                         "amamizu event-studies には、貯留施設の検証のフォルダを指定してください。")
    check_study(conf)
    g = JsonGuideline.required(conf, storage_dir)
    records = JmaStore(storage_dir.resolve().parent)
    study = StorageStudy(g, records, ModelConditions(**conf["conditions"]))
    c = study.cond
    plan = []
    for ev in conf["events"]:
        if not ev["stations"]:
            raise ValueError(f"{ev['name']}（study.json の events）の stations に、検討を作る地点がありません。地点名を書いてください。")
        for station in ev["stations"]:
            if records.tenmin(station, ev["start"], ev["end"]) is None:
                raise ValueError(f"{station} の {ev['start']}〜{ev['end']} の10分値がそろっていません。"
                                 f"amamizu fetch <検討フォルダ> tenmin --stations {station} --start {ev['start']} --end {ev['end']} で取得してください。")
            d = event_study_dir(studies, storage_dir, ev["name"], station)
            if d.exists() or any(d == p[2] for p in plan):     # 同じ名前の降雨が2つあると、同じフォルダを2回作ることになる
                raise ValueError(f"作る検討と同じ名前のフォルダがあります: {d}。作り直すなら、このフォルダを削除してから実行してください。"
                                 "study.json の events に同じ name の降雨が2つあるときは、name を変えてください。")
            plan.append((ev, station, d))
    now = datetime.datetime.now().replace(microsecond=0)
    out = []
    for i, (ev, station, d) in enumerate(plan):
        district = g.district_of(station)
        pond = study.facility(district).facility.pond
        d.mkdir(parents=True)
        # どの地区のモデル施設かを題名で示す（区域の呼び方は手引の district_term）
        new = {"kind": "design", "title": f"{ev['name']} {station}（{district}{g.district_term()}）",
               "guideline": relocated(conf["guideline"], storage_dir, d, ()),     # .json の相対パスは作る検討から見たパスに直す（コピーするファイルはない）
               # 一覧は作成日の新しい順なので、後の検討ほど1秒ずつ古くし、降雨・地点の順に並べる
               "created": (now - datetime.timedelta(seconds=i)).isoformat(),
               "tc_min": c.tc_min, "allowable_q_m3s": float(np.round(study.allowable_q(district), 6)),
               "catchments": [{"name": k.name, "area_ha": k.area_ha, "f": k.f} for k in study.catchments()],
               "observed_rain": {"station": station, "series": "tenmin", "start": ev["start"], "end": ev["end"]},
               "pond": {"prismatic": {"area_m2": pond.area, "hwl_m": pond.hwl}},
               "orifice": study.outlet_spec(district)}
        (d / "study.json").write_bytes(dump_study(new))
        out.append(d)
    return out
