"""貯留追跡計算（kind: design）の流入に実測降雨（studies/_jma/ の10分値・時間値）を使う試験。

例は studies/_jma/ の館山 2026-02-23〜25 の10分値（2/24 05:50〜06:40 が欠測）。
"""
import datetime
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from amamizu.adapters.design_inputs import load_observed, observed_from_frame
from amamizu.adapters.jma_store import JmaStore
from amamizu.app.design_runner import ConfigError, missing_text, prepare, trial
from amamizu.domain import rainfall as rf
from amamizu.domain.runoff import rational_inflow
from amamizu.usecases.design_study import Catchment, observed_inflow, rain_table

CASE = Path(__file__).parent / "fixtures" / "design_cases" / "observed_10min"
STORE = JmaStore(Path(__file__).parents[1] / "studies")
SPEC = {"station": "館山", "series": "tenmin", "start": "2026-02-23", "end": "2026-02-25"}


def tenmin() -> pd.DataFrame:
    return STORE.observed("館山", "tenmin", "2026-02-23", "2026-02-25")


def study(tmp_path, rows: pd.DataFrame | None = None, **inflow) -> Path:
    """例の検討のコピー。rows を渡すと、試験の studies/_jma/ に館山の10分値としてその行だけを置く（渡さなければリポジトリの値を読む）。"""
    d = tmp_path / "s"
    shutil.copytree(CASE, d)
    conf = json.loads((d / "study.json").read_text(encoding="utf-8"))
    conf.update(inflow)
    (d / "study.json").write_text(json.dumps(conf, ensure_ascii=False), encoding="utf-8")
    if rows is not None:
        (tmp_path / "_jma").mkdir()
        (tmp_path / "_jma" / "config.json").write_text('{"refresh_months": 1}', encoding="utf-8")
        JmaStore(tmp_path).put("tenmin", {"地点": "館山", "block_no": "47672", "pref": "千葉県"},
                               rows.drop(columns=["地点", "block_no"]), datetime.date(2026, 9, 22), "試験")
    return d


def hourly_frame(with_station=True) -> pd.DataFrame:
    """時間値（jma-rainfall hourly と amamizu fetch の出力の形。with_station=False は地点・block_no の列のない手作りのファイル）。
    2日、2日目の3時は欠測、4時は空欄。"""
    rows = []
    for day in ("2026-08-01", "2026-08-02"):
        for h in range(1, 25):
            v, q = (float(h % 5), "normal") if day == "2026-08-02" else (0.0, "none")
            if day == "2026-08-02" and h == 3:
                v, q = None, "missing"
            if day == "2026-08-02" and h == 4:
                v, q = None, "blank"
            rows.append({"地点": "千葉", "block_no": "47682", "日付": day, "時": h, "降水量_mm": v, "品質": q})
    df = pd.DataFrame(rows)
    return df if with_station else df.drop(columns=["地点", "block_no"])


def test_same_inflow_as_storage_study_and_volume_is_kept():
    """貯留施設の検証と同じ経路（10分値 → tc ごとの強度 → 合理式連続モデル）で、流入量の総量は f×雨量×面積。"""
    obs = load_observed(STORE, SPEC)
    t = tenmin()
    mm10 = pd.to_numeric(t["降水量_mm"], errors="coerce").fillna(0.0).to_numpy()
    cs = [Catchment("舗装", 0.6, 0.9), Catchment("緑地", 0.4, 0.3)]
    for tc in (10, 20, 30, 60):
        inflow = observed_inflow(obs, cs, tc)
        ref = sum(rational_inflow(rf.blocks_from_10min(mm10 * 6, tc), c.f, c.area_ha) for c in cs)
        assert np.array_equal(inflow.nodes, ref)
        volume = inflow.nodes.sum() * tc * 60                  # 三角形の和＝節点の値×tc の和
        assert volume == pytest.approx(sum(c.f * c.area_ha for c in cs) * mm10.sum() * 10, rel=1e-12)
        assert inflow.start == datetime.datetime(2026, 2, 23)


def test_tc_must_be_multiple_of_10():
    obs = load_observed(STORE, SPEC)
    with pytest.raises(ValueError, match="10の倍数"):
        observed_inflow(obs, [Catchment("a", 1.0, 0.9)], 15)


def test_missing_is_zero_and_listed():
    obs = load_observed(STORE, SPEC)
    assert obs.label == "館山 10分値 2026-02-23〜2026-02-25" and obs.step_min == 10 and len(obs.mm) == 3 * 144
    assert obs.missing == ((datetime.datetime(2026, 2, 24, 5, 50), datetime.datetime(2026, 2, 24, 6, 40)),)
    assert missing_text(obs.missing, 10) == "2026-02-24 05:50〜06:40"
    assert obs.mm[(24 * 60 + 5 * 60 + 50) // 10: (24 * 60 + 6 * 60 + 40) // 10].tolist() == [0.0] * 5


def test_period():
    obs = observed_from_frame(tenmin(), {"start": "2026-02-24", "end": "2026-02-24"})
    assert obs.start == datetime.datetime(2026, 2, 24) and len(obs.mm) == 144
    with pytest.raises(ValueError, match="行がありません.*2026-02-23〜2026-02-25"):
        observed_from_frame(tenmin(), {"start": "2026-03-01", "end": "2026-03-02"})
    with pytest.raises(ValueError, match="日付"):
        observed_from_frame(tenmin(), {"start": "2026/02/24"})
    with pytest.raises(ValueError, match="より後"):
        observed_from_frame(tenmin(), {"start": "2026-02-25", "end": "2026-02-24"})


def test_rows_not_in_table_are_zero_and_listed():
    """表にない区間（取得していない日など）は、欠測と同じく 0 mm として計算し、その時間を記録する。"""
    t = tenmin()
    full = observed_from_frame(t, {})
    gap = observed_from_frame(t.drop(index=list(range(144 + 72, 144 + 78))), {})     # 2/24 12:00〜13:00 の6コマ
    assert len(gap.mm) == len(full.mm) and gap.missing == full.missing
    assert gap.absent == ((datetime.datetime(2026, 2, 24, 12), datetime.datetime(2026, 2, 24, 13)),)
    assert gap.mm[144 + 72: 144 + 78].tolist() == [0.0] * 6
    assert np.array_equal(np.delete(gap.mm, range(144 + 72, 144 + 78)), np.delete(full.mm, range(144 + 72, 144 + 78)))
    day = observed_from_frame(t[t["日付"] != "2026-02-24"], {})                        # 1日まるごとない
    assert day.absent == ((datetime.datetime(2026, 2, 24), datetime.datetime(2026, 2, 25)),) and len(day.mm) == 3 * 144
    assert full.absent == ()


def test_duplicate_and_off_grid_are_errors():
    """同じ区間が2回ある、区切りに合わない時刻は、どの値を使うか決められないのでエラー。"""
    t = tenmin()
    with pytest.raises(ValueError, match="2回"):
        observed_from_frame(pd.concat([t, t.iloc[[10]]]), {})
    odd = t.copy()
    odd.loc[5, "時分"] = "00:55"
    with pytest.raises(ValueError, match="区切りになっていません"):
        observed_from_frame(odd, {})


def test_columns_are_found_by_name():
    with pytest.raises(KeyError, match="時分.*時"):
        observed_from_frame(pd.DataFrame({"日付": ["2026-01-01"], "降水量_mm": [1.0]}), {})
    t = tenmin()
    t = t[["品質", "降水量_mm", "時分", "日付", "地点"]]               # 列の順が違っても同じ
    assert np.array_equal(observed_from_frame(t, {}).mm, load_observed(STORE, SPEC).mm)


@pytest.mark.parametrize("with_station", [True, False])
def test_hourly_is_spread_over_10min(with_station):
    """時間値は1時間の雨量を10分ずつ均等に配分する（x mm の1時間 → 6区間とも x mm/hr）。欠測・空欄は0mm。"""
    obs = observed_from_frame(hourly_frame(with_station), {})
    assert obs.step_min == 60 and obs.start == datetime.datetime(2026, 8, 1)
    assert obs.missing == ((datetime.datetime(2026, 8, 2, 2), datetime.datetime(2026, 8, 2, 4)),)
    r = obs.mmhr_10min()
    assert len(r) == 48 * 6

    def hour(h):                                   # 2日目の h 時の区間（h-1 時〜h 時）
        return r[(24 + h - 1) * 6: (24 + h) * 6].tolist()
    assert hour(1) == [1.0] * 6 and hour(2) == [2.0] * 6 and hour(6) == [1.0] * 6
    assert hour(3) == [0.0] * 6 and hour(4) == [0.0] * 6      # 欠測・空欄
    inflow = observed_inflow(obs, [Catchment("a", 2.0, 0.5)], 60)
    assert inflow.nodes.sum() * 3600 == pytest.approx(0.5 * 2.0 * np.nansum(obs.mm) * 10, rel=1e-12)


def test_study_with_observed_rain(tmp_path):
    r = trial(json.loads((CASE / "study.json").read_text(encoding="utf-8")), CASE)
    assert r.prepared.inflow_rows[0] == ("実測降雨", "館山 10分値 2026-02-23〜2026-02-25")
    assert "| 欠測の時間（0 mm として計算） | 2026-02-24 05:50〜06:40 |" in r.lines
    assert r.notices[0] == "実測降雨の欠測の時間を 0 mm として計算しました: 2026-02-24 05:50〜06:40"
    assert not any("取得していない時間" in line for line in r.lines)
    assert any("2026-02-25 03:50" in line for line in r.lines if line.startswith("| 最大流入量"))


def test_long_gaps_are_summarized():
    spans = tuple((datetime.datetime(2026, 1, d), datetime.datetime(2026, 1, d, 2)) for d in range(1, 13))
    text = missing_text(spans, 10)
    assert text.count("〜") == 10 and text.endswith(" ほか 2 か所（合計 24 時間）")


def test_study_needs_catchments_and_one_inflow(tmp_path):
    d = study(tmp_path)
    conf = json.loads((d / "study.json").read_text(encoding="utf-8"))
    with pytest.raises(ConfigError, match="区域"):
        prepare({k: v for k, v in conf.items() if k != "catchments"}, d)
    with pytest.raises(ConfigError, match="どれか一つ"):
        prepare(conf | {"inflow_csv": {"file": "input/x.csv"}}, d)
    with pytest.raises(ConfigError, match=r"実測降雨の値の種類・開始日を選んでください（取得済みの期間の一覧から選ぶか、日付を入力します）。"
                                          r"study.json では observed_rain の series・start に書きます。"):
        prepare(conf | {"observed_rain": {"station": "館山", "end": "2026-02-25"}}, d)
    with pytest.raises(ConfigError, match="series）の「daily」は扱えません"):
        prepare(conf | {"observed_rain": SPEC | {"series": "daily"}}, d)
    with pytest.raises(ConfigError, match="地点「存在しない地点」の観測値がありません。amamizu fetch"):
        prepare(conf | {"observed_rain": SPEC | {"station": "存在しない地点"}}, d)
    with pytest.raises(ConfigError, match="館山の10分値の 2030-01-01〜2030-01-02 の値がありません。先に amamizu fetch"):
        prepare(conf | {"observed_rain": SPEC | {"start": "2030-01-01", "end": "2030-01-02"}}, d)
    with pytest.raises(ConfigError, match="--stations 館山 --dates-file"):                # 時間値は取得する日の一覧で取得する
        prepare(conf | {"observed_rain": SPEC | {"series": "hourly", "start": "2030-01-01", "end": "2030-01-02"}}, d)
    with pytest.raises(ConfigError, match="開始日（observed_rain.start）2026-02-25 が終了日（end）2026-02-23 より後"):
        prepare(conf | {"observed_rain": SPEC | {"start": "2026-02-25", "end": "2026-02-23"}}, d)
    with pytest.raises(ConfigError, match="「2026-2-23」を日付として読めません"):         # 取得していないと誤って示さない
        prepare(conf | {"observed_rain": SPEC | {"start": "2026-2-23"}}, d)


def test_rain_table_with_dates():
    t = rain_table([6.0, 12.0], 10, datetime.datetime(2026, 2, 24, 23, 50))
    assert list(t.columns)[0] == "区間の日時"
    assert t["区間の日時"].tolist() == ["2026-02-24 23:50〜00:00", "2026-02-25 00:00〜00:10"]
    assert list(rain_table([6.0], 10).columns) == ["区間", "時", "分", "降雨強度(mm/h)", "雨量(mm)"]


def test_study_with_a_day_not_fetched(tmp_path):
    """取得していない日があっても計算し、その時間を要約と注意に書く（時間値を候補の日だけ取得した地点など）。"""
    t = tenmin()
    d = study(tmp_path, t[t["日付"] != "2026-02-24"])
    r = trial(json.loads((d / "study.json").read_text(encoding="utf-8")), d)
    assert "| 取得していない時間（0 mm として計算） | 2026-02-24 00:00〜2026-02-25 00:00 |" in r.lines
    assert "実測降雨の取得していない時間を 0 mm として計算しました: 2026-02-24 00:00〜2026-02-25 00:00" in r.notices
    assert not any("欠測の時間" in line for line in r.lines)          # 欠測があった日がない


def test_study_shows_observations_changed_after_the_output(tmp_path, capsys):
    """amamizu study は、前の出力の後に変わった、使った観測値（studies/_jma/）を示す（出力し直した後は示さない）。"""
    from amamizu.cli import main
    rows = tenmin()
    d = study(tmp_path, rows)
    main(["study", str(d)])
    assert "使った観測値" not in capsys.readouterr().out
    fixed = rows[rows["時分"] == "05:40"].iloc[:1].drop(columns=["地点", "block_no"]).assign(降水量_mm=9.5, 品質="normal")
    JmaStore(tmp_path).put("tenmin", {"地点": "館山", "block_no": "47672", "pref": "千葉県"}, fixed, datetime.date.today(), "試験",   # 前の出力の日以後に取得した訂正（固定の日付にすると、翌日から訂正を示さない）
                           accept=True)
    main(["study", str(d)])
    out = capsys.readouterr().out
    assert "前の出力の後に、使った観測値（studies/_jma/）が変わりました" in out
    assert f"訂正 {fixed['日付'].iat[0]} 05:40:" in out
    main(["study", str(d)])
    assert "使った観測値" not in capsys.readouterr().out
