"""両方の検討のレポートが、新しい構成からバイト単位で同じに生成できること。"""
import shutil
from pathlib import Path

from amamizu.cli import main

ROOT = Path(__file__).parents[1]
GOLDEN = ROOT / "tests" / "golden"
STORAGE_GOLDEN = {"event_20260813.tsv": "chiba_event.tsv", "partial_year.tsv": "year2026.tsv"}   # out/ の名前 → golden の名前（違うものだけ）


def test_return_period_report(tmp_path):
    out = tmp_path / "return_period.md"
    main(["study", str(ROOT / "studies" / "2026-09_降雨強度の検証"), "--report", str(out)])
    assert out.read_bytes() == (GOLDEN / "return_period_report.md").read_bytes()


def test_storage_report(tmp_path):
    """--report を省略すると、検討フォルダの out/report.md に出力する。
    検討のコピーで実行する（リポジトリの検討フォルダの out/ を書き換えない）。"""
    study = tmp_path / "study"
    shutil.copytree(ROOT / "studies" / "2026-09_貯留施設の検証", study, ignore=shutil.ignore_patterns("out"))
    main(["study", str(study)])
    assert (study / "out" / "report.md").read_bytes() == (GOLDEN / "storage_report.md").read_bytes()
    tables = sorted(p.name for p in (study / "out").glob("*.tsv"))
    assert tables == ["daily_missing.tsv", "event_20260813.tsv", "event_20260920.tsv", "long_events.tsv", "long_events_final.tsv",
                      "partial_year.tsv", "screening.tsv", "tenmin_check.tsv"]
    for name in tables:                         # 表のファイルそのもの（列・桁・改行）も確かめる
        assert (study / "out" / name).read_bytes() == (GOLDEN / STORAGE_GOLDEN.get(name, name)).read_bytes(), name


def test_storage_study_writes_nothing_when_it_stops(tmp_path, monkeypatch):
    """計算が途中で止まったら、out/ を書き換えない（新しい表と、前の出力の表・レポート・conditions.json を混ぜない）。"""
    import pytest

    from amamizu.usecases.storage_study import StorageStudy
    study = tmp_path / "study"
    shutil.copytree(ROOT / "studies" / "2026-09_貯留施設の検証", study, ignore=shutil.ignore_patterns("out"))
    out = study / "out"
    out.mkdir()
    before = {"event_20260813.tsv": b"old", "report.md": b"old", "conditions.json": b'{"files": ["event_20260813.tsv"]}'}
    for name, data in before.items():
        (out / name).write_bytes(data)

    def stop(*args, **kwargs):
        raise ValueError("時間値がそろっていません")
    monkeypatch.setattr(StorageStudy, "long_record", stop)
    with pytest.raises(ValueError, match="時間値がそろっていません"):
        main(["study", str(study)])
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
