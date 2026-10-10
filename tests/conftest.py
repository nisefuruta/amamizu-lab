"""試験の共通の設定。

グラフ（PNG）と PDF を出力する試験には needs_japanese_font の印を付ける。日本語のフォントがない環境では、その試験をスキップし、
理由（フォントがないこと）を試験の結果に表示する。フォントのインストールは README の「日本語のフォント」にある。
環境変数 AMAMIZU_REQUIRE_JP_FONT が 1 のとき（CI）は、スキップせずに試験を始めない（フォントのインストールに失敗したまま、
グラフと PDF の試験が通ったように見えないようにする）。
"""
import os
from pathlib import Path

import pytest

REPO_GUIDELINES = Path(__file__).parents[1] / "studies" / "_guideline"
REPO_JMA = Path(__file__).parents[1] / "studies" / "_jma"


@pytest.fixture(autouse=True)
def guidelines_for_copied_studies(monkeypatch):
    """試験が作る検討のフォルダ（tmp_path、tests/fixtures/design_cases）には _guideline/ がない。
    そのときは、リポジトリの studies/_guideline/ の手引の定数を使う（_guideline/ を作った試験は、そのフォルダを使う）。"""
    from amamizu.adapters.guideline import json_guideline
    real = json_guideline.guideline_dir
    monkeypatch.setattr(json_guideline, "guideline_dir",
                        lambda studies: real(studies) if real(studies).is_dir() else REPO_GUIDELINES)


@pytest.fixture(autouse=True)
def observations_for_copied_studies(monkeypatch):
    """試験が作る検討のフォルダには _jma/ がない。そのときは、リポジトリの studies/_jma/ の観測値を読む
    （_jma/ を作った試験は、そのフォルダを使う）。リポジトリの studies/_jma/ に書こうとしたら、試験を失敗にする。"""
    from amamizu.adapters import jma_store
    real = jma_store.store_dir
    monkeypatch.setattr(jma_store, "store_dir", lambda studies: real(studies) if real(studies).is_dir() else REPO_JMA)
    write = jma_store._write_bytes

    def guarded(path, data):
        if Path(path).resolve().is_relative_to(REPO_JMA.resolve()):
            raise AssertionError("試験がリポジトリの studies/_jma/ に書こうとしました。試験の studies/ に _jma/config.json を作ってください。")
        write(path, data)
    monkeypatch.setattr(jma_store, "_write_bytes", guarded)


def pytest_collection_modifyitems(config, items):
    marked = [item for item in items if item.get_closest_marker("needs_japanese_font")]
    if not marked:
        return
    from amamizu.adapters.report.design_plots import GraphError, japanese_font
    try:
        japanese_font()
        return
    except GraphError as e:
        if os.environ.get("AMAMIZU_REQUIRE_JP_FONT") == "1":
            raise pytest.UsageError(f"AMAMIZU_REQUIRE_JP_FONT=1 ですが、{e}") from None
        skip = pytest.mark.skip(reason="日本語のフォントがないため、グラフと PDF を出力する試験をスキップする（README の「日本語のフォント」）")
    for item in marked:
        item.add_marker(skip)
