"""設計の検討（kind: design）の出力が、固定した正解（tests/golden/design/）とバイト単位で一致すること。

確認済みの出力を正解にしている（最初は作り替えの前の出力）。入力は tests/fixtures/design_cases/。
ポンプ・2段のオリフィス・実測降雨・5方式の強度式などは、手引の照合（amamizu verify、解説の計算例）では確かめられないので、ここで意図しない変化を見つける。
"""
import shutil
from pathlib import Path

import pytest

from amamizu.cli import main

pytestmark = pytest.mark.needs_japanese_font     # グラフと PDF を出力する（日本語のフォントが要る）

ROOT = Path(__file__).parent
CASES = sorted(p.name for p in (ROOT / "fixtures" / "design_cases").iterdir())


@pytest.mark.parametrize("case", CASES)
def test_design_outputs_match_golden(case, tmp_path):
    study = tmp_path / case
    shutil.copytree(ROOT / "fixtures" / "design_cases" / case, study)
    main(["study", str(study)])
    gold = ROOT / "golden" / "design" / case
    names = sorted(p.name for p in gold.iterdir())
    # out/ の CSV と要約は、どれも golden にある（golden にあるファイルだけを比べて、ほかの出力を確かめずに通さない）
    assert names == sorted(p.name for p in (study / "out").iterdir() if p.suffix in (".csv", ".md"))
    assert {"route.csv", "design_summary.md", "inflow.csv"} <= set(names)
    for name in names:
        assert (study / "out" / name).read_bytes() == (gold / name).read_bytes(), f"{case}/{name}"
    for png in ("hydrograph.png", "water_level.png", "pond_curves.png"):
        assert (study / "out" / png).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
