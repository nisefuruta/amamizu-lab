"""依存の向き（Clean Architecture）: 内側の層は外側の層を import しない。

domain   : 標準ライブラリと numpy だけ（pandas・ファイル・通信・表示を知らない）
usecases : domain と自分の ports だけ（adapters・app・cli・gui を知らない。pandas は表の受け渡しに使ってよい）
adapters : usecases と domain を使う（app・cli・gui を知らない）
app      : 組み立ての層。study.json と入力から使用ケースの入力を作り、計算し、out/ に書く（cli・gui を知らない）
cli・gui : 外側の入口。app を使う。gui は cli を知らない（共通の処理は app に置く）
"""
import ast
from pathlib import Path

PKG = Path(__file__).parents[1] / "src" / "amamizu"


def imports(layer: str) -> dict[str, set[str]]:
    """層のファイル（PKG からの相対パス）ごとの、import の名前。from A import b は「A」と「A.b」の両方を入れる
    （from .. import cli のように、外側の層の名前が import する名前の側にある形も確かめる）。"""
    out = {}
    for p in (PKG / layer).rglob("*.py"):
        names = set()
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                base = ("." * node.level) + (node.module or "")
                names.add(base)
                names |= {f"{base}.{a.name}" if node.module else base + a.name for a in node.names}
        out[p.relative_to(PKG).as_posix()] = names
    assert out, f"{layer} の層のファイルがありません"
    return out


def test_imported_names_are_checked(tmp_path, monkeypatch):
    """import する名前の側にある層の名前（from .. import cli）と、同じ名前のファイル（__init__.py）を見落とさない。"""
    import sys
    (tmp_path / "usecases" / "sub").mkdir(parents=True)
    (tmp_path / "usecases" / "__init__.py").write_text("from .. import cli\n", encoding="utf-8")
    (tmp_path / "usecases" / "sub" / "__init__.py").write_text("from amamizu import gui\nfrom ..ports import Guideline\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "PKG", tmp_path)
    found = imports("usecases")
    assert set(found) == {"usecases/__init__.py", "usecases/sub/__init__.py"}
    assert "cli" in parts("..cli") and "..cli" in found["usecases/__init__.py"]
    assert "amamizu.gui" in found["usecases/sub/__init__.py"] and "..ports.Guideline" in found["usecases/sub/__init__.py"]


def parts(name: str) -> set[str]:
    """import の名前を区切った部分（..app.design_runner → {app, design_runner}）。"""
    return set(name.strip(".").split("."))


def test_domain_depends_on_nothing_outside():
    allowed = {"math", "dataclasses", "typing", "collections", "numpy", "decimal"}   # decimal は四捨五入（Excel の ROUND と同じ）、collections は型の注釈（collections.abc）に使う
    for f, names in imports("domain").items():
        for n in names:
            assert n.startswith(".") or n.split(".")[0] in allowed, (f, n)
            assert not n.startswith("..")    # domain の外（usecases・adapters）へ出ない


def test_usecases_do_not_know_outer_layers():
    for f, names in imports("usecases").items():
        for n in names:
            assert not parts(n) & {"adapters", "app", "cli", "gui"}, (f, n)


def test_adapters_do_not_know_app_cli_gui():
    for f, names in imports("adapters").items():
        for n in names:
            assert not parts(n) & {"app", "cli", "gui"}, (f, n)


def test_app_does_not_know_cli_or_gui():
    for f, names in imports("app").items():
        for n in names:
            assert not parts(n) & {"cli", "gui"}, (f, n)


def test_gui_does_not_know_cli():
    for f, names in imports("gui").items():
        for n in names:
            assert "cli" not in parts(n), (f, n)
