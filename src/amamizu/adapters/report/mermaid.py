"""Mermaid xychart の部品。

注意: 同じ文字列のX軸ラベルは1つのカテゴリに統合される。間引いたラベルはゼロ幅スペースの個数を変えて一意にする。
%%{init: ...}%% は Python の % 書式で組まない（%% が % になる）。
棒は、値0でも縦軸の余白（目盛りの文字の高さの半分）の分だけ、0より下に着色される（図の下端から描くため。
mermaid-js/mermaid の issue #6517・#7061、未マージの PR #7083）。棒は xychart_bar_init と xychart_bars で描く。
"""

MASK = "#ffffff"     # 0より下を隠す棒の色。背景と同じにする
WIDTH = 900          # 図の幅


def xychart_init(height: int, palette: str, background: str | None = None) -> str:
    bg = f', "backgroundColor": "{background}"' if background else ""
    return ('%%{init: {"xyChart": {"width": ' + str(WIDTH) + ', "height": ' + str(height) + '}, '
            '"themeVariables": {"xyChart": {"plotColorPalette": "' + palette + '"' + bg + '}}}}%%')


def xychart_bar_init(height: int, color: str) -> str:
    """棒1系列の図。xychart_bars の2本目（0より下を隠す棒）の色を背景と同じにする。"""
    return xychart_init(height, f"{color}, {MASK}", background=MASK)


def xychart_bars(values: list[str]) -> list[str]:
    """棒と、その後に重ねる値0の棒。値0の棒は0の目盛りから図の下端まで（はみ出す部分と同じ範囲）に描かれるので、
    背景と同じ色にして、どの棒も0より下を隠す（降っていない時間は着色しない）。縦軸の下限は0にすること。"""
    return ["    bar [" + ", ".join(values) + "]", "    bar [" + ", ".join("0" for _ in values) + "]"]


def xychart_labels(labels: list[str | None]) -> list[str]:
    """None の位置は表示しないラベル（ゼロ幅スペースを位置ごとに違う個数だけ並べる）にする。"""
    return [lab if lab is not None else "​" * (j + 1) for j, lab in enumerate(labels)]
