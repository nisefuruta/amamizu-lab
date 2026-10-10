"""設計の検討の要約（design_summary.md）を PDF に清書する（out/design_summary.pdf）。

要約の中身は Markdown（見出し、項目と値の表、グラフの画像の参照）だけで組み立て、PDF はそれを A4 に並べる
（同じ中身を2か所で組み立てない）。日本語のフォントはグラフと同じもの（design_plots.japanese_font）を埋め込む。
WebGUI の「保存して出力」と amamizu study の out/ をバイト単位で一致させるため、作成日時などを入れない（invariant）。
"""
import re
from pathlib import Path
from xml.sax.saxutils import escape

from matplotlib import font_manager
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from .design_plots import japanese_font

FONT = "amamizu-jp"
IMAGE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)$")
HEAD_ROWS = 3           # 表の見出しと同じページに置く、表の最初の行の数（列名の行を除く）


def _register_font() -> str:
    """グラフと同じ日本語のフォントのファイルを ReportLab に登録する（TTC は最初のフォント）。"""
    if FONT not in pdfmetrics.getRegisteredFontNames():
        path = font_manager.findfont(font_manager.FontProperties(family=japanese_font()), fallback_to_default=False)
        pdfmetrics.registerFont(TTFont(FONT, path, subfontIndex=0))
    return FONT


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def write_pdf(md_lines: list[str], out_dir: Path, pdf_path: Path) -> Path:
    """Markdown の行（要約）から PDF を書く。画像の参照は out_dir から見たファイル。"""
    font = _register_font()
    h1 = ParagraphStyle("h1", fontName=font, fontSize=14, leading=19, spaceAfter=6)
    h2 = ParagraphStyle("h2", fontName=font, fontSize=11.5, leading=16, spaceBefore=6, spaceAfter=4)
    body = ParagraphStyle("body", fontName=font, fontSize=9, leading=13, alignment=TA_LEFT)
    cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=6.5, leading=8.5)          # 3列以上の表（時刻ごとの結果など）
    small_right = ParagraphStyle("small_right", parent=small, alignment=TA_RIGHT)
    small_head = ParagraphStyle("small_head", parent=small, alignment=TA_CENTER)            # 列名は中央
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, invariant=1,
                            title=md_lines[0].lstrip("# ").strip() if md_lines else "", author="amamizu")
    width = doc.width
    story, table = [], []

    def flush_table():
        if not table:
            return
        sep = [r for r in table if all(set(c) <= set("-: ") for c in r)]        # 区切りの行（| --- | ---: |）
        rows = [r for r in table if r not in sep]
        n = len(rows[0])
        if n == 2:                                          # 項目と値の表
            data = [[Paragraph(escape(c), cell) for c in r] for r in rows]
            widths = [width * 0.36, width * 0.64]
        else:                                               # 3列以上は等幅。本文の行は区切りの行の「---:」の列を右寄せ
            right = [c.endswith(":") for c in sep[0]] if sep else [False] * n
            # 列名の単位（「(min)」「(m3/s)」など）は改行して下に置く（列が狭いと単位の途中で折り返されるため）
            data = [[Paragraph(escape(c).replace("(", "<br/>(", 1), small_head) for c in rows[0]]] + [
                [Paragraph(escape(c), small_right if right[i] else small) for i, c in enumerate(r)] for r in rows[1:]]
            widths = [width / n] * n
        style = TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")),
                            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)])
        if story and isinstance(story[-1], Paragraph) and story[-1].style is h2:
            # 見出しだけがページの下に残らないよう、見出しと列名・最初の数行が入らなければ見出しから次のページにする
            head = story.pop()
            first = Table(data[:1 + HEAD_ROWS], colWidths=widths)
            first.setStyle(style)
            need = (h2.spaceBefore + head.wrap(width, doc.height)[1] + h2.spaceAfter
                    + first.wrap(width, doc.height)[1])
            story.extend([CondPageBreak(need), head])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(style)
        story.extend([t, Spacer(1, 4 * mm)])
        table.clear()

    for line in md_lines:
        s = line.rstrip()
        if s.startswith("|"):
            table.append(_cells(s))
            continue
        flush_table()
        if not s:
            continue
        m = IMAGE.match(s)
        if m:
            src = Path(out_dir) / m["src"]
            if not src.exists():
                raise FileNotFoundError(f"要約（design_summary.md）が参照するグラフがありません: {src}。もう一度出力してください"
                                        "（amamizu study、WebGUI では「保存して出力」）。")
            img = Image(str(src))
            # 幅に合わせて縮める（時間を横軸にする図は同じ幅なので同じ倍率になり、上下で横の位置がそろう）
            scale = min(1.0, width / img.drawWidth, doc.height * 0.9 / img.drawHeight)
            img.drawWidth, img.drawHeight = img.drawWidth * scale, img.drawHeight * scale
            story.append(KeepTogether([Paragraph(escape(m["alt"]), h2), img, Spacer(1, 3 * mm)]))
        elif s.startswith("## "):
            story.append(Paragraph(escape(s[3:]), h2))
        elif s.startswith("# "):
            story.append(Paragraph(escape(s[2:]), h1))
        else:
            story.append(Paragraph(escape(s), body))
    flush_table()
    doc.build(story)
    return Path(pdf_path)
