"""設計の検討（kind: design）のグラフ（PNG）。

1. hydrograph.png   降雨強度（上、下向きの棒）と流入量・放流量の時間変化（下）。許容放流量の線
2. water_level.png  水位の時間変化。オリフィスの敷高・洪水吐の敷高・満水位の線
3. pond_curves.png  池と放流施設の特性（水位と容量、水位と放流量）

文字が読めなくならないように、次を守る。
- 日本語のフォントがなければ出力しない。フォントにない文字（豆腐になる文字）が出たらエラーにする
- 凡例は図の外（下）に置き、線に重ねない。最大値は線の上に印を付け、値は凡例の文に書く（浮いた注記を置かない）
- 長いタイトルは折り返す。文字の大きさは 9pt 以上
"""
import contextlib
import re
import textwrap
import warnings
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from ...usecases.design_study import DesignResult, clock, hm  # noqa: E402

# PDF（design_pdf）にも同じフォントを埋め込むので、ReportLab が扱える TrueType のアウトラインのフォントに限る
# （Noto Sans CJK JP は PostScript のアウトラインで、ReportLab が埋め込めない）
JP_FONTS = ["Yu Gothic", "Meiryo", "MS Gothic", "IPAexGothic"]
GRAPH_NAMES = ("hydrograph.png", "water_level.png", "pond_curves.png")   # write_all が out/ に書く3枚（この順）
DPI = 150
# 時間を横軸にする図（降雨と流量、水位）は、幅と描画の領域の左右をそろえ、上下に並べて同じ時刻を読めるようにする
TIME_FIG_WIDTH = 8.0            # in
TIME_AXES_LR = (0.105, 0.975)   # 図の幅に対する描画の領域の左端・右端
TIME_YLABEL_X = -0.085          # 縦軸の名前の位置（描画の領域の幅に対する割合。目盛りの文字の幅によらずそろえる）
FONT_SIZE = 10
TITLE_WIDTH = 34          # タイトルを折り返す文字数（全角で数える目安）


class GraphError(RuntimeError):
    """グラフの文字が読めなくなるため出力しないときのエラー（日本語のフォントがない、フォントにない文字、凡例の重なりなど）。"""


def japanese_font() -> str:
    """使える日本語のフォント名。見つからなければ、文字化けした図を出力しないようエラーにする。"""
    for name in JP_FONTS:
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue
    raise GraphError(f"日本語のフォント（{', '.join(JP_FONTS)}）が見つかりません。グラフの文字が化けるため出力しません。"
                     "これらのフォントのどれかをインストールしてから、もう一度出力してください（README の「日本語のフォント」）。"
                     f"インストールした後も同じエラーになるときは、フォントの一覧のキャッシュ（{matplotlib.get_cachedir()} の fontlist-*.json）を"
                     "削除してください。")


@contextlib.contextmanager
def _readable_text(paths: list[Path]) -> Iterator[None]:
    """図の文字が読めることを保証して描く。日本語のフォントを設定し（なければ GraphError）、描画中の警告を捕まえて、
    フォントにない文字（豆腐になる文字）があれば paths の図を削除して GraphError にする。"""
    plt.rcParams["font.family"] = japanese_font()
    plt.rcParams["font.size"] = FONT_SIZE
    plt.rcParams["axes.unicode_minus"] = False           # 負号がフォントにないことがあるため
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    missing = [str(w.message) for w in caught if "missing from" in str(w.message)]
    if missing:
        for p in paths:
            p.unlink(missing_ok=True)
        raise GraphError(f"グラフに日本語のフォントにない文字があり、読めなくなるため出力しません（詳細: {missing[0]}）。"
                         "題名や区域・放流施設の名前から、特殊な記号や絵文字を除いてください。")


def _wrap(title: str) -> str:
    """長いタイトルを折り返す。語の途中で切らないよう、「、」の後、「（」の前、スペースの位置で区切り、1行 TITLE_WIDTH 字に詰める。
    区切りのない長い部分だけは、字数で切る。"""
    parts = re.findall(r"[^、（ ]+[、 ]?|（[^）]*）[、 ]?|[（ ]", title)
    lines, cur = [], ""
    for part in parts:
        if cur and len(cur) + len(part) > TITLE_WIDTH:
            lines.append(cur.rstrip())
            cur = ""
        cur += part
    if cur.strip():
        lines.append(cur.rstrip())
    out = [seg for line in lines for seg in (textwrap.wrap(line, TITLE_WIDTH) if len(line) > TITLE_WIDTH else [line])]
    return "\n".join(out) or title


def _when(res: DesignResult, minutes: float) -> str:
    """凡例の時刻。実測降雨のときは日時（経過時間は長いと読めないため）。"""
    return clock(res.inflow.start, minutes)


def _legend_below(ax, ncol: int = 2):
    """凡例を、あとで軸の下の外側に置く印を付ける（位置は _save で軸のラベルの下端を測って決める）。"""
    ax._amamizu_legend_ncol = ncol


def _lines(ax, res: DesignResult):
    """オリフィスの敷高・洪水吐の敷高・満水位の横線。"""
    many = len(res.orifices) > 1
    marks = [(o.sill, f"オリフィス{k if many else ''}の敷高", "#7a7a7a", ":") for k, o in enumerate(res.orifices, 1)]
    if res.spillway:
        marks.append((res.spillway.crest, "洪水吐の敷高", "#e08a00", "-."))
    marks.append((res.facility.pond.hwl, "満水位", "#d03b3b", "--"))
    for y, label, color, style in marks:
        ax.axhline(y, color=color, linestyle=style, linewidth=1, label=f"{label} {y:.3f} m")


def _time_axes(res: DesignResult, *axes):
    """時間を横軸にする軸をそろえる（横軸の範囲は追跡の時刻の範囲、縦軸の名前の位置は共通）。"""
    t = res.table["計算時間(min)"].to_numpy() / 60
    for ax in axes:
        ax.set_xlim(0, t[-1])
        ax.yaxis.set_label_coords(TIME_YLABEL_X, 0.5)


def _xlabel(res: DesignResult) -> str:
    """時間の軸の名前。実測降雨のときは起点の日時を書く。"""
    start = res.inflow.start
    return "時間 h" if start is None else f"時間 h（{start:%Y-%m-%d %H:%M} から）"


def hydrograph(res: DesignResult, path: Path, title: str):
    tc = res.tc_min
    s = res.summary
    t = res.table["計算時間(min)"].to_numpy() / 60
    rain = res.inflow.rain
    if rain is not None:
        fig, (ax_r, ax_q) = plt.subplots(2, 1, figsize=(TIME_FIG_WIDTH, 5.4), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 2]})
        left = np.arange(len(rain)) * tc / 60
        if len(rain) <= 2000:
            ax_r.bar(left, rain, width=tc / 60, align="edge", color="#2a78d6", edgecolor="#1a4f8f", linewidth=0.3)
        else:                                       # 長い期間の実測降雨は、棒の代わりに同じ形の階段を塗る（棒が多いと描けないため）
            edges = np.append(left, len(rain) * tc / 60)
            ax_r.stairs(rain, edges, fill=True, color="#2a78d6")
            ax_r.stairs(rain, edges, color="#1a4f8f", linewidth=0.8)
        ax_r.set_ylim(max(rain) * 1.15, 0)          # 下向きの棒
        ax_r.set_ylabel("降雨強度 mm/h")
        ax_r.grid(alpha=0.3)
        ax_r.set_title(_wrap(title))
    else:
        fig, ax_q = plt.subplots(figsize=(TIME_FIG_WIDTH, 4.5))
        ax_q.set_title(_wrap(title))
    if res.treated:
        ax_q.plot(t, res.table["流入量(m3/s)"], color="#d03b3b", linestyle=":", linewidth=1,
                  label=f"浸透前の流入量（最大 {s['浸透前の最大流入量_m3s']:.4f} m^3/s）")
        ax_q.plot(t, res.table["浸透後流入量(m3/s)"], color="#d03b3b",
                  label=f"浸透後の流入量（最大 {s['最大流入量_m3s']:.4f} m^3/s、{_when(res, s['最大流入量の時刻_min'])}）")
    else:
        ax_q.plot(t, res.table["流入量(m3/s)"], color="#d03b3b",
                  label=f"流入量（最大 {s['最大流入量_m3s']:.4f} m^3/s、{_when(res, s['最大流入量の時刻_min'])}）")
    ax_q.plot(t, res.table["放流量(m3/s)"], color="#1a4f8f",
              label=f"放流量（最大 {s['最大放流量_m3s']:.4f} m^3/s、{_when(res, s['最大放流量の時刻_min'])}）")
    ax_q.plot([s["最大放流量の時刻_min"] / 60], [s["最大放流量_m3s"]], "o", color="#1a4f8f", markersize=4)
    if res.allowable_q is not None:
        ax_q.axhline(res.allowable_q, color="#7a7a7a", linestyle="--", label=f"許容放流量 {res.allowable_q:g} m^3/s")
    ax_q.set_xlabel(_xlabel(res))
    ax_q.set_ylabel("流量 m^3/s")
    ax_q.set_ylim(bottom=0)
    ax_q.grid(alpha=0.3)
    _legend_below(ax_q, ncol=1)
    _time_axes(res, *fig.axes)
    _save(fig, path, time_axes=True)


def hyetograph(storm, tc_min: int, path: Path, title: str) -> Path:
    """計画降雨だけの図（amamizu rain）。区間ごとの降雨強度の棒。"""
    storm = np.asarray(storm, dtype=float)
    peak = int(storm.argmax())
    with _readable_text([path]):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        left = np.arange(len(storm)) * tc_min / 60
        ax.bar(left, storm, width=tc_min / 60, align="edge", color="#2a78d6", edgecolor="#1a4f8f", linewidth=0.3,
               label=f"降雨強度（最大 {storm[peak]:.1f} mm/h、{hm(peak * tc_min)}〜{hm((peak + 1) * tc_min)}）")
        ax.set_title(_wrap(title))
        ax.set_xlabel("時間 h")
        ax.set_ylabel("降雨強度 mm/h")
        ax.grid(alpha=0.3)
        _legend_below(ax, ncol=1)
        _save(fig, path)
    return path


def water_level(res: DesignResult, path: Path, title: str):
    s = res.summary
    t = res.table["計算時間(min)"].to_numpy() / 60
    fig, ax = plt.subplots(figsize=(TIME_FIG_WIDTH, 3.5))
    ax.plot(t, res.table["水位(m)"], color="#1a4f8f",
            label=f"水位（最大 {s['最大水位_m']:.3f} m、{_when(res, s['最大水位の時刻_min'])}）")
    ax.plot([s["最大水位の時刻_min"] / 60], [s["最大水位_m"]], "o", color="#1a4f8f", markersize=4)
    _lines(ax, res)
    ax.set_title(_wrap(title))
    ax.set_xlabel(_xlabel(res))
    ax.set_ylabel("水位 m")
    ax.grid(alpha=0.3)
    _legend_below(ax)
    _time_axes(res, ax)
    _save(fig, path, time_axes=True)


def pond_curves(res: DesignResult, path: Path, title: str):
    pond, outlet = res.facility.pond, res.facility.outlet
    s = res.summary
    h = np.linspace(pond.bottom, pond.hwl, 400)
    fig, (ax_v, ax_q) = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    ax_v.plot(pond.volume(h), h, color="#1a4f8f", label="容量")
    ax_v.set_xlabel("容量 m^3")
    ax_v.set_ylabel("水位 m")
    ax_v.set_title("水位と容量")
    ax_q.plot([outlet.discharge(x) for x in h], h, color="#1a4f8f",
              label="放流量（" + "＋".join(["オリフィス"] + (["洪水吐"] if res.spillway else []) + (["ポンプ"] if res.pump else [])) + "）")
    ax_q.set_xlabel("放流量 m^3/s")
    ax_q.set_title("水位と放流量")
    for ax in (ax_v, ax_q):
        _lines(ax, res)
        ax.axhline(s["最大水位_m"], color="#2a78d6", linewidth=0.8, label=f"最大水位 {s['最大水位_m']:.3f} m")
        ax.grid(alpha=0.3)
        _legend_below(ax, ncol=1)
    fig.suptitle(_wrap(f"{title}（オリフィス: {'、'.join(o.describe() for o in res.orifices)}）"))
    _save(fig, path)


def _place_legends(fig, time_axes: bool = False):
    """凡例を、軸の目盛りの文字と軸のラベルの下に置く。time_axes なら描画の領域の左右を TIME_AXES_LR にそろえる。"""
    fig.tight_layout()
    if time_axes:
        fig.subplots_adjust(left=TIME_AXES_LR[0], right=TIME_AXES_LR[1])
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        ncol = getattr(ax, "_amamizu_legend_ncol", None)
        if ncol is None:
            continue
        bottom = ax.get_tightbbox(renderer).y0                 # 目盛りの文字・軸のラベルを含む下端（画素）
        y = ax.transAxes.inverted().transform((0, bottom))[1]
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, y - 0.02), ncol=ncol, frameon=False)


def _check_legends(fig):
    """凡例が描画の領域、凡例の外の文字（タイトル・軸のラベル・目盛り）、ほかの凡例に重なっていればエラーにする。"""
    import matplotlib.text as mtext
    renderer = fig.canvas.get_renderer()
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    inside = {id(t) for leg in legends for t in leg.findobj(mtext.Text)}
    texts = [t for t in fig.findobj(mtext.Text) if t.get_visible() and t.get_text().strip() and id(t) not in inside]
    for i, leg in enumerate(legends):
        box = leg.get_window_extent(renderer)
        for other in legends[i + 1:]:
            if box.overlaps(other.get_window_extent(renderer)):
                raise GraphError("グラフの凡例どうしが重なり、文字が読めなくなるため出力しません。"
                                 "凡例に並ぶ区域・放流施設の名前を短くしてください。")
        for ax in fig.axes:
            if box.overlaps(ax.get_window_extent(renderer)):
                raise GraphError("グラフの凡例が描画の領域に重なり、文字が読めなくなるため出力しません。"
                                 "凡例に並ぶ区域・放流施設の名前を短くしてください。")
        for t in texts:
            if box.overlaps(t.get_window_extent(renderer)):
                raise GraphError(f"グラフの凡例が文字「{t.get_text()}」に重なり、文字が読めなくなるため出力しません。"
                                 "凡例に並ぶ区域・放流施設の名前を短くしてください。")


def _save(fig, path: Path, time_axes: bool = False):
    """図を保存する。図の外に置いた凡例も切らずに含める。time_axes の図は横を切り詰めず図の幅のままにし
    （切り詰める幅が縦軸の文字の幅で変わると、描画の領域の横の位置がずれる）、縦だけを切り詰める。"""
    from matplotlib.transforms import Bbox
    _place_legends(fig, time_axes)
    fig.canvas.draw()
    _check_legends(fig)
    box = "tight"
    if time_axes:
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        w = fig.get_figwidth()
        if tight.x0 < 0 or tight.x1 > w:
            raise GraphError("グラフの文字が図の幅に収まらず、文字が切れるため出力しません。"
                             "題名や区域・放流施設の名前を短くしてください。")
        box = Bbox.from_extents(0, tight.y0, w, tight.y1).padded(0.05)
    fig.savefig(path, dpi=DPI, bbox_inches=box)
    plt.close(fig)


def write_all(res: DesignResult, out_dir: Path, title: str) -> list[Path]:
    """3枚のグラフを out_dir に書き、書いたパスを返す（文字が読めなければ GraphError。_readable_text）。"""
    paths = [out_dir / name for name in GRAPH_NAMES]
    with _readable_text(paths):
        hydrograph(res, paths[0], title)
        water_level(res, paths[1], title)
        pond_curves(res, paths[2], title)
    return paths
