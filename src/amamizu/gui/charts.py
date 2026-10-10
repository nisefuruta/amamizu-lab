"""WebGUI のグラフ（ECharts の設定）。NiceGUI の部品には依存しない（設定の dict を返すだけ）。

文字が読めなくならないように、凡例は描画の領域の外（下）に置き、描画の領域の下に凡例の分の余白を取る。
値はマウスを重ねると出る（tooltip）。単位は m^3 のようにキャレットで書く。
点が多いとき（長い期間の実測降雨）は、ブラウザが止まらないよう区間ごとの最大の点だけを描く（_points。山は残る）。
"""
import numpy as np

BLUE, RED, GRAY, ORANGE, DARK = "#1a4f8f", "#d03b3b", "#7a7a7a", "#e08a00", "#2a78d6"
PALETTE = ["#1a4f8f", "#d03b3b", "#2e9e5b", "#e08a00", "#7b4fb3", "#1a9ab0", "#b35a1a", "#555555"]


MAX_POINTS = 4000


def _points(x, y, digits: int) -> list:
    """グラフの点 [x, y] の並び。MAX_POINTS を超えるときは、同じ数ずつの区間に分けて各区間の最大の点だけにする。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(y) > MAX_POINTS:
        k = -(-len(y) // MAX_POINTS)
        pad = -len(y) % k
        yy = np.concatenate([y, np.full(pad, -np.inf)]).reshape(-1, k)
        idx = np.arange(0, len(y) + pad, k) + yy.argmax(1)
        x, y = x[idx], y[idx]
    return [[round(float(a), 4), round(float(b), digits)] for a, b in zip(x, y)]


# 描画の領域の左右の余白 px。時間を横軸にするグラフ（流量、水位）を上下に並べたとき、同じ時刻が同じ横の位置になるよう共通にする
GRID_LEFT, GRID_RIGHT = 72, 24


def _base(legend_rows: int = 1) -> dict:
    return {
        "animation": False,
        "tooltip": {"trigger": "axis"},
        "legend": {"bottom": 0, "type": "scroll"},
        "grid": {"left": GRID_LEFT, "right": GRID_RIGHT, "top": 40, "bottom": 72 + 22 * (legend_rows - 1),
                 "containLabel": False},
        "toolbox": {"feature": {"dataZoom": {"yAxisIndex": "none"}, "restore": {}, "saveAsImage": {}}, "right": 8, "top": 0},
    }


def _hours(minutes) -> list:
    return [round(float(m) / 60, 4) for m in minutes]


def _time_axis(last_h: float) -> dict:
    """時間の軸。目盛りは 6 時間ごと（長いときは 12 時間、1日、…）にし、端も目盛りにそろえる（端の数字が重ならない）。"""
    step = next((s for s in (6, 12, 24, 48, 168, 336, 720, 1440, 4320, 8760, 43800) if last_h / s <= 16), 87600)
    return {"type": "value", "name": "時間 h", "nameLocation": "middle", "nameGap": 26, "min": 0,
            "max": float(np.ceil(last_h / step) * step) or step, "interval": step}


def rain_bars(storm, tc: int, title: str = "") -> dict:
    """降雨の棒グラフ（区間ごとの降雨強度。計画降雨と実測降雨）。"""
    storm = np.asarray(storm, dtype=float)
    x = [round((i + 0.5) * tc / 60, 4) for i in range(len(storm))]
    o = _base()
    o |= {"title": {"text": title, "left": "center", "textStyle": {"fontSize": 13}},
          "xAxis": _time_axis(len(storm) * tc / 60) | {"nameGap": 28},
          "yAxis": {"type": "value", "name": "mm/h"},
          "series": [{"type": "bar", "name": "降雨強度", "data": _points(x, storm, 3),
                      "barWidth": "95%", "itemStyle": {"color": DARK}}]}
    return o


def flows(t_min, series: dict, allowable: float | None = None, rain=None, tc: int | None = None) -> dict:
    """流量の時間変化（series は 名前 → 値の並び）。rain を渡すと上に降雨強度の棒（下向き）を足す。"""
    x = _hours(t_min)
    axis = _time_axis(x[-1])
    colors = [RED, BLUE, "#2e9e5b", ORANGE]
    o = _base(legend_rows=2 if len(series) > 2 else 1)
    grids, xaxes, yaxes, out = [], [], [], []
    top = 40
    if rain is not None:
        grids.append({"left": GRID_LEFT, "right": GRID_RIGHT, "top": top, "height": "22%"})
        xaxes.append({"type": "value", "gridIndex": 0, "axisLabel": {"show": False}, "min": 0, "max": axis["max"],
                      "interval": axis["interval"]})
        yaxes.append({"type": "value", "gridIndex": 0, "name": "降雨 mm/h", "inverse": True, "nameLocation": "middle",
                      "nameGap": 48, "splitNumber": 2})
        out.append({"type": "bar", "name": "降雨強度", "xAxisIndex": 0, "yAxisIndex": 0, "barWidth": "95%",
                    "itemStyle": {"color": DARK},
                    "data": _points((np.arange(len(rain)) + 0.5) * tc / 60, rain, 3)})
        top = "42%"
    k = len(grids)
    grids.append({"left": GRID_LEFT, "right": GRID_RIGHT, "top": top, "bottom": o["grid"]["bottom"]})
    xaxes.append(axis | {"gridIndex": k})
    yaxes.append({"type": "value", "gridIndex": k, "name": "流量 m^3/s", "nameLocation": "middle", "nameGap": 48})
    for i, (name, values) in enumerate(series.items()):
        out.append({"type": "line", "name": name, "xAxisIndex": k, "yAxisIndex": k, "showSymbol": False,
                    "lineStyle": {"width": 2 if i < 2 else 1.5, "type": "dotted" if "浸透前" in name else "solid"},
                    "itemStyle": {"color": colors[i % len(colors)]},
                    "data": _points(x, values, 5)})
    if allowable is not None and out:
        out[-1]["markLine"] = {"symbol": "none", "lineStyle": {"type": "dashed", "color": GRAY},
                               "label": {"formatter": f"許容放流量 {allowable:g} m^3/s", "position": "insideStartTop"},
                               "data": [{"yAxis": allowable}]}
    o.pop("grid")
    o |= {"grid": grids, "xAxis": xaxes, "yAxis": yaxes, "series": out,
          "axisPointer": {"link": [{"xAxisIndex": "all"}]}}
    return o


def _level_axis(values, lines: list) -> dict:
    """水位の縦軸。水位の線（満水位など）も範囲に入れる（範囲の外の線は描かれないため）。"""
    ys = [float(v) for v in values] + [float(v) for _, v, _ in lines]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.05 or 0.1
    return {"type": "value", "name": "水位 m", "nameLocation": "middle", "nameGap": 48,
            "min": round(lo - pad, 3), "max": round(hi + pad, 3)}


def _level_lines(lines: list) -> dict:
    """水位の線（名前, 値, 色）の markLine。"""
    return {"symbol": "none", "silent": True,
            "data": [{"yAxis": v, "lineStyle": {"type": "dashed", "color": c},
                      "label": {"formatter": f"{n} {v:.3f} m", "position": "insideStartTop", "color": c}}
                     for n, v, c in lines]}


def level(t_min, series: dict, lines: list) -> dict:
    """水位の時間変化（series は 名前 → 値の並び。比べるときは複数）。lines は（名前, 水位, 色）の横線。"""
    x = _hours(t_min) if not isinstance(t_min, dict) else None
    o = _base(legend_rows=2 if len(series) > 3 else 1)
    out, last = [], 1.0
    for i, (name, values) in enumerate(series.items()):
        xs = _hours(t_min[name]) if isinstance(t_min, dict) else x
        out.append({"type": "line", "name": name, "showSymbol": False, "itemStyle": {"color": PALETTE[i % len(PALETTE)]},
                    "data": _points(xs, values, 4)})
        last = max(last, xs[-1]) if len(xs) else last
    if out and lines:
        out[0]["markLine"] = _level_lines(lines)
    o |= {"xAxis": _time_axis(last), "yAxis": _level_axis([d[1] for x in out for d in x["data"]], lines), "series": out}
    return o


def curve(x, y, x_name: str, lines: list, name: str) -> dict:
    """水位を縦軸にした特性の線（水位と容量、水位と放流量）。"""
    o = _base()
    o["tooltip"] = {"trigger": "axis", "axisPointer": {"type": "cross"}}
    s = {"type": "line", "name": name, "showSymbol": False, "itemStyle": {"color": BLUE},
         "data": [[round(float(a), 5), round(float(b), 4)] for a, b in zip(x, y)]}
    if lines:
        s["markLine"] = _level_lines(lines)
    o |= {"xAxis": {"type": "value", "name": x_name, "nameLocation": "middle", "nameGap": 26},
          "yAxis": _level_axis(y, lines), "series": [s]}
    return o
