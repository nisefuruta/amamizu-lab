"""浸透施設（一定量差し引きモデル）。

- 比浸透量 K（m^2）は施設の種類と設計水頭・寸法の式で求める（specific_infiltration）。
- 施設の設計浸透量 m^3/h ＝ K × 飽和透水係数（m/h）× 数量 × 影響係数の積。
- 空隙貯留量 m^3 ＝ 1単位あたりの（体積 × 空隙率）の和 × 数量。
- 各時刻の流入量から設計浸透量（m^3/s）を差し引き、超えた分はまず空隙貯留を満たし、満ちてから流す。

式の適用範囲の外や、その種類で使わない寸法を書いたときは、黙って外挿や無視をせずエラーにする。
"""
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn

import numpy as np
import numpy.typing as npt

# 種類 → (名前, 使う寸法)
KINDS = {
    "pavement": ("透水性舗装（浸透池）", ("H",)),
    "trench": ("浸透側溝および浸透トレンチ", ("H", "W")),
    "circle_side_bottom": ("円筒ます（側面および底面）", ("H", "D")),
    "circle_bottom": ("円筒ます（底面）", ("H", "D")),
    "square_side_bottom": ("正方形ますおよび空隙貯留浸透施設（側面および底面）", ("H", "W")),
    "square_bottom": ("正方形ます（底面）", ("H", "W")),
    "rect_side_bottom": ("矩形のますおよび空隙貯留浸透施設（側面および底面）", ("H", "W", "L")),
    "large_side_bottom": ("大型貯留浸透槽（側面および底面）", ("H", "W", "L")),
    "large_bottom": ("大型貯留浸透槽（底面）", ("H", "W", "L")),
}

# 大型貯留浸透槽の係数表（W＝5,10,20,30,40,50 m ごとに a0, p, b。K＝(a0·X^p·H＋b)·L、X＝L/W）
_LARGE_W = (5.0, 10.0, 20.0, 30.0, 40.0, 50.0)
_LARGE = {
    "large_side_bottom": ((8.83, -0.461, 7.03), (7.88, -0.446, 14.0), (7.06, -0.452, 27.06),
                          (6.43, -0.444, 39.75), (5.97, -0.44, 52.25), (5.62, -0.442, 64.68)),
    "large_bottom": ((1.94, -0.328, 7.57), (2.29, -0.397, 13.84), (2.37, -0.488, 26.36),
                     (2.17, -0.518, 38.79), (1.96, -0.554, 51.16), (1.76, -0.609, 63.5)),
}


def _out_of_range(kind: str, dims: dict[str, float | None]) -> NoReturn:
    name = KINDS[kind][0]
    text = "、".join(f"{k}＝{v:g} m" for k, v in dims.items() if v is not None)
    raise ValueError(f"{name} の比浸透量の式の適用範囲外です（{text}）。寸法を式の適用範囲の中にするか、"
                     "比浸透量を直接入力してください（WebGUI では「比浸透量を直接」、study.json では specific_m2）。")


def _by_pressure(k_at: Callable[[float], float], h: float, p_at: Callable[[float], float]) -> float:
    """設計水頭が1.5 m を超えるときの比例配分: H＝1.0 と 1.5 m の K を静水圧指標 P の直線で結び、設計水頭の P で読む。"""
    k1, k2, p1, p2 = k_at(1.0), k_at(1.5), p_at(1.0), p_at(1.5)
    return k1 + (k2 - k1) * (p_at(h) - p1) / (p2 - p1)


def specific_infiltration(kind: str, H: float, W: float | None = None, D: float | None = None,
                          L: float | None = None) -> float:
    """比浸透量 K（m^2）。H は設計水頭、W は施設幅、D は施設直径、L は施設延長（m）。"""
    if kind not in KINDS:
        raise ValueError(f"浸透施設の種類「{kind}」は扱えません（{', '.join(KINDS)}）。")
    name, needed = KINDS[kind]
    dims = {"H": H, "W": W, "D": D, "L": L}
    lack = [k for k in needed if dims[k] is None]
    extra = [k for k in dims if k not in needed and dims[k] is not None]
    if lack:
        raise ValueError(f"{name} の比浸透量には {', '.join(lack)} が必要です。")
    if extra:
        raise ValueError(f"{name} の比浸透量の式は {', '.join(extra)} を使わないので、書かないでください。")
    # 使う寸法はすべてある（上で確かめた）。各種類の式は、確かめた寸法（float）だけを vals から取る
    vals = {k: v for k, v in dims.items() if v is not None}
    h = H
    if kind == "pavement":
        if 0 < h <= 1.5:
            return 0.014 * h + 1.287
    elif kind == "trench":
        W = vals["W"]
        if 0 < h <= 1.5 and 0 < W <= 1.5:
            return 3.093 * h + (1.34 * W + 0.677)
    elif kind == "circle_side_bottom":
        d = vals["D"]

        def p_at(x: float) -> float:
            return 0.5 * x ** 2 * math.pi * d + x * (d / 2) ** 2 * math.pi
        k_at: Callable[[float], float]
        if 0.2 <= d < 1.0:
            a, b, c = 0.475 * d + 0.945, 6.07 * d + 1.01, 2.57 * d - 0.188

            def k_at(x: float) -> float:
                return a * x ** 2 + b * x + c
        elif 1.0 <= d <= 10.0:
            a, b = 6.244 * d + 2.853, 0.93 * d ** 2 + 1.606 * d - 0.773

            def k_at(x: float) -> float:
                return a * x + b
        else:
            _out_of_range(kind, dims)
        if 0 < h <= 1.5:
            return k_at(h)
        if 1.5 < h <= 5.0:
            return _by_pressure(k_at, h, p_at)
    elif kind == "circle_bottom":
        D = vals["D"]
        if 0 < h <= 5.0 and 0.3 <= D <= 1.0:
            return (1.497 * D - 0.1) * h + (1.13 * D ** 2 + 0.638 * D - 0.011)
        if 0 < h <= 1.5 and 1.0 < D <= 10.0:
            return (2.556 * D - 2.052) * h + (0.924 * D ** 2 + 0.993 * D - 0.087)
    elif kind == "square_side_bottom":
        w = vals["W"]
        if 0 < w <= 1.0:
            a, b, c = 0.12 * w + 0.985, 7.837 * w + 0.82, 2.858 * w - 0.283

            def k_at(x: float) -> float:
                return a * x ** 2 + b * x + c
            if 0 < h <= 1.5:
                return k_at(h)
            if 1.5 < h <= 5.0:       # W＝1 ちょうども、H≦1.5 と同じくこの式で計算する（W≦1）
                return _by_pressure(k_at, h, lambda x: 2 * x ** 2 * w + x * w ** 2)
        elif 1.0 < w <= 10.0 and 0 < h <= 5.0:
            return (-0.453 * w ** 2 + 8.289 * w + 0.753) * h + (1.458 * w ** 2 + 1.27 * w + 0.362)
        elif 10.0 < w <= 80.0 and 0 < h <= 5.0:
            return (0.747 * w + 21.355) * h + (1.263 * w ** 2 + 4.295 * w - 7.649)
    elif kind == "square_bottom":
        W = vals["W"]
        if 0 < h <= 5.0 and 0 < W <= 1.0:
            return (1.676 * W - 0.137) * h + (1.496 * W ** 2 + 0.671 * W - 0.015)
        if 0 < h <= 5.0 and 1.0 < W <= 10.0:
            return (-0.204 * W ** 2 + 3.166 * W - 1.936) * h + (1.345 * W ** 2 + 0.736 * W + 0.251)
        if 0 < h <= 5.0 and 10.0 < W <= 80.0:
            return (1.265 * W - 15.67) * h + (1.259 * W ** 2 + 2.336 * W - 8.13)
    elif kind == "rect_side_bottom":
        W, L = vals["W"], vals["L"]
        if 0 < h <= 5.0 and 0 <= L <= 200.0 and 0 <= W <= 4.0:     # 施設規模の目安は延長 200 m、幅 4 m
            return (3.297 * L + (1.971 * W + 4.663)) * h + ((1.401 * W + 0.684) * L + (1.214 * W - 0.834))
    else:                            # 大型貯留浸透槽。W の表の間は K を W で線形補間する
        wl, ll = vals["W"], vals["L"]
        if wl > 0 and 0.5 < h <= 5.0 and 5.0 <= wl <= 50.0 and 1.0 <= ll / wl <= 5.0:
            table, x = _LARGE[kind], ll / wl

            def k_of(row: tuple[float, float, float]) -> float:
                a0, p, b = row
                return (a0 * float(x ** p) * h + b) * ll     # float(): x ＞ 0 なので実数（型の上では float ** float は Any）
            for i, w in enumerate(_LARGE_W):
                if wl == w:
                    return k_of(table[i])
                if w < wl < _LARGE_W[i + 1]:
                    k1, k2 = k_of(table[i]), k_of(table[i + 1])
                    return k1 + (k2 - k1) * (wl - w) / (_LARGE_W[i + 1] - w)
    _out_of_range(kind, dims)


PERMEABILITY_UNITS = {"m/h": 1.0, "m/s": 3600.0, "cm/s": 36.0}


@dataclass(frozen=True)
class Facility:
    """浸透施設1種類。specific_m2 は比浸透量 K、permeability_m_h は飽和透水係数（m/h）。
    影響係数は目づまり clogging、地下水位 groundwater、安全率 safety（3つを掛け合わせる）。
    void_per_unit_m3 は1単位あたりの空隙貯留量。"""
    name: str
    specific_m2: float
    permeability_m_h: float
    count: float
    clogging: float
    groundwater: float
    safety: float
    void_per_unit_m3: float = 0.0

    def __post_init__(self) -> None:
        if self.specific_m2 <= 0 or self.permeability_m_h < 0 or self.count < 0 or self.void_per_unit_m3 < 0:
            raise ValueError(f"浸透施設「{self.name}」の比浸透量は正、透水係数・数量・空隙貯留量は0以上にしてください。")
        for label, f in self.factors():
            if not 0 < f <= 1:
                raise ValueError(f"浸透施設「{self.name}」の影響係数（{label}）は 0 より大きく 1 以下にしてください（今は {f:g}）。")

    def factors(self) -> tuple[tuple[str, float], ...]:
        """影響係数の（名前, 値）。"""
        return (("目づまり", self.clogging), ("地下水位", self.groundwater), ("安全率 α", self.safety))

    @property
    def rate_m3_h(self) -> float:
        return self.specific_m2 * self.permeability_m_h * self.count * self.clogging * self.groundwater * self.safety

    @property
    def void_m3(self) -> float:
        return self.void_per_unit_m3 * self.count


def subtract_constant(inflow: npt.ArrayLike, rate_m3s: float, void_m3: float, dt: float) -> npt.NDArray[np.float64]:
    """各節点の流入量から設計浸透量 rate_m3s を差し引く（一定量差し引きモデル）。

    流入量が設計浸透量より小さい節点は0。超えた分（×dt の水量）はまず空隙貯留を満たし、
    満ちるまでは0、満ちた節点は残りを流量に直して流し、以後は超えた分をそのまま流す。
    超えた分がちょうど空隙の残りと等しい節点は、満ちたものとして扱う。
    """
    if rate_m3s < 0 or void_m3 < 0:
        raise ValueError("設計浸透量と空隙貯留量は0以上にしてください。")
    q = np.asarray(inflow, dtype=float)
    out = np.zeros_like(q)
    stored = 0.0
    for i, v in enumerate(q):
        excess = v - rate_m3s
        if excess <= 0:
            continue
        if stored >= void_m3:
            out[i] = excess
        elif stored + excess * dt < void_m3:
            stored += excess * dt
        else:
            out[i] = excess - (void_m3 - stored) / dt
            stored = void_m3
    return out


def design_rate_m3s(intensity_mmh: float, area_ha: float) -> float:
    """設計浸透強度 mm/h と浸透処理面積 ha から設計浸透量 m3/s を求める（手引 第10条(5)
    「設計浸透強度 ＝ 設計浸透量 m3/h ／（浸透処理面積 ha × 10）」の逆）。"""
    if intensity_mmh < 0:
        raise ValueError(f"設計浸透強度 {intensity_mmh} mm/h は0以上にしてください。")
    if area_ha <= 0:
        raise ValueError(f"浸透処理面積 {area_ha} ha は正の値にしてください。")
    return intensity_mmh * area_ha * 10 / 3600
