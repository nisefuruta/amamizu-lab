"""計画降雨の波形。"""
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import numpy.typing as npt

from .intensity import IntensityFormula

SHAPES = {"rear": "後方集中型", "center": "中央集中型"}


def ranked_blocks(formula: IntensityFormula, hours: int = 24, block_min: int = 10) -> npt.NDArray[np.float64]:
    """区間ごとの強度 mm/hr を、強い順（1番目、2番目 …）に返す。

    t＝block_min, 2·block_min … の累加雨量を強度式から求め、その差分（a, 2b-a, 3c-2b …）を区間の雨量とする。
    block_min は洪水到達時間 tc（合理式連続モデルの区切り）。降雨の継続時間を割り切れなければ ValueError。
    """
    if hours <= 0:                  # 0 以下では区間が1つもなく、流入 0 の計画降雨で黙って計算が進む
        raise ValueError(f"計画降雨の継続時間 {hours} 時間は正の値にしてください。")
    if block_min <= 0 or (hours * 60) % block_min:
        raise ValueError(f"計画降雨の区切り {block_min} 分が、継続時間 {hours} 時間を割り切りません。")
    k = np.arange(1, hours * 60 // block_min + 1)
    depth = np.array([formula.depth(float(block_min * i)) for i in k])
    block = np.diff(np.concatenate([[0.0], depth]))
    return block * 60 / block_min


def rear_loaded(formula: IntensityFormula, hours: int = 24, block_min: int = 10) -> npt.NDArray[np.float64]:
    """後方集中型の計画降雨（手引 図5－1）。block_min 分ごとの強度 mm/hr を時刻順に返す。最後がピーク。
    強い順の区間を逆に並べる（最後が1番目）。"""
    return ranked_blocks(formula, hours, block_min)[::-1]


def center_loaded(formula: IntensityFormula, hours: int = 24, block_min: int = 10) -> npt.NDArray[np.float64]:
    """中央集中型の計画降雨。block_min 分ごとの強度 mm/hr を時刻順に返す。

    1番目の区間を中央（区間が N 個なら (N-1)//2 番目。24時間・10分なら 11:50〜12:00）に置き、
    2・4・6 … 番目をその後ろへ順に、3・5・7 … 番目をその前へ順に（中央に近いほど強く）並べる。
    """
    ranked = ranked_blocks(formula, hours, block_min)
    n = len(ranked)
    before = (n - 1) // 2
    out = np.empty(n)
    out[before] = ranked[0]
    out[before + 1:] = ranked[1::2][:n - 1 - before]
    out[:before] = ranked[2::2][:before][::-1]
    return out


def round_half_up(values: npt.ArrayLike, digits: int) -> npt.NDArray[np.float64]:
    """小数点以下 digits 桁に四捨五入する（Excel の ROUND と同じ。numpy の round は偶数への丸めなので使わない）。

    値は表示に使う最短の10進表記（repr）から丸めるので、2.675 は 2.68 になる。
    """
    if isinstance(digits, bool) or not isinstance(digits, int) or digits < 0:
        raise ValueError(f"丸めの桁数は0以上の整数にしてください（{digits!r}）。")
    q = Decimal(1).scaleb(-digits)
    return np.array([float(Decimal(repr(float(v))).quantize(q, rounding=ROUND_HALF_UP)) for v in np.ravel(values)])


def planned_storm(formula: IntensityFormula, shape: str, hours: int = 24, block_min: int = 10,
                  round_digits: int | None = None) -> npt.NDArray[np.float64]:
    """波形（rear：後方集中型、center：中央集中型）を選んで計画降雨を作る。

    round_digits を与えると、区間の降雨強度 mm/hr を小数点以下その桁に四捨五入する（丸めた値で流入量を計算する）。
    省略すると丸めない。
    """
    if shape == "rear":
        storm = rear_loaded(formula, hours, block_min)
    elif shape == "center":
        storm = center_loaded(formula, hours, block_min)
    else:
        raise ValueError(f"降雨波形「{shape}」は扱えません（rear 後方集中型、center 中央集中型）。")
    return storm if round_digits is None else round_half_up(storm, round_digits)
