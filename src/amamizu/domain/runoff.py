"""流出の計算。"""
import numpy as np
import numpy.typing as npt


def rational_inflow(rain_mmhr: npt.ArrayLike, f: float, area_ha: float) -> npt.NDArray[np.float64]:
    """合理式連続モデル（手引 図5－3、洪水到達時間＝降雨の区切り）の流入量 m3/s。

    各区切りの降雨は底辺が区切り2つ分の三角形のハイドログラフになる。節点での流量は
    Q = f·r·A/360（r はその節点で終わる区切りの強度 mm/hr、A は ha）で、節点の間は直線。
    節点0は0、降雨の後に1節点足して最後の三角形が0に戻るまでを含める。
    """
    q = f * np.asarray(rain_mmhr, dtype=float) * area_ha / 360.0
    return np.concatenate([[0.0], q, [0.0]])
