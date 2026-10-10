"""浸透施設の雨水捕捉率。"""
import numpy as np
import numpy.typing as npt


def capture_rate(rain_mm: npt.ArrayLike, design_intensity: float) -> tuple[float, float, float]:
    """時間雨量の並びに対し、設計浸透強度 mm/hr までを捕捉できるとしたときの捕捉率 %。

    各時間の捕捉量は min(雨量, 設計浸透強度)。欠測（NaN）は除く。
    戻り値は (捕捉率 %, 総降雨量 mm, 捕捉雨量 mm)。
    """
    r = np.asarray(rain_mm, dtype=float)
    r = r[~np.isnan(r)]
    total = float(r.sum())
    captured = float(np.minimum(r, design_intensity).sum())
    return (captured / total * 100 if total > 0 else 0.0), total, captured
