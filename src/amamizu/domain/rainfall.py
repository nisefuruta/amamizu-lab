"""降雨の時系列の扱い（区切り、継続時間ごとの雨量）。"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# 値がない品質（jma_rainfall の品質のうち、降水量の値がないもの: 欠測・疑問値・空白）。欠測として数え、0mm で計算する
NO_VALUE = frozenset({"missing", "doubtful", "blank"})


def max_sum(values: npt.ArrayLike, k: int) -> float:
    """連続する k 個の和の最大。長さが k に満たなければエラー（記録の外を0とみなした値を、k 時間の雨量として黙って返さない）。"""
    x = np.asarray(values, dtype=float)
    if len(x) < k:
        raise ValueError(f"{k}時間の雨量を求めるには {k} 時間以上の記録が要りますが、{len(x)} 時間しかありません。")
    c = np.concatenate([[0.0], np.cumsum(x)])
    return float((c[k:] - c[:-k]).max())


def spread_hourly(hourly_mm: npt.ArrayLike) -> FloatArray:
    """1時間の雨量を6区間（10分）へ均等に配分し、10分ごとの強度 mm/hr にする（1時間 x mm → 各10分も x mm/hr）。"""
    return np.repeat(np.asarray(hourly_mm, dtype=float), 6)


def blocks_from_10min(rain_mmhr_10min: npt.ArrayLike, tc_min: int) -> FloatArray:
    """10分ごとの強度 mm/hr を、tc 分ごとの強度 mm/hr にまとめる（雨量を tc ごとに合計して強度に直す）。

    tc は10の倍数。末尾が tc に満たなければ0で埋める（雨の後の無降水）。tc＝10 ならそのまま返す。
    """
    x = np.asarray(rain_mmhr_10min, dtype=float)
    if tc_min == 10:
        return x
    if tc_min <= 0 or tc_min % 10:
        raise ValueError(f"tc {tc_min} 分は10分値から作れません（10の倍数にしてください）。")
    k = tc_min // 10
    x = np.concatenate([x, np.zeros(-len(x) % k)])
    return x.reshape(-1, k).mean(1)


def hourly_from_10min(values: npt.ArrayLike) -> FloatArray:
    """10分値（00:10〜24:00 のコマ）を毎正時で区切った1時間雨量にする。"""
    mm10 = np.asarray(values, dtype=float)
    if len(mm10) % 6:
        raise ValueError("10分値の個数が6の倍数ではないので、1時間ごとにまとめられません。")
    return mm10.reshape(-1, 6).sum(1)


@dataclass(frozen=True)
class DurationDepth:
    hours: int
    observed: float     # 実測 mm
    plan: float         # 計画（強度式）mm

    @property
    def ratio(self) -> float:
        return self.observed / self.plan


def duration_depths(hourly_mm: npt.ArrayLike, plan_depth: Callable[[int], float], hours: tuple[int, ...],
                    max1h_published: float | None = None) -> list[DurationDepth]:
    """継続時間ごとの最大雨量。毎正時で区切った1時間雨量の連続和の最大。

    1時間は公表値（日別値表の最大1時間降水量。1分ずつずらした最大）があればそれと大きいほうを採る。
    plan_depth(hours) は同じ継続時間の計画の雨量 mm を返す関数。
    """
    out = []
    for h in hours:
        v = max_sum(hourly_mm, h)
        if h == 1 and max1h_published is not None and not np.isnan(max1h_published):
            v = max(v, float(max1h_published))
        out.append(DurationDepth(h, v, plan_depth(h)))
    return out


@dataclass(frozen=True)
class RainEvent:
    start: int          # 開始日の番号（日別の並びの添字）
    end: int            # 終了日の番号（含む）
    total: float        # 総量 mm

    @property
    def days(self) -> int:
        return self.end - self.start + 1


def split_events(daily_totals: npt.ArrayLike, dry_gap: int = 2) -> list[RainEvent]:
    """日降水量の並びを降雨ごとに区切る。降水0の日が dry_gap 日以上続いたら別の降雨とする。
    欠測（NaN）の日は、雨量を0とするが無降水の日には数えない（その日で降雨を区切らない）。
    降った日かどうかわからないので、前後の降雨をつないで総量で判断し、計算する降雨を取りこぼさないため。
    欠測の日だけでは降雨を始めない。"""
    raw = np.asarray(daily_totals, dtype=float)
    totals = np.nan_to_num(raw)
    events: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None     # 降雨の途中なら（開始日, 最後に降った日）
    dry = 0
    for i, v in enumerate(raw):
        if np.isnan(v):
            dry = 0
        elif v > 0:
            cur = (i, i) if cur is None else (cur[0], i)
            dry = 0
        elif cur is not None:
            dry += 1
            if dry >= dry_gap:
                events.append(cur)
                cur = None
    if cur is not None:
        events.append(cur)
    return [RainEvent(a, b, round(float(totals[a:b + 1].sum()), 1)) for a, b in events]
