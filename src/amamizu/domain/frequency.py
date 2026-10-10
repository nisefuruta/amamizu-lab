"""確率分布（年最大値系列の当てはめ）。"""
import math
from collections.abc import Sequence
from dataclasses import dataclass

EULER = 0.5772157


@dataclass(frozen=True)
class Gumbel:
    mu: float       # 位置パラメータ
    beta: float     # 規模パラメータ

    def cdf(self, x: float) -> float:
        """非超過確率 F(x)。"""
        return math.exp(-math.exp(-(x - self.mu) / self.beta))

    def return_period(self, x: float) -> float:
        """再現期間 T = 1/(1-F(x)) 年。"""
        return 1 / (1 - self.cdf(x))

    @staticmethod
    def by_moments(sample: Sequence[float]) -> "Gumbel":
        """モーメント法。標本の平均 m と標準偏差 s（不偏）から beta = s·sqrt(6)/π、mu = m-0.5772157·beta。"""
        n = len(sample)
        if n < 2:
            raise ValueError("標本が2個未満です。再現期間を求めるには、年最大値が2年分以上必要です。")
        m = sum(sample) / n
        s = math.sqrt(sum((v - m) ** 2 for v in sample) / (n - 1))
        beta = s * math.sqrt(6) / math.pi
        return Gumbel(m - EULER * beta, beta)


def sample_stats(sample: Sequence[float]) -> tuple[float, float]:
    """平均と標準偏差（不偏）。"""
    n = len(sample)
    m = sum(sample) / n
    return m, math.sqrt(sum((v - m) ** 2 for v in sample) / (n - 1))
