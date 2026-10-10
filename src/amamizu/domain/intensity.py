"""確率降雨強度式。

r = a / (t^n + b)。r は mm/hr、t は分。次の5方式はどれもこの形で表せる。

| 種類 | 名前 | 式 | 係数 |
| talbot | タルボット | a / (t + b) | a, b（n＝1） |
| sherman | シャーマン | a / t^n | a, n（b＝0） |
| kuno_ishiguro | 久野・石黒 | a / (t^0.5 + b) | a, b（n＝1/2） |
| cleveland | クリーブランド | a / (t^n + b) | a, b, n |
| kuno_ishiguro_n | 久野・石黒（任意の n） | a / (t^n + b) | a, b, n（クリーブランドと同じ式） |

千葉県の手引 資料－1 の式はすべてクリーブランド型。
"""
from dataclasses import dataclass

# 種類 → (名前, 式の書き方, 書く係数, 決まっている係数)
FORMULA_TYPES = {
    "talbot": ("タルボット", "a/(t+b)", ("a", "b"), {"n": 1.0}),
    "sherman": ("シャーマン", "a/t^n", ("a", "n"), {"b": 0.0}),
    "kuno_ishiguro": ("久野・石黒", "a/(t^0.5+b)", ("a", "b"), {"n": 0.5}),
    "cleveland": ("クリーブランド", "a/(t^n+b)", ("a", "b", "n"), {}),
    "kuno_ishiguro_n": ("久野・石黒（任意の n）", "a/(t^n+b)", ("a", "b", "n"), {}),
}


@dataclass(frozen=True)
class IntensityFormula:
    a: float
    n: float
    b: float

    def intensity(self, t_min: float) -> float:
        """継続時間 t_min 分の平均降雨強度 mm/hr。"""
        d = float(t_min ** self.n) + self.b      # float(): t ＞ 0 なので実数（型の上では float ** float は Any）
        if not d > 0:                            # b が負のとき。0 以下では強度が負か無限大になる
            raise ValueError(f"強度式の分母 t^n＋b が t＝{t_min:g} 分で 0 以下（{d:g}）になり、降雨強度を求められません。"
                             f"係数 b（{self.b:g}）と n（{self.n:g}）を見直してください。")
        return self.a / d

    def depth(self, t_min: float) -> float:
        """継続時間 t_min 分の雨量 mm。"""
        return self.intensity(t_min) * t_min / 60


def formula_of(kind: str, a: float | None = None, b: float | None = None, n: float | None = None) -> IntensityFormula:
    """種類と係数から強度式を作る。

    その種類で使わない係数（タルボットの n、シャーマンの b、久野・石黒の n）を書いたら、
    黙って無視せずエラーにする。要る係数がないときもエラー。
    """
    if kind not in FORMULA_TYPES:
        raise ValueError(f"強度式の種類「{kind}」は扱えません（{', '.join(FORMULA_TYPES)}）。")
    name, text, needed, fixed = FORMULA_TYPES[kind]
    given = {"a": a, "b": b, "n": n}
    lack = [k for k in needed if given[k] is None]
    if lack:
        raise ValueError(f"{name}式 {text} には {', '.join(lack)} が必要です。")
    extra = [k for k in fixed if given[k] is not None]
    if extra:
        raise ValueError(f"{name}式 {text} では {', '.join(extra)} を書かないでください（{', '.join(f'{k}＝{v:g}' for k, v in fixed.items())} に決まっています）。")
    known = {k: v for k, v in given.items() if v is not None}     # 要る係数はすべてある（上で確かめた）
    if known["a"] <= 0:
        raise ValueError(f"強度式の a は正の値にしてください（{a}）。")
    vals = {**{k: known[k] for k in needed}, **fixed}
    return IntensityFormula(float(vals["a"]), float(vals["n"]), float(vals["b"]))
