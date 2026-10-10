"""レポート・要約・WebGUI で共通の数値の表記。"""
from decimal import Decimal


def spill_text(v: float, digits: int) -> str:
    """溢水量 m^3 の表示（設計の検討は小数1桁、貯留施設の検証は整数）。溢れているのに 0 と読める量は有効数字2桁で書き、
    判定（溢水量が0より大きければ N.G）や溢水の回数の表示と食い違わせない。"""
    text = f"{v:.{digits}f}"
    return format(Decimal(f"{v:.2g}"), "f") if v > 0 and float(text) == 0 else text      # 指数表記（4e-05）にしない
