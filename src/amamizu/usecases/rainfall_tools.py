"""時間降水量の道具（旧 jma_rainfall の analyzer / capturer から移設）。

入力は jma-rainfall の時間値 CSV（日付, 時(1〜24), 降水量_mm 列。旧形式の date, hour, rainfall_mm も可）。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..domain.capture import capture_rate


def capture_table(rain_mm, min_intensity: float = 5, max_intensity: float = 50, step: float = 5) -> pd.DataFrame:
    """設計浸透強度ごとの雨水捕捉率。欠測（空欄）の時間は除いて計算し、除いた時間数を「欠測の時間数」の列に書く。
    数として読めない値があればエラー（欠測として黙って除かない）。"""
    s = pd.Series(rain_mm).reset_index(drop=True)
    num = pd.to_numeric(s, errors="coerce")
    text = s.notna() & num.isna()
    if text.any():
        more = f"（ほかに {int(text.sum()) - 1} 行）" if text.sum() > 1 else ""
        raise ValueError(f"降水量の {int(np.flatnonzero(text.to_numpy())[0]) + 1} 行目の値「{s[text].iloc[0]}」を数として読めません{more}。"
                         "値を直すか、欠測なら空欄にしてください。")
    r = num.to_numpy(dtype=float)
    n_missing = int(np.isnan(r).sum())
    rows = []
    for i in np.arange(min_intensity, max_intensity + step, step):
        rate, total, captured = capture_rate(r, i)
        rows.append({"設計浸透強度 (mm/hr)": i, "捕捉率 (%)": rate, "総降雨量 (mm)": total,
                     "捕捉雨量 (mm)": captured, "超過雨量 (mm)": total - captured,
                     "計算した時間数": len(r) - n_missing, "欠測の時間数": n_missing})
    return pd.DataFrame(rows)


@dataclass
class HourlyStatistics:
    basic: dict
    monthly: pd.DataFrame      # 月, 平均, 最大
    yearly: pd.DataFrame       # 年, 平均, 年間総降水量, 最大


JP_COLUMNS = {"日付": "date", "時": "hour", "降水量_mm": "rainfall_mm"}


def normalize_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """jma-rainfall の時間値（日付, 時, 降水量_mm）も旧形式（date, hour, rainfall_mm）も受け付ける。"""
    return df.rename(columns={k: v for k, v in JP_COLUMNS.items() if k in df.columns})


def with_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """date と hour（1〜24、その時刻で終わる1時間）から、1時間の始まりの datetime・year・month を作る。"""
    df = normalize_hourly(df)
    if "datetime" not in df.columns:
        df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"].astype(int) - 1, unit="h")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    return df


def hourly_statistics(df: pd.DataFrame) -> HourlyStatistics:
    df = with_datetime(df)
    r = df["rainfall_mm"]
    basic = {
        "記録期間": f"{df['datetime'].min():%Y-%m-%d} から {df['datetime'].max():%Y-%m-%d}",
        "データポイント数": len(df),
        "欠損値数": int(r.isna().sum()),
        "欠損率 (%)": round(r.isna().mean() * 100, 2),
        "平均時間降水量 (mm/時)": round(r.mean(), 2),
        "最大時間降水量 (mm/時)": round(r.max(), 2),
        "最大時間降水量の発生日時": (df["datetime"].iloc[r.argmax()].strftime("%Y-%m-%d %H:%M")
                            if not pd.isna(r.max()) else "N/A"),
        "無降水時間の割合 (%)": round((r == 0).mean() * 100, 2),
        "10mm/時以上の強雨発生回数": int((r >= 10).sum()),
        "30mm/時以上の豪雨発生回数": int((r >= 30).sum()),
        "50mm/時以上の猛烈な雨発生回数": int((r >= 50).sum()),
    }
    g = df.groupby("month")["rainfall_mm"]
    monthly = pd.DataFrame({"平均": g.mean().round(2), "最大": g.max().round(2)})
    y = df.groupby("year")["rainfall_mm"]
    yearly = pd.DataFrame({"平均": y.mean().round(2), "年間総降水量": y.sum().round(2), "最大": y.max().round(2)})
    return HourlyStatistics(basic, monthly, yearly)
