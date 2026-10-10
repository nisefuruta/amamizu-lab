"""貯留施設の検証（kind＝storage）のレポートの Markdown を組み立てる。

文に記載する値（確率年、開発面積、地点数、降雨の名前と期間、判定の条件、出典など）は、条件・study.json・手引の定数から作る。
条件から導けない文は書かない（導けなければエラーにする）。条件を変えたときに、前の検討の文が黙って残らないようにするため。
"""
import datetime
import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pandas as pd

from ...usecases.guideline_check import HayamiResult
from ...usecases.storage_study import NO_PUBLISHED, StorageStudy
from . import numbers
from .mermaid import xychart_bar_init, xychart_bars, xychart_init, xychart_labels

DETAIL_FILL = 0.9   # 推移表と図を載せる降雨（最大貯留率がこれ以上）
INIT = xychart_init(360, "#d03b3b, #1a4f8f, #7a7a7a")
INIT1 = xychart_init(300, "#2a78d6")
INIT_RAIN = xychart_bar_init(300, "#d03b3b")
INIT_LEVEL = xychart_init(300, "#d03b3b, #1a4f8f, #7a7a7a")   # 水位（WebGUI の試算の比較と同じく、実測が赤、計画が青）


def section_numbers(n_events: int, n_overflow: int) -> dict:
    """節の番号。1 モデル施設、2 計画降雨の後に、降雨の表（n_events 節）、溢れた地点の水位（n_overflow 節）、長期記録、
    溢水した降雨の詳細、欠測のある降雨の順に並べる。レポートの文は、この番号を使う。"""
    overflow = 3 + n_events
    long = overflow + n_overflow
    return {"events": list(range(3, overflow)), "overflow": list(range(overflow, long)), "long": long,
            "details": long + 1, "unobserved": long + 2}


SPILL_DIGITS = 0        # 貯留施設の検証の溢水量の桁（整数）


def fill(v, spill):
    """最大貯留率の表示。溢れたら「溢水」。満水の直前は「100%」と丸めず、溢れていないと分かるようにする。"""
    if spill > 0:
        return "**溢水**"
    return f"{v:.1%}" if v >= 0.995 else f"{v:.0%}"


def spill_text(v):
    """貯留施設の検証の溢水量 m3 の表示（整数。adapters/report/numbers.spill_text）。"""
    return numbers.spill_text(v, SPILL_DIGITS)


DURATION_TEXT = ("1時間は、日別値表の最大1時間降水量（公表値）と、毎正時で区切った1時間雨量の最大の、大きいほう。"
                 "6〜24時間は毎正時で区切った1時間雨量の連続和の最大。")
# 公表値のない日を含む降雨の1時間に付ける印と、その説明（印を付けた行がある表の下に書く）
NO_PUBLISHED_MARK = "※"
NO_PUBLISHED_TEXT = (f"{NO_PUBLISHED_MARK} は、降雨の日に1時間の公表値のない日を含む降雨。1時間は、公表値のある日の公表値と、"
                     "毎正時で区切った1時間雨量の最大の、大きいほう（公表値のない日の最大1時間は含まない）。")


def dur(r, h, label):
    mark = NO_PUBLISHED_MARK if h == 1 and r[NO_PUBLISHED] > 0 else ""
    return f"{r[f'{h}h_mm']:.1f}{mark} / {r[f'{h}h_{label}_mm']:.1f}"


def no_published_note(rows) -> list[str]:
    """表に公表値のない日を含む降雨があれば、印の説明の行。なければ空。"""
    return [NO_PUBLISHED_TEXT, ""] if any(r[NO_PUBLISHED] > 0 for r in rows) else []


def period_text(start: str, end: str) -> str:
    """2026-08-13〜2026-08-15 → 2026年8月13〜15日。"""
    a, b = datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)
    if a == b:
        return f"{a.year}年{a.month}月{a.day}日"
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.year}年{a.month}月{a.day}〜{b.day}日"
    if a.year == b.year:
        return f"{a.year}年{a.month}月{a.day}日〜{b.month}月{b.day}日"
    return f"{a.year}年{a.month}月{a.day}日〜{b.year}年{b.month}月{b.day}日"


def date_from_text(d: str) -> str:
    """2008-01-01 → 2008年、それ以外は 2008年6月1日。"""
    x = datetime.date.fromisoformat(d)
    return f"{x.year}年" if (x.month, x.day) == (1, 1) else f"{x.year}年{x.month}月{x.day}日"


def ratio_text(r: float) -> str:
    """容量に対する割合の言い方（0.5 → 半分）。"""
    return "半分" if r == 0.5 else f"{r:.0%}"


def spans_text(spans: list[tuple[str, str]], year: int) -> str:
    """欠測の時間の表示。区間の終わりが0時なら前日の24:00と書く。year と違う年だけ年を書く。"""
    def day(t: datetime.datetime) -> str:
        return f"{t.month}/{t.day}" if t.year == year else f"{t.year}/{t.month}/{t.day}"

    out = []
    for a, b in spans:
        ta, tb = datetime.datetime.fromisoformat(a), datetime.datetime.fromisoformat(b)
        if tb.hour == 0 and tb.minute == 0:
            tb_day, tb_hm = tb - datetime.timedelta(days=1), "24:00"
        else:
            tb_day, tb_hm = tb, f"{tb.hour}:{tb.minute:02d}"
        head = f"{day(ta)} {ta.hour}:{ta.minute:02d}〜"
        out.append(head + (tb_hm if tb_day.date() == ta.date() else f"{day(tb_day)} {tb_hm}"))
    return "、".join(out)


def unobserved_text(study: StorageStudy, name: str, start: str, end: str, source: str, margin: int | None = None) -> str:
    return spans_text(study.unobserved(name, start, end, source, margin), int(start[:4]))


def impervious_ratio(study: StorageStudy) -> float:
    """流出係数 f から不浸透面積率を逆算する（f = 浸透域 + (不浸透域 - 浸透域) × 率）。範囲外ならエラー。"""
    ro = study.guideline.runoff()
    x = (study.cond.f - ro["pervious"]) / (ro["impervious"] - ro["pervious"])
    if not 0 <= x <= 1:
        raise ValueError(f"流出係数 f＝{study.cond.f} は手引の流出率（{ro['pervious']}〜{ro['impervious']}）の範囲外で、"
                         f"レポートに書く不浸透面積率が求められません。study.json の conditions.f を {ro['pervious']}〜{ro['impervious']} にしてください。")
    return x


def event_detail(study: StorageStudy, name, start, end, source, title):
    mm10, src, q, r, designed = study.trace(name, start, end, source)
    vplan = designed.capacity
    hwl = designed.facility.pond.hwl
    tc = study.cond.tc_min
    ph = 60 // tc                        # 1時間の節点の数
    ipk = int(np.argmax(r.h))            # 節点番号（節点 n は開始から n×tc 分）
    # 表と図は最大水位の前36時間〜後12時間（1時間ごと）
    h0 = max(0, (ipk - 36 * ph) // ph * ph)
    h1 = min(len(q) - 1, h0 + 48 * ph)
    t0 = datetime.datetime.fromisoformat(start)
    tpk = t0 + datetime.timedelta(minutes=tc * ipk)
    out = [f"### {title}", "",
           "| 項目 | 値 |", "| --- | ---: |",
           f"| 計画容量 | {vplan:.0f} m3 |",
           f"| 最大流入量 | {q.max():.4f} m3/s |",
           f"| 最大放流量 | {r.out.max():.4f} m3/s |",
           f"| 最大貯留量 | {r.vmax:.0f} m3（{fill(r.vmax / vplan, 0)}） |",
           f"| 溢水量 | {spill_text(r.spill)} m3 |"]
    miss = unobserved_text(study, name, start, end, source)
    if miss:
        out.append(f"| 欠測の時間 | {miss} |")
    out += ["",
           f"降雨 {start}〜{end}、総量 {mm10.sum():.1f} mm、出所 {src}。最大水位の時刻 {tpk:%m/%d %H:%M}。", ""]
    hourly = mm10[: (len(mm10) // 6) * 6].reshape(-1, 6).sum(1)      # mm10 は10分ごとの雨量
    a, b = h0 // ph, min(len(hourly), h1 // ph)
    out += ["1時間ごとの雨量（mm）。最大水位の前36時間〜後12時間。", "",
            "| 時間帯 | +0 | +1 | +2 | +3 | +4 | +5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k in range(a, b, 6):
        tk = t0 + datetime.timedelta(hours=k)
        vals = [f"{hourly[j]:.1f}" if j < b else "" for j in range(k, k + 6)]
        out.append(f"| {tk:%m/%d %H時}〜 | " + " | ".join(vals) + " |")
    out.append("")
    idx = list(range(h0, h1 + 1, ph))
    times = [t0 + datetime.timedelta(minutes=tc * i) for i in idx]
    labels = xychart_labels([f"{tk:%d日%H時}" if tk.hour % 6 == 0 else None for tk in times])
    allow = study.allowable_q(study.guideline.district_of(name))
    qh = [float(q[max(0, i - ph + 1):i + 1].mean()) for i in idx]    # 流入量は1時間の平均（節点を拾うと値が飛ぶ）
    xaxis = "    x-axis [" + ", ".join(f'"{x}"' for x in labels) + "]"
    out += ["```mermaid", INIT, "xychart-beta",
            f'    title "{name} {start} 流入量・放流量・許容放流量"', xaxis,
            f'    y-axis "m3/s" 0 --> {max(0.05, max(qh) * 1.1):.3f}',
            "    line [" + ", ".join(f"{v:.4f}" for v in qh) + "]",
            "    line [" + ", ".join(f"{r.out[i]:.4f}" for i in idx) + "]",
            "    line [" + ", ".join(f"{allow:.4f}" for _ in idx) + "]",
            "```", "",
            "赤が流入量（1時間の平均）、青が放流量、灰が許容放流量。", "",
            "```mermaid", INIT1, "xychart-beta",
            f'    title "{name} {start} 貯留率（貯留量÷計画容量）"', xaxis,
            '    y-axis "貯留率" 0 --> 1.2',
            "    line [" + ", ".join(f"{r.h[i] / hwl:.3f}" for i in idx) + "]",
            "```", "",
            "1.0 で満水。それを超えた分は溢れる（図では1.0で頭打ち）。", ""]
    return out


def sec_ponds(study: StorageStudy, hayami: list[HayamiResult]):
    """hayami は早見表の照合結果（流出率が f と上限のマス、許容放流比流量が下限値）。早見表がなければ空。"""
    c, g = study.cond, study.guideline
    own = {r.cell.district: r for r in hayami if r.cell.f == c.f}
    o = ["## 1 モデル施設", "",
         f"開発面積{c.area_ha:g}ha、柱状の貯留施設（深さ{c.depth}m）、自然放流（正方形のオリフィス、下端は池底）、浸透施設なし、"
         f"流出係数 f＝{c.f}（不浸透面積率{impervious_ratio(study):.0%}）。"
         f"許容放流比流量は手引き {g.cite('qc_min')} の下限値。",
         f"池の面積とオリフィスの寸法は、{study.guideline.district_term()}の{study.plan_label}計画降雨（後方集中型24時間、{c.tc_min}分ブロック）で、"
         f"最大水位が満水位（HWL＝{c.depth}m）にちょうど達する寸法を求め、溢れない向きに丸めた"
         "（オリフィスは1mm単位で切り捨て、池の面積はそのオリフィスで求め直して1m²単位で切り上げ）。"]
    if own:
        cell = next(iter(own.values())).cell
        o += ["早見表の列は、同じ計算を早見表の条件（池の面積を開発面積の"
              f"{g.hayami_conditions()['pond_area_ratio']:.0%}、深さを必要容量に合わせる）で行い、"
              f"手引き {g.cite('hayami')} の値（不浸透面積率{cell.impervious_pct}%、流出率{cell.f:.2f}、設計浸透強度0mm/h）と比べたもの。", "",
              f"| {study.guideline.district_term()} | Qc m3/s/ha | 計画容量 m3 | 池の面積 m² | オリフィス m | 早見表の条件での計算 m3/ha | 早見表 m3/ha |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    else:
        o += ["", f"| {study.guideline.district_term()} | Qc m3/s/ha | 計画容量 m3 | 池の面積 m² | オリフィス m |", "| --- | ---: | ---: | ---: | ---: |"]
    for d in g.districts():
        fac = study.facility(d)
        row = (f"| {d} | {g.qc_min(d)} | {fac.capacity:.0f} | {fac.facility.pond.area:.0f} | "
               f"{fac.facility.outlet.size:.3f} |")
        if own:
            h = own.get(d)
            row += f" {h.calc:.0f} | {h.cell.volume:.0f} |" if h else " — | — |"
        o.append(row)
    o.append("")
    if own:   # f の行を照合したときだけ書く（f の行がないのに「も含めた」と書かない）
        others = sorted({r.cell.f for r in hayami} - {c.f})
        also = "".join(f" f＝{f:g} の行も含めた" for f in others)
        worst = math.ceil(max(abs(r.rel_err) for r in hayami) * 1000 - 1e-9) / 10   # 0.1% 単位で切り上げ
        o += [f"手法を確認するため、{also.lstrip()}{len(hayami)}件すべてで早見表と照合し、±{worst:.1f}% 以内を確認した。", ""]
    return o


def sec_design_rain(study: StorageStudy, storms: dict[str, np.ndarray]):
    tc = study.cond.tc_min
    ph = 60 // tc
    heads = [f"0〜{tc}分"] + [f"{k * tc}〜{(k + 1) * tc}" for k in range(1, ph)]
    o = [f"## 2 計画降雨（{study.plan_label}、後方集中型24時間）", "",
         f"手引き {study.guideline.cite('design_storm')} の方法で作成した。{study.guideline.cite('formulas')} の{study.plan_label}の式から "
         f"t＝{tc}, {2 * tc} … {24 * 60}分の累加雨量を求め、その差分を小さい順に並べる（{tc}分ブロック×{24 * 60 // tc}個）。単位は mm/hr。",
         f"縦に開始からの時刻を表し、横方向に分を表す。1行で1時間（{ph}ブロック）を記載した。", ""]
    for d in study.guideline.districts():
        f = study.guideline.formula(d, study.cond.return_period)
        h = storms[d]
        o += [f"### {d}{study.guideline.district_term()} r＝{f.a}/(t^({Fraction(f.n).limit_denominator(10)})+{f.b})", "",
              f"24時間雨量 {h.sum() / (60 / tc):.1f} mm、最後の{tc}分 {h[-1]:.1f} mm/hr。", "",
              "| 時 | " + " | ".join(heads) + " |",
              "| ---: | " + " | ".join("---:" for _ in heads) + " |"]
        for k in range(24):
            o.append(f"| {k} | " + " | ".join(f"{v:.3f}" for v in h[k * ph:(k + 1) * ph]) + " |")
        o.append("")
    return o


def sec_event(study: StorageStudy, df: pd.DataFrame, event: dict, label: str, no: int):
    days = len(pd.date_range(event["start"], event["end"]))
    o = [f"## {no} {event['name']}（{period_text(event['start'], event['end'])}）{len(df)}地点", "",
         f"{len(df)}地点の10分値" + ("" if study.cond.tc_min == 10 else f"（{study.cond.tc_min}分ごとに合計）")
         + f"を用いて、それぞれの{study.guideline.district_term()}のモデル施設で厳密計算を行った。",
         f"継続時間ごとの雨量は「実測 / {label}」（mm）と表記した。{label} は、その地点が属する"
         f"{study.guideline.district_term()}の{label}の降雨強度式から求めた、"
         "同じ継続時間の雨量である。",
         DURATION_TEXT,
         "欠測の時間は、期間内の10分値のもの。欠測は0mm として計算した。", "",
         f"| 地点 | {study.guideline.district_term()} | {days}日合計 mm | 1時間 | 6時間 | 12時間 | 24時間 | 計画容量 m3 | 最大貯留率 | 溢水量 m3 | 欠測の時間 |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for _, r in df.iterrows():
        o.append(f"| {r['地点']} | {r['地区']} | {r[f'{days}日合計_mm']} | {dur(r, 1, label)} | {dur(r, 6, label)} | "
                 f"{dur(r, 12, label)} | {dur(r, 24, label)} | "
                 f"{r['計画容量_m3']} | {fill(r['最大貯留率'], r['溢水量_m3'])} | {spill_text(r['溢水量_m3'])} | "
                 f"{unobserved_text(study, r['地点'], event['start'], event['end'], '10分値', 0)} |")
    o.append("")
    return o + no_published_note([r for _, r in df.iterrows()])


def sec_long(study: StorageStudy, df: pd.DataFrame, y26: pd.DataFrame, long_record: dict, partial_to: str, nums: dict):
    c, label = study.cond, study.plan_label
    starts, end_year = long_record["starts"], long_record["end_year"]
    y, m, d = (int(x) for x in partial_to.split("-"))
    lo = c.fill_ratio    # 基準雨量未満の降雨は計算していないので、表に載せる下限は基準雨量の割合と同じにする
    sc = study.screening(starts, end_year, df)
    ex = sc.iloc[0]
    left = max(study.left_after_dry(d) for d in sc["地区"].unique())
    o = [f"## {nums['long']} {long_record.get('station_kind', '')}{len(starts)}地点の長期記録", "",
         "### 計算する降雨の絞り込み", "",
         "貯留追跡計算には時間値が要るが、闇雲にすべての解析を行わず、必要な降雨を絞り込んだ。気象庁へのアクセスも低減できる。"
         "絞り込みはまず、日別値で溢れる可能性のある降雨を絞り込み、その降雨の日だけ時間値を取得して計算した。", "",
         f"1. **降雨に区切る**：日別値の日降水量を、降水量0の日が{c.dry_gap}日以上続いたところで区切る。"
         f"計画降雨で満水にした池も、雨が止んで{c.dry_gap}日で容量の{math.ceil(left * 100)}%以下まで下がるので、"
         "前の降雨の水はほとんど残らない。"
         "日別値が欠測の日は、降ったかどうかわからないので無降水の日に数えず、その日では区切らないことにした。",
         "2. **総雨量で絞り込む**：日をまたいで降り続く雨は池に貯まる量も日をまたいで増えるので、"
         "1日ごとの雨量ではなく、降雨ごとの総雨量で判断する。"
         "1降雨で池に入る水量は「流出係数 × 総雨量 × 開発面積」（1haに1mmの雨は10m3）で、これが計画容量を超えなければ溢れない。"
         "したがって、総雨量が「計画容量 ÷ (流出係数 × 10 × 開発面積 ha)」（容量に相当する雨量）に満たない降雨は溢れない。"
         f"取りこぼしを防ぐ余裕として、その{ratio_text(lo)}を基準雨量とした（基準雨量に満たない降雨は最大貯留率が{lo:.0%}未満）。"
         f"例えば{ex['地点']}（{ex['地区']}{study.guideline.district_term()}）は、流出係数 {c.f}、開発面積 {c.area_ha:g}ha で、"
         f"容量に相当する雨量は {ex['計画容量_m3']:.0f} ÷ ({c.f} × 10 × {c.area_ha:g}) ＝ {ex['基準雨量_mm'] / lo:.1f} mm、"
         f"基準雨量は {lo:g} × {ex['基準雨量_mm'] / lo:.1f} ＝ {ex['基準雨量_mm']:.1f} mm。",
         "3. **時間値で計算する**：基準雨量以上となる降雨の日だけ時間値を取得し、貯留追跡計算を行った。"
         f"時間値は1時間の雨量を6区間へ均等に配分するので、{date_from_text(c.tenmin_from)}以降で"
         f"最大貯留率{c.tenmin_fill:.0%}以上の降雨は、10分値で計算し直した値に置き換えた（出所の列）。"
         + ("" if c.tc_min == 10 else f"追跡は、どちらも{c.tc_min}分ごとに合計した雨量で行った。"), "",
         f"| 地点 | 期間 | 対象期間の日数 | 降水のあった日 | 降雨 | 容量に相当する雨量 mm | 基準雨量 mm | 計算した降雨 | "
         f"計算した降雨の日数 | 最大貯留率{lo:.0%}以上 | 溢水 |",
         "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, s in sc.iterrows():
        o.append(f"| {s['地点']} | {s['開始年']}〜{s['終了年']}年 | {s['対象期間の日数']:,} | {s['降水のあった日']:,} | "
                 f"{s['降雨']:,} | {s['容量に相当する雨量_mm']:.1f} | {s['基準雨量_mm']:.1f} | {s['計算した降雨']} | "
                 f"{s['計算した降雨の日数']:,} | {s[f'最大貯留率{lo:.0%}以上']} | {s['溢水']} |")
    o += ["",
         f"各地点の表に載せるのは最大貯留率が{lo:.0%}以上の降雨とした。",
         f"継続時間ごとの雨量は「実測 / {label}」（mm）と表記した。" + DURATION_TEXT,
         f"{y}年（{m}月{d}日まで）は同じ基準雨量で選んだ降雨を10分値で計算し、各地点の表の最後に載せた。"
         "長期記録の回数には含めない。",
         f"欠測の時間は、期間内の欠測と、前後{c.dry_gap}日のうち日別値が欠測の日の欠測。扱いと一覧は{nums['unobserved']}に示す。", ""]
    for n, y0 in starts.items():
        e = df[df["地点"] == n]
        yrs_half = e[e["最大貯留率"] >= lo]["開始"].str[:4].nunique()
        o += [f"### {n}（{study.guideline.district_of(n)}{study.guideline.district_term()}、{y0}〜{end_year}年、{end_year - y0 + 1}年）", "",
              f"溢水 {int((e['溢水量_m3'] > 0).sum())} 回。最大貯留率{lo:.0%}以上の降雨があった年 {yrs_half} 年。", "",
              "| 開始 | 終了 | 総量 mm | 1時間 | 6時間 | 12時間 | 24時間 | 最大貯留率 | 溢水量 m3 | 出所 | 欠測の時間 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
        rows = [r for _, r in e[e["最大貯留率"] >= lo].sort_values("開始").iterrows()]
        rows += [r for _, r in y26[(y26["地点"] == n) & (y26["最大貯留率"] >= lo)].iterrows()]
        for r in rows:
            o.append(f"| {r['開始']} | {r['終了']} | {r['総量_mm']} | {dur(r, 1, label)} | {dur(r, 6, label)} | "
                     f"{dur(r, 12, label)} | {dur(r, 24, label)} | "
                     f"{fill(r['最大貯留率'], r['溢水量_m3'])} | {spill_text(r['溢水量_m3'])} | {r['出所']} | "
                     f"{unobserved_text(study, n, r['開始'], r['終了'], r['出所'])} |")
        o.append("")
        o += no_published_note(rows)
    return o

def sec_overflow_levels(study: StorageStudy, df: pd.DataFrame, event: dict, no: int, table_no: int):
    """降雨の表（table_no 節）で溢れた地点の、貯留施設の水位の推移。どの地点も同じ横軸（降雨の開始を0時間とし、
    降雨の期間とその後の24時間、1時間ごと）で、上に1時間ごとの雨量、下に水位（1時間ごとの最大）を描く。
    水位の図には、WebGUI の試算の比較と同じく、計画降雨を与えたときの水位を参考に重ねる。"""
    c, g, label = study.cond, study.guideline, study.plan_label
    tc, ph = c.tc_min, 60 // c.tc_min
    start, end = event["start"], event["end"]
    hours = len(pd.date_range(start, end)) * 24 + 24
    t0 = datetime.datetime.fromisoformat(start)
    over = df[df["溢水量_m3"] > 0].sort_values("溢水量_m3", ascending=False)
    n = len(over)
    o = [f"## {no} {event['name']}で溢れた{n}地点の貯留施設の水位", ""]
    if over.empty:
        return o + [f"{table_no}の表で溢れた地点はない。", ""]
    table, charts = [], []
    wet_h, rise_24, low_after, full_at = [], [], [], []   # 地点ごとの、雨のあった時間・24時間までの最大水位・その後満水位までの最小水位・満水位の節点
    for _, row in over.iterrows():
        name, d = row["地点"], row["地区"]
        mm10, _, _, r, designed = study.trace(name, start, end, "10分値")
        hwl = designed.facility.pond.hwl
        plan = study.planned(d)
        pr = study.run(d, plan, hours * ph - len(plan.rain)).routing
        if min(len(r.h), len(pr.h)) <= hours * ph:
            raise ValueError(f"{name}: 追跡の結果が図の横軸（{hours}時間）より短いので、図を描けません。study.json の conditions.tail_hours を大きくしてください。")
        full = np.flatnonzero(r.h >= hwl)
        tf = t0 + datetime.timedelta(minutes=tc * int(full[0]))
        wet = np.flatnonzero(mm10 > 0)                                     # mm10 は10分ごとの雨量
        wet_h.append((wet[-1] + 1 - wet[0]) / 6)
        full_at.append(int(full[0]))
        if full[0] > 24 * ph:
            rise_24.append(r.h[:24 * ph + 1].max())
            low_after.append(r.h[24 * ph:full[0]].min())
        table.append(f"| {name} | {tf.month}/{tf.day} {tf:%H:%M}（{full[0] * tc / 60:.1f}時間目） | "
                     f"{len(full) * tc / 60:.1f} | {spill_text(row['溢水量_m3'])} |")
        obs = np.zeros(hours * 6)
        obs[:min(len(mm10), hours * 6)] = mm10[:hours * 6]
        rain_h = obs.reshape(-1, 6).sum(1)
        lv_r, lv_p = (x[1:hours * ph + 1].reshape(-1, ph).max(1) for x in (r.h, pr.h))   # 1時間ごとの最大（ピークを欠かさない）
        ks = range(1, hours + 1)                                           # k 時間目（k-1〜k 時間の雨量の合計と、水位の最大）
        xaxis = ('    x-axis "開始からの時間 h" ['
                 + ", ".join(f'"{x}"' for x in xychart_labels([str(k) if k % 6 == 0 else None for k in ks])) + "]")
        charts += [f"### {name}（{d}{g.district_term()}）", "",
                   "```mermaid", INIT_RAIN, "xychart-beta",
                   f'    title "{name} 1時間ごとの雨量"', xaxis,
                   f'    y-axis "mm" 0 --> {math.ceil(rain_h.max() * 1.1 / 10) * 10}',
                   *xychart_bars([f"{v:.1f}" for v in rain_h]),
                   "```", "",
                   "```mermaid", INIT_LEVEL, "xychart-beta",
                   f'    title "{name} 貯留施設の水位"', xaxis,
                   f'    y-axis "水位 m" 0 --> {hwl * 1.2:g}',
                   "    line [" + ", ".join(f"{v:.3f}" for v in lv_r) + "]",
                   "    line [" + ", ".join(f"{v:.3f}" for v in lv_p) + "]",
                   "    line [" + ", ".join(f"{hwl:.3f}" for _ in ks) + "]",
                   "```", ""]

    def span(v, fmt, unit=""):
        a, b = fmt(min(v)), fmt(max(v))
        return f"{a}{unit}" if a == b else f"{a}〜{b}{unit}"

    def meters(v):
        return span(v, lambda x: f"{x:.2f}", " m")

    def when(v):
        a, b = (t0 + datetime.timedelta(minutes=tc * i) for i in (min(v), max(v)))
        head = f"{a.month}/{a.day} {a:%H:%M}"
        return head if a == b else head + "〜" + (f"{b:%H:%M}" if a.date() == b.date() else f"{b.month}/{b.day} {b:%H:%M}")

    each = f"{n}地点とも、" if n > 1 else ""
    reach = f"{when(full_at)}（{span(full_at, lambda i: f'{i * tc / 60:.0f}')}時間目）"
    wet_text = f"降り始めから降り終わりまでは{span(wet_h, lambda v: f'{v:.0f}')}時間だった。"
    if len(rise_24) == n:     # どの地点も24時間を過ぎてから満水位に達した（最初の24時間の水位とその後の最小を書ける）
        text = (f"{each}最初の24時間の雨で水位は{meters(rise_24)}まで上がり、その後も{meters(low_after)}を下回らないまま、"
                f"{reach}の雨で満水位（{c.depth} m）に達した。" + wet_text)
    else:
        text = f"{each}{reach}に満水位（{c.depth} m）に達した。" + wet_text
    o += [f"{table_no}の表で溢れた地点の、貯留施設の水位の推移を示す。", text, "",
          "満水の時間は、水位が満水位にあった時間の合計。", "",
          "| 地点 | 満水位に達した時刻 | 満水の時間 h | 溢水量 m3 |",
          "| --- | --- | ---: | ---: |"] + table
    return o + ["",
                "図は地点ごとに、上が1時間ごとの雨量、下が水位（1時間ごとの最大）。"
                f"水位の赤は実測、青は参考として{label}計画降雨を与えたときの水位、灰は満水位。"
                f"横軸は降雨の開始（{t0.month}/{t0.day} {t0.hour}時）からの時間で、どの地点も{hours}時間。"
                "水位は満水位で頭打ち（超えた分は溢水）。", ""] + charts


def sec_details(study: StorageStudy, df: pd.DataFrame, y26: pd.DataFrame, no: int):
    sel = pd.concat([df, y26])
    sel = sel[sel["最大貯留率"] >= DETAIL_FILL].sort_values("開始")
    o = [f"## {no} 溢水した降雨と満水に近い降雨", "",
         f"最大貯留率が{DETAIL_FILL:.0%}以上の降雨。降雨の推移表と、流入量・放流量・許容放流量のハイドログラフ、貯留率の推移。", ""]
    for _, r in sel.iterrows():
        tag = "溢水" if r["溢水量_m3"] > 0 else f"最大貯留率 {fill(r['最大貯留率'], 0)}"
        o += event_detail(study, r["地点"], r["開始"], r["終了"], r["出所"], f"{r['地点']} {r['開始']}〜{r['終了']}（{tag}）")
    return o


def sec_unobserved(study: StorageStudy, df: pd.DataFrame, y26: pd.DataFrame, long_record: dict, partial_to: str,
                   no: int):
    c = study.cond
    o = [f"## {no} 欠測のある降雨", "",
         f"期間内か前後{c.dry_gap}日に欠測がある降雨を、最大貯留率によらず示す。",
         "欠測は0mmとするが、日別値を欠測の日では区切らず、連続するものとした。", ""]
    sel = pd.concat([df, y26])
    for n, y0 in long_record["starts"].items():
        rows = []
        for _, r in sel[sel["地点"] == n].sort_values("開始").iterrows():
            miss = unobserved_text(study, n, r["開始"], r["終了"], r["出所"])
            if miss:
                rows.append(f"| {r['開始']} | {r['終了']} | {r['総量_mm']} | {fill(r['最大貯留率'], r['溢水量_m3'])} | "
                            f"{r['出所']} | {miss} |")
        skipped = study.uncomputed_near_missing(n, f"{y0}-01-01", partial_to)
        o += [f"### {n}", ""]
        if rows:
            o += ["計算した降雨。", "",
                  "| 開始 | 終了 | 総量 mm | 最大貯留率 | 出所 | 欠測の時間 |",
                  "| --- | --- | ---: | ---: | --- | --- |"] + rows + [""]
        if skipped:
            o += [f"計算しなかった降雨（観測できた総量が基準雨量 {skipped[0]['基準雨量_mm']} mm 未満）で、"
                  f"期間内か前後{c.dry_gap}日に日別値が欠測の日があるもの。", "",
                  "| 開始 | 終了 | 観測できた総量 mm | 日別値が欠測の日 |",
                  "| --- | --- | ---: | --- |"]
            for s in skipped:
                o.append(f"| {s['開始']} | {s['終了']} | {s['観測できた総量_mm']} | {'、'.join(s['日別値が欠測の日'])} |")
            o.append("")
        if not rows and not skipped:
            o += ["なし。", ""]
    return o


def observation_note(record: dict) -> list[str]:
    """使った観測値に準正常値・資料不足値が含まれていれば、その数を1文で示す（計算に使った値と、公表値との照合に使った
    年ごとの値（annual）を分けて数える）。なければ空。"""
    def counts(ranges):
        n = {k: sum(r[k] for r in ranges) for k in ("準正常値", "資料不足値")}
        return "、".join(p for p in (f"準正常値（`)`）が {n['準正常値']:,} 個" if n["準正常値"] else "",
                                     f"資料不足値（`]`）が {n['資料不足値']:,} 個" if n["資料不足値"] else "") if p)
    calc = counts([r for r in record["範囲"] if r["種類"] != "annual"])
    check = counts([r for r in record["範囲"] if r["種類"] == "annual"])
    parts = ([f"計算に使った気象庁の観測値には{calc}"] if calc else []) + ([f"公表値との照合に使った年ごとの値には{check}"] if check else [])
    return [f"{'、'.join(parts)}含まれる。", ""] if parts else []


def render(study: StorageStudy, conf: dict, hayami: list[HayamiResult], storms: dict[str, np.ndarray],
           events: list[tuple[dict, pd.DataFrame]], final: pd.DataFrame, y26: pd.DataFrame, observations: Callable[[], dict]) -> str:
    """conf は study.json の内容。events はレポートに表を載せる降雨（events のうち "in_report": true のもの）と、
    その降雨の地点ごとの結果の組の並び。そのうち "overflow_detail": true の降雨は、溢れた地点の貯留施設の水位の節も載せる。
    observations は使った観測値の記録（app/observations.observation_record）を返す関数。本文の節も観測値を読むので、節を作った後に呼ぶ。"""
    lr, to = conf["long_record"], conf["partial_year"]["to"]
    overflow = [k for k, (ev, _) in enumerate(events) if ev.get("overflow_detail")]
    nums = section_numbers(len(events), len(overflow))
    out = sec_ponds(study, hayami) + sec_design_rain(study, storms)
    for no, (ev, df) in zip(nums["events"], events):
        out += sec_event(study, df, ev, study.plan_label, no)
    for no, k in zip(nums["overflow"], overflow):
        out += sec_overflow_levels(study, events[k][1], events[k][0], no, nums["events"][k])
    out += sec_long(study, final, y26, lr, to, nums)
    out += sec_details(study, final, y26, nums["details"]) + sec_unobserved(study, final, y26, lr, to, nums["unobserved"])
    head = [f"# {conf['title']}", "", "この検討の計算を、モデル施設・計画降雨・地点・降雨ごとに示す。", ""] + observation_note(observations())
    return "\r\n".join(head + out) + "\r\n"
