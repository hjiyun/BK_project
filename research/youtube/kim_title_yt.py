#!/usr/bin/env python3
"""김정은 호칭 — 직함 부착률. KBS 4개 연도(인용 포함/제외) + 유튜브 2024.

referent_analysis_yt.py 와 같은 틀: Kiwi 토큰(사용자 사전), KBS 층1, quote_spans 내외 두 버전,
기사·영상 단위 부트스트랩 95% CI.
분류 (언급 1회당 하나):
  직함  : 앞뒤 2토큰 안에 TITLES (김정은 위원장 / 김정은 북한 국무위원장 / 국무위원장 김정은)
  동지  : 바로 뒤 '동지' (북한식 존칭, 주로 북한 매체 인용)
  맨이름: 그 외
※ 보고서 6.2절 '직함 부착 70→84%'의 원 코드는 남아 있지 않아 규칙이 같다고 보장할 수 없음.
"""
import json
import random
from pathlib import Path

import pandas as pd
from kiwipiepy import Kiwi

BASE = Path(__file__).parent
KBS = BASE.parent / "kbs" / "master"
kiwi = Kiwi(); kiwi.load_user_dictionary(str(BASE.parent / "kiwi_user_dict.tsv"))

TITLES = {"위원장", "국무위원장", "총비서", "원수", "총사령관", "대원수"}
CATS = ["직함", "동지", "맨이름"]


def mentions(text):
    """[(범주, 시작위치)]"""
    toks = kiwi.tokenize(str(text))
    out = []
    for i, tk in enumerate(toks):
        if tk.form != "김정은":
            continue
        after = [t.form for t in toks[i + 1:i + 3]]
        before = [t.form for t in toks[max(0, i - 2):i]]
        if after[:1] == ["동지"]:
            c = "동지"
        elif TITLES & set(after) or TITLES & set(before):
            c = "직함"
        else:
            c = "맨이름"
        out.append((c, tk.start))
    return out


def share(rows):
    agg = dict.fromkeys(CATS, 0)
    for r in rows:
        for c in CATS:
            agg[c] += r[c]
    n = sum(agg.values())
    return ({c: agg[c] / n * 100 for c in CATS}, n) if n else (None, 0)


def ci(rows, B=2000):
    vals = []
    for _ in range(B):
        p, n = share([rows[int(random.random() * len(rows))] for _ in rows])
        if n:
            vals.append(p["직함"])
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def main():
    random.seed(42)
    out = []
    sec = pd.read_parquet(KBS / "sections.parquet")
    sec = sec[sec.layer == 1]
    for y, gy in sec.groupby("year"):
        inc, exc = [], []
        for _, g in gy.groupby("ncd"):
            ci_, ce = dict.fromkeys(CATS, 0), dict.fromkeys(CATS, 0)
            for r in g.itertuples():
                spans = [(s["start"], s["end"]) for s in json.loads(r.quote_spans)]
                for c, st in mentions(r.text_raw):
                    ci_[c] += 1
                    if not any(a <= st < b for a, b in spans):
                        ce[c] += 1
            inc.append(ci_); exc.append(ce)
        for name, rows in [("인용 포함", inc), ("인용 제외", exc)]:
            p, n = share(rows); lo, hi = ci(rows)
            out.append(dict(매체="KBS", 연도=y, 버전=name, **{c: round(p[c], 1) for c in CATS},
                            N=n, 직함CI=f"[{lo:.1f}, {hi:.1f}]"))

    v = pd.read_parquet(BASE / "master" / "videos.parquet")
    v = v[v.year == 2024]
    rows, by_ch = [], {}
    for r in v.itertuples():
        c = dict.fromkeys(CATS, 0)
        for cat, _ in mentions(r.text_clean):
            c[cat] += 1
        rows.append(c); by_ch.setdefault(r.channel, []).append(c)
    p, n = share(rows); lo, hi = ci(rows)
    out.append(dict(매체="유튜브", 연도=2024, 버전="전체", **{c: round(p[c], 1) for c in CATS},
                    N=n, 직함CI=f"[{lo:.1f}, {hi:.1f}]"))
    for ch, rs in by_ch.items():
        p, n = share(rs)
        if n:
            out.append(dict(매체=f"  └ {ch}", 연도=2024, 버전="", **{c: round(p[c], 1) for c in CATS},
                            N=n, 직함CI=""))

    res = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print("김정은 호칭 범주 점유율 (%)\n" + res.to_string(index=False))
    res.to_csv(BASE / "master" / "kim_title_kbs_yt.csv", index=False)


if __name__ == "__main__":
    main()
