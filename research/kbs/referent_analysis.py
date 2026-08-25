#!/usr/bin/env python3
"""남측 지칭어 4시점 점유율 분석 (인용 포함/제외) + 기사단위 부트스트랩 CI.

폐쇄 개체군: 관계어(남측·남녘·남녘땅·남조선) / 국가명(한국·대한민국) / 지역명(남한) / 환유(청와대·용산)
- 인접명사 배제 규칙으로 한국전쟁·한국형 등 오탐 제거
- quote_spans 내외로 인용 포함/제외 두 버전
- 단위: 기사(ncd), 층위: 1
"""
import json
import random
import pandas as pd
from kiwipiepy import Kiwi

kiwi = Kiwi(); kiwi.load_user_dictionary('../kiwi_user_dict.tsv')
sec = pd.read_parquet('master/sections.parquet')
L1 = sec[sec.layer == 1]

VARIANT2TYPE = {}
for t, ws in {'관계어': ['남측', '남녘', '남녘땅', '남조선'],
              '국가명': ['한국', '대한민국'],
              '지역명': ['남한'],
              '환유': ['청와대', '용산']}.items():
    for w in ws:
        VARIANT2TYPE[w] = t
TYPES = ['관계어', '국가명', '지역명', '환유']
NOUNISH = ('NNG', 'NNP', 'XSN', 'XSV')   # 인접이면 복합어로 간주


def count_article(rows):
    """기사(층1 섹션들) → {type: [inc, exc]} 카운트. inc=인용포함, exc=인용제외."""
    cnt = {t: [0, 0] for t in TYPES}
    for _, r in rows.iterrows():
        t = str(r.text_raw)
        spans = [(s['start'], s['end']) for s in json.loads(r.quote_spans)]
        toks = kiwi.tokenize(t)
        for i, tk in enumerate(toks):
            if tk.form in VARIANT2TYPE and tk.tag.startswith('N'):
                # 인접명사 배제
                if i + 1 < len(toks):
                    nx = toks[i + 1]
                    if nx.start == tk.start + len(tk.form) and nx.tag in NOUNISH:
                        continue
                typ = VARIANT2TYPE[tk.form]
                inq = any(a <= tk.start < b for a, b in spans)
                cnt[typ][0] += 1              # 인용 포함
                if not inq:
                    cnt[typ][1] += 1          # 인용 제외
    return cnt


# 기사별 카운트
art_counts = {}   # (year, ncd) -> {type:[inc,exc]}
for (ncd, y), g in L1.groupby(['ncd', 'year']):
    art_counts[(y, ncd)] = count_article(g)

years = [2018, 2020, 2022, 2024]


def proportions(keys, ver):   # ver: 0=inc 1=exc
    agg = {t: 0 for t in TYPES}
    for k in keys:
        for t in TYPES:
            agg[t] += art_counts[k][t][ver]
    N = sum(agg.values())
    return ({t: agg[t] / N * 100 for t in TYPES}, N) if N else (None, 0)


def boot_ci(keys, ver, typ, B=1000):
    keys = list(keys)
    vals = []
    for _ in range(B):
        samp = [keys[int(random.random() * len(keys))] for _ in keys]
        agg = {t: 0 for t in TYPES}
        for k in samp:
            for t in TYPES:
                agg[t] += art_counts[k][t][ver]
        N = sum(agg.values())
        if N:
            vals.append(agg[typ] / N * 100)
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


random.seed(42)
for ver, name in [(0, '인용 포함 (방송 담론 총체)'), (1, '인용 제외 (KBS 자신의 언어)')]:
    print("=" * 74)
    print(f"[{name}]")
    print("=" * 74)
    print(f"{'연도':>5} {'관계어%':>8} {'국가명%':>8} {'지역명%':>8} {'환유%':>7} {'N':>6}")
    for y in years:
        keys = [k for k in art_counts if k[0] == y]
        p, N = proportions(keys, ver)
        print(f"{y:>5} {p['관계어']:>8.1f} {p['국가명']:>8.1f} {p['지역명']:>8.1f} {p['환유']:>7.1f} {N:>6}")
    # 핵심 대비: 국가명% CI + 인접연도 차이
    print("  국가명% 95% CI (부트스트랩 1000회, 기사 리샘플):")
    prev = None
    for y in years:
        keys = [k for k in art_counts if k[0] == y]
        lo, hi = boot_ci(keys, ver, '국가명')
        p, _ = proportions(keys, ver)
        print(f"    {y}: {p['국가명']:.1f}% [{lo:.1f}, {hi:.1f}]")
    print()
