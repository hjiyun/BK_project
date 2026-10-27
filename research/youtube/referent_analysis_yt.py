#!/usr/bin/env python3
"""남측 지칭어 점유율 — 유튜브 2024 vs KBS 2024 (매체 간 비교, 비교축 나).

KBS referent_analysis.py 와 동일한 폐쇄 개체군·인접명사 배제 규칙을 사용.
  관계어(남측·남녘·남녘땅·남조선) / 국가명(한국·대한민국) / 지역명(남한) / 환유(청와대·용산)
유튜브엔 quote_spans 가 없으므로 두 버전으로 대응 (KWIC 검수 근거는 referent_lexicon_yt.yaml):
  - 전체      : 모든 변이형 (≈ KBS 인용 포함)
  - 남조선제외 : 남조선은 KWIC상 거의 전부 북한 시절 발화·선전의 재현 → 제외 (≈ KBS 인용 제외)
민감도: 남쪽(방향어 혼재)을 관계어에 추가한 버전.
단위: 영상(유튜브) / 기사(KBS). CI: 영상·기사 부트스트랩 + 채널 군집 부트스트랩.
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd
from kiwipiepy import Kiwi

BASE = Path(__file__).parent
KBS = BASE.parent / "kbs" / "master"

kiwi = Kiwi(); kiwi.load_user_dictionary(str(BASE.parent / "kiwi_user_dict.tsv"))

LEXICON = {'관계어': ['남측', '남녘', '남녘땅', '남조선'],
           '국가명': ['한국', '대한민국'],
           '지역명': ['남한'],
           '환유': ['청와대', '용산']}
EXTRA = {'남쪽': '관계어'}                 # 민감도 전용
TYPES = ['관계어', '국가명', '지역명', '환유']
NOUNISH = ('NNG', 'NNP', 'XSN', 'XSV')   # 인접이면 복합어로 간주
VARIANT2TYPE = {w: t for t, ws in LEXICON.items() for w in ws} | EXTRA


def hits(text):
    """(변이형, 시작위치) 목록. 인접명사 배제 규칙은 KBS와 동일."""
    toks = kiwi.tokenize(text)
    out = []
    for i, tk in enumerate(toks):
        if tk.form in VARIANT2TYPE and tk.tag.startswith('N'):
            if i + 1 < len(toks):
                nx = toks[i + 1]
                if nx.start == tk.start + len(tk.form) and nx.tag in NOUNISH:
                    continue
            out.append((tk.form, tk.start))
    return out


def to_counts(forms, drop=(), extra=False):
    cnt = dict.fromkeys(TYPES, 0)
    for f in forms:
        if f in drop or (f in EXTRA and not extra):
            continue
        cnt[VARIANT2TYPE[f]] += 1
    return cnt


def share(rows, typ='국가명'):
    agg = dict.fromkeys(TYPES, 0)
    for c in rows:
        for t in TYPES:
            agg[t] += c[t]
    n = sum(agg.values())
    return ({t: agg[t] / n * 100 for t in TYPES}, n) if n else (None, 0)


def boot(rows, typ='국가명', B=2000):
    vals = []
    for _ in range(B):
        p, n = share([rows[int(random.random() * len(rows))] for _ in rows])
        if n:
            vals.append(p[typ])
    vals.sort()
    return vals


def boot_cluster(groups, typ='국가명', B=2000):
    """채널 군집 부트스트랩: 채널 리샘플 → 채널 내 영상 리샘플."""
    keys = list(groups)
    vals = []
    for _ in range(B):
        rows = []
        for _ in keys:
            g = groups[keys[int(random.random() * len(keys))]]
            rows += [g[int(random.random() * len(g))] for _ in g]
        p, n = share(rows)
        if n:
            vals.append(p[typ])
    vals.sort()
    return vals


def ci(vals):
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def kbs_2024():
    """KBS 2024 층1 기사별 카운트 (인용 포함/제외)."""
    sec = pd.read_parquet(KBS / "sections.parquet")
    sec = sec[(sec.layer == 1) & (sec.year == 2024)]
    inc, exc = {}, {}
    for ncd, g in sec.groupby('ncd'):
        fi, fe = [], []
        for _, r in g.iterrows():
            spans = [(s['start'], s['end']) for s in json.loads(r.quote_spans)]
            for f, st in hits(str(r.text_raw)):
                fi.append(f)
                if not any(a <= st < b for a, b in spans):
                    fe.append(f)
        inc[ncd], exc[ncd] = fi, fe
    return inc, exc


def main():
    random.seed(42)
    v = pd.read_parquet(BASE / "master" / "videos.parquet")
    v = v[v.year == 2024]
    forms = {r.video_id: [f for f, _ in hits(r.text_clean)] for r in v.itertuples()}
    chan = dict(zip(v.video_id, v.channel))
    k_inc, k_exc = kbs_2024()

    versions = {
        '전체':        dict(drop=()),
        '남조선제외':   dict(drop=('남조선',)),
        '민감도(+남쪽)': dict(drop=(), extra=True),
    }
    out = []
    for name, kw in versions.items():
        yt = {vid: to_counts(f, **kw) for vid, f in forms.items()}
        rows = list(yt.values())
        p, n = share(rows)
        lo, hi = ci(boot(rows))
        groups = {}
        for vid, c in yt.items():
            groups.setdefault(chan[vid], []).append(c)
        clo, chi = ci(boot_cluster(groups))
        ch_shares = {ch: share(g)[0]['국가명'] for ch, g in groups.items() if share(g)[1]}
        out.append(dict(매체='유튜브', 버전=name, **{t: round(p[t], 1) for t in TYPES}, N=n,
                        국가명CI=f"[{lo:.1f}, {hi:.1f}]", 채널군집CI=f"[{clo:.1f}, {chi:.1f}]",
                        채널균등평균=round(sum(ch_shares.values()) / len(ch_shares), 1)))
        if name == '전체':
            by_ch = []
            for ch, g in groups.items():
                pc, nc = share(g)
                if nc:
                    by_ch.append(dict(채널=ch, 영상=len(g), **{t: round(pc[t], 1) for t in TYPES}, N=nc))
            by_channel = pd.DataFrame(by_ch).sort_values('N', ascending=False)

    for name, src in [('인용 포함', k_inc), ('인용 제외', k_exc)]:
        rows = [to_counts(f) for f in src.values()]
        p, n = share(rows)
        lo, hi = ci(boot(rows))
        out.append(dict(매체='KBS', 버전=name, **{t: round(p[t], 1) for t in TYPES}, N=n,
                        국가명CI=f"[{lo:.1f}, {hi:.1f}]", 채널군집CI='', 채널균등평균=None))

    # 핵심 대비: 유튜브 − KBS 국가명%p 차이 (영상/기사 독립 부트스트랩)
    diffs = []
    for yv, kv, ksrc in [('전체', '인용 포함', k_inc), ('남조선제외', '인용 제외', k_exc)]:
        yrows = [to_counts(f, **versions[yv]) for f in forms.values()]
        krows = [to_counts(f) for f in ksrc.values()]
        d = [a - b for a, b in zip(boot(yrows), boot(krows))]
        d.sort()
        lo, hi = ci(d)
        est = share(yrows)[0]['국가명'] - share(krows)[0]['국가명']
        diffs.append(dict(대비=f"유튜브[{yv}] − KBS[{kv}]", 국가명차이pp=round(est, 1),
                          CI95=f"[{lo:.1f}, {hi:.1f}]"))

    res = pd.DataFrame(out)
    pd.set_option('display.width', 200)
    print("=" * 90, "\n남측 지칭어 유형별 점유율 (%), 2024\n" + "=" * 90)
    print(res.to_string(index=False))
    print("\n[채널별 — 유튜브 전체 버전]\n" + by_channel.to_string(index=False))
    print("\n[매체 간 차이 — 국가명 점유율]\n" + pd.DataFrame(diffs).to_string(index=False))

    res.to_csv(BASE / "master" / "referent_2024_yt_vs_kbs.csv", index=False)
    by_channel.to_csv(BASE / "master" / "referent_2024_by_channel.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
