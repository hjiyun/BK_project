#!/usr/bin/env python3
"""유튜브 채널 장르(생활·경험형 / 정세·체제형) × KBS 2024 — 7장 지표 재산출.

장르는 build_yt_master.py 의 CHANNELS 정의(videos.parquet genre 열).
1) 남측 지칭어 국가명%  — referent_analysis_yt.py 함수 재사용, 영상 부트스트랩 CI·차이 CI
2) 김정은 호칭 직함%    — kim_title_yt.py 함수 재사용
3) 합동 LDA 토픽 비중    — topic_joint_2024.py 산출 θ (K=6 주, K=10 보조)
4) 가중 로그오즈비       — 정세·체제형 vs KBS(같은 시사 소재), 생활·경험형 vs 정세·체제형
출력: master/genre_*.csv
"""
import collections
import random
from pathlib import Path

import pandas as pd

import kim_title_yt as kt
import logodds_yt_vs_kbs as lo
import referent_analysis_yt as ra

BASE = Path(__file__).parent
OUT = BASE / "master"
GENRES = ["생활·경험형", "정세·체제형"]
TOP = 20


def boot_diff(a, b, B=2000):
    d = sorted(x - y for x, y in zip(ra.boot(a, B=B), ra.boot(b, B=B)))
    return d[int(.025 * len(d))], d[int(.975 * len(d))]


def referent(v, k_inc, k_exc):
    forms = {r.video_id: [f for f, _ in ra.hits(r.text_clean)] for r in v.itertuples()}
    rows, sets = [], {}
    for ver, drop, kbs in [("전체", (), k_inc), ("남조선제외", ("남조선",), k_exc)]:
        krows = [ra.to_counts(f) for f in kbs.values()]
        sets[("KBS", ver)] = krows
        for g in GENRES:
            ids = v.video_id[v.genre == g]
            sets[(g, ver)] = [ra.to_counts(forms[i], drop=drop) for i in ids]
        for who in GENRES + ["KBS"]:
            p, n = ra.share(sets[(who, ver)])
            lo_, hi = ra.ci(ra.boot(sets[(who, ver)]))
            rows.append(dict(대상=who, 버전=ver if who != "KBS" else ("인용 포함" if ver == "전체" else "인용 제외"),
                             **{t: round(p[t], 1) for t in ra.TYPES}, N=n, 국가명CI=f"[{lo_:.1f}, {hi:.1f}]"))
    diffs = []
    for ver in ["전체", "남조선제외"]:
        for a, b in [("생활·경험형", "KBS"), ("정세·체제형", "KBS"), ("생활·경험형", "정세·체제형")]:
            est = ra.share(sets[(a, ver)])[0]["국가명"] - ra.share(sets[(b, ver)])[0]["국가명"]
            l, h = boot_diff(sets[(a, ver)], sets[(b, ver)])
            diffs.append(dict(버전=ver, 대비=f"{a} − {b}", 국가명차이pp=round(est, 1), CI95=f"[{l:.1f}, {h:.1f}]"))
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def kim(v):
    rows = []
    for g in GENRES:
        rs = []
        for t in v.text_clean[v.genre == g]:
            c = dict.fromkeys(kt.CATS, 0)
            for cat, _ in kt.mentions(t):
                c[cat] += 1
            rs.append(c)
        p, n = kt.share(rs); l, h = kt.ci(rs)
        rows.append(dict(대상=g, **{c: round(p[c], 1) for c in kt.CATS}, N=n, 직함CI=f"[{l:.1f}, {h:.1f}]"))
    return pd.DataFrame(rows)


def topics(chan2genre):
    out = {}
    for K in [6, 10]:
        th = pd.read_parquet(OUT / f"topic_joint2024_theta_K{K}.parquet")
        rep = pd.read_csv(OUT / f"topic_joint2024_K{K}.csv", index_col=0)
        cols = [c for c in th if c.startswith("T")]
        th["grp"] = th.channel.map(chan2genre).fillna("KBS")
        tab = th.groupby("grp")[cols].mean().T.round(1)[["KBS"] + GENRES]
        tab.insert(0, "상위어", rep["상위어"].str.split(" · ").str[:6].str.join(" · "))
        out[K] = tab
    return out


def logodds_tables(v):
    art = pd.read_parquet(lo.KBS / "articles.parquet")
    art = art[art.year == 2024]
    kbs = collections.Counter()
    for t in art.text:
        kbs.update(lo.toks(t))
    gc = {g: collections.Counter() for g in GENRES}
    for r in v.itertuples():
        gc[r.genre].update(lo.toks(r.text_clean))
    res = {}
    for a, b, an, bn in [(gc["정세·체제형"], kbs, "정세·체제형", "KBS"),
                         (gc["생활·경험형"], gc["정세·체제형"], "생활·경험형", "정세·체제형")]:
        prior, tot = lo.make_prior(a, b)
        z = lo.logodds(a, b, prior)
        df = pd.DataFrame([dict(word=w, z=round(s, 2), **{an: a[w], bn: b[w]})
                           for w, s in z.items() if tot[w] >= lo.MIN_FREQ]).sort_values("z")
        res[f"{an} vs {bn}"] = df
    return res


def main():
    random.seed(42)
    v = pd.read_parquet(OUT / "videos.parquet")
    v = v[v.year == 2024]
    chan2genre = dict(zip(v.channel, v.genre))
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 60)

    k_inc, k_exc = ra.kbs_2024()
    ref, ref_d = referent(v, k_inc, k_exc)
    print("[1] 남측 지칭어 점유율 (%)\n" + ref.to_string(index=False))
    print("\n    국가명 차이\n" + ref_d.to_string(index=False))

    km = kim(v)
    print("\n[2] 김정은 호칭 (%) — KBS 2024 직함 79.9(인용 제외)/59.4(포함)\n" + km.to_string(index=False))

    tp = topics(chan2genre)
    for K, tab in tp.items():
        print(f"\n[3] 합동 LDA K={K} 평균 토픽 비중 (%)\n" + tab.to_string())

    lt = logodds_tables(v)
    for name, df in lt.items():
        a, b = name.split(" vs ")
        print(f"\n[4] 로그오즈비 {name}\n  {a} 변별: "
              + " · ".join(df.tail(TOP).word[::-1]) + f"\n  {b} 변별: " + " · ".join(df.head(TOP).word))

    ref.to_csv(OUT / "genre_referent.csv", index=False)
    ref_d.to_csv(OUT / "genre_referent_diff.csv", index=False)
    km.to_csv(OUT / "genre_kim_title.csv", index=False)
    for K, tab in tp.items():
        tab.to_csv(OUT / f"genre_topic_K{K}.csv")
    for name, df in lt.items():
        df.to_csv(OUT / f"genre_logodds_{name.replace(' ', '_')}.csv", index=False)


if __name__ == "__main__":
    main()
