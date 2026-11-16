#!/usr/bin/env python3
"""연관어(공기어) 네트워크 — 7.13 회의 6-1 방법론. 핵심어의 담론적 맥락을 코퍼스별로 비교.

코퍼스 4개: KBS 2018 · KBS 2024 (층1 기사 text, 인용 포함 — 토픽모델과 동일) /
           유튜브 2024 생활·경험형 · 정세·체제형 (sentences.parquet)
- 단위: 문장. 문장 내 명사 집합(중복 제거)의 동시출현. 전처리는 topic_joint_2024.toks 와 동일
- 연관도: NPMI(문장 단위). 연관어는 공기 ≥ MIN_CO, 문장 빈도 ≥ min_df(n), NPMI 상위
- KBS는 인용 출처 표기("[이름/직함 :")를 지워 인물명 오염 방지(인용 내용은 유지)
- 맥락 비교: 핵심어별 PPMI 벡터(공통 어휘) 코사인, 상위 TOPN 연관어 자카드 — 코퍼스 쌍마다
- 네트워크: 코퍼스별 고빈도 명사 NET_N개, NPMI ≥ NET_MIN 간선 → GEXF(Gephi) + 커뮤니티(Louvain)
출력: master/cooc_assoc.csv, master/cooc_context_sim.csv, master/cooc_net_<코퍼스>.gexf, master/cooc_communities.csv
"""
import collections
import itertools
import math
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from topic_joint_2024 import toks

BASE = Path(__file__).parent
OUT = BASE / "master"
KBS = BASE.parent / "kbs" / "master"
KEYWORDS = ["김정은", "대한민국", "한국", "주민", "통일", "러시아", "탈북민", "미국"]
MIN_CO, TOPN = 5, 15
# NPMI 의 희귀어 편향 보정: 연관어(상대 단어)의 문장 빈도 하한 = max(10, 문장수/1000) — 코퍼스 크기 비례
# KBS 인용 출처 표기 "[이름/직함 : " 제거(인용 내용은 유지) — build_master.QUOTE_RE 의 출처부
ATTR_RE = re.compile(r'\[[^\]"“]{1,60}?\s*[:：]\s*(?=["“])')
NET_N, NET_MIN = 120, 0.15


def kbs_sents(year):
    art = pd.read_parquet(KBS / "articles.parquet")
    for t in art[art.year == year].text:
        t = ATTR_RE.sub(" ", str(t))
        for s in re.split(r"(?<=[다요까])[.?!]\s*|\n+", str(t)):
            if len(s.strip()) >= 5:
                yield s


def corpora():
    cache = OUT / "_cooc_sents.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        return {c: [set(x.split()) for x in g.nouns] for c, g in df.groupby("corpus", sort=False)}
    rows = [("KBS 2018", " ".join(set(toks(s)))) for s in kbs_sents(2018)]
    rows += [("KBS 2024", " ".join(set(toks(s)))) for s in kbs_sents(2024)]
    sen = pd.read_parquet(OUT / "sentences.parquet").merge(
        pd.read_parquet(OUT / "videos.parquet")[["video_id", "year", "genre"]])
    sen = sen[sen.year == 2024]
    for g in ["생활·경험형", "정세·체제형"]:
        rows += [(f"유튜브 {g}", " ".join(set(toks(s)))) for s in sen[sen.genre == g].sent]
    df = pd.DataFrame(rows, columns=["corpus", "nouns"])
    df = df[df.nouns.str.len() > 0]
    df.to_parquet(cache, index=False)
    return corpora()


def stats(sents):
    n = len(sents)
    df = collections.Counter(w for s in sents for w in s)
    co = collections.Counter()
    for s in sents:
        for a, b in itertools.combinations(sorted(s), 2):
            co[(a, b)] += 1
    return n, df, co


def npmi(n, fa, fb, fab):
    if fab == 0:
        return -1.0
    p = fab / n
    return math.log(p / (fa / n * fb / n)) / -math.log(p)


def min_df(n):
    return max(10, round(n / 1000))


def assoc(n, df, co, kw):
    out = {}
    for (a, b), c in co.items():
        if kw in (a, b) and c >= MIN_CO:
            o = b if a == kw else a
            if df[o] >= min_df(n):
                out[o] = (npmi(n, df[kw], df[o], c), c)
    return out


def main():
    C = corpora()
    S = {c: stats(s) for c, s in C.items()}
    for c, (n, df, _) in S.items():
        print(f"{c}: 문장 {n:,} · 연관어 최소빈도 {min_df(n)} · 어휘 {len(df):,} · " + " ".join(f"{k}{df[k]}" for k in KEYWORDS))

    # 1) 핵심어 연관어
    rows, A = [], {}
    for c, (n, df, co) in S.items():
        for kw in KEYWORDS:
            a = assoc(n, df, co, kw); A[(c, kw)] = a
            for rank, (w, (v, cnt)) in enumerate(sorted(a.items(), key=lambda x: -x[1][0])[:TOPN], 1):
                rows.append(dict(코퍼스=c, 핵심어=kw, 핵심어문장수=df[kw], 순위=rank, 연관어=w,
                                 NPMI=round(v, 3), 공기=cnt))
    assoc_df = pd.DataFrame(rows)
    assoc_df.to_csv(OUT / "cooc_assoc.csv", index=False)

    # 2) 맥락 유사도 (코퍼스 쌍 × 핵심어)
    sims = []
    names = list(S)
    for kw in KEYWORDS:
        for c1, c2 in itertools.combinations(names, 2):
            a1, a2 = A[(c1, kw)], A[(c2, kw)]
            vocab = sorted(set(a1) | set(a2))
            v1 = np.array([max(a1.get(w, (0, 0))[0], 0) for w in vocab])
            v2 = np.array([max(a2.get(w, (0, 0))[0], 0) for w in vocab])
            cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))) if v1.any() and v2.any() else np.nan
            t1 = {w for w, _ in sorted(a1.items(), key=lambda x: -x[1][0])[:TOPN]}
            t2 = {w for w, _ in sorted(a2.items(), key=lambda x: -x[1][0])[:TOPN]}
            jac = len(t1 & t2) / len(t1 | t2) if t1 | t2 else np.nan
            sims.append(dict(핵심어=kw, 코퍼스1=c1, 코퍼스2=c2, PPMI코사인=round(cos, 3),
                             상위연관어자카드=round(jac, 3), 공통연관어=" · ".join(sorted(t1 & t2))))
    sim_df = pd.DataFrame(sims)
    sim_df.to_csv(OUT / "cooc_context_sim.csv", index=False)

    # 3) 코퍼스별 네트워크 + 커뮤니티
    comm_rows = []
    for c, (n, df, co) in S.items():
        nodes = {w for w, _ in df.most_common(NET_N)}
        G = nx.Graph()
        for w in nodes:
            G.add_node(w, freq=df[w])
        for (a, b), cnt in co.items():
            if a in nodes and b in nodes and cnt >= MIN_CO:
                v = npmi(n, df[a], df[b], cnt)
                if v >= NET_MIN:
                    G.add_edge(a, b, weight=round(v, 3), cooc=cnt)
        G.remove_nodes_from([w for w in list(G) if G.degree(w) == 0])
        comms = nx.community.louvain_communities(G, weight="weight", seed=42)
        for i, cm in enumerate(sorted(comms, key=lambda x: -sum(df[w] for w in x))):
            for w in cm:
                G.nodes[w]["community"] = i
            comm_rows.append(dict(코퍼스=c, 커뮤니티=i, 크기=len(cm),
                                  상위어=" · ".join(sorted(cm, key=lambda w: -df[w])[:12])))
        nx.write_gexf(G, OUT / f"cooc_net_{c.replace(' ', '_')}.gexf")
        print(f"  네트워크 {c}: 노드 {G.number_of_nodes()} · 간선 {G.number_of_edges()} · 커뮤니티 {len(comms)}")
    pd.DataFrame(comm_rows).to_csv(OUT / "cooc_communities.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 140)
    print("\n[핵심어 상위 연관어 (NPMI 상위 10)]")
    for kw in KEYWORDS:
        print(f"■ {kw}")
        for c in names:
            sub = assoc_df[(assoc_df.코퍼스 == c) & (assoc_df.핵심어 == kw)].head(10)
            print(f"   {c:14} (n={int(sub.핵심어문장수.iloc[0]) if len(sub) else 0}): " + " · ".join(sub.연관어))
    piv = sim_df.assign(쌍=sim_df.코퍼스1 + " ↔ " + sim_df.코퍼스2).pivot(index="쌍", columns="핵심어", values="PPMI코사인")
    print("\n[맥락 유사도 — PPMI 코사인 (1=같은 맥락)]\n" + piv[KEYWORDS].to_string())
    print("\n[커뮤니티]\n" + pd.DataFrame(comm_rows).to_string(index=False))


if __name__ == "__main__":
    main()
