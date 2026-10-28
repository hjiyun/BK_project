#!/usr/bin/env python3
"""합동 LDA(2024 KBS+유튜브)의 K 선택 근거 — 응집도 외에 안정성·균형표본 응집도.

같은 토큰·벡터라이저·LDA 설정(topic_joint_2024.py). K ∈ {6, 8, 10, 12}.
- UMass(원본): topic_joint_2024.py 와 동일
- 안정성: random_state 5개로 재적합, 시드 쌍마다 토픽-단어 분포를 헝가리안 매칭한 코사인 유사도의 평균
  (1에 가까울수록 시드에 무관하게 같은 토픽이 나옴). 쌍별 최소 매칭 유사도 평균도 보고(가장 불안정한 토픽).
- UMass(균형): 유튜브를 채널 비례 147청크로 하위표본(topic_joint_balanced.py 와 같은 추출), 시드 5회 평균.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from topic_joint_2024 import umass

BASE = Path(__file__).parent
OUT = BASE / "master"
KS = [6, 8, 10, 12]
SEEDS = [0, 1, 2, 3, 4]


def vectorize(docs):
    vec = CountVectorizer(min_df=5, max_df=0.35, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(docs.text)
    Xb = (X > 0).astype(int)
    return X, Xb, np.asarray(Xb.sum(axis=0)).ravel()


def fit(K, X, seed):
    return LatentDirichletAllocation(n_components=K, max_iter=30, learning_method="batch",
                                     doc_topic_prior=0.1, topic_word_prior=0.01,
                                     random_state=seed).fit(X)


def stability(models):
    means, mins = [], []
    for a, b in combinations(models, 2):
        A = a.components_ / np.linalg.norm(a.components_, axis=1, keepdims=True)
        B = b.components_ / np.linalg.norm(b.components_, axis=1, keepdims=True)
        S = A @ B.T
        r, c = linear_sum_assignment(-S)
        means.append(S[r, c].mean()); mins.append(S[r, c].min())
    return np.mean(means), np.mean(mins)


def main():
    docs = pd.read_parquet(OUT / "_joint2024_tokens.parquet")
    X, Xb, Dw = vectorize(docs)
    kbs, yt = docs[docs.media == "KBS"], docs[docs.media == "유튜브"]
    quota = (yt.channel.value_counts() / len(yt) * len(kbs)).round().clip(lower=1).astype(int)
    balanced = [vectorize(pd.concat([kbs] + [g.sample(quota[ch], random_state=s)
                                             for ch, g in yt.groupby("channel")]))
                for s in [1, 2, 3, 4, 5]]

    rows = []
    for K in KS:
        ms = [fit(K, X, s) for s in SEEDS]
        st_mean, st_min = stability(ms)
        u_full = umass(fit(K, X, 42), Xb, Dw)
        u_bal = np.mean([umass(fit(K, bx, 42), bxb, bdw) for bx, bxb, bdw in balanced])
        rows.append(dict(K=K, UMass원본=round(u_full, 3), UMass균형평균=round(u_bal, 3),
                         안정성평균=round(st_mean, 3), 안정성최저토픽=round(st_min, 3)))
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    print("\n" + res.to_string(index=False))
    res.to_csv(OUT / "topic_joint2024_select_k.csv", index=False)


if __name__ == "__main__":
    main()
