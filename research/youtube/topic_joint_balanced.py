#!/usr/bin/env python3
"""합동 LDA 강건성 — 유튜브를 KBS 문서 수(147)로 채널 층화 하위표본 후 재학습.

topic_joint_2024.py 와 같은 토큰·벡터라이저·LDA 설정(K=6, K=10).
- 유튜브 147청크를 채널별 청크 수 비례(최소 1)로 무작위 추출, 시드 SEEDS개 반복
- 각 반복: 학습 후 '매체 쏠림' 지표를 산출
    KBS 집중도  = KBS 문서 평균 θ 의 상위 2토픽 합
    공유 토픽 수 = 두 매체 평균 θ 가 모두 5% 이상이고 비(큰/작은) ≤ 2 인 토픽 수
    매체 분리도 = Σ_k |KBS%_k − 유튜브%_k| / 2  (0=동일 구성, 100=완전 분리)
- 원본(불균형) 모델의 같은 지표와 비교. 상위어는 첫 시드만 출력.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

BASE = Path(__file__).parent
OUT = BASE / "master"
SEEDS = [1, 2, 3, 4, 5]


def fit(K, X):
    return LatentDirichletAllocation(n_components=K, max_iter=30, learning_method="batch",
                                     doc_topic_prior=0.1, topic_word_prior=0.01,
                                     random_state=42).fit(X)


def metrics(docs, K, show=False):
    vec = CountVectorizer(min_df=5, max_df=0.35, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(docs.text); vocab = np.array(vec.get_feature_names_out())
    m = fit(K, X)
    th = m.transform(X) * 100
    is_k = (docs.media == "KBS").values
    k, y = th[is_k].mean(0), th[~is_k].mean(0)
    shared = int(sum(1 for a, b in zip(k, y) if min(a, b) >= 5 and max(a, b) / min(a, b) <= 2))
    if show:
        for t in np.argsort(-(k - y)):
            print(f"    KBS {k[t]:5.1f} / YT {y[t]:5.1f}  {' · '.join(vocab[np.argsort(m.components_[t])[::-1][:10]])}")
    return dict(KBS집중도=round(np.sort(k)[-2:].sum(), 1), 공유토픽=shared,
                매체분리도=round(np.abs(k - y).sum() / 2, 1))


def main():
    docs = pd.read_parquet(OUT / "_joint2024_tokens.parquet")    # topic_joint_2024.py 가 생성
    kbs, yt = docs[docs.media == "KBS"], docs[docs.media == "유튜브"]
    n = len(kbs)
    quota = (yt.channel.value_counts() / len(yt) * n).round().clip(lower=1).astype(int)
    print(f"KBS {n} / 유튜브 하위표본 {quota.sum()} (채널별 {quota.to_dict()})")

    rows = []
    for K in [6, 10]:
        rows.append(dict(K=K, 표본="원본(불균형)", **metrics(docs, K)))
        for s in SEEDS:
            sub = pd.concat([g.sample(quota[ch], random_state=s) for ch, g in yt.groupby("channel")])
            show = s == SEEDS[0]
            if show:
                print(f"\n[K={K} 균형표본 seed={s} 토픽 (KBS−YT 차이 순)]")
            rows.append(dict(K=K, 표본=f"균형 seed{s}", **metrics(pd.concat([kbs, sub]), K, show)))
    res = pd.DataFrame(rows)
    summ = res[res.표본 != "원본(불균형)"].groupby("K")[["KBS집중도", "공유토픽", "매체분리도"]] \
        .agg(["mean", "min", "max"]).round(1)
    pd.set_option("display.width", 200)
    print("\n" + res.to_string(index=False))
    print("\n[균형표본 요약]\n" + summ.to_string())
    res.to_csv(OUT / "topic_joint2024_balanced.csv", index=False)


if __name__ == "__main__":
    main()
