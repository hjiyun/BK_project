#!/usr/bin/env python3
"""합동 LDA(2024 KBS+유튜브) 토픽 라벨 근거 지표 — K=6(주), K=10(보조).

topic_joint_2024.py 와 같은 토큰·벡터라이저·LDA(random_state=42) 모델을 재적합해 토픽별로 산출:
- UMass   : 토픽 상위 15어 UMass 응집도 (topic_joint_2024.umass 와 같은 식, 토픽 단위)
- 안정성   : random_state 0~4 모델 각각과 헝가리안 매칭한 코사인 유사도의 평균 (1=시드 무관 재현)
- 배타성   : 상위 15어가 다른 토픽 상위 15어와 겹치지 않는 비율
- relevance: Sievert & Shirley (2014), λ=0.6 — 빈도와 토픽 특이성의 절충 (라벨 1순위 근거)
- FREX     : 빈도 백분위와 배타성 백분위의 조화평균(w=0.5) — 토픽 고유어 (라벨 2순위 근거)
- 대표 문서 : θ 최고 KBS 기사·유튜브 영상 제목
라벨 규칙(LABELS): relevance·FREX 상위어에 공통으로 나오는 개념 + 대표 문서로 확인한 소재.
  안정성 < 0.5 이면 '불안정' 표시(해석 주의). 출력: master/topic_labels_K{K}.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

BASE = Path(__file__).parent
OUT = BASE / "master"
KBS = BASE.parent / "kbs"
LAMBDA, W_FREX, TOPN = 0.6, 0.5, 15
STABLE_MIN = 0.5

# 지표 확인 후 확정한 라벨 (근거: 아래 출력의 relevance·FREX·대표 문서)
LABELS = {
    6: ["외교·안보 정세", "체제 내부·권력층", "북한 사회·경제·스포츠(북한매체 보도)",
        "군사·무기(우크라이나 전쟁)", "탈북민 생활·가족", "북한군·접경지역(전단·홍수)"],
    10: ["외교·안보 정책", "김씨 일가·후계", "북한 사회·스포츠(북한매체 보도)", "미사일·무기기술",
         "탈북민 정착·교육", "보위부·권력층 비화", "접경 심리전(풍선·확성기)", "인프라·재해",
         "남한 생활 경험", "북한군 러시아 파병"],
}


def fit(K, X, seed):
    return LatentDirichletAllocation(n_components=K, max_iter=30, learning_method="batch",
                                     doc_topic_prior=0.1, topic_word_prior=0.01,
                                     random_state=seed).fit(X)


def per_topic_umass(comp, Xb, Dw):
    idx = np.argsort(comp)[::-1][:TOPN]
    co = np.asarray((Xb[:, idx].T @ Xb[:, idx]).todense())
    return np.mean([np.log((co[i, j] + 1) / Dw[idx[j]]) for i in range(1, TOPN) for j in range(i)])


def stability(ref, others):
    A = ref.components_ / np.linalg.norm(ref.components_, axis=1, keepdims=True)
    sims = []
    for m in others:
        B = m.components_ / np.linalg.norm(m.components_, axis=1, keepdims=True)
        S = A @ B.T
        r, c = linear_sum_assignment(-S)
        sims.append(S[r, c][np.argsort(r)])
    return np.mean(sims, axis=0)


def main():
    docs = pd.read_parquet(OUT / "_joint2024_tokens.parquet")
    vec = CountVectorizer(min_df=5, max_df=0.35, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(docs.text); vocab = np.array(vec.get_feature_names_out())
    Xb = (X > 0).astype(int); Dw = np.asarray(Xb.sum(axis=0)).ravel()
    pw = np.asarray(X.sum(axis=0)).ravel() / X.sum()

    title_k = pd.read_csv(KBS / "kbs_namnam_2024.csv", usecols=["ncd", "list_title"]) \
        .groupby("ncd").list_title.first()
    vids = pd.read_parquet(OUT / "videos.parquet").set_index("video_id")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)

    for K in [6, 10]:
        m = fit(K, X, 42)
        stab = stability(m, [fit(K, X, s) for s in range(5)])
        phi = m.components_ / m.components_.sum(axis=1, keepdims=True)
        theta = m.transform(X)
        tops = [set(np.argsort(c)[::-1][:TOPN]) for c in m.components_]
        is_k = (docs.media == "KBS").values
        rows = []
        for k in range(K):
            rel = LAMBDA * np.log(phi[k]) + (1 - LAMBDA) * np.log(phi[k] / pw)
            excl = phi[k] / phi.sum(axis=0)
            frex = 1 / (W_FREX / (rankdata(excl) / len(excl)) + (1 - W_FREX) / (rankdata(phi[k]) / len(phi[k])))
            others = set().union(*(tops[j] for j in range(K) if j != k))
            order = np.argsort(theta[:, k])[::-1]
            rk = [title_k.get(int(docs.doc_id[i][3:]), "") for i in order if is_k[i]][:2]
            ry = [vids.loc[docs.doc_id[i].rsplit("_", 1)[0], "title"] for i in order if not is_k[i]][:3]
            rows.append(dict(
                토픽=f"T{k}", 라벨=LABELS[K][k] + ("" if stab[k] >= STABLE_MIN else " (불안정)"),
                UMass=round(per_topic_umass(m.components_[k], Xb, Dw), 3), 안정성=round(stab[k], 3),
                배타성=round(1 - len(tops[k] & others) / TOPN, 2),
                KBS비중=round(theta[is_k, k].mean() * 100, 1), 유튜브비중=round(theta[~is_k, k].mean() * 100, 1),
                relevance=" · ".join(vocab[np.argsort(rel)[::-1][:10]]),
                FREX=" · ".join(vocab[np.argsort(frex)[::-1][:10]]),
                대표KBS=" | ".join(t[:35] for t in rk), 대표유튜브=" | ".join(t[:35] for t in ry)))
        df = pd.DataFrame(rows)
        print(f"\n===== K={K} =====")
        for r in df.itertuples():
            print(f"{r.토픽} [{r.라벨}] UMass {r.UMass} · 안정성 {r.안정성} · 배타성 {r.배타성} · "
                  f"KBS {r.KBS비중}% / YT {r.유튜브비중}%\n   rel : {r.relevance}\n   FREX: {r.FREX}\n"
                  f"   KBS : {r.대표KBS}\n   YT  : {r.대표유튜브}")
        df.to_csv(OUT / f"topic_labels_K{K}.csv", index=False)


if __name__ == "__main__":
    main()
