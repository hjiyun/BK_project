#!/usr/bin/env python3
"""2024 KBS + 유튜브 합동 LDA — 매체 고유 의제 탐색 (투영 분석 topic_project_yt.py 보완).

- 문서: KBS 2024 기사(articles.text, 층1) + 유튜브 2024 청크(chunks.parquet, ≈KBS 기사 길이)
- 전처리: KBS topic_*.py 와 동일(STOP, NNG·NNP, 2자↑) + 구어 상투 명사(SPOKEN_STOP, logodds_yt_vs_kbs.py와 동일)
- K 선택: KBS topic_select_k.py 와 동일한 UMass 응집도(top15), K ∈ {6..16}
- 보고: 토픽별 상위어, 매체별 평균 비중, 유튜브 내 최대 채널 점유(채널 정체성 토픽 판별)
사용: python3 topic_joint_2024.py [K]   (K 생략 시 응집도 최고 K)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

BASE = Path(__file__).parent
KBS = BASE.parent / "kbs" / "master"
OUT = BASE / "master"
KS = [6, 8, 10, 12, 14, 16]
kiwi = Kiwi(); kiwi.load_user_dictionary(str(BASE.parent / "kiwi_user_dict.tsv"))

STOP = set("""것 수 등 때 년 월 일 이번 지금 오늘 요즘 우리 저희 자신 서로 모습 정도 경우 문제 상황
이야기 부분 생각 이후 이전 지역 사람 사실 통해 위해 대해 관련 대한 여기 거기 하나 가지 방법 결과
시작 시간 동안 모두 다시 특히 최근 당시 그것 이것 무엇 얘기 말씀 때문 만큼 부터 까지 마디 자리
이날 지난 올해 내년 그동안 앞서 그런 이런 저런 어떤 무슨 관계 내용 상태 중심 규모 수준 차원 측면
가운데 속 앞 뒤 안 밖 위 아래 곳 데 점 면 채 뿐 등등 및 또 좀 잘 더 덜 각 별 총 약 명 개 번 차 회
분 초 시 년도 여러분 시청자 남북 북 한반도 북한""".split())
SPOKEN_STOP = set("거기 저기 이거 그거 저거 요거 그게 이게 뭔가 진짜 정말 그냥 이제 그때 이때 뭐".split())


def toks(t):
    return [x.form for x in kiwi.tokenize(str(t))
            if x.tag in ("NNG", "NNP") and len(x.form) >= 2
            and x.form not in STOP and x.form not in SPOKEN_STOP]


def umass(model, Xb, Dw, topn=15):
    cohs = []
    for comp in model.components_:
        idx = np.argsort(comp)[::-1][:topn]
        sub = Xb[:, idx]
        co = np.asarray((sub.T @ sub).todense())
        s = cnt = 0
        for i in range(1, topn):
            for j in range(i):
                s += np.log((co[i, j] + 1) / Dw[idx[j]]); cnt += 1
        cohs.append(s / cnt)
    return np.mean(cohs)


def load_docs():
    cache = OUT / "_joint2024_tokens.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    art = pd.read_parquet(KBS / "articles.parquet")
    art = art[art.year == 2024]
    ch = pd.read_parquet(OUT / "chunks.parquet")
    ch = ch[ch.year == 2024]
    docs = pd.concat([
        pd.DataFrame({"doc_id": "ncd" + art.ncd.astype(str), "media": "KBS", "channel": "KBS",
                      "text": [" ".join(toks(t)) for t in art.text]}),
        pd.DataFrame({"doc_id": ch.video_id + "_" + ch.chunk_idx.astype(str), "media": "유튜브",
                      "channel": ch.channel.values, "text": [" ".join(toks(t)) for t in ch.text]}),
    ], ignore_index=True)
    docs.to_parquet(cache, index=False)
    return docs


def fit(K, X):
    return LatentDirichletAllocation(n_components=K, max_iter=30, learning_method="batch",
                                     doc_topic_prior=0.1, topic_word_prior=0.01,
                                     random_state=42).fit(X)


def main():
    docs = load_docs()
    vec = CountVectorizer(min_df=5, max_df=0.35, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform(docs.text); vocab = np.array(vec.get_feature_names_out())
    Xb = (X > 0).astype(int); Dw = np.asarray(Xb.sum(axis=0)).ravel()
    print(f"문서 {X.shape[0]} (KBS {(docs.media == 'KBS').sum()} / 유튜브 {(docs.media == '유튜브').sum()})"
          f" · 어휘 {X.shape[1]}")

    print("\nK   UMass   perplexity")
    models = {}
    for K in KS:
        m = fit(K, X); models[K] = m
        print(f"{K:2d}  {umass(m, Xb, Dw):7.3f}  {m.perplexity(X):9.1f}")
    best = max(models, key=lambda k: umass(models[k], Xb, Dw))
    K = int(sys.argv[1]) if len(sys.argv) > 1 else best
    print(f"\n▶ 응집도 최고 K={best} / 사용 K={K}")

    m = models[K] if K in models else fit(K, X)
    theta = m.transform(X) * 100
    cols = [f"T{k}" for k in range(K)]
    th = pd.concat([docs[["doc_id", "media", "channel"]], pd.DataFrame(theta, columns=cols)], axis=1)

    med = th.groupby("media")[cols].mean().T
    yt = th[th.media == "유튜브"]
    ch_mass = yt.groupby("channel")[cols].sum()
    ch_share = ch_mass / ch_mass.sum()                     # 토픽 질량 중 채널 점유
    rep = pd.DataFrame({
        "상위어": [" · ".join(vocab[np.argsort(m.components_[k])[::-1][:12]]) for k in range(K)],
        "KBS%": med["KBS"].round(1).values,
        "유튜브%": med["유튜브"].round(1).values,
        "유튜브채널균등%": yt.groupby("channel")[cols].mean().mean().round(1).values,
        "최대채널": ch_share.idxmax().values,
        "최대채널점유%": (ch_share.max() * 100).round(0).values,
    }, index=cols)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90)
    print(rep.to_string())
    print("\n[채널별 평균 토픽 비중 (%)]\n"
          + th.groupby("channel")[cols].mean().round(1).to_string())

    # 대표 문서 (θ 최고 KBS 기사 제목 / 유튜브 영상 제목)
    # 마스터 테이블의 article_title 은 전부 결측 → 원본 CSV list_title 사용
    title = pd.read_csv(KBS.parent / "kbs_namnam_2024.csv", usecols=["ncd", "list_title"]) \
        .groupby("ncd").list_title.first()
    vids = pd.read_parquet(OUT / "videos.parquet").set_index("video_id")
    print("\n[토픽별 대표 문서]")
    for c in cols:
        top = th.sort_values(c, ascending=False)
        ks = top[top.media == "KBS"].head(2)
        ys = top[top.media == "유튜브"].head(3)
        kt = [f"{str(title.get(int(d[3:]), d))[:40]} θ{v:.0f}" for d, v in zip(ks.doc_id, ks[c])]
        yt_t = [f"[{vids.loc[d.rsplit('_', 1)[0], 'channel']}] "
                f"{vids.loc[d.rsplit('_', 1)[0], 'title'][:40]} θ{v:.0f}" for d, v in zip(ys.doc_id, ys[c])]
        print(f"{c}: KBS {kt}\n     YT  {yt_t}")

    rep.to_csv(OUT / f"topic_joint2024_K{K}.csv")
    th.round(2).to_parquet(OUT / f"topic_joint2024_theta_K{K}.parquet", index=False)


if __name__ == "__main__":
    main()
