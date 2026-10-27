#!/usr/bin/env python3
"""KBS LDA(K=8) 고정 모델에 유튜브 2024 청크를 투영 → 2024 매체 간 의제 구성비 비교.

- KBS 모델: kbs/topic_quarter.py 와 동일 설정으로 재적합(기사 572 + 정상회담 특집 18청크,
  CountVectorizer min_df=5 max_df=0.35, LDA K=8 batch random_state=42).
  특집 텍스트는 kbs/specials/*.txt (pdftotext -layout 추출; 9.10 회의록의 18청크와 일치).
- 유튜브: master/chunks.parquet (KBS 기사 길이 ≈3,500자 청크) → 같은 전처리·같은 어휘로 lda.transform.
  KBS 어휘 밖 토큰은 버려지므로 '어휘 적중률'을 함께 보고.
- 집계: 청크 평균(토큰 수 그대로), 채널 균등 평균(채널별 평균의 평균), 채널별.
- 출력: master/topic_2024_yt_vs_kbs.csv, master/topic_2024_by_channel.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

BASE = Path(__file__).parent
KBS = BASE.parent / "kbs"
kiwi = Kiwi(); kiwi.load_user_dictionary(str(BASE.parent / "kiwi_user_dict.tsv"))

# kbs/topic_quarter.py 와 동일
STOP = set("""것 수 등 때 년 월 일 이번 지금 오늘 요즘 우리 저희 자신 서로 모습 정도 경우 문제 상황
이야기 부분 생각 이후 이전 지역 사람 사실 통해 위해 대해 관련 대한 여기 거기 하나 가지 방법 결과
시작 시간 동안 모두 다시 특히 최근 당시 그것 이것 무엇 얘기 말씀 때문 만큼 부터 까지 마디 자리
이날 지난 올해 내년 그동안 앞서 그런 이런 저런 어떤 무슨 관계 내용 상태 중심 규모 수준 차원 측면
가운데 속 앞 뒤 안 밖 위 아래 곳 데 점 면 채 뿐 등등 및 또 좀 잘 더 덜 각 별 총 약 명 개 번 차 회
분 초 시 년도 여러분 시청자 남북 북 한반도 북한""".split())
LABELS = ["체제·우상화", "교육·산업", "문화·스포츠", "북러밀착", "비핵화외교", "농업·재해", "경제·사회", "미사일·도발"]


def toks(t):
    return [x.form for x in kiwi.tokenize(str(t))
            if x.tag in ("NNG", "NNP") and len(x.form) >= 2 and x.form not in STOP]


def chunk(t, size=1500):
    ps = [p.strip() for p in re.split(r"\n\s*\n", t) if len(p.strip()) > 20]; o = []; b = ""
    for p in ps:
        b += " " + p
        if len(b) >= size: o.append(b.strip()); b = ""
    if len(b.strip()) > 200: o.append(b.strip())
    return o


def fit_kbs():
    art = pd.read_parquet(KBS / "master" / "articles.parquet")
    docs = [(int(r.year), toks(r.text)) for r in art.itertuples()]
    n_sp = 0
    for fn in ["2018 남북 정상회담", "북미정상회담"]:
        for ch in chunk((KBS / "specials" / f"{fn}.txt").read_text(encoding="utf-8")):
            docs.append((2018, toks(ch))); n_sp += 1
    vec = CountVectorizer(min_df=5, max_df=0.35, token_pattern=r"(?u)\b\w+\b")
    X = vec.fit_transform([" ".join(d[1]) for d in docs])
    lda = LatentDirichletAllocation(n_components=8, max_iter=30, learning_method="batch",
                                    doc_topic_prior=0.1, topic_word_prior=0.01, random_state=42)
    theta = lda.fit_transform(X)
    return vec, lda, theta, np.array([d[0] for d in docs]), n_sp


def main():
    vec, lda, theta_k, years, n_sp = fit_kbs()
    vocab = np.array(vec.get_feature_names_out())
    print(f"KBS 문서 {len(years)} (특집 청크 {n_sp}) / 어휘 {len(vocab)}")
    print("토픽 상위어 (회의록 표2와 대조):")
    for k in range(8):
        print(f"  T{k} {LABELS[k]:7} {' · '.join(vocab[np.argsort(lda.components_[k])[::-1][:10]])}")
    yearly = pd.DataFrame(theta_k, columns=LABELS).groupby(years).mean() * 100
    print("KBS 연도별 평균 토픽 비중(%) (회의록 표4와 대조):\n" + yearly.round(1).to_string())

    ch = pd.read_parquet(BASE / "master" / "chunks.parquet")
    ch = ch[ch.year == 2024].reset_index(drop=True)
    ytoks = [toks(t) for t in ch.text]
    Xy = vec.transform([" ".join(t) for t in ytoks])
    ch["n_noun"] = [len(t) for t in ytoks]
    ch["n_inv"] = np.asarray(Xy.sum(axis=1)).ravel()
    theta_y = lda.transform(Xy)
    ty = pd.concat([ch[["video_id", "chunk_idx", "channel", "n_noun", "n_inv"]],
                    pd.DataFrame(theta_y * 100, columns=LABELS)], axis=1)

    hit_y = ch.n_inv.sum() / ch.n_noun.sum() * 100
    k24 = theta_k[years == 2024] * 100
    by_ch = ty.groupby("channel")[LABELS].mean()
    by_ch.insert(0, "청크", ty.groupby("channel").size())
    by_ch.insert(1, "어휘적중%", (ty.groupby("channel").n_inv.sum()
                               / ty.groupby("channel").n_noun.sum() * 100).round(1))

    comp = pd.DataFrame({
        "KBS 2024": k24.mean(axis=0),
        "유튜브(청크평균)": ty[LABELS].mean(),
        "유튜브(채널균등)": by_ch[LABELS].mean(),
    }, index=LABELS)
    comp["차이(청크-KBS)"] = comp["유튜브(청크평균)"] - comp["KBS 2024"]
    # 주토픽 분포 (θ 최대 토픽)
    comp["KBS 주토픽%"] = pd.Series(np.argmax(k24, 1)).map(dict(enumerate(LABELS))) \
        .value_counts(normalize=True).reindex(LABELS).fillna(0) * 100
    comp["유튜브 주토픽%"] = ty[LABELS].idxmax(axis=1).value_counts(normalize=True) \
        .reindex(LABELS).fillna(0) * 100

    pd.set_option("display.width", 200)
    print(f"\n유튜브 청크 {len(ch)} / 명사 중 KBS 어휘 적중률 {hit_y:.1f}%")
    print("\n[2024 평균 토픽 비중 (%)]\n" + comp.round(1).to_string())
    print("\n[채널별 평균 토픽 비중 (%)]\n" + by_ch.round(1).to_string())

    comp.round(2).to_csv(BASE / "master" / "topic_2024_yt_vs_kbs.csv")
    by_ch.round(2).to_csv(BASE / "master" / "topic_2024_by_channel.csv")
    ty.round(2).to_parquet(BASE / "master" / "topic_theta_yt_2024.parquet", index=False)


if __name__ == "__main__":
    main()
