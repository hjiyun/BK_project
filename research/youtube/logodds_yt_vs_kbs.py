#!/usr/bin/env python3
"""매체 간 변별 어휘 — 유튜브 2024 vs KBS 2024, 정보적 디리클레 사전 가중 로그오즈비 (Monroe et al. 2008).

- 전처리: KBS topic_model.py 와 동일 (Kiwi+사용자사전, NNG·NNP, 2자 이상, 동일 불용어)
  + 구어 상투 명사 소수 추가(SPOKEN_STOP). KBS는 articles.text(층1, 인용 포함), 유튜브는 videos.text_clean.
- 사전(prior): 두 코퍼스 합산 빈도를 배경으로 α_w = A · p_w(합산), A = PRIOR_TOTAL.
- 채널 일관성: 유튜브 각 채널 vs KBS 로 같은 계산을 반복해, 전체와 같은 방향으로 |z|>1.96 인
  채널 수(n_ch)를 기록. 유튜브 측 상위어는 n_ch ≥ MIN_CH 만 '공통 변별어'로 채택.
- 출력: master/logodds_2024_yt_vs_kbs.csv (전 어휘), 상위어 표준출력.
"""
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi

BASE = Path(__file__).parent
KBS = BASE.parent / "kbs" / "master"
PRIOR_TOTAL = 10_000
MIN_FREQ = 20          # 합산 빈도 하한
MIN_CH = 4             # 6개 채널 중 같은 방향 유의 채널 수 하한
TOP = 30

kiwi = Kiwi(); kiwi.load_user_dictionary(str(BASE.parent / "kiwi_user_dict.tsv"))

# KBS topic_model.py 의 STOP 과 동일
STOP = set("""것 수 등 때 년 월 일 이번 지금 오늘 요즘 우리 저희 자신 서로 모습 정도 경우 문제 상황
이야기 부분 생각 이후 이전 지역 사람 사실 통해 위해 대해 관련 대한 여기 거기 하나 가지 방법 결과
시작 시간 동안 모두 다시 특히 최근 당시 그것 이것 무엇 얘기 말씀 때문 만큼 부터 까지 마디 자리
이날 지난 올해 내년 그동안 앞서 그런 이런 저런 어떤 무슨 관계 내용 상태 중심 규모 수준 차원 측면
가운데 속 앞 뒤 안 밖 위 아래 곳 데 점 면 채 뿐 등등 및 또 좀 잘 더 덜 각 별 총 약 명 개 번 차 회
분 초 시 년도 여러분 시청자 남북 북 한반도 북한""".split())
# 구어 상투 명사 (의미 기여 없음, 매체 형식 차이만 반영)
SPOKEN_STOP = set("거기 저기 이거 그거 저거 요거 그게 이게 뭔가 진짜 정말 그냥 이제 그때 이때 뭐".split())


def toks(text):
    return [t.form for t in kiwi.tokenize(str(text))
            if t.tag in ("NNG", "NNP") and len(t.form) >= 2
            and t.form not in STOP and t.form not in SPOKEN_STOP]


def logodds(ca, cb, prior):
    """a 대비 b의 가중 로그오즈비 z. 양수 = a 쪽 변별."""
    na, nb, a0 = sum(ca.values()), sum(cb.values()), sum(prior.values())
    z = {}
    for w, aw in prior.items():
        ya, yb = ca.get(w, 0), cb.get(w, 0)
        la = np.log((ya + aw) / (na + a0 - ya - aw))
        lb = np.log((yb + aw) / (nb + a0 - yb - aw))
        z[w] = (la - lb) / np.sqrt(1 / (ya + aw) + 1 / (yb + aw))
    return z


def make_prior(*counters):
    tot = collections.Counter()
    for c in counters:
        tot.update(c)
    n = sum(tot.values())
    return {w: PRIOR_TOTAL * f / n for w, f in tot.items()}, tot


def main():
    art = pd.read_parquet(KBS / "articles.parquet")
    art = art[art.year == 2024]
    v = pd.read_parquet(BASE / "master" / "videos.parquet")
    v = v[v.year == 2024]

    kbs = collections.Counter()
    kbs_df = collections.Counter()                 # 문서 빈도(기사 수)
    for t in art.text:
        tk = toks(t)
        kbs.update(tk); kbs_df.update(set(tk))
    by_ch = collections.defaultdict(collections.Counter)
    yt_df = collections.Counter()                  # 문서 빈도(영상 수)
    for r in v.itertuples():
        tk = toks(r.text_clean)
        by_ch[r.channel].update(tk); yt_df.update(set(tk))
    yt = collections.Counter()
    for c in by_ch.values():
        yt.update(c)

    prior, tot = make_prior(yt, kbs)
    z = logodds(yt, kbs, prior)
    # 채널별 일관성 (각 채널 vs KBS, 같은 배경 사전)
    ch_z = {ch: logodds(c, kbs, prior) for ch, c in by_ch.items()}

    rows = []
    for w, zw in z.items():
        if tot[w] < MIN_FREQ:
            continue
        sign = np.sign(zw)
        n_ch = sum(1 for cz in ch_z.values() if np.sign(cz[w]) == sign and abs(cz[w]) > 1.96)
        rows.append(dict(word=w, z=round(zw, 2), yt=yt[w], kbs=kbs[w],
                         yt_per10k=round(yt[w] / sum(yt.values()) * 1e4, 2),
                         kbs_per10k=round(kbs[w] / sum(kbs.values()) * 1e4, 2),
                         yt_videos=yt_df[w], kbs_articles=kbs_df[w], n_ch=n_ch,
                         **{f"z_{ch}": round(cz[w], 1) for ch, cz in ch_z.items()}))
    df = pd.DataFrame(rows).sort_values("z", ascending=False)
    df.to_csv(BASE / "master" / "logodds_2024_yt_vs_kbs.csv", index=False)

    print(f"토큰(명사) — 유튜브 {sum(yt.values()):,} / KBS {sum(kbs.values()):,} | "
          f"어휘(빈도≥{MIN_FREQ}) {len(df):,} | prior α0={PRIOR_TOTAL:,}")
    cols = ["word", "z", "yt_per10k", "kbs_per10k", "yt_videos", "kbs_articles", "n_ch"]
    yt_top = df[(df.z > 0) & (df.n_ch >= MIN_CH)].head(TOP)
    yt_idio = df[(df.z > 1.96) & (df.n_ch < MIN_CH)].head(15)
    kbs_top = df[df.z < 0].sort_values("z").head(TOP)
    print(f"\n[유튜브 변별 — 공통(n_ch≥{MIN_CH}/6)]\n" + yt_top[cols].to_string(index=False))
    print(f"\n[유튜브 변별 — 채널 편중(n_ch<{MIN_CH}), 참고]\n" + yt_idio[cols].to_string(index=False))
    print(f"\n[KBS 변별]\n" + kbs_top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
