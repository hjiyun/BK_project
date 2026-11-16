#!/usr/bin/env python3
"""KBS 인접 시점 가중 로그오즈비 — 2018→2020, 2020→2022, 2022→2024 (+ 2018→2024 파일럿 대조). 8.25 회의 3절 과제.

- 방법: Monroe et al. (2008) 정보적 디리클레 사전. 사전 = 4개 연도 합산 빈도 × α0(=10,000) — youtube/logodds_yt_vs_kbs.py 와 같은 식
- 단위: 기사(articles.text, 층1, 인용 포함 — 파일럿과 동일). 전처리: KBS 토픽모델과 같은 STOP, Kiwi+사용자사전 v2
- 두 버전: ① 전체 명사(NNG+NNP) ② 일반명사(NNG)만 — 8.25 지적('상위어 대부분이 사건 고유명사')에 대한 보완
- 분산 필터: 변별 쪽 연도에서 기사 MIN_DF건 이상 등장한 단어만 채택(단일 기사 몰림 배제)
출력: master/logodds_adjacent.csv (전 어휘·z·연도별 빈도·기사빈도), 상위어 표준출력
"""
import collections
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from kiwipiepy import Kiwi

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE.parent / "youtube"))
from logodds_yt_vs_kbs import STOP, logodds, make_prior  # noqa: E402

kiwi = Kiwi(); kiwi.load_user_dictionary(str(BASE.parent / "kiwi_user_dict.tsv"))
PAIRS = [(2018, 2020), (2020, 2022), (2022, 2024), (2018, 2024)]
MIN_FREQ, MIN_DF, TOP = 10, 5, 15


def toks(text):
    return [(t.form, t.tag) for t in kiwi.tokenize(str(text))
            if t.tag in ("NNG", "NNP") and len(t.form) >= 2 and t.form not in STOP]


def main():
    art = pd.read_parquet(BASE / "master" / "articles.parquet")
    cnt = {v: {y: collections.Counter() for y in [2018, 2020, 2022, 2024]} for v in ["전체", "일반명사"]}
    dfreq = {y: collections.Counter() for y in [2018, 2020, 2022, 2024]}
    for r in art.itertuples():
        tk = toks(r.text)
        cnt["전체"][r.year].update(f for f, _ in tk)
        cnt["일반명사"][r.year].update(f for f, t in tk if t == "NNG")
        dfreq[r.year].update({f for f, _ in tk})

    rows = []
    for ver, C in cnt.items():
        prior, tot = make_prior(*C.values())
        for a, b in PAIRS:
            z = logodds(C[b], C[a], prior)          # 양수 = 뒤 연도(b) 변별
            for w, zw in z.items():
                if C[a][w] + C[b][w] < MIN_FREQ:
                    continue
                side = b if zw > 0 else a
                rows.append(dict(버전=ver, 구간=f"{a}→{b}", 단어=w, z=round(zw, 2), 변별연도=side,
                                 **{f"빈도{a}": C[a][w], f"빈도{b}": C[b][w]},
                                 **{f"기사{a}": dfreq[a][w], f"기사{b}": dfreq[b][w]},
                                 채택=dfreq[side][w] >= MIN_DF))
    df = pd.DataFrame(rows)
    df.to_csv(BASE / "master" / "logodds_adjacent.csv", index=False)

    for ver in ["전체", "일반명사"]:
        print(f"\n================ {ver} (기사 {MIN_DF}건 이상) ================")
        for a, b in PAIRS:
            d = df[(df.버전 == ver) & (df.구간 == f"{a}→{b}") & df.채택]
            up = d[d.z > 0].nlargest(TOP, "z"); dn = d[d.z < 0].nsmallest(TOP, "z")
            print(f"[{a}→{b}]\n  {b} 쪽: " + " · ".join(f"{w}({z:.1f})" for w, z in zip(up.단어, up.z))
                  + f"\n  {a} 쪽: " + " · ".join(f"{w}({-z:.1f})" for w, z in zip(dn.단어, dn.z)))


if __name__ == "__main__":
    main()
