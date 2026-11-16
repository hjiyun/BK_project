#!/usr/bin/env python3
"""Kiwi 사용자 사전 v3 후보 + ASR 오인식 점검 (사전 자체는 수정하지 않음 — 반영 시 전 분석 재실행 필요).

v2 기준(kiwi_user_dict.tsv 머리말): 코퍼스 빈도 ≥10 & Kiwi 오분해 & 명백한 북한 고유명사/제도어.
1) 분해 후보: 공백 없이 붙은 명사 토큰 연쇄(NNG/NNP/XSN 2~3개)의 표면형 빈도 — KBS 4개 연도 + 유튜브 2024
   (예: 김+주애, 보위+부). 매체별 빈도와 함께 출력, 사람이 고유명사 여부 판정.
2) ASR 변이: 북한 인명·기관의 알려진 표기 vs 유튜브 자막 변이형 빈도(예: 리설주/이설주).
출력: master/dict_v3_candidates.csv, master/asr_variants.csv
"""
import collections
import re
from pathlib import Path

import pandas as pd
from kiwipiepy import Kiwi

BASE = Path(__file__).parent
KBS = BASE.parent / "kbs" / "master"
kiwi = Kiwi(); kiwi.load_user_dictionary(str(BASE.parent / "kiwi_user_dict.tsv"))
NOUN = ("NNG", "NNP", "XSN")
V2 = {l.split("\t")[0] for l in (BASE.parent / "kiwi_user_dict.tsv").read_text(encoding="utf-8").splitlines()
      if l.strip() and not l.startswith("#")}
MIN_FREQ = 10

# 정자 표기 → 자막에서 나올 수 있는 변이형 (두음법칙·연음·유사음)
ASR = {
    "리설주": ["이설주", "이성주"], "김여정": ["김여종", "김연정"], "김주애": ["김주에", "김주해"],
    "리선권": ["이선권"], "리병철": ["이병철"], "최선희": ["최선이"], "조선중앙TV": ["조선 중앙 TV", "조선중앙 티비"],
    "보위부": ["보이부"], "노동신문": ["로동신문"], "장마당": ["장 마당"], "김정은": ["김정운", "김종은"],
}


def compounds(texts):
    c = collections.Counter()
    for t in texts:
        toks = kiwi.tokenize(str(t))
        i = 0
        while i < len(toks):
            j = i
            while (j + 1 < len(toks) and toks[j + 1].tag in NOUN and toks[j].tag in NOUN
                   and toks[j + 1].start == toks[j].start + toks[j].len and j - i < 2):
                j += 1
            if j > i and toks[i].tag in ("NNG", "NNP"):
                surf = t[toks[i].start:toks[j].start + toks[j].len]
                split = "+".join(x.form for x in toks[i:j + 1])
                c[(surf, split, toks[j].tag == "XSN")] += 1
            i = j + 1
    return c


def main():
    art = pd.read_parquet(KBS / "articles.parquet")
    v = pd.read_parquet(BASE / "master" / "videos.parquet")
    v = v[v.year == 2024]
    kc, yc = compounds(art.text), compounds(v.text_clean)
    rows = []
    for key in set(kc) | set(yc):
        k, y = kc.get(key, 0), yc.get(key, 0)
        if k + y >= MIN_FREQ and len(key[0]) >= 3:
            rows.append(dict(표면형=key[0], Kiwi분해=key[1], 접미사결합=key[2], v2등록=key[0] in V2,
                             KBS=k, 유튜브=y, 합계=k + y))
    df = pd.DataFrame(rows).sort_values("합계", ascending=False)
    nn = df[~df.접미사결합]
    df.to_csv(BASE / "master" / "dict_v3_candidates.csv", index=False)

    texts = {"KBS": " ".join(art.text.astype(str)), "유튜브": " ".join(v.text_clean)}
    arows = []
    for std, vars_ in ASR.items():
        for form in [std] + vars_:
            arows.append(dict(정자=std, 형태=form, 구분="정자" if form == std else "변이",
                              KBS=texts["KBS"].count(form), 유튜브=texts["유튜브"].count(form)))
    adf = pd.DataFrame(arows)
    adf.to_csv(BASE / "master" / "asr_variants.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"[분해 후보 — 명사+명사] 빈도≥{MIN_FREQ}, 3자 이상 — 상위 50 (명사+명사 {len(nn)} / 전체 {len(df)}; "
          f"접미사 결합 -들·-적·-님 등은 사전 대상 아님)\n" + nn.head(50).to_string(index=False))
    print("\n[v2 등록어인데 분해된 경우]\n" + df[df.v2등록].to_string(index=False))
    print("\n[ASR 변이형]\n" + adf.to_string(index=False))


if __name__ == "__main__":
    main()
