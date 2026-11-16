#!/usr/bin/env python3
"""형식 vs 화자 위치 분리 검토 — KBS 방송 안의 탈북민 인용 발화를 제3의 비교군으로.

KBS 층2(외부 발화)에는 탈북민이 없어(2024: 외부관료 20섹션), 대신 quote_spans 중 출처 표기에
'탈북·귀순·출신·북한이탈'이 들어간 스팬을 '방송 형식 + 탈북민 화자' 발화로 본다.
  KBS 제도 발화(인용 제외)  = 방송 형식 + 제3자 화자
  KBS 탈북민 인용            = 방송 형식 + 탈북민 화자   ← 이 스크립트
  유튜브                     = 유튜브 형식 + 탈북민 화자
지표: 남측 지칭 국가명%(referent_analysis_yt 규칙), 김정은 직함%(kim_title_yt 규칙). 기사 단위 부트스트랩.
한계: 인용은 KBS가 편집·선별한 짧은 발화(사운드바이트)라 순수한 화자 효과가 아니다.
출력: master/kbs_defector_quotes.csv
"""
import json
import random
import re
from pathlib import Path

import pandas as pd

import kim_title_yt as kt
import referent_analysis_yt as ra

BASE = Path(__file__).parent
KBS = BASE.parent / "kbs" / "master"
DEFECTOR = re.compile(r"탈북|귀순|북한이탈|출신")


def main():
    random.seed(42)
    sec = pd.read_parquet(KBS / "sections.parquet")
    rows = []
    for r in sec.itertuples():
        for q in json.loads(r.quote_spans):
            if DEFECTOR.search(q.get("attribution", "") or ""):
                rows.append((r.year, r.ncd, str(r.text_raw)[q["start"]:q["end"]]))
    d = pd.DataFrame(rows, columns=["year", "ncd", "text"])
    out = []
    for y, g in d.groupby("year"):
        ref, kim = [], []
        for _, ga in g.groupby("ncd"):
            t = " ".join(ga.text)
            ref.append(ra.to_counts([f for f, _ in ra.hits(t)]))
            c = dict.fromkeys(kt.CATS, 0)
            for cat, _ in kt.mentions(t):
                c[cat] += 1
            kim.append(c)
        p, n = ra.share(ref)
        lo, hi = ra.ci(ra.boot(ref)) if n else (float("nan"), float("nan"))
        kp, kn = kt.share(kim)
        out.append(dict(연도=y, 기사=g.ncd.nunique(), 스팬=len(g), 글자=int(g.text.str.len().sum()),
                        남측N=n, 국가명=round(p["국가명"], 1) if n else None,
                        국가명CI=f"[{lo:.1f}, {hi:.1f}]" if n else "", 지역명=round(p["지역명"], 1) if n else None,
                        관계어=round(p["관계어"], 1) if n else None,
                        김정은N=kn, 직함=round(kp["직함"], 1) if kn else None))
    res = pd.DataFrame(out)
    pd.set_option("display.width", 200)
    print("KBS 탈북민 인용 발화 — 남측 지칭(%)·김정은 직함(%)\n" + res.to_string(index=False))
    print("\n비교(2024): KBS 제도 발화 국가명 72.1 / 직함 79.9 · 유튜브 국가명 93.7 / 직함 0.3")
    res.to_csv(BASE / "master" / "kbs_defector_quotes.csv", index=False)


if __name__ == "__main__":
    main()
