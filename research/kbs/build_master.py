#!/usr/bin/env python3
"""KBS 남북의 창 4개 연도 → 마스터 테이블(sections / articles) 구축.

- 원본 CSV는 수정하지 않음. text_raw 보존.
- speaker/layer 파생(확정 규칙), quote_spans 는 위치만 마킹(제거하지 않음).
- 출력: master/sections.parquet, master/articles.parquet (+ 요약 csv)

speaker: 앵커 | 리포트 | 취재진 | 외부관료 | 외부전문가
layer  : 1(제도: 앵커·리포트·취재진) / 2(외부: 외부관료·외부전문가)
"""
import json
import re
from pathlib import Path

import pandas as pd
from kiwipiepy import Kiwi

BASE = Path(__file__).parent
OUT = BASE / "master"
FILES = {2018: "kbs_namnam_2018_final.csv", 2020: "kbs_namnam_2020.csv",
         2022: "kbs_namnam_2022.csv", 2024: "kbs_namnam_2024.csv"}

OFFICIAL = r"장관|대사|안보실장|국정원|의원|총리|차관|청장|사령관"     # 관료 신호(list_title)
REPORTER_END = r"전해드|전해 드|짚어봤|정리했|전해드립니다|특파원|에서 전"
# quote_spans: [출처 : "인용"] — 위치만 마킹
QUOTE_RE = re.compile(r'\[([^\]]{1,60}?)\s*[:：]\s*["“]([^"”\]]+)["”]?\]')
NKMEDIA = r"조선중앙|노동신문|조선신보|류경|우리민족|메아리|중앙통신|평양방송|조선의오늘"
EXPERT_A = r"교수|연구위원|소장|박사|연구원|위원"
OFFICIAL_A = r"장관|실장|대변인|부장|참모|사령관|청장|의원|대사|서기관|공보"

kiwi = Kiwi()
_UDICT = BASE.parent / "kiwi_user_dict.tsv"      # research/kiwi_user_dict.tsv
_n_udict = kiwi.load_user_dictionary(str(_UDICT))


def token_len(text):
    return sum(1 for _ in kiwi.tokenize(text)) if text.strip() else 0


def classify_speaker(section_type, list_title, text):
    lt, tx = str(list_title), str(text)
    if section_type == "앵커":
        return "앵커", 1
    if section_type == "리포트":
        return "리포트", 1
    if section_type == "답변":
        return ("외부관료" if re.search(OFFICIAL, lt) else "외부전문가"), 2
    if section_type == "기자":
        if re.search(OFFICIAL, lt):          # 예: [특별 대담] 조명균 통일부 장관
            return "외부관료", 2
        return "취재진", 1                    # 2022 특파원 등
    return "미상", 0


def src_type(attribution):
    a = attribution
    if re.search(NKMEDIA, a):
        return "NK매체"
    if re.search(EXPERT_A, a):
        return "전문가"
    if re.search(OFFICIAL_A, a):
        return "관료"
    return "기타"


def find_quote_spans(text):
    spans = []
    for m in QUOTE_RE.finditer(text):
        spans.append({"start": m.start(), "end": m.end(),
                      "src_type": src_type(m.group(1)),
                      "attribution": m.group(1).strip()[:40]})
    return spans


def main():
    OUT.mkdir(exist_ok=True)
    sec_rows = []
    for y, fn in FILES.items():
        df = pd.read_csv(BASE / fn)
        d = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
        iso_week = d.dt.isocalendar().week.astype(int)
        for i, r in df.iterrows():
            txt = str(r["text"])
            sp, lay = classify_speaker(r["section_type"], r["list_title"], txt)
            spans = find_quote_spans(txt)
            sec_rows.append({
                "year": y, "date": str(r["date"]), "iso_week": int(iso_week.iloc[i]),
                "ncd": r["ncd"], "section_order": r["section_order"],
                "article_title": r["article_title"], "url": r["url"],
                "section_type_raw": r["section_type"],
                "speaker": sp, "layer": lay,
                "text_raw": txt,
                "quote_spans": json.dumps(spans, ensure_ascii=False),
                "n_quotes": len(spans),
                "char_len": len(txt), "token_len": token_len(txt),
            })
    sections = pd.DataFrame(sec_rows)
    sections.to_parquet(OUT / "sections.parquet", index=False)

    # ---- articles: 층1 섹션만 연결 ----
    art_rows = []
    for (ncd, y), g in sections.groupby(["ncd", "year"], sort=False):
        g = g.sort_values("section_order")
        l1 = g[g.layer == 1]
        text = "\n".join(l1.text_raw)
        art_rows.append({
            "ncd": ncd, "year": y, "date": g.date.iloc[0],
            "iso_week": int(g.iso_week.iloc[0]),
            "article_title": g.article_title.iloc[0],
            "text": text,                      # 층1 text_raw 연결(인용 포함)
            "char_len": len(text),
            "token_len": int(l1.token_len.sum()),
            "n_anchor": int((g.speaker == "앵커").sum()),
            "n_report": int((g.speaker == "리포트").sum()),
            "n_l1_sections": int(len(l1)),
            "has_external": bool((g.layer == 2).any()),
            "l2_token_len": int(g[g.layer == 2].token_len.sum()),
            "quote_token_est": 0,   # 인용 토큰 추정은 후속(마킹만 우선)
        })
    articles = pd.DataFrame(art_rows)
    articles.to_parquet(OUT / "articles.parquet", index=False)

    # 요약
    print(f"sections: {len(sections)}행 → {OUT/'sections.parquet'}")
    print(f"articles: {len(articles)}건 → {OUT/'articles.parquet'}")
    print("\n[연도별 sections]")
    print(sections.groupby("year").agg(
        n=("ncd", "size"), 층1=("layer", lambda s: (s == 1).sum()),
        층2=("layer", lambda s: (s == 2).sum()),
        인용스팬=("n_quotes", "sum"), 토큰=("token_len", "sum")))
    print("\n[연도별 articles]")
    print(articles.groupby("year").agg(
        기사=("ncd", "size"), 층1토큰=("token_len", "sum"),
        기사당토큰=("token_len", "mean"), 외부포함기사=("has_external", "sum")))
    print("\n[speaker 분포]")
    print(sections.groupby(["year", "speaker"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
