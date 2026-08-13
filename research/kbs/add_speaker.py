#!/usr/bin/env python3
"""KBS 남북의 창 연도별 CSV에 화자(speaker)·층위(layer) 열을 추가한 파생본 생성.

section_type 라벨이 연도마다 다른 대상을 가리키므로(예: 2018 '기자'=통일부 장관,
2022 '기자'=KBS 특파원), 라벨을 그대로 쓰지 않고 화자를 재구성한다.

speaker: 앵커(KBS) | 리포트(KBS) | KBS취재진 | 외부발화
layer  : 층1(제도)  | 층2(외부)
ext_type(외부발화 한정): 관료 | 전문가 | 기타   ← 휴리스틱(참고용)

원본은 수정하지 않고 kbs_namnam_<연도>_speaker.csv 로 저장.
"""
import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
FILES = {
    2018: "kbs_namnam_2018_final.csv",
    2020: "kbs_namnam_2020.csv",
    2022: "kbs_namnam_2022.csv",
    2024: "kbs_namnam_2024.csv",
}

# list_title 신호
OFFICIAL = r"장관|대사|안보실장|국정원|의원|총리|차관"          # 정부 관료
EXPERT_T = r"교수|소장|박사|연구위원|자문위원|전문가|미니 ?대담"   # 학자/전문가 대담
# 기자 라벨이 KBS 취재진임을 보이는 종결/문맥
REPORTER = r"전해드|전해 드|짚어봤|정리했|전해드립니다|특파원|에서 전"


def classify(row):
    """(speaker, layer, ext_type) 반환."""
    st = row.section_type
    lt = str(row.list_title)
    tx = str(row.text)

    if st == "앵커":
        return "앵커(KBS)", "층1", ""
    if st == "리포트":
        return "리포트(KBS)", "층1", ""
    if st == "답변":
        # 답변은 항상 외부 발화(전문가 또는 관료)
        ext = "관료" if re.search(OFFICIAL, lt) else ("전문가" if re.search(EXPERT_T, lt) else "기타")
        return "외부발화", "층2", ext
    if st == "기자":
        # 기자 라벨 → KBS취재진 vs 외부인사 판별
        if re.search(OFFICIAL, lt):           # 예: [특별 대담] 조명균 통일부 장관
            return "외부발화", "층2", "관료"
        return "KBS취재진", "층1", ""          # 2022 특파원 등
    return "미상", "미상", ""


def main():
    summary = []
    for y, fn in FILES.items():
        df = pd.read_csv(BASE / fn)
        res = df.apply(classify, axis=1, result_type="expand")
        res.columns = ["speaker", "layer", "ext_type"]
        out = pd.concat([df, res], axis=1)
        outpath = BASE / f"kbs_namnam_{y}_speaker.csv"
        out.to_csv(outpath, index=False, encoding="utf-8-sig")

        L1 = (out.layer == "층1").sum()
        L2 = (out.layer == "층2").sum()
        mis = (out.layer == "미상").sum()
        summary.append((y, len(out), L1, L2, mis, out))
        print(f"[{y}] {len(out)}행 → {outpath.name} | 층1={L1} 층2={L2} 미상={mis}", flush=True)

    print("\n=== speaker 분포 ===")
    for y, n, L1, L2, mis, out in summary:
        print(f"  {y}: {out.speaker.value_counts().to_dict()}")
    print("\n=== 층2 외부발화 세부(ext_type) ===")
    for y, n, L1, L2, mis, out in summary:
        e = out[out.layer == "층2"].ext_type.value_counts().to_dict()
        print(f"  {y}: 층2 {L2}행 → {e}")


if __name__ == "__main__":
    sys.exit(main())
