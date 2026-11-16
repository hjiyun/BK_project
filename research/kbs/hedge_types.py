#!/usr/bin/env python3
"""KBS 리포트 나레이션 확신도(헤지) — 유형 세분 재산출 (8.25 회의 7절 단기 과제).

보고서 6.4절 초판 수치(12.5→18.1%)의 원 코드는 남아 있지 않아 규칙을 명시해 재구현한다.
- 대상: speaker=='리포트' 섹션, quote_spans 제거(인용 제외), 4개 연도
- 문장 분리: 종결 '다·요·까' 뒤 구두점. 5자 미만 제외. 분모 = 전체 나레이션 문장
- 유형 (문장 끝 서술어 기준, 상호배타):
    출처불명시형 : 알려졌습니다·전해졌습니다·알려집니다·전해집니다·~다고 합니다/한다 (+ 해라체)
    화자추정형   : 보입니다·보인다·풀이됩니다·해석됩니다·관측됩니다·예상됩니다·추정됩니다·전망됩니다·
                  전망입니다/이다·관측입니다·듯합니다·같습니다·가능성이 (높|있)습니다
- 소재(코너): 내부 = [요즘 북한은]·[클로즈업 북한] / 정세 = [이슈&한반도] / 기타
- 검정: Cochran-Armitage 경향(연도 2018,2020,2022,2024 점수), 연도별 문장 부트스트랩 95% CI
출력: master/hedge_types_year.csv, master/hedge_types_corner.csv
"""
import json
import math
import random
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
FILES = {2018: "kbs_namnam_2018_final.csv", 2020: "kbs_namnam_2020.csv",
         2022: "kbs_namnam_2022.csv", 2024: "kbs_namnam_2024.csv"}
YEARS = [2018, 2020, 2022, 2024]

HEARSAY = re.compile(r"((알려|전해)(졌|집)(습니다|니다|다)|다고\s*(합니다|한다)|라고\s*(합니다|한다))$")
INFER = re.compile(r"(보입니다|보인다|(풀이|해석|관측|예상|추정|전망)됩니다|(풀이|해석|관측|예상|추정|전망)된다|"
                   r"(전망|관측)(입니다|이다)|듯합니다|듯하다|같습니다|가능성이\s*(높|있)(습니다|다))$")
TYPES = ["출처불명시형", "화자추정형"]


def strip_quotes(text, qs):
    spans = sorted((q["start"], q["end"]) for q in json.loads(qs))
    out, p = [], 0
    for a, b in spans:
        out.append(text[p:a]); p = max(p, b)
    out.append(text[p:])
    return " ".join(out)


def corner(title):
    t = str(title)
    if "요즘 북한은" in t or "클로즈업 북한" in t:
        return "내부"
    if "이슈&한반도" in t or "이슈 & 한반도" in t:
        return "정세"
    return "기타"


def classify(sent):
    if HEARSAY.search(sent):
        return "출처불명시형"
    if INFER.search(sent):
        return "화자추정형"
    return "기타"


def cochran_armitage(counts):
    """counts: [(score, hits, n)] → (Z, 양측 p)"""
    N = sum(n for _, _, n in counts); H = sum(h for _, h, _ in counts); p = H / N
    sbar = sum(s * n for s, _, n in counts) / N
    T = sum(s * (h - n * p) for s, h, n in counts)
    V = p * (1 - p) * sum(n * (s - sbar) ** 2 for s, _, n in counts)
    z = T / math.sqrt(V)
    return z, math.erfc(abs(z) / math.sqrt(2))


def boot_ci(flags, B=2000):
    n = len(flags); vals = sorted(sum(flags[int(random.random() * n)] for _ in range(n)) / n * 100
                                  for _ in range(B))
    return vals[int(.025 * B)], vals[int(.975 * B)]


def main():
    random.seed(42)
    titles = pd.concat([pd.read_csv(BASE / f, usecols=["ncd", "list_title"]) for f in FILES.values()]) \
        .groupby("ncd").list_title.first()
    sec = pd.read_parquet(BASE / "master" / "sections.parquet")
    rows = []
    for r in sec[sec.speaker == "리포트"].itertuples():
        t = strip_quotes(str(r.text_raw), r.quote_spans)
        for s in re.split(r"(?<=[다요까])[.?!]\s*", t):
            s = s.strip()
            if len(s) >= 5:
                rows.append((r.year, r.ncd, corner(titles.get(r.ncd, "")), s, classify(s)))
    df = pd.DataFrame(rows, columns=["year", "ncd", "corner", "sent", "type"])
    df["헤지"] = df.type != "기타"

    def table(g, label):
        out = []
        for col in ["헤지"] + TYPES:
            flags = {y: (g[g.year == y].type == col).tolist() if col != "헤지" else g[g.year == y].헤지.tolist()
                     for y in YEARS}
            z, p = cochran_armitage([(y, sum(f), len(f)) for y, f in flags.items()])
            row = {"구분": label, "지표": col}
            for y, f in flags.items():
                lo, hi = boot_ci(f)
                row[str(y)] = f"{sum(f) / len(f) * 100:.1f} [{lo:.1f},{hi:.1f}]"
            row.update(N=" / ".join(str(len(f)) for f in flags.values()), CA_Z=round(z, 2), p=f"{p:.3g}")
            out.append(row)
        return out

    year_tab = pd.DataFrame(table(df, "전체"))
    corner_tab = pd.DataFrame(table(df[df.corner == "내부"], "내부(검증곤란)")
                              + table(df[df.corner == "정세"], "정세(관측가능)"))
    pd.set_option("display.width", 250)
    print("리포트 나레이션(인용 제외) 헤지 유형 비율 % [95% CI]\n" + year_tab.to_string(index=False))
    print("\n소재(코너)별\n" + corner_tab.to_string(index=False))
    mix = df.groupby(["year", "corner"]).size().unstack().apply(lambda r: (r / r.sum() * 100).round(1), axis=1)
    print("\n코너별 문장 구성비 %\n" + mix.to_string())
    year_tab.to_csv(BASE / "master" / "hedge_types_year.csv", index=False)
    corner_tab.to_csv(BASE / "master" / "hedge_types_corner.csv", index=False)


if __name__ == "__main__":
    main()
