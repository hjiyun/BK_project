#!/usr/bin/env python3
"""LLM 분류 검수 시트 생성(make)과 채점(score) — 내부 검수용(원문 입력 텍스트 포함, 외부 배포 금지).

표본: 무작위 60건 = KBS 30(연도별 8·8·7·7) + 유튜브 30(장르별 15) / 저확신 추가 6건(KBS 3·유튜브 3, confidence=low).
      오분류율은 '무작위' 행으로만 산출(저확신 추가분은 오류 패턴 파악용).
블라인드: 코더1·코더2 시트 분리, LLM 판정은 숨김 시트(코딩 완료 후 열람). 추출 구분도 코더에게 비표시.
채점: 코더별 LLM 일치율(=1−오분류율)과 Wilson 95% CI(전체·매체별), 코더 간·코더–LLM Cohen's κ.
  python3 review_sheet.py make    # → out/LLM분류_검수시트_v1.xlsx
  python3 review_sheet.py score [파일]   # 작성된 시트 채점 → out/review_score.csv
"""
import math
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

import classify as C

OUT = C.OUT / "LLM분류_검수시트_v1.xlsx"
SEED = 7
FONT = "맑은 고딕"
YELLOW = PatternFill("solid", fgColor="FFFF00")
HEAD = PatternFill("solid", fgColor="2A78D6")
THIN = Border(*(Side(style="thin", color="C3C2B7"),) * 4)
CODE_COLS = ["소재", "묶음", "프레임", "비고"]


def sample():
    lab = pd.read_csv(C.OUT / "labels.csv")
    docs = C.build_docs()[["doc_id", "text"]]
    lab = lab.merge(docs, on="doc_id")
    picks = []
    for y, n in zip([2018, 2020, 2022, 2024], [8, 8, 7, 7]):
        picks.append(lab[(lab.media == "KBS") & (lab.year == y)].sample(n, random_state=SEED))
    for g in ["생활·경험형", "정세·체제형"]:
        picks.append(lab[lab.genre == g].sample(15, random_state=SEED))
    rnd = pd.concat(picks).assign(추출="무작위")
    rest = lab[~lab.doc_id.isin(rnd.doc_id) & (lab.confidence == "low")]
    low = pd.concat([rest[rest.media == m].sample(min(3, (rest.media == m).sum()), random_state=SEED)
                     for m in ["KBS", "유튜브"]]).assign(추출="저확신 추가")
    s = pd.concat([rnd, low]).sample(frac=1, random_state=SEED).reset_index(drop=True)   # 순서 섞기
    # 제목·URL
    titles = pd.concat([pd.read_csv(C.RESEARCH / "kbs" / f, usecols=["ncd", "list_title", "url"])
                        for f in C.KBS_FILES.values()]).groupby("ncd").first()
    v = pd.read_parquet(C.RESEARCH / "youtube" / "master" / "videos.parquet").set_index("video_id")
    s["제목"] = [titles.list_title.get(int(d[4:]), "") if d.startswith("kbs_") else v.title.get(d[3:], "")
                for d in s.doc_id]
    s["URL"] = [titles.url.get(int(d[4:]), "") if d.startswith("kbs_") else v.url.get(d[3:], "") for d in s.doc_id]
    s["번호"] = range(1, len(s) + 1)
    return s


def head(ws, cols, widths):
    for i, (c, w) in enumerate(zip(cols, widths), 1):
        cell = ws.cell(1, i, c)
        cell.font = Font(name=FONT, bold=True, color="FFFFFF"); cell.fill = HEAD
        cell.alignment = Alignment(wrap_text=True, vertical="center"); cell.border = THIN
        ws.column_dimensions[cell.column_letter].width = w
    ws.freeze_panes = "B2"


def coder_sheet(wb, name, s):
    ws = wb.create_sheet(name)
    cols = ["번호", "doc_id", "매체", "연도", "장르·채널", "제목", "URL", "LLM이 본 입력(제목+앞 1,500자)"] + CODE_COLS
    head(ws, cols, [6, 16, 7, 7, 14, 30, 22, 70, 14, 8, 14, 30])
    for r, row in enumerate(s.itertuples(), 2):
        vals = [row.번호, row.doc_id, row.media, str(row.year), row.genre if row.media == "유튜브" else "KBS",
                row.제목, row.URL, row.text]
        for c, v in enumerate(vals, 1):
            cell = ws.cell(r, c, v)
            cell.font = Font(name=FONT, size=9); cell.border = THIN
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        for c in range(9, 13):
            cell = ws.cell(r, c); cell.fill = YELLOW; cell.border = THIN
            cell.font = Font(name=FONT, size=10, color="0000FF"); cell.alignment = Alignment(vertical="top", wrap_text=True)
        ws.row_dimensions[r].height = 150
    last = len(s) + 1
    for col, opts in [("I", C.TOPICS), ("J", ["예", "아니오"]), ("K", C.FRAMES)]:
        dv = DataValidation(type="list", formula1='"' + ",".join(opts) + '"', allow_blank=True,
                            showErrorMessage=True, errorTitle="범주 밖 값", error="목록에서 고르세요")
        dv.add(f"{col}2:{col}{last}"); ws.add_data_validation(dv)
    ws["I1"].comment = Comment("범주정의 시트 기준. 문서가 '주로' 다루는 대상 하나", "검수안내")
    ws["J1"].comment = Comment("서로 무관한 소식 2개 이상을 연달아 전하면 '예'", "검수안내")
    ws["K1"].comment = Comment("잠정 v0 체계. 체계 확정 전이면 비워 둬도 됨", "검수안내")
    return ws


def make():
    s = sample()
    wb = Workbook(); ws = wb.active; ws.title = "안내"
    guide = [
        ("LLM 분류 검수 시트 v1 — 내부용 (원문 포함, 외부 배포 금지)", True),
        ("", False),
        ("1. 목적: LLM(claude-opus-5-5, 프롬프트 6795c5d7693c)의 소재·묶음·프레임 판정이 맞는지 사람이 확인. 허용 오분류율 10% (8.25 회의).", False),
        ("2. 표본: 66건 (무작위 60 = KBS 30 + 유튜브 30, 그 외 6건은 오류 패턴 확인용). 순서는 섞여 있음.", False),
        ("3. 코더는 자기 시트(검수_코더1 또는 검수_코더2)만 연다. 다른 코더 시트와 LLM 판정(숨김 시트)은 코딩을 모두 마친 뒤에 본다.", False),
        ("4. 판정 근거는 'LLM이 본 입력' 열(제목 + 앞 1,500자). 원문 전체를 봐야 판정이 달라지면 비고에 '전체 보면 ○○'라고 적는다.", False),
        ("5. 입력 칸: 노란 칸(소재·묶음·프레임·비고)만. 소재·묶음·프레임은 드롭다운에서 고른다. 기준은 '범주정의' 시트.", False),
        ("6. 프레임은 잠정 v0 체계다. 팀이 체계를 바꿀 예정이면 프레임 열은 비워 두고 소재·묶음만 한다.", False),
        ("7. 끝나면 파일을 저장해 홍지윤에게 전달 → python3 review_sheet.py score 로 오분류율·Cohen's κ 산출.", False),
        ("", False),
        ("작성 예시 (실제 표본 아님):", True),
    ]
    for i, (t, b) in enumerate(guide, 1):
        c = ws.cell(i, 1, t); c.font = Font(name=FONT, bold=b, size=12 if i == 1 else 10)
    ex_cols = ["번호", "제목", "소재", "묶음", "프레임", "비고"]
    ex = [0, "[요즘 북한은] 평양 물가 급등…시장 단속 강화 외", "주민생활·경제", "예", "정보전달",
          "앞 소식(물가)이 비중 커서 주민생활로. 둘째 소식은 단속(내부정치)"]
    for c, (h, v) in enumerate(zip(ex_cols, ex), 1):
        ws.cell(12, c, h).font = Font(name=FONT, bold=True, size=10)
        cell = ws.cell(13, c, v); cell.font = Font(name=FONT, size=10, color="0000FF"); cell.fill = YELLOW
    ws.column_dimensions["A"].width = 120
    for col in "BCDEF":
        ws.column_dimensions[col].width = 22

    d = wb.create_sheet("범주정의")
    head(d, ["구분", "값", "정의·경계 규칙"], [10, 16, 110])
    body = C.PROMPT.read_text(encoding="utf-8")
    r = 2
    for sect, key in [("소재", "## 1."), ("프레임", "## 2.")]:
        block = body[body.index(key):]
        block = block[:block.index("\n## ", 3)] if "\n## " in block[3:] else block
        for line in block.splitlines():
            m = [x.strip() for x in line.strip().strip("|").split("|")] if line.startswith("|") else None
            if m and m[0] not in ("값", "---") and not set(m[0]) <= set("-"):
                d.cell(r, 1, sect); d.cell(r, 2, m[0]); d.cell(r, 3, " / ".join(x for x in m[1:] if x))
                for c in range(1, 4):
                    d.cell(r, c).font = Font(name=FONT, size=10); d.cell(r, c).alignment = Alignment(wrap_text=True)
                r += 1
    d.cell(r, 1, "묶음"); d.cell(r, 2, "예/아니오")
    d.cell(r, 3, "서로 무관한 독립 소식 2개 이상을 연달아 전하거나 여러 소재를 비슷한 비중으로 다루면 '예'. 이때 소재는 비중이 가장 큰 것")
    for c in range(1, 4):
        d.cell(r, c).font = Font(name=FONT, size=10); d.cell(r, c).alignment = Alignment(wrap_text=True)

    coder_sheet(wb, "검수_코더1", s)
    coder_sheet(wb, "검수_코더2", s)

    L = wb.create_sheet("LLM판정")
    head(L, ["번호", "doc_id", "추출", "매체", "소재", "묶음", "프레임", "확신도", "근거"], [6, 16, 10, 7, 14, 8, 14, 8, 50])
    for r, row in enumerate(s.itertuples(), 2):
        for c, v in enumerate([row.번호, row.doc_id, row.추출, row.media, row.topic,
                               "예" if row.is_bundle else "아니오", row.frame, row.confidence, row.rationale], 1):
            L.cell(r, c, v).font = Font(name=FONT, size=9)
    L.sheet_state = "hidden"

    A = wb.create_sheet("집계")
    n = len(s) + 1
    A["A1"] = "LLM 일치율 — 무작위 행만 (1 − 오분류율). 코딩이 끝난 행만 분모로 셈. κ·CI는 score 스크립트"
    A["A1"].font = Font(name=FONT, bold=True)
    head_row = ["코더", "항목", "매체", "완료 행", "LLM과 일치", "일치율", "오분류율"]
    for c, h in enumerate(head_row, 1):
        cell = A.cell(3, c, h); cell.font = Font(name=FONT, bold=True, color="FFFFFF"); cell.fill = HEAD
    r = 4
    for coder in ["검수_코더1", "검수_코더2"]:
        for item, cc, lc in [("소재", "I", "E"), ("묶음", "J", "F"), ("프레임", "K", "G")]:
            for media in ["전체", "KBS", "유튜브"]:
                rng_c = f"'{coder}'!${cc}$2:${cc}${n}"
                cond = f"('LLM판정'!$C$2:$C${n}=\"무작위\")*({rng_c}<>\"\")"
                if media != "전체":
                    cond += f"*('LLM판정'!$D$2:$D${n}=\"{media}\")"
                A.cell(r, 1, coder.replace("검수_", "")); A.cell(r, 2, item); A.cell(r, 3, media)
                A.cell(r, 4, f"=SUMPRODUCT({cond})")
                A.cell(r, 5, f"=SUMPRODUCT({cond}*({rng_c}='LLM판정'!${lc}$2:${lc}${n}))")
                A.cell(r, 6, f'=IF(D{r}=0,"",E{r}/D{r})').number_format = "0.0%"
                A.cell(r, 7, f'=IF(D{r}=0,"",1-F{r})').number_format = "0.0%"
                for c in range(1, 8):
                    A.cell(r, c).font = Font(name=FONT, size=10)
                r += 1
    r += 1
    A.cell(r, 1, "코더 간 일치율 (무작위 행, 두 코더 모두 완료)").font = Font(name=FONT, bold=True); r += 1
    for item, cc in [("소재", "I"), ("묶음", "J"), ("프레임", "K")]:
        a, b = f"'검수_코더1'!${cc}$2:${cc}${n}", f"'검수_코더2'!${cc}$2:${cc}${n}"
        cond = f"('LLM판정'!$C$2:$C${n}=\"무작위\")*({a}<>\"\")*({b}<>\"\")"
        A.cell(r, 2, item); A.cell(r, 4, f"=SUMPRODUCT({cond})"); A.cell(r, 5, f"=SUMPRODUCT({cond}*({a}={b}))")
        A.cell(r, 6, f'=IF(D{r}=0,"",E{r}/D{r})').number_format = "0.0%"
        for c in range(1, 7):
            A.cell(r, c).font = Font(name=FONT, size=10)
        r += 1
    A.cell(r + 1, 1, "기준: 오분류율 ≤ 10% (8.25 회의). 30건 기준 3건 이하. 표본이 작아 CI가 넓다는 점은 score 출력 참조.") \
        .font = Font(name=FONT, size=9, color="52514E")
    for col, w in zip("ABCDEFG", [10, 8, 8, 9, 11, 9, 9]):
        A.column_dimensions[col].width = w
    wb.save(OUT)
    print(f"→ {OUT}  ({len(s)}건: 무작위 {(s.추출 == '무작위').sum()}, 저확신 추가 {(s.추출 != '무작위').sum()}; "
          f"KBS {(s.media == 'KBS').sum()} / 유튜브 {(s.media == '유튜브').sum()})")


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def kappa(a, b):
    a, b = list(a), list(b)
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def score(path=OUT):
    wb = load_workbook(path)
    llm = pd.DataFrame(wb["LLM판정"].values).iloc[1:, :7]
    llm.columns = ["번호", "doc_id", "추출", "매체", "소재", "묶음", "프레임"]
    coders = {}
    for name in ["검수_코더1", "검수_코더2"]:
        df = pd.DataFrame(wb[name].values).iloc[1:, [0, 8, 9, 10]]
        df.columns = ["번호", "소재", "묶음", "프레임"]
        coders[name[-3:]] = df.set_index("번호")
    llm = llm.set_index("번호")
    rows = []
    for item in ["소재", "묶음", "프레임"]:
        for cname, df in coders.items():
            for media in ["전체", "KBS", "유튜브"]:
                m = (llm.추출 == "무작위") & df[item].notna() & (df[item] != "")
                if media != "전체":
                    m &= llm.매체 == media
                k = int((df.loc[m, item] == llm.loc[m, item]).sum()); n = int(m.sum())
                lo, hi = wilson(n - k, n)
                rows.append(dict(항목=item, 비교=f"{cname}–LLM", 매체=media, n=n, 오분류=n - k,
                                 오분류율=round((n - k) / n * 100, 1) if n else None,
                                 CI95=f"[{lo * 100:.1f}, {hi * 100:.1f}]" if n else "",
                                 kappa=round(kappa(df.loc[m, item], llm.loc[m, item]), 3) if n else None,
                                 기준충족=("예" if n and (n - k) / n <= 0.10 else "아니오") if n else ""))
        a, b = coders["코더1"][item], coders["코더2"][item]
        m = (llm.추출 == "무작위") & a.notna() & b.notna() & (a != "") & (b != "")
        rows.append(dict(항목=item, 비교="코더1–코더2", 매체="전체", n=int(m.sum()),
                         kappa=round(kappa(a[m], b[m]), 3) if m.any() else None))
    res = pd.DataFrame(rows)
    res.to_csv(C.OUT / "review_score.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    # 오분류 사례 (패턴 파악용, 저확신 추가분 포함)
    for cname, df in coders.items():
        bad = llm[(df.소재.notna()) & (df.소재 != "") & (df.소재 != llm.소재)]
        if len(bad):
            print(f"\n[{cname} 소재 불일치 {len(bad)}건]")
            print(pd.concat([bad[["doc_id", "추출", "소재"]], df.loc[bad.index, ["소재"]].add_prefix("코더_")], axis=1)
                  .to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "make"
    make() if cmd == "make" else score(Path(sys.argv[2]) if len(sys.argv) > 2 else OUT)
