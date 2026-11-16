#!/usr/bin/env python3
"""결과표 모음 — KBS 2018·2024 / 유튜브 2024 / 2024 KBS vs 유튜브. 해석 없이 표와 산출 조건만.

독자: 통일연구원(인문·사회계 북한 연구자, 코딩·통계 비전공) — 열 이름·주석은 평이한 우리말, 기술 열은 제외.
분석 산출물(CSV·parquet)에서 표를 직접 생성한다(수기 전사 없음). 남측 지칭 2018 vs 2024 직접 차이 CI만 여기서 새로 계산.
출력: 01_결과표_KBS2018·2024_유튜브2024.md / .xlsx  — handoff_build.py 가 호출(.docx 변환도 그쪽).
"""
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).parent
YT, KB, LLM = R / "youtube" / "master", R / "kbs" / "master", R / "llm_classify" / "out"
sys.path.insert(0, str(R / "youtube"))
STEM = "01_결과표_KBS2018·2024_유튜브2024"

# 열 이름·값 우리말 변환
SOUTH = {"관계어": "남측·남녘·남조선", "국가명": "한국·대한민국", "지역명": "남한", "환유": "청와대·용산"}
NORTH = {"국가명": "북한", "관계어": "북측·북녘·이북", "북한식자칭": "북조선·공화국 등", "적대어": "북괴"}
HEDGE = {"헤지": "추정·전언 전체", "출처불명시형": "출처 없는 전언", "화자추정형": "화자의 추정",
         "전체": "전체 기사", "내부(검증곤란)": "북한 내부 소식 코너", "정세(관측가능)": "정세 분석 코너"}
VER = {"남조선제외": "'남조선' 제외", "민감도(+남쪽)": "'남쪽' 포함", "북한식자칭 제외": "북조선·공화국 제외",
       "민감도(+북쪽)": "'북쪽' 포함"}
MODEL = {"koelectra": "모델 1 (KoELECTRA)", "mbert": "모델 2 (다국어 BERT)"}
SIG = {"예": "유의함", "아니오": "유의하지 않음"}


def md(df, floatfmt="{:.1f}"):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v).replace("|", "/"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


class Doc:
    def __init__(self):
        self.lines, self.sheets = [], []

    def h(self, text, lvl=2):
        self.lines += ["", "#" * lvl + " " + text, ""]

    def t(self, code, title, df, note="", fmt="{:.1f}"):
        self.lines += [f"**{code}. {title}**", "", md(df, fmt), ""]
        if note:
            self.lines += [f"※ {note}", ""]
        name = re.sub(r"[\[\]:*?/\\]", "", f"{code} {title}")[:31]
        self.sheets.append((name, df, note))


def ci_txt(s):
    """'[64.0, 80.5]' 또는 '5.3 [4.7,6.0]' → 대괄호 안 두 값 '64.0~80.5'"""
    m = re.search(r"\[([^\]]*)\]", str(s))
    v = re.findall(r"-?\d+\.?\d*", m.group(1) if m else str(s))
    return f"{v[0]}~{v[1]}" if len(v) >= 2 else ""


# ---------------------------------------------------------------- 계산 보조
def kbs_south(years=(2018, 2024)):
    import referent_analysis_yt as ra
    random.seed(42)
    sec = pd.read_parquet(KB / "sections.parquet")
    store, rows = {}, []
    for y in years:
        inc, exc = [], []
        for _, g in sec[(sec.layer == 1) & (sec.year == y)].groupby("ncd"):
            fi, fe = [], []
            for r in g.itertuples():
                spans = [(s["start"], s["end"]) for s in json.loads(r.quote_spans)]
                for f, st in ra.hits(str(r.text_raw)):
                    fi.append(f)
                    if not any(a <= st < b for a, b in spans):
                        fe.append(f)
            inc.append(ra.to_counts(fi)); exc.append(ra.to_counts(fe))
        for ver, rs in [("인용 제외", exc), ("인용 포함", inc)]:
            p, n = ra.share(rs); lo, hi = ra.ci(ra.boot(rs))
            rows.append({"연도": y, "기준": ver, **{SOUTH[t]: round(p[t], 1) for t in ra.TYPES},
                         "사례 수": n, "한국·대한민국 신뢰구간": f"{lo:.1f}~{hi:.1f}"})
            store[(y, ver)] = rs
    diffs = []
    for ver in ["인용 제외", "인용 포함"]:
        a, b = store[(years[0], ver)], store[(years[1], ver)]
        for t in ["국가명", "관계어", "지역명", "환유"]:
            d = sorted(x - y for x, y in zip(ra.boot(b, t), ra.boot(a, t)))
            est = ra.share(b)[0][t] - ra.share(a)[0][t]
            lo, hi = d[int(.025 * len(d))], d[int(.975 * len(d))]
            diffs.append({"기준": ver, "표현": SOUTH[t], "2018→2024 변화(%p)": round(est, 1),
                          "신뢰구간": f"{lo:.1f}~{hi:.1f}", "판정": "유의함" if lo > 0 or hi < 0 else "유의하지 않음"})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def topic_yearly():
    q = pd.read_csv(KB / "topic_quarter.csv")
    q["연도"] = q["bin"].str[:4].astype(int)
    cols = [c for c in q.columns if c not in ("bin", "N", "연도")]
    yr = q.groupby("연도").apply(lambda g: pd.Series({c: np.average(g[c], weights=g.N) for c in cols}),
                                include_groups=False)
    return yr.T.reset_index().rename(columns={"index": "주제"})


def llm_dist(col, groups, no_bundle=False):
    lab = pd.read_csv(LLM / "labels.csv").dropna(subset=[col])
    lab["집단"] = np.where(lab.media == "KBS", "KBS " + lab.year.astype(str), "유튜브 " + lab.genre.astype(str))
    if no_bundle:
        lab = lab[~lab.is_bundle]
    tab = (pd.crosstab(lab[col], lab.집단, normalize="columns") * 100)
    tab = tab[[g for g in groups if g in tab.columns]]
    n = lab.groupby("집단").size()
    tab = tab.apply(lambda c: c.map(lambda x: f"{x:.1f}%"))
    tab.loc["(기사·영상 수)"] = [str(int(n.get(g, 0))) for g in tab.columns]
    return tab.reset_index().rename(columns={col: "소재" if col == "topic" else "프레임"})


def llm_sig(groups, col, no_bundle=False):
    from scipy.stats import chi2_contingency
    lab = pd.read_csv(LLM / "labels.csv").dropna(subset=[col])
    if no_bundle:
        lab = lab[~lab.is_bundle]
    lab["집단"] = np.where(lab.media == "KBS", "KBS " + lab.year.astype(str), "유튜브 " + lab.genre.astype(str))
    sub = lab[lab.집단.isin(groups)]
    p = chi2_contingency(pd.crosstab(sub[col], sub.집단))[1]
    return "구성 차이: " + ("유의함" if p < 0.05 else "유의하지 않음") + f" (p={p:.2g})"


def assoc_table(corpora, keywords, k=8):
    a = pd.read_csv(YT / "cooc_assoc.csv")
    rows = []
    for kw in keywords:
        row = {"핵심어": kw}
        for c in corpora:
            s = a[(a.코퍼스 == c) & (a.핵심어 == kw)].head(k)
            n = int(s.핵심어문장수.iloc[0]) if len(s) else 0
            row[f"{c} — 함께 나온 단어 (핵심어가 나온 문장 수)"] = " · ".join(s.연관어) + f"  ({n:,})"
        rows.append(row)
    return pd.DataFrame(rows)


def context_sim(pairs):
    cs = pd.read_csv(YT / "cooc_context_sim.csv")
    rows = []
    for kw in ["김정은", "대한민국", "한국", "주민", "통일", "러시아", "탈북민", "미국"]:
        row = {"핵심어": kw}
        for a, b in pairs:
            m = cs[(cs.핵심어 == kw) & (((cs.코퍼스1 == a) & (cs.코퍼스2 == b)) | ((cs.코퍼스1 == b) & (cs.코퍼스2 == a)))]
            row[f"{a} ↔ {b}"] = round(float(m.PPMI코사인.iloc[0]), 2) if len(m) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def polarity(corpora):
    p = pd.read_csv(YT / "polarity_summary.csv")
    p = p[p.코퍼스.isin(corpora)]
    return pd.DataFrame({"모델": p.모델.map(MODEL), "자료": p.코퍼스, "문장 수": p.문장.map(lambda x: f"{x:,}"),
                         "점수의 흩어짐": p.분산.map(lambda s: s.split(" ")[0]),
                         "흩어짐 신뢰구간": p.분산.map(ci_txt),
                         "강한 부정 문장 비율(%)": p.강한부정.map(lambda s: s.split(" ")[0]),
                         "비율 신뢰구간": p.강한부정.map(ci_txt)})


def top_words(df_pos, df_neg, col_word, col_z, name_pos, name_neg, k=20, n_ch=None):
    a = df_pos.nlargest(k, col_z); b = df_neg.nsmallest(k, col_z)
    pos = [f"{w} ({z:.1f})" + (f" [{n}/6]" if n_ch is not None else "")
           for w, z, n in zip(a[col_word], a[col_z], a[n_ch] if n_ch else [None] * k)]
    return pd.DataFrame({"순위": range(1, k + 1), name_pos: pos,
                         name_neg: [f"{w} ({-z:.1f})" for w, z in zip(b[col_word], b[col_z])]})


POL_NOTE = ("문장마다 긍정·부정 정도를 −1~+1로 매긴 컴퓨터 판정. '점수의 흩어짐'이 클수록 긍정·부정이 강한 문장이 많다. "
            "'강한 부정'은 −0.9 이하. 영화 리뷰로 학습된 모델이라 보조 자료로만 본다. 두 모델 사이 결과가 엇갈린 지표는 싣지 않았다.")


# ---------------------------------------------------------------- 본문
def build():
    D = Doc()
    D.lines += ["# 결과표 모음 — KBS 2018·2024 / 유튜브 2024 / 2024년 KBS와 유튜브 비교", "",
                "- 작성: 2026년 11월 16일 · N.K.STATS",
                "- **결과에 대한 해석은 싣지 않았습니다.** 표와, 무엇을 어떻게 셌는지만 적었습니다.",
                "- 용어는 결과 보고서 맨 앞 '용어 풀이'를 참고해 주십시오. 자주 나오는 표현:",
                "  - **신뢰구간**: 자료를 다시 모았을 때 값이 놓일 범위(95%). 같은 값이라도 표마다 따로 계산해 소수점 첫째 자리가 조금 다를 수 있습니다.",
                "  - **유의함**: 차이가 우연으로 보기 어려움(차이의 신뢰구간이 0을 포함하지 않음).",
                "  - **KBS 자체 발화**: 앵커·리포터·KBS 기자의 말. **인용 제외**는 녹취·인용을 빼고 KBS 자신의 말만 센 것.",
                "  - **유튜브 두 유형**: 생활·경험형(유미카·이철은, 탈북민 게스트 인터뷰) / 정세·체제형(강철환tv·주성하TV·배나TV·이소연TV, 진행자 시사 해설).",
                "- 2018년 유튜브는 비교하지 않습니다(대상 채널 대부분이 2019년 이후 개설). KBS 2020·2022년 값은 결과 보고서에 있습니다.",
                "- **AI 분류 표(A9·A10, B8·B9, C10·C11)는 사람 검수 전 잠정치**이며, 프레임 범주도 확정 전 초안입니다.", ""]

    # ===================== A. KBS 2018·2024
    D.h("A. KBS 〈남북의 창〉 2018년 · 2024년")
    art = pd.read_parquet(KB / "articles.parquet")
    D.t("A1", "자료 규모", pd.DataFrame({
        "항목": ["기사 수", "방송 회차", "글자 수", "단어 수 (KBS 자체 발화)", "대본이 없어 빠진 회차"],
        "2018": ["131", "43", "522,618", f"{art[art.year == 2018].token_len.sum():,}", "9회 (17.3%)"],
        "2024": ["147", "49", "556,969", f"{art[art.year == 2024].token_len.sum():,}", "3회 (5.8%)"]}))

    south, sdiff = kbs_south()
    D.t("A2", "남한을 부르는 말 — 표현별 비율 (%)", south,
        "KBS 자체 발화. 남한을 가리킨 표현 전체를 100으로 했을 때 각 표현의 몫. '사례 수'는 남한을 가리킨 표현의 총 횟수.")
    D.t("A3", "남한을 부르는 말 — 2018년과 2024년의 차이", sdiff)

    nr = pd.read_csv(YT / "referent_north.csv")
    a4 = nr[nr.집단.isin(["KBS 2018", "KBS 2024"])]
    D.t("A4", "북한을 부르는 말 — 표현별 비율 (%)", pd.DataFrame({
        "연도": a4.집단.str[-4:], "기준": a4.버전, **{NORTH[t]: a4[t] for t in NORTH}, "사례 수": a4.N,
        "'북한' 신뢰구간": a4.국가명CI.map(ci_txt)}),
        "'조선' 단독형(조선시대·단체 이름 등이 섞임)과 '평양'은 세지 않았다.", fmt="{:.2f}")
    nd = pd.read_csv(YT / "referent_north_kbs_diff.csv")
    nd = nd[nd.구간 == "2018→2024"]
    D.t("A5", "북한을 부르는 말 — 2018년과 2024년의 차이", pd.DataFrame({
        "기준": nd.버전, "표현": nd.유형.map(NORTH), "변화(%p)": nd.차이pp, "신뢰구간": nd.CI95.map(ci_txt),
        "판정": nd.유의.map(SIG)}), fmt="{:.2f}")

    k = pd.read_csv(YT / "kim_title_kbs_yt.csv")
    a6 = k[(k.매체 == "KBS") & k.연도.isin([2018, 2024])]
    D.t("A6", "'김정은'을 부르는 방식 (%)", pd.DataFrame({
        "연도": a6.연도, "기준": a6.버전, "직함 붙임": a6.직함, "'동지' 붙임": a6.동지, "이름만": a6.맨이름,
        "언급 수": a6.N, "직함 비율 신뢰구간": a6.직함CI.map(ci_txt)}),
        "직함 = 김정은 앞뒤에 위원장·국무위원장·총비서·원수 등이 붙은 경우.")

    h = pd.concat([pd.read_csv(KB / "hedge_types_year.csv"), pd.read_csv(KB / "hedge_types_corner.csv")])
    D.t("A7", "추정·전언 표현으로 끝나는 문장 비율 (%)", pd.DataFrame({
        "코너": h.구분.map(HEDGE), "표현": h.지표.map(HEDGE),
        "2018": h["2018"].str.split(" ").str[0], "2018 신뢰구간": h["2018"].map(ci_txt),
        "2024": h["2024"].str.split(" ").str[0], "2024 신뢰구간": h["2024"].map(ci_txt),
        "네 시점 추세": [("증가 " if z > 0 else "감소 ") + ("(유의함)" if p < 0.05 else "(유의하지 않음)") if p < 0.05 else "추세 없음"
                     for z, p in zip(h.CA_Z, h.p)]}),
        "KBS 리포트의 리포터 문장(인용 제외). 출처 없는 전언 = '~로 알려졌습니다·~로 전해졌습니다·~다고 합니다', "
        "화자의 추정 = '~로 보입니다·~로 풀이됩니다·~로 전망됩니다' 등. 북한 내부 소식 코너 = [요즘 북한은]·[클로즈업 북한], "
        "정세 분석 코너 = [이슈&한반도]. '네 시점 추세'는 2018·2020·2022·2024 전체로 판정.")
    D.t("A8", "북한 발언을 전할 때 거리를 두는 동사의 비율", pd.DataFrame({
        "항목": ["비율", "네 시점 추세", "2018→2024 차이의 신뢰구간"],
        "2018": ["17.1%", "추세 없음 (p=0.150)", "−6.8~+25.4 (0 포함, 유의하지 않음)"], "2024": ["27.1%", "", ""]}),
        "거리를 두는 동사 = '주장했다·선전했다' 등. 2026년 8월 보고 수치(계산 기록이 남아 있지 않아 다시 계산하지 않음). "
        "인용 출처 구성(북한 매체 10~13%, 전문가 38~49%, 관료 7~8%)은 네 해 동안 비슷.")

    D.t("A9", "AI 분류 — 기사의 소재 (잠정)", llm_dist("topic", ["KBS 2018", "KBS 2024"]),
        llm_sig(["KBS 2018", "KBS 2024"], "topic") + ". 여러 소식을 묶어 전하는 기사도 포함(가장 비중이 큰 소식 기준).")
    D.t("A10", "AI 분류 — 기사의 프레임 (잠정)", llm_dist("frame", ["KBS 2018", "KBS 2024"]),
        llm_sig(["KBS 2018", "KBS 2024"], "frame") + ".")

    lo = pd.read_csv(KB / "logodds_adjacent.csv")
    s = lo[(lo.버전 == "전체") & (lo.구간 == "2018→2024") & lo.채택]
    D.t("A11", "2018년과 2024년의 특징어 (상위 20개)",
        top_words(s[s.z > 0], s[s.z < 0], "단어", "z", "2024년에 많은 단어 (점수)", "2018년에 많은 단어 (점수)"),
        "한쪽 연도에서 유난히 자주 쓰인 단어. 점수가 클수록 그 연도에 치우침. 기사 5건 이상에 나온 단어만.")

    ty = topic_yearly()
    D.t("A12", "주제 구성 — KBS 기사를 8개 주제로 나눈 비중 (%)", ty[["주제", 2018, 2024]].rename(columns={2018: "2018", 2024: "2024"}),
        "주제는 컴퓨터가 함께 쓰이는 단어 묶음으로 나누고 연구진이 이름을 붙임. 각 기사의 주제 비중을 연도별로 평균.")

    D.t("A13", "같은 문장에 함께 나온 단어 (연관어, 상위 8개)",
        assoc_table(["KBS 2018", "KBS 2024"], ["김정은", "주민", "통일", "러시아", "탈북민", "미국", "한국"]),
        "왼쪽일수록 함께 나오는 경향이 강함. 리포트의 '[이름/직함 :' 출처 표기는 빼고 셈.")
    D.t("A14", "같은 단어의 맥락 유사도 — KBS 2018년 ↔ 2024년", context_sim([("KBS 2018", "KBS 2024")]),
        "0~1, 1에 가까울수록 두 해에 같은 맥락에서 쓰임. '대한민국'은 KBS에서 문장 수가 적어(30~42) 값이 불안정.", fmt="{:.2f}")
    D.t("A15", "긍정·부정 점수 (보조 자료)", polarity(["KBS 2018", "KBS 2024"]), POL_NOTE)

    # ===================== B. 유튜브 2024
    D.h("B. 탈북민·북한 유튜브 2024년")
    sz = pd.read_csv(YT / "summary_by_channel.csv")
    sz = sz[sz.year == 2024]
    v = pd.read_parquet(YT / "videos.parquet")
    sz = sz.assign(유형=sz.channel.map(dict(zip(v.channel, v.genre)))).sort_values(["유형", "토큰"], ascending=[True, False])
    D.t("B1", "자료 규모", pd.DataFrame({
        "유형": sz.유형, "채널": sz.channel, "영상 수": sz.편수.map(lambda x: f"{int(x):,}"),
        "글자 수": sz.글자.map(lambda x: f"{int(x):,}"), "단어 수": sz.토큰.map(lambda x: f"{int(x):,}"),
        "영상당 단어 수": sz.영상당토큰.map(lambda x: f"{int(x):,}"), "전체 중 비중(%)": sz["토큰비중%"]}),
        "자막은 모두 유튜브 자동 생성(음성 인식).")
    gr = pd.read_csv(YT / "genre_referent.csv")
    gr = gr[gr.대상 != "KBS"]
    D.t("B2", "남한을 부르는 말 — 유형별 비율 (%)", pd.DataFrame({
        "유형": gr.대상, "기준": gr.버전.map(lambda x: VER.get(x, x)), **{SOUTH[t]: gr[t] for t in SOUTH},
        "사례 수": gr.N, "한국·대한민국 신뢰구간": gr.국가명CI.map(ci_txt)}),
        "'남조선' 제외 = 탈북민이 북한에 있을 때 듣던 말을 옮기는 경우가 대부분이라 뺀 기준(KBS '인용 제외'에 해당).")
    bc = pd.read_csv(YT / "referent_2024_by_channel.csv")
    D.t("B3", "남한을 부르는 말 — 채널별 비율 (%)", bc.rename(columns={**SOUTH, "영상": "영상 수", "N": "사례 수"}))
    b4 = nr[nr.집단.str.startswith("유튜브")]
    D.t("B4", "북한을 부르는 말 — 유형별 비율 (%)", pd.DataFrame({
        "유형": b4.집단.str.replace("유튜브 ", ""), "기준": b4.버전.map(lambda x: VER.get(x, x)),
        **{NORTH[t]: b4[t] for t in NORTH}, "사례 수": b4.N}), fmt="{:.2f}")
    gk = pd.read_csv(YT / "genre_kim_title.csv")
    ck = k[k.매체.str.contains("└")]
    D.t("B5", "'김정은'을 부르는 방식 (%)", pd.DataFrame({
        "유형·채널": list(gk.대상) + [c.replace("└", "").strip() + " (채널)" for c in ck.매체],
        "직함 붙임": list(gk.직함) + list(ck.직함), "'동지' 붙임": list(gk.동지) + list(ck.동지),
        "이름만": list(gk.맨이름) + list(ck.맨이름), "언급 수": list(gk.N) + list(ck.N)}))
    lab6 = pd.read_csv(YT / "topic_labels_K6.csv")
    g6 = pd.read_csv(YT / "genre_topic_K6.csv").rename(columns={"Unnamed: 0": "토픽"}).merge(lab6[["토픽", "라벨"]], on="토픽")
    D.t("B6", "주제 구성 — 6개 주제 비중 (%)", pd.DataFrame({
        "주제": g6.라벨, "대표 단어": g6.상위어, "생활·경험형": g6["생활·경험형"], "정세·체제형": g6["정세·체제형"]}),
        "2024년 KBS 기사와 유튜브를 함께 6개 주제로 나눔(KBS 값은 C7). 6개는 비교한 4가지(6·8·10·12개) 중 "
        "주제의 일관성과 반복 계산 시 안정성이 가장 좋았던 수.")
    gl = pd.read_csv(YT / "genre_logodds_생활·경험형_vs_정세·체제형.csv")
    D.t("B7", "두 유형의 특징어 (상위 20개)",
        top_words(gl[gl.z > 0], gl[gl.z < 0], "word", "z", "생활·경험형에 많은 단어 (점수)", "정세·체제형에 많은 단어 (점수)"))
    D.t("B8", "AI 분류 — 영상의 프레임 (잠정)", llm_dist("frame", ["유튜브 생활·경험형", "유튜브 정세·체제형"]),
        "AI가 제목과 자막 앞부분(약 1,500자)을 읽고 분류.")
    D.t("B9", "AI 분류 — 영상의 소재 (잠정)", llm_dist("topic", ["유튜브 생활·경험형", "유튜브 정세·체제형"]))
    D.t("B10", "같은 문장에 함께 나온 단어 (연관어, 상위 8개)",
        assoc_table(["유튜브 생활·경험형", "유튜브 정세·체제형"],
                    ["김정은", "대한민국", "한국", "주민", "통일", "러시아", "탈북민", "미국"]))
    D.t("B11", "긍정·부정 점수 (보조 자료)", polarity(["유튜브 생활·경험형", "유튜브 정세·체제형"]), POL_NOTE)

    # ===================== C. 2024 비교
    D.h("C. 2024년 KBS와 유튜브 비교")
    ref = pd.read_csv(YT / "referent_2024_yt_vs_kbs.csv")
    D.t("C1", "남한을 부르는 말 — 매체별 비율 (%)", pd.DataFrame({
        "매체": ref.매체, "기준": ref.버전.map(lambda x: VER.get(x, x)), **{SOUTH[t]: ref[t] for t in SOUTH},
        "사례 수": ref.N, "한국·대한민국 신뢰구간": ref.국가명CI.map(ci_txt),
        "채널 단위로 다시 계산한 신뢰구간": ref.채널군집CI.fillna("").map(ci_txt),
        "채널별 값의 평균": ref.채널균등평균}),
        "유튜브는 분량이 두 채널에 쏠려 있어, 채널 단위로 다시 계산한 값과 채널별 평균을 함께 실음.")
    gd = pd.read_csv(YT / "genre_referent_diff.csv")
    D.t("C2", "'한국·대한민국' 비율의 차이", pd.DataFrame({
        "기준": gd.버전.map(lambda x: "전체 (KBS는 인용 포함)" if x == "전체" else "'남조선' 제외 (KBS는 인용 제외)"),
        "비교": gd.대비, "차이(%p)": gd.국가명차이pp, "신뢰구간": gd.CI95.map(ci_txt),
        "판정": [SIG["예"] if (float(re.findall(r"-?\d+\.?\d*", c)[0]) > 0 or float(re.findall(r"-?\d+\.?\d*", c)[1]) < 0)
               else SIG["아니오"] for c in gd.CI95]}))
    dq = pd.read_csv(YT / "kbs_defector_quotes.csv")
    D.t("C3", "KBS 방송에 인용된 탈북민 발언만 따로 센 결과", pd.DataFrame({
        "연도": dq.연도, "기사 수": dq.기사, "녹취 수": dq.스팬, "남한 지칭 횟수": dq.남측N,
        "한국·대한민국(%)": dq.국가명, "신뢰구간": dq.국가명CI.map(ci_txt), "남한(%)": dq.지역명,
        "김정은 언급 수": dq.김정은N, "김정은 직함(%)": dq.직함}),
        "KBS 리포트의 녹취 가운데 출처 표기에 '탈북·귀순·○○ 출신'이 적힌 것만 모음. KBS가 편집한 짧은 발언이고 사례 수가 적음.")
    c4 = nr[nr.집단.isin(["KBS 2024", "KBS 2024 탈북민 인용", "유튜브 생활·경험형", "유튜브 정세·체제형"])]
    D.t("C4", "북한을 부르는 말 — 2024년 비교 (%)", pd.DataFrame({
        "자료": c4.집단, "기준": c4.버전.map(lambda x: VER.get(x, x)), **{NORTH[t]: c4[t] for t in NORTH}, "사례 수": c4.N}),
        fmt="{:.2f}")
    k24 = k[(k.매체 == "KBS") & (k.연도 == 2024)]
    D.t("C5", "'김정은'을 부르는 방식 — 2024년 비교 (%)", pd.DataFrame({
        "자료": ["KBS (" + b + ")" for b in k24.버전] + ["KBS 탈북민 인용"] + ["유튜브 " + g for g in gk.대상],
        "직함 붙임": list(k24.직함) + [float(dq[dq.연도 == 2024].직함.iloc[0])] + list(gk.직함),
        "이름만": list(k24.맨이름) + [np.nan] + list(gk.맨이름),
        "언급 수": list(k24.N) + [int(dq[dq.연도 == 2024].김정은N.iloc[0])] + list(gk.N),
        "직함 비율 신뢰구간": list(k24.직함CI.map(ci_txt)) + [""] + list(gk.직함CI.map(ci_txt))}))
    yl = pd.read_csv(YT / "logodds_2024_yt_vs_kbs.csv")
    D.t("C6", "유튜브와 KBS의 특징어 (상위 20개)",
        top_words(yl[(yl.z > 0) & (yl.n_ch >= 4)], yl[yl.z < 0], "word", "z",
                  "유튜브에 많은 단어 (점수) [같은 경향을 보인 채널 수]", "KBS에 많은 단어 (점수)", n_ch="n_ch"),
        "유튜브 쪽은 여섯 채널 중 네 채널 이상에서 같은 경향을 보인 단어만.")
    D.t("C7", "주제 구성 — 6개 주제 비중 (%)", pd.DataFrame({
        "주제": g6.라벨, "KBS": g6.KBS, "생활·경험형": g6["생활·경험형"], "정세·체제형": g6["정세·체제형"]}))
    bal = pd.read_csv(YT / "topic_joint2024_balanced.csv")
    D.t("C8", "분량 차이 점검 — 유튜브를 KBS와 같은 수(147건)로 맞춰 다시 계산", pd.DataFrame({
        "주제 수": bal.K, "계산": bal.표본.str.replace("원본(불균형)", "전체 자료").str.replace("균형 seed", "같은 수 표본 "),
        "KBS 기사가 상위 2개 주제에 몰린 정도(%)": bal.KBS집중도, "두 매체가 함께 다루는 주제 수": bal.공유토픽,
        "두 매체 주제 구성의 차이(0~100)": bal.매체분리도}),
        "'주제 구성의 차이'는 0이면 두 매체가 같은 주제 구성, 100이면 전혀 겹치지 않음. "
        "'함께 다루는 주제'는 두 매체 모두 5% 이상 다루고 비중 차이가 2배 이내인 주제.")
    proj = pd.read_csv(YT / "topic_2024_yt_vs_kbs.csv").rename(columns={"Unnamed: 0": "주제"})
    D.t("C9", "KBS로 만든 8개 주제를 유튜브에 적용한 결과 (%)", pd.DataFrame({
        "주제": proj.주제, "KBS 2024": proj["KBS 2024"], "유튜브": proj["유튜브(청크평균)"],
        "유튜브 (채널별 평균)": proj["유튜브(채널균등)"], "차이 (유튜브−KBS)": proj["차이(청크-KBS)"]}),
        "유튜브에 쓰인 단어의 약 3분의 1은 KBS 기사에 나오지 않아 이 방식으로는 반영되지 않음.")
    grp3 = ["KBS 2024", "유튜브 생활·경험형", "유튜브 정세·체제형"]
    D.t("C10", "AI 분류 — 프레임 비교 (잠정)", llm_dist("frame", grp3), llm_sig(grp3, "frame") + ".")
    D.t("C11", "AI 분류 — 소재 비교 (잠정)", llm_dist("topic", grp3), llm_sig(grp3, "topic") + ".")
    D.t("C12", "같은 단어의 맥락 유사도 — 2024년 비교",
        context_sim([("KBS 2024", "유튜브 정세·체제형"), ("KBS 2024", "유튜브 생활·경험형"),
                     ("유튜브 생활·경험형", "유튜브 정세·체제형")]),
        "0~1, 1에 가까울수록 같은 맥락에서 쓰임. '대한민국'은 KBS 문장 수가 적어 불안정.", fmt="{:.2f}")
    D.t("C13", "긍정·부정 점수 — 2024년 비교 (보조 자료)", polarity(grp3), POL_NOTE)
    return D


def write_xlsx(D, path):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame([(n, note) for n, _, note in D.sheets], columns=["표", "설명"]).to_excel(xw, sheet_name="목차", index=False)
        for name, df, note in D.sheets:
            df.to_excel(xw, sheet_name=name, index=False)
            if note:
                xw.sheets[name].cell(len(df) + 3, 1, "※ " + note)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for row in ws.iter_rows():
                for c in row:
                    c.font = Font(name="맑은 고딕", size=10)
            for c in ws[1]:
                c.font = Font(name="맑은 고딕", bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="2A78D6")
                c.alignment = Alignment(wrap_text=True, vertical="center")
            for ci, col in enumerate(ws.columns, 1):
                w = max(len(str(c.value or "")) for c in list(col)[:60] if not str(c.value or "").startswith("※"))
                ws.column_dimensions[get_column_letter(ci)].width = min(max(8, w * 1.4), 60)


def main(pkg):
    D = build()
    pkg = Path(pkg)
    (pkg / f"{STEM}.md").write_text("\n".join(D.lines) + "\n", encoding="utf-8")
    write_xlsx(D, pkg / f"{STEM}.xlsx")
    return pkg / f"{STEM}.md", len(D.sheets)


if __name__ == "__main__":
    print(main(Path(sys.argv[1]) if len(sys.argv) > 1 else R))
