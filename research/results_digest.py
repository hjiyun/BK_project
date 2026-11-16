#!/usr/bin/env python3
"""결과표 모음 — KBS 2018·2024 / 유튜브 2024 / 2024 KBS vs 유튜브. 해석 없이 표와 산출 조건만.

분석 산출물(CSV·parquet)에서 표를 직접 생성한다(수기 전사 없음). 남측 지칭 2018 vs 2024 직접 차이 CI만 여기서 새로 계산.
출력(패키지 루트): 01_결과표_KBS2018·2024_유튜브2024.md / .docx / .xlsx  — handoff_build.py 가 호출.
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

    def p(self, text):
        self.lines += [text, ""]

    def t(self, code, title, df, note="", fmt="{:.1f}"):
        self.lines += [f"**{code}. {title}**", "", md(df, fmt), ""]
        if note:
            self.lines += [f"주: {note}", ""]
        name = re.sub(r"[\[\]:*?/\\]", "", f"{code} {title}")[:31]
        self.sheets.append((name, df, note))


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
            rows.append({"연도": y, "버전": ver, **{t: round(p[t], 1) for t in ra.TYPES}, "N": n,
                         "국가명 95% CI": f"[{lo:.1f}, {hi:.1f}]"})
            store[(y, ver)] = rs
    diffs = []
    for ver in ["인용 제외", "인용 포함"]:
        a, b = store[(years[0], ver)], store[(years[1], ver)]
        for t in ["국가명", "관계어", "지역명", "환유"]:
            d = sorted(x - y for x, y in zip(ra.boot(b, t), ra.boot(a, t)))
            est = ra.share(b)[0][t] - ra.share(a)[0][t]
            lo, hi = d[int(.025 * len(d))], d[int(.975 * len(d))]
            diffs.append({"버전": ver, "유형": t, "2018→2024 차이(%p)": round(est, 1),
                          "95% CI": f"[{lo:.1f}, {hi:.1f}]", "유의": "예" if lo > 0 or hi < 0 else "아니오"})
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def topic_yearly():
    q = pd.read_csv(KB / "topic_quarter.csv")
    q["연도"] = q["bin"].str[:4].astype(int)
    cols = [c for c in q.columns if c not in ("bin", "N", "연도")]
    yr = q.groupby("연도").apply(lambda g: pd.Series({c: np.average(g[c], weights=g.N) for c in cols}))
    return yr.T.reset_index().rename(columns={"index": "토픽"})


def llm_dist(col, groups):
    lab = pd.read_csv(LLM / "labels.csv").dropna(subset=[col])
    lab["집단"] = np.where(lab.media == "KBS", "KBS " + lab.year.astype(str), "유튜브 " + lab.genre.astype(str))
    out = []
    for name, sub in [("전체 문서", lab), ("bundle 제외", lab[~lab.is_bundle])]:
        tab = (pd.crosstab(sub[col], sub.집단, normalize="columns") * 100)
        tab = tab[[g for g in groups if g in tab.columns]]
        n = sub.groupby("집단").size()
        tab = tab.apply(lambda c: c.map(lambda x: f"{x:.1f}"))
        tab.loc["(문서 수)"] = [str(int(n.get(g, 0))) for g in tab.columns]
        out.append((name, tab.reset_index().rename(columns={col: "범주"})))
    return out


def llm_chi2(groups, col, no_bundle=False):
    from scipy.stats import chi2_contingency
    lab = pd.read_csv(LLM / "labels.csv").dropna(subset=[col])
    if no_bundle:
        lab = lab[~lab.is_bundle]
    lab["집단"] = np.where(lab.media == "KBS", "KBS " + lab.year.astype(str), "유튜브 " + lab.genre.astype(str))
    sub = lab[lab.집단.isin(groups)]
    chi2, p, dof, exp = chi2_contingency(pd.crosstab(sub[col], sub.집단))
    return f"χ²={chi2:.1f}, df={dof}, p={p:.2g}" + (" (기대빈도<5 셀 있음)" if (exp < 5).any() else "")


def assoc_table(corpora, keywords, k=8):
    a = pd.read_csv(YT / "cooc_assoc.csv")
    rows = []
    for kw in keywords:
        row = {"핵심어": kw}
        for c in corpora:
            s = a[(a.코퍼스 == c) & (a.핵심어 == kw)].head(k)
            n = int(s.핵심어문장수.iloc[0]) if len(s) else 0
            row[f"{c} (문장 n)"] = f"({n}) " + " · ".join(s.연관어)
        rows.append(row)
    return pd.DataFrame(rows)


def polarity(corpora):
    p = pd.read_csv(YT / "polarity_summary.csv")
    p = p[p.코퍼스.isin(corpora)]
    return p[["모델", "코퍼스", "문서", "문장", "분산", "강한극성", "강한부정", "평균"]]


# ---------------------------------------------------------------- 본문
def build():
    D = Doc()
    D.lines += [f"# 결과표 모음 — KBS 2018·2024 / 유튜브 2024 / 2024 KBS–유튜브 비교", "",
                "- 작성: 2026-11-16 · N.K.STATS",
                "- **해석은 수록하지 않았다.** 표와 산출 조건만 기술한다. 방법 상세는 00_결과보고서 4절·코드북.",
                "- CI는 기사(KBS)·영상(유튜브) 단위 부트스트랩 95% 구간. 'KBS'는 층1(앵커·리포트·취재진) 발화.",
                "- 같은 점추정치의 CI가 표마다 소수점 첫째 자리에서 다를 수 있다(표별 독립 부트스트랩 실행).",
                "- 2018년 유튜브는 공백(대상 채널 대부분 2019년 이후 개설). 2020·2022 KBS 값은 00_결과보고서·통합 엑셀에 있음.",
                "- LLM 소재·프레임 분류(A9·A10, B8·B9, C10·C11)는 **사람 검수 전 잠정치**, 프레임 체계는 v0.", ""]

    # ===================== A. KBS 2018·2024
    D.h("A. KBS 〈남북의 창〉 2018 · 2024")
    art = pd.read_parquet(KB / "articles.parquet")
    size = pd.DataFrame({"항목": ["기사(분석 단위)", "방송 회차", "글자 수", "층1 토큰", "결측 회차"],
                         "2018": ["131", "43", "522,618", f"{art[art.year == 2018].token_len.sum():,}", "9 (17.3%)"],
                         "2024": ["147", "49", "556,969", f"{art[art.year == 2024].token_len.sum():,}", "3 (5.8%)"]})
    D.t("A1", "코퍼스 규모", size)

    south, sdiff = kbs_south()
    D.t("A2", "남측 지칭 유형 점유율 (%)", south,
        "관계어=남측·남녘·남조선, 국가명=한국·대한민국, 지역명=남한, 환유=청와대·용산. 분모=남측 지칭 총 언급.")
    D.t("A3", "남측 지칭 2018→2024 차이", sdiff, "두 연도 기사 부트스트랩(각 2,000회) 차의 분위수 구간.")

    nr = pd.read_csv(YT / "referent_north.csv")
    D.t("A4", "북측 지칭 유형 점유율 (%)",
        nr[nr.집단.isin(["KBS 2018", "KBS 2024"])][["집단", "버전", "국가명", "관계어", "북한식자칭", "적대어", "N",
                                                     "국가명CI", "관계어CI"]], "국가명=북한, 관계어=북측·북녘·이북, 북한식자칭=북조선·공화국·국호, 적대어=북괴.",
        fmt="{:.2f}")
    nd = pd.read_csv(YT / "referent_north_kbs_diff.csv")
    D.t("A5", "북측 지칭 2018→2024 차이", nd[nd.구간 == "2018→2024"], fmt="{:.2f}")

    k = pd.read_csv(YT / "kim_title_kbs_yt.csv")
    D.t("A6", "김정은 호칭 (%)", k[(k.매체 == "KBS") & k.연도.isin([2018, 2024])][["연도", "버전", "직함", "동지", "맨이름", "N", "직함CI"]],
        "직함=앞뒤 2토큰 내 위원장·국무위원장·총비서·원수·총사령관·대원수, 동지=바로 뒤 '동지'.")

    h = pd.read_csv(KB / "hedge_types_year.csv")
    hc = pd.read_csv(KB / "hedge_types_corner.csv")
    hh = pd.concat([h, hc])[["구분", "지표", "2018", "2024", "CA_Z", "p"]]
    hh["p"] = hh.p.map(lambda x: f"{x:.2g}")
    D.t("A7", "헤지 표현 비율 (%) [95% CI]", hh,
        "리포트 나레이션, 인용 제외. 출처불명시형=알려졌·전해졌·~다고 합니다, 화자추정형=보입니다·풀이·전망 등. "
        "CA_Z·p는 4개 연도(2018·2020·2022·2024) Cochran-Armitage 경향 검정. 내부=[요즘 북한은]·[클로즈업 북한], 정세=[이슈&한반도].",
        fmt="{:.2f}")
    D.t("A8", "보도동사 거리두기 (북한 발화)", pd.DataFrame({
        "지표": ["거리두기 비율(%)", "경향 검정", "2018→2024 차이 95% CI"],
        "2018": ["17.1", "Cochran-Armitage p=0.150 (4개 연도)", "[−6.8, +25.4]"], "2024": ["27.1", "", ""]}),
        "2026-08 초판 보고 수치(원 코드 미보존, 재산출하지 않음). 인용원 구성: 북한 매체 10~13%, 전문가 38~49%, 관료 7~8%(4시점).")

    for name, tab in llm_dist("topic", ["KBS 2018", "KBS 2024"]):
        D.t("A9" + ("a" if name == "전체 문서" else "b"), f"LLM 소재 분류 (%) — {name}", tab)
    for name, tab in llm_dist("frame", ["KBS 2018", "KBS 2024"]):
        nb = name != "전체 문서"
        D.t("A10" + ("b" if nb else "a"), f"LLM 프레임 분류 (%) — {name}", tab,
            "잠정치(검수 전, 프레임 v0). " + llm_chi2(["KBS 2018", "KBS 2024"], "frame", nb) + f" ({name}, 2집단)")

    lo = pd.read_csv(KB / "logodds_adjacent.csv")
    s = lo[(lo.버전 == "전체") & (lo.구간 == "2018→2024") & lo.채택]
    D.t("A11", "변별 어휘 2018 vs 2024 (가중 로그오즈비, 상위 20)", pd.DataFrame({
        "순위": range(1, 21),
        "2018 변별 (z)": [f"{w} ({-z:.1f})" for w, z in zip(s.nsmallest(20, "z").단어, s.nsmallest(20, "z").z)],
        "2024 변별 (z)": [f"{w} ({z:.1f})" for w, z in zip(s.nlargest(20, "z").단어, s.nlargest(20, "z").z)]}),
        "기사 text(층1, 인용 포함), 명사(NNG·NNP), 4개 연도 합산 사전, 변별 쪽 연도 기사 5건 이상 등장어. 인접 시점은 엑셀 '로그오즈비_KBS인접시점'.")

    ty = topic_yearly()
    D.t("A12", "의제 — KBS 단독 LDA(K=8) 연도 평균 토픽 비중 (%)", ty[["토픽", 2018, 2024]].rename(columns={2018: "2018", 2024: "2024"}),
        "기사 572 + 2018 정상회담 특집 청크 18, 4개 연도 함께 학습. 분기별 값은 엑셀 '토픽_KBS분기추세'.")

    D.t("A13", "연관어 — 핵심어별 NPMI 상위 8", assoc_table(["KBS 2018", "KBS 2024"],
        ["김정은", "주민", "통일", "러시아", "탈북민", "미국", "한국"]),
        "문장 단위 명사 공기, 공기≥5, 연관어 문장빈도≥10. 인용 출처 표기 제거.")
    cs = pd.read_csv(YT / "cooc_context_sim.csv")
    cs = cs[(cs.코퍼스1 == "KBS 2018") & (cs.코퍼스2 == "KBS 2024")][["핵심어", "PPMI코사인", "상위연관어자카드", "공통연관어"]]
    D.t("A14", "연관어 맥락 유사도 KBS 2018 ↔ 2024", cs, "PPMI 코사인 1=같은 맥락. '대한민국'은 문장 30~42개.", fmt="{:.3f}")
    D.t("A15", "극성 (보조 지표)", polarity(["KBS 2018", "KBS 2024"]),
        "문장 P(긍)−P(부). 강한극성 |s|≥0.9 %, 강한부정 s≤−0.9 %. NSMC(영화 리뷰) 학습 모델 — 도메인 불일치.")

    # ===================== B. 유튜브 2024
    D.h("B. 탈북민·북한 유튜브 2024")
    sz = pd.read_csv(YT / "summary_by_channel.csv")
    sz = sz[sz.year == 2024]
    v = pd.read_parquet(YT / "videos.parquet")
    sz["장르"] = sz.channel.map(dict(zip(v.channel, v.genre)))
    b1 = sz[["장르", "channel", "편수", "글자", "토큰", "영상당토큰", "문장", "청크", "토큰비중%"]] \
        .rename(columns={"channel": "채널"}).sort_values(["장르", "토큰"], ascending=[True, False])
    for c in ["편수", "글자", "토큰", "영상당토큰", "문장", "청크"]:
        b1[c] = b1[c].map(lambda x: f"{int(x):,}")
    D.t("B1", "코퍼스 규모", b1, "토큰비중% = 2024 유튜브 전체 토큰 대비.")
    gr = pd.read_csv(YT / "genre_referent.csv")
    D.t("B2", "남측 지칭 유형 점유율 (%) — 장르", gr[gr.대상 != "KBS"],
        "'남조선제외' = 북한 시절 발화·선전 재현으로 판정된 '남조선' 제외(KBS 인용 제외의 대응).")
    D.t("B3", "남측 지칭 — 채널 (전체 버전)", pd.read_csv(YT / "referent_2024_by_channel.csv"))
    D.t("B4", "북측 지칭 유형 점유율 (%) — 장르", nr[nr.집단.str.startswith("유튜브")][
        ["집단", "버전", "국가명", "관계어", "북한식자칭", "적대어", "N", "국가명CI", "북한식자칭CI"]], fmt="{:.2f}")
    D.t("B5", "김정은 호칭 (%)", pd.concat([
        pd.read_csv(YT / "genre_kim_title.csv").rename(columns={"대상": "집단"}),
        k[k.매체.str.contains("└")].rename(columns={"매체": "집단"})[["집단", "직함", "동지", "맨이름", "N"]]]))
    lab6 = pd.read_csv(YT / "topic_labels_K6.csv")
    g6 = pd.read_csv(YT / "genre_topic_K6.csv").rename(columns={"Unnamed: 0": "토픽"})
    g6 = g6.merge(lab6[["토픽", "라벨", "UMass", "안정성"]], on="토픽")
    D.t("B6", "의제 — 합동 LDA K=6 장르별 평균 비중 (%)", g6[["토픽", "라벨", "상위어", "생활·경험형", "정세·체제형", "UMass", "안정성"]],
        "2024 KBS 147 기사 + 유튜브 1,240 청크 함께 학습(K=6 = 응집도·시드 안정성 최고). K=10은 엑셀.", fmt="{:.2f}")
    gl = pd.read_csv(YT / "genre_logodds_생활·경험형_vs_정세·체제형.csv")
    D.t("B7", "변별 어휘 생활·경험형 vs 정세·체제형 (상위 20)", pd.DataFrame({
        "순위": range(1, 21),
        "생활·경험형 (z)": [f"{w} ({z:.1f})" for w, z in zip(gl.nlargest(20, "z").word, gl.nlargest(20, "z").z)],
        "정세·체제형 (z)": [f"{w} ({-z:.1f})" for w, z in zip(gl.nsmallest(20, "z").word, gl.nsmallest(20, "z").z)]}))
    for name, tab in llm_dist("frame", ["유튜브 생활·경험형", "유튜브 정세·체제형"]):
        if name == "전체 문서":
            D.t("B8", "LLM 프레임 분류 (%)", tab, "잠정치(검수 전, 프레임 v0). 입력=제목+자막 앞 1,500자.")
    for name, tab in llm_dist("topic", ["유튜브 생활·경험형", "유튜브 정세·체제형"]):
        if name == "전체 문서":
            D.t("B9", "LLM 소재 분류 (%)", tab, "잠정치(검수 전).")
    D.t("B10", "연관어 — 핵심어별 NPMI 상위 8", assoc_table(["유튜브 생활·경험형", "유튜브 정세·체제형"],
        ["김정은", "대한민국", "한국", "주민", "통일", "러시아", "탈북민", "미국"]),
        "연관어 문장빈도 ≥ 문장수/1000 (약 35).")
    D.t("B11", "극성 (보조 지표)", polarity(["유튜브 생활·경험형", "유튜브 정세·체제형"]))

    # ===================== C. 2024 비교
    D.h("C. 2024 KBS vs 유튜브 비교")
    ref = pd.read_csv(YT / "referent_2024_yt_vs_kbs.csv")
    D.t("C1", "남측 지칭 유형 점유율 (%) — 매체", ref, "유튜브 채널군집CI=채널 리샘플 후 영상 리샘플.")
    D.t("C2", "남측 국가명 점유율 차이", pd.read_csv(YT / "genre_referent_diff.csv"),
        "전체 버전 대비 KBS 인용 포함, 남조선제외 버전 대비 KBS 인용 제외.")
    dq = pd.read_csv(YT / "kbs_defector_quotes.csv")
    D.t("C3", "KBS 방송 내 탈북민 인용 발화", dq,
        "KBS quote_spans 중 출처 표기에 탈북·귀순·출신·북한이탈 포함 스팬. 편집된 발화, 소표본.")
    D.t("C4", "북측 지칭 — KBS 2024 vs 유튜브", nr[nr.집단.isin(["KBS 2024", "KBS 2024 탈북민 인용", "유튜브 생활·경험형", "유튜브 정세·체제형"])]
        [["집단", "버전", "국가명", "관계어", "북한식자칭", "적대어", "N"]], fmt="{:.2f}")
    D.t("C5", "김정은 호칭 — KBS 2024 vs 유튜브 (%)", pd.concat([
        k[(k.매체 == "KBS") & (k.연도 == 2024)].assign(집단=lambda d: "KBS 2024 " + d.버전)[["집단", "직함", "동지", "맨이름", "N", "직함CI"]],
        pd.read_csv(YT / "genre_kim_title.csv").rename(columns={"대상": "집단"}).assign(집단=lambda d: "유튜브 " + d.집단)]))
    yl = pd.read_csv(YT / "logodds_2024_yt_vs_kbs.csv")
    yt_top = yl[(yl.z > 0) & (yl.n_ch >= 4)].nlargest(20, "z"); kb_top = yl[yl.z < 0].nsmallest(20, "z")
    D.t("C6", "변별 어휘 유튜브 vs KBS 2024 (상위 20)", pd.DataFrame({
        "순위": range(1, 21),
        "유튜브 변별 (z, 일관 채널 수)": [f"{w} ({z:.1f}, {n}/6)" for w, z, n in zip(yt_top.word, yt_top.z, yt_top.n_ch)],
        "KBS 변별 (z)": [f"{w} ({-z:.1f})" for w, z in zip(kb_top.word, kb_top.z)]}),
        "유튜브 측은 6개 채널 중 4개 이상에서 같은 방향 |z|>1.96인 어휘만. 정세·체제형 vs KBS는 엑셀.")
    D.t("C7", "의제 — 합동 LDA K=6 (%)", g6[["토픽", "라벨", "KBS", "생활·경험형", "정세·체제형"]])
    bal = pd.read_csv(YT / "topic_joint2024_balanced.csv")
    D.t("C8", "합동 LDA 강건성 — 문서 수 균형", bal,
        "유튜브를 KBS 147건에 맞춰 채널 비례 하위표본(시드 5회). KBS집중도=KBS 상위 2토픽 비중 합, 공유토픽=양 매체 ≥5%·비율≤2, "
        "매체분리도=Σ|차|/2(0=동일, 100=완전 분리).")
    proj = pd.read_csv(YT / "topic_2024_yt_vs_kbs.csv").rename(columns={"Unnamed: 0": "토픽"})
    D.t("C9", "KBS 모델(K=8) 투영 — 유튜브 토픽 비중 (%)", proj, "유튜브 명사의 KBS 어휘 적중률 67.2%.")
    for name, tab in llm_dist("frame", ["KBS 2024", "유튜브 생활·경험형", "유튜브 정세·체제형"]):
        if name == "전체 문서":
            D.t("C10", "LLM 프레임 분류 (%) — 2024", tab,
                "잠정치. " + llm_chi2(["KBS 2024", "유튜브 생활·경험형", "유튜브 정세·체제형"], "frame"))
    for name, tab in llm_dist("topic", ["KBS 2024", "유튜브 생활·경험형", "유튜브 정세·체제형"]):
        if name == "전체 문서":
            D.t("C11", "LLM 소재 분류 (%) — 2024", tab,
                "잠정치. " + llm_chi2(["KBS 2024", "유튜브 생활·경험형", "유튜브 정세·체제형"], "topic"))
    cs = pd.read_csv(YT / "cooc_context_sim.csv")
    cs = cs[cs.코퍼스1.eq("KBS 2024") | (cs.코퍼스1.str.startswith("유튜브") & cs.코퍼스2.str.startswith("유튜브"))]
    D.t("C12", "연관어 맥락 유사도 — 2024", cs[["핵심어", "코퍼스1", "코퍼스2", "PPMI코사인", "상위연관어자카드"]], fmt="{:.3f}")
    D.t("C13", "극성 — 2024", polarity(["KBS 2024", "유튜브 생활·경험형", "유튜브 정세·체제형"]))
    return D


def write_xlsx(D, path):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame([(n, note) for n, _, note in D.sheets], columns=["시트", "주"]).to_excel(xw, sheet_name="목차", index=False)
        for name, df, note in D.sheets:
            df.to_excel(xw, sheet_name=name, index=False, startrow=0)
            if note:
                xw.sheets[name].cell(len(df) + 3, 1, "주: " + note)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for c in ws[1]:
                c.font = Font(name="맑은 고딕", bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="2A78D6")
                c.alignment = Alignment(wrap_text=True, vertical="center")
            for ci, col in enumerate(ws.columns, 1):
                w = max(len(str(c.value or "")) for c in list(col)[:60])
                ws.column_dimensions[get_column_letter(ci)].width = min(max(8, w * 1.2), 70)


def main(pkg):
    D = build()
    (pkg / f"{STEM}.md").write_text("\n".join(D.lines) + "\n", encoding="utf-8")
    write_xlsx(D, pkg / f"{STEM}.xlsx")
    return pkg / f"{STEM}.md", len(D.sheets)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else R
    print(main(out))
