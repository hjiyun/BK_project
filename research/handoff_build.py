#!/usr/bin/env python3
"""통일연구원 전달 패키지 빌더 — 결과·방법·재현 자료를 여러 형식으로 묶는다.

원칙: 원문 텍스트(KBS 스크립트·유튜브 자막)는 넣지 않는다(저작권·배포 제한). 집계 결과, 메타데이터(제목·URL·ID),
     어휘·코드·프롬프트만 포함. 해석은 수록하지 않는다(수치·방법·한계만).
출력: BK_project/handoff/통일연구원_전달자료_<날짜>/ (+ 같은 이름 .zip). handoff/ 는 .gitignore 대상.
실행: python3 handoff_build.py   (분석 산출물이 research/*/master 에 있어야 함)
"""
import base64
import datetime as dt
import io
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

R = Path(__file__).parent
YT, KB, LLM = R / "youtube" / "master", R / "kbs" / "master", R / "llm_classify" / "out"
TODAY = dt.date.today().strftime("%Y%m%d")
PKG = R.parent / "handoff" / f"통일연구원_전달자료_{TODAY}"
FIG = PKG / "02_그림"
APX = PKG / "부록_기술자료(읽지 않으셔도 됩니다)"          # 연구 재현용 — 비전공 독자는 열 필요 없음

# 차트 토큰 (dataviz 기준 팔레트, 라이트 표면 — 6슬롯 검증 통과, 대비 WARN → 값 직접 표기 + 표 제공)
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#b5b3aa"
from matplotlib import font_manager as _fm
for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    if Path(_f).exists():
        _fm.fontManager.addfont(_f)
FONT = "Noto Sans CJK JP"          # .ttc 컬렉션의 등록명 — 한글 글리프 포함
plt.rcParams.update({"font.family": FONT, "font.size": 10, "axes.edgecolor": BASE,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def num(s):
    """'93.6 [81.8, 100.0]' → (93.6, 81.8, 100.0)"""
    v = [float(x) for x in re.findall(r"-?\d+\.?\d*", str(s))]
    return (v + [np.nan, np.nan])[:3]


def style(ax, grid_axis="x"):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0)


RENAME = {'fig1_남측지칭_국가명': '그림1_남한을_부르는_말', 'fig2_김정은_직함부착': '그림2_김정은_직함', 'fig3_토픽K6_구성': '그림3_2024년_주제구성', 'fig4_토픽_K선택': '그림4_주제수_결정근거', 'fig5_KBS헤지_유형': '그림5_KBS_추정전언표현', 'fig6_극성_분산_강한부정': '그림6_긍정부정_점수', 'fig8_LLM_소재분포': '그림8_AI분류_소재', 'fig9_LLM_프레임분포': '그림9_AI분류_프레임'}


def save(fig, name):
    name = RENAME.get(name, name.replace("fig7_연관어네트워크_", "그림7_연관어연결망_"))
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


# ---------------------------------------------------------------- 데이터 표
def kbs_referent_years():
    """KBS 4개 연도 남측 국가명 점유율(인용 제외) — referent_analysis_yt 규칙."""
    sys.path.insert(0, str(R / "youtube"))
    import referent_analysis_yt as ra
    sec = pd.read_parquet(KB / "sections.parquet")
    rows = []
    for y, gy in sec[sec.layer == 1].groupby("year"):
        arts = []
        for _, g in gy.groupby("ncd"):
            fe = []
            for r in g.itertuples():
                spans = [(s["start"], s["end"]) for s in json.loads(r.quote_spans)]
                fe += [f for f, st in ra.hits(str(r.text_raw)) if not any(a <= st < b for a, b in spans)]
            arts.append(ra.to_counts(fe))
        p, n = ra.share(arts); lo, hi = ra.ci(ra.boot(arts))
        rows.append(dict(집단=f"KBS {y}", 국가명=round(p["국가명"], 1), lo=round(lo, 1), hi=round(hi, 1), N=n))
    return pd.DataFrame(rows)


def tables():
    T = {}
    T["코퍼스규모_채널"] = pd.read_csv(YT / "summary_by_channel.csv")
    ref = kbs_referent_years()
    g = pd.read_csv(YT / "genre_referent.csv")
    g = g[g.버전 == "남조선제외"]
    for r in g[g.대상 != "KBS"].itertuples():
        lo, hi = num(r.국가명CI)[:2]
        ref.loc[len(ref)] = dict(집단=f"유튜브 {r.대상}", 국가명=r.국가명, lo=lo, hi=hi, N=r.N)
    dq = pd.read_csv(YT / "kbs_defector_quotes.csv")
    d24 = dq[dq.연도 == 2024].iloc[0]
    lo, hi = num(d24.국가명CI)[:2]
    ref.loc[len(ref)] = dict(집단="KBS 2024 탈북민 인용", 국가명=d24.국가명, lo=lo, hi=hi, N=d24.남측N)
    T["지칭어_국가명"] = ref
    T["지칭어_유형전체"] = pd.read_csv(YT / "referent_2024_yt_vs_kbs.csv")
    T["지칭어_장르"] = pd.read_csv(YT / "genre_referent.csv")
    T["지칭어_장르차이"] = pd.read_csv(YT / "genre_referent_diff.csv")
    T["지칭어_채널"] = pd.read_csv(YT / "referent_2024_by_channel.csv")
    T["북측지칭"] = pd.read_csv(YT / "referent_north.csv")
    T["북측지칭_KBS인접연도"] = pd.read_csv(YT / "referent_north_kbs_diff.csv")
    T["북측지칭_채널"] = pd.read_csv(YT / "referent_north_by_channel.csv")
    T["북측지칭_표현빈도"] = pd.read_csv(YT / "referent_north_forms.csv").rename(columns={"Unnamed: 0": "표현"})
    T["김정은호칭"] = pd.read_csv(YT / "kim_title_kbs_yt.csv")
    T["김정은호칭_장르"] = pd.read_csv(YT / "genre_kim_title.csv")
    T["KBS탈북민인용"] = dq
    T["로그오즈비_KBS인접시점"] = pd.read_csv(KB / "logodds_adjacent.csv")
    T["로그오즈비_유튜브vsKBS"] = pd.read_csv(YT / "logodds_2024_yt_vs_kbs.csv")
    T["로그오즈비_정세형vsKBS"] = pd.read_csv(YT / "genre_logodds_정세·체제형_vs_KBS.csv")
    T["로그오즈비_경험형vs정세형"] = pd.read_csv(YT / "genre_logodds_생활·경험형_vs_정세·체제형.csv")
    T["토픽K6_라벨지표"] = pd.read_csv(YT / "topic_labels_K6.csv")
    T["토픽K10_라벨지표"] = pd.read_csv(YT / "topic_labels_K10.csv")
    T["토픽K6_장르비중"] = pd.read_csv(YT / "genre_topic_K6.csv").rename(columns={"Unnamed: 0": "토픽"})
    T["토픽K10_장르비중"] = pd.read_csv(YT / "genre_topic_K10.csv").rename(columns={"Unnamed: 0": "토픽"})
    T["토픽_K선택"] = pd.read_csv(YT / "topic_joint2024_select_k.csv")
    T["토픽_균형강건성"] = pd.read_csv(YT / "topic_joint2024_balanced.csv")
    T["토픽_KBS모델투영"] = pd.read_csv(YT / "topic_2024_yt_vs_kbs.csv").rename(columns={"Unnamed: 0": "토픽"})
    T["토픽_KBS분기추세"] = pd.read_csv(KB / "topic_quarter.csv")
    T["헤지_연도"] = pd.read_csv(KB / "hedge_types_year.csv")
    T["헤지_소재"] = pd.read_csv(KB / "hedge_types_corner.csv")
    T["연관어_상위"] = pd.read_csv(YT / "cooc_assoc.csv")
    T["연관어_맥락유사도"] = pd.read_csv(YT / "cooc_context_sim.csv")
    T["연관어_커뮤니티"] = pd.read_csv(YT / "cooc_communities.csv")
    T["극성"] = pd.read_csv(YT / "polarity_summary.csv")
    T["사전v3후보"] = pd.read_csv(YT / "dict_v3_candidates.csv")
    T["ASR변이형"] = pd.read_csv(YT / "asr_variants.csv")
    if (LLM / "labels.csv").exists():
        lab = pd.read_csv(LLM / "labels.csv")
        T["LLM분류_문서별"] = lab
        for col in ["topic", "frame"]:
            sub = lab.dropna(subset=[col]).assign(
                집단=lambda d: np.where(d.media == "KBS", "KBS " + d.year.astype(str), "유튜브 " + d.genre))
            T[f"LLM분류_{'소재' if col == 'topic' else '프레임'}분포"] = \
                (pd.crosstab(sub[col], sub.집단, normalize="columns") * 100).round(1).reset_index()
    return T


# ---------------------------------------------------------------- 그림
def fig_bar_ci(df, title, xlabel, name, note):
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(df) + 1.4))
    y = np.arange(len(df))[::-1]
    ax.barh(y, df.국가명 if "국가명" in df else df.val, height=0.5, color=CAT[0])
    vals = df.국가명 if "국가명" in df else df.val
    ax.errorbar(vals, y, xerr=[vals - df.lo, df.hi - vals], fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for yi, v, n in zip(y, vals, df.N):
        ax.text(min(v, 100) + 1.5 if np.isnan(df.hi.iloc[0]) else max(df.hi.max(), 0) + 2, yi,
                f"{v:.1f}%  (N={int(n):,})", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, df.집단); ax.set_xlim(0, 125); ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel(xlabel); style(ax)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    fig.text(0.01, -0.02, note, color=MUTED, fontsize=8)
    return save(fig, name)


def figures(T):
    out = []
    out.append(fig_bar_ci(T["지칭어_국가명"], "남한을 부르는 말 중 '한국·대한민국'의 비율",
                          "% (가로선 = 95% 신뢰구간)", "fig1_남측지칭_국가명",
                          "KBS: 앵커·리포터·기자의 말(인용 제외) · 유튜브: 북한 시절 말을 옮긴 '남조선' 제외 · 2018년 유튜브는 비교하지 않음"))
    k = T["김정은호칭"]; kk = k[(k.매체 == "KBS") & (k.버전 == "인용 제외")]
    rows = [dict(집단=f"KBS {r.연도}", val=r.직함, lo=num(r.직함CI)[0], hi=num(r.직함CI)[1], N=r.N)
            for r in kk.itertuples()]
    dq = T["KBS탈북민인용"]; d24 = dq[dq.연도 == 2024].iloc[0]
    rows.append(dict(집단="KBS 2024 탈북민 인용", val=d24.직함, lo=np.nan, hi=np.nan, N=d24.김정은N))
    for r in T["김정은호칭_장르"].itertuples():
        rows.append(dict(집단=f"유튜브 {r.대상}", val=r.직함, lo=num(r.직함CI)[0], hi=num(r.직함CI)[1], N=r.N))
    kd = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(kd) + 1.4))
    y = np.arange(len(kd))[::-1]
    ax.barh(y, kd.val, height=0.5, color=CAT[0])
    m = kd.lo.notna()
    ax.errorbar(kd.val[m], y[m.values], xerr=[kd.val[m] - kd.lo[m], kd.hi[m] - kd.val[m]],
                fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    for yi, v, n in zip(y, kd.val, kd.N):
        ax.text(102, yi, f"{v:.1f}%  (N={int(n):,})", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, kd.집단); ax.set_xlim(0, 125); ax.set_xticks(range(0, 101, 20)); style(ax)
    ax.set_xlabel("% (가로선 = 95% 신뢰구간; 탈북민 인용은 사례가 적어 생략)")
    ax.set_title("'김정은'에 직함(위원장·총비서 등)을 붙인 비율", loc="left", color=INK, fontsize=12, fontweight="bold")
    out.append(save(fig, "fig2_김정은_직함부착"))

    # 토픽 K=6 100% 누적
    lab = T["토픽K6_라벨지표"].set_index("토픽").라벨
    tp = T["토픽K6_장르비중"].set_index("토픽")[["KBS", "생활·경험형", "정세·체제형"]]
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    groups = ["KBS 2024", "유튜브 생활·경험형", "유튜브 정세·체제형"]
    left = np.zeros(3); yy = np.arange(3)[::-1]
    for i, t in enumerate(tp.index):
        v = tp.loc[t].values
        ax.barh(yy, v, left=left, height=0.55, color=CAT[i], edgecolor=SURF, linewidth=2, label=lab[t])
        for yi, l, w in zip(yy, left, v):
            if w >= 8:
                ax.text(l + w / 2, yi, f"{w:.0f}", ha="center", va="center", fontsize=9,
                        color="#0b0b0b" if CAT[i] == "#eda100" else "#ffffff")
        left += v
    ax.set_yticks(yy, groups); ax.set_xlim(0, 100); ax.set_xlabel("평균 주제 비중 (%)"); style(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncol=2, frameon=False, fontsize=8.5)
    ax.set_title("2024년 주제 구성 (KBS와 유튜브를 함께 6개 주제로 나눔)", loc="left", color=INK, fontsize=12, fontweight="bold")
    out.append(save(fig, "fig3_토픽K6_구성"))

    # K 선택 — 척도가 달라 두 패널
    ks = T["토픽_K선택"]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3))
    for ax, col, ttl in [(axs[0], "UMass원본", "주제 안 단어들의 일관성 (높을수록 좋음)"),
                         (axs[1], "안정성평균", "반복 계산 시 같은 주제가 나오는 정도 (1 = 항상 같음)")]:
        ax.plot(ks.K, ks[col], color=CAT[0], linewidth=2, marker="o", markersize=7)
        best = ks.loc[ks[col].idxmax()]
        ax.annotate(f"{int(best.K)}개: {best[col]:.3f}", (best.K, best[col]), xytext=(8, -14),
                    textcoords="offset points", color=INK, fontsize=9)
        ax.set_xticks(ks.K); ax.set_xlabel("주제 수"); ax.set_title(ttl, loc="left", color=INK, fontsize=10); style(ax, "y")
    fig.suptitle("주제 수를 6개로 정한 근거", x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(); out.append(save(fig, "fig4_토픽_K선택"))

    # 헤지 — 소재별 작은 다중
    h = T["헤지_소재"]; yrs = ["2018", "2020", "2022", "2024"]
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True)
    NAME = {"내부(검증곤란)": "북한 내부 소식 코너", "정세(관측가능)": "정세 분석 코너",
            "출처불명시형": "출처 없는 전언", "화자추정형": "화자의 추정"}
    for ax, grp in zip(axs, ["내부(검증곤란)", "정세(관측가능)"]):
        for i, typ in enumerate(["출처불명시형", "화자추정형"]):
            r = h[(h.구분 == grp) & (h.지표 == typ)].iloc[0]
            v = [num(r[y]) for y in yrs]
            x = [int(y) for y in yrs]
            ax.plot(x, [a[0] for a in v], color=CAT[i], linewidth=2, marker="o", markersize=7, label=NAME[typ])
            ax.fill_between(x, [a[1] for a in v], [a[2] for a in v], color=CAT[i], alpha=0.12, linewidth=0)
            ax.text(2024.3, v[-1][0], f"{NAME[typ]} {v[-1][0]:.1f}", color=INK2, fontsize=8, va="center")
            z = r.CA_Z
        ax.set_xticks([int(y) for y in yrs]); ax.set_xlim(2017.5, 2027.5); style(ax, "y")
        ax.set_title(NAME[grp], loc="left", color=INK, fontsize=10)
    axs[0].set_ylabel("리포터 문장 중 %")
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("KBS 추정·전언 표현의 비율 (인용 제외, 옅은 띠 = 95% 신뢰구간)", x=0.01, ha="left", color=INK, fontsize=12,
                 fontweight="bold")
    fig.tight_layout(); out.append(save(fig, "fig5_KBS헤지_유형"))

    # 극성 — 두 지표 두 패널, 모델 2계열
    pol = T["극성"]; corp = list(dict.fromkeys(pol.코퍼스))
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
    for ax, col, ttl in [(axs[0], "분산", "점수의 흩어짐"), (axs[1], "강한부정", "강한 부정 문장 비율 (%)")]:
        w = 0.36; x = np.arange(len(corp))
        for i, (m, mname) in enumerate([("koelectra", "모델 1 (KoELECTRA)"), ("mbert", "모델 2 (다국어 BERT)")]):
            v = [num(pol[(pol.모델 == m) & (pol.코퍼스 == c)][col].iloc[0]) for c in corp]
            ax.bar(x + (i - 0.5) * w, [a[0] for a in v], width=w, color=CAT[i], edgecolor=SURF, linewidth=2, label=mname)
            ax.errorbar(x + (i - 0.5) * w, [a[0] for a in v], yerr=[[a[0] - a[1] for a in v], [a[2] - a[0] for a in v]],
                        fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        ax.set_xticks(x, [c.replace("유튜브 ", "유튜브\n") for c in corp], fontsize=8)
        ax.set_title(ttl, loc="left", color=INK, fontsize=10); style(ax, "y")
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("긍정·부정 점수 (보조 자료, 세로선 = 95% 신뢰구간)", x=0.01, ha="left", color=INK, fontsize=12,
                 fontweight="bold")
    fig.tight_layout(); out.append(save(fig, "fig6_극성_분산_강한부정"))

    # 연관어 네트워크 4개
    for gexf in sorted(YT.glob("cooc_net_*.gexf")):
        G = nx.read_gexf(gexf)
        name = gexf.stem.replace("cooc_net_", "").replace("_", " ")
        comm = nx.get_node_attributes(G, "community")
        sizes = pd.Series(comm).value_counts()
        top = list(sizes.index[:6])
        col = {n: (CAT[top.index(c)] if c in top else OTHER) for n, c in comm.items()}
        freq = nx.get_node_attributes(G, "freq")
        pos = nx.kamada_kawai_layout(G, weight=None)          # 노드 겹침이 적은 배치
        fig, ax = plt.subplots(figsize=(9, 8))
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GRID, width=0.6)
        fv = np.array([float(freq[n]) for n in G])
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[col[n] for n in G],
                               node_size=40 + 600 * (fv / fv.max()) ** 0.6, edgecolors=SURF, linewidths=1.5)
        lab_nodes = sorted(G, key=lambda n: -float(freq[n]))[:30]
        nx.draw_networkx_labels(G, {n: pos[n] for n in lab_nodes}, {n: n for n in lab_nodes}, ax=ax,
                                font_family=FONT, font_size=8, font_color=INK,
                                bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.85))
        ax.set_axis_off()
        ax.set_title(f"연관어 연결망 — {name}\n(원 크기 = 많이 쓰인 정도, 선 = 같은 문장에 자주 함께 나옴, 색 = 함께 묶이는 단어 무리)", loc="left", color=INK, fontsize=11, fontweight="bold")
        out.append(save(fig, f"fig7_연관어네트워크_{name.replace(' ', '_')}"))

    if "LLM분류_소재분포" in T:
        for key, fname, ttl in [("LLM분류_소재분포", "fig8_LLM_소재분포", "AI 분류 — 소재 구성 (잠정, 사람 검수 전)"),
                                ("LLM분류_프레임분포", "fig9_LLM_프레임분포", "AI 분류 — 프레임 구성 (잠정, 사람 검수 전)")]:
            d = T[key].set_index(T[key].columns[0])
            cols = [c for c in ["KBS 2018", "KBS 2020", "KBS 2022", "KBS 2024", "유튜브 생활·경험형", "유튜브 정세·체제형"]
                    if c in d]
            d = d[cols]
            fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(cols) + 2.2))
            left = np.zeros(len(cols)); yy = np.arange(len(cols))[::-1]
            for i, cat in enumerate(d.index):
                v = d.loc[cat].values
                ax.barh(yy, v, left=left, height=0.55, color=CAT[i % 8] if i < 8 else OTHER,
                        edgecolor=SURF, linewidth=2, label=cat)
                for yi, l, w in zip(yy, left, v):
                    if w >= 8:
                        ax.text(l + w / 2, yi, f"{w:.0f}", ha="center", va="center", fontsize=8.5,
                                color="#0b0b0b" if CAT[i % 8] == "#eda100" else "#ffffff")
                left += v
            ax.set_yticks(yy, cols); ax.set_xlim(0, 100); ax.set_xlabel("%"); style(ax)
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=4, frameon=False, fontsize=8.5)
            ax.set_title(ttl, loc="left", color=INK, fontsize=12, fontweight="bold")
            out.append(save(fig, fname))
    return out


# ---------------------------------------------------------------- 문서
def md_to_docx(md_path, docx_path, base_dir=None):
    from docx import Document
    from docx.shared import Cm, Pt
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Noto Sans CJK KR"; st.font.size = Pt(10)
    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            rows = [l]; i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i]); i += 1
            cells = [[c.strip().replace("**", "") for c in r.strip().strip("|").split("|")] for r in rows]
            t = doc.add_table(rows=len(cells), cols=len(cells[0])); t.style = "Table Grid"
            for ri, r in enumerate(cells):
                for ci, c in enumerate(r[:len(cells[0])]):
                    t.cell(ri, ci).text = c
            continue
        txt = re.sub(r"\*\*(.+?)\*\*", r"\1", l).replace("`", "")
        img = re.match(r"^!\[(.*?)\]\((.+?)\)", l.strip())
        if img and base_dir is not None and (base_dir / img.group(2)).exists():
            doc.add_picture(str(base_dir / img.group(2)), width=Cm(16))
            cap = doc.add_paragraph(img.group(1)); cap.runs[0].font.size = Pt(8)
            i += 1
            continue
        if l.startswith("#"):
            lvl = len(l) - len(l.lstrip("#")); doc.add_heading(txt.lstrip("# "), level=min(lvl, 4))
        elif re.match(r"^\s*[-*] ", l):
            doc.add_paragraph(re.sub(r"^\s*[-*] ", "", txt), style="List Bullet")
        elif re.match(r"^\s*\d+\. ", l):
            doc.add_paragraph(re.sub(r"^\s*\d+\. ", "", txt), style="List Number")
        elif l.startswith(">"):
            doc.add_paragraph(txt.lstrip("> "), style="Intense Quote")
        elif l.strip() and l.strip() != "---":
            doc.add_paragraph(txt)
        i += 1
    doc.save(docx_path)


def write_xlsx(T, path):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        idx = pd.DataFrame([(k, len(v), ", ".join(map(str, v.columns[:8]))) for k, v in T.items()],
                           columns=["시트", "행수", "열(앞 8개)"])
        idx.to_excel(xw, "목차", index=False)
        for k, v in T.items():
            v.to_excel(xw, k[:31], index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="2A78D6")
                c.alignment = Alignment(wrap_text=True, vertical="center")
            for ci, col in enumerate(ws.columns, 1):
                w = max(len(str(c.value or "")) for c in list(col)[:200])
                ws.column_dimensions[get_column_letter(ci)].width = min(max(8, w * 1.3), 60)


CODEBOOK = {
    "지칭어_국가명": "남측 지칭 중 국가명(한국·대한민국) 비율. 국가명=%, lo/hi=95% 부트스트랩 CI(기사·영상 단위), N=남측 지칭 총 언급수",
    "지칭어_유형전체": "관계어(남측·남녘·남조선)/국가명/지역명(남한)/환유(청와대·용산) 점유율 %. 국가명CI=영상·기사 부트스트랩, 채널군집CI=채널 리샘플",
    "북측지칭": "북측 지칭 유형 점유율 %: 국가명(북한)/관계어(북측·북녘·이북)/북한식자칭(북조선·공화국·조선민주주의인민공화국)/적대어(북괴). CI=기사·영상 부트스트랩",
    "북측지칭_KBS인접연도": "KBS 인접 연도 점유율 차이(%p)와 기사 부트스트랩 95% CI, 유의=CI가 0을 포함하지 않음",
    "김정은호칭": "김정은 언급 범주: 직함(앞뒤 2토큰 내 위원장·국무위원장·총비서·원수)/동지/맨이름 %. KBS는 층1, 인용 포함·제외",
    "로그오즈비_유튜브vsKBS": "가중 로그오즈비 z(양수=유튜브 변별). per10k=1만 명사당 빈도, n_ch=같은 방향 |z|>1.96인 유튜브 채널 수(6개 중), z_<채널>=채널별 z",
    "로그오즈비_KBS인접시점": "KBS 인접 연도 가중 로그오즈비. z 양수=뒤 연도 변별, 버전=전체 명사/일반명사(NNG), 채택=변별 쪽 연도 기사 5건 이상",
    "토픽K6_라벨지표": "토픽별 라벨·UMass(토픽 상위15어)·안정성(시드 5개 매칭 코사인)·배타성(상위15어 비중복 비율)·relevance(λ=0.6)·FREX·대표 문서",
    "토픽_K선택": "K별 UMass(원본/균형표본 5회 평균), 안정성(시드 쌍 매칭 코사인 평균), 최저 토픽 안정성",
    "토픽_균형강건성": "유튜브를 KBS 문서 수(147)로 채널 비례 하위표본 후 재학습. KBS집중도=KBS 상위2토픽 비중합, 공유토픽=두 매체 모두 ≥5%·비율≤2, 매체분리도=Σ|차|/2",
    "헤지_소재": "KBS 리포트 나레이션(인용 제외) 문장 중 헤지 유형 % [95% CI]. 출처불명시형=알려졌·전해졌·~다고 합니다, 화자추정형=보입니다·풀이·전망 등. CA_Z=Cochran-Armitage 경향",
    "연관어_상위": "문장 단위 명사 공기 NPMI 상위 연관어. 공기≥5, 연관어 문장빈도≥max(10, 문장수/1000). KBS는 인용 출처 표기 제거",
    "연관어_맥락유사도": "핵심어별 코퍼스 쌍의 PPMI 벡터 코사인(1=같은 맥락)과 상위15 연관어 자카드",
    "극성": "문장 극성 P(긍)−P(부)∈[−1,1]. 분산, 강한극성=|s|≥0.9 %, 강한부정=s≤−0.9 %, 평균(참고). 모델 2종, 문서 부트스트랩 CI",
    "KBS탈북민인용": "KBS quote_spans 중 출처 표기에 탈북·귀순·출신·북한이탈 포함 스팬의 남측 지칭·김정은 직함 지표",
    "LLM분류_문서별": "문서별 소재(7범주)·bundle·프레임(6범주, 잠정 v0)·확신도·근거. model/prompt=재현용 모델 ID·프롬프트 해시",
}


def main():
    if PKG.exists():
        shutil.rmtree(PKG)        # 이 스크립트가 만든 산출물 폴더(원본 아님) — 매번 새로 생성
    A = {k: APX / v for k, v in dict(csv="표_CSV", net="연결망_Gephi", meth="방법·단어목록", lst="기사·영상_목록",
                                      rep="상세_보고서", md="문서_원본(md)").items()}
    for d in [FIG, *A.values()]:
        d.mkdir(parents=True)
    T = tables()
    for k, v in T.items():
        v.to_csv(A["csv"] / f"{k}.csv", index=False, encoding="utf-8-sig")
    write_xlsx(T, APX / "분석결과_전체표(기술용).xlsx")
    figs = figures(T)

    # ---- 읽을 자료 (패키지 맨 앞)
    md_to_docx(R / "통일연구원_결과보고서.md", PKG / "00_결과보고서.docx", base_dir=PKG)
    import results_digest                                  # 결과표 모음 (KBS 2018·2024 / 유튜브 2024 / 비교)
    dmd, nsheet = results_digest.main(PKG)
    md_to_docx(dmd, dmd.with_suffix(".docx"))
    shutil.move(str(dmd), A["md"] / dmd.name)
    shutil.copy(R / "통일연구원_결과보고서.md", A["md"] / "00_결과보고서.md")
    guide = APX.parent / "_안내.md"
    guide.write_text(GUIDE.format(date=TODAY, nfig=len(figs), ntab=nsheet), encoding="utf-8")
    md_to_docx(guide, PKG / "00_먼저_읽어주세요.docx")
    shutil.move(str(guide), A["md"] / "00_먼저_읽어주세요.md")

    # ---- 부록 (연구 재현용)
    for f in ["데이터수집_보고서.md", "유튜브_분석_요약.md", "토픽모델링_보고서.docx"]:
        if (R / f).exists():
            shutil.copy(R / f, A["rep"] / f)
    for f in ["데이터수집_보고서.md", "유튜브_분석_요약.md"]:
        md_to_docx(R / f, A["rep"] / f.replace(".md", ".docx"))
    for g in YT.glob("cooc_net_*.gexf"):
        shutil.copy(g, A["net"] / g.name)
    for src in [R / "kbs" / "referent_lexicon.yaml", R / "kbs" / "reporting_verbs.yaml",
                R / "youtube" / "referent_lexicon_yt.yaml", R / "youtube" / "referent_lexicon_north.yaml", R / "kiwi_user_dict.tsv",
                R / "kiwi_user_dict_v3_draft.tsv", R / "asr_normalize_v1.tsv", R / "llm_classify" / "prompt_v1.md"]:
        if src.exists():
            shutil.copy(src, A["meth"] / src.name)

    v = pd.read_parquet(R / "youtube" / "master" / "videos.parquet")
    v = v[["channel", "genre", "upload_date", "title", "url", "char_len"]].rename(columns={
        "channel": "채널", "genre": "유형", "upload_date": "업로드일", "title": "제목", "url": "주소", "char_len": "자막 글자 수"})
    v.to_excel(A["lst"] / "유튜브_영상목록.xlsx", index=False)
    art = pd.read_parquet(KB / "articles.parquet")[["ncd", "year", "date", "char_len"]]
    files = {2018: "kbs_namnam_2018_final.csv", 2020: "kbs_namnam_2020.csv", 2022: "kbs_namnam_2022.csv",
             2024: "kbs_namnam_2024.csv"}
    meta = pd.concat([pd.read_csv(R / "kbs" / f, usecols=["ncd", "list_title", "url"]) for f in files.values()]) \
        .groupby("ncd").first().reset_index()
    art.merge(meta, on="ncd", how="left")[["year", "date", "list_title", "url", "char_len"]].rename(columns={
        "year": "연도", "date": "방송일", "list_title": "코너·제목", "url": "주소", "char_len": "글자 수"}) \
        .to_excel(A["lst"] / "KBS_기사목록.xlsx", index=False)

    cb = ["# 코드북 — 부록 표(CSV·엑셀 시트) 설명\n", "| 표 | 설명 | 열 |", "|---|---|---|"]
    for k, df in T.items():
        cb.append(f"| {k} | {CODEBOOK.get(k, '')} | {', '.join(map(str, df.columns))} |")
    (A["meth"] / "코드북.md").write_text("\n".join(cb) + "\n", encoding="utf-8")
    (A["meth"] / "재현가이드.md").write_text(REPRO, encoding="utf-8")

    imgs = "".join(f'<figure><img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" '
                   f'alt="{p.stem}"><figcaption>{p.stem}</figcaption></figure>' for p in figs)
    (APX / "그림_한눈에보기.html").write_text(HTML.format(date=TODAY, imgs=imgs, tabs=""), encoding="utf-8")
    shutil.make_archive(str(PKG), "zip", PKG.parent, PKG.name)
    print(f"패키지: {PKG}  (+ .zip) · 결과표 {nsheet} · 그림 {len(figs)}")


GUIDE = """# 통일연구원 전달 자료 안내

N.K.STATS 연구그룹 · {date}

KBS 〈남북의 창〉(2018·2020·2022·2024년)과 탈북민·북한 유튜브(2024년, 6개 채널 592편)의 북한 관련 담론을 분석한 결과입니다.
**결과에 대한 해석은 담지 않았습니다.** 각 수치가 무엇을 어떻게 센 것인지만 설명했습니다.

## 읽는 순서

1. **00_결과보고서.docx** — 연구 설계, 자료, 방법, 결과, 한계를 정리한 본 보고서입니다. 맨 앞에 '용어 풀이와 표 읽는 법'이 있습니다.
2. **01_결과표 (Word 또는 Excel)** — 결과 표 {ntab}개를 KBS 2018·2024년 / 유튜브 2024년 / 2024년 비교 순서로 모았습니다. 내용은 Word와 Excel이 같습니다. 표를 다시 가공하실 때는 Excel 파일을 쓰십시오.
3. **02_그림** — 보고서에 쓴 그림을 포함해 {nfig}개입니다.

**'부록_기술자료' 폴더는 연구를 다시 계산해 볼 때 쓰는 자료**(계산 결과 원본, 단어 목록, 프로그램 설정 등)입니다. 읽지 않으셔도 됩니다.
다만 그 안의 '기사·영상_목록'에는 분석한 KBS 기사와 유튜브 영상의 제목과 주소가 있어, 원문을 직접 확인하실 때 쓸 수 있습니다.

## 함께 알아 두실 점

- 2018년 유튜브는 대상 채널 대부분이 2019년 이후 개설되어 비교하지 않았습니다. KBS와 유튜브 비교는 2024년만 했습니다.
- 유튜브 자막은 유튜브가 음성 인식으로 자동 생성한 것이라 받아쓰기 오류가 있을 수 있습니다.
- 'AI 분류'(기사·영상의 소재와 프레임) 결과는 사람이 맞는지 확인하기 전의 잠정치입니다. 프레임 범주도 아직 확정 전 초안입니다.
- 방송 대본과 자막 원문은 저작권 문제로 보내지 않습니다.
"""


README = """# 통일연구원 전달 자료 — N.K.STATS ({date})

KBS 〈남북의 창〉(2018·2020·2022·2024)과 탈북민·북한 유튜브(2024, 6개 채널 592편) 텍스트 분석의 **결과·방법·재현 자료**입니다.
해석은 수록하지 않았습니다(수치·방법·한계만). 원문 텍스트(방송 스크립트·자막)는 저작권상 포함하지 않으며, 표본 목록의 URL로 원 자료를 확인할 수 있습니다.

| 폴더/파일 | 내용 |
|---|---|
| **00_결과보고서.docx / .md** | **본 보고서** — 설계·자료·방법·결과·한계 (그림 포함, 해석 미수록) |
| **01_결과표_KBS2018·2024_유튜브2024 (.docx·.xlsx·.md)** | **결과표 모음** — A. KBS 2018·2024, B. 유튜브 2024, C. 2024 KBS–유튜브 비교 (표 41종, 해석 미수록) |
| 01_보고서/ | 데이터수집_보고서(md·docx) — 수집·정제·분석 전 과정, 유튜브_분석_요약(md·docx), 토픽모델링_보고서.docx |
| 02_분석결과_통합.xlsx | 모든 결과 표 {ntab}개 시트 (첫 시트 '목차') |
| 03_CSV/ | 같은 표의 CSV (UTF-8 BOM, 엑셀에서 바로 열림) |
| 04_그림/ | 그림 {nfig}개 (PNG 200dpi) |
| 05_네트워크_Gephi/ | 연관어 네트워크 GEXF 4개 (Gephi에서 열기; 노드 freq·community, 간선 NPMI weight·cooc) |
| 06_방법론_재현/ | 코드북, 재현가이드, 지칭어·보도동사 어휘집(yaml), Kiwi 사전 v2·v3 초안, ASR 정규화 초안, LLM 분류 프롬프트 |
| 07_표본목록/ | 유튜브 영상 목록(채널·장르·업로드일·제목·URL), KBS 기사 목록(날짜·코너 제목·URL) |
| 08_한눈에보기.html | 그림과 핵심 표를 한 파일로 (브라우저로 열기) |

## 꼭 함께 봐야 할 전제
- **2018 유튜브는 공백**으로 둔다(대상 채널 대부분이 2019년 이후 개설, 수집 3편). 매체 간 비교는 2024 단면만.
- 유튜브 자막은 100% 자동생성(ASR). 유튜브는 장르 2유형(생활·경험형: 유미카·이철은 / 정세·체제형: 강철환·주성하·배나·이소연)으로 층화.
- 토픽 라벨은 응집도·시드 안정성·배타성·relevance·FREX·대표 문서로 정함(토픽K6/K10_라벨지표 시트). 주 결과 K=6, K=10은 보조.
- 원 코드가 없어 재구현한 지표(김정은 호칭, 헤지)는 초판 보고서 수치와 수준이 다르다 — 보고서에 명시.
- 극성 모델은 NSMC(영화 리뷰)로 학습되어 도메인 불일치 — 분산·강한 부정만 두 모델에서 일관, 수기 검증 전 보조 지표.
- LLM 분류(소재·프레임): {llm}. 프레임 6범주는 잠정 v0, 내부 검수(66건, 코더 2인 블라인드) 진행 전.
- Kiwi 사전 v2는 일부 항목이 문맥에서 적용되지 않음(비핵화 55% 분해 등) — v3 초안은 미적용, 반영 시 토큰 기반 결과 재산출 필요.
"""

REPRO = """# 재현 가이드

환경: Python 3 + kiwipiepy 0.23, pandas, scikit-learn, networkx / 극성만 torch·transformers 환경(GPU).
코드: https://github.com/hjiyun/BK_project (research/). 원 데이터는 수집 스크립트로 재수집.

| 순서 | 스크립트 | 산출 |
|---|---|---|
| 1 | kbs/collect_kbs_year.py, youtube/build_manifest_multi.py·collect_subs_multi.py | 원자료 수집 |
| 2 | kbs/add_speaker.py → kbs/build_master.py | KBS 마스터(sections/articles) |
| 3 | youtube/build_yt_master.py | 유튜브 마스터(videos/sentences/chunks, genre) |
| 4 | kbs/referent_analysis.py, youtube/referent_analysis_yt.py | 남측 지칭어 |
| 5 | youtube/kim_title_yt.py, youtube/kbs_defector_quotes.py, youtube/referent_north.py | 김정은 호칭, KBS 탈북민 인용, 북측 지칭 |
| 6 | youtube/logodds_yt_vs_kbs.py, kbs/logodds_adjacent.py | 매체 간·KBS 인접 시점 로그오즈비 |
| 7 | kbs/topic_select_k.py·topic_model.py·topic_quarter.py | KBS 토픽(K=8) |
| 8 | youtube/topic_project_yt.py → topic_joint_2024.py 6 / 10 → topic_joint_balanced.py → topic_select_k_joint.py → topic_label_metrics.py | 투영·합동 토픽·강건성·K 선택·라벨 |
| 9 | youtube/genre_compare.py | 장르 층화 재산출 |
| 10 | kbs/hedge_types.py | 헤지 유형 |
| 11 | youtube/cooc_network.py | 연관어 네트워크 |
| 12 | youtube/polarity.py prepare → (~/.venv) score koelectra·mbert → report | 극성 |
| 13 | llm_classify/classify.py prepare → pilot → submit → collect → sample → report | LLM 소재·프레임 |
| 14 | youtube/dict_v3_candidates.py | 사전 v3 후보·ASR 변이 |
| 15 | handoff_build.py | 이 패키지 |

고정값: LDA random_state=42, 부트스트랩 seed=42, LLM 모델 ID·프롬프트 SHA-256(결과 파일 model·prompt 열).
"""

HTML = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>N.K.STATS 결과 한눈에 보기</title>
<style>body{{font-family:'Noto Sans CJK KR',sans-serif;background:#fcfcfb;color:#0b0b0b;max-width:1100px;margin:auto;padding:24px}}
figure{{margin:24px 0}}img{{max-width:100%;border:1px solid #e1e0d9}}figcaption{{color:#898781;font-size:12px}}
table{{border-collapse:collapse;font-size:12px;margin:8px 0 24px;display:block;overflow-x:auto}}
th{{background:#2a78d6;color:#fff;padding:4px 8px;text-align:left}}td{{border-bottom:1px solid #e1e0d9;padding:3px 8px}}
.note{{color:#52514e;font-size:13px}}</style></head><body>
<h1>N.K.STATS — 2024 KBS vs 유튜브 · KBS 2018–2024 결과 ({date})</h1>
<p class="note">해석 미수록. 원문 미포함. 2018 유튜브는 공백. 상세 표는 02_분석결과_통합.xlsx, 방법은 01_보고서.</p>
<h2>그림</h2>{imgs}<h2>핵심 표</h2>{tabs}</body></html>"""

if __name__ == "__main__":
    main()
