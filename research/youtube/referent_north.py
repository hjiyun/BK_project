#!/usr/bin/env python3
"""북측 지칭어 점유율 — KBS 4개 연도(시계열) + 유튜브 2024 장르별 + KBS 2024 탈북민 인용(매체 간).

어휘·규칙: referent_lexicon_north.yaml (KWIC 검수 근거 포함). 남측 분석(referent_analysis_yt.py)과 같은 틀:
Kiwi 토큰, 인접명사 배제, KBS 층1 인용 포함/제외, 유튜브 전체/북한식자칭 제외, 기사·영상 부트스트랩 95% CI.
민감도: '북쪽'을 관계어에 추가.
출력: master/referent_north.csv, _by_channel.csv, _kbs_diff.csv(인접 연도 차이), _forms.csv(표현별 빈도)
"""
import json
import random
import re
from pathlib import Path

import pandas as pd
from kiwipiepy import Kiwi

BASE = Path(__file__).parent
KBS = BASE.parent / "kbs" / "master"
kiwi = Kiwi(); kiwi.load_user_dictionary(str(BASE.parent / "kiwi_user_dict.tsv"))

LEX = {"국가명": ["북한"], "관계어": ["북측", "북녘", "이북"],
       "북한식자칭": ["북조선", "공화국", "조선민주주의인민공화국"], "적대어": ["북괴"]}
EXTRA = {"북쪽": "관계어"}
TYPES = list(LEX)
V2T = {w: t for t, ws in LEX.items() for w in ws} | EXTRA
NOUNISH = ("NNG", "NNP", "XSN", "XSV")
PRE_NOUN = ("NNG", "NNP", "SL", "SN", "SH")
DEFECTOR = re.compile(r"탈북|귀순|북한이탈|출신")


def hits(text):
    toks = kiwi.tokenize(str(text))
    out = []
    for i, tk in enumerate(toks):
        if tk.form not in V2T or not tk.tag.startswith("N"):
            continue
        if i + 1 < len(toks) and toks[i + 1].start == tk.start + tk.len and toks[i + 1].tag in NOUNISH:
            continue                                          # 북한군·북한산 등 복합어
        prev = toks[i - 1] if i else None
        if tk.form == "이북" and prev is not None and prev.tag in PRE_NOUN:
            continue                                          # NLL 이북·민통선 이북 (공간 표현)
        if tk.form == "공화국" and prev is not None and prev.tag == "NNP":
            continue                                          # 도미니카 공화국 등 외국 국호
        out.append((tk.form, tk.start))
    return out


def counts(forms, drop=(), extra=False):
    c = dict.fromkeys(TYPES, 0)
    for f in forms:
        if f in drop or (f in EXTRA and not extra):
            continue
        c[V2T[f]] += 1
    return c


def share(rows):
    agg = dict.fromkeys(TYPES, 0)
    for r in rows:
        for t in TYPES:
            agg[t] += r[t]
    n = sum(agg.values())
    return ({t: agg[t] / n * 100 for t in TYPES}, n) if n else (None, 0)


def ci(rows, typ, B=2000):
    vals = []
    for _ in range(B):
        p, n = share([rows[int(random.random() * len(rows))] for _ in rows])
        if n:
            vals.append(p[typ])
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def boot_vals(rows, typ, B=2000):
    vals = []
    for _ in range(B):
        p, n = share([rows[int(random.random() * len(rows))] for _ in rows])
        if n:
            vals.append(p[typ])
    return vals


def row(label, ver, rows):
    p, n = share(rows)
    if not n:
        return dict(집단=label, 버전=ver, N=0)
    out = dict(집단=label, 버전=ver, **{t: round(p[t], 2) for t in TYPES}, N=n, 문서=len(rows))
    for t in ["국가명", "관계어", "북한식자칭"]:
        lo, hi = ci(rows, t)
        out[f"{t}CI"] = f"[{lo:.2f}, {hi:.2f}]"
    return out


def main():
    random.seed(42)
    sec = pd.read_parquet(KBS / "sections.parquet")
    res, keep = [], {}
    for y, gy in sec[sec.layer == 1].groupby("year"):
        inc, exc, dq = [], [], []
        for _, g in gy.groupby("ncd"):
            fi, fe, fd = [], [], []
            for r in g.itertuples():
                qs = json.loads(r.quote_spans)
                spans = [(q["start"], q["end"]) for q in qs]
                dspans = [(q["start"], q["end"]) for q in qs if DEFECTOR.search(q.get("attribution", "") or "")]
                for f, st in hits(r.text_raw):
                    fi.append(f)
                    if not any(a <= st < b for a, b in spans):
                        fe.append(f)
                    if any(a <= st < b for a, b in dspans):
                        fd.append(f)
            inc.append(counts(fi)); exc.append(counts(fe))
            if fd:
                dq.append(counts(fd))
        res += [row(f"KBS {y}", "인용 포함", inc), row(f"KBS {y}", "인용 제외", exc)]
        keep[(y, "인용 포함")], keep[(y, "인용 제외")] = inc, exc
        if y == 2024:
            res.append(row("KBS 2024 탈북민 인용", "전체", dq))

    v = pd.read_parquet(BASE / "master" / "videos.parquet")
    v = v[v.year == 2024]
    forms = {r.video_id: [f for f, _ in hits(r.text_clean)] for r in v.itertuples()}
    by_ch = []
    for g in ["생활·경험형", "정세·체제형"]:
        ids = v.video_id[v.genre == g]
        res.append(row(f"유튜브 {g}", "전체", [counts(forms[i]) for i in ids]))
        res.append(row(f"유튜브 {g}", "북한식자칭 제외", [counts(forms[i], drop=tuple(LEX["북한식자칭"])) for i in ids]))
        res.append(row(f"유튜브 {g}", "민감도(+북쪽)", [counts(forms[i], extra=True) for i in ids]))
    for ch, g in v.groupby("channel"):
        p, n = share([counts(forms[i]) for i in g.video_id])
        if n:
            by_ch.append(dict(채널=ch, 장르=g.genre.iloc[0], **{t: round(p[t], 2) for t in TYPES}, N=n))

    diffs = []                                   # KBS 인접 연도 관계어·국가명 차이 (기사 부트스트랩)
    for ver in ["인용 포함", "인용 제외"]:
        for y1, y2 in [(2018, 2020), (2020, 2022), (2022, 2024), (2018, 2024)]:
            for t in ["관계어", "국가명"]:
                a, b = keep[(y1, ver)], keep[(y2, ver)]
                d = sorted(x - y for x, y in zip(boot_vals(b, t), boot_vals(a, t)))
                est = share(b)[0][t] - share(a)[0][t]
                lo, hi = d[int(.025 * len(d))], d[int(.975 * len(d))]
                diffs.append(dict(버전=ver, 구간=f"{y1}→{y2}", 유형=t, 차이pp=round(est, 2),
                                  CI95=f"[{lo:.2f}, {hi:.2f}]", 유의="예" if lo > 0 or hi < 0 else "아니오"))
    dd = pd.DataFrame(diffs)
    df = pd.DataFrame(res); ch = pd.DataFrame(by_ch).sort_values("N", ascending=False)
    pd.set_option("display.width", 250)
    print("북측 지칭 유형 점유율 (%)\n" + df.to_string(index=False))
    print("\n[KBS 인접 연도 차이]\n" + dd.to_string(index=False))
    print("\n[유튜브 채널별 — 전체]\n" + ch.to_string(index=False))
    # 표현별 원빈도 (해석용 참고)
    raw = {}
    for y, gy in sec[sec.layer == 1].groupby("year"):
        raw[f"KBS {y}"] = pd.Series([f for t in gy.text_raw for f, _ in hits(t)]).value_counts()
    for g in ["생활·경험형", "정세·체제형"]:
        raw[f"유튜브 {g}"] = pd.Series([f for i in v.video_id[v.genre == g] for f in forms[i]]).value_counts()
    rawdf = pd.DataFrame(raw).fillna(0).astype(int)
    print("\n[표현별 빈도 (인용 포함)]\n" + rawdf.to_string())
    df.to_csv(BASE / "master" / "referent_north.csv", index=False)
    ch.to_csv(BASE / "master" / "referent_north_by_channel.csv", index=False)
    dd.to_csv(BASE / "master" / "referent_north_kbs_diff.csv", index=False)
    rawdf.to_csv(BASE / "master" / "referent_north_forms.csv")


if __name__ == "__main__":
    main()
