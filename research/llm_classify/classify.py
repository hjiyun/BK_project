#!/usr/bin/env python3
"""LLM 기반 소재·프레임 분류 파이프라인 (8.25 회의 6-3절 제안 이행).

대상: KBS 〈남북의 창〉 기사 572건(4개 연도) + 유튜브 2024 영상 592편.
입력: KBS = 코너 제목 + 앵커 블록 + 리포트 앞 INPUT_CHARS자 / 유튜브 = 제목 + 자막 앞 INPUT_CHARS자.
프롬프트: prompt_v1.md (파일 SHA-256 앞 12자를 결과에 기록). 출력은 JSON 스키마로 강제.
재현성: 모델 ID 고정(MODEL), 프롬프트 파일 버전 관리, 결과에 모델·프롬프트 해시 기록.
  ※ 회의록의 '온도 0'은 현행 모델이 sampling 파라미터를 받지 않아(400) 적용 불가 — 위 장치로 대체.
비용: Batches API(정가의 50%). refusal 은 배치에서 서버측 fallback 불가 → 기록 후 재제출 대상.

단계:
  python3 classify.py prepare          # requests.jsonl 생성 + 규모·비용 추정(API 호출 없음)
  python3 classify.py pilot            # 6건 동기 호출로 형식·품질 사전 확인
  python3 classify.py submit           # 배치 제출 (ANTHROPIC_API_KEY 또는 ant auth 필요, 비용 발생)
  python3 classify.py collect <batch>  # 결과 수집 → labels.csv
  python3 classify.py sample           # 검수용 표본 30건(매체·연도 층화) → review_sample.csv
  python3 classify.py report           # 분포 + 카이제곱/Fisher (7.13 회의 6-3)
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
RESEARCH = BASE.parent
PROMPT = BASE / "prompt_v1.md"
OUT = BASE / "out"
MODEL = "claude-opus-5-5"
EFFORT = "low"
INPUT_CHARS = 1500
PRICE_IN, PRICE_OUT = 4.00, 20.00          # $/MTok (claude-opus-5-5 정가), 배치는 50%
TOPICS = ["미사일·군사", "북한내부정치", "주민생활·경제", "대외관계", "남북관계", "인권", "기타"]
FRAMES = ["평화·협력", "안보위협", "체제비판", "인권·인도", "생활·인간관심", "정보전달"]
SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string", "enum": TOPICS},
        "is_bundle": {"type": "boolean"},
        "frame": {"type": "string", "enum": FRAMES},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "rationale": {"type": "string"},
    },
    "required": ["topic", "is_bundle", "frame", "confidence", "rationale"],
    "additionalProperties": False,
}
KBS_FILES = {2018: "kbs_namnam_2018_final.csv", 2020: "kbs_namnam_2020.csv",
             2022: "kbs_namnam_2022.csv", 2024: "kbs_namnam_2024.csv"}


def prompt_hash():
    return hashlib.sha256(PROMPT.read_bytes()).hexdigest()[:12]


def build_docs():
    sec = pd.read_parquet(RESEARCH / "kbs" / "master" / "sections.parquet")
    titles = pd.concat([pd.read_csv(RESEARCH / "kbs" / f, usecols=["ncd", "list_title"])
                        for f in KBS_FILES.values()]).groupby("ncd").list_title.first()
    docs = []
    for ncd, g in sec[sec.layer == 1].sort_values("section_order").groupby("ncd"):
        anchor = " ".join(g[g.speaker == "앵커"].text_raw.astype(str))
        report = " ".join(g[g.speaker != "앵커"].text_raw.astype(str))
        body = (anchor + "\n\n" + report)[:INPUT_CHARS]
        docs.append(dict(doc_id=f"kbs_{ncd}", media="KBS", year=int(g.year.iloc[0]), channel="KBS",
                         genre="KBS", text=f"[KBS 기사] 코너 제목: {titles.get(ncd, '')}\n\n{body}"))
    v = pd.read_parquet(RESEARCH / "youtube" / "master" / "videos.parquet")
    for r in v[v.year == 2024].itertuples():
        docs.append(dict(doc_id=f"yt_{r.video_id}", media="유튜브", year=2024, channel=r.channel,
                         genre=r.genre, text=f"[유튜브 영상] 제목: {r.title}\n\n{r.text_clean[:INPUT_CHARS]}"))
    return pd.DataFrame(docs)


def params(text):
    return {
        "model": MODEL,
        "max_tokens": 2048,                 # 짧은 JSON + 적응형 사고 여유
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
        "system": [{"type": "text", "text": PROMPT.read_text(encoding="utf-8"),
                    "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": text}],
    }


def prepare():
    OUT.mkdir(exist_ok=True)
    docs = build_docs()
    docs.drop(columns="text").to_csv(OUT / "docs_meta.csv", index=False)
    with open(OUT / "requests.jsonl", "w", encoding="utf-8") as f:
        for r in docs.itertuples():
            f.write(json.dumps({"custom_id": r.doc_id, "params": params(r.text)}, ensure_ascii=False) + "\n")
    # 대략 추정: 한국어 1토큰 ≈ 1.2자 (정확한 값은 submit 전 count_tokens 로 확인)
    sys_tok = len(PROMPT.read_text(encoding="utf-8")) / 1.2
    in_tok = docs.text.str.len().sum() / 1.2 + sys_tok * len(docs)
    out_tok = 400 * len(docs)
    cost = (in_tok * PRICE_IN + out_tok * PRICE_OUT) / 1e6 * 0.5
    print(f"문서 {len(docs)} (KBS {(docs.media == 'KBS').sum()} / 유튜브 {(docs.media == '유튜브').sum()}) · "
          f"모델 {MODEL} effort={EFFORT} · 프롬프트 {prompt_hash()}")
    print(f"추정 입력 {in_tok / 1e6:.2f}M / 출력 {out_tok / 1e6:.2f}M 토큰 → 배치 비용 약 ${cost:.2f} "
          f"(캐시 미적용 상한, 시스템 프롬프트 캐시 시 더 낮음)")
    print(f"→ {OUT / 'requests.jsonl'}")


def client():
    """ANTHROPIC_API_KEY 환경변수 우선, 없으면 BK_project/api 파일(.gitignore 대상)에서 읽음."""
    import os
    import anthropic
    keyfile = RESEARCH.parent / "api"
    if not os.environ.get("ANTHROPIC_API_KEY") and keyfile.exists():
        return anthropic.Anthropic(api_key=keyfile.read_text().strip())
    return anthropic.Anthropic()


def pilot(n=6):
    """배치 전 소규모 동기 호출로 출력 형식·분류 품질 확인 (매체·연도 층화)."""
    docs = build_docs()
    pick = pd.concat([g.head(1) for _, g in docs.groupby(["media", "year"])] +
                     [docs[docs.genre == "정세·체제형"].head(1)]).head(n)
    c = client()
    for r in pick.itertuples():
        msg = c.messages.create(**params(r.text))
        text = next((b.text for b in msg.content if b.type == "text"), "")
        u = msg.usage
        print(f"{r.doc_id} [{r.media} {r.year} {r.genre}] stop={msg.stop_reason} "
              f"in={u.input_tokens} cache_w={u.cache_creation_input_tokens} cache_r={u.cache_read_input_tokens} out={u.output_tokens}")
        print("   제목:", r.text.split(chr(10))[0][:80])
        print("   →", text)


def submit():
    from anthropic.types.messages.batch_create_params import Request
    reqs = [json.loads(l) for l in open(OUT / "requests.jsonl", encoding="utf-8")]
    batch = client().messages.batches.create(
        requests=[Request(custom_id=r["custom_id"], params=r["params"]) for r in reqs])
    (OUT / "batch_id.txt").write_text(f"{batch.id}\t{MODEL}\t{prompt_hash()}\n")
    print(f"배치 제출: {batch.id} ({len(reqs)}건) — 완료 후 collect {batch.id}")


def collect(batch_id):
    c = client()
    b = c.messages.batches.retrieve(batch_id)
    if b.processing_status != "ended":
        print(f"아직 처리 중: {b.processing_status} {b.request_counts}"); return
    rows, failed = [], []
    for res in c.messages.batches.results(batch_id):
        if res.result.type != "succeeded":
            failed.append((res.custom_id, res.result.type)); continue
        msg = res.result.message
        if msg.stop_reason == "refusal":
            failed.append((res.custom_id, "refusal")); continue
        text = next((blk.text for blk in msg.content if blk.type == "text"), "")
        rows.append(dict(doc_id=res.custom_id, **json.loads(text), model=MODEL, prompt=prompt_hash()))
    meta = pd.read_csv(OUT / "docs_meta.csv")
    lab = meta.merge(pd.DataFrame(rows), on="doc_id", how="left")
    lab.to_csv(OUT / "labels.csv", index=False)
    pd.DataFrame(failed, columns=["doc_id", "reason"]).to_csv(OUT / "failed.csv", index=False)
    print(f"성공 {len(rows)} / 실패 {len(failed)} → labels.csv, failed.csv")


def sample(n=30):
    lab = pd.read_csv(OUT / "labels.csv").dropna(subset=["topic"])
    random.seed(42)
    strata = lab.groupby(["media", "year"])
    per = max(1, n // strata.ngroups)
    s = pd.concat([g.sample(min(per, len(g)), random_state=42) for _, g in strata]).head(n)
    s = s.assign(검수_topic="", 검수_frame="", 검수_bundle="", 비고="")
    s.to_csv(OUT / "review_sample.csv", index=False)
    print(f"검수 표본 {len(s)}건 → review_sample.csv (허용 오분류율 10%, 8.25 회의 기준)")


def perm_chi2(x, g, B=10000):
    """집단 라벨을 섞어 χ² 통계량의 귀무분포를 만든 Monte Carlo p (기대빈도<5 셀 대응)."""
    import numpy as np
    from scipy.stats import chi2_contingency
    rng = np.random.default_rng(42)
    obs = chi2_contingency(pd.crosstab(x, g))[0]
    hits = sum(chi2_contingency(pd.crosstab(x, rng.permutation(g)))[0] >= obs for _ in range(B))
    return (hits + 1) / (B + 1)


def report():
    from scipy.stats import chi2_contingency, fisher_exact
    lab = pd.read_csv(OUT / "labels.csv").dropna(subset=["topic"])
    lab["집단"] = lab.apply(lambda r: f"KBS {r.year}" if r.media == "KBS" else f"유튜브 {r.genre}", axis=1)
    for col in ["topic", "frame"]:
        for name, sub in [("1부 시계열(KBS, bundle 제외)", lab[(lab.media == "KBS") & ~lab.is_bundle]),
                          ("2부 매체 간(2024)", lab[lab.year == 2024])]:
            tab = pd.crosstab(sub[col], sub["집단"])
            chi2, p, dof, exp = chi2_contingency(tab)
            note = "기대빈도<5 셀 있음 → 셀별 Fisher 병행" if (exp < 5).any() else ""
            print(f"\n[{col}] {name}: χ²={chi2:.1f} df={dof} p={p:.3g} {note}")
            print((tab / tab.sum() * 100).round(1).to_string())
            if (exp < 5).any() and tab.shape[1] > 2:     # 다집단: 순열 χ² (Fisher 일반화 대용)
                print(f"   순열 χ² p={perm_chi2(sub[col].values, sub['집단'].values):.4g} (10,000회)")
            if (exp < 5).any() and tab.shape[1] == 2:
                for lvl in tab.index:          # 범주 vs 나머지 2×2 Fisher
                    t = [[tab.loc[lvl].iloc[0], tab.iloc[:, 0].sum() - tab.loc[lvl].iloc[0]],
                         [tab.loc[lvl].iloc[1], tab.iloc[:, 1].sum() - tab.loc[lvl].iloc[1]]]
                    print(f"   Fisher {lvl}: p={fisher_exact(t)[1]:.3g}")
    kb = lab[lab.media == "KBS"]
    print(f"\nbundle 비중(KBS): {kb.is_bundle.mean() * 100:.1f}% · 연도별 "
          + str((kb.groupby('year').is_bundle.mean() * 100).round(1).to_dict()))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    {"prepare": prepare, "submit": submit, "sample": sample, "report": report, "pilot": pilot,
     "collect": lambda: collect(sys.argv[2])}[cmd]()
