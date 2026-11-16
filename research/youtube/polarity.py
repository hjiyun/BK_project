#!/usr/bin/env python3
"""극성(감성) 보조 지표 — 7.13 회의 6-2: 평균이 아닌 '분산'과 '강한 극성 비율'을 매체 간 비교.

대상(문장 단위): KBS 2018·2024 층1 기사 text(인용 포함) / 유튜브 2024 장르 2유형(sentences.parquet)
모델: 주 monologg/koelectra-base-finetuned-nsmc (KoBERT 계열 한국어 인코더, NSMC 감성 미세조정)
      강건성 sangrimlee/bert-base-multilingual-cased-nsmc (mBERT)
      ※ 회의록의 'KoBERT'는 검증된 NSMC 미세조정 공개본이 없어 같은 계열의 KoELECTRA로 대체.
점수: P(긍정) − P(부정) ∈ [−1, 1]. 강한 극성 = |점수| ≥ STRONG.
지표: 문서(기사·영상) 단위 부트스트랩 95% CI — 분산, 강한 극성 비율, 강한 부정 비율, 평균(참고).
실행 (환경이 둘로 나뉨: Kiwi/pyarrow 환경 ↔ torch/transformers 환경):
  python3 polarity.py prepare                    # → master/_polarity_input.csv
  ~/.venv/bin/python polarity.py score <model>   # → master/_polarity_<tag>.csv  (GPU)
  python3 polarity.py report                     # → master/polarity_summary.csv
"""
import csv
import random
import re
import sys
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "master"
MODELS = {"koelectra": "monologg/koelectra-base-finetuned-nsmc",
          "mbert": "sangrimlee/bert-base-multilingual-cased-nsmc"}
STRONG = 0.9
INPUT = OUT / "_polarity_input.csv"


def prepare():
    import pandas as pd
    art = pd.read_parquet(BASE.parent / "kbs" / "master" / "articles.parquet")
    rows = []
    for y in [2018, 2024]:
        for r in art[art.year == y].itertuples():
            for s in re.split(r"(?<=[다요까])[.?!]\s*|\n+", str(r.text)):
                if len(s.strip()) >= 5:
                    rows.append((f"KBS {y}", f"ncd{r.ncd}", s.strip()))
    sen = pd.read_parquet(OUT / "sentences.parquet").merge(
        pd.read_parquet(OUT / "videos.parquet")[["video_id", "year", "genre"]])
    sen = sen[(sen.year == 2024) & (sen.char_len >= 5)]
    rows += [(f"유튜브 {r.genre}", r.video_id, r.sent) for r in sen.itertuples()]
    pd.DataFrame(rows, columns=["corpus", "doc", "sent"]).to_csv(INPUT, index=False)
    print(f"{len(rows):,} 문장 → {INPUT}")


def score(tag):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    name = MODELS[tag]
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).cuda().eval().half()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    pos = lab.get("positive", lab.get("label_1", 1)); neg = lab.get("negative", lab.get("label_0", 0))
    print(name, model.config.id2label, "pos", pos, "neg", neg)
    with open(INPUT, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            batch = [r["sent"] for r in rows[i:i + 256]]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
            p = torch.softmax(model(**enc).logits.float(), -1)
            out += (p[:, pos] - p[:, neg]).tolist()
    with open(OUT / f"_polarity_{tag}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["corpus", "doc", "score"])
        for r, s in zip(rows, out):
            w.writerow([r["corpus"], r["doc"], round(s, 4)])
    print(f"{len(out):,} 문장 채점 → _polarity_{tag}.csv")


def report():
    import numpy as np
    import pandas as pd
    random.seed(42)
    res = []
    for tag in MODELS:
        p = OUT / f"_polarity_{tag}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for c, g in df.groupby("corpus", sort=False):
            docs = [d.score.values for _, d in g.groupby("doc")]

            def stat(arrs):
                x = np.concatenate(arrs)
                return dict(분산=x.var(), 강한극성=(np.abs(x) >= STRONG).mean() * 100,
                            강한부정=(x <= -STRONG).mean() * 100, 평균=x.mean())
            est = stat(docs)
            boots = [stat([docs[int(random.random() * len(docs))] for _ in docs]) for _ in range(1000)]
            row = dict(모델=tag, 코퍼스=c, 문서=len(docs), 문장=len(g))
            for k, v in est.items():
                bs = sorted(b[k] for b in boots)
                row[k] = f"{v:.3f} [{bs[25]:.3f}, {bs[975]:.3f}]" if k in ("분산", "평균") \
                    else f"{v:.1f} [{bs[25]:.1f}, {bs[975]:.1f}]"
            res.append(row)
    out = pd.DataFrame(res)
    pd.set_option("display.width", 220)
    print(f"극성 점수 P(긍)−P(부) — 강한 극성 |s|≥{STRONG} (%) · 문서 부트스트랩 95% CI\n" + out.to_string(index=False))
    out.to_csv(OUT / "polarity_summary.csv", index=False)


if __name__ == "__main__":
    {"prepare": prepare, "report": report}.get(sys.argv[1], lambda: score(sys.argv[2]))()
