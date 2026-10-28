#!/usr/bin/env python3
"""유튜브 자막 TXT → 마스터 테이블(videos / sentences / chunks) 구축.

- 원본 TXT는 수정하지 않음. text_raw 보존.
- 정제(text_clean): ASR 줄바꿈 병합, [음악]류 효과음 태그 제거, 공백 정리.
  간투사(어·음·아)는 지우지 않음 — Kiwi가 IC로 태깅하므로 명사 기반 분석에 영향 없음.
- 문장 분리: 자동자막은 구두점이 없어 Kiwi split_into_sents(종결어미 기반) 사용.
- chunks: KBS 기사 길이(약 3,500자)에 맞춰 문장 경계로 분할 — 토픽모델 등 길이 민감 분석용.
- token_len 은 KBS build_master.py 와 동일한 Kiwi + kiwi_user_dict.tsv 로 산출(정규화 분모 일치).
- 출력: master/videos.parquet, master/sentences.parquet, master/chunks.parquet (+ 요약 csv)
"""
import re
from pathlib import Path

import pandas as pd
from kiwipiepy import Kiwi

BASE = Path(__file__).parent
OUT = BASE / "master"
CHUNK_CHARS = 3500

# 폴더 → (채널 표시명, 기본 재생목록명, 장르). 재생목록은 헤더 group/playlist 또는 하위 폴더명 우선.
# 장르: 생활·경험형 = 탈북민 게스트 인터뷰 중심 / 정세·체제형 = 진행자의 북한 시사 해설 중심.
#   합동 LDA(K=6·10) 채널 토픽 구성의 2군집 분할과 일치 (보고서 7.5절).
CHANNELS = {
    "yumi_talkbox_2024": ("유미카", "유미의 토크박스", "생활·경험형"),
    "강철환tv": ("강철환tv", None, "정세·체제형"),
    "이철은_Lee-CherUn": ("이철은", None, "생활·경험형"),
    "주성하TV_joosungha": ("주성하TV", None, "정세·체제형"),
    "배나TV_bnatv1004": ("배나TV", None, "정세·체제형"),
    "이소연TV": ("이소연TV", None, "정세·체제형"),
    "한송이_songyi": ("한송이", None, "생활·경험형"),
}

HEADER_RE = re.compile(r"^#\s*(\w+):\s*(.*)$")
TAG_RE = re.compile(r"\[[^\]]{1,10}\]")          # [음악] [박수] [콧방귀] 등

kiwi = Kiwi()
_UDICT = BASE.parent / "kiwi_user_dict.tsv"
_n_udict = kiwi.load_user_dictionary(str(_UDICT))


def token_len(text):
    return sum(1 for _ in kiwi.tokenize(text)) if text.strip() else 0


def parse_file(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    meta, body = {"title": lines[0].lstrip("# ").strip()}, []
    for i, line in enumerate(lines[1:], 1):
        m = HEADER_RE.match(line)
        if m and not body:
            meta[m.group(1)] = m.group(2).strip()
        elif line.strip() or body:
            body = lines[i:]
            break
    return meta, "\n".join(body).strip()


def clean(text):
    text = TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_sents(sents, limit=CHUNK_CHARS):
    chunks, cur = [], []
    for s in sents:
        if cur and sum(len(x) for x in cur) + len(s) > limit:
            chunks.append(cur)
            cur = []
        cur.append(s)
    if cur:
        # 꼬리가 너무 짧으면(한도의 1/3 미만) 직전 청크에 붙임
        if chunks and sum(len(x) for x in cur) < limit / 3:
            chunks[-1].extend(cur)
        else:
            chunks.append(cur)
    return chunks


def main():
    OUT.mkdir(exist_ok=True)
    videos, sent_rows, chunk_rows = [], [], []
    for folder, (channel, default_pl, genre) in CHANNELS.items():
        files = sorted(p for p in (BASE / folder).rglob("*.txt") if not p.name.startswith("_"))
        for p in files:
            meta, raw = parse_file(p)
            vid = meta["video_id"]
            ud = meta.get("upload_date", "")
            playlist = meta.get("group") or meta.get("playlist") or default_pl \
                or p.parent.name.replace("_", " ")
            text = clean(raw)
            sents = [s.text.strip() for s in kiwi.split_into_sents(text) if s.text.strip()]
            for i, s in enumerate(sents):
                sent_rows.append({"video_id": vid, "sent_idx": i, "sent": s,
                                  "char_len": len(s)})
            for j, c in enumerate(chunk_sents(sents)):
                ct = " ".join(c)
                chunk_rows.append({"video_id": vid, "chunk_idx": j, "channel": channel,
                                   "genre": genre, "year": int(ud[:4]), "text": ct, "char_len": len(ct),
                                   "token_len": token_len(ct)})
            videos.append({
                "video_id": vid, "channel": channel, "genre": genre, "playlist": playlist,
                "year": int(ud[:4]), "upload_date": pd.to_datetime(ud, format="%Y%m%d"),
                "title": meta["title"], "url": meta.get("url", ""),
                "sub_lang": meta.get("sub_lang") or meta.get("sub", ""),
                "text_raw": raw, "text_clean": text,
                "char_len": len(text), "token_len": token_len(text),
                "n_sents": len(sents), "src_path": str(p.relative_to(BASE)),
            })
        print(f"  {channel}: {len(files)}편")

    videos = pd.DataFrame(videos)
    dup = videos.video_id.duplicated(keep=False)
    if dup.any():
        print(f"⚠ 중복 video_id {dup.sum()}행:", videos[dup].video_id.unique().tolist())
    videos.to_parquet(OUT / "videos.parquet", index=False)
    pd.DataFrame(sent_rows).to_parquet(OUT / "sentences.parquet", index=False)
    chunks = pd.DataFrame(chunk_rows)
    chunks.to_parquet(OUT / "chunks.parquet", index=False)

    summary = videos.groupby(["year", "channel"]).agg(
        편수=("video_id", "size"), 글자=("char_len", "sum"), 토큰=("token_len", "sum"),
        영상당토큰=("token_len", "mean"), 문장=("n_sents", "sum")).round(0)
    summary["청크"] = chunks.groupby(["year", "channel"]).size()
    summary["토큰비중%"] = (100 * summary.토큰 / summary.groupby("year").토큰.transform("sum")).round(1)
    summary.to_csv(OUT / "summary_by_channel.csv")
    print(f"videos: {len(videos)}편 / sentences: {len(sent_rows)} / chunks: {len(chunks)}"
          f" → {OUT}  (사용자사전 {_n_udict}항목)")
    print(summary.to_string())


if __name__ == "__main__":
    main()
