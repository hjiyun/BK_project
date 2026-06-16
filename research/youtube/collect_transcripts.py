#!/usr/bin/env python3
"""유미카 '유미의 토크박스' 재생목록 자막(스크립트) 수집기.

manifest.jsonl 에서 지정 연도(기본 2018, 2024)의 영상만 골라
한국어 자막을 받아 TXT(순수 텍스트)로 저장한다.
수동 한국어 자막을 우선하고, 없으면 자동생성 한국어 자막을 사용한다.

사용: python3 collect_transcripts.py [연도 ...]
예:   python3 collect_transcripts.py 2018 2024
"""
import json
import re
import sys
import time
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

BASE = Path(__file__).parent
MANIFEST = BASE / "manifest.jsonl"
LANGS = ["ko", "ko-KR"]


def safe_filename(text: str, maxlen: int = 80) -> str:
    text = re.sub(r"[\\/:*?\"<>|\n\r\t]", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:maxlen].strip()


def load_manifest(years):
    rows = []
    with MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ud = d.get("upload_date") or ""
            if ud[:4] in years:
                rows.append(d)
    return rows


def fetch_transcript_text(api, video_id):
    tlist = api.list(video_id)
    try:
        tr = tlist.find_manually_created_transcript(LANGS)
    except NoTranscriptFound:
        tr = tlist.find_generated_transcript(LANGS)
    fetched = tr.fetch()
    text = "\n".join(snip.text for snip in fetched if snip.text.strip())
    kind = "auto" if tr.is_generated else "manual"
    return text, tr.language_code, kind


def main():
    years = set(sys.argv[1:]) or {"2018", "2024"}
    rows = load_manifest(years)
    print(f"대상 연도 {sorted(years)} → 영상 {len(rows)}개")

    out_dir = BASE / ("yumi_talkbox_" + "_".join(sorted(years)))
    out_dir.mkdir(parents=True, exist_ok=True)

    api = YouTubeTranscriptApi()
    ok, failed = [], []

    for n, d in enumerate(rows, 1):
        vid, title, idx, ud = d["id"], d["title"], d["index"], d.get("upload_date")
        fname = f"{ud}_{idx:04d}_{vid}_{safe_filename(title)}.txt"
        fpath = out_dir / fname
        if fpath.exists():
            print(f"[{n:4d}/{len(rows)}] SKIP {vid}")
            ok.append(vid)
            continue
        try:
            text, lang, kind = fetch_transcript_text(api, vid)
            header = (
                f"# {title}\n# video_id: {vid}\n# upload_date: {ud}\n"
                f"# lang: {lang} ({kind})\n# url: https://youtu.be/{vid}\n\n"
            )
            fpath.write_text(header + text, encoding="utf-8")
            print(f"[{n:4d}/{len(rows)}] OK   {vid} [{lang}/{kind}] {len(text)}자")
            ok.append(vid)
        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as ex:
            print(f"[{n:4d}/{len(rows)}] FAIL {vid} ({type(ex).__name__})")
            failed.append((vid, title, type(ex).__name__))
        except Exception as ex:
            print(f"[{n:4d}/{len(rows)}] ERR  {vid} ({type(ex).__name__}: {ex})")
            failed.append((vid, title, f"{type(ex).__name__}: {ex}"))
        time.sleep(0.4)

    print("\n===== 요약 =====")
    print(f"성공: {len(ok)} / 실패: {len(failed)}")
    if failed:
        log = out_dir / "_failed.txt"
        log.write_text("\n".join(f"{v}\t{t}\t{r}" for v, t, r in failed), encoding="utf-8")
        print(f"실패 목록 저장: {log}")


if __name__ == "__main__":
    sys.exit(main())
