#!/usr/bin/env python3
"""채널 전체 업로드(/videos)의 실제 업로드 날짜 매니페스트 생성.

재생목록에 담기지 않은 영상까지 포함해 연도 분포를 확인하기 위함.
저장: BASE/_channel_manifests/<handle>.jsonl
"""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "_channel_manifests"

HANDLES = ["YeonmiParkOfficial", "jooeuju-0815", "yoonseolmi"]


def flat_list(handle):
    proc = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s",
         f"https://www.youtube.com/@{handle}/videos"],
        capture_output=True, text=True,
    )
    rows = []
    for i, line in enumerate(proc.stdout.splitlines(), 1):
        if "\t" in line:
            vid, title = line.split("\t", 1)
        else:
            vid, title = line.strip(), ""
        if vid.strip():
            rows.append((i, vid.strip(), title))
    return rows


def fetch_dates(ids):
    out = {}
    CH = 200
    for i in range(0, len(ids), CH):
        batch = [f"https://youtu.be/{v}" for v in ids[i:i + CH]]
        proc = subprocess.run(
            ["yt-dlp", "--no-warnings", "--ignore-errors",
             "--sleep-requests", "0.5", "--retries", "5",
             "--print", "%(id)s\t%(upload_date)s", *batch],
            capture_output=True, text=True,
        )
        for line in proc.stdout.splitlines():
            if "\t" in line:
                vid, d = line.split("\t", 1)
                d = d.strip()
                out[vid.strip()] = d if d and d != "NA" else None
        print(f"    날짜 {min(i+CH, len(ids))}/{len(ids)}", flush=True)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    for h in HANDLES:
        rows = flat_list(h)
        print(f"[@{h}] 업로드 {len(rows)}개 — 날짜 조회 시작", flush=True)
        dates = fetch_dates([v for _, v, _ in rows])
        recs = [{"handle": h, "index": i, "id": v, "title": t,
                 "upload_date": dates.get(v)} for i, v, t in rows]
        with (OUT / f"{h}.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = Counter((r["upload_date"] or "????")[:4] for r in recs)
        print(f"[@{h}] 연도분포: " + " ".join(f"{y}:{c[y]}" for y in sorted(c)), flush=True)
    print("완료", flush=True)


if __name__ == "__main__":
    sys.exit(main())
