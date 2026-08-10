#!/usr/bin/env python3
"""config3 소스들의 영상목록 + 실제 업로드 날짜 매니페스트 생성.

소스별 폴더(BASE/<folder>)에 manifest.jsonl 저장.
날짜는 전 고유 영상ID를 단일 프로세스로 순차(배치) 조회(레이트리밋 회피).
"""
import json
import subprocess
from collections import Counter
from pathlib import Path

from config3 import SOURCES

BASE = Path(__file__).parent


def flat_list(url):
    proc = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s", url],
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
    src_recs = {}
    all_ids = set()
    for s in SOURCES:
        rows = flat_list(s["url"])
        print(f"[{s['folder']}] {s['group']}: {len(rows)}개", flush=True)
        recs = [{"folder": s["folder"], "group": s["group"], "index": i,
                 "id": v, "title": t, "upload_date": None} for i, v, t in rows]
        src_recs[s["folder"]] = recs
        all_ids.update(v for _, v, _ in rows)

    print(f"\n고유 영상 {len(all_ids)}개 날짜 조회", flush=True)
    dates = fetch_dates(sorted(all_ids))

    for folder, recs in src_recs.items():
        for r in recs:
            r["upload_date"] = dates.get(r["id"])
        d = BASE / folder
        d.mkdir(parents=True, exist_ok=True)
        with (d / "manifest.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = Counter((r.get("upload_date") or "????")[:4] for r in recs)
        print(f"[{folder}] 항목 {len(recs)} | 2018={c.get('2018',0)} "
              f"2024={c.get('2024',0)} | 미상={c.get('????',0)} | "
              f"범위 {min((r['upload_date'] for r in recs if r['upload_date']), default='-')}"
              f"~{max((r['upload_date'] for r in recs if r['upload_date']), default='-')}", flush=True)
    print("완료", flush=True)


if __name__ == "__main__":
    main()
