#!/usr/bin/env python3
"""재생목록의 실제 업로드 날짜를 병렬로 수집해 매니페스트(JSONL) 생성.

flat-playlist 의 날짜는 추정값이라 사용 불가 → 영상별 실제 upload_date 조회.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLk7cU7uDUhRXKuqy_JBYw1sZtxhE7nqaH"
OUT = Path(__file__).parent / "manifest.jsonl"
WORKERS = 8


def get_ids():
    proc = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s", PLAYLIST_URL],
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


def fetch_date(vid):
    proc = subprocess.run(
        ["yt-dlp", "--no-warnings", "--print", "%(upload_date)s", f"https://youtu.be/{vid}"],
        capture_output=True, text=True,
    )
    d = proc.stdout.strip()
    return d if d and d != "NA" else None


def main():
    rows = get_ids()
    print(f"총 영상: {len(rows)}", flush=True)
    results = {}
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch_date, vid): (idx, vid, title) for idx, vid, title in rows}
        for fut in as_completed(futs):
            idx, vid, title = futs[fut]
            try:
                date = fut.result()
            except Exception:
                date = None
            results[idx] = {"index": idx, "id": vid, "title": title, "upload_date": date}
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(rows)} 날짜 조회 완료", flush=True)

    with OUT.open("w", encoding="utf-8") as f:
        for idx in sorted(results):
            f.write(json.dumps(results[idx], ensure_ascii=False) + "\n")

    # 연도별 집계
    from collections import Counter
    years = Counter(
        (r["upload_date"] or "????")[:4] for r in results.values()
    )
    print("\n연도별 영상 수:")
    for y in sorted(years):
        print(f"  {y}: {years[y]}")
    print(f"\n매니페스트 저장: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
