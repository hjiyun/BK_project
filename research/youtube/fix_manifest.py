#!/usr/bin/env python3
"""manifest.jsonl 에서 upload_date 가 비어있는(None) 항목을 단일 yt-dlp
프로세스로 순차 재조회해 채운다. 병렬 레이트리밋 회피용."""
import json
import subprocess
from pathlib import Path

BASE = Path(__file__).parent
MANIFEST = BASE / "manifest.jsonl"


def load():
    rows = []
    with MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save(rows):
    with MANIFEST.open("w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda x: x["index"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fetch_dates(ids):
    """단일 yt-dlp 프로세스로 여러 영상의 'id\tupload_date' 출력."""
    urls = [f"https://youtu.be/{v}" for v in ids]
    proc = subprocess.run(
        ["yt-dlp", "--no-warnings", "--ignore-errors",
         "--sleep-requests", "0.4", "--retries", "5",
         "--print", "%(id)s\t%(upload_date)s", *urls],
        capture_output=True, text=True,
    )
    out = {}
    for line in proc.stdout.splitlines():
        if "\t" in line:
            vid, d = line.split("\t", 1)
            d = d.strip()
            out[vid.strip()] = d if d and d != "NA" else None
    return out


def main():
    rows = load()
    by_id = {r["id"]: r for r in rows}
    missing = [r["id"] for r in rows if not r.get("upload_date")]
    print(f"재조회 대상(날짜 없음): {len(missing)}개", flush=True)

    CHUNK = 100
    for i in range(0, len(missing), CHUNK):
        batch = missing[i:i + CHUNK]
        dates = fetch_dates(batch)
        filled = 0
        for vid in batch:
            d = dates.get(vid)
            if d:
                by_id[vid]["upload_date"] = d
                filled += 1
        save(list(by_id.values()))  # 중간 저장
        print(f"  {i + len(batch)}/{len(missing)} 처리, 이번 배치 {filled}/{len(batch)} 채움", flush=True)

    from collections import Counter
    c = Counter((r.get("upload_date") or "????")[:4] for r in by_id.values())
    print("\n연도별 영상 수:")
    for y in sorted(c):
        print(f"  {y}: {c[y]}")
    still = sum(1 for r in by_id.values() if not r.get("upload_date"))
    print(f"여전히 날짜 없음: {still}")


if __name__ == "__main__":
    main()
