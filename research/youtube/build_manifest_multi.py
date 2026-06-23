#!/usr/bin/env python3
"""4개 채널 지정 재생목록의 영상목록 + 실제 업로드 날짜 매니페스트 생성.

채널별 폴더(BASE/<채널>)에 manifest.jsonl 저장.
날짜는 모든 고유 영상ID를 단일 yt-dlp 프로세스로 순차 조회(레이트리밋 회피).
"""
import json
import subprocess
from pathlib import Path

from ysm_config import CHANNELS

BASE = Path(__file__).parent


def flat_list(plid):
    proc = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "%(id)s\t%(title)s",
         f"https://www.youtube.com/playlist?list={plid}"],
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
    urls = [f"https://youtu.be/{v}" for v in ids]
    out = {}
    CH = 200
    for i in range(0, len(urls), CH):
        batch = urls[i:i + CH]
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
        print(f"  날짜 {min(i+CH,len(urls))}/{len(urls)}", flush=True)
    return out


def main():
    # 1) 모든 채널의 flat 목록 수집
    all_records = {}  # channel -> list of records
    all_ids = set()
    for ch, pls in CHANNELS.items():
        recs = []
        for name, (plid, _said) in pls.items():
            rows = flat_list(plid)
            print(f"[{ch}] {name}: {len(rows)}", flush=True)
            for idx, vid, title in rows:
                recs.append({"channel": ch, "playlist": name, "plid": plid,
                             "index": idx, "id": vid, "title": title,
                             "upload_date": None})
                all_ids.add(vid)
        all_records[ch] = recs

    # 2) 고유 영상 날짜 일괄 조회
    print(f"\n고유 영상 {len(all_ids)}개 날짜 조회 시작", flush=True)
    dates = fetch_dates(sorted(all_ids))

    # 3) 채널별 매니페스트 저장
    from collections import Counter
    for ch, recs in all_records.items():
        for r in recs:
            r["upload_date"] = dates.get(r["id"])
        d = BASE / ch
        d.mkdir(parents=True, exist_ok=True)
        with (d / "manifest.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        c = Counter((r.get("upload_date") or "????")[:4] for r in recs)
        n18 = c.get("2018", 0)
        n24 = c.get("2024", 0)
        print(f"[{ch}] 항목 {len(recs)} | 2018={n18} 2024={n24} | 미상={c.get('????',0)}", flush=True)

    miss = sum(1 for v in dates.values() if not v)
    print(f"\n전체 날짜 미상(고유): {miss}", flush=True)
    print("완료", flush=True)


if __name__ == "__main__":
    main()
