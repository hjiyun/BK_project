#!/usr/bin/env python3
"""윤설미TV 지정 4개 재생목록의 영상 목록 + 실제 업로드 날짜 매니페스트 생성.

날짜는 단일 yt-dlp 프로세스로 순차 조회(병렬 레이트리밋 회피).
"""
import json
import subprocess
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / "윤설미TV"
MANIFEST = OUT_DIR / "manifest_ysm.jsonl"

PLAYLISTS = {
    "같이듣는_북한이야기": "PLfTbIb3O2ru08F2PniblnEGb5e_sQx3GK",
    "설미의_탈북스토리": "PLfTbIb3O2ru1P78_Hu-FUQsLwketXMB6j",
    "같이듣는_탈북스토리": "PLfTbIb3O2ru2Nx49hPTSEYUOagWqmQU75",
    "설미가말하는_북한이야기": "PLfTbIb3O2ru1rUHONFWFcAeLBYc8MdFx1",
}


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
    proc = subprocess.run(
        ["yt-dlp", "--no-warnings", "--ignore-errors",
         "--sleep-requests", "0.6", "--retries", "5",
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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for pname, plid in PLAYLISTS.items():
        rows = flat_list(plid)
        print(f"[{pname}] {len(rows)}개", flush=True)
        for idx, vid, title in rows:
            records.append({"playlist": pname, "plid": plid,
                            "index": idx, "id": vid, "title": title,
                            "upload_date": None})

    # 중복 영상ID(여러 재생목록에 같은 영상) 정리: 그대로 두되 날짜는 1회만 조회
    uniq_ids = sorted({r["id"] for r in records})
    print(f"고유 영상 {len(uniq_ids)}개 날짜 조회 중...", flush=True)
    dates = fetch_dates(uniq_ids)
    for r in records:
        r["upload_date"] = dates.get(r["id"])

    with MANIFEST.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    c = Counter((r.get("upload_date") or "????")[:4] for r in records)
    print("\n연도별(재생목록 항목 기준):")
    for y in sorted(c):
        print(f"  {y}: {c[y]}")
    miss = sum(1 for v in dates.values() if not v)
    print(f"날짜 없음(고유): {miss}")
    print(f"\n매니페스트: {MANIFEST}")


if __name__ == "__main__":
    main()
