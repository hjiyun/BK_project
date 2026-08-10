#!/usr/bin/env python3
"""config5 매니페스트에서 지정 연도(기본 2018,2024) 자막을 yt-dlp로 수집.

언어 우선순위 ko*>en*>기타. 저장: BASE/<folder>/<group>/<날짜>_<idx>_<id>_<제목>.txt
원본 vtt 는 _subs_vtt_multi 캐시 공유.
"""
import json
import re
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

from config5 import SOURCES

BASE = Path(__file__).parent
RAW = BASE / "_subs_vtt_multi"
SLEEP = 1.0


def safe_filename(text, maxlen=70):
    text = re.sub(r"[\\/:*?\"<>|\n\r\t]", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:maxlen].strip() or "untitled"


def vtt_to_text(vtt_path):
    lines = vtt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out = []
    for ln in lines:
        s = ln.strip()
        if not s or s == "WEBVTT":
            continue
        if s.startswith(("Kind:", "Language:", "NOTE")):
            continue
        if "-->" in s:
            continue
        s = re.sub(r"<[^>]+>", "", s).strip()
        if not s or (out and out[-1] == s):
            continue
        out.append(s)
    return "\n".join(out)


def download_sub(vid):
    RAW.mkdir(exist_ok=True)
    cached = sorted(RAW.glob(f"{vid}.*.vtt"))
    if not cached:
        subprocess.run(
            ["yt-dlp", "--skip-download", "--no-warnings",
             "--write-subs", "--write-auto-subs",
             "--sub-langs", "ko.*,en.*,ko,en",
             "--sub-format", "vtt/best",
             "--sleep-requests", "1",
             "-o", str(RAW / "%(id)s.%(ext)s"),
             f"https://youtu.be/{vid}"],
            capture_output=True, text=True,
        )
        cached = sorted(RAW.glob(f"{vid}.*.vtt"))
    if not cached:
        return None, None

    def rank(p):
        lang = p.suffixes[-2].lstrip(".") if len(p.suffixes) >= 2 else ""
        if lang.startswith("ko"):
            return (0, lang)
        if lang.startswith("en"):
            return (1, lang)
        return (2, lang)
    best = sorted(cached, key=rank)[0]
    lang = best.suffixes[-2].lstrip(".") if len(best.suffixes) >= 2 else "?"
    return best, lang


def main():
    years = set(sys.argv[1:]) or {"2018", "2024"}
    # 폴더 순서 유지
    folders = list(OrderedDict((s["folder"], None) for s in SOURCES))
    grand_ok = grand_fail = 0
    summary = []

    for folder in folders:
        man = BASE / folder / "manifest.jsonl"
        if not man.exists():
            print(f"[{folder}] manifest 없음", flush=True)
            continue
        rows = []
        with man.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if (d.get("upload_date") or "")[:4] in years:
                        rows.append(d)
        print(f"\n===== [{folder}] 대상 {len(rows)}개 =====", flush=True)
        ok = fail = 0
        for n, d in enumerate(rows, 1):
            vid, title, idx, ud, group = d["id"], d["title"], d["index"], d["upload_date"], d["group"]
            out_dir = BASE / folder / group
            out_dir.mkdir(parents=True, exist_ok=True)
            fpath = out_dir / f"{ud}_{idx:04d}_{vid}_{safe_filename(title)}.txt"
            if fpath.exists() and fpath.stat().st_size > 0:
                print(f"[{folder}][{n}/{len(rows)}] SKIP {vid}", flush=True)
                ok += 1
                continue
            try:
                vtt, lang = download_sub(vid)
                if not vtt:
                    print(f"[{folder}][{n}/{len(rows)}] NOSUB {vid}", flush=True)
                    fail += 1
                    time.sleep(SLEEP)
                    continue
                text = vtt_to_text(vtt)
                header = (f"# {title}\n# channel: {folder}\n# group: {group}\n"
                          f"# video_id: {vid}\n# upload_date: {ud}\n"
                          f"# sub_lang: {lang}\n# url: https://youtu.be/{vid}\n\n")
                fpath.write_text(header + text, encoding="utf-8")
                print(f"[{folder}][{n}/{len(rows)}] OK {vid} [{lang}] {len(text)}자", flush=True)
                ok += 1
            except Exception as ex:
                print(f"[{folder}][{n}/{len(rows)}] ERR {vid} ({ex})", flush=True)
                fail += 1
            time.sleep(SLEEP)
        summary.append((folder, len(rows), ok, fail))
        grand_ok += ok
        grand_fail += fail

    print("\n========== 전체 요약 ==========", flush=True)
    for folder, t, ok, fail in summary:
        print(f"  {folder}: 대상 {t}, 성공 {ok}, 실패 {fail}", flush=True)
    print(f"  합계: 성공 {grand_ok}, 실패 {grand_fail}", flush=True)


if __name__ == "__main__":
    main()
