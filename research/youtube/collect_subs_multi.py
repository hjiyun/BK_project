#!/usr/bin/env python3
"""4개 채널 매니페스트에서 지정 연도(기본 2018,2024) 영상 자막을 yt-dlp로 수집.

언어 우선순위: 한국어(ko*) > 영어(en*) > 영상에 존재하는 첫 자막.
박연미 채널은 영어 콘텐츠라 en 폴백 필요.
저장: BASE/<채널>/<재생목록>/<날짜>_<idx>_<id>_<제목>.txt
원본 vtt 는 _subs_vtt_multi 에 캐시(여러 재생목록 중복 영상 재다운로드 방지).
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from ysm_config import CHANNELS

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
    """vid 자막 vtt 다운로드(캐시 활용). (vtt_path, lang) 또는 (None,None)."""
    RAW.mkdir(exist_ok=True)
    # 이미 캐시된 vtt 있으면 사용
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
    # 우선순위: ko > en > 기타
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
    grand_ok = grand_fail = 0
    summary = []

    for ch in CHANNELS:
        man = BASE / ch / "manifest.jsonl"
        if not man.exists():
            print(f"[{ch}] manifest 없음, 건너뜀", flush=True)
            continue
        rows = []
        with man.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if (d.get("upload_date") or "")[:4] in years:
                        rows.append(d)
        print(f"\n===== [{ch}] 대상 {len(rows)}개 =====", flush=True)
        ok = fail = 0
        for n, d in enumerate(rows, 1):
            vid, title, idx, ud, pl = d["id"], d["title"], d["index"], d["upload_date"], d["playlist"]
            out_dir = BASE / ch / pl
            out_dir.mkdir(parents=True, exist_ok=True)
            fpath = out_dir / f"{ud}_{idx:04d}_{vid}_{safe_filename(title)}.txt"
            if fpath.exists() and fpath.stat().st_size > 0:
                print(f"[{ch}][{n}/{len(rows)}] SKIP {vid}", flush=True)
                ok += 1
                continue
            try:
                vtt, lang = download_sub(vid)
                if not vtt:
                    print(f"[{ch}][{n}/{len(rows)}] NOSUB {vid}", flush=True)
                    fail += 1
                    time.sleep(SLEEP)
                    continue
                text = vtt_to_text(vtt)
                header = (f"# {title}\n# channel: {ch}\n# playlist: {pl}\n"
                          f"# video_id: {vid}\n# upload_date: {ud}\n"
                          f"# sub_lang: {lang}\n# url: https://youtu.be/{vid}\n\n")
                fpath.write_text(header + text, encoding="utf-8")
                print(f"[{ch}][{n}/{len(rows)}] OK {vid} [{lang}] {len(text)}자", flush=True)
                ok += 1
            except Exception as ex:
                print(f"[{ch}][{n}/{len(rows)}] ERR {vid} ({ex})", flush=True)
                fail += 1
            time.sleep(SLEEP)
        summary.append((ch, len(rows), ok, fail))
        grand_ok += ok
        grand_fail += fail

    print("\n========== 전체 요약 ==========", flush=True)
    for ch, t, ok, fail in summary:
        print(f"  {ch}: 대상 {t}, 성공 {ok}, 실패 {fail}", flush=True)
    print(f"  합계: 성공 {grand_ok}, 실패 {grand_fail}", flush=True)


if __name__ == "__main__":
    main()
