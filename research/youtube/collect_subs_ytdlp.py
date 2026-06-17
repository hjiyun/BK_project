#!/usr/bin/env python3
"""yt-dlp 로 한국어 자막을 받아 순수 텍스트(TXT)로 저장.

youtube-transcript-api 가 IP 차단되어 yt-dlp 자막 다운로드로 대체.
manifest.jsonl 에서 지정 연도(기본 2024) 영상만 처리.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
MANIFEST = BASE / "manifest.jsonl"
RAW_DIR = BASE / "_subs_vtt"          # 원본 vtt 보관
SLEEP = 1.2                            # 요청 간 대기(차단 회피)


def safe_filename(text, maxlen=80):
    text = re.sub(r"[\\/:*?\"<>|\n\r\t]", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:maxlen].strip()


def vtt_to_text(vtt_path):
    """YouTube vtt(자동생성, word-timing 태그 포함)를 순수 텍스트로 변환."""
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
        # 인라인 타이밍/스타일 태그 제거: <00:00:01.000>, <c>, </c>
        s = re.sub(r"<[^>]+>", "", s)
        s = s.strip()
        if not s:
            continue
        # 직전과 동일하면(롤링 중복) 스킵
        if out and out[-1] == s:
            continue
        out.append(s)
    return "\n".join(out)


def download_sub(vid):
    """ko 자막 vtt 다운로드. 성공 시 (vtt_path, kind) 반환, 실패 시 (None, None)."""
    RAW_DIR.mkdir(exist_ok=True)
    subprocess.run(
        ["yt-dlp", "--skip-download", "--no-warnings",
         "--write-subs", "--write-auto-subs",
         "--sub-langs", "ko,ko-KR,ko-orig",
         "--sub-format", "vtt/best",
         "--sleep-requests", "1",
         "-o", str(RAW_DIR / "%(id)s.%(ext)s"),
         f"https://youtu.be/{vid}"],
        capture_output=True, text=True,
    )
    # 우선순위: 수동 ko > ko-KR > 자동 ko-orig > 기타 ko*
    cands = list(RAW_DIR.glob(f"{vid}.ko.vtt")) + \
            list(RAW_DIR.glob(f"{vid}.ko-KR.vtt")) + \
            list(RAW_DIR.glob(f"{vid}.ko-orig.vtt")) + \
            list(RAW_DIR.glob(f"{vid}.ko*.vtt"))
    seen = []
    for c in cands:
        if c not in seen:
            seen.append(c)
    if seen:
        return seen[0], seen[0].suffixes[0].lstrip(".")
    return None, None


def main():
    years = set(sys.argv[1:]) or {"2024"}
    rows = []
    with MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if (d.get("upload_date") or "")[:4] in years:
                rows.append(d)
    print(f"대상 연도 {sorted(years)} → 영상 {len(rows)}개", flush=True)

    out_dir = BASE / ("yumi_talkbox_" + "_".join(sorted(years)))
    out_dir.mkdir(parents=True, exist_ok=True)

    ok, failed = [], []
    for n, d in enumerate(rows, 1):
        vid, title, idx, ud = d["id"], d["title"], d["index"], d.get("upload_date")
        fname = f"{ud}_{idx:04d}_{vid}_{safe_filename(title)}.txt"
        fpath = out_dir / fname
        if fpath.exists() and fpath.stat().st_size > 0:
            print(f"[{n:4d}/{len(rows)}] SKIP {vid}", flush=True)
            ok.append(vid)
            continue
        try:
            vtt, kind = download_sub(vid)
            if not vtt:
                print(f"[{n:4d}/{len(rows)}] NOSUB {vid}", flush=True)
                failed.append((vid, title, "no_korean_sub"))
                time.sleep(SLEEP)
                continue
            text = vtt_to_text(vtt)
            header = (
                f"# {title}\n# video_id: {vid}\n# upload_date: {ud}\n"
                f"# sub: {kind}\n# url: https://youtu.be/{vid}\n\n"
            )
            fpath.write_text(header + text, encoding="utf-8")
            print(f"[{n:4d}/{len(rows)}] OK   {vid} [{kind}] {len(text)}자", flush=True)
            ok.append(vid)
        except Exception as ex:
            print(f"[{n:4d}/{len(rows)}] ERR  {vid} ({type(ex).__name__}: {ex})", flush=True)
            failed.append((vid, title, f"{type(ex).__name__}: {ex}"))
        time.sleep(SLEEP)

    print(f"\n===== 요약 =====\n성공: {len(ok)} / 실패: {len(failed)}", flush=True)
    if failed:
        (out_dir / "_failed.txt").write_text(
            "\n".join(f"{v}\t{t}\t{r}" for v, t, r in failed), encoding="utf-8")
        print(f"실패 목록: {out_dir / '_failed.txt'}", flush=True)


if __name__ == "__main__":
    main()
