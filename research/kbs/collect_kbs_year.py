r"""
KBS '남북의 창' 연도별 스크립트 크롤러 (연도 파라미터화 버전)
================================================================
collect_2024.py 와 **동일한 규칙**을 유지하고 TARGET_YEAR 만 인자로 받는다.

사용: python3 collect_kbs_year.py 2020
      python3 collect_kbs_year.py 2020 2022      (여러 연도 순차 처리)

[동일하게 유지되는 규칙]
  - 프로그램: bcd=0031 (남북의 창), 해당 연도의 모든 토요일
  - 제외 코너 정규식: \[\s*(북한\s*영상|통일로\s*미래로)\s*\]
  - 셀렉터/대기시간/재시도/무대본 판정 로직 전부 동일
  - 출력 스키마: date, ncd, list_title, article_title, url,
                section_order, section_type, text, char_len

[저작권 주의] 학술 연구(공정이용)용. 원문 그대로 외부 배포 금지.
"""

import time
import logging
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager


# ============================================
# 설정 (연도 외 전부 collect_2024.py 와 동일)
# ============================================
BCD = "0031"
BASE_DOMAIN = "https://news.kbs.co.kr"
LIST_URL = f"{BASE_DOMAIN}/news/pc/sisa/sisa.do?bcd={BCD}"
OUTPUT_DIR = Path(__file__).parent          # research/kbs (기존 CSV 위치)

PAGE_LOAD_WAIT = 6
ARTICLE_WAIT = 20
SCRIPT_WAIT = 12
EXTRA_SLEEP = 3
REQUEST_DELAY = 1.5

EXCLUDE_RE = re.compile(r"\[\s*(북한\s*영상|통일로\s*미래로)\s*\]")


def get_saturdays(year: int) -> list:
    """해당 연도의 모든 토요일을 YYYYMMDD로 반환."""
    dates, d = [], datetime(year, 1, 1)
    while d.year == year:
        if d.weekday() == 5:
            dates.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return dates


def setup_driver() -> webdriver.Chrome:
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(40)
    return driver


# ============================================
# Step 1: 각 토요일의 ncd 목록 수집
# ============================================
def collect_ncds_for_date(driver, date_str: str) -> list:
    url = f"{LIST_URL}#{date_str}"
    driver.get(url)
    time.sleep(2)
    driver.execute_script("location.reload();")
    time.sleep(PAGE_LOAD_WAIT)

    try:
        items = driver.find_elements(
            By.CSS_SELECTOR, "div.box-contents.has-wrap a.box-content")
        if not items:
            items = driver.find_elements(
                By.CSS_SELECTOR, "a.box-content[href*='ncd=']")
    except Exception as e:
        logging.error(f"[목록 {date_str}] 셀렉터 오류: {e}")
        return []

    results, seen = [], set()
    for it in items:
        href = it.get_attribute("href") or ""
        m = re.search(r"ncd=(\d+)", href)
        if not m:
            continue
        ncd = m.group(1)
        if ncd in seen:
            continue
        seen.add(ncd)

        try:
            title = it.find_element(By.CSS_SELECTOR, "p.title").text.strip()
        except NoSuchElementException:
            title = ""

        if EXCLUDE_RE.search(title):
            logging.info(f"[목록 {date_str}] 제외: {title}")
            continue

        results.append({"ncd": ncd, "title": title, "list_url": url})

    logging.info(f"[목록 {date_str}] {len(results)}개 영상 (필터 후)")
    return results


# ============================================
# Step 2: 개별 view.do 페이지에서 스크립트 추출
# ============================================
def has_filled_anchor_text(driver) -> bool:
    els = driver.find_elements(By.CSS_SELECTOR, "div.anchor-report p.text")
    return any(e.text.strip() for e in els)


def get_article_title(driver) -> str:
    try:
        return driver.find_element(By.CSS_SELECTOR, "p.article-title").text.strip()
    except NoSuchElementException:
        return ""


def extract_script(driver, ncd: str, retry: int = 1) -> dict:
    url = f"{BASE_DOMAIN}/news/pc/view/view.do?ncd={ncd}"

    for attempt in range(retry + 1):
        driver.get(url)

        try:
            WebDriverWait(driver, ARTICLE_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "p.article-title")))
        except TimeoutException:
            if attempt < retry:
                logging.warning(f"[ncd={ncd}] 페이지 로딩 타임아웃, 재시도")
                time.sleep(2)
                continue
            logging.warning(f"[ncd={ncd}] 페이지 로딩 최종 실패")
            return None

        try:
            WebDriverWait(driver, SCRIPT_WAIT).until(has_filled_anchor_text)
        except TimeoutException:
            inner = ""
            conts = driver.find_elements(By.CSS_SELECTOR, "#cont_newstext")
            if conts:
                inner = driver.execute_script(
                    "return arguments[0].innerHTML;", conts[0]) or ""
            if len(inner.strip()) < 120:
                logging.info(f"[ncd={ncd}] 무대본(영상전용) — #cont_newstext 빈값")
                return {"ncd": ncd, "url": url,
                        "title": get_article_title(driver),
                        "sections": [], "no_script": True}
            if attempt < retry:
                logging.warning(f"[ncd={ncd}] 본문 렌더링 지연, 재시도")
                time.sleep(2)
                continue
            logging.warning(f"[ncd={ncd}] 본문 렌더링 최종 실패 (대본 미게시 처리)")
            return {"ncd": ncd, "url": url,
                    "title": get_article_title(driver),
                    "sections": [], "no_script": True}

        time.sleep(EXTRA_SLEEP)

        title = get_article_title(driver)
        blocks = driver.find_elements(By.CSS_SELECTOR, "div.anchor-report")
        sections = []
        for idx, blk in enumerate(blocks):
            try:
                badge = blk.find_element(By.CSS_SELECTOR, "span.badge").text.strip()
            except NoSuchElementException:
                badge = "UNKNOWN"
            try:
                text = blk.find_element(By.CSS_SELECTOR, "p.text").text.strip()
            except NoSuchElementException:
                text = ""
            if text:
                sections.append({"order": idx + 1, "type": badge, "text": text})

        if sections:
            return {"ncd": ncd, "url": url, "title": title, "sections": sections}

        if attempt < retry:
            logging.warning(f"[ncd={ncd}] sections 비어있음, 재시도")
            time.sleep(2)
            continue

    return None


# ============================================
# 연도 1개 처리
# ============================================
def run_year(year: int):
    out_csv = OUTPUT_DIR / f"kbs_namnam_{year}.csv"
    list_json = OUTPUT_DIR / f"kbs_namnam_{year}_urls.json"
    log_path = OUTPUT_DIR / f"crawler_{year}.log"

    # 연도별 로그 파일로 교체
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        filename=log_path, level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s', encoding='utf-8')

    dates = get_saturdays(year)
    print(f"\n{'='*55}\n[설정] {year}년 토요일 {len(dates)}개", flush=True)

    driver = setup_driver()
    all_records, no_script, page_fail = [], [], []
    try:
        print("\n[Step 1] 영상 ncd 목록 수집", flush=True)
        date_to_items = {}
        for i, date in enumerate(dates, 1):
            try:
                items = collect_ncds_for_date(driver, date)
            except Exception as e:
                logging.error(f"[목록 {date}] {e}", exc_info=True)
                items = []
            date_to_items[date] = items
            print(f"  [{i:2d}/{len(dates)}] {date} ... {len(items)}건", flush=True)
            time.sleep(REQUEST_DELAY)

        with open(list_json, "w", encoding="utf-8") as f:
            json.dump(date_to_items, f, ensure_ascii=False, indent=2)
        total_items = sum(len(v) for v in date_to_items.values())
        print(f"\n  → 총 {total_items}개 영상 메타 → {list_json}", flush=True)

        print(f"\n[Step 2] 스크립트 추출", flush=True)
        count = 0
        for date, items in date_to_items.items():
            for item in items:
                count += 1
                ncd = item["ncd"]
                tag = f"[{count:3d}/{total_items}] {date} ncd={ncd}"
                try:
                    data = extract_script(driver, ncd)
                except Exception as e:
                    logging.error(f"[기사 ncd={ncd}] {e}", exc_info=True)
                    page_fail.append((date, ncd, item["title"]))
                    print(f"  {tag} ... 예외", flush=True)
                    time.sleep(REQUEST_DELAY)
                    continue

                if data is None:
                    page_fail.append((date, ncd, item["title"]))
                    print(f"  {tag} ... 페이지실패", flush=True)
                elif data.get("no_script"):
                    no_script.append((date, ncd, item["title"]))
                    print(f"  {tag} ... 무대본(영상전용)", flush=True)
                else:
                    for sec in data["sections"]:
                        all_records.append({
                            "date": date,
                            "ncd": ncd,
                            "list_title": item["title"],
                            "article_title": data["title"],
                            "url": data["url"],
                            "section_order": sec["order"],
                            "section_type": sec["type"],
                            "text": sec["text"],
                            "char_len": len(sec["text"]),
                        })
                    print(f"  {tag} ... {len(data['sections'])}개 코너", flush=True)

                if count % 30 == 0 and all_records:
                    pd.DataFrame(all_records).to_csv(
                        out_csv, index=False, encoding="utf-8-sig")
                time.sleep(REQUEST_DELAY)
    finally:
        driver.quit()

    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n[완료 {year}] 코너 텍스트 {len(df)}행 / 회차 {df['date'].nunique()}개")
        print(f"       저장: {out_csv}")
        print(f"\n[유형별]\n{df.groupby('section_type').size()}")
        df["month"] = df["date"].str[:6]
        print(f"\n[월별]\n{df.groupby('month').size()}")
    else:
        print(f"\n[경고] {year} 수집 데이터 없음. {log_path} 확인 필요.")

    if no_script:
        print(f"\n[{year} 무대본 {len(no_script)}건]")
        for d, n, t in no_script:
            print(f"   {d} ncd={n}  {t}")
    if page_fail:
        print(f"\n[{year} 페이지 실패 {len(page_fail)}건] (점검 필요)")
        for d, n, t in page_fail:
            print(f"   {d} ncd={n}  {t}")


def main():
    years = [int(a) for a in sys.argv[1:]] or [2020, 2022]
    for y in years:
        run_year(y)
    print("\n전체 완료", flush=True)


if __name__ == "__main__":
    main()
