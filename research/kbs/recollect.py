"""
2018년 KBS 남북의 창 스크립트 재수집 (Step 2 only)
====================================================
기존 JSON에서 ncd 목록을 읽어 view.do 페이지만 다시 방문.
대기 로직을 개선하여 JS 렌더링 완료 후 추출.

[변경점]
1. WebDriverWait 조건을 'p.text가 비어있지 않을 때까지'로 강화
2. 추가 sleep 3초로 안전 마진 확보
3. 한 페이지가 실패하면 1회 재시도

[실행]
  python recollect_2018.py
"""

import time
import logging
import json
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
# 경로
# ============================================
OUTPUT_DIR = Path("/home/jiyoon/BK_project/research/data")
INPUT_JSON = OUTPUT_DIR / "kbs_namnam_2018_urls.json"
OUTPUT_CSV = OUTPUT_DIR / "kbs_namnam_2018_v2.csv"      # ← v2로 저장 (기존 보존)
LOG_PATH = OUTPUT_DIR / "recollect_2018.log"

BASE_DOMAIN = "https://news.kbs.co.kr"
ARTICLE_WAIT = 15        # 본문 렌더링 대기 (증가)
EXTRA_SLEEP = 3          # 안전 마진
REQUEST_DELAY = 1.5


logging.basicConfig(
    filename=LOG_PATH, level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s', encoding='utf-8'
)


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
    driver.set_page_load_timeout(30)
    return driver


def has_filled_text(driver) -> bool:
    """anchor-report 안의 p.text 중 비어있지 않은 것이 있는지."""
    elements = driver.find_elements(By.CSS_SELECTOR, "div.anchor-report p.text")
    return any(e.text.strip() for e in elements)


def extract_script(driver, ncd: str, retry: int = 1) -> dict:
    """
    view.do?ncd=XXX 에서 스크립트 추출.
    대기 로직 강화 + 1회 재시도.
    """
    url = f"{BASE_DOMAIN}/news/pc/view/view.do?ncd={ncd}"

    for attempt in range(retry + 1):
        driver.get(url)

        try:
            # 1) anchor-report DOM 출현 대기
            WebDriverWait(driver, ARTICLE_WAIT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.anchor-report"))
            )
            # 2) p.text가 채워질 때까지 대기 — 핵심
            WebDriverWait(driver, ARTICLE_WAIT).until(
                lambda d: has_filled_text(d)
            )
        except TimeoutException:
            if attempt < retry:
                logging.warning(f"[ncd={ncd}] 타임아웃 재시도")
                time.sleep(2)
                continue
            logging.warning(f"[ncd={ncd}] 최종 실패: 본문 텍스트 없음")
            return None

        time.sleep(EXTRA_SLEEP)  # 안전 마진

        # 제목
        try:
            title = driver.find_element(By.CSS_SELECTOR, "p.article-title").text.strip()
        except NoSuchElementException:
            title = ""

        # 스크립트 블록
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

        # sections 비어있음 → 재시도
        if attempt < retry:
            logging.warning(f"[ncd={ncd}] sections 비어있음, 재시도")
            time.sleep(2)
            continue

    return None


def main():
    # 입력 JSON
    if not INPUT_JSON.exists():
        print(f"[오류] {INPUT_JSON} 없음. 먼저 본 크롤러 Step 1 실행 필요.")
        return

    with open(INPUT_JSON, encoding="utf-8") as f:
        date_to_items = json.load(f)

    total_items = sum(len(v) for v in date_to_items.values())
    print(f"[재수집] 대상 ncd {total_items}개 (날짜 {len(date_to_items)}개)")

    driver = setup_driver()
    all_records = []
    no_script_count = 0
    count = 0

    try:
        for date, items in date_to_items.items():
            for item in items:
                count += 1
                ncd = item["ncd"]
                list_title = item["title"]
                print(f"  [{count:3d}/{total_items}] {date} ncd={ncd}", end=" ... ", flush=True)
                try:
                    data = extract_script(driver, ncd)
                    if data and data["sections"]:
                        for sec in data["sections"]:
                            all_records.append({
                                "date": date,
                                "ncd": ncd,
                                "list_title": list_title,
                                "article_title": data["title"],
                                "url": data["url"],
                                "section_order": sec["order"],
                                "section_type": sec["type"],
                                "text": sec["text"],
                                "char_len": len(sec["text"]),
                            })
                        print(f"{len(data['sections'])}개 코너")
                    else:
                        no_script_count += 1
                        print("스크립트 없음")
                except Exception as e:
                    logging.error(f"[ncd={ncd}] 예외: {e}", exc_info=True)
                    print("예외")

                # 중간 저장 (50건마다)
                if count % 50 == 0 and all_records:
                    pd.DataFrame(all_records).to_csv(
                        OUTPUT_CSV, index=False, encoding="utf-8-sig"
                    )
                time.sleep(REQUEST_DELAY)
    finally:
        driver.quit()

    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n{'='*50}")
        print(f"[완료] 총 {len(df)}개 코너 수집")
        print(f"       스크립트 없음: {no_script_count}건")
        print(f"       저장: {OUTPUT_CSV}")
        print(f"\n[유형별 분포]")
        print(df.groupby("section_type").size())
        print(f"\n[월별 분포]")
        df["month"] = df["date"].str[:6]
        print(df.groupby("month").size())
    else:
        print("\n[경고] 수집 데이터 없음")


if __name__ == "__main__":
    main()