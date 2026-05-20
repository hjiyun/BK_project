"""
KBS '남북의 창' 2018년 스크립트 크롤러 (최종)
================================================
2단계 수집:
  Step 1) sisa.do?bcd=0031#YYYYMMDD 에서 그 날의 ncd 목록 수집
  Step 2) 각 view.do?ncd=XXX 에서 앵커/리포트 스크립트 추출

[확정된 셀렉터]
  목록:  a.box-content [href에 ncd]
         a.box-content p.title  [필터링용 제목]
  기사:  p.article-title       [기사 제목]
         div.anchor-report      [스크립트 블록]
           span.badge           [앵커/리포트]
           p.text               [본문]

[제외 패턴]
  [북한영상], [통일로 미래로]

[저작권 주의]
  학술 연구(공정이용)용. 원문 그대로 외부 배포 금지.
"""

import time
import logging
import json
import re
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
# 설정
# ============================================
TARGET_YEAR = 2018
BCD = "0031"
BASE_DOMAIN = "https://news.kbs.co.kr"
LIST_URL = f"{BASE_DOMAIN}/news/pc/sisa/sisa.do?bcd={BCD}"
OUTPUT_DIR = Path("/home/jiyoon/BK_project/research/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)   # 폴더 없으면 자동 생성

OUTPUT_CSV = OUTPUT_DIR / "kbs_namnam_2018.csv"
LIST_JSON = OUTPUT_DIR / "kbs_namnam_2018_urls.json"
LOG_PATH = OUTPUT_DIR / "crawler.log"

PAGE_LOAD_WAIT = 5     # 목록 페이지 JS 렌더링 대기
ARTICLE_WAIT = 8       # 개별 기사 로딩 대기 (타임아웃)
REQUEST_DELAY = 1.5    # 요청 간 대기

# 제외할 코너
EXCLUDE_PATTERNS = ["[북한영상]", "[통일로 미래로]"]


logging.basicConfig(
    filename=LOG_PATH, level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s', encoding='utf-8'
)


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
    driver.set_page_load_timeout(30)
    return driver


# ============================================
# Step 1: 각 토요일의 ncd 목록 수집
# ============================================
def collect_ncds_for_date(driver, date_str: str) -> list:
    """sisa.do#YYYYMMDD 에서 그 날 모든 영상의 (ncd, title) 추출."""
    url = f"{LIST_URL}#{date_str}"
    driver.get(url)
    time.sleep(2)
    driver.execute_script("location.reload();")
    time.sleep(PAGE_LOAD_WAIT)

    # box-contents has-wrap 안의 a.box-content 들이 그 날 영상 목록
    # (상단 캐러셀에도 같은 카드가 보일 수 있으므로 box-contents 컨테이너 내부로 한정)
    try:
        items = driver.find_elements(
            By.CSS_SELECTOR,
            "div.box-contents.has-wrap a.box-content"
        )
        if not items:
            # 폴백: 클래스명이 다를 경우 광범위 검색
            items = driver.find_elements(By.CSS_SELECTOR, "a.box-content[href*='ncd=']")
    except Exception as e:
        logging.error(f"[목록 {date_str}] 셀렉터 오류: {e}")
        return []

    results = []
    seen_ncds = set()
    for it in items:
        href = it.get_attribute("href") or ""
        m = re.search(r"ncd=(\d+)", href)
        if not m:
            continue
        ncd = m.group(1)
        if ncd in seen_ncds:
            continue
        seen_ncds.add(ncd)

        # 제목 추출 (필터링용)
        try:
            title = it.find_element(By.CSS_SELECTOR, "p.title").text.strip()
        except NoSuchElementException:
            title = ""

        # 제외 패턴 필터
        if any(p in title for p in EXCLUDE_PATTERNS):
            logging.info(f"[목록 {date_str}] 제외: {title}")
            continue

        results.append({"ncd": ncd, "title": title, "list_url": url})

    logging.info(f"[목록 {date_str}] {len(results)}개 영상 (필터 후)")
    return results


# ============================================
# Step 2: 개별 view.do 페이지에서 스크립트 추출
# ============================================
def extract_script_from_view(driver, ncd: str) -> dict:
    """view.do?ncd=XXX 에서 앵커/리포트 스크립트 추출."""
    url = f"{BASE_DOMAIN}/news/pc/view/view.do?ncd={ncd}"
    driver.get(url)

    # anchor-report 또는 article-title 중 하나는 떠야 함
    try:
        WebDriverWait(driver, ARTICLE_WAIT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "p.article-title, div.anchor-report"))
        )
    except TimeoutException:
        logging.warning(f"[기사 {ncd}] 페이지 로딩 실패")
        return None

    time.sleep(0.5)

    # 제목
    try:
        title = driver.find_element(By.CSS_SELECTOR, "p.article-title").text.strip()
    except NoSuchElementException:
        title = ""

    # 스크립트 블록 (없을 수 있음 — 특집/영상 클립 등)
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
            # p.text가 없으면 div 전체 텍스트
            text = blk.text.strip()
            # badge 텍스트가 본문 앞에 붙어 있을 수 있어 제거
            if badge and text.startswith(badge):
                text = text[len(badge):].strip()

        if text:
            sections.append({"order": idx + 1, "type": badge, "text": text})

    return {"ncd": ncd, "url": url, "title": title, "sections": sections}


# ============================================
# 메인
# ============================================
def main():
    dates = get_saturdays(TARGET_YEAR)
    print(f"[설정] {TARGET_YEAR}년 토요일 {len(dates)}개")

    # ─── 테스트 모드: 검증 후 주석 처리 ───
    # dates = dates[:2]

    driver = setup_driver()

    try:
        # ===== Step 1: ncd 목록 =====
        print("\n[Step 1] 영상 ncd 목록 수집")
        date_to_items = {}
        for i, date in enumerate(dates, 1):
            print(f"  [{i:2d}/{len(dates)}] {date}", end=" ... ", flush=True)
            try:
                items = collect_ncds_for_date(driver, date)
                date_to_items[date] = items
                print(f"{len(items)}건")
            except Exception as e:
                logging.error(f"[목록 {date}] {e}", exc_info=True)
                date_to_items[date] = []
                print("실패")
            time.sleep(REQUEST_DELAY)

        with open(LIST_JSON, "w", encoding="utf-8") as f:
            json.dump(date_to_items, f, ensure_ascii=False, indent=2)
        total_items = sum(len(v) for v in date_to_items.values())
        print(f"\n  → 총 {total_items}개 영상 메타데이터 → {LIST_JSON}")

        # ===== Step 2: 개별 스크립트 =====
        print(f"\n[Step 2] 스크립트 추출")
        all_records = []
        no_script_count = 0
        count = 0
        for date, items in date_to_items.items():
            for item in items:
                count += 1
                ncd = item["ncd"]
                print(f"  [{count:3d}/{total_items}] {date} ncd={ncd}", end=" ... ", flush=True)
                try:
                    data = extract_script_from_view(driver, ncd)
                    if data and data["sections"]:
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
                        print(f"{len(data['sections'])}개 코너")
                    else:
                        no_script_count += 1
                        print("스크립트 없음")
                except Exception as e:
                    logging.error(f"[기사 ncd={ncd}] {e}", exc_info=True)
                    print("실패")

                # 중간 저장
                if count % 30 == 0 and all_records:
                    pd.DataFrame(all_records).to_csv(
                        OUTPUT_CSV, index=False, encoding="utf-8-sig"
                    )
                time.sleep(REQUEST_DELAY)

    finally:
        driver.quit()

    # 최종 저장
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"\n{'='*50}")
        print(f"[완료] 총 {len(df)}개 코너 텍스트 수집")
        print(f"       스크립트 없음: {no_script_count}건")
        print(f"       저장 위치: {OUTPUT_CSV}")
        print(f"\n[유형별 분포]")
        print(df.groupby("section_type").size())
        print(f"\n[월별 분포]")
        df["month"] = df["date"].str[:6]
        print(df.groupby("month").size())
    else:
        print("\n[경고] 수집된 데이터 없음. crawler.log 확인 필요.")


if __name__ == "__main__":
    main()