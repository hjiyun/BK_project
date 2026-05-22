"""
KBS '남북의 창' 2018년 누락 의심 회차 판별
==========================================
결방 의심 날짜에 대해 목록 페이지(sisa.do#YYYYMMDD)를 열어
영상 카드(a.box-content) 존재 여부로 결방 vs 크롤러 누락을 판별.

  - 카드 0개      → 결방(또는 그 날 콘텐츠 없음)
  - 카드 있음     → 크롤러 누락 (재수집 필요)

제외 코너([북한영상]/[통일로 미래로])도 표시만 하고 카운트엔 포함.
"""

import time
import re

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

BASE_DOMAIN = "https://news.kbs.co.kr"
LIST_URL = f"{BASE_DOMAIN}/news/pc/sisa/sisa.do?bcd=0031"
PAGE_LOAD_WAIT = 6

# 점검 대상: 알려진 결방일 외 누락 의심 5개
SUSPECT_DATES = ["20180428", "20180616", "20180811", "20181103", "20181208"]

# 대조군: 정상 수집된 인접 회차 (스크립트/사이트 정상 동작 확인용)
CONTROL_DATES = ["20180421", "20181215"]

EXCLUDE_RE = re.compile(r"\[\s*(북한\s*영상|통일로\s*미래로)\s*\]")


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


def collect_cards(driver, date_str: str) -> list:
    """그 날 목록 페이지의 모든 영상 카드 (ncd, title, excluded)."""
    url = f"{LIST_URL}#{date_str}"
    driver.get(url)
    time.sleep(2)
    driver.execute_script("location.reload();")
    time.sleep(PAGE_LOAD_WAIT)

    items = driver.find_elements(
        By.CSS_SELECTOR, "div.box-contents.has-wrap a.box-content"
    )
    if not items:
        items = driver.find_elements(By.CSS_SELECTOR, "a.box-content[href*='ncd=']")

    cards, seen = [], set()
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
        cards.append({
            "ncd": ncd,
            "title": title,
            "excluded": bool(EXCLUDE_RE.search(title)),
        })
    return cards


def verdict(cards: list) -> str:
    if not cards:
        return "결방 (영상 카드 0개)"
    kept = [c for c in cards if not c["excluded"]]
    if not kept:
        return f"방영했으나 전부 제외 코너 ({len(cards)}개) — 정상적으로 0건"
    return f"★크롤러 누락★ 수집 대상 {len(kept)}개 존재 → 재수집 필요"


def main():
    driver = setup_driver()
    try:
        for label, dates in [("대조군 (정상 회차)", CONTROL_DATES),
                             ("누락 의심", SUSPECT_DATES)]:
            print(f"\n{'='*70}\n[{label}]\n{'='*70}")
            for date in dates:
                cards = collect_cards(driver, date)
                md = f"{date[4:6]}/{date[6:8]}"
                print(f"\n● {date} ({md})  →  {verdict(cards)}")
                for c in cards:
                    mark = " [제외]" if c["excluded"] else ""
                    print(f"    ncd={c['ncd']}{mark}  {c['title']}")
                time.sleep(1.5)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
