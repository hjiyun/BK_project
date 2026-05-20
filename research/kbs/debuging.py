"""
2018-11-10 개별 기사 페이지 디버깅
==================================
크롤러에서 4건 모두 "스크립트 없음"으로 떨어진 원인 추적.
JSON에서 11/10 ncd를 꺼내서 직접 분석한다.
"""

import json
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager


JSON_PATH = Path("/home/jiyoon/BK_project/research/data/kbs_namnam_2018_urls.json")
OUTPUT_DIR = Path("/home/jiyoon/BK_project/research/data")
TARGET_DATE = "20181110"


def main():
    # 1) JSON에서 11/10 ncd 목록 꺼내기
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get(TARGET_DATE, [])
    if not items:
        print(f"[오류] {TARGET_DATE} 데이터 없음")
        return

    print(f"[{TARGET_DATE}] {len(items)}개 영상 확인 예정")
    for it in items:
        print(f"  - ncd={it['ncd']}  {it['title']}")

    # 2) Driver 준비
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

    # 3) 첫 번째 영상에 대해 상세 진단
    try:
        first_ncd = items[0]["ncd"]
        first_title = items[0]["title"]
        url = f"https://news.kbs.co.kr/news/pc/view/view.do?ncd={first_ncd}"

        print(f"\n{'='*60}")
        print(f"진단 대상: {first_title}")
        print(f"URL: {url}")
        print(f"{'='*60}")

        driver.get(url)
        time.sleep(7)  # 충분히 대기

        # 페이지 제목
        print(f"\n[페이지 제목] {driver.title}")

        # 여러 셀렉터로 스크립트 영역 찾기
        print("\n[셀렉터별 검색 결과]")
        selectors_to_test = [
            "div.anchor-report",
            "div.anchor-text",
            "p.article-title",
            "p.text",
            "span.badge",
            "div.detail-cont",       # 추측
            "div.view-cont",         # 추측
            "div.cont-wrap",         # 추측
            "div[class*='anchor']",
            "div[class*='report']",
            "div[class*='script']",
            "div[class*='detail']",
            "div[class*='view']",
            "article",
            "main",
        ]
        for sel in selectors_to_test:
            elems = driver.find_elements(By.CSS_SELECTOR, sel)
            print(f"  '{sel}': {len(elems)}개")

        # 모든 div의 class 패턴 수집 (스크립트 영역 후보 탐색)
        print("\n[페이지의 주요 클래스명 — 상위 30개]")
        all_divs = driver.find_elements(By.TAG_NAME, "div")
        class_set = set()
        for d in all_divs[:300]:
            c = d.get_attribute("class") or ""
            if c and not c.startswith("modal") and not c.startswith("header"):
                class_set.add(c.split()[0] if c.split() else "")
        for c in sorted(class_set)[:30]:
            print(f"  .{c}")

        # body 텍스트 확인
        body = driver.find_element(By.TAG_NAME, "body").text
        print(f"\n[body 텍스트 길이] {len(body):,}자")
        print(f"\n[body 텍스트 미리보기 — 가운데 부분 500자]")
        mid = len(body) // 2
        print(body[max(0, mid-250):mid+250])

        # HTML 저장
        html_path = OUTPUT_DIR / f"debug_view_{first_ncd}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        png_path = OUTPUT_DIR / f"debug_view_{first_ncd}.png"
        driver.save_screenshot(str(png_path))
        print(f"\n[저장] {html_path.name}, {png_path.name}")

    finally:
        driver.quit()


if __name__ == "__main__":
    main()