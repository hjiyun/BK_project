"""
누락 의심 9개 기사 view.do 페이지 직접 점검
============================================
목록에 카드는 있으나 recollect가 '본문 텍스트 없음'으로 실패한 ncd들.
view 페이지에 실제 anchor-report 스크립트가 존재하는지 확인하여
'진짜 누락(스크립트 있음)' vs '스크립트 없는 영상클립'을 구분.
"""

import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

BASE_DOMAIN = "https://news.kbs.co.kr"

# (date, ncd, 목록 제목)
TARGETS = [
    ("20180428", "3641605", "[특집] ‘평화, 새로운 시작’ 2018 남북정상회담"),
    ("20180616", "3665265", "[북미정상회담] 평화를 향한 첫걸음…‘공동합의문’ 서명"),
    ("20180811", "4022957", "[북한의 문화] 김정은 시대 영화·드라마"),
    ("20181103", "4065760", "[이슈&한반도] 남북 교류 ‘무더기 지연’"),
    ("20181103", "4065763", "[요즘 북한은] 역사 품은 北 소나무"),
    ("20181103", "4065764", "[클로즈업 북한] 변화하는 북한 TV"),
    ("20181208", "4090869", "[이슈&한반도] ‘연내 답방’ 놓고 고심"),
    ("20181208", "4090871", "[요즘 북한은] 북한의 사슴 농장"),
    ("20181208", "4090872", "[클로즈업 북한] 무상 의료는 ‘옛말’"),
]

ARTICLE_WAIT = 20      # recollect(15s)보다 더 길게
EXTRA_SLEEP = 4


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


def probe(driver, ncd: str) -> dict:
    url = f"{BASE_DOMAIN}/news/pc/view/view.do?ncd={ncd}"
    driver.get(url)
    result = {"ncd": ncd, "title": "", "anchor_report": 0,
              "filled": 0, "badges": [], "note": ""}

    # article-title 또는 anchor-report 둘 중 하나는 떠야 함
    try:
        WebDriverWait(driver, ARTICLE_WAIT).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "p.article-title, div.anchor-report"))
        )
    except TimeoutException:
        result["note"] = "페이지 로딩 자체 실패"
        return result

    time.sleep(EXTRA_SLEEP)

    try:
        result["title"] = driver.find_element(
            By.CSS_SELECTOR, "p.article-title").text.strip()
    except NoSuchElementException:
        result["note"] = "article-title 없음"

    blocks = driver.find_elements(By.CSS_SELECTOR, "div.anchor-report")
    result["anchor_report"] = len(blocks)
    for blk in blocks:
        try:
            badge = blk.find_element(By.CSS_SELECTOR, "span.badge").text.strip()
        except NoSuchElementException:
            badge = "?"
        try:
            txt = blk.find_element(By.CSS_SELECTOR, "p.text").text.strip()
        except NoSuchElementException:
            txt = ""
        if txt:
            result["filled"] += 1
        result["badges"].append(f"{badge}({len(txt)}자)")
    return result


def main():
    driver = setup_driver()
    print(f"{'ncd':>9} | {'AR블록':>6} | {'본문채움':>7} | 판정 / 제목")
    print("-" * 90)
    try:
        for date, ncd, list_title in TARGETS:
            r = probe(driver, ncd)
            if r["filled"] > 0:
                verdict = f"★진짜 누락★ 스크립트 {r['filled']}블록 → 재수집 가능"
            elif r["anchor_report"] > 0:
                verdict = "AR블록 있으나 본문 빈값 → 렌더링 더 대기 필요"
            else:
                verdict = "스크립트 없는 영상클립 → 스크립트 코퍼스엔 정상 부재"
            print(f"{ncd:>9} | {r['anchor_report']:>6} | {r['filled']:>7} | {verdict}")
            print(f"          제목: {r['title'] or list_title}")
            if r["badges"]:
                print(f"          badges: {', '.join(r['badges'])}")
            if r["note"]:
                print(f"          note: {r['note']}")
            print()
            time.sleep(1.5)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
