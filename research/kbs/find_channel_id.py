"""
KBS News 채널에서 '남북의 창' 검색이 2018·2024에 영상을 얼마나 잡아주는지 사전 점검
- 각 연도 1분기만 테스트 → quota 약 800 units
"""

import os
import time
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()
youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

KBS_NEWS_CHANNEL_ID = "UCcQTRi69dsVYHN3exePtZ1A"

# 4개 검색어 × 2개 분기 테스트
QUERIES = ["남북의 창", "남북의창", "남북의 짤", "남북의 썰"]
TEST_PERIODS = [
    ("2018-01-01T00:00:00Z", "2018-04-01T00:00:00Z", "2018 Q1"),
    ("2024-01-01T00:00:00Z", "2024-04-01T00:00:00Z", "2024 Q1"),
]


def test_search(query, after, before, label):
    """1페이지(50개)만 가져와서 빠르게 확인. 100 units 소비."""
    try:
        res = youtube.search().list(
            part="snippet",
            channelId=KBS_NEWS_CHANNEL_ID,
            q=query,
            type="video",
            maxResults=50,
            publishedAfter=after,
            publishedBefore=before,
            order="date"
        ).execute()
    except Exception as e:
        print(f"  에러: {e}")
        return 0
    
    items = res.get("items", [])
    total_estimate = res.get("pageInfo", {}).get("totalResults", 0)
    has_next = "nextPageToken" in res
    
    print(f"\n[{label}] q='{query}'")
    print(f"  1페이지 결과: {len(items)}개")
    print(f"  전체 추정(부정확): {total_estimate}개")
    print(f"  추가 페이지 있음: {has_next}")
    
    # 샘플 3개 제목
    for i, item in enumerate(items[:3]):
        title = item["snippet"]["title"]
        pub = item["snippet"]["publishedAt"][:10]
        print(f"    {i+1}. [{pub}] {title}")
    
    return len(items)


print("=" * 70)
print("KBS News 채널에서 '남북의 창' 키워드 사전 점검")
print("=" * 70)
print(f"채널 ID: {KBS_NEWS_CHANNEL_ID}")
print(f"예상 quota 소비: 약 {len(QUERIES) * len(TEST_PERIODS) * 100} units")
print()

total_found = {}
for query in QUERIES:
    for after, before, label in TEST_PERIODS:
        n = test_search(query, after, before, label)
        total_found[(query, label)] = n
        time.sleep(0.3)

print("\n" + "=" * 70)
print("요약")
print("=" * 70)
for (q, lbl), n in total_found.items():
    print(f"  {lbl:10} | '{q:10}' | {n}개")

print("\n해석 가이드:")
print("- 2018 Q1에서 0개 → 2018년 이전 데이터 부재 가능성 높음")
print("- 추정값(totalResults)은 부정확함, 참고용")
print("- 제목에 '남북의 창'이 명시 안 된 클립도 있을 수 있음 (한계)")