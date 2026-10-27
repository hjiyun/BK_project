#!/usr/bin/env python3
"""K=8 LDA 재적합 후 분기(연도×Q1-Q4)별 평균 토픽 비중."""
import re
from pathlib import Path
import numpy as np, pandas as pd
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

BASE=Path(__file__).parent
SPECIALS = BASE / "specials"   # 정상회담 특집 텍스트(pdftotext -layout 추출, 미커밋 데이터)
OUT = BASE / "master"
kiwi=Kiwi(); kiwi.load_user_dictionary(str(BASE/"../kiwi_user_dict.tsv"))
STOP=set("""것 수 등 때 년 월 일 이번 지금 오늘 요즘 우리 저희 자신 서로 모습 정도 경우 문제 상황
이야기 부분 생각 이후 이전 지역 사람 사실 통해 위해 대해 관련 대한 여기 거기 하나 가지 방법 결과
시작 시간 동안 모두 다시 특히 최근 당시 그것 이것 무엇 얘기 말씀 때문 만큼 부터 까지 마디 자리
이날 지난 올해 내년 그동안 앞서 그런 이런 저런 어떤 무슨 관계 내용 상태 중심 규모 수준 차원 측면
가운데 속 앞 뒤 안 밖 위 아래 곳 데 점 면 채 뿐 등등 및 또 좀 잘 더 덜 각 별 총 약 명 개 번 차 회
분 초 시 년도 여러분 시청자 남북 북 한반도 북한""".split())
def toks(t):
    return [x.form for x in kiwi.tokenize(str(t)) if x.tag in("NNG","NNP") and len(x.form)>=2 and x.form not in STOP]

art=pd.read_parquet(BASE/"master/articles.parquet")
docs=[]  # (year, date, tokens)
for _,r in art.iterrows():
    docs.append((int(r.year), str(r.date), toks(r.text)))
def chunk(t,size=1500):
    ps=[p.strip() for p in re.split(r"\n\s*\n",t) if len(p.strip())>20];o=[];b=""
    for p in ps:
        b+=" "+p
        if len(b)>=size:o.append(b.strip());b=""
    if len(b.strip())>200:o.append(b.strip())
    return o
for fn,date in [("2018 남북 정상회담","20180427"),("북미정상회담","20180612")]:
    for ch in chunk((SPECIALS/f"{fn}.txt").read_text(encoding="utf-8")):
        docs.append((2018,date,toks(ch)))

texts=[" ".join(d[2]) for d in docs]
vec=CountVectorizer(min_df=5,max_df=0.35,token_pattern=r"(?u)\b\w+\b")
X=vec.fit_transform(texts)
lda=LatentDirichletAllocation(n_components=8,max_iter=30,learning_method="batch",
    doc_topic_prior=0.1,topic_word_prior=0.01,random_state=42)
theta=lda.fit_transform(X)
labels=["체제·우상화","교육·산업","문화·스포츠","북러밀착","비핵화외교","농업·재해","경제·사회","미사일·도발"]

def q(date):
    m=int(date[4:6]); return (int(date[:4]), (m-1)//3+1)
bins=[q(d[1]) for d in docs]
df=pd.DataFrame(theta,columns=labels); df["bin"]=[f"{y}Q{qq}" for (y,qq) in bins]
order=[f"{y}Q{qq}" for y in [2018,2020,2022,2024] for qq in [1,2,3,4]]
g=df.groupby("bin")
mean=(g[labels].mean()*100).reindex(order).round(1)
n=g.size().reindex(order).fillna(0).astype(int)
mean.insert(0,"N",n)
print("분기별 평균 토픽 비중(%) · N=기사수")
print(mean.to_string())
mean.to_csv(OUT/"topic_quarter.csv")
print("\n구간별 N 최소/최대:", int(n.min()), int(n.max()))
