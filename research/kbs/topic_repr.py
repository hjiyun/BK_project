#!/usr/bin/env python3
"""K=8 재적합 후 토픽별 대표 기사(θ 최고) 제목 추출."""
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

# list_title 로드
LT={}
for f in ["kbs_namnam_2018_final.csv","kbs_namnam_2020.csv","kbs_namnam_2022.csv","kbs_namnam_2024.csv"]:
    d=pd.read_csv(BASE/f)
    for ncd,g in d.groupby('ncd'): LT[ncd]=str(g.list_title.iloc[0])

art=pd.read_parquet(BASE/"master/articles.parquet")
docs=[]  # (title, year, tokens)
for _,r in art.iterrows():
    docs.append((LT.get(r.ncd,"(제목없음)"), int(r.year), toks(r.text)))
def chunk(t,size=1500):
    ps=[p.strip() for p in re.split(r"\n\s*\n",t) if len(p.strip())>20];out=[];b=""
    for p in ps:
        b+=" "+p
        if len(b)>=size:out.append(b.strip());b=""
    if len(b.strip())>200:out.append(b.strip())
    return out
for fn,lab in [("2018 남북 정상회담","[특집] 2018 남북 정상회담"),("북미정상회담","[특집] 북미 정상회담")]:
    for ch in chunk((SPECIALS/f"{fn}.txt").read_text(encoding="utf-8")):
        docs.append((lab,2018,toks(ch)))

texts=[" ".join(d[2]) for d in docs]
vec=CountVectorizer(min_df=5,max_df=0.35,token_pattern=r"(?u)\b\w+\b")
X=vec.fit_transform(texts);vocab=np.array(vec.get_feature_names_out())
lda=LatentDirichletAllocation(n_components=8,max_iter=30,learning_method="batch",
    doc_topic_prior=0.1,topic_word_prior=0.01,random_state=42)
theta=lda.fit_transform(X)
labels=["체제·우상화·역사","교육·산업·과학","문화·스포츠","북러 밀착·군사협력",
        "비핵화 외교·정상회담","농업·재해·식량","경제·사회·생활","미사일·군사도발"]
# 토픽 순서 확인용 상위어
for k in range(8):
    top=[vocab[i] for i in np.argsort(lda.components_[k])[::-1][:6]]
    print(f"T{k} [{labels[k]}]: {' '.join(top)}")
print("\n===== 토픽별 대표 기사 (θ 최고 5건) =====")
titles=[d[0] for d in docs]; years=[d[1] for d in docs]
for k in range(8):
    order=np.argsort(theta[:,k])[::-1]
    print(f"\n■ T{k} {labels[k]}")
    seen=set();cnt=0
    for i in order:
        t=titles[i]
        key=(t,years[i])
        if key in seen: continue
        seen.add(key)
        print(f"   {years[i]} ({theta[i,k]*100:.0f}%) {t[:56]}")
        cnt+=1
        if cnt>=5: break
