#!/usr/bin/env python3
"""토픽 수 K 탐색(UMass 응집도 + perplexity) 후 최적 K로 재적합·결과 저장."""
import re, json
from pathlib import Path
import numpy as np, pandas as pd
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

BASE = Path(__file__).parent
SPECIALS = BASE / "specials"   # 정상회담 특집 텍스트(pdftotext -layout 추출, 미커밋 데이터)
OUT = BASE / "master"
kiwi = Kiwi(); kiwi.load_user_dictionary(str(BASE/"../kiwi_user_dict.tsv"))
STOP = set("""것 수 등 때 년 월 일 이번 지금 오늘 요즘 우리 저희 자신 서로 모습 정도 경우 문제 상황
이야기 부분 생각 이후 이전 지역 사람 사실 통해 위해 대해 관련 대한 여기 거기 하나 가지 방법 결과
시작 시간 동안 모두 다시 특히 최근 당시 그것 이것 무엇 얘기 말씀 때문 만큼 부터 까지 마디 자리
이날 지난 올해 내년 그동안 앞서 그런 이런 저런 어떤 무슨 관계 내용 상태 중심 규모 수준 차원 측면
가운데 속 앞 뒤 안 밖 위 아래 곳 데 점 면 채 뿐 등등 및 또 좀 잘 더 덜 각 별 총 약 명 개 번 차 회
분 초 시 년도 여러분 시청자 남북 북 한반도 북한""".split())

def toks(text):
    return [t.form for t in kiwi.tokenize(str(text))
            if t.tag in ("NNG","NNP") and len(t.form)>=2 and t.form not in STOP]

# 코퍼스: 기사 + 특집 청크
art=pd.read_parquet(BASE/"master/articles.parquet")
docs=[(int(r.year),"기사",toks(r.text)) for _,r in art.iterrows()]
def chunk(t,size=1500):
    ps=[p.strip() for p in re.split(r"\n\s*\n",t) if len(p.strip())>20]
    out=[];b=""
    for p in ps:
        b+=" "+p
        if len(b)>=size:out.append(b.strip());b=""
    if len(b.strip())>200:out.append(b.strip())
    return out
for fn,lab in [("2018 남북 정상회담","남북정상회담특집"),("북미정상회담","북미정상회담특집")]:
    for ch in chunk((SPECIALS/f"{fn}.txt").read_text(encoding="utf-8")):
        docs.append((2018,lab,toks(ch)))

texts=[" ".join(d[2]) for d in docs]
vec=CountVectorizer(min_df=5,max_df=0.35,token_pattern=r"(?u)\b\w+\b")
X=vec.fit_transform(texts); vocab=np.array(vec.get_feature_names_out())
Xb=(X>0).astype(int)
Dw=np.asarray(Xb.sum(axis=0)).ravel()   # 단어별 문서빈도
print(f"문서 {X.shape[0]} · 어휘 {X.shape[1]}")

def umass(model,topn=15):
    cohs=[]
    for comp in model.components_:
        idx=np.argsort(comp)[::-1][:topn]
        sub=Xb[:,idx]
        co=np.asarray((sub.T@sub).todense())   # topn×topn 공동문서수
        s=0;cnt=0
        for i in range(1,topn):
            for j in range(i):
                s+=np.log((co[i,j]+1)/Dw[idx[j]]); cnt+=1
        cohs.append(s/cnt)
    return np.mean(cohs)

print("\nK   UMass응집도   perplexity   (응집도 높을수록=0에가까울수록 좋음)")
results={}
for K in [6,8,10,12,14,16]:
    m=LatentDirichletAllocation(n_components=K,max_iter=30,learning_method="batch",
        doc_topic_prior=0.1,topic_word_prior=0.01,random_state=42).fit(X)
    c=umass(m); p=m.perplexity(X)
    results[K]=(c,p,m)
    print(f"{K:2d}   {c:8.3f}    {p:9.1f}")

bestK=max(results,key=lambda k:results[k][0])
print(f"\n▶ 최적 K = {bestK} (응집도 최고)")

# 최적 K 재적합 결과 저장
m=results[bestK][2]; theta=m.transform(X)
topics=[]
for k in range(bestK):
    top=[vocab[i] for i in np.argsort(m.components_[k])[::-1][:12]]
    topics.append(top)
    print(f"T{k:2d}: {' · '.join(top)}")
df=pd.DataFrame(theta,columns=[f"T{k}" for k in range(bestK)]); df["year"]=[d[0] for d in docs]
yearly=df.groupby("year").mean()
print("\n연도별 평균 토픽 비중(%)\n"+(yearly*100).round(1).to_string())
yearly.to_csv(OUT/"topic_yearly_bestK.csv")
json.dump({"K":bestK,"topics":topics,"yearly":(yearly*100).round(2).to_dict()},
          open(OUT/"topics_bestK.json","w"),ensure_ascii=False,indent=1)
print(f"\n저장: topics_bestK.json, topic_yearly_bestK.csv")
