#!/usr/bin/env python3
"""KBS 남북의 창 전체 코퍼스(기사 + 2018 정상회담 특집) LDA 토픽모델링.
연도별 토픽 추세(평균 토픽 비중) 산출."""
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
    out=[]
    for t in kiwi.tokenize(str(text)):
        if t.tag in ("NNG","NNP") and len(t.form)>=2 and t.form not in STOP:
            out.append(t.form)
    return out

# 1) 기사 코퍼스
art = pd.read_parquet(BASE/"master/articles.parquet")
docs=[]  # (id, year, date, label, tokens)
for _,r in art.iterrows():
    docs.append((f"ncd{r.ncd}", int(r.year), str(r.date), "기사", toks(r.text)))

# 2) 특집 PDF(텍스트) 청크 추가
specials=[("2018 남북 정상회담","20180427","남북정상회담특집"),
          ("북미정상회담","20180612","북미정상회담특집")]
def chunk(text, size=1500):
    paras=[p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip())>20]
    chunks=[]; buf=""
    for p in paras:
        buf += " "+p
        if len(buf)>=size: chunks.append(buf.strip()); buf=""
    if len(buf.strip())>200: chunks.append(buf.strip())
    return chunks
for fname,date,label in specials:
    txt=(SPECIALS/f"{fname}.txt").read_text(encoding="utf-8")
    for i,ch in enumerate(chunk(txt)):
        docs.append((f"{label}_{i}",2018,date,label,toks(ch)))

print(f"문서 총 {len(docs)}개 (기사 {sum(1 for d in docs if d[3]=='기사')} + 특집 {sum(1 for d in docs if d[3]!='기사')})")

# 3) 벡터화 + LDA
texts=[" ".join(d[4]) for d in docs]
vec=CountVectorizer(min_df=5, max_df=0.35, token_pattern=r"(?u)\b\w+\b")
X=vec.fit_transform(texts); vocab=np.array(vec.get_feature_names_out())
print(f"어휘 {len(vocab)}개, 문서-단어 행렬 {X.shape}")

K=10
lda=LatentDirichletAllocation(n_components=K, max_iter=30, learning_method="batch",
                              doc_topic_prior=0.1, topic_word_prior=0.01, random_state=42)
theta=lda.fit_transform(X)   # 문서×토픽 비중

# 4) 토픽별 상위어
print("\n===== 토픽별 상위어 (K=10) =====")
for k in range(K):
    top=vocab[np.argsort(lda.components_[k])[::-1][:12]]
    print(f"T{k:2d}: {' · '.join(top)}")

# 5) 연도별 평균 토픽 비중
df=pd.DataFrame(theta, columns=[f"T{k}" for k in range(K)])
df["year"]=[d[1] for d in docs]
yearly=df.groupby("year").mean()
print("\n===== 연도별 평균 토픽 비중(%) =====")
print((yearly*100).round(1).to_string())
yearly.to_csv(OUT/"topic_yearly.csv")

# 6) 특집이 실린 토픽
print("\n===== 정상회담 특집 문서의 주(主)토픽 =====")
for i,d in enumerate(docs):
    if d[3]!="기사":
        dom=int(np.argmax(theta[i]))
        print(f"  {d[3]:14} chunk → T{dom} ({theta[i][dom]*100:.0f}%)")
