#!/usr/bin/env python
# coding: utf-8

# **AIFB Associate 자격인증 문제**
# ### **고객의 VOC 정보를 통한 <span style="color:darkgreen">상품변경여부</span> 예측 문제**
# ---

# #### **<span style="color:red">[유의사항]</span>**
# - 각 문항의 답안코드는 반드시 <span style="color:darkgreen">'# 여기에 답안코드를 작성하세요'</span> 로 표시된 cell에 작성해야 합니다. 
# - 제공된 cell을 추가/삭제하고 다른 cell에 답안코드를 작성 시 채점되지 않습니다.
# - <span style="color:darkgreen">본인 핸드폰번호.ipynb로 된 본 문제지를 저장하면 시험종료 시 자동 제출됩니다.</span><br>
# (저장방법: File > Save Notebook 또는 본 문제지 상단의 저장 아이콘 클릭)
# - 반드시 문제에 제시된 가이드를 읽고 답안 작성하세요.
# - 문제에 변수명이 제시된 경우 반드시 해당 변수명을 사용하세요.
# - 자격인증 문제와 데이터는 제 3자에게 공유하거나 개인적인 용도로 사용하는 등 외부로 유출할 수 없으며 유출로 인한 책임은 응시자 본인에게 있습니다.
# ---

# **[ 데이터 컬럼 설명 (데이터 파일명: VOC.csv) ]**
# - 일자: VOC 발생 일자(YYYY-MM-DD)
# - VOC 유형 1레벨: 변경/조회, 청구 수/미납, 해지 3가지 유형으로 구분
# - VOC 유형 2레벨: 상품변경, 요금확인, 해지문의, 해지요청(A사) 등 10가지 유형으로 구분 
# - VOC 유형 3레벨: VOC 상세유형 7가지로 구분
# - VOC 유형 4레벨: VOC 상세유형 23가지로 상세구분
# - 신규가입일자: 상품의 최초 가입일자
# - 상품명: VOC를 제기한 상품
# - 주소: 고객 상품 설치주소
# - 총 사용기간(Month): VOC를 제기한 상품의 총 사용기간(개월 수)
# - 약정기간(Year): VOC를 제기한 상품에 대해 고객이 약정한 기간(년)
# - 연령: 연령대 정보(20s, 30s 등)
# - 요금: VOC를 제기한 상품의 월 이용요금
# - 요금납부방법: Bank Transfer, Credit Card, Cash 등
# - 약정만료여부: VOC를 제기한 상품에 대한 약정만료여부
# - 연계상품 여부: 다른 상품과 연계 여부
# - 고객Care그룹: 고객Care그룹 정보(1~7)
# - 프로모션 정보: 고객에게 제공된 프로모션 정보
# - 상품변경여부: VOC를 제기한 상품의 변경여부

#   
#   

# ### **1. Pandas는 데이터 분석에 가장 많이 쓰이는 파이썬 라이브러리입니다.**
# ### **Pandas를 별칭(alias) pd로 임포트하는 코드를 작성하고 실행하세요.**
# ---

# In[146]:


# 여기에 답안코드를 작성하세요

import pandas as pd


# <br>
# <br>

# ### **2. Matplotlib은 데이터를 다양한 방법으로 시각화할 수 있도록 하는 파이썬 라이브러리입니다.**
# ### **Matplotlib의 pyplot을 사용하기 위해 별칭(alias) plt로 임포트하는 코드를 작성하고 실행하세요.**
# ---

# In[147]:


# 여기에 답안코드를 작성하세요

import matplotlib.pyplot as plt


# <br>
# <br>

# > **<span style="color:red">다음 문항을 풀기 전에 </span>아래 코드를 실행하세요.**
# >

# In[148]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('pip install seaborn')
import seaborn as sns

# tensorflow 관련 라이브러리
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# <br>

# ### **3. 모델링을 위해 분석 및 처리할 데이터 파일을 읽어오려고 합니다.** 
# ### **Pandas 함수로 데이터 파일을 읽어 dataframe 변수에 할당하는 코드를 작성하세요.**

# * **
# - VOC.csv 파일을 읽어 데이터 프레임 변수명 df 에 할당하세요.
# ---

# In[165]:


# 여기에 답안코드를 작성하세요

df = pd.read_csv("VOC.csv")
df.info()


# <br>
# <br>

# ### **4. 예측하려는 결과값의 분포를 확인하려고 합니다.**
# ### **'상품변경여부' 컬럼의 분포를 확인하는 histogram 그래프를 만드세요.**

# * **
# - matplotlib을 활용하세요
# ---

# In[166]:


# 여기에 답안코드를 작성하세요

sns.histplot(df['상품변경여부'])


# <br>
# <br>

# ### **5. 연령별로 사용기간의 차이가 있는지 확인해보고자 합니다.**
# ### **'연령'과 '총 사용기간(Month)'컬럼을 이용하여 barplot 그래프를 만드세요.**

# * **
# - Seaborn을 활용하세요
# - X축에는 '연령'이라고 표시하고 Y축에는 '총 사용기간(Month)'이라고 표시하세요.
# ---

# In[167]:


# 여기에 답안코드를 작성하세요
sns.barplot(x="연령",y="총 사용기간(Month)", data=df)


# <br>
# <br>

# ### **6. 모델링 성능을 제대로 얻기 위해서는 데이터 결측치 처리가 필요합니다.**
# ### **아래 가이드에 따라 결측치를 처리하세요.**

# * **
# - 먼저, 각 컬럼의 결측치를 확인하는 코드를 작성하고 실행하세요.
# - 다음으로, 결측치가 있는 컬럼 중에서 float 타입의 결측치 데이터는 0으로, object 타입의 결측치 데이터는 공백(' ')으로 처리하세요.
# ---

# In[168]:


# 여기에 답안코드를 작성하세요

print(df.isnull().sum())
df['약정만료여부']=df['약정만료여부'].fillna('')
df['고객Care그룹']=df['고객Care그룹'].fillna(0)
df['프로모션 정보']=df['프로모션 정보'].fillna('')


df.info()


# <br>
# <br>

# ### **7. 범주형 데이터를 수치형 데이터로 변환해주는 데이터 전처리 방법 중 하나로 label encoder를 사용합니다.**
# ### **Scikit-learn의 label encoder를 사용하여 범주형 데이터를 수치형 데이터로 변환하세요.** 

# * **
# - 전처리 대상 컬럼: 신규가입일자, 연계상품 여부, 프로모션 정보, 상품변경여부
# - fit_transform을 활용하세요.
# - '일자' 컬럼은 삭제하세요.
# 
# ---

# In[169]:


# 여기에 답안코드를 작성하세요
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['신규가입일자']=le.fit_transform(df['신규가입일자'])
df['연계상품 여부']=le.fit_transform(df['연계상품 여부'])
df['프로모션 정보']=le.fit_transform(df['프로모션 정보'])
df['상품변경여부']=le.fit_transform(df['상품변경여부'])
df=df.drop('일자',axis=1)


# <br>
# <br>

# ### **8. 원-핫 인코딩은 범주형 변수를 1과 0의 이진형 벡터로 변환하기 위하여 사용하는 방법입니다.**
# ### **원-핫 인코딩(One-hot encoding)으로 아래 조건에 해당하는 컬럼 데이터를 변환하세요.**

# * **
# - 원-핫 인코딩 대상: object 타입의 컬럼
# - Pandas의 get_dummies 함수를 활용하고 drop_first는 'True'로 옵션을 적용하세요.
# ---

# In[171]:


# 여기에 답안코드를 작성하세요


df = pd.get_dummies(df, columns = ['VOC 유형 1레벨','VOC 유형 2레벨','VOC 유형 3레벨','VOC 유형 4레벨','상품명','주소','연령','요금납부방법','약정만료여부'],drop_first=True)


# ### **9. 훈련과 검증 각각에 사용할 데이터셋을 분리하려고 합니다.**
# ### **'상품변경여부'컬럼을 label값 y로, 나머지 컬럼을 feature값 X로 할당한 후 훈련데이터셋과 검증데이터셋으로 분리하세요.**
# 

# * **
# - Scikit-learn의 train_test_split 함수를 활용하세요.
# - X, y 데이터로부터 훈련데이터셋과 검증데이터셋을 80:20 비율로 분리하세요
# - 데이터 분리시 y 데이터를 원래 데이터의 분포와 유사하게 추출되도록 옵션을 적용하세요.
# ---

# In[172]:


# 여기에 답안코드를 작성하세요

from sklearn.model_selection import train_test_split
X =df.drop('상품변경여부',axis = 1)
y= df['상품변경여부']
x_train,x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2021, stratify=y)


# ### **10. 데이터들이 동일한 중요도로 반영되도록 하기 위해 데이터 정규화를 하려고 합니다.**
# ### **StandardScaler를 사용하여 아래 조건에 따라 데이터 변수를 정규분포화, 표준화 하세요.**

# * **
# - Scikit-learn의 StandardScaler를 사용하세요.
# - train set은 정규분포화(fit_transform)를 하세요. 
# - valid set은 표준화(transform)를 하세요.
# ---

# In[175]:


# 여기에 답안코드를 작성하세요

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# <br>
# <br>

# ### **11. 랜덤 포레스트(Random Forest)는 분류, 회귀 분석 등에 사용되는 앙상블 학습 방법입니다.**
# ### **아래 하이퍼 파라미터 설정값을 적용하여 Randomforest 모델로 학습을 진행하세요.**

# * **
# - 결정트리의 개수는 100개로 설정하세요.
# - 최대 feature 개수는 9로 설정하세요.
# - 트리의 최대 깊이는 5로 설정하세요.
# - random_state는 42로 설정하세요.
# ---

# In[176]:


# 여기에 답안코드를 작성하세요
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

forest=RandomForestClassifier(n_estimators=100, max_features=9, max_depth=5, random_state =42)
forest.fit(x_train, y_train)


# <br>
# <br>

# ### **12. 위 모델의 성능을 평가하려고 합니다.**
# ### **y값을 예측하여 confusion matrix를 구하고 heatmap그래프로 시각화하세요.**
# ### **또한, Scikit-learn의 classification report 기능을 사용하여 성능을 출력하세요.**

# * **
# - 11번 문제에서 만든 모델로 y값을 예측(predict)하여 y_pred에 저장하세요.
# - Confusion_matrix를 구하고 heatmap 그래프로 시각화하세요. 이때 annotation을 포함시키세요.
# - Scikit-learn의 classification report 기능을 사용하여 클래스별 precision, recall, f1-score를 출력하세요.
# ---

# In[180]:


# 여기에 답안코드를 작성하세요
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

y_pred=forest.predict(x_test)
cf = confusion_matrix(y_test,y_pred)
sns.heatmap(cf,annot=True)

print(classification_report(y_test,y_pred))


# <br>
# <br>

# ### **13. 고객의 상품변경여부를 예측하는 딥러닝 모델을 만들려고 합니다.**
# ### **아래 가이드에 따라 모델링하고 학습을 진행하세요.**

# * **
# - Tensorflow framework를 사용하여 딥러닝 모델을 만드세요.
# - 히든레이어(hidden layer) 3개이상으로 모델을 구성하고 과적합 방지하는 dropout을 설정하세요. 
# - EarlyStopping 콜백으로 정해진 epoch 동안 모니터링 지표가 향상되지 않을 때 훈련을 중지하도록 설정하세요.
# - ModelCheckpoint 콜백으로 validation performance가 좋은 모델을 best_model.h5 파일로 저장하세요.
# ---

# In[177]:


# 여기에 답안코드를 작성하세요
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(98,)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))


model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])


modelpath = "best_model.h5"
checkpointer = ModelCheckpoint( filepath=modelpath, monitor='val_loss', verbose=1,save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10 )

history = model.fit(x_test, y_test, epochs=100, validation_split=0.2, callbacks=[es, checkpointer])
model.load_weights(modelpath)


# <br>
# <br>

# ### **14. 위 딥러닝 모델의 성능을 평가하려고 합니다.**
# ### **학습 정확도 및 손실, 검증 정확도 및 손실을 그래프로 표시하세요.**

# * **
# - 1개의 그래프에 학습 정확도 및 손실, 검증 정확도 및 손실 4가지를 모두 표시하세요.
# - 위 4가지 각각의 범례를 'acc', 'loss', 'val_acc', 'val_loss'로 표시하세요.
# - 그래프의 타이틀은 'Accuracy'로 표시하세요.
# - X축에는 'Epochs'라고 표시하고 Y축에는 'Acc'라고 표시하세요.
# ---

# In[178]:


# 여기에 답안코드를 작성하세요

h = history
plt.plot(h.history['acc'])

plt.plot(h.history['val_acc'])
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['acc','val_acc','loss','val_loss'])
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel("Acc")
plt.show()


# <br><br>

# ### **1번부터 14번까지 모든 문제를 풀었습니다. 수고하셨습니다.**
# ### **핸드폰번호.ipynb 파일을 저장하면 시험종료 시 자동제출됩니다.**