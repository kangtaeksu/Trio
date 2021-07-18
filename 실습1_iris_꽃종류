#!/usr/bin/env python
# coding: utf-8

# **실습.1**
# ### **iris 데이터를 이용한 <span style="color:darkgreen">AI분류</span> 문제**
# ---
# 

# #### 붓꽃의 종류, 길이, 너비 등의 데이터를 이용하여 붓꽃(iris)의 종류를 분류하는 AI 모델 제작 문제입니다. 
# #### AI코딩 단계에 따라 주어지는 문제를 읽고 답안을 작성하세요.
#  - 데이터 : 분류(카테고리)
#  - 모델 : RandomForest, DeepLerning
#  - 주요 전처리 : 분석 Column 추가, label 전처리(카테고리 → 수치화)
#  - 주요 학습 내용 : 산점도, 히스토그램, 분류형 모델 생성(분류방법, input, output 처리, 손실함수 등)
# ---
# 

# **iris.csv / iris(붓꽃) 데이터 컬럼 설명**
# - sepal.length : 큰 꽃잎의 길이
# - sepal.width : 큰 꽃잎의 너비
# - petal.length : 작은 꽃잎의 길이
# - petal.width : 작은 꽃잎의 너비
# - variety : 클래스, target, label
#     * Setosa, Versicolor, Virginica

# ---
# > **<span style="color:red">다음 문항을 풀기 전에 </span>아래 코드를 실행해주시기 바랍니다.**<br>
# > - AIDU 사용을 위한 AIDU 환경변수를 선언을 하는 코드. <span style="color:darkgreen"></span><br>
# 
# ---

# In[1]:


# AIDU 내부 연동을 위한 라이브러리
from aicentro.session import Session
from aicentro.framework.keras import Keras as AiduFrm
# AIDU와 연동을 위한 변수
aidu_session = Session(verify=False)
aidu_framework = AiduFrm(session=aidu_session)


# ### **Q1. Pandas를 pd로 alias하여 사용할 수 있도록 불러오는 코드를 작성하고 실행하시기 바랍니다.**
# ---

# In[2]:


# 여기에 답안코드를 작성하세요
import pandas as pd


# ### **Q2.Matplotlib의 pyplot을 plt로 alias하여 사용할 수 있도록 불러오는 코드를 작성하고 실행하시기 바랍니다.**
# ---

# In[3]:


# pip install seaborn


# In[4]:


# 여기에 답안코드를 작성하세요
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


# ### **Q3.iris.csv를 판다스 데이터 프레임으로 불러와서 iris에 선언하는 코드를 작성하고 실행하시기 바랍니다.**
# ---

# In[60]:


# 여기에 답안코드를 작성하세요
iris = pd.read_csv("iris.csv")
iris = pd.DataFrame(data=iris)
iris


# ### **Q4. 데이터 프레임 iris의 처음 6개 행을 조회하는 코드를 작성하고 실행하시기 바랍니다.**
# ---

# In[61]:


# 여기에 답안코드를 작성하세요
iris.head()


# ### **Q5. 데이터 프레임 iris의 variety 컬럼을 바 플롯(bar plot)을 이용하여 시각화 하시기 바랍니다.**

# In[74]:


# 여기에 답안코드를 작성하세요
x = pd.value_counts(iris['variety'])
index = iris['variety'].unique()

plt.bar(index,x)


# ### **Q6. 데이터 프레임 iris의 sepal.length 컬럼을 히스토 그램을 이용하여 시각화 하시기 바랍니다.**
# ---

# In[76]:


# 여기에 답안코드를 작성하세요
sns.histplot(iris["sepal.length"])


# ### **Q7. 데이터 프레임 iris의 petal.width 컬럼을 히스토 그램을 이용하여 시각화 하시기 바랍니다.**
# 
# * **
# - 다섯개 구간으로 나누어 시각화 하시오.
# ---

# In[63]:


# 여기에 답안코드를 작성하세요
sns.histplot(iris["petal.width"],bins=5)


# ### **Q8. 데이터 프레임 iris의 sepal.width를 x축으로 petal.width를 y축으로 하는 산점도를 시각화 하시기 바랍니다.**
# ---

# In[10]:


# 여기에 답안코드를 작성하세요
plt.plot('sepal.width','petal.width',data=iris,linestyle='none',marker='o')
plt.xlabel('sepal.width')
plt.ylabel('petal.width')
plt.show()


# ### **Q9. 데이터 프레임 iris의 sepal.length를 x축으로 petal.length를 y축으로 하는 산점도를 시각화 하시기 바랍니다.**
# 
# * **
# - class에 따라 다른색을 띄도록 시각화 하시오.
# ---

# In[11]:


# 여기에 답안코드를 작성하세요
sns.lmplot(x='sepal.length',y='petal.length',fit_reg=False,data=iris,hue='variety')
#FIt reg : 선
plt.show()


# ### **Q10. 다음 조건에 맞추어 데이터 프레임 iris에 새로운 컬럼 sepal_ratio 를 추가하시기 바랍니다.**
# * **
# - sepal.length를 분자로 sepal.width를 분모로 하는 비율을 sepal_ratio로 정의한다.
# ---

# In[64]:


# 여기에 답안코드를 작성하세요
iris['sepal_ratio'] = iris['sepal.length']/iris['sepal.width']
iris


# ### **Q11. 다음 조건에 맞추어 데이터 프레임 iris에 새로운 컬럼 length_diff 를 추가하시기 바랍니다.**
# 
# * **
# - sepal.length와 petal.length의 차이의 크기를 length_diff로 정의한다.
# ---

# In[65]:


# 여기에 답안코드를 작성하세요
iris['length_diff'] = abs(iris['sepal.length']-iris['petal.length'])
iris


# ### **Q12. 데이터 프레임 iris의 컬럼 variety를 label encoding하시기 바랍니다.**
# ---

# In[66]:


# 여기에 답안코드를 작성하세요
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
iris['variety']=le.fit_transform(iris['variety'])
# iris['variety'].unique()


# In[67]:


iris


# In[15]:


iris.iloc[:,0:4]


# ### **Q13. 데이터를 트레이닝셋 / 테스트셋으로 분할하시기 바랍니다.**
# * **
# - y는 iris데이터 프레임의 'variety'컬럼이고 x는 그 나머지 컬럼이다.
# - train : test = 9 : 1
# - y의 클래스가 골고루 분할되도록 stratify하게 분할한다.
# - 변수명 규칙은 다음과 같다.
#     * x_train, y_train
#     * x_test, y_test
# - random state, seed 등은 2021로 설정한다.
# ---

# In[77]:


# 여기에 답안코드를 작성하세요

from sklearn.model_selection import train_test_split

x=iris.drop('variety',axis=1)
y = iris['variety']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,stratify=y,random_state=2021)


# ### **Q15. Random Forest 모델들을 학습시키시기 바랍니다.**
# * **
# - RandomForestClassifier 하이퍼파라미터 설정 :  n_estimators=50, max_depth=13, random_state=30, min_samples_leaf=5
# - n_estimators 종합한 전체 트리의 가지수, max_depth : 각 Tree의 가장 깊은 높이, min_samples_leaf: 각 끝의 노드에는 최소 5개의 트레이닝 샘플이 있어야함
# - 트레이닝 셋 (x_train, y_train)을 이용하여 학습시킨다.
# - Forest를 이루는 tree의 leaf안에는 최소한 5개의 트레이닝셋 샘플이 있어야 한다.
# - seed나 random_state는 2021로 고정한다.
# ---

# In[78]:


# 여기에 답안코드를 작성하세요
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50,  max_depth=13, random_state=30, min_samples_leaf=5)
model.fit(x_train,y_train)


# > **<span style="color:red">다음 문항을 풀기 전에 </span>아래 코드를 실행하세요.**
# >

# In[70]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping


# ### **Q16. 아래 조건에 맞추어 뉴럴네트워크 모델을 학습시키시기 바랍니다.**
# * **
# - Tensorflow framework를 사용한다.
# - 히든레이어는 아래와 같은 규칙에 맞추어 구성합니다.
#     * 2개의 fully connected layer를 사용할 것
#     * Batchnormalization을 반드시 활용한다.
# - Early stopping을 이용하여, validation loss가 5번 이상 개선되지 않으면 학습을 중단 시키고, 가장 성능이 좋았을 때의 가중치를 복구한다.
# - 학습과정의 로그(loss, accuracy)를 history에 선언하여 남긴다.
# - y를 별도로 원핫인코딩 하지 않고 분류모델을 학습시킬 수 있도록 한다.(해당 형태의 경우 loss function은 sparse_categorical_crossentropy를 활용해야한다.)
# - epochs는 2000번을 지정한다.
# ---

# In[52]:


y_train


# In[85]:


# 여기에 답안코드를 작성하세요
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.callbacks import ModelCheckpoint
# from keras.optimizer_v2.adam import Adam
# from tensorflow.keras.wrappers.scikit_leran import KerasClassifier

model = Sequential()


model.add(Dense(10, activation='relu',input_shape=(6,)))
model.add(BatchNormalization())
model.add(Dense(3, activation='softmax'))

opt=keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)

history=model.fit(x_train,y_train, epochs=2000,batch_size=10, verbose=2, validation_data=(x_test,y_test), callbacks=[es,mc])


# ### **Q17. 다음 조건에 맞추어 뉴럴네트워크의 학습 로그를 시각화 하시기 바랍니다.**
# * **
# - 필요한 라이브러리가 있다면 따로 불러온다.
# - epochs에 따른 accuracy의 변화를 시각화 한다.
# - train accuracy와 validation accuracy를 전부 시각화하고, 구별가능해야 한다.
# - 그래프의 타이틀은 'Accuracy'로 표시한다.
# - x축에는 'epochs'라고 표시하고 y축에는 'accuracy'라고 표시한다.
# ---

# In[86]:


# 여기에 답안코드를 작성하세요
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train_accuracy','validation_accuracy'])

plt.show()


# In[ ]:




