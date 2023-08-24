import streamlit as st
import pandas as pd
import numpy as np
import xgboost
from sklearn import datasets
from xgboost.sklearn import XGBClassifier

data3 = pd.read_csv("https://raw.githubusercontent.com/nkuwangkai/app-for-mortality-prediction/main/laboratory.csv",thousands=',',encoding='GBK')
Xtrain = (data3.iloc[:,1:19]) 
Ytrain = (data3.iloc[:,0])

clf = XGBClassifier(objective='binary:logistic',
              booster='gbtree',
              colsample_bytree=0.7823379,
              gamma=0.07321518,
              learning_rate=0.05336876,
              max_delta_step=7,
              max_depth=3,
              min_child_weight=50,
              n_estimators=259,
              subsample=0.797208)

clf.fit(Xtrain,Ytrain)

# Title
st.header("Machine learning app for integration in laboratory test")								

# 用户输入数据
def user_input_features():
    wbc = st.sidebar.slider('wbc', 0.1, 100, 10)
    hemoglobin = st.sidebar.slider('hemoglobin', 2.0, 30, 10)
    rdw = st.sidebar.slider('rdw', 10, 40, 20)
    albumin = st.sidebar.slider('albumin', 0.1, 6.0, 3.0)
    total_bilirubin = st.sidebar.slider('total bilirubin', 0.1, 50, 10)
    bun = st.sidebar.slider('bun',1.0, 250, 10)
    creatinine = st.sidebar.slider('creatinine', 0.1, 25, 5)
    glucose = st.sidebar.slider('glucose', 15, 500, 50)
    sodium = st.sidebar.slider('sodium', 80, 180, 150)
    chloride = st.sidebar.slider('chloride', 60, 150, 100)
    neutrophils_lymphocytes_ratio = st.sidebar.slider('neutrophils - lymphocytes ratio', 0.1, 10000, 100)
    hematocrit_rdw_ratio = st.sidebar.slider('hematocrit - rdw ratio', 0.1, 5.0, 3.0)
    rdw_albumin_ratio = st.sidebar.slider('rdw - albumin ratio', 2.0, 42, 5.0)
    neutrophils_albumin_ratio = st.sidebar.slider('neutrophils - albumin ratio', 0.1, 60, 4.0)
    bun_creatinine_ratio = st.sidebar.slider('bun - creatinine ratio', 1.0, 190, 15)
    albumin_total_bilirubin_ratio = st.sidebar.slider('albumin - total bilirubin ratio', 1.0, 45, 15)
    alt_ast_ratio = st.sidebar.slider('alt - ast ratio', 0.1, 18, 2.0)
    bun_albumin_ratio = st.sidebar.slider('bun - albumin ratio', 0.1, 90, 20)
    data = {'wbc': wbc, 'hemoglobin': hemoglobin,
            'rdw': rdw,'albumin': albumin,
            'total_bilirubin': total_bilirubin,'bun': bun,
            'creatinine': creatinine,'glucose': glucose,
            'sodium': sodium,'chloride': chloride,
            'neutrophils_lymphocytes_ratio': neutrophils_lymphocytes_ratio,'hematocrit_rdw_ratio': hematocrit_rdw_ratio,
            'rdw_albumin_ratio': rdw_albumin_ratio,'neutrophils_albumin_ratio': neutrophils_albumin_ratio,
            'bun_creatinine_ratio': bun_creatinine_ratio,'albumin_total_bilirubin_ratio': albumin_total_bilirubin_ratio,
            'alt_ast_ratio': alt_ast_ratio,'bun_albumin_ratio': bun_albumin_ratio}
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)
st.text(f"In-hospital mortality risk score is {prediction_proba}")
